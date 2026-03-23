# stack.py
# ------------------------------------------------------------
# v4: Group-aware Prefit Calibration + Composite Class Weights + Subject-Grouped Stacking
#  - Fixed schema: train.csv must contain ['sequence_id', 'subject'].
#  - Every OOF CSV must contain ['sequence_id', 'gesture_true', <class probability columns>].
#  - Stage two uses GroupKFold(groups=subject) to prevent subject leakage.
#  - Supported meta-models: 'ridge', 'logistic', 'rf', 'lgbm', 'catboost', 'xgb'.
#  - Ridge supports group-aware prefit calibration as a safer alternative to cv=3 calibration.
#  - Logits and linear-feature scaling stay off by default because they underperformed here.
#  - Exports meta_model.pkl + meta_info.json, with optional second-stage OOF probabilities.
# ------------------------------------------------------------

import os
import json
from functools import reduce
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Optional libs
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "INPUT_DIR": "",             # Directory holding the base-model OOF probability CSVs.
    "TRAIN_CSV": "",             # Training metadata with at least sequence_id and subject.
    "SUBJECT_COL": "subject",    # Subject column used to build GroupKFold splits.
    "STRICT_SUBJECT_MATCH": True,     # Error if any sequence_id cannot be matched back to a subject.

    # Stacking / features
    "META_MODEL": "ridge",            # 'ridge'|'logistic'|'rf'|'lgbm'|'catboost'|'xgb'
    "N_SPLITS": 5,                    # GroupKFold splits, capped by the number of unique subjects.
    "USE_LOGITS": False,              # Disabled because raw probabilities worked better here.
    "SCALE_LINEAR": False,            # Disabled because linear scaling hurt the public score.
    "RANDOM_STATE": 42,

    # Calibration options. Only the ridge path uses them.
    "RIDGE_CALIBRATE": False,         # Uses non-group-aware inner CV and is slightly optimistic.
    "RIDGE_CALIBRATE_PREFIT": False,  # Splits calibration holdout by subject inside each outer fold.
    "CALIBRATION_METHOD": "sigmoid",  # 'sigmoid' or 'isotonic'
    "CALIBRATION_HOLDOUT": 0.2,       # Fraction of each outer-train fold reserved for calibration.

    # Competition-shaped class weighting.
    "USE_COMPOSITE_CLASS_WEIGHTS": True,

    # Meta-model hyperparameters.
    "PARAMS": {
        "ridge": {
            "alpha": 11,             # Primary ridge knob; a small log grid usually suffices.
            "fit_intercept": True,
            "solver": "auto",         # Try 'lbfgs' or 'saga' if your sklearn build supports them.
            "tol": 1e-4,
            "max_iter": None,
            "positive": False,
            "class_weight": None,     # Sample weights already encode the competition weighting.
        },
        "logistic": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "multi_class": "auto"},
        "rf": {"n_estimators": 500, "max_depth": None, "n_jobs": -1, "random_state": 42},
        "lgbm": {"n_estimators": 1200, "learning_rate": 0.05, "num_leaves": 63, "objective": "multiclass", "random_state": 42},
        "xgb": {"n_estimators": 1200, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
                "tree_method": "hist", "objective": "multi:softprob", "eval_metric": "mlogloss", "random_state": 42},
        "catboost": {"iterations": 1200, "learning_rate": 0.05, "depth": 6, "loss_function": "MultiClass", "verbose": False, "random_state": 42},
    },

    # Export paths.
    "SAVE_DIR": "./stack_artifacts",          # Directory for the trained meta-model artifacts.
    "SAVE_OOF_STACK_CSV": "./stack_oof.csv",  # Optional second-stage OOF probability dump.
}

# ============================================================
# Competition metric constants
# ============================================================
TARGET_GESTURES = [
    'Forehead - scratch', 'Forehead - pull hairline', 'Neck - scratch',
    'Neck - pinch skin', 'Eyelash - pull hair', 'Above ear - pull hair',
    'Eyebrow - pull hair', 'Cheek - pinch skin'
]
NON_TARGET_CLASS_NAME = 'Other'


def calculate_competition_metric_from_9_classes(
    y_true_9_encoded: np.ndarray,
    y_pred_9_encoded: np.ndarray,
    label_encoder_9_class: LabelEncoder,
    non_target_index: int,
):
    """Return ``0.5 * binary_f1 + 0.5 * macro_f1`` on the observed labels."""
    present_labels = np.unique(np.concatenate((y_true_9_encoded, y_pred_9_encoded)))
    macro_f1 = f1_score(
        y_true_9_encoded, y_pred_9_encoded,
        average='macro', labels=present_labels, zero_division=0
    )
    y_true_binary = (y_true_9_encoded != non_target_index).astype(int)
    y_pred_binary = (y_pred_9_encoded != non_target_index).astype(int)
    binary_f1 = f1_score(
        y_true_binary, y_pred_binary,
        average='binary', pos_label=1, zero_division=0
    )
    return 0.5 * binary_f1 + 0.5 * macro_f1, binary_f1, macro_f1


# ============================================================
# Competition-shaped 18-class weighting
# ============================================================
def calculate_composite_weights_18_class(
    y_18_class_series: pd.Series,
    label_encoder_18_class: LabelEncoder,
    target_gesture_names: List[str],
) -> Dict[int, float]:
    """
    Compute the competition-shaped class weights for the 18-class model.
    """
    IMP_BFRB = 25 / 288
    IMP_NON_BFRB_INDIVIDUAL = 11 / 360

    class_counts = y_18_class_series.value_counts()

    raw_weights = {}
    for name in label_encoder_18_class.classes_:
        count = class_counts.get(name, 1)
        if name in target_gesture_names:
            raw_weights[name] = IMP_BFRB / count
        else:
            raw_weights[name] = IMP_NON_BFRB_INDIVIDUAL / count

    total_raw_weight = sum(raw_weights.values())
    num_classes = len(raw_weights)
    avg_raw_weight = total_raw_weight / num_classes if num_classes > 0 else 1.0
    if avg_raw_weight <= 1e-9:
        avg_raw_weight = 1.0

    normalized_weights = {name: w / avg_raw_weight for name, w in raw_weights.items()}
    class_weight_dict = {
        idx: normalized_weights.get(name, 1.0)
        for idx, name in enumerate(label_encoder_18_class.classes_)
    }
    return class_weight_dict


# ============================================================
# Helper functions
# ============================================================
def _safe_normalize_probs(probs: np.ndarray) -> np.ndarray:
    """Row-normalize probabilities after scrubbing NaN/Inf values."""
    probs = np.nan_to_num(probs, nan=1e-15, posinf=1e-15, neginf=1e-15)
    rs = probs.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return probs / rs


def _prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply a numerically safe logit transform."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def build_feature_matrix(prob_matrices, model_names, class_names_full, use_logits, with_model_name_prefix=True):
    """
    Stack ``M`` probability matrices into a ``[N, M*C]`` feature matrix.
    """
    M = len(prob_matrices)
    N, C = prob_matrices[0].shape
    mats = []
    colnames = []
    for name, mat in zip(model_names, prob_matrices):
        mat = _safe_normalize_probs(mat)
        if use_logits:
            mat = _prob_to_logit(mat)
        mats.append(mat)
        prefix = (name + "::") if with_model_name_prefix else ""
        colnames += [f"{prefix}{cls}" for cls in class_names_full]
    X = np.concatenate(mats, axis=1)  # [N, M*C]
    return X, colnames


def _extract_final_estimator(estimator):
    """Return the final estimator when the model is wrapped in a Pipeline."""
    if hasattr(estimator, "steps"):  # sklearn Pipeline
        return estimator.steps[-1][1]
    return estimator


def _proba_in_full_order(estimator, X, n_classes: int, le_full: LabelEncoder = None):
    """
    Return probabilities aligned to the full class order ``[0..n_classes-1]``.

    Models without ``predict_proba`` fall back to ``decision_function`` plus a
    softmax approximation. If ``classes_`` is available, probabilities are
    reordered into the full class space.
    """
    est_final = _extract_final_estimator(estimator)

    # Prefer native probabilities when the estimator exposes them.
    if hasattr(estimator, "predict_proba"):
        proba_list = estimator.predict_proba(X)
        if isinstance(proba_list, list):
            proba = np.column_stack([p[:, 1] if p.ndim == 2 else p for p in proba_list])
        else:
            proba = np.asarray(proba_list)
    else:
        # Uncalibrated ridge models still expose decision_function.
        if hasattr(estimator, "decision_function"):
            df = estimator.decision_function(X)
            df = np.asarray(df)
            if df.ndim == 1:
                df = np.stack([-df, df], axis=1)
            proba = _softmax(df)
        else:
            raise RuntimeError("该元模型既无 predict_proba 也无 decision_function，无法得到概率。")

    proba = _safe_normalize_probs(proba)

    if proba.shape[1] == n_classes:
        return proba

    if proba.shape[1] == 1 and n_classes == 2:
        return _safe_normalize_probs(np.hstack([1 - proba, proba]))

    classes = getattr(est_final, "classes_", None)
    if classes is not None:
        classes = np.array(classes)
        full = np.zeros((proba.shape[0], n_classes), dtype=float)
        for j, cls in enumerate(classes):
            if isinstance(cls, (np.integer, int)) and 0 <= int(cls) < n_classes:
                full[:, int(cls)] = proba[:, j]
            elif le_full is not None:
                idx = int(le_full.transform([cls])[0])
                full[:, idx] = proba[:, j]
            else:
                return proba
        return _safe_normalize_probs(full)

    return proba


def make_meta_model(name: str, n_classes: int, cfg: dict):
    """
    Construct the configured meta-model.

    Linear models can be wrapped in a scaler depending on ``SCALE_LINEAR``.
    """
    name = name.lower()
    scale = cfg["SCALE_LINEAR"]
    rs = cfg["RANDOM_STATE"]
    params = cfg["PARAMS"].get(name, {})

    if name == "ridge":
        base = RidgeClassifier(
            alpha=params.get("alpha", 1.0),
            fit_intercept=params.get("fit_intercept", True),
            copy_X=True,
            max_iter=params.get("max_iter", None),
            tol=params.get("tol", 1e-4),
            class_weight=params.get("class_weight", None),
            solver=params.get("solver", "auto"),
            positive=params.get("positive", False),
            random_state=rs,
        )
        if scale:
            return Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                             ("est", base)])
        return base

    if name == "logistic":
        est = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            solver=params.get("solver", "lbfgs"),
            multi_class=params.get("multi_class", "auto"),
            random_state=rs,
            n_jobs=-1 if "n_jobs" in params else None,
        )
        if scale:
            return Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                             ("est", est)])
        return est

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", None),
            n_jobs=params.get("n_jobs", -1),
            random_state=rs
        )

    if name == "lgbm":
        if LGBMClassifier is None:
            raise ImportError("需要 lightgbm：pip install lightgbm")
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 1200),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 63),
            objective=params.get("objective", "multiclass"),
            random_state=rs
        )

    if name == "xgb":
        if XGBClassifier is None:
            raise ImportError("需要 xgboost：pip install xgboost")
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 1200),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            tree_method=params.get("tree_method", "hist"),
            objective=params.get("objective", "multi:softprob"),
            eval_metric=params.get("eval_metric", "mlogloss"),
            random_state=rs
        )

    if name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("需要 catboost：pip install catboost")
        return CatBoostClassifier(
            iterations=params.get("iterations", 1200),
            learning_rate=params.get("learning_rate", 0.05),
            depth=params.get("depth", 6),
            loss_function=params.get("loss_function", "MultiClass"),
            verbose=params.get("verbose", False),
            random_state=rs
        )

    raise ValueError(f"未知 META_MODEL: {name}")


def _fit_with_sample_weight(estimator, X, y, sample_weight: np.ndarray = None):
    """
    Fit an estimator while handling ``sample_weight`` for both pipelines and raw models.
    """
    if sample_weight is None:
        return estimator.fit(X, y)

    try:
        if hasattr(estimator, "steps"):  # Pipeline
            return estimator.fit(X, y, **{"est__sample_weight": sample_weight})
        else:
            return estimator.fit(X, y, sample_weight=sample_weight)
    except TypeError as e:
        print(f"[WARN] {type(_extract_final_estimator(estimator)).__name__} 不支持 sample_weight，已回退无权训练。 ({e})")
        return estimator.fit(X, y)


# ============================================================
# Main flow
# ============================================================
def main():
    # Read the base-model OOF predictions.
    input_dir = CONFIG["INPUT_DIR"]
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"目录不存在：{input_dir}")

    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if len(csv_files) < 2:
        raise ValueError("至少需要 2 个 OOF 概率文件用于 stacking。")
    print(f"[Info] 发现 {len(csv_files)} 个 OOF CSV：{csv_files}")

    all_dfs = {f: pd.read_csv(os.path.join(input_dir, f)) for f in csv_files}

    # Keep only sequence_ids shared by every base model.
    for name, df in all_dfs.items():
        if "sequence_id" not in df.columns or "gesture_true" not in df.columns:
            raise ValueError(f"文件 {name} 必须包含 'sequence_id' 和 'gesture_true' 列。")
    common_seq = reduce(np.intersect1d, (df["sequence_id"].values for df in all_dfs.values()))
    if len(common_seq) == 0:
        raise ValueError("不同模型的 OOF 文件之间没有共同的 sequence_id。")
    master_df = pd.DataFrame({"sequence_id": common_seq})

    # Freeze class order from the first file so every matrix shares one schema.
    first_df = next(iter(all_dfs.values()))
    class_names_full = sorted([c for c in first_df.columns if c not in ["sequence_id", "gesture_true"]])
    if not class_names_full:
        raise ValueError("未检测到类别概率列。")
    n_classes = len(class_names_full)

    # Encode the ground-truth labels in the full class space.
    truth = master_df.merge(first_df[["sequence_id", "gesture_true"]], on="sequence_id", how="left")
    y_true_names_all = truth["gesture_true"].values
    le_full = LabelEncoder().fit(class_names_full)
    y_all = le_full.transform(y_true_names_all)

    # Join subject ids so stage-two CV can stay group-aware.
    train = pd.read_csv(CONFIG["TRAIN_CSV"])
    need_cols = {"sequence_id", CONFIG["SUBJECT_COL"]}
    if not need_cols.issubset(train.columns):
        raise ValueError(f"train.csv 必须含列：'sequence_id' 和 '{CONFIG['SUBJECT_COL']}'")
    subj_map = train[["sequence_id", CONFIG["SUBJECT_COL"]]].drop_duplicates("sequence_id").rename(
        columns={CONFIG["SUBJECT_COL"]: "subject"}
    )
    merged = master_df.merge(subj_map, on="sequence_id", how="left")
    if merged["subject"].isna().any():
        miss = merged.loc[merged["subject"].isna(), "sequence_id"].head(10).tolist()
        if CONFIG["STRICT_SUBJECT_MATCH"]:
            raise ValueError(f"{merged['subject'].isna().sum()} 个 sequence_id 找不到 subject，例如：{miss}")
        else:
            print(f"[警告] 有 {merged['subject'].isna().sum()} 个 sequence_id 无 subject，将剔除这些行。")
            merged = merged.loc[~merged["subject"].isna()].copy()
    groups = merged["subject"].astype(str).values
    if len(merged) != len(master_df):
        # Any dropped sequence_ids must be removed from every derived array.
        master_df = merged[["sequence_id"]].reset_index(drop=True)
        truth = master_df.merge(first_df[["sequence_id", "gesture_true"]], on="sequence_id", how="left")
        y_true_names_all = truth["gesture_true"].values
        y_all = le_full.transform(y_true_names_all)

    # Align each model's probabilities to the shared sequence_id order.
    prob_matrices, model_names = [], []
    for name, df in all_dfs.items():
        dfm = master_df.merge(df, on="sequence_id", how="left")
        if not set(class_names_full).issubset(dfm.columns):
            miss_cols = [c for c in class_names_full if c not in dfm.columns]
            raise ValueError(f"{name} 缺少类别列：{miss_cols}")
        probs = dfm[class_names_full].to_numpy(dtype=float)
        if np.isnan(probs).any():
            print(f"[警告] {name} 含 NaN/Inf 概率，已修正并归一化。")
        probs = _safe_normalize_probs(probs)
        prob_matrices.append(probs)
        model_names.append(name)

    # Build the second-stage feature matrix.
    use_logits = CONFIG["USE_LOGITS"] and CONFIG["META_MODEL"].lower() in ("logistic", "ridge")
    X, feature_cols = build_feature_matrix(prob_matrices, model_names, class_names_full, use_logits, with_model_name_prefix=True)

    # Train the meta-model with subject-grouped CV and collect second-stage OOF probabilities.
    g_unique = np.unique(groups)
    folds = min(max(2, CONFIG["N_SPLITS"]), len(g_unique))
    gkf = GroupKFold(n_splits=folds)
    oof_stack = np.zeros((len(master_df), n_classes), dtype=float)
    fold_scores = []
    fold_logloss = []

    print(f"[Info] META_MODEL = {CONFIG['META_MODEL']}, folds = {folds}, use_logits = {use_logits}")
    print(f"       SCALE_LINEAR={CONFIG['SCALE_LINEAR']}, COMPOSITE_WEIGHTS={CONFIG['USE_COMPOSITE_CLASS_WEIGHTS']}")
    print(f"       RIDGE_CALIBRATE={CONFIG['RIDGE_CALIBRATE']}, RIDGE_CALIBRATE_PREFIT={CONFIG['RIDGE_CALIBRATE_PREFIT']} ({CONFIG['CALIBRATION_METHOD']}, holdout={CONFIG['CALIBRATION_HOLDOUT']})")

    for k, (tr_idx, va_idx) in enumerate(gkf.split(X=np.zeros(len(master_df)), y=y_all, groups=groups), 1):
        # Compute fold-specific sample weights from the training labels only.
        if CONFIG["USE_COMPOSITE_CLASS_WEIGHTS"]:
            y_tr_names = y_true_names_all[tr_idx]
            class_weight_dict = calculate_composite_weights_18_class(
                pd.Series(y_tr_names), le_full, TARGET_GESTURES
            )
            sw_tr = np.array([class_weight_dict[int(c)] for c in y_all[tr_idx]], dtype=float)
        else:
            sw_tr = None

        meta_name = CONFIG["META_MODEL"].lower()

        # Ridge can optionally reserve a subject-grouped holdout for calibration.
        if meta_name == "ridge" and CONFIG.get("RIDGE_CALIBRATE_PREFIT", False):
            gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG.get("CALIBRATION_HOLDOUT", 0.2),
                                    random_state=CONFIG["RANDOM_STATE"])
            inner_tr_rel, calib_rel = next(gss.split(X[tr_idx], y_all[tr_idx], groups=groups[tr_idx]))
            inner_tr_idx = tr_idx[inner_tr_rel]
            calib_idx    = tr_idx[calib_rel]

            ridge_base = make_meta_model("ridge", n_classes, CONFIG)
            _fit_with_sample_weight(ridge_base, X[inner_tr_idx], y_all[inner_tr_idx],
                                    sample_weight=(sw_tr[inner_tr_rel] if sw_tr is not None else None))

            # CalibratedClassifierCV ignores sample_weight in most versions anyway.
            est = CalibratedClassifierCV(ridge_base, method=CONFIG.get("CALIBRATION_METHOD", "sigmoid"), cv="prefit")
            est.fit(X[calib_idx], y_all[calib_idx])

        else:
            # Optional inner-fold calibration is slightly optimistic because it is not group-aware.
            if meta_name == "ridge" and CONFIG.get("RIDGE_CALIBRATE", False):
                base = make_meta_model("ridge", n_classes, CONFIG)
                _fit_with_sample_weight(base, X[tr_idx], y_all[tr_idx], sample_weight=sw_tr)
                est = CalibratedClassifierCV(base, method=CONFIG.get("CALIBRATION_METHOD", "sigmoid"), cv=3)
                est.fit(X[tr_idx], y_all[tr_idx])
            else:
                est = make_meta_model(CONFIG["META_MODEL"], n_classes, CONFIG)
                _fit_with_sample_weight(est, X[tr_idx], y_all[tr_idx], sample_weight=sw_tr)

        # Validation-set probabilities for this fold.
        proba = _proba_in_full_order(est, X[va_idx], n_classes, le_full)
        oof_stack[va_idx] = proba

        # Report both log-loss and the competition score.
        ll = log_loss(y_all[va_idx], proba, labels=np.arange(n_classes))
        y_pred_idx = np.argmax(proba, axis=1)
        y_pred_names = [class_names_full[i] for i in y_pred_idx]
        y_true_names_va = le_full.inverse_transform(y_all[va_idx])

        le9 = LabelEncoder().fit(TARGET_GESTURES + [NON_TARGET_CLASS_NAME])
        y_true_9 = le9.transform([n if n in TARGET_GESTURES else NON_TARGET_CLASS_NAME for n in y_true_names_va])
        y_pred_9 = le9.transform([n if n in TARGET_GESTURES else NON_TARGET_CLASS_NAME for n in y_pred_names])
        non_target_index = list(le9.classes_).index(NON_TARGET_CLASS_NAME)
        comp, binf1, macf1 = calculate_competition_metric_from_9_classes(y_true_9, y_pred_9, le9, non_target_index)

        print(f"[Fold {k}/{folds}] LogLoss={ll:.6f}  Comp={comp:.6f}  binF1={binf1:.6f}  macF1={macf1:.6f}  |va|={len(va_idx)}")
        fold_logloss.append(ll)
        fold_scores.append(comp)

    # Aggregate OOF metrics across all folds.
    oof_ll = log_loss(y_all, oof_stack, labels=np.arange(n_classes))
    oof_pred_idx = np.argmax(oof_stack, axis=1)
    oof_pred_names = [class_names_full[i] for i in oof_pred_idx]
    y_true_names_all_dec = le_full.inverse_transform(y_all)
    le9_all = LabelEncoder().fit(TARGET_GESTURES + [NON_TARGET_CLASS_NAME])
    y_true_9_all = le9_all.transform([n if n in TARGET_GESTURES else NON_TARGET_CLASS_NAME for n in y_true_names_all_dec])
    y_pred_9_all = le9_all.transform([n if n in TARGET_GESTURES else NON_TARGET_CLASS_NAME for n in oof_pred_names])
    non_target_index_all = list(le9_all.classes_).index(NON_TARGET_CLASS_NAME)
    oof_comp, oof_binf1, oof_macf1 = calculate_competition_metric_from_9_classes(
        y_true_9_all, y_pred_9_all, le9_all, non_target_index_all
    )

    print("\n========== OOF Summary ==========")
    print(f"OOF LogLoss : {oof_ll:.6f}")
    print(f"OOF Comp    : {oof_comp:.6f}  (binF1={oof_binf1:.6f}, macroF1={oof_macf1:.6f})")
    print(f"CV LogLoss  : {np.mean(fold_logloss):.6f} ± {np.std(fold_logloss):.6f}")
    print(f"CV Comp     : {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    print("=================================\n")

    # Optionally persist the second-stage OOF probabilities.
    if CONFIG["SAVE_OOF_STACK_CSV"]:
        out = master_df.copy()
        for i, cls in enumerate(class_names_full):
            out[cls] = oof_stack[:, i]
        out["gesture_true"] = y_true_names_all_dec
        out.to_csv(CONFIG["SAVE_OOF_STACK_CSV"], index=False)
        print(f"[Saved] 二阶段 OOF 已写入 {CONFIG['SAVE_OOF_STACK_CSV']}")

    # Refit the meta-model on the full dataset for offline/competition inference.
    if CONFIG["USE_COMPOSITE_CLASS_WEIGHTS"]:
        class_weight_full = calculate_composite_weights_18_class(
            pd.Series(y_true_names_all_dec), le_full, TARGET_GESTURES
        )
        sw_full = np.array([class_weight_full[int(c)] for c in y_all], dtype=float)
    else:
        class_weight_full = None
        sw_full = None

    meta_name = CONFIG["META_MODEL"].lower()

    if meta_name == "ridge" and CONFIG.get("RIDGE_CALIBRATE_PREFIT", False):
        # Reuse the safer subject-grouped prefit calibration path for the final artifact.
        gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG.get("CALIBRATION_HOLDOUT", 0.2),
                                random_state=CONFIG["RANDOM_STATE"])
        inner_tr_rel, calib_rel = next(gss.split(X, y_all, groups=groups))
        ridge_base = make_meta_model("ridge", n_classes, CONFIG)
        _fit_with_sample_weight(ridge_base, X[inner_tr_rel], y_all[inner_tr_rel],
                                sample_weight=(sw_full[inner_tr_rel] if sw_full is not None else None))
        est_final = CalibratedClassifierCV(ridge_base, method=CONFIG.get("CALIBRATION_METHOD", "sigmoid"), cv="prefit")
        est_final.fit(X[calib_rel], y_all[calib_rel])

        calibration_info = {
            "type": "prefit",
            "method": CONFIG.get("CALIBRATION_METHOD", "sigmoid"),
            "holdout": CONFIG.get("CALIBRATION_HOLDOUT", 0.2)
        }

    elif meta_name == "ridge" and CONFIG.get("RIDGE_CALIBRATE", False):
        base = make_meta_model("ridge", n_classes, CONFIG)
        _fit_with_sample_weight(base, X, y_all, sample_weight=sw_full)
        est_final = CalibratedClassifierCV(base, method=CONFIG.get("CALIBRATION_METHOD", "sigmoid"), cv=3)
        est_final.fit(X, y_all)
        calibration_info = {"type": "kfold", "method": CONFIG.get("CALIBRATION_METHOD", "sigmoid"), "cv": 3}

    else:
        est_final = make_meta_model(CONFIG["META_MODEL"], n_classes, CONFIG)
        _fit_with_sample_weight(est_final, X, y_all, sample_weight=sw_full)
        calibration_info = None

    print("[Info] 元模型已在全量数据上重拟合，用于线上推理。")

    # Export inference artifacts.
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    model_path = os.path.join(CONFIG["SAVE_DIR"], "meta_model.pkl")
    info_path = os.path.join(CONFIG["SAVE_DIR"], "meta_info.json")

    joblib.dump(est_final, model_path)

    if CONFIG["USE_COMPOSITE_CLASS_WEIGHTS"]:
        class_weight_export = {name: float(class_weight_full[idx]) for idx, name in enumerate(le_full.classes_)}
    else:
        class_weight_export = None

    meta_info = {
        "meta_model": CONFIG["META_MODEL"],
        "feature_columns": X.shape[1],
    }
    meta_info = {
        "meta_model": CONFIG["META_MODEL"],
        "feature_columns": build_feature_matrix(prob_matrices, model_names, class_names_full, use_logits, True)[1],
        "model_names": model_names,
        "class_names_full": class_names_full,
        "use_logits": use_logits,
        "scale_linear": CONFIG["SCALE_LINEAR"],
        "ridge_params": CONFIG["PARAMS"]["ridge"] if CONFIG["META_MODEL"].lower()=="ridge" else None,
        "ridge_calibrate": CONFIG.get("RIDGE_CALIBRATE", False),
        "ridge_calibrate_prefit": CONFIG.get("RIDGE_CALIBRATE_PREFIT", False),
        "calibration": calibration_info,
        "label_encoder_full_classes": list(le_full.classes_),
        "random_state": CONFIG["RANDOM_STATE"],
        "use_composite_class_weights": CONFIG["USE_COMPOSITE_CLASS_WEIGHTS"],
        "composite_class_weights_by_name": class_weight_export,
        "oof_metrics": {
            "logloss": float(oof_ll),
            "competition": float(oof_comp),
            "bin_f1": float(oof_binf1),
            "macro_f1": float(oof_macf1),
            "cv_logloss_mean": float(np.mean(fold_logloss)),
            "cv_logloss_std": float(np.std(fold_logloss)),
            "cv_comp_mean": float(np.mean(fold_scores)),
            "cv_comp_std": float(np.std(fold_scores)),
        }
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
    print(f"[Saved] 元模型: {model_path}")
    print(f"[Saved] 元信息: {info_path}")

    print("\n【线上推理要点】")
    print("- 线上把各基模型概率按 meta_info['feature_columns'] 的列顺序拼成特征；")
    print("- 若 use_logits=True，要先对每个概率做 logit（本配置默认 False）；")
    print("- 加载 est = joblib.load('meta_model.pkl') 后：")
    print("    * 如果模型有 predict_proba：y_proba = est.predict_proba(X_stack)")
    print("    * 否则（如未校准的 RidgeClassifier）：用 decision_function -> softmax 得到概率。")
    if calibration_info:
        print(f"- 已启用概率校准：{calibration_info}")


if __name__ == "__main__":
    main()
