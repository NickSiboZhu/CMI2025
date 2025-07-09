#!/usr/bin/env python3

import os
import pickle
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import polars as pl

warnings.filterwarnings("ignore")

# ------------------ 路径常量 ------------------
BASE_DIR   = os.path.dirname(__file__)
WEIGHT_DIR = os.path.join(BASE_DIR, "weights")

# ------------------ 自定义模块 ------------------
from models.cnn import Simple1DCNN, GestureDataset
from data_utils.data_preprocessing import pad_sequences
from data_utils.tof_utils import interpolate_tof

# ------------------ 全局资源加载 ------------------
# We support two variants: "full" (uses THM/TOF sensors) and "imu" (IMU-only).
# Each variant has its own scaler and model weight files.

MAP_NON_TARGET = "Drink from bottle/cup"
SEQ_LEN        = 100          # 与训练保持一致

def _load_preprocessing_objects(variant: str):
    """Load label encoder and (optionally) a scaler for the given variant.

    A scaler file is no longer required because we load per-fold scalers later.
    We attempt to load one for backward compatibility but simply return None if it
    does not exist.
    """
    le_path = os.path.join(WEIGHT_DIR, f"label_encoder_{variant}.pkl")
    if not os.path.exists(le_path):
        # Fallback to generic file name (old submissions)
        le_path = os.path.join(WEIGHT_DIR, "label_encoder.pkl")

    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # Try to load an accompanying scaler but don't fail if it's missing
    possible_scalers = [
        os.path.join(WEIGHT_DIR, f"scaler_{variant}.pkl"),
        os.path.join(WEIGHT_DIR, "scaler.pkl"),
    ]
    scaler = None
    for path in possible_scalers:
        if os.path.exists(path):
            with open(path, "rb") as f:
                scaler = pickle.load(f)
            break

    return le, scaler

def _load_models(device, num_classes, variant: str) -> List[Tuple[torch.nn.Module, "StandardScaler"]]:
    """Load models together with their matching scaler.

    Returns a list of (model, scaler) tuples so that each model gets the
    exact feature scaling it was trained with (Option B). For single-model
    variants we still return a one-element list.
    """
    pairs = []

    # 搜索 5-fold 文件
    fold_paths = [
        os.path.join(WEIGHT_DIR, f"model_fold_{i}_{variant}.pth") for i in range(1, 6)
    ]
    fold_paths = [p for p in fold_paths if os.path.exists(p)]

    if fold_paths:
        print(f"🧩  [{variant}] Detected {len(fold_paths)} fold models → ensemble")
        for p in fold_paths:
            basename = os.path.basename(p)
            # Derive fold number to locate matching scaler
            # Pattern: model_fold_{i}_{variant}.pth
            parts = basename.split("_")
            try:
                fold_num = int(parts[2])  # parts: ['model','fold','{i}','{variant}.pth']
            except (IndexError, ValueError):
                raise ValueError(f"Unexpected model filename format: {basename}")

            scaler_path = os.path.join(WEIGHT_DIR, f"scaler_fold_{fold_num}_{variant}.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Expected scaler file '{scaler_path}' for model '{basename}' not found.")

            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            in_channels = scaler.mean_.shape[0]
            m = Simple1DCNN(in_channels, num_classes, SEQ_LEN)
            m.load_state_dict(torch.load(p, map_location=device))
            m.to(device).eval()

            pairs.append((m, scaler))
    else:
        single = os.path.join(WEIGHT_DIR, f"best_model_{variant}.pth")
        if not os.path.exists(single):
            raise FileNotFoundError(
                f"No model weights found for variant '{variant}' in weights/"
            )
        print(f"🧩  [{variant}] Using single model {os.path.basename(single)}")

        # Load single scaler
        scaler_path = os.path.join(WEIGHT_DIR, f"scaler_{variant}.pkl")
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(WEIGHT_DIR, "scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        in_channels = scaler.mean_.shape[0]
        m = Simple1DCNN(in_channels, num_classes, SEQ_LEN)
        m.load_state_dict(torch.load(single, map_location=device))
        m.to(device).eval()

        pairs.append((m, scaler))

    return pairs

print("🔧  Initialising inference resources …")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANTS = ["full", "imu"]
RESOURCES = {}

for v in VARIANTS:
    le, _ = _load_preprocessing_objects(v)  # scaler is optional / unused here
    num_classes = len(le.classes_)
    model_scaler_pairs = _load_models(DEVICE, num_classes, v)

    # Use the first scaler just to extract in_channels for bookkeeping
    in_channels = model_scaler_pairs[0][1].mean_.shape[0]

    RESOURCES[v] = {
        "label_encoder": le,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "model_scaler_pairs": model_scaler_pairs,
    }

print("✅  Models / Scalers / LabelEncoders for all variants loaded. Ready.")


# ------------------ 单序列预处理 ------------------
def _decide_variant(seq_df: "pd.DataFrame") -> str:
    """
    Return 'imu' if all rows in THM OR all rows in TOF columns are NaN/-1, else 'full'.
    
    Logic: The hidden dataset has all columns, but some sequences have all null/-1 values
    in either THM or TOF sensor columns. If either sensor type has NO valid data across
    all rows in the sequence, use the IMU-only model.
    """
    thm_cols = [c for c in seq_df.columns if c.startswith("thm_")]
    tof_cols = [c for c in seq_df.columns if c.startswith("tof_")]

    # If no THM/TOF columns exist at all, use IMU (defensive fallback)
    if not thm_cols and not tof_cols:
        return "imu"
    
    # Check if ALL rows in THM columns are null/-1
    thm_all_missing = True
    if thm_cols:
        thm_df = seq_df[thm_cols].replace(-1.0, np.nan)
        # If ANY cell in THM columns has valid data, then THM is not all missing
        thm_all_missing = not thm_df.notna().values.any()
    
    # Check if ALL rows in TOF columns are null/-1  
    tof_all_missing = True
    if tof_cols:
        tof_df = seq_df[tof_cols].replace(-1.0, np.nan)
        # If ANY cell in TOF columns has valid data, then TOF is not all missing
        tof_all_missing = not tof_df.notna().values.any()
    
    # Use IMU model if either sensor type is completely missing across all rows
    if (thm_cols and thm_all_missing) or (tof_cols and tof_all_missing):
        return "imu"
        
    return "full"  # Both sensor types have at least some valid data

def preprocess_single_sequence(seq_pl: pl.DataFrame, demog_pl: pl.DataFrame):
    """
    Convert a single sequence to a DataFrame suitable for the chosen variant.
    Padding is now handled AFTER scaling in the predict function.
    Returns (variant, pd.DataFrame[L, F])
    """

    seq_df = seq_pl.to_pandas()
    if not demog_pl.is_empty():
        seq_df = seq_df.merge(demog_pl.to_pandas(), on="subject", how="left")

    variant = _decide_variant(seq_df)

    meta_cols = [
        "row_id", "sequence_id", "sequence_type", "sequence_counter",
        "subject", "orientation", "behavior", "phase",
    ]

    feat_cols = [c for c in seq_df.columns if c not in meta_cols]

    if variant == "imu":
        feat_cols = [c for c in feat_cols if not (c.startswith("thm_") or c.startswith("tof_"))]

    # seq_df[feat_cols] = seq_df[feat_cols].replace(-1.0, np.nan)

    # 2-D interpolation for TOF sensor grids (per row) – skip if IMU variant
    if variant != "imu":
        seq_df = interpolate_tof(seq_df)

    # Ensure chronological order before temporal interpolation
    seq_df = seq_df.sort_values("sequence_counter")

    # Linear interpolation forward/backward along the time axis
    seq_df[feat_cols] = seq_df[feat_cols].interpolate(method="linear", limit_direction="both")

    # Fallback to median for any column still containing NaN, then 0 as a last resort
    # Note: Using the median of the single sequence might be noisy.
    # A more robust approach would be to load pre-computed medians from the training set.
    col_medians = seq_df[feat_cols].median()
    seq_df[feat_cols] = seq_df[feat_cols].fillna(col_medians)
    seq_df[feat_cols] = seq_df[feat_cols].fillna(0)
    
    # ---- Return the feature DataFrame (unscaled, unpadded)
    # The scaler requires a DataFrame or a NumPy array with the correct feature order.
    # Slicing with feat_cols ensures the column order is correct.
    final_features_df = seq_df.sort_values("sequence_counter")[feat_cols]

    return variant, final_features_df


# ------------------ 预测逻辑 ------------------
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Entry point that Kaggle calls for each sequence.
    """
    if sequence.is_empty():
        return MAP_NON_TARGET

    # 1. Preprocess sequence to get an unpadded feature DataFrame
    variant, features_df = preprocess_single_sequence(sequence, demographics)

    res      = RESOURCES[variant]
    le       = res["label_encoder"]
    pairs    = res["model_scaler_pairs"]  # List[(model, scaler)]
    num_cls  = res["num_classes"]

    with torch.no_grad():
        probs_sum = np.zeros((1, num_cls))
        
        # 2. Loop through each model and its corresponding scaler
        for model, scaler in pairs:
            # 3. Apply scaling to the unpadded 2D DataFrame
            X_scaled_unpadded = scaler.transform(features_df)

            # 4. ✨ CORRECTED PADDING CALL ✨
            # Apply padding AFTER scaling, using the correct function signature.
            # Your function takes a list of sequences and max_length.
            X_padded = pad_sequences(
                [X_scaled_unpadded], 
                max_length=SEQ_LEN
            )

            # 5. Convert to tensor and predict
            # The shape from pad_sequences is (1, SEQ_LEN, num_features)
            xb = torch.from_numpy(X_padded.astype(np.float32)).to(DEVICE)
            
            probs = torch.softmax(model(xb), dim=1).cpu().numpy()
            probs_sum += probs

        # Average the probabilities for the ensemble
        probs = probs_sum / len(pairs)

    pred_idx = int(np.argmax(probs, axis=1)[0])
    label    = le.inverse_transform([pred_idx])[0]
    return label if label in le.classes_ else MAP_NON_TARGET


# ------------------ 启动评测服务器 ------------------
if __name__ == "__main__":
    import kaggle_evaluation.cmi_inference_server as kis
    print("🚀 Starting CMIInferenceServer …")
    inference_server = kis.CMIInferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            data_paths=(
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
            )
        )