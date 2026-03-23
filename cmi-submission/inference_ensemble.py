#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust stacking inference for the Kaggle submission package.

This script is intentionally strict about artifact availability and class-space
alignment. It only falls back when numerical issues appear at inference time,
for example by zero-filling missing spectrogram sources or replacing non-finite
probabilities with a uniform distribution.
"""

import os
import re
import json
import pickle
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from scipy import signal

warnings.filterwarnings("ignore")

# Hard-coded Kaggle paths expected by the published submission bundle.
PACKS_PARENT_DIR = "/kaggle/input/cmi2025-ensemble/ensemble learning"
STACK_DIR        = "/kaggle/input/cmi-link/"

# Project-local imports.
from models.multimodality import MultimodalityModel
from data_utils.data_preprocessing import (
    pad_sequences, feature_engineering, STATIC_FEATURE_COLS,
    generate_spectrogram, generate_feature_columns, add_tof_missing_flags
)
from data_utils.tof_utils import interpolate_tof

MAP_NON_TARGET = "Drink from bottle/cup"
VARIANTS       = ["full", "imu"]

# Helper utilities.
def softmax_nd(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)

def _safe_clip01(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.clip(x, eps, 1 - eps)

def _ensure(p: str, what: str):
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少 {what}: {p}")

def _list_submission_dirs(root: str) -> Dict[str, str]:
    """Return a leaderboard-tag to submission-directory mapping."""
    dmap = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"根路径不存在：{root}")
    for name in os.listdir(root):
        m = re.fullmatch(r"cmi-submission \((\d+\.\d{3})\)", name)
        if m:
            dmap[m.group(1)] = os.path.join(root, name)
    if not dmap:
        raise FileNotFoundError(f"在 {root} 下未找到任何 cmi-submission (X.XXX) 目录")
    return dmap

# Meta-model loading.
def _load_meta_for_variant(variant: str):
    mdir = os.path.join(STACK_DIR, "stack_artifacts", variant)
    model_path = os.path.join(mdir, "meta_model.pkl")
    info_path  = os.path.join(mdir, "meta_info.json")
    _ensure(model_path, f"[{variant}] 元模型")
    _ensure(info_path,  f"[{variant}] 元信息")
    import joblib
    est = joblib.load(model_path)
    with open(info_path, "r", encoding="utf-8") as f:
        meta_info = json.load(f)
    for k in ["feature_columns", "class_names_full"]:
        if k not in meta_info:
            raise ValueError(f"[{variant}] meta_info.json 缺少字段: {k} ({info_path})")
    print(f"✅ [{variant}] 元模型来自: {mdir}")
    return est, meta_info


def _list_bases_from_meta(meta_info: Dict) -> List[str]:
    """Extract unique base-model prefixes from ``meta_info['feature_columns']``."""
    bases = []
    for col in meta_info["feature_columns"]:
        prefix = col.split("::", 1)[0]
        if prefix not in bases:
            bases.append(prefix)
    if not bases:
        raise ValueError("meta_info['feature_columns'] 为空，无法确定基模型列表")
    return bases

# Map stacking feature prefixes back to submission packs and variants.
def _parse_base_key(base_key: str) -> Tuple[str, str]:
    m = re.fullmatch(r"(\d+\.\d{3})_(imu|full)\.csv", base_key)
    if not m:
        raise ValueError(f"无法解析基模型前缀: {base_key}（期望格式 '0.849_imu.csv'）")
    return m.group(1), m.group(2)


def _weights_dir_for_key(base_key: str) -> Tuple[str, str]:
    lb, variant = _parse_base_key(base_key)
    dmap = _list_submission_dirs(PACKS_PARENT_DIR)
    if lb not in dmap:
        raise FileNotFoundError(f"未找到提交包目录：cmi-submission ({lb})")
    pack_root = dmap[lb]
    wdir = os.path.join(pack_root, "weights")
    if not os.path.isdir(wdir):
        raise FileNotFoundError(f"权重目录不存在：{wdir}")
    return wdir, pack_root

# Base-model class metadata lives next to the fold weights.
def _load_base_classes(weights_dir: str, variant: str) -> List[str]:
    cand = [
        os.path.join(weights_dir, f"label_encoder_{variant}.pkl"),
        os.path.join(weights_dir, f"classes_{variant}.json"),
        os.path.join(weights_dir, "label_encoder.pkl"),
        os.path.join(weights_dir, "classes.json"),
    ]
    for p in cand:
        if not os.path.exists(p):
            continue
        if p.endswith(".pkl"):
            with open(p, "rb") as f:
                le = pickle.load(f)
            cls = list(le.classes_)
            if not cls:
                raise ValueError(f"[{p}] classes_ 为空")
            return [str(x) for x in cls]
        else:
            with open(p, "r", encoding="utf-8") as f:
                arr = json.load(f)
            if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr) or not arr:
                raise ValueError(f"[{p}] 必须是非空字符串数组")
            return arr
    raise FileNotFoundError(
        f"[{weights_dir}] 未找到类名文件（需要 label_encoder_{variant}.pkl 或 classes_{variant}.json；"
        f"也可提供 label_encoder.pkl / classes.json）"
    )


def _load_label_map(weights_dir: str, variant: str) -> Dict[str, str]:
    for name in [f"label_map_{variant}.json", "label_map.json"]:
        p = os.path.join(weights_dir, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                raise ValueError(f"[{p}] 需为字典 {基类: 元类}")
            return {str(k): str(v) for k, v in d.items()}
    return {}

# Load all fold artifacts for one base model.
def _load_models_from_weights_dir(device, weights_dir: str, variant: str) -> List[Tuple]:
    files = os.listdir(weights_dir)

    # Discover whichever fold numbers are present instead of assuming 1..K.
    fold_nums = set()
    pat_model = re.compile(rf"^model_fold_?(\d+)_({variant})\.pth$")
    pat_model2 = re.compile(rf"^model_fold(\d+)_({variant})\.pth$")
    for f in files:
        m = pat_model.match(f) or pat_model2.match(f)
        if m:
            fold_nums.add(int(m.group(1)))
    fold_nums = sorted(list(fold_nums))
    if not fold_nums:
        raise FileNotFoundError(f"[{weights_dir}] 未发现任何 model_foldK_{variant}.pth")

    def _pick(tag: str, k: int, must_ext: str):
        cand = [f for f in files if f.endswith(must_ext) and re.fullmatch(rf"{tag}_fold_?{k}_{variant}\{must_ext}", f)]
        if not cand:
            cand = [f for f in files if f.endswith(must_ext) and re.fullmatch(rf"{tag}_fold{k}_{variant}\{must_ext}", f)]
        if not cand:
            raise FileNotFoundError(f"[{weights_dir}] 缺少 {tag}_fold{k}_{variant}{must_ext}")
        return os.path.join(weights_dir, cand[0])

    folds = []
    for k in fold_nums:
        model_path = None
        for f in files:
            if pat_model.match(f) or pat_model2.match(f):
                m = (pat_model.match(f) or pat_model2.match(f))
                if int(m.group(1)) == k:
                    model_path = os.path.join(weights_dir, f)
                    break
        if model_path is None:
            raise FileNotFoundError(f"[{weights_dir}] 未找到 model_fold{k}_{variant}.pth")

        scaler_path      = _pick("scaler",      k, ".pkl")
        spec_stats_path  = _pick("spec_stats",  k, ".pkl")
        spec_params_path = _pick("spec_params", k, ".pkl")

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(spec_stats_path, "rb") as f:
            spec_stats = pickle.load(f)
        with open(spec_params_path, "rb") as f:
            spec_params = pickle.load(f)

        ckpt = torch.load(model_path, map_location=device)
        if 'model_cfg' not in ckpt:
            raise ValueError(f"[{model_path}] 缺少 'model_cfg'")
        model_cfg = {k2: v2 for k2, v2 in ckpt['model_cfg'].items() if k2 != 'type'}
        state_dict = ckpt['state_dict']
        model = MultimodalityModel(**model_cfg)

        # torch.compile checkpoints prefix parameters with ``_orig_mod.``.
        is_compiled = any(key.startswith('_orig_mod.') for key in state_dict.keys())
        if is_compiled:
            from collections import OrderedDict
            state_dict = OrderedDict((k.replace('_orig_mod.', '', 1), v) for k, v in state_dict.items())
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Recover the expected sequence length from whichever config field exists.
        seq_len = None
        if 'sequence_length' in model_cfg: seq_len = model_cfg['sequence_length']
        elif 'seq_len' in model_cfg:      seq_len = model_cfg['seq_len']
        elif 'tof_branch_cfg' in model_cfg and 'seq_len' in model_cfg['tof_branch_cfg']:
            seq_len = model_cfg['tof_branch_cfg']['seq_len']

        # The saved spectrogram length must agree with the model's sequence length.
        try:
            max_len_from_spec = spec_params.get('max_length', None)
        except Exception:
            max_len_from_spec = None
        if (seq_len is not None) and (max_len_from_spec is not None) and (max_len_from_spec != seq_len):
            raise ValueError(f"[{weights_dir}] fold {k} 长度不一致: model seq_len={seq_len}, spec.max_length={max_len_from_spec}")

        folds.append((model, scaler, spec_stats, spec_params, seq_len))
    return folds

# Variant detection and per-sequence preprocessing.
def _decide_variant(seq_df: "pd.DataFrame") -> str:
    thm_cols = [c for c in seq_df.columns if c.startswith("thm_")]
    tof_cols = [c for c in seq_df.columns if c.startswith("tof_")]
    if not thm_cols and not tof_cols:
        return "imu"
    thm_all_missing = not seq_df[thm_cols].notna().values.any() if thm_cols else True
    tof_all_missing = not seq_df[tof_cols].notna().values.any() if tof_cols else True
    return "imu" if thm_all_missing and tof_all_missing else "full"


def preprocess_single_sequence(seq_pl: pl.DataFrame, demog_pl: pl.DataFrame):
    seq_df = seq_pl.to_pandas()
    if not demog_pl.is_empty():
        seq_df = seq_df.merge(demog_pl.to_pandas(), on="subject", how="left")

    variant = _decide_variant(seq_df)
    if variant != "imu":
        seq_df = add_tof_missing_flags(seq_df)
        seq_df = interpolate_tof(seq_df)

    processed_df, feature_cols = feature_engineering(seq_df)

    existing_static_cols = [c for c in STATIC_FEATURE_COLS if c in processed_df.columns]
    for col in existing_static_cols:
        if col not in feature_cols:
            feature_cols.append(col)

    dynamic_missing_flags = [c for c in processed_df.columns if c.endswith('_missing')]
    for col in dynamic_missing_flags:
        if col not in feature_cols:
            feature_cols.append(col)

    if variant == "imu":
        feature_cols = [c for c in feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]

    final_features_df = processed_df.sort_values("sequence_counter")
    final_features_df = final_features_df[[c for c in feature_cols if c in final_features_df.columns]]
    return variant, final_features_df

# Safe spectrogram helpers.
_BASE_SPEC_ORDER = ['linear_acc_x','linear_acc_y','linear_acc_z','angular_vel_x','angular_vel_y','angular_vel_z']

def _make_zero_spec(seq_len: int, spec_params: dict) -> np.ndarray:
    zero_sig = np.zeros(seq_len, dtype=np.float32)
    spec = generate_spectrogram(zero_sig, fs=spec_params['fs'], nperseg=spec_params['nperseg'],
                                noverlap=spec_params['noverlap'], max_length=seq_len)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return spec

# Run one fold using the scaler and spectrogram settings saved with that fold.
def _predict_one_fold(
    fold_entry: Tuple, features_df: pd.DataFrame, device: torch.device
) -> np.ndarray:
    model, scaler, spec_stats, spec_params, sequence_length = fold_entry

    # Align to the training feature order and synthesize missing columns as NaN.
    if hasattr(scaler, 'feature_names_'):
        needed_cols = list(scaler.feature_names_)
        missing_cols = [c for c in needed_cols if c not in features_df.columns]
        if missing_cols:
            for c in missing_cols:
                features_df[c] = np.nan
        features_df = features_df[needed_cols]

    # Sanitize immediately after scaling so every downstream branch sees finite data.
    X_scaled_unpadded = scaler.transform(features_df).astype(np.float32)
    X_scaled_unpadded = np.nan_to_num(X_scaled_unpadded, nan=0.0, posinf=0.0, neginf=0.0)
    scaled_feature_names = scaler.get_feature_names_out().tolist()

    # Re-split the flattened frame into the modality groups expected by the model.
    static_cols = [c for c in scaled_feature_names if c in STATIC_FEATURE_COLS or c.endswith('_missing')]
    thm_cols, tof_cols = generate_feature_columns(scaled_feature_names)
    thm_cols = [c for c in thm_cols if c in scaled_feature_names]
    tof_cols = [c for c in tof_cols if c in scaled_feature_names]
    imu_cols = [c for c in scaled_feature_names if c not in static_cols + tof_cols + thm_cols]

    # Keep spectrogram channels in the same fixed order used during training.
    spec_source_cols = [c for c in _BASE_SPEC_ORDER]

    static_idx = [scaled_feature_names.index(c) for c in static_cols]
    static_arr = X_scaled_unpadded[0:1, static_idx]

    tof_idx    = [scaled_feature_names.index(c) for c in tof_cols]
    thm_idx    = [scaled_feature_names.index(c) for c in thm_cols]
    imu_idx    = [scaled_feature_names.index(c) for c in imu_cols]

    tof_arr, thm_arr, imu_arr = (
        X_scaled_unpadded[:, tof_idx] if len(tof_idx) else np.zeros((X_scaled_unpadded.shape[0], 0), dtype=np.float32),
        X_scaled_unpadded[:, thm_idx] if len(thm_idx) else np.zeros((X_scaled_unpadded.shape[0], 0), dtype=np.float32),
        X_scaled_unpadded[:, imu_idx] if len(imu_idx) else np.zeros((X_scaled_unpadded.shape[0], 0), dtype=np.float32),
    )

    # ToF masking is sensor-level, not pixel-level, so one flag gates 64 values.
    tof_sensor_ids = []
    for c in scaled_feature_names:
        if c.startswith('tof_') and c.endswith('_missing') and c.count('_') == 2:
            try:
                sid = int(c.split('_')[1])
                if sid not in tof_sensor_ids:
                    tof_sensor_ids.append(sid)
            except Exception:
                pass
    if not tof_sensor_ids:
        seen = set()
        for c in scaled_feature_names:
            if c.startswith('tof_') and '_v' in c and not c.endswith('_missing'):
                try:
                    seen.add(int(c.split('_')[1]))
                except Exception:
                    pass
        tof_sensor_ids = sorted(list(seen))

    ch_mask_vals = []
    for sid in sorted(tof_sensor_ids):
        flag = f"tof_{sid}_missing"
        if flag in scaled_feature_names:
            flag_idx = scaled_feature_names.index(flag)
            valid = 1.0 - float(X_scaled_unpadded[0, flag_idx])
        else:
            valid = 1.0
        ch_mask_vals.append(valid)
    tof_channel_mask_arr = np.array(ch_mask_vals, dtype=np.float32)[np.newaxis, :] if ch_mask_vals else np.ones((1,0), dtype=np.float32)

    # Pad to the exact sequence length saved with the fold artifacts.
    X_imu_pad, imu_mask = pad_sequences([imu_arr], max_length=sequence_length)
    X_thm_pad, _        = pad_sequences([thm_arr], max_length=sequence_length)
    X_tof_pad, _        = pad_sequences([tof_arr], max_length=sequence_length)

    # Missing channels become zero signals; any non-finite STFT values are scrubbed.
    sequence_spectrograms = []
    spec_mean, spec_std = spec_stats['mean'], spec_stats['std']

    # Precompute one zero spectrogram so every missing source uses the same shape.
    zero_spec = _make_zero_spec(sequence_length, spec_params)

    for name in spec_source_cols:
        if name in scaled_feature_names:
            idx = scaled_feature_names.index(name)
            signal_1d = X_scaled_unpadded[:, idx]
            signal_1d = np.nan_to_num(signal_1d, nan=0.0, posinf=0.0, neginf=0.0)
            if len(signal_1d) >= sequence_length:
                padded_signal = signal_1d[-sequence_length:]
            else:
                padded_signal = np.pad(signal_1d, (sequence_length - len(signal_1d), 0), 'constant')
            spec = generate_spectrogram(
                padded_signal,
                fs=spec_params['fs'],
                nperseg=spec_params['nperseg'],
                noverlap=spec_params['noverlap'],
                max_length=sequence_length,
            )
            spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            spec = zero_spec
        spec_norm = ((spec - spec_mean) / (spec_std + 1e-6)).astype(np.float32)
        spec_norm = np.nan_to_num(spec_norm, nan=0.0, posinf=0.0, neginf=0.0)
        sequence_spectrograms.append(spec_norm)

    X_spec = np.stack(sequence_spectrograms, axis=0).astype(np.float32)[np.newaxis, ...]

    # 5) Tensor
    xb_imu = torch.from_numpy(X_imu_pad).to(DEVICE)
    xb_thm = torch.from_numpy(X_thm_pad).to(DEVICE)
    xb_tof = torch.from_numpy(X_tof_pad).to(DEVICE)
    xb_spec = torch.from_numpy(X_spec).to(DEVICE)
    xb_static = torch.from_numpy(static_arr).to(DEVICE)
    xb_mask = torch.from_numpy(imu_mask).to(DEVICE)

    # THM mask
    thm_sensor_ids_inf = []
    for c in scaled_feature_names:
        if c.startswith('thm_') and c.endswith('_missing'):
            try:
                thm_sensor_ids_inf.append(int(c.split('_')[1]))
            except Exception:
                pass
    thm_sensor_ids_inf = sorted(list(set(thm_sensor_ids_inf)))
    thm_mask_vals = []
    for sid in thm_sensor_ids_inf:
        flag = f"thm_{sid}_missing"
        if flag in scaled_feature_names:
            idx_flag = scaled_feature_names.index(flag)
            thm_mask_vals.append(1.0 - float(X_scaled_unpadded[0, idx_flag]))
        else:
            thm_mask_vals.append(1.0)
    thm_channel_mask_arr = np.array(thm_mask_vals, dtype=np.float32)[np.newaxis, :] if len(thm_mask_vals) > 0 else np.ones((1,0), dtype=np.float32)
    xb_thm_ch_mask = torch.from_numpy(thm_channel_mask_arr).to(DEVICE)

    # IMU rot-only mask
    imu_feature_names_inf = [c for c in scaled_feature_names if c not in static_cols + tof_cols + thm_cols]
    rot_fields_inf = ['rot_w','rot_x','rot_y','rot_z']
    imu_rot_mask = np.ones((1, len(imu_feature_names_inf)), dtype=np.float32)
    if 'rot_missing' in scaled_feature_names:
        idx_rm = scaled_feature_names.index('rot_missing')
        if float(X_scaled_unpadded[0, idx_rm]) == 1.0:
            for i, name in enumerate(imu_feature_names_inf):
                if name in rot_fields_inf:
                    imu_rot_mask[0, i] = 0.0
    xb_imu_ch_mask = torch.from_numpy(imu_rot_mask).to(DEVICE)

    model = model.to(DEVICE)
    # Forward pass.
    with torch.no_grad():
        logits = model(
            xb_imu, xb_thm, xb_tof, xb_spec, xb_static,
            mask=xb_mask,
            tof_channel_mask=torch.from_numpy(tof_channel_mask_arr).to(DEVICE),
            thm_channel_mask=xb_thm_ch_mask,
            imu_channel_mask=xb_imu_ch_mask
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Never propagate NaN/Inf probabilities into the stacking model.
    if not np.isfinite(probs).all():
        C = probs.shape[1]
        probs = np.full((1, C), 1.0 / C, dtype=np.float32)

    model = model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs  # [1, C_base]


def _avg_proba_for_base(fold_entries: List[Tuple], features_df: pd.DataFrame) -> np.ndarray:
    acc = None
    for fe in fold_entries:
        p = _predict_one_fold(fe, features_df, DEVICE)  # [1, C_base]
        acc = p if acc is None else acc + p
    avg = acc / len(fold_entries)
    avg = np.nan_to_num(avg, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(avg.sum())
    if s > 0:
        avg = avg / s
    return avg[0]  # [C_base]


def _map_base_to_meta_proba(
    proba_base: np.ndarray,
    base_classes: List[str],
    meta_classes: List[str],
    base_to_meta_map: Dict[str, str]
) -> np.ndarray:
    meta_idx = {name: i for i, name in enumerate(meta_classes)}
    out = np.zeros((len(meta_classes),), dtype=np.float32)

    for j, bname in enumerate(base_classes):
        mapped = base_to_meta_map.get(bname, bname)
        if mapped not in meta_idx:
            raise ValueError(
                f"类名映射失败：基类 '{bname}'（映射为 '{mapped}'）不在元模型类空间中。"
            )
        out[meta_idx[mapped]] = float(proba_base[j])

    s = out.sum()
    if s > 0:
        out = out / s
    return out

# Resource initialization stays strict: missing artifacts should fail loudly.
print("🔧  Initialising inference resources …")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

RESOURCES: Dict[str, Dict] = {}
META: Dict[str, Dict] = {}  # {variant: {"est":..., "info":...}}

# Load meta-model artifacts only from STACK_DIR.
for v in VARIANTS:
    est, meta_info = _load_meta_for_variant(v)
    META[v] = {"est": est, "info": meta_info}
    print(f"✅ [{v}] meta loaded with {len(_list_bases_from_meta(meta_info))} base(s).")

# Meta-info decides which base models to load across submission packs.
packs = _list_submission_dirs(PACKS_PARENT_DIR)

for v in VARIANTS:
    meta_info = META[v]["info"]
    base_keys = _list_bases_from_meta(meta_info)

    bases: Dict[str, Dict] = {}
    for bk in base_keys:
        lb, variant_b = _parse_base_key(bk)
        if variant_b != v:
            continue

        weights_dir, pack_root = _weights_dir_for_key(bk)
        base_classes = _load_base_classes(weights_dir, v)
        label_map    = _load_label_map(weights_dir, v)
        fold_entries = _load_models_from_weights_dir(DEVICE, weights_dir, v)

        bases[bk] = {
            "fold_entries": fold_entries,
            "base_classes": base_classes,
            "label_map": label_map,
            "weights_dir": weights_dir,
        }
        print(f"  • [{v}] {bk} -> {weights_dir} (folds={len(fold_entries)})")

    if not bases:
        raise RuntimeError(f"[{v}] 未找到任何基模型，请检查 meta_info['feature_columns'] 与目录结构。")

    RESOURCES[v] = {
        "bases": bases,
        "meta": META[v],
    }

print("✅  Resource initialization complete. Ready for inference.")

# Strict stacking prediction.

def _predict_with_stacking(variant: str, features_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    res_v = RESOURCES[variant]
    est = res_v["meta"]["est"]
    meta_info = res_v["meta"]["info"]
    meta_classes = meta_info["class_names_full"]

    # Collect one probability vector per base model in meta-feature order.
    probs_by_base: Dict[str, np.ndarray] = {}
    for col in meta_info["feature_columns"]:
        bk = col.split("::", 1)[0]
        if bk in probs_by_base:
            continue
        if bk not in res_v["bases"]:
            continue
        base_entry = res_v["bases"][bk]
        p_base = _avg_proba_for_base(base_entry["fold_entries"], features_df)  # [C_base]
        p_meta = _map_base_to_meta_proba(p_base, base_entry["base_classes"], meta_classes, base_entry["label_map"])
        probs_by_base[bk] = p_meta

    # Rebuild the exact stacking feature layout used by the trained meta-model.
    x_stack = np.zeros((1, len(meta_info["feature_columns"])), dtype=np.float32)
    cls_to_idx = {c: i for i, c in enumerate(meta_classes)}
    for j, col in enumerate(meta_info["feature_columns"]):
        prefix, cls = col.split("::", 1)
        if prefix not in probs_by_base:
            x_stack[0, j] = 0.0
        else:
            x_stack[0, j] = float(probs_by_base[prefix][cls_to_idx[cls]])

    # Final guard before handing features to scikit-learn.
    x_stack = np.nan_to_num(x_stack, nan=1e-6, posinf=1e-6, neginf=1e-6)

    # Meta-model probabilities.
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(x_stack)
        if isinstance(proba, list):
            proba = np.column_stack([p[:, 1] if p.ndim == 2 else p for p in proba])
    else:
        if hasattr(est, "decision_function"):
            df = est.decision_function(x_stack)
            if df.ndim == 1:
                df = np.stack([-df, df], axis=1)
            proba = softmax_nd(df)
        else:
            raise RuntimeError("元模型既无 predict_proba 也无 decision_function。")

    proba = _safe_clip01(np.asarray(proba))
    proba = np.nan_to_num(proba, nan=1e-9, posinf=1.0, neginf=0.0)
    # Normalize again to protect against downstream numeric drift.
    s = proba.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    proba = proba / s
    return proba, meta_classes


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    if sequence.is_empty():
        return MAP_NON_TARGET

    variant, features_df = preprocess_single_sequence(sequence, demographics)
    proba, class_names = _predict_with_stacking(variant, features_df)
    pred_idx = int(np.argmax(proba, axis=1)[0])
    pred_name = class_names[pred_idx]
    if pred_name.strip().lower() == "other":
        pred_name = "Drink from bottle/cup"
    return pred_name


# Launch the Kaggle evaluation server.
if __name__ == "__main__":
    import kaggle_evaluation.cmi_inference_server as kis
    print("🚀 Starting CMIInferenceServer …")
    inference_server = kis.CMIInferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        # Local testing gateway.
        os.chdir("/kaggle/working")
        inference_server.run_local_gateway(
            data_paths=(
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
            )
        )
