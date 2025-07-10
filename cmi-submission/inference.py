#!/usr/bin/env python3
"""
CMI – Detect Behavior with Sensor Data
--------------------------------------
推理脚本（Code Competition 版本）

目录结构::
    cmi-submission/
        data/
            __init__.py
            data_preprocessing.py   ← pad_sequences 在此
        models/
            __init__.py
            cnn.py                  ← CNN1D & GestureDataset
        weights/
            best_model.pth  或  model_fold_1.pth …
            label_encoder.pkl
            scaler.pkl
        inference.py                ← 当前文件
"""

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
from models.multimodality import MultimodalityModel
from data_utils.data_preprocessing import pad_sequences, STATIC_FEATURE_COLS, preprocess_single_sequence_multimodal
from data_utils.tof_utils import interpolate_tof

# --- NEW: helper utilities for scaler handling ---
from typing import Union
from sklearn.preprocessing import StandardScaler

def _get_tof_in_channels(tof_scaler: Union[StandardScaler, list, None]) -> int:
    """Return the feature dimension (channels) handled by a TOF scaler.

    The training pipeline stores TOF scalers as a *list* of 5 per-sensor
    StandardScaler objects (one for each 8×8 grid).  Old checkpoints store a
    single StandardScaler covering all 320 pixels.  This helper unifies both
    cases.
    """
    if tof_scaler is None:
        return 0
    # New format → list[StandardScaler] (one per sensor, 64 pixels each)
    if isinstance(tof_scaler, list):
        return len(tof_scaler) * 64
    # Legacy format → single StandardScaler for all pixels
    return tof_scaler.mean_.shape[0]


def _transform_tof(flat_tof: np.ndarray, tof_scaler: Union[StandardScaler, list, None]) -> np.ndarray:
    """Scale a 2-D (N, 320) TOF array with either a single or a list of scalers."""
    if tof_scaler is None or flat_tof.shape[1] == 0:
        return flat_tof  # nothing to do

    if isinstance(tof_scaler, list):
        # Expect len == 5, each scaler handles 64 columns
        n_sensors = len(tof_scaler)
        assert flat_tof.shape[1] == n_sensors * 64, (
            f"Expected {n_sensors*64} TOF channels, got {flat_tof.shape[1]}")
        scaled_parts = []
        for i, scaler in enumerate(tof_scaler):
            start = i * 64
            end = start + 64
            scaled_parts.append(scaler.transform(flat_tof[:, start:end]))
        return np.concatenate(scaled_parts, axis=1)
    else:
        # Single scaler covering all channels
        return tof_scaler.transform(flat_tof)


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

def _load_models(device, num_classes, variant: str):
    """Load multimodal Fusion models together with their matching scalers.

    Returns a list of (model, non_tof_scaler, tof_scaler, static_scaler) tuples. 
    Works for both 5-fold ensembles and single-model submissions.
    """
    pairs = []

    # Look for 5-fold models first
    fold_paths = [os.path.join(WEIGHT_DIR, f"model_fold_{i}_{variant}.pth") for i in range(1, 6)]
    fold_paths = [p for p in fold_paths if os.path.exists(p)]

    if fold_paths:
        print(f"🧩  [{variant}] Detected {len(fold_paths)} fold models → ensemble")
        for p in fold_paths:
            basename = os.path.basename(p)
            # Extract fold index
            parts = basename.split("_")
            fold_num = int(parts[2])  # e.g. model_fold_3_full.pth → 3

            non_tof_scaler_path = os.path.join(WEIGHT_DIR, f"non_tof_scaler_fold_{fold_num}_{variant}.pkl")
            tof_scaler_path     = os.path.join(WEIGHT_DIR, f"tof_scaler_fold_{fold_num}_{variant}.pkl")
            static_scaler_path  = os.path.join(WEIGHT_DIR, f"static_scaler_fold_{fold_num}_{variant}.pkl")

            # Check if all required scalers exist
            required_scalers = [non_tof_scaler_path, static_scaler_path]
            if variant == "full":
                required_scalers.append(tof_scaler_path)
            
            missing_scalers = [p for p in required_scalers if not os.path.exists(p)]
            if missing_scalers:
                raise FileNotFoundError(f"Missing scaler(s) for fold {fold_num} ({variant}): {missing_scalers}")

            # Load scalers
            with open(non_tof_scaler_path, "rb") as f:
                non_tof_scaler = pickle.load(f)
            with open(static_scaler_path, "rb") as f:
                static_scaler = pickle.load(f)
            
            if variant == "full":
                with open(tof_scaler_path, "rb") as f:
                    tof_scaler = pickle.load(f)
                tof_in_channels = _get_tof_in_channels(tof_scaler)
            else:
                tof_scaler = None
                tof_in_channels = 0

            non_tof_in_channels = non_tof_scaler.mean_.shape[0]
            static_in_features  = static_scaler.mean_.shape[0]

            model = MultimodalityModel(
                seq_input_channels=non_tof_in_channels,
                tof_input_channels=tof_in_channels,
                static_input_features=static_in_features,
                num_classes=num_classes,
                sequence_length=SEQ_LEN
            )
            model.load_state_dict(torch.load(p, map_location=device))
            model.to(device).eval()

            pairs.append((model, non_tof_scaler, tof_scaler, static_scaler))
    else:
        # Single-model fallback
        weight_path = os.path.join(WEIGHT_DIR, f"best_model_{variant}.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"No weight file found for variant '{variant}'.")

        non_tof_scaler_path = os.path.join(WEIGHT_DIR, f"non_tof_scaler_{variant}.pkl")
        tof_scaler_path     = os.path.join(WEIGHT_DIR, f"tof_scaler_{variant}.pkl")
        static_scaler_path  = os.path.join(WEIGHT_DIR, f"static_scaler_{variant}.pkl")

        # Check required scalers
        required_scalers = [non_tof_scaler_path, static_scaler_path]
        if variant == "full":
            required_scalers.append(tof_scaler_path)
            
        missing_scalers = [p for p in required_scalers if not os.path.exists(p)]
        if missing_scalers:
            raise FileNotFoundError(f"Missing scaler(s) for single-model variant '{variant}': {missing_scalers}")

        # Load scalers
        with open(non_tof_scaler_path, "rb") as f:
            non_tof_scaler = pickle.load(f)
        with open(static_scaler_path, "rb") as f:
            static_scaler = pickle.load(f)
        
        if variant == "full":
            with open(tof_scaler_path, "rb") as f:
                tof_scaler = pickle.load(f)
            tof_in_channels = _get_tof_in_channels(tof_scaler)
        else:
            tof_scaler = None
            tof_in_channels = 0

        non_tof_in_channels = non_tof_scaler.mean_.shape[0]
        static_in_features  = static_scaler.mean_.shape[0]

        model = MultimodalityModel(
            seq_input_channels=non_tof_in_channels,
            tof_input_channels=tof_in_channels,
            static_input_features=static_in_features,
            num_classes=num_classes,
            sequence_length=SEQ_LEN
        )
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device).eval()

        pairs.append((model, non_tof_scaler, tof_scaler, static_scaler))

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
    """Convert one sequence to sequential & static numpy arrays ready for inference.

    Returns (variant, seq_arr[L,F], static_arr[features])"""

    seq_df = seq_pl.to_pandas()
    if not demog_pl.is_empty():
        seq_df = seq_df.merge(demog_pl.to_pandas(), on="subject", how="left")

    variant = _decide_variant(seq_df)

    # --- MODIFIED: Define and separate static vs. sequential features (same as training) ---
    metadata_cols = [
        "row_id", "sequence_id", "sequence_type", "sequence_counter",
        "subject", "orientation", "behavior", "phase",
    ]
    
    # Sequential features: exclude metadata AND static features
    seq_feat_cols = [c for c in seq_df.columns if c not in metadata_cols and c not in STATIC_FEATURE_COLS]

    if variant == "imu":
        seq_feat_cols = [c for c in seq_feat_cols if not (c.startswith("thm_") or c.startswith("tof_"))]

    # seq_df[seq_feat_cols] = seq_df[seq_feat_cols].replace(-1.0, np.nan)

    # 2-D interpolation for TOF sensor grids (per row) – skip if IMU variant
    if variant != "imu":
        seq_df = interpolate_tof(seq_df)

    # Ensure chronological order before temporal interpolation
    seq_df = seq_df.sort_values("sequence_counter")

    # Linear interpolation forward/backward along the time axis
    seq_df[seq_feat_cols] = seq_df[seq_feat_cols].interpolate(method="linear", limit_direction="both")

    # Fallback to median for any column still containing NaN, then 0 as a last resort
    seq_df[seq_feat_cols] = seq_df[seq_feat_cols].fillna(seq_df[seq_feat_cols].median())
    seq_df[seq_feat_cols] = seq_df[seq_feat_cols].fillna(0)

    # ---- build padded sequential tensor (unscaled)
    seq_arr = seq_df.sort_values("sequence_counter")[seq_feat_cols].to_numpy()
    seq_arr = pad_sequences([seq_arr], max_length=SEQ_LEN)[0]  # (L, F)
    seq_arr = seq_arr.astype(np.float32)  # Ensure float32

    # ---- static demographic features (order must match training)
    if STATIC_FEATURE_COLS[0] in seq_df.columns:
        static_vec = seq_df.iloc[0][STATIC_FEATURE_COLS].to_numpy()
    else:
        # If not merged correctly, fall back to zeros
        static_vec = np.zeros(len(STATIC_FEATURE_COLS), dtype=np.float32)

    static_vec = static_vec.astype(np.float32)

    return variant, seq_arr, static_vec


# ------------------ 预测逻辑 ------------------
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Entry point that Kaggle calls for each sequence.
    """
    if sequence.is_empty():
        return MAP_NON_TARGET

    variant, non_tof_arr, tof_arr, static_vec = preprocess_single_sequence_multimodal(sequence, demographics)

    res      = RESOURCES[variant]
    le       = res["label_encoder"]
    pairs    = res["model_scaler_pairs"]  # List[(model, non_tof_scaler, tof_scaler, static_scaler)]
    num_cls  = res["num_classes"]

    with torch.no_grad():
        probs_sum = np.zeros((1, num_cls))
        for model, non_tof_scaler, tof_scaler, static_scaler in pairs:
            # ----- scale inputs -----
            # 1. Scale non-TOF sequential features
            if non_tof_scaler is not None and non_tof_arr.shape[1] > 0:
                non_tof_channels = non_tof_scaler.mean_.shape[0]
                non_tof_scaled = non_tof_scaler.transform(non_tof_arr.reshape(-1, non_tof_channels)).reshape(1, SEQ_LEN, non_tof_channels)
                non_tof_scaled = non_tof_scaled.astype(np.float32)
            else:
                non_tof_scaled = np.empty((1, SEQ_LEN, 0), dtype=np.float32)

            # 2. Scale TOF sequential features
            if tof_scaler is not None and tof_arr.shape[1] > 0:
                tof_channels = _get_tof_in_channels(tof_scaler)
                # Flatten to 2-D, apply per-sensor or global scaler, then reshape back
                tof_flat = tof_arr.reshape(-1, tof_channels)
                tof_scaled_flat = _transform_tof(tof_flat, tof_scaler)
                tof_scaled = tof_scaled_flat.reshape(1, SEQ_LEN, tof_channels).astype(np.float32)
            else:
                tof_scaled = np.empty((1, SEQ_LEN, 0), dtype=np.float32)

            # 3. Scale static features
            static_scaled = static_scaler.transform(static_vec.reshape(1, -1))  # (1, static_features)
            static_scaled = static_scaled.astype(np.float32)

            # Convert to tensors
            xb_non_tof = torch.from_numpy(non_tof_scaled).to(DEVICE)
            xb_tof     = torch.from_numpy(tof_scaled).to(DEVICE)
            xb_static  = torch.from_numpy(static_scaled).to(DEVICE)

            # Forward pass through multimodal model
            probs = torch.softmax(model(xb_non_tof, xb_tof, xb_static), dim=1).cpu().numpy()
            probs_sum += probs

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
