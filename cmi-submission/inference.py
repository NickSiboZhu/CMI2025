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
            cnn.py                  ← Simple1DCNN & GestureDataset
        weights/
            best_model.pth  或  model_fold_1.pth …
            label_encoder.pkl
            scaler.pkl
        inference.py                ← 当前文件
"""

import os
import pickle
import warnings
from typing import List

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
from data.data_preprocessing import pad_sequences

# ------------------ 全局资源加载 ------------------
# We support two variants: "full" (uses THM/TOF sensors) and "imu" (IMU-only).
# Each variant has its own scaler and model weight files.

MAP_NON_TARGET = "Drink from bottle/cup"
SEQ_LEN        = 100          # 与训练保持一致

def _load_preprocessing_objects(variant: str):
    """Load label encoder & scaler for the given variant."""
    le_path     = os.path.join(WEIGHT_DIR, f"label_encoder_{variant}.pkl")
    scaler_path = os.path.join(WEIGHT_DIR, f"scaler_{variant}.pkl")

    # Back-compat: fall back to un-suffixed filenames if variant files not found
    if not os.path.exists(le_path):
        le_path = os.path.join(WEIGHT_DIR, "label_encoder.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(WEIGHT_DIR, "scaler.pkl")

    with open(le_path, "rb") as f:
        le = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return le, scaler

def _load_models(device, in_channels, num_classes, variant: str) -> List[torch.nn.Module]:
    models = []

    # 搜索 5-fold 文件
    fold_paths = [
        os.path.join(WEIGHT_DIR, f"model_fold_{i}_{variant}.pth") for i in range(1, 6)
    ]
    fold_paths = [p for p in fold_paths if os.path.exists(p)]

    if fold_paths:
        print(f"🧩  [{variant}] Detected {len(fold_paths)} fold models → ensemble")
        for p in fold_paths:
            m = Simple1DCNN(in_channels, num_classes, SEQ_LEN)
            m.load_state_dict(torch.load(p, map_location=device))
            m.to(device).eval()
            models.append(m)
    else:
        single = os.path.join(WEIGHT_DIR, f"best_model_{variant}.pth")
        if not os.path.exists(single):
            raise FileNotFoundError(
                f"No model weights found for variant '{variant}' in weights/"
            )
        print(f"🧩  [{variant}] Using single model {os.path.basename(single)}")
        m = Simple1DCNN(in_channels, num_classes, SEQ_LEN)
        m.load_state_dict(torch.load(single, map_location=device))
        m.to(device).eval()
        models.append(m)

    return models

print("🔧  Initialising inference resources …")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANTS = ["full", "imu"]
RESOURCES = {}

for v in VARIANTS:
    le, scaler = _load_preprocessing_objects(v)
    in_channels = scaler.mean_.shape[0]
    num_classes = len(le.classes_)
    models = _load_models(DEVICE, in_channels, num_classes, v)

    RESOURCES[v] = {
        "label_encoder": le,
        "scaler": scaler,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "models": models,
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
    Convert a single sequence to a numpy tensor suitable for the chosen variant.
    Returns (variant, np.ndarray[1, L, F])
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

    seq_df[feat_cols] = seq_df[feat_cols].replace(-1.0, np.nan)
    for c in feat_cols:
        seq_df[c] = seq_df[c].fillna(seq_df[c].median())

    # ---- build padded tensor
    arr = seq_df.sort_values("sequence_counter")[feat_cols].to_numpy()
    arr = pad_sequences([arr], max_length=SEQ_LEN)[0]  # (L, F)

    scaler = RESOURCES[variant]["scaler"]
    in_channels = RESOURCES[variant]["in_channels"]
    arr = scaler.transform(arr.reshape(-1, in_channels)).reshape(1, SEQ_LEN, in_channels)

    return variant, arr.astype(np.float32)


# ------------------ 预测逻辑 ------------------
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Kaggle 评测机会多次调用此函数，每次一条 sequence。
    """
    if sequence.is_empty():
        return MAP_NON_TARGET

    variant, X = preprocess_single_sequence(sequence, demographics)

    res      = RESOURCES[variant]
    le       = res["label_encoder"]
    models   = res["models"]
    num_cls  = res["num_classes"]

    ds = GestureDataset(X, np.zeros(1))
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # 多模型平均
    with torch.no_grad():
        probs_sum = np.zeros((1, num_cls))
        for model in models:
            for xb, _ in dl:
                xb = xb.to(DEVICE)
                probs = torch.softmax(model(xb), dim=1).cpu().numpy()
                probs_sum += probs
        probs = probs_sum / len(models)

    pred_idx = int(np.argmax(probs, axis=1)[0])
    label    = le.inverse_transform([pred_idx])[0]
    return label if label in le.classes_ else MAP_NON_TARGET


# ------------------ 启动评测服务器 ------------------
if __name__ == "__main__":
    import kaggle_evaluation.cmi_inference_server as kis
    print("🚀  Starting CMIInferenceServer …")
    kis.CMIInferenceServer(predict).serve()
