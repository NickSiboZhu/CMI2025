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
MAP_NON_TARGET = "Drink from bottle/cup"
SEQ_LEN        = 100          # 与训练保持一致

def _load_preprocessing_objects():
    with open(os.path.join(WEIGHT_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(WEIGHT_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return le, scaler

def _load_models(device, in_channels, num_classes) -> List[torch.nn.Module]:
    models = []
    # 先找 5-fold
    fold_paths = [
        os.path.join(WEIGHT_DIR, f"model_fold_{i}.pth") for i in range(1, 6)
    ]
    fold_paths = [p for p in fold_paths if os.path.exists(p)]
    if fold_paths:
        print(f"🧩  Detected {len(fold_paths)} fold models → ensemble")
        for p in fold_paths:
            m = Simple1DCNN(in_channels, num_classes, SEQ_LEN)
            m.load_state_dict(torch.load(p, map_location=device))
            m.to(device).eval()
            models.append(m)
    else:
        single = os.path.join(WEIGHT_DIR, "best_model.pth")
        if not os.path.exists(single):
            raise FileNotFoundError(
                "No model weights found in weights/ (best_model.pth or model_fold_*.pth)"
            )
        print("🧩  Using single model best_model.pth")
        m = Simple1DCNN(in_channels, num_classes, SEQ_LEN)
        m.load_state_dict(torch.load(single, map_location=device))
        m.to(device).eval()
        models.append(m)
    return models

print("🔧  Initialising inference resources …")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_ENC, SCALER = _load_preprocessing_objects()

# 推断输入维度：借助 label_encoder 里的类数 & 训练时记录的 scaler 特征数
IN_CHANNELS = SCALER.mean_.shape[0]
NUM_CLASSES = len(LABEL_ENC.classes_)
MODELS      = _load_models(DEVICE, IN_CHANNELS, NUM_CLASSES)
print("✅  Models / Scaler / LabelEncoder loaded. Ready.")


# ------------------ 单序列预处理 ------------------
def preprocess_single_sequence(
    seq_pl: pl.DataFrame,
    demog_pl: pl.DataFrame,
) -> np.ndarray:
    """返回 shape (1, SEQ_LEN, IN_CHANNELS) 的 numpy 数组"""

    seq_df = seq_pl.to_pandas()
    if not demog_pl.is_empty():
        seq_df = seq_df.merge(demog_pl.to_pandas(), on="subject", how="left")

    meta_cols = [
        "row_id", "sequence_id", "sequence_type", "sequence_counter",
        "subject", "orientation", "behavior", "phase",
    ]
    feat_cols = [c for c in seq_df.columns if c not in meta_cols]

    seq_df[feat_cols] = seq_df[feat_cols].replace(-1.0, np.nan)
    for c in feat_cols:
        seq_df[c] = seq_df[c].fillna(seq_df[c].median())

    arr = seq_df.sort_values("sequence_counter")[feat_cols].to_numpy()
    arr = pad_sequences([arr], max_length=SEQ_LEN)[0]           # (L, F)
    arr = SCALER.transform(arr.reshape(-1, IN_CHANNELS)).reshape(1, SEQ_LEN, IN_CHANNELS)
    return arr.astype(np.float32)


# ------------------ 预测逻辑 ------------------
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Kaggle 评测机会多次调用此函数，每次一条 sequence。
    """
    if sequence.is_empty():
        return MAP_NON_TARGET

    X = preprocess_single_sequence(sequence, demographics)      # (1, L, F)
    ds = GestureDataset(X, np.zeros(1))
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # 多模型平均
    with torch.no_grad():
        probs_sum = np.zeros((1, NUM_CLASSES))
        for model in MODELS:
            for xb, _ in dl:
                xb = xb.to(DEVICE)
                probs = torch.softmax(model(xb), dim=1).cpu().numpy()
                probs_sum += probs
        probs = probs_sum / len(MODELS)

    pred_idx = int(np.argmax(probs, axis=1)[0])
    label    = LABEL_ENC.inverse_transform([pred_idx])[0]
    return label if label in LABEL_ENC.classes_ else MAP_NON_TARGET


# ------------------ 启动评测服务器 ------------------
if __name__ == "__main__":
    import kaggle_evaluation.cmi_inference_server as kis
    print("🚀  Starting CMIInferenceServer …")
    kis.CMIInferenceServer(predict).serve()
