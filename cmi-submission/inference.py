#!/usr/bin/env python3
"""
CMI – Detect Behavior with Sensor Data
--------------------------------------
推理脚本 (最终混合模型版本 - 已修复padding错误)

此版本为最终整合版，包含以下核心功能：
- 加载并运行一个混合1D+2D的深度学习模型 (MultimodalityModel)。
- 在推理时动态地从时域信号生成频谱图 (Spectrograms)。
- 使用与每个模型折叠相匹配的预处理器 (ColumnTransformer) 和频谱图统计量 (spec_stats)
  来确保数据处理与训练时完全一致。
- 健壮地处理任意长度的输入序列（过长则截断，过短则填充）。
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
from scipy.spatial.transform import Rotation as R
from scipy import signal

warnings.filterwarnings("ignore")

# ------------------ 路径常量 ------------------
BASE_DIR   = os.path.dirname(__file__)
WEIGHT_DIR = os.path.join(BASE_DIR, "weights")

# ------------------ 自定义模块 ------------------
from models.multimodality import MultimodalityModel
from data_utils.data_preprocessing import pad_sequences, feature_engineering, STATIC_FEATURE_COLS, generate_spectrogram
from data_utils.tof_utils import interpolate_tof

# ------------------ 全局资源加载 ------------------
MAP_NON_TARGET = "Drink from bottle/cup"

def _load_preprocessing_objects(variant: str):
    """为给定变体加载标签编码器 (label encoder)。"""
    le_path = os.path.join(WEIGHT_DIR, f"label_encoder_{variant}.pkl")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Label encoder for variant '{variant}' not found at {le_path}")
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    return le

def _load_models(device, variant: str):
    """Load models, their scalers, spectrogram stats, and capture model config (once).

    Additionally, if spec_params were saved during training as
    'spec_params_fold_{i}_{variant}.pkl', load them and return one set.
    """
    triplets = []
    model_cfg_once = None
    spec_params_once = None
    fold_paths = [os.path.join(WEIGHT_DIR, f"model_fold_{i}_{variant}.pth") for i in range(1, 6)]
    fold_paths = [p for p in fold_paths if os.path.exists(p)]

    if not fold_paths:
        print(f"No K-Fold models found for variant '{variant}'. This variant will be unavailable.")
        return [], None
        
    print(f"🧩  [{variant}] Detected {len(fold_paths)} fold models → ensemble")
    for p in fold_paths:
        basename = os.path.basename(p)
        fold_num = int(basename.split("_")[2])
        
        scaler_path = os.path.join(WEIGHT_DIR, f"scaler_fold_{fold_num}_{variant}.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler for fold {fold_num} ({variant}): {scaler_path}")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        spec_stats_path = os.path.join(WEIGHT_DIR, f"spec_stats_fold_{fold_num}_{variant}.pkl")
        if not os.path.exists(spec_stats_path):
            raise FileNotFoundError(f"Missing spectrogram stats for fold {fold_num} ({variant}): {spec_stats_path}")
        with open(spec_stats_path, "rb") as f:
            spec_stats = pickle.load(f)

        # Try to load spectrogram generation params used in training for this fold
        spec_params_path = os.path.join(WEIGHT_DIR, f"spec_params_fold_{fold_num}_{variant}.pkl")
        if spec_params_once is None and os.path.exists(spec_params_path):
            with open(spec_params_path, "rb") as f:
                spec_params_once = pickle.load(f)

        ckpt = torch.load(p, map_location=device)
        if 'model_cfg' not in ckpt:
            raise ValueError(f"Checkpoint for {p} is missing 'model_cfg'. Please retrain.")
        model_cfg = {k: v for k, v in ckpt['model_cfg'].items() if k != 'type'}
        if model_cfg_once is None:
            model_cfg_once = model_cfg
        state_dict = ckpt['state_dict']
        
        model = MultimodalityModel(**model_cfg)
        
        is_compiled = any(key.startswith('_orig_mod.') for key in state_dict.keys())
        if is_compiled:
            from collections import OrderedDict
            state_dict = OrderedDict((k.replace('_orig_mod.', '', 1), v) for k, v in state_dict.items())
        model.load_state_dict(state_dict, strict=True)

        model.to(device).eval()
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"⚠️  torch.compile failed during inference setup: {e}")

        triplets.append((model, scaler, spec_stats))
    return triplets, model_cfg_once, spec_params_once

# REMOVED: _load_spec_params_for_variant function
# We now strictly require spec_params to be loaded from weights directory

print("🔧  Initialising inference resources …")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

VARIANTS = ["full", "imu"]
RESOURCES = {}

for v in VARIANTS:
    try:
        le = _load_preprocessing_objects(v)
        model_scaler_stats_triplets, model_cfg_once, spec_params_from_model = _load_models(DEVICE, v)
        if model_scaler_stats_triplets:
            # STRICT: Require spec_params from weights (no config fallback)
            if spec_params_from_model is None:
                raise FileNotFoundError(
                    f"spec_params not found in weights for variant '{v}'. "
                    f"Please retrain with the updated train.py that saves spec_params."
                )
            spec_params = spec_params_from_model
            # Ensure sequence length used for padding matches the trained model
            seq_len_from_model = model_cfg_once['sequence_length'] if model_cfg_once and 'sequence_length' in model_cfg_once else spec_params['max_length']
            spec_params['max_length'] = seq_len_from_model
            RESOURCES[v] = {
                "label_encoder": le,
                "model_scaler_stats_triplets": model_scaler_stats_triplets,
                "spec_params": spec_params,
                "sequence_length": seq_len_from_model,
            }
            print(f"✅  Resources for '{v}' variant loaded successfully.")
            print(f"    Using spec_params from training: nperseg={spec_params['nperseg']}, noverlap={spec_params['noverlap']}")
    except FileNotFoundError as e:
        print(f"⚠️  Could not load resources for '{v}' variant: {e}.")

print("✅  Resource initialization complete. Ready for inference.")

def _decide_variant(seq_df: "pd.DataFrame") -> str:
    """如果 THM 或 TOF 数据缺失，则返回 'imu'，否则返回 'full'。"""
    thm_cols = [c for c in seq_df.columns if c.startswith("thm_")]
    tof_cols = [c for c in seq_df.columns if c.startswith("tof_")]
    if not thm_cols and not tof_cols: return "imu"
    thm_all_missing = not seq_df[thm_cols].notna().values.any() if thm_cols else True
    tof_all_missing = not seq_df[tof_cols].notna().values.any() if tof_cols else True
    return "imu" if thm_all_missing and tof_all_missing else "full"

def preprocess_single_sequence(seq_pl: pl.DataFrame, demog_pl: pl.DataFrame):
    """应用完整的特征工程流程来预处理单个序列。"""
    seq_df = seq_pl.to_pandas()
    if not demog_pl.is_empty():
        seq_df = seq_df.merge(demog_pl.to_pandas(), on="subject", how="left")

    variant = _decide_variant(seq_df)
    
    if variant not in RESOURCES:
        raise FileNotFoundError(f"Resources for variant '{variant}' not available. Ensure models and scalers are exported for this variant.")
    else:
        print(f"🧬 Preprocessing with variant: {variant}")

    if variant != "imu":
        seq_df = interpolate_tof(seq_df)

    processed_df, feature_cols = feature_engineering(seq_df)

    existing_static_cols = [c for c in STATIC_FEATURE_COLS if c in processed_df.columns]
    for col in existing_static_cols:
        if col not in feature_cols:
            feature_cols.append(col)

    if variant == "imu":
        feature_cols = [c for c in feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]

    final_features_df = processed_df.sort_values("sequence_counter")
    final_features_df = final_features_df[[c for c in feature_cols if c in final_features_df.columns]]

    return variant, final_features_df


# ------------------ 预测逻辑 ------------------
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Kaggle为每个序列调用的入口点。"""
    if sequence.is_empty():
        return MAP_NON_TARGET

    variant, features_df = preprocess_single_sequence(sequence, demographics)
    
    if variant not in RESOURCES:
        raise RuntimeError(f"Attempted to predict with variant '{variant}', but its resources were not loaded successfully at startup.")

    res = RESOURCES[variant]
    le = res["label_encoder"]
    model_scaler_stats_triplets = res["model_scaler_stats_triplets"]
    spec_params = res["spec_params"]
    sequence_length = res["sequence_length"]

    with torch.no_grad():
        probs_sum = None
        
        for model, scaler, spec_stats in model_scaler_stats_triplets:
            
            # 1. 标准化时域数据
            # --- MODIFIED: 在此处添加 .astype(np.float32) 来确保数据类型一致 ---
            X_scaled_unpadded = scaler.transform(features_df).astype(np.float32)
            
            scaled_feature_names = scaler.get_feature_names_out()

            # 2. 拆分多模态数据
            static_cols = [c for c in scaled_feature_names if c in STATIC_FEATURE_COLS]
            tof_cols = [c for c in scaled_feature_names if c.startswith('tof_')]
            thm_cols = [c for c in scaled_feature_names if c.startswith('thm_')]
            spec_source_cols = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z']
            spec_source_cols = [c for c in spec_source_cols if c in scaled_feature_names]
            imu_cols = [c for c in scaled_feature_names if c not in static_cols + tof_cols + thm_cols]

            static_idx = [list(scaled_feature_names).index(c) for c in static_cols]
            tof_idx = [list(scaled_feature_names).index(c) for c in tof_cols]
            thm_idx = [list(scaled_feature_names).index(c) for c in thm_cols]
            imu_idx = [list(scaled_feature_names).index(c) for c in imu_cols]
            spec_idx = [list(scaled_feature_names).index(c) for c in spec_source_cols]

            static_arr = X_scaled_unpadded[0:1, static_idx]
            tof_arr, thm_arr, imu_arr = X_scaled_unpadded[:, tof_idx], X_scaled_unpadded[:, thm_idx], X_scaled_unpadded[:, imu_idx]
            spec_source_arr = X_scaled_unpadded[:, spec_idx]

            # 3. 对时域数据进行 Padding (strict length from trained model)
            X_imu_pad, imu_mask = pad_sequences([imu_arr], max_length=sequence_length)
            X_thm_pad, _ = pad_sequences([thm_arr], max_length=sequence_length)
            X_tof_pad, _ = pad_sequences([tof_arr], max_length=sequence_length)

            # 4. 从标准化的时域数据动态生成并标准化频谱图
            sequence_spectrograms = []
            spec_mean, spec_std = spec_stats['mean'], spec_stats['std']
            for i in range(spec_source_arr.shape[1]):
                signal_1d = spec_source_arr[:, i]
                seq_len_current = len(signal_1d)
                if seq_len_current >= sequence_length:
                    padded_signal = signal_1d[-sequence_length:]
                else:
                    padded_signal = np.pad(signal_1d, (sequence_length - seq_len_current, 0), 'constant')
                
                spec = generate_spectrogram(
                    padded_signal,
                    fs=spec_params['fs'],
                    nperseg=spec_params['nperseg'],
                    noverlap=spec_params['noverlap'],
                    max_length=sequence_length,
                )
                spec_norm = (spec - spec_mean) / (spec_std + 1e-6)
                sequence_spectrograms.append(spec_norm)
            
            X_spec = np.stack(sequence_spectrograms, axis=0)[np.newaxis, ...]
            
            # 5. 转换为Tensor (由于上游已是float32, 这里生成的也是FloatTensor)
            xb_imu = torch.from_numpy(X_imu_pad).to(DEVICE)
            xb_thm = torch.from_numpy(X_thm_pad).to(DEVICE)
            xb_tof = torch.from_numpy(X_tof_pad).to(DEVICE)
            xb_spec = torch.from_numpy(X_spec).to(DEVICE)
            xb_static = torch.from_numpy(static_arr).to(DEVICE)
            xb_mask = torch.from_numpy(imu_mask).to(DEVICE)

            # 6. 使用混合模型进行前向传播
            probs = torch.softmax(model(xb_imu, xb_thm, xb_tof, xb_spec, xb_static, mask=xb_mask), dim=1).cpu().numpy()
            
            if probs_sum is None: probs_sum = probs
            else: probs_sum += probs

        avg_probs = probs_sum / len(model_scaler_stats_triplets)

    pred_idx = int(np.argmax(avg_probs, axis=1)[0])
    label = le.inverse_transform([pred_idx])[0]
    return label if label in le.classes_ else MAP_NON_TARGET


# ------------------ 启动评测服务器 ------------------
if __name__ == "__main__":
    import kaggle_evaluation.cmi_inference_server as kis
    print("🚀 Starting CMIInferenceServer …")
    inference_server = kis.CMIInferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        # 本地测试网关
        os.chdir("/kaggle/working")
        inference_server.run_local_gateway(
            data_paths=(
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
            )
        )