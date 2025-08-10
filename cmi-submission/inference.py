#!/usr/bin/env python3
"""
CMI – Detect Behavior with Sensor Data
--------------------------------------
推理脚本（Code Competition 版本）

此版本为合并版本，整合了以下流程：
- 使用高级特征工程 (feature_engineering)
- 采用“先标准化后Padding”的正确数据处理流程
- 支持多模态模型 (MultimodalityModel)
- 使用 ColumnTransformer 进行统一的特征缩放

目录结构::
    cmi-submission/
        data_utils/
            __init__.py
            data_preprocessing.py   ← 核心预处理逻辑在此
            tof_utils.py
        models/
            __init__.py
            multimodality.py        ← 多模态模型在此
        weights/
            model_fold_1_full.pth …
            scaler_fold_1_full.pkl  ← 每个折叠对应一个ColumnTransformer
            label_encoder_full.pkl
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
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore")

# ------------------ 路径常量 ------------------
BASE_DIR   = os.path.dirname(__file__)
WEIGHT_DIR = os.path.join(BASE_DIR, "weights")

# ------------------ 自定义模块 ------------------
# 从zsb分支引入多模态模型
from models.multimodality import MultimodalityModel 
# 从你的分支(HEAD)引入核心预处理逻辑和常量
from data_utils.data_preprocessing import pad_sequences, feature_engineering, STATIC_FEATURE_COLS
from data_utils.tof_utils import interpolate_tof

# ------------------ 全局资源加载 ------------------
# 我们支持两种变体: "full" (使用 THM/TOF 传感器) 和 "imu" (仅IMU).
# 每种变体都有其自己的scaler和模型权重文件。

MAP_NON_TARGET = "Drink from bottle/cup"

def _load_preprocessing_objects(variant: str):
    """
    为给定变体加载标签编码器 (label encoder)。
    """
    le_path = os.path.join(WEIGHT_DIR, f"label_encoder_{variant}.pkl")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Label encoder for variant '{variant}' not found at {le_path}")

    with open(le_path, "rb") as f:
        le = pickle.load(f)
    
    return le

def _load_models(device, num_classes, variant: str):
    """
    加载多模态模型及其匹配的 ColumnTransformer scalers。
    返回一个 (model, scaler) 元组的列表。
    适用于 K-Fold 集成和单一模型提交。
    """
    pairs = []
    model_cfg_once = None  # To store the config from the first loaded model

    # 首先查找 K-Fold 模型
    fold_paths = [os.path.join(WEIGHT_DIR, f"model_fold_{i}_{variant}.pth") for i in range(1, 6)]
    fold_paths = [p for p in fold_paths if os.path.exists(p)]

    if fold_paths:
        print(f"🧩  [{variant}] Detected {len(fold_paths)} fold models → ensemble")
        for p in fold_paths:
            basename = os.path.basename(p)
            parts = basename.split("_")
            fold_num = int(parts[2])  # e.g. model_fold_3_full.pth → 3

            # <--- 核心改动：加载与每个模型对应的单个 ColumnTransformer scaler
            scaler_path = os.path.join(WEIGHT_DIR, f"scaler_fold_{fold_num}_{variant}.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Missing scaler for fold {fold_num} ({variant}): {scaler_path}")
            
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            # 使用训练时保存的配置构建模型
            ckpt = torch.load(p, map_location=device)
            if isinstance(ckpt, dict) and 'model_cfg' in ckpt:
                model_cfg = ckpt['model_cfg']
                if model_cfg_once is None:
                    model_cfg_once = model_cfg  # Save the config
                
                # 移除'type'键，因为它用于注册表，而不是构造函数
                init_kwargs = {k: v for k, v in model_cfg.items() if k != 'type'}
                state_dict = ckpt['state_dict']
                model = MultimodalityModel(**init_kwargs)
                
                # 移除 torch.compile 产生的 `_orig_mod.` 前缀
                is_compiled = any(key.startswith('_orig_mod.') for key in state_dict.keys())
                if is_compiled:
                    print("Model was trained with torch.compile(). Cleaning state_dict keys...")
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace('_orig_mod.', '', 1) 
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)
            else:
                raise ValueError(f"Checkpoint for {p} is in a legacy format without 'model_cfg'. Please retrain and save with model config.")

            model.to(device).eval()
            model = torch.compile(model, mode="reduce-overhead")
            pairs.append((model, scaler))
    else:
        print(f"No K-Fold models found")

    if not pairs:
        raise FileNotFoundError(f"No valid models found for variant '{variant}'.")
        
    return pairs, model_cfg_once

print("🔧  Initialising inference resources …")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

VARIANTS = ["full", "imu"]
RESOURCES = {}

for v in VARIANTS:
    try:
        le = _load_preprocessing_objects(v)
        num_classes = len(le.classes_)
        model_scaler_pairs, model_cfg = _load_models(DEVICE, num_classes, v)

        RESOURCES[v] = {
            "label_encoder": le,
            "num_classes": num_classes,
            "model_scaler_pairs": model_scaler_pairs,
            "model_cfg": model_cfg  # Store the model config
        }
        print(f"✅  Resources for '{v}' variant loaded successfully.")
    except FileNotFoundError as e:
        print(f"⚠️  Could not load resources for '{v}' variant: {e}. This variant will be unavailable.")

print("✅  Resource initialization complete. Ready for inference.")


# ------------------ 单序列预处理 ------------------
def _decide_variant(seq_df: "pd.DataFrame") -> str:
    """
    如果 THM 或 TOF 列中的所有行都是 NaN/-1，则返回 'imu'，否则返回 'full'。
    """
    thm_cols = [c for c in seq_df.columns if c.startswith("thm_")]
    tof_cols = [c for c in seq_df.columns if c.startswith("tof_")]

    if not thm_cols and not tof_cols:
        return "imu"
    
    thm_all_missing = True
    if thm_cols:
        thm_df = seq_df[thm_cols].replace(-1.0, np.nan)
        thm_all_missing = not thm_df.notna().values.any()
    
    tof_all_missing = True
    if tof_cols:
        tof_df = seq_df[tof_cols].replace(-1.0, np.nan)
        tof_all_missing = not tof_df.notna().values.any()
    
    if (thm_cols and thm_all_missing) or (tof_cols and tof_all_missing):
        return "imu"
        
    return "full"

def preprocess_single_sequence(seq_pl: pl.DataFrame, demog_pl: pl.DataFrame):
    """
    通过应用完整的 ToF 和 IMU 特征工程流程来预处理单个序列，
    确保它与训练过程完全匹配。
    """
    seq_df = seq_pl.to_pandas()
    if not demog_pl.is_empty():
        seq_df = seq_df.merge(demog_pl.to_pandas(), on="subject", how="left")

    variant = _decide_variant(seq_df)
    
    if variant not in RESOURCES:
        raise FileNotFoundError(f"Resources for variant '{variant}' not available. Ensure models and scalers are exported for this variant.")
    else:
        print(f"🧬 Preprocessing with variant: {variant}")

    # 1. 首先，如果需要，处理所有 ToF 插值。
    if variant != "imu":
        seq_df = interpolate_tof(seq_df)

    # 2. 接下来，应用高级特征工程。
    processed_df, feature_cols = feature_engineering(seq_df)

    # --- ！！！关键修复：将静态特征列添加回总特征列表！！！ ---
    # 找出数据中实际存在的静态列
    existing_static_cols = [c for c in STATIC_FEATURE_COLS if c in processed_df.columns]
    
    # 将它们添加到 feature_cols 列表中，并去重
    for col in existing_static_cols:
        if col not in feature_cols:
            feature_cols.append(col)
    # -----------------------------------------------------------------

    # 3. 如果确定的变体是仅IMU，则再次确认过滤
    if variant == "imu":
        imu_engineered_cols = [c for c in feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
        demographic_cols = [c for c in STATIC_FEATURE_COLS if c in processed_df.columns]
        feature_cols = sorted(list(set(imu_engineered_cols + demographic_cols)))

    # 4. 返回最终的特征DataFrame，它现在包含了所有scaler需要的列
    #    并确保列的顺序与训练时一致（虽然ColumnTransformer不强求顺序，但这是个好习惯）
    final_features_df = processed_df.sort_values("sequence_counter")
    
    # 确保返回的DataFrame只包含feature_cols中的列，并按此顺序排列
    final_features_df = final_features_df[[c for c in feature_cols if c in final_features_df.columns]]

    return variant, final_features_df


# ------------------ 预测逻辑 ------------------
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Entry point that Kaggle calls for each sequence.
    """
    if sequence.is_empty():
        return MAP_NON_TARGET

    # 1. 预处理序列以获得一个未填充的、待标准化的特征DataFrame
    variant, features_df = preprocess_single_sequence(sequence, demographics)

    res                  = RESOURCES[variant]
    le                   = res["label_encoder"]
    model_scaler_pairs   = res["model_scaler_pairs"]
    num_cls              = res["num_classes"]
    # ✨ Get sequence length dynamically from the model's config
    model_cfg            = res["model_cfg"]
    sequence_length      = model_cfg['sequence_length']

    with torch.no_grad():
        probs_sum = np.zeros((1, num_cls))
        
        # 2. 循环遍历每个模型及其对应的 ColumnTransformer scaler
        for model, scaler in model_scaler_pairs:
            
            # 3. ✨ 标准化: 对整个特征DataFrame应用scaler
            X_scaled_unpadded = scaler.transform(features_df)
            scaled_feature_names = scaler.get_feature_names_out()

            # 4. ✨ 拆分多模态数据 (在标准化之后)
            static_cols = [c for c in scaled_feature_names if c in STATIC_FEATURE_COLS]
            tof_cols   = [c for c in scaled_feature_names if c.startswith('tof_')]
            thm_cols   = [c for c in scaled_feature_names if c.startswith('thm_')]
            imu_cols   = [c for c in scaled_feature_names if (c not in static_cols and not c.startswith('tof_') and not c.startswith('thm_'))]

            static_idx = [list(scaled_feature_names).index(c) for c in static_cols]
            tof_idx    = [list(scaled_feature_names).index(c) for c in tof_cols]
            thm_idx    = [list(scaled_feature_names).index(c) for c in thm_cols]
            imu_idx    = [list(scaled_feature_names).index(c) for c in imu_cols]

            static_arr = X_scaled_unpadded[:, static_idx]
            tof_arr    = X_scaled_unpadded[:, tof_idx]
            thm_arr    = X_scaled_unpadded[:, thm_idx]
            imu_arr    = X_scaled_unpadded[:, imu_idx]

            # 5. ✨ 分别对 IMU 和 THM 进行 Padding 并生成 mask
            X_imu_pad, imu_mask = pad_sequences([imu_arr], max_length=sequence_length)
            X_thm_pad, thm_mask = pad_sequences([thm_arr], max_length=sequence_length)
            X_tof_pad, _                = pad_sequences([tof_arr], max_length=sequence_length)
            X_static                    = static_arr[0:1, :]  # 静态特征取第一行即可

            # 6. 转换为Tensor并预测
            xb_imu     = torch.from_numpy(X_imu_pad.astype(np.float32)).to(DEVICE)
            xb_thm     = torch.from_numpy(X_thm_pad.astype(np.float32)).to(DEVICE)
            xb_tof     = torch.from_numpy(X_tof_pad.astype(np.float32)).to(DEVICE)
            xb_static  = torch.from_numpy(X_static.astype(np.float32)).to(DEVICE)
            xb_mask    = torch.from_numpy(imu_mask.astype(np.float32)).to(DEVICE)

            # Forward pass through multimodal model
            probs = torch.softmax(model(xb_imu, xb_thm, xb_tof, xb_static, mask=xb_mask), dim=1).cpu().numpy()
            probs_sum += probs

        # Average the probabilities for the ensemble
        probs = probs_sum / len(model_scaler_pairs)

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
        os.chdir("/kaggle/working")
        inference_server.run_local_gateway(
            data_paths=(
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
            )
        )