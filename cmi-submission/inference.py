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
from data_utils.data_preprocessing import pad_sequences, feature_engineering, create_sequence_level_features, STATIC_FEATURE_COLS
from data_utils.tof_utils import interpolate_tof

# ------------------ 全局资源加载 ------------------
# 我们支持两种变体: "full" (使用 THM/TOF 传感器) 和 "imu" (仅IMU).
# 每种变体都有其自己的scaler和模型权重文件。

MAP_NON_TARGET = "Drink from bottle/cup"
SEQ_LEN        = 100      # 与训练保持一致

def _load_preprocessing_objects(variant: str):
    """
    为给定变体加载标签编码器 (label encoder)。
    """
    le_path = os.path.join(WEIGHT_DIR, f"label_encoder_{variant}.pkl")
    if not os.path.exists(le_path):
        # 回退到通用文件名 (旧的提交)
        le_path = os.path.join(WEIGHT_DIR, "label_encoder.pkl")
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

            # 从scaler获取输入维度信息
            feature_names = scaler.get_feature_names_out()
            static_in_features = len([c for c in feature_names if c in STATIC_FEATURE_COLS])
            tof_in_channels = len([c for c in feature_names if c.startswith('tof_')])
            non_tof_in_channels = len(feature_names) - static_in_features - tof_in_channels

            # 使用训练时保存的配置构建模型
            ckpt = torch.load(p, map_location=device)
            if isinstance(ckpt, dict) and 'model_cfg' in ckpt:
                model_cfg = ckpt['model_cfg']
                # 移除'type'键，因为它用于注册表，而不是构造函数
                model_cfg = {k: v for k, v in model_cfg.items() if k != 'type'}
                state_dict = ckpt['state_dict']
                model = MultimodalityModel(**model_cfg)
                # 移除 torch.compile 产生的 `_orig_mod.` 前缀
                # 检查键是否以 `_orig_mod.` 开头
                is_compiled = any(key.startswith('_orig_mod.') for key in state_dict.keys())
                if is_compiled:
                    print("Model was trained with torch.compile(). Cleaning state_dict keys...")
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        # 去掉 '_orig_mod.' 前缀
                        name = k.replace('_orig_mod.', '', 1) 
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                else:
                    # 如果没有前缀，则正常加载
                    model.load_state_dict(state_dict)
            else:
                # 为没有配置的旧checkpoint提供回退
                raise ValueError(f"Checkpoint for {p} is in a legacy format without 'model_cfg'. Please retrain and save with model config.")

            model.to(device).eval()
            model = torch.compile(model, mode="reduce-overhead")
            pairs.append((model, scaler))
    else:
        print(f"No K-Fold models found")
        # # 单一模型回退逻辑
        # print(f"SINGLE MODEL for variant: {variant}")
        # weight_path = os.path.join(WEIGHT_DIR, f"best_model_{variant}.pth")
        # scaler_path = os.path.join(WEIGHT_DIR, f"scaler_{variant}.pkl")

        # if not os.path.exists(weight_path):
        #      raise FileNotFoundError(f"No weight file found for variant '{variant}'.")
        # if not os.path.exists(scaler_path):
        #     raise FileNotFoundError(f"No scaler file found for variant '{variant}'.")

        # with open(scaler_path, "rb") as f:
        #     scaler = pickle.load(f)

        # feature_names = scaler.get_feature_names_out()
        # static_in_features = len([c for c in feature_names if c in STATIC_FEATURE_COLS])
        # tof_in_channels = len([c for c in feature_names if c.startswith('tof_')])
        # non_tof_in_channels = len(feature_names) - static_in_features - tof_in_channels
        
        # # 使用与 K-Fold 相同的逻辑加载模型
        # ckpt = torch.load(weight_path, map_location=device)
        # if isinstance(ckpt, dict) and 'model_cfg' in ckpt:
        #     model_cfg = ckpt['model_cfg']
        #     model_cfg = {k: v for k, v in model_cfg.items() if k != 'type'}
        #     state_dict = ckpt['state_dict']
        #     model = MultimodalityModel(**model_cfg)
        #     model.load_state_dict(state_dict)
        # else:
        #     raise ValueError(f"Checkpoint for {weight_path} is in a legacy format without 'model_cfg'. Please retrain and save with model config.")
        
        # model.to(device).eval()
        # pairs.append((model, scaler))

    return pairs

print("🔧  Initialising inference resources …")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

VARIANTS = ["full", "imu"]
RESOURCES = {}

for v in VARIANTS:
    try:
        le = _load_preprocessing_objects(v)
        num_classes = len(le.classes_)
        model_scaler_pairs = _load_models(DEVICE, num_classes, v)

        RESOURCES[v] = {
            "label_encoder": le,
            "num_classes": num_classes,
            "model_scaler_pairs": model_scaler_pairs, # <-- 现在是 (model, ColumnTransformer) 对
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
    [修正版] 修复了最后返回的DataFrame被错误覆盖的问题。
    """
    seq_df = seq_pl.to_pandas()
    if not demog_pl.is_empty():
        demog_pandas_df = demog_pl.to_pandas()
        if 'subject_id' in demog_pandas_df.columns and 'subject' not in demog_pandas_df.columns:
            demog_pandas_df = demog_pandas_df.rename(columns={'subject_id': 'subject'})
        seq_df = seq_df.merge(demog_pandas_df, on="subject", how="left")

    variant = _decide_variant(seq_df)
    
    if variant not in RESOURCES:
        fallback_variant = "imu" if "imu" in RESOURCES else "full"
        print(f"🧬 Variant '{variant}' not available, falling back to '{fallback_variant}'")
        variant = fallback_variant
    else:
        print(f"🧬 Preprocessing with variant: {variant}")

    # 1. ToF 插值
    if variant != "imu":
        seq_df = interpolate_tof(seq_df)

    # 2. 生成时间点级别特征
    processed_df, ts_feature_cols = feature_engineering(seq_df)

    # 3. 生成序列级别聚合特征
    agg_features_df = create_sequence_level_features(processed_df)

    # 4. 合并所有特征，得到最终的完整DataFrame
    merged_df = processed_df.merge(agg_features_df, on='sequence_id', how='left').fillna(0.0)
    
    # 5. 定义最终需要的所有特征列
    all_feature_cols = ts_feature_cols + [c for c in agg_features_df.columns if c != 'sequence_id'] + STATIC_FEATURE_COLS
    
    if variant == "imu":
        # 如果是IMU模式，则过滤掉ToF/Thm特征
        feature_cols = [c for c in all_feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
    else:
        feature_cols = all_feature_cols

    # 确保我们只保留实际存在的列
    feature_cols = [c for c in feature_cols if c in merged_df.columns]

    # ✨ 6. 核心修复：对合并后的 `merged_df` 进行排序和列选择
    #    确保我们操作的是包含了所有特征的正确DataFrame
    final_df_to_return = merged_df.sort_values("sequence_counter")
    
    # 返回一个只包含最终特征列，并且顺序正确的DataFrame
    final_df_to_return = final_df_to_return[feature_cols]

    return variant, final_df_to_return


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

    with torch.no_grad():
        probs_sum = np.zeros((1, num_cls))
        
        # 2. 循环遍历每个模型及其对应的 ColumnTransformer scaler
        for model, scaler in model_scaler_pairs:
            
            # 3. ✨ 标准化: 对整个特征DataFrame应用scaler
            X_scaled_unpadded = scaler.transform(features_df)
            scaled_feature_names = scaler.get_feature_names_out()

            # 4. ✨ 拆分多模态数据 (在标准化之后)
            agg_freq_cols = [c for c in scaled_feature_names if c.startswith('agg_') or c.startswith('freq_')]
            static_cols = [c for c in scaled_feature_names if c in STATIC_FEATURE_COLS] + agg_freq_cols
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
            X_imu_pad, imu_mask = pad_sequences([imu_arr], max_length=SEQ_LEN)
            X_thm_pad, thm_mask = pad_sequences([thm_arr], max_length=SEQ_LEN)
            X_tof_pad, _                = pad_sequences([tof_arr], max_length=SEQ_LEN)
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