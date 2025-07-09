# This file is copied from development/data/tof_utils.py so that the submission
# package is self-contained when uploaded to Kaggle.

#!/usr/bin/env python3
"""Utility functions for 2-D interpolation of the 8×8 TOF sensor grids."""
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import QhullError

_TOF_SENSORS = [1, 2, 3, 4, 5]


def get_tof_columns() -> dict:
    """Return a mapping: sensor_id -> list of 64 column names (tof_X_v0 … v63)."""
    mapping = {}
    for sid in _TOF_SENSORS:
        mapping[sid] = [f"tof_{sid}_v{i}" for i in range(64)]
    return mapping

def _interpolate_block(values: np.ndarray, replacement_value: int = 255) -> np.ndarray:
    """
    Robustly interpolates a single 8x8 spatial grid. Handles various edge cases.
    """
    values = values.copy()
    grid = values.astype(float).reshape(8, 8)

    # 替换 -1
    grid[grid == -1] = replacement_value
    
    # 如果没有NaN，直接返回
    nan_mask = np.isnan(grid)
    if not np.any(nan_mask):
        return grid.flatten()

    # 如果全部是NaN，也直接返回（让主函数处理）
    if np.all(nan_mask):
        return values # 返回原始的全NaN数组

    x, y = np.mgrid[0:8, 0:8]
    valid_points_coords = np.column_stack((x[~nan_mask], y[~nan_mask]))
    valid_points_values = grid[~nan_mask]
    points_to_interpolate = np.column_stack((x[nan_mask], y[nan_mask]))

    # 3D线性插值至少需要3个非共线的点
    if valid_points_coords.shape[0] < 3:
        method = 'nearest'
    else:
        method = 'linear'

    try:
        interpolated_values = griddata(valid_points_coords, valid_points_values, points_to_interpolate, method=method)
        nan_in_result = np.isnan(interpolated_values)
        if np.any(nan_in_result):
            nearest_values = griddata(valid_points_coords, valid_points_values, points_to_interpolate[nan_in_result], method='nearest')
            interpolated_values[nan_in_result] = nearest_values
    except (QhullError, ValueError):
        interpolated_values = griddata(valid_points_coords, valid_points_values, points_to_interpolate, method='nearest')

    grid[nan_mask] = interpolated_values
    
    # 最终保险：如果还有NaN，用中位数或替换值填充
    if np.isnan(grid).any():
        grid[np.isnan(grid)] = 128

    return grid.flatten()


def interpolate_tof(df: pd.DataFrame, replacement_value: int = 255) -> pd.DataFrame:
    """
    A self-contained, robust function to handle all ToF data interpolation using
    a two-stage hierarchical strategy: Temporal fill -> Spatial interpolation.
    """
    print("\nStarting robust ToF interpolation...")
    df_processed = df.copy()
    
    # 假设 get_tof_columns() 存在且能正常工作
    tof_mapping = get_tof_columns() 
    all_tof_cols = [col for sensor_cols in tof_mapping.values() for col in sensor_cols]

    # STAGE 1: Temporal Filling for large gaps (all-NaN blocks)
    if all_tof_cols:
        print("  Stage 1 (Temporal): Applying forward/backward fill to handle large gaps...")
        df_processed = df_processed.sort_values(['sequence_id', 'sequence_counter'])
        df_processed[all_tof_cols] = df_processed.groupby('sequence_id')[all_tof_cols].ffill()
        df_processed[all_tof_cols] = df_processed.groupby('sequence_id')[all_tof_cols].bfill()
        # 对于整个序列都是NaN的极端情况，填充一个中性值
        df_processed[all_tof_cols] = df_processed[all_tof_cols].fillna(128)


    # STAGE 2: Spatial Interpolation for scattered NaNs
    print("  Stage 2 (Spatial): Applying 2D interpolation for scattered NaNs...")
    for sensor_id, cols in tof_mapping.items():
        sensor_data_block = df_processed[cols].to_numpy()
        processed_block = np.empty_like(sensor_data_block, dtype=float)
        
        for i in range(sensor_data_block.shape[0]):
            processed_block[i, :] = _interpolate_block(sensor_data_block[i, :], replacement_value)
            
        df_processed[cols] = processed_block

    print("✅ Robust ToF Interpolation complete.")
    return df_processed