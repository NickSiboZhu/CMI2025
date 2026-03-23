# This file is copied from development/data/tof_utils.py so the submission
# package remains self-contained on Kaggle.

#!/usr/bin/env python3
"""Utility functions for 2-D interpolation of the 8x8 ToF sensor grids."""
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import QhullError

def get_tof_columns(df_columns) -> dict:
    """
    Return a mapping from sensor id to the matching ``tof_<id>_v*`` columns.

    The layout is derived from the dataframe itself so the submission package
    does not depend on a hard-coded sensor configuration.
    """
    if df_columns is None:
        raise ValueError("df_columns is required for get_tof_columns")
    from .data_preprocessing import get_sensor_config
    config = get_sensor_config(df_columns)
    sensor_ids = config['tof_sensor_ids']
    pixels_per_sensor = config['tof_pixels_per_sensor']
    
    mapping = {}
    for sid in sensor_ids:
        mapping[sid] = [f"tof_{sid}_v{i}" for i in range(pixels_per_sensor)]
    return mapping

def _interpolate_block(values: np.ndarray, replacement_value: int = 255) -> np.ndarray:
    """
    Interpolate one 8x8 sensor frame while guarding sparse and degenerate cases.
    """
    values = values.copy()
    grid = values.astype(float).reshape(8, 8)

    # Treat the sentinel as a concrete distance before spatial interpolation.
    grid[grid == -1] = replacement_value
    
    nan_mask = np.isnan(grid)
    if not np.any(nan_mask):
        return grid.flatten()

    # Defer completely missing frames to the sequence-level fallback.
    if np.all(nan_mask):
        return values

    x, y = np.mgrid[0:8, 0:8]
    valid_points_coords = np.column_stack((x[~nan_mask], y[~nan_mask]))
    valid_points_values = grid[~nan_mask]
    points_to_interpolate = np.column_stack((x[nan_mask], y[nan_mask]))

    # Linear interpolation needs enough spatial support; otherwise nearest is safer.
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
    
    # Leave no non-finite values behind before the model sees the frame.
    if np.isnan(grid).any():
        grid[np.isnan(grid)] = 128

    return grid.flatten()


def interpolate_tof(df: pd.DataFrame, replacement_value: int = 255) -> pd.DataFrame:
    """
    Interpolate ToF values in two stages: temporal fill, then per-frame spatial fill.
    """
    print("\nStarting robust ToF interpolation...")
    df_processed = df.copy()
    
    tof_mapping = get_tof_columns(df.columns) 
    all_tof_cols = [col for sensor_cols in tof_mapping.values() for col in sensor_cols]

    if all_tof_cols:
        print("  Stage 1 (Temporal): Applying forward/backward fill to handle large gaps...")
        df_processed = df_processed.sort_values(['sequence_id', 'sequence_counter'])
        df_processed[all_tof_cols] = df_processed.groupby('sequence_id')[all_tof_cols].ffill()
        df_processed[all_tof_cols] = df_processed.groupby('sequence_id')[all_tof_cols].bfill()
        # Use a neutral midpoint when an entire sequence is missing for a sensor.
        df_processed[all_tof_cols] = df_processed[all_tof_cols].fillna(128)


    print("  Stage 2 (Spatial): Applying 2D interpolation for scattered NaNs...")
    for sensor_id, cols in tof_mapping.items():
        sensor_data_block = df_processed[cols].to_numpy()
        processed_block = np.empty_like(sensor_data_block, dtype=float)
        
        for i in range(sensor_data_block.shape[0]):
            processed_block[i, :] = _interpolate_block(sensor_data_block[i, :], replacement_value)
            
        df_processed[cols] = processed_block

    print("Robust ToF interpolation complete.")
    return df_processed
