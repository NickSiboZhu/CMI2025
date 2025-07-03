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
    Handles complex data cleaning with added debugging for QhullError.
    """
    values = values.copy()
    grid = values.astype(float).reshape(8, 8)

    if np.all(np.isnan(grid)):
        return values

    grid[grid == -1] = replacement_value

    nan_mask = np.isnan(grid)
    if not nan_mask.any():
        return grid.flatten()

    x, y = np.mgrid[0:8, 0:8]
    valid_points_coords = np.column_stack((x[~nan_mask], y[~nan_mask]))
    valid_points_values = grid[~nan_mask]
    
    if valid_points_coords.shape[0] < 3:
        points_to_interpolate = np.column_stack((x[nan_mask], y[nan_mask]))
        interpolated_values = griddata(
            valid_points_coords, valid_points_values, points_to_interpolate, method='nearest'
        )
    else:
        points_to_interpolate = np.column_stack((x[nan_mask], y[nan_mask]))
        try:
            interpolated_values = griddata(
                valid_points_coords, valid_points_values, points_to_interpolate, method='linear'
            )
            
            nan_in_result_mask = np.isnan(interpolated_values)
            if nan_in_result_mask.any():
                nearest_values = griddata(
                    valid_points_coords, valid_points_values, points_to_interpolate[nan_in_result_mask], method='nearest'
                )
                interpolated_values[nan_in_result_mask] = nearest_values

        except QhullError:
            # --- DEBUGGING BLOCK STARTS HERE ---
            print("This error occurs when all valid points are collinear (on the same line).")
            
            # We print the grid state *before* interpolation is attempted.
            # This is the grid after -1 has been replaced by large value.
            print("Problematic 8x8 Grid (after -1 -> large value replacement):")
            # Use numpy print options for better formatting
            with np.printoptions(precision=1, suppress=True, linewidth=120):
                print(grid)
            
            print("\nList of valid points' coordinates passed to the algorithm:")
            print(valid_points_coords)
            print("="*60 + "\n")
            # --- DEBUGGING BLOCK ENDS HERE ---

            # Fallback to nearest neighbor interpolation
            interpolated_values = griddata(
                valid_points_coords, valid_points_values, points_to_interpolate, method='nearest'
            )

    grid[nan_mask] = interpolated_values
    
    if np.isnan(grid).any():
        median_val = np.nanmedian(grid)
        grid[np.isnan(grid)] = median_val if not np.isnan(median_val) else float(replacement_value)

    return grid.flatten()

def interpolate_tof(df: pd.DataFrame) -> pd.DataFrame:
    """
    An optimized processing function that operates on whole data blocks (sensors)
    instead of row-by-row.
    """

    tof_mapping = get_tof_columns()
    df_processed = df.copy()

    print(f"Processing tof df for all {len(df)} rows...")
    # Loop through the 5 sensors (this loop is fine, it only runs 5 times)
    for sensor_id, cols in tof_mapping.items():
        
        
        # 1. Extract the entire data block for the sensor at once.
        # Shape will be (8000, 64)
        sensor_data_block = df_processed[cols].to_numpy()
        
        # 2. Pre-allocate an array for the results
        processed_block = np.empty_like(sensor_data_block, dtype=float)
        
        # 3. Loop over the NumPy array. This is MUCH faster than looping over a DataFrame.
        for i in range(sensor_data_block.shape[0]):
            processed_block[i, :] = _interpolate_block(sensor_data_block[i, :])
            
        # 4. Assign the processed data back to the DataFrame.
        df_processed[cols] = processed_block

    print("2D Interpolation complete.")
    
    return df_processed