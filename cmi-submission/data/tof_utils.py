# This file is copied from development/data/tof_utils.py so that the submission
# package is self-contained when uploaded to Kaggle.

#!/usr/bin/env python3
"""Utility functions for 2-D interpolation of the 8×8 TOF sensor grids."""
import numpy as np

_TOF_SENSORS = [1, 2, 3, 4, 5]


def get_tof_columns() -> dict:
    """Return a mapping: sensor_id -> list of 64 column names (tof_X_v0 … v63)."""
    mapping = {}
    for sid in _TOF_SENSORS:
        mapping[sid] = [f"tof_{sid}_v{i}" for i in range(64)]
    return mapping


def _interpolate_block(values: np.ndarray) -> np.ndarray:
    """Fill NaNs of a single 8×8 grid by neighbour averaging (≤3 passes)."""
    grid = values.reshape(8, 8).astype(float)
    mask = np.isnan(grid)
    if not mask.any():
        return values

    for _ in range(3):
        new_grid = grid.copy()
        for i in range(8):
            for j in range(8):
                if mask[i, j]:
                    neigh = grid[max(0, i - 1): i + 2, max(0, j - 1): j + 2]
                    valid = neigh[~np.isnan(neigh)]
                    if valid.size:
                        new_grid[i, j] = valid.mean()
        grid = new_grid
        mask = np.isnan(grid)
        if not mask.any():
            break

    if np.isnan(grid).any():
        median_val = np.nanmedian(grid)
        if np.isnan(median_val):
            median_val = 0.0
        grid[np.isnan(grid)] = median_val

    return grid.reshape(-1)


def interpolate_tof_row(row, mapping=None):
    """Apply 2-D interpolation to every TOF sensor block in a pandas row."""
    if mapping is None:
        mapping = get_tof_columns()
    for cols in mapping.values():
        sub = row[cols]
        if sub.isna().any():
            row[cols] = _interpolate_block(sub.to_numpy())
    return row 