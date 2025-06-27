#!/usr/bin/env python3
"""Utility functions for 2-D interpolation of 8×8 TOF sensor grids."""
import numpy as np

_TOF_SENSORS = [1, 2, 3, 4, 5]


def get_tof_columns() -> dict:
    """Return a dict mapping sensor_id -> column list names for the 64 pixels."""
    mapping = {}
    for sid in _TOF_SENSORS:
        cols = [f"tof_{sid}_v{i}" for i in range(64)]
        mapping[sid] = cols
    return mapping


def _interpolate_block(values: np.ndarray) -> np.ndarray:
    """Interpolate a single 8×8 grid (values.shape == (64,)).

    NaNs are filled by averaging valid 8-neighbour cells; repeat up to 3 passes.
    Remaining NaNs are replaced by the median of the grid.
    """
    grid = values.reshape(8, 8).astype(float)
    mask = np.isnan(grid)
    if not mask.any():
        return values  # nothing to do

    for _ in range(3):
        new_grid = grid.copy()
        for i in range(8):
            for j in range(8):
                if mask[i, j]:
                    neigh = grid[max(0, i - 1): i + 2, max(0, j - 1): j + 2]
                    neigh_valid = neigh[~np.isnan(neigh)]
                    if neigh_valid.size:
                        new_grid[i, j] = neigh_valid.mean()
        grid = new_grid
        mask = np.isnan(grid)
        if not mask.any():
            break

    # final fallback – median of valid pixels, else zero
    if np.isnan(grid).any():
        median_val = np.nanmedian(grid)
        if np.isnan(median_val):
            median_val = 0.0
        grid[np.isnan(grid)] = median_val

    return grid.reshape(-1)


def interpolate_tof_row(row, mapping=None):
    """Interpolate all TOF sensor blocks within a pandas Series row.

    Parameters
    ----------
    row : pandas.Series
    mapping : dict sensor_id -> column list. If None will call get_tof_columns().
    Returns the modified row (in-place modification for speed).
    """
    if mapping is None:
        mapping = get_tof_columns()

    for sid, cols in mapping.items():
        sub = row[cols]
        if sub.isna().any():
            arr = sub.to_numpy()
            row[cols] = _interpolate_block(arr)
    return row 