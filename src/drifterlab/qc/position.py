"""Position and combined analysis-validity masks."""

import numpy as np


def position_valid(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)


def analysis_valid(positions: np.ndarray, drogue: np.ndarray, time: np.ndarray) -> np.ndarray:
    return positions & drogue & ~np.isnat(time)
