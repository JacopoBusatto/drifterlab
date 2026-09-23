"""Compute advisory time and speed flags without changing observations."""

from dataclasses import dataclass

import numpy as np

from .position import position_valid

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class AuditConfig:
    nominal_interval_seconds: float = 300.0
    interval_tolerance_seconds: float = 1.0
    large_gap_seconds: float = 1800.0
    residual_jump_speed_m_s: float = 3.0


def interval_seconds(time: np.ndarray) -> np.ndarray:
    result = np.full(len(time), np.nan)
    pairs = np.flatnonzero(~np.isnat(time[:-1]) & ~np.isnat(time[1:]))
    # Subtract microsecond endpoints to avoid int64 nanosecond overflow for
    # valid dates centuries apart, then restore the nanosecond remainders.
    if len(pairs):
        delta = (time[pairs + 1].astype("datetime64[us]") - time[pairs].astype("datetime64[us]"))
        result[pairs + 1] = delta / np.timedelta64(1, "s")
        # Correct the sub-microsecond remainder without subtracting int64 epochs.
        remainder = time.astype("datetime64[ns]").astype(np.int64) % 1000
        result[pairs + 1] += (remainder[pairs + 1] - remainder[pairs]) / 1e9
    return result


def consecutive_speed(time: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Haversine speed assigned to the later row; no bridging missing fixes."""
    result = np.full(len(time), np.nan)
    dt = interval_seconds(time)
    good = position_valid(lon, lat) & ~np.isnat(time)
    right = np.flatnonzero(good[:-1] & good[1:] & (dt[1:] > 0)) + 1
    if not len(right):
        return result
    left = right - 1
    phi1, phi2 = np.radians(lat[left]), np.radians(lat[right])
    dphi = phi2 - phi1
    dlon = np.radians(lon[right] - lon[left])
    hav = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
    result[right] = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0, 1))) / dt[right]
    return result


def audit_time(time: np.ndarray, config: AuditConfig) -> dict[str, np.ndarray]:
    dt = interval_seconds(time)
    duplicate = np.zeros(len(time), dtype=bool)
    indices = np.flatnonzero(~np.isnat(time))
    if len(indices):
        _, inverse, counts = np.unique(time[indices], return_inverse=True, return_counts=True)
        duplicate[indices] = counts[inverse] > 1
    return {
        "invalid_time_flag": np.isnat(time),
        "duplicate_time_flag": duplicate,
        "interval_seconds": dt,
        "sampling_interval_flag": np.isfinite(dt) & (
            np.abs(dt - config.nominal_interval_seconds) > config.interval_tolerance_seconds
        ),
        "large_gap_flag": np.isfinite(dt) & (
            dt > config.large_gap_seconds + config.interval_tolerance_seconds
        ),
    }
