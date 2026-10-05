"""Overshoot-aware finite-size Lyapunov exponent calculations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

REACHED = "reached"
RIGHT_CENSORED_TRACK_END = "right_censored_track_end"
RIGHT_CENSORED_GAP = "right_censored_gap"
NOT_OBSERVED_IN_SHELL = "not_observed_in_shell"
NOT_OBSERVED_BEFORE_GAP = "not_observed_before_gap"


def build_scale_thresholds(
    minimum_km: float,
    maximum_km: float,
    rho: float,
    anchor_km: float = 1.0,
) -> np.ndarray:
    """Build geometric thresholds that contain ``anchor_km`` exactly."""
    values = (minimum_km, maximum_km, rho, anchor_km)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("FSLE scales, rho, and anchor must be finite and positive")
    if maximum_km <= minimum_km:
        raise ValueError("FSLE maximum scale must exceed the minimum scale")
    if rho <= 1:
        raise ValueError("FSLE rho must be greater than 1")
    if not minimum_km <= anchor_km <= maximum_km:
        raise ValueError("FSLE anchor scale must lie inside the configured scale range")

    log_rho = math.log(rho)
    lower_power = math.ceil(math.log(minimum_km / anchor_km) / log_rho - 1e-12)
    upper_power = math.floor(math.log(maximum_km / anchor_km) / log_rho + 1e-12)
    powers = np.arange(lower_power, upper_power + 1, dtype=np.int64)
    thresholds = anchor_km * np.power(rho, powers.astype(float))
    thresholds[np.where(powers == 0)] = anchor_km
    if len(thresholds) < 2:
        raise ValueError("FSLE configuration must produce at least two thresholds")
    return thresholds


def _contiguous_prefix(
    times: np.ndarray,
    distance_km: np.ndarray,
    expected_interval_seconds: float,
) -> tuple[int, str]:
    if len(times) != len(distance_km):
        raise ValueError("FSLE time and distance arrays must have equal length")
    if not len(times):
        raise ValueError("FSLE pair trajectory is empty at its encounter")
    valid = (~np.isnat(times)) & np.isfinite(distance_km)
    if not valid[0]:
        raise ValueError("FSLE pair is invalid at encounter_observation")
    for index in range(1, len(times)):
        if not valid[index]:
            return index, "missing_coordinate_or_time"
        elapsed = float((times[index] - times[index - 1]) / np.timedelta64(1, "s"))
        if not math.isfinite(elapsed) or not math.isclose(
            elapsed,
            expected_interval_seconds,
            rel_tol=0,
            abs_tol=1e-6,
        ):
            return index, "time_step_mismatch"
    return len(times), "track_end"


def collect_pair_first_passages(
    *,
    pair_id: str,
    array_id: int,
    cluster_id: str,
    platform_code_1: str,
    platform_code_2: str,
    times: np.ndarray,
    distance_km: np.ndarray,
    thresholds_km: np.ndarray,
    expected_interval_seconds: float,
) -> pd.DataFrame:
    """Collect one ordered first-passage or censoring record per pair and shell.

    The input begins at the selected encounter. Analysis stops at the first
    missing coordinate, missing time, or unexpected time step and never restarts.
    A shell clock begins at the first stored sample in ``[delta, rho * delta)``;
    its exit is the first later stored sample strictly above the upper threshold.
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    distance_km = np.asarray(distance_km, dtype=float)
    thresholds_km = np.asarray(thresholds_km, dtype=float)
    if expected_interval_seconds <= 0 or not math.isfinite(expected_interval_seconds):
        raise ValueError("FSLE expected interval must be finite and positive")
    prefix_length, termination_reason = _contiguous_prefix(
        times,
        distance_km,
        expected_interval_seconds,
    )
    usable_times = times[:prefix_length]
    usable_distance = distance_km[:prefix_length]
    censor_time = usable_times[-1]
    rows: list[dict[str, Any]] = []
    for lower, upper in pairwise(thresholds_km):
        entry_indices = np.flatnonzero(
            (usable_distance >= lower) & (usable_distance < upper),
        )
        common = {
            "pair_id": str(pair_id),
            "array_id": int(array_id),
            "cluster_id": str(cluster_id),
            "platform_code_1": str(platform_code_1),
            "platform_code_2": str(platform_code_2),
            "scale_lower_km": float(lower),
            "scale_upper_km": float(upper),
            "termination_reason": termination_reason,
        }
        if not len(entry_indices):
            rows.append(
                {
                    **common,
                    "status": (
                        NOT_OBSERVED_BEFORE_GAP
                        if termination_reason != "track_end"
                        else NOT_OBSERVED_IN_SHELL
                    ),
                    "shell_entry_time": np.datetime64("NaT", "ns"),
                    "shell_exit_time": np.datetime64("NaT", "ns"),
                    "censor_time": censor_time,
                    "entry_distance_km": np.nan,
                    "exit_distance_km": np.nan,
                    "passage_time_days": np.nan,
                    "log_growth": np.nan,
                }
            )
            continue

        entry = int(entry_indices[0])
        exit_candidates = np.flatnonzero(
            (np.arange(len(usable_distance)) > entry) & (usable_distance > upper),
        )
        if not len(exit_candidates):
            rows.append(
                {
                    **common,
                    "status": (
                        RIGHT_CENSORED_GAP
                        if termination_reason != "track_end"
                        else RIGHT_CENSORED_TRACK_END
                    ),
                    "shell_entry_time": usable_times[entry],
                    "shell_exit_time": np.datetime64("NaT", "ns"),
                    "censor_time": censor_time,
                    "entry_distance_km": float(usable_distance[entry]),
                    "exit_distance_km": np.nan,
                    "passage_time_days": np.nan,
                    "log_growth": np.nan,
                }
            )
            continue

        exit_index = int(exit_candidates[0])
        duration_days = float(
            (usable_times[exit_index] - usable_times[entry]) / np.timedelta64(1, "D")
        )
        if duration_days <= 0 or not math.isfinite(duration_days):
            raise ValueError(f"FSLE pair {pair_id!r} has a nonpositive passage time")
        entry_distance = float(usable_distance[entry])
        exit_distance = float(usable_distance[exit_index])
        rows.append(
            {
                **common,
                "status": REACHED,
                "shell_entry_time": usable_times[entry],
                "shell_exit_time": usable_times[exit_index],
                "censor_time": np.datetime64("NaT", "ns"),
                "entry_distance_km": entry_distance,
                "exit_distance_km": exit_distance,
                "passage_time_days": duration_days,
                "log_growth": math.log(exit_distance / entry_distance),
            }
        )
    return pd.DataFrame(rows)


def _unique_platform_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return len(
        set(frame.platform_code_1.astype(str)) | set(frame.platform_code_2.astype(str))
    )


def _overshoot_fsle_standard_error(
    reached: pd.DataFrame,
    *,
    fsle_day_inverse: float,
    mean_passage_time_days: float,
) -> float:
    """Return the PDF-style standard error generalized to observed overshoot."""
    if len(reached) < 2:
        return np.nan
    passage_time = reached.passage_time_days.to_numpy(dtype=float)
    log_growth = reached.log_growth.to_numpy(dtype=float)
    second_moment = float(
        np.mean(np.square(log_growth) / passage_time) / mean_passage_time_days
    )
    variance = max(second_moment - fsle_day_inverse**2, 0.0)
    return math.sqrt(variance / len(reached))


def _spectrum_rows(
    frame: pd.DataFrame,
    thresholds_km: np.ndarray,
    *,
    array_id: int,
    scope: str,
    cluster_id: str | None,
    minimum_reached_pairs: int,
) -> Iterable[dict[str, Any]]:
    total_pairs = int(frame.pair_id.nunique()) if not frame.empty else 0
    for lower, upper in pairwise(thresholds_km):
        shell = frame[np.isclose(frame.scale_lower_km, lower, rtol=1e-12, atol=1e-12)]
        at_risk = shell[
            shell.status.isin(
                {
                    REACHED,
                    RIGHT_CENSORED_TRACK_END,
                    RIGHT_CENSORED_GAP,
                }
            )
        ]
        reached = shell[shell.status == REACHED]
        mean_time = float(reached.passage_time_days.mean()) if len(reached) else np.nan
        mean_log = float(reached.log_growth.mean()) if len(reached) else np.nan
        fsle = (
            mean_log / mean_time
            if math.isfinite(mean_log) and math.isfinite(mean_time) and mean_time > 0
            else np.nan
        )
        standard_error = (
            _overshoot_fsle_standard_error(
                reached,
                fsle_day_inverse=fsle,
                mean_passage_time_days=mean_time,
            )
            if math.isfinite(fsle) and math.isfinite(mean_time)
            else np.nan
        )
        yield {
            "array_id": int(array_id),
            "scope": scope,
            "cluster_id": cluster_id,
            "scale_lower_km": float(lower),
            "scale_upper_km": float(upper),
            "fsle_day_inverse": fsle,
            "fsle_standard_error_day_inverse": standard_error,
            "mean_log_growth": mean_log,
            "mean_passage_time_days": mean_time,
            "median_passage_time_days": (
                float(reached.passage_time_days.median()) if len(reached) else np.nan
            ),
            "std_passage_time_days": (
                float(reached.passage_time_days.std(ddof=1))
                if len(reached) > 1
                else np.nan
            ),
            "total_pair_count": total_pairs,
            "at_risk_pair_count": len(at_risk),
            "reached_pair_count": len(reached),
            "censored_pair_count": int(len(at_risk) - len(reached)),
            "censored_fraction": (
                float(1 - len(reached) / len(at_risk)) if len(at_risk) else np.nan
            ),
            "not_observed_pair_count": int(len(shell) - len(at_risk)),
            "at_risk_platform_count": _unique_platform_count(at_risk),
            "reached_platform_count": _unique_platform_count(reached),
            "plot_included": bool(len(reached) >= minimum_reached_pairs),
        }


def aggregate_fsle_spectra(
    passages: pd.DataFrame,
    thresholds_km: np.ndarray,
    *,
    array_ids: Iterable[int],
    minimum_reached_pairs: int,
) -> pd.DataFrame:
    """Build pooled and individual-cluster overshoot spectra."""
    if minimum_reached_pairs < 1:
        raise ValueError("minimum_reached_pairs must be positive")
    rows: list[dict[str, Any]] = []
    for array_id in sorted({int(value) for value in array_ids}):
        array_frame = passages[passages.array_id == array_id]
        rows.extend(
            _spectrum_rows(
                array_frame,
                thresholds_km,
                array_id=array_id,
                scope="same_cluster_pooled",
                cluster_id=None,
                minimum_reached_pairs=minimum_reached_pairs,
            )
        )
        for cluster_id in sorted(array_frame.cluster_id.astype(str).unique()):
            rows.extend(
                _spectrum_rows(
                    array_frame[array_frame.cluster_id.astype(str) == cluster_id],
                    thresholds_km,
                    array_id=array_id,
                    scope="individual_cluster",
                    cluster_id=cluster_id,
                    minimum_reached_pairs=minimum_reached_pairs,
                )
            )
    return pd.DataFrame(rows)
