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


PASSAGE_POLICIES = {"strict_observed", "bridge_missing"}


def collect_observed_first_passages(
    *,
    times: np.ndarray,
    distance_km: np.ndarray,
    thresholds_km: np.ndarray,
    expected_interval_seconds: float,
    policy: str = "strict_observed",
    source_gap_minutes: np.ndarray | None = None,
    maximum_source_gap_minutes: float | None = None,
    **identity: Any,
) -> pd.DataFrame:
    """Collect v2 passages, requiring a witnessed outward shell entry.

    An entry is the first sample in [lower, upper) immediately preceded by a
    valid sample below lower. An exact lower threshold at the analysis start
    also establishes an entry. Entry timestamps and distances are sampled,
    not interpolated. Exits reach or exceed upper. Strict passages cannot cross
    a missing sample; bridge_missing may close an existing clock across missing
    samples, but cannot establish a new entry across that gap. Higher shells can
    acquire fresh entries after a gap. A shell is attempted at most once.

    A finite maximum source gap is a hard limit in both policies: rejected
    source support cannot be bridged. It also limits elapsed gaps between valid
    observations in bridge_missing. None adds no source-gap restriction.
    """
    if policy not in PASSAGE_POLICIES:
        raise ValueError(f"Unknown FSLE passage policy: {policy}")
    times = np.asarray(times, dtype="datetime64[ns]")
    distance = np.asarray(distance_km, dtype=float)
    thresholds = np.asarray(thresholds_km, dtype=float)
    if times.ndim != 1 or distance.shape != times.shape or not len(times):
        raise ValueError("FSLE needs equal, nonempty one-dimensional times and distances")
    if np.isnat(times).any() or np.any(np.diff(times.astype(np.int64)) <= 0):
        raise ValueError("FSLE times must be finite and strictly increasing")
    if not math.isfinite(expected_interval_seconds) or expected_interval_seconds <= 0:
        raise ValueError("FSLE expected interval must be finite and positive")
    if (
        thresholds.ndim != 1 or len(thresholds) < 2
        or not np.isfinite(thresholds).all() or np.any(thresholds <= 0)
        or np.any(np.diff(thresholds) <= 0)
    ):
        raise ValueError("FSLE thresholds must be positive and strictly increasing")
    hard_invalid = np.zeros(len(times), dtype=bool)
    if maximum_source_gap_minutes is not None:
        if not math.isfinite(maximum_source_gap_minutes) or maximum_source_gap_minutes <= 0:
            raise ValueError("FSLE maximum source gap must be finite and positive or null")
        if source_gap_minutes is None:
            raise ValueError("A source-gap limit requires source_gap_minutes")
        source_gap = np.asarray(source_gap_minutes, dtype=float)
        if source_gap.shape != times.shape:
            raise ValueError("FSLE source gaps must match times")
        hard_invalid = np.isfinite(distance) & (
            ~np.isfinite(source_gap) | (source_gap > maximum_source_gap_minutes)
        )
    valid = np.isfinite(distance) & (distance >= 0) & ~hard_invalid
    steps = np.diff(times) / np.timedelta64(1, "s")
    adjacent = np.isclose(steps, expected_interval_seconds, rtol=0, atol=1e-6)
    witnessed = np.zeros(len(times), dtype=bool)
    witnessed[1:] = valid[:-1] & valid[1:] & adjacent
    rows = []
    for lower, upper in pairwise(thresholds):
        inside = valid & (distance >= lower) & (distance < upper)
        entries = np.zeros(len(times), dtype=bool)
        entries[1:] = inside[1:] & witnessed[1:] & (distance[:-1] < lower)
        entries[0] = inside[0] and distance[0] == lower
        entry_indices = np.flatnonzero(entries)
        row = {
            **identity,
            "passage_policy": policy,
            "scale_lower_km": float(lower), "scale_upper_km": float(upper),
            "status": "not_observed_entry", "termination_reason": "no_witnessed_entry",
            "shell_entry_time": np.datetime64("NaT", "ns"),
            "entry_preceding_time": np.datetime64("NaT", "ns"),
            "shell_exit_time": np.datetime64("NaT", "ns"),
            "censor_time": np.datetime64("NaT", "ns"),
            "entry_distance_km": np.nan, "exit_distance_km": np.nan,
            "passage_time_days": np.nan, "log_growth": np.nan,
            "missing_observation_count": 0, "maximum_observation_gap_minutes": np.nan,
        }
        if not len(entry_indices):
            rows.append(row)
            continue
        entry = int(entry_indices[0])
        row.update({
            "shell_entry_time": times[entry], "entry_distance_km": float(distance[entry]),
            "entry_preceding_time": times[entry - 1] if entry else np.datetime64("NaT", "ns"),
        })
        exits = np.flatnonzero(valid & (distance >= upper) & (np.arange(len(times)) > entry))
        exit_index = int(exits[0]) if len(exits) else None
        end = exit_index if exit_index is not None else len(times) - 1
        segment = np.arange(entry, end + 1)
        observed = segment[valid[segment]]
        gaps = np.diff(times[observed]) / np.timedelta64(1, "m")
        max_gap = float(gaps.max()) if len(gaps) else 0.0
        row["missing_observation_count"] = int((~valid[segment]).sum())
        row["maximum_observation_gap_minutes"] = max_gap
        failures: list[tuple[int, str]] = []
        bad_source = segment[hard_invalid[segment]]
        if len(bad_source):
            failures.append((int(bad_source[0]), "source_gap_limit"))
        if policy == "strict_observed":
            bad = segment[~valid[segment]]
            if len(bad):
                failures.append((int(bad[0]), "missing_pair_position"))
            bad_step = np.flatnonzero(~adjacent[entry:end])
            if len(bad_step):
                failures.append((entry + int(bad_step[0]) + 1, "time_step_mismatch"))
        elif maximum_source_gap_minutes is not None:
            excessive = np.flatnonzero(gaps > maximum_source_gap_minutes)
            if len(excessive):
                failures.append((int(observed[int(excessive[0]) + 1]), "elapsed_gap_limit"))
        if failures:
            failure_index, reason = min(failures, key=lambda item: item[0])
            before_failure = observed[observed < failure_index]
            row.update({
                "status": RIGHT_CENSORED_GAP, "termination_reason": reason,
                "censor_time": times[before_failure[-1]] if len(before_failure) else times[entry],
            })
        elif exit_index is None:
            row.update({
                "status": RIGHT_CENSORED_TRACK_END, "termination_reason": "track_end",
                "censor_time": times[observed[-1]],
            })
        else:
            row.update({
                "status": REACHED, "termination_reason": "upper_threshold_reached",
                "shell_exit_time": times[exit_index],
                "exit_distance_km": float(distance[exit_index]),
                "passage_time_days": float((times[exit_index] - times[entry]) / np.timedelta64(1, "D")),
                "log_growth": math.log(distance[exit_index] / distance[entry]),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_fsle_populations(
    passages: pd.DataFrame,
    thresholds_km: np.ndarray,
    *,
    array_id: int,
    cluster_ids: Iterable[str],
    minimum_reached_pairs: int,
) -> pd.DataFrame:
    """Summarize two disjoint array populations and every selected cluster."""
    rows = []
    for population, scope in (
        ("within_cluster", "within_cluster_pooled"),
        ("cross_cluster", "cross_cluster"),
    ):
        frame = passages[passages.population == population]
        rows.extend(_spectrum_rows(
            frame, thresholds_km, array_id=array_id, scope=scope, cluster_id=None,
            minimum_reached_pairs=minimum_reached_pairs,
        ))
    for cluster in sorted(set(cluster_ids)):
        frame = passages[(passages.population == "within_cluster") & (passages.cluster_id == cluster)]
        rows.extend(_spectrum_rows(
            frame, thresholds_km, array_id=array_id, scope="individual_cluster", cluster_id=cluster,
            minimum_reached_pairs=minimum_reached_pairs,
        ))
    return pd.DataFrame(rows)


def fit_fsle_power_laws(spectra: pd.DataFrame, fits: list[dict[str, Any]]) -> pd.DataFrame:
    """Fit log(lambda_s^-1) = log(A_SI) + power * log(delta_m).

    Equal weight is assigned to supported shells wholly inside each fit range.
    At least two shells are required. No independence-based confidence interval
    or conversion to epsilon is implied; A^3 is saved only for power -2/3.
    """
    columns = [
        "array_id", "scope", "cluster_id", "fit_id", "power", "scale_min_km", "scale_max_km",
        "status", "shell_count", "used_scale_lower_km", "used_scale_upper_km",
        "coefficient_si", "coefficient_units", "coefficient_cubed_m2_s3",
        "log_rmse", "actual_scale_min_km", "actual_scale_max_km",
    ]
    rows = []
    for (array_id, scope, cluster_id), group in spectra.groupby(
        ["array_id", "scope", "cluster_id"], dropna=False, sort=False,
    ):
        for spec in fits:
            lower = group.scale_lower_km.to_numpy(dtype=float)
            upper = group.scale_upper_km.to_numpy(dtype=float)
            eligible = (
                group.plot_included.astype(bool).to_numpy()
                & np.isfinite(group.fsle_day_inverse) & (group.fsle_day_inverse > 0)
                & ((lower >= spec["scale_min_km"]) | np.isclose(lower, spec["scale_min_km"], rtol=1e-12, atol=0))
                & ((upper <= spec["scale_max_km"]) | np.isclose(upper, spec["scale_max_km"], rtol=1e-12, atol=0))
            )
            selected = group[eligible]
            power = spec["power"]
            row = {
                "array_id": int(array_id), "scope": scope,
                "cluster_id": None if pd.isna(cluster_id) else cluster_id,
                "fit_id": spec["id"], "power": power,
                "scale_min_km": spec["scale_min_km"], "scale_max_km": spec["scale_max_km"],
                "status": "insufficient_shells", "shell_count": len(selected),
                "used_scale_lower_km": selected.scale_lower_km.tolist(),
                "used_scale_upper_km": selected.scale_upper_km.tolist(),
                "coefficient_si": np.nan, "coefficient_units": f"m^({-power:g}) s^-1",
                "coefficient_cubed_m2_s3": np.nan, "log_rmse": np.nan,
                "actual_scale_min_km": float(selected.scale_lower_km.min()) if len(selected) else np.nan,
                "actual_scale_max_km": float(selected.scale_upper_km.max()) if len(selected) else np.nan,
            }
            if len(selected) >= 2:
                log_x = np.log(selected.scale_lower_km.to_numpy(dtype=float) * 1000)
                log_y = np.log(selected.fsle_day_inverse.to_numpy(dtype=float) / 86400)
                log_a = float(np.mean(log_y - power * log_x))
                coefficient = math.exp(log_a)
                row.update({
                    "status": "fitted", "coefficient_si": coefficient,
                    "log_rmse": float(np.sqrt(np.mean(np.square(log_y - log_a - power * log_x)))),
                    "coefficient_cubed_m2_s3": coefficient**3 if math.isclose(power, -2 / 3, abs_tol=1e-12, rel_tol=0) else np.nan,
                })
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)
