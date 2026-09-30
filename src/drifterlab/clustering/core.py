"""Numerical operations for observed-start diagnostics and candidate clusters."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod


WGS84 = Geod(ellps="WGS84")


def _required_starts(starts: pd.DataFrame) -> None:
    required = {
        "platform_id", "observed_start_time", "observed_start_lon", "observed_start_lat",
    }
    missing = required - set(starts)
    if missing:
        raise ValueError(f"Observed starts are missing columns: {sorted(missing)}")
    if starts.empty:
        raise ValueError("Observed starts are empty")
    if starts.platform_id.astype(str).duplicated().any():
        raise ValueError("Observed starts contain duplicate platform identifiers")


def pairwise_start_metrics(starts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return WGS84 distance metres and absolute observed-start gaps in minutes."""
    _required_starts(starts)
    longitude = starts.observed_start_lon.to_numpy(dtype=float)
    latitude = starts.observed_start_lat.to_numpy(dtype=float)
    time = pd.to_datetime(starts.observed_start_time, utc=True).to_numpy(dtype="datetime64[ns]")
    if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
        raise ValueError("Observed starts contain non-finite coordinates")
    if np.any((longitude < -180) | (longitude >= 180) | (latitude < -90) | (latitude > 90)):
        raise ValueError("Observed starts contain invalid coordinates")
    if np.isnat(time).any():
        raise ValueError("Observed starts contain invalid timestamps")
    lon1 = np.broadcast_to(longitude[:, None], (len(starts), len(starts)))
    lat1 = np.broadcast_to(latitude[:, None], (len(starts), len(starts)))
    _forward, _backward, distance = WGS84.inv(lon1, lat1, lon1.T, lat1.T)
    distance = np.asarray(distance, dtype=float)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0.0)
    time_ns = time.astype(np.int64)
    gap = np.abs(time_ns[:, None] - time_ns[None, :]) / 60_000_000_000
    return distance, gap.astype(float)


def infer_start_cohorts(
    starts: pd.DataFrame, *, maximum_adjacent_gap_hours: float,
) -> pd.DataFrame:
    """Propose broad cohorts by splitting sorted starts at large adjacent gaps."""
    _required_starts(starts)
    if not math.isfinite(maximum_adjacent_gap_hours) or maximum_adjacent_gap_hours <= 0:
        raise ValueError("maximum_adjacent_gap_hours must be finite and positive")
    result = starts.copy()
    result["observed_start_time"] = pd.to_datetime(result.observed_start_time, utc=True)
    result = result.sort_values(["observed_start_time", "platform_id"], kind="stable").reset_index(drop=True)
    times = result.observed_start_time
    previous = times.diff().dt.total_seconds() / 3600
    following = (times.shift(-1) - times).dt.total_seconds() / 3600
    breaks = previous.gt(maximum_adjacent_gap_hours).fillna(False).to_numpy(dtype=bool)
    cohort_number = np.cumsum(breaks) + 1
    result["gap_from_previous_start_hours"] = previous
    result["gap_to_next_start_hours"] = following
    result["proposed_cohort_id"] = [f"cohort_{number:03d}" for number in cohort_number]
    return result


def nearest_neighbor_diagnostics(
    starts: pd.DataFrame, *, ranks: int = 5, cohort_column: str = "cohort_id",
) -> pd.DataFrame:
    """Return spatial neighbor ranks and their corresponding observed-start gaps."""
    _required_starts(starts)
    if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1:
        raise ValueError("ranks must be a positive integer")
    if cohort_column not in starts:
        raise ValueError(f"Observed starts have no {cohort_column!r} column")
    rows: list[dict[str, Any]] = []
    for cohort, raw_group in starts.groupby(cohort_column, sort=True, dropna=False):
        if pd.isna(cohort) or not str(cohort).strip():
            raise ValueError("Every included observed start needs a cohort")
        group = raw_group.sort_values("platform_id", kind="stable").reset_index(drop=True)
        if len(group) < 2:
            continue
        distances, time_gaps = pairwise_start_metrics(group)
        platforms = group.platform_id.astype(str).to_numpy()
        ordered: list[np.ndarray] = []
        inverse_ranks: list[dict[int, int]] = []
        for index in range(len(group)):
            candidates = np.flatnonzero(np.arange(len(group)) != index)
            order = candidates[np.lexsort((platforms[candidates], distances[index, candidates]))]
            ordered.append(order)
            inverse_ranks.append({int(other): rank for rank, other in enumerate(order, start=1)})
        for index, order in enumerate(ordered):
            for rank, other in enumerate(order[:ranks], start=1):
                rows.append({
                    "cohort_id": str(cohort),
                    "platform_id": platforms[index],
                    "observed_start_time_utc": _utc(group.at[index, "observed_start_time"]),
                    "neighbor_rank": rank,
                    "neighbor_platform_id": platforms[other],
                    "neighbor_observed_start_time_utc": _utc(group.at[other, "observed_start_time"]),
                    "distance_m": float(distances[index, other]),
                    "start_time_gap_minutes": float(time_gaps[index, other]),
                    "reciprocal_neighbor_rank": inverse_ranks[other][index],
                })
    columns = [
        "cohort_id", "platform_id", "observed_start_time_utc", "neighbor_rank",
        "neighbor_platform_id", "neighbor_observed_start_time_utc", "distance_m",
        "start_time_gap_minutes", "reciprocal_neighbor_rank",
    ]
    return pd.DataFrame(rows, columns=columns)


def candidate_clusters(
    starts: pd.DataFrame, *, maximum_diameter_m: float,
    maximum_start_spread_minutes: float, maximum_members: int,
    cohort_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition one cohort using capacity-constrained joint complete linkage."""
    _required_starts(starts)
    if not math.isfinite(maximum_diameter_m) or maximum_diameter_m <= 0:
        raise ValueError("maximum_diameter_m must be finite and positive")
    if not math.isfinite(maximum_start_spread_minutes) or maximum_start_spread_minutes <= 0:
        raise ValueError("maximum_start_spread_minutes must be finite and positive")
    if isinstance(maximum_members, bool) or not isinstance(maximum_members, int) or maximum_members < 1:
        raise ValueError("maximum_members must be a positive integer")
    group = starts.sort_values("platform_id", kind="stable").reset_index(drop=True).copy()
    distances, time_gaps = pairwise_start_metrics(group)
    platforms = group.platform_id.astype(str).to_numpy()
    clusters: list[tuple[int, ...]] = [(index,) for index in range(len(group))]
    merge_rows: list[dict[str, Any]] = []
    step = 0
    while True:
        choices: list[tuple[tuple[Any, ...], int, int, tuple[int, ...], float, float, float]] = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                merged = tuple(sorted((*clusters[left], *clusters[right])))
                if len(merged) > maximum_members:
                    continue
                cells = np.ix_(merged, merged)
                diameter = float(np.max(distances[cells]))
                time_span = float(np.max(time_gaps[cells]))
                score = max(
                    diameter / maximum_diameter_m,
                    time_span / maximum_start_spread_minutes,
                )
                if score > 1 + 1e-12:
                    continue
                member_ids = tuple(sorted(platforms[list(merged)].tolist()))
                key = (score, diameter, time_span, member_ids)
                choices.append((key, left, right, merged, diameter, time_span, score))
        if not choices:
            break
        _key, left, right, merged, diameter, time_span, score = min(choices, key=lambda item: item[0])
        step += 1
        merge_rows.append({
            "cohort_id": cohort_id,
            "merge_step": step,
            "left_members": json.dumps(sorted(platforms[list(clusters[left])].tolist()), separators=(",", ":")),
            "right_members": json.dumps(sorted(platforms[list(clusters[right])].tolist()), separators=(",", ":")),
            "merged_members": json.dumps(sorted(platforms[list(merged)].tolist()), separators=(",", ":")),
            "member_count": len(merged),
            "diameter_m": diameter,
            "observed_start_spread_minutes": time_span,
            "joint_complete_linkage_score": score,
        })
        clusters = [value for index, value in enumerate(clusters) if index not in {left, right}]
        clusters.append(merged)

    time_ns = pd.to_datetime(group.observed_start_time, utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    clusters.sort(key=lambda members: (int(np.min(time_ns[list(members)])), min(platforms[list(members)])))
    member_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for number, members in enumerate(clusters, start=1):
        cluster_id = f"{cohort_id}__cluster_{number:03d}"
        cells = np.ix_(members, members)
        diameter = float(np.max(distances[cells]))
        time_span = float(np.max(time_gaps[cells]))
        subset = group.iloc[list(members)]
        start_times = pd.to_datetime(subset.observed_start_time, utc=True)
        status = "singleton" if len(members) == 1 else "candidate"
        summary_rows.append({
            "cohort_id": cohort_id,
            "candidate_cluster_id": cluster_id,
            "member_count": len(members),
            "members_json": json.dumps(sorted(subset.platform_id.astype(str).tolist()), separators=(",", ":")),
            "diameter_m": diameter,
            "maximum_diameter_m": maximum_diameter_m,
            "diameter_fraction": diameter / maximum_diameter_m,
            "earliest_observed_start_utc": _utc(start_times.min()),
            "latest_observed_start_utc": _utc(start_times.max()),
            "observed_start_spread_minutes": time_span,
            "maximum_observed_start_spread_minutes": maximum_start_spread_minutes,
            "start_spread_fraction": time_span / maximum_start_spread_minutes,
            "maximum_members": maximum_members,
            "candidate_status": status,
        })
        for index in members:
            row = group.iloc[index]
            member_rows.append({
                "cohort_id": cohort_id,
                "candidate_cluster_id": cluster_id,
                "platform_id": str(row.platform_id),
                "observed_start_time_utc": _utc(row.observed_start_time),
                "observed_start_lon": float(row.observed_start_lon),
                "observed_start_lat": float(row.observed_start_lat),
                "member_count": len(members),
                "cluster_diameter_m": diameter,
                "cluster_observed_start_spread_minutes": time_span,
                "candidate_status": status,
            })
    return (
        pd.DataFrame(member_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(merge_rows, columns=[
            "cohort_id", "merge_step", "left_members", "right_members",
            "merged_members", "member_count", "diameter_m",
            "observed_start_spread_minutes", "joint_complete_linkage_score",
        ]),
    )


def _utc(value: Any) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")
