"""Deterministic candidate deployment clustering with dual distance diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Mapping, Sequence

import numpy as np
from pyproj import Geod


WGS84 = Geod(ellps="WGS84")
VALID_ASSIGNMENTS = {"infer", "single_cluster"}
VALID_DISTANCE_REFERENCES = {"observed_starts", "first_common_grid"}


@dataclass(frozen=True)
class InitialClusterRule:
    """Effective rules for one array's initial cluster assignment."""

    assignment: str
    maximum_start_time_difference_seconds: float
    maximum_pair_distance_m: float
    maximum_cluster_diameter_m: float
    maximum_members: int | None


@dataclass(frozen=True)
class ClusterPairDiagnostic:
    """Both deployment-oriented and time-aligned pairwise distances."""

    platform_a: str
    platform_b: str
    array_id: int
    start_time_difference_seconds: float
    first_common_time: np.datetime64 | None
    observed_start_distance_m: float | None
    first_common_grid_distance_m: float | None
    distance_reference: str
    distance_m: float | None
    candidate_link: bool


@dataclass(frozen=True)
class InitialClusterAssignment:
    """Cluster metadata aligned to one platform."""

    platform_id: str
    array_id: int
    cluster_id: str
    member_id: int
    cluster_size: int
    cluster_assignment_status: str
    cluster_start_spread_seconds: float
    cluster_diameter_m: float | None
    cluster_maximum_available_pair_distance_m: float | None
    cluster_observed_start_diameter_m: float | None
    cluster_first_common_grid_diameter_m: float | None
    cluster_first_common_grid_unavailable_pair_count: int
    cluster_candidate_link_count: int
    cluster_unavailable_pair_count: int


@dataclass(frozen=True)
class InitialClusterResult:
    """Assignments plus auditable array and pair diagnostics."""

    assignments: tuple[InitialClusterAssignment, ...]
    pair_diagnostics: tuple[ClusterPairDiagnostic, ...]
    array_summary: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _PairMetric:
    left: int
    right: int
    start_gap_seconds: float
    first_common_time: np.datetime64 | None
    observed_start_distance_m: float | None
    first_common_grid_distance_m: float | None
    distance_m: float | None
    candidate_link: bool


def validate_cluster_rule(rule: InitialClusterRule, *, name: str = "cluster rule") -> None:
    """Validate one fully inherited cluster rule."""
    if rule.assignment not in VALID_ASSIGNMENTS:
        raise ValueError(f"{name}.assignment must be 'infer' or 'single_cluster'")
    for field, value in (
        ("maximum_start_time_difference_seconds", rule.maximum_start_time_difference_seconds),
        ("maximum_pair_distance_m", rule.maximum_pair_distance_m),
        ("maximum_cluster_diameter_m", rule.maximum_cluster_diameter_m),
    ):
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name}.{field} must be finite and positive")
    if rule.maximum_cluster_diameter_m < rule.maximum_pair_distance_m:
        raise ValueError(
            f"{name}.maximum_cluster_diameter_m must be at least "
            "maximum_pair_distance_m"
        )
    if (rule.maximum_members is not None
            and (isinstance(rule.maximum_members, bool) or rule.maximum_members < 1)):
        raise ValueError(f"{name}.maximum_members must be null or a positive integer")


def _distance_m(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    _, _, value = WGS84.inv(lon_a, lat_a, lon_b, lat_b)
    return float(value)


def _member_key(index: int, platform_ids: Sequence[str], starts: np.ndarray) -> tuple[int, str]:
    return int(starts[index].astype(np.int64)), platform_ids[index]


def _cluster_statistics(
    members: tuple[int, ...], starts: np.ndarray,
    metrics: Mapping[tuple[int, int], _PairMetric],
) -> tuple[float, float | None, float | None, int, int]:
    member_starts = starts[np.asarray(members, dtype=int)].astype("datetime64[ns]")
    spread = float((member_starts.max() - member_starts.min()) / np.timedelta64(1, "s"))
    distances: list[float] = []
    links = 0
    unavailable = 0
    for left, right in combinations(members, 2):
        metric = metrics[min(left, right), max(left, right)]
        if metric.distance_m is None:
            unavailable += 1
        else:
            distances.append(metric.distance_m)
        links += int(metric.candidate_link)
    maximum_available = max(distances) if distances else (
        0.0 if len(members) == 1 else None
    )
    diameter = None if unavailable else maximum_available
    return spread, diameter, maximum_available, links, unavailable


def _diagnostic_diameter(
    members: tuple[int, ...], metrics: Mapping[tuple[int, int], _PairMetric],
    attribute: str,
) -> tuple[float | None, int]:
    if len(members) == 1:
        return 0.0, 0
    values: list[float] = []
    unavailable = 0
    for left, right in combinations(members, 2):
        value = getattr(metrics[min(left, right), max(left, right)], attribute)
        if value is None:
            unavailable += 1
        else:
            values.append(float(value))
    return (None if unavailable else max(values)), unavailable


def _infer_array_clusters(
    members: tuple[int, ...], platform_ids: Sequence[str], starts: np.ndarray,
    metrics: Mapping[tuple[int, int], _PairMetric], rule: InitialClusterRule,
) -> list[tuple[int, ...]]:
    clusters = [(index,) for index in members]
    while True:
        choices: list[
            tuple[
                tuple[float, float, float, tuple[str, ...]], int, int, tuple[int, ...]
            ]
        ] = []
        for left_cluster, right_cluster in combinations(range(len(clusters)), 2):
            left = clusters[left_cluster]
            right = clusters[right_cluster]
            joining = [
                metrics[min(i, j), max(i, j)]
                for i in left for j in right
                if metrics[min(i, j), max(i, j)].candidate_link
            ]
            if not joining:
                continue
            merged = tuple(sorted((*left, *right)))
            if rule.maximum_members is not None and len(merged) > rule.maximum_members:
                continue
            spread, diameter, _, _, unavailable = _cluster_statistics(
                merged, starts, metrics,
            )
            if (spread > rule.maximum_start_time_difference_seconds
                    or diameter is None or unavailable
                    or diameter > rule.maximum_cluster_diameter_m):
                continue
            joining_distance = min(
                metric.distance_m for metric in joining if metric.distance_m is not None
            )
            platforms = tuple(sorted(platform_ids[index] for index in merged))
            key = (joining_distance, diameter, spread, platforms)
            choices.append((key, left_cluster, right_cluster, merged))
        if not choices:
            return clusters
        _, left_cluster, right_cluster, merged = min(choices, key=lambda item: item[0])
        clusters = [
            cluster for index, cluster in enumerate(clusters)
            if index not in {left_cluster, right_cluster}
        ]
        clusters.append(merged)


def assign_initial_clusters(
    platform_ids: Sequence[str], array_ids: Sequence[int], start_times: Sequence[np.datetime64],
    grid: Sequence[np.datetime64], longitude: np.ndarray, latitude: np.ndarray,
    default_rule: InitialClusterRule,
    array_overrides: Mapping[int, InitialClusterRule] | None = None,
    *, distance_reference: str = "first_common_grid",
    start_longitude: Sequence[float] | None = None,
    start_latitude: Sequence[float] | None = None,
) -> InitialClusterResult:
    """Assign candidate deployment clusters without modifying trajectory coverage.

    Assignment may use distances between observed starts or distances at the first
    common finite grid instant. Both are retained as diagnostics when supplied.
    """
    count = len(platform_ids)
    if not count:
        raise ValueError("At least one platform is required for cluster assignment")
    if len(set(platform_ids)) != count or any(not str(value) for value in platform_ids):
        raise ValueError("platform_ids must be unique nonempty strings")
    if len(array_ids) != count or len(start_times) != count:
        raise ValueError("Cluster inputs must align with platform_ids")
    starts = np.asarray(start_times, dtype="datetime64[ns]")
    times = np.asarray(grid, dtype="datetime64[ns]")
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    if distance_reference not in VALID_DISTANCE_REFERENCES:
        raise ValueError(
            "distance_reference must be 'observed_starts' or 'first_common_grid'"
        )
    if (start_longitude is None) != (start_latitude is None):
        raise ValueError("start_longitude and start_latitude must be supplied together")
    start_lon = (
        None if start_longitude is None else np.asarray(start_longitude, dtype=float)
    )
    start_lat = (
        None if start_latitude is None else np.asarray(start_latitude, dtype=float)
    )
    if start_lon is not None:
        if start_lon.shape != (count,) or start_lat.shape != (count,):
            raise ValueError("Observed-start coordinates must align with platform_ids")
        if (not np.isfinite(start_lon).all() or not np.isfinite(start_lat).all()
                or np.any((start_lon < -180) | (start_lon > 180))
                or np.any((start_lat < -90) | (start_lat > 90))):
            raise ValueError("Observed-start coordinates contain invalid values")
    if distance_reference == "observed_starts" and start_lon is None:
        raise ValueError("observed_starts distance_reference requires start coordinates")
    if np.isnat(starts).any() or np.isnat(times).any() or len(times) == 0:
        raise ValueError("Cluster start times and grid times must be valid")
    if len(times) > 1 and np.any(np.diff(times.astype(np.int64)) <= 0):
        raise ValueError("Cluster grid must be strictly increasing")
    if lon.shape != (count, len(times)) or lat.shape != lon.shape:
        raise ValueError("Cluster coordinates must have shape (platform, time)")
    if not np.array_equal(np.isfinite(lon), np.isfinite(lat)):
        raise ValueError("Cluster longitude and latitude finite masks differ")
    finite = np.isfinite(lon)
    if np.any(finite & ((lon < -180) | (lon >= 180))):
        raise ValueError("Cluster longitudes must be in [-180, 180)")
    if np.any(finite & ((lat < -90) | (lat > 90))):
        raise ValueError("Cluster latitudes must be in [-90, 90]")
    if np.any(~finite.any(axis=1)):
        raise ValueError("Every platform must have at least one finite cluster coordinate")
    parsed_arrays = np.asarray(array_ids, dtype=np.int64)
    if np.any(parsed_arrays < 1):
        raise ValueError("array_ids must be positive integers")

    validate_cluster_rule(default_rule, name="initial_clustering.defaults")
    overrides = dict(array_overrides or {})
    unknown = set(overrides) - set(parsed_arrays.tolist())
    if unknown:
        raise ValueError(f"Cluster overrides name absent arrays: {sorted(unknown)}")
    for array_id, rule in overrides.items():
        validate_cluster_rule(rule, name=f"initial_clustering.array_overrides.array_{array_id:03d}")

    pair_metrics: dict[tuple[int, int], _PairMetric] = {}
    public_pairs: list[ClusterPairDiagnostic] = []
    assignments_by_index: dict[int, InitialClusterAssignment] = {}
    summaries: list[dict[str, object]] = []
    for array_id in sorted(set(parsed_arrays.tolist())):
        rule = overrides.get(array_id, default_rule)
        members = tuple(np.flatnonzero(parsed_arrays == array_id).tolist())
        for left, right in combinations(members, 2):
            start_gap = float(abs(starts[right] - starts[left]) / np.timedelta64(1, "s"))
            observed_distance = (
                None if start_lon is None else _distance_m(
                    start_lon[left], start_lat[left], start_lon[right], start_lat[right],
                )
            )
            common = np.flatnonzero(finite[left] & finite[right])
            if len(common):
                time_index = int(common[0])
                first_common: np.datetime64 | None = times[time_index]
                common_distance: float | None = _distance_m(
                    lon[left, time_index], lat[left, time_index],
                    lon[right, time_index], lat[right, time_index],
                )
            else:
                first_common = None
                common_distance = None
            distance = (
                observed_distance if distance_reference == "observed_starts"
                else common_distance
            )
            candidate = bool(
                start_gap <= rule.maximum_start_time_difference_seconds
                and distance is not None
                and distance <= rule.maximum_pair_distance_m
            )
            metric = _PairMetric(
                left, right, start_gap, first_common, observed_distance,
                common_distance, distance, candidate,
            )
            pair_metrics[left, right] = metric
            public_pairs.append(ClusterPairDiagnostic(
                str(platform_ids[left]), str(platform_ids[right]), int(array_id),
                start_gap, first_common, observed_distance, common_distance,
                distance_reference, distance, candidate,
            ))

        if rule.assignment == "single_cluster":
            clusters = [members]
        else:
            clusters = _infer_array_clusters(
                members, platform_ids, starts, pair_metrics, rule,
            )
        clusters.sort(key=lambda cluster: min(
            _member_key(index, platform_ids, starts) for index in cluster
        ))
        array_unavailable = 0
        array_distances: list[float] = []
        array_available_distances: list[float] = []
        for cluster_number, cluster in enumerate(clusters, start=1):
            ordered = tuple(sorted(
                cluster, key=lambda index: _member_key(index, platform_ids, starts),
            ))
            spread, diameter, maximum_available, links, unavailable = _cluster_statistics(
                ordered, starts, pair_metrics,
            )
            observed_diameter, _ = _diagnostic_diameter(
                ordered, pair_metrics, "observed_start_distance_m",
            )
            common_diameter, common_unavailable = _diagnostic_diameter(
                ordered, pair_metrics, "first_common_grid_distance_m",
            )
            array_unavailable += unavailable
            if diameter is not None:
                array_distances.append(diameter)
            if maximum_available is not None:
                array_available_distances.append(maximum_available)
            cluster_id = f"array_{array_id:03d}__cluster_{cluster_number:03d}"
            if rule.assignment == "single_cluster":
                status = "configured_single_cluster"
            elif len(ordered) == 1:
                status = "algorithmic_singleton"
            else:
                status = "candidate_cluster"
            for member_number, index in enumerate(ordered, start=1):
                assignments_by_index[index] = InitialClusterAssignment(
                    str(platform_ids[index]), int(array_id), cluster_id, member_number,
                    len(ordered), status, spread, diameter, maximum_available,
                    observed_diameter, common_diameter, common_unavailable,
                    links, unavailable,
                )
        summaries.append({
            "array_id": int(array_id),
            "assignment": rule.assignment,
            "distance_reference": distance_reference,
            "platform_count": len(members),
            "cluster_count": len(clusters),
            "algorithmic_singleton_count": sum(len(cluster) == 1 for cluster in clusters)
            if rule.assignment == "infer" else 0,
            "maximum_assigned_cluster_diameter_m": max(array_distances)
            if array_distances else None,
            "maximum_available_within_cluster_pair_distance_m":
                max(array_available_distances) if array_available_distances else None,
            "unavailable_within_cluster_pair_count": array_unavailable,
            "effective_rule": {
                "maximum_start_time_difference_seconds":
                    rule.maximum_start_time_difference_seconds,
                "maximum_pair_distance_m": rule.maximum_pair_distance_m,
                "maximum_cluster_diameter_m": rule.maximum_cluster_diameter_m,
                "maximum_members": rule.maximum_members,
            },
        })
    return InitialClusterResult(
        tuple(assignments_by_index[index] for index in range(count)),
        tuple(public_pairs), tuple(summaries),
    )
