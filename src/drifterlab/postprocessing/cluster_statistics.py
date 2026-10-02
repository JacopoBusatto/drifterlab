"""Numerical within-cluster trajectory statistics in a fixed metric frame."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import ConvexHull, QhullError

from .config import ClusterStatisticsConfig, percentile_suffix
from .trajectories import ClusterArrayTrajectoryData

FLOATING_TOLERANCE_FACTOR = 256.0
"""Multiplier on machine epsilon for scale-aware geometry comparisons."""

ISOTROPY_RELATIVE_TOLERANCE = 1e-8
"""Relative eigenvalue-gap tolerance below which orientation is not identifiable."""

COLLINEAR_RELATIVE_TOLERANCE = 1e-8
"""Relative singular-value tolerance used to identify a zero-area line geometry."""

PROJECTION_ORIGIN_METHOD = (
    "wrap-safe spherical mean of one first-valid selected-coordinate position "
    "per platform over the full input trajectory"
)

CLUSTER_STATISTICS_TABLES = ("cluster_timeseries.csv", "cluster_summary.csv")


@dataclass(frozen=True)
class VelocityDistributionSamples:
    absolute_speed_m_s: np.ndarray
    internal_speed_m_s: np.ndarray


@dataclass(frozen=True)
class ClusterStatisticsArrayResult:
    array_id: int
    cluster_ids: tuple[str, ...]
    timeseries: pd.DataFrame
    summary: pd.DataFrame
    velocity_samples: dict[str, VelocityDistributionSamples]
    projection_origin_longitude: float
    projection_origin_latitude: float
    crs_definition: str
    projection_origin_method: str


def _format_utc(value: np.datetime64) -> str:
    return pd.Timestamp(value, tz="UTC").isoformat().replace("+00:00", "Z")


def _finite(values: np.ndarray | list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    return result[np.isfinite(result)]


def _mean(values: np.ndarray | list[float]) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if len(finite) else np.nan


def _maximum(values: np.ndarray | list[float]) -> float:
    finite = _finite(values)
    return float(np.max(finite)) if len(finite) else np.nan


def _percentile_values(
    values: np.ndarray | list[float], percentiles: tuple[float, ...],
) -> dict[str, float]:
    finite = _finite(values)
    if not len(finite):
        return {percentile_suffix(percentile): np.nan for percentile in percentiles}
    calculated = np.percentile(finite, percentiles)
    return {
        percentile_suffix(percentile): float(value)
        for percentile, value in zip(percentiles, calculated)
    }


def _add_percentiles(
    target: dict[str, object], *, prefix: str, unit: str,
    values: np.ndarray | list[float], percentiles: tuple[float, ...],
) -> None:
    for suffix, value in _percentile_values(values, percentiles).items():
        target[f"{prefix}_{suffix}_{unit}"] = value


def _metric_columns(percentiles: tuple[float, ...]) -> tuple[str, ...]:
    fixed = (
        "centroid_x_m", "centroid_y_m", "centroid_displacement_m",
        "covariance_xx_m2", "covariance_xy_m2", "covariance_yy_m2",
        "lambda_major_m2", "lambda_minor_m2", "major_scale_m", "minor_scale_m",
        "eigenvalue_ratio", "aspect_ratio", "major_axis_orientation_deg",
        "radius_of_gyration_m", "convex_hull_area_m2", "convex_hull_area_ratio",
        "mean_velocity_u_m_s", "mean_velocity_v_m_s", "mean_platform_speed_m_s",
        "mean_internal_speed_m_s", "mean_absolute_displacement_m",
        "mean_pair_separation_m", "mean_squared_pair_separation_m2",
        "relative_dispersion_D2_m2",
    )
    percentile_columns: list[str] = []
    definitions = (
        ("speed", "m_s"),
        ("internal_speed", "m_s"),
        ("absolute_displacement", "m"),
        ("pair_separation", "m"),
        ("relative_displacement_q", "m2"),
    )
    for prefix, unit in definitions:
        percentile_columns.extend(
            f"{prefix}_{percentile_suffix(value)}_{unit}" for value in percentiles
        )
    return (*fixed, *percentile_columns)


def _empty_metric_values(percentiles: tuple[float, ...]) -> dict[str, float]:
    return {name: np.nan for name in _metric_columns(percentiles)}


def _projection(data: ClusterArrayTrajectoryData) -> tuple[np.ndarray, np.ndarray, str]:
    definition = (
        f"+proj=aeqd +lat_0={data.projection_origin_latitude:.15g} "
        f"+lon_0={data.projection_origin_longitude:.15g} "
        "+datum=WGS84 +units=m +no_defs +type=crs"
    )
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_proj4(definition), always_xy=True,
    )
    x = np.full(data.longitude.shape, np.nan, dtype=float)
    y = np.full(data.latitude.shape, np.nan, dtype=float)
    finite = np.isfinite(data.longitude) & np.isfinite(data.latitude)
    x[finite], y[finite] = transformer.transform(
        data.longitude[finite], data.latitude[finite],
    )
    if not np.isfinite(x[finite]).all() or not np.isfinite(y[finite]).all():
        raise ValueError(f"Array {data.array_id} projection produced non-finite coordinates")
    return x, y, definition


def _membership_interval(
    active_count: np.ndarray, assigned_size: int, stop_on_member_loss: bool,
) -> tuple[np.ndarray, int | None, int | None, bool, str]:
    included = np.ones(len(active_count), dtype=bool)
    if not stop_on_member_loss:
        return included, 0, len(active_count) - 1, False, "configured_window_end"
    complete = np.flatnonzero(active_count == assigned_size)
    if not len(complete):
        included[:] = False
        return included, None, None, False, "complete_cluster_never_available"
    start = int(complete[0])
    subsequent_loss = np.flatnonzero(
        (np.arange(len(active_count)) > start) & (active_count < assigned_size)
    )
    included[:] = False
    if len(subsequent_loss):
        loss = int(subsequent_loss[0])
        included[start:loss] = True
        return included, start, loss - 1, True, "member_loss"
    included[start:] = True
    return included, start, len(active_count) - 1, False, "configured_window_end"


def _velocity_components(
    x: np.ndarray, y: np.ndarray, times: np.ndarray, included: np.ndarray,
    difference_interval_minutes: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Use exact centered spans, with one-sided stencils only near policy boundaries."""
    u = np.full(x.shape, np.nan, dtype=float)
    v = np.full(y.shape, np.nan, dtype=float)
    admitted = np.flatnonzero(included)
    if not len(admitted):
        return u, v
    delta_ns = round(difference_interval_minutes * 60.0 * 1_000_000_000.0)
    if delta_ns <= 0:
        raise ValueError("Velocity difference interval is too small to represent in nanoseconds")
    time_ns = times.astype("datetime64[ns]").astype(np.int64)
    lookup = {int(value): index for index, value in enumerate(time_ns)}
    start_ns = int(time_ns[admitted[0]])
    end_ns = int(time_ns[admitted[-1]])

    for time_index in admitted:
        current_ns = int(time_ns[time_index])
        endpoints: tuple[int, int] | None = None
        if delta_ns % 2 == 0:
            half = delta_ns // 2
            left_ns = current_ns - half
            right_ns = current_ns + half
            left_index = lookup.get(left_ns)
            right_index = lookup.get(right_ns)
            if (
                left_ns >= start_ns and right_ns <= end_ns
                and left_index is not None and right_index is not None
                and included[left_index] and included[right_index]
            ):
                endpoints = left_index, right_index

        left_crosses_boundary = 2 * (current_ns - start_ns) < delta_ns
        right_crosses_boundary = 2 * (end_ns - current_ns) < delta_ns
        if endpoints is None and left_crosses_boundary:
            right_index = lookup.get(current_ns + delta_ns)
            if right_index is not None and included[right_index]:
                endpoints = time_index, right_index
        if endpoints is None and right_crosses_boundary:
            left_index = lookup.get(current_ns - delta_ns)
            if left_index is not None and included[left_index]:
                endpoints = left_index, time_index
        if endpoints is None:
            continue

        left_index, right_index = endpoints
        elapsed_ns = int(time_ns[right_index] - time_ns[left_index])
        if elapsed_ns != delta_ns:
            continue
        # A finite-difference stencil may not bridge an internal missing
        # observation even when its two endpoints happen to be finite.
        stencil_valid = np.all(
            np.isfinite(x[:, left_index:right_index + 1])
            & np.isfinite(y[:, left_index:right_index + 1]),
            axis=1,
        )
        valid = (
            stencil_valid
            & np.isfinite(x[:, time_index]) & np.isfinite(y[:, time_index])
        )
        elapsed_seconds = elapsed_ns / 1_000_000_000.0
        u[valid, time_index] = (
            x[valid, right_index] - x[valid, left_index]
        ) / elapsed_seconds
        v[valid, time_index] = (
            y[valid, right_index] - y[valid, left_index]
        ) / elapsed_seconds
    return u, v


def _geometry(points: np.ndarray, *, cluster_id: str, time: np.datetime64) -> dict[str, float]:
    result = {
        "covariance_xx_m2": np.nan,
        "covariance_xy_m2": np.nan,
        "covariance_yy_m2": np.nan,
        "lambda_major_m2": np.nan,
        "lambda_minor_m2": np.nan,
        "major_scale_m": np.nan,
        "minor_scale_m": np.nan,
        "eigenvalue_ratio": np.nan,
        "aspect_ratio": np.nan,
        "major_axis_orientation_deg": np.nan,
        "radius_of_gyration_m": np.nan,
        "convex_hull_area_m2": np.nan,
    }
    count = len(points)
    if count < 2:
        return result
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / count
    covariance = (covariance + covariance.T) / 2.0
    scale = max(1.0, float(np.max(np.abs(covariance))))
    tolerance = FLOATING_TOLERANCE_FACTOR * np.finfo(float).eps * scale
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] < -tolerance:
        raise FloatingPointError(
            f"Materially negative covariance eigenvalue for cluster {cluster_id!r} "
            f"at {_format_utc(time)}: {eigenvalues[0]:.12g} m2"
        )
    eigenvalues = np.where(eigenvalues < 0, 0.0, eigenvalues)
    minor = float(eigenvalues[0])
    major = float(eigenvalues[1])
    radius_squared = float(np.mean(np.sum(centered * centered, axis=1)))
    if not np.isclose(radius_squared, major + minor, rtol=1e-12, atol=tolerance):
        raise FloatingPointError(
            f"Radius-of-gyration identity failed for cluster {cluster_id!r} "
            f"at {_format_utc(time)}"
        )
    result.update({
        "covariance_xx_m2": float(covariance[0, 0]),
        "covariance_xy_m2": float(covariance[0, 1]),
        "covariance_yy_m2": float(covariance[1, 1]),
        "lambda_major_m2": major,
        "lambda_minor_m2": minor,
        "major_scale_m": float(np.sqrt(major)),
        "minor_scale_m": float(np.sqrt(minor)),
        "radius_of_gyration_m": float(np.sqrt(max(0.0, radius_squared))),
    })
    if major > tolerance:
        ratio = min(1.0, max(0.0, minor / major))
        result["eigenvalue_ratio"] = ratio
        result["aspect_ratio"] = float(np.sqrt(ratio))
        if abs(major - minor) > max(tolerance, ISOTROPY_RELATIVE_TOLERANCE * major):
            vector = eigenvectors[:, 1]
            result["major_axis_orientation_deg"] = float(
                np.degrees(np.arctan2(vector[1], vector[0])) % 180.0
            )

    if count >= 3:
        singular = np.linalg.svd(centered, compute_uv=False)
        rank_tolerance = (
            max(centered.shape) * np.finfo(float).eps * singular[0]
            if len(singular) and singular[0] > 0 else 0.0
        )
        if (
            len(singular) < 2
            or singular[1] <= max(
                rank_tolerance, COLLINEAR_RELATIVE_TOLERANCE * singular[0],
            )
        ):
            result["convex_hull_area_m2"] = 0.0
        else:
            try:
                result["convex_hull_area_m2"] = float(ConvexHull(points).volume)
            except QhullError as exc:
                raise FloatingPointError(
                    f"Convex hull failed for cluster {cluster_id!r} at {_format_utc(time)}"
                ) from exc
    return result


def _summary_distribution(
    target: dict[str, object], *, prefix: str, unit: str | None,
    values: np.ndarray | list[float], percentiles: tuple[float, ...],
) -> None:
    unit_suffix = "" if unit is None else f"_{unit}"
    target[f"mean_{prefix}{unit_suffix}"] = _mean(values)
    for suffix, value in _percentile_values(values, percentiles).items():
        target[f"{prefix}_{suffix}{unit_suffix}"] = value


def calculate_array_cluster_statistics(
    data: ClusterArrayTrajectoryData, config: ClusterStatisticsConfig,
) -> ClusterStatisticsArrayResult:
    """Calculate all requested statistics without combining arrays or clusters."""
    x, y, crs_definition = _projection(data)
    cluster_order = tuple(dict.fromkeys(data.cluster_ids))
    timeseries_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    distributions: dict[str, VelocityDistributionSamples] = {}

    for cluster_id in cluster_order:
        member_indices = np.asarray([
            index for index, value in enumerate(data.cluster_ids) if value == cluster_id
        ], dtype=int)
        assigned_size = int(data.cluster_sizes[member_indices[0]])
        cluster_x = x[member_indices]
        cluster_y = y[member_indices]
        finite = np.isfinite(cluster_x) & np.isfinite(cluster_y)
        active_count = finite.sum(axis=0).astype(int)
        pair_count = (active_count * (active_count - 1) // 2).astype(int)
        included, start_index, end_index, truncated, termination_reason = _membership_interval(
            active_count, assigned_size, config.stop_on_member_loss,
        )
        if start_index is None:
            warnings.warn(
                f"Array {data.array_id} cluster {cluster_id!r} is unavailable with "
                "stop_on_member_loss=true because the complete assigned cluster is never "
                "simultaneously available",
                RuntimeWarning,
                stacklevel=2,
            )

        u, v = _velocity_components(
            cluster_x, cluster_y, data.times, included,
            config.velocity.difference_interval_minutes,
        )
        speed = np.hypot(u, v)
        internal_speed = np.full(speed.shape, np.nan, dtype=float)
        mean_u = np.full(len(data.times), np.nan, dtype=float)
        mean_v = np.full(len(data.times), np.nan, dtype=float)
        for time_index in np.flatnonzero(included):
            valid_velocity = np.isfinite(u[:, time_index]) & np.isfinite(v[:, time_index])
            if valid_velocity.any():
                mean_u[time_index] = float(np.mean(u[valid_velocity, time_index]))
                mean_v[time_index] = float(np.mean(v[valid_velocity, time_index]))
                internal_speed[valid_velocity, time_index] = np.hypot(
                    u[valid_velocity, time_index] - mean_u[time_index],
                    v[valid_velocity, time_index] - mean_v[time_index],
                )
        distributions[cluster_id] = VelocityDistributionSamples(
            _finite(speed[:, included]), _finite(internal_speed[:, included]),
        )

        platform_origins = np.full((assigned_size, 2), np.nan, dtype=float)
        for platform_index in range(assigned_size):
            valid = np.flatnonzero(included & finite[platform_index])
            if len(valid):
                first = int(valid[0])
                platform_origins[platform_index] = (
                    cluster_x[platform_index, first], cluster_y[platform_index, first],
                )

        centroid_x = np.full(len(data.times), np.nan, dtype=float)
        centroid_y = np.full(len(data.times), np.nan, dtype=float)
        for time_index in np.flatnonzero(included):
            active = finite[:, time_index]
            if active.any():
                centroid_x[time_index] = float(np.mean(cluster_x[active, time_index]))
                centroid_y[time_index] = float(np.mean(cluster_y[active, time_index]))
        centroid_valid = np.flatnonzero(np.isfinite(centroid_x) & np.isfinite(centroid_y))
        centroid_origin = (
            None if not len(centroid_valid)
            else np.asarray([
                centroid_x[int(centroid_valid[0])], centroid_y[int(centroid_valid[0])],
            ])
        )

        pair_members = tuple(combinations(range(assigned_size), 2))
        pair_origins: dict[tuple[int, int], np.ndarray] = {}
        for left, right in pair_members:
            simultaneous = np.flatnonzero(included & finite[left] & finite[right])
            if len(simultaneous):
                first = int(simultaneous[0])
                pair_origins[left, right] = np.asarray([
                    cluster_x[right, first] - cluster_x[left, first],
                    cluster_y[right, first] - cluster_y[left, first],
                ])

        hull_reference_area: float | None = None
        hull_reference_time: np.datetime64 | None = None

        for time_index, time in enumerate(data.times):
            row: dict[str, object] = {
                "array_id": data.array_id,
                "cluster_id": cluster_id,
                "time": _format_utc(time),
                "assigned_cluster_size": assigned_size,
                "active_member_count": int(active_count[time_index]),
                "valid_pair_count": int(pair_count[time_index]),
                "analysis_included": bool(included[time_index]),
                "valid_velocity_sample_count": 0,
                **_empty_metric_values(config.percentiles),
            }
            if not included[time_index]:
                timeseries_rows.append(row)
                continue

            active = finite[:, time_index]
            if active.any():
                points = np.column_stack((
                    cluster_x[active, time_index], cluster_y[active, time_index],
                ))
                row["centroid_x_m"] = centroid_x[time_index]
                row["centroid_y_m"] = centroid_y[time_index]
                if centroid_origin is not None:
                    row["centroid_displacement_m"] = float(np.hypot(
                        centroid_x[time_index] - centroid_origin[0],
                        centroid_y[time_index] - centroid_origin[1],
                    ))
                geometry = _geometry(points, cluster_id=cluster_id, time=time)
                row.update(geometry)
                hull_area = geometry["convex_hull_area_m2"]
                if hull_reference_area is None and np.isfinite(hull_area):
                    hull_reference_area = hull_area
                    hull_reference_time = time
                if (
                    hull_reference_area is not None
                    and hull_reference_area > 0
                    and np.isfinite(hull_area)
                ):
                    row["convex_hull_area_ratio"] = hull_area / hull_reference_area

                displacement = np.hypot(
                    cluster_x[active, time_index] - platform_origins[active, 0],
                    cluster_y[active, time_index] - platform_origins[active, 1],
                )
                row["mean_absolute_displacement_m"] = _mean(displacement)
                _add_percentiles(
                    row, prefix="absolute_displacement", unit="m",
                    values=displacement, percentiles=config.percentiles,
                )

            valid_velocity = np.isfinite(speed[:, time_index])
            row["valid_velocity_sample_count"] = int(valid_velocity.sum())
            if valid_velocity.any():
                current_speed = speed[valid_velocity, time_index]
                current_internal = internal_speed[valid_velocity, time_index]
                row["mean_velocity_u_m_s"] = mean_u[time_index]
                row["mean_velocity_v_m_s"] = mean_v[time_index]
                row["mean_platform_speed_m_s"] = _mean(current_speed)
                row["mean_internal_speed_m_s"] = _mean(current_internal)
                _add_percentiles(
                    row, prefix="speed", unit="m_s", values=current_speed,
                    percentiles=config.percentiles,
                )
                _add_percentiles(
                    row, prefix="internal_speed", unit="m_s", values=current_internal,
                    percentiles=config.percentiles,
                )

            separations: list[float] = []
            squared_separations: list[float] = []
            relative_displacements: list[float] = []
            for left, right in pair_members:
                if not (finite[left, time_index] and finite[right, time_index]):
                    continue
                relative = np.asarray([
                    cluster_x[right, time_index] - cluster_x[left, time_index],
                    cluster_y[right, time_index] - cluster_y[left, time_index],
                ])
                squared = float(relative @ relative)
                separations.append(float(np.sqrt(squared)))
                squared_separations.append(squared)
                change = relative - pair_origins[left, right]
                relative_displacements.append(float(change @ change))
            if separations:
                row["mean_pair_separation_m"] = _mean(separations)
                row["mean_squared_pair_separation_m2"] = _mean(squared_separations)
                row["relative_dispersion_D2_m2"] = _mean(relative_displacements)
                _add_percentiles(
                    row, prefix="pair_separation", unit="m", values=separations,
                    percentiles=config.percentiles,
                )
                _add_percentiles(
                    row, prefix="relative_displacement_q", unit="m2",
                    values=relative_displacements, percentiles=config.percentiles,
                )
            timeseries_rows.append(row)

        cluster_frame = pd.DataFrame(
            timeseries_rows[-len(data.times):],
        )
        admitted_frame = cluster_frame[cluster_frame.analysis_included]
        contributing = int(np.sum(np.any(finite[:, included], axis=1))) if included.any() else 0
        summary: dict[str, object] = {
            "array_id": data.array_id,
            "cluster_id": cluster_id,
            "assigned_cluster_size": assigned_size,
            "contributing_platform_count": contributing,
            "membership_policy": (
                "stop_on_member_loss" if config.stop_on_member_loss
                else "dynamic_active_membership"
            ),
            "analysis_start_time": (
                np.nan if start_index is None else _format_utc(data.times[start_index])
            ),
            "analysis_end_time": (
                np.nan if end_index is None else _format_utc(data.times[end_index])
            ),
            "analysis_truncated": truncated,
            "termination_reason": termination_reason,
            "convex_hull_reference_time": (
                np.nan
                if hull_reference_time is None
                else _format_utc(hull_reference_time)
            ),
            "convex_hull_reference_area_m2": (
                np.nan if hull_reference_area is None else hull_reference_area
            ),
            "valid_velocity_sample_count": int(np.isfinite(speed[:, included]).sum()),
            "maximum_centroid_displacement_m": _maximum(
                admitted_frame["centroid_displacement_m"].to_numpy()
            ),
            "maximum_valid_pair_count": (
                int(admitted_frame["valid_pair_count"].max())
                if len(admitted_frame) else 0
            ),
        }
        _summary_distribution(
            summary, prefix="speed", unit="m_s",
            values=distributions[cluster_id].absolute_speed_m_s,
            percentiles=config.percentiles,
        )
        _summary_distribution(
            summary, prefix="internal_speed", unit="m_s",
            values=distributions[cluster_id].internal_speed_m_s,
            percentiles=config.percentiles,
        )
        for prefix, column, unit in (
            ("radius_of_gyration", "radius_of_gyration_m", "m"),
            ("major_scale", "major_scale_m", "m"),
            ("minor_scale", "minor_scale_m", "m"),
            ("aspect_ratio", "aspect_ratio", None),
            ("convex_hull_area", "convex_hull_area_m2", "m2"),
            ("convex_hull_area_ratio", "convex_hull_area_ratio", None),
        ):
            _summary_distribution(
                summary, prefix=prefix, unit=unit,
                values=admitted_frame[column].to_numpy(), percentiles=config.percentiles,
            )
        summary_rows.append(summary)

    return ClusterStatisticsArrayResult(
        array_id=data.array_id,
        cluster_ids=cluster_order,
        timeseries=pd.DataFrame(timeseries_rows),
        summary=pd.DataFrame(summary_rows),
        velocity_samples=distributions,
        projection_origin_longitude=data.projection_origin_longitude,
        projection_origin_latitude=data.projection_origin_latitude,
        crs_definition=crs_definition,
        projection_origin_method=PROJECTION_ORIGIN_METHOD,
    )


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, na_rep="NaN")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_cluster_statistics_tables(
    result: ClusterStatisticsArrayResult, directory: str | Path,
) -> tuple[Path, Path]:
    """Atomically write exactly the two declared tidy numerical products."""
    directory = Path(directory)
    return (
        _write_csv_atomic(result.timeseries, directory / CLUSTER_STATISTICS_TABLES[0]),
        _write_csv_atomic(result.summary, directory / CLUSTER_STATISTICS_TABLES[1]),
    )
