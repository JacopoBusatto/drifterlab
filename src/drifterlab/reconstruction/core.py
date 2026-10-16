"""Pure reconstruction calculations on a pre-defined common time grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from pyproj import Geod

WGS84 = Geod(ellps="WGS84")

MINUTES_PER_NS = 1 / 60_000_000_000
OVERSHOOT_TOLERANCE_DEGREES = 1e-10


@dataclass(frozen=True)
class SplineFallbackStats:
    """Counts for one spline period; point counts are unique grid cells."""

    long_gap_portions: int = 0
    long_gap_points: int = 0
    insufficient_support_portions: int = 0
    insufficient_support_points: int = 0
    overshoot_portions: int = 0
    overshoot_points: int = 0
    total_fallback_points: int = 0


@dataclass(frozen=True)
class PlatformReconstruction:
    longitude_linear: np.ndarray
    latitude_linear: np.ndarray
    longitude_spline: dict[int, np.ndarray]
    latitude_spline: dict[int, np.ndarray]
    source_gap_minutes: np.ndarray
    exact_grid_points: int
    filled_grid_points: int
    fallback: dict[int, SplineFallbackStats]


@dataclass(frozen=True)
class LinearGridTrack:
    """One source trajectory linearly resampled onto a bounded common grid."""

    longitude: np.ndarray
    latitude: np.ndarray
    source_gap_minutes: np.ndarray
    exact_grid_points: int
    filled_grid_points: int

@dataclass(frozen=True)
class ForwardVelocity:
    """Forward velocity assigned to the earlier position of each edge."""

    dt_seconds: np.ndarray
    east_m_s: np.ndarray
    north_m_s: np.ndarray
    speed_m_s: np.ndarray

@dataclass(frozen=True)
class RegularTrajectoryProduct:
    """One trajectory resampled onto a regular grid with forward velocity."""

    time: np.ndarray
    position: LinearGridTrack
    velocity: ForwardVelocity

@dataclass(frozen=True)
class LinearGridSeries:
    """One scalar observation series linearly resampled without extrapolation."""

    values: np.ndarray
    source_gap_minutes: np.ndarray
    exact_grid_points: int
    filled_grid_points: int


def resample_linear_series(
    time: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    *,
    maximum_gap_minutes: float | None = None,
) -> LinearGridSeries:
    """Interpolate a scalar series and mask intervals wider than ``maximum_gap_minutes``."""
    time = np.asarray(time, dtype="datetime64[ns]")
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype="datetime64[ns]")
    if not (time.ndim == values.ndim == grid.ndim == 1):
        raise ValueError("Scalar resampling inputs must be one-dimensional")
    if len(time) != len(values):
        raise ValueError("Scalar source times and values must have equal length")
    if maximum_gap_minutes is not None:
        maximum_gap_minutes = float(maximum_gap_minutes)
        if not np.isfinite(maximum_gap_minutes) or maximum_gap_minutes <= 0:
            raise ValueError("maximum_gap_minutes must be finite and positive")

    result = np.full(len(grid), np.nan, dtype=np.float64)
    gap = np.full(len(grid), np.nan, dtype=np.float64)
    if len(time) < 2:
        return LinearGridSeries(result, gap, 0, 0)
    time_ns = time.astype(np.int64)
    if np.isnat(time).any() or np.any(np.diff(time_ns) <= 0):
        raise ValueError("Scalar source times must be valid and strictly increasing")
    if not np.isfinite(values).all():
        raise ValueError("Scalar source values must be finite")

    grid_ns = grid.astype(np.int64)
    inside = (grid_ns >= time_ns[0]) & (grid_ns <= time_ns[-1])
    inside_indices = np.flatnonzero(inside)
    if not len(inside_indices):
        return LinearGridSeries(result, gap, 0, 0)
    relative_time = (time_ns - time_ns[0]) / 1e9
    relative_grid = (grid_ns[inside] - time_ns[0]) / 1e9
    result[inside] = np.interp(relative_grid, relative_time, values)

    targets = grid_ns[inside]
    right = np.searchsorted(time_ns, targets, side="left")
    clipped = np.minimum(right, len(time_ns) - 1)
    exact = (right < len(time_ns)) & (time_ns[clipped] == targets)
    gap[inside_indices[exact]] = 0.0
    between = ~exact
    if between.any():
        gap[inside_indices[between]] = (
            time_ns[right[between]] - time_ns[right[between] - 1]
        ) * MINUTES_PER_NS
    if maximum_gap_minutes is not None:
        result[gap > maximum_gap_minutes] = np.nan
    return LinearGridSeries(
        result, gap, int(exact.sum()), int(np.isfinite(result).sum()),
    )


def wrap_longitude(longitude: np.ndarray) -> np.ndarray:
    """Wrap finite longitude to the half-open interval [-180, 180)."""
    result = np.asarray(longitude, dtype=float).copy()
    finite = np.isfinite(result)
    result[finite] = (result[finite] + 180.0) % 360.0 - 180.0
    return result


def resample_linear_track(
    time: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    grid: np.ndarray,
    *,
    coverage_start: np.datetime64,
    coverage_end: np.datetime64,
    maximum_gap_minutes: float | None = None,
) -> LinearGridTrack:
    """Linearly resample fixes only inside an explicit, source-bracketed span."""
    time = np.asarray(time, dtype="datetime64[ns]")
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    grid = np.asarray(grid, dtype="datetime64[ns]")
    coverage_start = np.datetime64(coverage_start, "ns")
    coverage_end = np.datetime64(coverage_end, "ns")
    if not (time.ndim == longitude.ndim == latitude.ndim == grid.ndim == 1):
        raise ValueError("Linear resampling inputs must be one-dimensional")
    if len({len(time), len(longitude), len(latitude)}) != 1 or len(time) < 2:
        raise ValueError("At least two equally sized source fixes are required")
    time_ns = time.astype(np.int64)
    if np.any(np.diff(time_ns) <= 0):
        raise ValueError("Source fix times must be strictly increasing")
    if (not np.isfinite(longitude).all() or not np.isfinite(latitude).all()
            or np.any((longitude < -180) | (longitude > 180))
            or np.any((latitude < -90) | (latitude > 90))):
        raise ValueError("Source coordinates must be finite and geographically valid")
    if (np.isnat(coverage_start) or np.isnat(coverage_end)
            or coverage_start > coverage_end
            or coverage_start < time[0] or coverage_end > time[-1]):
        raise ValueError("Linear coverage must be ordered and bracketed by source fixes")
    if maximum_gap_minutes is not None:
        maximum_gap_minutes = float(maximum_gap_minutes)
        if not np.isfinite(maximum_gap_minutes) or maximum_gap_minutes <= 0:
            raise ValueError(
                "maximum_gap_minutes must be finite and positive"
            )

    grid_ns = grid.astype(np.int64)
    inside = (grid >= coverage_start) & (grid <= coverage_end)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(longitude)))
    relative_time = (time_ns - time_ns[0]) / 1e9
    relative_grid = (grid_ns[inside] - time_ns[0]) / 1e9
    result_lon = np.full(len(grid), np.nan, dtype=np.float64)
    result_lat = np.full(len(grid), np.nan, dtype=np.float64)
    result_lon[inside] = np.interp(relative_grid, relative_time, unwrapped)
    result_lat[inside] = np.interp(relative_grid, relative_time, latitude)

    gap = np.full(len(grid), np.nan, dtype=np.float64)
    exact = np.zeros(len(grid), dtype=bool)
    inside_indices = np.flatnonzero(inside)
    if len(inside_indices):
        values = grid_ns[inside]
        right = np.searchsorted(time_ns, values, side="left")
        clipped = np.minimum(right, len(time_ns) - 1)
        is_exact = (right < len(time_ns)) & (time_ns[clipped] == values)
        exact[inside_indices[is_exact]] = True
        gap[inside_indices[is_exact]] = 0.0
        between = ~is_exact
        if between.any():
            gap_ns = time_ns[right[between]] - time_ns[right[between] - 1]
            gap[inside_indices[between]] = gap_ns * MINUTES_PER_NS
    if maximum_gap_minutes is not None:
        too_wide = gap > maximum_gap_minutes
        result_lon[too_wide] = np.nan
        result_lat[too_wide] = np.nan
    return LinearGridTrack(
        wrap_longitude(result_lon), result_lat, gap,
        int(exact.sum()), int((np.isfinite(result_lon) & np.isfinite(result_lat)).sum()),
    )


def _unique_knots(
    time_ns: np.ndarray, longitude: np.ndarray, latitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(time_ns, kind="stable")
    time_ns = time_ns[order]
    longitude = longitude[order]
    latitude = latitude[order]
    keep = np.r_[True, np.diff(time_ns) != 0]
    return time_ns[keep], longitude[keep], latitude[keep]


def _outside_envelope(values: np.ndarray, source: np.ndarray) -> bool:
    low = float(np.min(source)) - OVERSHOOT_TOLERANCE_DEGREES
    high = float(np.max(source)) + OVERSHOOT_TOLERANCE_DEGREES
    return bool(np.any((values < low) | (values > high)))


def reconstruct_platform(
    time: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    grid: np.ndarray,
    grid_indices: np.ndarray,
    *,
    dt_minutes: float,
    periods_minutes: tuple[int, ...],
    long_gap_threshold_minutes: dict[int, float],
) -> PlatformReconstruction:
    """Reconstruct one accepted trajectory without extrapolating its endpoints."""
    time = np.asarray(time, dtype="datetime64[ns]")
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    grid = np.asarray(grid, dtype="datetime64[ns]")
    grid_indices = np.asarray(grid_indices, dtype=np.int64)
    if not (time.ndim == longitude.ndim == latitude.ndim == 1):
        raise ValueError("Accepted trajectory inputs must be one-dimensional")
    if len({len(time), len(longitude), len(latitude)}) != 1 or len(time) < 2:
        raise ValueError("At least two equally sized accepted fixes are required")
    if len(grid) != len(grid_indices):
        raise ValueError("grid and grid_indices must have equal length")
    time_ns = time.astype(np.int64)
    if np.any(np.diff(time_ns) <= 0):
        raise ValueError("Accepted fix times must be strictly increasing")
    if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
        raise ValueError("Accepted coordinates must be finite")

    unwrapped_lon = np.rad2deg(np.unwrap(np.deg2rad(longitude)))
    grid_ns = grid.astype(np.int64)
    inside = (grid_ns >= time_ns[0]) & (grid_ns <= time_ns[-1])
    relative_time = (time_ns - time_ns[0]) / 1e9
    relative_grid = (grid_ns[inside] - time_ns[0]) / 1e9
    linear_lon = np.full(len(grid), np.nan, dtype=np.float64)
    linear_lat = np.full(len(grid), np.nan, dtype=np.float64)
    linear_lon[inside] = np.interp(relative_grid, relative_time, unwrapped_lon)
    linear_lat[inside] = np.interp(relative_grid, relative_time, latitude)

    source_gap = np.full(len(grid), np.nan, dtype=np.float64)
    exact = np.zeros(len(grid), dtype=bool)
    inside_indices = np.flatnonzero(inside)
    if len(inside_indices):
        values = grid_ns[inside]
        right = np.searchsorted(time_ns, values, side="left")
        is_exact = (right < len(time_ns)) & (time_ns[np.minimum(right, len(time_ns) - 1)] == values)
        exact[inside_indices[is_exact]] = True
        source_gap[inside_indices[is_exact]] = 0.0
        between = ~is_exact
        if between.any():
            gap_ns = time_ns[right[between]] - time_ns[right[between] - 1]
            source_gap[inside_indices[between]] = gap_ns * MINUTES_PER_NS

    spline_lon: dict[int, np.ndarray] = {}
    spline_lat: dict[int, np.ndarray] = {}
    fallback_stats: dict[int, SplineFallbackStats] = {}
    source_gaps_minutes = np.diff(time_ns) * MINUTES_PER_NS

    for period in periods_minutes:
        steps = int(round(period / dt_minutes))
        threshold = float(long_gap_threshold_minutes[period])
        candidate_lon = linear_lon.copy()
        candidate_lat = linear_lat.copy()
        long_edges = source_gaps_minutes > threshold
        long_mask = np.zeros(len(grid), dtype=bool)
        for edge in np.flatnonzero(long_edges):
            long_mask |= inside & (grid_ns >= time_ns[edge]) & (grid_ns <= time_ns[edge + 1])

        split_edges = np.flatnonzero(long_edges)
        starts = np.r_[0, split_edges + 1]
        ends = np.r_[split_edges, len(time_ns) - 1]
        insufficient_mask = np.zeros(len(grid), dtype=bool)
        overshoot_mask = np.zeros(len(grid), dtype=bool)
        insufficient_portions = 0
        overshoot_portions = 0

        for lo, hi in zip(starts, ends):
            portion = inside & (grid_ns >= time_ns[lo]) & (grid_ns <= time_ns[hi])
            portion_indices = np.flatnonzero(portion)
            if not len(portion_indices):
                continue
            phase_tracks: list[np.ndarray] = []
            supported = True
            for phase in range(steps):
                phase_indices = portion_indices[grid_indices[portion_indices] % steps == phase]
                knot_time = np.r_[time_ns[lo], grid_ns[phase_indices], time_ns[hi]]
                knot_lon = np.r_[unwrapped_lon[lo], linear_lon[phase_indices], unwrapped_lon[hi]]
                knot_lat = np.r_[latitude[lo], linear_lat[phase_indices], latitude[hi]]
                knot_time, knot_lon, knot_lat = _unique_knots(knot_time, knot_lon, knot_lat)
                if len(knot_time) < 4:
                    supported = False
                    break
                x = (knot_time - time_ns[lo]) / 1e9
                target = (grid_ns[portion_indices] - time_ns[lo]) / 1e9
                values = CubicSpline(
                    x, np.column_stack([knot_lon, knot_lat]),
                    axis=0, bc_type="natural", extrapolate=False,
                )(target)
                if (not np.isfinite(values).all()
                        or np.any((values[:, 1] < -90) | (values[:, 1] > 90))
                        or _outside_envelope(values[:, 0], unwrapped_lon[lo:hi + 1])
                        or _outside_envelope(values[:, 1], latitude[lo:hi + 1])):
                    supported = False
                    overshoot_portions += 1
                    overshoot_mask[portion_indices] = True
                    break
                phase_tracks.append(values)
            if not supported:
                if not overshoot_mask[portion_indices].any():
                    insufficient_portions += 1
                    insufficient_mask[portion_indices] = True
                continue
            averaged = np.mean(np.stack(phase_tracks), axis=0)
            if (not np.isfinite(averaged).all()
                    or np.any((averaged[:, 1] < -90) | (averaged[:, 1] > 90))
                    or _outside_envelope(averaged[:, 0], unwrapped_lon[lo:hi + 1])
                    or _outside_envelope(averaged[:, 1], latitude[lo:hi + 1])):
                overshoot_portions += 1
                overshoot_mask[portion_indices] = True
                continue
            candidate_lon[portion_indices] = averaged[:, 0]
            candidate_lat[portion_indices] = averaged[:, 1]

        total_fallback = long_mask | insufficient_mask | overshoot_mask
        spline_lon[period] = wrap_longitude(candidate_lon)
        spline_lat[period] = candidate_lat
        fallback_stats[period] = SplineFallbackStats(
            long_gap_portions=int(long_edges.sum()),
            long_gap_points=int(long_mask.sum()),
            insufficient_support_portions=insufficient_portions,
            insufficient_support_points=int(insufficient_mask.sum()),
            overshoot_portions=overshoot_portions,
            overshoot_points=int(overshoot_mask.sum()),
            total_fallback_points=int(total_fallback.sum()),
        )

    return PlatformReconstruction(
        wrap_longitude(linear_lon), linear_lat, spline_lon, spline_lat,
        source_gap, int(exact.sum()), int(inside.sum()), fallback_stats,
    )


def forward_velocity(
    time: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> ForwardVelocity:
    """Calculate forward velocity without bridging missing positions."""

    time = np.asarray(time, dtype="datetime64[ns]")
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)

    if not (
        time.ndim == longitude.ndim == latitude.ndim == 1
        and len({len(time), len(longitude), len(latitude)}) == 1
    ):
        raise ValueError(
            "Velocity inputs must be equally sized one-dimensional arrays"
        )

    count = len(time)
    dt = np.full(count, np.nan)
    east = np.full(count, np.nan)
    north = np.full(count, np.nan)
    speed = np.full(count, np.nan)

    if count < 2:
        return ForwardVelocity(dt, east, north, speed)

    time_ns = time.astype(np.int64)
    edge_dt = np.asarray(
        time_ns[1:].astype(object) - time_ns[:-1].astype(object),
        dtype=float,
    ) / 1e9

    valid_coordinates = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180)
        & (longitude <= 180)
        & (latitude >= -90)
        & (latitude <= 90)
    )
    valid_edges = (
        ~np.isnat(time[:-1])
        & ~np.isnat(time[1:])
        & valid_coordinates[:-1]
        & valid_coordinates[1:]
        & (edge_dt > 0)
    )

    indices = np.flatnonzero(valid_edges)
    if len(indices):
        azimuth, _back, distance = WGS84.inv(
            longitude[indices],
            latitude[indices],
            longitude[indices + 1],
            latitude[indices + 1],
        )
        values = np.asarray(distance) / edge_dt[indices]
        radians = np.deg2rad(azimuth)

        # Store at the earlier endpoint: velocity from t to t + dt.
        dt[indices] = edge_dt[indices]
        speed[indices] = values
        east[indices] = values * np.sin(radians)
        north[indices] = values * np.cos(radians)

    return ForwardVelocity(dt, east, north, speed)



def build_regular_product(
    source_time: np.ndarray,
    source_longitude: np.ndarray,
    source_latitude: np.ndarray,
    grid: np.ndarray,
    *,
    coverage_start: np.datetime64,
    coverage_end: np.datetime64,
    maximum_gap_minutes: float,
) -> RegularTrajectoryProduct:
    """Build gap-limited positions and their forward velocity."""

    grid = np.asarray(grid, dtype="datetime64[ns]")

    position = resample_linear_track(
        source_time,
        source_longitude,
        source_latitude,
        grid,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        maximum_gap_minutes=maximum_gap_minutes,
    )
    velocity = forward_velocity(
        grid,
        position.longitude,
        position.latitude,
    )

    return RegularTrajectoryProduct(
        time=grid,
        position=position,
        velocity=velocity,
    )