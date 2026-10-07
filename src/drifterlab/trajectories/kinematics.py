"""Metric projection and explicitly supported trajectory velocities."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from pyproj import CRS, Transformer


PROJECTION_ORIGIN_METHOD = "equal_weight_spherical_mean_of_platform_first_finite_positions"


@dataclass(frozen=True)
class ArrayProjection:
    """One fixed local metric projection and its transformed positions."""

    x_m: np.ndarray
    y_m: np.ndarray
    origin_longitude: float
    origin_latitude: float
    crs_definition: str
    forward: Transformer
    inverse: Transformer


@dataclass(frozen=True)
class CenteredVelocityResult:
    """Velocity values and the reason each platform-time sample is invalid."""

    u_m_s: np.ndarray
    v_m_s: np.ndarray
    valid: np.ndarray
    invalid_reason: np.ndarray
    cadence_seconds: float
    total_span_seconds: float
    half_span_steps: int
    stencil_point_count: int
    required_validity_point_count: int
    full_stencil_support_required: bool


def spherical_mean(longitude: np.ndarray, latitude: np.ndarray) -> tuple[float, float]:
    """Return a wrap-safe equal-weight spherical mean of coordinate pairs."""
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    if lon.shape != lat.shape or lon.ndim != 1 or not len(lon):
        raise ValueError("Spherical-mean longitude and latitude must be nonempty 1D peers")
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("Spherical-mean coordinates must be finite")
    lon_radians = np.deg2rad(lon)
    lat_radians = np.deg2rad(lat)
    vectors = np.column_stack((
        np.cos(lat_radians) * np.cos(lon_radians),
        np.cos(lat_radians) * np.sin(lon_radians),
        np.sin(lat_radians),
    ))
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot derive a unique spherical-mean projection origin")
    mean /= norm
    mean_lon = float(np.rad2deg(np.arctan2(mean[1], mean[0])))
    mean_lat = float(np.rad2deg(np.arctan2(mean[2], np.hypot(mean[0], mean[1]))))
    if mean_lon >= 180:
        mean_lon -= 360
    return mean_lon, mean_lat


def project_array_positions(
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    origin_longitude: float | None = None,
    origin_latitude: float | None = None,
) -> ArrayProjection:
    """Project platform-by-time WGS84 positions to one fixed array AEQD CRS."""
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    if lon.shape != lat.shape or lon.ndim != 2:
        raise ValueError("Array longitude and latitude must be platform-by-time peers")
    finite = np.isfinite(lon) & np.isfinite(lat)
    if not np.array_equal(np.isfinite(lon), np.isfinite(lat)):
        raise ValueError("Longitude and latitude finite masks differ")
    if np.any(~finite.any(axis=1)):
        raise ValueError("Every projected platform must have a finite position")
    if np.any(finite & ((lon < -180) | (lon >= 180))):
        raise ValueError("Longitude must lie in [-180, 180)")
    if np.any(finite & ((lat < -90) | (lat > 90))):
        raise ValueError("Latitude must lie in [-90, 90]")

    if origin_longitude is None or origin_latitude is None:
        first = np.argmax(finite, axis=1)
        rows = np.arange(lon.shape[0])
        derived_lon, derived_lat = spherical_mean(lon[rows, first], lat[rows, first])
        origin_longitude = derived_lon if origin_longitude is None else origin_longitude
        origin_latitude = derived_lat if origin_latitude is None else origin_latitude
    if not math.isfinite(origin_longitude) or not math.isfinite(origin_latitude):
        raise ValueError("Projection origin must be finite")

    definition = (
        f"+proj=aeqd +lat_0={origin_latitude:.15g} "
        f"+lon_0={origin_longitude:.15g} +datum=WGS84 +units=m +no_defs +type=crs"
    )
    geographic = CRS.from_epsg(4326)
    metric = CRS.from_proj4(definition)
    forward = Transformer.from_crs(geographic, metric, always_xy=True)
    inverse = Transformer.from_crs(metric, geographic, always_xy=True)
    x = np.full(lon.shape, np.nan, dtype=float)
    y = np.full(lat.shape, np.nan, dtype=float)
    x[finite], y[finite] = forward.transform(lon[finite], lat[finite])
    if not np.isfinite(x[finite]).all() or not np.isfinite(y[finite]).all():
        raise ValueError("Array projection produced non-finite metric coordinates")
    return ArrayProjection(
        x_m=x,
        y_m=y,
        origin_longitude=float(origin_longitude),
        origin_latitude=float(origin_latitude),
        crs_definition=definition,
        forward=forward,
        inverse=inverse,
    )


def centered_supported_velocity(
    x_m: np.ndarray,
    y_m: np.ndarray,
    times: np.ndarray,
    source_gap_minutes: np.ndarray,
    *,
    total_span_minutes: float,
    maximum_source_gap_minutes: float,
    speed_mask_m_s: float | None = None,
    require_full_stencil_support: bool = True,
) -> CenteredVelocityResult:
    r"""Calculate a two-endpoint centered derivative with full-stencil validity.

    For the Stage-2 baseline, ``total_span_minutes=30`` on a five-minute grid,

    .. math::

        \mathbf{u}(t) =
        \frac{\mathbf{x}(t+15\,\mathrm{min})-
              \mathbf{x}(t-15\,\mathrm{min})}{1800\,\mathrm{s}}.

    When ``require_full_stencil_support`` is true, all seven intervening grid
    positions, including both endpoints, must have finite positions and finite
    source support no larger than the configured threshold.  When false, source
    support and intervening positions are trusted, while the two derivative
    endpoints must still be finite.  Only those two endpoints enter the
    derivative numerator in either mode.
    """
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    grid = np.asarray(times, dtype="datetime64[ns]")
    gaps = np.asarray(source_gap_minutes, dtype=float)
    if x.shape != y.shape or x.shape != gaps.shape or x.ndim != 2:
        raise ValueError("Velocity positions and source gaps must be platform-by-time peers")
    if grid.shape != (x.shape[1],) or len(grid) < 3 or np.isnat(grid).any():
        raise ValueError("Velocity time must be a finite 1D coordinate matching positions")
    differences_ns = np.diff(grid.astype(np.int64))
    if np.any(differences_ns <= 0) or len(np.unique(differences_ns)) != 1:
        raise ValueError("Velocity time must have one positive regular cadence")
    cadence_ns = int(differences_ns[0])
    cadence_seconds = cadence_ns / 1_000_000_000.0
    if not math.isfinite(total_span_minutes) or total_span_minutes <= 0:
        raise ValueError("Velocity total span must be finite and positive")
    if not math.isfinite(maximum_source_gap_minutes) or maximum_source_gap_minutes <= 0:
        raise ValueError("Maximum source gap must be finite and positive")
    if not isinstance(require_full_stencil_support, bool):
        raise ValueError("require_full_stencil_support must be true or false")
    span_ns = round(total_span_minutes * 60.0 * 1_000_000_000.0)
    if span_ns % 2:
        raise ValueError("Velocity total span cannot be represented by a centered nanosecond span")
    half_ns = span_ns // 2
    if half_ns % cadence_ns:
        raise ValueError("Half the velocity span must be an integer number of grid steps")
    half_steps = half_ns // cadence_ns
    if half_steps < 1:
        raise ValueError("Velocity half span must contain at least one grid step")
    stencil_points = 2 * half_steps + 1

    u = np.full(x.shape, np.nan, dtype=float)
    v = np.full(y.shape, np.nan, dtype=float)
    valid = np.zeros(x.shape, dtype=bool)
    reason = np.full(x.shape, "outside_centered_stencil", dtype="<U40")
    for time_index in range(half_steps, x.shape[1] - half_steps):
        left = time_index - half_steps
        right = time_index + half_steps
        if require_full_stencil_support:
            position_ok = np.all(
                np.isfinite(x[:, left:right + 1]) & np.isfinite(y[:, left:right + 1]),
                axis=1,
            )
            gap_finite = np.all(np.isfinite(gaps[:, left:right + 1]), axis=1)
            gap_supported = np.all(
                gaps[:, left:right + 1] <= maximum_source_gap_minutes,
                axis=1,
            )
        else:
            position_ok = (
                np.isfinite(x[:, left]) & np.isfinite(y[:, left])
                & np.isfinite(x[:, right]) & np.isfinite(y[:, right])
            )
            gap_finite = np.ones(x.shape[0], dtype=bool)
            gap_supported = np.ones(x.shape[0], dtype=bool)
        current_reason = np.full(x.shape[0], "valid", dtype="<U40")
        current_reason[~position_ok] = "nonfinite_position_in_stencil"
        current_reason[position_ok & ~gap_finite] = "missing_source_support_in_stencil"
        current_reason[position_ok & gap_finite & ~gap_supported] = (
            "source_gap_exceeded_in_stencil"
        )
        current_valid = position_ok & gap_finite & gap_supported
        elapsed_seconds = span_ns / 1_000_000_000.0
        u[current_valid, time_index] = (
            x[current_valid, right] - x[current_valid, left]
        ) / elapsed_seconds
        v[current_valid, time_index] = (
            y[current_valid, right] - y[current_valid, left]
        ) / elapsed_seconds
        finite_velocity = np.isfinite(u[:, time_index]) & np.isfinite(v[:, time_index])
        current_reason[current_valid & ~finite_velocity] = "nonfinite_velocity"
        current_valid &= finite_velocity
        if speed_mask_m_s is not None:
            if not math.isfinite(speed_mask_m_s) or speed_mask_m_s <= 0:
                raise ValueError("Speed mask must be finite and positive or null")
            speed_exceeded = current_valid & (
                np.hypot(u[:, time_index], v[:, time_index]) > speed_mask_m_s
            )
            current_reason[speed_exceeded] = "speed_mask_exceeded"
            current_valid &= ~speed_exceeded
            u[speed_exceeded, time_index] = np.nan
            v[speed_exceeded, time_index] = np.nan
        valid[:, time_index] = current_valid
        reason[:, time_index] = current_reason

    return CenteredVelocityResult(
        u_m_s=u,
        v_m_s=v,
        valid=valid,
        invalid_reason=reason,
        cadence_seconds=cadence_seconds,
        total_span_seconds=span_ns / 1_000_000_000.0,
        half_span_steps=int(half_steps),
        stencil_point_count=int(stencil_points),
        required_validity_point_count=(int(stencil_points) if require_full_stencil_support else 2),
        full_stencil_support_required=require_full_stencil_support,
    )
