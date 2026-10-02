"""Read and validate per-array views of reconstructed trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .config import PostprocessingConfig, TimeWindowConfig


@dataclass(frozen=True)
class ArrayTrajectoryData:
    """One in-memory array view used consistently by static and movie renderers."""

    array_id: int
    dataset_label: str
    coordinate_method: str
    platform_ids: tuple[str, ...]
    times: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    color_by: str
    color_values: np.ndarray
    reference_time: np.datetime64
    reference_kind: str
    first_retained_fix: np.datetime64
    configured_extent: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class TrajectoryInputMetadata:
    schema_version: str | None
    algorithm_version: str | None
    product_status: str | None
    build_report_sha256: str | None
    array_assignment_sha256: str | None
    initial_cluster_assignment_sha256: str | None
    array_ids: tuple[int, ...]


@dataclass(frozen=True)
class ClusterArrayTrajectoryData:
    """Independent per-array input view for within-cluster statistics."""

    array_id: int
    dataset_label: str
    coordinate_method: str
    platform_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    cluster_sizes: np.ndarray
    times: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    projection_origin_longitude: float
    projection_origin_latitude: float


def _coordinate_names(dataset: xr.Dataset, method: str) -> tuple[str, str]:
    candidates = (
        (("longitude", "latitude"),) if method == "native" else ()
    ) + ((f"longitude_{method}", f"latitude_{method}"),)
    for longitude, latitude in candidates:
        if longitude in dataset.variables and latitude in dataset.variables:
            return longitude, latitude
    available = sorted(
        name.removeprefix("longitude_")
        for name in dataset.variables
        if name.startswith("longitude_") and f"latitude_{name.removeprefix('longitude_')}" in dataset
    )
    raise ValueError(
        f"Trajectory store has no coordinate method {method!r}; available methods are {available}"
    )


def _merged_time_window(
    base: TimeWindowConfig, override: TimeWindowConfig | None,
) -> TimeWindowConfig:
    if override is None:
        return base
    return TimeWindowConfig(
        override.start if not np.isnat(override.start) else base.start,
        override.end if not np.isnat(override.end) else base.end,
    )


def _validate_base_dataset(dataset: xr.Dataset, path: Path) -> tuple[np.ndarray, np.ndarray]:
    if "platform" not in dataset.sizes or "time" not in dataset.sizes:
        raise ValueError(f"Trajectory store must have platform and time dimensions: {path}")
    required = {
        "platform_id": ("platform",),
        "time": ("time",),
        "start_time": ("platform",),
        "array_id": ("platform",),
    }
    for name, dims in required.items():
        if name not in dataset.variables or dataset[name].dims != dims:
            raise ValueError(f"Trajectory store has no valid {name} variable")
    platform_ids = dataset.platform_id.values.astype(str)
    if len(set(platform_ids.tolist())) != len(platform_ids):
        raise ValueError("Trajectory store contains duplicate platform_id values")
    times = dataset.time.values.astype("datetime64[ns]")
    if len(times) == 0 or np.isnat(times).any() or np.any(np.diff(times.astype(np.int64)) <= 0):
        raise ValueError("Trajectory time must be nonempty, finite, and strictly increasing")
    array_values = dataset.array_id.values
    if not np.issubdtype(array_values.dtype, np.integer) or np.any(array_values < 1):
        raise ValueError("Trajectory array_id must contain positive integers")
    starts = dataset.start_time.values.astype("datetime64[ns]")
    if np.isnat(starts).any():
        raise ValueError("Trajectory start_time contains missing values")
    return platform_ids, times


def _input_metadata(dataset: xr.Dataset, array_ids: tuple[int, ...]) -> TrajectoryInputMetadata:
    attrs: dict[str, Any] = dataset.attrs
    return TrajectoryInputMetadata(
        schema_version=_optional_attr(attrs, "schema_version"),
        algorithm_version=_optional_attr(attrs, "algorithm_version"),
        product_status=_optional_attr(attrs, "product_status"),
        build_report_sha256=_optional_attr(attrs, "build_report_sha256"),
        array_assignment_sha256=_optional_attr(attrs, "array_assignment_sha256"),
        initial_cluster_assignment_sha256=_optional_attr(
            attrs, "initial_cluster_assignment_sha256",
        ),
        array_ids=array_ids,
    )


def _validate_coordinates(
    longitude: np.ndarray, latitude: np.ndarray, *, method: str,
) -> np.ndarray:
    longitude_finite = np.isfinite(longitude)
    latitude_finite = np.isfinite(latitude)
    if not np.array_equal(longitude_finite, latitude_finite):
        raise ValueError(
            f"Trajectory {method} longitude and latitude finite masks differ"
        )
    if np.any(longitude_finite & ((longitude < -180) | (longitude >= 180))):
        raise ValueError(f"Trajectory {method} longitude must lie in [-180, 180)")
    if np.any(latitude_finite & ((latitude < -90) | (latitude > 90))):
        raise ValueError(f"Trajectory {method} latitude must lie in [-90, 90]")
    return longitude_finite


def _spherical_mean(longitude: np.ndarray, latitude: np.ndarray) -> tuple[float, float]:
    """Return the wrap-safe equal-weight spherical mean of coordinate pairs."""
    lon_radians = np.deg2rad(np.asarray(longitude, dtype=float))
    lat_radians = np.deg2rad(np.asarray(latitude, dtype=float))
    vectors = np.column_stack((
        np.cos(lat_radians) * np.cos(lon_radians),
        np.cos(lat_radians) * np.sin(lon_radians),
        np.sin(lat_radians),
    ))
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot derive a unique spherical-mean array projection origin")
    mean /= norm
    longitude_mean = float(np.rad2deg(np.arctan2(mean[1], mean[0])))
    latitude_mean = float(np.rad2deg(np.arctan2(mean[2], np.hypot(mean[0], mean[1]))))
    if longitude_mean >= 180:
        longitude_mean -= 360
    return longitude_mean, latitude_mean


def load_cluster_statistic_trajectories(
    config: PostprocessingConfig,
) -> tuple[tuple[ClusterArrayTrajectoryData, ...], TrajectoryInputMetadata]:
    """Load cluster metadata and an independent statistics window for every array."""
    path = config.trajectory_path
    if not path.is_dir():
        raise ValueError(f"Trajectory Zarr does not exist: {path}")
    try:
        context = xr.open_zarr(path, consolidated=True, chunks=None)
    except Exception as exc:
        raise ValueError(f"Cannot open consolidated trajectory Zarr {path}: {exc}") from exc
    with context as dataset:
        platform_ids, times = _validate_base_dataset(dataset, path)
        longitude_name, latitude_name = _coordinate_names(dataset, config.coordinate_method)
        required = {
            "cluster_id": ("platform",),
            "cluster_size": ("platform",),
            longitude_name: ("platform", "time"),
            latitude_name: ("platform", "time"),
        }
        for name, dimensions in required.items():
            if name not in dataset.variables or dataset[name].dims != dimensions:
                raise ValueError(
                    f"Trajectory store has no valid {name} variable with dimensions {dimensions}"
                )

        longitude_all = dataset[longitude_name].values.astype(float)
        latitude_all = dataset[latitude_name].values.astype(float)
        finite_all = _validate_coordinates(
            longitude_all, latitude_all, method=config.coordinate_method,
        )
        if np.any(~finite_all.any(axis=1)):
            missing = platform_ids[~finite_all.any(axis=1)].tolist()
            raise ValueError(
                "Every platform must have a finite selected coordinate for the fixed "
                f"array projection; missing: {missing}"
            )

        raw_cluster_ids = np.asarray(dataset.cluster_id.values)
        if any(
            value is None
            or (isinstance(value, (float, np.floating)) and np.isnan(value))
            or not str(value)
            for value in raw_cluster_ids.tolist()
        ):
            raise ValueError("Trajectory cluster_id must contain nonempty identifiers")
        cluster_ids_all = np.asarray([str(value) for value in raw_cluster_ids], dtype=object)
        try:
            raw_sizes = np.asarray(dataset.cluster_size.values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Trajectory cluster_size must be numeric") from exc
        if (
            not np.isfinite(raw_sizes).all()
            or np.any(raw_sizes < 1)
            or np.any(raw_sizes != np.floor(raw_sizes))
        ):
            raise ValueError(
                "Trajectory cluster_size must contain finite positive integer values"
            )
        cluster_sizes_all = raw_sizes.astype(np.int64)

        array_values = dataset.array_id.values.astype(int)
        array_ids = tuple(sorted(set(array_values.tolist())))
        configured_ids = set(config.arrays) | set(config.cluster_statistics.array_overrides)
        unknown = sorted(configured_ids - set(array_ids))
        if unknown:
            raise ValueError(f"Postprocessing configuration references unknown arrays: {unknown}")

        for array_id in array_ids:
            array_mask = array_values == array_id
            for cluster_id in dict.fromkeys(cluster_ids_all[array_mask].tolist()):
                group = array_mask & (cluster_ids_all == cluster_id)
                stored_sizes = np.unique(cluster_sizes_all[group])
                if len(stored_sizes) != 1:
                    raise ValueError(
                        f"Array {array_id} cluster {cluster_id!r} has inconsistent cluster_size"
                    )
                assigned = int(group.sum())
                if int(stored_sizes[0]) != assigned:
                    raise ValueError(
                        f"Array {array_id} cluster {cluster_id!r} stores cluster_size "
                        f"{int(stored_sizes[0])}, but {assigned} platforms carry that assignment"
                    )

        result: list[ClusterArrayTrajectoryData] = []
        statistics = config.cluster_statistics
        for array_id in array_ids:
            platform_mask = array_values == array_id
            longitude_full = longitude_all[platform_mask]
            latitude_full = latitude_all[platform_mask]
            finite_full = finite_all[platform_mask]
            first_indices = np.argmax(finite_full, axis=1)
            row_indices = np.arange(int(platform_mask.sum()))
            origin_longitude, origin_latitude = _spherical_mean(
                longitude_full[row_indices, first_indices],
                latitude_full[row_indices, first_indices],
            )

            override = statistics.array_overrides.get(array_id)
            window = _merged_time_window(statistics.time, override)
            active_times = finite_full.any(axis=0)
            start = times[np.flatnonzero(active_times)[0]] if np.isnat(window.start) else window.start
            end = times[np.flatnonzero(active_times)[-1]] if np.isnat(window.end) else window.end
            time_mask = (times >= start) & (times <= end)
            if not time_mask.any():
                raise ValueError(
                    f"Array {array_id} cluster_statistics time window contains no source timestamps"
                )
            result.append(ClusterArrayTrajectoryData(
                array_id=array_id,
                dataset_label=config.dataset_label,
                coordinate_method=config.coordinate_method,
                platform_ids=tuple(platform_ids[platform_mask].tolist()),
                cluster_ids=tuple(cluster_ids_all[platform_mask].tolist()),
                cluster_sizes=cluster_sizes_all[platform_mask].copy(),
                times=times[time_mask].copy(),
                longitude=longitude_full[:, time_mask].copy(),
                latitude=latitude_full[:, time_mask].copy(),
                projection_origin_longitude=origin_longitude,
                projection_origin_latitude=origin_latitude,
            ))
        return tuple(result), _input_metadata(dataset, array_ids)


def load_array_trajectories(
    config: PostprocessingConfig,
) -> tuple[tuple[ArrayTrajectoryData, ...], TrajectoryInputMetadata]:
    """Load selected coordinates and split them into independent array populations."""
    path = config.trajectory_path
    if not path.is_dir():
        raise ValueError(f"Trajectory Zarr does not exist: {path}")
    plotting = config.trajectory_plotting
    try:
        context = xr.open_zarr(path, consolidated=True, chunks=None)
    except Exception as exc:
        raise ValueError(f"Cannot open consolidated trajectory Zarr {path}: {exc}") from exc
    with context as dataset:
        platform_ids, times = _validate_base_dataset(dataset, path)
        longitude_name, latitude_name = _coordinate_names(dataset, config.coordinate_method)
        for name in (longitude_name, latitude_name):
            if dataset[name].dims != ("platform", "time"):
                raise ValueError(f"Trajectory variable {name} must have dimensions (platform, time)")
        if plotting.color_by not in dataset.variables:
            raise ValueError(
                f"trajectory_plotting.color_by={plotting.color_by!r} is not in the trajectory store"
            )
        if dataset[plotting.color_by].dims != ("platform",):
            raise ValueError(
                f"Color variable {plotting.color_by!r} must be constant per platform"
            )

        array_values = dataset.array_id.values.astype(int)
        array_ids = tuple(sorted(set(array_values.tolist())))
        configured_ids = set(config.arrays) | set(plotting.array_overrides)
        unknown = sorted(configured_ids - set(array_ids))
        if unknown:
            raise ValueError(f"Postprocessing configuration references unknown arrays: {unknown}")

        longitude_all = dataset[longitude_name].values.astype(float)
        latitude_all = dataset[latitude_name].values.astype(float)
        starts_all = dataset.start_time.values.astype("datetime64[ns]")
        colors_all = np.asarray(dataset[plotting.color_by].values)
        result: list[ArrayTrajectoryData] = []
        for array_id in array_ids:
            platform_mask = array_values == array_id
            longitude = longitude_all[platform_mask]
            latitude = latitude_all[platform_mask]
            finite = np.isfinite(longitude) & np.isfinite(latitude)
            active_times = finite.any(axis=0)
            if not active_times.any():
                raise ValueError(
                    f"Array {array_id} has no finite {config.coordinate_method} coordinates"
                )

            override = plotting.array_overrides.get(array_id)
            window = _merged_time_window(
                plotting.time, None if override is None else override.time,
            )
            start = times[np.flatnonzero(active_times)[0]] if np.isnat(window.start) else window.start
            end = times[np.flatnonzero(active_times)[-1]] if np.isnat(window.end) else window.end
            time_mask = (times >= start) & (times <= end)
            if not time_mask.any() or not finite[:, time_mask].any():
                raise ValueError(
                    f"Array {array_id} has no finite coordinates in its configured time window"
                )

            first_fix = starts_all[platform_mask].min()
            metadata = config.arrays.get(array_id)
            nominal = (
                np.datetime64("NaT", "ns")
                if metadata is None else metadata.nominal_deployment_time
            )
            if np.isnat(nominal):
                reference_time = first_fix
                reference_kind = "first_retained_fix"
            else:
                reference_time = nominal
                reference_kind = "nominal_deployment"
            configured_extent = plotting.map.extent
            if override is not None and override.map_configured:
                configured_extent = override.extent
            result.append(ArrayTrajectoryData(
                array_id=array_id,
                dataset_label=config.dataset_label,
                coordinate_method=config.coordinate_method,
                platform_ids=tuple(platform_ids[platform_mask].tolist()),
                times=times[time_mask].copy(),
                longitude=longitude[:, time_mask].copy(),
                latitude=latitude[:, time_mask].copy(),
                color_by=plotting.color_by,
                color_values=colors_all[platform_mask].copy(),
                reference_time=reference_time,
                reference_kind=reference_kind,
                first_retained_fix=first_fix,
                configured_extent=configured_extent,
            ))

        input_metadata = _input_metadata(dataset, array_ids)
    return tuple(result), input_metadata


def _optional_attr(attrs: dict[str, Any], name: str) -> str | None:
    value = attrs.get(name)
    return None if value is None else str(value)
