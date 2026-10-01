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

        attrs: dict[str, Any] = dataset.attrs
        input_metadata = TrajectoryInputMetadata(
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
    return tuple(result), input_metadata


def _optional_attr(attrs: dict[str, Any], name: str) -> str | None:
    value = attrs.get(name)
    return None if value is None else str(value)
