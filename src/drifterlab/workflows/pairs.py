"""Build candidate encounter pairs in the kinematicParcels grouped layout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import math
from numbers import Real
from pathlib import Path
import re
import shutil
from typing import Any, Callable
from uuid import uuid4

from numcodecs import Blosc
import numpy as np
import pandas as pd
from pyproj import Geod
import xarray as xr
import yaml
import zarr

from drifterlab import __version__
from drifterlab.pairs import (
    PairCandidate,
    PairSearchResult,
    circular_mean_longitude,
    find_candidate_pairs,
)


PAIR_ALGORITHM_VERSION = "first-geodesic-threshold-crossing-v1"
PAIR_SCHEMA_VERSION = "1.0"
SUPPORTED_RECONSTRUCTION_SCHEMA_VERSION = "1.4"
ZARR_NAME = "pairs.zarr"
CATALOG_NAME = "pair_catalog.csv"
WGS84 = Geod(ellps="WGS84")
_COORDINATE_PATTERN = re.compile(r"^longitude_(.+)$")


@dataclass(frozen=True)
class PairWorkflowConfig:
    input_zarr: Path
    selection_method: str
    maximum_distance_m: float
    maximum_seconds_from_each_observed_start: float | None
    output_directory: Path
    chunk_trajectory: int
    chunk_obs: int

    def effective(self) -> dict[str, Any]:
        return {
            "input": {"zarr": str(self.input_zarr)},
            "coordinates": {"selection_method": self.selection_method},
            "selection": {
                "maximum_distance_m": self.maximum_distance_m,
                "maximum_seconds_from_each_observed_start": (
                    self.maximum_seconds_from_each_observed_start
                ),
            },
            "output": {
                "directory": str(self.output_directory),
                "chunks": {
                    "trajectory": self.chunk_trajectory,
                    "obs": self.chunk_obs,
                },
            },
        }


@dataclass(frozen=True)
class PairWorkflowResult:
    output_directory: Path
    zarr_path: Path
    catalog_path: Path
    platform_count: int
    possible_pair_count: int
    overlapping_pair_count: int
    selected_pair_count: int
    maximum_observations: int
    coordinate_methods: tuple[str, ...]

    def format(self) -> str:
        lines = [
            f"Candidate pairs: {self.selected_pair_count:,} selected from "
            f"{self.possible_pair_count:,} possible platform pairs",
            f"Pairs with temporal overlap: {self.overlapping_pair_count:,}",
            f"Grouped trajectories: {self.zarr_path}",
            f"Pair catalog: {self.catalog_path}",
            f"Coordinate representations: {', '.join(self.coordinate_methods)}",
            "Input trajectories, coordinate values, and valid lifetimes were read only",
        ]
        return "\n".join(lines)


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        if value is None:
            raise ValueError(f"{name} is required and cannot be null")
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _optional_nonnegative(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric or null")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative or null")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_pair_config(path: str | Path) -> PairWorkflowConfig:
    """Read and strictly validate a candidate-pair YAML file."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8-sig") as stream:
        values = yaml.safe_load(stream)
    values = _mapping(values, {"input", "coordinates", "selection", "output"}, "configuration")
    source = _mapping(values.get("input", {}), {"zarr"}, "input")
    coordinates = _mapping(
        values.get("coordinates", {}), {"selection_method"}, "coordinates",
    )
    selection = _mapping(
        values.get("selection", {}),
        {"maximum_distance_m", "maximum_seconds_from_each_observed_start"},
        "selection",
    )
    output = _mapping(values.get("output", {}), {"directory", "chunks"}, "output")
    chunks = _mapping(output.get("chunks", {}), {"trajectory", "obs"}, "output.chunks")

    def resolved(value: Any, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string")
        return (config_path.parent / Path(value).expanduser()).resolve()

    input_zarr = resolved(source.get("zarr"), "input.zarr")
    output_directory = resolved(output.get("directory"), "output.directory")
    if (
        input_zarr == output_directory
        or input_zarr in output_directory.parents
        or output_directory in input_zarr.parents
    ):
        raise ValueError("Pair output and input Zarr paths must not overlap")
    method = coordinates.get("selection_method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("coordinates.selection_method must be a nonempty method name")
    if not re.fullmatch(r"[A-Za-z0-9_]+", method.strip()):
        raise ValueError("coordinates.selection_method contains unsupported characters")
    return PairWorkflowConfig(
        input_zarr=input_zarr,
        selection_method=method.strip(),
        maximum_distance_m=_positive(
            selection.get("maximum_distance_m"), "selection.maximum_distance_m",
        ),
        maximum_seconds_from_each_observed_start=_optional_nonnegative(
            selection.get("maximum_seconds_from_each_observed_start"),
            "selection.maximum_seconds_from_each_observed_start",
        ),
        output_directory=output_directory,
        chunk_trajectory=_positive_integer(
            chunks.get("trajectory", 1), "output.chunks.trajectory",
        ),
        chunk_obs=_positive_integer(chunks.get("obs", 2016), "output.chunks.obs"),
    )


def _method_sort_key(method: str) -> tuple[int, int, str]:
    if method == "linear":
        return (0, 0, method)
    match = re.fullmatch(r"spline_(\d+)", method)
    if match:
        return (1, int(match.group(1)), method)
    return (2, 0, method)


def _coordinate_methods(dataset: xr.Dataset) -> tuple[str, ...]:
    longitude_methods: set[str] = set()
    latitude_methods: set[str] = set()
    for name in dataset.data_vars:
        match = _COORDINATE_PATTERN.fullmatch(str(name))
        if match:
            longitude_methods.add(match.group(1))
        latitude_match = re.fullmatch(r"^latitude_(.+)$", str(name))
        if latitude_match:
            latitude_methods.add(latitude_match.group(1))
    if longitude_methods != latitude_methods:
        raise ValueError(
            "Input Zarr has incomplete coordinate representations; "
            f"longitude-only={sorted(longitude_methods - latitude_methods)}, "
            f"latitude-only={sorted(latitude_methods - longitude_methods)}"
        )
    return tuple(sorted(longitude_methods, key=_method_sort_key))


def _validate_dataset(dataset: xr.Dataset, config: PairWorkflowConfig) -> tuple[str, ...]:
    if dataset.attrs.get("schema_version") != SUPPORTED_RECONSTRUCTION_SCHEMA_VERSION:
        raise ValueError(
            "Input has unsupported reconstruction schema "
            f"{dataset.attrs.get('schema_version')!r}; expected "
            f"{SUPPORTED_RECONSTRUCTION_SCHEMA_VERSION!r}"
        )
    required = {"platform_id", "start_time", "start_lon", "start_lat", "time"}
    missing = required - set(dataset.variables)
    if missing:
        raise ValueError(f"Input Zarr is missing required variables: {sorted(missing)}")
    if set(dataset.sizes) != {"platform", "time"}:
        raise ValueError("Input Zarr must have exactly platform and time dimensions")
    if dataset.sizes["platform"] < 2:
        raise ValueError("Pair building requires at least two platforms")
    expected_dims = {
        "platform_id": ("platform",), "start_time": ("platform",),
        "start_lon": ("platform",), "start_lat": ("platform",), "time": ("time",),
    }
    for name, dimensions in expected_dims.items():
        if dataset[name].dims != dimensions:
            raise ValueError(f"{name} must have dimensions {dimensions}")
    methods = _coordinate_methods(dataset)
    if not methods:
        raise ValueError("Input Zarr has no complete longitude/latitude coordinate representation")
    if config.selection_method not in methods:
        raise ValueError(
            f"Selection method {config.selection_method!r} is unavailable; found {list(methods)}"
        )
    for method in methods:
        for prefix in ("longitude", "latitude"):
            name = f"{prefix}_{method}"
            if dataset[name].dims != ("platform", "time"):
                raise ValueError(f"{name} must have dimensions ('platform', 'time')")
    identifiers = dataset.platform_id.values.astype(str)
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("Input Zarr contains duplicate platform identifiers")
    if any(not value.strip() for value in identifiers):
        raise ValueError("Input Zarr contains an empty platform identifier")
    start_time = dataset.start_time.values.astype("datetime64[ns]")
    start_lon = dataset.start_lon.values.astype(float)
    start_lat = dataset.start_lat.values.astype(float)
    if np.isnat(start_time).any() or not np.isfinite(start_lon).all() or not np.isfinite(start_lat).all():
        raise ValueError("Observed starts must contain finite coordinates and valid timestamps")
    if np.any((start_lon < -180) | (start_lon >= 180) | (start_lat < -90) | (start_lat > 90)):
        raise ValueError("Observed starts contain coordinates outside valid longitude/latitude bounds")
    return methods


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_metadata_hash(path: Path) -> str:
    metadata = path / ".zmetadata"
    if not metadata.is_file():
        raise ValueError(f"Input Zarr is not consolidated: {path}")
    return _file_sha256(metadata)


def _format_utc(value: np.datetime64) -> str:
    return pd.Timestamp(value).tz_localize("UTC").isoformat().replace("+00:00", "Z")


def _one_dimensional(
    root: zarr.Group, name: str, values: np.ndarray, compressor: Blosc,
    *, chunks: int, attributes: dict[str, Any] | None = None,
) -> zarr.Array:
    array = root.create_dataset(
        name, data=values, chunks=(min(chunks, len(values)),),
        compressor=compressor, fill_value=None,
    )
    array.attrs.update({"_ARRAY_DIMENSIONS": ["trajectory"], **(attributes or {})})
    return array


def _position_attributes(name: str, method: str | None = None) -> dict[str, Any]:
    coordinate = "longitude" if "lon" in name else "latitude"
    result: dict[str, Any] = {
        "_ARRAY_DIMENSIONS": ["trajectory", "obs"],
        "standard_name": coordinate,
        "units": "degrees_east" if coordinate == "longitude" else "degrees_north",
    }
    if method is not None:
        result["coordinate_method"] = method
    return result


def _create_store(
    path: Path, candidates: tuple[PairCandidate, ...], methods: tuple[str, ...],
    time: np.ndarray, starts: np.ndarray, config: PairWorkflowConfig,
    search: PairSearchResult, input_attributes: dict[str, Any], input_metadata_sha256: str,
) -> zarr.Group:
    pair_count = len(candidates)
    maximum_obs = max(item.post_encounter_observations for item in candidates)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update({
        "title": "Candidate drifter encounter pairs",
        "product_status": "candidate_pairs",
        "schema_version": PAIR_SCHEMA_VERSION,
        "algorithm_version": PAIR_ALGORITHM_VERSION,
        "drifterlab_version": __version__,
        "processing_time_utc": datetime.now(timezone.utc).isoformat(),
        "feature_type": "trajectory",
        "Conventions": "CF-1.10",
        "grouped_trajectory_layout": "kinematicParcels grouped entity; one trajectory per pair",
        "canonical_coordinate_method": config.selection_method,
        "available_coordinate_methods": list(methods),
        "encounter_policy": (
            "First common stored timestamp at which WGS84 geodesic member distance is less "
            "than or equal to maximum_distance_m"
        ),
        "deployment_window_policy": (
            "When finite, encounter_time - observed_start_time must not exceed "
            "maximum_seconds_from_each_observed_start for either member; null searches the "
            "full common valid lifetime"
        ),
        "post_encounter_policy": (
            "obs=0 is the selected encounter; rows continue on the source time grid through "
            "the final common valid timestamp of the canonical coordinate method"
        ),
        "time_convention": "UTC; timezone-naive datetime64 values represent UTC",
        "distance_method": "WGS84 geodesic (pyproj.Geod.inv)",
        "maximum_distance_m": config.maximum_distance_m,
        "maximum_seconds_from_each_observed_start": (
            config.maximum_seconds_from_each_observed_start
        ),
        "platform_count": len(starts),
        "possible_pair_count": search.possible_pair_count,
        "overlapping_pair_count": search.overlapping_pair_count,
        "selected_pair_count": pair_count,
        "effective_configuration": config.effective(),
        "source_zarr": str(config.input_zarr),
        "source_zarr_metadata_sha256": input_metadata_sha256,
        "source_reconstruction_schema_version": input_attributes.get("schema_version"),
        "source_reconstruction_algorithm_version": input_attributes.get("algorithm_version"),
        "source_build_report_sha256": input_attributes.get("build_report_sha256"),
    })
    trajectory_chunk = min(config.chunk_trajectory, pair_count)
    obs_chunk = min(config.chunk_obs, maximum_obs)
    chunks = (trajectory_chunk, obs_chunk)

    trajectory = root.create_dataset(
        "trajectory", data=np.arange(pair_count, dtype=np.int64),
        chunks=(trajectory_chunk,), compressor=compressor, fill_value=None,
    )
    trajectory.attrs.update({"_ARRAY_DIMENSIONS": ["trajectory"], "cf_role": "trajectory_id"})
    obs = root.create_dataset(
        "obs", data=np.arange(maximum_obs, dtype=np.int32),
        chunks=(obs_chunk,), compressor=compressor, fill_value=None,
    )
    obs.attrs.update({"_ARRAY_DIMENSIONS": ["obs"], "long_name": "observation index after encounter"})

    _one_dimensional(
        root, "group_id", np.arange(pair_count, dtype=np.int64), compressor,
        chunks=config.chunk_trajectory,
    )
    _one_dimensional(
        root, "group_size", np.full(pair_count, 2, dtype=np.int32), compressor,
        chunks=config.chunk_trajectory,
    )
    width = max(
        1,
        *(len(item.platform_id_1) for item in candidates),
        *(len(item.platform_id_2) for item in candidates),
    )
    _one_dimensional(
        root, "platform_code_1",
        np.asarray([item.platform_id_1 for item in candidates], dtype=f"<U{width}"),
        compressor, chunks=config.chunk_trajectory,
        attributes={"long_name": "platform identifier of deterministic first pair member"},
    )
    _one_dimensional(
        root, "platform_code_2",
        np.asarray([item.platform_id_2 for item in candidates], dtype=f"<U{width}"),
        compressor, chunks=config.chunk_trajectory,
        attributes={"long_name": "platform identifier of deterministic second pair member"},
    )

    encounter_times = np.asarray(
        [time[item.encounter_index] for item in candidates], dtype="datetime64[ns]",
    )
    overlap_starts = np.asarray(
        [time[item.overlap_start_index] for item in candidates], dtype="datetime64[ns]",
    )
    overlap_ends = np.asarray(
        [time[item.overlap_end_index] for item in candidates], dtype="datetime64[ns]",
    )
    for name, values, long_name in (
        ("encounter_time", encounter_times, "selected first distance-threshold crossing"),
        (
            "observed_start_time_1",
            np.asarray([starts[item.platform_index_1] for item in candidates], dtype="datetime64[ns]"),
            "first retained QC fix of member 1",
        ),
        (
            "observed_start_time_2",
            np.asarray([starts[item.platform_index_2] for item in candidates], dtype="datetime64[ns]"),
            "first retained QC fix of member 2",
        ),
        ("overlap_start_time", overlap_starts, "first common valid canonical timestamp"),
        ("overlap_end_time", overlap_ends, "last common valid canonical timestamp"),
    ):
        _one_dimensional(
            root, name, values, compressor, chunks=config.chunk_trajectory,
            attributes={"standard_name": "time", "timezone": "UTC", "long_name": long_name},
        )
    duration_seconds = (
        overlap_ends.astype(np.int64) - encounter_times.astype(np.int64)
    ) / 1_000_000_000
    one_dimensional_values = (
        (
            "encounter_distance_m",
            np.asarray([item.encounter_distance_m for item in candidates], dtype=np.float64),
            "m", "canonical member distance at selected encounter",
        ),
        (
            "encounter_delay_seconds_1",
            np.asarray([item.encounter_delay_seconds_1 for item in candidates], dtype=np.float64),
            "s", "encounter delay from observed start of member 1",
        ),
        (
            "encounter_delay_seconds_2",
            np.asarray([item.encounter_delay_seconds_2 for item in candidates], dtype=np.float64),
            "s", "encounter delay from observed start of member 2",
        ),
        (
            "post_encounter_duration_seconds", duration_seconds.astype(np.float64),
            "s", "duration from encounter through final common valid timestamp",
        ),
        (
            "post_encounter_observations",
            np.asarray([item.post_encounter_observations for item in candidates], dtype=np.int64),
            "1", "number of published observations for this pair",
        ),
    )
    for name, values, units, long_name in one_dimensional_values:
        _one_dimensional(
            root, name, values, compressor, chunks=config.chunk_trajectory,
            attributes={"units": units, "long_name": long_name},
        )
    for method in methods:
        _one_dimensional(
            root, f"encounter_distance_{method}_m",
            np.full(pair_count, np.nan, dtype=np.float64), compressor,
            chunks=config.chunk_trajectory,
            attributes={
                "units": "m",
                "long_name": f"{method} member distance at the canonical selected encounter",
                "coordinate_method": method,
            },
        )

    time_array = root.create_dataset(
        "time", shape=(pair_count, maximum_obs), chunks=chunks, dtype="datetime64[ns]",
        fill_value=np.datetime64("NaT", "ns"), compressor=compressor,
    )
    time_array.attrs.update({
        "_ARRAY_DIMENSIONS": ["trajectory", "obs"],
        "standard_name": "time", "timezone": "UTC",
    })
    all_position_names = [
        "lon", "lat", "center_lon", "center_lat", "lon_1", "lat_1", "lon_2", "lat_2",
    ]
    for method in methods:
        all_position_names.extend([
            f"lon_{method}_1", f"lat_{method}_1", f"lon_{method}_2", f"lat_{method}_2",
            f"center_lon_{method}", f"center_lat_{method}",
        ])
    for name in all_position_names:
        method = next((value for value in methods if f"_{value}" in name), None)
        array = root.create_dataset(
            name, shape=(pair_count, maximum_obs), chunks=chunks, dtype="f8",
            fill_value=np.nan, compressor=compressor,
        )
        array.attrs.update(_position_attributes(name, method))
    z_values = root.create_dataset(
        "z", shape=(pair_count, maximum_obs), chunks=chunks, dtype="f8",
        fill_value=np.nan, compressor=compressor,
    )
    z_values.attrs.update({
        "_ARRAY_DIMENSIONS": ["trajectory", "obs"], "units": "m",
        "positive": "down", "long_name": "group center depth",
    })
    return root


def _write_pairs(
    root: zarr.Group, dataset: xr.Dataset, candidates: tuple[PairCandidate, ...],
    methods: tuple[str, ...], config: PairWorkflowConfig, time: np.ndarray,
    report: Callable[[str], None],
) -> pd.DataFrame:
    selection_method = config.selection_method
    rows: list[dict[str, Any]] = []
    starts = dataset.start_time.values.astype("datetime64[ns]")
    for trajectory, candidate in enumerate(candidates):
        source_slice = slice(candidate.encounter_index, candidate.overlap_end_index + 1)
        count = candidate.post_encounter_observations
        root["time"][trajectory, :count] = time[source_slice]
        root["z"][trajectory, :count] = 0.0
        method_distances: dict[str, float] = {}
        for method in methods:
            lon_1 = np.asarray(dataset[f"longitude_{method}"][
                candidate.platform_index_1, source_slice
            ].values, dtype=float)
            lat_1 = np.asarray(dataset[f"latitude_{method}"][
                candidate.platform_index_1, source_slice
            ].values, dtype=float)
            lon_2 = np.asarray(dataset[f"longitude_{method}"][
                candidate.platform_index_2, source_slice
            ].values, dtype=float)
            lat_2 = np.asarray(dataset[f"latitude_{method}"][
                candidate.platform_index_2, source_slice
            ].values, dtype=float)
            if not (
                np.isfinite(lon_1[0]) and np.isfinite(lat_1[0])
                and np.isfinite(lon_2[0]) and np.isfinite(lat_2[0])
            ):
                raise ValueError(
                    f"Coordinate method {method!r} is invalid at the selected encounter for "
                    f"{candidate.platform_id_1}/{candidate.platform_id_2}"
                )
            center_lon = circular_mean_longitude(lon_1, lon_2)
            center_lat = (lat_1 + lat_2) / 2.0
            values = {
                f"lon_{method}_1": lon_1, f"lat_{method}_1": lat_1,
                f"lon_{method}_2": lon_2, f"lat_{method}_2": lat_2,
                f"center_lon_{method}": center_lon, f"center_lat_{method}": center_lat,
            }
            for name, data in values.items():
                root[name][trajectory, :count] = data
            _azimuth_1, _azimuth_2, distance = WGS84.inv(
                float(lon_1[0]), float(lat_1[0]), float(lon_2[0]), float(lat_2[0]),
            )
            method_distances[method] = float(distance)
            root[f"encounter_distance_{method}_m"][trajectory] = float(distance)
            if method == selection_method:
                for canonical, data in (
                    ("lon_1", lon_1), ("lat_1", lat_1), ("lon_2", lon_2),
                    ("lat_2", lat_2), ("center_lon", center_lon),
                    ("center_lat", center_lat), ("lon", center_lon), ("lat", center_lat),
                ):
                    root[canonical][trajectory, :count] = data

        encounter_time = time[candidate.encounter_index]
        overlap_start = time[candidate.overlap_start_index]
        overlap_end = time[candidate.overlap_end_index]
        row: dict[str, Any] = {
            "trajectory": trajectory,
            "group_id": trajectory,
            "group_size": 2,
            "platform_code_1": candidate.platform_id_1,
            "platform_code_2": candidate.platform_id_2,
            "observed_start_time_1_utc": _format_utc(starts[candidate.platform_index_1]),
            "observed_start_time_2_utc": _format_utc(starts[candidate.platform_index_2]),
            "overlap_start_time_utc": _format_utc(overlap_start),
            "overlap_end_time_utc": _format_utc(overlap_end),
            "encounter_time_utc": _format_utc(encounter_time),
            "encounter_distance_m": candidate.encounter_distance_m,
            "encounter_delay_seconds_1": candidate.encounter_delay_seconds_1,
            "encounter_delay_seconds_2": candidate.encounter_delay_seconds_2,
            "post_encounter_duration_seconds": float(
                (overlap_end - encounter_time) / np.timedelta64(1, "s")
            ),
            "post_encounter_observations": count,
            "canonical_coordinate_method": selection_method,
            "maximum_distance_m": config.maximum_distance_m,
            "maximum_seconds_from_each_observed_start": (
                config.maximum_seconds_from_each_observed_start
            ),
        }
        for method in methods:
            row[f"encounter_distance_{method}_m"] = method_distances[method]
        rows.append(row)
        if (trajectory + 1) % 100 == 0 or trajectory + 1 == len(candidates):
            report(f"Wrote grouped pair {trajectory + 1:,}/{len(candidates):,}")
    return pd.DataFrame(rows)


def _expected_data_variables(methods: tuple[str, ...]) -> set[str]:
    names = {
        "time", "lon", "lat", "z", "group_id", "group_size", "center_lon",
        "center_lat", "lon_1", "lat_1", "lon_2", "lat_2", "platform_code_1",
        "platform_code_2", "encounter_time", "encounter_distance_m",
        "encounter_delay_seconds_1", "encounter_delay_seconds_2",
        "observed_start_time_1", "observed_start_time_2", "overlap_start_time",
        "overlap_end_time", "post_encounter_duration_seconds",
        "post_encounter_observations",
    }
    for method in methods:
        names.update({
            f"lon_{method}_1", f"lat_{method}_1", f"lon_{method}_2",
            f"lat_{method}_2", f"center_lon_{method}", f"center_lat_{method}",
            f"encounter_distance_{method}_m",
        })
    return names


def _verify(
    zarr_path: Path, catalog_path: Path, candidates: tuple[PairCandidate, ...],
    methods: tuple[str, ...], selection_method: str,
) -> None:
    expected = _expected_data_variables(methods)
    maximum_obs = max(item.post_encounter_observations for item in candidates)
    with xr.open_zarr(zarr_path, consolidated=True, chunks=None) as dataset:
        expected_sizes = {"trajectory": len(candidates), "obs": maximum_obs}
        if dict(dataset.sizes) != expected_sizes:
            raise ValueError(f"Pair Zarr readback changed dimensions: {dict(dataset.sizes)}")
        if set(dataset.data_vars) != expected:
            missing = expected - set(dataset.data_vars)
            extra = set(dataset.data_vars) - expected
            raise ValueError(
                f"Pair Zarr readback schema differs; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if dataset.attrs.get("canonical_coordinate_method") != selection_method:
            raise ValueError("Pair Zarr readback changed the canonical coordinate method")
        if tuple(dataset.attrs.get("available_coordinate_methods", [])) != methods:
            raise ValueError("Pair Zarr readback changed the available coordinate methods")
        expected_1 = [item.platform_id_1 for item in candidates]
        expected_2 = [item.platform_id_2 for item in candidates]
        if dataset.platform_code_1.values.astype(str).tolist() != expected_1:
            raise ValueError("Pair Zarr readback changed first member identifiers")
        if dataset.platform_code_2.values.astype(str).tolist() != expected_2:
            raise ValueError("Pair Zarr readback changed second member identifiers")
        if not np.array_equal(dataset.group_id.values, np.arange(len(candidates))):
            raise ValueError("Pair Zarr readback changed deterministic group identifiers")
        if not np.all(dataset.group_size.values == 2):
            raise ValueError("Pair Zarr readback changed pair group sizes")
        for canonical, retained in (
            ("lon_1", f"lon_{selection_method}_1"),
            ("lat_1", f"lat_{selection_method}_1"),
            ("lon_2", f"lon_{selection_method}_2"),
            ("lat_2", f"lat_{selection_method}_2"),
            ("center_lon", f"center_lon_{selection_method}"),
            ("center_lat", f"center_lat_{selection_method}"),
        ):
            np.testing.assert_allclose(
                dataset[canonical].values, dataset[retained].values, equal_nan=True,
                err_msg=f"Canonical {canonical} differs from {retained}",
            )
        np.testing.assert_allclose(dataset.lon.values, dataset.center_lon.values, equal_nan=True)
        np.testing.assert_allclose(dataset.lat.values, dataset.center_lat.values, equal_nan=True)
        for index, candidate in enumerate(candidates):
            count = candidate.post_encounter_observations
            if dataset.time.values[index, 0] != dataset.encounter_time.values[index]:
                raise ValueError(f"Pair {index} does not begin at its selected encounter")
            if not np.isfinite(dataset.lon_1.values[index, 0]):
                raise ValueError(f"Pair {index} has invalid coordinates at obs=0")
            if count < maximum_obs:
                if not np.isnat(dataset.time.values[index, count:]).all():
                    raise ValueError(f"Pair {index} has non-NaT time padding")
                if not np.isnan(dataset.lon.values[index, count:]).all():
                    raise ValueError(f"Pair {index} has non-NaN coordinate padding")

    catalog = pd.read_csv(
        catalog_path, dtype={"platform_code_1": str, "platform_code_2": str},
    )
    if len(catalog) != len(candidates) or catalog.trajectory.tolist() != list(range(len(candidates))):
        raise ValueError("Pair catalog readback changed the selected pair inventory")


def _remove_generated(path: Path, parent: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith("."):
        raise RuntimeError(f"Refusing cleanup outside generated paths in {parent}")
    shutil.rmtree(resolved)


def _publish(temporary: Path, target: Path, *, token: str, overwrite: bool) -> None:
    if not target.exists():
        temporary.rename(target)
        return
    if not overwrite:
        raise FileExistsError(f"Pair output already exists: {target}; use --overwrite")
    backup = target.with_name(f".{target.name}.{token}.backup")
    if backup.exists():
        raise FileExistsError(f"Pair backup path already exists: {backup}")
    target.rename(backup)
    try:
        temporary.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    _remove_generated(backup, target.parent)


def _build_bundle(
    dataset: xr.Dataset, config: PairWorkflowConfig, methods: tuple[str, ...],
    search: PairSearchResult, time: np.ndarray, starts: np.ndarray,
    input_metadata_sha256: str, report: Callable[[str], None], *, overwrite: bool,
) -> None:
    target = config.output_directory
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = target.with_name(f".{target.name}.{token}.tmp")
    temporary.mkdir()
    zarr_path = temporary / ZARR_NAME
    catalog_path = temporary / CATALOG_NAME
    try:
        root = _create_store(
            zarr_path, search.candidates, methods, time, starts, config, search,
            dict(dataset.attrs), input_metadata_sha256,
        )
        catalog = _write_pairs(
            root, dataset, search.candidates, methods, config, time, report,
        )
        catalog.to_csv(catalog_path, index=False)
        root.attrs["pair_catalog_sha256"] = _file_sha256(catalog_path)
        if _input_metadata_hash(config.input_zarr) != input_metadata_sha256:
            raise ValueError("Input Zarr metadata changed during pair building")
        zarr.consolidate_metadata(str(zarr_path))
        _verify(
            zarr_path, catalog_path, search.candidates, methods, config.selection_method,
        )
        _publish(temporary, target, token=token, overwrite=overwrite)
    finally:
        if temporary.exists():
            _remove_generated(temporary, target.parent)


def run_pair_workflow(
    config_path: str | Path, *, overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PairWorkflowResult:
    """Discover candidate encounter pairs and publish a grouped-trajectory bundle."""
    config = load_pair_config(config_path)
    report = progress or (lambda message: None)
    if not config.input_zarr.is_dir():
        raise ValueError(f"Input Zarr does not exist: {config.input_zarr}")
    if config.output_directory.exists() and not overwrite:
        raise FileExistsError(
            f"Pair output already exists: {config.output_directory}; use --overwrite"
        )
    metadata_sha256 = _input_metadata_hash(config.input_zarr)
    with xr.open_zarr(config.input_zarr, consolidated=True, chunks=None) as dataset:
        methods = _validate_dataset(dataset, config)
        platform_ids = dataset.platform_id.values.astype(str)
        time = dataset.time.values.astype("datetime64[ns]")
        starts = dataset.start_time.values.astype("datetime64[ns]")
        report(
            f"Searching {len(platform_ids) * (len(platform_ids) - 1) // 2:,} pairs "
            f"with canonical coordinates {config.selection_method!r}"
        )
        search = find_candidate_pairs(
            platform_ids, time, starts,
            dataset[f"longitude_{config.selection_method}"].values,
            dataset[f"latitude_{config.selection_method}"].values,
            maximum_distance_m=config.maximum_distance_m,
            maximum_seconds_from_each_observed_start=(
                config.maximum_seconds_from_each_observed_start
            ),
            progress=report,
        )
        if not search.candidates:
            raise ValueError(
                "No candidate pairs met the temporal-overlap, deployment-window, and distance "
                "criteria; increase maximum_distance_m or set "
                "maximum_seconds_from_each_observed_start to null to search chance encounters"
            )
        report(
            f"Selected {len(search.candidates):,} candidate pairs; writing all coordinate methods"
        )
        _build_bundle(
            dataset, config, methods, search, time, starts, metadata_sha256, report,
            overwrite=overwrite,
        )
    maximum_obs = max(item.post_encounter_observations for item in search.candidates)
    return PairWorkflowResult(
        output_directory=config.output_directory,
        zarr_path=config.output_directory / ZARR_NAME,
        catalog_path=config.output_directory / CATALOG_NAME,
        platform_count=len(platform_ids),
        possible_pair_count=search.possible_pair_count,
        overlapping_pair_count=search.overlapping_pair_count,
        selected_pair_count=len(search.candidates),
        maximum_observations=maximum_obs,
        coordinate_methods=methods,
    )
