"""Build a common-grid trajectory Zarr from finalized position-QC Parquets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
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
import pyarrow.parquet as pq
import xarray as xr
import yaml
import zarr

from drifterlab import __version__
from drifterlab.reconstruction import PlatformReconstruction, reconstruct_platform
from drifterlab.workflows.position import POSITION_QC_SCHEMA_VERSION


RECONSTRUCTION_SCHEMA_VERSION = "1.1"
RECONSTRUCTION_ALGORITHM_VERSION = "phase-ensemble-natural-cubic-v1"
POSITION_QC_METADATA_KEY = b"drifterlab_position_qc"
ZARR_NAME = "trajectories.zarr"
REPORT_NAME = "build_report.csv"
REQUIRED_COLUMNS = {
    "platform_code", "source_sha256", "time", "source_lon", "source_lat",
    "final_position_status", "final_position_valid", "drogue_eligible",
    "final_drogue_status", "analysis_cutoff_time", "platform_qc_complete",
}
SUPPORTED_MAP_PROJECTIONS = {
    "PlateCarree", "Mercator", "SouthPolarStereo", "NorthPolarStereo", "Robinson",
}
METHOD_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class PlotPlatformSelection:
    dataset: str
    platform_id: str


@dataclass(frozen=True)
class AdditionalTrajectoryStore:
    label: str
    path: Path
    method: str


@dataclass(frozen=True)
class TrajectoryMapConfig:
    projection: str
    central_longitude: float
    extent: tuple[float, float, float, float] | None
    figsize: tuple[float, float]
    dpi: int
    land: bool
    coastlines: bool
    gridlines: bool
    label_starts: bool


@dataclass(frozen=True)
class TrajectoryPlottingConfig:
    enabled: bool
    output_directory: Path
    dataset_label: str
    method: str
    start_time: np.datetime64
    end_time: np.datetime64
    platforms: tuple[PlotPlatformSelection, ...]
    check_platforms: tuple[PlotPlatformSelection, ...]
    map: TrajectoryMapConfig
    additional_stores: tuple[AdditionalTrajectoryStore, ...]


@dataclass(frozen=True)
class ReconstructionWorkflowConfig:
    input_directory: Path
    input_pattern: str
    start_time: np.datetime64
    end_time: np.datetime64
    dt_minutes: float
    periods_minutes: tuple[int, ...]
    long_gap_threshold_minutes: dict[int, float]
    output_directory: Path
    chunk_platform: int
    chunk_time: int
    plotting: TrajectoryPlottingConfig

    def effective(self) -> dict[str, Any]:
        return {
            "input": {
                "directory": str(self.input_directory), "pattern": self.input_pattern,
            },
            "grid": {
                "start_time": _format_utc(self.start_time),
                "end_time": None if np.isnat(self.end_time) else _format_utc(self.end_time),
                "dt_minutes": self.dt_minutes,
                "trim_all_nan_leading": True,
            },
            "spline": {
                "period_minutes": list(self.periods_minutes),
                "long_gap_threshold_minutes": {
                    str(key): value for key, value in self.long_gap_threshold_minutes.items()
                },
            },
            "output": {
                "directory": str(self.output_directory),
                "chunks": {"platform": self.chunk_platform, "time": self.chunk_time},
            },
        }


@dataclass(frozen=True)
class ReconstructionWorkflowResult:
    output_directory: Path
    zarr_path: Path
    report_path: Path
    platform_count: int
    time_count: int
    configured_start_time: np.datetime64
    published_start_time: np.datetime64
    end_time: np.datetime64
    action: str
    figure_paths: tuple[Path, ...]

    def format(self) -> str:
        return "\n".join([
            f"Candidate trajectories ({self.action}): {self.platform_count} platforms x {self.time_count} times",
            f"Published UTC grid: {_format_utc(self.published_start_time)} to {_format_utc(self.end_time)}",
            f"Zarr: {self.zarr_path}", f"Build report: {self.report_path}",
            *( [f"Figures: {len(self.figure_paths)} in {self.figure_paths[0].parent}"]
               if self.figure_paths else [] ),
            "Product status: candidate_pending_gap_review",
        ])


@dataclass(frozen=True)
class QCInventory:
    path: Path
    sha256: str
    platform_id: str
    source_sha256: str
    first_time: np.datetime64
    last_time: np.datetime64
    accepted_count: int
    first_longitude: float
    first_latitude: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AcceptedTrack:
    inventory: QCInventory
    time: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _method(value: Any, name: str) -> str:
    result = _nonempty_string(value, name)
    if not METHOD_PATTERN.fullmatch(result):
        raise ValueError(f"{name} must contain only letters, digits, and underscores")
    return result


def _utc(value: Any, name: str, *, optional: bool = False) -> np.datetime64:
    if optional and (value is None or str(value).strip() in {"", "null", "None"}):
        return np.datetime64("NaT", "ns")
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an explicit UTC timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must be an explicit UTC timestamp")
    return stamp.tz_convert("UTC").tz_localize(None).to_datetime64().astype("datetime64[ns]")


def _format_utc(value: np.datetime64) -> str:
    stamp = pd.Timestamp(value, tz="UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def load_reconstruction_config(path: str | Path) -> ReconstructionWorkflowConfig:
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    data = _mapping(
        data, {"input", "grid", "spline", "output", "plotting"}, "configuration",
    )
    source = _mapping(data.get("input", {}), {"directory", "pattern"}, "input")
    grid = _mapping(data.get("grid", {}), {"start_time", "end_time", "dt_minutes"}, "grid")
    spline = _mapping(
        data.get("spline", {}), {"period_minutes", "long_gap_threshold_minutes"}, "spline",
    )
    output = _mapping(data.get("output", {}), {"directory", "chunks"}, "output")
    chunks = _mapping(output.get("chunks", {}), {"platform", "time"}, "output.chunks")
    plotting = _mapping(
        data.get("plotting", {}),
        {
            "enabled", "output_directory", "dataset_label", "method", "time",
            "platforms", "check_platforms", "map", "additional_stores",
        },
        "plotting",
    )
    plot_time = _mapping(plotting.get("time", {}), {"start", "end"}, "plotting.time")
    plot_map = _mapping(
        plotting.get("map", {}),
        {
            "projection", "central_longitude", "extent", "figsize", "dpi",
            "land", "coastlines", "gridlines", "label_starts",
        },
        "plotting.map",
    )

    def resolved(value: Any, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string")
        candidate = Path(value).expanduser()
        return (source_path.parent / candidate).resolve()

    pattern = source.get("pattern", "*.parquet")
    if (not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute()
            or ".." in Path(pattern).parts):
        raise ValueError("input.pattern must be a relative glob without '..'")
    start = _utc(grid.get("start_time"), "grid.start_time")
    end = _utc(grid.get("end_time"), "grid.end_time", optional=True)
    dt_minutes = _positive_number(grid.get("dt_minutes", 5), "grid.dt_minutes")
    dt_ns = int(round(dt_minutes * 60_000_000_000))
    if not np.isclose(dt_ns, dt_minutes * 60_000_000_000):
        raise ValueError("grid.dt_minutes must be exactly representable in nanoseconds")

    raw_periods = spline.get("period_minutes", [15, 30, 60])
    if not isinstance(raw_periods, list) or not raw_periods:
        raise ValueError("spline.period_minutes must be a nonempty list")
    periods: list[int] = []
    for raw in raw_periods:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError("Every spline period must be a positive integer number of minutes")
        if (raw * 60_000_000_000) % dt_ns:
            raise ValueError(f"Spline period {raw} minutes is not an integer multiple of dt")
        periods.append(raw)
    if len(periods) != len(set(periods)):
        raise ValueError("Spline periods must be unique")
    periods = sorted(periods)

    raw_thresholds = spline.get("long_gap_threshold_minutes", {})
    if not isinstance(raw_thresholds, dict):
        raise ValueError("spline.long_gap_threshold_minutes must be a mapping")
    thresholds: dict[int, float] = {period: float(period) for period in periods}
    for raw_key, raw_value in raw_thresholds.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError("Long-gap threshold keys must be configured spline periods") from exc
        if key not in periods or str(key) != str(raw_key):
            raise ValueError(f"Long-gap threshold key {raw_key!r} is not a configured period")
        thresholds[key] = _positive_number(
            raw_value, f"spline.long_gap_threshold_minutes.{key}",
        )

    chunk_values: list[int] = []
    for name, default in (("platform", 1), ("time", 2016)):
        value = chunks.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"output.chunks.{name} must be a positive integer")
        chunk_values.append(value)
    input_directory = resolved(source.get("directory"), "input.directory")
    output_directory = resolved(output.get("directory"), "output.directory")
    if output_directory == input_directory or input_directory in output_directory.parents:
        raise ValueError("Reconstruction output must be outside the QC input directory")

    plotting_enabled = _boolean(plotting.get("enabled", False), "plotting.enabled")
    raw_plot_output = plotting.get("output_directory")
    if raw_plot_output is None:
        plot_output_directory = output_directory.with_name(output_directory.name + "_figures")
    else:
        plot_output_directory = resolved(raw_plot_output, "plotting.output_directory")
    if (plot_output_directory == output_directory
            or output_directory in plot_output_directory.parents):
        raise ValueError("plotting.output_directory must be outside the reconstruction bundle")
    dataset_label = _nonempty_string(
        plotting.get("dataset_label", "current"), "plotting.dataset_label",
    )
    plot_method = _method(plotting.get("method", "linear"), "plotting.method")
    plot_start = _utc(plot_time.get("start"), "plotting.time.start", optional=True)
    plot_end = _utc(plot_time.get("end"), "plotting.time.end", optional=True)
    if not np.isnat(plot_start) and not np.isnat(plot_end) and plot_end < plot_start:
        raise ValueError("plotting.time.end must not precede plotting.time.start")

    projection = _nonempty_string(
        plot_map.get("projection", "PlateCarree"), "plotting.map.projection",
    )
    if projection not in SUPPORTED_MAP_PROJECTIONS:
        raise ValueError(
            f"plotting.map.projection must be one of {sorted(SUPPORTED_MAP_PROJECTIONS)}"
        )
    central_longitude = float(plot_map.get("central_longitude", 0))
    if not math.isfinite(central_longitude) or not -180 <= central_longitude <= 180:
        raise ValueError("plotting.map.central_longitude must be between -180 and 180")
    raw_extent = plot_map.get("extent")
    extent: tuple[float, float, float, float] | None = None
    if raw_extent is not None:
        if not isinstance(raw_extent, list) or len(raw_extent) != 4:
            raise ValueError("plotting.map.extent must be null or [west, east, south, north]")
        extent = tuple(float(value) for value in raw_extent)
        if (not all(math.isfinite(value) for value in extent)
                or extent[0] >= extent[1] or extent[2] >= extent[3]
                or extent[2] < -90 or extent[3] > 90):
            raise ValueError("plotting.map.extent contains invalid geographic bounds")
    raw_figsize = plot_map.get("figsize", [12, 8])
    if not isinstance(raw_figsize, list) or len(raw_figsize) != 2:
        raise ValueError("plotting.map.figsize must contain width and height")
    figsize = tuple(_positive_number(value, "plotting.map.figsize") for value in raw_figsize)
    dpi = plot_map.get("dpi", 150)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("plotting.map.dpi must be a positive integer")
    map_config = TrajectoryMapConfig(
        projection, central_longitude, extent, figsize, dpi,
        _boolean(plot_map.get("land", True), "plotting.map.land"),
        _boolean(plot_map.get("coastlines", True), "plotting.map.coastlines"),
        _boolean(plot_map.get("gridlines", True), "plotting.map.gridlines"),
        _boolean(plot_map.get("label_starts", False), "plotting.map.label_starts"),
    )

    raw_stores = plotting.get("additional_stores", [])
    if not isinstance(raw_stores, list):
        raise ValueError("plotting.additional_stores must be a list")
    additional_stores: list[AdditionalTrajectoryStore] = []
    labels = {dataset_label}
    for index, raw_store in enumerate(raw_stores):
        item = _mapping(raw_store, {"label", "path", "method"}, f"plotting.additional_stores[{index}]")
        label = _nonempty_string(item.get("label"), f"plotting.additional_stores[{index}].label")
        if label in labels:
            raise ValueError(f"Duplicate plotting dataset label: {label}")
        labels.add(label)
        additional_stores.append(AdditionalTrajectoryStore(
            label, resolved(item.get("path"), f"plotting.additional_stores[{index}].path"),
            _method(item.get("method"), f"plotting.additional_stores[{index}].method"),
        ))

    def selections(key: str) -> tuple[PlotPlatformSelection, ...]:
        raw_values = plotting.get(key, [])
        if not isinstance(raw_values, list):
            raise ValueError(f"plotting.{key} must be a list")
        values: list[PlotPlatformSelection] = []
        seen: set[tuple[str, str]] = set()
        for index, raw_value in enumerate(raw_values):
            item = _mapping(raw_value, {"dataset", "platform_id"}, f"plotting.{key}[{index}]")
            label = _nonempty_string(item.get("dataset"), f"plotting.{key}[{index}].dataset")
            platform = _nonempty_string(
                item.get("platform_id"), f"plotting.{key}[{index}].platform_id",
            )
            if label not in labels:
                raise ValueError(f"plotting.{key}[{index}] uses unknown dataset label {label!r}")
            pair = (label, platform)
            if pair in seen:
                raise ValueError(f"Duplicate plotting platform selection: {pair}")
            seen.add(pair)
            values.append(PlotPlatformSelection(*pair))
        return tuple(values)

    plotting_config = TrajectoryPlottingConfig(
        plotting_enabled, plot_output_directory, dataset_label, plot_method,
        plot_start, plot_end, selections("platforms"), selections("check_platforms"),
        map_config, tuple(additional_stores),
    )
    return ReconstructionWorkflowConfig(
        input_directory, pattern, start, end, dt_minutes, tuple(periods), thresholds,
        output_directory, *chunk_values, plotting_config,
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parquet_metadata(path: Path) -> dict[str, Any]:
    try:
        raw = (pq.read_metadata(path).metadata or {}).get(POSITION_QC_METADATA_KEY)
        return json.loads(raw.decode()) if raw else {}
    except Exception as exc:
        raise ValueError(f"Cannot read position-QC metadata: {path}") from exc


def _load_track(path: Path) -> AcceptedTrack:
    before = _file_sha256(path)
    metadata = _parquet_metadata(path)
    if metadata.get("schema_version") != POSITION_QC_SCHEMA_VERSION:
        raise ValueError(
            f"Position QC file has unsupported schema {metadata.get('schema_version')!r}; "
            f"expected {POSITION_QC_SCHEMA_VERSION}: {path.name}"
        )
    required_footer = {
        "algorithm_version", "config_sha256", "platform_code", "product_layout",
        "resolution_policy", "source_sha256", "platform_qc_complete",
    }
    missing_footer = required_footer - set(metadata)
    if missing_footer:
        raise ValueError(
            f"Position QC footer is missing fields {sorted(missing_footer)}: {path.name}"
        )
    if metadata.get("product_layout") != "per-trajectory-v2":
        raise ValueError(f"Position QC file has unsupported product layout: {path.name}")
    for name in ("algorithm_version", "config_sha256", "resolution_policy", "source_sha256"):
        if not isinstance(metadata.get(name), str) or not metadata[name].strip():
            raise ValueError(f"Position QC footer has invalid {name}: {path.name}")
    if metadata.get("platform_qc_complete") is not True:
        raise ValueError(f"Position QC footer is incomplete for {path.name}")
    columns = set(pq.read_schema(path).names)
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"Position QC file is missing columns {sorted(missing)}: {path.name}")
    frame = pd.read_parquet(path, columns=sorted(REQUIRED_COLUMNS))
    after = _file_sha256(path)
    if after != before:
        raise ValueError(f"Position QC input changed while being read: {path}")
    if frame.empty:
        raise ValueError(f"Position QC file is empty: {path.name}")

    identities = frame.platform_code.astype(str).unique().tolist()
    footer_platform = str(metadata.get("platform_code", ""))
    if len(identities) != 1 or not footer_platform or identities[0] != footer_platform:
        raise ValueError(f"Inconsistent platform identity in {path.name}")
    platform = identities[0]
    if path.stem != platform:
        raise ValueError(
            f"Position QC filename, footer, and rows disagree for platform {platform}: {path.name}"
        )
    source_hashes = frame.source_sha256.astype(str).unique().tolist()
    if len(source_hashes) != 1 or source_hashes[0] != str(metadata.get("source_sha256", "")):
        raise ValueError(f"Inconsistent native-source identity in {path.name}")

    status = frame.final_position_status.fillna("").astype(str)
    unresolved = status.isin(["unresolved", "uncertain"])
    if unresolved.any() or not bool(metadata.get("platform_qc_complete", False)):
        raise ValueError(f"Position QC is unresolved for platform {platform}")
    if frame.final_position_valid.isna().any() or frame.drogue_eligible.isna().any():
        raise ValueError(f"Position QC validity contains missing values for platform {platform}")
    final_valid = frame.final_position_valid.to_numpy(dtype=bool)
    if not np.array_equal(final_valid, status.eq("valid").to_numpy()):
        raise ValueError(f"Final position status/validity disagree for platform {platform}")
    if frame.platform_qc_complete.isna().any() or not frame.platform_qc_complete.to_numpy(dtype=bool).all():
        raise ValueError(f"Position QC is incomplete for platform {platform}")

    drogue_states = frame.final_drogue_status.fillna("").astype(str).unique().tolist()
    if len(drogue_states) != 1 or drogue_states[0] not in {"lost", "not_lost"}:
        raise ValueError(f"Drogue decision is unresolved or inconsistent for platform {platform}")
    accepted = status.eq("valid").to_numpy() & final_valid & frame.drogue_eligible.to_numpy(dtype=bool)
    selected = frame.loc[accepted, ["time", "source_lon", "source_lat"]].copy()
    parsed = pd.to_datetime(selected.time, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"Accepted QC positions have missing timestamps for platform {platform}")
    selected["time"] = parsed
    selected = selected.sort_values("time", kind="stable")
    if selected.time.duplicated().any():
        raise ValueError(f"Accepted QC positions have duplicate timestamps for platform {platform}")
    if len(selected) < 2:
        raise ValueError(f"Platform {platform} has fewer than two accepted QC positions")
    lon = selected.source_lon.to_numpy(dtype=float)
    lat = selected.source_lat.to_numpy(dtype=float)
    if (not np.isfinite(lon).all() or not np.isfinite(lat).all()
            or np.any((lon < -180) | (lon > 180)) or np.any((lat < -90) | (lat > 90))):
        raise ValueError(f"Accepted QC positions contain invalid coordinates for platform {platform}")
    time = selected.time.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    if drogue_states[0] == "lost":
        cutoffs = pd.to_datetime(frame.analysis_cutoff_time, errors="coerce", utc=True).dropna().unique()
        if len(cutoffs) != 1:
            raise ValueError(f"Lost drogue has no single recorded cutoff for platform {platform}")
        cutoff = pd.Timestamp(cutoffs[0]).tz_localize(None).to_datetime64().astype("datetime64[ns]")
        if np.any(time >= cutoff):
            raise ValueError(f"Accepted QC position reaches or exceeds the drogue cutoff for {platform}")
    inventory = QCInventory(
        path.resolve(), before, platform, source_hashes[0], time[0], time[-1], len(time),
        float(lon[0]), float(lat[0]), metadata,
    )
    return AcceptedTrack(inventory, time, lon, lat)


def _discover(config: ReconstructionWorkflowConfig) -> tuple[Path, ...]:
    if not config.input_directory.is_dir():
        raise ValueError(f"Position-QC input directory does not exist: {config.input_directory}")
    paths = tuple(sorted(
        (path.resolve() for path in config.input_directory.glob(config.input_pattern) if path.is_file()),
        key=lambda path: path.name,
    ))
    if not paths:
        raise ValueError(
            f"No position-QC files match {config.input_pattern!r} in {config.input_directory}"
        )
    return paths


def _ceil_div(value: int, divisor: int) -> int:
    return -((-value) // divisor)


def _grid(
    config: ReconstructionWorkflowConfig, inventories: list[QCInventory],
) -> tuple[np.ndarray, np.ndarray, np.datetime64]:
    anchor_ns = int(config.start_time.astype(np.int64))
    dt_ns = int(round(config.dt_minutes * 60_000_000_000))
    early = [item for item in inventories if item.first_time < config.start_time]
    if early:
        details = ", ".join(
            f"{item.platform_id}={_format_utc(item.first_time)}" for item in early
        )
        raise ValueError(f"Eligible QC positions predate configured start_time: {details}")
    latest = max(item.last_time for item in inventories)
    latest_ns = int(latest.astype(np.int64))
    if np.isnat(config.end_time):
        end_index = (latest_ns - anchor_ns) // dt_ns
    else:
        end_ns = int(config.end_time.astype(np.int64))
        if end_ns < anchor_ns or (end_ns - anchor_ns) % dt_ns:
            raise ValueError("Explicit end_time must be on the grid anchored at start_time")
        if end_ns < latest_ns:
            raise ValueError(
                f"Explicit end_time {_format_utc(config.end_time)} would omit eligible data ending "
                f"at {_format_utc(latest)}"
            )
        end_index = (end_ns - anchor_ns) // dt_ns

    first_indices: list[int] = []
    no_grid: list[str] = []
    for item in inventories:
        first_index = _ceil_div(int(item.first_time.astype(np.int64)) - anchor_ns, dt_ns)
        last_index = (int(item.last_time.astype(np.int64)) - anchor_ns) // dt_ns
        if first_index > last_index:
            no_grid.append(item.platform_id)
        else:
            first_indices.append(first_index)
    if no_grid:
        raise ValueError(f"Platforms have no grid instant inside their eligible lifespan: {no_grid}")
    published_start_index = min(first_indices)
    if published_start_index > end_index:
        raise ValueError("The configured time grid contains no reconstructible positions")
    indices = np.arange(published_start_index, end_index + 1, dtype=np.int64)
    values = anchor_ns + indices * dt_ns
    grid = values.astype("datetime64[ns]")
    return grid, indices, grid[0]


def _position_attrs(method: str, coordinate: str) -> dict[str, Any]:
    return {
        "_ARRAY_DIMENSIONS": ["platform", "time"],
        "coordinates": "platform_id",
        "standard_name": "longitude" if coordinate == "longitude" else "latitude",
        "units": "degrees_east" if coordinate == "longitude" else "degrees_north",
        "interpolation_method": method,
    }


def _create_store(
    path: Path, inventories: list[QCInventory], grid: np.ndarray,
    config: ReconstructionWorkflowConfig, attributes: dict[str, Any],
) -> zarr.Group:
    platforms = [item.platform_id for item in inventories]
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(attributes)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    platform_width = max(1, *(len(value) for value in platforms))
    platform = root.create_dataset(
        "platform_id", data=np.asarray(platforms, dtype=f"<U{platform_width}"),
        chunks=(min(config.chunk_platform, len(platforms)),), compressor=compressor,
    )
    platform.attrs.update({"_ARRAY_DIMENSIONS": ["platform"], "cf_role": "trajectory_id"})
    time = root.create_dataset(
        "time", data=grid.astype("datetime64[ns]"),
        chunks=(min(config.chunk_time, len(grid)),), compressor=compressor,
    )
    time.attrs.update({
        "_ARRAY_DIMENSIONS": ["time"], "standard_name": "time", "timezone": "UTC",
    })
    start_time = root.create_dataset(
        "start_time", data=np.asarray([item.first_time for item in inventories], dtype="datetime64[ns]"),
        chunks=(min(config.chunk_platform, len(platforms)),), compressor=compressor,
        fill_value=None,
    )
    start_time.attrs.update({
        "_ARRAY_DIMENSIONS": ["platform"], "standard_name": "time", "timezone": "UTC",
        "long_name": "actual timestamp of first retained QC position",
    })
    for name, coordinate, values in (
        ("start_lon", "longitude", [item.first_longitude for item in inventories]),
        ("start_lat", "latitude", [item.first_latitude for item in inventories]),
    ):
        array = root.create_dataset(
            name, data=np.asarray(values, dtype=np.float64),
            chunks=(min(config.chunk_platform, len(platforms)),), compressor=compressor,
            fill_value=None,
        )
        array.attrs.update({
            "_ARRAY_DIMENSIONS": ["platform"],
            "standard_name": coordinate,
            "units": "degrees_east" if coordinate == "longitude" else "degrees_north",
            "long_name": f"{coordinate} of first retained QC position",
        })
    chunks = (
        min(config.chunk_platform, len(platforms)), min(config.chunk_time, len(grid)),
    )
    variables = ["longitude_linear", "latitude_linear", "source_gap_minutes"]
    for period in config.periods_minutes:
        variables.extend([f"longitude_spline_{period}", f"latitude_spline_{period}"])
    for name in variables:
        array = root.create_dataset(
            name, shape=(len(platforms), len(grid)), chunks=chunks,
            dtype="f8", fill_value=np.nan, compressor=compressor,
        )
        if name == "source_gap_minutes":
            array.attrs.update({
                "_ARRAY_DIMENSIONS": ["platform", "time"], "coordinates": "platform_id",
                "units": "minutes",
                "long_name": "Separation of accepted QC fixes bracketing each reconstructed point; zero at an exact accepted fix",
            })
        else:
            coordinate, method = name.split("_", 1)
            array.attrs.update(_position_attrs(method, coordinate))
    return root


def _gap_fields(track: AcceptedTrack, config: ReconstructionWorkflowConfig) -> dict[str, Any]:
    gaps = np.diff(track.time) / np.timedelta64(1, "m")
    quantiles = {
        "gap_min_minutes": float(np.min(gaps)),
        "gap_p25_minutes": float(np.quantile(gaps, .25)),
        "gap_p50_minutes": float(np.quantile(gaps, .50)),
        "gap_p75_minutes": float(np.quantile(gaps, .75)),
        "gap_p90_minutes": float(np.quantile(gaps, .90)),
        "gap_p95_minutes": float(np.quantile(gaps, .95)),
        "gap_p99_minutes": float(np.quantile(gaps, .99)),
        "gap_max_minutes": float(np.max(gaps)),
    }
    for period in config.periods_minutes:
        threshold = config.long_gap_threshold_minutes[period]
        quantiles[f"spline_{period}_long_gap_threshold_minutes"] = threshold
        quantiles[f"spline_{period}_gaps_over_threshold"] = int((gaps > threshold).sum())
    return {"gap_count": len(gaps), **quantiles}


def _report_row(
    track: AcceptedTrack, result: PlatformReconstruction, grid: np.ndarray,
    config: ReconstructionWorkflowConfig,
) -> dict[str, Any]:
    first_grid = grid[(grid >= track.time[0]) & (grid <= track.time[-1])][0]
    last_grid = grid[(grid >= track.time[0]) & (grid <= track.time[-1])][-1]
    warnings: list[str] = []
    if first_grid != track.time[0]:
        warnings.append("first accepted fix is off-grid; no backward extrapolation")
    if last_grid != track.time[-1]:
        warnings.append("last accepted fix is off-grid; no forward extrapolation")
    if grid[-1] > track.time[-1]:
        warnings.append("shared grid continues after this platform; trailing cells are NaN")
    row: dict[str, Any] = {
        "platform_id": track.inventory.platform_id,
        "qc_filename": track.inventory.path.name,
        "qc_input_sha256": track.inventory.sha256,
        "accepted_fix_count": track.inventory.accepted_count,
        "eligible_start_utc": _format_utc(track.time[0]),
        "eligible_end_utc": _format_utc(track.time[-1]),
        "first_filled_grid_utc": _format_utc(first_grid),
        "last_filled_grid_utc": _format_utc(last_grid),
        "filled_grid_points": result.filled_grid_points,
        "exact_fix_grid_points": result.exact_grid_points,
        "interpolated_grid_points": result.filled_grid_points - result.exact_grid_points,
        **_gap_fields(track, config),
    }
    for period in config.periods_minutes:
        stats = asdict(result.fallback[period])
        row.update({f"spline_{period}_{name}": value for name, value in stats.items()})
    row["boundary_warnings_json"] = json.dumps(warnings, separators=(",", ":"))
    return row


def _write_platform(root: zarr.Group, index: int, result: PlatformReconstruction) -> None:
    root["longitude_linear"][index, :] = result.longitude_linear
    root["latitude_linear"][index, :] = result.latitude_linear
    root["source_gap_minutes"][index, :] = result.source_gap_minutes
    for period, values in result.longitude_spline.items():
        root[f"longitude_spline_{period}"][index, :] = values
        root[f"latitude_spline_{period}"][index, :] = result.latitude_spline[period]


def _verify(
    path: Path, platforms: list[str], grid: np.ndarray, inventories: list[QCInventory],
    config: ReconstructionWorkflowConfig,
) -> None:
    required = {
        "longitude_linear", "latitude_linear", "source_gap_minutes",
        "start_time", "start_lon", "start_lat",
        *(f"longitude_spline_{period}" for period in config.periods_minutes),
        *(f"latitude_spline_{period}" for period in config.periods_minutes),
    }
    with xr.open_zarr(path, consolidated=True, chunks=None) as dataset:
        if dict(dataset.sizes) != {"platform": len(platforms), "time": len(grid)}:
            raise ValueError("Zarr readback changed the common-grid dimensions")
        if dataset.platform_id.values.astype(str).tolist() != platforms:
            raise ValueError("Zarr readback changed platform identifiers")
        if not np.array_equal(dataset.time.values.astype("datetime64[ns]"), grid):
            raise ValueError("Zarr readback changed the common time coordinate")
        if set(dataset.data_vars) != required:
            raise ValueError("Zarr readback found an unexpected lean-product schema")
    root = zarr.open_group(str(path), mode="r")
    position_names = sorted(
        required - {"source_gap_minutes", "start_time", "start_lon", "start_lat"}
    )
    for index, inventory in enumerate(inventories):
        if np.asarray(root["start_time"][index]).astype("datetime64[ns]") != inventory.first_time:
            raise ValueError(f"Zarr readback changed start_time for {inventory.platform_id}")
        if (float(root["start_lon"][index]) != inventory.first_longitude
                or float(root["start_lat"][index]) != inventory.first_latitude):
            raise ValueError(f"Zarr readback changed exact start coordinates for {inventory.platform_id}")
        inside = (grid >= inventory.first_time) & (grid <= inventory.last_time)
        gap = np.asarray(root["source_gap_minutes"][index, :], dtype=float)
        if not np.isfinite(gap[inside]).all() or not np.isnan(gap[~inside]).all():
            raise ValueError(f"Zarr readback found invalid source-gap coverage for {inventory.platform_id}")
        for name in position_names:
            values = np.asarray(root[name][index, :], dtype=float)
            if not np.isfinite(values[inside]).all() or not np.isnan(values[~inside]).all():
                raise ValueError(f"Zarr readback found invalid lifespan coverage in {name}")
            if name.startswith("longitude") and np.any((values[inside] < -180) | (values[inside] >= 180)):
                raise ValueError(f"Zarr readback found invalid longitude in {name}")
            if name.startswith("latitude") and np.any((values[inside] < -90) | (values[inside] > 90)):
                raise ValueError(f"Zarr readback found invalid latitude in {name}")


def _inventory_inputs(
    config: ReconstructionWorkflowConfig, report: Callable[[str], None],
) -> tuple[list[QCInventory], tuple[Path, ...]]:
    paths = _discover(config)
    inventories: list[QCInventory] = []
    identities: set[str] = set()
    consistency: tuple[Any, ...] | None = None
    for number, path in enumerate(paths, start=1):
        track = _load_track(path)
        item = track.inventory
        if item.platform_id in identities:
            raise ValueError(f"Duplicate position-QC platform: {item.platform_id}")
        identities.add(item.platform_id)
        values = tuple(item.metadata.get(name) for name in (
            "schema_version", "algorithm_version", "config_sha256", "resolution_policy",
        ))
        if consistency is None:
            consistency = values
        elif values != consistency:
            raise ValueError("Position-QC inputs mix schema, algorithm, configuration, or resolution policy")
        inventories.append(item)
        report(f"Validated position QC {number}/{len(paths)}: {path.name}")
    order = np.argsort([item.platform_id for item in inventories], kind="stable")
    inventories = [inventories[int(index)] for index in order]
    paths = tuple(item.path for item in inventories)
    return inventories, paths


def _input_hash_metadata(inventories: list[QCInventory]) -> list[dict[str, str]]:
    return [
        {"platform_id": item.platform_id, "filename": item.path.name, "sha256": item.sha256}
        for item in inventories
    ]


def _existing_problem(
    config: ReconstructionWorkflowConfig, inventories: list[QCInventory],
    grid: np.ndarray, platforms: list[str],
) -> str | None:
    zarr_path = config.output_directory / ZARR_NAME
    report_path = config.output_directory / REPORT_NAME
    if not zarr_path.is_dir() or not report_path.is_file():
        return "the atomic bundle is missing trajectories.zarr or build_report.csv"
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
        attrs = dict(root.attrs)
        if attrs.get("schema_version") != RECONSTRUCTION_SCHEMA_VERSION:
            return (
                f"schema is {attrs.get('schema_version')!r}, expected "
                f"{RECONSTRUCTION_SCHEMA_VERSION!r}"
            )
        if attrs.get("algorithm_version") != RECONSTRUCTION_ALGORITHM_VERSION:
            return "the recorded reconstruction algorithm differs"
        if attrs.get("effective_configuration") != config.effective():
            return "the recorded reconstruction configuration differs"
        if attrs.get("qc_input_hashes") != _input_hash_metadata(inventories):
            return "the recorded QC input provenance differs"
        if attrs.get("build_report_sha256") != _file_sha256(report_path):
            return "the build report hash differs"
        _verify(zarr_path, platforms, grid, inventories, config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"schema/readback validation failed: {exc}"
    return None


def _remove_exact_directory(path: Path, *, parent: Path, expected_name: str) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or resolved.name != expected_name:
        raise RuntimeError(f"Refusing cleanup outside {parent}")
    shutil.rmtree(resolved)


def _publish_bundle(
    temporary: Path, output: Path, *, token: str, overwrite: bool,
) -> None:
    if not output.exists():
        temporary.rename(output)
        return
    if not overwrite:
        raise FileExistsError(f"Reconstruction output already exists: {output}")
    backup = output.with_name(f".{output.name}.{token}.backup")
    if backup.exists():
        raise FileExistsError(f"Reconstruction backup path already exists: {backup}")
    output.rename(backup)
    try:
        temporary.rename(output)
    except Exception:
        if not output.exists() and backup.exists():
            backup.rename(output)
        raise
    _remove_exact_directory(backup, parent=output.parent, expected_name=backup.name)


def _build_bundle(
    config: ReconstructionWorkflowConfig, inventories: list[QCInventory], paths: tuple[Path, ...],
    grid: np.ndarray, grid_indices: np.ndarray, published_start: np.datetime64,
    report: Callable[[str], None], *, overwrite: bool,
) -> None:
    platforms = [item.platform_id for item in inventories]

    config.output_directory.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = config.output_directory.with_name(f".{config.output_directory.name}.{token}.tmp")
    temporary.mkdir()
    zarr_path = temporary / ZARR_NAME
    report_path = temporary / REPORT_NAME
    try:
        attributes = {
            "title": "Drifter trajectories reconstructed from finalized position QC",
            "product_status": "candidate_pending_gap_review",
            "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
            "algorithm_version": RECONSTRUCTION_ALGORITHM_VERSION,
            "position_qc_schema_version": POSITION_QC_SCHEMA_VERSION,
            "drifterlab_version": __version__,
            "processing_time_utc": datetime.now(timezone.utc).isoformat(),
            "time_convention": "UTC; timezone-naive datetime64 values represent UTC",
            "configured_start_time": _format_utc(config.start_time),
            "published_start_time": _format_utc(published_start),
            "end_time": _format_utc(grid[-1]),
            "interpolation_policy": {
                "linear": "Unwrapped-longitude linear interpolation between consecutive accepted QC fixes; all internal gaps filled; no endpoint extrapolation",
                "spline": "Every grid phase for each period; exact portion endpoints added; natural cubic interpolation; equal phase mean",
                "long_gap": "Accepted-fix gaps strictly greater than the configured per-period threshold use the linear track",
                "insufficient_support": "Fewer than four unique knots in any phase makes the complete portion linear",
                "overshoot": "Non-finite, invalid, or accepted-fix-envelope-exceeding phase/mean makes the complete portion linear",
                "longitude": "Fit unwrapped longitude, then wrap published values to [-180, 180)",
                "median_filter": "none",
            },
            "effective_configuration": config.effective(),
            "qc_input_hashes": _input_hash_metadata(inventories),
        }
        root = _create_store(zarr_path, inventories, grid, config, attributes)
        rows: list[dict[str, Any]] = []
        for index, inventory in enumerate(inventories):
            if _file_sha256(inventory.path) != inventory.sha256:
                raise ValueError(f"Position-QC input changed after inventory: {inventory.path}")
            track = _load_track(inventory.path)
            if track.inventory != inventory:
                raise ValueError(f"Position-QC identity changed after inventory: {inventory.path}")
            reconstructed = reconstruct_platform(
                track.time, track.longitude, track.latitude, grid, grid_indices,
                dt_minutes=config.dt_minutes, periods_minutes=config.periods_minutes,
                long_gap_threshold_minutes=config.long_gap_threshold_minutes,
            )
            _write_platform(root, index, reconstructed)
            rows.append(_report_row(track, reconstructed, grid, config))
            report(f"Reconstructed {index + 1}/{len(inventories)}: {inventory.platform_id}")
        pd.DataFrame(rows).to_csv(report_path, index=False)
        root.attrs["build_report_sha256"] = _file_sha256(report_path)

        current_paths = _discover(config)
        if current_paths != tuple(sorted(paths, key=lambda path: path.name)):
            raise ValueError("Position-QC input file set changed during reconstruction")
        changed = [item.platform_id for item in inventories if _file_sha256(item.path) != item.sha256]
        if changed:
            raise ValueError(f"Position-QC inputs changed during reconstruction: {changed}")
        zarr.consolidate_metadata(str(zarr_path))
        _verify(zarr_path, platforms, grid, inventories, config)
        _publish_bundle(
            temporary, config.output_directory, token=token, overwrite=overwrite,
        )
    finally:
        if temporary.exists():
            _remove_exact_directory(
                temporary, parent=config.output_directory.parent,
                expected_name=f".{config.output_directory.name}.{token}.tmp",
            )


def run_reconstruction_workflow(
    config_path: str | Path, *, overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ReconstructionWorkflowResult:
    """Build or reuse a validated trajectory bundle, then optionally plot it."""
    config = load_reconstruction_config(config_path)
    report = progress or (lambda message: None)
    inventories, paths = _inventory_inputs(config, report)
    grid, grid_indices, published_start = _grid(config, inventories)
    platforms = [item.platform_id for item in inventories]
    action = "built"
    if config.output_directory.exists() and not overwrite:
        problem = _existing_problem(config, inventories, grid, platforms)
        if problem is not None:
            raise ValueError(
                f"Existing trajectory product cannot be reused because {problem}; "
                "rerun with --overwrite to rebuild it"
            )
        action = "reused"
        report("Validated and reused existing candidate trajectory bundle")
    else:
        _build_bundle(
            config, inventories, paths, grid, grid_indices, published_start, report,
            overwrite=overwrite,
        )
        report("Verified and published candidate trajectory bundle")

    figure_paths: tuple[Path, ...] = ()
    if config.plotting.enabled:
        try:
            from drifterlab.plotting.trajectories import generate_trajectory_figures
        except ImportError as exc:
            raise ImportError(
                "Trajectory plotting requires the optional plotting dependencies; "
                "install drifterlab[plotting]"
            ) from exc
        figure_paths = generate_trajectory_figures(
            config.output_directory / ZARR_NAME, config.plotting, progress=report,
        )
    result = ReconstructionWorkflowResult(
        config.output_directory, config.output_directory / ZARR_NAME,
        config.output_directory / REPORT_NAME, len(platforms), len(grid),
        config.start_time, published_start, grid[-1], action, figure_paths,
    )
    return result
