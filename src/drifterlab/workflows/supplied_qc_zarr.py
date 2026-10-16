"""Convert supplied ARCTERX QC MATLAB files to grouped analysis products."""

from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from drifterlab import __version__
from drifterlab.experiments.arcterx import read_microsvp
from drifterlab.reconstruction import (
    InitialClusterResult,
    InitialClusterRule,
    assign_initial_clusters,
    assign_start_arrays,
    build_regular_grids,
    build_regular_product,
    forward_velocity,
    resample_linear_track,
    validate_cluster_rule,
)
from drifterlab.trajectories.product_store import (
    SUPPORTED_PRODUCT_STORE_SCHEMA_VERSION,
    load_trajectory_product,
)
from drifterlab.trajectories.supplied_product_store import (
    write_supplied_product_store,
)

CONFIG_SCHEMA_VERSION = 1
BUILD_ALGORITHM_VERSION = "supplied-arcterx-qc-grouped-products-v1"
VALID_ALIGNMENT_MODES = {"automatic_native", "manual_reference"}
VALID_REPRESENTATIONS = {
    "native_qc",
    "supplied_spline30",
    "supplied_spline60",
}
VALID_END_POLICIES = {"record_end", "source_drogue_off"}
VALID_UNKNOWN_LOSS_POLICIES = {"record_end", "error"}
VALID_SPLINE_TIME_ALIGNMENTS = {"strict_snap", "linear_resample"}


@dataclass(frozen=True)
class SuppliedProductDefinition:
    name: str
    representation: str
    cadence_minutes: int
    maximum_gap_minutes: float


@dataclass(frozen=True)
class CoverageConfig:
    end_policy: str
    drogue_buffer_hours: float
    unknown_drogue_loss: str


@dataclass(frozen=True)
class SuppliedQCConfig:
    source_path: Path
    input_directory: Path
    pattern: str
    missing_value: float
    alignment_mode: str
    reference_zarr: Path | None
    reference_products: dict[int, str]
    grid_anchor: np.datetime64
    spline_snap_tolerance_seconds: float
    spline_time_alignment: str
    coverage: CoverageConfig
    products: tuple[SuppliedProductDefinition, ...]
    maximum_adjacent_start_gap_hours: float
    cluster_distance_reference: str
    cluster_default_rule: InitialClusterRule
    cluster_array_overrides: dict[int, InitialClusterRule]
    output_zarr: Path
    chunk_platform: int
    chunk_time: int

    def effective(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "input": {
                "directory": str(self.input_directory),
                "pattern": self.pattern,
                "missing_value": self.missing_value,
            },
            "reference_alignment": {
                "mode": self.alignment_mode,
                "reference_zarr": (
                    None if self.reference_zarr is None else str(self.reference_zarr)
                ),
                "reference_products": {
                    str(key): value
                    for key, value in sorted(self.reference_products.items())
                },
            },
            "grid": {
                "anchor_time": _utc_text(self.grid_anchor),
                "spline_snap_tolerance_seconds": (
                    self.spline_snap_tolerance_seconds
                ),
                "spline_time_alignment": self.spline_time_alignment,
            },
            "coverage": asdict(self.coverage),
            "products": [asdict(item) for item in self.products],
            "arrays": {
                "maximum_adjacent_start_gap_hours": (
                    self.maximum_adjacent_start_gap_hours
                ),
            },
            "initial_clustering": {
                "distance_reference": self.cluster_distance_reference,
                "defaults": _rule_values(self.cluster_default_rule),
                "array_overrides": {
                    f"array_{key:03d}": _rule_values(value)
                    for key, value in sorted(self.cluster_array_overrides.items())
                },
            },
            "output": {
                "zarr": str(self.output_zarr),
                "chunks": {
                    "platform": self.chunk_platform,
                    "time": self.chunk_time,
                },
            },
        }


@dataclass(frozen=True)
class SuppliedQCBuildResult:
    zarr_path: Path
    platform_count: int
    product_count: int
    array_count: int
    cluster_count: int
    alignment_mode: str

    def format(self) -> str:
        return "\n".join((
            (
                f"Supplied-QC Zarr complete: {self.platform_count} platform(s), "
                f"{self.product_count} product(s)"
            ),
            (
                f"Assignments: {self.array_count} array(s), "
                f"{self.cluster_count} cluster(s)"
            ),
            f"Alignment: {self.alignment_mode}",
            f"Zarr: {self.zarr_path}",
        ))


@dataclass(frozen=True)
class _PreparedTrack:
    platform_id: str
    source_path: Path
    source_sha256: str
    warnings: tuple[str, ...]
    native_time: np.ndarray
    native_longitude: np.ndarray
    native_latitude: np.ndarray
    spline_time: np.ndarray
    spline30_longitude: np.ndarray
    spline30_latitude: np.ndarray
    spline60_longitude: np.ndarray
    spline60_latitude: np.ndarray
    automatic_start: np.datetime64
    automatic_end: np.datetime64
    automatic_start_longitude: float
    automatic_start_latitude: float
    source_drogue_off_time: np.datetime64
    sanitization: dict[str, dict[str, int | float]]


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _utc(value: Any, name: str) -> np.datetime64:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty UTC timestamp")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid UTC timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.tz_localize(None).to_datetime64().astype("datetime64[ns]")


def _utc_text(value: np.datetime64) -> str:
    return np.datetime_as_string(
        np.datetime64(value, "ns"), unit="s", timezone="UTC",
    )


def _rule_values(rule: InitialClusterRule) -> dict[str, Any]:
    return {
        "assignment": rule.assignment,
        "maximum_start_time_difference_seconds": (
            rule.maximum_start_time_difference_seconds
        ),
        "maximum_pair_distance_m": rule.maximum_pair_distance_m,
        "maximum_cluster_diameter_m": rule.maximum_cluster_diameter_m,
        "maximum_members": rule.maximum_members,
    }


def _maximum_members(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be null or a positive integer")
    return value


def _cluster_rule(
    values: Mapping[str, Any],
    name: str,
    *,
    base: InitialClusterRule | None = None,
) -> InitialClusterRule:
    allowed = {
        "assignment",
        "maximum_start_time_difference_seconds",
        "maximum_pair_distance_m",
        "maximum_cluster_diameter_m",
        "maximum_members",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    if base is None:
        required = allowed - {"assignment"}
        missing = required - set(values)
        if missing:
            raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    assignment = values.get("assignment", "infer" if base is None else base.assignment)
    rule = InitialClusterRule(
        str(assignment),
        _positive(
            values.get(
                "maximum_start_time_difference_seconds",
                None if base is None else base.maximum_start_time_difference_seconds,
            ),
            f"{name}.maximum_start_time_difference_seconds",
        ),
        _positive(
            values.get(
                "maximum_pair_distance_m",
                None if base is None else base.maximum_pair_distance_m,
            ),
            f"{name}.maximum_pair_distance_m",
        ),
        _positive(
            values.get(
                "maximum_cluster_diameter_m",
                None if base is None else base.maximum_cluster_diameter_m,
            ),
            f"{name}.maximum_cluster_diameter_m",
        ),
        _maximum_members(
            values.get(
                "maximum_members",
                None if base is None else base.maximum_members,
            ),
            f"{name}.maximum_members",
        ),
    )
    validate_cluster_rule(rule, name=name)
    return rule


def load_supplied_qc_config(path: str | Path) -> SuppliedQCConfig:
    """Load the strict supplied-QC import configuration."""

    source_path = Path(path).expanduser().resolve()
    with source_path.open(encoding="utf-8-sig") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(
        raw,
        {
            "schema_version", "input", "reference_alignment", "grid",
            "coverage", "products", "arrays", "initial_clustering", "output",
        },
        "configuration",
    )
    if root.get("schema_version", CONFIG_SCHEMA_VERSION) != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Supplied-QC configuration schema must be {CONFIG_SCHEMA_VERSION}"
        )
    input_section = _mapping(
        root.get("input", {}), {"directory", "pattern", "missing_value"}, "input",
    )
    alignment = _mapping(
        root.get("reference_alignment", {}),
        {"mode", "reference_zarr", "reference_products"},
        "reference_alignment",
    )
    grid = _mapping(
        root.get("grid", {}),
        {
            "anchor_time", "spline_snap_tolerance_seconds",
            "spline_time_alignment",
        },
        "grid",
    )
    coverage_values = _mapping(
        root.get("coverage", {}),
        {"end_policy", "drogue_buffer_hours", "unknown_drogue_loss"},
        "coverage",
    )
    arrays = _mapping(
        root.get("arrays", {}), {"maximum_adjacent_start_gap_hours"}, "arrays",
    )
    clustering = _mapping(
        root.get("initial_clustering", {}),
        {"distance_reference", "defaults", "array_overrides"},
        "initial_clustering",
    )
    output = _mapping(root.get("output", {}), {"zarr", "chunks"}, "output")
    chunks = _mapping(
        output.get("chunks", {}), {"platform", "time"}, "output.chunks",
    )

    def resolved(value: Any, name: str) -> Path:
        candidate = Path(_nonempty(value, name)).expanduser()
        return (source_path.parent / candidate).resolve()

    input_directory = resolved(input_section.get("directory"), "input.directory")
    output_zarr = resolved(output.get("zarr"), "output.zarr")
    if output_zarr == input_directory or input_directory in output_zarr.parents:
        raise ValueError("output.zarr must be outside the MAT input directory")
    pattern = input_section.get("pattern", "*.mat")
    if (
        not isinstance(pattern, str)
        or not pattern.strip()
        or Path(pattern).is_absolute()
        or ".." in Path(pattern).parts
    ):
        raise ValueError("input.pattern must be a relative glob without '..'")

    mode = alignment.get("mode", "automatic_native")
    if mode not in VALID_ALIGNMENT_MODES:
        raise ValueError(
            f"reference_alignment.mode must be one of {sorted(VALID_ALIGNMENT_MODES)}"
        )
    raw_reference = alignment.get("reference_zarr")
    reference_zarr = (
        None if raw_reference is None else resolved(
            raw_reference, "reference_alignment.reference_zarr",
        )
    )
    if mode == "manual_reference" and reference_zarr is None:
        raise ValueError(
            "manual_reference mode requires reference_alignment.reference_zarr"
        )
    raw_reference_products = alignment.get("reference_products", {})
    if not isinstance(raw_reference_products, dict):
        raise TypeError("reference_alignment.reference_products must be a mapping")
    reference_products: dict[int, str] = {}
    for raw_cadence, product_name in raw_reference_products.items():
        try:
            cadence = int(raw_cadence)
        except (TypeError, ValueError) as exc:
            raise ValueError("Reference-product keys must be positive cadences") from exc
        if cadence < 1 or str(cadence) != str(raw_cadence):
            raise ValueError("Reference-product keys must be positive integer strings")
        reference_products[cadence] = _nonempty(
            product_name, f"reference_alignment.reference_products.{cadence}",
        )

    end_policy = coverage_values.get("end_policy", "record_end")
    if end_policy not in VALID_END_POLICIES:
        raise ValueError(f"coverage.end_policy must be one of {sorted(VALID_END_POLICIES)}")
    unknown_loss = coverage_values.get("unknown_drogue_loss", "record_end")
    if unknown_loss not in VALID_UNKNOWN_LOSS_POLICIES:
        raise ValueError(
            "coverage.unknown_drogue_loss must be 'record_end' or 'error'"
        )
    raw_buffer = coverage_values.get("drogue_buffer_hours", 0)
    if isinstance(raw_buffer, bool):
        raise TypeError("coverage.drogue_buffer_hours must be finite and nonnegative")
    buffer_hours = float(raw_buffer)
    if not math.isfinite(buffer_hours) or buffer_hours < 0:
        raise ValueError("coverage.drogue_buffer_hours must be finite and nonnegative")
    coverage = CoverageConfig(end_policy, buffer_hours, unknown_loss)

    raw_products = root.get("products")
    if not isinstance(raw_products, list) or not raw_products:
        raise ValueError("products must be a nonempty list")
    products: list[SuppliedProductDefinition] = []
    for index, raw_product in enumerate(raw_products):
        values = _mapping(
            raw_product,
            {"name", "representation", "cadence_minutes", "maximum_gap_minutes"},
            f"products[{index}]",
        )
        name = _nonempty(values.get("name"), f"products[{index}].name")
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"products[{index}].name must be one direct group name")
        representation = values.get("representation")
        if representation not in VALID_REPRESENTATIONS:
            raise ValueError(
                f"products[{index}].representation must be one of "
                f"{sorted(VALID_REPRESENTATIONS)}"
            )
        products.append(SuppliedProductDefinition(
            name,
            str(representation),
            _positive_integer(
                values.get("cadence_minutes"),
                f"products[{index}].cadence_minutes",
            ),
            _positive(
                values.get("maximum_gap_minutes"),
                f"products[{index}].maximum_gap_minutes",
            ),
        ))
    if len({item.name for item in products}) != len(products):
        raise ValueError("Product names must be unique")
    if not any(item.representation == "native_qc" for item in products):
        raise ValueError("At least one native_qc product is required for clustering")

    defaults = clustering.get("defaults", {})
    if not isinstance(defaults, dict):
        raise TypeError("initial_clustering.defaults must be a mapping")
    default_rule = _cluster_rule(defaults, "initial_clustering.defaults")
    raw_overrides = clustering.get("array_overrides", {})
    if not isinstance(raw_overrides, dict):
        raise TypeError("initial_clustering.array_overrides must be a mapping")
    overrides: dict[int, InitialClusterRule] = {}
    key_pattern = re.compile(r"^array_([0-9]{3})$")
    for raw_key, raw_value in raw_overrides.items():
        matched = key_pattern.fullmatch(str(raw_key))
        if matched is None or int(matched.group(1)) < 1:
            raise ValueError(
                "initial_clustering.array_overrides keys must use array_NNN"
            )
        if not isinstance(raw_value, dict):
            raise TypeError(
                f"initial_clustering.array_overrides.{raw_key} must be a mapping"
            )
        overrides[int(matched.group(1))] = _cluster_rule(
            raw_value,
            f"initial_clustering.array_overrides.{raw_key}",
            base=default_rule,
        )
    distance_reference = clustering.get("distance_reference", "observed_starts")
    if distance_reference not in {"observed_starts", "first_common_grid"}:
        raise ValueError(
            "initial_clustering.distance_reference must be 'observed_starts' "
            "or 'first_common_grid'"
        )

    spline_time_alignment = grid.get("spline_time_alignment", "strict_snap")
    if spline_time_alignment not in VALID_SPLINE_TIME_ALIGNMENTS:
        raise ValueError(
            "grid.spline_time_alignment must be 'strict_snap' or "
            "'linear_resample'"
        )

    return SuppliedQCConfig(
        source_path=source_path,
        input_directory=input_directory,
        pattern=pattern,
        missing_value=float(input_section.get("missing_value", -999)),
        alignment_mode=mode,
        reference_zarr=reference_zarr,
        reference_products=reference_products,
        grid_anchor=_utc(grid.get("anchor_time"), "grid.anchor_time"),
        spline_snap_tolerance_seconds=_positive(
            grid.get("spline_snap_tolerance_seconds", 1),
            "grid.spline_snap_tolerance_seconds",
        ),
        spline_time_alignment=str(spline_time_alignment),
        coverage=coverage,
        products=tuple(products),
        maximum_adjacent_start_gap_hours=_positive(
            arrays.get("maximum_adjacent_start_gap_hours", 24),
            "arrays.maximum_adjacent_start_gap_hours",
        ),
        cluster_distance_reference=str(distance_reference),
        cluster_default_rule=default_rule,
        cluster_array_overrides=overrides,
        output_zarr=output_zarr,
        chunk_platform=_positive_integer(
            chunks.get("platform", 1), "output.chunks.platform",
        ),
        chunk_time=_positive_integer(
            chunks.get("time", 2016), "output.chunks.time",
        ),
    )


def _snap_times(
    times: np.ndarray,
    *,
    anchor: np.datetime64,
    cadence_minutes: int,
    tolerance_seconds: float,
    alignment: str,
) -> tuple[np.ndarray, float, bool]:
    values = np.asarray(times, dtype="datetime64[ns]")
    if np.isnat(values).any():
        raise ValueError("Supplied spline time contains NaT")
    step = cadence_minutes * 60_000_000_000
    delta = values.astype(np.int64) - np.datetime64(anchor, "ns").astype(np.int64)
    quotient, remainder = np.divmod(delta, step)
    indices = quotient + (remainder * 2 >= step)
    snapped_ns = np.datetime64(anchor, "ns").astype(np.int64) + indices * step
    residual = np.abs(values.astype(np.int64) - snapped_ns) / 1_000_000_000
    maximum = float(residual.max(initial=0.0))
    differences = np.diff(values.astype(np.int64)) / 1_000_000_000
    expected = cadence_minutes * 60.0
    cadence_residual = float(
        np.max(np.abs(differences - expected), initial=0.0)
    )
    if cadence_residual > tolerance_seconds:
        raise ValueError(
            f"Supplied spline cadence differs from {cadence_minutes} minutes by "
            f"{cadence_residual:g} seconds; tolerance is {tolerance_seconds:g}"
        )
    if maximum > tolerance_seconds and alignment == "strict_snap":
        raise ValueError(
            f"Supplied spline timestamp differs from the {cadence_minutes}-minute "
            f"grid by {maximum:g} seconds; tolerance is {tolerance_seconds:g}"
        )
    if maximum > tolerance_seconds:
        # Preserve a real platform phase (for example +60 seconds), while
        # removing only MATLAB floating-point jitter around that phase.  The
        # supplied positions are later evaluated on the common analysis grid;
        # they are never relabelled as exact common-grid observations.
        signed_offset = values.astype(np.int64) - snapped_ns
        phase_offset = round(float(np.median(signed_offset)))
        phase_residual = np.abs(signed_offset - phase_offset) / 1_000_000_000
        if float(phase_residual.max(initial=0.0)) > tolerance_seconds:
            raise ValueError(
                "Supplied spline timestamps do not share one stable five-minute "
                "phase within the configured tolerance"
            )
        phase_aligned = (snapped_ns + phase_offset).astype("datetime64[ns]")
        return phase_aligned, maximum, False
    snapped = snapped_ns.astype("datetime64[ns]")
    if len(snapped) > 1 and np.any(np.diff(snapped.astype(np.int64)) <= 0):
        raise ValueError("Supplied spline timestamps collide after grid snapping")
    return snapped, maximum, True


def _sanitize_coordinates(
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    lon = np.asarray(longitude, dtype=float).copy()
    lat = np.asarray(latitude, dtype=float).copy()
    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError("Supplied coordinate pairs must be aligned vectors")
    lon_finite = np.isfinite(lon)
    lat_finite = np.isfinite(lat)
    mismatch = lon_finite != lat_finite
    pair = lon_finite & lat_finite
    out_of_range = pair & (
        (lon < -180) | (lon >= 180) | (lat < -90) | (lat > 90)
    )
    invalid = mismatch | out_of_range
    lon[invalid] = np.nan
    lat[invalid] = np.nan
    return lon, lat, {
        "source_rows": len(lon),
        "finite_pair_rows": int(np.count_nonzero(np.isfinite(lon))),
        "finite_mask_mismatch_rows": int(np.count_nonzero(mismatch)),
        "out_of_range_pair_rows": int(np.count_nonzero(out_of_range)),
    }


def _prepare_track(path: Path, config: SuppliedQCConfig) -> _PreparedTrack:
    record = read_microsvp(path, missing_value=config.missing_value)
    native = record.series["qc"]
    interp = record.series["interp"]
    native_lon, native_lat, native_counts = _sanitize_coordinates(
        native.variables["lon_qc"], native.variables["lat_qc"],
    )
    spline30_lon, spline30_lat, spline30_counts = _sanitize_coordinates(
        interp.variables["lon_interp_30m"], interp.variables["lat_interp_30m"],
    )
    spline60_lon, spline60_lat, spline60_counts = _sanitize_coordinates(
        interp.variables["lon_interp_60m"], interp.variables["lat_interp_60m"],
    )
    spline_time, residual, snapped = _snap_times(
        interp.time,
        anchor=config.grid_anchor,
        cadence_minutes=5,
        tolerance_seconds=config.spline_snap_tolerance_seconds,
        alignment=config.spline_time_alignment,
    )
    finite_native = np.isfinite(native_lon) & np.isfinite(native_lat)
    if np.count_nonzero(finite_native) < 2:
        raise ValueError(
            f"{path.name} has fewer than two valid automatic native-QC positions"
        )
    good_time = native.time[finite_native].astype("datetime64[ns]")
    good_lon = native_lon[finite_native]
    good_lat = native_lat[finite_native]
    if np.isnat(good_time).any() or np.any(np.diff(good_time.astype(np.int64)) <= 0):
        raise ValueError(f"{path.name} native-QC times are not strictly increasing")
    loss = np.datetime64(record.metadata["drogue_off_time"], "ns")
    return _PreparedTrack(
        platform_id=record.platform_code,
        source_path=path.resolve(),
        source_sha256=str(record.metadata["source_sha256"]),
        warnings=tuple(record.warnings),
        native_time=native.time.astype("datetime64[ns]"),
        native_longitude=native_lon,
        native_latitude=native_lat,
        spline_time=spline_time,
        spline30_longitude=spline30_lon,
        spline30_latitude=spline30_lat,
        spline60_longitude=spline60_lon,
        spline60_latitude=spline60_lat,
        automatic_start=good_time[0],
        automatic_end=good_time[-1],
        automatic_start_longitude=float(good_lon[0]),
        automatic_start_latitude=float(good_lat[0]),
        source_drogue_off_time=loss,
        sanitization={
            "native_qc": native_counts,
            "supplied_spline30": {
                **spline30_counts,
                "maximum_time_snap_residual_seconds": residual,
                "time_grid_snapped": int(snapped),
            },
            "supplied_spline60": {
                **spline60_counts,
                "maximum_time_snap_residual_seconds": residual,
                "time_grid_snapped": int(snapped),
            },
        },
    )


def _automatic_coverage_end(
    track: _PreparedTrack,
    coverage: CoverageConfig,
) -> np.datetime64:
    if coverage.end_policy == "record_end":
        return track.automatic_end
    if np.isnat(track.source_drogue_off_time):
        if coverage.unknown_drogue_loss == "error":
            raise ValueError(
                f"{track.platform_id} has no supplied drogue-loss time"
            )
        return track.automatic_end
    buffer_ns = round(coverage.drogue_buffer_hours * 3_600_000_000_000)
    cutoff = track.source_drogue_off_time - np.timedelta64(buffer_ns, "ns")
    end = min(track.automatic_end, cutoff)
    if end <= track.automatic_start:
        raise ValueError(
            f"{track.platform_id} drogue cutoff is not after its automatic start"
        )
    return end.astype("datetime64[ns]")


def _load_reference_root(
    config: SuppliedQCConfig,
    platforms: Sequence[str],
) -> xr.Dataset | None:
    if config.alignment_mode != "manual_reference":
        return None
    assert config.reference_zarr is not None
    if not config.reference_zarr.is_dir():
        raise ValueError(f"Reference Zarr does not exist: {config.reference_zarr}")
    with xr.open_zarr(
        config.reference_zarr, consolidated=True, chunks=None,
    ) as source:
        reference = source.load()
    required = {
        "platform_id", "array_id", "cluster_id", "cluster_size",
        "cluster_assignment_status", "deployment_start_time",
    }
    missing = required - set(reference.variables)
    if missing:
        raise ValueError(f"Reference Zarr lacks root variables: {sorted(missing)}")
    reference_platforms = reference.platform_id.values.astype(str).tolist()
    if set(reference_platforms) != set(platforms):
        raise ValueError(
            "Reference and automatic inputs do not contain the same platform IDs"
        )
    index = {platform: i for i, platform in enumerate(reference_platforms)}
    order = [index[platform] for platform in platforms]
    return reference.isel(platform=order)


def _reference_grid(
    config: SuppliedQCConfig,
    cadence: int,
    platforms: Sequence[str],
) -> np.ndarray:
    assert config.reference_zarr is not None
    product = config.reference_products.get(cadence, f"qc_{cadence}min")
    with xr.open_zarr(
        config.reference_zarr,
        group=f"products/{product}",
        consolidated=True,
        chunks=None,
    ) as dataset:
        if "time" not in dataset or "platform_id" not in dataset:
            raise ValueError(f"Reference product {product!r} lacks time or platform_id")
        reference_platforms = dataset.platform_id.values.astype(str).tolist()
        if set(reference_platforms) != set(platforms):
            raise ValueError(
                f"Reference product {product!r} has different platform IDs"
            )
        grid = dataset.time.values.astype("datetime64[ns]").copy()
    if len(grid) < 2 or np.isnat(grid).any():
        raise ValueError(f"Reference product {product!r} has an invalid grid")
    differences = np.diff(grid.astype(np.int64))
    expected = cadence * 60_000_000_000
    if np.any(differences != expected):
        raise ValueError(
            f"Reference product {product!r} is not on a {cadence}-minute grid"
        )
    return grid


def _track_source(
    track: _PreparedTrack,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if representation == "native_qc":
        return track.native_time, track.native_longitude, track.native_latitude
    if representation == "supplied_spline30":
        return track.spline_time, track.spline30_longitude, track.spline30_latitude
    if representation == "supplied_spline60":
        return track.spline_time, track.spline60_longitude, track.spline60_latitude
    raise ValueError(f"Unknown supplied representation: {representation}")


def _support_gap(
    source_time: np.ndarray,
    grid: np.ndarray,
    *,
    coverage_start: np.datetime64,
    coverage_end: np.datetime64,
) -> np.ndarray:
    time = np.asarray(source_time, dtype="datetime64[ns]")
    target = np.asarray(grid, dtype="datetime64[ns]")
    result = np.full(len(target), np.nan)
    if len(time) < 2:
        return result
    time_ns = time.astype(np.int64)
    grid_ns = target.astype(np.int64)
    inside = (
        (target >= coverage_start) & (target <= coverage_end)
        & (target >= time[0]) & (target <= time[-1])
    )
    positions = np.flatnonzero(inside)
    if not len(positions):
        return result
    values = grid_ns[positions]
    right = np.searchsorted(time_ns, values, side="left")
    clipped = np.minimum(right, len(time_ns) - 1)
    exact = (right < len(time_ns)) & (time_ns[clipped] == values)
    result[positions[exact]] = 0.0
    between = ~exact
    if between.any():
        result[positions[between]] = (
            time_ns[right[between]] - time_ns[right[between] - 1]
        ) / 60_000_000_000
    return result


def _aligned_supplied_product(
    source_time: np.ndarray,
    source_longitude: np.ndarray,
    source_latitude: np.ndarray,
    native_time: np.ndarray,
    native_longitude: np.ndarray,
    native_latitude: np.ndarray,
    grid: np.ndarray,
    *,
    coverage_start: np.datetime64,
    coverage_end: np.datetime64,
    maximum_gap_minutes: float,
) -> dict[str, np.ndarray]:
    lon = np.full(len(grid), np.nan)
    lat = np.full(len(grid), np.nan)
    gaps = np.full(len(grid), np.nan)
    finite = np.isfinite(source_longitude) & np.isfinite(source_latitude)
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices):
        split = np.flatnonzero(np.diff(finite_indices) > 1) + 1
        for portion in np.split(finite_indices, split):
            if not len(portion):
                continue
            if len(portion) == 1:
                match = np.flatnonzero(grid == source_time[portion[0]])
                if (
                    len(match)
                    and coverage_start <= source_time[portion[0]] <= coverage_end
                ):
                    lon[match[0]] = source_longitude[portion[0]]
                    lat[match[0]] = source_latitude[portion[0]]
                    gaps[match[0]] = 0.0
                continue
            start = max(coverage_start, source_time[portion[0]])
            end = min(coverage_end, source_time[portion[-1]])
            if end < start:
                continue
            aligned = resample_linear_track(
                source_time[portion],
                source_longitude[portion],
                source_latitude[portion],
                grid,
                coverage_start=start,
                coverage_end=end,
                maximum_gap_minutes=maximum_gap_minutes,
            )
            available = np.isfinite(aligned.longitude) & np.isfinite(aligned.latitude)
            lon[available] = aligned.longitude[available]
            lat[available] = aligned.latitude[available]
            gaps[available] = aligned.source_gap_minutes[available]
    velocity = forward_velocity(grid, lon, lat)
    native_finite = np.isfinite(native_longitude) & np.isfinite(native_latitude)
    native_gap = _support_gap(
        native_time[native_finite],
        grid,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
    )
    return {
        "longitude": lon,
        "latitude": lat,
        "source_gap_minutes": gaps,
        "native_qc_support_gap_minutes": native_gap,
        "eastward_velocity": velocity.east_m_s,
        "northward_velocity": velocity.north_m_s,
        "speed": velocity.speed_m_s,
        "velocity_dt_seconds": velocity.dt_seconds,
    }


def _empty_product(grid: np.ndarray, *, include_native_gap: bool) -> dict[str, np.ndarray]:
    result = {
        name: np.full(len(grid), np.nan)
        for name in (
            "longitude", "latitude", "source_gap_minutes", "eastward_velocity",
            "northward_velocity", "speed", "velocity_dt_seconds",
        )
    }
    if include_native_gap:
        result["native_qc_support_gap_minutes"] = np.full(len(grid), np.nan)
    return result


def _build_product_dataset(
    specification: SuppliedProductDefinition,
    tracks: Sequence[_PreparedTrack],
    platforms: Sequence[str],
    grid: np.ndarray,
    coverage_starts: np.ndarray,
    coverage_ends: np.ndarray,
) -> xr.Dataset:
    include_native_gap = specification.representation != "native_qc"
    names = [
        "longitude", "latitude", "source_gap_minutes", "eastward_velocity",
        "northward_velocity", "speed", "velocity_dt_seconds",
    ]
    if include_native_gap:
        names.append("native_qc_support_gap_minutes")
    values = {
        name: np.full((len(tracks), len(grid)), np.nan)
        for name in names
    }
    for index, track in enumerate(tracks):
        source_time, source_lon, source_lat = _track_source(
            track, specification.representation,
        )
        finite = np.isfinite(source_lon) & np.isfinite(source_lat)
        if np.count_nonzero(finite) < 2:
            product = _empty_product(grid, include_native_gap=include_native_gap)
        else:
            usable_time = source_time[finite]
            start = max(coverage_starts[index], usable_time[0])
            end = min(coverage_ends[index], usable_time[-1])
            if end < start:
                product = _empty_product(grid, include_native_gap=include_native_gap)
            elif specification.representation == "native_qc":
                regular = build_regular_product(
                    usable_time,
                    source_lon[finite],
                    source_lat[finite],
                    grid,
                    coverage_start=start,
                    coverage_end=end,
                    maximum_gap_minutes=specification.maximum_gap_minutes,
                )
                product = {
                    "longitude": regular.position.longitude,
                    "latitude": regular.position.latitude,
                    "source_gap_minutes": regular.position.source_gap_minutes,
                    "eastward_velocity": regular.velocity.east_m_s,
                    "northward_velocity": regular.velocity.north_m_s,
                    "speed": regular.velocity.speed_m_s,
                    "velocity_dt_seconds": regular.velocity.dt_seconds,
                }
            else:
                product = _aligned_supplied_product(
                    source_time,
                    source_lon,
                    source_lat,
                    track.native_time,
                    track.native_longitude,
                    track.native_latitude,
                    grid,
                    coverage_start=coverage_starts[index],
                    coverage_end=coverage_ends[index],
                    maximum_gap_minutes=specification.maximum_gap_minutes,
                )
        for name in names:
            values[name][index, :] = product[name]

    variables: dict[str, Any] = {
        "platform_id": ("platform", np.asarray(platforms)),
        "time": ("time", grid),
    }
    variables.update({
        name: (("platform", "time"), array)
        for name, array in values.items()
    })
    interpolation = (
        "linear_gap_limited" if specification.representation == "native_qc"
        else "linear_time_alignment_without_spline_refit"
    )
    dataset = xr.Dataset(variables, attrs={
        "product_name": specification.name,
        "source_kind": "qc",
        "source_representation": specification.representation,
        "cadence_minutes": specification.cadence_minutes,
        "maximum_source_gap_minutes": specification.maximum_gap_minutes,
        "interpolation": interpolation,
        "endpoint_extrapolation": "none",
        "velocity_method": "forward_difference",
        "velocity_assignment": "earlier_endpoint",
        "source_gap_definition": (
            "bracketing finite automatic native-QC fixes"
            if specification.representation == "native_qc"
            else "bracketing finite supplied spline samples"
        ),
    })
    units = {
        "longitude": "degrees_east",
        "latitude": "degrees_north",
        "source_gap_minutes": "minutes",
        "native_qc_support_gap_minutes": "minutes",
        "eastward_velocity": "m s-1",
        "northward_velocity": "m s-1",
        "speed": "m s-1",
        "velocity_dt_seconds": "seconds",
    }
    for name, unit in units.items():
        if name in dataset:
            dataset[name].attrs["units"] = unit
    dataset.platform_id.attrs["cf_role"] = "trajectory_id"
    dataset.time.attrs.update({"standard_name": "time", "timezone": "UTC"})
    return dataset


def _reference_values(
    reference: xr.Dataset,
    name: str,
    *,
    default: np.ndarray | None = None,
) -> np.ndarray:
    if name in reference:
        return reference[name].values.copy()
    if default is None:
        raise ValueError(f"Reference Zarr lacks required variable {name!r}")
    return np.asarray(default).copy()


def _derive_assignments(
    config: SuppliedQCConfig,
    tracks: Sequence[_PreparedTrack],
    platforms: Sequence[str],
    starts: np.ndarray,
    coordinate_product: xr.Dataset,
) -> tuple[dict[str, np.ndarray], InitialClusterResult]:
    arrays = assign_start_arrays(
        platforms,
        starts,
        maximum_adjacent_start_gap_hours=config.maximum_adjacent_start_gap_hours,
    )
    active = {item.array_id for item in arrays}
    overrides = {
        key: value
        for key, value in config.cluster_array_overrides.items()
        if key in active
    }
    clusters = assign_initial_clusters(
        platforms,
        [item.array_id for item in arrays],
        starts,
        coordinate_product.time.values.astype("datetime64[ns]"),
        coordinate_product.longitude.values.astype(float),
        coordinate_product.latitude.values.astype(float),
        config.cluster_default_rule,
        overrides,
        distance_reference=config.cluster_distance_reference,
        start_longitude=[item.automatic_start_longitude for item in tracks],
        start_latitude=[item.automatic_start_latitude for item in tracks],
    )
    values: dict[str, np.ndarray] = {
        "array_id": np.asarray([item.array_id for item in arrays], dtype=np.int32),
        "gap_from_previous_start_hours": np.asarray([
            np.nan if item.gap_from_previous_start_hours is None
            else item.gap_from_previous_start_hours
            for item in arrays
        ]),
        "gap_to_next_start_hours": np.asarray([
            np.nan if item.gap_to_next_start_hours is None
            else item.gap_to_next_start_hours
            for item in arrays
        ]),
    }
    for name, dtype in (
        ("cluster_id", str),
        ("member_id", np.int32),
        ("cluster_size", np.int32),
        ("cluster_assignment_status", str),
    ):
        values[name] = np.asarray(
            [getattr(item, name) for item in clusters.assignments], dtype=dtype,
        )
    return values, clusters


def _reference_assignments(
    reference: xr.Dataset,
    platforms: Sequence[str],
    starts: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = reference.array_id.values.astype(np.int32)
    cluster_ids = reference.cluster_id.values.astype(str)
    cluster_sizes = reference.cluster_size.values.astype(np.int32)
    status = reference.cluster_assignment_status.values.astype(str)
    if "member_id" in reference:
        member_ids = reference.member_id.values.astype(np.int32)
    else:
        member_ids = np.zeros(len(platforms), dtype=np.int32)
        for cluster in dict.fromkeys(cluster_ids.tolist()):
            members = np.flatnonzero(cluster_ids == cluster)
            ordered = sorted(members, key=lambda index: (starts[index], platforms[index]))
            for number, index in enumerate(ordered, start=1):
                member_ids[index] = number
    chronological = assign_start_arrays(
        platforms,
        starts,
        # The threshold does not affect the adjacent-gap diagnostics.  Use a
        # very large finite value because the assignment helper deliberately
        # rejects infinity.
        maximum_adjacent_start_gap_hours=1.0e12,
    )
    previous = _reference_values(
        reference,
        "gap_from_previous_start_hours",
        default=np.asarray([
            np.nan if item.gap_from_previous_start_hours is None
            else item.gap_from_previous_start_hours
            for item in chronological
        ]),
    )
    following = _reference_values(
        reference,
        "gap_to_next_start_hours",
        default=np.asarray([
            np.nan if item.gap_to_next_start_hours is None
            else item.gap_to_next_start_hours
            for item in chronological
        ]),
    )
    return {
        "array_id": arrays,
        "cluster_id": cluster_ids,
        "member_id": member_ids,
        "cluster_size": cluster_sizes,
        "cluster_assignment_status": status,
        "gap_from_previous_start_hours": previous.astype(float),
        "gap_to_next_start_hours": following.astype(float),
    }


def _root_dataset(
    config: SuppliedQCConfig,
    tracks: Sequence[_PreparedTrack],
    platforms: Sequence[str],
    starts: np.ndarray,
    ends: np.ndarray,
    assignments: Mapping[str, np.ndarray],
    *,
    manifest: Mapping[str, Any],
) -> xr.Dataset:
    automatic_starts = np.asarray(
        [item.automatic_start for item in tracks], dtype="datetime64[ns]",
    )
    automatic_ends = np.asarray(
        [item.automatic_end for item in tracks], dtype="datetime64[ns]",
    )
    root = xr.Dataset({
        "platform_id": ("platform", np.asarray(platforms)),
        "array_id": ("platform", assignments["array_id"]),
        "cluster_id": ("platform", assignments["cluster_id"]),
        "member_id": ("platform", assignments["member_id"]),
        "cluster_size": ("platform", assignments["cluster_size"]),
        "cluster_assignment_status": (
            "platform", assignments["cluster_assignment_status"],
        ),
        "gap_from_previous_start_hours": (
            "platform", assignments["gap_from_previous_start_hours"],
        ),
        "gap_to_next_start_hours": (
            "platform", assignments["gap_to_next_start_hours"],
        ),
        "deployment_start_time": ("platform", starts),
        "automatic_native_start_time": ("platform", automatic_starts),
        "automatic_native_end_time": ("platform", automatic_ends),
        "deployment_start_longitude": (
            "platform",
            np.asarray([item.automatic_start_longitude for item in tracks]),
        ),
        "deployment_start_latitude": (
            "platform",
            np.asarray([item.automatic_start_latitude for item in tracks]),
        ),
        "trajectory_end_time": ("platform", ends),
        "review_end_time": ("platform", ends),
        "source_filename": (
            "platform", np.asarray([item.source_path.name for item in tracks]),
        ),
        "source_sha256": (
            "platform", np.asarray([item.source_sha256 for item in tracks]),
        ),
    }, attrs={
        "schema_version": SUPPORTED_PRODUCT_STORE_SCHEMA_VERSION,
        "description": "Campaign-supplied automatic-QC trajectory products",
        "algorithm_version": BUILD_ALGORITHM_VERSION,
        "drifterlab_version": __version__,
        "processing_time_utc": datetime.now(timezone.utc).isoformat(),
        "product_status": "final_supplied_automatic_qc",
        "assignment_scope_status": "final",
        "alignment_mode": config.alignment_mode,
        "configured_grid_anchor": _utc_text(config.grid_anchor),
        "included_platform_count": len(platforms),
        "configured_platform_count": len(platforms),
        "included_platform_ids": list(platforms),
        "input_qc_sha256": {
            item.platform_id: item.source_sha256 for item in tracks
        },
        "product_names": [item.name for item in config.products],
        "effective_configuration": config.effective(),
        "build_manifest": dict(manifest),
        "build_manifest_sha256": _canonical_sha256(manifest),
        "velocity_method": "forward_difference",
        "velocity_assignment": "earlier_endpoint",
    })
    root.platform_id.attrs["cf_role"] = "trajectory_id"
    for name in (
        "deployment_start_time", "automatic_native_start_time",
        "automatic_native_end_time", "trajectory_end_time", "review_end_time",
    ):
        root[name].attrs.update({"standard_name": "time", "timezone": "UTC"})
    return root


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str,
        ).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_table(
    tracks: Sequence[_PreparedTrack],
    starts: np.ndarray,
    ends: np.ndarray,
    assignments: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, track in enumerate(tracks):
        row: dict[str, Any] = {
            "platform_id": track.platform_id,
            "source_filename": track.source_path.name,
            "source_sha256": track.source_sha256,
            "alignment_start_utc": _utc_text(starts[index]),
            "automatic_native_start_utc": _utc_text(track.automatic_start),
            "automatic_native_end_utc": _utc_text(track.automatic_end),
            "coverage_end_utc": _utc_text(ends[index]),
            "source_drogue_off_utc": (
                "" if np.isnat(track.source_drogue_off_time)
                else _utc_text(track.source_drogue_off_time)
            ),
            "array_id": int(assignments["array_id"][index]),
            "cluster_id": str(assignments["cluster_id"][index]),
            "member_id": int(assignments["member_id"][index]),
            "cluster_size": int(assignments["cluster_size"][index]),
            "warnings": "|".join(track.warnings),
        }
        for representation, counts in track.sanitization.items():
            for name, value in counts.items():
                row[f"{representation}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _manifest(
    config: SuppliedQCConfig,
    tracks: Sequence[_PreparedTrack],
    starts: np.ndarray,
    ends: np.ndarray,
    assignments: Mapping[str, np.ndarray],
    audit_sha256: str,
) -> dict[str, Any]:
    records = [
        {
            "platform_id": track.platform_id,
            "array_id": int(assignments["array_id"][index]),
            "cluster_id": str(assignments["cluster_id"][index]),
            "start": _utc_text(starts[index]),
            "end": _utc_text(ends[index]),
        }
        for index, track in enumerate(tracks)
    ]
    reference_metadata_hash = None
    if config.reference_zarr is not None:
        metadata = config.reference_zarr / ".zmetadata"
        if metadata.is_file():
            reference_metadata_hash = _file_sha256(metadata)
    return {
        "schema_version": "1.0",
        "product_store_schema_version": SUPPORTED_PRODUCT_STORE_SCHEMA_VERSION,
        "algorithm_version": BUILD_ALGORITHM_VERSION,
        "alignment_mode": config.alignment_mode,
        "reference_zarr": (
            None if config.reference_zarr is None else str(config.reference_zarr)
        ),
        "reference_zarr_metadata_sha256": reference_metadata_hash,
        "platform_count": len(tracks),
        "input_sha256": {
            item.platform_id: item.source_sha256 for item in tracks
        },
        "products": [asdict(item) for item in config.products],
        "assignment_sha256": _canonical_sha256(records),
        "platform_audit_sha256": audit_sha256,
        "effective_configuration": config.effective(),
    }


def _remove_exact_directory(path: Path, *, parent: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or resolved.name != path.name:
        raise RuntimeError(f"Refusing cleanup outside {parent}")
    shutil.rmtree(resolved)


def _publish(temporary: Path, destination: Path, *, overwrite: bool, token: str) -> None:
    if not destination.exists():
        temporary.rename(destination)
        return
    if not overwrite:
        raise FileExistsError(f"Product store already exists: {destination}")
    if not destination.is_dir():
        raise ValueError(f"Zarr output exists and is not a directory: {destination}")
    backup = destination.with_name(f".{destination.name}.{token}.backup")
    if backup.exists():
        raise FileExistsError(f"Backup path already exists: {backup}")
    destination.rename(backup)
    try:
        temporary.rename(destination)
    except Exception:
        if not destination.exists() and backup.exists():
            backup.rename(destination)
        raise
    _remove_exact_directory(backup, parent=destination.parent)


def run_supplied_qc_import(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SuppliedQCBuildResult:
    """Build and atomically publish automatic-QC trajectory products."""

    config = load_supplied_qc_config(config_path)
    report = progress or (lambda _message: None)
    if not config.input_directory.is_dir():
        raise ValueError(f"MAT input directory does not exist: {config.input_directory}")
    files = sorted(config.input_directory.glob(config.pattern))
    if not files:
        raise ValueError(
            f"No MAT files match {config.pattern!r} in {config.input_directory}"
        )
    report(f"Reading {len(files)} supplied-QC MAT files")
    tracks = sorted(
        (_prepare_track(path, config) for path in files),
        key=lambda item: item.platform_id,
    )
    platforms = [item.platform_id for item in tracks]
    if len(set(platforms)) != len(platforms):
        raise ValueError("Supplied-QC MAT files contain duplicate platform IDs")
    reference = _load_reference_root(config, platforms)

    automatic_starts = np.asarray(
        [item.automatic_start for item in tracks], dtype="datetime64[ns]",
    )
    automatic_ends = np.asarray([
        _automatic_coverage_end(item, config.coverage) for item in tracks
    ], dtype="datetime64[ns]")
    if reference is None:
        starts = automatic_starts
        ends = automatic_ends
    else:
        starts = reference.deployment_start_time.values.astype("datetime64[ns]")
        if "review_end_time" in reference:
            ends = reference.review_end_time.values.astype("datetime64[ns]")
        elif "trajectory_end_time" in reference:
            ends = reference.trajectory_end_time.values.astype("datetime64[ns]")
        else:
            ends = automatic_ends
    if np.isnat(starts).any() or np.isnat(ends).any() or np.any(ends <= starts):
        raise ValueError("Every alignment interval must have finite start < end")

    cadences = tuple(sorted({item.cadence_minutes for item in config.products}))
    if reference is None:
        grids = build_regular_grids(
            config.grid_anchor,
            starts.min(),
            ends.max(),
            cadences_minutes=cadences,
        )
    else:
        grids = {
            cadence: _reference_grid(config, cadence, platforms)
            for cadence in cadences
        }

    report("Building configured regular trajectory products")
    products = {
        specification.name: _build_product_dataset(
            specification,
            tracks,
            platforms,
            grids[specification.cadence_minutes],
            starts,
            ends,
        )
        for specification in config.products
    }
    cluster_spec = min(
        (
            item for item in config.products
            if item.representation == "native_qc"
        ),
        key=lambda item: item.cadence_minutes,
    )
    if reference is None:
        assignments, cluster_result = _derive_assignments(
            config,
            tracks,
            platforms,
            starts,
            products[cluster_spec.name],
        )
    else:
        assignments = _reference_assignments(reference, platforms, starts)
        cluster_result = None
    report(
        f"Prepared {len(set(assignments['array_id'].tolist()))} arrays and "
        f"{len(set(assignments['cluster_id'].astype(str).tolist()))} clusters"
    )

    audit = _audit_table(tracks, starts, ends, assignments)
    audit_text = audit.to_csv(index=False, lineterminator="\n")
    audit_hash = sha256(audit_text.encode()).hexdigest()
    manifest = _manifest(
        config, tracks, starts, ends, assignments, audit_hash,
    )
    root = _root_dataset(
        config,
        tracks,
        platforms,
        starts,
        ends,
        assignments,
        manifest=manifest,
    )

    destination = config.output_zarr
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Product store already exists: {destination}; use --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = destination.with_name(f".{destination.name}.{token}.tmp")
    try:
        write_supplied_product_store(
            temporary,
            root,
            products,
            manifest=manifest,
            chunk_platform=config.chunk_platform,
            chunk_time=config.chunk_time,
        )
        # Write bytes so the manifest hash is independent of host newline
        # translation (notably CRLF on Windows).
        (temporary / "platform_audit.csv").write_bytes(audit_text.encode())
        if cluster_result is not None:
            pd.DataFrame([
                asdict(item) for item in cluster_result.pair_diagnostics
            ]).to_csv(temporary / "cluster_pair_diagnostics.csv", index=False)
        for product in config.products:
            loaded, info = load_trajectory_product(temporary, product.name)
            if loaded.sizes["platform"] != len(platforms):
                raise ValueError(f"Readback changed platforms for {product.name}")
            if info.cadence_minutes != product.cadence_minutes:
                raise ValueError(f"Readback changed cadence for {product.name}")
        if _file_sha256(temporary / "platform_audit.csv") != audit_hash:
            raise ValueError("Platform audit changed during write")
        _publish(temporary, destination, overwrite=overwrite, token=token)
    finally:
        if temporary.exists():
            _remove_exact_directory(temporary, parent=destination.parent)

    report(f"Published supplied-QC Zarr: {destination}")
    return SuppliedQCBuildResult(
        zarr_path=destination,
        platform_count=len(platforms),
        product_count=len(products),
        array_count=len(set(assignments["array_id"].tolist())),
        cluster_count=len(set(assignments["cluster_id"].astype(str).tolist())),
        alignment_mode=config.alignment_mode,
    )

