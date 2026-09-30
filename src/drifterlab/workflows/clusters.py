"""Inspect observed starts and build candidate deployment clusters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
import shutil
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from drifterlab import __version__
from drifterlab.clustering import (
    candidate_clusters,
    infer_start_cohorts,
    nearest_neighbor_diagnostics,
)


CLUSTER_ALGORITHM_VERSION = "joint-complete-linkage-v1"
SUPPORTED_RECONSTRUCTION_SCHEMA_VERSION = "1.1"
INSPECTION_DIRECTORY = "inspection"
CANDIDATE_DIRECTORY = "candidate"
REVIEW_NAME = "cohort_review.csv"
NEIGHBOR_NAME = "neighbor_diagnostics.csv"
MEMBERS_NAME = "cluster_members.csv"
SUMMARY_NAME = "cluster_summary.csv"
MERGE_LOG_NAME = "merge_log.csv"
MANIFEST_NAME = "manifest.json"
REVIEW_COLUMNS = [
    "platform_id", "observed_start_time_utc", "observed_start_lon",
    "observed_start_lat", "observed_start_sha256",
    "gap_from_previous_start_hours", "gap_to_next_start_hours",
    "proposed_cohort_id", "reviewed_cohort_id", "map_marker", "include",
    "review_note",
]
SUPPORTED_MAP_PROJECTIONS = {
    "PlateCarree", "Mercator", "SouthPolarStereo", "NorthPolarStereo", "Robinson",
}


@dataclass(frozen=True)
class CohortGroupingOverride:
    maximum_cluster_diameter_m: float | None
    maximum_members_per_cluster: int | None
    maximum_observed_start_spread_minutes: float | None


@dataclass(frozen=True)
class ClusterPlottingConfig:
    enabled: bool
    projection: str
    central_longitude: float
    extent: tuple[float, float, float, float] | None
    figsize: tuple[float, float]
    dpi: int
    land: bool
    coastlines: bool
    gridlines: bool


@dataclass(frozen=True)
class ClusterWorkflowConfig:
    input_zarr: Path
    maximum_adjacent_start_gap_hours: float
    reviewed_assignments: Path | None
    maximum_cluster_diameter_m: float | None
    maximum_members_per_cluster: int
    maximum_observed_start_spread_minutes: float | None
    cohort_overrides: dict[str, CohortGroupingOverride]
    nearest_neighbor_ranks: int
    output_directory: Path
    plotting: ClusterPlottingConfig

    def effective(self) -> dict[str, Any]:
        return {
            "input": {"zarr": str(self.input_zarr)},
            "cohorts": {
                "maximum_adjacent_start_gap_hours": self.maximum_adjacent_start_gap_hours,
                "reviewed_assignments": (
                    None if self.reviewed_assignments is None else str(self.reviewed_assignments)
                ),
            },
            "grouping": {
                "maximum_cluster_diameter_m": self.maximum_cluster_diameter_m,
                "maximum_members_per_cluster": self.maximum_members_per_cluster,
                "maximum_observed_start_spread_minutes": (
                    self.maximum_observed_start_spread_minutes
                ),
                "cohort_overrides": {
                    key: {
                        "maximum_cluster_diameter_m": value.maximum_cluster_diameter_m,
                        "maximum_members_per_cluster": value.maximum_members_per_cluster,
                        "maximum_observed_start_spread_minutes": (
                            value.maximum_observed_start_spread_minutes
                        ),
                    }
                    for key, value in sorted(self.cohort_overrides.items())
                },
            },
            "diagnostics": {"nearest_neighbor_ranks": self.nearest_neighbor_ranks},
            "output": {"directory": str(self.output_directory)},
            "plotting": {
                "enabled": self.plotting.enabled,
                "projection": self.plotting.projection,
                "central_longitude": self.plotting.central_longitude,
                "extent": None if self.plotting.extent is None else list(self.plotting.extent),
                "figsize": list(self.plotting.figsize),
                "dpi": self.plotting.dpi,
                "land": self.plotting.land,
                "coastlines": self.plotting.coastlines,
                "gridlines": self.plotting.gridlines,
            },
        }


@dataclass(frozen=True)
class ClusterWorkflowResult:
    mode: str
    output_directory: Path
    platform_count: int
    cohort_count: int
    cluster_count: int | None
    figure_paths: tuple[Path, ...]

    def format(self) -> str:
        lines = [
            f"Candidate cluster {self.mode}: {self.platform_count} platforms in "
            f"{self.cohort_count} observed-start cohorts",
            f"Output: {self.output_directory}",
        ]
        if self.cluster_count is not None:
            lines.append(f"Candidate clusters: {self.cluster_count}")
        if self.figure_paths:
            lines.append(f"Figures: {len(self.figure_paths)} PNG/PDF files")
        lines.append("Input trajectories and lifetimes were read only and remain unchanged")
        return "\n".join(lines)


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _positive(value: Any, name: str, *, optional: bool = False) -> float | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def load_cluster_config(path: str | Path) -> ClusterWorkflowConfig:
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    data = _mapping(
        data, {"input", "cohorts", "grouping", "diagnostics", "output", "plotting"},
        "configuration",
    )
    source = _mapping(data.get("input", {}), {"zarr"}, "input")
    cohorts = _mapping(
        data.get("cohorts", {}),
        {"maximum_adjacent_start_gap_hours", "reviewed_assignments"}, "cohorts",
    )
    grouping = _mapping(
        data.get("grouping", {}),
        {
            "maximum_cluster_diameter_m", "maximum_members_per_cluster",
            "maximum_observed_start_spread_minutes", "cohort_overrides",
        },
        "grouping",
    )
    diagnostics = _mapping(
        data.get("diagnostics", {}), {"nearest_neighbor_ranks"}, "diagnostics",
    )
    output = _mapping(data.get("output", {}), {"directory"}, "output")
    plotting = _mapping(
        data.get("plotting", {}),
        {
            "enabled", "projection", "central_longitude", "extent", "figsize", "dpi",
            "land", "coastlines", "gridlines",
        },
        "plotting",
    )

    def resolved(value: Any, name: str, *, optional: bool = False) -> Path | None:
        if optional and (value is None or str(value).strip() in {"", "null", "None"}):
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string")
        return (source_path.parent / Path(value).expanduser()).resolve()

    input_zarr = resolved(source.get("zarr"), "input.zarr")
    output_directory = resolved(output.get("directory"), "output.directory")
    assert input_zarr is not None and output_directory is not None
    if output_directory == input_zarr or input_zarr in output_directory.parents:
        raise ValueError("Cluster output must be outside the input Zarr")

    raw_overrides = grouping.get("cohort_overrides", {})
    if not isinstance(raw_overrides, dict):
        raise ValueError("grouping.cohort_overrides must be a mapping")
    overrides: dict[str, CohortGroupingOverride] = {}
    for raw_name, raw_value in raw_overrides.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Every grouping.cohort_overrides key must be a cohort identifier")
        name = raw_name.strip()
        item = _mapping(
            raw_value,
            {
                "maximum_cluster_diameter_m", "maximum_members_per_cluster",
                "maximum_observed_start_spread_minutes",
            },
            f"grouping.cohort_overrides.{name}",
        )
        raw_members = item.get("maximum_members_per_cluster")
        overrides[name] = CohortGroupingOverride(
            _positive(
                item.get("maximum_cluster_diameter_m"),
                f"grouping.cohort_overrides.{name}.maximum_cluster_diameter_m",
                optional=True,
            ),
            None if raw_members is None else _positive_integer(
                raw_members, f"grouping.cohort_overrides.{name}.maximum_members_per_cluster",
            ),
            _positive(
                item.get("maximum_observed_start_spread_minutes"),
                f"grouping.cohort_overrides.{name}.maximum_observed_start_spread_minutes",
                optional=True,
            ),
        )

    projection = plotting.get("projection", "PlateCarree")
    if not isinstance(projection, str) or projection not in SUPPORTED_MAP_PROJECTIONS:
        raise ValueError(f"plotting.projection must be one of {sorted(SUPPORTED_MAP_PROJECTIONS)}")
    central_longitude = float(plotting.get("central_longitude", 0))
    if not math.isfinite(central_longitude) or not -180 <= central_longitude <= 180:
        raise ValueError("plotting.central_longitude must be between -180 and 180")
    raw_extent = plotting.get("extent")
    extent: tuple[float, float, float, float] | None = None
    if raw_extent is not None:
        if not isinstance(raw_extent, list) or len(raw_extent) != 4:
            raise ValueError("plotting.extent must be null or [west, east, south, north]")
        extent = tuple(float(value) for value in raw_extent)
        if (not all(math.isfinite(value) for value in extent)
                or extent[0] >= extent[1] or extent[2] >= extent[3]
                or extent[2] < -90 or extent[3] > 90):
            raise ValueError("plotting.extent contains invalid geographic bounds")
    raw_figsize = plotting.get("figsize", [12, 9])
    if not isinstance(raw_figsize, list) or len(raw_figsize) != 2:
        raise ValueError("plotting.figsize must contain width and height")
    figsize = tuple(float(_positive(value, "plotting.figsize")) for value in raw_figsize)
    plot_config = ClusterPlottingConfig(
        _boolean(plotting.get("enabled", True), "plotting.enabled"), projection,
        central_longitude, extent, figsize,
        _positive_integer(plotting.get("dpi", 200), "plotting.dpi"),
        _boolean(plotting.get("land", True), "plotting.land"),
        _boolean(plotting.get("coastlines", True), "plotting.coastlines"),
        _boolean(plotting.get("gridlines", True), "plotting.gridlines"),
    )
    return ClusterWorkflowConfig(
        input_zarr=input_zarr,
        maximum_adjacent_start_gap_hours=float(_positive(
            cohorts.get("maximum_adjacent_start_gap_hours", 24),
            "cohorts.maximum_adjacent_start_gap_hours",
        )),
        reviewed_assignments=resolved(
            cohorts.get("reviewed_assignments"), "cohorts.reviewed_assignments", optional=True,
        ),
        maximum_cluster_diameter_m=_positive(
            grouping.get("maximum_cluster_diameter_m"),
            "grouping.maximum_cluster_diameter_m", optional=True,
        ),
        maximum_members_per_cluster=_positive_integer(
            grouping.get("maximum_members_per_cluster", 5),
            "grouping.maximum_members_per_cluster",
        ),
        maximum_observed_start_spread_minutes=_positive(
            grouping.get("maximum_observed_start_spread_minutes"),
            "grouping.maximum_observed_start_spread_minutes", optional=True,
        ),
        cohort_overrides=overrides,
        nearest_neighbor_ranks=_positive_integer(
            diagnostics.get("nearest_neighbor_ranks", 5),
            "diagnostics.nearest_neighbor_ranks",
        ),
        output_directory=output_directory,
        plotting=plot_config,
    )


def _start_sha256(platform: str, time: np.datetime64, longitude: float, latitude: float) -> str:
    digest = sha256()
    digest.update(platform.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(time, dtype="datetime64[ns]").astype("<i8").tobytes())
    digest.update(np.asarray([longitude, latitude], dtype="<f8").tobytes())
    return digest.hexdigest()


def _inventory_sha256(starts: pd.DataFrame) -> str:
    values = starts.sort_values("platform_id", kind="stable").observed_start_sha256.astype(str)
    return sha256("\n".join(values).encode()).hexdigest()


def _load_observed_starts(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_dir():
        raise ValueError(f"Input reconstructed Zarr does not exist: {path}")
    try:
        with xr.open_zarr(path, consolidated=True, chunks=None) as dataset:
            required = {"platform_id", "start_time", "start_lon", "start_lat"}
            missing = required - set(dataset.variables)
            if missing:
                raise ValueError(f"Reconstructed Zarr is missing variables: {sorted(missing)}")
            for name in required:
                if dataset[name].dims != ("platform",):
                    raise ValueError(f"Reconstructed Zarr {name} must have dimensions ('platform',)")
            platform = dataset.platform_id.values.astype(str)
            time = dataset.start_time.values.astype("datetime64[ns]")
            longitude = dataset.start_lon.values.astype(float)
            latitude = dataset.start_lat.values.astype(float)
            attrs = dict(dataset.attrs)
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot read reconstructed Zarr {path}: {exc}") from exc
    if not len(platform):
        raise ValueError("Reconstructed Zarr contains no platforms")
    if attrs.get("schema_version") != SUPPORTED_RECONSTRUCTION_SCHEMA_VERSION:
        raise ValueError(
            f"Reconstructed Zarr schema is {attrs.get('schema_version')!r}; expected "
            f"{SUPPORTED_RECONSTRUCTION_SCHEMA_VERSION!r}"
        )
    if len(set(platform.tolist())) != len(platform) or any(not value.strip() for value in platform):
        raise ValueError("Reconstructed Zarr platform identifiers must be unique and nonempty")
    if np.isnat(time).any():
        raise ValueError("Reconstructed Zarr contains missing observed start times")
    if (not np.isfinite(longitude).all() or not np.isfinite(latitude).all()
            or np.any((longitude < -180) | (longitude >= 180))
            or np.any((latitude < -90) | (latitude > 90))):
        raise ValueError("Reconstructed Zarr contains invalid observed start coordinates")
    starts = pd.DataFrame({
        "platform_id": platform,
        "observed_start_time": pd.to_datetime(time, utc=True),
        "observed_start_lon": longitude,
        "observed_start_lat": latitude,
    }).sort_values("platform_id", kind="stable").reset_index(drop=True)
    starts["observed_start_sha256"] = [
        _start_sha256(str(row.platform_id), row.observed_start_time.to_datetime64(),
                      float(row.observed_start_lon), float(row.observed_start_lat))
        for row in starts.itertuples(index=False)
    ]
    provenance = {
        "path": str(path),
        "schema_version": attrs.get("schema_version"),
        "algorithm_version": attrs.get("algorithm_version"),
        "product_status": attrs.get("product_status"),
        "build_report_sha256": attrs.get("build_report_sha256"),
        "observed_start_inventory_sha256": _inventory_sha256(starts),
    }
    return starts, provenance


def _marker_values(frame: pd.DataFrame, cohort_column: str) -> pd.Series:
    result = pd.Series(index=frame.index, dtype="Int64")
    for _cohort, group in frame.groupby(cohort_column, sort=True):
        order = group.sort_values(["observed_start_time", "platform_id"], kind="stable").index
        result.loc[order] = np.arange(1, len(order) + 1)
    return result


def _proposed_review(starts: pd.DataFrame, config: ClusterWorkflowConfig) -> pd.DataFrame:
    proposed = infer_start_cohorts(
        starts, maximum_adjacent_gap_hours=config.maximum_adjacent_start_gap_hours,
    )
    proposed["reviewed_cohort_id"] = proposed.proposed_cohort_id
    proposed["map_marker"] = _marker_values(proposed, "reviewed_cohort_id")
    proposed["include"] = True
    proposed["review_note"] = ""
    proposed["observed_start_time_utc"] = proposed.observed_start_time.map(_utc)
    return proposed[REVIEW_COLUMNS].copy()


def _parse_include(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cohort review include must be true or false, got {value!r}")


def _load_review(path: Path, starts: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Reviewed cohort assignments do not exist: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.columns.tolist() != REVIEW_COLUMNS:
        raise ValueError(f"Cohort review columns must be exactly {REVIEW_COLUMNS}")
    if frame.platform_id.duplicated().any():
        raise ValueError("Cohort review contains duplicate platform identifiers")
    expected = set(starts.platform_id.astype(str))
    actual = set(frame.platform_id.astype(str))
    if expected != actual:
        raise ValueError(
            f"Cohort review platform inventory differs; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    fingerprints = dict(zip(starts.platform_id.astype(str), starts.observed_start_sha256.astype(str)))
    stale = [
        row.platform_id for row in frame.itertuples(index=False)
        if row.observed_start_sha256 != fingerprints[row.platform_id]
    ]
    if stale:
        raise ValueError(f"Cohort review is stale for observed starts: {sorted(stale)}")
    frame["include"] = frame.include.map(_parse_include)
    included = frame.include.to_numpy(dtype=bool)
    if not included.any():
        raise ValueError("Cohort review excludes every platform")
    empty = frame.reviewed_cohort_id.astype(str).str.strip().eq("") & included
    if empty.any():
        raise ValueError(
            f"Included platforms need reviewed_cohort_id: {frame.loc[empty, 'platform_id'].tolist()}"
        )
    actual_starts = starts[[
        "platform_id", "observed_start_time", "observed_start_lon", "observed_start_lat",
        "observed_start_sha256",
    ]]
    review_fields = frame[[
        "platform_id", "gap_from_previous_start_hours", "gap_to_next_start_hours",
        "proposed_cohort_id", "reviewed_cohort_id", "include", "review_note",
    ]]
    result = actual_starts.merge(review_fields, on="platform_id", how="left", validate="one_to_one")
    result["map_marker"] = pd.NA
    selected = result.include.to_numpy(dtype=bool)
    result.loc[selected, "map_marker"] = _marker_values(
        result.loc[selected], "reviewed_cohort_id",
    )
    return result


def _review_output(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["observed_start_time_utc"] = pd.to_datetime(
        result.observed_start_time, utc=True,
    ).map(_utc)
    for name in ("gap_from_previous_start_hours", "gap_to_next_start_hours"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    return result[REVIEW_COLUMNS]


def _starts_for_diagnostics(review: pd.DataFrame) -> pd.DataFrame:
    selected = review.loc[review.include.to_numpy(dtype=bool)].copy()
    selected["cohort_id"] = selected.reviewed_cohort_id.astype(str)
    return selected[[
        "platform_id", "observed_start_time", "observed_start_lon", "observed_start_lat",
        "cohort_id", "map_marker",
    ]]


def _grouping_parameters(
    config: ClusterWorkflowConfig, cohort_id: str,
) -> tuple[float, float, int]:
    override = config.cohort_overrides.get(cohort_id)
    diameter = (
        override.maximum_cluster_diameter_m
        if override is not None and override.maximum_cluster_diameter_m is not None
        else config.maximum_cluster_diameter_m
    )
    spread = (
        override.maximum_observed_start_spread_minutes
        if override is not None and override.maximum_observed_start_spread_minutes is not None
        else config.maximum_observed_start_spread_minutes
    )
    members = (
        override.maximum_members_per_cluster
        if override is not None and override.maximum_members_per_cluster is not None
        else config.maximum_members_per_cluster
    )
    if diameter is None:
        raise ValueError(
            f"Grouping needs maximum_cluster_diameter_m for cohort {cohort_id!r}"
        )
    if spread is None:
        raise ValueError(
            f"Grouping needs maximum_observed_start_spread_minutes for cohort {cohort_id!r}"
        )
    return diameter, spread, members


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(
    config: ClusterWorkflowConfig, mode: str, provenance: dict[str, Any],
    review_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "title": "Observed-start inspection" if mode == "inspection" else "Candidate deployment clusters",
        "mode": mode,
        "algorithm_version": CLUSTER_ALGORITHM_VERSION,
        "drifterlab_version": __version__,
        "processing_time_utc": datetime.now(timezone.utc).isoformat(),
        "start_semantics": "Exact first retained QC fixes; not verified deployment coordinates",
        "lifetime_policy": "Trajectory values and valid spans are not modified or used for grouping",
        "effective_configuration": config.effective(),
        "input": provenance,
        "review_sha256": None if review_path is None else _file_sha256(review_path),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remove_generated(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith("."):
        raise RuntimeError(f"Refusing cleanup outside generated paths in {parent}")
    shutil.rmtree(resolved)


def _publish(
    target: Path, *, overwrite: bool, writer: Callable[[Path], tuple[Path, ...]],
) -> tuple[Path, ...]:
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = target.with_name(f".{target.name}.{token}.tmp")
    backup = target.with_name(f".{target.name}.{token}.backup")
    temporary.mkdir()
    try:
        relative_figures = writer(temporary)
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"Cluster output already exists: {target}; use --overwrite")
            target.rename(backup)
        try:
            temporary.rename(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
        if backup.exists():
            _remove_generated(backup, target.parent)
        return tuple(target / path for path in relative_figures)
    finally:
        if temporary.exists():
            _remove_generated(temporary, target.parent)


def run_cluster_workflow(
    config_path: str | Path, *, inspection: bool = False, overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ClusterWorkflowResult:
    """Inspect exact observed starts or publish reviewed candidate clusters."""
    config = load_cluster_config(config_path)
    report = progress or (lambda message: None)
    starts, provenance = _load_observed_starts(config.input_zarr)
    proposed = _proposed_review(starts, config)
    if config.reviewed_assignments is None:
        review = proposed.copy()
        review["observed_start_time"] = pd.to_datetime(review.observed_start_time_utc, utc=True)
    else:
        review = _load_review(config.reviewed_assignments, starts)
    diagnostic_starts = _starts_for_diagnostics(review)
    cohort_count = int(diagnostic_starts.cohort_id.nunique())

    if inspection:
        neighbors = nearest_neighbor_diagnostics(
            diagnostic_starts, ranks=config.nearest_neighbor_ranks,
        )
        target = config.output_directory / INSPECTION_DIRECTORY

        def write_inspection(output: Path) -> tuple[Path, ...]:
            _review_output(review).to_csv(output / REVIEW_NAME, index=False)
            neighbors.to_csv(output / NEIGHBOR_NAME, index=False)
            _write_json(
                output / MANIFEST_NAME,
                _manifest(
                    config, "inspection", provenance,
                    config.reviewed_assignments,
                ),
            )
            figures: tuple[Path, ...] = ()
            if config.plotting.enabled:
                try:
                    from drifterlab.plotting.clusters import generate_inspection_figures
                except ImportError as exc:
                    raise ImportError(
                        "Cluster plotting requires the plotting dependencies; install drifterlab[plotting]"
                    ) from exc
                figures = generate_inspection_figures(
                    diagnostic_starts, neighbors, config.plotting, output / "figures",
                )
            return tuple(path.relative_to(output) for path in figures)

        figure_paths = _publish(target, overwrite=overwrite, writer=write_inspection)
        report(f"Published observed-start inspection for {len(starts)} platforms")
        return ClusterWorkflowResult(
            "inspection", target, len(starts), cohort_count, None, figure_paths,
        )

    if config.reviewed_assignments is None:
        raise ValueError(
            "Grouping requires cohorts.reviewed_assignments; run --inspection, review "
            "cohort_review.csv, and reference it from the YAML"
        )
    unknown_overrides = set(config.cohort_overrides) - set(diagnostic_starts.cohort_id)
    if unknown_overrides:
        raise ValueError(
            f"Grouping overrides name unknown reviewed cohorts: {sorted(unknown_overrides)}"
        )
    member_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    merge_frames: list[pd.DataFrame] = []
    for cohort_id, group in diagnostic_starts.groupby("cohort_id", sort=True):
        diameter, spread, members = _grouping_parameters(config, str(cohort_id))
        member_frame, summary_frame, merge_frame = candidate_clusters(
            group, maximum_diameter_m=diameter,
            maximum_start_spread_minutes=spread,
            maximum_members=members, cohort_id=str(cohort_id),
        )
        member_frames.append(member_frame)
        summary_frames.append(summary_frame)
        merge_frames.append(merge_frame)
    cluster_members = pd.concat(member_frames, ignore_index=True)
    cluster_summary = pd.concat(summary_frames, ignore_index=True)
    merge_log = pd.concat(merge_frames, ignore_index=True)
    excluded = review.loc[~review.include.to_numpy(dtype=bool)]
    if not excluded.empty:
        excluded_rows = pd.DataFrame({
            "cohort_id": excluded.reviewed_cohort_id.astype(str),
            "candidate_cluster_id": "",
            "platform_id": excluded.platform_id.astype(str),
            "observed_start_time_utc": excluded.observed_start_time.map(_utc),
            "observed_start_lon": excluded.observed_start_lon.astype(float),
            "observed_start_lat": excluded.observed_start_lat.astype(float),
            "member_count": 0,
            "cluster_diameter_m": np.nan,
            "cluster_observed_start_spread_minutes": np.nan,
            "candidate_status": "excluded_by_review",
        })
        cluster_members = pd.concat([cluster_members, excluded_rows], ignore_index=True)
    cluster_members = cluster_members.sort_values(
        ["cohort_id", "candidate_cluster_id", "platform_id"], kind="stable",
    )
    target = config.output_directory / CANDIDATE_DIRECTORY

    def write_candidate(output: Path) -> tuple[Path, ...]:
        cluster_members.to_csv(output / MEMBERS_NAME, index=False)
        cluster_summary.to_csv(output / SUMMARY_NAME, index=False)
        merge_log.to_csv(output / MERGE_LOG_NAME, index=False)
        _write_json(
            output / MANIFEST_NAME,
            _manifest(config, "candidate", provenance, config.reviewed_assignments),
        )
        figures: tuple[Path, ...] = ()
        if config.plotting.enabled:
            try:
                from drifterlab.plotting.clusters import generate_candidate_figures
            except ImportError as exc:
                raise ImportError(
                    "Cluster plotting requires the plotting dependencies; install drifterlab[plotting]"
                ) from exc
            figures = generate_candidate_figures(
                cluster_members, cluster_summary, config.plotting, output / "figures",
            )
        return tuple(path.relative_to(output) for path in figures)

    figure_paths = _publish(target, overwrite=overwrite, writer=write_candidate)
    report(f"Published {len(cluster_summary)} candidate clusters")
    return ClusterWorkflowResult(
        "build", target, len(starts), cohort_count, len(cluster_summary), figure_paths,
    )


def _utc(value: Any) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")
