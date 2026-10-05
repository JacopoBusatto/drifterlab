"""Build overshoot-aware FSLE spectra from an authoritative pair Zarr."""

from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import Geod

from drifterlab import __version__
from drifterlab.fsle import (
    aggregate_fsle_spectra,
    build_scale_thresholds,
    collect_pair_first_passages,
)
from drifterlab.plotting.fsle import render_array_fsle_figures

FSLE_SCHEMA_VERSION = "1.1"
FSLE_ALGORITHM_VERSION = "overshoot-first-observed-shell-exit-v2"
SUPPORTED_PAIR_SCHEMA_VERSION = "1.2"
SPECTRUM_NAME = "fsle_spectra.csv"
PASSAGES_NAME = "fsle_first_passages.parquet"
MANIFEST_NAME = "fsle_manifest.json"
WGS84 = Geod(ellps="WGS84")


@dataclass(frozen=True)
class FSLEReferenceConfig:
    enabled: bool
    exponent: float
    label: str


@dataclass(frozen=True)
class FSLEPlotConfig:
    dpi: int
    x_range_km: tuple[float | None, float | None]
    y_range_day_inverse: tuple[float | None, float | None]
    standard_error_bars: bool
    reference: FSLEReferenceConfig


@dataclass(frozen=True)
class FSLEWorkflowConfig:
    source_path: Path
    pair_zarr: Path
    coordinate_method: str
    output_directory: Path
    minimum_scale_km: float
    maximum_scale_km: float
    rho: float
    anchor_scale_km: float
    expected_interval_minutes: float
    minimum_reached_pairs_per_scale: int
    plotting: FSLEPlotConfig

    def effective(self) -> dict[str, Any]:
        return {
            "input": {
                "pairs_zarr": str(self.pair_zarr),
                "coordinate_method": self.coordinate_method,
            },
            "analysis": {
                "minimum_scale_km": self.minimum_scale_km,
                "maximum_scale_km": self.maximum_scale_km,
                "rho": self.rho,
                "anchor_scale_km": self.anchor_scale_km,
                "expected_interval_minutes": self.expected_interval_minutes,
                "minimum_reached_pairs_per_scale": (
                    self.minimum_reached_pairs_per_scale
                ),
                "estimator": "mean(log(exit_distance / entry_distance)) / mean(exit_time - entry_time)",
                "start": "encounter_observation",
                "membership": "same_array and same_cluster pairs only",
                "standard_error": (
                    "sqrt((mean(log_growth^2 / passage_time) / mean(passage_time) "
                    "- fsle^2) / reached_pair_count)"
                ),
            },
            "plotting": {
                "dpi": self.plotting.dpi,
                "x_range_km": list(self.plotting.x_range_km),
                "y_range_day_inverse": list(self.plotting.y_range_day_inverse),
                "standard_error_bars": self.plotting.standard_error_bars,
                "reference_slope": asdict(self.plotting.reference),
            },
            "output": {"directory": str(self.output_directory)},
        }


@dataclass(frozen=True)
class FSLEWorkflowResult:
    output_directory: Path
    spectrum_path: Path
    passages_path: Path
    manifest_path: Path
    figure_paths: tuple[Path, ...]
    array_count: int
    analyzed_pair_count: int
    passage_count: int

    def format(self) -> str:
        return "\n".join(
            [
                (
                    f"FSLE complete: {self.array_count} arrays, "
                    f"{self.analyzed_pair_count:,} same-cluster pairs"
                ),
                f"Pair-scale records: {self.passage_count:,}",
                f"Figures: {len(self.figure_paths)}",
                f"Spectrum: {self.spectrum_path}",
                f"First passages: {self.passages_path}",
                f"Manifest: {self.manifest_path}",
            ]
        )


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _range(
    value: Any,
    name: str,
) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be null or [minimum, maximum]")
    parsed: list[float | None] = []
    for index, item in enumerate(value):
        if item is None:
            parsed.append(None)
        else:
            parsed.append(_positive(item, f"{name}[{index}]"))
    lower, upper = parsed
    if lower is not None and upper is not None and upper <= lower:
        raise ValueError(f"{name} maximum must exceed its minimum")
    return lower, upper


def load_fsle_config(path: str | Path) -> FSLEWorkflowConfig:
    """Read and strictly validate an FSLE YAML file."""
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8-sig") as stream:
        values = yaml.safe_load(stream)
    values = _mapping(
        values, {"input", "analysis", "plotting", "output"}, "configuration"
    )
    input_section = _mapping(
        values.get("input", {}),
        {"pairs_zarr", "coordinate_method"},
        "input",
    )
    analysis = _mapping(
        values.get("analysis", {}),
        {
            "minimum_scale_km",
            "maximum_scale_km",
            "rho",
            "anchor_scale_km",
            "expected_interval_minutes",
            "minimum_reached_pairs_per_scale",
        },
        "analysis",
    )
    plotting = _mapping(
        values.get("plotting", {}),
        {
            "dpi",
            "x_range_km",
            "y_range_day_inverse",
            "standard_error_bars",
            "reference_slope",
        },
        "plotting",
    )
    reference = _mapping(
        plotting.get("reference_slope", {}),
        {"enabled", "exponent", "label"},
        "plotting.reference_slope",
    )
    output = _mapping(values.get("output", {}), {"directory"}, "output")

    def resolved(value: Any, name: str) -> Path:
        text = _nonempty(value, name)
        candidate = Path(text).expanduser()
        return (source_path.parent / candidate).resolve()

    pair_zarr = resolved(input_section.get("pairs_zarr"), "input.pairs_zarr")
    output_directory = resolved(output.get("directory"), "output.directory")
    if (
        pair_zarr == output_directory
        or pair_zarr in output_directory.parents
        or output_directory in pair_zarr.parents
    ):
        raise ValueError("FSLE output and input pair Zarr paths must not overlap")
    method = _nonempty(
        input_section.get("coordinate_method"), "input.coordinate_method"
    )
    if not re.fullmatch(r"[A-Za-z0-9_]+", method):
        raise ValueError("input.coordinate_method contains unsupported characters")

    minimum = _positive(analysis.get("minimum_scale_km"), "analysis.minimum_scale_km")
    maximum = _positive(analysis.get("maximum_scale_km"), "analysis.maximum_scale_km")
    rho = _positive(analysis.get("rho"), "analysis.rho")
    anchor = _positive(analysis.get("anchor_scale_km", 1), "analysis.anchor_scale_km")
    build_scale_thresholds(minimum, maximum, rho, anchor)
    return FSLEWorkflowConfig(
        source_path=source_path,
        pair_zarr=pair_zarr,
        coordinate_method=method,
        output_directory=output_directory,
        minimum_scale_km=minimum,
        maximum_scale_km=maximum,
        rho=rho,
        anchor_scale_km=anchor,
        expected_interval_minutes=_positive(
            analysis.get("expected_interval_minutes"),
            "analysis.expected_interval_minutes",
        ),
        minimum_reached_pairs_per_scale=_positive_integer(
            analysis.get("minimum_reached_pairs_per_scale", 3),
            "analysis.minimum_reached_pairs_per_scale",
        ),
        plotting=FSLEPlotConfig(
            dpi=_positive_integer(plotting.get("dpi", 150), "plotting.dpi"),
            x_range_km=_range(plotting.get("x_range_km"), "plotting.x_range_km"),
            y_range_day_inverse=_range(
                plotting.get("y_range_day_inverse"),
                "plotting.y_range_day_inverse",
            ),
            standard_error_bars=_boolean(
                plotting.get("standard_error_bars", True),
                "plotting.standard_error_bars",
            ),
            reference=FSLEReferenceConfig(
                enabled=_boolean(
                    reference.get("enabled", True),
                    "plotting.reference_slope.enabled",
                ),
                exponent=_finite(
                    reference.get("exponent", -2 / 3),
                    "plotting.reference_slope.exponent",
                ),
                label=_nonempty(
                    reference.get("label", "delta^-2/3"),
                    "plotting.reference_slope.label",
                ),
            ),
        ),
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_metadata_hash(path: Path) -> str:
    metadata = path / ".zmetadata"
    if not metadata.is_file():
        raise ValueError(f"Pair Zarr lacks consolidated metadata: {metadata}")
    return _file_sha256(metadata)


def _validate_dataset(dataset: xr.Dataset, config: FSLEWorkflowConfig) -> None:
    if dataset.attrs.get("schema_version") != SUPPORTED_PAIR_SCHEMA_VERSION:
        raise ValueError(
            f"Input pair schema is {dataset.attrs.get('schema_version')!r}; "
            f"expected {SUPPORTED_PAIR_SCHEMA_VERSION!r}"
        )
    if dataset.attrs.get("product_status") != "candidate_pairs":
        raise ValueError("Input is not a candidate-pair product")
    required = {
        "time",
        "group_id",
        "group_size",
        "platform_code_1",
        "platform_code_2",
        "array_id_1",
        "array_id_2",
        "cluster_id_1",
        "cluster_id_2",
        "cluster_size_1",
        "cluster_size_2",
        "same_array",
        "same_cluster",
        "encounter_observation",
        "common_overlap_observations",
        f"lon_{config.coordinate_method}_1",
        f"lat_{config.coordinate_method}_1",
        f"lon_{config.coordinate_method}_2",
        f"lat_{config.coordinate_method}_2",
    }
    missing = required - set(dataset.variables)
    if missing:
        raise ValueError(f"Pair Zarr is missing FSLE variables: {sorted(missing)}")
    if set(dataset.sizes) != {"trajectory", "obs"}:
        raise ValueError(
            "Pair Zarr must use exactly trajectory and obs dimensions; "
            f"found {dict(dataset.sizes)}"
        )
    methods = tuple(
        str(value) for value in dataset.attrs.get("available_coordinate_methods", ())
    )
    if config.coordinate_method not in methods:
        raise ValueError(
            f"Coordinate method {config.coordinate_method!r} is not declared by the pair Zarr; "
            f"available={list(methods)}"
        )
    if not np.all(dataset.group_size.values == 2):
        raise ValueError("FSLE pair Zarr must have group_size=2 for every trajectory")

    array_1 = dataset.array_id_1.values.astype(np.int64)
    array_2 = dataset.array_id_2.values.astype(np.int64)
    cluster_1 = dataset.cluster_id_1.values.astype(str)
    cluster_2 = dataset.cluster_id_2.values.astype(str)
    expected_same_array = array_1 == array_2
    expected_same_cluster = expected_same_array & (cluster_1 == cluster_2)
    if not np.array_equal(dataset.same_array.values.astype(bool), expected_same_array):
        raise ValueError("Pair Zarr same_array flags disagree with array identifiers")
    if not np.array_equal(
        dataset.same_cluster.values.astype(bool), expected_same_cluster
    ):
        raise ValueError(
            "Pair Zarr same_cluster flags disagree with cluster identifiers"
        )
    size_1 = dataset.cluster_size_1.values.astype(float)
    size_2 = dataset.cluster_size_2.values.astype(float)
    if (
        not np.isfinite(size_1).all()
        or not np.isfinite(size_2).all()
        or not np.equal(size_1, np.floor(size_1)).all()
        or not np.equal(size_2, np.floor(size_2)).all()
        or (size_1 < 1).any()
        or (size_2 < 1).any()
    ):
        raise ValueError(
            "Pair Zarr cluster_size values must be finite positive integers"
        )
    same_cluster = expected_same_cluster
    if not np.array_equal(size_1[same_cluster], size_2[same_cluster]):
        raise ValueError("Same-cluster pair members disagree on cluster_size")


def _pair_inventory(dataset: xr.Dataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": dataset.group_id.values.astype(str),
            "platform_code_1": dataset.platform_code_1.values.astype(str),
            "platform_code_2": dataset.platform_code_2.values.astype(str),
            "array_id_1": dataset.array_id_1.values.astype(np.int64),
            "array_id_2": dataset.array_id_2.values.astype(np.int64),
            "cluster_id_1": dataset.cluster_id_1.values.astype(str),
            "cluster_id_2": dataset.cluster_id_2.values.astype(str),
            "cluster_size_1": dataset.cluster_size_1.values.astype(np.int64),
            "cluster_size_2": dataset.cluster_size_2.values.astype(np.int64),
            "same_array": dataset.same_array.values.astype(bool),
            "same_cluster": dataset.same_cluster.values.astype(bool),
            "encounter_observation": dataset.encounter_observation.values.astype(
                np.int64
            ),
            "common_overlap_observations": (
                dataset.common_overlap_observations.values.astype(np.int64)
            ),
            "trajectory_index": np.arange(dataset.sizes["trajectory"], dtype=np.int64),
        }
    )


def _validate_inventory(inventory: pd.DataFrame, maximum_obs: int) -> None:
    if inventory.pair_id.duplicated().any():
        raise ValueError("Pair Zarr group_id values must be unique")
    if not inventory.same_array.any():
        raise ValueError("Pair Zarr contains no selected same-array pairs")
    invalid = (
        (inventory.encounter_observation < 0)
        | (inventory.common_overlap_observations < 1)
        | (inventory.common_overlap_observations > maximum_obs)
        | (inventory.encounter_observation >= inventory.common_overlap_observations)
    )
    if invalid.any():
        pair_id = inventory.loc[invalid, "pair_id"].iloc[0]
        raise ValueError(
            f"Pair {pair_id!r} has invalid encounter/common-window indices"
        )
    duplicate_members = inventory.apply(
        lambda row: tuple(sorted((row.platform_code_1, row.platform_code_2))),
        axis=1,
    ).duplicated()
    if duplicate_members.any():
        raise ValueError("Pair Zarr contains duplicate unordered platform pairs")


def _collect_passages(
    dataset: xr.Dataset,
    inventory: pd.DataFrame,
    config: FSLEWorkflowConfig,
    thresholds: np.ndarray,
    report: Callable[[str], None],
) -> pd.DataFrame:
    selected = inventory[inventory.same_cluster].reset_index(drop=True)
    if selected.empty:
        raise ValueError("Pair Zarr contains no selected same-cluster pairs")
    lon_1_name = f"lon_{config.coordinate_method}_1"
    lat_1_name = f"lat_{config.coordinate_method}_1"
    lon_2_name = f"lon_{config.coordinate_method}_2"
    lat_2_name = f"lat_{config.coordinate_method}_2"
    frames: list[pd.DataFrame] = []
    for number, row in enumerate(selected.itertuples(index=False), start=1):
        trajectory_index = int(row.trajectory_index)
        start = int(row.encounter_observation)
        stop = int(row.common_overlap_observations)
        selector = {"trajectory": trajectory_index, "obs": slice(start, stop)}
        times = dataset.time.isel(selector).values.astype("datetime64[ns]")
        lon_1 = dataset[lon_1_name].isel(selector).values.astype(float)
        lat_1 = dataset[lat_1_name].isel(selector).values.astype(float)
        lon_2 = dataset[lon_2_name].isel(selector).values.astype(float)
        lat_2 = dataset[lat_2_name].isel(selector).values.astype(float)
        _, _, distance_m = WGS84.inv(lon_1, lat_1, lon_2, lat_2)
        frames.append(
            collect_pair_first_passages(
                pair_id=row.pair_id,
                array_id=int(row.array_id_1),
                cluster_id=row.cluster_id_1,
                platform_code_1=row.platform_code_1,
                platform_code_2=row.platform_code_2,
                times=times,
                distance_km=np.abs(np.asarray(distance_m, dtype=float)) / 1000,
                thresholds_km=thresholds,
                expected_interval_seconds=config.expected_interval_minutes * 60,
            )
        )
        if number % 50 == 0 or number == len(selected):
            report(f"Calculated first passages for {number:,}/{len(selected):,} pairs")
    return pd.concat(frames, ignore_index=True)


def _cluster_inventory(inventory: pd.DataFrame) -> list[dict[str, Any]]:
    same = inventory[inventory.same_cluster].copy()
    rows: list[dict[str, Any]] = []
    for (array_id, cluster_id), frame in same.groupby(
        ["array_id_1", "cluster_id_1"],
        sort=True,
    ):
        sizes = set(frame.cluster_size_1.astype(int)) | set(
            frame.cluster_size_2.astype(int)
        )
        if len(sizes) != 1:
            raise ValueError(
                f"Array {array_id} cluster {cluster_id!r} has inconsistent cluster_size values"
            )
        assigned_size = sizes.pop()
        expected = assigned_size * (assigned_size - 1) // 2
        selected = len(frame)
        if selected > expected:
            raise ValueError(
                f"Array {array_id} cluster {cluster_id!r} contains more pairs than cluster_size permits"
            )
        rows.append(
            {
                "array_id": int(array_id),
                "cluster_id": str(cluster_id),
                "assigned_cluster_size": int(assigned_size),
                "possible_within_cluster_pair_count": int(expected),
                "selected_within_cluster_pair_count": selected,
                "selected_pair_fraction": selected / expected if expected else None,
            }
        )
    return rows


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
        raise FileExistsError(f"FSLE output already exists: {target}; use --overwrite")
    backup = target.with_name(f".{target.name}.{token}.backup")
    if backup.exists():
        raise FileExistsError(f"FSLE backup path already exists: {backup}")
    target.rename(backup)
    try:
        temporary.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    _remove_generated(backup, target.parent)


def _write_bundle(
    config: FSLEWorkflowConfig,
    inventory: pd.DataFrame,
    passages: pd.DataFrame,
    spectrum: pd.DataFrame,
    thresholds: np.ndarray,
    input_attrs: dict[str, Any],
    input_metadata_sha256: str,
    *,
    overwrite: bool,
) -> FSLEWorkflowResult:
    target = config.output_directory
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = target.with_name(f".{target.name}.{token}.tmp")
    temporary.mkdir()
    try:
        spectrum_path = temporary / SPECTRUM_NAME
        passages_path = temporary / PASSAGES_NAME
        spectrum.to_csv(spectrum_path, index=False)
        passages.to_parquet(passages_path, index=False)
        array_ids = sorted(
            inventory.loc[inventory.same_array, "array_id_1"].astype(int).unique()
        )
        figure_paths: list[Path] = []
        array_manifest: list[dict[str, Any]] = []
        for array_id in array_ids:
            directory = temporary / f"array_{array_id:03d}"
            rendered = render_array_fsle_figures(
                spectrum,
                array_id=array_id,
                coordinate_method=config.coordinate_method,
                directory=directory,
                dpi=config.plotting.dpi,
                x_range_km=config.plotting.x_range_km,
                y_range_day_inverse=config.plotting.y_range_day_inverse,
                reference_enabled=config.plotting.reference.enabled,
                reference_exponent=config.plotting.reference.exponent,
                reference_label=config.plotting.reference.label,
                anchor_scale_km=config.anchor_scale_km,
                standard_error_bars_enabled=config.plotting.standard_error_bars,
            )
            figure_paths.extend(rendered)
            pooled_anchor = spectrum[
                (spectrum.array_id == array_id)
                & (spectrum.scope == "same_cluster_pooled")
                & np.isclose(spectrum.scale_lower_km, config.anchor_scale_km)
            ]
            anchor_value = None
            anchor_available = False
            if len(pooled_anchor):
                value = float(pooled_anchor.iloc[0].fsle_day_inverse)
                anchor_available = bool(
                    pooled_anchor.iloc[0].plot_included
                    and math.isfinite(value)
                    and value > 0
                )
                anchor_value = value if anchor_available else None
            array_inventory = inventory[
                inventory.same_array & (inventory.array_id_1 == array_id)
            ]
            same_cluster = array_inventory[array_inventory.same_cluster]
            array_manifest.append(
                {
                    "array_id": int(array_id),
                    "selected_same_array_pair_count": len(array_inventory),
                    "selected_same_cluster_pair_count": len(same_cluster),
                    "selected_between_cluster_pair_count": int(
                        len(array_inventory) - len(same_cluster)
                    ),
                    "represented_cluster_count": int(
                        same_cluster.cluster_id_1.nunique()
                    ),
                    "reference_anchor_available": anchor_available,
                    "reference_anchor_fsle_day_inverse": anchor_value,
                    "figures": [str(path.relative_to(temporary)) for path in rendered],
                }
            )

        product_paths = [spectrum_path, passages_path, *figure_paths]
        manifest = {
            "schema_version": FSLE_SCHEMA_VERSION,
            "algorithm_version": FSLE_ALGORITHM_VERSION,
            "drifterlab_version": __version__,
            "processing_time_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_path": str(config.source_path),
            "effective_configuration": config.effective(),
            "input": {
                "pair_zarr": str(config.pair_zarr),
                "consolidated_metadata_sha256": input_metadata_sha256,
                "pair_schema_version": input_attrs.get("schema_version"),
                "pair_algorithm_version": input_attrs.get("algorithm_version"),
                "pair_coordinate_method": input_attrs.get(
                    "canonical_coordinate_method"
                ),
                "pair_maximum_distance_m": input_attrs.get("maximum_distance_m"),
                "possible_pair_count": input_attrs.get("possible_pair_count"),
                "overlapping_pair_count": input_attrs.get("overlapping_pair_count"),
                "selected_pair_count": input_attrs.get("selected_pair_count"),
            },
            "analysis": {
                "thresholds_km": thresholds.tolist(),
                "shell_count": len(thresholds) - 1,
                "selected_same_array_pair_count": int(inventory.same_array.sum()),
                "analyzed_same_cluster_pair_count": int(inventory.same_cluster.sum()),
                "ignored_between_cluster_pair_count": int(
                    (inventory.same_array & ~inventory.same_cluster).sum()
                ),
                "ignored_cross_array_pair_count": int((~inventory.same_array).sum()),
                "pair_scale_record_count": len(passages),
                "uncertainty": {
                    "column": "fsle_standard_error_day_inverse",
                    "method": "duration-weighted overshoot standard error",
                    "minimum_reached_passages": 2,
                    "interval_shown": "fsle plus or minus one standard error",
                    "dependency_caveat": (
                        "Pairs sharing platforms are correlated, so this nominal "
                        "standard error is not a confidence interval."
                    ),
                },
                "cluster_inventory": _cluster_inventory(inventory),
                "arrays": array_manifest,
            },
            "products": {
                str(path.relative_to(temporary)): {
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in product_paths
            },
        }
        manifest_path = temporary / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if _input_metadata_hash(config.pair_zarr) != input_metadata_sha256:
            raise ValueError("Input pair Zarr metadata changed during FSLE processing")
        if len(pd.read_csv(spectrum_path)) != len(spectrum):
            raise RuntimeError("FSLE spectrum CSV failed readback validation")
        if len(pd.read_parquet(passages_path)) != len(passages):
            raise RuntimeError("FSLE passage Parquet failed readback validation")
        if any(
            not path.is_file() or path.stat().st_size == 0 for path in product_paths
        ):
            raise RuntimeError("One or more FSLE products are missing or empty")
        json.loads(manifest_path.read_text(encoding="utf-8"))
        relative_figures = tuple(path.relative_to(temporary) for path in figure_paths)
        _publish(temporary, target, token=token, overwrite=overwrite)
        return FSLEWorkflowResult(
            output_directory=target,
            spectrum_path=target / SPECTRUM_NAME,
            passages_path=target / PASSAGES_NAME,
            manifest_path=target / MANIFEST_NAME,
            figure_paths=tuple(target / path for path in relative_figures),
            array_count=len(array_ids),
            analyzed_pair_count=int(inventory.same_cluster.sum()),
            passage_count=len(passages),
        )
    finally:
        if temporary.exists():
            _remove_generated(temporary, target.parent)


def run_fsle_workflow(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> FSLEWorkflowResult:
    """Calculate and atomically publish FSLE spectra from selected pairs."""
    config = load_fsle_config(config_path)
    report = progress or (lambda message: None)
    if not config.pair_zarr.is_dir():
        raise ValueError(f"Input pair Zarr does not exist: {config.pair_zarr}")
    if config.output_directory.exists() and not overwrite:
        raise FileExistsError(
            f"FSLE output already exists: {config.output_directory}; use --overwrite"
        )
    metadata_sha256 = _input_metadata_hash(config.pair_zarr)
    thresholds = build_scale_thresholds(
        config.minimum_scale_km,
        config.maximum_scale_km,
        config.rho,
        config.anchor_scale_km,
    )
    report(
        f"Opening pair product with {len(thresholds) - 1} FSLE shells; "
        f"coordinates={config.coordinate_method!r}"
    )
    with xr.open_zarr(config.pair_zarr, consolidated=True, chunks=None) as dataset:
        _validate_dataset(dataset, config)
        inventory = _pair_inventory(dataset)
        _validate_inventory(inventory, dataset.sizes["obs"])
        report(
            f"Filtering {len(inventory):,} selected pairs to "
            f"{int(inventory.same_cluster.sum()):,} same-array/same-cluster pairs"
        )
        passages = _collect_passages(
            dataset,
            inventory,
            config,
            thresholds,
            report,
        )
        input_attrs = dict(dataset.attrs)
    array_ids = sorted(
        inventory.loc[inventory.same_array, "array_id_1"].astype(int).unique()
    )
    spectrum = aggregate_fsle_spectra(
        passages,
        thresholds,
        array_ids=array_ids,
        minimum_reached_pairs=config.minimum_reached_pairs_per_scale,
    )
    return _write_bundle(
        config,
        inventory,
        passages,
        spectrum,
        thresholds,
        input_attrs,
        metadata_sha256,
        overwrite=overwrite,
    )
