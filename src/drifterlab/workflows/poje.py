"""Provenance-linked reproduction of trajectory-only Poje et al. analyses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
import shutil
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
import yaml

from drifterlab import __version__
from drifterlab.poje import (
    IncrementDistributionScale,
    SeparationMemoryGroup,
    calculate_figure4_dispersion,
    calculate_figure5_self_similarity,
    calculate_figure6_increment_distributions,
    calculate_figure7_structure_functions,
)
from drifterlab.poje_figures import (
    plot_figure4_dispersion,
    plot_figure5_self_similarity,
    plot_figure6_increment_distributions,
    plot_figure7_structure_functions,
)


POJE_SCHEMA_VERSION = "1.0"
POJE_ALGORITHM_VERSION = "poje-trajectory-figure04-07-v6"


@dataclass(frozen=True)
class Figure4Config:
    initial_pair_separation_maximum_m: float
    absolute_plot_end_days: float
    ballistic_fit_end_hours: float
    richardson_fit_start_days: float
    richardson_fit_end_days: float
    bootstrap_replicates: int
    bootstrap_confidence_level: float
    bootstrap_random_seed: int
    dpi: int


@dataclass(frozen=True)
class Figure5Config:
    memory_groups: tuple[SeparationMemoryGroup, ...]
    similarity_bin_count: int
    correlation_time_fraction: float
    kolmogorov_constant: float
    dpi: int


@dataclass(frozen=True)
class Figure6Config:
    scales: tuple[IncrementDistributionScale, ...]
    maximum_center_relative_difference: float
    normalized_increment_limit: float
    histogram_bin_count: int
    gaussian_core_maximum_absolute_normalized_increment: float
    dpi: int


@dataclass(frozen=True)
class Figure7Config:
    fit_minimum_separation_m: float
    fit_maximum_separation_m: float
    panel_a_maximum_separation_m: float
    panel_b_maximum_separation_m: float
    rossby_reference_values: tuple[float, ...]
    epsilon_linear_threshold_m2_s3: float
    dpi: int


@dataclass(frozen=True)
class PojeConfig:
    source_path: Path
    epsilon_run: Path
    arrays: str | tuple[int, ...]
    window: str
    figure4: Figure4Config
    figure5: Figure5Config
    figure6: Figure6Config
    figure7: Figure7Config
    output_root: Path

    def effective(self) -> dict[str, Any]:
        return {
            "input": {"epsilon_run": str(self.epsilon_run)},
            "selection": {
                "arrays": self.arrays if isinstance(self.arrays, str) else list(self.arrays),
            },
            "time": {"window": self.window},
            "figure4": asdict(self.figure4),
            "figure5": asdict(self.figure5),
            "figure6": asdict(self.figure6),
            "figure7": asdict(self.figure7),
            "output": {"root": str(self.output_root)},
        }


@dataclass(frozen=True)
class PojeResult:
    run_directory: Path
    manifest_path: Path
    array_count: int
    figure_count: int
    run_id: str

    def format(self) -> str:
        return "\n".join((
            f"Poje trajectory analysis: {self.run_directory}",
            f"Arrays: {self.array_count}",
            f"Figures: {self.figure_count}",
            f"Run ID: {self.run_id}",
            f"Manifest: {self.manifest_path}",
        ))


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _positive(value: Any, name: str) -> float:
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


def _probability(value: Any, name: str) -> float:
    result = _positive(value, name)
    if result >= 1:
        raise ValueError(f"{name} must be smaller than one")
    return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_poje_config(path: str | Path) -> PojeConfig:
    source_path = Path(path).expanduser().resolve()
    values = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    root = _mapping(
        values,
        {
            "input", "selection", "time", "figure4", "figure5", "figure6",
            "figure7", "output",
        },
        "configuration",
    )
    input_section = _mapping(root.get("input", {}), {"epsilon_run"}, "input")
    selection = _mapping(root.get("selection", {}), {"arrays"}, "selection")
    time = _mapping(root.get("time", {}), {"window"}, "time")
    figure4 = _mapping(
        root.get("figure4", {}),
        {
            "initial_pair_separation_maximum_m", "absolute_plot_end_days",
            "ballistic_fit_end_hours", "richardson_fit_start_days",
            "richardson_fit_end_days", "bootstrap", "figure",
        },
        "figure4",
    )
    bootstrap = _mapping(
        figure4.get("bootstrap", {}),
        {"replicates", "confidence_level", "random_seed", "unit"},
        "figure4.bootstrap",
    )
    figure = _mapping(figure4.get("figure", {}), {"dpi"}, "figure4.figure")
    figure5 = _mapping(
        root.get("figure5", {}), {"separation_memory", "lagrangian", "figure"},
        "figure5",
    )
    separation_memory = _mapping(
        figure5.get("separation_memory", {}),
        {"groups", "similarity_bin_count", "correlation_time_fraction"},
        "figure5.separation_memory",
    )
    lagrangian = _mapping(
        figure5.get("lagrangian", {}), {"kolmogorov_constant"},
        "figure5.lagrangian",
    )
    figure5_figure = _mapping(
        figure5.get("figure", {}), {"dpi"}, "figure5.figure",
    )
    figure6 = _mapping(
        root.get("figure6", {}), {"distributions", "figure"}, "figure6",
    )
    distributions = _mapping(
        figure6.get("distributions", {}),
        {
            "scales", "maximum_center_relative_difference",
            "normalized_increment_limit", "histogram_bin_count",
            "gaussian_core_maximum_absolute_normalized_increment",
        },
        "figure6.distributions",
    )
    figure6_figure = _mapping(
        figure6.get("figure", {}), {"dpi"}, "figure6.figure",
    )
    figure7 = _mapping(
        root.get("figure7", {}), {"structure_functions", "figure"}, "figure7",
    )
    structure_functions = _mapping(
        figure7.get("structure_functions", {}),
        {
            "fit_minimum_separation_m", "fit_maximum_separation_m",
            "panel_a_maximum_separation_m", "panel_b_maximum_separation_m",
            "rossby_reference_values",
        },
        "figure7.structure_functions",
    )
    figure7_figure = _mapping(
        figure7.get("figure", {}), {"epsilon_linear_threshold_m2_s3", "dpi"},
        "figure7.figure",
    )
    output = _mapping(root.get("output", {}), {"root"}, "output")

    def resolved(value: Any, name: str) -> Path:
        candidate = Path(_nonempty(value, name)).expanduser()
        return (source_path.parent / candidate).resolve()

    arrays_raw = selection.get("arrays", "all")
    if arrays_raw == "all":
        arrays: str | tuple[int, ...] = "all"
    elif isinstance(arrays_raw, list) and arrays_raw:
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in arrays_raw):
            raise ValueError("selection.arrays must contain positive integers")
        arrays = tuple(arrays_raw)
        if len(set(arrays)) != len(arrays):
            raise ValueError("selection.arrays contains duplicates")
    else:
        raise ValueError("selection.arrays must be 'all' or a nonempty list")
    window = _nonempty(time.get("window", "maximum_configured"), "time.window")
    if window != "maximum_configured":
        raise ValueError("time.window currently supports maximum_configured only")
    unit = _nonempty(
        bootstrap.get("unit", "fixed_initial_pair_identity_synchronized_across_times"),
        "figure4.bootstrap.unit",
    )
    if unit != "fixed_initial_pair_identity_synchronized_across_times":
        raise ValueError(
            "figure4.bootstrap.unit currently supports only "
            "fixed_initial_pair_identity_synchronized_across_times"
        )
    random_seed = bootstrap.get("random_seed", 41723)
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("figure4.bootstrap.random_seed must be a nonnegative integer")
    epsilon_run = resolved(input_section.get("epsilon_run"), "input.epsilon_run")
    output_root = resolved(output.get("root"), "output.root")
    if epsilon_run == output_root or epsilon_run in output_root.parents:
        raise ValueError("Poje output root must be outside the source epsilon run")
    groups_raw = separation_memory.get("groups", [
        {"id": "days_1_2", "start_day": 1, "end_day": 2},
        {"id": "days_5_8", "start_day": 5, "end_day": 8},
        {"id": "days_9_12", "start_day": 9, "end_day": 12},
    ])
    if not isinstance(groups_raw, list) or not groups_raw:
        raise ValueError("figure5.separation_memory.groups must be a nonempty list")
    groups: list[SeparationMemoryGroup] = []
    for index, raw_group in enumerate(groups_raw):
        group = _mapping(raw_group, {"id", "start_day", "end_day"}, f"figure5 group {index}")
        groups.append(SeparationMemoryGroup(
            identifier=_nonempty(group.get("id"), f"figure5 group {index}.id"),
            start_day=_positive(group.get("start_day"), f"figure5 group {index}.start_day"),
            end_day=_positive(group.get("end_day"), f"figure5 group {index}.end_day"),
        ))
    if len({group.identifier for group in groups}) != len(groups):
        raise ValueError("figure5 separation-memory group identifiers must be unique")
    for group in groups:
        if group.start_day >= group.end_day:
            raise ValueError(f"Figure 5 group must have start_day < end_day: {group.identifier}")
    scales_raw = distributions.get("scales", [
        {"id": "small_0250m", "panel_group": "small", "target_center_m": 250},
        {"id": "small_0500m", "panel_group": "small", "target_center_m": 500},
        {"id": "small_1000m", "panel_group": "small", "target_center_m": 1000},
        {"id": "large_5000m", "panel_group": "large", "target_center_m": 5000},
        {"id": "large_7000m", "panel_group": "large", "target_center_m": 7000},
        {"id": "large_10000m", "panel_group": "large", "target_center_m": 10000},
    ])
    if not isinstance(scales_raw, list) or not scales_raw:
        raise ValueError("figure6.distributions.scales must be a nonempty list")
    scales: list[IncrementDistributionScale] = []
    for index, raw_scale in enumerate(scales_raw):
        scale = _mapping(
            raw_scale, {"id", "panel_group", "target_center_m"},
            f"figure6 scale {index}",
        )
        panel_group = _nonempty(
            scale.get("panel_group"), f"figure6 scale {index}.panel_group",
        )
        if panel_group not in {"small", "large"}:
            raise ValueError("Figure 6 panel_group must be small or large")
        scales.append(IncrementDistributionScale(
            identifier=_nonempty(scale.get("id"), f"figure6 scale {index}.id"),
            panel_group=panel_group,
            target_center_m=_positive(
                scale.get("target_center_m"), f"figure6 scale {index}.target_center_m",
            ),
        ))
    if len({scale.identifier for scale in scales}) != len(scales):
        raise ValueError("Figure 6 scale identifiers must be unique")
    group_counts = {
        group: sum(scale.panel_group == group for scale in scales)
        for group in ("small", "large")
    }
    if group_counts != {"small": 3, "large": 3}:
        raise ValueError("Figure 6 requires exactly three small and three large scales")
    rossby_values_raw = structure_functions.get("rossby_reference_values", [1.0, 0.1])
    if not isinstance(rossby_values_raw, list) or not rossby_values_raw:
        raise ValueError("figure7 rossby_reference_values must be a nonempty list")
    rossby_values = tuple(
        _positive(value, f"figure7 rossby_reference_values[{index}]")
        for index, value in enumerate(rossby_values_raw)
    )
    if len(set(rossby_values)) != len(rossby_values):
        raise ValueError("figure7 rossby_reference_values contains duplicates")
    return PojeConfig(
        source_path=source_path,
        epsilon_run=epsilon_run,
        arrays=arrays,
        window=window,
        figure4=Figure4Config(
            initial_pair_separation_maximum_m=_positive(
                figure4.get("initial_pair_separation_maximum_m", 300),
                "figure4.initial_pair_separation_maximum_m",
            ),
            absolute_plot_end_days=_positive(
                figure4.get("absolute_plot_end_days", 8),
                "figure4.absolute_plot_end_days",
            ),
            ballistic_fit_end_hours=_positive(
                figure4.get("ballistic_fit_end_hours", 12),
                "figure4.ballistic_fit_end_hours",
            ),
            richardson_fit_start_days=_positive(
                figure4.get("richardson_fit_start_days", 2),
                "figure4.richardson_fit_start_days",
            ),
            richardson_fit_end_days=_positive(
                figure4.get("richardson_fit_end_days", 8),
                "figure4.richardson_fit_end_days",
            ),
            bootstrap_replicates=_positive_integer(
                bootstrap.get("replicates", 10000), "figure4.bootstrap.replicates",
            ),
            bootstrap_confidence_level=_probability(
                bootstrap.get("confidence_level", 0.95),
                "figure4.bootstrap.confidence_level",
            ),
            bootstrap_random_seed=random_seed,
            dpi=_positive_integer(figure.get("dpi", 160), "figure4.figure.dpi"),
        ),
        figure5=Figure5Config(
            memory_groups=tuple(groups),
            similarity_bin_count=_positive_integer(
                separation_memory.get("similarity_bin_count", 100),
                "figure5.separation_memory.similarity_bin_count",
            ),
            correlation_time_fraction=_probability(
                separation_memory.get("correlation_time_fraction", 1 / 3),
                "figure5.separation_memory.correlation_time_fraction",
            ),
            kolmogorov_constant=_positive(
                lagrangian.get("kolmogorov_constant", 6.5),
                "figure5.lagrangian.kolmogorov_constant",
            ),
            dpi=_positive_integer(
                figure5_figure.get("dpi", 160), "figure5.figure.dpi",
            ),
        ),
        figure6=Figure6Config(
            scales=tuple(scales),
            maximum_center_relative_difference=_probability(
                distributions.get("maximum_center_relative_difference", 0.1),
                "figure6.distributions.maximum_center_relative_difference",
            ),
            normalized_increment_limit=_positive(
                distributions.get("normalized_increment_limit", 5),
                "figure6.distributions.normalized_increment_limit",
            ),
            histogram_bin_count=_positive_integer(
                distributions.get("histogram_bin_count", 100),
                "figure6.distributions.histogram_bin_count",
            ),
            gaussian_core_maximum_absolute_normalized_increment=_positive(
                distributions.get(
                    "gaussian_core_maximum_absolute_normalized_increment", 1.5,
                ),
                "figure6.distributions.gaussian_core_maximum_absolute_normalized_increment",
            ),
            dpi=_positive_integer(
                figure6_figure.get("dpi", 160), "figure6.figure.dpi",
            ),
        ),
        figure7=Figure7Config(
            fit_minimum_separation_m=_positive(
                structure_functions.get("fit_minimum_separation_m", 100),
                "figure7.structure_functions.fit_minimum_separation_m",
            ),
            fit_maximum_separation_m=_positive(
                structure_functions.get("fit_maximum_separation_m", 10000),
                "figure7.structure_functions.fit_maximum_separation_m",
            ),
            panel_a_maximum_separation_m=_positive(
                structure_functions.get("panel_a_maximum_separation_m", 50000),
                "figure7.structure_functions.panel_a_maximum_separation_m",
            ),
            panel_b_maximum_separation_m=_positive(
                structure_functions.get("panel_b_maximum_separation_m", 10000),
                "figure7.structure_functions.panel_b_maximum_separation_m",
            ),
            rossby_reference_values=rossby_values,
            epsilon_linear_threshold_m2_s3=_positive(
                figure7_figure.get("epsilon_linear_threshold_m2_s3", 1e-8),
                "figure7.figure.epsilon_linear_threshold_m2_s3",
            ),
            dpi=_positive_integer(
                figure7_figure.get("dpi", 160), "figure7.figure.dpi",
            ),
        ),
        output_root=output_root,
    )


def _source_window(resolved: dict[str, Any]) -> tuple[str, float]:
    windows = resolved.get("analysis", {}).get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Source resolved config has no analysis windows")
    maximum = max(windows, key=lambda item: float(item["end_hour"]))
    if float(maximum.get("start_hour", 0)) != 0:
        raise ValueError("Poje dispersion requires a maximum source window starting at zero")
    return _nonempty(maximum.get("id"), "source window id"), _positive(
        maximum.get("end_hour"), "source window end_hour",
    )


def _source_separation_bins(resolved: dict[str, Any]) -> tuple[list[float], list[float]]:
    section = resolved.get("separation_bins", {})
    edges = section.get("resolved_edges_m")
    centers = section.get("resolved_nominal_centers_m")
    if not isinstance(edges, list) or not isinstance(centers, list):
        raise ValueError("Source resolved config lacks resolved separation bins")
    if len(edges) != len(centers) + 1:
        raise ValueError("Source resolved separation edges and centers are inconsistent")
    return edges, centers


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
        raise FileExistsError(f"Poje output already exists: {target}; use --overwrite")
    backup = target.with_name(f".{target.name}.{token}.backup")
    target.rename(backup)
    try:
        temporary.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    _remove_generated(backup, target.parent)


def run_poje_workflow(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PojeResult:
    """Create Figure 4 through 7 products from a completed all-array epsilon run."""
    config = load_poje_config(config_path)
    report = progress or (lambda message: None)
    source_manifest_path = config.epsilon_run / "manifest.json"
    source_resolved_path = config.epsilon_run / "config_resolved.yaml"
    if not source_manifest_path.is_file() or not source_resolved_path.is_file():
        raise ValueError("input.epsilon_run is not a completed epsilon product")
    source_manifest_sha256 = _file_sha256(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_resolved = yaml.safe_load(source_resolved_path.read_text(encoding="utf-8"))
    source_run_id = _nonempty(source_manifest.get("run_id"), "source manifest run_id")
    if source_resolved.get("selection", {}).get("pair_scope") != "all_array":
        raise ValueError("Poje dispersion source must use selection.pair_scope=all_array")
    window_id, window_end_hours = _source_window(source_resolved)
    source_edges_m, source_centers_m = _source_separation_bins(source_resolved)
    if config.figure4.absolute_plot_end_days * 24 > window_end_hours:
        raise ValueError("Figure 4 absolute plot interval exceeds the source window")
    if max(group.end_day for group in config.figure5.memory_groups) * 24 > window_end_hours:
        raise ValueError("Figure 5 separation-memory groups exceed the source window")

    identity_payload = {
        "schema_version": POJE_SCHEMA_VERSION,
        "algorithm_version": POJE_ALGORITHM_VERSION,
        "drifterlab_version": __version__,
        "source_manifest_sha256": source_manifest_sha256,
        "configuration": config.effective(),
    }
    digest = sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    run_id = f"poje-v1-{digest}"
    target_parent = config.output_root / source_run_id
    target = target_parent / run_id
    if target.exists() and not overwrite:
        raise FileExistsError(f"Poje output already exists: {target}; use --overwrite")
    target_parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = target.with_name(f".{target.name}.{token}.tmp")
    temporary.mkdir()

    products: list[Path] = []
    summaries: list[dict[str, object]] = []
    source_files: dict[str, dict[str, object]] = {}
    try:
        array_directories = sorted(config.epsilon_run.glob("array_[0-9][0-9][0-9]"))
        available = {int(path.name.removeprefix("array_")): path for path in array_directories}
        selected_arrays = tuple(sorted(available)) if config.arrays == "all" else config.arrays
        missing_arrays = sorted(set(selected_arrays) - set(available))
        if missing_arrays:
            raise ValueError(f"Selected arrays are absent from source run: {missing_arrays}")

        for array_id in selected_arrays:
            source_array = available[array_id]
            clusters_path = source_array / "clusters.csv"
            diagnostics_path = source_array / "data" / "platform_diagnostics.parquet"
            pair_observations_path = source_array / "data" / "pair_observations.parquet"
            epsilon_path = source_array / "epsilon_by_scale.csv"
            if (
                not clusters_path.is_file()
                or not diagnostics_path.is_file()
                or not pair_observations_path.is_file()
                or not epsilon_path.is_file()
            ):
                raise ValueError(f"Array {array_id} lacks required Poje source tables")
            for path in (
                clusters_path, diagnostics_path, pair_observations_path, epsilon_path,
            ):
                relative_native = str(path.relative_to(config.epsilon_run))
                relative = relative_native.replace("\\", "/")
                actual_hash = _file_sha256(path)
                source_products = source_manifest.get("products", {})
                expected = source_products.get(
                    relative, source_products.get(relative_native, {}),
                ).get("sha256")
                if expected is not None and actual_hash != expected:
                    raise ValueError(f"Source epsilon product hash mismatch: {relative}")
                source_files[relative] = {
                    "sha256": actual_hash, "bytes": path.stat().st_size,
                }
            diagnostics = pd.read_parquet(diagnostics_path)
            figure4_result = calculate_figure4_dispersion(
                diagnostics,
                array_id=array_id,
                window_id=window_id,
                window_end_hours=window_end_hours,
                absolute_plot_end_days=config.figure4.absolute_plot_end_days,
                initial_pair_separation_maximum_m=(
                    config.figure4.initial_pair_separation_maximum_m
                ),
                ballistic_fit_end_hours=config.figure4.ballistic_fit_end_hours,
                richardson_fit_start_days=config.figure4.richardson_fit_start_days,
                richardson_fit_end_days=config.figure4.richardson_fit_end_days,
                bootstrap_replicates=config.figure4.bootstrap_replicates,
                bootstrap_confidence_level=config.figure4.bootstrap_confidence_level,
                bootstrap_random_seed=config.figure4.bootstrap_random_seed + array_id,
            )
            figure5_result = calculate_figure5_self_similarity(
                diagnostics,
                array_id=array_id,
                window_id=window_id,
                window_end_hours=window_end_hours,
                memory_groups=config.figure5.memory_groups,
                similarity_bin_count=config.figure5.similarity_bin_count,
                kolmogorov_constant=config.figure5.kolmogorov_constant,
            )
            pair_observations = pd.read_parquet(
                pair_observations_path,
                columns=[
                    "array_id", "pair_id", "platform_id_1", "platform_id_2",
                    "time_utc", "cluster_age_hours", "separation_m", "delta_u_l_m_s",
                    "delta_u_l_delta_u_t_squared_m3_s3",
                ],
            )
            figure6_result = calculate_figure6_increment_distributions(
                pair_observations,
                array_id=array_id,
                window_id=window_id,
                window_end_hours=window_end_hours,
                source_edges_m=source_edges_m,
                source_centers_m=source_centers_m,
                scales=config.figure6.scales,
                maximum_center_relative_difference=(
                    config.figure6.maximum_center_relative_difference
                ),
                normalized_increment_limit=config.figure6.normalized_increment_limit,
                histogram_bin_count=config.figure6.histogram_bin_count,
                gaussian_core_maximum_absolute_normalized_increment=(
                    config.figure6.gaussian_core_maximum_absolute_normalized_increment
                ),
            )
            epsilon_by_scale = pd.read_csv(epsilon_path)
            figure7_result = calculate_figure7_structure_functions(
                pair_observations,
                epsilon_by_scale,
                array_id=array_id,
                window_id=window_id,
                window_end_hours=window_end_hours,
                source_edges_m=source_edges_m,
                source_centers_m=source_centers_m,
                fit_minimum_separation_m=config.figure7.fit_minimum_separation_m,
                fit_maximum_separation_m=config.figure7.fit_maximum_separation_m,
                panel_a_maximum_separation_m=(
                    config.figure7.panel_a_maximum_separation_m
                ),
                panel_b_maximum_separation_m=(
                    config.figure7.panel_b_maximum_separation_m
                ),
            )
            output_array = temporary / f"array_{array_id:03d}"
            dispersion_path = output_array / "dispersion.csv"
            pairs_path = output_array / "pairs.csv"
            observations_path = output_array / "data" / "pair_dispersion.parquet"
            figure_path = output_array / "figures" / "figure_04_dispersion.png"
            memory_path = output_array / "separation_memory.csv"
            lagrangian_path = output_array / "lagrangian_structure.csv"
            memory_samples_path = output_array / "data" / "separation_memory_samples.parquet"
            figure5_path = output_array / "figures" / "figure_05_self_similarity.png"
            distributions_path = output_array / "increment_distributions.csv"
            increment_statistics_path = output_array / "increment_statistics.csv"
            increment_observations_path = (
                output_array / "data" / "figure_06_pair_observations.parquet"
            )
            figure6_path = output_array / "figures" / "figure_06_increment_distributions.png"
            structure_functions_path = output_array / "structure_functions.csv"
            figure7_path = output_array / "figures" / "figure_07_structure_functions.png"
            observations_path.parent.mkdir(parents=True, exist_ok=True)
            figure4_result.timeseries.to_csv(dispersion_path, index=False)
            figure4_result.pair_inventory.to_csv(pairs_path, index=False)
            figure4_result.pair_observations.to_parquet(observations_path, index=False)
            figure5_result.separation_memory.to_csv(memory_path, index=False)
            figure5_result.lagrangian_structure.to_csv(lagrangian_path, index=False)
            figure5_result.separation_memory_samples.to_parquet(
                memory_samples_path, index=False,
            )
            figure6_result.distributions.to_csv(distributions_path, index=False)
            figure6_result.statistics.to_csv(increment_statistics_path, index=False)
            figure6_result.observations.to_parquet(increment_observations_path, index=False)
            figure7_result.structure_functions.to_csv(structure_functions_path, index=False)
            combined_summary = {
                **figure4_result.summary,
                **figure5_result.summary,
                **figure6_result.summary,
                **figure7_result.summary,
            }
            plot_figure4_dispersion(
                figure4_result.timeseries, figure4_result.summary,
                output_path=figure_path, dpi=config.figure4.dpi,
            )
            plot_figure5_self_similarity(
                figure5_result.separation_memory,
                figure5_result.lagrangian_structure,
                combined_summary,
                correlation_time_fraction=config.figure5.correlation_time_fraction,
                output_path=figure5_path,
                dpi=config.figure5.dpi,
            )
            plot_figure6_increment_distributions(
                figure6_result.distributions,
                figure6_result.statistics,
                combined_summary,
                normalized_increment_limit=config.figure6.normalized_increment_limit,
                output_path=figure6_path,
                dpi=config.figure6.dpi,
            )
            plot_figure7_structure_functions(
                figure7_result.structure_functions,
                combined_summary,
                rossby_reference_values=config.figure7.rossby_reference_values,
                epsilon_linear_threshold_m2_s3=(
                    config.figure7.epsilon_linear_threshold_m2_s3
                ),
                output_path=figure7_path,
                dpi=config.figure7.dpi,
            )
            products.extend((
                dispersion_path, pairs_path, observations_path, figure_path,
                memory_path, lagrangian_path, memory_samples_path, figure5_path,
                distributions_path, increment_statistics_path,
                increment_observations_path, figure6_path,
                structure_functions_path, figure7_path,
            ))
            summaries.append(combined_summary)
            report(
                f"Array {array_id}: Figures 4-7 rendered from "
                f"{figure4_result.summary['initial_pair_count']} initial Figure 4 pairs "
                f"and {figure5_result.summary['figure5_complete_pair_count']} complete pairs"
            )

        summary_path = temporary / "summary.csv"
        summary = pd.DataFrame(summaries).sort_values("array_id")
        summary.to_csv(summary_path, index=False)
        products.append(summary_path)
        resolved_path = temporary / "config_resolved.yaml"
        resolved = {
            **config.effective(),
            "run_id": run_id,
            "schema_version": POJE_SCHEMA_VERSION,
            "algorithm_version": POJE_ALGORITHM_VERSION,
            "source_run_id": source_run_id,
            "source_manifest_sha256": source_manifest_sha256,
            "resolved_window_id": window_id,
            "resolved_window_end_hours": window_end_hours,
        }
        resolved_path.write_text(
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8",
        )
        products.append(resolved_path)
        product_manifest = {
            str(path.relative_to(temporary)).replace("\\", "/"): {
                "sha256": _file_sha256(path), "bytes": path.stat().st_size,
            }
            for path in products
        }
        manifest = {
            "schema_version": POJE_SCHEMA_VERSION,
            "algorithm_version": POJE_ALGORITHM_VERSION,
            "drifterlab_version": __version__,
            "processing_time_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "configuration_path": str(config.source_path),
            "effective_configuration": config.effective(),
            "source": {
                "epsilon_run": str(config.epsilon_run),
                "run_id": source_run_id,
                "manifest_sha256": source_manifest_sha256,
                "candidate_input_allowed": source_resolved.get("input", {}).get(
                    "allow_candidate_input"
                ),
                "files": source_files,
            },
            "analysis": {
                "implemented_figures": [
                    "figure_04", "figure_05", "figure_06", "figure_07",
                ],
                "array_count": len(summary),
                "figure_count": len(summary) * 4,
                "initial_pair_count": int(summary.initial_pair_count.sum()),
                "window_id": window_id,
                "window_end_hours": window_end_hours,
            },
            "products": product_manifest,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        json.loads(manifest_path.read_text(encoding="utf-8"))
        _publish(temporary, target, token=token, overwrite=overwrite)
        return PojeResult(
            run_directory=target,
            manifest_path=target / "manifest.json",
            array_count=len(summary),
            figure_count=len(summary) * 4,
            run_id=run_id,
        )
    finally:
        if temporary.exists():
            _remove_generated(temporary, target.parent)
