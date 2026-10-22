"""Atomic orchestration for the unified v2 scientific analysis."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np
import yaml

from drifterlab import __version__
from drifterlab.postprocessing import run_postprocessing
from drifterlab.trajectories.product_store import load_trajectory_product
from drifterlab.workflows.epsilon import load_epsilon_config, run_epsilon_workflow
from drifterlab.workflows.epsilon_diagnostics import run_epsilon_diagnostics

from .config import ANALYSIS_SCHEMA_VERSION, AnalysisConfig, load_analysis_config


ANALYSIS_ALGORITHM_VERSION = "unified-grouped-products-v1"
MANIFEST_NAME = "manifest.json"
RESOLVED_CONFIG_NAME = "config_resolved.yml"


@dataclass(frozen=True)
class AnalysisResult:
    output_directory: Path
    manifest_path: Path
    array_count: int
    cluster_count: int
    product_count: int
    run_id: str

    def format(self) -> str:
        return "\n".join((
            f"Unified analysis complete: {self.array_count} array(s), "
            f"{self.cluster_count} cluster(s)",
            f"Products: {self.product_count}",
            f"Run ID: {self.run_id}",
            f"Output: {self.output_directory}",
            f"Manifest: {self.manifest_path}",
        ))


def run_analysis(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> AnalysisResult:
    """Run every enabled module and atomically publish one numbered product tree."""

    config = load_analysis_config(config_path)
    report = progress or (lambda message: None)
    report(
        f"Opening trajectory product {config.trajectory_product!r}: "
        f"{config.trajectory_zarr}"
    )
    dataset, product_info = load_trajectory_product(
        config.trajectory_zarr, config.trajectory_product,
    )
    status = product_info.product_status
    if _is_provisional(status) and not config.allow_provisional:
        raise ValueError(
            f"Trajectory product status is {status!r}; set input.allow_provisional: true "
            "for an explicitly provisional analysis"
        )

    arrays_available = tuple(sorted(set(dataset.array_id.values.astype(int).tolist())))
    arrays = _resolve_arrays(config, arrays_available)
    clusters_available = tuple(dict.fromkeys(
        dataset.cluster_id.values.astype(str)[
            np.isin(dataset.array_id.values.astype(int), np.asarray(arrays))
        ].tolist()
    ))
    clusters = _resolve_clusters(config, clusters_available)
    selected_platform = np.isin(dataset.array_id.values.astype(int), np.asarray(arrays))
    if isinstance(clusters, tuple):
        selected_platform &= np.isin(
            dataset.cluster_id.values.astype(str), np.asarray(clusters),
        )
    if not selected_platform.any():
        raise ValueError("The configured array and cluster selection contains no platforms")

    effective_span = _effective_span_minutes(config, product_info.cadence_minutes)
    if any(block.get("enabled", False) for block in (
        config.trajectory_plots, config.trajectory_statistics,
        config.epsilon, config.epsilon_diagnostics,
    )):
        _validate_velocity_grid(config, product_info.cadence_minutes, effective_span)
    metadata_sha256 = _file_sha256(config.trajectory_zarr / ".zmetadata")
    run_id = _run_id(config, metadata_sha256, product_info.product_attributes)
    target = config.output_root
    if target.exists() and not overwrite:
        raise FileExistsError(f"Analysis output already exists: {target}; use --overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = target.with_name(f".{target.name}.{token}.tmp")
    work = temporary / "_work"
    temporary.mkdir()
    work.mkdir()

    module_summary: dict[str, Any] = {}
    try:
        if (
            config.trajectory_plots.get("enabled", False)
            or config.trajectory_statistics.get("enabled", False)
        ):
            report("Running trajectory plots and descriptive cluster statistics")
            post_values = _postprocessing_values(
                config, arrays, clusters, work, effective_span,
                product_info.cadence_minutes,
            )
            post_config = _write_yaml(work / "postprocessing.yml", post_values)
            post_result = run_postprocessing(
                post_config, overwrite=False, progress=report,
            )
            _publish_postprocessing_modules(
                post_result.output_directory,
                temporary,
                trajectory_plots=config.trajectory_plots.get("enabled", False),
                trajectory_statistics=config.trajectory_statistics.get("enabled", False),
            )
            module_summary["trajectory_plots"] = {
                "enabled": config.trajectory_plots.get("enabled", False),
                "figure_count": sum(
                    path.suffix.lower() == ".png" for path in post_result.figure_paths
                    if "trajectory_plotting" in path.parts
                ),
                "movie_count": len(post_result.movie_paths),
            }
            module_summary["trajectory_statistics"] = {
                "enabled": config.trajectory_statistics.get("enabled", False),
                "figure_count": sum(
                    "cluster_statistics" in path.parts for path in post_result.figure_paths
                ),
                "table_count": len(post_result.numerical_paths),
            }
        else:
            module_summary["trajectory_plots"] = {"enabled": False}
            module_summary["trajectory_statistics"] = {"enabled": False}

        epsilon_result = None
        if config.epsilon.get("enabled", False):
            report("Running epsilon estimation, uncertainty, distributions, and figures")
            epsilon_values = _epsilon_values(
                config, arrays, clusters, work, effective_span,
                product_info.cadence_minutes,
            )
            epsilon_config = _write_yaml(work / "epsilon.yml", epsilon_values)
            epsilon_result = run_epsilon_workflow(
                epsilon_config, overwrite=False, progress=report,
            )
            spatial_summary = {"enabled": False}
            if config.epsilon.get("spatial_maps", {}).get("enabled", False):
                from .spatial import run_spatial_maps

                spatial_label = (
                    "daily spatial epsilon" if config.epsilon["spatial_maps"].get("daily_maps")
                    else "spatial pairwise-q"
                )
                report(f"Rendering {spatial_label} maps on a common UTC clock")
                spatial_summary = run_spatial_maps(
                    dataset,
                    load_epsilon_config(epsilon_config),
                    config.epsilon["spatial_maps"],
                    epsilon_result.run_directory,
                    progress=report,
                )
            module_summary["epsilon"] = {
                "enabled": True,
                "spatial_maps": spatial_summary,
                "run_id": epsilon_result.run_id,
                "activated_cluster_count": epsilon_result.activated_cluster_count,
                "snapshot_pair_observation_count": (
                    epsilon_result.snapshot_observation_count
                ),
                "evolution_pair_observation_count": (
                    epsilon_result.evolution_observation_count
                ),
            }
        else:
            module_summary["epsilon"] = {"enabled": False}

        if config.epsilon_diagnostics.get("enabled", False):
            if epsilon_result is None:
                raise RuntimeError("Epsilon diagnostics have no epsilon source run")
            report("Rendering epsilon diagnostics")
            diagnostic_values = _diagnostic_values(
                config, arrays, clusters, work, epsilon_result.run_directory,
            )
            diagnostic_config = _write_yaml(work / "epsilon_diagnostics.yml", diagnostic_values)
            diagnostic_result = run_epsilon_diagnostics(
                diagnostic_config, overwrite=False, progress=report,
            )
            _publish_diagnostic_module(diagnostic_result.run_directory, temporary)
            module_summary["epsilon_diagnostics"] = {
                "enabled": True,
                "figure_count": diagnostic_result.figure_count,
                "inventory_row_count": diagnostic_result.cluster_count,
            }
        else:
            module_summary["epsilon_diagnostics"] = {"enabled": False}

        if epsilon_result is not None:
            _publish_epsilon_module(epsilon_result.run_directory, temporary)

        if config.fsle["enabled"]:
            from .fsle import run_fsle_analysis

            report("Running FSLE directly from selected trajectory positions")
            module_summary["fsle"] = run_fsle_analysis(
                dataset.isel(platform=np.flatnonzero(selected_platform)),
                config.fsle, temporary / "05_fsle",
                cadence_minutes=product_info.cadence_minutes, progress=report,
            )
        else:
            module_summary["fsle"] = {"enabled": False}

        resolved = config.effective()
        resolved["run_id"] = run_id
        resolved["resolved_input"] = {
            "product_source_kind": product_info.source_kind,
            "product_cadence_minutes": product_info.cadence_minutes,
            "product_maximum_source_gap_minutes": (
                product_info.maximum_source_gap_minutes
            ),
            "product_status": status,
            "effective_velocity_span_minutes": effective_span,
        }
        _write_yaml(temporary / RESOLVED_CONFIG_NAME, resolved)
        shutil.rmtree(work)

        product_records = _product_records(temporary)
        selected_cluster_values = set(
            dataset.cluster_id.values.astype(str)[selected_platform].tolist()
        )
        manifest = {
            "schema_version": str(ANALYSIS_SCHEMA_VERSION),
            "algorithm_version": ANALYSIS_ALGORITHM_VERSION,
            "drifterlab_version": __version__,
            "processing_time_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "configuration_path": str(config.source_path),
            "input": {
                "trajectory_zarr": str(config.trajectory_zarr),
                "trajectory_product": config.trajectory_product,
                "product_status": status,
                "provisional": _is_provisional(status),
                "source_kind": product_info.source_kind,
                "cadence_minutes": product_info.cadence_minutes,
                "maximum_source_gap_minutes": (
                    product_info.maximum_source_gap_minutes
                ),
                "consolidated_metadata_sha256": metadata_sha256,
                "build_manifest_sha256": product_info.root_attributes.get(
                    "build_manifest_sha256"
                ),
                "array_assignment_sha256": product_info.root_attributes.get(
                    "array_assignment_sha256"
                ),
                "initial_cluster_assignment_sha256": (
                    product_info.root_attributes.get(
                        "initial_cluster_assignment_sha256"
                    )
                ),
            },
            "selection": {
                "arrays": list(arrays),
                "clusters": (
                    "all" if clusters == "all" else list(clusters)
                ),
                "platform_count": int(selected_platform.sum()),
                "cluster_count": len(selected_cluster_values),
            },
            "kinematics": {
                "method": config.kinematics.method,
                "effective_span_minutes": effective_span,
                "maximum_source_gap_minutes": (
                    config.kinematics.maximum_source_gap_minutes
                ),
                "time_assignment": (
                    "earlier_endpoint"
                    if config.kinematics.method == "stored_forward"
                    else "center_timestamp"
                ),
            },
            "modules": module_summary,
            "output_layout": [
                "01_trajectory_plots",
                "02_trajectory_statistics",
                "03_epsilon",
                "04_epsilon_diagnostics",
                "05_fsle",
            ],
            "products": product_records,
        }
        manifest_path = temporary / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        if _file_sha256(config.trajectory_zarr / ".zmetadata") != metadata_sha256:
            raise ValueError("Trajectory Zarr metadata changed during analysis")
        _publish(temporary, target, overwrite=overwrite, token=token)
        return AnalysisResult(
            output_directory=target,
            manifest_path=target / MANIFEST_NAME,
            array_count=len(arrays),
            cluster_count=len(selected_cluster_values),
            product_count=len(product_records),
            run_id=run_id,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _postprocessing_values(
    config: AnalysisConfig,
    arrays: tuple[int, ...],
    clusters: str | tuple[str, ...],
    work: Path,
    effective_span: float,
    cadence_minutes: int,
) -> dict[str, Any]:
    plotting = deepcopy(config.trajectory_plots)
    statistics = deepcopy(config.trajectory_statistics)
    velocity = statistics.setdefault("velocity", {})
    velocity["method"] = (
        "stored_forward"
        if config.kinematics.method == "stored_forward"
        else "centered_difference"
    )
    velocity.setdefault("difference_interval_minutes", effective_span or cadence_minutes)
    velocity["maximum_source_gap_minutes"] = (
        config.kinematics.maximum_source_gap_minutes
    )
    return {
        "input": {
            "trajectories": str(config.trajectory_zarr),
            "trajectory_product": config.trajectory_product,
            "dataset_label": config.dataset_label,
            "coordinate_method": "product",
        },
        "selection": {
            "arrays": list(arrays),
            "clusters": clusters if isinstance(clusters, str) else list(clusters),
        },
        "output": {"directory": str(work / "postprocessing")},
        "arrays": {},
        "trajectory_plotting": plotting,
        "cluster_statistics": statistics,
    }


def _epsilon_values(
    config: AnalysisConfig,
    arrays: tuple[int, ...],
    clusters: str | tuple[str, ...],
    work: Path,
    effective_span: float,
    cadence_minutes: int,
) -> dict[str, Any]:
    epsilon = deepcopy(config.epsilon)
    _check_keys(
        epsilon,
        {
            "enabled", "pair_scope", "array_anchor", "minimum_active_members",
            "sample_cadence_minutes", "sample_phase_reference", "windows",
            "minimum_reliable_separation_m", "separation_bins", "support",
            "uncertainty", "q_distribution", "figures", "spatial_maps",
        },
        "analyses.epsilon",
    )
    pair_scope = epsilon.get("pair_scope", "within_cluster")
    velocity_method = (
        "stored_forward"
        if config.kinematics.method == "stored_forward"
        else "centered_difference"
    )
    velocity = {
        "position_method": "product",
        "velocity_method": velocity_method,
        "maximum_source_gap_minutes": config.kinematics.maximum_source_gap_minutes,
        "require_full_stencil_support": (
            config.kinematics.require_full_stencil_support
        ),
        "speed_mask_m_s": config.kinematics.speed_mask_m_s,
    }
    if velocity_method == "centered_difference":
        velocity["total_span_minutes"] = effective_span

    uncertainty = deepcopy(epsilon.get("uncertainty", {}))
    q_distribution = deepcopy(epsilon.get("q_distribution", {}))
    figures = deepcopy(epsilon.get("figures", {}))
    enabled_flags = [
        bool(section.pop("enabled", True))
        for section in (uncertainty, q_distribution, figures)
    ]
    if len(set(enabled_flags)) != 1:
        raise ValueError(
            "epsilon uncertainty, q_distribution, and figures must currently be "
            "enabled or disabled together"
        )
    return {
        "input": {
            "trajectories": str(config.trajectory_zarr),
            "trajectory_product": config.trajectory_product,
            "dataset_label": config.dataset_label,
            "allow_candidate_input": config.allow_provisional,
        },
        "selection": {
            "arrays": list(arrays),
            "clusters": clusters if isinstance(clusters, str) else list(clusters),
            "pair_scope": pair_scope,
            "array_anchor": epsilon.get("array_anchor"),
        },
        "activation": {
            "minimum_active_members": epsilon.get("minimum_active_members", 2),
            "reference": {**velocity, "speed_mask_m_s": None},
        },
        "analysis": {
            **velocity,
            "sample_cadence_minutes": epsilon.get(
                "sample_cadence_minutes", cadence_minutes,
            ),
            "sample_phase_reference": epsilon.get(
                "sample_phase_reference",
                "array_anchor" if pair_scope == "all_array" else "activation_anchor",
            ),
            "windows": epsilon.get("windows", [
                {"id": "initial_0_6h", "start_hour": 0, "end_hour": 6},
                {"id": "initial_0_12h", "start_hour": 0, "end_hour": 12},
            ]),
            "minimum_reliable_separation_m": epsilon.get(
                "minimum_reliable_separation_m"
            ),
        },
        "separation_bins": epsilon.get("separation_bins", {
            "mode": "geometric",
            "minimum_m": 62.5,
            "maximum_m": 128000,
            "ratio": 2 ** 0.5,
            "anchor_m": 1000,
            "axis_scale": "log",
        }),
        "support": epsilon.get("support", {
            "observations": 12,
            "unique_pairs": 3,
            "platforms": 3,
            "unique_utc": 12,
        }),
        "uncertainty": {"enabled": enabled_flags[0], **uncertainty},
        "q_distribution": {"enabled": enabled_flags[0], **q_distribution},
        "figures": {"enabled": enabled_flags[0], **figures},
        "output": {"root": str(work / "epsilon")},
    }


def _diagnostic_values(
    config: AnalysisConfig,
    arrays: tuple[int, ...],
    clusters: str | tuple[str, ...],
    work: Path,
    epsilon_run: Path,
) -> dict[str, Any]:
    diagnostics = deepcopy(config.epsilon_diagnostics)
    _check_keys(
        diagnostics,
        {
            "enabled", "time", "trajectory_windows", "velocity_availability",
            "pair_relative_motion",
        },
        "analyses.epsilon_diagnostics",
    )
    time = diagnostics.get("time", {})
    return {
        "input": {"epsilon_run": str(epsilon_run)},
        "selection": {
            "arrays": list(arrays),
            "clusters": clusters if isinstance(clusters, str) else list(clusters),
        },
        "time": {
            "window": time.get("window", "maximum_configured"),
            "mark_elapsed_hours": time.get("mark_elapsed_hours", [0, 6, 12]),
        },
        "figures": {
            "trajectory_window": diagnostics.get("trajectory_windows", {"enabled": True}),
            "velocity_availability": diagnostics.get(
                "velocity_availability", {"enabled": True},
            ),
            "pair_relative_motion": diagnostics.get(
                "pair_relative_motion", {"enabled": True},
            ),
        },
        "output": {"root": str(work / "epsilon_diagnostics")},
    }


def _publish_postprocessing_modules(
    source: Path,
    target: Path,
    *,
    trajectory_plots: bool,
    trajectory_statistics: bool,
) -> None:
    for array in sorted(source.glob("array_*")):
        if trajectory_plots:
            source_plots = array / "trajectory_plotting"
            if source_plots.is_dir():
                _move_directory_contents(
                    source_plots,
                    target / "01_trajectory_plots" / array.name,
                )
        if trajectory_statistics:
            source_statistics = array / "cluster_statistics"
            if source_statistics.is_dir():
                destination = target / "02_trajectory_statistics" / array.name
                figures = destination / "figures"
                for item in source_statistics.iterdir():
                    move_target = figures / item.name if item.suffix.lower() == ".png" else destination / item.name
                    move_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(move_target))


def _publish_epsilon_module(source: Path, target: Path) -> None:
    root = target / "03_epsilon"
    for array in sorted(source.glob("array_*")):
        destination = root / array.name
        tables = destination / "tables"
        for item in array.iterdir():
            move_target = tables / item.name if item.suffix.lower() == ".csv" else destination / item.name
            move_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(move_target))


def _publish_diagnostic_module(source: Path, target: Path) -> None:
    root = target / "04_epsilon_diagnostics"
    for array in sorted(source.glob("array_*")):
        _move_directory_contents(array, root / array.name)
    summary = source / "diagnostic_figures.csv"
    if summary.is_file():
        root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(summary), str(root / summary.name))


def _move_directory_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        shutil.move(str(item), str(destination / item.name))


def _resolve_arrays(
    config: AnalysisConfig,
    available: tuple[int, ...],
) -> tuple[int, ...]:
    if config.arrays == "all":
        return available
    missing = sorted(set(config.arrays) - set(available))
    if missing:
        raise ValueError(f"Selected arrays are unavailable: {missing}")
    return config.arrays


def _resolve_clusters(
    config: AnalysisConfig,
    available: tuple[str, ...],
) -> str | tuple[str, ...]:
    if config.clusters == "all":
        return "all"
    missing = sorted(set(config.clusters) - set(available))
    if missing:
        raise ValueError(f"Selected clusters are unavailable: {missing}")
    return config.clusters


def _effective_span_minutes(config: AnalysisConfig, cadence_minutes: int) -> float:
    if config.kinematics.method == "centered_fixed_span":
        assert config.kinematics.total_span_minutes is not None
        return config.kinematics.total_span_minutes
    if config.kinematics.method == "centered_adjacent":
        return float(2 * cadence_minutes)
    return float(cadence_minutes)


def _validate_velocity_grid(
    config: AnalysisConfig,
    cadence_minutes: int,
    effective_span: float,
) -> None:
    if config.kinematics.method == "stored_forward":
        return
    half_span = effective_span / 2
    steps = half_span / cadence_minutes
    if not math_is_integer(steps) or steps < 1:
        raise ValueError(
            f"A {effective_span:g}-minute centered velocity is incompatible with "
            f"the {cadence_minutes}-minute product grid; half the span must be a "
            "positive whole number of grid steps"
        )


def math_is_integer(value: float) -> bool:
    return np.isfinite(value) and abs(value - round(value)) < 1e-12


def _is_provisional(status: str) -> bool:
    lowered = status.lower()
    return any(marker in lowered for marker in ("candidate", "provisional", "partial"))


def _check_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")


def _write_yaml(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8",
    )
    return path


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Required provenance file is absent: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_id(
    config: AnalysisConfig,
    metadata_sha256: str,
    product_attributes: dict[str, Any],
) -> str:
    payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "algorithm_version": ANALYSIS_ALGORITHM_VERSION,
        "drifterlab_version": __version__,
        "configuration": config.effective(),
        "input_metadata_sha256": metadata_sha256,
        "product_attributes": product_attributes,
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"analysis-v1-{digest}"


def _product_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_NAME:
            continue
        records[relative] = {
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return records


def _publish(temporary: Path, target: Path, *, overwrite: bool, token: str) -> None:
    if not target.exists():
        temporary.rename(target)
        return
    if not overwrite:
        raise FileExistsError(f"Analysis output already exists: {target}; use --overwrite")
    backup = target.with_name(f".{target.name}.{token}.backup")
    if backup.exists():
        raise FileExistsError(f"Analysis backup already exists: {backup}")
    target.rename(backup)
    try:
        temporary.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    shutil.rmtree(backup)
