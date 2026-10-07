"""Numerical cluster-clock workflow for signed longitudinal statistics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
import math
from numbers import Real
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from drifterlab import __version__
from drifterlab.epsilon import (
    BinSupportRequirements,
    bin_indices,
    first_cluster_activation,
    geometric_bin_edges,
    linear_bin_edges,
    numeric_distribution_summary,
    pair_kinematics,
    separation_range_counts,
    stable_pair_id,
    summarize_separation_bins,
)
from drifterlab.epsilon_figures import (
    plot_cluster_comparison,
    plot_cluster_product,
    plot_window_small_multiples,
)
from drifterlab.epsilon_stage3 import (
    BootstrapSpec,
    conditional_q_histogram,
    leave_one_platform_out,
    summarize_platform_influence,
    symmetric_log_edges,
    synchronized_block_bootstrap,
)
from drifterlab.trajectories.kinematics import (
    PROJECTION_ORIGIN_METHOD,
    CenteredVelocityResult,
    centered_supported_velocity,
    project_array_positions,
)


EPSILON_SCHEMA_VERSION = "3.0"
EPSILON_ALGORITHM_VERSION = "array-clock-sampling-bootstrap-figures-v4"
SUPPORTED_RECONSTRUCTION_SCHEMA_VERSION = "1.4"
MANIFEST_NAME = "manifest.json"
RESOLVED_CONFIG_NAME = "config_resolved.yaml"
PROGRESS_NAME = "progress.md"


@dataclass(frozen=True)
class VelocitySpec:
    position_method: str
    total_span_minutes: float
    maximum_source_gap_minutes: float
    require_full_stencil_support: bool
    speed_mask_m_s: float | None


@dataclass(frozen=True)
class ActivationConfig:
    minimum_active_members: int | str
    reference: VelocitySpec


@dataclass(frozen=True)
class WindowConfig:
    identifier: str
    start_hour: float
    end_hour: float


@dataclass(frozen=True)
class SeparationBinConfig:
    mode: str
    edges_m: tuple[float, ...]
    centers_m: tuple[float, ...]
    source: dict[str, Any]
    axis_scale: str


@dataclass(frozen=True)
class QHistogramConfig:
    maximum_absolute_m2_s3: float
    linear_threshold_m2_s3: float
    bins_per_decade: int


@dataclass(frozen=True)
class FigureConfig:
    dpi: int
    scatter_maximum_points: int
    epsilon_linear_threshold_m2_s3: float
    heatmap_probability_maximum: float


@dataclass(frozen=True)
class Stage3Config:
    enabled: bool
    bootstrap: BootstrapSpec
    q_histogram: QHistogramConfig
    figures: FigureConfig


@dataclass(frozen=True)
class EpsilonWorkflowConfig:
    source_path: Path
    trajectory_path: Path
    dataset_label: str
    allow_candidate_input: bool
    arrays: tuple[int, ...]
    clusters: str | tuple[str, ...]
    pair_scope: str
    array_anchor: dict[str, Any] | None
    activation: ActivationConfig
    analysis: VelocitySpec
    sample_cadence_minutes: int
    sample_phase_reference: str
    windows: tuple[WindowConfig, ...]
    bins: SeparationBinConfig
    minimum_reliable_separation_m: float | None
    support: BinSupportRequirements
    stage3: Stage3Config
    output_root: Path

    def effective(self) -> dict[str, Any]:
        return {
            "input": {
                "trajectories": str(self.trajectory_path),
                "dataset_label": self.dataset_label,
                "allow_candidate_input": self.allow_candidate_input,
            },
            "selection": {
                "arrays": list(self.arrays),
                "clusters": self.clusters if isinstance(self.clusters, str) else list(self.clusters),
                "pair_scope": self.pair_scope,
                "array_anchor": self.array_anchor,
            },
            "activation": {
                "minimum_active_members": self.activation.minimum_active_members,
                "reference": _velocity_effective(self.activation.reference),
                "anchor_reuse": (
                    "reference anchors remain fixed for reconstruction, speed-mask, "
                    "and source-gap sensitivities"
                ),
            },
            "analysis": {
                **_velocity_effective(self.analysis),
                "sample_cadence_minutes": self.sample_cadence_minutes,
                "sample_phase_reference": self.sample_phase_reference,
                "sampling_operation": (
                    "select synchronized timestamps after vector-velocity calculation; "
                    "do not average scalar speed"
                ),
                "windows": [
                    {
                        "id": item.identifier,
                        "start_hour": item.start_hour,
                        "end_hour": item.end_hour,
                        "interval": "left_closed_right_open",
                    }
                    for item in self.windows
                ],
                "minimum_reliable_separation_m": self.minimum_reliable_separation_m,
                "four_fifths_coefficient": -1.25,
                "pair_time_weighting": "equal weight per valid pair-time observation",
                "transverse_unit_vector": "(-r_y, r_x) / separation",
            },
            "separation_bins": {
                "mode": self.bins.mode,
                **self.bins.source,
                "resolved_edges_m": list(self.bins.edges_m),
                "resolved_nominal_centers_m": list(self.bins.centers_m),
                "axis_scale": self.bins.axis_scale,
                "membership": "left_closed_right_open_with_final_edge_inclusive",
            },
            "support": asdict(self.support),
            "stage3": {
                "enabled": self.stage3.enabled,
                "uncertainty": {
                    "method": "synchronized_utc_block_bootstrap",
                    **asdict(self.stage3.bootstrap),
                },
                "q_histogram": asdict(self.stage3.q_histogram),
                "figures": asdict(self.stage3.figures),
            },
            "output": {"root": str(self.output_root)},
        }

    def identity_payload(self) -> dict[str, Any]:
        value = self.effective()
        value = json.loads(json.dumps(value))
        value.pop("output")
        return value


@dataclass(frozen=True)
class EpsilonWorkflowResult:
    run_directory: Path
    manifest_path: Path
    clusters_paths: tuple[Path, ...]
    array_count: int
    activated_cluster_count: int
    snapshot_observation_count: int
    evolution_observation_count: int
    run_id: str

    def format(self) -> str:
        return "\n".join((
            f"Epsilon analysis complete: {self.array_count} array(s), "
            f"{self.activated_cluster_count} activated cluster(s)",
            f"Snapshot pair observations: {self.snapshot_observation_count:,}",
            f"Evolution pair observations: {self.evolution_observation_count:,}",
            f"Run ID: {self.run_id}",
            f"Output: {self.run_directory}",
            f"Manifest: {self.manifest_path}",
        ))


def _velocity_effective(value: VelocitySpec) -> dict[str, Any]:
    return {
        "position_method": value.position_method,
        "velocity_method": "centered_two_endpoint_finite_difference",
        "total_span_minutes": value.total_span_minutes,
        "maximum_source_gap_minutes": value.maximum_source_gap_minutes,
        "require_full_stencil_support": value.require_full_stencil_support,
        "maximum_source_gap_applied": value.require_full_stencil_support,
        "speed_mask_m_s": value.speed_mask_m_s,
        "stencil_validity": (
            "all intervening positions must be finite and source-supported"
            if value.require_full_stencil_support else
            "only derivative endpoints must be finite; source support is trusted"
        ),
        "endpoint_policy": "centered_only_no_one_sided_estimates",
    }


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


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _unit_interval(value: Any, name: str) -> float:
    result = _positive(value, name)
    if result >= 1:
        raise ValueError(f"{name} must be less than one")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _parse_velocity(value: Any, name: str, *, allow_speed_mask: bool) -> VelocitySpec:
    section = _mapping(
        value,
        {
            "position_method", "velocity_method", "total_span_minutes",
            "maximum_source_gap_minutes", "require_full_stencil_support",
            "speed_mask_m_s",
        },
        name,
    )
    method = _nonempty(section.get("position_method"), f"{name}.position_method")
    if not re.fullmatch(r"[A-Za-z0-9_]+", method):
        raise ValueError(f"{name}.position_method contains unsupported characters")
    velocity_method = _nonempty(
        section.get("velocity_method", "centered_difference"),
        f"{name}.velocity_method",
    )
    if velocity_method != "centered_difference":
        raise ValueError(f"{name}.velocity_method currently supports centered_difference only")
    speed_raw = section.get("speed_mask_m_s")
    if speed_raw is not None and not allow_speed_mask:
        raise ValueError(f"{name}.speed_mask_m_s must be null for reference activation")
    return VelocitySpec(
        position_method=method,
        total_span_minutes=_positive(
            section.get("total_span_minutes", 30), f"{name}.total_span_minutes",
        ),
        maximum_source_gap_minutes=_positive(
            section.get("maximum_source_gap_minutes", 30),
            f"{name}.maximum_source_gap_minutes",
        ),
        require_full_stencil_support=_boolean(
            section.get("require_full_stencil_support", True),
            f"{name}.require_full_stencil_support",
        ),
        speed_mask_m_s=(
            None if speed_raw is None else _positive(speed_raw, f"{name}.speed_mask_m_s")
        ),
    )


def _parse_bins(value: Any) -> SeparationBinConfig:
    section = _mapping(
        value,
        {
            "mode", "edges_m", "minimum_m", "maximum_m", "ratio", "anchor_m",
            "width_m", "axis_scale",
        },
        "separation_bins",
    )
    mode = _nonempty(section.get("mode", "geometric"), "separation_bins.mode").lower()
    axis_scale = _nonempty(
        section.get("axis_scale", "log"), "separation_bins.axis_scale",
    ).lower()
    if axis_scale not in {"linear", "log"}:
        raise ValueError("separation_bins.axis_scale must be linear or log")
    if mode == "explicit":
        raw = section.get("edges_m")
        if not isinstance(raw, list) or len(raw) < 2:
            raise ValueError("separation_bins.edges_m must contain at least two edges")
        edges = np.asarray([_nonnegative(item, "separation_bins.edges_m") for item in raw])
        if np.any(np.diff(edges) <= 0):
            raise ValueError("Explicit separation edges must increase strictly")
        source = {"edges_m": edges.tolist()}
    elif mode == "geometric":
        minimum = _positive(section.get("minimum_m"), "separation_bins.minimum_m")
        maximum = _positive(section.get("maximum_m"), "separation_bins.maximum_m")
        ratio = _positive(section.get("ratio"), "separation_bins.ratio")
        anchor = _positive(section.get("anchor_m", 1000), "separation_bins.anchor_m")
        edges = geometric_bin_edges(minimum, maximum, ratio, anchor)
        source = {
            "minimum_m": minimum, "maximum_m": maximum,
            "ratio": ratio, "anchor_m": anchor,
        }
    elif mode == "linear":
        minimum = _nonnegative(section.get("minimum_m"), "separation_bins.minimum_m")
        maximum = _positive(section.get("maximum_m"), "separation_bins.maximum_m")
        width = _positive(section.get("width_m"), "separation_bins.width_m")
        edges = linear_bin_edges(minimum, maximum, width)
        source = {"minimum_m": minimum, "maximum_m": maximum, "width_m": width}
    else:
        raise ValueError("separation_bins.mode must be explicit, geometric, or linear")
    centers = (
        np.sqrt(edges[:-1] * edges[1:])
        if mode == "geometric" else (edges[:-1] + edges[1:]) / 2
    )
    return SeparationBinConfig(
        mode=mode,
        edges_m=tuple(float(item) for item in edges),
        centers_m=tuple(float(item) for item in centers),
        source=source,
        axis_scale=axis_scale,
    )


def _parse_stage3(value: Any, windows: tuple[WindowConfig, ...]) -> Stage3Config:
    section = _mapping(
        value, {"enabled", "uncertainty", "q_histogram", "figures"}, "stage3",
    )
    uncertainty = _mapping(
        section.get("uncertainty", {}),
        {
            "method", "confidence_level", "replicates", "random_seed",
            "primary_block_minutes", "sensitivity_block_minutes",
            "minimum_contributing_blocks", "minimum_successful_replicate_fraction",
        },
        "stage3.uncertainty",
    )
    method = _nonempty(
        uncertainty.get("method", "synchronized_utc_block_bootstrap"),
        "stage3.uncertainty.method",
    )
    if method != "synchronized_utc_block_bootstrap":
        raise ValueError("stage3.uncertainty.method is not supported")
    primary_block = _positive_integer(
        uncertainty.get("primary_block_minutes", 30),
        "stage3.uncertainty.primary_block_minutes",
    )
    raw_sensitivity = uncertainty.get("sensitivity_block_minutes", [60])
    if not isinstance(raw_sensitivity, list):
        raise ValueError("stage3.uncertainty.sensitivity_block_minutes must be a list")
    sensitivity = tuple(
        _positive_integer(item, "stage3.uncertainty.sensitivity_block_minutes")
        for item in raw_sensitivity
    )
    if len(set(sensitivity)) != len(sensitivity) or primary_block in sensitivity:
        raise ValueError("Bootstrap block durations must be unique")
    for window in windows:
        duration_minutes = (window.end_hour - window.start_hour) * 60
        for block in (primary_block, *sensitivity):
            quotient = duration_minutes / block
            if not math.isclose(quotient, round(quotient), abs_tol=1e-10):
                raise ValueError(
                    f"Bootstrap block duration {block} does not divide window {window.identifier}"
                )
    histogram = _mapping(
        section.get("q_histogram", {}),
        {"maximum_absolute_m2_s3", "linear_threshold_m2_s3", "bins_per_decade"},
        "stage3.q_histogram",
    )
    maximum_q = _positive(
        histogram.get("maximum_absolute_m2_s3", 0.005),
        "stage3.q_histogram.maximum_absolute_m2_s3",
    )
    q_threshold = _positive(
        histogram.get("linear_threshold_m2_s3", 1e-7),
        "stage3.q_histogram.linear_threshold_m2_s3",
    )
    if q_threshold >= maximum_q:
        raise ValueError("stage3 q histogram threshold must be below its maximum")
    figures = _mapping(
        section.get("figures", {}),
        {
            "dpi", "scatter_maximum_points", "epsilon_linear_threshold_m2_s3",
            "heatmap_probability_maximum",
        },
        "stage3.figures",
    )
    heatmap_maximum = _unit_interval(
        figures.get("heatmap_probability_maximum", 0.25),
        "stage3.figures.heatmap_probability_maximum",
    )
    return Stage3Config(
        enabled=_boolean(section.get("enabled", False), "stage3.enabled"),
        bootstrap=BootstrapSpec(
            confidence_level=_unit_interval(
                uncertainty.get("confidence_level", 0.95),
                "stage3.uncertainty.confidence_level",
            ),
            replicates=_positive_integer(
                uncertainty.get("replicates", 2000), "stage3.uncertainty.replicates",
            ),
            random_seed=_nonnegative_integer(
                uncertainty.get("random_seed", 41723), "stage3.uncertainty.random_seed",
            ),
            primary_block_minutes=primary_block,
            sensitivity_block_minutes=sensitivity,
            minimum_contributing_blocks=_positive_integer(
                uncertainty.get("minimum_contributing_blocks", 6),
                "stage3.uncertainty.minimum_contributing_blocks",
            ),
            minimum_successful_replicate_fraction=_unit_interval(
                uncertainty.get("minimum_successful_replicate_fraction", 0.95),
                "stage3.uncertainty.minimum_successful_replicate_fraction",
            ),
        ),
        q_histogram=QHistogramConfig(
            maximum_absolute_m2_s3=maximum_q,
            linear_threshold_m2_s3=q_threshold,
            bins_per_decade=_positive_integer(
                histogram.get("bins_per_decade", 6),
                "stage3.q_histogram.bins_per_decade",
            ),
        ),
        figures=FigureConfig(
            dpi=_positive_integer(figures.get("dpi", 160), "stage3.figures.dpi"),
            scatter_maximum_points=_positive_integer(
                figures.get("scatter_maximum_points", 20000),
                "stage3.figures.scatter_maximum_points",
            ),
            epsilon_linear_threshold_m2_s3=_positive(
                figures.get("epsilon_linear_threshold_m2_s3", 1e-7),
                "stage3.figures.epsilon_linear_threshold_m2_s3",
            ),
            heatmap_probability_maximum=heatmap_maximum,
        ),
    )


def load_epsilon_config(path: str | Path) -> EpsilonWorkflowConfig:
    """Read and strictly validate an epsilon-analysis YAML file."""
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8-sig") as stream:
        raw = yaml.safe_load(stream)
    values = _mapping(
        raw,
        {
            "input", "selection", "activation", "analysis", "separation_bins",
            "support", "stage3", "output",
        },
        "configuration",
    )
    input_section = _mapping(
        values.get("input", {}),
        {"trajectories", "dataset_label", "allow_candidate_input"},
        "input",
    )
    selection = _mapping(
        values.get("selection", {}),
        {"arrays", "clusters", "pair_scope", "array_anchor"},
        "selection",
    )
    activation_section = _mapping(
        values.get("activation", {}), {"minimum_active_members", "reference"}, "activation",
    )
    analysis = _mapping(
        values.get("analysis", {}),
        {
            "position_method", "velocity_method", "total_span_minutes",
            "maximum_source_gap_minutes", "require_full_stencil_support",
            "speed_mask_m_s", "sample_cadence_minutes", "sample_phase_reference",
            "windows", "minimum_reliable_separation_m",
        },
        "analysis",
    )
    support = _mapping(
        values.get("support", {}),
        {"observations", "unique_pairs", "platforms", "unique_utc"},
        "support",
    )
    output = _mapping(values.get("output", {}), {"root"}, "output")

    def resolved(value: Any, name: str) -> Path:
        candidate = Path(_nonempty(value, name)).expanduser()
        return (source_path.parent / candidate).resolve()

    trajectory_path = resolved(input_section.get("trajectories"), "input.trajectories")
    output_root = resolved(output.get("root"), "output.root")
    if trajectory_path == output_root or trajectory_path in output_root.parents:
        raise ValueError("Epsilon output root must be outside the input trajectory store")

    raw_arrays = selection.get("arrays")
    if not isinstance(raw_arrays, list) or not raw_arrays:
        raise ValueError("selection.arrays must be a nonempty list")
    arrays = tuple(_positive_integer(item, "selection.arrays") for item in raw_arrays)
    if len(set(arrays)) != len(arrays):
        raise ValueError("selection.arrays contains duplicates")
    raw_clusters = selection.get("clusters", "all")
    if raw_clusters == "all":
        clusters: str | tuple[str, ...] = "all"
    elif isinstance(raw_clusters, list) and raw_clusters:
        clusters = tuple(_nonempty(item, "selection.clusters") for item in raw_clusters)
    else:
        raise ValueError("selection.clusters must be 'all' or a nonempty list")
    pair_scope = _nonempty(
        selection.get("pair_scope", "within_cluster"), "selection.pair_scope",
    )
    if pair_scope not in {"within_cluster", "all_array"}:
        raise ValueError("selection.pair_scope must be within_cluster or all_array")
    array_anchor = selection.get("array_anchor")
    if pair_scope == "all_array":
        anchor = _mapping(
            array_anchor,
            {"method"},
            "selection.array_anchor",
        )
        anchor_method = _nonempty(
            anchor.get("method"), "selection.array_anchor.method",
        )
        if anchor_method != "first_time_all_selected_platforms_reference_valid":
            raise ValueError(
                "selection.array_anchor.method currently supports only "
                "first_time_all_selected_platforms_reference_valid"
            )
        array_anchor = {"method": anchor_method}
    elif array_anchor is not None:
        raise ValueError("selection.array_anchor must be null for within_cluster")

    minimum_active = activation_section.get("minimum_active_members", 2)
    if minimum_active != "all":
        minimum_active = _positive_integer(
            minimum_active, "activation.minimum_active_members",
        )
        if minimum_active < 2:
            raise ValueError("activation.minimum_active_members must be >=2 or 'all'")
    if pair_scope == "all_array" and minimum_active != "all":
        raise ValueError(
            "The first_time_all_selected_platforms_reference_valid array anchor "
            "requires activation.minimum_active_members: all"
        )
    reference = _parse_velocity(
        activation_section.get("reference", {}), "activation.reference",
        allow_speed_mask=False,
    )
    analysis_velocity = _parse_velocity(
        {
            key: analysis[key]
            for key in (
                "position_method", "velocity_method", "total_span_minutes",
                "maximum_source_gap_minutes", "require_full_stencil_support",
                "speed_mask_m_s",
            )
            if key in analysis
        },
        "analysis",
        allow_speed_mask=True,
    )
    sample_cadence_minutes = _positive_integer(
        analysis.get("sample_cadence_minutes", 5),
        "analysis.sample_cadence_minutes",
    )
    default_phase = "array_anchor" if pair_scope == "all_array" else "activation_anchor"
    sample_phase_reference = _nonempty(
        analysis.get("sample_phase_reference", default_phase),
        "analysis.sample_phase_reference",
    )
    expected_phase = "array_anchor" if pair_scope == "all_array" else "activation_anchor"
    if sample_phase_reference != expected_phase:
        raise ValueError(
            f"analysis.sample_phase_reference must be {expected_phase!r} for "
            f"selection.pair_scope {pair_scope!r}"
        )

    raw_windows = analysis.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError("analysis.windows must be a nonempty list")
    windows: list[WindowConfig] = []
    for index, raw_window in enumerate(raw_windows):
        section = _mapping(
            raw_window, {"id", "start_hour", "end_hour"}, f"analysis.windows[{index}]",
        )
        identifier = _nonempty(section.get("id"), f"analysis.windows[{index}].id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier):
            raise ValueError(f"analysis.windows[{index}].id is not path-safe")
        start = _nonnegative(section.get("start_hour"), f"analysis.windows[{index}].start_hour")
        end = _positive(section.get("end_hour"), f"analysis.windows[{index}].end_hour")
        if end <= start:
            raise ValueError(f"analysis.windows[{index}] must have end_hour > start_hour")
        windows.append(WindowConfig(identifier, start, end))
    if len({item.identifier for item in windows}) != len(windows):
        raise ValueError("analysis window IDs must be unique")

    minimum_reliable = analysis.get("minimum_reliable_separation_m")
    if minimum_reliable is not None:
        minimum_reliable = _positive(
            minimum_reliable, "analysis.minimum_reliable_separation_m",
        )
    resolved_windows = tuple(windows)
    return EpsilonWorkflowConfig(
        source_path=source_path,
        trajectory_path=trajectory_path,
        dataset_label=_nonempty(
            input_section.get("dataset_label", "current"), "input.dataset_label",
        ),
        allow_candidate_input=_boolean(
            input_section.get("allow_candidate_input", False),
            "input.allow_candidate_input",
        ),
        arrays=arrays,
        clusters=clusters,
        pair_scope=pair_scope,
        array_anchor=array_anchor,
        activation=ActivationConfig(minimum_active, reference),
        analysis=analysis_velocity,
        sample_cadence_minutes=sample_cadence_minutes,
        sample_phase_reference=sample_phase_reference,
        windows=resolved_windows,
        bins=_parse_bins(values.get("separation_bins", {})),
        minimum_reliable_separation_m=minimum_reliable,
        support=BinSupportRequirements(
            observations=_positive_integer(support.get("observations", 12), "support.observations"),
            unique_pairs=_positive_integer(support.get("unique_pairs", 3), "support.unique_pairs"),
            platforms=_positive_integer(support.get("platforms", 3), "support.platforms"),
            unique_utc=_positive_integer(support.get("unique_utc", 12), "support.unique_utc"),
        ),
        stage3=_parse_stage3(values.get("stage3", {}), resolved_windows),
        output_root=output_root,
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
        raise ValueError(f"Trajectory Zarr lacks consolidated metadata: {metadata}")
    return _file_sha256(metadata)


def _format_utc(value: np.datetime64) -> str:
    return pd.Timestamp(value).tz_localize("UTC").isoformat().replace("+00:00", "Z")


def _coordinate_names(dataset: xr.Dataset, method: str) -> tuple[str, str]:
    longitude = f"longitude_{method}"
    latitude = f"latitude_{method}"
    if longitude not in dataset or latitude not in dataset:
        methods = sorted(
            name.removeprefix("longitude_")
            for name in dataset.variables
            if name.startswith("longitude_")
            and f"latitude_{name.removeprefix('longitude_')}" in dataset
        )
        raise ValueError(f"Trajectory method {method!r} is unavailable; found {methods}")
    if dataset[longitude].dims != ("platform", "time") or dataset[latitude].dims != (
        "platform", "time",
    ):
        raise ValueError(f"Trajectory coordinates for {method!r} have invalid dimensions")
    return longitude, latitude


def _validate_dataset(dataset: xr.Dataset, config: EpsilonWorkflowConfig) -> None:
    if dataset.attrs.get("schema_version") != SUPPORTED_RECONSTRUCTION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported reconstruction schema {dataset.attrs.get('schema_version')!r}; "
            f"expected {SUPPORTED_RECONSTRUCTION_SCHEMA_VERSION!r}"
        )
    status = str(dataset.attrs.get("product_status", ""))
    if "candidate" in status.lower() and not config.allow_candidate_input:
        raise ValueError(
            f"Trajectory product status is {status!r}; set allow_candidate_input explicitly"
        )
    required = {
        "platform_id": ("platform",), "time": ("time",), "start_time": ("platform",),
        "array_id": ("platform",), "cluster_id": ("platform",),
        "cluster_size": ("platform",), "cluster_assignment_status": ("platform",),
        "source_gap_minutes": ("platform", "time"),
    }
    for name, dimensions in required.items():
        if name not in dataset or dataset[name].dims != dimensions:
            raise ValueError(f"Trajectory store lacks valid {name} with dimensions {dimensions}")
    _coordinate_names(dataset, config.activation.reference.position_method)
    _coordinate_names(dataset, config.analysis.position_method)
    ids = dataset.platform_id.values.astype(str)
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("Trajectory platform IDs must be unique and nonempty")
    times = dataset.time.values.astype("datetime64[ns]")
    if not len(times) or np.isnat(times).any() or np.any(np.diff(times.astype(np.int64)) <= 0):
        raise ValueError("Trajectory time must be finite and increase strictly")
    arrays = dataset.array_id.values.astype(int)
    unknown_arrays = sorted(set(config.arrays) - set(arrays.tolist()))
    if unknown_arrays:
        raise ValueError(f"Selected arrays are unavailable: {unknown_arrays}")
    cluster_ids = dataset.cluster_id.values.astype(str)
    if isinstance(config.clusters, tuple):
        unknown_clusters = sorted(set(config.clusters) - set(cluster_ids.tolist()))
        if unknown_clusters:
            raise ValueError(f"Selected clusters are unavailable: {unknown_clusters}")


def _sample_indices(
    times: np.ndarray,
    anchor: np.datetime64,
    cadence_minutes: int,
) -> np.ndarray:
    """Select one common-grid phase without interpolating or averaging velocity."""
    values = np.asarray(times, dtype="datetime64[ns]")
    if values.ndim != 1 or not len(values) or np.isnat(values).any():
        raise ValueError("Sampling requires a finite one-dimensional time grid")
    differences = np.diff(values.astype(np.int64))
    if not len(differences) or np.any(differences != differences[0]):
        raise ValueError("Sampling requires a uniform input time grid")
    cadence_ns = int(pd.to_timedelta(cadence_minutes, unit="m").value)
    native_ns = int(differences[0])
    if cadence_ns < native_ns or cadence_ns % native_ns:
        native_minutes = native_ns / (60 * 1_000_000_000)
        raise ValueError(
            "analysis.sample_cadence_minutes must be an integer multiple of the "
            f"native {native_minutes:g}-minute grid"
        )
    anchor_ns = np.datetime64(anchor, "ns").astype(np.int64)
    matching = np.flatnonzero(values.astype(np.int64) == anchor_ns)
    if len(matching) != 1:
        raise ValueError("The configured sampling anchor must be one input-grid timestamp")
    offsets = values.astype(np.int64) - anchor_ns
    return np.flatnonzero(np.mod(offsets, cadence_ns) == 0)


def _cluster_selected(config: EpsilonWorkflowConfig, cluster_id: str) -> bool:
    return config.clusters == "all" or cluster_id in config.clusters


def _pair_observations(
    *,
    array_id: int,
    cluster_id: str,
    member_indices: np.ndarray,
    time_indices: np.ndarray,
    platform_ids: np.ndarray,
    member_cluster_ids: np.ndarray,
    times: np.ndarray,
    t0: np.datetime64,
    longitude: np.ndarray,
    latitude: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    velocity: CenteredVelocityResult,
    source_gap_minutes: np.ndarray,
    inverse_transformer,
    minimum_reliable_separation_m: float | None,
) -> pd.DataFrame:
    column_names = (
        "array_id", "cluster_id", "pair_id", "platform_id_1", "platform_id_2",
        "source_cluster_id_1", "source_cluster_id_2", "pair_cluster_relation",
        "time", "cluster_age_hours", "longitude_1", "latitude_1", "longitude_2",
        "latitude_2", "x_1_m", "y_1_m", "x_2_m", "y_2_m", "u_1_m_s",
        "v_1_m_s", "u_2_m_s", "v_2_m_s", "speed_1_m_s", "speed_2_m_s",
        "source_gap_1_minutes", "source_gap_2_minutes", "midpoint_x_m",
        "midpoint_y_m", "midpoint_longitude", "midpoint_latitude", "delta_x_m",
        "delta_y_m", "separation_m", "delta_u_m_s", "delta_v_m_s",
        "delta_u_l_m_s", "delta_u_t_m_s", "delta_u_l_cube_m3_s3",
        "delta_u_l_delta_u_t_squared_m3_s3", "q_m2_s3",
        "below_minimum_reliable_separation",
    )
    frames: list[pd.DataFrame] = []
    ordered_members = sorted(member_indices.tolist(), key=lambda item: str(platform_ids[item]))
    for first, second in combinations(ordered_members, 2):
        selected = time_indices[
            velocity.valid[first, time_indices] & velocity.valid[second, time_indices]
        ]
        if not len(selected):
            continue
        values = pair_kinematics(
            x_m[first, selected], y_m[first, selected],
            velocity.u_m_s[first, selected], velocity.v_m_s[first, selected],
            x_m[second, selected], y_m[second, selected],
            velocity.u_m_s[second, selected], velocity.v_m_s[second, selected],
        )
        valid = values.pop("valid")
        selected = selected[valid]
        if not len(selected):
            continue
        values = {name: item[valid] for name, item in values.items()}
        midpoint_x = (x_m[first, selected] + x_m[second, selected]) / 2
        midpoint_y = (y_m[first, selected] + y_m[second, selected]) / 2
        midpoint_lon, midpoint_lat = inverse_transformer.transform(midpoint_x, midpoint_y)
        count = len(selected)
        first_id = str(platform_ids[first])
        second_id = str(platform_ids[second])
        pair_id = stable_pair_id(first_id, second_id)
        source_cluster_1 = str(member_cluster_ids[first])
        source_cluster_2 = str(member_cluster_ids[second])
        below = (
            np.zeros(count, dtype=bool)
            if minimum_reliable_separation_m is None
            else values["separation_m"] < minimum_reliable_separation_m
        )
        frames.append(pd.DataFrame({
            "array_id": array_id,
            "cluster_id": cluster_id,
            "pair_id": pair_id,
            "platform_id_1": first_id,
            "platform_id_2": second_id,
            "source_cluster_id_1": source_cluster_1,
            "source_cluster_id_2": source_cluster_2,
            "pair_cluster_relation": (
                "within_cluster"
                if source_cluster_1 == source_cluster_2
                else "between_clusters"
            ),
            "time": times[selected],
            "cluster_age_hours": np.asarray(
                (times[selected] - t0) / np.timedelta64(1, "h"), dtype=float,
            ),
            "longitude_1": longitude[first, selected],
            "latitude_1": latitude[first, selected],
            "longitude_2": longitude[second, selected],
            "latitude_2": latitude[second, selected],
            "x_1_m": x_m[first, selected], "y_1_m": y_m[first, selected],
            "x_2_m": x_m[second, selected], "y_2_m": y_m[second, selected],
            "u_1_m_s": velocity.u_m_s[first, selected],
            "v_1_m_s": velocity.v_m_s[first, selected],
            "u_2_m_s": velocity.u_m_s[second, selected],
            "v_2_m_s": velocity.v_m_s[second, selected],
            "speed_1_m_s": np.hypot(
                velocity.u_m_s[first, selected], velocity.v_m_s[first, selected],
            ),
            "speed_2_m_s": np.hypot(
                velocity.u_m_s[second, selected], velocity.v_m_s[second, selected],
            ),
            "source_gap_1_minutes": source_gap_minutes[first, selected],
            "source_gap_2_minutes": source_gap_minutes[second, selected],
            "midpoint_x_m": midpoint_x, "midpoint_y_m": midpoint_y,
            "midpoint_longitude": np.asarray(midpoint_lon, dtype=float),
            "midpoint_latitude": np.asarray(midpoint_lat, dtype=float),
            **values,
            "below_minimum_reliable_separation": below,
        }, columns=column_names))
    frame = (
        pd.concat(frames, ignore_index=True)
        if frames else pd.DataFrame(columns=column_names)
    )
    if len(frame):
        frame["time"] = pd.to_datetime(frame.time).to_numpy(dtype="datetime64[ns]")
    return frame


def _pair_valid_at_times(
    first: int,
    second: int,
    indices: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    velocity: CenteredVelocityResult,
    minimum_reliable_separation_m: float | None,
) -> np.ndarray:
    valid = velocity.valid[first, indices] & velocity.valid[second, indices]
    separation = np.hypot(
        x_m[second, indices] - x_m[first, indices],
        y_m[second, indices] - y_m[first, indices],
    )
    valid &= np.isfinite(separation) & (separation > 0)
    if minimum_reliable_separation_m is not None:
        valid &= separation >= minimum_reliable_separation_m
    return valid


def _primary_pair_exclusion_reason(
    first: int,
    second: int,
    index: int,
    x_m: np.ndarray,
    y_m: np.ndarray,
    velocity: CenteredVelocityResult,
    minimum_reliable_separation_m: float | None,
) -> str:
    if velocity.valid[first, index] and velocity.valid[second, index]:
        separation = math.hypot(
            x_m[second, index] - x_m[first, index],
            y_m[second, index] - y_m[first, index],
        )
        if not math.isfinite(separation) or separation <= 0:
            return "zero_or_invalid_separation"
        if (
            minimum_reliable_separation_m is not None
            and separation < minimum_reliable_separation_m
        ):
            return "below_minimum_reliable_separation"
        return "valid"
    reasons = {str(velocity.invalid_reason[first, index]), str(velocity.invalid_reason[second, index])}
    priority = (
        "nonfinite_position_in_stencil", "missing_source_support_in_stencil",
        "source_gap_exceeded_in_stencil", "speed_mask_exceeded",
        "nonfinite_velocity", "outside_centered_stencil",
    )
    for reason in priority:
        if reason in reasons:
            return reason
    return "other_invalid_velocity"


def _coverage_tables(
    *,
    array_id: int,
    cluster_id: str,
    member_indices: np.ndarray,
    platform_ids: np.ndarray,
    member_cluster_ids: np.ndarray,
    times: np.ndarray,
    sample_indices: np.ndarray,
    t0: np.datetime64,
    windows: tuple[WindowConfig, ...],
    x_m: np.ndarray,
    y_m: np.ndarray,
    velocity: CenteredVelocityResult,
    minimum_reliable_separation_m: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    ordered = sorted(member_indices.tolist(), key=lambda item: str(platform_ids[item]))
    members = list(combinations(ordered, 2))
    sample_indices = np.asarray(sample_indices, dtype=int)
    sample_ages = (times[sample_indices] - t0) / np.timedelta64(1, "h")
    for window in windows:
        indices = sample_indices[
            (sample_ages >= window.start_hour) & (sample_ages < window.end_hour)
        ]
        expected = len(indices)
        reason_counts: Counter[str] = Counter()
        for first, second in members:
            pair_valid = _pair_valid_at_times(
                first, second, indices, x_m, y_m, velocity,
                minimum_reliable_separation_m,
            )
            valid_locations = np.flatnonzero(pair_valid)
            if len(valid_locations):
                first_valid = int(valid_locations[0])
                last_valid = int(valid_locations[-1])
                availability_count = last_valid - first_valid + 1
                before = first_valid
                after = expected - last_valid - 1
                internal_missing = availability_count - len(valid_locations)
            else:
                availability_count = 0
                before = expected
                after = 0
                internal_missing = 0
            pair_rows.append({
                "array_id": array_id,
                "cluster_id": cluster_id,
                "window_id": window.identifier,
                "window_start_hour": window.start_hour,
                "window_end_hour": window.end_hour,
                "pair_id": stable_pair_id(platform_ids[first], platform_ids[second]),
                "platform_id_1": str(platform_ids[first]),
                "platform_id_2": str(platform_ids[second]),
                "source_cluster_id_1": str(member_cluster_ids[first]),
                "source_cluster_id_2": str(member_cluster_ids[second]),
                "pair_cluster_relation": (
                    "within_cluster"
                    if str(member_cluster_ids[first]) == str(member_cluster_ids[second])
                    else "between_clusters"
                ),
                "scheduled_observation_count": expected,
                "valid_observation_count": int(pair_valid.sum()),
                "full_window_coverage_fraction": (
                    float(pair_valid.sum() / expected) if expected else np.nan
                ),
                "availability_scheduled_count": availability_count,
                "availability_conditional_coverage_fraction": (
                    float(pair_valid.sum() / availability_count)
                    if availability_count else np.nan
                ),
                "scheduled_before_first_valid_count": before,
                "internal_missing_count": internal_missing,
                "scheduled_after_last_valid_count": after,
                "complete_full_window": bool(expected and pair_valid.all()),
            })
            for local, index in enumerate(indices):
                if pair_valid[local]:
                    reason_counts["valid"] += 1
                else:
                    reason_counts[_primary_pair_exclusion_reason(
                        first, second, int(index), x_m, y_m, velocity,
                        minimum_reliable_separation_m,
                    )] += 1
        expected_pair_observations = len(members) * expected
        valid_pair_observations = reason_counts["valid"]
        cluster_rows.append({
            "array_id": array_id,
            "cluster_id": cluster_id,
            "window_id": window.identifier,
            "window_start_hour": window.start_hour,
            "window_end_hour": window.end_hour,
            "scheduled_utc_count": expected,
            "assigned_member_count": len(ordered),
            "possible_pair_count": len(members),
            "expected_pair_observation_count": expected_pair_observations,
            "valid_pair_observation_count": valid_pair_observations,
            "aggregate_full_window_coverage_fraction": (
                valid_pair_observations / expected_pair_observations
                if expected_pair_observations else np.nan
            ),
            "unique_contributing_pair_count": sum(
                row["valid_observation_count"] > 0
                for row in pair_rows if row["window_id"] == window.identifier
            ),
            "complete_pair_count": sum(
                row["complete_full_window"]
                for row in pair_rows if row["window_id"] == window.identifier
            ),
        })
        for reason, count in sorted(reason_counts.items()):
            exclusion_rows.append({
                "array_id": array_id,
                "cluster_id": cluster_id,
                "window_id": window.identifier,
                "reason": reason,
                "pair_time_count": count,
                "fraction_of_scheduled_pair_times": (
                    count / expected_pair_observations if expected_pair_observations else np.nan
                ),
                "classification": "exclusive_primary_reason",
            })
    return pd.DataFrame(pair_rows), pd.DataFrame(cluster_rows), pd.DataFrame(exclusion_rows)


def _platform_diagnostics(
    *,
    array_id: int,
    cluster_id: str,
    member_indices: np.ndarray,
    platform_ids: np.ndarray,
    member_cluster_ids: np.ndarray,
    times: np.ndarray,
    time_indices: np.ndarray,
    t0: np.datetime64 | None,
    longitude: np.ndarray,
    latitude: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    source_gap_minutes: np.ndarray,
    reference: CenteredVelocityResult,
    analysis: CenteredVelocityResult,
) -> pd.DataFrame:
    """Return normalized platform-time validity and velocity diagnostics."""
    indices = np.asarray(time_indices, dtype=int)
    if t0 is None or np.isnat(t0):
        ages = np.full(len(indices), np.nan, dtype=float)
        is_snapshot = np.zeros(len(indices), dtype=bool)
    else:
        ages = np.asarray((times[indices] - t0) / np.timedelta64(1, "h"), dtype=float)
        is_snapshot = times[indices] == t0
    reference_counts = reference.valid[np.ix_(member_indices, indices)].sum(axis=0).astype(int)
    analysis_counts = analysis.valid[np.ix_(member_indices, indices)].sum(axis=0).astype(int)
    frames: list[pd.DataFrame] = []
    for member in member_indices:
        reference_u = reference.u_m_s[member, indices]
        reference_v = reference.v_m_s[member, indices]
        analysis_u = analysis.u_m_s[member, indices]
        analysis_v = analysis.v_m_s[member, indices]
        frames.append(pd.DataFrame({
            "array_id": array_id,
            "cluster_id": cluster_id,
            "source_cluster_id": str(member_cluster_ids[member]),
            "platform_id": str(platform_ids[member]),
            "time_utc": times[indices],
            "cluster_age_hours": ages,
            "is_snapshot": is_snapshot,
            "longitude": longitude[member, indices],
            "latitude": latitude[member, indices],
            "x_m": x_m[member, indices],
            "y_m": y_m[member, indices],
            "source_gap_minutes": source_gap_minutes[member, indices],
            "assigned_member_count": len(member_indices),
            "reference_active_member_count": reference_counts,
            "analysis_active_member_count": analysis_counts,
            "reference_valid": reference.valid[member, indices],
            "reference_invalid_reason": reference.invalid_reason[member, indices],
            "reference_u_m_s": reference_u,
            "reference_v_m_s": reference_v,
            "reference_speed_m_s": np.hypot(reference_u, reference_v),
            "analysis_valid": analysis.valid[member, indices],
            "analysis_invalid_reason": analysis.invalid_reason[member, indices],
            "analysis_u_m_s": analysis_u,
            "analysis_v_m_s": analysis_v,
            "analysis_speed_m_s": np.hypot(analysis_u, analysis_v),
        }))
    return pd.concat(frames, ignore_index=True)


def _velocity_observations(
    *,
    array_id: int,
    cluster_id: str,
    member_indices: np.ndarray,
    platform_ids: np.ndarray,
    times: np.ndarray,
    indices: np.ndarray,
    velocity: CenteredVelocityResult,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for member in member_indices:
        selected = indices[velocity.valid[member, indices]]
        if not len(selected):
            continue
        u = velocity.u_m_s[member, selected]
        v = velocity.v_m_s[member, selected]
        rows.append(pd.DataFrame({
            "array_id": array_id,
            "cluster_id": cluster_id,
            "platform_id": str(platform_ids[member]),
            "time": times[selected],
            "u_m_s": u,
            "v_m_s": v,
            "speed_m_s": np.hypot(u, v),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["array_id", "cluster_id", "platform_id", "time", "u_m_s", "v_m_s", "speed_m_s"]
    )


def _offset_utc(t0: np.datetime64, hours: float) -> str:
    value = pd.Timestamp(t0).tz_localize("UTC") + pd.to_timedelta(hours, unit="h")
    return value.isoformat().replace("+00:00", "Z")


def _analysis_identity(
    frame: pd.DataFrame,
    *,
    array_id: int,
    cluster_id: str,
    analysis_type: str,
    t0: np.datetime64,
    window: WindowConfig | None = None,
) -> pd.DataFrame:
    """Add one consistent readable-table identity without conflating time types."""
    if analysis_type not in {"snapshot", "window"}:
        raise ValueError(f"Unknown analysis type {analysis_type!r}")
    if analysis_type == "window" and window is None:
        raise ValueError("Window analysis identity requires a window")
    identity = {
        "array_id": array_id,
        "cluster_id": cluster_id,
        "analysis_type": analysis_type,
        "window_id": "snapshot" if window is None else window.identifier,
        "activation_time_utc": _format_utc(t0),
        "snapshot_time_utc": _format_utc(t0) if window is None else None,
        "window_start_hour": np.nan if window is None else window.start_hour,
        "window_end_hour": np.nan if window is None else window.end_hour,
        "window_start_utc": None if window is None else _offset_utc(t0, window.start_hour),
        "window_end_utc": None if window is None else _offset_utc(t0, window.end_hour),
    }
    result = frame.drop(columns=list(identity), errors="ignore").reset_index(drop=True)
    metadata = pd.DataFrame(
        {name: [value] * len(result) for name, value in identity.items()}
    )
    return pd.concat((metadata, result), axis=1)


def _coverage_identity(
    frame: pd.DataFrame,
    *,
    coverage_level: str,
    t0: np.datetime64,
    windows: tuple[WindowConfig, ...],
) -> pd.DataFrame:
    """Normalize cluster, pair, exclusion, and range records into one table."""
    result = frame.copy()
    window_by_id = {window.identifier: window for window in windows}
    result.insert(2, "analysis_type", "window")
    result.insert(4, "coverage_level", coverage_level)
    if "pair_id" not in result:
        result["pair_id"] = pd.NA
    if "reason" in result:
        result = result.rename(columns={"reason": "exclusion_reason"})
    elif "exclusion_reason" not in result:
        result["exclusion_reason"] = pd.NA
    result["activation_time_utc"] = _format_utc(t0)
    result["window_start_hour"] = result.window_id.map(
        lambda identifier: window_by_id[str(identifier)].start_hour
    )
    result["window_end_hour"] = result.window_id.map(
        lambda identifier: window_by_id[str(identifier)].end_hour
    )
    result["window_start_utc"] = result.window_id.map(
        lambda identifier: _offset_utc(t0, window_by_id[str(identifier)].start_hour)
    )
    result["window_end_utc"] = result.window_id.map(
        lambda identifier: _offset_utc(t0, window_by_id[str(identifier)].end_hour)
    )
    leading = [
        "array_id", "cluster_id", "analysis_type", "window_id", "coverage_level",
        "pair_id", "activation_time_utc", "window_start_hour", "window_end_hour",
        "window_start_utc", "window_end_utc", "exclusion_reason",
    ]
    return result[leading + [name for name in result if name not in leading]]


def _stage3_array_products(
    *,
    array_id: int,
    pair_observations: pd.DataFrame,
    epsilon_by_scale: pd.DataFrame,
    config: EpsilonWorkflowConfig,
    array_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    """Add Stage-3 uncertainty, influence, histograms, and review figures."""
    edges = np.asarray(config.bins.edges_m, dtype=float)
    q_edges = symmetric_log_edges(
        config.stage3.q_histogram.maximum_absolute_m2_s3,
        config.stage3.q_histogram.linear_threshold_m2_s3,
        config.stage3.q_histogram.bins_per_decade,
    )
    histogram_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    influence_frames: list[pd.DataFrame] = []
    enriched_frames: list[pd.DataFrame] = []
    figure_paths: list[Path] = []
    window_lookup = {item.identifier: item for item in config.windows}
    identity_columns = ["array_id", "cluster_id", "analysis_type", "window_id"]

    for identity, summary in epsilon_by_scale.groupby(identity_columns, sort=False):
        _, cluster_id, analysis_type, window_id = identity
        cluster_rows = pair_observations[pair_observations.cluster_id == cluster_id]
        if analysis_type == "snapshot":
            selected = cluster_rows[cluster_rows.is_snapshot.astype(bool)].copy()
            window = None
        else:
            window = window_lookup[str(window_id)]
            selected = cluster_rows[
                (cluster_rows.cluster_age_hours >= window.start_hour)
                & (cluster_rows.cluster_age_hours < window.end_hour)
            ].copy()
        if "below_minimum_reliable_separation" in selected:
            selected = selected[~selected.below_minimum_reliable_separation.astype(bool)]

        histogram = conditional_q_histogram(selected, edges, q_edges)
        for name, value in zip(identity_columns, identity):
            histogram.insert(len(histogram.columns) if name in histogram else 0, name, value)
        # Restore the common identifiers to the front after the insertions above.
        histogram = histogram[
            identity_columns + [name for name in histogram if name not in identity_columns]
        ]
        histogram_frames.append(histogram)

        influence = leave_one_platform_out(
            selected, edges,
            support_minimum_observations=config.support.observations,
            support_minimum_pairs=config.support.unique_pairs,
            support_minimum_platforms=config.support.platforms,
            support_minimum_utc=config.support.unique_utc,
        )
        for name, value in reversed(tuple(zip(identity_columns, identity))):
            influence.insert(0, name, value)
        influence_frames.append(influence)
        influence_summary = summarize_platform_influence(influence)

        enriched = summary.copy()
        enriched = enriched.drop(columns=["confidence_interval_status"], errors="ignore")
        if not influence_summary.empty:
            enriched = enriched.merge(
                influence_summary, on="separation_bin_id", how="left", validate="one_to_one",
            )
        if analysis_type == "snapshot":
            enriched["bootstrap_block_minutes"] = np.nan
            enriched["scheduled_block_count"] = np.nan
            enriched["contributing_block_count"] = np.nan
            enriched["bootstrap_replicates_requested"] = 0
            enriched["successful_bootstrap_replicates"] = 0
            enriched["successful_bootstrap_fraction"] = np.nan
            enriched["confidence_level"] = config.stage3.bootstrap.confidence_level
            enriched["epsilon_ci_lower_m2_s3"] = np.nan
            enriched["epsilon_ci_upper_m2_s3"] = np.nan
            enriched["confidence_interval_status"] = "not_applicable_snapshot"
        else:
            support_by_bin = dict(zip(
                enriched.separation_bin_id.astype(str), enriched.support_ok.astype(bool),
            ))
            primary: pd.DataFrame | None = None
            for block_minutes in (
                config.stage3.bootstrap.primary_block_minutes,
                *config.stage3.bootstrap.sensitivity_block_minutes,
            ):
                bootstrap = synchronized_block_bootstrap(
                    selected, edges,
                    window_start_hour=window.start_hour,
                    window_end_hour=window.end_hour,
                    block_minutes=block_minutes,
                    confidence_level=config.stage3.bootstrap.confidence_level,
                    replicates=config.stage3.bootstrap.replicates,
                    random_seed=config.stage3.bootstrap.random_seed,
                    identity="|".join(map(str, identity)),
                    minimum_contributing_blocks=(
                        config.stage3.bootstrap.minimum_contributing_blocks
                    ),
                    minimum_successful_replicate_fraction=(
                        config.stage3.bootstrap.minimum_successful_replicate_fraction
                    ),
                    support_by_bin=support_by_bin,
                )
                bootstrap["is_primary_block_duration"] = (
                    block_minutes == config.stage3.bootstrap.primary_block_minutes
                )
                for name, value in reversed(tuple(zip(identity_columns, identity))):
                    bootstrap.insert(0, name, value)
                bootstrap_frames.append(bootstrap)
                if block_minutes == config.stage3.bootstrap.primary_block_minutes:
                    primary = bootstrap
            if primary is None:
                raise RuntimeError("Primary bootstrap duration was not evaluated")
            bootstrap_columns = [
                "separation_bin_id", "bootstrap_block_minutes", "scheduled_block_count",
                "contributing_block_count", "bootstrap_replicates_requested",
                "successful_bootstrap_replicates", "successful_bootstrap_fraction",
                "confidence_level", "epsilon_ci_lower_m2_s3", "epsilon_ci_upper_m2_s3",
                "confidence_interval_status",
            ]
            enriched = enriched.merge(
                primary[bootstrap_columns], on="separation_bin_id", how="left",
                validate="one_to_one",
            )
        enriched_frames.append(enriched)

        safe_cluster = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cluster_id))
        figure_name = "snapshot.png" if analysis_type == "snapshot" else f"{window_id}.png"
        figure_path = array_directory / "figures" / safe_cluster / figure_name
        activation = enriched.activation_time_utc.dropna()
        plot_cluster_product(
            selected, enriched, histogram,
            output_path=figure_path,
            array_id=array_id,
            cluster_id=str(cluster_id),
            analysis_type=str(analysis_type),
            window_id=str(window_id),
            activation_time_utc=str(activation.iloc[0]) if len(activation) else "unavailable",
            separation_edges_m=edges,
            q_edges_m2_s3=q_edges,
            q_linear_threshold_m2_s3=config.stage3.q_histogram.linear_threshold_m2_s3,
            epsilon_linear_threshold_m2_s3=(
                config.stage3.figures.epsilon_linear_threshold_m2_s3
            ),
            scatter_maximum_points=config.stage3.figures.scatter_maximum_points,
            dpi=config.stage3.figures.dpi,
            heatmap_probability_maximum=(
                config.stage3.figures.heatmap_probability_maximum
            ),
        )
        figure_paths.append(figure_path)

    enriched_all = pd.concat(enriched_frames, ignore_index=True)
    histograms_all = pd.concat(histogram_frames, ignore_index=True)
    bootstrap_all = (
        pd.concat(bootstrap_frames, ignore_index=True)
        if bootstrap_frames else pd.DataFrame(columns=(
            *identity_columns, "separation_bin_id", "bootstrap_block_minutes",
        ))
    )
    influence_all = (
        pd.concat(influence_frames, ignore_index=True)
        if influence_frames else pd.DataFrame(columns=(
            *identity_columns, "separation_bin_id", "removed_platform_id",
        ))
    )

    temporal = enriched_all[enriched_all.analysis_type == "window"]
    comparison_directory = array_directory / "figures" / "comparisons"
    comparison_subject = "population" if config.pair_scope == "all_array" else "cluster"
    for window_id, group in temporal.groupby("window_id", sort=True):
        path = comparison_directory / f"epsilon_by_{comparison_subject}__{window_id}.png"
        plot_cluster_comparison(
            group, output_path=path, array_id=array_id, window_id=str(window_id),
            separation_edges_m=edges,
            epsilon_linear_threshold_m2_s3=(
                config.stage3.figures.epsilon_linear_threshold_m2_s3
            ),
            dpi=config.stage3.figures.dpi,
        )
        figure_paths.append(path)
    if len(temporal):
        path = comparison_directory / "epsilon_by_window_small_multiples.png"
        plot_window_small_multiples(
            temporal, output_path=path, array_id=array_id,
            separation_edges_m=edges,
            epsilon_linear_threshold_m2_s3=(
                config.stage3.figures.epsilon_linear_threshold_m2_s3
            ),
            dpi=config.stage3.figures.dpi,
        )
        figure_paths.append(path)
    guide_path = array_directory / "figures" / "README.md"
    guide_path.write_text(
        "# Figure guide\n\n"
        "See `docs/epsilon_analysis/figure_guide.md` in the drifterlab repository. "
        "Population folders contain snapshot and elapsed-window diagnostics; "
        "for all-array mode the population is `array_NNN__all_array` and includes "
        "simultaneous within- and between-cluster pairs.\n",
        encoding="utf-8",
    )
    figure_paths.append(guide_path)
    return enriched_all, histograms_all, bootstrap_all, influence_all, figure_paths


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


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
        raise FileExistsError(f"Epsilon output already exists: {target}; use --overwrite")
    backup = target.with_name(f".{target.name}.{token}.backup")
    if backup.exists():
        raise FileExistsError(f"Epsilon backup already exists: {backup}")
    target.rename(backup)
    try:
        temporary.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    _remove_generated(backup, target.parent)


def _run_id(
    config: EpsilonWorkflowConfig,
    input_metadata_sha256: str,
    build_report_sha256: str | None,
) -> str:
    payload = {
        "schema_version": EPSILON_SCHEMA_VERSION,
        "algorithm_version": EPSILON_ALGORITHM_VERSION,
        "drifterlab_version": __version__,
        "configuration": config.identity_payload(),
        "input_metadata_sha256": input_metadata_sha256,
        "build_report_sha256": build_report_sha256,
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"epsilon-v1-{digest}"


def _velocity_directory_name(config: EpsilonWorkflowConfig) -> str:
    span = f"{config.analysis.total_span_minutes:g}".replace(".", "p")
    cadence = f"{config.sample_cadence_minutes}min"
    return f"centered_difference_{span}min_on_{cadence}_grid"


def run_epsilon_workflow(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> EpsilonWorkflowResult:
    """Run numerical epsilon products and optional Stage-3 review products."""
    config = load_epsilon_config(config_path)
    report = progress or (lambda message: None)
    if not config.trajectory_path.is_dir():
        raise ValueError(f"Trajectory Zarr does not exist: {config.trajectory_path}")
    metadata_sha256 = _input_metadata_hash(config.trajectory_path)
    build_report_path = config.trajectory_path.parent / "build_report.csv"
    build_report_sha256 = _file_sha256(build_report_path) if build_report_path.is_file() else None
    run_id = _run_id(config, metadata_sha256, build_report_sha256)
    target = (
        config.output_root
        / config.analysis.position_method
        / _velocity_directory_name(config)
        / run_id
    )
    if target.exists() and not overwrite:
        raise FileExistsError(f"Epsilon output already exists: {target}; use --overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = target.with_name(f".{target.name}.{token}.tmp")
    temporary.mkdir()

    product_paths: list[Path] = []
    relative_clusters_paths: list[Path] = []
    array_manifests: list[dict[str, Any]] = []
    snapshot_count = 0
    evolution_count = 0
    unified_pair_count = 0
    activated_count = 0
    input_attrs: dict[str, Any] = {}
    try:
        report(f"Opening {config.trajectory_path}")
        with xr.open_zarr(config.trajectory_path, consolidated=True, chunks=None) as dataset:
            _validate_dataset(dataset, config)
            input_attrs = dict(dataset.attrs)
            times = dataset.time.values.astype("datetime64[ns]")
            platform_ids_all = dataset.platform_id.values.astype(str)
            arrays_all = dataset.array_id.values.astype(int)
            clusters_all = dataset.cluster_id.values.astype(str)
            cluster_sizes_all = dataset.cluster_size.values.astype(int)
            cluster_status_all = dataset.cluster_assignment_status.values.astype(str)
            starts_all = dataset.start_time.values.astype("datetime64[ns]")
            gaps_all = dataset.source_gap_minutes.values.astype(float)
            ref_lon_name, ref_lat_name = _coordinate_names(
                dataset, config.activation.reference.position_method,
            )
            analysis_lon_name, analysis_lat_name = _coordinate_names(
                dataset, config.analysis.position_method,
            )
            ref_lon_all = dataset[ref_lon_name].values.astype(float)
            ref_lat_all = dataset[ref_lat_name].values.astype(float)
            analysis_lon_all = dataset[analysis_lon_name].values.astype(float)
            analysis_lat_all = dataset[analysis_lat_name].values.astype(float)

            for array_id in config.arrays:
                array_mask = arrays_all == array_id
                global_indices = np.flatnonzero(array_mask)
                platform_ids = platform_ids_all[array_mask]
                cluster_ids = clusters_all[array_mask]
                cluster_sizes = cluster_sizes_all[array_mask]
                cluster_status = cluster_status_all[array_mask]
                starts = starts_all[array_mask]
                gaps = gaps_all[array_mask]
                analysis_longitude = analysis_lon_all[array_mask]
                analysis_latitude = analysis_lat_all[array_mask]
                reference_projection = project_array_positions(
                    ref_lon_all[array_mask], ref_lat_all[array_mask],
                )
                analysis_projection = project_array_positions(
                    analysis_longitude, analysis_latitude,
                    origin_longitude=reference_projection.origin_longitude,
                    origin_latitude=reference_projection.origin_latitude,
                )
                reference_velocity = centered_supported_velocity(
                    reference_projection.x_m, reference_projection.y_m, times, gaps,
                    total_span_minutes=config.activation.reference.total_span_minutes,
                    maximum_source_gap_minutes=(
                        config.activation.reference.maximum_source_gap_minutes
                    ),
                    speed_mask_m_s=None,
                    require_full_stencil_support=(
                        config.activation.reference.require_full_stencil_support
                    ),
                )
                analysis_velocity = centered_supported_velocity(
                    analysis_projection.x_m, analysis_projection.y_m, times, gaps,
                    total_span_minutes=config.analysis.total_span_minutes,
                    maximum_source_gap_minutes=config.analysis.maximum_source_gap_minutes,
                    speed_mask_m_s=config.analysis.speed_mask_m_s,
                    require_full_stencil_support=config.analysis.require_full_stencil_support,
                )
                array_directory = temporary / f"array_{array_id:03d}"
                data_directory = array_directory / "data"
                (array_directory / "figures").mkdir(parents=True, exist_ok=True)
                cluster_entries: list[dict[str, Any]] = []
                inventory_rows: list[dict[str, object]] = []
                pair_frames: list[pd.DataFrame] = []
                platform_frames: list[pd.DataFrame] = []
                epsilon_frames: list[pd.DataFrame] = []
                statistic_frames: list[pd.DataFrame] = []
                coverage_frames: list[pd.DataFrame] = []
                edges = np.asarray(config.bins.edges_m)
                centers = np.asarray(config.bins.centers_m)
                maximum_end = max(item.end_hour for item in config.windows)
                diagnostic_duration = pd.to_timedelta(maximum_end, unit="h").to_timedelta64()

                if config.pair_scope == "all_array":
                    selected_member_mask = np.asarray([
                        _cluster_selected(config, item) for item in cluster_ids
                    ], dtype=bool)
                    selected_members = np.flatnonzero(selected_member_mask)
                    if not len(selected_members):
                        raise ValueError(f"Array {array_id} contains no selected platforms")
                    population_definitions = [
                        (f"array_{array_id:03d}__all_array", selected_members)
                    ]
                else:
                    population_definitions = [
                        (cluster_id, np.flatnonzero(cluster_ids == cluster_id))
                        for cluster_id in dict.fromkeys(cluster_ids.tolist())
                        if _cluster_selected(config, cluster_id)
                    ]

                for cluster_id, members in population_definitions:
                    assigned = len(members)
                    if config.pair_scope == "within_cluster":
                        stored_sizes = set(cluster_sizes[members].tolist())
                        if stored_sizes != {assigned}:
                            raise ValueError(
                                f"Array {array_id} cluster {cluster_id} has inconsistent cluster_size"
                            )
                    statuses = sorted(set(cluster_status[members].tolist()))
                    activation_index = first_cluster_activation(
                        reference_velocity.valid[members],
                        config.activation.minimum_active_members,
                    )
                    activation_time = (
                        np.datetime64("NaT", "ns")
                        if activation_index is None else times[activation_index]
                    )
                    reference_active = (
                        np.zeros(assigned, dtype=bool)
                        if activation_index is None
                        else reference_velocity.valid[members, activation_index]
                    )
                    analysis_active = (
                        np.zeros(assigned, dtype=bool)
                        if activation_index is None
                        else analysis_velocity.valid[members, activation_index]
                    )
                    if activation_index is None:
                        snapshot_status = (
                            "singleton_no_internal_pair" if assigned == 1
                            else "never_satisfies_reference_activation"
                        )
                    elif int(analysis_active.sum()) < 2:
                        snapshot_status = "unavailable_analysis_has_fewer_than_two_members"
                    elif np.any(reference_active & ~analysis_active):
                        snapshot_status = "available_but_reference_members_lost_in_analysis"
                    elif int(analysis_active.sum()) < assigned:
                        snapshot_status = "available_partial_assigned_members"
                    else:
                        snapshot_status = "available_all_assigned_members"
                    inventory_row = {
                        "array_id": array_id,
                        "cluster_id": cluster_id,
                        "analysis_type": "inventory",
                        "window_id": "all",
                        "pair_scope": config.pair_scope,
                        "assigned_member_count": assigned,
                        "possible_pair_count": assigned * (assigned - 1) // 2,
                        "possible_internal_pair_count": assigned * (assigned - 1) // 2,
                        "source_cluster_count": int(len(set(cluster_ids[members].tolist()))),
                        "source_cluster_ids": "|".join(sorted(set(cluster_ids[members].tolist()))),
                        "cluster_assignment_status": "|".join(statuses),
                        "platform_ids": "|".join(platform_ids[members].astype(str).tolist()),
                        "first_recorded_position_time_min": _format_utc(starts[members].min()),
                        "first_recorded_position_time_max": _format_utc(starts[members].max()),
                        "activation_time_utc": (
                            None if activation_index is None else _format_utc(activation_time)
                        ),
                        "activation_delay_from_first_recorded_minutes": (
                            np.nan if activation_index is None else float(
                                (activation_time - starts[members].min()) / np.timedelta64(1, "m")
                            )
                        ),
                        "reference_active_member_count_at_activation": int(reference_active.sum()),
                        "analysis_active_member_count_at_reference_activation": int(analysis_active.sum()),
                        "reference_active_platform_ids": "|".join(
                            platform_ids[members[reference_active]].astype(str).tolist()
                        ),
                        "analysis_active_platform_ids": "|".join(
                            platform_ids[members[analysis_active]].astype(str).tolist()
                        ),
                        "snapshot_status": snapshot_status,
                        "snapshot_pair_observation_count": 0,
                        "evolution_pair_observation_count": 0,
                    }
                    inventory_rows.append(inventory_row)
                    diagnostic_start = starts[members].min()
                    diagnostic_origin = (
                        diagnostic_start if activation_index is None else activation_time
                    )
                    diagnostic_stop = diagnostic_origin + diagnostic_duration
                    phase_anchor = times[0] if activation_index is None else activation_time
                    population_sample_indices = _sample_indices(
                        times, phase_anchor, config.sample_cadence_minutes,
                    )
                    diagnostic_indices = population_sample_indices[
                        (times[population_sample_indices] >= diagnostic_start)
                        & (times[population_sample_indices] < diagnostic_stop)
                    ]
                    platform_frames.append(_platform_diagnostics(
                        array_id=array_id,
                        cluster_id=cluster_id,
                        member_indices=members,
                        platform_ids=platform_ids,
                        member_cluster_ids=cluster_ids,
                        times=times,
                        time_indices=diagnostic_indices,
                        t0=None if activation_index is None else activation_time,
                        longitude=analysis_longitude,
                        latitude=analysis_latitude,
                        x_m=analysis_projection.x_m,
                        y_m=analysis_projection.y_m,
                        source_gap_minutes=gaps,
                        reference=reference_velocity,
                        analysis=analysis_velocity,
                    ))
                    if activation_index is None:
                        cluster_entries.append({
                            "cluster_id": cluster_id,
                            "assigned_member_count": assigned,
                            "activation_status": snapshot_status,
                            "snapshot_pair_observation_count": 0,
                            "evolution_pair_observation_count": 0,
                        })
                        continue

                    activated_count += 1
                    minimum_start = min(item.start_hour for item in config.windows)
                    ages = (times - activation_time) / np.timedelta64(1, "h")
                    evolution_indices = population_sample_indices[
                        (ages[population_sample_indices] >= minimum_start)
                        & (ages[population_sample_indices] < maximum_end)
                    ]
                    snapshot_indices = np.asarray([activation_index], dtype=int)
                    observation_arguments = {
                        "array_id": array_id,
                        "cluster_id": cluster_id,
                        "member_indices": members,
                        "platform_ids": platform_ids,
                        "member_cluster_ids": cluster_ids,
                        "times": times,
                        "t0": activation_time,
                        "longitude": analysis_longitude,
                        "latitude": analysis_latitude,
                        "x_m": analysis_projection.x_m,
                        "y_m": analysis_projection.y_m,
                        "velocity": analysis_velocity,
                        "source_gap_minutes": gaps,
                        "inverse_transformer": analysis_projection.inverse,
                        "minimum_reliable_separation_m": config.minimum_reliable_separation_m,
                    }
                    snapshot = _pair_observations(
                        time_indices=snapshot_indices, **observation_arguments,
                    )
                    evolution = _pair_observations(
                        time_indices=evolution_indices, **observation_arguments,
                    )
                    snapshot_count += len(snapshot)
                    evolution_count += len(evolution)
                    inventory_row["snapshot_pair_observation_count"] = len(snapshot)
                    inventory_row["evolution_pair_observation_count"] = len(evolution)

                    evolution_rows = evolution.copy()
                    evolution_rows["is_snapshot"] = (
                        evolution_rows.time.to_numpy(dtype="datetime64[ns]") == activation_time
                    )
                    snapshot_rows = snapshot.copy()
                    snapshot_rows["is_snapshot"] = True
                    unified = pd.concat((evolution_rows, snapshot_rows), ignore_index=True)
                    if len(unified):
                        unified = unified.rename(columns={"time": "time_utc"})
                        unified = unified.sort_values(
                            ["cluster_id", "pair_id", "time_utc", "is_snapshot"]
                        ).drop_duplicates(
                            ["array_id", "cluster_id", "pair_id", "time_utc"], keep="last"
                        ).reset_index(drop=True)
                    else:
                        unified = unified.rename(columns={"time": "time_utc"})
                    pair_frames.append(unified)

                    snapshot_scientific = snapshot[
                        ~snapshot.below_minimum_reliable_separation.astype(bool)
                    ]
                    snapshot_summary = summarize_separation_bins(
                        snapshot_scientific, edges, centers, product_kind="snapshot",
                        scheduled_utc_count=1, requirements=config.support,
                    )
                    snapshot_summary.insert(
                        0, "separation_bin_id",
                        snapshot_summary.separation_bin_index.map(lambda value: f"bin_{int(value):03d}"),
                    )
                    epsilon_frames.append(_analysis_identity(
                        snapshot_summary, array_id=array_id, cluster_id=cluster_id,
                        analysis_type="snapshot", t0=activation_time,
                    ))

                    snapshot_pair_statistics = numeric_distribution_summary(
                        snapshot_scientific,
                        ("separation_m", "delta_u_l_m_s", "delta_u_t_m_s", "q_m2_s3"),
                    )
                    snapshot_pair_statistics.insert(0, "population_level", "pair")
                    statistic_frames.append(_analysis_identity(
                        snapshot_pair_statistics, array_id=array_id, cluster_id=cluster_id,
                        analysis_type="snapshot", t0=activation_time,
                    ))
                    snapshot_velocity = _velocity_observations(
                        array_id=array_id, cluster_id=cluster_id,
                        member_indices=members, platform_ids=platform_ids,
                        times=times, indices=snapshot_indices, velocity=analysis_velocity,
                    )
                    snapshot_velocity_statistics = numeric_distribution_summary(
                        snapshot_velocity, ("u_m_s", "v_m_s", "speed_m_s"),
                    )
                    snapshot_velocity_statistics.insert(0, "population_level", "platform")
                    statistic_frames.append(_analysis_identity(
                        snapshot_velocity_statistics, array_id=array_id, cluster_id=cluster_id,
                        analysis_type="snapshot", t0=activation_time,
                    ))

                    pair_coverage, cluster_coverage, exclusions = _coverage_tables(
                        array_id=array_id, cluster_id=cluster_id,
                        member_indices=members, platform_ids=platform_ids,
                        member_cluster_ids=cluster_ids,
                        times=times, t0=activation_time, windows=config.windows,
                        sample_indices=population_sample_indices,
                        x_m=analysis_projection.x_m, y_m=analysis_projection.y_m,
                        velocity=analysis_velocity,
                        minimum_reliable_separation_m=config.minimum_reliable_separation_m,
                    )
                    coverage_frames.extend((
                        _coverage_identity(
                            cluster_coverage, coverage_level="cluster", t0=activation_time,
                            windows=config.windows,
                        ),
                        _coverage_identity(
                            pair_coverage, coverage_level="pair", t0=activation_time,
                            windows=config.windows,
                        ),
                        _coverage_identity(
                            exclusions, coverage_level="exclusion", t0=activation_time,
                            windows=config.windows,
                        ),
                    ))

                    for window in config.windows:
                        selected = evolution[
                            (evolution.cluster_age_hours >= window.start_hour)
                            & (evolution.cluster_age_hours < window.end_hour)
                            & ~evolution.below_minimum_reliable_separation.astype(bool)
                        ].copy()
                        scheduled_indices = population_sample_indices[
                            (ages[population_sample_indices] >= window.start_hour)
                            & (ages[population_sample_indices] < window.end_hour)
                        ]
                        summary = summarize_separation_bins(
                            selected, edges, centers, product_kind="window",
                            scheduled_utc_count=len(scheduled_indices),
                            requirements=config.support,
                        )
                        summary.insert(
                            0, "separation_bin_id",
                            summary.separation_bin_index.map(lambda value: f"bin_{int(value):03d}"),
                        )
                        epsilon_frames.append(_analysis_identity(
                            summary, array_id=array_id, cluster_id=cluster_id,
                            analysis_type="window", t0=activation_time, window=window,
                        ))

                        distribution = numeric_distribution_summary(
                            selected,
                            ("separation_m", "delta_u_l_m_s", "delta_u_t_m_s", "q_m2_s3"),
                        )
                        distribution.insert(0, "population_level", "pair")
                        statistic_frames.append(_analysis_identity(
                            distribution, array_id=array_id, cluster_id=cluster_id,
                            analysis_type="window", t0=activation_time, window=window,
                        ))
                        velocity_frame = _velocity_observations(
                            array_id=array_id, cluster_id=cluster_id,
                            member_indices=members, platform_ids=platform_ids,
                            times=times, indices=scheduled_indices,
                            velocity=analysis_velocity,
                        )
                        velocity_summary = numeric_distribution_summary(
                            velocity_frame, ("u_m_s", "v_m_s", "speed_m_s"),
                        )
                        velocity_summary.insert(0, "population_level", "platform")
                        statistic_frames.append(_analysis_identity(
                            velocity_summary, array_id=array_id, cluster_id=cluster_id,
                            analysis_type="window", t0=activation_time, window=window,
                        ))
                        range_frame = pd.DataFrame(({
                            "array_id": array_id,
                            "cluster_id": cluster_id,
                            "window_id": window.identifier,
                            **separation_range_counts(selected, edges),
                        },))
                        coverage_frames.append(_coverage_identity(
                            range_frame, coverage_level="separation_range",
                            t0=activation_time, windows=config.windows,
                        ))

                    cluster_entries.append({
                        "cluster_id": cluster_id,
                        "assigned_member_count": assigned,
                        "activation_time_utc": _format_utc(activation_time),
                        "reference_active_member_count": int(reference_active.sum()),
                        "analysis_active_member_count": int(analysis_active.sum()),
                        "snapshot_status": snapshot_status,
                        "snapshot_pair_observation_count": len(snapshot),
                        "evolution_pair_observation_count": len(evolution),
                    })
                    report(
                        f"Array {array_id} {cluster_id}: t0={_format_utc(activation_time)}, "
                        f"snapshot={len(snapshot)}, evolution={len(evolution):,}"
                    )

                if not inventory_rows:
                    raise ValueError(f"Array {array_id} contains no selected clusters")
                clusters_frame = pd.DataFrame(inventory_rows).sort_values("cluster_id")
                pair_observations = (
                    pd.concat(pair_frames, ignore_index=True)
                    if pair_frames else pd.DataFrame(columns=(
                        "array_id", "cluster_id", "pair_id", "platform_id_1",
                        "platform_id_2", "source_cluster_id_1", "source_cluster_id_2",
                        "pair_cluster_relation", "time_utc", "cluster_age_hours",
                        "is_snapshot",
                    ))
                )
                if len(pair_observations):
                    pair_observations = pair_observations.sort_values(
                        ["cluster_id", "pair_id", "time_utc"]
                    ).reset_index(drop=True)
                    duplicated = pair_observations.duplicated(
                        ["array_id", "cluster_id", "pair_id", "time_utc"]
                    )
                    if duplicated.any():
                        raise RuntimeError("Unified pair observations contain duplicate pair-times")
                unified_pair_count += len(pair_observations)
                platform_diagnostics = (
                    pd.concat(platform_frames, ignore_index=True)
                    if platform_frames else pd.DataFrame(columns=(
                        "array_id", "cluster_id", "platform_id", "time_utc",
                    ))
                )
                if len(platform_diagnostics):
                    platform_diagnostics = platform_diagnostics.sort_values(
                        ["cluster_id", "platform_id", "time_utc"]
                    ).reset_index(drop=True)
                    duplicated = platform_diagnostics.duplicated(
                        ["array_id", "cluster_id", "platform_id", "time_utc"]
                    )
                    if duplicated.any():
                        raise RuntimeError("Platform diagnostics contain duplicate platform-times")
                epsilon_by_scale = (
                    pd.concat(epsilon_frames, ignore_index=True)
                    if epsilon_frames else pd.DataFrame(columns=(
                        "array_id", "cluster_id", "analysis_type", "window_id",
                        "separation_bin_id",
                    ))
                )
                statistics = (
                    pd.concat(statistic_frames, ignore_index=True)
                    if statistic_frames else pd.DataFrame(columns=(
                        "array_id", "cluster_id", "analysis_type", "window_id",
                        "population_level", "metric",
                    ))
                )
                coverage = (
                    pd.concat(coverage_frames, ignore_index=True, sort=False)
                    if coverage_frames else pd.DataFrame(columns=(
                        "array_id", "cluster_id", "analysis_type", "window_id",
                        "coverage_level", "pair_id",
                    ))
                )
                stage3_detail_paths: list[Path] = []
                stage3_figure_paths: list[Path] = []
                if config.stage3.enabled:
                    report(f"Array {array_id}: Stage 3 bootstrap, influence, and figures")
                    (
                        epsilon_by_scale,
                        q_histograms,
                        bootstrap_sensitivity,
                        platform_influence,
                        stage3_figure_paths,
                    ) = _stage3_array_products(
                        array_id=array_id,
                        pair_observations=pair_observations,
                        epsilon_by_scale=epsilon_by_scale,
                        config=config,
                        array_directory=array_directory,
                    )
                    stage3_detail_paths = [
                        _write_parquet(
                            q_histograms, data_directory / "q_histograms.parquet",
                        ),
                        _write_parquet(
                            bootstrap_sensitivity,
                            data_directory / "bootstrap_sensitivity.parquet",
                        ),
                        _write_parquet(
                            platform_influence,
                            data_directory / "leave_one_platform_out.parquet",
                        ),
                    ]
                array_products = [
                    _write_csv(clusters_frame, array_directory / "clusters.csv"),
                    _write_csv(epsilon_by_scale, array_directory / "epsilon_by_scale.csv"),
                    _write_csv(statistics, array_directory / "statistics.csv"),
                    _write_csv(coverage, array_directory / "coverage.csv"),
                    _write_parquet(pair_observations, data_directory / "pair_observations.parquet"),
                    _write_parquet(
                        platform_diagnostics, data_directory / "platform_diagnostics.parquet",
                    ),
                    *stage3_detail_paths,
                    *stage3_figure_paths,
                ]
                product_paths.extend(array_products)
                relative_clusters_paths.append(array_products[0].relative_to(temporary))
                for frame, path in (
                    (clusters_frame, array_products[0]),
                    (epsilon_by_scale, array_products[1]),
                    (statistics, array_products[2]),
                    (coverage, array_products[3]),
                ):
                    if len(pd.read_csv(path)) != len(frame):
                        raise RuntimeError(f"Readable table failed readback validation: {path.name}")
                array_manifests.append({
                    "array_id": array_id,
                    "platform_count": len(global_indices),
                    "projection": {
                        "type": "azimuthal_equidistant",
                        "origin_method": PROJECTION_ORIGIN_METHOD,
                        "origin_longitude": reference_projection.origin_longitude,
                        "origin_latitude": reference_projection.origin_latitude,
                        "crs_definition": reference_projection.crs_definition,
                    },
                    "files": [str(path.relative_to(temporary)) for path in array_products],
                    "figures_directory": str((array_directory / "figures").relative_to(temporary)),
                    "stage3_figure_count": len(stage3_figure_paths),
                    "clusters": cluster_entries,
                })

        resolved = {
            **config.effective(),
            "run_id": run_id,
            "epsilon_schema_version": EPSILON_SCHEMA_VERSION,
            "epsilon_algorithm_version": EPSILON_ALGORITHM_VERSION,
            "input_metadata_sha256": metadata_sha256,
            "input_build_report_sha256": build_report_sha256,
        }
        resolved_path = temporary / RESOLVED_CONFIG_NAME
        resolved_path.write_text(
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8",
        )
        product_paths.append(resolved_path)
        progress_path = temporary / PROGRESS_NAME
        stage_label = "Stage 3 reviewable pilot" if config.stage3.enabled else "Stage 2 numerical pilot"
        population_label = (
            "array-wide populations" if config.pair_scope == "all_array" else "clusters"
        )
        stage_lines = (
            (
                "- Synchronized UTC block-bootstrap intervals with configured duration sensitivity.",
                "- Conditional signed-q histograms, leave-one-platform-out checks, and figures.",
                "- Snapshot products remain descriptive without temporal confidence intervals.",
            )
            if config.stage3.enabled else
            ("- Confidence intervals and plots are intentionally not computed in Stage 2.",)
        )
        progress_path.write_text(
            "\n".join((
                "# Epsilon analysis progress",
                "",
                "## Completed",
                "",
                f"- {stage_label}.",
                "- Per-array readable tables and normalized observation/diagnostic Parquet data.",
                f"- Fixed reference anchors for {population_label}, elapsed windows, support, coverage,",
                "  structure functions, signed epsilon, moments, and influence diagnostics.",
                *stage_lines,
                "",
                "## Validation",
                "",
                f"- Activated {population_label}: {activated_count}",
                f"- Snapshot pair observations: {snapshot_count}",
                f"- Evolution pair observations: {evolution_count}",
                f"- Unique stored pair observations: {unified_pair_count}",
                "- Snapshot/evolution duplicates are represented once with is_snapshot=true.",
                "",
                "## Open scientific choices",
                "",
                f"- Input product status: {input_attrs.get('product_status')}.",
                "- Positional accuracy and a validated minimum reliable separation are unresolved.",
                "- Scientific primary reconstruction awaits linear versus spline_30 comparison.",
                "- q histogram edges and bootstrap block lengths remain provisional choices.",
                "",
                "## Next stage",
                "",
                (
                    "- Stage 4 configured reconstruction and mask comparisons."
                    if config.stage3.enabled else
                    "- Stage 3 plots, synchronized UTC block bootstrap, and leave-one-platform-out checks."
                ),
                "",
            )) + "\n",
            encoding="utf-8",
        )
        product_paths.append(progress_path)

        products = {
            str(path.relative_to(temporary)): {
                "sha256": _file_sha256(path), "bytes": path.stat().st_size,
            }
            for path in product_paths
        }
        manifest = {
            "schema_version": EPSILON_SCHEMA_VERSION,
            "algorithm_version": EPSILON_ALGORITHM_VERSION,
            "drifterlab_version": __version__,
            "processing_time_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "configuration_path": str(config.source_path),
            "effective_configuration": config.effective(),
            "output_layout": {
                "scope": "one consolidated product set per array",
                "readable_tables": [
                    "clusters.csv", "epsilon_by_scale.csv", "statistics.csv", "coverage.csv",
                ],
                "normalized_data": [
                    "data/pair_observations.parquet",
                    "data/platform_diagnostics.parquet",
                    *(
                        [
                            "data/q_histograms.parquet",
                            "data/bootstrap_sensitivity.parquet",
                            "data/leave_one_platform_out.parquet",
                        ]
                        if config.stage3.enabled else []
                    ),
                ],
                "raw_window_membership": (
                    "derived from cluster_age_hours; in all_array mode this compatibility "
                    "field is elapsed array-anchor age; rows are not duplicated"
                ),
            },
            "input": {
                "trajectory_zarr": str(config.trajectory_path),
                "consolidated_metadata_sha256": metadata_sha256,
                "build_report_sha256": build_report_sha256,
                "schema_version": input_attrs.get("schema_version"),
                "algorithm_version": input_attrs.get("algorithm_version"),
                "product_status": input_attrs.get("product_status"),
                "initial_cluster_assignment_sha256": input_attrs.get(
                    "initial_cluster_assignment_sha256"
                ),
            },
            "scientific_conventions": {
                "coefficient": -1.25,
                "coefficient_label": "3D four-fifths-law convention",
                "epsilon_interpretation": "signed effective transfer diagnostic, not local viscous dissipation",
                "velocity_formula_baseline": (
                    "u(t) = [x(t + 15 min) - x(t - 15 min)] / 1800 s"
                ),
                "sample_cadence_minutes": config.sample_cadence_minutes,
                "sample_phase_reference": config.sample_phase_reference,
                "sampling_operation": (
                    "synchronized timestamp selection after vector-velocity calculation; "
                    "no scalar-speed averaging"
                ),
                "velocity_stencil_point_count": 7,
                "velocity_numerator_point_count": 2,
                "reference_required_validity_point_count": (
                    7 if config.activation.reference.require_full_stencil_support else 2
                ),
                "analysis_required_validity_point_count": (
                    7 if config.analysis.require_full_stencil_support else 2
                ),
                "transverse_unit_vector": "(-r_y, r_x) / separation",
                "units": {
                    "separation": "m", "velocity": "m s-1",
                    "q": "m2 s-3", "epsilon_eff": "m2 s-3",
                },
            },
            "analysis": {
                "array_count": len(config.arrays),
                "activated_cluster_count": activated_count,
                "activated_population_count": activated_count,
                "pair_scope": config.pair_scope,
                "snapshot_pair_observation_count": snapshot_count,
                "evolution_pair_observation_count": evolution_count,
                "stored_unique_pair_observation_count": unified_pair_count,
                "confidence_intervals": (
                    "synchronized_utc_block_bootstrap"
                    if config.stage3.enabled else "not_computed_stage_2"
                ),
                "figures": (
                    (
                        "stage3_population_products_and_comparisons"
                        if config.pair_scope == "all_array"
                        else "stage3_cluster_products_and_comparisons"
                    )
                    if config.stage3.enabled
                    else "directory_created_empty_stage_2"
                ),
                "arrays": array_manifests,
            },
            "products": products,
        }
        manifest_path = temporary / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        if _input_metadata_hash(config.trajectory_path) != metadata_sha256:
            raise ValueError("Trajectory Zarr metadata changed during epsilon processing")
        json.loads(manifest_path.read_text(encoding="utf-8"))
        _publish(temporary, target, token=token, overwrite=overwrite)
        return EpsilonWorkflowResult(
            run_directory=target,
            manifest_path=target / MANIFEST_NAME,
            clusters_paths=tuple(target / path for path in relative_clusters_paths),
            array_count=len(config.arrays),
            activated_cluster_count=activated_count,
            snapshot_observation_count=snapshot_count,
            evolution_observation_count=evolution_count,
            run_id=run_id,
        )
    finally:
        if temporary.exists():
            _remove_generated(temporary, target.parent)
