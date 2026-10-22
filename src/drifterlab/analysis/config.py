"""Strict configuration for the unified v2 trajectory analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any

import yaml


ANALYSIS_SCHEMA_VERSION = 1
KINEMATIC_METHODS = {
    "centered_fixed_span",
    "centered_adjacent",
    "stored_forward",
}


@dataclass(frozen=True)
class KinematicsConfig:
    method: str
    total_span_minutes: float | None
    maximum_source_gap_minutes: float
    require_full_stencil_support: bool
    speed_mask_m_s: float | None


@dataclass(frozen=True)
class AnalysisConfig:
    source_path: Path
    trajectory_zarr: Path
    trajectory_product: str
    dataset_label: str
    allow_provisional: bool
    arrays: str | tuple[int, ...]
    clusters: str | tuple[str, ...]
    kinematics: KinematicsConfig
    trajectory_plots: dict[str, Any]
    trajectory_statistics: dict[str, Any]
    epsilon: dict[str, Any]
    epsilon_diagnostics: dict[str, Any]
    fsle: dict[str, Any]
    output_root: Path

    def effective(self) -> dict[str, Any]:
        return {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "input": {
                "trajectory_zarr": str(self.trajectory_zarr),
                "trajectory_product": self.trajectory_product,
                "dataset_label": self.dataset_label,
                "allow_provisional": self.allow_provisional,
            },
            "selection": {
                "arrays": self.arrays if isinstance(self.arrays, str) else list(self.arrays),
                "clusters": (
                    self.clusters if isinstance(self.clusters, str) else list(self.clusters)
                ),
            },
            "kinematics": {
                "method": self.kinematics.method,
                "total_span_minutes": self.kinematics.total_span_minutes,
                "maximum_source_gap_minutes": (
                    self.kinematics.maximum_source_gap_minutes
                ),
                "require_full_stencil_support": (
                    self.kinematics.require_full_stencil_support
                ),
                "speed_mask_m_s": self.kinematics.speed_mask_m_s,
            },
            "analyses": {
                "trajectory_plots": self.trajectory_plots,
                "trajectory_statistics": self.trajectory_statistics,
                "epsilon": self.epsilon,
                "epsilon_diagnostics": self.epsilon_diagnostics,
                "fsle": self.fsle,
            },
            "output": {"root": str(self.output_root)},
        }


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    source_path = Path(path).expanduser().resolve()
    with source_path.open(encoding="utf-8-sig") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(
        raw,
        {"schema_version", "input", "selection", "kinematics", "analyses", "output"},
        "configuration",
    )
    schema = root.get("schema_version", ANALYSIS_SCHEMA_VERSION)
    if schema != ANALYSIS_SCHEMA_VERSION:
        raise ValueError(
            f"Analysis configuration schema {schema!r} is unsupported; "
            f"expected {ANALYSIS_SCHEMA_VERSION}"
        )

    input_section = _mapping(
        root.get("input", {}),
        {
            "trajectory_zarr", "trajectory_product", "dataset_label",
            "allow_provisional",
        },
        "input",
    )
    selection = _mapping(
        root.get("selection", {}), {"arrays", "clusters"}, "selection",
    )
    kinematics = _mapping(
        root.get("kinematics", {}),
        {
            "method", "total_span_minutes", "maximum_source_gap_minutes",
            "require_full_stencil_support", "speed_mask_m_s",
        },
        "kinematics",
    )
    analyses = _mapping(
        root.get("analyses", {}),
        {
            "trajectory_plots", "trajectory_statistics", "epsilon",
            "epsilon_diagnostics",
            "fsle",
        },
        "analyses",
    )
    output = _mapping(root.get("output", {}), {"root"}, "output")

    def resolved(value: Any, name: str) -> Path:
        candidate = Path(_nonempty(value, name)).expanduser()
        return (source_path.parent / candidate).resolve()

    trajectory_zarr = resolved(input_section.get("trajectory_zarr"), "input.trajectory_zarr")
    output_root = resolved(output.get("root"), "output.root")
    if output_root == trajectory_zarr or trajectory_zarr in output_root.parents:
        raise ValueError("Analysis output must be outside the input trajectory Zarr")
    product = _nonempty(
        input_section.get("trajectory_product", "qc_5min"),
        "input.trajectory_product",
    )
    if "/" in product or "\\" in product or product in {".", ".."}:
        raise ValueError("input.trajectory_product must be one direct product-group name")

    method = _nonempty(
        kinematics.get("method", "centered_fixed_span"), "kinematics.method",
    )
    if method not in KINEMATIC_METHODS:
        raise ValueError(f"kinematics.method must be one of {sorted(KINEMATIC_METHODS)}")
    span_raw = kinematics.get("total_span_minutes")
    if method == "centered_fixed_span":
        total_span = _positive(
            30 if span_raw is None else span_raw,
            "kinematics.total_span_minutes",
        )
    else:
        if span_raw is not None:
            raise ValueError(
                "kinematics.total_span_minutes is only valid for centered_fixed_span"
            )
        total_span = None
    full_support = _boolean(
        kinematics.get("require_full_stencil_support", True),
        "kinematics.require_full_stencil_support",
    )
    if method == "stored_forward" and not full_support:
        raise ValueError("stored_forward requires require_full_stencil_support: true")

    trajectory_plots = _analysis_block(
        analyses.get("trajectory_plots", {}), "analyses.trajectory_plots",
    )
    trajectory_statistics = _analysis_block(
        analyses.get("trajectory_statistics", {}),
        "analyses.trajectory_statistics",
    )
    epsilon = _analysis_block(analyses.get("epsilon", {}), "analyses.epsilon")
    if "spatial_maps" in epsilon:
        from .spatial import resolve_spatial_maps

        epsilon["spatial_maps"] = resolve_spatial_maps(epsilon["spatial_maps"], epsilon)
    diagnostics = _analysis_block(
        analyses.get("epsilon_diagnostics", {}),
        "analyses.epsilon_diagnostics",
    )
    from .fsle import resolve_fsle

    fsle = resolve_fsle(analyses.get("fsle", {}))
    if diagnostics.get("enabled", False) and not epsilon.get("enabled", False):
        raise ValueError("epsilon_diagnostics requires epsilon.enabled: true")
    if not any(
        block.get("enabled", False)
        for block in (trajectory_plots, trajectory_statistics, epsilon, diagnostics, fsle)
    ):
        raise ValueError("At least one analysis block must be enabled")

    return AnalysisConfig(
        source_path=source_path,
        trajectory_zarr=trajectory_zarr,
        trajectory_product=product,
        dataset_label=_nonempty(
            input_section.get("dataset_label", "current"), "input.dataset_label",
        ),
        allow_provisional=_boolean(
            input_section.get("allow_provisional", False), "input.allow_provisional",
        ),
        arrays=_selection(selection.get("arrays", "all"), "selection.arrays", integer=True),
        clusters=_selection(
            selection.get("clusters", "all"), "selection.clusters", integer=False,
        ),
        kinematics=KinematicsConfig(
            method=method,
            total_span_minutes=total_span,
            maximum_source_gap_minutes=_positive(
                kinematics.get("maximum_source_gap_minutes", 30),
                "kinematics.maximum_source_gap_minutes",
            ),
            require_full_stencil_support=full_support,
            speed_mask_m_s=_optional_positive(
                kinematics.get("speed_mask_m_s"), "kinematics.speed_mask_m_s",
            ),
        ),
        trajectory_plots=trajectory_plots,
        trajectory_statistics=trajectory_statistics,
        epsilon=epsilon,
        epsilon_diagnostics=diagnostics,
        fsle=fsle,
        output_root=output_root,
    )


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _analysis_block(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    result = dict(value)
    result["enabled"] = _boolean(result.get("enabled", False), f"{name}.enabled")
    return result


def _selection(
    value: Any,
    name: str,
    *,
    integer: bool,
) -> str | tuple[Any, ...]:
    if value == "all":
        return "all"
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be 'all' or a nonempty list")
    if integer:
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
            raise ValueError(f"{name} must contain positive integers")
        parsed = tuple(value)
    else:
        parsed = tuple(_nonempty(item, name) for item in value)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} contains duplicates")
    return parsed


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _optional_positive(value: Any, name: str) -> float | None:
    return None if value is None else _positive(value, name)
