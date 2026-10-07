"""Derived, provenance-linked diagnostics for completed epsilon runs."""

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

import numpy as np
import pandas as pd
import yaml

from drifterlab import __version__
from drifterlab.epsilon import bin_indices
from drifterlab.epsilon_figures import (
    plot_cluster_influence_audit,
    plot_cluster_pair_relative_motion,
    plot_cluster_time_scale_contributions,
    plot_cluster_trajectory_window,
    plot_cluster_velocity_availability,
)


DIAGNOSTIC_SCHEMA_VERSION = "1.0"
DIAGNOSTIC_ALGORITHM_VERSION = "epsilon-cluster-diagnostics-v5"


@dataclass(frozen=True)
class TrajectoryFigureConfig:
    coordinates: str
    equal_aspect: bool
    show_analysis_valid: bool
    show_invalid_reconstructed_positions: bool
    label_platforms: bool
    mark_activation: bool
    mark_window_boundaries: bool
    dpi: int


@dataclass(frozen=True)
class VelocityAvailabilityFigureConfig:
    enabled: bool
    dpi: int


@dataclass(frozen=True)
class PairRelativeMotionFigureConfig:
    enabled: bool
    derivative_span: str
    dpi: int


@dataclass(frozen=True)
class TimeScaleContributionsFigureConfig:
    enabled: bool
    dpi: int


@dataclass(frozen=True)
class InfluenceAuditFigureConfig:
    enabled: bool
    small_scale_maximum_m: float
    dominant_observation_fraction: float
    dominant_platform_relative_change: float
    dpi: int


@dataclass(frozen=True)
class EpsilonDiagnosticsConfig:
    source_path: Path
    epsilon_run: Path
    arrays: str | tuple[int, ...]
    clusters: str | tuple[str, ...]
    window: str
    mark_elapsed_hours: tuple[float, ...]
    trajectory: TrajectoryFigureConfig
    velocity_availability: VelocityAvailabilityFigureConfig
    pair_relative_motion: PairRelativeMotionFigureConfig
    time_scale_contributions: TimeScaleContributionsFigureConfig
    influence_audit: InfluenceAuditFigureConfig
    output_root: Path

    def effective(self) -> dict[str, Any]:
        return {
            "input": {"epsilon_run": str(self.epsilon_run)},
            "selection": {
                "arrays": self.arrays if isinstance(self.arrays, str) else list(self.arrays),
                "clusters": (
                    self.clusters if isinstance(self.clusters, str) else list(self.clusters)
                ),
            },
            "time": {
                "window": self.window,
                "mark_elapsed_hours": list(self.mark_elapsed_hours),
            },
            "figures": {
                "trajectory_window": asdict(self.trajectory),
                "velocity_availability": asdict(self.velocity_availability),
                "pair_relative_motion": asdict(self.pair_relative_motion),
                "time_scale_contributions": asdict(self.time_scale_contributions),
                "influence_audit": asdict(self.influence_audit),
            },
            "output": {"root": str(self.output_root)},
        }


@dataclass(frozen=True)
class EpsilonDiagnosticsResult:
    run_directory: Path
    manifest_path: Path
    figure_count: int
    cluster_count: int
    diagnostic_id: str

    def format(self) -> str:
        return "\n".join((
            f"Diagnostic run: {self.run_directory}",
            f"Figures: {self.figure_count}",
            f"Selected inventory rows: {self.cluster_count}",
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


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_number(value: Any, name: str) -> float:
    result = _nonnegative(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _nonnegative(value, name)
    if result > 1:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_epsilon_diagnostics_config(path: str | Path) -> EpsilonDiagnosticsConfig:
    source_path = Path(path).expanduser().resolve()
    values = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    root = _mapping(
        values,
        {"input", "selection", "time", "figures", "output"},
        "configuration",
    )
    input_section = _mapping(root.get("input", {}), {"epsilon_run"}, "input")
    selection = _mapping(
        root.get("selection", {}), {"arrays", "clusters"}, "selection",
    )
    time = _mapping(
        root.get("time", {}), {"window", "mark_elapsed_hours"}, "time",
    )
    figures = _mapping(
        root.get("figures", {}),
        {
            "trajectory_window", "velocity_availability", "pair_relative_motion",
            "time_scale_contributions", "influence_audit",
        },
        "figures",
    )
    trajectory = _mapping(
        figures.get("trajectory_window", {}),
        {
            "enabled", "coordinates", "equal_aspect", "show_analysis_valid",
            "show_invalid_reconstructed_positions", "label_platforms",
            "mark_activation", "mark_window_boundaries", "dpi",
        },
        "figures.trajectory_window",
    )
    velocity_availability = _mapping(
        figures.get("velocity_availability", {}),
        {"enabled", "dpi"},
        "figures.velocity_availability",
    )
    pair_relative_motion = _mapping(
        figures.get("pair_relative_motion", {}),
        {"enabled", "derivative_span", "dpi"},
        "figures.pair_relative_motion",
    )
    time_scale_contributions = _mapping(
        figures.get("time_scale_contributions", {}),
        {"enabled", "dpi"},
        "figures.time_scale_contributions",
    )
    influence_audit = _mapping(
        figures.get("influence_audit", {}),
        {
            "enabled", "small_scale_maximum_m",
            "dominant_observation_fraction", "dominant_platform_relative_change",
            "dpi",
        },
        "figures.influence_audit",
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

    clusters_raw = selection.get("clusters", "all")
    if clusters_raw == "all":
        clusters: str | tuple[str, ...] = "all"
    elif isinstance(clusters_raw, list) and clusters_raw:
        clusters = tuple(_nonempty(item, "selection.clusters") for item in clusters_raw)
        if len(set(clusters)) != len(clusters):
            raise ValueError("selection.clusters contains duplicates")
    else:
        raise ValueError("selection.clusters must be 'all' or a nonempty list")

    window = _nonempty(time.get("window", "maximum_configured"), "time.window")
    if window != "maximum_configured":
        raise ValueError("time.window currently supports maximum_configured only")
    marks_raw = time.get("mark_elapsed_hours", [0, 6, 12])
    if not isinstance(marks_raw, list):
        raise ValueError("time.mark_elapsed_hours must be a list")
    marks = tuple(_nonnegative(value, "time.mark_elapsed_hours") for value in marks_raw)
    if tuple(sorted(set(marks))) != marks:
        raise ValueError("time.mark_elapsed_hours must be unique and increasing")

    if not _boolean(trajectory.get("enabled", True), "figures.trajectory_window.enabled"):
        raise ValueError("The trajectory_window diagnostic must be enabled")
    coordinates = _nonempty(
        trajectory.get("coordinates", "local_projected_xy"),
        "figures.trajectory_window.coordinates",
    )
    if coordinates != "local_projected_xy":
        raise ValueError("trajectory coordinates currently support local_projected_xy only")
    if not _boolean(
        trajectory.get("equal_aspect", True), "figures.trajectory_window.equal_aspect",
    ):
        raise ValueError("trajectory_window.equal_aspect must remain true")
    if not _boolean(
        trajectory.get("show_analysis_valid", True),
        "figures.trajectory_window.show_analysis_valid",
    ):
        raise ValueError("trajectory_window.show_analysis_valid must remain true")

    epsilon_run = resolved(input_section.get("epsilon_run"), "input.epsilon_run")
    output_root = resolved(output.get("root"), "output.root")
    if epsilon_run == output_root or epsilon_run in output_root.parents:
        raise ValueError("Diagnostic output root must be outside the source epsilon run")
    return EpsilonDiagnosticsConfig(
        source_path=source_path,
        epsilon_run=epsilon_run,
        arrays=arrays,
        clusters=clusters,
        window=window,
        mark_elapsed_hours=marks,
        trajectory=TrajectoryFigureConfig(
            coordinates=coordinates,
            equal_aspect=True,
            show_analysis_valid=True,
            show_invalid_reconstructed_positions=_boolean(
                trajectory.get("show_invalid_reconstructed_positions", True),
                "figures.trajectory_window.show_invalid_reconstructed_positions",
            ),
            label_platforms=_boolean(
                trajectory.get("label_platforms", True),
                "figures.trajectory_window.label_platforms",
            ),
            mark_activation=_boolean(
                trajectory.get("mark_activation", True),
                "figures.trajectory_window.mark_activation",
            ),
            mark_window_boundaries=_boolean(
                trajectory.get("mark_window_boundaries", True),
                "figures.trajectory_window.mark_window_boundaries",
            ),
            dpi=_positive_integer(trajectory.get("dpi", 160), "figures.trajectory_window.dpi"),
        ),
        velocity_availability=VelocityAvailabilityFigureConfig(
            enabled=_boolean(
                velocity_availability.get("enabled", False),
                "figures.velocity_availability.enabled",
            ),
            dpi=_positive_integer(
                velocity_availability.get("dpi", 160),
                "figures.velocity_availability.dpi",
            ),
        ),
        pair_relative_motion=PairRelativeMotionFigureConfig(
            enabled=_boolean(
                pair_relative_motion.get("enabled", False),
                "figures.pair_relative_motion.enabled",
            ),
            derivative_span=_nonempty(
                pair_relative_motion.get(
                    "derivative_span", "source_analysis_velocity",
                ),
                "figures.pair_relative_motion.derivative_span",
            ),
            dpi=_positive_integer(
                pair_relative_motion.get("dpi", 160),
                "figures.pair_relative_motion.dpi",
            ),
        ),
        time_scale_contributions=TimeScaleContributionsFigureConfig(
            enabled=_boolean(
                time_scale_contributions.get("enabled", False),
                "figures.time_scale_contributions.enabled",
            ),
            dpi=_positive_integer(
                time_scale_contributions.get("dpi", 160),
                "figures.time_scale_contributions.dpi",
            ),
        ),
        influence_audit=InfluenceAuditFigureConfig(
            enabled=_boolean(
                influence_audit.get("enabled", False),
                "figures.influence_audit.enabled",
            ),
            small_scale_maximum_m=_positive_number(
                influence_audit.get("small_scale_maximum_m", 1000),
                "figures.influence_audit.small_scale_maximum_m",
            ),
            dominant_observation_fraction=_fraction(
                influence_audit.get("dominant_observation_fraction", 0.5),
                "figures.influence_audit.dominant_observation_fraction",
            ),
            dominant_platform_relative_change=_nonnegative(
                influence_audit.get("dominant_platform_relative_change", 0.5),
                "figures.influence_audit.dominant_platform_relative_change",
            ),
            dpi=_positive_integer(
                influence_audit.get("dpi", 160),
                "figures.influence_audit.dpi",
            ),
        ),
        output_root=output_root,
    )


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
        raise FileExistsError(f"Diagnostic output already exists: {target}; use --overwrite")
    backup = target.with_name(f".{target.name}.{token}.backup")
    target.rename(backup)
    try:
        temporary.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    _remove_generated(backup, target.parent)


def _selected(value: str, selection: str | tuple[str, ...]) -> bool:
    return selection == "all" or value in selection


def _source_window_end(resolved: dict[str, Any]) -> float:
    windows = resolved.get("analysis", {}).get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Source resolved config has no analysis windows")
    ends = [float(item["end_hour"]) for item in windows]
    if any(not pd.notna(value) or value <= 0 for value in ends):
        raise ValueError("Source resolved config contains invalid window endpoints")
    return max(ends)


def _source_activation_threshold(resolved: dict[str, Any], assigned: int) -> int:
    raw = resolved.get("activation", {}).get("minimum_active_members")
    if raw == "all":
        return assigned
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 2:
        raise ValueError("Source resolved config has an invalid activation threshold")
    return raw


def _source_derivative_span_minutes(
    resolved: dict[str, Any], derivative_span: str,
) -> float:
    if derivative_span != "source_analysis_velocity":
        raise ValueError(
            "pair_relative_motion.derivative_span currently supports "
            "source_analysis_velocity only"
        )
    raw = resolved.get("analysis", {}).get("total_span_minutes")
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise ValueError("Source resolved config has no numeric analysis velocity span")
    result = float(raw)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Source resolved config has an invalid analysis velocity span")
    return result


def _source_time_scale_parameters(
    resolved: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, str]:
    bins = resolved.get("separation_bins", {})
    edges = np.asarray(bins.get("resolved_edges_m", []), dtype=float)
    centers = np.asarray(bins.get("resolved_nominal_centers_m", []), dtype=float)
    if (
        len(edges) < 2
        or centers.shape != (len(edges) - 1,)
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("Source resolved config has invalid separation-bin geometry")
    coefficient_raw = resolved.get("analysis", {}).get("four_fifths_coefficient")
    if isinstance(coefficient_raw, bool) or not isinstance(coefficient_raw, Real):
        raise ValueError("Source resolved config has no numeric four-fifths coefficient")
    coefficient = float(coefficient_raw)
    if not math.isfinite(coefficient):
        raise ValueError("Source resolved config has an invalid four-fifths coefficient")
    windows = resolved.get("analysis", {}).get("windows", [])
    maximum_window = max(windows, key=lambda item: float(item["end_hour"]))
    if float(maximum_window.get("start_hour", 0)) != 0:
        raise ValueError("Time-scale diagnostic requires a maximum window starting at zero")
    window_id = _nonempty(maximum_window.get("id"), "source maximum window id")
    return edges, centers, coefficient, window_id


def _build_time_scale_contributions(
    observations: pd.DataFrame,
    *,
    edges_m: np.ndarray,
    centers_m: np.ndarray,
    coefficient: float,
    source_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    frame = observations.copy().reset_index(drop=True)
    frame["separation_bin_index"] = bin_indices(
        frame.separation_m.to_numpy(dtype=float), edges_m,
    )
    frame = frame[
        (frame.separation_bin_index >= 0)
        & (frame.separation_bin_index < len(edges_m) - 1)
    ].copy()
    if frame.empty:
        return pd.DataFrame(), 0.0
    frame["cube_m3_s3"] = frame.delta_u_l_m_s.to_numpy(dtype=float) ** 3
    frame["absolute_cube_m3_s3"] = frame.cube_m3_s3.abs()
    separation_denominators = frame.groupby("separation_bin_index").separation_m.sum()
    rows: list[dict[str, Any]] = []
    for (time_utc, bin_index), group in frame.groupby(
        ["time_utc", "separation_bin_index"], sort=True,
    ):
        bin_index = int(bin_index)
        cube_sum = float(group.cube_m3_s3.sum())
        absolute_cube_sum = float(group.absolute_cube_m3_s3.sum())
        influential = group.sort_values(
            ["absolute_cube_m3_s3", "pair_id"], ascending=[False, True],
        ).iloc[0]
        denominator = float(separation_denominators.loc[bin_index])
        platform_ids = set(group.platform_id_1.astype(str)) | set(
            group.platform_id_2.astype(str)
        )
        rows.append({
            "time_utc": time_utc,
            "cluster_age_hours": float(group.cluster_age_hours.iloc[0]),
            "separation_bin_id": f"bin_{bin_index:03d}",
            "separation_bin_index": bin_index,
            "separation_bin_lower_m": float(edges_m[bin_index]),
            "separation_bin_upper_m": float(edges_m[bin_index + 1]),
            "separation_bin_nominal_center_m": float(centers_m[bin_index]),
            "pair_observation_count": len(group),
            "unique_pair_count": int(group.pair_id.nunique()),
            "platform_count": len(platform_ids),
            "separation_sum_m": float(group.separation_m.sum()),
            "full_window_bin_separation_sum_m": denominator,
            "longitudinal_cube_sum_m3_s3": cube_sum,
            "positive_longitudinal_cube_sum_m3_s3": float(
                group.loc[group.cube_m3_s3 > 0, "cube_m3_s3"].sum()
            ),
            "negative_longitudinal_cube_sum_m3_s3": float(
                group.loc[group.cube_m3_s3 < 0, "cube_m3_s3"].sum()
            ),
            "absolute_longitudinal_cube_sum_m3_s3": absolute_cube_sum,
            "cube_cancellation_ratio": (
                abs(cube_sum) / absolute_cube_sum if absolute_cube_sum > 0 else None
            ),
            "maximum_absolute_cube_fraction": (
                float(influential.absolute_cube_m3_s3) / absolute_cube_sum
                if absolute_cube_sum > 0 else None
            ),
            "most_influential_pair_id": str(influential.pair_id),
            "most_influential_platform_id_1": str(influential.platform_id_1),
            "most_influential_platform_id_2": str(influential.platform_id_2),
            "most_influential_separation_m": float(influential.separation_m),
            "most_influential_delta_u_l_m_s": float(influential.delta_u_l_m_s),
            "most_influential_cube_m3_s3": float(influential.cube_m3_s3),
            "epsilon_additive_contribution_m2_s3": coefficient * cube_sum / denominator,
        })
    contributions = pd.DataFrame(rows)
    reconstructed = contributions.groupby("separation_bin_index")[
        "epsilon_additive_contribution_m2_s3"
    ].sum()
    finite_source = source_summary[np.isfinite(source_summary.epsilon_eff_m2_s3)]
    errors: list[float] = []
    for row in finite_source.itertuples(index=False):
        bin_index = int(row.separation_bin_index)
        if bin_index not in reconstructed.index:
            raise ValueError(
                f"Saved epsilon exists for bin {bin_index}, but its contribution is absent"
            )
        errors.append(abs(
            float(reconstructed.loc[bin_index]) - float(row.epsilon_eff_m2_s3)
        ))
    maximum_error = max(errors, default=0.0)
    if maximum_error > 1e-12:
        raise ValueError(
            "Time-scale contribution identity does not reproduce saved epsilon; "
            f"maximum absolute error={maximum_error:.6g}"
        )
    contributions["epsilon_identity_maximum_absolute_error_m2_s3"] = maximum_error
    return contributions, maximum_error


def _as_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _build_influence_audit(
    source_summary: pd.DataFrame,
    lopo_details: pd.DataFrame,
    *,
    small_scale_maximum_m: float,
    dominant_observation_fraction: float,
    dominant_platform_relative_change: float,
) -> pd.DataFrame:
    selected = source_summary[
        np.isfinite(source_summary.epsilon_eff_m2_s3)
        & (source_summary.epsilon_eff_m2_s3 < 0)
        & (source_summary.separation_bin_upper_m <= small_scale_maximum_m)
    ].copy()
    rows: list[dict[str, Any]] = []
    for source_row in selected.to_dict("records"):
        bin_id = str(source_row["separation_bin_id"])
        removals = lopo_details[lopo_details.separation_bin_id.astype(str) == bin_id].copy()
        finite_removals = removals[np.isfinite(removals.epsilon_after_removal_m2_s3)]
        reversal_mask = removals.sign_reversal.map(_as_boolean) if len(removals) else pd.Series(
            dtype=bool,
        )
        supported_mask = (
            removals.support_ok_after_removal.map(_as_boolean)
            if len(removals) else pd.Series(dtype=bool)
        )
        event_after = source_row.get("epsilon_without_most_extreme_m2_s3")
        event_reversal = bool(pd.notna(event_after) and float(event_after) > 0)
        platform_reversal = bool(reversal_mask.any())
        low_support = not _as_boolean(source_row.get("support_ok", False))
        observation_fraction = source_row.get("maximum_absolute_cube_fraction")
        platform_relative_change = source_row.get("maximum_lopo_relative_change")
        dominant_observation = bool(
            pd.notna(observation_fraction)
            and float(observation_fraction) >= dominant_observation_fraction
        )
        dominant_platform = bool(
            pd.notna(platform_relative_change)
            and float(platform_relative_change) >= dominant_platform_relative_change
        )
        event_evaluable = pd.notna(event_after)
        platform_evaluable = len(finite_removals) > 0
        persistent = bool(
            event_evaluable and platform_evaluable
            and not event_reversal and not platform_reversal
        )
        if low_support:
            classification = "low_support"
        elif event_reversal and platform_reversal:
            classification = "event_and_platform_sign_sensitive"
        elif event_reversal:
            classification = "event_sign_sensitive"
        elif platform_reversal:
            classification = "platform_sign_sensitive"
        elif dominant_observation and dominant_platform:
            classification = "observation_and_platform_concentrated"
        elif dominant_observation:
            classification = "observation_concentrated"
        elif dominant_platform:
            classification = "platform_sensitive_without_sign_reversal"
        elif persistent:
            classification = "persistent_negative_under_tested_removals"
        else:
            classification = "influence_test_incomplete"
        reversal_platforms = (
            removals.loc[reversal_mask, "removed_platform_id"].astype(str).tolist()
            if len(removals) else []
        )
        retained_columns = (
            "array_id", "cluster_id", "analysis_type", "window_id",
            "activation_time_utc", "window_start_hour", "window_end_hour",
            "separation_bin_id", "separation_bin_index", "separation_bin_lower_m",
            "separation_bin_upper_m", "separation_bin_nominal_center_m",
            "mean_separation_m", "epsilon_eff_m2_s3", "support_ok",
            "support_status", "observation_count", "unique_pair_count",
            "platform_count", "unique_utc_count", "maximum_absolute_cube_fraction",
            "top_five_absolute_cube_fraction",
            "absolute_cube_effective_observation_count", "cube_cancellation_ratio",
            "most_influential_pair_id", "most_influential_time",
            "epsilon_without_most_extreme_m2_s3",
            "epsilon_change_without_most_extreme_m2_s3",
            "leave_one_platform_out_evaluable_count", "most_influential_platform_id",
            "maximum_lopo_absolute_change_m2_s3", "maximum_lopo_relative_change",
            "most_influential_platform_removal_support_ok", "lopo_any_sign_reversal",
        )
        result = {column: source_row.get(column) for column in retained_columns}
        result.update({
            "small_scale_maximum_m": small_scale_maximum_m,
            "dominant_observation_fraction_threshold": dominant_observation_fraction,
            "dominant_platform_relative_change_threshold": (
                dominant_platform_relative_change
            ),
            "low_support": low_support,
            "extreme_observation_evaluable": event_evaluable,
            "extreme_observation_sign_reversal": event_reversal,
            "dominant_observation": dominant_observation,
            "platform_removal_evaluable_count": len(finite_removals),
            "platform_removal_sign_reversal": platform_reversal,
            "platform_removal_sign_reversal_count": int(reversal_mask.sum()),
            "supported_platform_removal_sign_reversal_count": int(
                (reversal_mask & supported_mask).sum()
            ),
            "sign_reversal_removed_platform_ids": ";".join(reversal_platforms),
            "lopo_minimum_epsilon_m2_s3": (
                float(finite_removals.epsilon_after_removal_m2_s3.min())
                if len(finite_removals) else None
            ),
            "lopo_maximum_epsilon_m2_s3": (
                float(finite_removals.epsilon_after_removal_m2_s3.max())
                if len(finite_removals) else None
            ),
            "dominant_platform": dominant_platform,
            "persistent_negative_under_tested_removals": persistent,
            "audit_classification": classification,
        })
        rows.append(result)
    return pd.DataFrame(rows)


def run_epsilon_diagnostics(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> EpsilonDiagnosticsResult:
    """Render trajectory-window diagnostics from an immutable epsilon product."""
    config = load_epsilon_diagnostics_config(config_path)
    report = progress or (lambda message: None)
    source_manifest_path = config.epsilon_run / "manifest.json"
    source_resolved_path = config.epsilon_run / "config_resolved.yaml"
    if not source_manifest_path.is_file() or not source_resolved_path.is_file():
        raise ValueError("input.epsilon_run is not a completed epsilon product")
    source_manifest_sha256 = _file_sha256(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_resolved = yaml.safe_load(source_resolved_path.read_text(encoding="utf-8"))
    source_run_id = _nonempty(source_manifest.get("run_id"), "source manifest run_id")
    window_end_hour = _source_window_end(source_resolved)
    pair_derivative_span_minutes = _source_derivative_span_minutes(
        source_resolved, config.pair_relative_motion.derivative_span,
    ) if config.pair_relative_motion.enabled else None
    if config.time_scale_contributions.enabled or config.influence_audit.enabled:
        (
            separation_edges_m,
            separation_centers_m,
            four_fifths_coefficient,
            maximum_window_id,
        ) = _source_time_scale_parameters(source_resolved)
    else:
        separation_edges_m = separation_centers_m = None
        four_fifths_coefficient = None
        maximum_window_id = None
    if any(value > window_end_hour for value in config.mark_elapsed_hours):
        raise ValueError("Marked elapsed hours cannot exceed the maximum source window")

    identity_payload = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "algorithm_version": DIAGNOSTIC_ALGORITHM_VERSION,
        "drifterlab_version": __version__,
        "source_manifest_sha256": source_manifest_sha256,
        "configuration": config.effective(),
    }
    digest = sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    diagnostic_id = f"epsilon-diagnostics-v1-{digest}"
    target_parent = config.output_root / source_run_id
    target = target_parent / diagnostic_id
    if target.exists() and not overwrite:
        raise FileExistsError(f"Diagnostic output already exists: {target}; use --overwrite")
    target_parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = target.with_name(f".{target.name}.{token}.tmp")
    temporary.mkdir()

    products: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    contribution_parts: list[pd.DataFrame] = []
    influence_audit_parts: list[pd.DataFrame] = []
    source_files: dict[str, dict[str, Any]] = {}
    try:
        array_directories = sorted(config.epsilon_run.glob("array_[0-9][0-9][0-9]"))
        available_arrays = {
            int(path.name.removeprefix("array_")): path for path in array_directories
        }
        selected_arrays = (
            tuple(sorted(available_arrays))
            if config.arrays == "all" else config.arrays
        )
        missing_arrays = sorted(set(selected_arrays) - set(available_arrays))
        if missing_arrays:
            raise ValueError(f"Selected arrays are absent from source run: {missing_arrays}")

        for array_id in selected_arrays:
            source_array = available_arrays[array_id]
            clusters_path = source_array / "clusters.csv"
            diagnostics_path = source_array / "data" / "platform_diagnostics.parquet"
            pair_observations_path = source_array / "data" / "pair_observations.parquet"
            epsilon_by_scale_path = source_array / "epsilon_by_scale.csv"
            lopo_path = source_array / "data" / "leave_one_platform_out.parquet"
            if not clusters_path.is_file() or not diagnostics_path.is_file():
                raise ValueError(f"Array {array_id} lacks required diagnostic source tables")
            needs_pairs = (
                config.pair_relative_motion.enabled
                or config.time_scale_contributions.enabled
            )
            if needs_pairs and not pair_observations_path.is_file():
                raise ValueError(
                    f"Array {array_id} lacks pair_observations.parquet required by "
                    "the configured pair diagnostics"
                )
            needs_scale_summary = (
                config.time_scale_contributions.enabled or config.influence_audit.enabled
            )
            if needs_scale_summary and not epsilon_by_scale_path.is_file():
                raise ValueError(
                    f"Array {array_id} lacks epsilon_by_scale.csv required by "
                    "the configured scale diagnostics"
                )
            if config.influence_audit.enabled and not lopo_path.is_file():
                raise ValueError(
                    f"Array {array_id} lacks leave_one_platform_out.parquet required by "
                    "influence_audit"
                )
            source_paths = [clusters_path, diagnostics_path]
            if needs_pairs:
                source_paths.append(pair_observations_path)
            if needs_scale_summary:
                source_paths.append(epsilon_by_scale_path)
            if config.influence_audit.enabled:
                source_paths.append(lopo_path)
            for path in source_paths:
                relative_native = str(path.relative_to(config.epsilon_run))
                relative = relative_native.replace("\\", "/")
                actual_hash = _file_sha256(path)
                source_products = source_manifest.get("products", {})
                expected = source_products.get(relative, source_products.get(relative_native, {})).get(
                    "sha256"
                )
                if expected is not None and actual_hash != expected:
                    raise ValueError(f"Source epsilon product hash mismatch: {relative}")
                source_files[relative] = {"sha256": actual_hash, "bytes": path.stat().st_size}

            inventory = pd.read_csv(clusters_path, dtype={"cluster_id": str})
            diagnostics = pd.read_parquet(diagnostics_path)
            required = {
                "array_id", "cluster_id", "platform_id", "time_utc",
                "cluster_age_hours", "x_m", "y_m", "analysis_valid",
            }
            if config.velocity_availability.enabled:
                required |= {
                    "reference_valid", "analysis_u_m_s", "analysis_v_m_s",
                }
            missing = required - set(diagnostics.columns)
            if missing:
                raise ValueError(
                    f"Array {array_id} platform diagnostics lack columns: {sorted(missing)}"
                )
            diagnostics["cluster_id"] = diagnostics.cluster_id.astype(str)
            diagnostics["platform_id"] = diagnostics.platform_id.astype(str)
            diagnostics["time_utc"] = pd.to_datetime(diagnostics.time_utc, utc=True)
            pair_observations: pd.DataFrame | None = None
            if needs_pairs:
                pair_observations = pd.read_parquet(pair_observations_path)
                pair_required = {
                    "array_id", "cluster_id", "pair_id", "platform_id_1",
                    "platform_id_2", "time_utc", "cluster_age_hours",
                    "separation_m", "delta_u_l_m_s", "delta_u_t_m_s",
                }
                pair_missing = pair_required - set(pair_observations.columns)
                if pair_missing:
                    raise ValueError(
                        f"Array {array_id} pair observations lack columns: "
                        f"{sorted(pair_missing)}"
                    )
                pair_observations["cluster_id"] = pair_observations.cluster_id.astype(str)
                pair_observations["pair_id"] = pair_observations.pair_id.astype(str)
                pair_observations["platform_id_1"] = (
                    pair_observations.platform_id_1.astype(str)
                )
                pair_observations["platform_id_2"] = (
                    pair_observations.platform_id_2.astype(str)
                )
                pair_observations["time_utc"] = pd.to_datetime(
                    pair_observations.time_utc, utc=True,
                )
            scale_summary: pd.DataFrame | None = None
            if needs_scale_summary:
                scale_summary = pd.read_csv(
                    epsilon_by_scale_path, dtype={"cluster_id": str},
                )
                scale_required = {
                    "cluster_id", "analysis_type", "window_id",
                    "separation_bin_id", "separation_bin_index",
                    "separation_bin_lower_m", "separation_bin_upper_m",
                    "separation_bin_nominal_center_m", "mean_separation_m",
                    "epsilon_eff_m2_s3",
                }
                if config.influence_audit.enabled:
                    scale_required |= {
                        "epsilon_without_most_extreme_m2_s3", "support_ok",
                        "observation_count", "unique_pair_count", "platform_count",
                        "unique_utc_count", "maximum_absolute_cube_fraction",
                        "maximum_lopo_relative_change",
                    }
                scale_missing = scale_required - set(scale_summary.columns)
                if scale_missing:
                    raise ValueError(
                        f"Array {array_id} epsilon_by_scale lacks columns: "
                        f"{sorted(scale_missing)}"
                    )
            lopo_details: pd.DataFrame | None = None
            if config.influence_audit.enabled:
                lopo_details = pd.read_parquet(lopo_path)
                lopo_required = {
                    "cluster_id", "analysis_type", "window_id", "separation_bin_id",
                    "removed_platform_id", "epsilon_after_removal_m2_s3",
                    "sign_reversal", "support_ok_after_removal",
                }
                lopo_missing = lopo_required - set(lopo_details.columns)
                if lopo_missing:
                    raise ValueError(
                        f"Array {array_id} leave-one-platform-out data lack columns: "
                        f"{sorted(lopo_missing)}"
                    )
                lopo_details["cluster_id"] = lopo_details.cluster_id.astype(str)

            for row in inventory.itertuples(index=False):
                cluster_id = str(row.cluster_id)
                if not _selected(cluster_id, config.clusters):
                    continue
                activation = getattr(row, "activation_time_utc", None)
                if activation is None or pd.isna(activation) or not str(activation).strip():
                    summary_rows.append({
                        "array_id": array_id,
                        "cluster_id": cluster_id,
                        "status": "skipped_no_activation",
                        "figure": None,
                        "velocity_availability_figure": None,
                        "pair_relative_motion_figure": None,
                        "time_scale_contributions_figure": None,
                        "influence_audit_figure": None,
                        "retained_platform_count": 0,
                        "retained_row_count": 0,
                    })
                    continue
                selected = diagnostics[
                    (diagnostics.cluster_id == cluster_id)
                    & (diagnostics.cluster_age_hours >= 0)
                    & (diagnostics.cluster_age_hours < window_end_hour)
                ].copy()
                if selected.empty:
                    summary_rows.append({
                        "array_id": array_id,
                        "cluster_id": cluster_id,
                        "status": "skipped_no_rows_in_window",
                        "figure": None,
                        "velocity_availability_figure": None,
                        "pair_relative_motion_figure": None,
                        "time_scale_contributions_figure": None,
                        "influence_audit_figure": None,
                        "retained_platform_count": 0,
                        "retained_row_count": 0,
                    })
                    continue
                safe_cluster = re.sub(r"[^A-Za-z0-9_.-]+", "_", cluster_id)
                figure_path = (
                    temporary / f"array_{array_id:03d}" / "trajectory_window"
                    / f"{safe_cluster}.png"
                )
                plot_cluster_trajectory_window(
                    selected,
                    output_path=figure_path,
                    array_id=array_id,
                    cluster_id=cluster_id,
                    activation_time_utc=str(activation),
                    window_end_hour=window_end_hour,
                    marked_elapsed_hours=config.mark_elapsed_hours,
                    show_invalid_reconstructed_positions=(
                        config.trajectory.show_invalid_reconstructed_positions
                    ),
                    label_platforms=config.trajectory.label_platforms,
                    mark_activation=config.trajectory.mark_activation,
                    mark_window_boundaries=config.trajectory.mark_window_boundaries,
                    dpi=config.trajectory.dpi,
                )
                products.append(figure_path)
                velocity_figure_path: Path | None = None
                if config.velocity_availability.enabled:
                    assigned_member_count = int(getattr(
                        row, "assigned_member_count", selected.platform_id.nunique(),
                    ))
                    activation_threshold = _source_activation_threshold(
                        source_resolved, assigned_member_count,
                    )
                    velocity_figure_path = (
                        temporary / f"array_{array_id:03d}" / "velocity_availability"
                        / f"{safe_cluster}.png"
                    )
                    plot_cluster_velocity_availability(
                        selected,
                        output_path=velocity_figure_path,
                        array_id=array_id,
                        cluster_id=cluster_id,
                        activation_time_utc=str(activation),
                        window_end_hour=window_end_hour,
                        assigned_member_count=assigned_member_count,
                        activation_threshold=activation_threshold,
                        dpi=config.velocity_availability.dpi,
                    )
                    products.append(velocity_figure_path)
                pair_figure_path: Path | None = None
                selected_pairs = pd.DataFrame()
                if pair_observations is not None:
                    selected_pairs = pair_observations[
                        (pair_observations.cluster_id == cluster_id)
                        & (pair_observations.cluster_age_hours >= 0)
                        & (pair_observations.cluster_age_hours < window_end_hour)
                    ].copy()
                    if config.pair_relative_motion.enabled and not selected_pairs.empty:
                        pair_figure_path = (
                            temporary / f"array_{array_id:03d}" / "pair_relative_motion"
                            / f"{safe_cluster}.png"
                        )
                        plot_cluster_pair_relative_motion(
                            selected_pairs,
                            selected[["time_utc", "cluster_age_hours"]],
                            output_path=pair_figure_path,
                            array_id=array_id,
                            cluster_id=cluster_id,
                            activation_time_utc=str(activation),
                            window_end_hour=window_end_hour,
                            derivative_span_minutes=float(pair_derivative_span_minutes),
                            dpi=config.pair_relative_motion.dpi,
                        )
                        products.append(pair_figure_path)
                time_scale_figure_path: Path | None = None
                identity_maximum_error: float | None = None
                top_negative_cell: pd.Series | None = None
                source_cluster_summary = pd.DataFrame()
                if scale_summary is not None:
                    source_cluster_summary = scale_summary[
                        (scale_summary.cluster_id == cluster_id)
                        & (scale_summary.analysis_type == "window")
                        & (scale_summary.window_id == maximum_window_id)
                    ].copy()
                    if source_cluster_summary.empty:
                        raise ValueError(
                            f"No saved scale summary for {cluster_id} {maximum_window_id}"
                        )
                if (
                    config.time_scale_contributions.enabled
                    and not selected_pairs.empty
                    and not source_cluster_summary.empty
                ):
                    contributions, identity_maximum_error = (
                        _build_time_scale_contributions(
                            selected_pairs,
                            edges_m=separation_edges_m,
                            centers_m=separation_centers_m,
                            coefficient=float(four_fifths_coefficient),
                            source_summary=source_cluster_summary,
                        )
                    )
                    if not contributions.empty:
                        contributions.insert(0, "window_id", maximum_window_id)
                        contributions.insert(0, "cluster_id", cluster_id)
                        contributions.insert(0, "array_id", array_id)
                        contribution_parts.append(contributions)
                        top_negative_cell = contributions.loc[
                            contributions.epsilon_additive_contribution_m2_s3.idxmin()
                        ]
                        time_scale_figure_path = (
                            temporary / f"array_{array_id:03d}"
                            / "time_scale_contributions" / f"{safe_cluster}.png"
                        )
                        plot_cluster_time_scale_contributions(
                            contributions,
                            selected[["time_utc", "cluster_age_hours"]],
                            source_cluster_summary,
                            output_path=time_scale_figure_path,
                            array_id=array_id,
                            cluster_id=cluster_id,
                            activation_time_utc=str(activation),
                            window_id=str(maximum_window_id),
                            window_end_hour=window_end_hour,
                            separation_edges_m=separation_edges_m,
                            identity_maximum_absolute_error_m2_s3=(
                                identity_maximum_error
                            ),
                            dpi=config.time_scale_contributions.dpi,
                        )
                        products.append(time_scale_figure_path)
                influence_figure_path: Path | None = None
                cluster_audit = pd.DataFrame()
                if (
                    config.influence_audit.enabled
                    and not source_cluster_summary.empty
                    and lopo_details is not None
                ):
                    cluster_lopo = lopo_details[
                        (lopo_details.cluster_id == cluster_id)
                        & (lopo_details.analysis_type == "window")
                        & (lopo_details.window_id == maximum_window_id)
                    ].copy()
                    cluster_audit = _build_influence_audit(
                        source_cluster_summary,
                        cluster_lopo,
                        small_scale_maximum_m=(
                            config.influence_audit.small_scale_maximum_m
                        ),
                        dominant_observation_fraction=(
                            config.influence_audit.dominant_observation_fraction
                        ),
                        dominant_platform_relative_change=(
                            config.influence_audit.dominant_platform_relative_change
                        ),
                    )
                    if not cluster_audit.empty:
                        influence_audit_parts.append(cluster_audit)
                        influence_figure_path = (
                            temporary / f"array_{array_id:03d}" / "influence_audit"
                            / f"{safe_cluster}.png"
                        )
                        plot_cluster_influence_audit(
                            cluster_audit,
                            output_path=influence_figure_path,
                            array_id=array_id,
                            cluster_id=cluster_id,
                            window_id=str(maximum_window_id),
                            small_scale_maximum_m=(
                                config.influence_audit.small_scale_maximum_m
                            ),
                            dominant_observation_fraction=(
                                config.influence_audit.dominant_observation_fraction
                            ),
                            dominant_platform_relative_change=(
                                config.influence_audit.dominant_platform_relative_change
                            ),
                            dpi=config.influence_audit.dpi,
                        )
                        products.append(influence_figure_path)
                longitudinal = (
                    selected_pairs.delta_u_l_m_s.to_numpy(dtype=float)
                    if not selected_pairs.empty else []
                )
                summary_rows.append({
                    "array_id": array_id,
                    "cluster_id": cluster_id,
                    "status": "available",
                    "figure": str(figure_path.relative_to(temporary)).replace("\\", "/"),
                    "velocity_availability_figure": (
                        None if velocity_figure_path is None else
                        str(velocity_figure_path.relative_to(temporary)).replace("\\", "/")
                    ),
                    "pair_relative_motion_figure": (
                        None if pair_figure_path is None else
                        str(pair_figure_path.relative_to(temporary)).replace("\\", "/")
                    ),
                    "time_scale_contributions_figure": (
                        None if time_scale_figure_path is None else
                        str(time_scale_figure_path.relative_to(temporary)).replace("\\", "/")
                    ),
                    "influence_audit_figure": (
                        None if influence_figure_path is None else
                        str(influence_figure_path.relative_to(temporary)).replace("\\", "/")
                    ),
                    "activation_time_utc": str(activation),
                    "window_end_hour": window_end_hour,
                    "retained_platform_count": int(selected.platform_id.nunique()),
                    "retained_row_count": len(selected),
                    "analysis_valid_row_count": int(selected.analysis_valid.astype(bool).sum()),
                    "pair_observation_count": len(selected_pairs),
                    "pair_count": (
                        int(selected_pairs.pair_id.nunique()) if not selected_pairs.empty else 0
                    ),
                    "separating_pair_observation_fraction": (
                        float((longitudinal > 0).mean()) if len(longitudinal) else None
                    ),
                    "net_longitudinal_cube_sum_m3_s3": (
                        float((longitudinal ** 3).sum()) if len(longitudinal) else None
                    ),
                    "time_scale_identity_maximum_absolute_error_m2_s3": (
                        identity_maximum_error
                    ),
                    "largest_negative_epsilon_cell_time_utc": (
                        None if top_negative_cell is None else top_negative_cell.time_utc
                    ),
                    "largest_negative_epsilon_cell_separation_bin_id": (
                        None if top_negative_cell is None else
                        top_negative_cell.separation_bin_id
                    ),
                    "largest_negative_epsilon_cell_contribution_m2_s3": (
                        None if top_negative_cell is None else
                        top_negative_cell.epsilon_additive_contribution_m2_s3
                    ),
                    "largest_negative_epsilon_cell_most_influential_pair_id": (
                        None if top_negative_cell is None else
                        top_negative_cell.most_influential_pair_id
                    ),
                    "largest_negative_epsilon_cell_platform_id_1": (
                        None if top_negative_cell is None else
                        top_negative_cell.most_influential_platform_id_1
                    ),
                    "largest_negative_epsilon_cell_platform_id_2": (
                        None if top_negative_cell is None else
                        top_negative_cell.most_influential_platform_id_2
                    ),
                    "negative_small_scale_bin_count": len(cluster_audit),
                    "low_support_negative_small_scale_bin_count": (
                        int(cluster_audit.low_support.sum())
                        if not cluster_audit.empty else 0
                    ),
                    "event_sign_sensitive_bin_count": (
                        int(cluster_audit.extreme_observation_sign_reversal.sum())
                        if not cluster_audit.empty else 0
                    ),
                    "platform_sign_sensitive_bin_count": (
                        int(cluster_audit.platform_removal_sign_reversal.sum())
                        if not cluster_audit.empty else 0
                    ),
                    "persistent_negative_bin_count": (
                        int(cluster_audit.persistent_negative_under_tested_removals.sum())
                        if not cluster_audit.empty else 0
                    ),
                    "first_time_utc": selected.time_utc.min().isoformat(),
                    "last_time_utc": selected.time_utc.max().isoformat(),
                })
                report(f"Array {array_id} {cluster_id}: configured diagnostics rendered")

        summary = pd.DataFrame(summary_rows)
        if summary.empty:
            raise ValueError("No clusters matched the diagnostic selection")
        summary_path = temporary / "diagnostic_figures.csv"
        summary.to_csv(summary_path, index=False)
        products.append(summary_path)
        time_scale_row_count = 0
        if config.time_scale_contributions.enabled:
            if not contribution_parts:
                raise ValueError("No time-scale contributions were available")
            contribution_table = pd.concat(contribution_parts, ignore_index=True)
            contribution_table = contribution_table.sort_values(
                ["array_id", "cluster_id", "time_utc", "separation_bin_index"],
            ).reset_index(drop=True)
            contribution_path = temporary / "data" / "time_scale_contributions.parquet"
            contribution_path.parent.mkdir(parents=True, exist_ok=True)
            contribution_table.to_parquet(contribution_path, index=False)
            products.append(contribution_path)
            time_scale_row_count = len(contribution_table)
        influence_audit_row_count = 0
        influence_classification_counts: dict[str, int] = {}
        if config.influence_audit.enabled:
            if not influence_audit_parts:
                raise ValueError("No negative small-scale bins were available for audit")
            influence_table = pd.concat(influence_audit_parts, ignore_index=True)
            influence_table = influence_table.sort_values(
                ["array_id", "cluster_id", "separation_bin_index"],
            ).reset_index(drop=True)
            influence_path = temporary / "influence_audit.csv"
            influence_table.to_csv(influence_path, index=False)
            products.append(influence_path)
            influence_audit_row_count = len(influence_table)
            influence_classification_counts = {
                str(key): int(value)
                for key, value in influence_table.audit_classification.value_counts().items()
            }
        resolved_path = temporary / "config_resolved.yaml"
        resolved = {
            **config.effective(),
            "diagnostic_id": diagnostic_id,
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "diagnostic_algorithm_version": DIAGNOSTIC_ALGORITHM_VERSION,
            "resolved_window_end_hour": window_end_hour,
            "source_run_id": source_run_id,
            "source_manifest_sha256": source_manifest_sha256,
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
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "algorithm_version": DIAGNOSTIC_ALGORITHM_VERSION,
            "drifterlab_version": __version__,
            "processing_time_utc": datetime.now(timezone.utc).isoformat(),
            "diagnostic_id": diagnostic_id,
            "configuration_path": str(config.source_path),
            "effective_configuration": config.effective(),
            "source": {
                "epsilon_run": str(config.epsilon_run),
                "run_id": source_run_id,
                "manifest_sha256": source_manifest_sha256,
                "files": source_files,
            },
            "analysis": {
                "window_end_hour": window_end_hour,
                "available_cluster_count": int((summary.status == "available").sum()),
                "trajectory_figure_count": int(summary.figure.notna().sum()),
                "velocity_availability_figure_count": int(
                    summary.velocity_availability_figure.notna().sum()
                ),
                "pair_relative_motion_figure_count": int(
                    summary.pair_relative_motion_figure.notna().sum()
                ),
                "time_scale_contributions_figure_count": int(
                    summary.time_scale_contributions_figure.notna().sum()
                ),
                "time_scale_contribution_row_count": time_scale_row_count,
                "influence_audit_figure_count": int(
                    summary.influence_audit_figure.notna().sum()
                ),
                "influence_audit_row_count": influence_audit_row_count,
                "influence_audit_classification_counts": (
                    influence_classification_counts
                ),
                "available_figure_count": int(
                    summary.figure.notna().sum()
                    + summary.velocity_availability_figure.notna().sum()
                    + summary.pair_relative_motion_figure.notna().sum()
                    + summary.time_scale_contributions_figure.notna().sum()
                    + summary.influence_audit_figure.notna().sum()
                ),
                "selected_inventory_count": len(summary),
            },
            "products": product_manifest,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        json.loads(manifest_path.read_text(encoding="utf-8"))
        _publish(temporary, target, token=token, overwrite=overwrite)
        return EpsilonDiagnosticsResult(
            run_directory=target,
            manifest_path=target / "manifest.json",
            figure_count=int(
                summary.figure.notna().sum()
                + summary.velocity_availability_figure.notna().sum()
                + summary.pair_relative_motion_figure.notna().sum()
                + summary.time_scale_contributions_figure.notna().sum()
                + summary.influence_audit_figure.notna().sum()
            ),
            cluster_count=len(summary),
            diagnostic_id=diagnostic_id,
        )
    finally:
        if temporary.exists():
            _remove_generated(temporary, target.parent)
