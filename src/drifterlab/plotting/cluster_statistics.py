"""Small-multiple figures for within-array cluster statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

from drifterlab.postprocessing.cluster_statistics import ClusterStatisticsArrayResult
from drifterlab.postprocessing.config import ClusterStatisticsConfig, percentile_suffix

CLUSTER_STATISTICS_FIGURES = (
    "velocity_distributions.png",
    "absolute_displacement.png",
    "pair_separation.png",
    "relative_dispersion.png",
    "cluster_extent_and_shape.png",
    "cluster_orientation.png",
)


def _save(figure: Any, path: Path, *, dpi: int, policy: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")
    try:
        figure.savefig(
            temporary,
            dpi=dpi,
            bbox_inches="tight",
            metadata={"Description": f"Cluster membership policy: {policy}"},
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return path


def _policy(config: ClusterStatisticsConfig) -> str:
    return (
        "stop at first member loss" if config.stop_on_member_loss
        else "dynamic active membership"
    )


def _colors(result: ClusterStatisticsArrayResult) -> dict[str, Any]:
    cmap = colormaps["tab20"]
    samples = (
        np.linspace(0, 1, len(result.cluster_ids))
        if len(result.cluster_ids) > 1 else np.asarray([0.05])
    )
    return {
        cluster_id: cmap(float(sample))
        for cluster_id, sample in zip(result.cluster_ids, samples)
    }


def _cluster_frame(result: ClusterStatisticsArrayResult, cluster_id: str) -> pd.DataFrame:
    frame = result.timeseries[result.timeseries.cluster_id == cluster_id].copy()
    frame["plot_time"] = pd.to_datetime(frame.time, utc=True)
    return frame


def _cluster_title(result: ClusterStatisticsArrayResult, cluster_id: str) -> str:
    row = result.summary[result.summary.cluster_id == cluster_id].iloc[0]
    suffix = " · singleton" if int(row.assigned_cluster_size) == 1 else ""
    return f"{cluster_id} · assigned n={int(row.assigned_cluster_size)}{suffix}"


def _mark_unavailable(axes: Any, text: str) -> None:
    axes.text(
        0.5, 0.5, text, transform=axes.transAxes, ha="center", va="center",
        fontsize=9, color="0.35",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )


def _series(frame: pd.DataFrame, column: str, *, positive: bool = False) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=float, copy=True,
    )
    if positive:
        values[values <= 0] = np.nan
    return values


def _plot_percentiles(
    axes: Any, frame: pd.DataFrame, *, prefix: str, unit: str,
    percentiles: tuple[float, ...], color: Any, positive: bool = False,
) -> None:
    requested = set(percentiles)
    times = frame.plot_time
    special = {0.0, 25.0, 50.0, 75.0, 100.0}.issubset(requested)
    if special:
        p000 = _series(frame, f"{prefix}_p000_{unit}", positive=positive)
        p025 = _series(frame, f"{prefix}_p025_{unit}", positive=positive)
        p050 = _series(frame, f"{prefix}_p050_{unit}", positive=positive)
        p075 = _series(frame, f"{prefix}_p075_{unit}", positive=positive)
        p100 = _series(frame, f"{prefix}_p100_{unit}", positive=positive)
        axes.fill_between(times, p000, p100, color=color, alpha=0.12, label="min–max")
        axes.fill_between(times, p025, p075, color=color, alpha=0.28, label="25–75%")
        axes.plot(times, p050, color=color, linewidth=1.5, label="median")
        extras = [value for value in percentiles if value not in {0, 25, 50, 75, 100}]
    else:
        extras = list(percentiles)
    for percentile in extras:
        suffix = percentile_suffix(percentile)
        axes.plot(
            times, _series(frame, f"{prefix}_{suffix}_{unit}", positive=positive),
            linewidth=0.9, alpha=0.75, label=f"p{percentile:g}", color=color,
        )


def _plot_count(
    axes: Any, frame: pd.DataFrame, column: str, maximum: int, label: str,
) -> Any:
    twin = axes.twinx()
    twin.step(
        frame.plot_time, frame[column].to_numpy(), where="mid",
        color="0.35", linewidth=0.8, alpha=0.65, label=label,
    )
    twin.set_ylim(-0.05, max(1, maximum) + 0.4)
    twin.set_ylabel(label, color="0.35", fontsize=8)
    twin.tick_params(axis="y", labelsize=7, colors="0.35")
    return twin


def _finish_time_figure(
    figure: Any, axes: np.ndarray, *, title: str, policy: str,
) -> None:
    figure.suptitle(f"{title}\nMembership policy: {policy}", fontsize=12)
    for axes_item in axes.flat:
        axes_item.grid(True, alpha=0.22)
        axes_item.tick_params(axis="x", rotation=20)
    figure.tight_layout(rect=(0, 0, 1, 0.95))


def _velocity_distributions(
    result: ClusterStatisticsArrayResult, config: ClusterStatisticsConfig,
    colors: dict[str, Any], path: Path,
) -> Path:
    values = np.concatenate([
        sample
        for cluster_id in result.cluster_ids
        for sample in (
            result.velocity_samples[cluster_id].absolute_speed_m_s,
            result.velocity_samples[cluster_id].internal_speed_m_s,
        )
        if len(sample)
    ]) if any(
        len(result.velocity_samples[cluster_id].absolute_speed_m_s)
        or len(result.velocity_samples[cluster_id].internal_speed_m_s)
        for cluster_id in result.cluster_ids
    ) else np.asarray([], dtype=float)
    if config.velocity.speed_range_m_s is not None:
        lower, upper = config.velocity.speed_range_m_s
    elif len(values):
        lower, upper = float(np.min(values)), float(np.max(values))
        if np.isclose(lower, upper):
            padding = max(1e-9, abs(lower) * 0.05)
            lower = max(0.0, lower - padding)
            upper += padding
    else:
        lower, upper = 0.0, 1.0
    bins = np.linspace(lower, upper, config.velocity.histogram_bins + 1)
    rows = len(result.cluster_ids)
    figure, axes = plt.subplots(
        rows, 1, figsize=(10, max(3.2, 2.7 * rows)), sharex=True,
        sharey=config.velocity.share_probability_density_y_axis,
        squeeze=False,
    )
    for row, cluster_id in enumerate(result.cluster_ids):
        ax = axes[row, 0]
        samples = result.velocity_samples[cluster_id]
        absolute = samples.absolute_speed_m_s
        internal = samples.internal_speed_m_s
        absolute_in = absolute[(absolute >= lower) & (absolute <= upper)]
        internal_in = internal[(internal >= lower) & (internal <= upper)]
        if len(absolute_in):
            ax.hist(
                absolute_in, bins=bins, density=True, histtype="step", linewidth=1.5,
                color=colors[cluster_id], label=f"absolute (n={len(absolute)})",
            )
        if len(internal_in):
            ax.hist(
                internal_in, bins=bins, density=True, alpha=0.28,
                color=colors[cluster_id], label=f"internal (n={len(internal)})",
            )
        if not len(absolute_in) and not len(internal_in):
            _mark_unavailable(ax, "No valid velocity samples")
        elif len(absolute_in) != len(absolute) or len(internal_in) != len(internal):
            ax.text(
                0.99, 0.96,
                f"In range: absolute {len(absolute_in)}/{len(absolute)}; "
                f"internal {len(internal_in)}/{len(internal)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
            )
        ax.set_title(_cluster_title(result, cluster_id), color=colors[cluster_id], fontsize=9)
        ax.set_ylabel("Probability density")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("Speed (m s⁻¹)")
    figure.suptitle(
        f"Array {result.array_id} velocity distributions\n"
        f"Membership policy: {_policy(config)}; common bins",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(figure, path, dpi=config.plotting.dpi, policy=_policy(config))


def _absolute_displacement(
    result: ClusterStatisticsArrayResult, config: ClusterStatisticsConfig,
    colors: dict[str, Any], path: Path,
) -> Path:
    rows = len(result.cluster_ids)
    figure, axes = plt.subplots(
        rows, 1, figsize=(11, max(3.4, 2.8 * rows)), sharex=True, sharey=True,
        squeeze=False,
    )
    maximum_members = int(result.summary.assigned_cluster_size.max())
    for row, cluster_id in enumerate(result.cluster_ids):
        ax = axes[row, 0]
        frame = _cluster_frame(result, cluster_id)
        _plot_percentiles(
            ax, frame, prefix="absolute_displacement", unit="m",
            percentiles=config.percentiles, color=colors[cluster_id],
        )
        ax.plot(
            frame.plot_time, frame.mean_absolute_displacement_m,
            color="black", linewidth=1.2, label="member mean",
        )
        ax.plot(
            frame.plot_time, frame.centroid_displacement_m,
            color=colors[cluster_id], linestyle="--", linewidth=1.2,
            label="centroid displacement",
        )
        _plot_count(
            ax, frame, "active_member_count", maximum_members, "active members",
        )
        if not np.isfinite(_series(frame, "mean_absolute_displacement_m")).any():
            _mark_unavailable(ax, "Displacement unavailable in admitted interval")
        ax.set_title(_cluster_title(result, cluster_id), color=colors[cluster_id], fontsize=9)
        ax.set_ylabel("Displacement (m)")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
    _finish_time_figure(
        figure, axes,
        title=f"Array {result.array_id} absolute and centroid displacement",
        policy=_policy(config),
    )
    return _save(figure, path, dpi=config.plotting.dpi, policy=_policy(config))


def _pair_separation(
    result: ClusterStatisticsArrayResult, config: ClusterStatisticsConfig,
    colors: dict[str, Any], path: Path,
) -> Path:
    rows = len(result.cluster_ids)
    figure, axes = plt.subplots(
        rows, 1, figsize=(11, max(3.4, 2.8 * rows)), sharex=True, sharey=True,
        squeeze=False,
    )
    maximum_members = int(result.summary.assigned_cluster_size.max())
    maximum_pairs = maximum_members * (maximum_members - 1) // 2
    for row, cluster_id in enumerate(result.cluster_ids):
        ax = axes[row, 0]
        frame = _cluster_frame(result, cluster_id)
        _plot_percentiles(
            ax, frame, prefix="pair_separation", unit="m",
            percentiles=config.percentiles, color=colors[cluster_id],
        )
        ax.plot(
            frame.plot_time, frame.mean_pair_separation_m,
            color="black", linewidth=1.2, label="pair mean",
        )
        assigned = int(frame.assigned_cluster_size.iloc[0])
        _plot_count(
            ax, frame, "valid_pair_count", maximum_pairs,
            "valid pairs",
        )
        if assigned < 2:
            _mark_unavailable(ax, "Pair statistics unavailable: singleton cluster")
        elif not np.isfinite(_series(frame, "mean_pair_separation_m")).any():
            _mark_unavailable(ax, "Pair statistics unavailable in admitted interval")
        ax.set_title(_cluster_title(result, cluster_id), color=colors[cluster_id], fontsize=9)
        ax.set_ylabel("Pair separation (m)")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
    _finish_time_figure(
        figure, axes, title=f"Array {result.array_id} within-cluster pair separation",
        policy=_policy(config),
    )
    return _save(figure, path, dpi=config.plotting.dpi, policy=_policy(config))


def _relative_dispersion(
    result: ClusterStatisticsArrayResult, config: ClusterStatisticsConfig,
    colors: dict[str, Any], path: Path,
) -> Path:
    rows = len(result.cluster_ids)
    figure, axes = plt.subplots(
        rows, 1, figsize=(11, max(3.4, 2.9 * rows)), sharex=True, sharey=True,
        squeeze=False,
    )
    logarithmic = config.plotting.relative_dispersion_yscale == "log"
    dispersion_columns = [
        "mean_squared_pair_separation_m2",
        "relative_dispersion_D2_m2",
        *(
            f"relative_displacement_q_{percentile_suffix(value)}_m2"
            for value in config.percentiles
        ),
    ]
    has_positive_data = any(
        np.any(_series(result.timeseries, column) > 0)
        for column in dispersion_columns
    )
    maximum_members = int(result.summary.assigned_cluster_size.max())
    maximum_pairs = maximum_members * (maximum_members - 1) // 2
    for row, cluster_id in enumerate(result.cluster_ids):
        ax = axes[row, 0]
        frame = _cluster_frame(result, cluster_id)
        _plot_percentiles(
            ax, frame, prefix="relative_displacement_q", unit="m2",
            percentiles=config.percentiles, color=colors[cluster_id],
            positive=logarithmic,
        )
        ax.plot(
            frame.plot_time,
            _series(frame, "mean_squared_pair_separation_m2", positive=logarithmic),
            color="black", linewidth=1.2, label="mean squared pair separation",
        )
        ax.plot(
            frame.plot_time,
            _series(frame, "relative_dispersion_D2_m2", positive=logarithmic),
            color=colors[cluster_id], linestyle="--", linewidth=1.4,
            label="mean D² (vector change)",
        )
        assigned = int(frame.assigned_cluster_size.iloc[0])
        _plot_count(
            ax, frame, "valid_pair_count", maximum_pairs,
            "valid pairs",
        )
        if logarithmic:
            ax.set_yscale("log")
        if assigned < 2:
            _mark_unavailable(ax, "Relative dispersion unavailable: singleton cluster")
        ax.set_title(_cluster_title(result, cluster_id), color=colors[cluster_id], fontsize=9)
        ax.set_ylabel("Squared distance (m²)")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
    if logarithmic and not has_positive_data:
        # Matplotlib cannot auto-locate ticks on an entirely empty log axis.
        # A fixed placeholder range carries no data and keeps unavailable panels renderable.
        axes[0, 0].set_ylim(1.0, 10.0)
    _finish_time_figure(
        figure, axes, title=f"Array {result.array_id} relative dispersion",
        policy=f"{_policy(config)}; {config.plotting.relative_dispersion_yscale} scale",
    )
    return _save(figure, path, dpi=config.plotting.dpi, policy=_policy(config))


def _extent_and_shape(
    result: ClusterStatisticsArrayResult, config: ClusterStatisticsConfig,
    colors: dict[str, Any], path: Path,
) -> Path:
    rows = len(result.cluster_ids)
    figure, axes = plt.subplots(
        rows, 4, figsize=(17, max(3.6, 2.8 * rows)), sharex="col", sharey="col",
        squeeze=False,
    )
    maximum_members = int(result.summary.assigned_cluster_size.max())
    for row, cluster_id in enumerate(result.cluster_ids):
        frame = _cluster_frame(result, cluster_id)
        color = colors[cluster_id]
        linear, aspect, area, count = axes[row]
        area_ratio = area.twinx()
        linear.plot(frame.plot_time, frame.radius_of_gyration_m, color="black", label="Rg")
        linear.plot(frame.plot_time, frame.major_scale_m, color=color, label="major")
        linear.plot(
            frame.plot_time, frame.minor_scale_m, color=color, linestyle="--", label="minor",
        )
        aspect.plot(frame.plot_time, frame.aspect_ratio, color=color, label="aspect ratio")
        aspect.plot(
            frame.plot_time, frame.eigenvalue_ratio, color="black", linestyle=":",
            label="eigenvalue ratio",
        )
        area.plot(frame.plot_time, frame.convex_hull_area_m2, color=color, label="hull area")
        area_ratio.plot(
            frame.plot_time, frame.convex_hull_area_ratio, color="black", linestyle="--",
            linewidth=1.1, label="A(t)/A(t₀)",
        )
        count.step(
            frame.plot_time, frame.active_member_count, where="mid", color=color,
            label="active members",
        )
        assigned = int(frame.assigned_cluster_size.iloc[0])
        count.set_ylim(-0.05, max(1, maximum_members) + 0.4)
        linear.set_ylabel("Scale (m)")
        aspect.set_ylabel("Ratio")
        aspect.set_ylim(-0.02, 1.02)
        area.set_ylabel("Area (m²)")
        area_ratio.set_ylabel("A(t)/A(t₀)")
        count.set_ylabel("Active members")
        linear.set_title(_cluster_title(result, cluster_id), color=color, fontsize=9)
        aspect.set_title("Shape ratios", fontsize=9)
        area.set_title("Convex-hull footprint", fontsize=9)
        count.set_title("Membership support", fontsize=9)
        if assigned < 2:
            _mark_unavailable(linear, "Covariance geometry unavailable\nfor singleton")
            _mark_unavailable(aspect, "Shape unavailable\nfor singleton")
        if assigned < 3:
            _mark_unavailable(area, "Hull area unavailable\nfor fewer than 3 members")
        elif not np.isfinite(_series(frame, "convex_hull_area_ratio")).any():
            area_ratio.text(
                0.98, 0.04, "Area ratio unavailable: A(t₀) is zero or missing",
                transform=area_ratio.transAxes, ha="right", va="bottom", fontsize=7,
                color="0.35",
            )
        for ax in (linear, aspect, area, count):
            ax.legend(loc="best", fontsize=7)
        area_ratio.legend(loc="upper right", fontsize=7)
    _finish_time_figure(
        figure, axes, title=f"Array {result.array_id} cluster extent and shape",
        policy=_policy(config),
    )
    return _save(figure, path, dpi=config.plotting.dpi, policy=_policy(config))


def _orientation(
    result: ClusterStatisticsArrayResult, config: ClusterStatisticsConfig,
    colors: dict[str, Any], path: Path,
) -> Path:
    rows = len(result.cluster_ids)
    figure, axes = plt.subplots(
        rows, 1, figsize=(11, max(3.4, 2.6 * rows)), sharex=True, sharey=True,
        squeeze=False,
    )
    for row, cluster_id in enumerate(result.cluster_ids):
        ax = axes[row, 0]
        frame = _cluster_frame(result, cluster_id)
        values = _series(frame, "major_axis_orientation_deg")
        finite = np.isfinite(values)
        ax.scatter(
            frame.plot_time[finite], values[finite], s=12,
            color=colors[cluster_id], label="major axis",
        )
        ax.set_ylim(0, 180)
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.set_ylabel("CCW from east (°)")
        ax.set_title(_cluster_title(result, cluster_id), color=colors[cluster_id], fontsize=9)
        if not finite.any():
            _mark_unavailable(
                ax, "Orientation unavailable (insufficient, zero-size, or isotropic geometry)",
            )
    _finish_time_figure(
        figure, axes,
        title=f"Array {result.array_id} axial major-axis orientation (points avoid 0/180° wraps)",
        policy=_policy(config),
    )
    return _save(figure, path, dpi=config.plotting.dpi, policy=_policy(config))


def render_cluster_statistics_figures(
    result: ClusterStatisticsArrayResult, config: ClusterStatisticsConfig,
    directory: str | Path,
) -> tuple[Path, ...]:
    """Atomically render the six declared figures for one independent array."""
    directory = Path(directory)
    colors = _colors(result)
    paths = tuple(directory / name for name in CLUSTER_STATISTICS_FIGURES)
    renderers = (
        _velocity_distributions,
        _absolute_displacement,
        _pair_separation,
        _relative_dispersion,
        _extent_and_shape,
        _orientation,
    )
    for renderer, path in zip(renderers, paths):
        renderer(result, config, colors, path)
    return paths
