"""FSLE spectrum figures for pooled and individual deployment clusters."""

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

POOLED_FIGURE_NAME = "fsle_same_cluster_pooled.png"
CLUSTER_FIGURE_NAME = "fsle_individual_clusters.png"


def _save(figure: Any, path: Path, *, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")
    try:
        figure.savefig(
            temporary,
            dpi=dpi,
            bbox_inches="tight",
            metadata={"Description": "Overshoot-aware finite-size Lyapunov exponent"},
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return path


def _valid_spectrum(frame: pd.DataFrame) -> pd.DataFrame:
    valid = (
        frame.plot_included.astype(bool)
        & np.isfinite(frame.scale_lower_km)
        & np.isfinite(frame.fsle_day_inverse)
        & (frame.scale_lower_km > 0)
        & (frame.fsle_day_inverse > 0)
    )
    return frame[valid].sort_values("scale_lower_km")


def _reference_values(
    pooled: pd.DataFrame,
    *,
    anchor_scale_km: float,
    exponent: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    anchor = pooled[
        np.isclose(
            pooled.scale_lower_km.to_numpy(dtype=float),
            anchor_scale_km,
            rtol=1e-12,
            atol=1e-12,
        )
        & pooled.plot_included.astype(bool).to_numpy()
    ]
    if anchor.empty:
        return None
    anchor_value = float(anchor.iloc[0].fsle_day_inverse)
    finite = _valid_spectrum(pooled)
    if not np.isfinite(anchor_value) or anchor_value <= 0 or finite.empty:
        return None
    lower = min(float(finite.scale_lower_km.min()), anchor_scale_km)
    upper = max(float(finite.scale_lower_km.max()), anchor_scale_km)
    x = np.geomspace(lower, upper, 200)
    y = anchor_value * np.power(x / anchor_scale_km, exponent)
    return x, y, anchor_value


def _finish_axis(
    ax: Any,
    *,
    x_range_km: tuple[float | None, float | None],
    y_range_day_inverse: tuple[float | None, float | None],
) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Separation scale $\delta$ (km)")
    ax.set_ylabel(r"FSLE $\lambda(\delta)$ (day$^{-1}$)")
    ax.grid(which="major", alpha=0.35)
    ax.grid(which="minor", linestyle="--", alpha=0.18)
    if x_range_km != (None, None):
        ax.set_xlim(left=x_range_km[0], right=x_range_km[1])
    if y_range_day_inverse != (None, None):
        ax.set_ylim(bottom=y_range_day_inverse[0], top=y_range_day_inverse[1])


def _add_reference(
    ax: Any,
    pooled: pd.DataFrame,
    *,
    enabled: bool,
    anchor_scale_km: float,
    exponent: float,
    label: str,
) -> bool:
    if not enabled:
        return False
    values = _reference_values(
        pooled,
        anchor_scale_km=anchor_scale_km,
        exponent=exponent,
    )
    if values is None:
        return False
    x, y, _ = values
    ax.plot(
        x,
        y,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=rf"{label}, anchored to pooled $\lambda$ at {anchor_scale_km:g} km",
    )
    return True


def _add_standard_error_bars(
    ax: Any,
    frame: pd.DataFrame,
    *,
    color: Any,
    enabled: bool,
) -> None:
    if not enabled or "fsle_standard_error_day_inverse" not in frame:
        return
    finite = np.isfinite(frame.fsle_standard_error_day_inverse.to_numpy(dtype=float))
    if not finite.any():
        return
    selected = frame.iloc[np.flatnonzero(finite)]
    ax.errorbar(
        selected.scale_lower_km,
        selected.fsle_day_inverse,
        yerr=selected.fsle_standard_error_day_inverse,
        fmt="none",
        ecolor=color,
        elinewidth=0.9,
        capsize=2.5,
        alpha=0.55,
        zorder=1,
    )


def render_array_fsle_figures(
    spectrum: pd.DataFrame,
    *,
    array_id: int,
    coordinate_method: str,
    directory: str | Path,
    dpi: int,
    x_range_km: tuple[float | None, float | None],
    y_range_day_inverse: tuple[float | None, float | None],
    reference_enabled: bool,
    reference_exponent: float,
    reference_label: str,
    anchor_scale_km: float,
    standard_error_bars_enabled: bool,
) -> tuple[Path, Path]:
    """Render one pooled and one individual-cluster figure for an array."""
    directory = Path(directory)
    array_frame = spectrum[spectrum.array_id == array_id]
    pooled = array_frame[array_frame.scope == "same_cluster_pooled"]

    pooled_path = directory / POOLED_FIGURE_NAME
    figure, ax = plt.subplots(figsize=(9, 6))
    valid = _valid_spectrum(pooled)
    if valid.empty:
        ax.text(
            0.5,
            0.5,
            "No scales meet the configured reached-pair minimum",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        _add_standard_error_bars(
            ax,
            valid,
            color="C0",
            enabled=standard_error_bars_enabled,
        )
        ax.scatter(
            valid.scale_lower_km,
            valid.fsle_day_inverse,
            marker="o",
            s=28,
            facecolors="none",
            edgecolors="C0",
            label="same-cluster pairs pooled",
            zorder=2,
        )
    pooled_reference_added = _add_reference(
        ax,
        pooled,
        enabled=reference_enabled,
        anchor_scale_km=anchor_scale_km,
        exponent=reference_exponent,
        label=reference_label,
    )
    if reference_enabled and not pooled_reference_added:
        ax.text(
            0.01,
            0.01,
            f"Reference unavailable: pooled {anchor_scale_km:g} km scale is unsupported",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )
    _finish_axis(
        ax,
        x_range_km=x_range_km,
        y_range_day_inverse=y_range_day_inverse,
    )
    ax.set_title(
        f"Array {array_id}: pooled within-cluster FSLE\n"
        f"Overshoot estimator; coordinates: {coordinate_method}",
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    figure.tight_layout()
    _save(figure, pooled_path, dpi=dpi)

    cluster_path = directory / CLUSTER_FIGURE_NAME
    figure, ax = plt.subplots(figsize=(11, 7))
    clusters = array_frame[array_frame.scope == "individual_cluster"]
    cluster_ids = sorted(clusters.cluster_id.dropna().astype(str).unique())
    cmap = colormaps["tab20"]
    colors = (
        np.linspace(0, 1, len(cluster_ids))
        if len(cluster_ids) > 1
        else np.asarray([0.05])
    )
    plotted = 0
    for cluster_id, color_index in zip(cluster_ids, colors):
        cluster = _valid_spectrum(
            clusters[clusters.cluster_id.astype(str) == cluster_id]
        )
        if cluster.empty:
            continue
        color = cmap(float(color_index))
        _add_standard_error_bars(
            ax,
            cluster,
            color=color,
            enabled=standard_error_bars_enabled,
        )
        ax.scatter(
            cluster.scale_lower_km,
            cluster.fsle_day_inverse,
            marker="o",
            s=22,
            facecolors="none",
            edgecolors=color,
            label=cluster_id,
            zorder=2,
        )
        plotted += 1
    cluster_reference_added = _add_reference(
        ax,
        pooled,
        enabled=reference_enabled,
        anchor_scale_km=anchor_scale_km,
        exponent=reference_exponent,
        label=reference_label,
    )
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No individual clusters meet the reached-pair minimum",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    _finish_axis(
        ax,
        x_range_km=x_range_km,
        y_range_day_inverse=y_range_day_inverse,
    )
    reference_title = (
        "common reference anchored to pooled spectrum"
        if cluster_reference_added
        else "pooled reference unavailable"
    )
    ax.set_title(
        f"Array {array_id}: individual within-cluster FSLE spectra\n"
        f"Overshoot estimator; {reference_title}; coordinates: {coordinate_method}",
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    figure.tight_layout()
    _save(figure, cluster_path, dpi=dpi)
    return pooled_path, cluster_path
