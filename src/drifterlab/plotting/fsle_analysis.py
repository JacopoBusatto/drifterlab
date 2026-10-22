"""Separate v2 FSLE figures with common limits and prescribed-power compensation."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fsle import _save


def _positive_limits(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return [0.1, 1.0]
    lower, upper = float(values.min()), float(values.max())
    padding = max(np.log(upper / lower) * 0.08, np.log(1.25))
    return [float(lower * np.exp(-padding)), float(upper * np.exp(padding))]


def _supported(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame.plot_included.astype(bool) & np.isfinite(frame.fsle_day_inverse)
        & (frame.fsle_day_inverse > 0)
    ].sort_values("scale_lower_km")


def _values(frame: pd.DataFrame, power: float | None) -> tuple[np.ndarray, np.ndarray]:
    factor = np.ones(len(frame)) if power is None else np.power(frame.scale_lower_km.to_numpy() * 1000, -power) / 86400
    return frame.fsle_day_inverse.to_numpy() * factor, frame.fsle_standard_error_day_inverse.to_numpy() * factor


def _fit_line(row: Any, power: float | None) -> tuple[np.ndarray, np.ndarray]:
    x = np.geomspace(row.actual_scale_min_km, row.actual_scale_max_km, 100)
    y = row.coefficient_si * np.power(x * 1000, row.power) * 86400
    if power is not None:
        y *= np.power(x * 1000, -power) / 86400
    return x, y


def _shared_y_limits(frame: pd.DataFrame, fits: pd.DataFrame, power: float | None, error_bars: bool) -> list[float]:
    valid = _supported(frame)
    y, error = _values(valid, power)
    parts = [y]
    if error_bars:
        parts.extend([y - error, y + error])
    for row in fits[fits.status == "fitted"].itertuples():
        if power is None or row.power == power:
            parts.append(_fit_line(row, power)[1])
    return _positive_limits(np.concatenate(parts))


def _cluster_name(cluster: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", cluster)[:80] or "cluster"
    return f"{slug}_{sha256(cluster.encode()).hexdigest()[:8]}"


def render_fsle_analysis(
    spectra: pd.DataFrame,
    fits: pd.DataFrame,
    settings: dict[str, Any],
    directory: Path,
) -> dict[str, Any]:
    """Render one standard and one compensated plot per exponent/spectrum.

    Limits are resolved once across all selected arrays and clusters. Different
    powers have distinct compensated y limits because their dimensions differ.
    """
    plotting = settings["plotting"]
    x_limits = plotting["x_range_km"] or [float(spectra.scale_lower_km.min()), float(spectra.scale_upper_km.max())]
    y_limits = plotting["y_range_day_inverse"] or _shared_y_limits(spectra, fits, None, plotting["standard_error_bars"])
    powers = list(dict.fromkeys(spec["power"] for spec in settings["fits"]))
    compensated_limits = {
        power: _shared_y_limits(spectra, fits, power, plotting["standard_error_bars"])
        for power in powers
    }
    records = []
    for (array_id, scope, cluster_id), frame in spectra.groupby(
        ["array_id", "scope", "cluster_id"], sort=False, dropna=False,
    ):
        same_cluster = fits.cluster_id.isna() if pd.isna(cluster_id) else fits.cluster_id == cluster_id
        spectrum_fits = fits[(fits.array_id == array_id) & (fits.scope == scope) & same_cluster]
        if scope == "individual_cluster":
            stem = f"clusters/{_cluster_name(str(cluster_id))}"
            title = f"Array {array_id}: cluster {cluster_id}"
        else:
            stem = scope
            title = f"Array {array_id}: " + ("pooled within-cluster pairs" if scope == "within_cluster_pooled" else "cross-cluster chance pairs")
        valid = _supported(frame)
        for power in [None, *powers]:
            figure, ax = plt.subplots(figsize=(9, 6))
            try:
                y, error = _values(valid, power)
                if len(valid):
                    ax.scatter(valid.scale_lower_km, y, facecolors="none", edgecolors="C0", s=28, label="FSLE" if power is None else "Compensated FSLE", zorder=3)
                    if plotting["standard_error_bars"]:
                        finite_error = np.isfinite(error)
                        ax.errorbar(valid.scale_lower_km.to_numpy()[finite_error], y[finite_error], yerr=error[finite_error], fmt="none", ecolor="C0", alpha=0.6, capsize=2)
                else:
                    ax.text(0.5, 0.5, "No shells meet the completed-pair minimum", ha="center", va="center", transform=ax.transAxes)
                relevant = spectrum_fits if power is None else spectrum_fits[spectrum_fits.power == power]
                for index, row in enumerate(relevant.itertuples()):
                    if row.status != "fitted":
                        continue
                    x, line = _fit_line(row, power)
                    ax.plot(x, line, "--", color=f"C{(index + 1) % 10}", label=f"{row.fit_id}: p={row.power:.3g}, A={row.coefficient_si:.3g} (SI)")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits if power is None else compensated_limits[power])
                ax.set_xlabel(r"Separation scale $\delta$ (km)")
                ax.set_ylabel(r"FSLE $\lambda$ (day$^{-1}$)" if power is None else rf"$\lambda\delta^{{{-power:g}}}$ (s$^{{-1}}$ m$^{{{-power:g}}}$)")
                ax.set_title(title + ("" if power is None else f"\nCompensated for power {power:.6g}"))
                ax.grid(which="major", alpha=0.35)
                ax.grid(which="minor", linestyle="--", alpha=0.15)
                footer = f"{settings['passage_policy']}; minimum {settings['minimum_reached_pairs_per_scale']} completed pairs per shell"
                if scope == "cross_cluster":
                    footer += f"; closest approach ≤ {settings['chance_pair_threshold_km']:g} km"
                figure.text(0.5, 0.01, footer, ha="center", fontsize=8)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=8)
                figure.tight_layout(rect=(0, 0.035, 1, 1))
                suffix = "" if power is None else "__compensated_" + str(power).replace("-", "m").replace(".", "p").replace("+", "")
                path = directory / f"array_{array_id:03d}/figures/{stem}{suffix}.png"
                _save(figure, path, dpi=plotting["dpi"])
                records.append({
                    "array_id": int(array_id), "scope": scope,
                    "cluster_id": None if pd.isna(cluster_id) else cluster_id,
                    "power": power, "path": path.relative_to(directory).as_posix(),
                    "x_range_km": list(x_limits), "y_range": list(y_limits if power is None else compensated_limits[power]),
                })
            finally:
                plt.close(figure)
    return {
        "figure_count": len(records), "standard_y_range_day_inverse": list(y_limits),
        "x_range_km": list(x_limits),
        "compensated_y_ranges_si": {str(key): value for key, value in compensated_limits.items()},
        "records": records,
    }
