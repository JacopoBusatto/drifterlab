"""Figures for the trajectory-only Poje et al. method comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


def _log_spaced_indices(values: np.ndarray, count: int) -> np.ndarray:
    positive = np.flatnonzero(np.isfinite(values) & (values > 0))
    if len(positive) <= count:
        return positive
    targets = np.geomspace(values[positive[0]], values[positive[-1]], count)
    indices = [positive[int(np.argmin(np.abs(values[positive] - target)))] for target in targets]
    return np.unique(indices)


def plot_figure4_dispersion(
    timeseries: pd.DataFrame,
    summary: dict[str, object],
    *,
    output_path: Path,
    dpi: int,
) -> None:
    """Render a Poje Figure 4-style absolute and relative dispersion figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = timeseries.sort_values("elapsed_days")
    days = frame.elapsed_days.to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    figure.subplots_adjust(
        left=0.07, right=0.98, bottom=0.16, top=0.82, wspace=0.14,
    )
    absolute_axis, relative_axis = axes

    absolute = frame.absolute_rms_dispersion_km.to_numpy(dtype=float)
    absolute_mask = (
        frame.absolute_panel_in_range.astype(bool).to_numpy()
        & (days > 0) & np.isfinite(absolute) & (absolute > 0)
    )
    absolute_axis.plot(days[absolute_mask], absolute[absolute_mask], color="black", linewidth=1.4)
    ballistic_mask = (
        frame.ballistic_fit_in_range.astype(bool).to_numpy()
        & (days > 0)
    )
    v0_squared = float(summary["ballistic_v0_squared_m2_s2"])
    if np.isfinite(v0_squared) and v0_squared > 0 and ballistic_mask.any():
        ballistic_seconds = days[ballistic_mask] * 86400
        ballistic_rms_km = np.sqrt(v0_squared) * ballistic_seconds / 1000
        absolute_axis.plot(
            days[ballistic_mask], ballistic_rms_km,
            color="#dc2626", linestyle="--", linewidth=1.2,
            label=r"ballistic $\sqrt{A^2}=\sqrt{v_0^2}\,t$",
        )
        absolute_axis.legend(fontsize=8, loc="best")
    absolute_axis.set_title("A  RMS absolute dispersion", loc="left")
    absolute_axis.set_xlabel("Elapsed time [days]")
    absolute_axis.set_ylabel(r"$\sqrt{A^2(t)}$ [km]")

    relative = frame.adjusted_radial_relative_dispersion_km2.to_numpy(dtype=float)
    lower = frame.relative_ci_lower_m2.to_numpy(dtype=float) / 1_000_000
    upper = frame.relative_ci_upper_m2.to_numpy(dtype=float) / 1_000_000
    relative_mask = (days > 0) & np.isfinite(relative) & (relative > 0)
    relative_axis.plot(
        days[relative_mask], relative[relative_mask], color="black", linewidth=1.3,
        label=r"$\langle[r(t)-r(0)]^2\rangle$",
    )
    relative_axis.fill_between(
        days[relative_mask], lower[relative_mask], upper[relative_mask],
        color="#9ca3af", alpha=0.3, linewidth=0, label="95% pair-bootstrap CI",
    )
    error_indices = _log_spaced_indices(np.where(relative_mask, days, np.nan), 18)
    if len(error_indices):
        relative_axis.errorbar(
            days[error_indices], relative[error_indices],
            yerr=np.vstack((
                relative[error_indices] - lower[error_indices],
                upper[error_indices] - relative[error_indices],
            )),
            fmt="none", ecolor="#374151", elinewidth=0.8, capsize=2, alpha=0.75,
        )
    richardson_mask = frame.richardson_fit_in_range.astype(bool).to_numpy()
    coefficient = float(summary["richardson_coefficient_m2_s3"])
    if np.isfinite(coefficient) and coefficient > 0 and richardson_mask.any():
        seconds = days[richardson_mask] * 86400
        reference_km2 = coefficient * seconds ** 3 / 1_000_000
        relative_axis.plot(
            days[richardson_mask], reference_km2,
            color="#dc2626", linewidth=1.4, label=r"$C_R t^3$",
        )
    relative_axis.set_title("B  Adjusted radial relative dispersion", loc="left")
    relative_axis.set_xlabel("Elapsed time [days]")
    relative_axis.set_ylabel(r"$\langle[r(t)-r(0)]^2\rangle$ [km$^2$]")
    relative_axis.legend(fontsize=8, loc="lower right")

    inset = relative_axis.inset_axes([0.08, 0.57, 0.38, 0.34])
    compensated = frame.richardson_compensated_m2_s3.to_numpy(dtype=float) / 1e-9
    compensated_mask = (days > 0) & np.isfinite(compensated) & (compensated > 0)
    inset.plot(days[compensated_mask], compensated[compensated_mask], color="black", linewidth=0.9)
    if np.isfinite(coefficient) and coefficient > 0:
        inset.axhline(coefficient / 1e-9, color="#dc2626", linewidth=0.8, linestyle="--")
    inset.set_xscale("log")
    inset.set_yscale("log")
    inset.set_xlabel("days", fontsize=7)
    inset.set_ylabel(r"$D^2/t^3$ [$10^{-9}$ m$^2$ s$^{-3}$]", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(True, which="major", color="#d1d5db", linewidth=0.35)

    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="major", color="#d1d5db", linewidth=0.5)
        axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.3, linestyle=":")

    minimum_day = float(np.nanmin(days[days > 0]))
    absolute_axis.set_xlim(minimum_day, float(summary["absolute_plot_end_days"]))
    relative_axis.set_xlim(minimum_day, float(summary["window_end_hours"]) / 24)
    figure.suptitle(
        f"Array {int(summary['array_id']):03d} · Poje Figure 4 method reproduction\n"
        f"common anchor {summary['anchor_time_utc']}; "
        f"{int(summary['assigned_platform_count'])} platforms; "
        f"{int(summary['initial_pair_count'])} fixed pairs with "
        f"$r(0)<{float(summary['initial_separation_maximum_m']):g}$ m",
        fontsize=13, y=0.96,
    )
    figure.text(
        0.01, 0.035,
        "Panel A uses displacement from each platform's own anchor position. "
        "Panel B uses scalar separation change, matching Poje Eq. (3), not vector relative displacement. "
        "Confidence intervals synchronously resample the fixed initial-pair cohort.",
        fontsize=7, color="#374151",
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_figure5_self_similarity(
    separation_memory: pd.DataFrame,
    lagrangian_structure: pd.DataFrame,
    summary: dict[str, object],
    *,
    correlation_time_fraction: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Render Poje Figure 5-style separation memory and velocity structure plots."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (memory_axis, velocity_axis) = plt.subplots(1, 2, figsize=(14, 6))
    figure.subplots_adjust(
        left=0.07, right=0.98, bottom=0.17, top=0.80, wspace=0.18,
    )

    colors = ("#dc2626", "#2563eb", "#111827", "#059669", "#7c3aed")
    for color, (_, group) in zip(
        colors, separation_memory.groupby("memory_group_id", sort=False),
    ):
        ordered = group.sort_values("similarity_bin_center")
        finite = np.isfinite(ordered.mean_normalized_separation_correlation)
        start = float(ordered.memory_group_start_day.iloc[0])
        end = float(ordered.memory_group_end_day.iloc[0])
        x_values = ordered.loc[finite, "similarity_bin_center"].to_numpy(dtype=float)
        y_values = ordered.loc[
            finite, "mean_normalized_separation_correlation"
        ].to_numpy(dtype=float)
        memory_axis.plot(
            np.append(x_values, 0.0), np.append(y_values, 1.0),
            color=color, linewidth=1.25, marker="o", markersize=2.5,
            label=f"{start:g}–{end:g} days",
        )
    reference_x = np.asarray([-correlation_time_fraction, 0.0])
    reference_y = 1 + reference_x / correlation_time_fraction
    memory_axis.plot(
        reference_x, reference_y, color="#6b7280", linestyle="--", linewidth=1,
        label=rf"linear memory scale $|\tau|=t/{1 / correlation_time_fraction:g}$",
    )
    memory_axis.set_xlim(-1, 0)
    memory_axis.set_ylim(bottom=0)
    memory_axis.set_title("A  Scale-independent separation memory", loc="left")
    memory_axis.set_xlabel(r"Normalized backward lag $\tau/t$")
    memory_axis.set_ylabel(r"$R(t,\tau)/R(t,0)$")
    memory_axis.legend(fontsize=8, loc="upper left")

    lagrangian = lagrangian_structure.sort_values("elapsed_days")
    lag_days = lagrangian.elapsed_days.to_numpy(dtype=float)
    epsilon_proxy = lagrangian.epsilon_proxy_m2_s3.to_numpy(dtype=float)
    finite_proxy = (lag_days > 0) & np.isfinite(epsilon_proxy)
    velocity_axis.plot(
        lag_days[finite_proxy], epsilon_proxy[finite_proxy],
        color="black", linewidth=1.3,
    )
    velocity_axis.set_xscale("log")
    velocity_axis.set_title("B  Lagrangian velocity structure", loc="left")
    velocity_axis.set_xlabel(r"Elapsed time $\tau$ [days]")
    constant = float(summary["figure5_kolmogorov_constant"])
    velocity_axis.set_ylabel(
        rf"$S_L^2(\tau,0)/({constant:g}\,\tau)$ [m$^2$ s$^{{-3}}$]"
    )
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    velocity_axis.yaxis.set_major_formatter(formatter)

    for axis in (memory_axis, velocity_axis):
        axis.grid(True, which="major", color="#d1d5db", linewidth=0.5)
        axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.3, linestyle=":")
    figure.suptitle(
        f"Array {int(summary['array_id']):03d} · Poje Figure 5 method reproduction\n"
        f"common anchor {summary['anchor_time_utc']}; "
        f"{int(summary['assigned_platform_count'])} platforms; "
        f"{int(summary['figure5_complete_pair_count'])} complete-matrix pairs",
        fontsize=13, y=0.95,
    )
    figure.text(
        0.01, 0.035,
        "Panel A averages normalized correlations within the configured target-time groups; "
        "each target/lag estimate uses only pairs finite at both times. "
        f"Panel B uses valid platform velocities referenced to the common anchor and C0={constant:g}.",
        fontsize=7, color="#374151",
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_figure6_increment_distributions(
    distributions: pd.DataFrame,
    statistics: pd.DataFrame,
    summary: dict[str, object],
    *,
    normalized_increment_limit: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Render ARCTERX analogues of the observational panels in Poje Figure 6."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.07, right=0.98, bottom=0.18, top=0.79, wspace=0.08,
    )
    colors = ("#111827", "#2563eb", "#dc2626")
    markers = ("o", "s", "^")
    panel_definitions = (
        ("small", "A  Small separation bins"),
        ("large", "B  Large separation bins"),
    )
    positive_density: list[float] = []
    for axis, (panel_group, title) in zip(axes, panel_definitions):
        panel_statistics = statistics[statistics.panel_group == panel_group]
        scale_handles: list[Line2D] = []
        for color, marker, (_, statistic) in zip(
            colors, markers, panel_statistics.iterrows(),
        ):
            scale = distributions[
                distributions.figure6_scale_id == statistic.figure6_scale_id
            ].sort_values("normalized_increment_bin_center")
            density = scale.probability_density.to_numpy(dtype=float)
            finite = np.isfinite(density) & (density > 0)
            positive_density.extend(density[finite].tolist())
            resolved_km = float(statistic.separation_bin_nominal_center_m) / 1000
            scale_handle, = axis.plot(
                scale.loc[finite, "normalized_increment_bin_center"], density[finite],
                color=color, linewidth=0.9, marker=marker, markersize=2.7,
                label=(
                    f"{resolved_km:g} km; n={int(statistic.observation_count):,}; "
                    f"skew={float(statistic.central_skewness):.2f}; "
                    f"fit width={float(statistic.gaussian_core_fit_width):.2f}"
                ),
            )
            scale_handles.append(scale_handle)
            axis.plot(
                scale.normalized_increment_bin_center,
                scale.fitted_core_gaussian_density,
                color=color, linewidth=1.0, linestyle="--", alpha=0.85,
            )
            axis.plot(
                scale.normalized_increment_bin_center,
                scale.unit_variance_gaussian_density,
                color=color, linewidth=0.8, linestyle=":", alpha=0.55,
            )
        axis.set_title(title, loc="left")
        axis.set_xlabel(r"Normalized longitudinal increment $\Delta u_l/\sigma$")
        axis.set_xlim(-normalized_increment_limit, normalized_increment_limit)
        axis.set_yscale("log")
        axis.grid(True, which="major", color="#d1d5db", linewidth=0.5)
        axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.3, linestyle=":")
        style_handles = [
            Line2D([0], [0], color="#4b5563", linewidth=1, linestyle="--",
                   label="central-core Gaussian fit"),
            Line2D([0], [0], color="#4b5563", linewidth=1, linestyle=":",
                   label="unit-variance reference"),
        ]
        axis.legend(handles=scale_handles + style_handles, fontsize=7.2, loc="lower center")
    axes[0].set_ylabel(r"Probability density $p(\Delta u_l/\sigma)$")
    if positive_density:
        lower = max(1e-6, min(positive_density) / 1.5)
        axes[0].set_ylim(lower, 2)
    figure.suptitle(
        f"Array {int(summary['array_id']):03d} · Poje Figure 6 observational analogue\n"
        f"common array window {summary['window_id']}; "
        f"equal-weight valid pair-time observations",
        fontsize=13, y=0.95,
    )
    figure.text(
        0.01, 0.035,
        "Solid curves are ARCTERX histograms; dashed curves are central-core Gaussian fits; "
        "dotted curves retain the shifted unit-variance reference.\n"
        "Each scale uses the source separation bin nearest Poje's reported nominal center. "
        "Raw increments are divided by their population standard deviation without mean subtraction. "
        "AVISO comparison panels are intentionally omitted.",
        fontsize=7, color="#374151",
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_figure7_structure_functions(
    structure_functions: pd.DataFrame,
    summary: dict[str, object],
    *,
    rossby_reference_values: tuple[float, ...],
    epsilon_linear_threshold_m2_s3: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Render the ARCTERX observational analogue of Poje Figure 7."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = structure_functions.sort_values("mean_separation_m")
    figure, (second_axis, epsilon_axis) = plt.subplots(1, 2, figsize=(14, 6))
    figure.subplots_adjust(
        left=0.075, right=0.98, bottom=0.19, top=0.79, wspace=0.18,
    )

    panel_a = frame[
        frame.panel_a_in_range.astype(bool)
        & np.isfinite(frame.mean_separation_m)
        & np.isfinite(frame.longitudinal_second_order_m2_s2)
        & (frame.longitudinal_second_order_m2_s2 > 0)
    ]
    supported_a = panel_a[panel_a.support_ok.astype(bool)]
    low_a = panel_a[~panel_a.support_ok.astype(bool)]
    second_axis.plot(
        supported_a.mean_separation_m / 1000,
        supported_a.longitudinal_second_order_m2_s2,
        color="black", marker="o", markersize=4, linewidth=1,
        label=r"ARCTERX $S_{ll}^{2}$",
    )
    if len(low_a):
        second_axis.scatter(
            low_a.mean_separation_m / 1000,
            low_a.longitudinal_second_order_m2_s2,
            color="#9ca3af", marker="x", s=28, label="low support",
        )
    fit = panel_a[panel_a.scaling_fit_in_range.astype(bool)]
    if len(fit):
        second_axis.plot(
            fit.mean_separation_m / 1000,
            fit.k41_fixed_fit_second_order_m2_s2,
            color="#dc2626", linewidth=1.3,
            label=r"fixed $r^{2/3}$ fit",
        )
        exponent = float(summary["figure7_second_order_free_fit_exponent"])
        standard_error = float(
            summary["figure7_second_order_free_fit_exponent_standard_error"]
        )
        second_axis.plot(
            fit.mean_separation_m / 1000,
            fit.free_fit_second_order_m2_s2,
            color="#2563eb", linewidth=1, linestyle="--",
            label=rf"free fit $r^{{{exponent:.2f}\pm{standard_error:.2f}}}$",
        )
    if len(panel_a):
        reference_r_m = panel_a.mean_separation_m.to_numpy(dtype=float)
        reference_f = panel_a.bin_mean_coriolis_s_inverse.to_numpy(dtype=float)
        reference_values: list[np.ndarray] = []
        for index, rossby in enumerate(rossby_reference_values):
            rossby_curve = (rossby * reference_f * reference_r_m) ** 2
            reference_values.append(rossby_curve)
            second_axis.plot(
                reference_r_m / 1000,
                rossby_curve,
                color="#6b7280", linewidth=0.85,
                linestyle="-." if index % 2 == 0 else ":",
                label=rf"$Ro={rossby:g}$, bin-mean $f(\phi)$",
            )
        observed = panel_a.longitudinal_second_order_m2_s2.to_numpy(dtype=float)
        lower_candidates = [float(np.min(observed))]
        lower_candidates.extend(float(values[0]) for values in reference_values)
        second_axis.set_ylim(
            max(np.finfo(float).tiny, min(lower_candidates) / 2),
            float(np.max(observed)) * 2,
        )
    second_axis.set_xscale("log")
    second_axis.set_yscale("log")
    second_axis.set_title("A  Second-order longitudinal structure", loc="left")
    second_axis.set_xlabel("Mean observed separation [km]")
    second_axis.set_ylabel(r"$S_{ll}^{2}(r)$ [m$^2$ s$^{-2}$]")
    second_axis.legend(fontsize=7.5, loc="best")

    panel_b = frame[
        frame.panel_b_in_range.astype(bool) & np.isfinite(frame.mean_separation_m)
    ]
    supported_b = panel_b[panel_b.support_ok.astype(bool)]
    low_b = panel_b[~panel_b.support_ok.astype(bool)]
    epsilon_axis.plot(
        supported_b.mean_separation_m / 1000,
        supported_b.epsilon_four_fifths_m2_s3,
        color="black", marker="o", markerfacecolor="black", markersize=4,
        linewidth=0.9, label="four-fifths relation",
    )
    epsilon_axis.plot(
        supported_b.mean_separation_m / 1000,
        supported_b.epsilon_mixed_relation_m2_s3,
        color="#2563eb", marker="o", markerfacecolor="white", markersize=4,
        linewidth=0.9, label="mixed longitudinal-transverse relation",
    )
    if len(low_b):
        epsilon_axis.scatter(
            low_b.mean_separation_m / 1000,
            low_b.epsilon_four_fifths_m2_s3,
            color="#9ca3af", marker="x", s=28, label="low support",
        )
    epsilon_axis.axhline(0, color="#6b7280", linewidth=0.8)
    epsilon_axis.set_xscale("log")
    epsilon_axis.set_yscale(
        "symlog", linthresh=epsilon_linear_threshold_m2_s3, linscale=0.8,
    )
    epsilon_axis.set_title("B  Signed scaled third-order estimates", loc="left")
    epsilon_axis.set_xlabel("Mean observed separation [km]")
    epsilon_axis.set_ylabel(r"$\widehat{\varepsilon}(r)$ [m$^2$ s$^{-3}$]")
    epsilon_values = np.concatenate((
        panel_b.epsilon_four_fifths_m2_s3.to_numpy(dtype=float),
        panel_b.epsilon_mixed_relation_m2_s3.to_numpy(dtype=float),
    ))
    finite_epsilon = epsilon_values[np.isfinite(epsilon_values)]
    if len(finite_epsilon):
        epsilon_limit = max(
            float(np.max(np.abs(finite_epsilon))) * 1.25,
            epsilon_linear_threshold_m2_s3 * 2,
        )
        epsilon_axis.set_ylim(-epsilon_limit, epsilon_limit)
    epsilon_axis.legend(fontsize=7.5, loc="best")

    for axis in (second_axis, epsilon_axis):
        axis.grid(True, which="major", color="#d1d5db", linewidth=0.5)
        axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.3, linestyle=":")
    figure.suptitle(
        f"Array {int(summary['array_id']):03d} · Poje Figure 7 observational analogue\n"
        f"window {summary['window_id']}; source-supported bins retain exact Stage 2 weighting",
        fontsize=13, y=0.95,
    )
    figure.text(
        0.01, 0.035,
        r"Panel B: filled points use $-\frac{5}{4}\langle\Delta u_l^3\rangle/\langle r\rangle$; "
        r"open points use $-[\langle\Delta u_l^3\rangle+\langle\Delta u_l\Delta u_t^2\rangle]/(2\langle r\rangle)$."
        "\nThe signed symlog axis is an ARCTERX departure from Poje's positive-only log axis; negative estimates are not omitted.",
        fontsize=7, color="#374151",
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
