"""Review figures for signed ARCTERX epsilon diagnostics."""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


Q_LABEL = r"$q=-\frac{5}{4}(\Delta u_L)^3/\delta$  [m$^2$ s$^{-3}$]"
EPSILON_LABEL = r"$\widehat{\varepsilon}_{\mathrm{eff}}$  [m$^2$ s$^{-3}$]"


def _display_sample(frame: pd.DataFrame, maximum: int, identity: str) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    seed = int.from_bytes(sha256(identity.encode("utf-8")).digest()[:8], "big")
    return frame.sample(n=maximum, random_state=seed & 0xFFFFFFFF).sort_index()


def _symlog_limit(values: np.ndarray, configured_threshold: float) -> tuple[float, float]:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    maximum = float(np.max(finite)) if len(finite) else configured_threshold * 10
    maximum = max(maximum * 1.08, configured_threshold * 10)
    return -maximum, maximum


def plot_cluster_trajectory_window(
    diagnostics: pd.DataFrame,
    *,
    output_path: Path,
    array_id: int,
    cluster_id: str,
    activation_time_utc: str,
    window_end_hour: float,
    marked_elapsed_hours: tuple[float, ...],
    show_invalid_reconstructed_positions: bool,
    label_platforms: bool,
    mark_activation: bool,
    mark_window_boundaries: bool,
    dpi: int,
) -> None:
    """Plot reconstructed and analysis-valid cluster trajectories in metric space."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = diagnostics.copy()
    frame = frame[
        (frame.cluster_age_hours >= 0)
        & (frame.cluster_age_hours < window_end_hour)
    ].sort_values(["platform_id", "time_utc"])
    figure, axis = plt.subplots(figsize=(10, 8))
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.88)
    colors = plt.get_cmap("tab10")

    for color_index, (platform_id, track) in enumerate(
        frame.groupby("platform_id", sort=True)
    ):
        color = colors(color_index % 10)
        x = track.x_m.to_numpy(dtype=float)
        y = track.y_m.to_numpy(dtype=float)
        valid_position = np.isfinite(x) & np.isfinite(y)
        analysis_valid = track.analysis_valid.astype(bool).to_numpy() & valid_position
        axis.plot(
            np.where(valid_position, x, np.nan),
            np.where(valid_position, y, np.nan),
            color=color, alpha=0.22, linewidth=1.0,
        )
        axis.plot(
            np.where(analysis_valid, x, np.nan),
            np.where(analysis_valid, y, np.nan),
            color=color, linewidth=1.8, label=str(platform_id),
        )
        if show_invalid_reconstructed_positions:
            invalid = valid_position & ~analysis_valid
            if invalid.any():
                axis.scatter(
                    x[invalid], y[invalid], s=13, marker="x", color=color,
                    linewidths=0.7, alpha=0.65,
                )

        valid_track = track.loc[analysis_valid]
        if len(valid_track):
            start = valid_track.iloc[0]
            end = valid_track.iloc[-1]
            if mark_activation:
                axis.scatter(
                    [start.x_m], [start.y_m], s=42, marker="o", facecolors="white",
                    edgecolors=[color], linewidths=1.4, zorder=4,
                )
            if mark_window_boundaries:
                axis.scatter(
                    [end.x_m], [end.y_m], s=42, marker="D", color=[color], zorder=4,
                )
            if label_platforms:
                axis.annotate(
                    str(platform_id), (end.x_m, end.y_m),
                    xytext=(6, -12 + 6 * (color_index % 5)),
                    textcoords="offset points", fontsize=7, color=color,
                )
        for elapsed_hour in marked_elapsed_hours if mark_window_boundaries else ():
            if elapsed_hour <= 0 or elapsed_hour >= window_end_hour or not len(track):
                continue
            distance = np.abs(
                track.cluster_age_hours.to_numpy(dtype=float) - elapsed_hour
            )
            location = int(np.argmin(distance))
            if distance[location] <= 1e-8 and bool(analysis_valid[location]):
                row = track.iloc[location]
                axis.scatter(
                    [row.x_m], [row.y_m], s=32, marker="s", facecolors="white",
                    edgecolors=[color], linewidths=1.2, zorder=4,
                )

    valid_rows = frame[frame.analysis_valid.astype(bool)].copy()
    valid_rows = valid_rows[np.isfinite(valid_rows.x_m) & np.isfinite(valid_rows.y_m)]
    if len(valid_rows):
        centroid = valid_rows.groupby("time_utc", sort=True)[["x_m", "y_m"]].mean()
        axis.plot(
            centroid.x_m, centroid.y_m, color="black", linestyle="--",
            linewidth=1.25, alpha=0.85, label="valid-member centroid",
        )

    axis.set_aspect("equal", adjustable="datalim")
    axis.margins(x=0.10, y=0.08)
    axis.set_xlabel("Projected eastward coordinate x [m]")
    axis.set_ylabel("Projected northward coordinate y [m]")
    axis.set_title(
        f"Array {array_id:03d} · {cluster_id}\n"
        f"activation {activation_time_utc}; 0 ≤ elapsed time < {window_end_hour:g} h"
    )
    axis.grid(True, which="major", color="#d1d5db", linewidth=0.55, alpha=0.75)
    axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.4, linestyle=":")
    axis.minorticks_on()
    axis.legend(fontsize=7, loc="best", ncol=2)
    figure.text(
        0.02, 0.025,
        "Open circle: first valid point · square: configured intermediate time · "
        "diamond: last retained valid point\n"
        "x: reconstructed position with invalid analysis velocity",
        fontsize=7, color="#374151",
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_cluster_velocity_availability(
    diagnostics: pd.DataFrame,
    *,
    output_path: Path,
    array_id: int,
    cluster_id: str,
    activation_time_utc: str,
    window_end_hour: float,
    assigned_member_count: int,
    activation_threshold: int,
    dpi: int,
) -> None:
    """Plot valid platform velocities, relative motion, and availability counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = diagnostics[
        (diagnostics.cluster_age_hours >= 0)
        & (diagnostics.cluster_age_hours < window_end_hour)
    ].copy()
    frame = frame.sort_values(["platform_id", "time_utc"])
    frame["analysis_speed_m_s"] = np.hypot(
        frame.analysis_u_m_s.to_numpy(dtype=float),
        frame.analysis_v_m_s.to_numpy(dtype=float),
    )
    valid = frame.analysis_valid.astype(bool)
    valid_frame = frame[valid].copy()
    means = valid_frame.groupby("time_utc", sort=True)[
        ["analysis_u_m_s", "analysis_v_m_s", "analysis_speed_m_s"]
    ].mean()
    frame = frame.join(
        means[["analysis_u_m_s", "analysis_v_m_s"]],
        on="time_utc", rsuffix="_mean",
    )
    frame["relative_speed_m_s"] = np.hypot(
        frame.analysis_u_m_s - frame.analysis_u_m_s_mean,
        frame.analysis_v_m_s - frame.analysis_v_m_s_mean,
    )
    frame.loc[~frame.analysis_valid.astype(bool), "relative_speed_m_s"] = np.nan

    figure, axes = plt.subplots(
        5, 1, figsize=(14, 13), sharex=True,
        gridspec_kw={"height_ratios": (1, 1, 1, 1, 1.15)},
    )
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.09, top=0.91, hspace=0.16)
    colors = plt.get_cmap("tab10")
    metrics = (
        ("analysis_u_m_s", "A  Eastward velocity", r"$u$ [m s$^{-1}$]"),
        ("analysis_v_m_s", "B  Northward velocity", r"$v$ [m s$^{-1}$]"),
        ("analysis_speed_m_s", "C  Speed", r"$|\mathbf{u}|$ [m s$^{-1}$]"),
        (
            "relative_speed_m_s", "D  Velocity relative to valid-member mean",
            r"$|\mathbf{u}_i-\overline{\mathbf{u}}|$ [m s$^{-1}$]",
        ),
    )
    for color_index, (platform_id, track) in enumerate(
        frame.groupby("platform_id", sort=True)
    ):
        color = colors(color_index % 10)
        age = track.cluster_age_hours.to_numpy(dtype=float)
        for axis, (metric, _, _) in zip(axes[:4], metrics):
            values = track[metric].to_numpy(dtype=float)
            values = np.where(track.analysis_valid.astype(bool), values, np.nan)
            axis.plot(
                age, values, color=color,
                linewidth=1.0, alpha=0.88, label=str(platform_id),
            )
    if len(means):
        mean_timeline = (
            frame[["time_utc", "cluster_age_hours"]]
            .drop_duplicates("time_utc")
            .sort_values("time_utc")
            .set_index("time_utc")
        )
        means_for_plot = means.reindex(mean_timeline.index)
        mean_ages = mean_timeline.cluster_age_hours.to_numpy(dtype=float)
        for axis, metric in zip(
            axes[:3], ("analysis_u_m_s", "analysis_v_m_s", "analysis_speed_m_s")
        ):
            axis.plot(
                mean_ages, means_for_plot[metric].to_numpy(dtype=float), color="black",
                linewidth=1.8, label="valid-member mean", zorder=4,
            )

    for axis, (_, title, ylabel) in zip(axes[:4], metrics):
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_ylabel(ylabel)
        axis.axhline(0, color="#6b7280", linewidth=0.6, alpha=0.65)

    unique_times = (
        frame[["time_utc", "cluster_age_hours"]]
        .drop_duplicates("time_utc")
        .sort_values("time_utc")
    )
    time_index = pd.Index(unique_times.time_utc)
    ages = unique_times.cluster_age_hours.to_numpy(dtype=float)
    position_valid = (
        frame.assign(position_valid=np.isfinite(frame.x_m) & np.isfinite(frame.y_m))
        .groupby("time_utc").position_valid.sum()
        .reindex(time_index, fill_value=0)
        .to_numpy(dtype=int)
    )
    reference_valid = (
        frame.groupby("time_utc").reference_valid.sum()
        .reindex(time_index, fill_value=0).to_numpy(dtype=int)
    )
    analysis_valid_count = (
        frame.groupby("time_utc").analysis_valid.sum()
        .reindex(time_index, fill_value=0).to_numpy(dtype=int)
    )
    possible_pairs = analysis_valid_count * (analysis_valid_count - 1) // 2
    count_axis = axes[4]
    count_axis.step(
        ages, position_valid, where="post", linewidth=1.25,
        label="finite reconstructed positions",
    )
    count_axis.step(
        ages, reference_valid, where="post", linewidth=1.25,
        label="reference-valid velocities",
    )
    count_axis.step(
        ages, analysis_valid_count, where="post", linewidth=1.5,
        label="analysis-valid velocities",
    )
    count_axis.axhline(
        activation_threshold, color="#6b7280", linestyle=":", linewidth=1.1,
        label=f"activation threshold ({activation_threshold})",
    )
    count_axis.set_ylim(-0.15, max(assigned_member_count, activation_threshold) + 0.5)
    count_axis.set_ylabel("Platform count")
    count_axis.set_title("E  Availability and possible simultaneous pairs", loc="left", fontsize=10)
    pair_axis = count_axis.twinx()
    pair_axis.step(
        ages, possible_pairs, where="post", color="black", linestyle="--",
        linewidth=1.25, label="possible analysis-valid pairs",
    )
    pair_axis.set_ylabel("Pair count")
    pair_axis.set_ylim(-0.3, max(1, assigned_member_count * (assigned_member_count - 1) // 2) + 0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=7, ncol=3, loc="best")
    count_handles, count_labels = count_axis.get_legend_handles_labels()
    pair_handles, pair_labels = pair_axis.get_legend_handles_labels()
    count_axis.legend(
        count_handles + pair_handles, count_labels + pair_labels,
        fontsize=7, ncol=3, loc="best",
    )
    for axis in axes:
        axis.set_xlim(0, window_end_hour)
        axis.grid(True, which="major", color="#d1d5db", linewidth=0.5, alpha=0.75)
        axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.35, linestyle=":")
        axis.minorticks_on()
    axes[-1].set_xlabel("Elapsed time from fixed reference activation [h]")
    figure.suptitle(
        f"Array {array_id:03d} · {cluster_id} · valid velocity history\n"
        f"activation {activation_time_utc}; 0 ≤ elapsed time < {window_end_hour:g} h",
        fontsize=14,
    )
    figure.text(
        0.02, 0.025,
        "Velocity curves contain analysis-valid samples only. Relative speed uses the "
        "instantaneous mean of currently analysis-valid members.\n"
        "Possible pair count is n(n−1)/2 and does not apply separation-bin or minimum-distance filtering.",
        fontsize=7, color="#374151",
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_cluster_pair_relative_motion(
    observations: pd.DataFrame,
    scheduled_times: pd.DataFrame,
    *,
    output_path: Path,
    array_id: int,
    cluster_id: str,
    activation_time_utc: str,
    window_end_hour: float,
    derivative_span_minutes: float,
    dpi: int,
) -> None:
    """Plot pair separation, relative velocities, availability, and cubic sign driver."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = observations[
        (observations.cluster_age_hours >= 0)
        & (observations.cluster_age_hours < window_end_hour)
    ].copy()
    frame["time_utc"] = pd.to_datetime(frame.time_utc, utc=True)
    frame = frame.sort_values(["pair_id", "time_utc"])
    timeline = (
        scheduled_times[["time_utc", "cluster_age_hours"]]
        .drop_duplicates("time_utc")
        .sort_values("time_utc")
    )
    timeline["time_utc"] = pd.to_datetime(timeline.time_utc, utc=True)
    timeline = timeline[
        (timeline.cluster_age_hours >= 0)
        & (timeline.cluster_age_hours < window_end_hour)
    ]
    timeline_index = pd.DatetimeIndex(timeline.time_utc)
    ages = timeline.cluster_age_hours.to_numpy(dtype=float)
    if len(timeline_index) < 2:
        raise ValueError("Pair-relative diagnostic requires at least two scheduled times")
    cadence_seconds = float(
        np.median(np.diff(timeline_index.asi8)) / 1_000_000_000
    )
    half_span_seconds = derivative_span_minutes * 60 / 2
    half_steps_float = half_span_seconds / cadence_seconds
    half_steps = int(round(half_steps_float))
    if half_steps < 1 or not math.isclose(half_steps_float, half_steps, abs_tol=1e-9):
        raise ValueError("Velocity half-span must be an integer multiple of diagnostic cadence")

    figure, axes = plt.subplots(
        6, 1, figsize=(14, 16), sharex=True,
        gridspec_kw={"height_ratios": (1.2, 1, 1, 1, 0.9, 1.05)},
    )
    figure.subplots_adjust(left=0.09, right=0.95, bottom=0.08, top=0.92, hspace=0.18)
    colors = plt.get_cmap("tab20")
    for color_index, (pair_id, pair) in enumerate(frame.groupby("pair_id", sort=True)):
        pair = pair.drop_duplicates("time_utc", keep="last").set_index("time_utc")
        aligned = pair.reindex(timeline_index)
        separation = aligned.separation_m.to_numpy(dtype=float)
        longitudinal = aligned.delta_u_l_m_s.to_numpy(dtype=float)
        transverse = aligned.delta_u_t_m_s.to_numpy(dtype=float)
        separation_rate = np.full(len(aligned), np.nan, dtype=float)
        if len(aligned) > 2 * half_steps:
            separation_rate[half_steps:-half_steps] = (
                separation[2 * half_steps:] - separation[:-2 * half_steps]
            ) / (2 * half_span_seconds)
        first_id = str(pair.platform_id_1.dropna().iloc[0])
        second_id = str(pair.platform_id_2.dropna().iloc[0])
        label = f"{first_id[-5:]}–{second_id[-5:]}"
        color = colors(color_index % 20)
        axes[0].plot(ages, separation, color=color, linewidth=1.0, label=label)
        axes[1].plot(ages, longitudinal, color=color, linewidth=0.95)
        axes[2].plot(ages, separation_rate, color=color, linewidth=0.95)
        axes[3].plot(ages, transverse, color=color, linewidth=0.95)
    axes[0].set_yscale("log")
    axes[0].set_title("A  Pair separation", loc="left", fontsize=10)
    axes[0].set_ylabel(r"$\delta$ [m]")
    axes[0].legend(fontsize=6.5, ncol=4, loc="best", title="endpoint suffixes")
    axes[1].set_title("B  Longitudinal relative velocity", loc="left", fontsize=10)
    axes[1].set_ylabel(r"$\Delta u_L$ [m s$^{-1}$]")
    axes[2].set_title(
        f"C  Centered separation rate ({derivative_span_minutes:g}-minute span)",
        loc="left", fontsize=10,
    )
    axes[2].set_ylabel(r"$d\delta/dt$ [m s$^{-1}$]")
    axes[3].set_title("D  Transverse relative velocity", loc="left", fontsize=10)
    axes[3].set_ylabel(r"$\Delta u_T$ [m s$^{-1}$]")
    for axis in axes[1:4]:
        axis.axhline(0, color="black", linewidth=0.7, alpha=0.7)

    time_groups = frame.groupby("time_utc", sort=True)
    valid_pair_count = time_groups.size().reindex(timeline_index, fill_value=0).to_numpy(dtype=int)
    separating_count = time_groups.delta_u_l_m_s.apply(
        lambda values: int(np.sum(np.asarray(values, dtype=float) > 0))
    ).reindex(timeline_index, fill_value=0).to_numpy(dtype=int)
    approaching_count = time_groups.delta_u_l_m_s.apply(
        lambda values: int(np.sum(np.asarray(values, dtype=float) < 0))
    ).reindex(timeline_index, fill_value=0).to_numpy(dtype=int)
    count_axis = axes[4]
    count_axis.step(ages, valid_pair_count, where="post", color="black", label="valid pairs")
    count_axis.step(
        ages, separating_count, where="post", color="#dc2626", label=r"separating ($\Delta u_L>0$)",
    )
    count_axis.step(
        ages, approaching_count, where="post", color="#2563eb", label=r"approaching ($\Delta u_L<0$)",
    )
    count_axis.set_ylabel("Pair count")
    count_axis.set_title("E  Valid-pair direction counts", loc="left", fontsize=10)
    count_axis.legend(fontsize=7, ncol=3, loc="best")

    cube = frame.delta_u_l_m_s.to_numpy(dtype=float) ** 3
    cube_frame = pd.DataFrame({"time_utc": frame.time_utc, "cube": cube})
    net_cube = cube_frame.groupby("time_utc").cube.sum().reindex(
        timeline_index, fill_value=0.0,
    ).to_numpy(dtype=float)
    positive_cube = cube_frame.assign(value=np.maximum(cube_frame.cube, 0)).groupby(
        "time_utc"
    ).value.sum().reindex(timeline_index, fill_value=0.0).to_numpy(dtype=float)
    negative_cube = cube_frame.assign(value=np.minimum(cube_frame.cube, 0)).groupby(
        "time_utc"
    ).value.sum().reindex(timeline_index, fill_value=0.0).to_numpy(dtype=float)
    cube_axis = axes[5]
    cube_axis.plot(ages, positive_cube, color="#dc2626", linewidth=1.0, label="separating cube sum")
    cube_axis.plot(ages, negative_cube, color="#2563eb", linewidth=1.0, label="approaching cube sum")
    cube_axis.plot(ages, net_cube, color="black", linewidth=1.5, label="net cube sum")
    maximum_cube = float(np.nanmax(np.abs(np.concatenate((positive_cube, negative_cube, net_cube)))))
    cube_axis.set_yscale("symlog", linthresh=max(maximum_cube * 1e-4, 1e-12))
    cube_axis.axhline(0, color="black", linewidth=0.7)
    cube_axis.set_ylabel(r"$\sum(\Delta u_L)^3$ [m$^3$ s$^{-3}$]")
    cube_axis.set_title(
        "F  Signed cubic numerator across all separations",
        loc="left", fontsize=10,
    )
    cube_axis.legend(fontsize=7, ncol=3, loc="best")

    for axis in axes:
        axis.set_xlim(0, window_end_hour)
        axis.grid(True, which="major", color="#d1d5db", linewidth=0.5, alpha=0.75)
        axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.35, linestyle=":")
        axis.minorticks_on()
    axes[-1].set_xlabel("Elapsed time from fixed reference activation [h]")
    figure.suptitle(
        f"Array {array_id:03d} · {cluster_id} · pair-relative motion\n"
        f"activation {activation_time_utc}; 0 ≤ elapsed time < {window_end_hour:g} h",
        fontsize=14,
    )
    figure.text(
        0.02, 0.018,
        r"Positive $\Delta u_L$ means separation and contributes a positive cube, which drives "
        r"the four-fifths epsilon estimate negative. Panel F pools scales only as a time diagnostic."
        "\nPair curves contain gaps whenever the saved simultaneous pair observation is unavailable.",
        fontsize=7, color="#374151",
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_cluster_time_scale_contributions(
    contributions: pd.DataFrame,
    scheduled_times: pd.DataFrame,
    source_summary: pd.DataFrame,
    *,
    output_path: Path,
    array_id: int,
    cluster_id: str,
    activation_time_utc: str,
    window_id: str,
    window_end_hour: float,
    separation_edges_m: np.ndarray,
    identity_maximum_absolute_error_m2_s3: float,
    dpi: int,
) -> None:
    """Plot the exact time-scale decomposition of the signed epsilon estimator."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = (
        scheduled_times[["time_utc", "cluster_age_hours"]]
        .drop_duplicates("time_utc")
        .sort_values("time_utc")
    )
    timeline["time_utc"] = pd.to_datetime(timeline.time_utc, utc=True)
    timeline = timeline[
        (timeline.cluster_age_hours >= 0)
        & (timeline.cluster_age_hours < window_end_hour)
    ]
    ages = timeline.cluster_age_hours.to_numpy(dtype=float)
    if len(ages) < 2:
        raise ValueError("Time-scale diagnostic requires at least two scheduled times")
    midpoints = (ages[:-1] + ages[1:]) / 2
    time_edges = np.concatenate((
        [max(0.0, ages[0] - (midpoints[0] - ages[0]))],
        midpoints,
        [min(window_end_hour, ages[-1] + (ages[-1] - midpoints[-1]))],
    ))
    timeline_index = pd.Index(timeline.time_utc)
    bin_count = len(separation_edges_m) - 1
    contribution_matrix = np.full((bin_count, len(timeline)), np.nan)
    count_matrix = np.full((bin_count, len(timeline)), np.nan)
    influence_matrix = np.full((bin_count, len(timeline)), np.nan)
    time_lookup = {value: index for index, value in enumerate(timeline_index)}
    for row in contributions.itertuples(index=False):
        column = time_lookup.get(pd.Timestamp(row.time_utc))
        if column is None:
            continue
        bin_index = int(row.separation_bin_index)
        contribution_matrix[bin_index, column] = row.epsilon_additive_contribution_m2_s3
        count_matrix[bin_index, column] = row.pair_observation_count
        influence_matrix[bin_index, column] = row.maximum_absolute_cube_fraction

    occupied = np.flatnonzero(np.isfinite(count_matrix).any(axis=1))
    if not len(occupied):
        raise ValueError("Time-scale diagnostic has no occupied separation bins")
    first_bin = max(0, int(occupied.min()) - 1)
    final_bin = min(bin_count, int(occupied.max()) + 2)
    y_edges = np.asarray(separation_edges_m[first_bin:final_bin + 1], dtype=float)
    contribution_display = contribution_matrix[first_bin:final_bin]
    count_display = count_matrix[first_bin:final_bin]
    influence_display = influence_matrix[first_bin:final_bin]

    finite_contributions = np.abs(contribution_display[np.isfinite(contribution_display)])
    contribution_maximum = (
        float(np.max(finite_contributions)) if len(finite_contributions) else 1e-12
    )
    contribution_maximum = max(contribution_maximum, 1e-14)
    contribution_norm = SymLogNorm(
        linthresh=max(contribution_maximum * 1e-3, 1e-14),
        vmin=-contribution_maximum,
        vmax=contribution_maximum,
        base=10,
    )

    figure = plt.figure(figsize=(17, 11))
    grid = figure.add_gridspec(
        3, 2, width_ratios=(5.2, 1.35),
        left=0.08, right=0.94, bottom=0.1, top=0.89, hspace=0.16, wspace=0.2,
    )
    contribution_axis = figure.add_subplot(grid[0, 0])
    count_axis = figure.add_subplot(grid[1, 0], sharex=contribution_axis)
    influence_axis = figure.add_subplot(grid[2, 0], sharex=contribution_axis)
    profile_axis = figure.add_subplot(grid[:, 1])

    contribution_colormap = plt.get_cmap("RdBu_r").copy()
    contribution_colormap.set_bad("#e5e7eb")
    contribution_mesh = contribution_axis.pcolormesh(
        time_edges, y_edges, np.ma.masked_invalid(contribution_display),
        cmap=contribution_colormap, norm=contribution_norm, shading="flat",
    )
    contribution_axis.set_title(
        "A  Additive contribution to final epsilon (blue negative, red positive)",
        loc="left", fontsize=10,
    )
    contribution_colorbar = figure.colorbar(
        contribution_mesh, ax=contribution_axis, pad=0.012,
    )
    contribution_colorbar.set_label(r"$c_{t,b}$ [m$^2$ s$^{-3}$]")

    finite_counts = count_display[np.isfinite(count_display)]
    count_maximum = max(1.01, float(np.max(finite_counts)))
    count_colormap = plt.get_cmap("viridis").copy()
    count_colormap.set_bad("#e5e7eb")
    count_mesh = count_axis.pcolormesh(
        time_edges, y_edges, np.ma.masked_invalid(count_display),
        cmap=count_colormap, norm=LogNorm(vmin=1, vmax=count_maximum), shading="flat",
    )
    count_axis.set_title("B  Pair-time support in each cell", loc="left", fontsize=10)
    count_colorbar = figure.colorbar(count_mesh, ax=count_axis, pad=0.012)
    count_colorbar.set_label("Pair observations")

    influence_colormap = plt.get_cmap("magma").copy()
    influence_colormap.set_bad("#e5e7eb")
    influence_mesh = influence_axis.pcolormesh(
        time_edges, y_edges, np.ma.masked_invalid(influence_display),
        cmap=influence_colormap, norm=Normalize(0, 1), shading="flat",
    )
    influence_axis.set_title(
        "C  Largest single-observation share of absolute cubic magnitude",
        loc="left", fontsize=10,
    )
    influence_colorbar = figure.colorbar(influence_mesh, ax=influence_axis, pad=0.012)
    influence_colorbar.set_label("Maximum absolute-cube fraction")

    reconstructed = contributions.groupby("separation_bin_index").agg(
        epsilon_reconstructed_m2_s3=("epsilon_additive_contribution_m2_s3", "sum"),
        separation_bin_nominal_center_m=("separation_bin_nominal_center_m", "first"),
    ).reset_index()
    profile_axis.plot(
        reconstructed.epsilon_reconstructed_m2_s3,
        reconstructed.separation_bin_nominal_center_m,
        color="#111827", marker="o", markersize=4, linewidth=1.2,
        label=r"$\sum_t c_{t,b}$",
    )
    finite_source = source_summary[np.isfinite(source_summary.epsilon_eff_m2_s3)]
    profile_axis.scatter(
        finite_source.epsilon_eff_m2_s3,
        finite_source.separation_bin_nominal_center_m,
        marker="x", color="#dc2626", s=38, label="saved epsilon",
    )
    profile_values = np.concatenate((
        reconstructed.epsilon_reconstructed_m2_s3.to_numpy(dtype=float),
        finite_source.epsilon_eff_m2_s3.to_numpy(dtype=float),
    ))
    profile_maximum = max(float(np.nanmax(np.abs(profile_values))), 1e-12)
    profile_threshold = max(profile_maximum * 1e-4, 1e-12)
    profile_axis.set_xscale("symlog", linthresh=profile_threshold)
    profile_middle = math.sqrt(profile_maximum * profile_threshold)
    profile_axis.set_xticks([
        -profile_maximum, -profile_middle, 0, profile_middle, profile_maximum,
    ])
    profile_axis.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _: "0" if value == 0 else f"{value:.0e}",
    ))
    profile_axis.tick_params(axis="x", labelrotation=35, labelsize=7)
    profile_axis.minorticks_off()
    profile_axis.axvline(0, color="#6b7280", linewidth=0.8)
    profile_axis.set_title("D  Exact time sum", fontsize=10)
    profile_axis.set_xlabel(EPSILON_LABEL)
    profile_axis.legend(fontsize=7, loc="best")

    for axis in (contribution_axis, count_axis, influence_axis, profile_axis):
        axis.set_yscale("log")
        axis.set_ylim(y_edges[0], y_edges[-1])
        axis.grid(True, which="major", color="#d1d5db", linewidth=0.4, alpha=0.55)
    for axis in (contribution_axis, count_axis, influence_axis):
        axis.set_xlim(0, window_end_hour)
        axis.set_ylabel(r"Separation $\delta$ [m]")
    contribution_axis.tick_params(labelbottom=False)
    count_axis.tick_params(labelbottom=False)
    influence_axis.set_xlabel("Elapsed time from fixed reference activation [h]")
    profile_axis.set_ylabel(r"Separation $\delta$ [m]")

    figure.suptitle(
        f"Array {array_id:03d} · {cluster_id} · {window_id.replace('_', ' ')}\n"
        f"activation {activation_time_utc}; exact decomposition maximum identity error "
        f"{identity_maximum_absolute_error_m2_s3:.3g} m² s⁻³",
        fontsize=14,
    )
    figure.text(
        0.08, 0.025,
        r"Each occupied cell is $c_{t,b}=-(5/4)\,\sum_{i\in(t,b)}(\Delta u_{L,i})^3/"
        r"\sum_{i\in b}\delta_i$, so $\sum_t c_{t,b}=\widehat{\varepsilon}_b$ exactly. "
        "Grey cells have no valid pair observation.\n"
        "Panel C equals one when a single observation supplies all absolute cubic magnitude "
        "in that time-scale cell; identities are recorded in the derived table.",
        fontsize=7, color="#374151",
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_cluster_influence_audit(
    audit: pd.DataFrame,
    *,
    output_path: Path,
    array_id: int,
    cluster_id: str,
    window_id: str,
    small_scale_maximum_m: float,
    dominant_observation_fraction: float,
    dominant_platform_relative_change: float,
    dpi: int,
) -> None:
    """Plot observation and platform influence for negative small-scale bins."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = audit.sort_values("separation_bin_index").copy()
    separation = frame.mean_separation_m.to_numpy(dtype=float)
    baseline = frame.epsilon_eff_m2_s3.to_numpy(dtype=float)
    after_extreme = frame.epsilon_without_most_extreme_m2_s3.to_numpy(dtype=float)
    lopo_lower = frame.lopo_minimum_epsilon_m2_s3.to_numpy(dtype=float)
    lopo_upper = frame.lopo_maximum_epsilon_m2_s3.to_numpy(dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    epsilon_axis, influence_axis, support_axis, classification_axis = axes.ravel()

    epsilon_axis.fill_between(
        separation, lopo_lower, lopo_upper, color="#bfdbfe", alpha=0.65,
        label="leave-one-platform range",
    )
    epsilon_axis.plot(
        separation, baseline, "o-", color="#111827", linewidth=1.25,
        markersize=4, label="saved epsilon",
    )
    epsilon_axis.plot(
        separation, after_extreme, "s--", color="#d97706", linewidth=1,
        markersize=4, label="without largest |cube| observation",
    )
    epsilon_axis.axhline(0, color="#6b7280", linewidth=0.8)
    epsilon_values = np.concatenate((baseline, after_extreme, lopo_lower, lopo_upper))
    epsilon_maximum = max(
        float(np.nanmax(np.abs(epsilon_values[np.isfinite(epsilon_values)]))), 1e-12,
    )
    epsilon_axis.set_yscale("symlog", linthresh=max(epsilon_maximum * 1e-4, 1e-12))
    epsilon_axis.set_ylabel(EPSILON_LABEL)
    epsilon_axis.set_title("A  Removal sensitivity", loc="left", fontsize=10)
    epsilon_axis.legend(fontsize=7, loc="best")

    influence_axis.plot(
        separation, frame.maximum_absolute_cube_fraction, "o-",
        color="#7c3aed", linewidth=1.1, markersize=4,
        label="largest observation / total |cube|",
    )
    influence_axis.plot(
        separation, frame.maximum_lopo_relative_change, "s-",
        color="#0891b2", linewidth=1.1, markersize=4,
        label="maximum platform-removal relative change",
    )
    influence_axis.axhline(
        dominant_observation_fraction, color="#7c3aed", linestyle=":",
        linewidth=1, label=f"observation threshold ({dominant_observation_fraction:g})",
    )
    influence_axis.axhline(
        dominant_platform_relative_change, color="#0891b2", linestyle="--",
        linewidth=1, label=f"platform threshold ({dominant_platform_relative_change:g})",
    )
    influence_axis.set_yscale("symlog", linthresh=0.1)
    influence_axis.set_ylabel("Influence fraction or relative change")
    influence_axis.set_title("B  Influence magnitude", loc="left", fontsize=10)
    influence_axis.legend(fontsize=7, loc="best")

    support_axis.plot(
        separation, frame.observation_count, "o-", markersize=4,
        label="pair-time observations",
    )
    support_axis.plot(
        separation, frame.unique_pair_count, "s-", markersize=4,
        label="distinct pairs",
    )
    support_axis.plot(
        separation, frame.platform_count, "^-", markersize=4,
        label="platforms",
    )
    support_axis.plot(
        separation, frame.unique_utc_count, "d-", markersize=4,
        label="UTC timestamps",
    )
    support_axis.set_yscale("symlog", linthresh=1)
    support_axis.set_ylabel("Support count")
    support_axis.set_title("C  Original within-bin support", loc="left", fontsize=10)
    support_axis.legend(fontsize=7, loc="best")

    flag_columns = [
        ("low_support", "low support"),
        ("extreme_observation_sign_reversal", "event sign reversal"),
        ("platform_removal_sign_reversal", "platform sign reversal"),
        ("dominant_observation", "observation concentrated"),
        ("dominant_platform", "platform sensitive"),
        ("persistent_negative_under_tested_removals", "persistent in tests"),
    ]
    for row_index, (column, _) in enumerate(flag_columns):
        selected = frame[frame[column].astype(bool)]
        if len(selected):
            classification_axis.scatter(
                selected.mean_separation_m,
                np.full(len(selected), row_index),
                marker="s", s=80, color="#1f2937",
            )
    classification_axis.set_yticks(
        np.arange(len(flag_columns)), [label for _, label in flag_columns],
    )
    classification_axis.set_ylim(len(flag_columns) - 0.5, -0.5)
    classification_axis.set_title("D  Non-exclusive audit flags", loc="left", fontsize=10)
    classification_axis.set_ylabel("Flag")

    for axis in axes.ravel():
        axis.set_xscale("log")
        axis.set_xlim(
            max(1e-9, float(frame.separation_bin_lower_m.min()) / 1.08),
            min(
                small_scale_maximum_m,
                float(frame.separation_bin_upper_m.max()) * 1.08,
            ),
        )
        axis.set_xlabel("Mean observed separation [m]")
        axis.grid(True, which="major", color="#d1d5db", linewidth=0.5, alpha=0.7)
        axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.35, linestyle=":")
    figure.suptitle(
        f"Array {array_id:03d} · {cluster_id} · negative small-scale influence audit\n"
        f"{window_id.replace('_', ' ')}; configured small-scale upper edge ≤ "
        f"{small_scale_maximum_m:g} m",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_cluster_product(
    observations: pd.DataFrame,
    summary: pd.DataFrame,
    histogram: pd.DataFrame,
    *,
    output_path: Path,
    array_id: int,
    cluster_id: str,
    analysis_type: str,
    window_id: str,
    activation_time_utc: str,
    separation_edges_m: np.ndarray,
    q_edges_m2_s3: np.ndarray,
    q_linear_threshold_m2_s3: float,
    epsilon_linear_threshold_m2_s3: float,
    scatter_maximum_points: int,
    dpi: int,
    heatmap_probability_maximum: float,
) -> None:
    """Render scatter, epsilon, conditional q, and support in one figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identity = f"{array_id}|{cluster_id}|{analysis_type}|{window_id}"
    displayed = _display_sample(observations, scatter_maximum_points, identity)
    figure = plt.figure(figsize=(17, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(4.0, 1.45))
    scatter_axis = figure.add_subplot(grid[0, 0])
    epsilon_axis = figure.add_subplot(grid[0, 1])
    heatmap_axis = figure.add_subplot(grid[0, 2])
    support_axis = figure.add_subplot(grid[1, :])

    scatter_axis.scatter(
        displayed.separation_m, displayed.q_m2_s3,
        s=9, alpha=0.22, linewidths=0, color="#155e75", rasterized=True,
    )
    scatter_axis.axhline(0, color="black", linewidth=0.8)
    scatter_axis.set_title(f"A  Pair contributions ({len(displayed):,}/{len(observations):,} shown)")
    scatter_axis.set_xlabel("Instantaneous separation [m]")
    scatter_axis.set_ylabel(Q_LABEL)
    scatter_axis.set_yscale("symlog", linthresh=q_linear_threshold_m2_s3)
    scatter_axis.set_ylim(*_symlog_limit(observations.q_m2_s3, q_linear_threshold_m2_s3))

    finite = summary[np.isfinite(summary.epsilon_eff_m2_s3)].copy()
    supported = finite[finite.support_ok.astype(bool)]
    low = finite[~finite.support_ok.astype(bool)]
    if len(low):
        epsilon_axis.scatter(
            low.mean_separation_m, low.epsilon_eff_m2_s3,
            marker="o", facecolors="none", edgecolors="#9ca3af", s=38,
            label="descriptive/low support", zorder=2,
        )
    if len(supported):
        epsilon_axis.plot(
            supported.mean_separation_m, supported.epsilon_eff_m2_s3,
            color="#b91c1c", marker="o", linewidth=1.1, markersize=4,
            label="supported point estimate", zorder=3,
        )
    available = finite[finite.confidence_interval_status == "available"]
    if len(available):
        lower = available.epsilon_eff_m2_s3 - available.epsilon_ci_lower_m2_s3
        upper = available.epsilon_ci_upper_m2_s3 - available.epsilon_eff_m2_s3
        epsilon_axis.errorbar(
            available.mean_separation_m, available.epsilon_eff_m2_s3,
            yerr=np.vstack((lower, upper)), fmt="none", ecolor="#7f1d1d",
            elinewidth=1, capsize=2, alpha=0.85, label="95% block-bootstrap CI",
            zorder=1,
        )
    epsilon_axis.axhline(0, color="black", linewidth=0.9)
    epsilon_axis.set_title("B  Four-fifths signed estimate")
    epsilon_axis.set_xlabel("Mean observed separation [m]")
    epsilon_axis.set_ylabel(EPSILON_LABEL)
    epsilon_axis.set_yscale("symlog", linthresh=epsilon_linear_threshold_m2_s3)
    epsilon_values = np.concatenate((
        finite.epsilon_eff_m2_s3.to_numpy(dtype=float),
        finite.epsilon_ci_lower_m2_s3.to_numpy(dtype=float),
        finite.epsilon_ci_upper_m2_s3.to_numpy(dtype=float),
    )) if len(finite) else np.asarray([])
    epsilon_axis.set_ylim(*_symlog_limit(epsilon_values, epsilon_linear_threshold_m2_s3))
    if len(finite):
        epsilon_axis.legend(fontsize=8, loc="best")

    separation_count = len(separation_edges_m) - 1
    q_count = len(q_edges_m2_s3) - 1
    matrix = np.full((q_count, separation_count), np.nan, dtype=float)
    for row in histogram.itertuples(index=False):
        matrix[int(row.q_bin_index), int(row.separation_bin_index)] = (
            row.conditional_probability_mass
        )
    masked = np.ma.masked_invalid(matrix)
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#dedede")
    mesh = heatmap_axis.pcolormesh(
        separation_edges_m, q_edges_m2_s3, masked,
        cmap=colormap, norm=Normalize(0, heatmap_probability_maximum), shading="flat",
    )
    heatmap_axis.axhline(0, color="white", linewidth=0.7, alpha=0.9)
    heatmap_axis.set_yscale("symlog", linthresh=q_linear_threshold_m2_s3)
    heatmap_axis.set_title("C  Conditional signed-q distribution")
    heatmap_axis.set_xlabel("Instantaneous separation [m]")
    heatmap_axis.set_ylabel(Q_LABEL)
    colorbar = figure.colorbar(mesh, ax=heatmap_axis, pad=0.02)
    colorbar.set_label("Conditional probability mass per q bin")

    x = summary.separation_bin_nominal_center_m.to_numpy(dtype=float)
    observation_count = summary.observation_count.to_numpy(dtype=float)
    support_axis.step(x, observation_count, where="mid", color="#111827", label="pair-time observations")
    support_axis.plot(
        x, summary.unique_utc_count, "o-", ms=3, lw=1,
        label="distinct UTC timestamps",
    )
    support_axis.plot(
        x, summary.unique_pair_count, "s-", ms=3, lw=1,
        label="distinct pair IDs",
    )
    support_axis.plot(
        x, summary.platform_count, "^-", ms=3, lw=1,
        label="distinct platform IDs",
    )
    support_axis.set_yscale("symlog", linthresh=1, subs=np.arange(2, 10))
    support_axis.set_ylabel("Support count")
    support_axis.set_xlabel("Separation-bin nominal center [m]")
    support_axis.set_title("D  Sampling support by separation bin")
    support_axis.legend(ncol=4, fontsize=8, loc="upper right")

    for axis in (scatter_axis, epsilon_axis, heatmap_axis, support_axis):
        axis.set_xscale("log")
        axis.set_xlim(float(separation_edges_m[0]), float(separation_edges_m[-1]))
        axis.grid(True, which="major", color="#d1d5db", linewidth=0.5, alpha=0.65)
    support_axis.grid(
        True, which="minor", axis="y", color="#d1d5db", linewidth=0.45,
        linestyle=":", alpha=0.8,
    )

    tail = histogram.groupby("separation_bin_id", sort=False).first()
    underflow = int(tail.q_underflow_count.sum()) if len(tail) else 0
    overflow = int(tail.q_overflow_count.sum()) if len(tail) else 0
    if analysis_type == "snapshot":
        product_label = "initial snapshot — descriptive; temporal CI not applicable"
    else:
        product_label = window_id.replace("_", " ")
    figure.suptitle(
        f"Array {array_id:03d} · {cluster_id} · {product_label}\n"
        f"reference activation {activation_time_utc}; q display tails below/above edges: "
        f"{underflow}/{overflow}",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_cluster_comparison(
    summaries: pd.DataFrame,
    *,
    output_path: Path,
    array_id: int,
    window_id: str,
    separation_edges_m: np.ndarray,
    epsilon_linear_threshold_m2_s3: float,
    dpi: int,
) -> None:
    """Compare supported cluster curves for one temporal window."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    colors = plt.get_cmap("tab20")
    plotted_values: list[float] = []
    for index, (cluster_id, group) in enumerate(summaries.groupby("cluster_id", sort=True)):
        supported = group[
            group.support_ok.astype(bool) & np.isfinite(group.epsilon_eff_m2_s3)
        ].sort_values("mean_separation_m")
        if supported.empty:
            continue
        values = supported.epsilon_eff_m2_s3.to_numpy(dtype=float)
        plotted_values.extend(values.tolist())
        axis.plot(
            supported.mean_separation_m, values, marker="o", markersize=3,
            linewidth=1, color=colors(index % 20), label=str(cluster_id),
        )
    axis.axhline(0, color="black", linewidth=0.9)
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=epsilon_linear_threshold_m2_s3)
    axis.set_xlim(float(separation_edges_m[0]), float(separation_edges_m[-1]))
    axis.set_ylim(*_symlog_limit(np.asarray(plotted_values), epsilon_linear_threshold_m2_s3))
    axis.set_xlabel("Mean observed separation [m]")
    axis.set_ylabel(EPSILON_LABEL)
    axis.set_title(
        f"Array {array_id:03d} · {window_id.replace('_', ' ')} · supported cluster estimates"
    )
    axis.grid(True, which="major", color="#d1d5db", linewidth=0.5)
    axis.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_window_small_multiples(
    summaries: pd.DataFrame,
    *,
    output_path: Path,
    array_id: int,
    separation_edges_m: np.ndarray,
    epsilon_linear_threshold_m2_s3: float,
    dpi: int,
) -> None:
    """Show configured temporal windows within every cluster without pooling."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clusters = sorted(summaries.cluster_id.unique().tolist())
    columns = 4
    rows = max(1, math.ceil(len(clusters) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.2 * rows),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    values = summaries.epsilon_eff_m2_s3.to_numpy(dtype=float)
    y_limits = _symlog_limit(values, epsilon_linear_threshold_m2_s3)
    colors = plt.get_cmap("Dark2")
    for cluster_index, cluster_id in enumerate(clusters):
        axis = axes_array[cluster_index]
        cluster = summaries[summaries.cluster_id == cluster_id]
        for window_index, (window_id, group) in enumerate(cluster.groupby("window_id", sort=True)):
            supported = group[
                group.support_ok.astype(bool) & np.isfinite(group.epsilon_eff_m2_s3)
            ].sort_values("mean_separation_m")
            if len(supported):
                axis.plot(
                    supported.mean_separation_m, supported.epsilon_eff_m2_s3,
                    marker="o", markersize=2.5, linewidth=1,
                    color=colors(window_index % 8), label=str(window_id).replace("initial_", ""),
                )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=epsilon_linear_threshold_m2_s3)
        axis.set_xlim(float(separation_edges_m[0]), float(separation_edges_m[-1]))
        axis.set_ylim(*y_limits)
        axis.set_title(str(cluster_id), fontsize=9)
        axis.grid(True, which="major", color="#e5e7eb", linewidth=0.45)
        if cluster_index == 0:
            axis.legend(fontsize=7)
    for axis in axes_array[len(clusters):]:
        axis.set_visible(False)
    figure.supxlabel("Mean observed separation [m]")
    figure.supylabel(EPSILON_LABEL)
    figure.suptitle(f"Array {array_id:03d} · supported temporal-window estimates by cluster")
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
