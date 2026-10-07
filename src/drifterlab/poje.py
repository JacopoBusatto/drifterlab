"""Trajectory-only statistics used in the Poje et al. comparison."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from drifterlab.epsilon import bin_indices, stable_pair_id


@dataclass(frozen=True)
class DispersionResult:
    timeseries: pd.DataFrame
    pair_inventory: pd.DataFrame
    pair_observations: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True)
class SeparationMemoryGroup:
    identifier: str
    start_day: float
    end_day: float


@dataclass(frozen=True)
class Figure5Result:
    separation_memory: pd.DataFrame
    separation_memory_samples: pd.DataFrame
    lagrangian_structure: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True)
class IncrementDistributionScale:
    identifier: str
    panel_group: str
    target_center_m: float


@dataclass(frozen=True)
class Figure6Result:
    distributions: pd.DataFrame
    statistics: pd.DataFrame
    observations: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True)
class Figure7Result:
    structure_functions: pd.DataFrame
    summary: dict[str, object]


def _bootstrap_mean_intervals(
    values: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    random_seed: int,
    time_batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample fixed pair identities synchronously across the full time curve."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[0]:
        raise ValueError("Bootstrap values must be a nonempty pair-by-time matrix")
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("Bootstrap confidence level must lie between zero and one")
    pair_count, time_count = values.shape
    random = np.random.default_rng(random_seed)
    weights = random.multinomial(
        pair_count, np.full(pair_count, 1 / pair_count), size=replicates,
    ).astype(float)
    lower = np.full(time_count, np.nan)
    median = np.full(time_count, np.nan)
    upper = np.full(time_count, np.nan)
    alpha = (1 - confidence_level) / 2
    percentiles = (100 * alpha, 50.0, 100 * (1 - alpha))
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    for start in range(0, time_count, time_batch_size):
        stop = min(time_count, start + time_batch_size)
        numerator = weights @ filled[:, start:stop]
        denominator = weights @ finite[:, start:stop].astype(float)
        bootstrap = np.divide(
            numerator, denominator,
            out=np.full(numerator.shape, np.nan), where=denominator > 0,
        )
        lower[start:stop], median[start:stop], upper[start:stop] = np.nanpercentile(
            bootstrap, percentiles, axis=0,
        )
    return lower, median, upper


def calculate_figure4_dispersion(
    platform_diagnostics: pd.DataFrame,
    *,
    array_id: int,
    window_id: str,
    window_end_hours: float,
    absolute_plot_end_days: float,
    initial_pair_separation_maximum_m: float,
    ballistic_fit_end_hours: float,
    richardson_fit_start_days: float,
    richardson_fit_end_days: float,
    bootstrap_replicates: int,
    bootstrap_confidence_level: float,
    bootstrap_random_seed: int,
) -> DispersionResult:
    """Calculate Poje Figure 4 absolute and adjusted radial pair dispersion."""
    required = {
        "array_id", "cluster_id", "source_cluster_id", "platform_id", "time_utc",
        "cluster_age_hours", "x_m", "y_m",
    }
    missing = required - set(platform_diagnostics.columns)
    if missing:
        raise ValueError(f"Platform diagnostics lack columns: {sorted(missing)}")
    if window_end_hours <= 0 or absolute_plot_end_days <= 0:
        raise ValueError("Dispersion windows must be positive")
    if initial_pair_separation_maximum_m <= 0:
        raise ValueError("Initial pair-separation maximum must be positive")
    if not 0 < ballistic_fit_end_hours <= window_end_hours:
        raise ValueError("Ballistic fit end must lie inside the analysis window")
    if not 0 < richardson_fit_start_days < richardson_fit_end_days:
        raise ValueError("Richardson fit interval must be positive and increasing")
    if richardson_fit_end_days * 24 > window_end_hours:
        raise ValueError("Richardson fit interval exceeds the analysis window")

    frame = platform_diagnostics.copy()
    frame["platform_id"] = frame.platform_id.astype(str)
    frame["source_cluster_id"] = frame.source_cluster_id.astype(str)
    frame["time_utc"] = pd.to_datetime(frame.time_utc, utc=True)
    frame = frame[
        (frame.cluster_age_hours >= 0)
        & (frame.cluster_age_hours < window_end_hours)
    ].copy()
    if frame.empty:
        raise ValueError(f"Array {array_id} has no positions in the dispersion window")
    if frame.duplicated(["platform_id", "time_utc"]).any():
        raise ValueError("Platform diagnostics contain duplicate platform-time rows")

    time_table = (
        frame[["time_utc", "cluster_age_hours"]]
        .drop_duplicates()
        .sort_values("time_utc")
    )
    if time_table.groupby("time_utc").cluster_age_hours.nunique().max() != 1:
        raise ValueError("A UTC timestamp maps to multiple array ages")
    time_table = time_table.drop_duplicates("time_utc")
    times = pd.DatetimeIndex(time_table.time_utc)
    elapsed_hours = time_table.cluster_age_hours.to_numpy(dtype=float)
    elapsed_days = elapsed_hours / 24
    anchor_matches = np.flatnonzero(np.isclose(elapsed_hours, 0, atol=1e-12))
    if len(anchor_matches) != 1:
        raise ValueError("Dispersion input must contain exactly one common anchor time")
    anchor_index = int(anchor_matches[0])

    platforms = sorted(frame.platform_id.unique().tolist())
    platform_lookup = {value: index for index, value in enumerate(platforms)}
    x_table = frame.pivot(index="platform_id", columns="time_utc", values="x_m")
    y_table = frame.pivot(index="platform_id", columns="time_utc", values="y_m")
    x = x_table.reindex(index=platforms, columns=times).to_numpy(dtype=float)
    y = y_table.reindex(index=platforms, columns=times).to_numpy(dtype=float)
    source_clusters = (
        frame.sort_values("time_utc").drop_duplicates("platform_id")
        .set_index("platform_id").source_cluster_id.reindex(platforms).astype(str)
    )
    x0 = x[:, anchor_index]
    y0 = y[:, anchor_index]
    anchor_finite = np.isfinite(x0) & np.isfinite(y0)
    if not anchor_finite.all():
        missing_platforms = np.asarray(platforms)[~anchor_finite].tolist()
        raise ValueError(f"Platforms lack finite positions at the common anchor: {missing_platforms}")

    displacement_squared = (x - x0[:, None]) ** 2 + (y - y0[:, None]) ** 2
    absolute_count = np.isfinite(displacement_squared).sum(axis=0)
    absolute_mean_squared = np.nanmean(displacement_squared, axis=0)
    absolute_rms = np.sqrt(absolute_mean_squared)

    first_indices: list[int] = []
    second_indices: list[int] = []
    initial_separations: list[float] = []
    for first, second in combinations(range(len(platforms)), 2):
        separation = math_hypot(x0[second] - x0[first], y0[second] - y0[first])
        if separation < initial_pair_separation_maximum_m:
            first_indices.append(first)
            second_indices.append(second)
            initial_separations.append(separation)
    if not first_indices:
        raise ValueError(
            f"Array {array_id} has no anchor pairs below "
            f"{initial_pair_separation_maximum_m:g} m"
        )
    first_index = np.asarray(first_indices, dtype=int)
    second_index = np.asarray(second_indices, dtype=int)
    initial_separation = np.asarray(initial_separations, dtype=float)
    separation = np.hypot(
        x[second_index] - x[first_index], y[second_index] - y[first_index],
    )
    radial_change = separation - initial_separation[:, None]
    radial_change_squared = radial_change ** 2
    relative_count = np.isfinite(radial_change_squared).sum(axis=0)
    relative_dispersion = np.nanmean(radial_change_squared, axis=0)
    ci_lower, bootstrap_median, ci_upper = _bootstrap_mean_intervals(
        radial_change_squared,
        replicates=bootstrap_replicates,
        confidence_level=bootstrap_confidence_level,
        random_seed=bootstrap_random_seed,
    )

    elapsed_seconds = elapsed_hours * 3600
    compensated = np.divide(
        relative_dispersion, elapsed_seconds ** 3,
        out=np.full(relative_dispersion.shape, np.nan), where=elapsed_seconds > 0,
    )
    compensated_lower = np.divide(
        ci_lower, elapsed_seconds ** 3,
        out=np.full(ci_lower.shape, np.nan), where=elapsed_seconds > 0,
    )
    compensated_upper = np.divide(
        ci_upper, elapsed_seconds ** 3,
        out=np.full(ci_upper.shape, np.nan), where=elapsed_seconds > 0,
    )

    ballistic_mask = (
        (elapsed_hours > 0) & (elapsed_hours <= ballistic_fit_end_hours)
        & np.isfinite(absolute_mean_squared) & (absolute_mean_squared > 0)
    )
    ballistic_values = absolute_mean_squared[ballistic_mask] / (
        elapsed_seconds[ballistic_mask] ** 2
    )
    ballistic_v0_squared = (
        float(np.median(ballistic_values)) if len(ballistic_values) else np.nan
    )
    richardson_mask = (
        (elapsed_days >= richardson_fit_start_days)
        & (elapsed_days <= richardson_fit_end_days)
        & np.isfinite(relative_dispersion) & (relative_dispersion > 0)
    )
    richardson_values = compensated[richardson_mask]
    richardson_coefficient = (
        float(np.median(richardson_values)) if len(richardson_values) else np.nan
    )
    if richardson_mask.sum() >= 2:
        free_exponent, log_intercept = np.polyfit(
            np.log(elapsed_seconds[richardson_mask]),
            np.log(relative_dispersion[richardson_mask]), 1,
        )
        free_coefficient = float(np.exp(log_intercept))
    else:
        free_exponent = free_coefficient = np.nan

    timeseries = pd.DataFrame({
        "array_id": array_id,
        "analysis_type": "poje_figure_04",
        "window_id": window_id,
        "time_utc": times,
        "elapsed_hours": elapsed_hours,
        "elapsed_days": elapsed_days,
        "absolute_platform_count": absolute_count,
        "absolute_mean_squared_displacement_m2": absolute_mean_squared,
        "absolute_rms_dispersion_m": absolute_rms,
        "absolute_rms_dispersion_km": absolute_rms / 1000,
        "relative_pair_count": relative_count,
        "relative_pair_coverage_fraction": relative_count / len(first_index),
        "adjusted_radial_relative_dispersion_m2": relative_dispersion,
        "adjusted_radial_relative_dispersion_km2": relative_dispersion / 1_000_000,
        "relative_bootstrap_median_m2": bootstrap_median,
        "relative_ci_lower_m2": ci_lower,
        "relative_ci_upper_m2": ci_upper,
        "richardson_compensated_m2_s3": compensated,
        "richardson_compensated_ci_lower_m2_s3": compensated_lower,
        "richardson_compensated_ci_upper_m2_s3": compensated_upper,
        "absolute_panel_in_range": elapsed_days <= absolute_plot_end_days,
        "ballistic_fit_in_range": ballistic_mask,
        "richardson_fit_in_range": richardson_mask,
    })

    pair_rows: list[dict[str, object]] = []
    pair_ids: list[str] = []
    for pair_position, (first, second, initial) in enumerate(zip(
        first_index, second_index, initial_separation,
    )):
        first_id = platforms[int(first)]
        second_id = platforms[int(second)]
        pair_id = stable_pair_id(first_id, second_id)
        pair_ids.append(pair_id)
        pair_rows.append({
            "array_id": array_id,
            "analysis_type": "poje_figure_04",
            "window_id": window_id,
            "pair_id": pair_id,
            "platform_id_1": first_id,
            "platform_id_2": second_id,
            "source_cluster_id_1": source_clusters.loc[first_id],
            "source_cluster_id_2": source_clusters.loc[second_id],
            "pair_cluster_relation": (
                "within_cluster" if source_clusters.loc[first_id] == source_clusters.loc[second_id]
                else "between_clusters"
            ),
            "initial_separation_m": initial,
            "initial_separation_maximum_m": initial_pair_separation_maximum_m,
            "available_time_count": int(np.isfinite(radial_change_squared[pair_position]).sum()),
        })
    pair_inventory = pd.DataFrame(pair_rows)

    pair_index, time_index = np.nonzero(np.isfinite(radial_change_squared))
    pair_observations = pd.DataFrame({
        "array_id": array_id,
        "analysis_type": "poje_figure_04",
        "window_id": window_id,
        "pair_id": np.asarray(pair_ids, dtype=object)[pair_index],
        "platform_id_1": np.asarray(platforms, dtype=object)[first_index[pair_index]],
        "platform_id_2": np.asarray(platforms, dtype=object)[second_index[pair_index]],
        "time_utc": times.to_numpy()[time_index],
        "elapsed_days": elapsed_days[time_index],
        "initial_separation_m": initial_separation[pair_index],
        "separation_m": separation[pair_index, time_index],
        "radial_separation_change_m": radial_change[pair_index, time_index],
        "radial_separation_change_squared_m2": radial_change_squared[pair_index, time_index],
    })

    summary = {
        "array_id": array_id,
        "analysis_type": "poje_figure_04",
        "window_id": window_id,
        "anchor_time_utc": times[anchor_index].isoformat(),
        "assigned_platform_count": len(platforms),
        "initial_pair_count": len(first_index),
        "between_cluster_initial_pair_count": int(
            (pair_inventory.pair_cluster_relation == "between_clusters").sum()
        ),
        "initial_separation_maximum_m": initial_pair_separation_maximum_m,
        "window_end_hours": window_end_hours,
        "absolute_plot_end_days": absolute_plot_end_days,
        "ballistic_fit_end_hours": ballistic_fit_end_hours,
        "ballistic_v0_squared_m2_s2": ballistic_v0_squared,
        "richardson_fit_start_days": richardson_fit_start_days,
        "richardson_fit_end_days": richardson_fit_end_days,
        "richardson_coefficient_m2_s3": richardson_coefficient,
        "relative_dispersion_free_fit_exponent": float(free_exponent),
        "relative_dispersion_free_fit_coefficient": free_coefficient,
        "bootstrap_unit": "fixed initial pair identity synchronized across times",
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_confidence_level": bootstrap_confidence_level,
    }
    return DispersionResult(timeseries, pair_inventory, pair_observations, summary)


def calculate_figure5_self_similarity(
    platform_diagnostics: pd.DataFrame,
    *,
    array_id: int,
    window_id: str,
    window_end_hours: float,
    memory_groups: tuple[SeparationMemoryGroup, ...],
    similarity_bin_count: int,
    kolmogorov_constant: float,
) -> Figure5Result:
    """Calculate Poje Figure 5 separation memory and Lagrangian velocity statistics."""
    required = {
        "array_id", "cluster_id", "platform_id", "time_utc", "cluster_age_hours",
        "x_m", "y_m", "analysis_valid", "analysis_u_m_s", "analysis_v_m_s",
    }
    missing = required - set(platform_diagnostics.columns)
    if missing:
        raise ValueError(f"Platform diagnostics lack Figure 5 columns: {sorted(missing)}")
    if similarity_bin_count < 2:
        raise ValueError("Similarity-bin count must be at least two")
    if not np.isfinite(kolmogorov_constant) or kolmogorov_constant <= 0:
        raise ValueError("Kolmogorov constant must be finite and positive")
    if not memory_groups:
        raise ValueError("At least one separation-memory time group is required")

    frame = platform_diagnostics.copy()
    frame["platform_id"] = frame.platform_id.astype(str)
    frame["time_utc"] = pd.to_datetime(frame.time_utc, utc=True)
    frame = frame[
        (frame.cluster_age_hours >= 0)
        & (frame.cluster_age_hours < window_end_hours)
    ].copy()
    if frame.empty:
        raise ValueError(f"Array {array_id} has no Figure 5 observations")
    if frame.duplicated(["platform_id", "time_utc"]).any():
        raise ValueError("Platform diagnostics contain duplicate platform-time rows")
    time_table = (
        frame[["time_utc", "cluster_age_hours"]]
        .drop_duplicates().sort_values("time_utc")
    )
    if time_table.groupby("time_utc").cluster_age_hours.nunique().max() != 1:
        raise ValueError("A UTC timestamp maps to multiple array ages")
    time_table = time_table.drop_duplicates("time_utc")
    times = pd.DatetimeIndex(time_table.time_utc)
    elapsed_hours = time_table.cluster_age_hours.to_numpy(dtype=float)
    elapsed_days = elapsed_hours / 24
    anchor_matches = np.flatnonzero(np.isclose(elapsed_hours, 0, atol=1e-12))
    if len(anchor_matches) != 1 or int(anchor_matches[0]) != 0:
        raise ValueError("Figure 5 requires the common anchor as the first selected time")
    platforms = sorted(frame.platform_id.unique().tolist())
    if len(platforms) < 2:
        raise ValueError("Figure 5 separation memory requires at least two platforms")

    def matrix(column: str) -> np.ndarray:
        return (
            frame.pivot(index="platform_id", columns="time_utc", values=column)
            .reindex(index=platforms, columns=times).to_numpy(dtype=float)
        )

    x = matrix("x_m")
    y = matrix("y_m")
    pair_members = np.asarray(list(combinations(range(len(platforms)), 2)), dtype=int)
    first_index = pair_members[:, 0]
    second_index = pair_members[:, 1]
    separation = np.hypot(
        x[second_index] - x[first_index], y[second_index] - y[first_index],
    )

    similarity_edges = np.linspace(-1, 0, similarity_bin_count + 1)
    similarity_centers = (similarity_edges[:-1] + similarity_edges[1:]) / 2
    sample_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for group in memory_groups:
        if not 0 < group.start_day < group.end_day <= window_end_hours / 24:
            raise ValueError(f"Invalid Figure 5 memory group: {group.identifier}")
        target_indices = np.flatnonzero(
            (elapsed_days >= group.start_day) & (elapsed_days < group.end_day)
        )
        if not len(target_indices):
            raise ValueError(f"Figure 5 memory group has no samples: {group.identifier}")
        group_samples: list[dict[str, object]] = []
        for target_index in target_indices:
            current = separation[:, target_index]
            current_finite = np.isfinite(current)
            past = separation[:, :target_index + 1]
            joint = current_finite[:, None] & np.isfinite(past)
            product_sum = np.sum(
                np.where(joint, current[:, None] * past, 0.0), axis=0,
            )
            current_square_sum = np.sum(
                np.where(joint, current[:, None] ** 2, 0.0), axis=0,
            )
            pair_count = joint.sum(axis=0)
            normalized = np.divide(
                product_sum, current_square_sum,
                out=np.full(product_sum.shape, np.nan), where=current_square_sum > 0,
            )
            past_indices = np.arange(target_index + 1)
            lag_days = elapsed_days[past_indices] - elapsed_days[target_index]
            normalized_lag = lag_days / elapsed_days[target_index]
            for past_index, similarity, lag, correlation, count in zip(
                past_indices, normalized_lag, lag_days, normalized, pair_count,
            ):
                if not np.isfinite(correlation):
                    continue
                group_samples.append({
                    "array_id": array_id,
                    "analysis_type": "poje_figure_05_separation_memory",
                    "window_id": window_id,
                    "memory_group_id": group.identifier,
                    "memory_group_start_day": group.start_day,
                    "memory_group_end_day": group.end_day,
                    "target_time_utc": times[target_index],
                    "target_elapsed_days": elapsed_days[target_index],
                    "lagged_time_utc": times[past_index],
                    "lag_days": lag,
                    "normalized_lag_tau_over_t": similarity,
                    "normalized_separation_correlation": correlation,
                    "contributing_pair_count": int(count),
                })
        group_frame = pd.DataFrame(group_samples)
        bin_index = np.searchsorted(
            similarity_edges, group_frame.normalized_lag_tau_over_t, side="right",
        ) - 1
        bin_index[np.isclose(group_frame.normalized_lag_tau_over_t, 0)] = (
            similarity_bin_count - 1
        )
        group_frame["similarity_bin_index"] = bin_index
        sample_rows.extend(group_frame.to_dict("records"))
        for index, center in enumerate(similarity_centers):
            selected = group_frame[group_frame.similarity_bin_index == index]
            aggregate_rows.append({
                "array_id": array_id,
                "analysis_type": "poje_figure_05_separation_memory",
                "window_id": window_id,
                "memory_group_id": group.identifier,
                "memory_group_start_day": group.start_day,
                "memory_group_end_day": group.end_day,
                "similarity_bin_index": index,
                "similarity_bin_lower": similarity_edges[index],
                "similarity_bin_upper": similarity_edges[index + 1],
                "similarity_bin_center": center,
                "correlation_sample_count": len(selected),
                "target_time_count": int(selected.target_time_utc.nunique()) if len(selected) else 0,
                "mean_contributing_pair_count": (
                    float(selected.contributing_pair_count.mean()) if len(selected) else np.nan
                ),
                "mean_normalized_separation_correlation": (
                    float(selected.normalized_separation_correlation.mean())
                    if len(selected) else np.nan
                ),
                "standard_deviation_normalized_separation_correlation": (
                    float(selected.normalized_separation_correlation.std(ddof=1))
                    if len(selected) > 1 else np.nan
                ),
            })

    analysis_valid = matrix("analysis_valid") > 0.5
    u = matrix("analysis_u_m_s")
    v = matrix("analysis_v_m_s")
    valid = analysis_valid & np.isfinite(u) & np.isfinite(v)
    anchor_valid = valid[:, 0]
    delta_squared = (u - u[:, [0]]) ** 2 + (v - v[:, [0]]) ** 2
    delta_squared[~(valid & anchor_valid[:, None])] = np.nan
    velocity_count = np.isfinite(delta_squared).sum(axis=0)
    lagrangian_second = np.divide(
        np.nansum(delta_squared, axis=0), velocity_count,
        out=np.full(velocity_count.shape, np.nan, dtype=float),
        where=velocity_count > 0,
    )
    elapsed_seconds = elapsed_hours * 3600
    epsilon_proxy = np.divide(
        lagrangian_second, kolmogorov_constant * elapsed_seconds,
        out=np.full(lagrangian_second.shape, np.nan), where=elapsed_seconds > 0,
    )
    lagrangian = pd.DataFrame({
        "array_id": array_id,
        "analysis_type": "poje_figure_05_lagrangian_structure",
        "window_id": window_id,
        "time_utc": times,
        "elapsed_hours": elapsed_hours,
        "elapsed_days": elapsed_days,
        "contributing_platform_count": velocity_count,
        "lagrangian_velocity_structure_second_m2_s2": lagrangian_second,
        "kolmogorov_constant": kolmogorov_constant,
        "epsilon_proxy_m2_s3": epsilon_proxy,
    })
    finite_proxy = np.flatnonzero(np.isfinite(epsilon_proxy))
    peak_index = (
        int(finite_proxy[np.argmax(epsilon_proxy[finite_proxy])])
        if len(finite_proxy) else None
    )
    summary = {
        "figure5_complete_pair_count": len(pair_members),
        "figure5_similarity_bin_count": similarity_bin_count,
        "figure5_kolmogorov_constant": kolmogorov_constant,
        "figure5_peak_epsilon_proxy_m2_s3": (
            float(epsilon_proxy[peak_index]) if peak_index is not None else np.nan
        ),
        "figure5_peak_epsilon_proxy_elapsed_days": (
            float(elapsed_days[peak_index]) if peak_index is not None else np.nan
        ),
        "figure5_minimum_velocity_platform_count": int(velocity_count.min()),
        "figure5_memory_group_count": len(memory_groups),
    }
    return Figure5Result(
        separation_memory=pd.DataFrame(aggregate_rows),
        separation_memory_samples=pd.DataFrame(sample_rows),
        lagrangian_structure=lagrangian,
        summary=summary,
    )


def calculate_figure6_increment_distributions(
    pair_observations: pd.DataFrame,
    *,
    array_id: int,
    window_id: str,
    window_end_hours: float,
    source_edges_m: np.ndarray,
    source_centers_m: np.ndarray,
    scales: tuple[IncrementDistributionScale, ...],
    maximum_center_relative_difference: float,
    normalized_increment_limit: float,
    histogram_bin_count: int,
    gaussian_core_maximum_absolute_normalized_increment: float,
) -> Figure6Result:
    """Calculate Poje Figure 6 longitudinal-increment distributions."""
    required = {
        "array_id", "pair_id", "platform_id_1", "platform_id_2", "time_utc",
        "cluster_age_hours", "separation_m", "delta_u_l_m_s",
    }
    missing = required - set(pair_observations.columns)
    if missing:
        raise ValueError(f"Pair observations lack Figure 6 columns: {sorted(missing)}")
    edges = np.asarray(source_edges_m, dtype=float)
    centers = np.asarray(source_centers_m, dtype=float)
    if edges.ndim != 1 or centers.shape != (len(edges) - 1,):
        raise ValueError("Figure 6 source separation bins are inconsistent")
    if not np.isfinite(edges).all() or not np.isfinite(centers).all():
        raise ValueError("Figure 6 source separation bins must be finite")
    if np.any(np.diff(edges) <= 0) or np.any(np.diff(centers) <= 0):
        raise ValueError("Figure 6 source separation bins must increase")
    if not scales:
        raise ValueError("Figure 6 requires at least one separation scale")
    if not 0 < maximum_center_relative_difference < 1:
        raise ValueError("Figure 6 center tolerance must lie between zero and one")
    if not np.isfinite(normalized_increment_limit) or normalized_increment_limit <= 0:
        raise ValueError("Figure 6 normalized-increment limit must be positive")
    if histogram_bin_count < 10:
        raise ValueError("Figure 6 histogram-bin count must be at least ten")
    if not (
        0 < gaussian_core_maximum_absolute_normalized_increment
        < normalized_increment_limit
    ):
        raise ValueError("Figure 6 Gaussian core limit must lie inside the histogram range")

    frame = pair_observations.copy()
    frame["time_utc"] = pd.to_datetime(frame.time_utc, utc=True)
    frame = frame[
        (frame.cluster_age_hours >= 0)
        & (frame.cluster_age_hours < window_end_hours)
        & np.isfinite(frame.separation_m)
        & (frame.separation_m > 0)
        & np.isfinite(frame.delta_u_l_m_s)
    ].copy()
    if frame.empty:
        raise ValueError(f"Array {array_id} has no valid Figure 6 observations")
    frame["source_separation_bin_index"] = bin_indices(
        frame.separation_m.to_numpy(dtype=float), edges,
    )

    histogram_edges = np.linspace(
        -normalized_increment_limit,
        normalized_increment_limit,
        histogram_bin_count + 1,
    )
    histogram_centers = (histogram_edges[:-1] + histogram_edges[1:]) / 2
    histogram_widths = np.diff(histogram_edges)
    distribution_rows: list[dict[str, object]] = []
    statistic_rows: list[dict[str, object]] = []
    observation_parts: list[pd.DataFrame] = []
    selected_source_indices: set[int] = set()

    for scale in scales:
        source_index = int(np.argmin(np.abs(centers - scale.target_center_m)))
        relative_difference = abs(centers[source_index] - scale.target_center_m) / (
            scale.target_center_m
        )
        if relative_difference > maximum_center_relative_difference:
            raise ValueError(
                f"No source separation bin is close enough to Figure 6 scale "
                f"{scale.identifier}: target {scale.target_center_m:g} m, nearest "
                f"{centers[source_index]:g} m"
            )
        if source_index in selected_source_indices:
            raise ValueError("Figure 6 target centers resolve to duplicate source bins")
        selected_source_indices.add(source_index)
        selected = frame[frame.source_separation_bin_index == source_index].copy()
        values = selected.delta_u_l_m_s.to_numpy(dtype=float)
        if len(values) < 2:
            raise ValueError(f"Figure 6 scale has fewer than two samples: {scale.identifier}")
        mean = float(np.mean(values))
        centered = values - mean
        variance = float(np.mean(centered ** 2))
        standard_deviation = math.sqrt(variance)
        if not np.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError(f"Figure 6 scale has zero velocity variance: {scale.identifier}")
        normalized = values / standard_deviation
        normalized_mean = mean / standard_deviation
        normalized_centered = normalized - normalized_mean
        raw_second = float(np.mean(values ** 2))
        raw_third = float(np.mean(values ** 3))
        raw_fourth = float(np.mean(values ** 4))
        skewness = float(np.mean(normalized_centered ** 3))
        flatness = float(np.mean(normalized_centered ** 4))
        counts, _ = np.histogram(normalized, bins=histogram_edges)
        probability_mass = counts / len(values)
        density = probability_mass / histogram_widths
        unit_gaussian_density = np.exp(
            -0.5 * (histogram_centers - normalized_mean) ** 2
        ) / (
            math.sqrt(2 * math.pi)
        )
        core_mask = (
            np.abs(histogram_centers)
            <= gaussian_core_maximum_absolute_normalized_increment
        )
        core_x = histogram_centers[core_mask]
        core_density = density[core_mask]

        def gaussian_curve(
            values: np.ndarray, amplitude: float, location: float, width: float,
        ) -> np.ndarray:
            return amplitude * np.exp(-0.5 * ((values - location) / width) ** 2)

        fit_status = "success"
        try:
            fitted_parameters, _ = curve_fit(
                gaussian_curve,
                core_x,
                core_density,
                p0=(float(np.max(core_density)), normalized_mean, 0.75),
                bounds=(
                    (0.0, -gaussian_core_maximum_absolute_normalized_increment, 0.01),
                    (
                        np.inf,
                        gaussian_core_maximum_absolute_normalized_increment,
                        normalized_increment_limit,
                    ),
                ),
                maxfev=20000,
            )
            fit_amplitude, fit_location, fit_width = map(float, fitted_parameters)
            fitted_gaussian_density = gaussian_curve(
                histogram_centers, fit_amplitude, fit_location, fit_width,
            )
            fit_rmse = float(np.sqrt(np.mean(
                (core_density - gaussian_curve(
                    core_x, fit_amplitude, fit_location, fit_width,
                )) ** 2
            )))
            fit_area = float(fit_amplitude * math.sqrt(2 * math.pi) * fit_width)
        except (RuntimeError, ValueError, FloatingPointError):
            fit_status = "fit_failed"
            fit_amplitude = fit_location = fit_width = fit_rmse = fit_area = np.nan
            fitted_gaussian_density = np.full(histogram_centers.shape, np.nan)
        underflow_count = int(np.sum(normalized < histogram_edges[0]))
        overflow_count = int(np.sum(normalized > histogram_edges[-1]))
        scale_values = {
            "array_id": array_id,
            "analysis_type": "poje_figure_06",
            "window_id": window_id,
            "figure6_scale_id": scale.identifier,
            "panel_group": scale.panel_group,
            "target_separation_center_m": scale.target_center_m,
            "resolved_source_bin_index": source_index,
            "separation_bin_lower_m": float(edges[source_index]),
            "separation_bin_upper_m": float(edges[source_index + 1]),
            "separation_bin_nominal_center_m": float(centers[source_index]),
            "center_relative_difference": float(relative_difference),
        }
        for histogram_index, (
            lower, upper, center, count, mass, value_density, unit_gaussian,
            fitted_gaussian,
        ) in enumerate(
            zip(
                histogram_edges[:-1], histogram_edges[1:], histogram_centers,
                counts, probability_mass, density, unit_gaussian_density,
                fitted_gaussian_density,
            )
        ):
            distribution_rows.append({
                **scale_values,
                "normalized_increment_bin_index": histogram_index,
                "normalized_increment_bin_lower": lower,
                "normalized_increment_bin_upper": upper,
                "normalized_increment_bin_center": center,
                "observation_count": int(count),
                "probability_mass": float(mass),
                "probability_density": float(value_density),
                "unit_variance_gaussian_density": float(unit_gaussian),
                "fitted_core_gaussian_density": float(fitted_gaussian),
            })
        platforms = set(selected.platform_id_1.astype(str)) | set(
            selected.platform_id_2.astype(str)
        )
        statistic_rows.append({
            **scale_values,
            "mean_observed_separation_m": float(selected.separation_m.mean()),
            "observation_count": len(selected),
            "unique_pair_count": int(selected.pair_id.astype(str).nunique()),
            "platform_count": len(platforms),
            "unique_utc_count": int(selected.time_utc.nunique()),
            "mean_delta_u_l_m_s": mean,
            "population_standard_deviation_delta_u_l_m_s": standard_deviation,
            "normalized_mean_delta_u_l": normalized_mean,
            "raw_second_moment_delta_u_l_m2_s2": raw_second,
            "raw_third_moment_delta_u_l_m3_s3": raw_third,
            "raw_fourth_moment_delta_u_l_m4_s4": raw_fourth,
            "raw_third_moment_ratio": (
                raw_third / raw_second ** 1.5 if raw_second > 0 else np.nan
            ),
            "raw_fourth_moment_ratio": (
                raw_fourth / raw_second ** 2 if raw_second > 0 else np.nan
            ),
            "central_skewness": skewness,
            "central_flatness": flatness,
            "histogram_underflow_count": underflow_count,
            "histogram_overflow_count": overflow_count,
            "histogram_retained_fraction": float(np.sum(counts) / len(values)),
            "gaussian_core_maximum_absolute_normalized_increment": (
                gaussian_core_maximum_absolute_normalized_increment
            ),
            "gaussian_core_fit_bin_count": int(core_mask.sum()),
            "gaussian_core_fit_status": fit_status,
            "gaussian_core_fit_amplitude": fit_amplitude,
            "gaussian_core_fit_location": fit_location,
            "gaussian_core_fit_width": fit_width,
            "gaussian_core_fit_area": fit_area,
            "gaussian_core_fit_rmse": fit_rmse,
        })
        selected["analysis_type"] = "poje_figure_06"
        selected["window_id"] = window_id
        selected["figure6_scale_id"] = scale.identifier
        selected["panel_group"] = scale.panel_group
        selected["normalized_delta_u_l_by_sigma"] = normalized
        observation_parts.append(selected[[
            "array_id", "analysis_type", "window_id", "figure6_scale_id",
            "panel_group", "pair_id", "platform_id_1", "platform_id_2",
            "time_utc", "cluster_age_hours", "separation_m", "delta_u_l_m_s",
            "normalized_delta_u_l_by_sigma", "source_separation_bin_index",
        ]])

    statistics = pd.DataFrame(statistic_rows)
    summary = {
        "figure6_scale_count": len(statistics),
        "figure6_minimum_observation_count": int(statistics.observation_count.min()),
        "figure6_minimum_unique_pair_count": int(statistics.unique_pair_count.min()),
        "figure6_minimum_platform_count": int(statistics.platform_count.min()),
        "figure6_minimum_unique_utc_count": int(statistics.unique_utc_count.min()),
        "figure6_minimum_skewness": float(statistics.central_skewness.min()),
        "figure6_maximum_skewness": float(statistics.central_skewness.max()),
        "figure6_minimum_raw_third_moment_ratio": float(
            statistics.raw_third_moment_ratio.min()
        ),
        "figure6_maximum_raw_third_moment_ratio": float(
            statistics.raw_third_moment_ratio.max()
        ),
        "figure6_minimum_histogram_retained_fraction": float(
            statistics.histogram_retained_fraction.min()
        ),
        "figure6_gaussian_fit_failure_count": int(
            (statistics.gaussian_core_fit_status != "success").sum()
        ),
        "figure6_minimum_fitted_gaussian_width": float(
            statistics.gaussian_core_fit_width.min()
        ),
        "figure6_maximum_fitted_gaussian_width": float(
            statistics.gaussian_core_fit_width.max()
        ),
    }
    return Figure6Result(
        distributions=pd.DataFrame(distribution_rows),
        statistics=statistics,
        observations=pd.concat(observation_parts, ignore_index=True),
        summary=summary,
    )


def calculate_figure7_structure_functions(
    pair_observations: pd.DataFrame,
    epsilon_by_scale: pd.DataFrame,
    *,
    array_id: int,
    window_id: str,
    window_end_hours: float,
    source_edges_m: np.ndarray,
    source_centers_m: np.ndarray,
    fit_minimum_separation_m: float,
    fit_maximum_separation_m: float,
    panel_a_maximum_separation_m: float,
    panel_b_maximum_separation_m: float,
) -> Figure7Result:
    """Calculate Poje Figure 7 second- and third-order structure statistics."""
    observation_required = {
        "pair_id", "platform_id_1", "platform_id_2", "time_utc",
        "cluster_age_hours", "separation_m", "delta_u_l_m_s",
        "delta_u_l_delta_u_t_squared_m3_s3",
    }
    observation_missing = observation_required - set(pair_observations.columns)
    if observation_missing:
        raise ValueError(
            f"Pair observations lack Figure 7 columns: {sorted(observation_missing)}"
        )
    summary_required = {
        "analysis_type", "window_id", "separation_bin_index",
        "separation_bin_lower_m", "separation_bin_upper_m",
        "separation_bin_nominal_center_m", "mean_separation_m",
        "raw_delta_u_l_m_s_second", "raw_delta_u_l_m_s_third",
        "epsilon_eff_m2_s3", "observation_count", "unique_pair_count",
        "platform_count", "unique_utc_count", "support_ok", "support_status",
        "mean_bin_coriolis_s_inverse", "poje_longitudinal_rossby_number",
        "epsilon_ci_lower_m2_s3",
        "epsilon_ci_upper_m2_s3", "confidence_interval_status",
    }
    summary_missing = summary_required - set(epsilon_by_scale.columns)
    if summary_missing:
        raise ValueError(
            f"Epsilon summary lacks Figure 7 columns: {sorted(summary_missing)}"
        )
    edges = np.asarray(source_edges_m, dtype=float)
    centers = np.asarray(source_centers_m, dtype=float)
    if centers.shape != (len(edges) - 1,) or np.any(np.diff(edges) <= 0):
        raise ValueError("Figure 7 source separation bins are inconsistent")
    if not 0 < fit_minimum_separation_m < fit_maximum_separation_m:
        raise ValueError("Figure 7 fit interval must be positive and increasing")
    if panel_a_maximum_separation_m < fit_maximum_separation_m:
        raise ValueError("Figure 7 panel A must include the complete fit interval")
    if panel_b_maximum_separation_m <= fit_minimum_separation_m:
        raise ValueError("Figure 7 panel B maximum separation is too small")
    source_summary = epsilon_by_scale[
        (epsilon_by_scale.analysis_type == "window")
        & (epsilon_by_scale.window_id.astype(str) == window_id)
    ].copy()
    if len(source_summary) != len(centers):
        raise ValueError("Figure 7 requires exactly one window summary row per source bin")
    source_summary = source_summary.sort_values("separation_bin_index")
    if not np.array_equal(
        source_summary.separation_bin_index.to_numpy(dtype=int),
        np.arange(len(centers)),
    ):
        raise ValueError("Figure 7 source summary bin indices are incomplete")
    if not np.allclose(
        source_summary.separation_bin_nominal_center_m.to_numpy(dtype=float), centers,
        rtol=1e-12, atol=1e-12,
    ):
        raise ValueError("Figure 7 source summary centers differ from resolved config")

    observations = pair_observations.copy()
    observations["time_utc"] = pd.to_datetime(observations.time_utc, utc=True)
    observations = observations[
        (observations.cluster_age_hours >= 0)
        & (observations.cluster_age_hours < window_end_hours)
        & np.isfinite(observations.separation_m)
        & (observations.separation_m > 0)
        & np.isfinite(observations.delta_u_l_m_s)
        & np.isfinite(observations.delta_u_l_delta_u_t_squared_m3_s3)
    ].copy()
    observations["source_separation_bin_index"] = bin_indices(
        observations.separation_m.to_numpy(dtype=float), edges,
    )

    rows: list[dict[str, object]] = []
    for source_row in source_summary.itertuples(index=False):
        index = int(source_row.separation_bin_index)
        selected = observations[observations.source_separation_bin_index == index]
        mean_separation = float(source_row.mean_separation_m)
        s_ll = float(source_row.raw_delta_u_l_m_s_second)
        s_lll = float(source_row.raw_delta_u_l_m_s_third)
        mixed = (
            float(selected.delta_u_l_delta_u_t_squared_m3_s3.mean())
            if len(selected) else np.nan
        )
        direct_recomputed = (
            -1.25 * s_lll / mean_separation
            if np.isfinite(s_lll) and np.isfinite(mean_separation) and mean_separation > 0
            else np.nan
        )
        mixed_epsilon = (
            -(s_lll + mixed) / (2 * mean_separation)
            if np.isfinite(s_lll) and np.isfinite(mixed)
            and np.isfinite(mean_separation) and mean_separation > 0
            else np.nan
        )
        bin_coriolis = float(source_row.mean_bin_coriolis_s_inverse)
        latitude_rossby_recomputed = (
            math.sqrt(s_ll) / (abs(bin_coriolis) * mean_separation)
            if np.isfinite(s_ll) and s_ll >= 0
            and np.isfinite(bin_coriolis) and bin_coriolis != 0
            and np.isfinite(mean_separation) and mean_separation > 0
            else np.nan
        )
        latitude_rossby = float(source_row.poje_longitudinal_rossby_number)
        rows.append({
            "array_id": array_id,
            "analysis_type": "poje_figure_07",
            "window_id": window_id,
            "separation_bin_index": index,
            "separation_bin_lower_m": float(source_row.separation_bin_lower_m),
            "separation_bin_upper_m": float(source_row.separation_bin_upper_m),
            "separation_bin_nominal_center_m": float(
                source_row.separation_bin_nominal_center_m
            ),
            "mean_separation_m": mean_separation,
            "longitudinal_second_order_m2_s2": s_ll,
            "longitudinal_third_order_m3_s3": s_lll,
            "mixed_longitudinal_transverse_third_order_m3_s3": mixed,
            "epsilon_four_fifths_m2_s3": float(source_row.epsilon_eff_m2_s3),
            "epsilon_four_fifths_recomputed_m2_s3": direct_recomputed,
            "epsilon_four_fifths_identity_absolute_error_m2_s3": (
                abs(float(source_row.epsilon_eff_m2_s3) - direct_recomputed)
                if np.isfinite(float(source_row.epsilon_eff_m2_s3))
                and np.isfinite(direct_recomputed) else np.nan
            ),
            "epsilon_mixed_relation_m2_s3": mixed_epsilon,
            "bin_mean_coriolis_s_inverse": bin_coriolis,
            "latitude_dependent_rossby_number": latitude_rossby,
            "latitude_dependent_rossby_number_recomputed": (
                latitude_rossby_recomputed
            ),
            "latitude_dependent_rossby_identity_absolute_error": (
                abs(latitude_rossby - latitude_rossby_recomputed)
                if np.isfinite(latitude_rossby)
                and np.isfinite(latitude_rossby_recomputed) else np.nan
            ),
            "observation_count": int(source_row.observation_count),
            "unique_pair_count": int(source_row.unique_pair_count),
            "platform_count": int(source_row.platform_count),
            "unique_utc_count": int(source_row.unique_utc_count),
            "support_ok": bool(source_row.support_ok),
            "support_status": str(source_row.support_status),
            "epsilon_ci_lower_m2_s3": float(source_row.epsilon_ci_lower_m2_s3),
            "epsilon_ci_upper_m2_s3": float(source_row.epsilon_ci_upper_m2_s3),
            "confidence_interval_status": str(source_row.confidence_interval_status),
            "panel_a_in_range": mean_separation <= panel_a_maximum_separation_m,
            "panel_b_in_range": mean_separation <= panel_b_maximum_separation_m,
            "scaling_fit_in_range": (
                fit_minimum_separation_m <= mean_separation <= fit_maximum_separation_m
            ),
        })
    result = pd.DataFrame(rows)
    fit_mask = (
        result.scaling_fit_in_range.astype(bool)
        & result.support_ok.astype(bool)
        & np.isfinite(result.mean_separation_m)
        & np.isfinite(result.longitudinal_second_order_m2_s2)
        & (result.longitudinal_second_order_m2_s2 > 0)
    )
    fit = result[fit_mask]
    if len(fit) < 3:
        raise ValueError("Figure 7 has fewer than three supported scaling-fit bins")
    log_r = np.log(fit.mean_separation_m.to_numpy(dtype=float))
    log_s = np.log(fit.longitudinal_second_order_m2_s2.to_numpy(dtype=float))
    coefficients, covariance = np.polyfit(log_r, log_s, 1, cov=True)
    exponent = float(coefficients[0])
    free_coefficient = float(np.exp(coefficients[1]))
    exponent_standard_error = float(np.sqrt(covariance[0, 0]))
    k41_exponent = 2 / 3
    k41_coefficient = float(np.exp(np.mean(log_s - k41_exponent * log_r)))
    result["free_fit_second_order_m2_s2"] = (
        free_coefficient * result.mean_separation_m ** exponent
    )
    result["k41_fixed_fit_second_order_m2_s2"] = (
        k41_coefficient * result.mean_separation_m ** k41_exponent
    )
    panel_b = result[result.panel_b_in_range.astype(bool)]
    direct = panel_b.epsilon_four_fifths_m2_s3.to_numpy(dtype=float)
    mixed_values = panel_b.epsilon_mixed_relation_m2_s3.to_numpy(dtype=float)
    identity_errors = result.epsilon_four_fifths_identity_absolute_error_m2_s3.to_numpy(
        dtype=float
    )
    summary = {
        "figure7_fit_minimum_separation_m": fit_minimum_separation_m,
        "figure7_fit_maximum_separation_m": fit_maximum_separation_m,
        "figure7_scaling_fit_bin_count": len(fit),
        "figure7_second_order_free_fit_exponent": exponent,
        "figure7_second_order_free_fit_exponent_standard_error": exponent_standard_error,
        "figure7_second_order_free_fit_coefficient": free_coefficient,
        "figure7_k41_fixed_exponent": k41_exponent,
        "figure7_k41_fixed_fit_coefficient": k41_coefficient,
        "figure7_minimum_bin_mean_coriolis_s_inverse": float(
            result.bin_mean_coriolis_s_inverse.min()
        ),
        "figure7_maximum_bin_mean_coriolis_s_inverse": float(
            result.bin_mean_coriolis_s_inverse.max()
        ),
        "figure7_maximum_latitude_dependent_rossby_identity_error": float(
            result.latitude_dependent_rossby_identity_absolute_error.max()
        ),
        "figure7_maximum_four_fifths_identity_error_m2_s3": float(
            np.nanmax(identity_errors)
        ),
        "figure7_panel_b_positive_four_fifths_bin_count": int(np.sum(direct > 0)),
        "figure7_panel_b_negative_four_fifths_bin_count": int(np.sum(direct < 0)),
        "figure7_panel_b_positive_mixed_bin_count": int(np.sum(mixed_values > 0)),
        "figure7_panel_b_negative_mixed_bin_count": int(np.sum(mixed_values < 0)),
    }
    return Figure7Result(structure_functions=result, summary=summary)


def math_hypot(x: float, y: float) -> float:
    """Return a scalar Euclidean norm without leaking NumPy scalar types."""
    return float(np.hypot(x, y))
