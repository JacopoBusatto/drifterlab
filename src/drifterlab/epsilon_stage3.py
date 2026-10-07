"""Dependence-aware uncertainty, histograms, and influence for epsilon products."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
import pandas as pd

from drifterlab.epsilon import FOUR_FIFTHS_COEFFICIENT, bin_indices, epsilon_from_samples


@dataclass(frozen=True)
class BootstrapSpec:
    """Configuration for synchronized cluster-window UTC block resampling."""

    confidence_level: float
    replicates: int
    random_seed: int
    primary_block_minutes: int
    sensitivity_block_minutes: tuple[int, ...]
    minimum_contributing_blocks: int
    minimum_successful_replicate_fraction: float


def symmetric_log_edges(
    maximum_absolute: float,
    linear_threshold: float,
    bins_per_decade: int,
) -> np.ndarray:
    """Return explicit symmetric physical edges with one central linear bin."""
    if not math.isfinite(maximum_absolute) or maximum_absolute <= 0:
        raise ValueError("Histogram maximum absolute q must be finite and positive")
    if (
        not math.isfinite(linear_threshold)
        or linear_threshold <= 0
        or linear_threshold >= maximum_absolute
    ):
        raise ValueError("Histogram linear threshold must lie between zero and its maximum")
    if isinstance(bins_per_decade, bool) or bins_per_decade < 1:
        raise ValueError("Histogram bins_per_decade must be a positive integer")
    decades = math.log10(maximum_absolute / linear_threshold)
    count = max(1, int(math.ceil(decades * bins_per_decade)))
    positive = np.geomspace(linear_threshold, maximum_absolute, count + 1)
    edges = np.concatenate((-positive[::-1], positive))
    if np.any(np.diff(edges) <= 0):
        raise RuntimeError("Generated q histogram edges are not strictly increasing")
    return edges


def conditional_q_histogram(
    observations: pd.DataFrame,
    separation_edges_m: np.ndarray,
    q_edges_m2_s3: np.ndarray,
) -> pd.DataFrame:
    """Save conditional q probability mass for every separation and q bin."""
    separation_edges = np.asarray(separation_edges_m, dtype=float)
    q_edges = np.asarray(q_edges_m2_s3, dtype=float)
    if np.any(np.diff(q_edges) <= 0) or not np.isfinite(q_edges).all():
        raise ValueError("q histogram edges must be finite and strictly increasing")
    separation = observations.separation_m.to_numpy(dtype=float)
    q_values = observations.q_m2_s3.to_numpy(dtype=float)
    separation_bin_index = bin_indices(separation, separation_edges)
    rows: list[dict[str, object]] = []
    for separation_index in range(len(separation_edges) - 1):
        q = q_values[separation_bin_index == separation_index]
        q = q[np.isfinite(q)]
        counts, _ = np.histogram(q, bins=q_edges)
        underflow = int(np.sum(q < q_edges[0]))
        overflow = int(np.sum(q > q_edges[-1]))
        in_range = int(counts.sum())
        mass = counts / in_range if in_range else np.full(len(counts), np.nan)
        for q_index, count in enumerate(counts):
            rows.append({
                "separation_bin_id": f"bin_{separation_index:03d}",
                "separation_bin_index": separation_index,
                "separation_bin_lower_m": float(separation_edges[separation_index]),
                "separation_bin_upper_m": float(separation_edges[separation_index + 1]),
                "q_bin_id": f"qbin_{q_index:03d}",
                "q_bin_index": q_index,
                "q_bin_lower_m2_s3": float(q_edges[q_index]),
                "q_bin_upper_m2_s3": float(q_edges[q_index + 1]),
                "observation_count": int(count),
                "conditional_probability_mass": float(mass[q_index]),
                "separation_column_observation_count": len(q),
                "q_in_range_count": in_range,
                "q_underflow_count": underflow,
                "q_overflow_count": overflow,
            })
    return pd.DataFrame(rows)


def _stable_rng(seed: int, identity: str) -> np.random.Generator:
    digest = int.from_bytes(sha256(identity.encode("utf-8")).digest()[:8], "big")
    sequence = np.random.SeedSequence((seed, digest & 0xFFFFFFFF, digest >> 32))
    return np.random.default_rng(sequence)


def synchronized_block_bootstrap(
    observations: pd.DataFrame,
    separation_edges_m: np.ndarray,
    *,
    window_start_hour: float,
    window_end_hour: float,
    block_minutes: int,
    confidence_level: float,
    replicates: int,
    random_seed: int,
    identity: str,
    minimum_contributing_blocks: int,
    minimum_successful_replicate_fraction: float,
    support_by_bin: dict[str, bool],
) -> pd.DataFrame:
    """Bootstrap complete synchronized UTC blocks after fixed window selection.

    Blocks are aligned to the selected cluster window start. Every pair row at a
    UTC timestamp therefore receives the same block index. Scheduled empty blocks
    remain in the resampling population so intermittent bin occupation is retained.
    """
    duration_minutes = (window_end_hour - window_start_hour) * 60.0
    total_blocks_float = duration_minutes / block_minutes
    total_blocks = int(round(total_blocks_float))
    if not math.isclose(total_blocks_float, total_blocks, abs_tol=1e-10):
        raise ValueError("Bootstrap block duration must divide every configured window")
    if total_blocks < 1:
        raise ValueError("Bootstrap requires at least one scheduled block")
    edges = np.asarray(separation_edges_m, dtype=float)
    separation = observations.separation_m.to_numpy(dtype=float)
    longitudinal = observations.delta_u_l_m_s.to_numpy(dtype=float)
    separation_bin_index = bin_indices(separation, edges)
    relative_minutes = (
        observations.cluster_age_hours.to_numpy(dtype=float) - window_start_hour
    ) * 60
    bootstrap_block_index = np.floor(
        relative_minutes / block_minutes + 1e-10
    ).astype(int)
    if len(observations) and (
        bootstrap_block_index.min() < 0
        or bootstrap_block_index.max() >= total_blocks
    ):
        raise ValueError("Observations fall outside the declared bootstrap window")

    bin_count = len(edges) - 1
    cube_sums = np.zeros((total_blocks, bin_count), dtype=float)
    separation_sums = np.zeros((total_blocks, bin_count), dtype=float)
    observation_counts = np.zeros((total_blocks, bin_count), dtype=np.int64)
    valid = (
        np.isfinite(longitudinal)
        & np.isfinite(separation)
        & (separation > 0)
        & (separation_bin_index >= 0)
        & (separation_bin_index < bin_count)
    )
    block = bootstrap_block_index[valid]
    separation_bin = separation_bin_index[valid]
    np.add.at(cube_sums, (block, separation_bin), longitudinal[valid] ** 3)
    np.add.at(separation_sums, (block, separation_bin), separation[valid])
    np.add.at(observation_counts, (block, separation_bin), 1)

    rng = _stable_rng(random_seed, f"{identity}|{block_minutes}")
    draws = rng.integers(0, total_blocks, size=(replicates, total_blocks))
    weights = np.zeros((replicates, total_blocks), dtype=np.int32)
    replicate_indices = np.repeat(np.arange(replicates), total_blocks)
    np.add.at(weights, (replicate_indices, draws.ravel()), 1)
    replicate_cube = weights @ cube_sums
    replicate_separation = weights @ separation_sums
    replicate_epsilon = np.full((replicates, bin_count), np.nan, dtype=float)
    valid = replicate_separation > 0
    replicate_epsilon[valid] = (
        FOUR_FIFTHS_COEFFICIENT
        * replicate_cube[valid]
        / replicate_separation[valid]
    )
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, object]] = []
    for bin_index in range(bin_count):
        bin_id = f"bin_{bin_index:03d}"
        contributing = int(np.sum(observation_counts[:, bin_index] > 0))
        values = replicate_epsilon[:, bin_index]
        finite = values[np.isfinite(values)]
        successful_fraction = len(finite) / replicates
        if not support_by_bin.get(bin_id, False):
            status = "suppressed_descriptive_support"
        elif contributing < minimum_contributing_blocks:
            status = "suppressed_insufficient_contributing_blocks"
        elif successful_fraction < minimum_successful_replicate_fraction:
            status = "suppressed_insufficient_successful_replicates"
        else:
            status = "available"
        lower = upper = np.nan
        if status == "available":
            lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
        rows.append({
            "separation_bin_id": bin_id,
            "separation_bin_index": bin_index,
            "bootstrap_block_minutes": block_minutes,
            "scheduled_block_count": total_blocks,
            "contributing_block_count": contributing,
            "bootstrap_replicates_requested": replicates,
            "successful_bootstrap_replicates": len(finite),
            "successful_bootstrap_fraction": successful_fraction,
            "confidence_level": confidence_level,
            "epsilon_ci_lower_m2_s3": float(lower),
            "epsilon_ci_upper_m2_s3": float(upper),
            "confidence_interval_status": status,
        })
    return pd.DataFrame(rows)


def leave_one_platform_out(
    observations: pd.DataFrame,
    separation_edges_m: np.ndarray,
    *,
    support_minimum_observations: int,
    support_minimum_pairs: int,
    support_minimum_platforms: int,
    support_minimum_utc: int,
) -> pd.DataFrame:
    """Recalculate each separation-bin estimate after removing one platform."""
    edges = np.asarray(separation_edges_m, dtype=float)
    frame = observations.loc[:, [
        "separation_m", "delta_u_l_m_s", "pair_id", "platform_id_1",
        "platform_id_2", "time_utc",
    ]].copy()
    frame["separation_bin_index"] = bin_indices(
        frame.separation_m.to_numpy(dtype=float), edges,
    )
    rows: list[dict[str, object]] = []
    for bin_index in range(len(edges) - 1):
        selected = frame[frame.separation_bin_index == bin_index]
        baseline = epsilon_from_samples(
            selected.separation_m.to_numpy(dtype=float),
            selected.delta_u_l_m_s.to_numpy(dtype=float),
        )["epsilon_eff_m2_s3"]
        platforms = sorted(
            set(selected.platform_id_1.astype(str))
            | set(selected.platform_id_2.astype(str))
        )
        for platform_id in platforms:
            kept = selected[
                (selected.platform_id_1.astype(str) != platform_id)
                & (selected.platform_id_2.astype(str) != platform_id)
            ]
            estimate = epsilon_from_samples(
                kept.separation_m.to_numpy(dtype=float),
                kept.delta_u_l_m_s.to_numpy(dtype=float),
            )["epsilon_eff_m2_s3"]
            kept_platforms = (
                set(kept.platform_id_1.astype(str))
                | set(kept.platform_id_2.astype(str))
            )
            support_ok = (
                len(kept) >= support_minimum_observations
                and kept.pair_id.nunique() >= support_minimum_pairs
                and len(kept_platforms) >= support_minimum_platforms
                and kept.time_utc.nunique() >= support_minimum_utc
            )
            change = (
                estimate - baseline
                if math.isfinite(estimate) and math.isfinite(baseline) else np.nan
            )
            relative = (
                abs(change) / abs(baseline)
                if math.isfinite(change) and math.isfinite(baseline) and baseline != 0
                else np.nan
            )
            sign_reversal = (
                bool(np.sign(estimate) != np.sign(baseline))
                if math.isfinite(estimate) and math.isfinite(baseline)
                and estimate != 0 and baseline != 0 else False
            )
            rows.append({
                "separation_bin_id": f"bin_{bin_index:03d}",
                "separation_bin_index": bin_index,
                "removed_platform_id": platform_id,
                "baseline_epsilon_eff_m2_s3": baseline,
                "epsilon_after_removal_m2_s3": estimate,
                "epsilon_change_m2_s3": change,
                "absolute_epsilon_change_m2_s3": abs(change) if math.isfinite(change) else np.nan,
                "relative_absolute_change": relative,
                "sign_reversal": sign_reversal,
                "remaining_observation_count": len(kept),
                "remaining_unique_pair_count": int(kept.pair_id.nunique()),
                "remaining_platform_count": len(kept_platforms),
                "remaining_unique_utc_count": int(kept.time_utc.nunique()),
                "support_ok_after_removal": support_ok,
            })
    return pd.DataFrame(rows)


def summarize_platform_influence(details: pd.DataFrame) -> pd.DataFrame:
    """Collapse detailed leave-one-platform-out estimates to readable columns."""
    rows: list[dict[str, object]] = []
    if details.empty:
        return pd.DataFrame()
    for bin_id, group in details.groupby("separation_bin_id", sort=False):
        finite = group[np.isfinite(group.absolute_epsilon_change_m2_s3)]
        if finite.empty:
            rows.append({
                "separation_bin_id": bin_id,
                "leave_one_platform_out_evaluable_count": 0,
                "most_influential_platform_id": None,
                "maximum_lopo_absolute_change_m2_s3": np.nan,
                "maximum_lopo_relative_change": np.nan,
                "most_influential_platform_removal_support_ok": False,
                "lopo_any_sign_reversal": False,
            })
            continue
        leading = finite.loc[finite.absolute_epsilon_change_m2_s3.idxmax()]
        rows.append({
            "separation_bin_id": bin_id,
            "leave_one_platform_out_evaluable_count": len(finite),
            "most_influential_platform_id": str(leading.removed_platform_id),
            "maximum_lopo_absolute_change_m2_s3": float(
                leading.absolute_epsilon_change_m2_s3
            ),
            "maximum_lopo_relative_change": float(leading.relative_absolute_change),
            "most_influential_platform_removal_support_ok": bool(
                leading.support_ok_after_removal
            ),
            "lopo_any_sign_reversal": bool(finite.sign_reversal.any()),
        })
    return pd.DataFrame(rows)
