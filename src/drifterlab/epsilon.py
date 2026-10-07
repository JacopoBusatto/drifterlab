"""Pure calculations for signed longitudinal pair statistics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Literal

import numpy as np
import pandas as pd


FOUR_FIFTHS_COEFFICIENT = -5.0 / 4.0
EARTH_ROTATION_RATE_RAD_S = 7.292115e-5


@dataclass(frozen=True)
class BinSupportRequirements:
    """Minimum descriptive support for one temporal separation-bin estimate."""

    observations: int
    unique_pairs: int
    platforms: int
    unique_utc: int


def stable_pair_id(platform_id_1: str, platform_id_2: str) -> str:
    """Return an order-independent identifier while retaining IDs in saved rows."""
    members = sorted((str(platform_id_1), str(platform_id_2)))
    if not members[0] or not members[1] or members[0] == members[1]:
        raise ValueError("A pair requires two different nonempty platform IDs")
    encoded = json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode()
    return f"pair_{sha256(encoded).hexdigest()[:20]}"


def first_cluster_activation(
    usable: np.ndarray,
    minimum_active_members: int | Literal["all"],
) -> int | None:
    """Return the first grid index satisfying a cluster membership threshold."""
    values = np.asarray(usable, dtype=bool)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("Cluster usability must be a nonempty member-by-time matrix")
    if minimum_active_members == "all":
        threshold = values.shape[0]
    elif (
        isinstance(minimum_active_members, bool)
        or not isinstance(minimum_active_members, int)
        or minimum_active_members < 2
    ):
        raise ValueError("minimum_active_members must be an integer >=2 or 'all'")
    else:
        threshold = minimum_active_members
    if threshold > values.shape[0]:
        return None
    candidates = np.flatnonzero(values.sum(axis=0) >= threshold)
    return None if not len(candidates) else int(candidates[0])


def geometric_bin_edges(
    minimum_m: float,
    maximum_m: float,
    ratio: float,
    anchor_m: float,
) -> np.ndarray:
    """Build stable geometric edges containing the configured anchor exactly."""
    values = (minimum_m, maximum_m, ratio, anchor_m)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Geometric-bin parameters must be finite and positive")
    if maximum_m <= minimum_m:
        raise ValueError("Maximum bin edge must exceed the minimum")
    if ratio <= 1:
        raise ValueError("Geometric-bin ratio must exceed one")
    if not minimum_m <= anchor_m <= maximum_m:
        raise ValueError("Geometric-bin anchor must lie inside the edge range")
    log_ratio = math.log(ratio)
    lower = math.ceil(math.log(minimum_m / anchor_m) / log_ratio - 1e-12)
    upper = math.floor(math.log(maximum_m / anchor_m) / log_ratio + 1e-12)
    powers = np.arange(lower, upper + 1, dtype=np.int64)
    edges = anchor_m * np.power(ratio, powers.astype(float))
    edges[powers == 0] = anchor_m
    if math.isclose(edges[0], minimum_m, rel_tol=1e-12, abs_tol=0.0):
        edges[0] = minimum_m
    if math.isclose(edges[-1], maximum_m, rel_tol=1e-12, abs_tol=0.0):
        edges[-1] = maximum_m
    if len(edges) < 2:
        raise ValueError("Geometric configuration must produce at least two edges")
    return edges


def linear_bin_edges(minimum_m: float, maximum_m: float, width_m: float) -> np.ndarray:
    """Build linear edges without silently shortening the final configured range."""
    if not all(math.isfinite(value) for value in (minimum_m, maximum_m, width_m)):
        raise ValueError("Linear-bin parameters must be finite")
    if minimum_m < 0 or maximum_m <= minimum_m or width_m <= 0:
        raise ValueError("Linear-bin bounds and width are invalid")
    count_float = (maximum_m - minimum_m) / width_m
    count = round(count_float)
    if not math.isclose(count_float, count, rel_tol=0, abs_tol=1e-10) or count < 1:
        raise ValueError("Linear-bin width must divide the configured range exactly")
    return minimum_m + width_m * np.arange(count + 1, dtype=float)


def bin_indices(separation_m: np.ndarray, edges_m: np.ndarray) -> np.ndarray:
    """Assign left-closed bins, including the final edge in the final bin."""
    values = np.asarray(separation_m, dtype=float)
    edges = np.asarray(edges_m, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all():
        raise ValueError("Separation edges must be a finite 1D array")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Separation edges must increase strictly")
    result = np.searchsorted(edges, values, side="right") - 1
    result[np.isclose(values, edges[-1], rtol=0, atol=0)] = len(edges) - 2
    result[~np.isfinite(values)] = -2
    return result.astype(np.int64)


def pair_kinematics(
    x_1_m: np.ndarray,
    y_1_m: np.ndarray,
    u_1_m_s: np.ndarray,
    v_1_m_s: np.ndarray,
    x_2_m: np.ndarray,
    y_2_m: np.ndarray,
    u_2_m_s: np.ndarray,
    v_2_m_s: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate signed longitudinal/transverse increments and individual q."""
    arrays = [
        np.asarray(value, dtype=float)
        for value in (x_1_m, y_1_m, u_1_m_s, v_1_m_s, x_2_m, y_2_m, u_2_m_s, v_2_m_s)
    ]
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays):
        raise ValueError("Pair kinematic inputs must have equal shapes")
    x1, y1, u1, v1, x2, y2, u2, v2 = arrays
    dx = x2 - x1
    dy = y2 - y1
    separation = np.hypot(dx, dy)
    du = u2 - u1
    dv = v2 - v1
    valid = (
        np.isfinite(separation)
        & (separation > 0)
        & np.isfinite(du)
        & np.isfinite(dv)
    )
    longitudinal = np.full(shape, np.nan, dtype=float)
    transverse = np.full(shape, np.nan, dtype=float)
    longitudinal[valid] = (
        du[valid] * dx[valid] + dv[valid] * dy[valid]
    ) / separation[valid]
    # The declared counter-clockwise transverse unit vector is (-r_y, r_x)/|r|.
    transverse[valid] = (
        -du[valid] * dy[valid] + dv[valid] * dx[valid]
    ) / separation[valid]
    cube = longitudinal ** 3
    q = np.full(shape, np.nan, dtype=float)
    q[valid] = FOUR_FIFTHS_COEFFICIENT * cube[valid] / separation[valid]
    return {
        "delta_x_m": dx,
        "delta_y_m": dy,
        "separation_m": separation,
        "delta_u_m_s": du,
        "delta_v_m_s": dv,
        "delta_u_l_m_s": longitudinal,
        "delta_u_t_m_s": transverse,
        "delta_u_l_cube_m3_s3": cube,
        "delta_u_l_delta_u_t_squared_m3_s3": longitudinal * transverse ** 2,
        "q_m2_s3": q,
        "valid": valid,
    }


def epsilon_from_samples(separation_m: np.ndarray, delta_u_l_m_s: np.ndarray) -> dict[str, float]:
    """Calculate the exact finite-bin estimator and its weighted-q identity."""
    separation = np.asarray(separation_m, dtype=float)
    longitudinal = np.asarray(delta_u_l_m_s, dtype=float)
    valid = np.isfinite(separation) & (separation > 0) & np.isfinite(longitudinal)
    if not valid.any():
        return {
            "mean_separation_m": np.nan,
            "s_lll_m3_s3": np.nan,
            "epsilon_eff_m2_s3": np.nan,
            "mean_q_m2_s3": np.nan,
            "separation_weighted_mean_q_m2_s3": np.nan,
            "identity_absolute_error_m2_s3": np.nan,
        }
    separation = separation[valid]
    cube = longitudinal[valid] ** 3
    q = FOUR_FIFTHS_COEFFICIENT * cube / separation
    mean_separation = float(np.mean(separation))
    s_lll = float(np.mean(cube))
    epsilon = FOUR_FIFTHS_COEFFICIENT * s_lll / mean_separation
    weighted = float(np.sum(separation * q) / np.sum(separation))
    return {
        "mean_separation_m": mean_separation,
        "s_lll_m3_s3": s_lll,
        "epsilon_eff_m2_s3": epsilon,
        "mean_q_m2_s3": float(np.mean(q)),
        "separation_weighted_mean_q_m2_s3": weighted,
        "identity_absolute_error_m2_s3": abs(epsilon - weighted),
    }


def _moments(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    result = {
        f"mean_{prefix}": np.nan,
        f"raw_{prefix}_second": np.nan,
        f"raw_{prefix}_third": np.nan,
        f"raw_{prefix}_fourth": np.nan,
        f"raw_{prefix}_third_ratio": np.nan,
        f"raw_{prefix}_fourth_ratio": np.nan,
        f"central_{prefix}_variance": np.nan,
        f"central_{prefix}_third": np.nan,
        f"central_{prefix}_fourth": np.nan,
        f"central_{prefix}_skewness": np.nan,
        f"central_{prefix}_flatness": np.nan,
    }
    if not len(finite):
        return result
    mean = float(np.mean(finite))
    raw_second = float(np.mean(finite ** 2))
    raw_third = float(np.mean(finite ** 3))
    raw_fourth = float(np.mean(finite ** 4))
    centered = finite - mean
    central_second = float(np.mean(centered ** 2))
    central_third = float(np.mean(centered ** 3))
    central_fourth = float(np.mean(centered ** 4))
    result.update({
        f"mean_{prefix}": mean,
        f"raw_{prefix}_second": raw_second,
        f"raw_{prefix}_third": raw_third,
        f"raw_{prefix}_fourth": raw_fourth,
        f"raw_{prefix}_third_ratio": (
            raw_third / raw_second ** 1.5 if raw_second > 0 else np.nan
        ),
        f"raw_{prefix}_fourth_ratio": (
            raw_fourth / raw_second ** 2 if raw_second > 0 else np.nan
        ),
        f"central_{prefix}_variance": central_second,
        f"central_{prefix}_third": central_third,
        f"central_{prefix}_fourth": central_fourth,
        f"central_{prefix}_skewness": (
            central_third / central_second ** 1.5 if central_second > 0 else np.nan
        ),
        f"central_{prefix}_flatness": (
            central_fourth / central_second ** 2 if central_second > 0 else np.nan
        ),
    })
    return result


def _support_status(
    frame: pd.DataFrame,
    *,
    product_kind: str,
    requirements: BinSupportRequirements,
) -> tuple[bool, str]:
    observations = len(frame)
    pairs = int(frame.pair_id.nunique()) if observations else 0
    platforms = (
        len(set(frame.platform_id_1.astype(str)) | set(frame.platform_id_2.astype(str)))
        if observations else 0
    )
    utc = int(frame.time.nunique()) if observations else 0
    if observations == 0:
        return False, "missing_no_observations"
    if product_kind == "snapshot":
        if pairs == 1:
            return False, "descriptive_snapshot_single_pair"
        if pairs < requirements.unique_pairs or platforms < requirements.platforms:
            return False, "descriptive_snapshot_low_pair_or_platform_support"
        return False, "descriptive_snapshot_no_temporal_uncertainty"
    reasons: list[str] = []
    if observations < requirements.observations:
        reasons.append("observations")
    if pairs < requirements.unique_pairs:
        reasons.append("pairs")
    if platforms < requirements.platforms:
        reasons.append("platforms")
    if utc < requirements.unique_utc:
        reasons.append("utc")
    return (not reasons, "supported" if not reasons else "low_support_" + "_".join(reasons))


def summarize_separation_bins(
    observations: pd.DataFrame,
    edges_m: np.ndarray,
    centers_m: np.ndarray,
    *,
    product_kind: Literal["snapshot", "window"],
    scheduled_utc_count: int,
    requirements: BinSupportRequirements,
) -> pd.DataFrame:
    """Summarize signed pair statistics in every configured separation bin."""
    required = {
        "time", "cluster_age_hours", "pair_id", "platform_id_1", "platform_id_2",
        "separation_m", "delta_u_l_m_s", "delta_u_t_m_s", "q_m2_s3",
        "midpoint_latitude",
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Pair observations are missing summary columns: {sorted(missing)}")
    edges = np.asarray(edges_m, dtype=float)
    centers = np.asarray(centers_m, dtype=float)
    if centers.shape != (len(edges) - 1,):
        raise ValueError("Nominal bin centers must match separation intervals")
    frame = observations.copy()
    frame["separation_bin_index"] = bin_indices(frame.separation_m.to_numpy(), edges)
    rows: list[dict[str, object]] = []
    for index, (lower, upper, center) in enumerate(zip(edges[:-1], edges[1:], centers)):
        selected = frame[frame.separation_bin_index == index]
        support_ok, support_status = _support_status(
            selected, product_kind=product_kind, requirements=requirements,
        )
        pair_ids = selected.pair_id.astype(str) if len(selected) else pd.Series(dtype=str)
        platform_values = (
            set(selected.platform_id_1.astype(str)) | set(selected.platform_id_2.astype(str))
            if len(selected) else set()
        )
        utc_values = np.sort(selected.time.unique()) if len(selected) else np.asarray([])
        estimator = epsilon_from_samples(
            selected.separation_m.to_numpy(dtype=float),
            selected.delta_u_l_m_s.to_numpy(dtype=float),
        )
        longitudinal_moments = _moments(
            selected.delta_u_l_m_s.to_numpy(dtype=float), "delta_u_l_m_s",
        )
        transverse_moments = _moments(
            selected.delta_u_t_m_s.to_numpy(dtype=float), "delta_u_t_m_s",
        )
        midpoint_latitude = (
            float(selected.midpoint_latitude.mean()) if len(selected) else np.nan
        )
        coriolis = (
            2 * EARTH_ROTATION_RATE_RAD_S * math.sin(math.radians(midpoint_latitude))
            if math.isfinite(midpoint_latitude) else np.nan
        )
        s_ll = longitudinal_moments["raw_delta_u_l_m_s_second"]
        mean_separation = estimator["mean_separation_m"]
        rossby = (
            math.sqrt(s_ll) / (abs(coriolis) * mean_separation)
            if math.isfinite(s_ll) and s_ll >= 0
            and math.isfinite(coriolis) and coriolis != 0
            and math.isfinite(mean_separation) and mean_separation > 0
            else np.nan
        )
        cube = (
            selected.delta_u_l_m_s.to_numpy(dtype=float) ** 3
            if len(selected) else np.asarray([], dtype=float)
        )
        absolute_cube = np.abs(cube)
        absolute_sum = float(np.sum(absolute_cube)) if len(cube) else 0.0
        ordered = np.sort(absolute_cube)[::-1] if len(cube) else absolute_cube
        max_index = int(np.argmax(absolute_cube)) if len(cube) else None
        max_fraction = float(ordered[0] / absolute_sum) if absolute_sum > 0 else 0.0
        top_five_fraction = (
            float(np.sum(ordered[:5]) / absolute_sum) if absolute_sum > 0 else 0.0
        )
        effective_cube_count = (
            absolute_sum ** 2 / float(np.sum(absolute_cube ** 2))
            if len(cube) and np.sum(absolute_cube ** 2) > 0 else np.nan
        )
        epsilon_without_extreme = np.nan
        if len(selected) > 1 and max_index is not None:
            keep = np.ones(len(selected), dtype=bool)
            keep[max_index] = False
            epsilon_without_extreme = epsilon_from_samples(
                selected.separation_m.to_numpy(dtype=float)[keep],
                selected.delta_u_l_m_s.to_numpy(dtype=float)[keep],
            )["epsilon_eff_m2_s3"]
        maximum_row = selected.iloc[max_index] if max_index is not None else None
        unique_utc = len(utc_values)
        if unique_utc > 1:
            utc_span_hours = float(
                (utc_values[-1] - utc_values[0]) / np.timedelta64(1, "h")
            )
        else:
            utc_span_hours = 0.0 if unique_utc == 1 else np.nan
        occupied_blocks = (
            int(pd.to_datetime(selected.time).dt.floor("h").nunique())
            if len(selected) else 0
        )
        ages = selected.cluster_age_hours.to_numpy(dtype=float)
        rows.append({
            "separation_bin_index": index,
            "separation_bin_lower_m": float(lower),
            "separation_bin_upper_m": float(upper),
            "separation_bin_nominal_center_m": float(center),
            **estimator,
            **longitudinal_moments,
            **transverse_moments,
            "mean_midpoint_latitude": midpoint_latitude,
            "mean_bin_coriolis_s_inverse": coriolis,
            "poje_longitudinal_rossby_number": rossby,
            "observation_count": len(selected),
            "unique_pair_count": int(pair_ids.nunique()),
            "platform_count": len(platform_values),
            "unique_utc_count": unique_utc,
            "scheduled_utc_count": scheduled_utc_count,
            "occupied_utc_fraction": (
                unique_utc / scheduled_utc_count if scheduled_utc_count else np.nan
            ),
            "occupied_one_hour_block_count": occupied_blocks,
            "utc_span_hours": utc_span_hours,
            "minimum_cluster_age_hours": float(np.min(ages)) if len(ages) else np.nan,
            "maximum_cluster_age_hours": float(np.max(ages)) if len(ages) else np.nan,
            "support_ok": support_ok,
            "support_status": support_status,
            "confidence_interval_status": "not_computed_stage_2",
            "maximum_absolute_cube_fraction": max_fraction,
            "top_five_absolute_cube_fraction": top_five_fraction,
            "absolute_cube_effective_observation_count": effective_cube_count,
            "cube_cancellation_ratio": (
                abs(float(np.sum(cube))) / absolute_sum if absolute_sum > 0 else 0.0
            ),
            "most_influential_pair_id": (
                None if maximum_row is None else str(maximum_row.pair_id)
            ),
            "most_influential_time": (
                None if maximum_row is None else maximum_row.time
            ),
            "epsilon_without_most_extreme_m2_s3": epsilon_without_extreme,
            "epsilon_change_without_most_extreme_m2_s3": (
                epsilon_without_extreme - estimator["epsilon_eff_m2_s3"]
                if math.isfinite(epsilon_without_extreme)
                and math.isfinite(estimator["epsilon_eff_m2_s3"])
                else np.nan
            ),
        })
    return pd.DataFrame(rows)


def separation_range_counts(observations: pd.DataFrame, edges_m: np.ndarray) -> dict[str, int]:
    """Count configured-bin, underflow, overflow, and invalid observations."""
    indices = bin_indices(observations.separation_m.to_numpy(dtype=float), edges_m)
    final_bin = len(edges_m) - 2
    return {
        "total_observation_count": len(indices),
        "in_configured_bins_count": int(((indices >= 0) & (indices <= final_bin)).sum()),
        "underflow_count": int((indices == -1).sum()),
        "overflow_count": int((indices > final_bin).sum()),
        "invalid_separation_count": int((indices == -2).sum()),
    }


def numeric_distribution_summary(
    frame: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Return labelled descriptive summaries without calling them structure functions."""
    rows: list[dict[str, object]] = []
    for metric in metrics:
        if metric not in frame:
            raise ValueError(f"Distribution metric {metric!r} is unavailable")
        values = frame[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        moments = _moments(values, "value")
        row: dict[str, object] = {"metric": metric, "count": len(values)}
        if len(values):
            quantiles = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1])
            for label, value in zip(
                ("minimum", "p01", "p05", "p25", "median", "p75", "p95", "p99", "maximum"),
                quantiles,
            ):
                row[label] = float(value)
            row["standard_deviation"] = float(np.std(values))
        else:
            for label in (
                "minimum", "p01", "p05", "p25", "median", "p75", "p95", "p99",
                "maximum", "standard_deviation",
            ):
                row[label] = np.nan
        row.update(moments)
        rows.append(row)
    return pd.DataFrame(rows)
