"""Finite-interval endpoint affine fits, independent of velocity estimators."""

from __future__ import annotations

import json
from collections.abc import Callable
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import pdist

from drifterlab.reconstruction.core import resample_linear_track
from drifterlab.trajectories.kinematics import (
    PROJECTION_ORIGIN_METHOD,
    project_array_positions,
)

from .config import _boolean, _mapping, _optional_positive, _positive
from .fsle import _utc_string

ALGORITHM_VERSION = "original-member-endpoint-affine-v1"


def resolve_affine(value: Any, base: Path) -> dict[str, Any]:
    name = "analyses.affine_deformation"
    raw = _mapping(
        value,
        {
            "enabled",
            "map_window_hours",
            "deformation_window_hours",
            "maximum_elapsed_hours",
            "reference_time_utc",
            "maximum_source_gap_minutes",
            "quality",
            "plotting",
            "sst",
        },
        name,
    )
    result = {"enabled": _boolean(raw.get("enabled", False), name + ".enabled")}
    for key, default in (("map_window_hours", 24), ("deformation_window_hours", 6)):
        result[key] = _positive(raw.get(key, default), name + "." + key)
    duration = duration_ns(result["deformation_window_hours"])
    if result["map_window_hours"] != 24 or duration > duration_ns(24) or duration_ns(24) % duration:
        raise ValueError("Affine daily maps require T=24 h, T_prime<=T and integer T/T_prime")
    for key in ("maximum_elapsed_hours", "maximum_source_gap_minutes"):
        result[key] = _optional_positive(raw.get(key), name + "." + key)
    if result["maximum_elapsed_hours"] is not None:
        duration_ns(result["maximum_elapsed_hours"])
    result["reference_time_utc"] = _utc_string(
        raw.get("reference_time_utc"), name + ".reference_time_utc"
    )
    quality = _mapping(
        raw.get("quality", {}),
        {
            "minimum_aspect_ratio",
            "rank_relative_tolerance",
            "near_isotropy_relative_tolerance",
            "near_collapse_stretch",
            "minimum_rms_radius_m",
            "maximum_normalized_residual",
        },
        name + ".quality",
    )
    defaults = {
        "minimum_aspect_ratio": 0.1,
        "rank_relative_tolerance": 1e-12,
        "near_isotropy_relative_tolerance": 1e-3,
        "near_collapse_stretch": 1e-12,
    }
    result["quality"] = {
        key: _positive(quality.get(key, default), name + ".quality." + key)
        for key, default in defaults.items()
    }
    for key in (
        "minimum_aspect_ratio",
        "rank_relative_tolerance",
        "near_isotropy_relative_tolerance",
    ):
        if result["quality"][key] >= 1:
            raise ValueError(f"{name}.quality.{key} must be less than one")
    for key in ("minimum_rms_radius_m", "maximum_normalized_residual"):
        result["quality"][key] = _optional_positive(quality.get(key), name + ".quality." + key)
    plot = _mapping(
        raw.get("plotting", {}),
        {
            "enabled",
            "dpi",
            "maximum_absolute_rate_h_inverse",
            "axis_length_km",
            "connect_centroids",
            "label_clusters",
            "show_poor_geometry",
            "fixed_extent",
            "longitude_bounds",
            "latitude_bounds",
        },
        name + ".plotting",
    )
    result["plotting"] = {
        key: _boolean(plot.get(key, default), name + ".plotting." + key)
        for key, default in (
            ("enabled", True),
            ("connect_centroids", False),
            ("label_clusters", True),
            ("show_poor_geometry", False),
            ("fixed_extent", True),
        )
    }
    dpi = plot.get("dpi", 140)
    for key, limit in (("longitude_bounds", 180), ("latitude_bounds", 90)):
        bounds = plot.get(key)
        if bounds is not None:
            if (
                not isinstance(bounds, (list, tuple))
                or len(bounds) != 2
                or any(isinstance(v, bool) or not isinstance(v, Real) for v in bounds)
                or not np.isfinite(bounds).all()
                or not -limit <= bounds[0] < bounds[1] <= limit
            ):
                raise ValueError(
                    f"{name}.plotting.{key} must be [minimum, maximum] in degrees "
                    f"within [-{limit}, {limit}], or null"
                )
            if not result["plotting"]["fixed_extent"]:
                raise ValueError(f"{name}.plotting.{key} requires fixed_extent: true")
            bounds = [float(v) for v in bounds]
        result["plotting"][key] = bounds
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("Affine plotting.dpi must be a positive integer")
    result["plotting"].update(
        {
            "dpi": dpi,
            "maximum_absolute_rate_h_inverse": _positive(
                plot.get("maximum_absolute_rate_h_inverse", 0.1), "affine rate limit"
            ),
            "axis_length_km": _optional_positive(plot.get("axis_length_km"), "affine axis length"),
        }
    )
    from .affine_sst import resolve_sst

    result["sst"] = resolve_sst(raw.get("sst", {}), base)
    return result


def duration_ns(hours: float) -> int:
    """Convert exactly representable nanosecond durations, without float ratios."""
    from decimal import Decimal

    value = Decimal(str(hours)) * Decimal(3_600_000_000_000)
    if not value.is_finite() or value <= 0 or value != value.to_integral_value():
        raise ValueError("Durations must be positive whole numbers of nanoseconds")
    return int(value)


def utc(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def schedule(start: Any, end: Any, hours: float) -> pd.DataFrame:
    """Retain all complete midnight-phased intervals inside analysis bounds."""
    start, end = utc(start), utc(end)
    step = pd.Timedelta(duration_ns(hours), unit="ns")
    origin = start.normalize()
    first = origin + ((start.value - origin.value + step.value - 1) // step.value) * step
    rows = []
    current = first
    while current + step <= end:
        rows.append(
            {
                "map_start_utc": current.normalize().isoformat(),
                "map_end_utc": (current.normalize() + pd.Timedelta(days=1)).isoformat(),
                "fit_start_utc": current.isoformat(),
                "fit_end_utc": (current + step).isoformat(),
            }
        )
        current += step
    return pd.DataFrame(
        rows, columns=["map_start_utc", "map_end_utc", "fit_start_utc", "fit_end_utc"]
    )


def fit_affine(
    initial: np.ndarray, final: np.ndarray, hours: float, quality: dict[str, Any]
) -> dict[str, Any]:
    """Solve X @ F.T = Y with equal member weights and an explicit rank test."""
    initial, final = np.asarray(initial, float), np.asarray(final, float)
    if initial.ndim != 2 or initial.shape[1] != 2 or final.shape != initial.shape:
        raise ValueError("Affine configurations must be matching N-by-2 arrays")
    duration_ns(hours)
    numeric = [
        "initial_singular_value_1_m",
        "initial_singular_value_2_m",
        "initial_condition_number",
        "initial_aspect_ratio",
        "initial_major_scale_m",
        "initial_minor_scale_m",
        "initial_rms_radius_m",
        "initial_diameter_m",
        "final_rms_radius_m",
        "residual_sum_squares_m2",
        "rmse_m",
        "normalized_residual",
        "determinant",
        "relative_eigenvalue_separation",
        "stretch_anisotropy_ratio",
    ]
    numeric += [
        f"{prefix}_{i}{j}" for prefix in ("f", "cauchy_green") for i in (1, 2) for j in (1, 2)
    ]
    numeric += [
        f"{prefix}_{i}"
        for prefix in ("cauchy_green_eigenvalue", "stretch", "rate_h_inverse")
        for i in (1, 2)
    ]
    numeric += [f"initial_axis_{i}_{component}" for i in (1, 2) for component in ("x", "y")]
    numeric += [
        f"{stage}_centroid_{component}"
        for stage in ("initial", "final")
        for component in ("x_m", "y_m", "longitude", "latitude")
    ]
    numeric += ["centroid_displacement_m"]
    result = {key: np.nan for key in numeric}
    result.update(
        numerically_valid=False,
        axis_displayable=False,
        isotropic=False,
        status="invalid",
        reason="",
        quality_flags="",
        initial_rank=0,
        eigenvalues_roundoff_clipped=False,
        original_member_count=len(initial),
    )
    if len(initial) < 3:
        result["reason"] = "fewer_than_three_original_members"
        return result
    if not np.isfinite(initial).all() or not np.isfinite(final).all():
        result["reason"] = "missing_endpoint_positions"
        return result
    x, y = initial - initial.mean(axis=0), final - final.mean(axis=0)
    for stage, centroid in (("initial", initial.mean(axis=0)), ("final", final.mean(axis=0))):
        result.update({f"{stage}_centroid_x_m": centroid[0], f"{stage}_centroid_y_m": centroid[1]})
    result["centroid_displacement_m"] = float(
        np.linalg.norm(final.mean(axis=0) - initial.mean(axis=0))
    )
    try:
        solution, _, rank, singular = np.linalg.lstsq(
            x, y, rcond=quality["rank_relative_tolerance"]
        )
    except np.linalg.LinAlgError:
        result["reason"] = "least_squares_failure"
        return result
    result.update(
        initial_rank=int(rank),
        initial_singular_value_1_m=singular[0],
        initial_singular_value_2_m=singular[1],
        initial_condition_number=float(singular[0] / singular[1]) if singular[1] else np.inf,
        initial_aspect_ratio=float(singular[1] / singular[0]) if singular[0] else 0.0,
        initial_major_scale_m=singular[0] / np.sqrt(len(x)),
        initial_minor_scale_m=singular[1] / np.sqrt(len(x)),
        initial_rms_radius_m=float(np.sqrt(np.mean(np.sum(x * x, axis=1)))),
        initial_diameter_m=float(pdist(initial).max()),
        final_rms_radius_m=float(np.sqrt(np.mean(np.sum(y * y, axis=1)))),
    )
    if rank < 2:
        result["reason"] = "rank_deficient_initial_geometry"
        return result
    f = solution.T
    c = f.T @ f
    if not np.isfinite(f).all() or not np.isfinite(c).all():
        result["reason"] = "nonfinite_matrix"
        return result
    for prefix, matrix in (("f", f), ("cauchy_green", c)):
        result.update(
            {f"{prefix}_{i + 1}{j + 1}": float(matrix[i, j]) for i in range(2) for j in range(2)}
        )
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(c)
    except np.linalg.LinAlgError:
        result["reason"] = "eigendecomposition_failure"
        return result
    tolerance = 64 * np.finfo(float).eps * float(np.max(np.abs(eigenvalues)))
    if eigenvalues[0] < -tolerance:
        result["reason"] = "negative_cauchy_green_eigenvalue"
        return result
    result["eigenvalues_roundoff_clipped"] = bool(np.any(eigenvalues < 0))
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    eigenvectors = eigenvectors[:, ::-1]
    stretches = np.sqrt(eigenvalues)
    rates = np.full(2, -np.inf)
    np.log(stretches, out=rates, where=stretches > 0)
    rates /= hours
    error = y - x @ f.T
    rss = float(np.sum(error * error))
    rmse = np.sqrt(rss / len(x))
    normalized = rmse / result["final_rms_radius_m"] if result["final_rms_radius_m"] > 0 else np.nan
    determinant = float(np.linalg.det(f))
    separation = (
        float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]) if eigenvalues[0] > 0 else np.nan
    )
    isotropic = bool(
        np.isfinite(separation) and separation <= quality["near_isotropy_relative_tolerance"]
    )
    flags = []
    if result["initial_aspect_ratio"] < quality["minimum_aspect_ratio"]:
        flags.append("poor_initial_geometry")
    if len(x) == 3:
        flags.append("three_members_no_residual_degrees_of_freedom")
    if determinant < 0:
        flags.append("orientation_reversal")
    if determinant == 0 or stretches[1] <= quality["near_collapse_stretch"]:
        flags.append("collapse_or_near_collapse")
    if isotropic:
        flags.append("near_isotropic_axes")
    if (
        quality["minimum_rms_radius_m"] is not None
        and result["initial_rms_radius_m"] < quality["minimum_rms_radius_m"]
    ):
        flags.append("below_configured_size")
    if (
        quality["maximum_normalized_residual"] is not None
        and np.isfinite(normalized)
        and normalized > quality["maximum_normalized_residual"]
    ):
        flags.append("above_configured_residual")
    for i in range(2):
        result.update(
            {
                f"cauchy_green_eigenvalue_{i + 1}": float(eigenvalues[i]),
                f"stretch_{i + 1}": float(stretches[i]),
                f"rate_h_inverse_{i + 1}": float(rates[i]),
                f"initial_axis_{i + 1}_x": float(eigenvectors[0, i]),
                f"initial_axis_{i + 1}_y": float(eigenvectors[1, i]),
            }
        )
    hidden = {
        "poor_initial_geometry",
        "orientation_reversal",
        "collapse_or_near_collapse",
        "near_isotropic_axes",
        "below_configured_size",
        "above_configured_residual",
    }
    result.update(
        numerically_valid=True,
        status="flagged" if flags else "valid",
        reason="",
        quality_flags=";".join(flags),
        axis_displayable=not bool(hidden.intersection(flags)),
        isotropic=isotropic,
        determinant=determinant,
        residual_sum_squares_m2=rss,
        rmse_m=rmse,
        normalized_residual=normalized,
        relative_eigenvalue_separation=separation,
        stretch_anisotropy_ratio=float(stretches[0] / stretches[1]) if stretches[1] else np.inf,
    )
    return result


def endpoint_positions(
    times: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    source_gap: np.ndarray,
    targets: np.ndarray,
    maximum_gap_minutes: float,
    cutoff: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the approved resampler and preserve upstream gap provenance.

    The effective support gap is max(product bracket span, inherited support at
    both bracket samples). A source gap containing a target either contains one
    bracket sample or lies entirely between them, so this is an upper bound.
    Samples beyond a supplied analysis cutoff cannot support interpolation.
    """
    times, targets = (
        np.asarray(times, dtype="datetime64[ns]"),
        np.asarray(targets, dtype="datetime64[ns]"),
    )
    shape = (len(longitude), len(targets))
    out_lon, out_lat, gaps = [np.full(shape, np.nan) for _ in range(3)]
    reasons = np.full(shape, "no_usable_positions", dtype=object)
    for member in range(len(longitude)):
        usable = (
            np.isfinite(longitude[member])
            & np.isfinite(latitude[member])
            & np.isfinite(source_gap[member])
        )
        usable &= (source_gap[member] >= 0) & (source_gap[member] <= maximum_gap_minutes)
        if cutoff is not None:
            usable &= times <= utc(cutoff).tz_localize(None).to_datetime64()
        indices = np.flatnonzero(usable)
        if not len(indices):
            continue
        t = times[indices]
        if len(indices) == 1:
            exact = targets == t[0]
            out_lon[member, exact], out_lat[member, exact] = (
                longitude[member, indices[0]],
                latitude[member, indices[0]],
            )
            gaps[member, exact] = source_gap[member, indices[0]]
            reasons[member, exact] = "valid"
            continue
        sampled = resample_linear_track(
            t,
            longitude[member, indices],
            latitude[member, indices],
            targets,
            coverage_start=t[0],
            coverage_end=t[-1],
            maximum_gap_minutes=maximum_gap_minutes,
        )
        right = np.searchsorted(t, targets, side="left")
        right = np.clip(right, 0, len(t) - 1)
        exact = targets == t[right]
        left = np.where(exact, right, np.maximum(right - 1, 0))
        inherited = np.maximum(
            source_gap[member, indices[left]], source_gap[member, indices[right]]
        )
        effective = np.maximum(inherited, sampled.source_gap_minutes)
        supported = np.isfinite(sampled.longitude) & (effective <= maximum_gap_minutes)
        out_lon[member] = np.where(supported, sampled.longitude, np.nan)
        out_lat[member] = np.where(supported, sampled.latitude, np.nan)
        gaps[member] = effective
        reasons[member] = np.where(supported, "valid", "endpoint_gap_exceeds_threshold")
        reasons[member, (targets < t[0]) | (targets > t[-1])] = "endpoint_outside_usable_trajectory"
    return out_lon, out_lat, gaps, reasons


def calculate_array(
    array: xr.Dataset,
    settings: dict[str, Any],
    maximum_gap: float,
    selected_clusters: str | tuple[str, ...] = "all",
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Any]:
    """Calculate all scheduled attempts; SST never enters the estimator."""
    times = array.time.values.astype("datetime64[ns]")
    lon, lat, gaps = [
        array[key].values.astype(float) for key in ("longitude", "latitude", "source_gap_minutes")
    ]
    clusters = array.cluster_id.values.astype(str)
    ids = array.platform_id.values.astype(str)
    # Fill only allowed internal holes for lifecycle/activation, never require
    # continuous support between the actual diagnostic endpoints.
    supported_lon, supported_lat, _, _ = endpoint_positions(
        times, lon, lat, gaps, times, maximum_gap
    )
    finite = np.isfinite(supported_lon) & np.isfinite(supported_lat)
    catalogue = []
    activations = []
    for cluster in sorted(set(clusters)):
        members = np.flatnonzero(clusters == cluster)
        declared = array.cluster_size.values[members]
        if not np.all(np.isfinite(declared) & (declared == len(members))):
            raise ValueError(f"Cluster {cluster}: original membership disagrees with cluster_size")
        complete = np.flatnonzero(finite[members].all(axis=0))
        activation = utc(times[complete[0]]) if len(complete) else None
        last = [np.flatnonzero(finite[m]) for m in members]
        lifetime_end = utc(min(times[v[-1]] for v in last)) if all(len(v) for v in last) else None
        reason = (
            "fewer_than_three_original_members"
            if len(members) < 3
            else "complete_cluster_never_available"
            if activation is None
            else "eligible"
        )
        if reason == "eligible":
            activations.append(activation)
        catalogue.append(
            {
                "array_id": int(array.array_id.values[0]),
                "cluster_id": cluster,
                "original_member_count": len(members),
                "original_member_ids": json.dumps(ids[members].tolist()),
                "activation_time_utc": activation.isoformat() if activation is not None else None,
                "lifetime_end_utc": lifetime_end.isoformat() if lifetime_end is not None else None,
                "eligibility": reason,
                "selected": selected_clusters == "all" or cluster in selected_clusters,
            }
        )
    start = (
        utc(settings["reference_time_utc"])
        if settings["reference_time_utc"]
        else min(activations)
        if activations
        else None
    )
    metadata = {
        "analysis_start_utc": start.isoformat() if start is not None else None,
        "maximum_source_gap_minutes": maximum_gap,
        "reference_population": "all_original_clusters_in_selected_array_before_cluster_selection",
    }
    if start is None:
        return pd.DataFrame(), pd.DataFrame(catalogue), metadata, None
    end = (
        start + pd.Timedelta(duration_ns(settings["maximum_elapsed_hours"]), unit="ns")
        if settings["maximum_elapsed_hours"]
        else max(
            [start]
            + [
                utc(entry["lifetime_end_utc"])
                for entry in catalogue
                if entry["selected"] and entry["eligibility"] == "eligible"
            ]
        )
    )
    # An explicit Tmax may extend past data; retain attempted rows and lifecycle
    # reasons, but never extrapolate positions or shorten an interval.
    intervals = schedule(start, end, settings["deformation_window_hours"])
    metadata.update(analysis_end_utc=end.isoformat(), map_origin_utc=start.normalize().isoformat())
    # Projection uses first available per-platform positions and is independent
    # of cluster selection. Never use post-cutoff positions to derive the origin.
    before_end = times <= end.tz_localize(None).to_datetime64()
    projectable = np.isfinite(lon[:, before_end]).any(axis=1)
    projection = (
        project_array_positions(lon[projectable][:, before_end], lat[projectable][:, before_end])
        if projectable.any()
        else None
    )
    if projection is not None:
        metadata["projection"] = {
            "crs_definition": projection.crs_definition,
            "origin_longitude": projection.origin_longitude,
            "origin_latitude": projection.origin_latitude,
            "origin_method": PROJECTION_ORIGIN_METHOD,
        }
        px, py = projection.x_m, projection.y_m
        valid_xy = np.isfinite(px) & np.isfinite(py)
        bounds = [
            float(px[valid_xy].min()),
            float(px[valid_xy].max()),
            float(py[valid_xy].min()),
            float(py[valid_xy].max()),
        ]
        margin = max(bounds[1] - bounds[0], bounds[3] - bounds[2], 1000) * 0.06
        metadata["extent_m"] = [
            bounds[0] - margin,
            bounds[1] + margin,
            bounds[2] - margin,
            bounds[3] + margin,
        ]
    targets = np.unique(
        np.asarray(
            [
                utc(v).tz_localize(None).to_datetime64()
                for key in ("fit_start_utc", "fit_end_utc")
                for v in intervals[key]
            ],
            dtype="datetime64[ns]",
        )
    )
    target_lon, target_lat, target_gaps, target_reasons = endpoint_positions(
        times, lon, lat, gaps, targets, maximum_gap, cutoff=end
    )
    rows = []
    for entry in catalogue:
        if not entry["selected"]:
            continue
        members = np.flatnonzero(clusters == entry["cluster_id"])
        for interval in intervals.to_dict("records"):
            empty = fit_affine(
                np.empty((0, 2)),
                np.empty((0, 2)),
                settings["deformation_window_hours"],
                settings["quality"],
            )
            row = {
                **empty,
                **entry,
                **interval,
                "deformation_window_hours": settings["deformation_window_hours"],
                "numerically_valid": False,
                "axis_displayable": False,
                "isotropic": False,
                "status": "invalid",
                "reason": "",
                "quality_flags": "",
                "sst_status": "not_loaded",
            }
            i, j = [
                int(np.searchsorted(targets, utc(interval[key]).tz_localize(None).to_datetime64()))
                for key in ("fit_start_utc", "fit_end_utc")
            ]
            row.update(
                initial_endpoint_count=int(np.isfinite(target_lon[members, i]).sum()),
                final_endpoint_count=int(np.isfinite(target_lon[members, j]).sum()),
                interior_support_required=False,
                maximum_endpoint_support_gap_minutes=float(
                    np.nanmax(target_gaps[members][:, [i, j]])
                )
                if np.isfinite(target_gaps[members][:, [i, j]]).any()
                else np.nan,
            )
            if entry["eligibility"] != "eligible":
                row["reason"] = entry["eligibility"]
            elif utc(interval["fit_start_utc"]) < utc(entry["activation_time_utc"]):
                row["reason"] = "before_cluster_activation"
            elif utc(interval["fit_end_utc"]) > utc(entry["lifetime_end_utc"]):
                row["reason"] = "original_member_usable_trajectory_ended"
            elif row["initial_endpoint_count"] != len(members) or row[
                "final_endpoint_count"
            ] != len(members):
                row["reason"] = ";".join(
                    sorted(set(target_reasons[members][:, [i, j]].ravel()) - {"valid"})
                )
            elif projection is None:
                row["reason"] = "projection_unavailable"
            else:
                configurations = [
                    np.column_stack(
                        projection.forward.transform(target_lon[members, k], target_lat[members, k])
                    )
                    for k in (i, j)
                ]
                row.update(
                    fit_affine(
                        *configurations, settings["deformation_window_hours"], settings["quality"]
                    )
                )
                for stage in ("initial", "final"):
                    if f"{stage}_centroid_x_m" in row:
                        clon, clat = projection.inverse.transform(
                            row[f"{stage}_centroid_x_m"], row[f"{stage}_centroid_y_m"]
                        )
                        row.update(
                            {
                                f"{stage}_centroid_longitude": clon,
                                f"{stage}_centroid_latitude": clat,
                            }
                        )
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(catalogue), metadata, projection


def run_affine_analysis(
    dataset: xr.Dataset,
    settings: dict[str, Any],
    directory: Path,
    *,
    selected_clusters: str | tuple[str, ...],
    product_maximum_gap_minutes: float,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from drifterlab.plotting.affine_deformation import geographic_extent, map_extent, render_maps

    from .affine_sst import PreparedSST

    report = progress or (lambda message: None)
    configured = settings["maximum_source_gap_minutes"]
    maximum_gap = (
        min(configured, product_maximum_gap_minutes)
        if configured is not None
        else product_maximum_gap_minutes
    )
    backgrounds = PreparedSST(settings["sst"])
    counts = {
        "enabled": True,
        "attempted_fit_count": 0,
        "numerically_valid_fit_count": 0,
        "figure_count": 0,
    }
    for array_id in sorted(set(dataset.array_id.values.astype(int))):
        report(
            f"Affine deformation array {array_id}: reconstructing endpoints and fitting original members"
        )
        array = dataset.isel(platform=np.flatnonzero(dataset.array_id.values == array_id))
        if (
            selected_clusters != "all"
            and not np.isin(array.cluster_id.values.astype(str), selected_clusters).any()
        ):
            continue
        rows, membership, metadata, projection = calculate_array(
            array, settings, maximum_gap, selected_clusters
        )
        root = directory / f"array_{array_id:03d}"
        (root / "data").mkdir(parents=True)
        membership.to_csv(root / "data" / "original_membership.csv", index=False)
        windows = []
        if metadata["analysis_start_utc"] is not None:
            start, end = utc(metadata["analysis_start_utc"]), utc(metadata["analysis_end_utc"])
            day = start.normalize()
            while day < end:
                requested_end = day + pd.Timedelta(days=1)
                daily_rows = rows[rows.map_start_utc == day.isoformat()] if len(rows) else rows
                frame_extent = map_extent(daily_rows, metadata, settings["plotting"])
                geographic_bounds = geographic_extent(frame_extent, metadata, settings["plotting"])
                explicit_bounds = any(
                    settings["plotting"].get(key) is not None
                    for key in ("longitude_bounds", "latitude_bounds")
                )
                background, sst_meta = backgrounds.load(
                    day, frame_extent, projection,
                    geographic_bounds=geographic_bounds if explicit_bounds else None,
                )
                report(f"Affine array {array_id}, {day:%Y-%m-%d}: {sst_meta['status']}")
                if background is not None:
                    (root / "sst").mkdir(exist_ok=True)
                    background.to_netcdf(root / "sst" / f"{day:%Y%m%d}.nc")
                if len(rows):
                    rows.loc[rows.map_start_utc == day.isoformat(), "sst_status"] = sst_meta[
                        "status"
                    ]
                valid_count = int(daily_rows.numerically_valid.sum()) if len(daily_rows) else 0
                valid_rows = (
                    daily_rows[daily_rows.numerically_valid] if len(daily_rows) else daily_rows
                )
                window = {
                    "extent_m": frame_extent,
                    "geographic_extent_deg": geographic_bounds,
                    "map_start_utc": day.isoformat(),
                    "requested_sst_end_utc": requested_end.isoformat(),
                    "analysis_overlap_start_utc": max(start, day).isoformat(),
                    "analysis_overlap_end_utc": min(end, requested_end).isoformat(),
                    "scheduled_coverage_start_utc": daily_rows.fit_start_utc.min()
                    if len(daily_rows)
                    else None,
                    "scheduled_coverage_end_utc": daily_rows.fit_end_utc.max()
                    if len(daily_rows)
                    else None,
                    "diagnostic_coverage_start_utc": valid_rows.fit_start_utc.min()
                    if len(valid_rows)
                    else None,
                    "diagnostic_coverage_end_utc": valid_rows.fit_end_utc.max()
                    if len(valid_rows)
                    else None,
                    "attempted_fit_count": len(daily_rows),
                    "valid_fit_count": valid_count,
                    "sst": sst_meta,
                }
                windows.append(window)
                if settings["plotting"]["enabled"] and projection is not None:
                    render_maps(
                        daily_rows,
                        background,
                        window,
                        metadata,
                        settings,
                        array_id,
                        root / "figures" / f"{day:%Y%m%d}.png",
                    )
                    counts["figure_count"] += 1
                day = requested_end
        if not len(rows):
            rows = pd.DataFrame(
                columns=[
                    "array_id",
                    "cluster_id",
                    "fit_start_utc",
                    "fit_end_utc",
                    "status",
                    "reason",
                    "numerically_valid",
                    "axis_displayable",
                    "quality_flags",
                    "sst_status",
                ]
            )
        rows.to_parquet(root / "data" / "affine_intervals.parquet", index=False)
        rows.to_csv(root / "data" / "affine_intervals.csv", index=False)
        valid = int(rows.numerically_valid.sum())
        counts["attempted_fit_count"] += len(rows)
        counts["numerically_valid_fit_count"] += valid
        metadata.update(
            algorithm_version=ALGORITHM_VERSION,
            settings=settings,
            membership_policy="original_members_endpoint_only_stop_at_first_member_usable_end",
            input_product=dataset.attrs,
            attempted_fit_count=len(rows),
            numerically_valid_fit_count=valid,
            endpoint_interpolation="approved_linear_resampler_with_inherited_source_gap_provenance_no_extrapolation",
            exclusion_counts=rows.loc[~rows.numerically_valid, "reason"].value_counts().to_dict(),
            membership_exclusion_counts=membership.loc[
                membership.eligibility != "eligible", "eligibility"
            ]
            .value_counts()
            .to_dict(),
            quality_flag_counts=rows.quality_flags.str.split(";")
            .explode()
            .loc[lambda s: s.ne("")]
            .value_counts()
            .to_dict(),
            windows=windows,
        )
        (root / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
        )
    return counts
