"""Direct FSLE analysis of selected grouped v2 trajectory products."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod
import xarray as xr

from drifterlab.fsle import (
    PASSAGE_POLICIES,
    aggregate_fsle_populations,
    build_scale_thresholds,
    collect_observed_first_passages,
    fit_fsle_power_laws,
)

from .config import _boolean, _mapping, _optional_positive, _positive


FSLE_ALGORITHM_VERSION = "direct-observed-shells-v1"
WGS84 = Geod(ellps="WGS84")


def resolve_fsle(value: Any) -> dict[str, Any]:
    """Validate FSLE independently of epsilon and velocity configuration."""
    raw = _mapping(value, {
        "enabled", "passage_policy", "maximum_source_gap_minutes", "chance_pair_threshold_km",
        "minimum_scale_km", "maximum_scale_km", "rho", "anchor_scale_km",
        "minimum_reached_pairs_per_scale", "time", "fits", "plotting",
    }, "analyses.fsle")
    result = {
        "enabled": _boolean(raw.get("enabled", False), "analyses.fsle.enabled"),
        "passage_policy": raw.get("passage_policy", "strict_observed"),
        "maximum_source_gap_minutes": _optional_positive(
            raw.get("maximum_source_gap_minutes"), "analyses.fsle.maximum_source_gap_minutes",
        ),
        "chance_pair_threshold_km": _positive(
            raw.get("chance_pair_threshold_km", 10), "analyses.fsle.chance_pair_threshold_km",
        ),
    }
    if not isinstance(result["passage_policy"], str) or result["passage_policy"] not in PASSAGE_POLICIES:
        raise ValueError(f"analyses.fsle.passage_policy must be one of {sorted(PASSAGE_POLICIES)}")
    for key, default in (
        ("minimum_scale_km", 0.0625), ("maximum_scale_km", 128),
        ("rho", math.sqrt(2)), ("anchor_scale_km", 1),
    ):
        result[key] = _positive(raw.get(key, default), f"analyses.fsle.{key}")
    build_scale_thresholds(
        result["minimum_scale_km"], result["maximum_scale_km"], result["rho"], result["anchor_scale_km"],
    )
    minimum = raw.get("minimum_reached_pairs_per_scale", 3)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("analyses.fsle.minimum_reached_pairs_per_scale must be a positive integer")
    result["minimum_reached_pairs_per_scale"] = minimum
    time = _mapping(raw.get("time", {}), {"start", "end"}, "analyses.fsle.time")
    result["time"] = {key: _utc_string(time.get(key), f"analyses.fsle.time.{key}") for key in ("start", "end")}
    if all(result["time"].values()) and result["time"]["start"] >= result["time"]["end"]:
        raise ValueError("analyses.fsle.time.end must follow start")
    specs = raw.get("fits", [])
    if not isinstance(specs, list):
        raise ValueError("analyses.fsle.fits must be a list")
    fits, ids = [], set()
    for spec in specs:
        spec = _mapping(spec, {"id", "power", "scale_min_km", "scale_max_km"}, "analyses.fsle.fits entry")
        fit_id = spec.get("id")
        if not isinstance(fit_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", fit_id):
            raise ValueError("FSLE fit id must contain only letters, digits, underscores, and hyphens")
        if fit_id in ids:
            raise ValueError(f"Duplicate FSLE fit id: {fit_id}")
        ids.add(fit_id)
        power = spec.get("power")
        if isinstance(power, bool) or not isinstance(power, (int, float)) or not math.isfinite(power):
            raise ValueError("FSLE fit power must be finite and numeric; write -2/3 as -0.6666666666666666")
        lower = _positive(spec.get("scale_min_km"), "FSLE fit scale_min_km")
        upper = _positive(spec.get("scale_max_km"), "FSLE fit scale_max_km")
        if upper <= lower:
            raise ValueError("FSLE fit scale_max_km must exceed scale_min_km")
        fits.append({"id": fit_id, "power": float(power), "scale_min_km": lower, "scale_max_km": upper})
    result["fits"] = fits
    plotting = _mapping(raw.get("plotting", {}), {
        "enabled", "dpi", "standard_error_bars", "x_range_km", "y_range_day_inverse",
    }, "analyses.fsle.plotting")
    dpi = plotting.get("dpi", 150)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("analyses.fsle.plotting.dpi must be a positive integer")
    result["plotting"] = {
        "enabled": _boolean(plotting.get("enabled", True), "analyses.fsle.plotting.enabled"),
        "dpi": dpi,
        "standard_error_bars": _boolean(plotting.get("standard_error_bars", True), "FSLE standard_error_bars"),
        "x_range_km": _axis_range(plotting.get("x_range_km"), "FSLE x_range_km"),
        "y_range_day_inverse": _axis_range(plotting.get("y_range_day_inverse"), "FSLE y_range_day_inverse"),
    }
    return result


def _utc_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    try:
        if not isinstance(value, str) or not value.strip():
            raise ValueError()
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError()
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        return timestamp.isoformat()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be an ISO timestamp or null") from exc


def _axis_range(value: Any, name: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [minimum, maximum] or null")
    lower, upper = [_positive(item, name) for item in value]
    if upper <= lower:
        raise ValueError(f"{name} maximum must exceed minimum")
    return [lower, upper]


def run_fsle_analysis(
    dataset: xr.Dataset,
    settings: dict[str, Any],
    directory: Path,
    *,
    cadence_minutes: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Publish compact pair/shell tables, spectra, fits, and separate figures.

    Only one pair distance vector is constructed at a time. Trajectory positions
    are never copied into pair stores. The enclosing unified workflow handles
    atomic publication and input/product hashes.
    """
    report = progress or (lambda message: None)
    thresholds = build_scale_thresholds(
        settings["minimum_scale_km"], settings["maximum_scale_km"],
        settings["rho"], settings["anchor_scale_km"],
    )
    times = dataset.time.values.astype("datetime64[ns]")
    selected_time = np.ones(len(times), dtype=bool)
    for key, comparison in (("start", np.greater_equal), ("end", np.less)):
        if settings["time"][key] is not None:
            boundary = pd.Timestamp(settings["time"][key]).tz_convert("UTC").tz_localize(None).to_datetime64()
            selected_time &= comparison(times, boundary)
    if not selected_time.any():
        raise ValueError("FSLE time selection contains no trajectory samples")
    dataset = dataset.isel(time=np.flatnonzero(selected_time))
    times = dataset.time.values.astype("datetime64[ns]")
    directory.mkdir(parents=True, exist_ok=True)
    spectra_by_array, inventories = [], []
    pair_counts = {"within_cluster": 0, "cross_cluster": 0}
    for array_id in sorted(set(dataset.array_id.values.astype(int))):
        array = dataset.isel(platform=np.flatnonzero(dataset.array_id.values == array_id))
        platforms = array.platform_id.values.astype(str)
        clusters = array.cluster_id.values.astype(str)
        lon, lat = array.longitude.values.astype(float), array.latitude.values.astype(float)
        source_gap = array.source_gap_minutes.values.astype(float)
        finite = np.isfinite(lon) & np.isfinite(lat)
        supported = finite.copy()
        maximum_gap = settings["maximum_source_gap_minutes"]
        if maximum_gap is not None:
            supported &= np.isfinite(source_gap) & (source_gap <= maximum_gap)
        catalogue, passages = [], []
        order = sorted(range(len(platforms)), key=lambda index: platforms[index])
        report(f"FSLE array {array_id}: examining {len(order) * (len(order) - 1) // 2:,} pairs")
        for pair_number, (first, second) in enumerate(combinations(order, 2), start=1):
            within = clusters[first] == clusters[second]
            population = "within_cluster" if within else "cross_cluster"
            pair_id = sha256(json.dumps([platforms[first], platforms[second]]).encode()).hexdigest()[:24]
            common = np.flatnonzero(finite[first] & finite[second])
            distance = np.full(len(times), np.nan)
            if len(common):
                _, _, metres = WGS84.inv(lon[first, common], lat[first, common], lon[second, common], lat[second, common])
                distance[common] = np.asarray(metres) / 1000
            available = np.flatnonzero(supported[first] & supported[second] & np.isfinite(distance))
            closest = int(available[np.argmin(distance[available])]) if len(available) else None
            selected = closest is not None and (within or distance[closest] <= settings["chance_pair_threshold_km"])
            start = int(available[0]) if within and len(available) else closest
            identity = {
                "pair_id": pair_id, "array_id": int(array_id), "population": population,
                "cluster_id": clusters[first] if within else None,
                "cluster_id_1": clusters[first], "cluster_id_2": clusters[second],
                "platform_code_1": platforms[first], "platform_code_2": platforms[second],
            }
            catalogue.append({
                **identity, "selected": selected,
                "selection_reason": "selected" if selected else "no_valid_overlap" if closest is None else "outside_chance_threshold",
                "minimum_distance_km": float(distance[closest]) if closest is not None else np.nan,
                "minimum_distance_time": times[closest] if closest is not None else np.datetime64("NaT", "ns"),
                "analysis_start_time": times[start] if selected else np.datetime64("NaT", "ns"),
                "analysis_end_time": times[common[-1]] if selected else np.datetime64("NaT", "ns"),
                "start_policy": "first_common_valid_position" if within else "earliest_global_minimum",
            })
            if selected:
                pair_counts[population] += 1
                # Padding after either member's final position is a track end,
                # not an internal missing-data interruption.
                end = int(common[-1]) + 1
                pair_gaps = np.maximum(source_gap[first, start:end], source_gap[second, start:end])
                passages.append(collect_observed_first_passages(
                    times=times[start:end], distance_km=distance[start:end], thresholds_km=thresholds,
                    expected_interval_seconds=cadence_minutes * 60,
                    policy=settings["passage_policy"], source_gap_minutes=pair_gaps,
                    maximum_source_gap_minutes=maximum_gap, **identity,
                ))
            if pair_number % 1000 == 0:
                report(f"FSLE array {array_id}: examined {pair_number:,} pairs")
        # An explicit schema also supports singleton arrays and empty populations.
        if passages:
            passage_table = pd.concat(passages, ignore_index=True)
        else:
            passage_table = collect_observed_first_passages(
                times=times[:1], distance_km=np.asarray([0.0]), thresholds_km=thresholds,
                expected_interval_seconds=cadence_minutes * 60,
                pair_id="", array_id=int(array_id), population="", cluster_id=None,
                cluster_id_1="", cluster_id_2="", platform_code_1="", platform_code_2="",
            ).iloc[:0]
        catalogue_columns = [
            "pair_id", "array_id", "population", "cluster_id", "cluster_id_1", "cluster_id_2",
            "platform_code_1", "platform_code_2", "selected", "selection_reason",
            "minimum_distance_km", "minimum_distance_time", "analysis_start_time", "analysis_end_time", "start_policy",
        ]
        catalogue_table = pd.DataFrame(catalogue, columns=catalogue_columns)
        spectra = aggregate_fsle_populations(
            passage_table, thresholds, array_id=int(array_id), cluster_ids=clusters,
            minimum_reached_pairs=settings["minimum_reached_pairs_per_scale"],
        )
        fits = fit_fsle_power_laws(spectra, settings["fits"])
        target = directory / f"array_{array_id:03d}"
        (target / "data").mkdir(parents=True)
        (target / "tables").mkdir()
        passage_table.to_parquet(target / "data/first_passages.parquet", index=False)
        catalogue_table.to_csv(target / "tables/pair_selection.csv", index=False)
        spectra.to_csv(target / "tables/fsle_spectra.csv", index=False)
        fit_export = fits.copy()
        for key in ("used_scale_lower_km", "used_scale_upper_km"):
            fit_export[key] = fit_export[key].map(json.dumps)
        fit_export.to_csv(target / "tables/power_law_fits.csv", index=False)
        inventories.append({
            "array_id": int(array_id), "cluster_count": len(set(clusters)),
            "possible_pair_count": len(catalogue),
            "selected_within_cluster_pair_count": int(sum(row["selected"] and row["population"] == "within_cluster" for row in catalogue)),
            "selected_cross_cluster_pair_count": int(sum(row["selected"] and row["population"] == "cross_cluster" for row in catalogue)),
            "reached_passage_count": int((passage_table.status == "reached").sum()),
            "fitted_law_count": int((fits.status == "fitted").sum()),
        })
        spectra_by_array.append(spectra)
    spectra = pd.concat(spectra_by_array, ignore_index=True)
    fits = fit_fsle_power_laws(spectra, settings["fits"])
    figure_metadata = {"figure_count": 0}
    if settings["plotting"]["enabled"]:
        from drifterlab.plotting.fsle_analysis import render_fsle_analysis

        report("FSLE: rendering separate spectra and compensated plots with shared axes")
        figure_metadata = render_fsle_analysis(spectra, fits, settings, directory)
    metadata = {
        "algorithm_version": FSLE_ALGORITHM_VERSION,
        "trajectory_product": dataset.attrs.get("selected_trajectory_product"),
        "distance_definition": "WGS84_geodesic", "cadence_minutes": cadence_minutes,
        "search_interval": {"first_sample_utc": str(times[0]), "last_sample_utc": str(times[-1]), "configured_bounds": settings["time"]},
        "thresholds_km": thresholds.tolist(), "configuration": settings,
        "estimator": "mean(log(r_exit/r_entry))/mean(passage_time_days)",
        "lower_entry": "observed_outward_crossing_or_exact_lower_at_start",
        "upper_exit": "first_later_sample_at_or_above_upper",
        "uncertainty": "nominal_overshoot_standard_error_shared_platform_pairs_are_correlated",
        "fit": "equal_shell_weight_least_squares_in_natural_log_space_SI_units",
        "fit_minimum_shell_count": 2,
        "epsilon_conversion": "none; coefficient_cubed_m2_s3_is_a_proxy_only_for_power_minus_two_thirds",
        "arrays": inventories, "figures": figure_metadata,
    }
    (directory / "fsle_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "enabled": True, "algorithm_version": FSLE_ALGORITHM_VERSION,
        "passage_policy": settings["passage_policy"], "array_count": len(inventories),
        "selected_pair_counts": pair_counts, "figure_count": figure_metadata["figure_count"],
        "reached_passage_count": sum(item["reached_passage_count"] for item in inventories),
    }
