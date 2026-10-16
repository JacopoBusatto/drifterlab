"""Spatial pairwise-q diagnostics with a shared array clock and display stride."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from drifterlab.epsilon import pair_kinematics, stable_pair_id
from drifterlab.trajectories.kinematics import (
    ArrayProjection,
    CenteredVelocityResult,
    project_array_positions,
)
from drifterlab.workflows.epsilon import EpsilonWorkflowConfig, _velocity_from_spec

from .config import _boolean, _mapping, _optional_positive, _positive


def resolve_spatial_maps(raw: Any, epsilon: dict[str, Any]) -> dict[str, Any]:
    """Validate the opt-in map block and resolve inherited window definitions."""
    name = "analyses.epsilon.spatial_maps"
    section = _mapping(
        raw, {"enabled", "couples", "windows", "resample_stride", "color", "dpi"}, name,
    )
    couples = _mapping(
        section.get("couples", {}), {"between", "max_initial_distance_m"}, f"{name}.couples",
    )
    between = couples.get("between", "cluster")
    if between not in ("cluster", "array"):
        raise ValueError(f"{name}.couples.between must be cluster or array")
    maximum_distance = _optional_positive(
        couples.get("max_initial_distance_m"), f"{name}.couples.max_initial_distance_m",
    )
    stride = section.get("resample_stride", 1)
    dpi = section.get("dpi", 160)
    for key, value in (("resample_stride", stride), ("dpi", dpi)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name}.{key} must be a positive integer")

    color = _mapping(
        section.get("color", {}),
        {"maximum_absolute_m2_s3", "linear_threshold_m2_s3"}, f"{name}.color",
    )
    distribution = epsilon.get("q_distribution", {})
    maximum = _positive(
        color.get("maximum_absolute_m2_s3", distribution.get("maximum_absolute_m2_s3", 0.005)),
        f"{name}.color.maximum_absolute_m2_s3",
    )
    threshold = _positive(
        color.get("linear_threshold_m2_s3", distribution.get("linear_threshold_m2_s3", 1e-8)),
        f"{name}.color.linear_threshold_m2_s3",
    )
    if threshold >= maximum:
        raise ValueError(f"{name}.color.linear_threshold_m2_s3 must be less than maximum_absolute_m2_s3")

    inherited = epsilon.get("windows", [
        {"id": "initial_0_6h", "start_hour": 0, "end_hour": 6},
        {"id": "initial_0_12h", "start_hour": 0, "end_hour": 12},
    ])
    requested = section.get("windows", "all")
    if requested == "all":
        requested = inherited
    if not isinstance(requested, list) or not requested:
        raise ValueError(f"{name}.windows must be all or a nonempty list of IDs or window definitions")
    lookup = {item["id"]: item for item in inherited}
    windows = []
    for index, item in enumerate(requested):
        if isinstance(item, str):
            if item not in lookup:
                raise ValueError(f"Unknown {name}.windows ID: {item!r}")
            item = lookup[item]
        window_name = f"{name}.windows[{index}]"
        window = _mapping(item, {"id", "start_hour", "end_hour"}, window_name)
        identifier = window.get("id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier)
        ):
            raise ValueError(f"{window_name}.id must be a path-safe nonempty identifier")
        start_raw = window.get("start_hour")
        # Reuse the positive-number checks while permitting a zero start.
        start = 0.0 if type(start_raw) in (int, float) and start_raw == 0 else _positive(
            start_raw, f"{window_name}.start_hour",
        )
        end = _positive(window.get("end_hour"), f"{window_name}.end_hour")
        if end <= start:
            raise ValueError(f"{window_name}.end_hour must exceed start_hour")
        windows.append({"id": identifier, "start_hour": start, "end_hour": end})
    if len({item["id"] for item in windows}) != len(windows):
        raise ValueError(f"{name}.windows contains duplicate IDs")
    return {
        "enabled": _boolean(section.get("enabled", False), f"{name}.enabled"),
        "couples": {"between": between, "max_initial_distance_m": maximum_distance},
        "windows": windows,
        "resample_stride": stride,
        "color": {
            "maximum_absolute_m2_s3": maximum,
            "linear_threshold_m2_s3": threshold,
        },
        "dpi": dpi,
    }


def spatial_pair_observations(
    *,
    platform_ids: np.ndarray,
    cluster_ids: np.ndarray,
    times: np.ndarray,
    projection: ArrayProjection,
    velocity: CenteredVelocityResult,
    reference_index: int,
    sample_steps: int,
    windows: list[dict[str, Any]],
    between: str,
    max_initial_distance_m: float | None,
    minimum_separation_m: float,
    maximum_separation_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select pairs once, then calculate q before any display thinning.

    Cross-cluster distance is measured at the first common finite position,
    independently of velocity support, cadence, and requested windows.
    """
    x, y = projection.x_m, projection.y_m
    finite = np.isfinite(x) & np.isfinite(y)
    indices = np.arange(reference_index, len(times), sample_steps)
    ages = (times - times[reference_index]) / np.timedelta64(1, "h")
    in_windows = np.zeros(len(times), dtype=bool)
    for window in windows:
        in_windows |= (ages >= window["start_hour"]) & (ages < window["end_hour"])
    indices = indices[in_windows[indices]]
    frames, inventory = [], []
    ordered = sorted(range(len(platform_ids)), key=lambda index: str(platform_ids[index]))
    for first, second in combinations(ordered, 2):
        within = cluster_ids[first] == cluster_ids[second]
        if between == "cluster" and not within:
            continue
        pair_id = stable_pair_id(str(platform_ids[first]), str(platform_ids[second]))
        common = np.flatnonzero(finite[first] & finite[second])
        initial_distance = np.nan
        initial_time = None
        if len(common):
            initial_index = int(common[0])
            initial_time = pd.Timestamp(times[initial_index]).isoformat() + "Z"
            initial_distance = float(np.hypot(
                x[second, initial_index] - x[first, initial_index],
                y[second, initial_index] - y[first, initial_index],
            ))
        selected = bool(len(common)) and (
            within or max_initial_distance_m is None or initial_distance <= max_initial_distance_m
        )
        inventory.append({
            "pair_id": pair_id,
            "platform_id_1": platform_ids[first], "platform_id_2": platform_ids[second],
            "source_cluster_id_1": cluster_ids[first], "source_cluster_id_2": cluster_ids[second],
            "pair_cluster_relation": "within_cluster" if within else "between_clusters",
            "initial_time_utc": initial_time, "initial_separation_m": initial_distance,
            "selected": selected,
            "selection_status": (
                "no_common_position" if not len(common)
                else "selected" if selected else "above_maximum_initial_distance"
            ),
        })
        if not selected:
            continue
        valid_indices = indices[velocity.valid[first, indices] & velocity.valid[second, indices]]
        values = pair_kinematics(
            x[first, valid_indices], y[first, valid_indices],
            velocity.u_m_s[first, valid_indices], velocity.v_m_s[first, valid_indices],
            x[second, valid_indices], y[second, valid_indices],
            velocity.u_m_s[second, valid_indices], velocity.v_m_s[second, valid_indices],
        )
        valid = (
            values["valid"] & np.isfinite(values["q_m2_s3"])
            & (values["separation_m"] >= minimum_separation_m)
            & (values["separation_m"] <= maximum_separation_m)
        )
        valid_indices = valid_indices[valid]
        if not len(valid_indices):
            continue
        midpoint_x = (x[first, valid_indices] + x[second, valid_indices]) / 2
        midpoint_y = (y[first, valid_indices] + y[second, valid_indices]) / 2
        midpoint_lon, midpoint_lat = projection.inverse.transform(midpoint_x, midpoint_y)
        frames.append(pd.DataFrame({
            "pair_id": pair_id,
            "platform_id_1": platform_ids[first], "platform_id_2": platform_ids[second],
            "source_cluster_id_1": cluster_ids[first], "source_cluster_id_2": cluster_ids[second],
            "pair_cluster_relation": "within_cluster" if within else "between_clusters",
            "time_utc": pd.to_datetime(times[valid_indices], utc=True),
            "array_age_hours": np.asarray(ages[valid_indices], dtype=float),
            "schedule_index": (valid_indices - reference_index) // sample_steps,
            "midpoint_x_m": midpoint_x, "midpoint_y_m": midpoint_y,
            "midpoint_longitude": midpoint_lon, "midpoint_latitude": midpoint_lat,
            "delta_x_m": values["delta_x_m"][valid], "delta_y_m": values["delta_y_m"][valid],
            "separation_m": values["separation_m"][valid], "q_m2_s3": values["q_m2_s3"][valid],
        }))
    columns = [
        "pair_id", "platform_id_1", "platform_id_2", "source_cluster_id_1", "source_cluster_id_2",
        "pair_cluster_relation", "time_utc", "array_age_hours", "schedule_index", "midpoint_x_m",
        "midpoint_y_m", "midpoint_longitude", "midpoint_latitude", "delta_x_m", "delta_y_m",
        "separation_m", "q_m2_s3",
    ]
    observations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    inventory_columns = [
        "pair_id", "platform_id_1", "platform_id_2", "source_cluster_id_1", "source_cluster_id_2",
        "pair_cluster_relation", "initial_time_utc", "initial_separation_m", "selected", "selection_status",
    ]
    return observations, pd.DataFrame(inventory, columns=inventory_columns)


def _bounds(frame: pd.DataFrame, fallback: tuple[np.ndarray, np.ndarray]) -> tuple[float, ...]:
    x, y = (
        (frame.midpoint_x_m.to_numpy(dtype=float), frame.midpoint_y_m.to_numpy(dtype=float))
        if len(frame) else fallback
    )
    result = []
    for values in (x, y):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)] / 1000
        lower, upper = (float(finite.min()), float(finite.max())) if len(finite) else (-0.5, 0.5)
        padding = max((upper - lower) * 0.05, 0.05)
        result.extend((lower - padding, upper + padding))
    return tuple(result)


def run_spatial_maps(
    dataset: xr.Dataset,
    epsilon: EpsilonWorkflowConfig,
    settings: dict[str, Any],
    output: Path,
    *,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """Write independent spatial observations, maps, and an audit per array."""
    from drifterlab.epsilon_figures import plot_spatial_q

    times = dataset.time.values.astype("datetime64[ns]")
    cadence_minutes = float((times[1] - times[0]) / np.timedelta64(1, "m"))
    sample_steps = round(epsilon.sample_cadence_minutes / cadence_minutes)
    figure_count = observation_count = 0
    for array_id in epsilon.arrays:
        progress(f"Spatial q maps: array {array_id:03d}")
        array = dataset.isel(platform=np.flatnonzero(dataset.array_id.values == array_id))
        longitude = array.longitude.values.astype(float)
        latitude = array.latitude.values.astype(float)
        # Match the epsilon projection even when a subset of clusters is selected.
        projection = project_array_positions(longitude, latitude)
        finite = np.isfinite(longitude) & np.isfinite(latitude)
        reference_index = int(np.flatnonzero(finite.any(axis=0))[0])
        reference = pd.Timestamp(times[reference_index]).tz_localize("UTC")
        gaps = array.source_gap_minutes.values.astype(float)
        velocity = _velocity_from_spec(
            epsilon.analysis, longitude=longitude, latitude=latitude, projection=projection,
            times=times, source_gap_minutes=gaps,
            eastward_velocity=array.eastward_velocity.values.astype(float),
            northward_velocity=array.northward_velocity.values.astype(float),
            velocity_dt_seconds=array.velocity_dt_seconds.values.astype(float),
        )
        cluster_ids = array.cluster_id.values.astype(str)
        members = np.flatnonzero(
            np.ones(len(cluster_ids), dtype=bool) if epsilon.clusters == "all"
            else np.isin(cluster_ids, epsilon.clusters)
        )
        # Keep the array CRS and time origin fixed while restricting candidate pairs.
        selected_projection = replace(projection, x_m=projection.x_m[members], y_m=projection.y_m[members])
        selected_velocity = replace(
            velocity, u_m_s=velocity.u_m_s[members], v_m_s=velocity.v_m_s[members],
            valid=velocity.valid[members], invalid_reason=velocity.invalid_reason[members],
        )
        minimum = max(epsilon.bins.edges_m[0], epsilon.minimum_reliable_separation_m or 0)
        observations, inventory = spatial_pair_observations(
            platform_ids=array.platform_id.values.astype(str)[members],
            cluster_ids=cluster_ids[members], times=times, projection=selected_projection,
            velocity=selected_velocity, reference_index=reference_index, sample_steps=sample_steps,
            windows=settings["windows"], between=settings["couples"]["between"],
            max_initial_distance_m=settings["couples"]["max_initial_distance_m"],
            minimum_separation_m=minimum, maximum_separation_m=epsilon.bins.edges_m[-1],
        )
        observations.insert(0, "array_id", array_id)
        inventory.insert(0, "array_id", array_id)
        observation_count += len(observations)
        directory = output / f"array_{array_id:03d}"
        data = directory / "data"
        data.mkdir(parents=True, exist_ok=True)
        observations.to_parquet(data / "spatial_pair_observations.parquet", index=False)
        inventory.to_csv(directory / "spatial_pair_selection.csv", index=False)
        figures = directory / "figures" / "spatial_q"
        figures.mkdir(parents=True, exist_ok=True)
        populations = [("array", "all_selected_pairs", observations, members)]
        for cluster_id in sorted(set(cluster_ids[members])):
            within = observations[
                (observations.source_cluster_id_1 == cluster_id)
                & (observations.source_cluster_id_2 == cluster_id)
            ]
            populations.append(("cluster", cluster_id, within, members[cluster_ids[members] == cluster_id]))
        records = []
        for scope, identifier, population, population_members in populations:
            bounds = _bounds(population, (projection.x_m[population_members], projection.y_m[population_members]))
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier)
            if scope == "cluster":
                safe_id += "_" + sha256(identifier.encode()).hexdigest()[:8]
            population_directory = figures / ("array" if scope == "array" else f"cluster_{safe_id}")
            for window in settings["windows"]:
                selected = population[
                    (population.array_age_hours >= window["start_hour"])
                    & (population.array_age_hours < window["end_hour"])
                ]
                displayed = selected[selected.schedule_index % settings["resample_stride"] == 0]
                path = population_directory / f"{window['id']}.png"
                start = reference + pd.Timedelta(hours=window["start_hour"])
                end = reference + pd.Timedelta(hours=window["end_hour"])
                plot_spatial_q(
                    displayed, output_path=path, bounds_km=bounds,
                    title=(f"{epsilon.dataset_label} | Array {array_id:03d} | "
                           + ("Whole array" if scope == "array" else identifier)),
                    start_utc=start.isoformat(), end_utc=end.isoformat(),
                    available_count=len(selected), stride=settings["resample_stride"],
                    maximum_absolute_m2_s3=settings["color"]["maximum_absolute_m2_s3"],
                    linear_threshold_m2_s3=settings["color"]["linear_threshold_m2_s3"],
                    dpi=settings["dpi"],
                )
                maximum = settings["color"]["maximum_absolute_m2_s3"]
                records.append({
                    "array_id": array_id, "scope": scope, "population_id": identifier,
                    "window_id": window["id"], "reference_time_utc": reference.isoformat(),
                    "window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(),
                    "available_observations": len(selected), "displayed_observations": len(displayed),
                    "resample_stride": settings["resample_stride"],
                    "negative_saturated_count": int((displayed.q_m2_s3 < -maximum).sum()),
                    "positive_saturated_count": int((displayed.q_m2_s3 > maximum).sum()),
                    "figure": path.relative_to(directory).as_posix(),
                })
                figure_count += 1
        pd.DataFrame(records).to_csv(directory / "spatial_figures.csv", index=False)
        metadata = {
            "settings": settings, "reference_time_utc": reference.isoformat(),
            "time_reference": "first_finite_position_entry_of_array",
            "initial_distance_reference": "first_simultaneous_finite_position_per_pair",
            "sample_cadence_minutes": epsilon.sample_cadence_minutes,
            "minimum_separation_m": minimum, "maximum_separation_m": epsilon.bins.edges_m[-1],
            "cluster_maps": "within_cluster_pairs_only",
            "projection": {
                "crs_definition": projection.crs_definition,
                "origin_longitude": projection.origin_longitude,
                "origin_latitude": projection.origin_latitude,
            },
        }
        (data / "spatial_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"enabled": True, "figure_count": figure_count, "observation_count": observation_count}
