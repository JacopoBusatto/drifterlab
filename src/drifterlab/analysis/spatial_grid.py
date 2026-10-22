"""Sparse spatial epsilon estimates for fixed geographic daily maps."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drifterlab.epsilon import FOUR_FIFTHS_COEFFICIENT, BinSupportRequirements
from drifterlab.trajectories.kinematics import ArrayProjection

from .config import _boolean, _mapping, _positive


def resolve_daily_maps(raw: Any) -> dict[str, Any]:
    name = "analyses.epsilon.spatial_maps.daily_maps"
    section = _mapping(raw, {
        "window_hours", "separation_edges_m", "cell_size_factor",
        "minimum_cell_size_m", "manual_cell_size_m", "enforce_support",
    }, name)
    edges = section.get("separation_edges_m")
    if edges is not None:
        if not isinstance(edges, list):
            raise ValueError(f"{name}.separation_edges_m must be null or a list")
        edges = [_positive(value, f"{name}.separation_edges_m") for value in edges]
        if any(upper <= lower for lower, upper in pairwise(edges)):
            raise ValueError(f"{name}.separation_edges_m must increase strictly")
        edges = edges or None
    manual = section.get("manual_cell_size_m")
    if section.get("minimum_cell_size_m") is not None:
        raise ValueError(f"{name}.minimum_cell_size_m is no longer used; choose manual_cell_size_m instead")
    if manual is not None:
        if section.get("cell_size_factor") is not None:
            raise ValueError(f"{name}.manual_cell_size_m is mutually exclusive with automatic sizing")
        if isinstance(manual, list):
            if len(manual) != len(edges or []) + 1:
                raise ValueError(f"{name}.manual_cell_size_m needs one size per separation bin")
            manual = [_positive(value, f"{name}.manual_cell_size_m") for value in manual]
        else:
            manual = _positive(manual, f"{name}.manual_cell_size_m")
    elif edges is None:
        raise ValueError(f"{name}.manual_cell_size_m is required without separation edges")
    return {
        "window_hours": _positive(section.get("window_hours", 24), f"{name}.window_hours"),
        "separation_edges_m": edges,
        "cell_size_factor": (
            _positive(section.get("cell_size_factor", 1), f"{name}.cell_size_factor")
            if manual is None else None
        ),
        "manual_cell_size_m": manual,
        "enforce_support": _boolean(section.get("enforce_support", True), f"{name}.enforce_support"),
    }


def daily_windows(windows: list[dict[str, Any]], hours: float) -> list[dict[str, Any]]:
    """Split the configured overall time span into consecutive half-open windows."""
    start = min(item["start_hour"] for item in windows)
    end = max(item["end_hour"] for item in windows)
    ratio = (end - start) / hours
    count = round(ratio) if math.isclose(ratio, round(ratio), abs_tol=1e-12) else math.ceil(ratio)
    return [
        {"id": f"day_{index + 1:03d}", "start_hour": start + index * hours,
         "end_hour": min(start + (index + 1) * hours, end)}
        for index in range(max(count, 1))
    ]


@dataclass(frozen=True)
class SpatialBand:
    index: int
    lower_m: float
    upper_m: float | None
    cell_size_m: float | None
    sizing_reference_m: float | None

    @property
    def label(self) -> str:
        if self.lower_m == 0 and self.upper_m is None:
            return "All valid separations"
        if self.lower_m == 0:
            return f"0 < separation < {self.upper_m:g} m"
        if self.upper_m is None:
            return f"Separation >= {self.lower_m:g} m"
        return f"{self.lower_m:g} <= separation < {self.upper_m:g} m"


def resolve_bands(observations: pd.DataFrame, settings: dict[str, Any]) -> list[SpatialBand]:
    """Use c times each lower edge, with the first positive edge for the first bin."""
    edges = settings["separation_edges_m"] or []
    bounds = [0.0, *edges, None]
    manual = settings["manual_cell_size_m"]
    result = []
    for index, (lower, upper) in enumerate(pairwise(bounds)):
        reference = lower if lower > 0 else upper
        if manual is not None:
            size = manual[index] if isinstance(manual, list) else manual
        elif reference is None:
            raise ValueError("Comprehensive spatial maps require manual_cell_size_m")
        else:
            size = settings["cell_size_factor"] * reference
        result.append(SpatialBand(index, lower, upper, size, reference))
    return result


CELL_COLUMNS = [
    "window_id", "separation_bin_index", "cell_x_index", "cell_y_index", "cell_size_m",
    "x_lower_m", "x_upper_m", "y_lower_m", "y_upper_m", "center_longitude", "center_latitude",
    "observation_count", "unique_pair_count", "platform_count", "unique_utc_count", "cluster_count",
    "dominant_cluster_fraction", "separation_sum_m", "longitudinal_cube_sum_m3_s3",
    "mean_separation_m", "epsilon_eff_m2_s3", "support_ok", "support_status", "display_ok",
]
CONTRIBUTION_COLUMNS = [
    "window_id", "separation_bin_index", "cell_x_index", "cell_y_index",
    "cluster_id", "observation_weight", "observation_fraction",
]


def estimate_spatial_cells(
    observations: pd.DataFrame,
    *,
    windows: list[dict[str, Any]],
    bands: list[SpatialBand],
    support: BinSupportRequirements,
    projection: ArrayProjection,
    enforce_support: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate the ratio of sums; optionally mask cells below support thresholds.

    Empty cells have no rows. No display stride, interpolation, or smoothing is
    involved. Cross-cluster observations attribute half their support to each
    endpoint's cluster, without duplicating their contribution to epsilon.
    """
    records, contributions = [], []
    for window in windows:
        day = observations[
            (observations.array_age_hours >= window["start_hour"])
            & (observations.array_age_hours < window["end_hour"])
        ]
        for band in bands:
            if band.cell_size_m is None:
                continue
            frame = day[day.separation_m >= band.lower_m]
            if band.upper_m is not None:
                frame = frame[frame.separation_m < band.upper_m]
            if frame.empty:
                continue
            frame = frame.copy()
            frame["cell_x"] = np.floor(frame.midpoint_x_m / band.cell_size_m).astype(np.int64)
            frame["cell_y"] = np.floor(frame.midpoint_y_m / band.cell_size_m).astype(np.int64)
            for (cell_x, cell_y), group in frame.groupby(["cell_x", "cell_y"], sort=True):
                identity = {
                    "window_id": window["id"], "separation_bin_index": band.index,
                    "cell_x_index": int(cell_x), "cell_y_index": int(cell_y),
                }
                platforms = pd.concat([group.platform_id_1, group.platform_id_2]).nunique()
                clusters = pd.concat([group.source_cluster_id_1, group.source_cluster_id_2]).value_counts() / 2
                count, pairs, utc = len(group), group.pair_id.nunique(), group.time_utc.nunique()
                reasons = [
                    label for condition, label in (
                        (count < support.observations, "observations"),
                        (pairs < support.unique_pairs, "unique_pairs"),
                        (platforms < support.platforms, "platforms"),
                        (utc < support.unique_utc, "unique_utc"),
                    ) if condition
                ]
                separation_sum = float(group.separation_m.sum())
                cube_sum = float(group.delta_u_l_cube_m3_s3.sum())
                estimate = FOUR_FIFTHS_COEFFICIENT * cube_sum / separation_sum
                finite_estimate = math.isfinite(estimate)
                if not finite_estimate:
                    reasons.append("nonfinite_estimate")
                display_ok = finite_estimate and (not enforce_support or not reasons)
                size = band.cell_size_m
                center_lon, center_lat = projection.inverse.transform((cell_x + 0.5) * size, (cell_y + 0.5) * size)
                records.append({
                    **identity, "cell_size_m": size,
                    "x_lower_m": cell_x * size, "x_upper_m": (cell_x + 1) * size,
                    "y_lower_m": cell_y * size, "y_upper_m": (cell_y + 1) * size,
                    "center_longitude": center_lon, "center_latitude": center_lat,
                    "observation_count": count, "unique_pair_count": pairs,
                    "platform_count": platforms, "unique_utc_count": utc, "cluster_count": len(clusters),
                    "dominant_cluster_fraction": float(clusters.max() / count),
                    "separation_sum_m": separation_sum, "longitudinal_cube_sum_m3_s3": cube_sum,
                    "mean_separation_m": separation_sum / count,
                    "epsilon_eff_m2_s3": estimate if display_ok else np.nan,
                    "support_ok": not reasons,
                    "support_status": "supported" if not reasons else "below_minimum:" + ",".join(reasons),
                    "display_ok": display_ok,
                })
                contributions.extend({
                    **identity, "cluster_id": str(cluster_id), "observation_weight": float(weight),
                    "observation_fraction": float(weight / count),
                } for cluster_id, weight in clusters.items())
    return pd.DataFrame(records, columns=CELL_COLUMNS), pd.DataFrame(contributions, columns=CONTRIBUTION_COLUMNS)


def cell_polygons(cells: pd.DataFrame, projection: ArrayProjection) -> np.ndarray:
    """Transform occupied square cells into geographic polygon vertices."""
    if cells.empty:
        return np.empty((0, 4, 2))
    x = np.column_stack([cells.x_lower_m, cells.x_upper_m, cells.x_upper_m, cells.x_lower_m])
    y = np.column_stack([cells.y_lower_m, cells.y_lower_m, cells.y_upper_m, cells.y_upper_m])
    lon, lat = projection.inverse.transform(x, y)
    lon = projection.origin_longitude + (np.asarray(lon) - projection.origin_longitude + 180) % 360 - 180
    return np.stack([lon, lat], axis=-1)


def fixed_geographic_extent(
    cells: pd.DataFrame,
    projection: ArrayProjection,
    fallback: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float, float, float]:
    # Cell size must not enlarge the region and hide the finer bands. Use the
    # selected period's platform positions; polygons are clipped to this viewport.
    lon, lat = projection.inverse.transform(*fallback)
    if not np.isfinite(lon).any() or not np.isfinite(lat).any():
        polygons = cell_polygons(cells, projection)
        lon, lat = polygons[:, :, 0].ravel(), polygons[:, :, 1].ravel()
    lon = projection.origin_longitude + (np.asarray(lon) - projection.origin_longitude + 180) % 360 - 180
    result = []
    for values in (lon, lat):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError("Daily spatial maps require a finite geographic extent")
        lower, upper = float(finite.min()), float(finite.max())
        pad = max((upper - lower) * 0.05, 0.01)
        result.extend([lower - pad, upper + pad])
    result[2] = max(result[2], -89.99)
    result[3] = min(result[3], 89.99)
    return tuple(result)


def write_daily_maps(
    observations: pd.DataFrame,
    *,
    projection: ArrayProjection,
    reference: pd.Timestamp,
    windows: list[dict[str, Any]],
    settings: dict[str, Any],
    support: BinSupportRequirements,
    directory: Path,
    array_id: int,
    dataset_label: str,
    fallback: tuple[np.ndarray, np.ndarray],
    minimum_reliable_separation_m: float | None,
    sample_cadence_minutes: int,
    progress: Callable[[str], None],
) -> int:
    from drifterlab.plotting.spatial_epsilon import plot_daily_epsilon

    bands = resolve_bands(observations, settings["daily_maps"])
    cells, contributions = estimate_spatial_cells(
        observations, windows=windows, bands=bands, support=support, projection=projection,
        enforce_support=settings["daily_maps"]["enforce_support"],
    )
    cells.insert(0, "array_id", array_id)
    contributions.insert(0, "array_id", array_id)
    extent = fixed_geographic_extent(cells, projection, fallback)
    data = directory / "data"
    cells.to_parquet(data / "spatial_epsilon_cells.parquet", index=False)
    contributions.to_parquet(data / "spatial_cluster_contributions.parquet", index=False)
    records = []
    for day_index, window in enumerate(windows, start=1):
        progress(f"Array {array_id:03d}: daily epsilon map {day_index}/{len(windows)}")
        start = reference + pd.Timedelta(hours=window["start_hour"])
        end = reference + pd.Timedelta(hours=window["end_hour"])
        selected = observations[
            (observations.array_age_hours >= window["start_hour"])
            & (observations.array_age_hours < window["end_hour"])
        ]
        day_cells = cells[cells.window_id == window["id"]]
        path = directory / "figures" / "spatial_epsilon" / f"{window['id']}.png"
        plot_daily_epsilon(
            day_cells, observations=selected, bands=bands, projection=projection,
            extent=extent, title=f"{dataset_label} | Array {array_id:03d} | Day {day_index:02d}",
            start_utc=start.isoformat(), end_utc=end.isoformat(), output_path=path,
            color=settings["color"], dpi=settings["dpi"], support=support,
            enforce_support=settings["daily_maps"]["enforce_support"],
        )
        for band in bands:
            band_cells = day_cells[day_cells.separation_bin_index == band.index]
            supported = band_cells[band_cells.support_ok.astype(bool)]
            displayed = band_cells[band_cells.display_ok.astype(bool)]
            maximum = settings["color"]["maximum_absolute_m2_s3"]
            records.append({
                "array_id": array_id, "window_id": window["id"],
                "window_start_hour": window["start_hour"], "window_end_hour": window["end_hour"],
                "reference_time_utc": reference.isoformat(), "window_start_utc": start.isoformat(),
                "window_end_utc": end.isoformat(), "separation_bin_index": band.index,
                "separation_lower_m": band.lower_m, "separation_upper_m": band.upper_m,
                "cell_size_m": band.cell_size_m,
                "observation_count": int(band_cells.observation_count.sum()),
                "occupied_cell_count": len(band_cells), "supported_cell_count": len(supported),
                "displayed_cell_count": len(displayed),
                "support_enforced": settings["daily_maps"]["enforce_support"],
                "below_observations_cell_count": int((band_cells.observation_count < support.observations).sum()),
                "below_unique_pairs_cell_count": int((band_cells.unique_pair_count < support.unique_pairs).sum()),
                "below_platforms_cell_count": int((band_cells.platform_count < support.platforms).sum()),
                "below_unique_utc_cell_count": int((band_cells.unique_utc_count < support.unique_utc).sum()),
                "maximum_unique_utc_per_cell": int(band_cells.unique_utc_count.max()) if len(band_cells) else 0,
                "negative_saturated_count": int((displayed.epsilon_eff_m2_s3 < -maximum).sum()),
                "positive_saturated_count": int((displayed.epsilon_eff_m2_s3 > maximum).sum()),
                "figure": path.relative_to(directory).as_posix(),
            })
    pd.DataFrame(records).to_csv(directory / "spatial_daily_figures.csv", index=False)
    metadata = {
        "mode": "daily_spatial_epsilon", "settings": settings,
        "reference_time_utc": reference.isoformat(),
        "time_reference": "first_finite_position_entry_of_array", "daily_windows": windows,
        "bands": [asdict(band) for band in bands], "support": asdict(support),
        "support_enforced": settings["daily_maps"]["enforce_support"],
        "automatic_cell_sizing": "factor * lower edge; first bin uses its upper edge",
        "minimum_reliable_separation_m": minimum_reliable_separation_m,
        "sample_cadence_minutes": sample_cadence_minutes,
        "initial_distance_reference": "first_simultaneous_finite_position_per_pair",
        "estimator": "(-5/4) * sum(delta_u_l_cube_m3_s3) / sum(separation_m)",
        "interpretation": "signed effective transfer conditional on sampled region, period, and separation band",
        "sampling": "all valid scheduled pair observations; display stride is not applied",
        "empty_cells": "absent from sparse table and blank on maps",
        "unsupported_cells": (
            "saved with epsilon_eff_m2_s3=NaN and blank on maps"
            if settings["daily_maps"]["enforce_support"]
            else "finite estimates displayed; support_ok and support_status retain screening results"
        ),
        "cluster_attribution": "one half observation weight per endpoint cluster",
        "interpolation": "none", "smoothing": "none", "contours": False,
        "grid_origin_projected_m": [0, 0], "fixed_geographic_extent": extent,
        "geographic_extent_reference": "platform positions in selected period; independent of cell size",
        "projection": {
            "crs_definition": projection.crs_definition,
            "origin_longitude": projection.origin_longitude,
            "origin_latitude": projection.origin_latitude,
        },
    }
    (data / "spatial_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return len(windows)
