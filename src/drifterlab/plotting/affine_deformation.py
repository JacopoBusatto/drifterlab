"""Matched geographic axis maps and replotting from saved affine diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer


def geographic_extent(extent_m: list | None, metadata: dict, plotting: dict) -> list[float] | None:
    """Return exact displayed [west, east, south, north] bounds in degrees."""
    if extent_m is None or "projection" not in metadata:
        return None
    custom = plotting.get("fixed_extent", True) and any(
        plotting.get(key) is not None for key in ("longitude_bounds", "latitude_bounds")
    )
    if custom:
        extent_m = metadata["extent_m"]
    inverse = Transformer.from_crs(
        metadata["projection"]["crs_definition"], "EPSG:4326", always_xy=True
    )
    x0, x1, y0, y1 = extent_m
    x, y = np.meshgrid(np.linspace(x0, x1, 20), np.linspace(y0, y1, 20))
    lon, lat = inverse.transform(x, y)
    origin = metadata["projection"]["origin_longitude"]
    lon = origin + (np.asarray(lon) - origin + 180) % 360 - 180
    bounds = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    if custom:
        if plotting.get("longitude_bounds") is not None:
            west, east = plotting["longitude_bounds"]
            midpoint = (west + east) / 2
            shift = 360 * np.floor((origin - midpoint + 180) / 360)
            bounds[:2] = [west + float(shift), east + float(shift)]
        if plotting.get("latitude_bounds") is not None:
            bounds[2:] = plotting["latitude_bounds"]
    return bounds


def map_extent(rows: pd.DataFrame, metadata: dict, plotting: dict) -> list[float] | None:
    """Choose a shared view for both panels without using SST availability.

    Adaptive views include every finite, numerically valid initial centroid,
    including flagged fits. Empty frames fall back to the fixed array view.
    A one-kilometre minimum span and six-percent margin keep small frames legible.
    """
    fallback = metadata.get("extent_m")
    if plotting.get("fixed_extent", True) and any(
        plotting.get(key) is not None for key in ("longitude_bounds", "latitude_bounds")
    ):
        bounds = geographic_extent(fallback, metadata, plotting)
        if bounds is None:
            return fallback
        forward = Transformer.from_crs(
            "EPSG:4326", metadata["projection"]["crs_definition"], always_xy=True
        )
        lon, lat = np.meshgrid(np.linspace(*bounds[:2], 41), np.linspace(*bounds[2:], 41))
        x, y = forward.transform(lon, lat)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Configured map bounds cannot be represented in the array projection")
        return [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
    if plotting.get("fixed_extent", True) or rows.empty:
        return fallback
    valid = rows[rows.numerically_valid.fillna(False)]
    if valid.empty:
        return fallback
    centres = valid[["initial_centroid_x_m", "initial_centroid_y_m"]].to_numpy(dtype=float)
    centres = centres[np.isfinite(centres).all(axis=1)]
    if not len(centres):
        return fallback
    low, high = centres.min(axis=0), centres.max(axis=0)
    midpoint = (low + high) / 2
    span = np.maximum(high - low, 1000.0)
    padding = float(span.max()) * 0.06
    if plotting.get("axis_length_km") is not None:
        padding = max(padding, plotting["axis_length_km"] * 1000 * 0.55)
    low, high = midpoint - span / 2 - padding, midpoint + span / 2 + padding
    return [float(low[0]), float(high[0]), float(low[1]), float(high[1])]


def render_maps(
    rows: pd.DataFrame,
    sst: xr.Dataset | None,
    window: dict,
    metadata: dict,
    settings: dict,
    array_id: int,
    path: Path,
) -> None:
    """Transform short metric axis segments back to longitude/latitude.

    Local equirectangular display uses a fixed reference-latitude aspect ratio.
    Eigenvectors belong to the fixed AEQD basis, not to geographic degrees.
    This needs neither Cartopy nor coastline downloads.
    """
    plot = settings["plotting"]
    inverse = Transformer.from_crs(
        metadata["projection"]["crs_definition"], "EPSG:4326", always_xy=True
    )
    origin = metadata["projection"]["origin_longitude"]
    x0, x1, y0, y1 = window.get("extent_m", metadata["extent_m"])

    def wrap(values: Any) -> np.ndarray:
        return origin + (np.asarray(values) - origin + 180) % 360 - 180

    bounds = window.get("geographic_extent_deg") or geographic_extent(
        [x0, x1, y0, y1], metadata, plot
    )
    length = (
        plot["axis_length_km"] * 1000 if plot["axis_length_km"] else max(x1 - x0, y1 - y0) * 0.025
    )
    limit = plot["maximum_absolute_rate_h_inverse"]
    norm = Normalize(-limit, limit)
    latitude_factor = max(np.cos(np.deg2rad(np.mean(bounds[2:]))), 0.05)
    map_ratio = (bounds[3] - bounds[2]) / max((bounds[1] - bounds[0]) * latitude_factor, 1e-9)
    figure_height = min(max(6 * map_ratio + 2.2, 4.5), 9)
    fig, axes = plt.subplots(
        1, 2, figsize=(14, figure_height), layout="compressed", sharex=True, sharey=True
    )
    sst_artist = None
    eligible = rows[rows.numerically_valid.fillna(False)] if len(rows) else rows
    slots = (
        sorted(set(rows.fit_start_utc.map(lambda v: pd.Timestamp(v).strftime("%H:%M"))))
        if len(rows)
        else []
    )
    symbols = ("o", "s", "P", "X", "d", "+")
    # Standard 6/12/24-hour windows need at most four distinct markers. For a
    # finer explicitly requested schedule, retain exact times in labels.
    slot_markers = {slot: symbols[i % len(symbols)] for i, slot in enumerate(slots)}
    saturation = []
    for index, ax in enumerate(axes, start=1):
        ax.set_facecolor("#eef0f2")
        if sst is not None and np.isfinite(sst.sst_celsius.values).any():
            sst_artist = ax.pcolormesh(
                sst.longitude.values,
                sst.latitude.values,
                sst.sst_celsius.values,
                cmap="Greys",
                vmin=settings["sst"]["minimum_celsius"],
                vmax=settings["sst"]["maximum_celsius"],
                shading="auto",
                alpha=0.65,
                rasterized=True,
            )
        segments, rates = [], []
        labelled_clusters = set()
        for _, row in eligible.iterrows():
            flags = set(str(row.quality_flags).split(";")) - {
                "",
                "three_members_no_residual_degrees_of_freedom",
            }
            if plot["show_poor_geometry"]:
                flags.discard("poor_initial_geometry")
            centre = np.array([row.initial_centroid_x_m, row.initial_centroid_y_m])
            if not np.isfinite(centre).all():
                continue
            clon, clat = inverse.transform(*centre)
            displayable = not flags
            hour = pd.Timestamp(row.fit_start_utc).strftime("%H:%M")
            if displayable:
                vector = np.array([row[f"initial_axis_{index}_x"], row[f"initial_axis_{index}_y"]])
                endpoints = centre + np.array([-1, 1])[:, None] * length / 2 * vector
                a, b = inverse.transform(endpoints[:, 0], endpoints[:, 1])
                segments.append(np.column_stack((wrap(a), b)))
                rates.append(row[f"rate_h_inverse_{index}"])
                ax.plot(
                    float(wrap(clon)),
                    clat,
                    marker=slot_markers[hour],
                    color="#333333",
                    ms=2.5,
                    linestyle="",
                    zorder=4,
                )
            elif flags == {"near_isotropic_axes"}:
                ax.plot(float(wrap(clon)), clat, "o", color="#666666", ms=3)
            if plot["label_clusters"] and (displayable or flags == {"near_isotropic_axes"}):
                cluster = str(row.cluster_id).split("cluster_")[-1].lstrip("0") or "0"
                if row.cluster_id not in labelled_clusters or len(slots) > len(symbols):
                    label = f"C{cluster}" + (f" {hour}" if len(slots) > len(symbols) else "")
                    ax.annotate(
                        label,
                        (float(wrap(clon)), clat),
                        xytext=(3, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color="#303030",
                    )
                    labelled_clusters.add(row.cluster_id)
        if plot["connect_centroids"] and len(eligible):
            for _, group in eligible.groupby("cluster_id"):
                group = group.sort_values("fit_start_utc")
                a, b = inverse.transform(
                    group.initial_centroid_x_m.to_numpy(), group.initial_centroid_y_m.to_numpy()
                )
                ax.plot(wrap(a), b, linewidth=0.5, color="#555555", alpha=0.5, zorder=1)
        collection = LineCollection(segments, cmap="coolwarm", norm=norm, linewidths=2.5, zorder=3)
        collection.set_array(np.asarray(rates))
        ax.add_collection(collection)
        saturation.append(sum(abs(v) > limit for v in rates))
        ax.set_title(f"Initial principal axis $e_{index}$ | $\\sigma_{index}$")
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_aspect(1 / latitude_factor)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.set_xlabel("Longitude (°)")
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{(value + 180) % 360 - 180:.2f}")
        )
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Latitude (°)")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"),
        ax=axes.tolist(),
        location="right",
        shrink=0.65,
        extend="both",
        label="Logarithmic stretching rate (h⁻¹)",
    )
    if sst_artist is not None:
        fig.colorbar(
            sst_artist,
            ax=axes.tolist(),
            location="bottom",
            shrink=0.65,
            pad=0.05,
            label="Prepared SST background (°C)",
        )
    axes[0].legend(
        handles=[
            Line2D(
                [],
                [],
                color="#666666",
                marker="o",
                linestyle="",
                markersize=4,
                label="Near-isotropic estimate: direction suppressed",
            )
        ],
        loc="lower left",
        fontsize=7,
    )
    if slots and len(slots) <= len(symbols):
        axes[1].legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color="#333333",
                    marker=slot_markers[slot],
                    linestyle="",
                    markersize=4,
                    label=slot,
                )
                for slot in slots
            ],
            title="Interval start UTC",
            loc="lower right",
            fontsize=7,
            title_fontsize=7,
            ncol=2,
        )
    date = pd.Timestamp(window["map_start_utc"]).strftime("%Y-%m-%d")
    coverage = window["sst"]["coverage_fraction"]
    fig.suptitle(
        f"Array {array_id} | UTC {date} 00:00–24:00 | T′={settings['deformation_window_hours']:g} h\n"
        f"Numerically valid/attempted fits: {window['valid_fit_count']}/{window['attempted_fit_count']} | "
        f"{window['sst']['status']} | regional SST coverage {coverage:.1%}",
        fontsize=12,
    )
    start = window["diagnostic_coverage_start_utc"] or "none"
    end = window["diagnostic_coverage_end_utc"] or "none"
    view_mode = "Fixed" if plot.get("fixed_extent", True) else "Per-frame"
    source_label = window["sst"].get("upstream_metadata", {}).get("title", "Prepared SST")
    fig.supxlabel(
        f"{view_mode} view. Initial locations and directions; segments have fixed length within each frame. Labels: cluster; symbols: interval start UTC.\n"
        f"Valid diagnostic span: {start} to {end}. Saturated axes (e₁/e₂): {saturation[0]}/{saturation[1]}.\n"
        f"{source_label}. SST is contextual, not simultaneous with every fit. Visual correspondence is exploratory.",
        fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=plot["dpi"])
    plt.close(fig)


def replot_array(directory: str | Path) -> int:
    """Replot one published array using saved intervals, settings and SST subsets."""
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if "projection" not in metadata:
        return 0
    rows = pd.read_parquet(root / "data" / "affine_intervals.parquet")
    membership = pd.read_csv(root / "data" / "original_membership.csv")
    count = 0
    for window in metadata["windows"]:
        day = pd.Timestamp(window["map_start_utc"])
        sst_path = root / "sst" / f"{day:%Y%m%d}.nc"
        background = None
        if sst_path.exists():
            with xr.open_dataset(sst_path) as product:
                background = product.load()
        selected = rows[rows.map_start_utc == window["map_start_utc"]] if len(rows) else rows
        render_maps(
            selected,
            background,
            window,
            metadata,
            metadata["settings"],
            int(membership.array_id.iloc[0]),
            root / "replots" / f"{day:%Y%m%d}.png",
        )
        count += 1
    return count
