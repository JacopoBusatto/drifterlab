"""Static maps and streamed MP4 movies for one experimental drifter array."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Callable
from uuid import uuid4

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import animation, colormaps
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pyproj import Geod

from drifterlab.plotting.projections import get_projection
from drifterlab.plotting.trajectories import split_longitude_wrapped_path
from drifterlab.postprocessing.config import MapConfig, TrajectoryPlottingConfig
from drifterlab.postprocessing.trajectories import ArrayTrajectoryData


_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class ColorEncoding:
    mode: str
    label: str
    rgba: np.ndarray
    cmap: Any
    norm: Any
    categories: tuple[str, ...]
    category_colors: tuple[Any, ...]


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def _category_label(value: str) -> str:
    match = re.fullmatch(r"array_[0-9]+__cluster_([0-9]+)", value)
    return f"Cluster {int(match.group(1))}" if match else value


def build_color_encoding(
    values: np.ndarray, *, name: str, mode: str, cmap_name: str,
    label: str | None, vmin: float | None, vmax: float | None,
) -> ColorEncoding:
    """Resolve one stable categorical or numeric color assignment for an array."""
    raw = np.asarray(values)
    identity_name = name == "platform_id" or name.endswith("_id")
    if mode == "auto":
        mode = "categorical" if identity_name or not np.issubdtype(raw.dtype, np.number) else "numeric"
    try:
        cmap = colormaps[cmap_name]
    except KeyError as exc:
        raise ValueError(f"Unknown Matplotlib colormap {cmap_name!r}") from exc

    display_label = label or name
    if mode == "categorical":
        strings = np.asarray(
            [None if pd.isna(value) else str(value) for value in raw], dtype=object,
        )
        categories = tuple(sorted({value for value in strings if value is not None}, key=_natural_key))
        if not categories:
            raise ValueError(f"Color variable {name!r} has no valid values")
        samples = np.linspace(0, 1, len(categories), endpoint=True) if len(categories) > 1 else [0.5]
        category_colors = tuple(cmap(float(sample)) for sample in samples)
        lookup = dict(zip(categories, category_colors))
        rgba = np.asarray([
            (0.55, 0.55, 0.55, 1.0) if value is None else lookup[value]
            for value in strings
        ])
        listed = mcolors.ListedColormap(category_colors, name=f"{cmap_name}_categories")
        norm = mcolors.BoundaryNorm(np.arange(len(categories) + 1) - 0.5, len(categories))
        return ColorEncoding(
            mode, display_label, rgba, listed, norm, categories, category_colors,
        )

    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Color variable {name!r} cannot be represented numerically") from exc
    finite = np.isfinite(numeric)
    if not finite.any():
        raise ValueError(f"Color variable {name!r} has no finite numeric values")
    lower = float(np.nanmin(numeric)) if vmin is None else vmin
    upper = float(np.nanmax(numeric)) if vmax is None else vmax
    if upper < lower:
        raise ValueError("trajectory_plotting.vmax must be greater than or equal to vmin")
    if np.isclose(lower, upper):
        upper = lower + 1.0
    norm = mcolors.Normalize(vmin=lower, vmax=upper, clip=False)
    rgba = np.empty((len(numeric), 4), dtype=float)
    rgba[:] = (0.55, 0.55, 0.55, 1.0)
    rgba[finite] = cmap(norm(numeric[finite]))
    return ColorEncoding(mode, display_label, rgba, cmap, norm, (), ())


def resolve_map_extent(
    data: ArrayTrajectoryData, map_config: MapConfig,
) -> tuple[float, float, float, float]:
    """Return one fixed explicit or full-window auto extent for an array."""
    if data.configured_extent is not None:
        return data.configured_extent
    finite = np.isfinite(data.longitude) & np.isfinite(data.latitude)
    longitude = data.longitude[finite]
    latitude = data.latitude[finite]
    if len(longitude) == 0:
        raise ValueError(f"Array {data.array_id} has no finite coordinates for an extent")
    west = float(np.min(longitude))
    east = float(np.max(longitude))
    south = float(np.min(latitude))
    north = float(np.max(latitude))
    if east - west > 180:
        raise ValueError(
            f"Array {data.array_id} crosses the longitude wrap; configure an explicit map extent"
        )
    lon_padding = max(0.05, (east - west) * map_config.padding_fraction)
    lat_padding = max(0.05, (north - south) * map_config.padding_fraction)
    return (
        west - lon_padding,
        east + lon_padding,
        max(-90.0, south - lat_padding),
        min(90.0, north + lat_padding),
    )


def _new_map(
    figure: Any, map_config: MapConfig, extent: tuple[float, float, float, float],
) -> Any:
    axes = figure.add_subplot(
        1, 1, 1,
        projection=get_projection(
            map_config.projection, central_longitude=map_config.central_longitude,
        ),
    )
    if map_config.land:
        axes.add_feature(cfeature.LAND.with_scale("110m"), facecolor="0.92", zorder=0)
    if map_config.coastlines:
        axes.coastlines(resolution="110m", linewidth=0.7, color="0.3")
    if map_config.gridlines:
        lines = axes.gridlines(draw_labels=True, linestyle="--", alpha=0.35, linewidth=0.6)
        lines.top_labels = False
        lines.right_labels = False
    axes.set_extent(extent, crs=ccrs.PlateCarree())
    if map_config.scale_bar:
        _add_scale_bar(axes, extent)
    return axes


def _nice_scale_length(target_km: float) -> float:
    if not math.isfinite(target_km) or target_km <= 0:
        return 1.0
    exponent = math.floor(math.log10(target_km))
    factor = 10.0 ** exponent
    candidates = [1.0 * factor, 2.0 * factor, 5.0 * factor, 10.0 * factor]
    return max((value for value in candidates if value <= target_km), default=candidates[0])


def _add_scale_bar(axes: Any, extent: tuple[float, float, float, float]) -> None:
    west, east, south, north = extent
    longitude = west + 0.08 * (east - west)
    latitude = south + 0.08 * (north - south)
    _, _, width_m = _GEOD.inv(west, latitude, east, latitude)
    length_km = _nice_scale_length(abs(width_m) / 1000.0 * 0.20)
    end_lon, end_lat, _ = _GEOD.fwd(longitude, latitude, 90.0, length_km * 1000.0)
    transform = ccrs.PlateCarree()
    axes.plot(
        [longitude, end_lon], [latitude, end_lat], transform=transform,
        color="black", linewidth=2.5, solid_capstyle="butt", zorder=8,
    )
    cap = max(0.002, 0.012 * (north - south))
    for current_lon, current_lat in ((longitude, latitude), (end_lon, end_lat)):
        axes.plot(
            [current_lon, current_lon], [current_lat - cap, current_lat + cap],
            transform=transform, color="black", linewidth=1.4, zorder=8,
        )
    axes.text(
        (longitude + end_lon) / 2,
        (latitude + end_lat) / 2 + 1.7 * cap,
        f"{length_km:g} km",
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
        zorder=9,
    )


def _color_key(figure: Any, axes: Any, encoding: ColorEncoding) -> None:
    if encoding.mode == "numeric":
        colorbar = figure.colorbar(
            ScalarMappable(norm=encoding.norm, cmap=encoding.cmap),
            ax=axes, pad=0.03, shrink=0.82,
        )
        colorbar.set_label(encoding.label)
        return
    if len(encoding.categories) <= 20:
        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=6,
                markerfacecolor=color, markeredgecolor="black", markeredgewidth=0.3,
                label=_category_label(category),
            )
            for category, color in zip(encoding.categories, encoding.category_colors)
        ]
        axes.legend(
            handles=handles, title=encoding.label, loc="upper left",
            bbox_to_anchor=(1.01, 1.0), borderaxespad=0, frameon=True, fontsize=8,
        )
    else:
        axes.text(
            1.01, 1.0, f"Color: {encoding.label}\n{len(encoding.categories)} categories",
            transform=axes.transAxes, ha="left", va="top", fontsize=8,
        )


def _format_utc(value: np.datetime64) -> str:
    return pd.Timestamp(value, tz="UTC").strftime("%Y-%m-%d %H:%M UTC")


def _format_elapsed(current: np.datetime64, reference: np.datetime64) -> str:
    total_minutes = int(round(float((current - reference) / np.timedelta64(1, "m"))))
    sign = "+" if total_minutes >= 0 else "-"
    remaining = abs(total_minutes)
    days, remaining = divmod(remaining, 24 * 60)
    hours, minutes = divmod(remaining, 60)
    return f"T{sign}{days:d}d {hours:02d}h {minutes:02d}m"


def _reference_label(data: ArrayTrajectoryData) -> str:
    return (
        "Elapsed since nominal deployment"
        if data.reference_kind == "nominal_deployment"
        else "Elapsed since first retained fix"
    )


def _save_figure(figure: Any, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")
    try:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight")
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        plt.close(figure)


def render_array_trajectory_figure(
    data: ArrayTrajectoryData,
    config: TrajectoryPlottingConfig,
    path: str | Path,
) -> Path:
    """Render the complete configured trajectory window for one array."""
    path = Path(path)
    encoding = build_color_encoding(
        data.color_values,
        name=data.color_by,
        mode=config.color_mode,
        cmap_name=config.cmap,
        label=config.color_label,
        vmin=config.vmin,
        vmax=config.vmax,
    )
    extent = resolve_map_extent(data, config.map)
    figure = plt.figure(figsize=config.map.figsize)
    axes = _new_map(figure, config.map, extent)
    transform = ccrs.PlateCarree()
    for index in range(len(data.platform_ids)):
        longitude = data.longitude[index]
        latitude = data.latitude[index]
        for segment_lon, segment_lat in split_longitude_wrapped_path(longitude, latitude):
            axes.plot(
                segment_lon, segment_lat, transform=transform,
                color=encoding.rgba[index], linewidth=1.0, alpha=0.75, zorder=2,
            )
        finite = np.flatnonzero(np.isfinite(longitude) & np.isfinite(latitude))
        if len(finite):
            first, last = int(finite[0]), int(finite[-1])
            axes.scatter(
                longitude[first], latitude[first], transform=transform,
                marker="o", s=24, facecolor=encoding.rgba[index], edgecolor="black",
                linewidth=0.4, zorder=4,
            )
            axes.scatter(
                longitude[last], latitude[last], transform=transform,
                marker="x", s=25, color=encoding.rgba[index], linewidth=0.8, zorder=4,
            )
    _color_key(figure, axes, encoding)
    axes.set_title(
        f"{data.dataset_label} — Array {data.array_id}\n"
        f"{_format_utc(data.times[0])} to {_format_utc(data.times[-1])} · "
        f"{data.coordinate_method}"
    )
    axes.text(
        0.01, 0.01,
        f"○ first position in window   × last position in window\n"
        f"Time reference: {_format_utc(data.reference_time)} "
        f"({'nominal deployment' if data.reference_kind == 'nominal_deployment' else 'first retained fix'})",
        transform=axes.transAxes, ha="left", va="bottom", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.82, "pad": 3},
        zorder=10,
    )
    figure.subplots_adjust(left=0.07, right=0.79, bottom=0.08, top=0.90)
    _save_figure(figure, path, dpi=config.map.dpi)
    return path


def movie_frame_indices(times: np.ndarray, frame_interval_minutes: int) -> np.ndarray:
    """Select regular source-grid frames and always include the configured final time."""
    values = np.asarray(times).astype("datetime64[ns]")
    if len(values) < 2:
        raise ValueError("A trajectory movie requires at least two source-grid times")
    differences = np.diff(values.astype(np.int64))
    if np.any(differences <= 0) or not np.all(differences == differences[0]):
        raise ValueError("Trajectory movie input time must use a regular increasing grid")
    requested_ns = frame_interval_minutes * 60_000_000_000
    source_ns = int(differences[0])
    if requested_ns < source_ns or requested_ns % source_ns:
        source_minutes = source_ns / 60_000_000_000
        raise ValueError(
            "trajectory_plotting.movie.frame_interval_minutes must be an integer multiple "
            f"of the {source_minutes:g}-minute source grid"
        )
    step = requested_ns // source_ns
    indices = np.arange(0, len(values), step, dtype=int)
    if indices[-1] != len(values) - 1:
        indices = np.append(indices, len(values) - 1)
    return indices


def _draw_movie_frame(
    figure: Any,
    data: ArrayTrajectoryData,
    config: TrajectoryPlottingConfig,
    encoding: ColorEncoding,
    extent: tuple[float, float, float, float],
    frame_index: int,
) -> None:
    figure.clear()
    axes = _new_map(figure, config.map, extent)
    transform = ccrs.PlateCarree()
    current_time = data.times[frame_index]
    tail_start = current_time - np.timedelta64(
        int(round(config.movie.tail_hours * 3_600_000_000_000)), "ns",
    )
    tail_mask = (data.times >= tail_start) & (data.times <= current_time)
    tail_indices = np.flatnonzero(tail_mask)
    for platform_index in range(len(data.platform_ids)):
        longitude = data.longitude[platform_index, tail_indices]
        latitude = data.latitude[platform_index, tail_indices]
        for segment_lon, segment_lat in split_longitude_wrapped_path(longitude, latitude):
            axes.plot(
                segment_lon, segment_lat, transform=transform,
                color=encoding.rgba[platform_index], linewidth=1.0,
                alpha=0.55, zorder=2,
            )
    active = (
        np.isfinite(data.longitude[:, frame_index])
        & np.isfinite(data.latitude[:, frame_index])
    )
    if active.any():
        axes.scatter(
            data.longitude[active, frame_index],
            data.latitude[active, frame_index],
            transform=transform,
            s=30,
            c=encoding.rgba[active],
            edgecolor="black",
            linewidth=0.35,
            zorder=4,
        )
    _color_key(figure, axes, encoding)
    axes.set_title(f"{data.dataset_label} — Array {data.array_id} · {data.coordinate_method}")
    axes.text(
        0.01, 0.99,
        f"{_format_utc(current_time)}\n"
        f"{_reference_label(data)}: {_format_elapsed(current_time, data.reference_time)}\n"
        f"Active: {int(active.sum())}/{len(active)}",
        transform=axes.transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "pad": 3},
        zorder=10,
    )
    figure.subplots_adjust(left=0.07, right=0.79, bottom=0.08, top=0.91)


def render_array_trajectory_movie(
    data: ArrayTrajectoryData,
    config: TrajectoryPlottingConfig,
    path: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Stream one array movie directly to an H.264 MP4 file."""
    if not animation.writers.is_available("ffmpeg"):
        raise ValueError(
            "MP4 rendering requires FFmpeg on PATH; install FFmpeg and rerun postprocessing"
        )
    report = progress or (lambda message: None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = build_color_encoding(
        data.color_values,
        name=data.color_by,
        mode=config.color_mode,
        cmap_name=config.cmap,
        label=config.color_label,
        vmin=config.vmin,
        vmax=config.vmax,
    )
    extent = resolve_map_extent(data, config.map)
    indices = movie_frame_indices(data.times, config.movie.frame_interval_minutes)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")
    figure = plt.figure(figsize=config.map.figsize)
    writer = animation.FFMpegWriter(
        fps=config.movie.fps,
        codec="h264",
        metadata={
            "title": f"{data.dataset_label} Array {data.array_id} trajectories",
            "artist": "drifterlab",
        },
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        with writer.saving(figure, str(temporary), dpi=config.movie.dpi):
            for output_index, frame_index in enumerate(indices):
                _draw_movie_frame(
                    figure, data, config, encoding, extent, int(frame_index),
                )
                writer.grab_frame()
                if (
                    output_index == 0
                    or output_index + 1 == len(indices)
                    or (output_index + 1) % max(1, len(indices) // 10) == 0
                ):
                    report(
                        f"Array {data.array_id}: rendered movie frame "
                        f"{output_index + 1}/{len(indices)}"
                    )
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        plt.close(figure)
    return path
