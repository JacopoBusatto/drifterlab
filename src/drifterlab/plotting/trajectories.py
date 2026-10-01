"""Batch figures for one or more lean reconstructed-trajectory Zarr stores."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable
from uuid import uuid4

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr

from drifterlab.plotting.projections import get_projection


@dataclass(frozen=True)
class _Dataset:
    label: str
    path: Path
    selected_method: str
    dataset: xr.Dataset
    platform_ids: tuple[str, ...]
    platform_index: dict[str, int]
    methods: dict[str, tuple[str, str]]
    array_ids: tuple[int, ...] | None
    cluster_ids: tuple[str, ...] | None


CLUSTER_MARKERS = (
    "o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "8", "*", "H", "d",
)


def split_longitude_wrapped_path(
    longitude: np.ndarray, latitude: np.ndarray, *, max_lon_step: float = 180,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return finite contiguous line segments without drawing across a wrap."""
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    if lon.ndim != 1 or lat.ndim != 1 or lon.shape != lat.shape:
        raise ValueError("longitude and latitude must be equally sized one-dimensional arrays")
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start: int | None = None
    for index in range(len(lon)):
        finite = np.isfinite(lon[index]) and np.isfinite(lat[index])
        wrapped = (
            index > 0 and finite and np.isfinite(lon[index - 1])
            and np.isfinite(lat[index - 1])
            and abs(lon[index] - lon[index - 1]) > max_lon_step
        )
        if not finite or wrapped:
            if start is not None and index - start >= 2:
                segments.append((lon[start:index].copy(), lat[start:index].copy()))
            start = index if finite else None
        elif start is None:
            start = index
    if start is not None and len(lon) - start >= 2:
        segments.append((lon[start:].copy(), lat[start:].copy()))
    return segments


def _available_methods(dataset: xr.Dataset) -> dict[str, tuple[str, str]]:
    methods: dict[str, tuple[str, str]] = {}
    if {"longitude", "latitude"} <= set(dataset.data_vars):
        methods["native"] = ("longitude", "latitude")
    for name in dataset.data_vars:
        if not name.startswith("longitude_"):
            continue
        method = name.removeprefix("longitude_")
        latitude = f"latitude_{method}"
        if latitude in dataset.data_vars:
            methods[method] = (name, latitude)
    return methods


def _open_dataset(stack: ExitStack, label: str, path: Path, method: str) -> _Dataset:
    if not path.is_dir():
        raise ValueError(f"Plotting Zarr for dataset {label!r} does not exist: {path}")
    dataset = stack.enter_context(xr.open_zarr(path, consolidated=True, chunks=None))
    if "platform" not in dataset.sizes or "time" not in dataset.sizes:
        raise ValueError(f"Plotting dataset {label!r} must have platform and time dimensions")
    for name, dims in {
        "platform_id": ("platform",), "time": ("time",),
        "start_time": ("platform",), "start_lon": ("platform",), "start_lat": ("platform",),
    }.items():
        if name not in dataset.variables or dataset[name].dims != dims:
            raise ValueError(f"Plotting dataset {label!r} has no valid {name} variable")
    platform_ids = tuple(dataset.platform_id.values.astype(str).tolist())
    if len(platform_ids) != len(set(platform_ids)):
        raise ValueError(f"Plotting dataset {label!r} contains duplicate platform IDs")
    methods = _available_methods(dataset)
    if method not in methods:
        raise ValueError(
            f"Plotting dataset {label!r} has no method {method!r}; "
            f"available methods are {sorted(methods)}"
        )
    for method_name, (longitude, latitude) in methods.items():
        if dataset[longitude].dims != ("platform", "time"):
            raise ValueError(f"{label!r} method {method_name!r} longitude has invalid dimensions")
        if dataset[latitude].dims != ("platform", "time"):
            raise ValueError(f"{label!r} method {method_name!r} latitude has invalid dimensions")
    times = dataset.time.values.astype("datetime64[ns]")
    if np.isnat(times).any() or np.any(np.diff(times.astype(np.int64)) <= 0):
        raise ValueError(f"Plotting dataset {label!r} time coordinate is not strictly increasing")
    starts = dataset.start_time.values.astype("datetime64[ns]")
    start_lon = dataset.start_lon.values.astype(float)
    start_lat = dataset.start_lat.values.astype(float)
    if (np.isnat(starts).any() or not np.isfinite(start_lon).all()
            or not np.isfinite(start_lat).all()
            or np.any((start_lon < -180) | (start_lon > 180))
            or np.any((start_lat < -90) | (start_lat > 90))):
        raise ValueError(f"Plotting dataset {label!r} contains invalid exact starts")
    array_ids: tuple[int, ...] | None = None
    if "array_id" in dataset.variables:
        if dataset.array_id.dims != ("platform",):
            raise ValueError(f"Plotting dataset {label!r} has invalid array_id dimensions")
        values = dataset.array_id.values
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Plotting dataset {label!r} array_id must be integer")
        array_ids = tuple(int(value) for value in values)
        if any(value < 1 for value in array_ids):
            raise ValueError(f"Plotting dataset {label!r} has invalid array_id values")
    cluster_ids: tuple[str, ...] | None = None
    if "cluster_id" in dataset.variables:
        if dataset.cluster_id.dims != ("platform",):
            raise ValueError(f"Plotting dataset {label!r} has invalid cluster_id dimensions")
        cluster_ids = tuple(dataset.cluster_id.values.astype(str).tolist())
        if any(not value.strip() for value in cluster_ids):
            raise ValueError(f"Plotting dataset {label!r} has empty cluster_id values")
        if array_ids is None:
            raise ValueError(
                f"Plotting dataset {label!r} has cluster_id but no array_id variable"
            )
    return _Dataset(
        label, path, method, dataset, platform_ids,
        {platform: index for index, platform in enumerate(platform_ids)}, methods,
        array_ids, cluster_ids,
    )


def _time_mask(times: np.ndarray, config: Any) -> np.ndarray:
    mask = np.ones(len(times), dtype=bool)
    if not np.isnat(config.start_time):
        mask &= times >= config.start_time
    if not np.isnat(config.end_time):
        mask &= times <= config.end_time
    return mask


def _selected_platforms(source: _Dataset, selected: set[tuple[str, str]] | None) -> tuple[str, ...]:
    if selected is None:
        return source.platform_ids
    return tuple(
        platform for platform in source.platform_ids if (source.label, platform) in selected
    )


def _start_in_window(source: _Dataset, index: int, config: Any) -> bool:
    start = source.dataset.start_time.isel(platform=index).values.astype("datetime64[ns]")
    return bool(
        (np.isnat(config.start_time) or start >= config.start_time)
        and (np.isnat(config.end_time) or start <= config.end_time)
    )


def _new_map(config: Any):
    figure = plt.figure(figsize=config.map.figsize)
    axes = figure.add_subplot(
        1, 1, 1,
        projection=get_projection(
            config.map.projection, central_longitude=config.map.central_longitude,
        ),
    )
    if config.map.land:
        axes.add_feature(cfeature.LAND.with_scale("110m"), facecolor="0.92", zorder=0)
    if config.map.coastlines:
        axes.coastlines(resolution="110m", linewidth=.7, color="0.3")
    if config.map.gridlines:
        lines = axes.gridlines(draw_labels=True, linestyle="--", alpha=.35, linewidth=.6)
        lines.top_labels = False
        lines.right_labels = False
    if config.map.extent is not None:
        axes.set_extent(config.map.extent, crs=ccrs.PlateCarree())
    return figure, axes


def _save_figure(figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")
    try:
        figure.tight_layout()
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight")
        temporary.replace(path)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise ValueError(f"Failed to create figure {path}: {exc}") from exc
    finally:
        plt.close(figure)


def _plot_overview(
    sources: list[_Dataset], selected: set[tuple[str, str]] | None,
    config: Any, colors: dict[str, Any], path: Path,
) -> None:
    figure, axes = _new_map(config)
    transform = ccrs.PlateCarree()
    handles: list[Line2D] = []
    plotted = 0
    for source in sources:
        color = colors[source.label]
        longitude_name, latitude_name = source.methods[source.selected_method]
        times = source.dataset.time.values.astype("datetime64[ns]")
        time_mask = _time_mask(times, config)
        for platform in _selected_platforms(source, selected):
            index = source.platform_index[platform]
            longitude = source.dataset[longitude_name].isel(platform=index).values.astype(float)
            latitude = source.dataset[latitude_name].isel(platform=index).values.astype(float)
            longitude = np.where(time_mask, longitude, np.nan)
            latitude = np.where(time_mask, latitude, np.nan)
            segments = split_longitude_wrapped_path(longitude, latitude)
            for segment_lon, segment_lat in segments:
                axes.plot(
                    segment_lon, segment_lat, transform=transform, color=color,
                    linewidth=.8, alpha=.75, zorder=2,
                )
            if segments:
                if _start_in_window(source, index, config):
                    start_lon = float(source.dataset.start_lon.isel(platform=index).values)
                    start_lat = float(source.dataset.start_lat.isel(platform=index).values)
                    axes.scatter(
                        start_lon, start_lat, transform=transform, marker="*", s=38,
                        facecolor=color, edgecolor="black", linewidth=.45, zorder=4,
                    )
                plotted += 1
        handles.append(Line2D([0], [0], color=color, lw=1.5,
                              label=f"{source.label}: {source.selected_method}"))
    if not plotted:
        plt.close(figure)
        raise ValueError("No finite trajectory segments remain after plotting selections")
    handles.append(Line2D(
        [0], [0], linestyle="none", marker="*", markersize=9,
        markerfacecolor="white", markeredgecolor="black",
        label="First retained QC position",
    ))
    axes.legend(handles=handles, loc="best", frameon=True)
    axes.set_title("Trajectory overview")
    axes.autoscale_view()
    _save_figure(figure, path, dpi=config.map.dpi)


def _plot_starts(
    sources: list[_Dataset], selected: set[tuple[str, str]] | None,
    config: Any, colors: dict[str, Any], path: Path,
) -> None:
    figure, axes = _new_map(config)
    transform = ccrs.PlateCarree()
    handles: dict[tuple[str, int | None], Line2D] = {}
    plotted = 0
    array_keys = [
        (source.label, array_id)
        for source in sources if source.array_ids is not None
        for array_id in sorted(set(source.array_ids))
    ]
    palette = plt.get_cmap("tab20")
    array_colors = {
        key: palette(index % 20) for index, key in enumerate(array_keys)
    }
    for source in sources:
        for platform in _selected_platforms(source, selected):
            index = source.platform_index[platform]
            if not _start_in_window(source, index, config):
                continue
            array_id = None if source.array_ids is None else source.array_ids[index]
            key = (source.label, array_id)
            color = array_colors.get(key, colors[source.label])
            longitude = float(source.dataset.start_lon.isel(platform=index).values)
            latitude = float(source.dataset.start_lat.isel(platform=index).values)
            axes.scatter(
                longitude, latitude, transform=transform, marker="*", s=56,
                facecolor=color, edgecolor="black", linewidth=.5, zorder=4,
            )
            if config.map.label_starts:
                axes.annotate(
                    f"{source.label}:{platform}", (longitude, latitude), xytext=(4, 4),
                    textcoords="offset points", fontsize=6, transform=transform, zorder=5,
                )
            plotted += 1
            label = source.label if array_id is None else f"{source.label}: Array {array_id}"
            handles.setdefault(key, Line2D(
                [0], [0], linestyle="none", marker="*", markersize=9,
                markerfacecolor=color, markeredgecolor="black", label=label,
            ))
    if not plotted:
        plt.close(figure)
        raise ValueError("No starting positions remain after platform selection")
    axes.legend(handles=list(handles.values()), title="Deployment array", loc="best", frameon=True)
    axes.set_title("First retained QC positions")
    axes.autoscale_view()
    _save_figure(figure, path, dpi=config.map.dpi)


def _plot_array_starts(
    source: _Dataset, array_id: int, platforms: tuple[str, ...], config: Any, path: Path,
) -> None:
    figure, axes = _new_map(config)
    transform = ccrs.PlateCarree()
    indices = [source.platform_index[platform] for platform in platforms]
    indices = [index for index in indices if _start_in_window(source, index, config)]
    if not indices:
        plt.close(figure)
        raise ValueError(f"No starting positions remain for {source.label}:Array {array_id}")
    longitude = source.dataset.start_lon.isel(platform=indices).values.astype(float)
    latitude = source.dataset.start_lat.isel(platform=indices).values.astype(float)
    times = source.dataset.start_time.isel(platform=indices).values.astype("datetime64[ns]")
    elapsed_hours = (times - times.min()) / np.timedelta64(1, "h")
    maximum = max(1.0, float(np.max(elapsed_hours)))
    if source.cluster_ids is None:
        cluster_ids = ("unassigned",) * len(indices)
    else:
        cluster_ids = tuple(source.cluster_ids[index] for index in indices)
    unique_clusters = sorted(set(cluster_ids), key=_cluster_sort_key)
    marker_by_cluster = {
        cluster_id: CLUSTER_MARKERS[index % len(CLUSTER_MARKERS)]
        for index, cluster_id in enumerate(unique_clusters)
    }
    points = None
    handles: list[Line2D] = []
    for cluster_id in unique_clusters:
        selected = np.asarray([value == cluster_id for value in cluster_ids], dtype=bool)
        current = axes.scatter(
            longitude[selected], latitude[selected], c=elapsed_hours[selected],
            cmap="viridis", vmin=0, vmax=maximum, transform=transform,
            marker=marker_by_cluster[cluster_id], s=62, edgecolor="black",
            linewidth=.5, zorder=4,
        )
        if points is None:
            points = current
        handles.append(Line2D(
            [0], [0], linestyle="none", marker=marker_by_cluster[cluster_id],
            markersize=8, markerfacecolor="white", markeredgecolor="black",
            label=_cluster_display_name(cluster_id),
        ))
    if config.map.label_starts:
        for index, platform in enumerate(platforms):
            if source.platform_index[platform] not in indices:
                continue
            position = indices.index(source.platform_index[platform])
            axes.annotate(
                platform, (longitude[position], latitude[position]), xytext=(4, 4),
                textcoords="offset points", fontsize=6, transform=transform, zorder=5,
            )
    colorbar = figure.colorbar(points, ax=axes, pad=.04, shrink=.8)
    colorbar.set_label("Hours after first start in array")
    axes.legend(handles=handles, title="Candidate cluster", loc="best", frameon=True)
    axes.set_title(
        f"{source.label} — Array {array_id}: candidate deployment clusters"
    )
    axes.autoscale_view()
    _save_figure(figure, path, dpi=config.map.dpi)


def _method_order(name: str) -> tuple[int, int | str]:
    if name == "native":
        return 0, 0
    if name == "linear":
        return 1, 0
    match = re.fullmatch(r"spline_(\d+)", name)
    return (2, int(match.group(1))) if match else (3, name)


def _cluster_sort_key(cluster_id: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"array_[0-9]+__cluster_([0-9]+)", cluster_id)
    return (0, int(match.group(1))) if match else (1, cluster_id)


def _cluster_display_name(cluster_id: str) -> str:
    match = re.fullmatch(r"array_[0-9]+__cluster_([0-9]+)", cluster_id)
    return f"Cluster {int(match.group(1))}" if match else cluster_id


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def _plot_check(source: _Dataset, platform: str, config: Any, path: Path) -> None:
    if "linear" not in source.methods:
        raise ValueError(f"Dataset {source.label!r} has no linear method for reconstruction checks")
    figure, axes = _new_map(config)
    transform = ccrs.PlateCarree()
    index = source.platform_index[platform]
    times = source.dataset.time.values.astype("datetime64[ns]")
    time_mask = _time_mask(times, config)
    names = sorted(source.methods, key=_method_order)
    palette = plt.get_cmap("tab10")
    plotted = 0
    for method_index, method in enumerate(names):
        longitude_name, latitude_name = source.methods[method]
        longitude = source.dataset[longitude_name].isel(platform=index).values.astype(float)
        latitude = source.dataset[latitude_name].isel(platform=index).values.astype(float)
        longitude = np.where(time_mask, longitude, np.nan)
        latitude = np.where(time_mask, latitude, np.nan)
        color = palette(method_index % 10)
        for segment_lon, segment_lat in split_longitude_wrapped_path(longitude, latitude):
            axes.plot(
                segment_lon, segment_lat, transform=transform, color=color,
                linewidth=1.7 if method == source.selected_method else .9,
                alpha=.9 if method == source.selected_method else .7,
                label=method if plotted == 0 else None, zorder=2,
            )
            plotted += 1
        # Add one deterministic legend handle even when a method has multiple wrap segments.
        axes.plot([], [], color=color, linewidth=1.7 if method == source.selected_method else .9,
                  label=method)
    if not plotted:
        plt.close(figure)
        raise ValueError(f"No finite reconstruction data for {source.label}:{platform}")
    start_lon = float(source.dataset.start_lon.isel(platform=index).values)
    start_lat = float(source.dataset.start_lat.isel(platform=index).values)
    axes.scatter(
        start_lon, start_lat, transform=transform, marker="*", s=58,
        facecolor="white", edgecolor="black", linewidth=.6, zorder=4,
        label="First retained QC position",
    )
    handles, labels = axes.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes.legend(unique.values(), unique.keys(), loc="best", frameon=True)
    axes.set_title(f"Reconstruction check — {source.label}:{platform}")
    axes.autoscale_view()
    _save_figure(figure, path, dpi=config.map.dpi)


def generate_trajectory_figures(
    current_zarr: str | Path, config: Any, *, progress: Callable[[str], None] | None = None,
) -> tuple[Path, ...]:
    """Validate selected stores and atomically regenerate configured PNG figures."""
    report = progress or (lambda message: None)
    output = Path(config.output_directory)
    specifications = [
        (config.dataset_label, Path(current_zarr), config.method),
        *((store.label, Path(store.path), store.method) for store in config.additional_stores),
    ]
    with ExitStack() as stack:
        sources = [_open_dataset(stack, *specification) for specification in specifications]
        by_label = {source.label: source for source in sources}
        all_pairs = {
            (source.label, platform) for source in sources for platform in source.platform_ids
        }
        selected = (
            {(item.dataset, item.platform_id) for item in config.platforms}
            if config.platforms else None
        )
        requested = (selected or set()) | {
            (item.dataset, item.platform_id) for item in config.check_platforms
        }
        missing = sorted(requested - all_pairs)
        if missing:
            raise ValueError(f"Unknown plotting platform selections: {missing}")
        palette = plt.get_cmap("tab10")
        colors = {source.label: palette(index % 10) for index, source in enumerate(sources)}
        paths = [output / "trajectory_overview.png", output / "starting_positions.png"]
        _plot_overview(sources, selected, config, colors, paths[0])
        report(f"Created figure: {paths[0]}")
        _plot_starts(sources, selected, config, colors, paths[1])
        report(f"Created figure: {paths[1]}")
        current = sources[0]
        if current.array_ids is None:
            raise ValueError(
                f"Current reconstruction dataset {current.label!r} has no array_id variable"
            )
        current_platforms = _selected_platforms(current, selected)
        for array_id in sorted(set(current.array_ids)):
            members = tuple(
                platform for platform in current_platforms
                if current.array_ids[current.platform_index[platform]] == array_id
                and _start_in_window(current, current.platform_index[platform], config)
            )
            if not members:
                continue
            path = output / f"starting_positions__array_{array_id:02d}.png"
            _plot_array_starts(current, array_id, members, config, path)
            paths.append(path)
            report(f"Created figure: {path}")
        for item in config.check_platforms:
            source = by_label[item.dataset]
            path = output / (
                f"reconstruction_check__{_safe_filename(item.dataset)}__"
                f"{_safe_filename(item.platform_id)}.png"
            )
            _plot_check(source, item.platform_id, config, path)
            paths.append(path)
            report(f"Created figure: {path}")
    return tuple(paths)
