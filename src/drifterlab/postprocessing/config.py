"""Strict YAML configuration for trajectory postprocessing."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SUPPORTED_MAP_PROJECTIONS = {
    "PlateCarree", "Mercator", "SouthPolarStereo", "NorthPolarStereo", "Robinson",
}
METHOD_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")
ARRAY_KEY_PATTERN = re.compile(r"^array_([0-9]{3})$")


@dataclass(frozen=True)
class TimeWindowConfig:
    start: np.datetime64
    end: np.datetime64


@dataclass(frozen=True)
class MovieConfig:
    enabled: bool
    format: str
    frame_interval_minutes: int
    fps: int
    tail_hours: float
    dpi: int


@dataclass(frozen=True)
class MapConfig:
    projection: str
    central_longitude: float
    extent: tuple[float, float, float, float] | None
    padding_fraction: float
    figsize: tuple[float, float]
    dpi: int
    land: bool
    coastlines: bool
    gridlines: bool
    scale_bar: bool


@dataclass(frozen=True)
class ArrayMetadataConfig:
    nominal_deployment_time: np.datetime64


@dataclass(frozen=True)
class ArrayPlotOverride:
    time: TimeWindowConfig | None
    extent: tuple[float, float, float, float] | None
    map_configured: bool


@dataclass(frozen=True)
class TrajectoryPlottingConfig:
    enabled: bool
    color_by: str
    color_mode: str
    cmap: str
    color_label: str | None
    vmin: float | None
    vmax: float | None
    time: TimeWindowConfig
    movie: MovieConfig
    map: MapConfig
    array_overrides: dict[int, ArrayPlotOverride]


@dataclass(frozen=True)
class ClusterVelocityConfig:
    difference_interval_minutes: float
    histogram_bins: int
    speed_range_m_s: tuple[float, float] | None


@dataclass(frozen=True)
class ClusterStatisticsPlottingConfig:
    dpi: int
    relative_dispersion_yscale: str


@dataclass(frozen=True)
class ClusterStatisticsConfig:
    enabled: bool
    time: TimeWindowConfig
    stop_on_member_loss: bool
    percentiles: tuple[float, ...]
    velocity: ClusterVelocityConfig
    plotting: ClusterStatisticsPlottingConfig
    array_overrides: dict[int, TimeWindowConfig]


@dataclass(frozen=True)
class PostprocessingConfig:
    source_path: Path
    trajectory_path: Path
    dataset_label: str
    coordinate_method: str
    output_directory: Path
    arrays: dict[int, ArrayMetadataConfig]
    trajectory_plotting: TrajectoryPlottingConfig
    cluster_statistics: ClusterStatisticsConfig

    def effective(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the resolved configuration."""
        plotting = self.trajectory_plotting
        statistics = self.cluster_statistics
        return {
            "input": {
                "trajectories": str(self.trajectory_path),
                "dataset_label": self.dataset_label,
                "coordinate_method": self.coordinate_method,
            },
            "output": {"directory": str(self.output_directory)},
            "arrays": {
                _array_key(array_id): {
                    "nominal_deployment_time": _optional_utc(item.nominal_deployment_time),
                }
                for array_id, item in sorted(self.arrays.items())
            },
            "trajectory_plotting": {
                "enabled": plotting.enabled,
                "color_by": plotting.color_by,
                "color_mode": plotting.color_mode,
                "cmap": plotting.cmap,
                "color_label": plotting.color_label,
                "vmin": plotting.vmin,
                "vmax": plotting.vmax,
                "time": _time_effective(plotting.time),
                "movie": {
                    "enabled": plotting.movie.enabled,
                    "format": plotting.movie.format,
                    "frame_interval_minutes": plotting.movie.frame_interval_minutes,
                    "fps": plotting.movie.fps,
                    "tail_hours": plotting.movie.tail_hours,
                    "dpi": plotting.movie.dpi,
                },
                "map": {
                    "projection": plotting.map.projection,
                    "central_longitude": plotting.map.central_longitude,
                    "extent": (
                        None if plotting.map.extent is None else list(plotting.map.extent)
                    ),
                    "padding_fraction": plotting.map.padding_fraction,
                    "figsize": list(plotting.map.figsize),
                    "dpi": plotting.map.dpi,
                    "land": plotting.map.land,
                    "coastlines": plotting.map.coastlines,
                    "gridlines": plotting.map.gridlines,
                    "scale_bar": plotting.map.scale_bar,
                },
                "array_overrides": {
                    _array_key(array_id): {
                        **({"time": _time_effective(item.time)} if item.time is not None else {}),
                        **(
                            {"map": {"extent": None if item.extent is None else list(item.extent)}}
                            if item.map_configured else {}
                        ),
                    }
                    for array_id, item in sorted(plotting.array_overrides.items())
                },
            },
            "cluster_statistics": {
                "enabled": statistics.enabled,
                "time": _time_effective(statistics.time),
                "stop_on_member_loss": statistics.stop_on_member_loss,
                "percentiles": list(statistics.percentiles),
                "velocity": {
                    "difference_interval_minutes": (
                        statistics.velocity.difference_interval_minutes
                    ),
                    "histogram_bins": statistics.velocity.histogram_bins,
                    "speed_range_m_s": (
                        None if statistics.velocity.speed_range_m_s is None
                        else list(statistics.velocity.speed_range_m_s)
                    ),
                },
                "plotting": {
                    "dpi": statistics.plotting.dpi,
                    "relative_dispersion_yscale": (
                        statistics.plotting.relative_dispersion_yscale
                    ),
                },
                "array_overrides": {
                    _array_key(array_id): {"time": _time_effective(time)}
                    for array_id, time in sorted(statistics.array_overrides.items())
                },
            },
        }


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be null or numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be null or finite")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _utc(value: Any, name: str, *, optional: bool = True) -> np.datetime64:
    if optional and (value is None or str(value).strip() in {"", "null", "None"}):
        return np.datetime64("NaT", "ns")
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an explicit UTC timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must be an explicit UTC timestamp")
    return stamp.tz_convert("UTC").tz_localize(None).to_datetime64().astype("datetime64[ns]")


def _format_utc(value: np.datetime64) -> str:
    return pd.Timestamp(value, tz="UTC").isoformat().replace("+00:00", "Z")


def _optional_utc(value: np.datetime64) -> str | None:
    return None if np.isnat(value) else _format_utc(value)


def _time_effective(value: TimeWindowConfig) -> dict[str, str | None]:
    return {"start": _optional_utc(value.start), "end": _optional_utc(value.end)}


def _array_key(array_id: int) -> str:
    return f"array_{array_id:03d}"


def _parse_time(value: Any, name: str) -> TimeWindowConfig:
    section = _mapping(value, {"start", "end"}, name)
    start = _utc(section.get("start"), f"{name}.start")
    end = _utc(section.get("end"), f"{name}.end")
    if not np.isnat(start) and not np.isnat(end) and end < start:
        raise ValueError(f"{name}.end must not precede {name}.start")
    return TimeWindowConfig(start, end)


def _parse_extent(value: Any, name: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must be null or [west, east, south, north]")
    extent = tuple(float(item) for item in value)
    if (
        not all(math.isfinite(item) for item in extent)
        or extent[0] >= extent[1]
        or extent[2] >= extent[3]
        or extent[2] < -90
        or extent[3] > 90
    ):
        raise ValueError(f"{name} contains invalid geographic bounds")
    return extent


def percentile_suffix(value: float) -> str:
    """Return a deterministic CSV-safe percentile suffix such as ``p012p5``."""
    decimal = Decimal(str(value))
    rendered = format(decimal, "f")
    whole, separator, fraction = rendered.partition(".")
    fraction = fraction.rstrip("0")
    suffix = f"p{int(whole):03d}"
    return suffix if not separator or not fraction else f"{suffix}p{fraction}"


def _parse_percentiles(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{name} values must be numeric")
        number = float(item)
        if not math.isfinite(number) or not 0 <= number <= 100:
            raise ValueError(f"{name} values must lie in [0, 100]")
        parsed.append(number)
    if any(right <= left for left, right in pairwise(parsed)):
        raise ValueError(f"{name} values must be unique and strictly increasing")
    suffixes = [percentile_suffix(item) for item in parsed]
    if len(set(suffixes)) != len(suffixes):
        raise ValueError(f"{name} values produce colliding canonical column names")
    return tuple(parsed)


def _parse_speed_range(
    value: Any, name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be null or [minimum, maximum]")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise ValueError(f"{name} bounds must be numeric")
    lower, upper = (float(item) for item in value)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0 or upper <= lower:
        raise ValueError(f"{name} must satisfy 0 <= minimum < maximum")
    return lower, upper


def _parse_array_key(value: Any, name: str) -> int:
    if not isinstance(value, str) or not (match := ARRAY_KEY_PATTERN.fullmatch(value)):
        raise ValueError(f"{name} keys must use array_NNN")
    result = int(match.group(1))
    if result < 1:
        raise ValueError(f"{name} identifiers start at array_001")
    return result


def load_postprocessing_config(path: str | Path) -> PostprocessingConfig:
    """Load and strictly validate one postprocessing YAML file."""
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    data = _mapping(
        data,
        {"input", "output", "arrays", "trajectory_plotting", "cluster_statistics"},
        "configuration",
    )
    input_section = _mapping(
        data.get("input", {}),
        {"trajectories", "dataset_label", "coordinate_method"},
        "input",
    )
    output_section = _mapping(data.get("output", {}), {"directory"}, "output")

    def resolved(value: Any, name: str) -> Path:
        text = _nonempty_string(value, name)
        candidate = Path(text).expanduser()
        return (source_path.parent / candidate).resolve()

    trajectory_path = resolved(input_section.get("trajectories"), "input.trajectories")
    dataset_label = _nonempty_string(
        input_section.get("dataset_label", "current"), "input.dataset_label",
    )
    coordinate_method = _nonempty_string(
        input_section.get("coordinate_method", "linear"), "input.coordinate_method",
    )
    if not METHOD_PATTERN.fullmatch(coordinate_method):
        raise ValueError("input.coordinate_method must contain only letters, digits, and underscores")
    output_directory = resolved(output_section.get("directory"), "output.directory")
    if output_directory == trajectory_path or trajectory_path in output_directory.parents:
        raise ValueError("Postprocessing output must be outside the input trajectory store")

    raw_arrays = data.get("arrays", {})
    if not isinstance(raw_arrays, dict):
        raise ValueError("arrays must be a mapping")
    arrays: dict[int, ArrayMetadataConfig] = {}
    for raw_key, raw_value in raw_arrays.items():
        array_id = _parse_array_key(raw_key, "arrays")
        section = _mapping(raw_value, {"nominal_deployment_time"}, f"arrays.{raw_key}")
        arrays[array_id] = ArrayMetadataConfig(_utc(
            section.get("nominal_deployment_time"),
            f"arrays.{raw_key}.nominal_deployment_time",
        ))

    plotting_section = _mapping(
        data.get("trajectory_plotting", {}),
        {
            "enabled", "color_by", "color_mode", "cmap", "color_label", "vmin", "vmax",
            "time", "movie", "map", "array_overrides",
        },
        "trajectory_plotting",
    )
    color_mode = _nonempty_string(
        plotting_section.get("color_mode", "auto"), "trajectory_plotting.color_mode",
    )
    if color_mode not in {"auto", "categorical", "numeric"}:
        raise ValueError("trajectory_plotting.color_mode must be auto, categorical, or numeric")
    color_label_raw = plotting_section.get("color_label")
    color_label = (
        None if color_label_raw is None
        else _nonempty_string(color_label_raw, "trajectory_plotting.color_label")
    )
    vmin = _optional_number(plotting_section.get("vmin"), "trajectory_plotting.vmin")
    vmax = _optional_number(plotting_section.get("vmax"), "trajectory_plotting.vmax")
    if vmin is not None and vmax is not None and vmax <= vmin:
        raise ValueError("trajectory_plotting.vmax must be greater than vmin")
    time = _parse_time(plotting_section.get("time", {}), "trajectory_plotting.time")

    movie_section = _mapping(
        plotting_section.get("movie", {}),
        {"enabled", "format", "frame_interval_minutes", "fps", "tail_hours", "dpi"},
        "trajectory_plotting.movie",
    )
    movie_format = _nonempty_string(
        movie_section.get("format", "mp4"), "trajectory_plotting.movie.format",
    ).lower()
    if movie_format != "mp4":
        raise ValueError("trajectory_plotting.movie.format currently supports only mp4")
    movie = MovieConfig(
        _boolean(movie_section.get("enabled", True), "trajectory_plotting.movie.enabled"),
        movie_format,
        _positive_integer(
            movie_section.get("frame_interval_minutes", 60),
            "trajectory_plotting.movie.frame_interval_minutes",
        ),
        _positive_integer(movie_section.get("fps", 15), "trajectory_plotting.movie.fps"),
        _positive_number(
            movie_section.get("tail_hours", 24), "trajectory_plotting.movie.tail_hours",
        ),
        _positive_integer(movie_section.get("dpi", 120), "trajectory_plotting.movie.dpi"),
    )

    map_section = _mapping(
        plotting_section.get("map", {}),
        {
            "projection", "central_longitude", "extent", "padding_fraction", "figsize",
            "dpi", "land", "coastlines", "gridlines", "scale_bar",
        },
        "trajectory_plotting.map",
    )
    projection = _nonempty_string(
        map_section.get("projection", "PlateCarree"), "trajectory_plotting.map.projection",
    )
    if projection not in SUPPORTED_MAP_PROJECTIONS:
        raise ValueError(
            "trajectory_plotting.map.projection must be one of "
            f"{sorted(SUPPORTED_MAP_PROJECTIONS)}"
        )
    central_longitude = float(map_section.get("central_longitude", 0))
    if not math.isfinite(central_longitude) or not -180 <= central_longitude <= 180:
        raise ValueError("trajectory_plotting.map.central_longitude must be between -180 and 180")
    padding_fraction = float(map_section.get("padding_fraction", 0.05))
    if not math.isfinite(padding_fraction) or not 0 <= padding_fraction <= 1:
        raise ValueError("trajectory_plotting.map.padding_fraction must be between 0 and 1")
    raw_figsize = map_section.get("figsize", [12, 8])
    if not isinstance(raw_figsize, list) or len(raw_figsize) != 2:
        raise ValueError("trajectory_plotting.map.figsize must contain width and height")
    map_config = MapConfig(
        projection,
        central_longitude,
        _parse_extent(map_section.get("extent"), "trajectory_plotting.map.extent"),
        padding_fraction,
        tuple(_positive_number(item, "trajectory_plotting.map.figsize") for item in raw_figsize),
        _positive_integer(map_section.get("dpi", 150), "trajectory_plotting.map.dpi"),
        _boolean(map_section.get("land", True), "trajectory_plotting.map.land"),
        _boolean(map_section.get("coastlines", True), "trajectory_plotting.map.coastlines"),
        _boolean(map_section.get("gridlines", True), "trajectory_plotting.map.gridlines"),
        _boolean(map_section.get("scale_bar", True), "trajectory_plotting.map.scale_bar"),
    )

    raw_overrides = plotting_section.get("array_overrides", {})
    if not isinstance(raw_overrides, dict):
        raise ValueError("trajectory_plotting.array_overrides must be a mapping")
    overrides: dict[int, ArrayPlotOverride] = {}
    for raw_key, raw_value in raw_overrides.items():
        array_id = _parse_array_key(raw_key, "trajectory_plotting.array_overrides")
        section = _mapping(
            raw_value, {"time", "map"}, f"trajectory_plotting.array_overrides.{raw_key}",
        )
        override_time = (
            _parse_time(section["time"], f"trajectory_plotting.array_overrides.{raw_key}.time")
            if "time" in section else None
        )
        override_map = None
        if "map" in section:
            override_map = _mapping(
                section["map"], {"extent"},
                f"trajectory_plotting.array_overrides.{raw_key}.map",
            )
        overrides[array_id] = ArrayPlotOverride(
            override_time,
            None if override_map is None else _parse_extent(
                override_map.get("extent"),
                f"trajectory_plotting.array_overrides.{raw_key}.map.extent",
            ),
            override_map is not None,
        )

    trajectory_plotting = TrajectoryPlottingConfig(
        _boolean(plotting_section.get("enabled", False), "trajectory_plotting.enabled"),
        _nonempty_string(
            plotting_section.get("color_by", "cluster_id"),
            "trajectory_plotting.color_by",
        ),
        color_mode,
        _nonempty_string(plotting_section.get("cmap", "tab20"), "trajectory_plotting.cmap"),
        color_label,
        vmin,
        vmax,
        time,
        movie,
        map_config,
        overrides,
    )

    statistics_section = _mapping(
        data.get("cluster_statistics", {}),
        {
            "enabled", "time", "stop_on_member_loss", "percentiles", "velocity",
            "plotting", "array_overrides",
        },
        "cluster_statistics",
    )
    statistics_time = _parse_time(
        statistics_section.get("time", {}), "cluster_statistics.time",
    )
    velocity_section = _mapping(
        statistics_section.get("velocity", {}),
        {"difference_interval_minutes", "histogram_bins", "speed_range_m_s"},
        "cluster_statistics.velocity",
    )
    velocity = ClusterVelocityConfig(
        _positive_number(
            velocity_section.get("difference_interval_minutes", 30),
            "cluster_statistics.velocity.difference_interval_minutes",
        ),
        _positive_integer(
            velocity_section.get("histogram_bins", 50),
            "cluster_statistics.velocity.histogram_bins",
        ),
        _parse_speed_range(
            velocity_section.get("speed_range_m_s"),
            "cluster_statistics.velocity.speed_range_m_s",
        ),
    )
    statistics_plotting_section = _mapping(
        statistics_section.get("plotting", {}),
        {"dpi", "relative_dispersion_yscale"},
        "cluster_statistics.plotting",
    )
    relative_dispersion_yscale = _nonempty_string(
        statistics_plotting_section.get("relative_dispersion_yscale", "log"),
        "cluster_statistics.plotting.relative_dispersion_yscale",
    ).lower()
    if relative_dispersion_yscale not in {"linear", "log"}:
        raise ValueError(
            "cluster_statistics.plotting.relative_dispersion_yscale must be linear or log"
        )
    statistics_plotting = ClusterStatisticsPlottingConfig(
        _positive_integer(
            statistics_plotting_section.get("dpi", 150),
            "cluster_statistics.plotting.dpi",
        ),
        relative_dispersion_yscale,
    )
    raw_statistics_overrides = statistics_section.get("array_overrides", {})
    if not isinstance(raw_statistics_overrides, dict):
        raise ValueError("cluster_statistics.array_overrides must be a mapping")
    statistics_overrides: dict[int, TimeWindowConfig] = {}
    for raw_key, raw_value in raw_statistics_overrides.items():
        array_id = _parse_array_key(raw_key, "cluster_statistics.array_overrides")
        section = _mapping(
            raw_value, {"time"}, f"cluster_statistics.array_overrides.{raw_key}",
        )
        statistics_overrides[array_id] = _parse_time(
            section.get("time", {}),
            f"cluster_statistics.array_overrides.{raw_key}.time",
        )
    cluster_statistics = ClusterStatisticsConfig(
        _boolean(statistics_section.get("enabled", False), "cluster_statistics.enabled"),
        statistics_time,
        _boolean(
            statistics_section.get("stop_on_member_loss", False),
            "cluster_statistics.stop_on_member_loss",
        ),
        _parse_percentiles(
            statistics_section.get("percentiles", [0, 25, 50, 75, 100]),
            "cluster_statistics.percentiles",
        ),
        velocity,
        statistics_plotting,
        statistics_overrides,
    )
    return PostprocessingConfig(
        source_path, trajectory_path, dataset_label, coordinate_method,
        output_directory, arrays, trajectory_plotting, cluster_statistics,
    )
