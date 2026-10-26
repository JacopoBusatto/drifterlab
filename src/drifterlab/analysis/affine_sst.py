"""Read prepared Celsius SST time bins independently of source-product details."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .config import _boolean, _mapping


def resolve_sst(raw_value: Any, base: Path) -> dict[str, Any]:
    raw = _mapping(
        raw_value, {"enabled", "file", "minimum_celsius", "maximum_celsius"}, "affine.sst"
    )
    path = raw.get("file")
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("affine.sst.file must be a prepared NetCDF path or null")
    limits = []
    for key, default in (("minimum_celsius", -2), ("maximum_celsius", 35)):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"affine.sst.{key} must be finite")
        limits.append(float(value))
    if limits[1] <= limits[0]:
        raise ValueError("SST colour maximum must exceed minimum")
    return {
        "enabled": _boolean(raw.get("enabled", False), "affine.sst.enabled"),
        "file": str((base / path).resolve()) if path is not None else None,
        "minimum_celsius": limits[0],
        "maximum_celsius": limits[1],
    }


def _open(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_timedelta=False)


def _validate(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray, int]:
    for name in ("time", "lat", "lon"):
        if name not in dataset or dataset[name].dims != (name,):
            raise ValueError(f"Prepared SST requires a one-dimensional {name!r} coordinate")
    if "sea_surface_temperature" not in dataset or dataset.sea_surface_temperature.dims != (
        "time",
        "lat",
        "lon",
    ):
        raise ValueError("Prepared SST requires sea_surface_temperature(time, lat, lon)")
    units = str(dataset.sea_surface_temperature.attrs.get("units", "")).strip().lower()
    if units not in {
        "degree_c",
        "degrees_c",
        "celsius",
        "degree_celsius",
        "degrees_celsius",
        "degc",
    }:
        raise ValueError(f"Prepared SST must be in Celsius, got {units!r}; convert units upstream")
    for name, limit in (("lat", 90), ("lon", 360)):
        values = dataset[name].values.astype(float)
        if len(values) < 2 or not np.isfinite(values).all() or np.any(np.abs(values) > limit):
            raise ValueError(
                f"Prepared SST {name} needs at least two finite geographic coordinates"
            )
        delta = np.diff(values)
        if not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError(f"Prepared SST {name} must be strictly monotonic")
        if name == "lon" and np.ptp(values) >= 360:
            raise ValueError("Prepared SST longitude must not repeat the periodic grid boundary")
    bounds_name = dataset.time.attrs.get("bounds", "time_bounds")
    if (
        bounds_name not in dataset
        or dataset[bounds_name].ndim != 2
        or dataset[bounds_name].dims[0] != "time"
        or dataset[bounds_name].shape != (dataset.sizes["time"], 2)
    ):
        raise ValueError("Prepared SST requires CF time bounds with shape (time, 2)")
    bounds = dataset[bounds_name].values
    if not np.issubdtype(bounds.dtype, np.datetime64) or not np.issubdtype(
        dataset.time.dtype, np.datetime64
    ):
        raise ValueError("Prepared SST time and bounds must decode to Gregorian UTC datetimes")
    bounds = bounds.astype("datetime64[ns]")
    times = dataset.time.values.astype("datetime64[ns]")
    if not len(times) or np.isnat(bounds).any() or np.isnat(times).any():
        raise ValueError("Prepared SST time and bounds must be nonempty and finite")
    widths = bounds[:, 1].astype(np.int64) - bounds[:, 0].astype(np.int64)
    if np.any(widths <= 0) or np.any(widths != widths[0]):
        raise ValueError("Prepared SST bins must have a common positive duration")
    if np.any(bounds[1:, 0] < bounds[:-1, 1]) or np.any(np.diff(times.astype(np.int64)) <= 0):
        raise ValueError("Prepared SST bins must be ordered and nonoverlapping")
    if np.any((times < bounds[:, 0]) | (times >= bounds[:, 1])):
        raise ValueError("Each SST reference time must lie inside its time bounds")
    if "observation_count" in dataset and dataset.observation_count.dims != ("time", "lat", "lon"):
        raise ValueError("Prepared observation_count must have dimensions (time, lat, lon)")
    return bounds[:, 0], bounds[:, 1], int(widths[0])


class PreparedSST:
    """Read already masked bins; average equal-duration bins within each daily map.

    Daily composites enter unchanged. Hourly inputs receive equal hour weights,
    never observation-count weights. No source-specific QC or gap filling occurs.
    """

    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.path = Path(settings["file"]) if settings["file"] else None
        self.cache: dict[tuple, tuple[xr.Dataset | None, dict]] = {}
        self.provenance: dict[str, Any] | None = None
        self.starts, self.ends = (
            np.array([], dtype="datetime64[ns]"),
            np.array([], dtype="datetime64[ns]"),
        )
        self.width_ns = 0
        self.attributes: dict[str, Any] = {}
        if not settings["enabled"] or self.path is None:
            return
        try:
            with _open(self.path) as dataset:
                self.starts, self.ends, self.width_ns = _validate(dataset)
                self.attributes = dict(dataset.attrs)
        except Exception as exc:
            raise ValueError(f"Invalid prepared SST input {self.path}: {exc}") from exc

    def load(
        self, day: pd.Timestamp, extent_m: list | None, projection: Any,
        *, geographic_bounds: list | None = None,
    ) -> tuple[xr.Dataset | None, dict]:
        start = pd.Timestamp(day)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = start + pd.Timedelta(days=1)
        meta = {
            "status": "sst_disabled" if not self.settings["enabled"] else "sst_unavailable",
            "kind": "prepared_sst",
            "window_start_utc": start.isoformat(),
            "window_end_utc": end.isoformat(),
            "coverage_fraction": 0.0,
            "coverage_denominator": "regional_subset_grid_pixel_count",
            "temporal_coverage_denominator": "requested_window_duration_divided_by_input_bin_duration",
            "upstream_metadata": self.attributes,
        }
        if (
            not self.settings["enabled"]
            or self.path is None
            or extent_m is None
            or projection is None
        ):
            return None, meta
        key = (
            start.isoformat(), tuple(extent_m), projection.crs_definition,
            tuple(geographic_bounds) if geographic_bounds is not None else None,
        )
        if key in self.cache:
            return self.cache[key]
        low, high = [
            t.tz_localize(None).to_datetime64().astype("datetime64[ns]") for t in (start, end)
        ]
        overlaps = (self.starts < high) & (self.ends > low)
        contained = (self.starts >= low) & (self.ends <= high)
        if np.any(overlaps & ~contained):
            raise ValueError(
                "Prepared SST bins cross a map boundary; prepare aligned bins without splitting a composite"
            )
        duration_ns = int((high - low).astype("timedelta64[ns]").astype(np.int64))
        if duration_ns % self.width_ns:
            raise ValueError(
                "Map duration must be an integer multiple of prepared SST bin duration"
            )
        expected = duration_ns // self.width_ns
        indices = np.flatnonzero(contained)
        meta.update(
            expected_bin_count=expected,
            available_bin_count=len(indices),
            input_bin_hours=self.width_ns / 3.6e12,
            aggregation="supplied_bin_unchanged"
            if expected == 1
            else "equal_bin_nan_ignoring_mean",
        )
        if self.provenance is None:
            digest = sha256()
            with self.path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.provenance = {
                "source_path": str(self.path),
                "source_bytes": self.path.stat().st_size,
                "source_mtime_ns": self.path.stat().st_mtime_ns,
                "source_sha256": digest.hexdigest(),
            }
        meta.update(self.provenance)
        if not len(indices):
            self.cache[key] = (None, meta)
            return None, meta
        x0, x1, y0, y1 = extent_m
        xx, yy = np.meshgrid(np.linspace(x0, x1, 15), np.linspace(y0, y1, 15))
        region_lon, region_lat = projection.inverse.transform(xx, yy)
        origin = projection.origin_longitude
        region_lon = origin + (np.asarray(region_lon) - origin + 180) % 360 - 180
        if geographic_bounds is not None:
            region_lon = np.asarray(geographic_bounds[:2])
            region_lat = np.asarray(geographic_bounds[2:])
        try:
            with _open(self.path) as dataset:
                raw_lon, lat = dataset.lon.values.astype(float), dataset.lat.values.astype(float)
                lon = origin + (raw_lon - origin + 180) % 360 - 180
                dlon, dlat = np.max(np.abs(np.diff(raw_lon))), np.max(np.abs(np.diff(lat)))
                li = np.flatnonzero(
                    (lon >= np.min(region_lon) - dlon) & (lon <= np.max(region_lon) + dlon)
                )
                bi = np.flatnonzero(
                    (lat >= np.min(region_lat) - dlat) & (lat <= np.max(region_lat) + dlat)
                )
                if not len(li) or not len(bi):
                    self.cache[key] = (None, meta)
                    return None, meta
                li, bi = li[np.argsort(lon[li])], bi[np.argsort(lat[bi])]
                subset = dataset.isel(time=indices, lat=bi, lon=li)
                values = subset.sea_surface_temperature.values.astype(float)
                good = np.isfinite(values)
                counts = good.sum(axis=0).astype(np.int32)
                temperature = np.full(counts.shape, np.nan)
                np.divide(
                    np.where(good, values, 0).sum(axis=0), counts, out=temperature, where=counts > 0
                )
                product = xr.Dataset(
                    {
                        "sst_celsius": (("latitude", "longitude"), temperature),
                        "contributing_bin_count": (("latitude", "longitude"), counts),
                        "temporal_coverage_fraction": (
                            ("latitude", "longitude"),
                            counts / expected,
                        ),
                    },
                    coords={"longitude": lon[li], "latitude": lat[bi]},
                )
                product.sst_celsius.attrs["units"] = "degree_Celsius"
                product.contributing_bin_count.attrs["description"] = (
                    "Number of finite prepared SST bins contributing to the window"
                )
                product.temporal_coverage_fraction.attrs["denominator"] = (
                    "Expected bins in the full requested map window, including missing bins"
                )
                product.temporal_coverage_fraction.attrs["description"] = (
                    "Fraction of prepared bins with finite SST; not continuous acquisition-time coverage"
                )
                if "observation_count" in subset:
                    observations = subset.observation_count.values.astype(float)
                    if np.any(
                        np.isfinite(observations)
                        & ((observations < 0) | (observations != np.floor(observations)))
                    ):
                        raise ValueError(
                            "Prepared observation_count must be nonnegative integer-valued"
                        )
                    product["observation_count"] = (
                        ("latitude", "longitude"),
                        np.where(good & np.isfinite(observations), observations, 0).sum(axis=0),
                    )
                    product.observation_count.attrs.update(subset.observation_count.attrs)
                    product.observation_count.attrs["aggregation"] = (
                        "Sum over finite SST bins; never used as SST averaging weights"
                    )
                product.attrs.update(
                    source_path=str(self.path),
                    product_kind="prepared SST background",
                    aggregation=meta["aggregation"],
                    expected_bin_count=expected,
                    input_bin_hours=self.width_ns / 3.6e12,
                    source_title=str(self.attributes.get("title", "Prepared SST")),
                )
                meta.update(
                    status="sst_available" if counts.any() else "sst_unavailable",
                    valid_pixel_count=int(np.count_nonzero(counts)),
                    regional_grid_pixel_count=int(counts.size),
                    coverage_fraction=float(np.mean(counts > 0)),
                    mean_temporal_coverage_fraction=float(np.mean(counts / expected)),
                )
        except Exception as exc:
            raise ValueError(f"Cannot read prepared SST {self.path}: {exc}") from exc
        self.cache[key] = (product, meta)
        return product, meta
