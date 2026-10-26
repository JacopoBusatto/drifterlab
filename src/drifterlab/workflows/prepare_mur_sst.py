"""Convert local MUR v4.1 subsets to the common prepared-SST interface.

MUR-specific units, masks and nominal-date assignment belong here. The affine
reader remains unaware of the satellite product. No downloads or regridding.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr

from drifterlab import __version__

TIME_UNITS = "seconds since 1970-01-01 00:00:00 UTC"
MUR_URL = "https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1"
FIELDS = ("analysis_error", "dt_1km_data", "sea_ice_fraction")
TIME_POLICY = (
    "Use the UTC calendar date of the nominal analysis time for map assignment. "
    "time_bounds are plotting-assignment bins, not observation or source coverage bounds. "
    "Original source coverage is preserved in source_time_bounds. No daily averaging."
)


def _utc(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("MUR timestamps must be finite")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _seconds(value: object) -> int:
    return int(_utc(value).value // 1_000_000_000)


def _decoded(variable: xr.DataArray) -> np.ndarray:
    """Mask packed fill/range values before applying CF scale and offset."""
    raw = np.asarray(variable.values, dtype=float)
    good = np.isfinite(raw)
    for key in ("_FillValue", "missing_value"):
        for fill in np.asarray(variable.attrs.get(key, [])).ravel():
            good &= raw != fill
    if "valid_range" in variable.attrs:
        low, high = variable.attrs["valid_range"]
        good &= (raw >= low) & (raw <= high)
    for key, comparison in (("valid_min", np.greater_equal), ("valid_max", np.less_equal)):
        if key in variable.attrs:
            good &= comparison(raw, variable.attrs[key])
    return np.where(
        good,
        raw * variable.attrs.get("scale_factor", 1) + variable.attrs.get("add_offset", 0),
        np.nan,
    )


def _attributes(variable: xr.DataArray) -> dict:
    packed = {
        "_FillValue",
        "missing_value",
        "scale_factor",
        "add_offset",
        "valid_min",
        "valid_max",
        "valid_range",
        "coordinates",
    }
    return {key: value for key, value in variable.attrs.items() if key not in packed}


def _metadata(dataset: xr.Dataset) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    if str(dataset.attrs.get("processing_level", "")).upper() != "L4" or not str(
        dataset.attrs.get("id", "")
    ).startswith("MUR-JPL-L4-GLOB-v"):
        raise ValueError("Expected a MUR-JPL-L4-GLOB daily analysis")
    version = str(dataset.attrs.get("product_version", "")).lstrip("0")
    if version != "4.1":
        raise ValueError("This converter supports MUR version 4.1")
    for name in ("lat", "lon", "time"):
        if name not in dataset or dataset[name].dims != (name,):
            raise ValueError(f"MUR requires one-dimensional {name}")
    for name in ("analysed_sst", "mask", *FIELDS):
        if name not in dataset or dataset[name].dims != ("time", "lat", "lon"):
            raise ValueError(f"MUR requires {name}(time, lat, lon)")
    if dataset.sizes["time"] != 1:
        raise ValueError("Each input MUR file must contain one nominal daily analysis")
    time = xr.decode_cf(dataset[["time"]]).time.values[0]
    reference = _utc(time)
    try:
        low, high = [
            _utc(dataset.attrs[key]) for key in ("time_coverage_start", "time_coverage_end")
        ]
    except KeyError as exc:
        raise ValueError("MUR source time_coverage_start/end metadata are required") from exc
    if not low <= reference < high:
        raise ValueError("MUR nominal analysis time must lie inside its stated source coverage")
    if str(dataset.analysed_sst.attrs.get("units", "")).strip().lower() not in {
        "kelvin",
        "k",
        "degree_kelvin",
        "degrees_kelvin",
    }:
        raise ValueError("MUR analysed_sst must declare Kelvin units")
    if str(dataset.analysis_error.attrs.get("units", "")).strip().lower() not in {"kelvin", "k"}:
        raise ValueError("MUR analysis_error must declare Kelvin uncertainty units")
    if str(dataset.dt_1km_data.attrs.get("units", "")).strip().lower() not in {
        "hours",
        "hour",
        "h",
    }:
        raise ValueError("MUR dt_1km_data must declare hours")
    return reference, low, high


def _grid(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for name, limit in (("lat", 90), ("lon", 360)):
        values = _decoded(dataset[name])
        delta = np.diff(values)
        if (
            len(values) < 2
            or not np.isfinite(values).all()
            or np.any(np.abs(values) > limit)
            or not (np.all(delta > 0) or np.all(delta < 0))
        ):
            raise ValueError(f"MUR {name} must be a finite monotonic geographic grid")
        result.append(values)
    return result[0], result[1]


def prepare_mur_sst(
    input_directory: str | Path,
    output: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    maximum_analysis_error_k: float | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Stream daily subsets to one prepared file; preserve requested missing days."""
    import h5netcdf
    import h5py

    report = progress or (lambda message: None)
    root, target = Path(input_directory).resolve(), Path(output).resolve()
    if maximum_analysis_error_k is not None and (
        isinstance(maximum_analysis_error_k, bool)
        or not np.isfinite(maximum_analysis_error_k)
        or maximum_analysis_error_k <= 0
    ):
        raise ValueError("maximum_analysis_error_k must be finite and positive or unset")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {target}; use --overwrite deliberately")
    paths = sorted(root.rglob("*-L4_GHRSST-SSTfnd-MUR-*.nc"))
    if not paths:
        raise ValueError(f"No MUR daily NetCDF files found under {root}")
    if target in paths:
        raise ValueError("Prepared output must not replace a source MUR file")
    requested_start = _utc(start).normalize() if start is not None else None
    requested_end = _utc(end).normalize() if end is not None else None
    if (
        requested_start is not None
        and requested_end is not None
        and requested_end < requested_start
    ):
        raise ValueError("end must be on or after start")
    records = {}
    for path in paths:
        try:
            with xr.open_dataset(path, decode_cf=False) as dataset:
                reference, low, high = _metadata(dataset)
        except Exception as exc:
            raise ValueError(f"Invalid MUR input {path}: {exc}") from exc
        day = reference.normalize()
        if (requested_start is not None and day < requested_start) or (
            requested_end is not None and day > requested_end
        ):
            continue
        if day in records:
            raise ValueError(f"Multiple MUR analyses for {day.date()}; choose one source/version")
        records[day] = (path, reference, low, high)
    if not records:
        raise ValueError("No MUR analyses fall inside the requested dates")
    first, last = (
        requested_start if requested_start is not None else min(records),
        requested_end if requested_end is not None else max(records),
    )
    if last < first:
        raise ValueError("Requested date range contains no MUR analyses")
    days = pd.date_range(first, last, freq="D")
    with xr.open_dataset(records[min(records)][0], decode_cf=False) as sample:
        latitude, longitude = _grid(sample)
        field_attrs = {name: _attributes(sample[name]) for name in FIELDS}
        field_attrs["sea_ice_fraction"].setdefault("units", "1")
        mask_attrs = _attributes(sample["mask"])
        product_id = str(sample.attrs["id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.part")
    sources = []
    try:
        with h5netcdf.File(temporary, "w") as out:
            out.dimensions = {
                "time": len(days),
                "lat": len(latitude),
                "lon": len(longitude),
                "nv": 2,
            }
            out.attrs.update(
                Conventions="CF-1.8",
                title="MUR L4 daily foundation SST analysis",
                source_product=product_id,
                references=MUR_URL,
                temporal_representation="daily_analysis_at_nominal_reference_time",
                calendar_assignment_policy=TIME_POLICY,
                processing="Decode packed values; convert SST Kelvin to Celsius; retain the source grid. No temporal averaging or additional interpolation.",
                quality_policy="Finite SST; open-sea mask bit 1 set; land/lake/ice bits 2,4,8,16 clear. Optional analysis-error ceiling recorded separately.",
                maximum_analysis_error_k="unset"
                if maximum_analysis_error_k is None
                else maximum_analysis_error_k,
                source_directory=str(root),
                source_file_count=len(records),
                time_coverage_start=first.isoformat(),
                time_coverage_end=(last + pd.Timedelta(days=1)).isoformat(),
                source_coverage_note="Global time_coverage_* describe calendar assignment; source_time_bounds retain original per-file coverage.",
                history=datetime.now(timezone.utc).isoformat() + " drifterlab prepare_mur_sst",
                drifterlab_version=__version__,
            )
            for name, values, units, standard in (
                ("lat", latitude, "degrees_north", "latitude"),
                ("lon", longitude, "degrees_east", "longitude"),
            ):
                var = out.create_variable(name, (name,), dtype="f8", data=values)
                var.attrs.update(units=units, standard_name=standard)
            time = out.create_variable("time", ("time",), dtype="i8")
            time.attrs.update(
                units=TIME_UNITS,
                calendar="standard",
                standard_name="time",
                bounds="time_bounds",
                long_name="Nominal MUR analysis time; not a daily mean",
            )
            bounds = out.create_variable("time_bounds", ("time", "nv"), dtype="i8")
            bounds.attrs.update(
                units=TIME_UNITS,
                calendar="standard",
                long_name="UTC calendar-day assignment bounds for plotting",
                comment=TIME_POLICY,
            )
            source_bounds = out.create_variable(
                "source_time_bounds", ("time", "nv"), dtype="f8", fillvalue=np.nan
            )
            source_bounds.attrs.update(
                units=TIME_UNITS, calendar="standard", long_name="Original MUR source time coverage"
            )
            present = out.create_variable("source_present", ("time",), dtype="u1")
            present.attrs.update(
                flag_values=np.array([0, 1], dtype=np.uint8), flag_meanings="missing available"
            )
            filenames = out.create_variable(
                "source_file", ("time",), dtype=h5py.string_dtype("utf-8")
            )
            layout = {
                "chunks": (1, min(128, len(latitude)), min(128, len(longitude))),
                "compression": "gzip",
                "compression_opts": 4,
            }
            dims = ("time", "lat", "lon")
            temperature = out.create_variable(
                "sea_surface_temperature", dims, dtype="f4", fillvalue=np.nan, **layout
            )
            temperature.attrs.update(
                units="degree_C",
                standard_name="sea_surface_foundation_temperature",
                long_name="MUR analysed foundation SST",
                ancillary_variables="valid_mask analysis_error dt_1km_data mask sea_ice_fraction",
                comment="Interpolated L4 analysis supplied by JPL; finite values are not direct-observation counts.",
            )
            valid_mask = out.create_variable("valid_mask", dims, dtype="u1", **layout)
            valid_mask.attrs.update(
                flag_values=np.array([0, 1], dtype=np.uint8), flag_meanings="invalid valid"
            )
            raw_mask = out.create_variable("mask", dims, dtype="i2", fillvalue=-32768, **layout)
            raw_mask.attrs.update(mask_attrs)
            fields = {}
            for name in FIELDS:
                fields[name] = out.create_variable(
                    name, dims, dtype="f4", fillvalue=np.nan, **layout
                )
                fields[name].attrs.update(field_attrs[name])
            for index, day in enumerate(days):
                origin = _seconds(day)
                bounds[index] = [origin, origin + 86400]
                time[index] = origin + 9 * 3600
                present[index] = int(day in records)
                filenames[index] = records[day][0].name if day in records else ""
                if day not in records:
                    valid_mask[index] = np.zeros((len(latitude), len(longitude)), dtype=np.uint8)
                    report(f"{day:%Y-%m-%d}: MUR source missing; retained as NaN")
                    continue
                path, reference, low, high = records[day]
                try:
                    with xr.open_dataset(path, decode_cf=False) as dataset:
                        lat, lon = _grid(dataset)
                        if not np.array_equal(lat, latitude) or not np.array_equal(lon, longitude):
                            raise ValueError(
                                "Input MUR grids differ; explicit regridding is required"
                            )
                        if str(dataset.attrs["id"]) != product_id:
                            raise ValueError("Input MUR product identifiers differ")
                        sst = _decoded(dataset.analysed_sst.isel(time=0)) - 273.15
                        mask = _decoded(dataset["mask"].isel(time=0))
                        bitmask = np.where(np.isfinite(mask), mask, 0).astype(np.int64)
                        good = (
                            np.isfinite(sst)
                            & np.isfinite(mask)
                            & ((bitmask & 1) != 0)
                            & ((bitmask & 30) == 0)
                        )
                        ancillary = {name: _decoded(dataset[name].isel(time=0)) for name in FIELDS}
                        if maximum_analysis_error_k is not None:
                            error = ancillary["analysis_error"]
                            good &= np.isfinite(error) & (error <= maximum_analysis_error_k)
                        time[index] = _seconds(reference)
                        source_bounds[index] = [_seconds(low), _seconds(high)]
                        temperature[index] = np.where(good, sst, np.nan).astype(np.float32)
                        valid_mask[index] = good.astype(np.uint8)
                        raw_mask[index] = np.where(np.isfinite(mask), mask, -32768).astype(np.int16)
                        for name in FIELDS:
                            fields[name][index] = ancillary[name].astype(np.float32)
                except Exception as exc:
                    raise ValueError(f"Cannot prepare MUR input {path}: {exc}") from exc
                digest = sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                sources.append(
                    {
                        "date": day.date().isoformat(),
                        "path": str(path),
                        "sha256": digest.hexdigest(),
                        "size_bytes": path.stat().st_size,
                        "nominal_time_utc": reference.isoformat(),
                        "source_coverage_start_utc": low.isoformat(),
                        "source_coverage_end_utc": high.isoformat(),
                    }
                )
                report(f"{day:%Y-%m-%d}: MUR prepared; {good.mean():.1%} valid regional pixels")
            out.attrs["source_provenance_json"] = json.dumps(sources, sort_keys=True)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during preparation: {target}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "output": str(target),
        "requested_day_count": len(days),
        "source_day_count": len(records),
        "missing_day_count": len(days) - len(records),
        "grid_shape": [len(latitude), len(longitude)],
    }
