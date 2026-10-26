"""Prepared-file contract, equal-bin averaging and explicit coverage checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from drifterlab.analysis.affine_sst import PreparedSST, resolve_sst
from drifterlab.trajectories.kinematics import project_array_positions


def write_bins(path: Path, hours=(0, 1, 24), width=1, **changes) -> Path:
    epoch = pd.Timestamp("2025-02-01").value // 1_000_000_000
    starts = epoch + np.asarray(hours) * 3600
    values = np.broadcast_to(
        np.array([10.0, 30.0, 40.0])[: len(hours), None, None], (len(hours), 2, 2)
    ).copy()
    values[1, 0, 0] = np.nan
    values[:, 0, 1] = np.nan
    counts = np.broadcast_to(np.array([1, 9, 100])[: len(hours), None, None], values.shape).copy()
    ds = xr.Dataset(
        {
            "sea_surface_temperature": (("time", "lat", "lon"), values),
            "observation_count": (("time", "lat", "lon"), counts),
            "time_bounds": (("time", "nv"), np.column_stack((starts, starts + width * 3600))),
        },
        coords={"time": starts, "lat": [70.01, 69.99], "lon": [209.99, 210.01]},
        attrs={"title": "Prepared test SST", "quality_policy": "Applied upstream"},
    )
    ds.time.attrs.update(units="seconds since 1970-01-01 00:00:00 UTC", bounds="time_bounds")
    ds.sea_surface_temperature.attrs["units"] = changes.get("units", "degree_C")
    if changes.get("overlap"):
        ds.time_bounds.values[1, 0] = starts[0] + 1800
        ds.time_bounds.values[1, 1] = starts[0] + 5400
    if changes.get("unequal"):
        ds.time_bounds.values[-1, 1] += 3600
    if changes.get("no_bounds"):
        ds = ds.drop_vars("time_bounds")
    ds.to_netcdf(path)
    return path


def reader(path: Path) -> PreparedSST:
    return PreparedSST(resolve_sst({"enabled": True, "file": str(path)}, path.parent))


def load(prepared: PreparedSST, day="2025-02-01"):
    projection = project_array_positions(np.array([[-150.0]]), np.array([[70.0]]))
    return prepared.load(pd.Timestamp(day, tz="UTC"), [-2500, 2500, -2500, 2500], projection)


def test_equal_hour_weights_missing_bins_counts_and_map_boundary(tmp_path):
    prepared = reader(write_bins(tmp_path / "hourly.nc"))
    product, meta = load(prepared)
    # Latitude is sorted ascending: input row 1 is output row 0.
    assert product.sst_celsius.values[0, 0] == 20  # Not observation-weighted 28.
    assert product.observation_count.values[0, 0] == 10
    assert product.contributing_bin_count.values[0, 0] == 2
    assert product.temporal_coverage_fraction.values[0, 0] == pytest.approx(2 / 24)
    assert product.sst_celsius.values[1, 0] == 10
    assert np.isnan(product.sst_celsius.values[1, 1])
    assert meta["expected_bin_count"] == 24 and meta["available_bin_count"] == 2
    assert meta["coverage_fraction"] == 0.75
    assert meta["mean_temporal_coverage_fraction"] == pytest.approx(5 / 96)
    assert meta["upstream_metadata"]["quality_policy"] == "Applied upstream"
    assert len(meta["source_sha256"]) == 64
    next_day, meta = load(prepared, "2025-02-02")
    assert next_day.sst_celsius.values[0, 0] == 40  # Next midnight belongs only to the next map.
    assert meta["available_bin_count"] == 1
    missing, meta = load(prepared, "2025-02-03")
    assert missing is None and meta["status"] == "sst_unavailable"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"units": "kelvin"}, "Celsius"),
        ({"overlap": True}, "nonoverlapping"),
        ({"unequal": True}, "common positive duration"),
        ({"no_bounds": True}, "bounds"),
    ],
)
def test_malformed_prepared_file_rejected(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        reader(write_bins(tmp_path / "bad.nc", **changes))


def test_daily_composite_cannot_be_split_at_a_map_boundary(tmp_path):
    prepared = reader(write_bins(tmp_path / "offset_days.nc", hours=(1, 25), width=24))
    with pytest.raises(ValueError, match="cross a map boundary"):
        load(prepared)


def test_absent_disabled_and_missing_explicit_files(tmp_path):
    product, meta = load(PreparedSST(resolve_sst({}, tmp_path)))
    assert product is None and meta["status"] == "sst_disabled"
    product, meta = load(PreparedSST(resolve_sst({"enabled": True}, tmp_path)))
    assert product is None and meta["status"] == "sst_unavailable"
    with pytest.raises(ValueError, match="Invalid prepared SST"):
        reader(tmp_path / "missing.nc")
