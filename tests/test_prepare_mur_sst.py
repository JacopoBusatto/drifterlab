"""MUR preparation preserves source semantics and the common SST contract."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from drifterlab.analysis.affine_sst import PreparedSST, resolve_sst
from drifterlab.trajectories.kinematics import project_array_positions
from drifterlab.workflows.prepare_mur_sst import prepare_mur_sst


def mur_file(root: Path, day="2025-02-01", *, units="kelvin", shifted=False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(day)
    path = root / f"{stamp:%Y%m%d}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_subset.nc"
    shape = (1, 3, 4)
    sst = np.zeros(shape, dtype=np.int16)
    sst[0, 1, 0] = -32768
    mask = np.array([[[1, 2, 4, 9], [1, 1, 1, 1], [1, 16, 1, 1]]], dtype=np.int8)
    error = np.full(shape, 20, dtype=np.int16)
    error[0, 1, 2] = 200
    age = np.full(shape, 4, dtype=np.int8)
    age[0, 1, 1] = -128
    epoch = pd.Timestamp("1981-01-01")
    time = int((stamp + pd.Timedelta(hours=9) - epoch).total_seconds())
    ds = xr.Dataset(
        {
            "analysed_sst": (("time", "lat", "lon"), sst),
            "mask": (("time", "lat", "lon"), mask),
            "analysis_error": (("time", "lat", "lon"), error),
            "dt_1km_data": (("time", "lat", "lon"), age),
            "sea_ice_fraction": (("time", "lat", "lon"), np.zeros(shape, dtype=np.int8)),
        },
        coords={
            "time": [time],
            "lat": [20.5, 20.6, 20.7],
            "lon": np.array([129.9, 130.0, 130.1, 130.2]) + int(shifted),
        },
        attrs={
            "processing_level": "L4",
            "id": "MUR-JPL-L4-GLOB-v04.1",
            "product_version": "04.1",
            "time_coverage_start": (stamp - pd.Timedelta(hours=3)).isoformat() + "Z",
            "time_coverage_end": (stamp + pd.Timedelta(hours=21)).isoformat() + "Z",
        },
    )
    ds.time.attrs["units"] = "seconds since 1981-01-01 00:00:00 UTC"
    ds.analysed_sst.attrs.update(
        units=units,
        scale_factor=0.001,
        add_offset=298.15,
        _FillValue=-32768,
        valid_min=-32767,
        valid_max=32767,
    )
    ds["mask"].attrs.update(
        _FillValue=-128,
        flag_masks=np.array([1, 2, 4, 8, 16], dtype=np.int8),
        flag_meanings="open_sea land open_lake open_sea_with_ice_in_the_grid open_lake_with_ice_in_the_grid",
    )
    ds.analysis_error.attrs.update(
        units="kelvin", scale_factor=0.01, _FillValue=-32768, valid_min=0, valid_max=32767
    )
    ds.dt_1km_data.attrs.update(units="hours", _FillValue=-128)
    ds.sea_ice_fraction.attrs.update(scale_factor=0.01, _FillValue=-128)
    ds.to_netcdf(path)
    return path


def test_mur_conversion_masks_missing_days_uncertainty_and_source_bounds(tmp_path):
    root = tmp_path / "raw"
    mur_file(root)
    mur_file(root, "2025-02-03")
    output = tmp_path / "prepared.nc"
    result = prepare_mur_sst(root, output, start="2025-02-01", end="2025-02-04")
    assert result["source_day_count"] == 2 and result["missing_day_count"] == 2
    with xr.open_dataset(output, decode_timedelta=False) as ds:
        assert ds.sizes["time"] == 4
        assert float(ds.sea_surface_temperature[0, 0, 0]) == pytest.approx(25)
        assert np.isnan(ds.sea_surface_temperature[0, 0, 1:]).all()
        assert np.isnan(ds.sea_surface_temperature[0, 1, 0])
        assert np.isfinite(
            ds.sea_surface_temperature[0, 1, 1]
        )  # Missing 1km input does not invalidate an L4 analysis.
        assert np.isnan(ds.dt_1km_data[0, 1, 1])
        assert float(ds.analysis_error[0, 1, 1]) == pytest.approx(
            0.2
        )  # No subtraction of 273.15 from uncertainty.
        assert float(ds.analysis_error[0, 1, 2]) == pytest.approx(2)
        assert np.isfinite(
            ds.sea_surface_temperature[0, 1, 2]
        )  # No default scientific uncertainty ceiling.
        assert np.isnan(ds.sea_surface_temperature[1]).all()
        assert np.isnan(ds.sea_surface_temperature[3]).all()
        np.testing.assert_array_equal(ds.source_present, [1, 0, 1, 0])
        assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("2025-02-01T09:00")
        assert pd.Timestamp(ds.time_bounds.values[0, 0]) == pd.Timestamp("2025-02-01T00:00")
        assert pd.Timestamp(ds.source_time_bounds.values[0, 0]) == pd.Timestamp("2025-01-31T21:00")
        assert (
            ds.sea_surface_temperature.attrs["standard_name"]
            == "sea_surface_foundation_temperature"
        )
        assert "No daily averaging" in ds.attrs["calendar_assignment_policy"]
        assert "observation_count" not in ds
    prepared = PreparedSST(resolve_sst({"enabled": True, "file": str(output)}, tmp_path))
    projection = project_array_positions(np.array([[130.05]]), np.array([[20.6]]))
    background, metadata = prepared.load(
        pd.Timestamp("2025-02-01", tz="UTC"), [-30000, 30000, -30000, 30000], projection
    )
    assert metadata["status"] == "sst_available" and metadata["expected_bin_count"] == 1
    assert metadata["aggregation"] == "supplied_bin_unchanged"
    assert background.sst_celsius.values[0, 0] == pytest.approx(25)


def test_optional_uncertainty_mask_and_overwrite_protection(tmp_path):
    root = tmp_path / "raw"
    mur_file(root)
    output = tmp_path / "prepared.nc"
    prepare_mur_sst(root, output, maximum_analysis_error_k=0.5)
    with xr.open_dataset(output) as ds:
        assert np.isnan(ds.sea_surface_temperature[0, 1, 2])
    with pytest.raises(FileExistsError):
        prepare_mur_sst(root, output)
    with pytest.raises(ValueError, match="positive"):
        prepare_mur_sst(root, tmp_path / "invalid.nc", maximum_analysis_error_k=0)


def test_duplicate_dates_bad_units_and_atomic_failure(tmp_path):
    root = tmp_path / "raw"
    first = mur_file(root)
    mur_file(root / "duplicate")
    with pytest.raises(ValueError, match="Multiple MUR"):
        prepare_mur_sst(root, tmp_path / "duplicate.nc")
    bad_root = tmp_path / "bad_units"
    mur_file(bad_root, units="unknown")
    with pytest.raises(ValueError, match="Kelvin"):
        prepare_mur_sst(bad_root, tmp_path / "bad.nc")
    changed_root = tmp_path / "changed_grid"
    mur_file(changed_root)
    output = tmp_path / "retained.nc"
    prepare_mur_sst(changed_root, output)
    original = output.read_bytes()
    mur_file(changed_root, "2025-02-02", shifted=True)
    with pytest.raises(ValueError, match="grids differ"):
        prepare_mur_sst(changed_root, output, overwrite=True)
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".*.part"))
    assert first.exists()
