import copy
from hashlib import sha256
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
import xarray as xr
import zarr

from drifterlab.experiments.arcterx import preprocess, read_microsvp
from drifterlab.trajectories.records import ObservationSeries
from drifterlab.trajectories.zarr import DatasetLayout, write_master


def test_complete_roundtrip_and_inventory(mat_factory, config_factory, tmp_path, monkeypatch):
    p1, payload1 = mat_factory("1001", native_changes={"battery": np.array([8., -999., 7., 6.])})
    p2, payload2 = mat_factory("1002", native_changes={"drogue_off": -999})
    # Shorter second trajectory exercises padding independently on both axes.
    for obj in payload2.values():
        for key, value in obj.items():
            if isinstance(value, np.ndarray) and value.ndim == 1:
                obj[key] = value[:2]
    savemat(p2, payload2)
    config = config_factory()
    monkeypatch.chdir(tmp_path.parent)
    summary = preprocess(config)
    assert summary.trajectories == 2
    assert summary.native_observations == 5
    assert summary.reconstructed_observations == 7
    assert summary.source_loss_unknown == 1
    assert summary.unaligned_battery_trajectories == 1
    inventory = pd.read_parquet(summary.inventory_path)
    assert inventory.n_obs_qc.sum() == 5
    assert inventory.status.tolist() == ["ok", "ok"]
    assert inventory.source_sha256.tolist() == [sha256(p.read_bytes()).hexdigest() for p in [p1, p2]]
    with xr.open_zarr(summary.zarr_path, consolidated=True, chunks=None) as ds:
        assert dict(ds.sizes) == {"trajectory": 2, "obs_qc": 3, "obs_interp": 5}
        assert ds.platform_code.values.tolist() == ["1001", "1002"]
        assert ds.drogue_off_time.values[0] == np.datetime64("1970-01-05")
        assert np.isnat(ds.drogue_off_time.values[1])
        assert ds.position_valid_qc.dtype == bool
        assert ds.source_obs_index_qc.dtype.kind == "i"
        assert ds.source_obs_index_qc.values[0, -1] == 0
        assert ds.source_obs_index_qc.values[1, -1] == -1
        assert not ds.position_valid_qc.values[1, -1]
        assert not ds.analysis_valid_qc.values[1].any()
        assert np.isnat(ds.time_qc.values[1, -1])
        assert np.isnan(ds.lon_qc.values[1, -1])
        np.testing.assert_array_equal(ds.sst_qc.values[0], payload1["drifter"]["SST"][::-1])
        np.testing.assert_array_equal(ds.sst_interp.values[0], payload1["drifter_interp"]["SST"][::-1])
        for suffix in ("30m", "60m"):
            np.testing.assert_array_equal(ds[f"lon_interp_{suffix}"].values[0], payload1["drifter_interp"][f"longitude_{suffix.replace('m', 'min')}"][::-1])
        assert ds.attrs["effective_configuration"]["drogue_buffer_hours"] == 24
        assert ds.time_qc.attrs["time_reference"] == "UTC"
    root = zarr.open_group(summary.zarr_path, mode="r")
    assert root["lon_qc"].chunks == (1, 2)
    with xr.open_zarr(summary.zarr_path, group="source_unaligned/battery", consolidated=True, chunks=None) as aux:
        assert aux.platform_code.item() == "1001"
        assert aux.n_source_values.item() == 4
        np.testing.assert_array_equal(aux.battery.values[0], [8., np.nan, 7., 6.])
    before = Path(summary.inventory_path).read_bytes()
    with pytest.raises(FileExistsError):
        preprocess(config)
    assert Path(summary.inventory_path).read_bytes() == before


def test_missing_interpolation_has_zero_length_axis(mat_factory, config_factory):
    mat_factory(no_interp=True)
    summary = preprocess(config_factory())
    assert summary.reconstruction_30m_available == 0
    with xr.open_zarr(summary.zarr_path, chunks=None) as ds:
        assert ds.sizes["obs_interp"] == 0
        assert ds.lon_interp_30m.shape == (1, 0)
        assert ds.n_obs_interp.item() == 0


def test_inventory_records_every_input_before_failure(mat_factory, config_factory, tmp_path):
    mat_factory("1001")
    mat_factory("1002", native_changes={"SST": np.array([1., 2.])})
    with pytest.raises(ValueError, match="after writing inventory"):
        preprocess(config_factory())
    inventory = pd.read_parquet(tmp_path / "out/inventory.parquet")
    assert inventory.status.tolist() == ["ok", "error"]
    assert "SST" in inventory.error.iloc[1]
    assert not (tmp_path / "out/master.zarr").exists()


def test_duplicate_platforms_fail_without_merging(mat_factory, config_factory, tmp_path):
    path, _ = mat_factory()
    path.with_name("duplicate.mat").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="Duplicate platform ID"):
        preprocess(config_factory())
    assert not (tmp_path / "out/master.zarr").exists()


def test_writer_supports_different_reconstruction_axes(mat_factory, tmp_path):
    path, _ = mat_factory()
    record = read_microsvp(path)
    original = record.series.pop("interp")
    for representation, selection in [("interp_30m", np.arange(5)), ("interp_60m", np.array([0, 2, 4]))]:
        variables = {name: values[selection] for name, values in original.variables.items() if representation in name}
        record.series[representation] = ObservationSeries(
            original.time[selection], original.time_matlab[selection], original.source_index[selection], variables, "descending")
        record.position_axes[representation] = representation
    layout = DatasetLayout()
    layout.observe(record)
    output = tmp_path / "different_axes.zarr"
    write_master([record], layout, output, attributes={})
    with xr.open_zarr(output, chunks=None) as ds:
        assert ds.sizes["obs_interp_30m"] == 5
        assert ds.sizes["obs_interp_60m"] == 3
        assert ds.lon_interp_30m.dims == ("trajectory", "obs_interp_30m")
        assert ds.lon_interp_60m.dims == ("trajectory", "obs_interp_60m")
        np.testing.assert_array_equal(ds.time_interp_60m.values[0], original.time[[0, 2, 4]])


def test_failed_write_does_not_publish_partial_store(mat_factory, tmp_path):
    path, _ = mat_factory()
    record = read_microsvp(path)
    layout = DatasetLayout()
    layout.observe(record)
    output = tmp_path / "broken.zarr"

    def broken():
        yield record
        raise ValueError("simulated failure")

    with pytest.raises(ValueError, match="simulated failure"):
        write_master(broken(), layout, output, attributes={})
    assert not output.exists()
    assert not list(tmp_path.glob(".broken.zarr.*.tmp"))


def test_cli_works_outside_repository(mat_factory, config_factory, tmp_path):
    mat_factory()
    config = config_factory()
    result = subprocess.run([sys.executable, "-m", "drifterlab.cli.preprocess_arcterx_microsvp", str(config)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Native observations: 3" in result.stdout
    assert "Verified and published" in result.stdout
