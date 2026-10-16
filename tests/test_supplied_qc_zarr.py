from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.io import savemat

from drifterlab.trajectories.product_store import load_trajectory_product
from drifterlab.workflows.qc_comparison import run_qc_comparison
from drifterlab.workflows.supplied_qc_zarr import run_supplied_qc_import

DAY_NS = 86_400_000_000_000


def _datenum(times: np.ndarray) -> np.ndarray:
    return 719529 + times.astype("datetime64[ns]").astype(np.int64) / DAY_NS


def _write_mat_files(directory: Path) -> tuple[list[str], np.ndarray]:
    directory.mkdir()
    platforms = ["1001", "1002", "1003"]
    times = (
        np.datetime64("2025-02-06T00:00:00", "ns")
        + np.arange(13).astype("timedelta64[m]") * 5
    )
    source_time = _datenum(times)[::-1]
    for index, platform in enumerate(platforms):
        elapsed = np.arange(len(times), dtype=float)
        longitude = -150 + index * 0.001 + elapsed * 0.00002
        latitude = 70 + index * 0.0005 + elapsed * 0.00001
        spline30 = longitude + 0.0001
        spline60 = longitude + 0.0002
        if index == 0:
            spline30 = spline30.copy()
            spline30[5] = 999
        interp_times = times + (
            np.timedelta64(1, "m") if index == 1 else np.timedelta64(0, "m")
        )
        payload = {
            "drifter": {
                "PlatformId": np.int64(platform),
                "ID": np.int64(platform),
                "type": "MicroSVP",
                "time": source_time,
                "longitude": longitude[::-1],
                "latitude": latitude[::-1],
                "SST": np.full(len(times), 2.0),
                "SLP": np.full(len(times), -999),
                "battery": np.full(len(times), 10.0),
                "drogue": np.full(len(times), 1.0),
                "speed": np.full(len(times), 0.1),
                "drogue_off": -999,
            },
            "drifter_interp": {
                "time": _datenum(interp_times)[::-1],
                "longitude_30min": spline30[::-1],
                "latitude_30min": latitude[::-1],
                "longitude_60min": spline60[::-1],
                "latitude_60min": latitude[::-1],
                "SST": np.full(len(times), 2.0),
                "speed_30min": np.full(len(times), 0.1),
                "speed_60min": np.full(len(times), 0.1),
            },
        }
        savemat(directory / f"ARCTERYX_microsvp_{platform}.mat", payload)
    return platforms, times


def _config(input_directory: Path, output: Path) -> dict:
    return {
        "schema_version": 1,
        "input": {
            "directory": str(input_directory),
            "pattern": "*.mat",
            "missing_value": -999,
        },
        "reference_alignment": {
            "mode": "automatic_native",
            "reference_zarr": None,
            "reference_products": {},
        },
        "grid": {
            "anchor_time": "2025-02-06T00:00:00Z",
            "spline_snap_tolerance_seconds": 1,
            "spline_time_alignment": "linear_resample",
        },
        "coverage": {
            "end_policy": "record_end",
            "drogue_buffer_hours": 0,
            "unknown_drogue_loss": "record_end",
        },
        "products": [
            {
                "name": "qc_5min",
                "representation": "native_qc",
                "cadence_minutes": 5,
                "maximum_gap_minutes": 30,
            },
            {
                "name": "supplied_spline30_5min",
                "representation": "supplied_spline30",
                "cadence_minutes": 5,
                "maximum_gap_minutes": 5,
            },
        ],
        "arrays": {"maximum_adjacent_start_gap_hours": 24},
        "initial_clustering": {
            "distance_reference": "observed_starts",
            "defaults": {
                "maximum_start_time_difference_seconds": 21600,
                "maximum_pair_distance_m": 500,
                "maximum_cluster_diameter_m": 1000,
                "maximum_members": 5,
            },
            "array_overrides": {},
        },
        "output": {
            "zarr": str(output),
            "chunks": {"platform": 1, "time": 32},
        },
    }


def _write_yaml(path: Path, values: dict) -> Path:
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_reference(path: Path, platforms: list[str], times: np.ndarray) -> Path:
    starts = times[0] + np.arange(len(platforms)).astype("timedelta64[m]")
    root = xr.Dataset({
        "platform_id": ("platform", np.asarray(platforms)),
        "array_id": ("platform", np.asarray([4, 4, 4], dtype=np.int32)),
        "cluster_id": (
            "platform", np.asarray(["array_004__cluster_002"] * 3),
        ),
        "member_id": ("platform", np.asarray([1, 2, 3], dtype=np.int32)),
        "cluster_size": ("platform", np.asarray([3, 3, 3], dtype=np.int32)),
        "cluster_assignment_status": (
            "platform", np.asarray(["candidate_cluster"] * 3),
        ),
        "deployment_start_time": ("platform", starts),
        "review_end_time": ("platform", np.asarray([times[-1]] * 3)),
    }, attrs={"schema_version": "2.0"})
    root.to_zarr(path, mode="w", consolidated=False, zarr_format=2)
    product = xr.Dataset({
        "platform_id": ("platform", np.asarray(platforms)),
        "time": ("time", times),
    })
    product.to_zarr(
        path, group="products/qc_5min", mode="a", consolidated=True,
        zarr_format=2,
    )
    return path


def test_automatic_native_import_is_v2_analysis_compatible(tmp_path: Path) -> None:
    source = tmp_path / "mat"
    platforms, times = _write_mat_files(source)
    source_hashes = {path.name: _digest(path) for path in source.glob("*.mat")}
    output = tmp_path / "automatic.zarr"
    result = run_supplied_qc_import(
        _write_yaml(tmp_path / "import.yml", _config(source, output)),
    )

    assert result.platform_count == 3
    assert result.product_count == 2
    assert result.alignment_mode == "automatic_native"
    assert (output / "platform_audit.csv").is_file()
    assert (output / "build_manifest.json").is_file()
    assert source_hashes == {
        path.name: _digest(path) for path in source.glob("*.mat")
    }

    native, native_info = load_trajectory_product(output, "qc_5min")
    spline, spline_info = load_trajectory_product(
        output, "supplied_spline30_5min",
    )
    assert native_info.product_status == "final_supplied_automatic_qc"
    assert spline_info.product_attributes["source_representation"] == (
        "supplied_spline30"
    )
    assert native.platform_id.values.astype(str).tolist() == platforms
    np.testing.assert_array_equal(native.start_time.values, [times[0]] * 3)
    assert np.isnan(spline.longitude.values[0, 5])
    assert np.isnan(spline.latitude.values[0, 5])
    assert np.isfinite(spline.longitude.values[0, 4])
    assert np.isfinite(spline.longitude.values[0, 6])
    assert np.isfinite(spline.longitude.values[1, 1:-1]).all()
    audit = pd.read_csv(output / "platform_audit.csv")
    assert int((audit.supplied_spline30_time_grid_snapped == 0).sum()) == 1


def test_manual_reference_mode_copies_starts_and_assignments(tmp_path: Path) -> None:
    source = tmp_path / "mat"
    platforms, times = _write_mat_files(source)
    reference = _write_reference(tmp_path / "reference.zarr", platforms, times)
    output = tmp_path / "aligned.zarr"
    values = _config(source, output)
    values["reference_alignment"] = {
        "mode": "manual_reference",
        "reference_zarr": str(reference),
        "reference_products": {"5": "qc_5min"},
    }
    result = run_supplied_qc_import(
        _write_yaml(tmp_path / "aligned.yml", values),
    )

    assert result.alignment_mode == "manual_reference"
    dataset, _ = load_trajectory_product(output, "qc_5min")
    expected_starts = times[0] + np.arange(3).astype("timedelta64[m]")
    np.testing.assert_array_equal(dataset.start_time.values, expected_starts)
    np.testing.assert_array_equal(dataset.array_id.values, [4, 4, 4])
    assert set(dataset.cluster_id.values.astype(str)) == {
        "array_004__cluster_002"
    }


def test_comparison_keeps_each_products_native_support(tmp_path: Path) -> None:
    source = tmp_path / "mat"
    platforms, times = _write_mat_files(source)
    native_store = tmp_path / "native.zarr"
    run_supplied_qc_import(_write_yaml(
        tmp_path / "native.yml", _config(source, native_store),
    ))

    reference = _write_reference(tmp_path / "reference.zarr", platforms, times)
    aligned_store = tmp_path / "aligned.zarr"
    aligned = _config(source, aligned_store)
    aligned["reference_alignment"] = {
        "mode": "manual_reference",
        "reference_zarr": str(reference),
        "reference_products": {"5": "qc_5min"},
    }
    run_supplied_qc_import(_write_yaml(tmp_path / "aligned.yml", aligned))

    output = tmp_path / "comparison"
    comparison_config = {
        "schema_version": 1,
        "manual": {
            "label": "reference aligned",
            "trajectory_zarr": str(aligned_store),
            "trajectory_product": "qc_5min",
            "analysis_root": None,
        },
        "automatic": {
            "label": "automatic native",
            "trajectory_zarr": str(native_store),
            "trajectory_product": "qc_5min",
            "analysis_root": None,
        },
        "output": {"root": str(output)},
    }
    result = run_qc_comparison(_write_yaml(
        tmp_path / "comparison.yml", comparison_config,
    ))

    assert result.platform_count == 3
    assert result.changed_start_count == 2
    assert result.changed_array_count == 3
    table = pd.read_csv(output / "platform_comparison.csv")
    assert "manual_finite_position_count" in table
    assert "automatic_finite_position_count" in table
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert "no common-support mask" in summary["support_policy"]

