"""Validation, publication, and readback tests for reconstruction workflow."""

import json
import importlib.util
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr
import yaml
import zarr

import drifterlab.experiments.arcterx as arcterx
from drifterlab.cli.reconstruct_trajectories import main
from drifterlab.plotting.trajectories import split_longitude_wrapped_path
from drifterlab.reconstruction import assign_start_arrays
import drifterlab.workflows.reconstruction as reconstruction_workflow
from drifterlab.workflows.reconstruction import (
    REPORT_NAME, ZARR_NAME, load_reconstruction_config, run_reconstruction_workflow,
)


def write_qc(
    path: Path, platform: str, minutes, lon=None, lat=None, *, schema="4.0",
    drogue_status="not_lost", cutoff_minute=None, statuses=None, eligible=None,
    source_valid=None, deployment_eligible=None, segments=None, footer_changes=None,
):
    minutes = np.asarray(minutes)
    n = len(minutes)
    status = np.asarray(statuses if statuses is not None else ["valid"] * n)
    valid = status == "valid"
    drogue = np.asarray(eligible if eligible is not None else [True] * n, dtype=bool)
    source_valid = np.asarray(
        source_valid if source_valid is not None else [True] * n, dtype=bool,
    )
    deployment_eligible = np.asarray(
        deployment_eligible if deployment_eligible is not None else [True] * n,
        dtype=bool,
    )
    cutoff = (
        pd.NaT if cutoff_minute is None
        else pd.Timestamp("2025-01-12T00:00:00Z") + pd.Timedelta(minutes=cutoff_minute)
    )
    frame = pd.DataFrame({
        "platform_code": [platform] * n,
        "source_sha256": [f"source-{platform}"] * n,
        "time": pd.Timestamp("2025-01-12T00:00:00Z") + pd.to_timedelta(minutes, unit="m"),
        "source_lon": np.asarray(lon if lon is not None else minutes / 100, dtype=float),
        "source_lat": np.asarray(lat if lat is not None else minutes / 200, dtype=float),
        "final_position_status": status,
        "final_position_valid": valid,
        "drogue_eligible": drogue,
        "valid_timestamp": [True] * n,
        "source_position_valid": source_valid,
        "deployment_eligible": deployment_eligible,
        "segment_status": np.asarray(
            segments if segments is not None else ["retained"] * n,
        ),
        "final_drogue_status": [drogue_status] * n,
        "analysis_cutoff_time": [cutoff] * n,
        "platform_qc_complete": [not np.isin(status, ["unresolved", "uncertain"]).any()] * n,
    })
    metadata = {
        "schema_version": schema, "algorithm_version": "native-position-qc-v2.5",
        "product_layout": "per-trajectory-v3", "platform_code": platform,
        "config_sha256": "qc-config", "resolution_policy": "aggressive",
        "source_sha256": f"source-{platform}", "platform_qc_complete": True,
    }
    metadata.update(footer_changes or {})
    table = pa.Table.from_pandas(frame, preserve_index=False)
    arrow_metadata = dict(table.schema.metadata or {})
    arrow_metadata[b"drifterlab_position_qc"] = json.dumps(metadata).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(arrow_metadata), path)


def write_config(
    tmp_path: Path, *, end=None, periods=(15,), thresholds=None, plotting=None,
    array_gap_hours=24,
):
    values = {
        "input": {"directory": "qc", "pattern": "*.parquet"},
        "grid": {"start_time": "2025-01-12T00:00:00Z", "end_time": end, "dt_minutes": 5},
        "spline": {
            "period_minutes": list(periods),
            "long_gap_threshold_minutes": thresholds or {period: period for period in periods},
        },
        "arrays": {"maximum_adjacent_start_gap_hours": array_gap_hours},
        "output": {"directory": "candidate", "chunks": {"platform": 1, "time": 4}},
    }
    if plotting is not None:
        values["plotting"] = plotting
    path = tmp_path / "reconstruction.yml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_common_grid_leading_trim_report_and_readback(tmp_path):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [7, 22, 37, 52, 67])
    write_qc(tmp_path / "qc/1002.parquet", "1002", [15, 30, 45, 60, 75])
    result = run_reconstruction_workflow(write_config(tmp_path))
    assert result.zarr_path.name == ZARR_NAME
    assert result.report_path.name == REPORT_NAME
    assert result.published_start_time == np.datetime64("2025-01-12T00:10:00")
    assert result.end_time == np.datetime64("2025-01-12T01:15:00")
    report = pd.read_csv(result.report_path)
    assert report.platform_id.astype(str).tolist() == ["1001", "1002"]
    assert {"gap_p50_minutes", "spline_15_total_fallback_points"} <= set(report)
    with xr.open_zarr(result.zarr_path, consolidated=True, chunks=None) as dataset:
        assert dict(dataset.sizes) == {"platform": 2, "time": 14}
        assert dataset.platform_id.values.tolist() == ["1001", "1002"]
        assert set(dataset.data_vars) == {
            "longitude_native", "latitude_native", "native_source_gap_minutes",
            "longitude_linear", "latitude_linear", "source_gap_minutes",
            "longitude_spline_15", "latitude_spline_15",
            "start_time", "start_lon", "start_lat",
            "array_id",
        }
        np.testing.assert_array_equal(dataset.array_id, [1, 1])
        np.testing.assert_array_equal(dataset.start_time.values, np.asarray([
            "2025-01-12T00:07:00", "2025-01-12T00:15:00",
        ], dtype="datetime64[ns]"))
        np.testing.assert_allclose(dataset.start_lon, [.07, .15])
        np.testing.assert_allclose(dataset.start_lat, [.035, .075])
        assert dataset.attrs["product_status"] == "candidate_pending_gap_review"
        assert dataset.attrs["configured_start_time"] == "2025-01-12T00:00:00Z"
        assert dataset.attrs["array_assignment_policy"]["maximum_adjacent_start_gap_hours"] == 24
        assert dataset.attrs["array_summary"][0]["platform_count"] == 2
    assert report.array_id.tolist() == [1, 1]
    assert {"gap_from_previous_start_hours", "gap_to_next_start_hours"} <= set(report)
    root = zarr.open_group(result.zarr_path, mode="r")
    assert root["longitude_linear"].chunks == (1, 4)
    assert np.isnan(root["longitude_linear"][0, -2:]).all()
    assert np.array_equal(
        np.isfinite(root["longitude_native"][:]),
        np.isfinite(root["longitude_linear"][:]),
    )


def test_native_uses_pre_point_qc_fixes_on_exact_qc_grid_span(tmp_path):
    write_qc(
        tmp_path / "qc/1001.parquet", "1001", [2, 7, 12, 17, 22],
        lon=[0, 50, .1, 60, .2], lat=[0, 10, .1, 20, .2],
        statuses=["valid", "rejected", "valid", "rejected", "valid"],
    )
    result = run_reconstruction_workflow(write_config(tmp_path))
    report = pd.read_csv(result.report_path).iloc[0]
    assert report.accepted_fix_count == 3
    assert report.native_fix_count == 5
    assert report.native_only_fix_count == 2
    with xr.open_zarr(result.zarr_path, consolidated=True, chunks=None) as dataset:
        native_lon = dataset.longitude_native.isel(platform=0).values
        linear_lon = dataset.longitude_linear.isel(platform=0).values
        np.testing.assert_array_equal(np.isfinite(native_lon), np.isfinite(linear_lon))
        # The common cells are 00:05 through 00:20; rejected native fixes at
        # 00:07 and 00:17 affect only the before-point-QC representation.
        assert not np.allclose(native_lon, linear_lon, equal_nan=True)
        np.testing.assert_allclose(dataset.native_source_gap_minutes, [[5, 5, 5, 5]])
        np.testing.assert_allclose(dataset.source_gap_minutes, [[10, 10, 10, 10]])


def test_native_excludes_nonretained_and_source_invalid_rows(tmp_path):
    write_qc(
        tmp_path / "qc/1001.parquet", "1001", [0, 5, 10, 15, 20],
        lon=[0, 100, .1, 120, .2],
        statuses=["valid", "rejected", "valid", "rejected", "valid"],
        source_valid=[True, False, True, True, True],
        segments=["retained", "retained", "retained", "excluded", "retained"],
    )
    result = run_reconstruction_workflow(write_config(tmp_path))
    with xr.open_zarr(result.zarr_path, consolidated=True, chunks=None) as dataset:
        # Only retained, source-valid fixes 0/10/20 enter native resampling.
        np.testing.assert_allclose(dataset.longitude_native, [[0, .05, .1, .15, .2]])
        np.testing.assert_allclose(dataset.longitude_native, dataset.longitude_linear)


def test_duplicate_native_timestamp_is_rejected_even_when_accepted_times_are_unique(tmp_path):
    write_qc(
        tmp_path / "qc/1001.parquet", "1001", [0, 10, 10, 20],
        statuses=["valid", "valid", "rejected", "valid"],
    )
    with pytest.raises(ValueError, match="Native pre-point-QC.*duplicate timestamps"):
        run_reconstruction_workflow(write_config(tmp_path))


def test_prestart_reports_every_affected_platform_before_writing(tmp_path):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [-10, 0, 10, 20])
    write_qc(tmp_path / "qc/1002.parquet", "1002", [-5, 5, 15, 25])
    with pytest.raises(ValueError, match=r"1001=.*23:50.*1002=.*23:55"):
        run_reconstruction_workflow(write_config(tmp_path))
    assert not (tmp_path / "candidate").exists()


def test_explicit_end_cannot_omit_eligible_fix(tmp_path):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [0, 10, 20])
    with pytest.raises(ValueError, match="would omit eligible data"):
        run_reconstruction_workflow(write_config(tmp_path, end="2025-01-12T00:15:00Z"))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"schema": "2.0"}, "unsupported schema"),
        ({"drogue_status": "uncertain"}, "Drogue decision"),
        ({"statuses": ["valid", "unresolved", "valid"]}, "unresolved"),
    ],
)
def test_rejects_old_schema_and_unresolved_inputs(tmp_path, change, match):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [0, 10, 20], **change)
    with pytest.raises(ValueError, match=match):
        run_reconstruction_workflow(write_config(tmp_path))


def test_rejects_missing_coordinates_identity_duplicates_and_invalid_accepted(tmp_path):
    path = tmp_path / "qc/1001.parquet"
    write_qc(path, "1001", [0, 10, 20])
    table = pq.read_table(path).drop(["source_lon"])
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="missing columns"):
        run_reconstruction_workflow(write_config(tmp_path))

    path.unlink()
    write_qc(path, "1002", [0, 10, 20])
    with pytest.raises(ValueError, match="filename, footer, and rows disagree"):
        run_reconstruction_workflow(write_config(tmp_path))

    path.unlink()
    write_qc(path, "1001", [0, 10, 10])
    with pytest.raises(ValueError, match="duplicate timestamps"):
        run_reconstruction_workflow(write_config(tmp_path))

    path.unlink()
    write_qc(path, "1001", [0, 10, 20], lon=[0, 181, 2])
    with pytest.raises(ValueError, match="invalid coordinates"):
        run_reconstruction_workflow(write_config(tmp_path))


def test_cutoff_ineligible_row_is_not_reconstructed(tmp_path):
    write_qc(
        tmp_path / "qc/1001.parquet", "1001", [0, 10, 20, 30],
        drogue_status="lost", cutoff_minute=20,
        statuses=["valid", "valid", "outside_drogue_window", "outside_drogue_window"],
        eligible=[True, True, False, False],
    )
    result = run_reconstruction_workflow(write_config(tmp_path))
    with xr.open_zarr(result.zarr_path, consolidated=True, chunks=None) as dataset:
        assert dataset.time.values[-1] == np.datetime64("2025-01-12T00:10:00")


def test_cli_reuses_valid_existing_output(tmp_path, capsys):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [0, 15, 30, 45])
    config = write_config(tmp_path)
    assert main([str(config)]) == 0
    assert "candidate_pending_gap_review" in capsys.readouterr().out
    before = (tmp_path / "candidate/trajectories.zarr/.zmetadata").read_bytes()
    result = run_reconstruction_workflow(config)
    assert result.action == "reused"
    assert (tmp_path / "candidate/trajectories.zarr/.zmetadata").read_bytes() == before


def test_changed_input_aborts_and_removes_temporary_bundle(tmp_path, monkeypatch):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [0, 15, 30, 45])
    config = write_config(tmp_path)
    original = reconstruction_workflow._file_sha256
    calls = 0

    def changing(path):
        nonlocal calls
        value = original(path)
        if Path(path).suffix == ".parquet":
            calls += 1
            if calls >= 3:
                return "changed-" + value
        return value

    monkeypatch.setattr(reconstruction_workflow, "_file_sha256", changing)
    with pytest.raises(ValueError, match="changed after inventory"):
        run_reconstruction_workflow(config)
    assert not (tmp_path / "candidate").exists()
    assert not list(tmp_path.glob(".candidate.*.tmp"))


@pytest.mark.parametrize("change", ["added", "removed"])
def test_changed_input_set_aborts_and_removes_temporary_bundle(tmp_path, monkeypatch, change):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [0, 15, 30, 45])
    write_qc(tmp_path / "qc/1002.parquet", "1002", [0, 15, 30, 45])
    config = write_config(tmp_path)
    original = reconstruction_workflow._discover
    calls = 0

    def changing(configuration):
        nonlocal calls
        calls += 1
        paths = original(configuration)
        if calls == 1:
            return paths
        if change == "added":
            return tuple(sorted((*paths, (tmp_path / "qc/1003.parquet").resolve())))
        return paths[:-1]

    monkeypatch.setattr(reconstruction_workflow, "_discover", changing)
    with pytest.raises(ValueError, match="file set changed"):
        run_reconstruction_workflow(config)
    assert not (tmp_path / "candidate").exists()
    assert not list(tmp_path.glob(".candidate.*.tmp"))


def test_old_mat_to_zarr_api_is_removed():
    assert not hasattr(arcterx, "preprocess")
    assert importlib.util.find_spec("drifterlab.cli.preprocess_arcterx_microsvp") is None
    assert importlib.util.find_spec("drifterlab.trajectories.zarr") is None


def test_obsolete_cluster_command_and_api_are_removed():
    assert importlib.util.find_spec("drifterlab.cli.clusters") is None
    assert importlib.util.find_spec("drifterlab.workflows.clusters") is None
    assert importlib.util.find_spec("drifterlab.clustering.core") is None


def test_config_requires_period_multiple_and_utc(tmp_path):
    config = write_config(tmp_path, periods=(12,))
    with pytest.raises(ValueError, match="integer multiple"):
        load_reconstruction_config(config)
    values = yaml.safe_load(config.read_text(encoding="utf-8"))
    values["spline"] = {"period_minutes": [15]}
    values["grid"]["start_time"] = "2025-01-12 00:00:00"
    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit UTC"):
        load_reconstruction_config(config)


def plotting_config(*, method="spline_15", checks=True, additional=None, dpi=90):
    return {
        "enabled": True,
        "output_directory": "figures",
        "dataset_label": "MicroSVP",
        "method": method,
        "time": {"start": None, "end": None},
        "platforms": [],
        "check_platforms": (
            [{"dataset": "MicroSVP", "platform_id": "1001"}] if checks else []
        ),
        "map": {
            "projection": "PlateCarree", "central_longitude": 0,
            "extent": None, "figsize": [5, 4], "dpi": dpi,
            "land": False, "coastlines": False, "gridlines": False,
            "label_starts": False,
        },
        "additional_stores": additional or [],
    }


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_plotting_after_build_then_regenerates_on_reuse(tmp_path):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [7, 22, 37, 52, 67])
    config = write_config(tmp_path, plotting=plotting_config(dpi=90))
    first = run_reconstruction_workflow(config)
    assert first.action == "built"
    assert [path.name for path in first.figure_paths] == [
        "trajectory_overview.png", "starting_positions.png",
        "starting_positions__array_01.png",
        "reconstruction_check__MicroSVP__1001.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 1000 for path in first.figure_paths)
    zarr_metadata = digest(first.zarr_path / ".zmetadata")
    overview = digest(first.figure_paths[0])

    write_config(tmp_path, plotting=plotting_config(dpi=120))
    second = run_reconstruction_workflow(config)
    assert second.action == "reused"
    assert digest(second.zarr_path / ".zmetadata") == zarr_metadata
    assert digest(second.figure_paths[0]) != overview


def test_plotting_can_select_current_products_native_grid_method(tmp_path):
    write_qc(
        tmp_path / "qc/1001.parquet", "1001", [0, 5, 10, 15, 20],
        lon=[0, .01, .02, .03, .04],
        statuses=["valid", "rejected", "valid", "rejected", "valid"],
    )
    plotting = plotting_config(method="native", checks=True)
    result = run_reconstruction_workflow(write_config(tmp_path, plotting=plotting))
    assert len(result.figure_paths) == 4
    assert all(path.is_file() and path.stat().st_size > 1000 for path in result.figure_paths)


def test_plotting_failure_leaves_new_valid_zarr_for_later_reuse(tmp_path):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [0, 15, 30, 45])
    config = write_config(tmp_path, plotting=plotting_config(method="spline_999"))
    with pytest.raises(ValueError, match="no method 'spline_999'"):
        run_reconstruction_workflow(config)
    metadata = digest(tmp_path / "candidate/trajectories.zarr/.zmetadata")

    write_config(tmp_path, plotting=plotting_config())
    result = run_reconstruction_workflow(config)
    assert result.action == "reused"
    assert digest(result.zarr_path / ".zmetadata") == metadata


def test_stale_configuration_requires_overwrite_and_failed_rebuild_keeps_old(tmp_path, monkeypatch):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [0, 15, 30, 45])
    config = write_config(tmp_path, thresholds={15: 15})
    first = run_reconstruction_workflow(config)
    metadata = digest(first.zarr_path / ".zmetadata")
    write_config(tmp_path, thresholds={15: 20})
    with pytest.raises(ValueError, match=r"configuration differs.*--overwrite"):
        run_reconstruction_workflow(config)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic reconstruction failure")

    monkeypatch.setattr(reconstruction_workflow, "reconstruct_platform", fail)
    with pytest.raises(RuntimeError, match="synthetic reconstruction failure"):
        run_reconstruction_workflow(config, overwrite=True)
    assert digest(first.zarr_path / ".zmetadata") == metadata
    assert not list(tmp_path.glob(".candidate.*.tmp"))


def test_overwrite_replaces_stale_bundle_after_validation(tmp_path):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [0, 15, 30, 45])
    config = write_config(tmp_path, thresholds={15: 15})
    first = run_reconstruction_workflow(config)
    first_metadata = digest(first.zarr_path / ".zmetadata")
    write_config(tmp_path, thresholds={15: 20})
    second = run_reconstruction_workflow(config, overwrite=True)
    assert second.action == "built"
    assert digest(second.zarr_path / ".zmetadata") != first_metadata
    attrs = dict(zarr.open_group(second.zarr_path, mode="r").attrs)
    assert attrs["effective_configuration"]["spline"]["long_gap_threshold_minutes"]["15"] == 20
    assert not list(tmp_path.glob(".candidate.*.backup"))


def test_additional_store_may_have_distinct_grid_method_and_overlapping_id(tmp_path):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [0, 15, 30, 45])
    other = xr.Dataset(
        {
            "longitude": (("platform", "time"), [[20., 20.1, 20.2]]),
            "latitude": (("platform", "time"), [[-5., -4.9, -4.8]]),
            "start_time": ("platform", [np.datetime64("2025-01-12T00:03:00", "ns")]),
            "start_lon": ("platform", [19.97]),
            "start_lat": ("platform", [-5.03]),
            "platform_id": ("platform", ["1001"]),
        },
        coords={
            "platform": [0],
            "time": np.asarray([
                "2025-01-12T00:10:00", "2025-01-12T00:20:00", "2025-01-12T00:30:00",
            ], dtype="datetime64[ns]"),
        },
    )
    other.to_zarr(tmp_path / "svp.zarr", mode="w", consolidated=True, zarr_format=2)
    other_metadata = digest(tmp_path / "svp.zarr/.zmetadata")
    plotting = plotting_config(
        checks=False,
        additional=[{"label": "SVP", "path": "svp.zarr", "method": "native"}],
    )
    plotting["platforms"] = [
        {"dataset": "MicroSVP", "platform_id": "1001"},
        {"dataset": "SVP", "platform_id": "1001"},
    ]
    result = run_reconstruction_workflow(write_config(tmp_path, plotting=plotting))
    assert len(result.figure_paths) == 3
    assert all(path.stat().st_size > 1000 for path in result.figure_paths)
    assert digest(tmp_path / "svp.zarr/.zmetadata") == other_metadata


def test_array_assignment_uses_strict_gap_and_stable_ties():
    platforms = ["b", "a", "c", "d"]
    starts = np.asarray([
        "2025-01-12T01:00:00", "2025-01-12T01:00:00",
        "2025-01-13T01:00:00", "2025-01-14T01:00:01",
    ], dtype="datetime64[ns]")
    assignments = assign_start_arrays(
        platforms, starts, maximum_adjacent_start_gap_hours=24,
    )
    by_platform = {item.platform_id: item for item in assignments}
    assert [item.platform_id for item in assignments] == platforms
    assert {platform: item.array_id for platform, item in by_platform.items()} == {
        "a": 1, "b": 1, "c": 1, "d": 2,
    }
    assert by_platform["a"].gap_from_previous_start_hours is None
    assert by_platform["a"].gap_to_next_start_hours == 0
    assert by_platform["b"].gap_from_previous_start_hours == 0
    assert by_platform["c"].gap_from_previous_start_hours == 24


def test_array_threshold_is_reconstruction_provenance(tmp_path):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [0, 15, 30, 45])
    config = write_config(tmp_path, array_gap_hours=24)
    run_reconstruction_workflow(config)
    write_config(tmp_path, array_gap_hours=12)
    with pytest.raises(ValueError, match=r"configuration differs.*--overwrite"):
        run_reconstruction_workflow(config)


def test_array_ids_follow_exact_start_times_and_are_in_report(tmp_path):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [7, 22, 37, 52])
    write_qc(tmp_path / "qc/1002.parquet", "1002", [67, 82, 97, 112])
    write_qc(tmp_path / "qc/1003.parquet", "1003", [127, 142, 157, 172])
    result = run_reconstruction_workflow(write_config(tmp_path, array_gap_hours=1))
    with xr.open_zarr(result.zarr_path, consolidated=True, chunks=None) as dataset:
        np.testing.assert_array_equal(dataset.array_id, [1, 1, 1])
        np.testing.assert_array_equal(dataset.start_time.values, np.asarray([
            "2025-01-12T00:07:00", "2025-01-12T01:07:00", "2025-01-12T02:07:00",
        ], dtype="datetime64[ns]"))
    report = pd.read_csv(result.report_path)
    assert report.array_id.tolist() == [1, 1, 1]


def test_plotting_writes_one_exact_start_map_per_array(tmp_path):
    write_qc(tmp_path / "qc/1001.parquet", "1001", [7, 22, 37, 52])
    write_qc(tmp_path / "qc/1002.parquet", "1002", [1507, 1522, 1537, 1552])
    plotting = plotting_config(checks=False)
    result = run_reconstruction_workflow(
        write_config(tmp_path, plotting=plotting, array_gap_hours=24),
    )
    assert [path.name for path in result.figure_paths] == [
        "trajectory_overview.png", "starting_positions.png",
        "starting_positions__array_01.png", "starting_positions__array_02.png",
    ]
    with xr.open_zarr(result.zarr_path, consolidated=True, chunks=None) as dataset:
        np.testing.assert_array_equal(dataset.array_id, [1, 2])


def test_longitude_wrap_and_missing_values_split_paths_without_bridge():
    segments = split_longitude_wrapped_path(
        np.asarray([178., 179., -179., -178., np.nan, 10., 11.]),
        np.asarray([0., 1., 2., 3., np.nan, 4., 5.]),
    )
    assert [longitude.tolist() for longitude, _ in segments] == [
        [178., 179.], [-179., -178.], [10., 11.],
    ]
