"""Configuration, array separation, and rendering tests for postprocessing."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from drifterlab.plotting.array_trajectories import (
    build_color_encoding,
    movie_frame_indices,
)
from drifterlab.postprocessing import load_postprocessing_config, run_postprocessing
from drifterlab.postprocessing.trajectories import load_array_trajectories


def _write_store(path: Path, *, invalid_color_dims: bool = False) -> Path:
    times = np.arange(
        np.datetime64("2025-02-01T00:00:00", "m"),
        np.datetime64("2025-02-01T00:20:00", "m"),
        np.timedelta64(5, "m"),
    ).astype("datetime64[ns]")
    longitude = np.asarray([
        [130.00, 130.01, 130.02, 130.03],
        [130.02, 130.03, 130.04, 130.05],
        [131.00, 131.02, 131.04, 131.06],
    ])
    latitude = np.asarray([
        [20.00, 20.01, 20.02, 20.03],
        [20.02, 20.03, 20.04, 20.05],
        [21.00, 21.01, 21.02, 21.03],
    ])
    variables = {
        "platform_id": (("platform",), np.asarray(["p1", "p2", "p3"])),
        "time": (("time",), times),
        "start_time": (("platform",), np.asarray([
            "2025-02-01T00:01:00", "2025-02-01T00:02:00", "2025-02-01T00:03:00",
        ], dtype="datetime64[ns]")),
        "array_id": (("platform",), np.asarray([1, 1, 2], dtype=np.int32)),
        "cluster_id": (
            ("time",) if invalid_color_dims else ("platform",),
            np.asarray(["bad"] * 4) if invalid_color_dims else np.asarray([
                "array_001__cluster_001", "array_001__cluster_002",
                "array_002__cluster_001",
            ]),
        ),
        "member_id": (("platform",), np.asarray([1, 1, 1], dtype=np.int32)),
        "longitude_spline_30": (("platform", "time"), longitude),
        "latitude_spline_30": (("platform", "time"), latitude),
    }
    dataset = xr.Dataset(
        variables,
        attrs={
            "schema_version": "1.4",
            "algorithm_version": "test-reconstruction",
            "product_status": "candidate_pending_gap_review",
            "build_report_sha256": "build-hash",
            "array_assignment_sha256": "array-hash",
            "initial_cluster_assignment_sha256": "cluster-hash",
        },
    )
    dataset.to_zarr(path, mode="w", consolidated=True)
    return path


def _configuration(tmp_path: Path, **plotting_changes) -> dict:
    plotting = {
        "enabled": True,
        "color_by": "cluster_id",
        "color_mode": "categorical",
        "cmap": "tab20",
        "movie": {"enabled": False},
        "map": {
            "land": False,
            "coastlines": False,
            "gridlines": False,
            "scale_bar": False,
            "dpi": 50,
            "figsize": [4, 3],
        },
    }
    plotting.update(plotting_changes)
    return {
        "input": {
            "trajectories": "trajectories.zarr",
            "dataset_label": "Synthetic",
            "coordinate_method": "spline_30",
        },
        "output": {"directory": "products"},
        "arrays": {
            "array_001": {"nominal_deployment_time": "2025-02-01T00:00:00Z"},
        },
        "trajectory_plotting": plotting,
    }


def _write_config(tmp_path: Path, values: dict) -> Path:
    path = tmp_path / "postprocessing.yml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_config_defaults_and_resolved_paths(tmp_path):
    config_path = _write_config(tmp_path, _configuration(tmp_path))
    config = load_postprocessing_config(config_path)
    assert config.trajectory_path == (tmp_path / "trajectories.zarr").resolve()
    assert config.output_directory == (tmp_path / "products").resolve()
    assert config.trajectory_plotting.movie.frame_interval_minutes == 60
    assert config.trajectory_plotting.movie.fps == 15
    assert config.trajectory_plotting.movie.tail_hours == 24
    assert str(config.arrays[1].nominal_deployment_time) == "2025-02-01T00:00:00.000000000"
    assert config.effective()["trajectory_plotting"]["color_by"] == "cluster_id"


@pytest.mark.parametrize(
    "change,match",
    [
        ({"unknown": True}, "Unknown trajectory_plotting keys"),
        ({"color_mode": "rainbow"}, "color_mode must be"),
        ({"movie": {"format": "gif"}}, "supports only mp4"),
        ({"time": {"start": "2025-02-02T00:00:00Z", "end": "2025-02-01T00:00:00Z"}},
         "end must not precede"),
    ],
)
def test_config_rejects_invalid_plotting_values(tmp_path, change, match):
    values = _configuration(tmp_path)
    values["trajectory_plotting"].update(change)
    with pytest.raises(ValueError, match=match):
        load_postprocessing_config(_write_config(tmp_path, values))


def test_loader_separates_arrays_and_labels_time_reference(tmp_path):
    _write_store(tmp_path / "trajectories.zarr")
    config = load_postprocessing_config(_write_config(tmp_path, _configuration(tmp_path)))
    arrays, metadata = load_array_trajectories(config)
    assert [item.array_id for item in arrays] == [1, 2]
    assert [len(item.platform_ids) for item in arrays] == [2, 1]
    assert arrays[0].reference_kind == "nominal_deployment"
    assert arrays[1].reference_kind == "first_retained_fix"
    assert str(arrays[1].reference_time) == "2025-02-01T00:03:00.000000000"
    assert metadata.array_ids == (1, 2)
    assert metadata.initial_cluster_assignment_sha256 == "cluster-hash"


def test_loader_requires_platform_level_color_variable(tmp_path):
    _write_store(tmp_path / "trajectories.zarr", invalid_color_dims=True)
    config = load_postprocessing_config(_write_config(tmp_path, _configuration(tmp_path)))
    with pytest.raises(ValueError, match="must be constant per platform"):
        load_array_trajectories(config)


def test_movie_frame_indices_require_grid_multiple_and_include_end():
    times = np.arange(
        np.datetime64("2025-01-01T00:00", "m"),
        np.datetime64("2025-01-01T00:20", "m"),
        np.timedelta64(5, "m"),
    )
    assert movie_frame_indices(times, 10).tolist() == [0, 2, 3]
    with pytest.raises(ValueError, match="integer multiple"):
        movie_frame_indices(times, 7)


def test_identifier_auto_color_is_categorical():
    encoding = build_color_encoding(
        np.asarray([1, 2, 1]), name="member_id", mode="auto", cmap_name="tab10",
        label=None, vmin=None, vmax=None,
    )
    assert encoding.mode == "categorical"
    assert encoding.categories == ("1", "2")
    np.testing.assert_allclose(encoding.rgba[0], encoding.rgba[2])


def test_workflow_creates_one_png_per_array_and_manifest_without_changing_input(tmp_path):
    store = _write_store(tmp_path / "trajectories.zarr")
    metadata_path = store / ".zmetadata"
    before = _sha256(metadata_path)
    config_path = _write_config(tmp_path, _configuration(tmp_path))

    result = run_postprocessing(config_path)

    assert result.array_count == 2
    assert result.movie_paths == ()
    assert len(result.figure_paths) == 2
    assert all(path.is_file() and path.stat().st_size > 0 for path in result.figure_paths)
    assert _sha256(metadata_path) == before
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    entries = manifest["analyses"]["trajectory_plotting"]["arrays"]
    assert [entry["array_id"] for entry in entries] == [1, 2]
    assert entries[0]["time_reference_kind"] == "nominal_deployment"
    assert entries[1]["time_reference_kind"] == "first_retained_fix"
    assert manifest["input"]["consolidated_metadata_sha256"] == before

    with pytest.raises(ValueError, match="already exist"):
        run_postprocessing(config_path)
