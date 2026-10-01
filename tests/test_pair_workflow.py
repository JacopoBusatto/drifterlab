"""Grouped candidate-pair workflow tests."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from drifterlab.cli.pairs import main
from drifterlab.workflows.pairs import (
    CATALOG_NAME, ZARR_NAME, load_pair_config, run_pair_workflow,
)


def write_zarr(path: Path) -> None:
    time = np.asarray([
        "2025-01-01T00:00:00", "2025-01-01T00:05:00",
        "2025-01-01T00:10:00", "2025-01-01T00:15:00",
        "2025-01-01T00:20:00",
    ], dtype="datetime64[ns]")
    spline_lon = np.asarray([
        [0.000, 0.000, 0.000, 0.000, 0.000],
        [0.020, 0.004, 0.003, 0.002, 0.001],
        [0.004, 0.004, 0.004, np.nan, np.nan],
    ])
    linear_lon = np.asarray([
        [0.000, 0.000, 0.000, 0.000, 0.000],
        [0.010, 0.001, 0.001, 0.001, 0.001],
        [0.002, 0.002, 0.002, np.nan, np.nan],
    ])
    latitude = np.asarray([
        [70.0, 70.0, 70.0, 70.0, 70.0],
        [70.0, 70.0, 70.0, 70.0, 70.0],
        [70.0, 70.0, 70.0, np.nan, np.nan],
    ])
    dataset = xr.Dataset(
        {
            "platform_id": ("platform", ["100", "200", "300"]),
            "start_time": ("platform", np.repeat(time[:1], 3)),
            "start_lon": ("platform", spline_lon[:, 0]),
            "start_lat": ("platform", [70.0, 70.0, 70.0]),
            "array_id": ("platform", np.asarray([1, 1, 2], dtype=np.int32)),
            "cluster_id": ("platform", ["A01_C01", "A01_C01", "A02_C01"]),
            "member_id": ("platform", np.asarray([1, 2, 1], dtype=np.int32)),
            "cluster_size": ("platform", np.asarray([2, 2, 1], dtype=np.int32)),
            "cluster_assignment_status": (
                "platform", ["inferred", "inferred", "algorithmic_singleton"],
            ),
            "longitude_native": (("platform", "time"), linear_lon),
            "latitude_native": (("platform", "time"), latitude),
            "longitude_linear": (("platform", "time"), linear_lon),
            "latitude_linear": (("platform", "time"), latitude),
            "longitude_spline_30": (("platform", "time"), spline_lon),
            "latitude_spline_30": (("platform", "time"), latitude),
        },
        coords={"platform": np.arange(3), "time": time},
        attrs={
            "schema_version": "1.4", "algorithm_version": "test-reconstruction",
            "build_report_sha256": "test-report",
        },
    )
    dataset.to_zarr(path, mode="w", consolidated=True, zarr_format=2)


def config_values(
    *, window=None, distance=200, output="pairs",
    same_array=False, same_cluster=False,
):
    return {
        "input": {"zarr": "trajectories.zarr"},
        "coordinates": {"selection_method": "spline_30"},
        "selection": {
            "maximum_distance_m": distance,
            "maximum_seconds_from_each_observed_start": window,
        },
        "filters": {
            "same_array": same_array,
            "same_cluster": same_cluster,
        },
        "output": {
            "directory": output,
            "chunks": {"trajectory": 1, "obs": 2},
        },
    }


def write_config(tmp_path: Path, **kwargs) -> Path:
    path = tmp_path / "pairs.yml"
    path.write_text(yaml.safe_dump(config_values(**kwargs)), encoding="utf-8")
    return path


def hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): sha256(item.read_bytes()).hexdigest()
        for item in path.rglob("*") if item.is_file()
    }


def test_grouped_zarr_retains_methods_membership_and_padding(tmp_path):
    source = tmp_path / "trajectories.zarr"
    write_zarr(source)
    before = hashes(source)
    result = run_pair_workflow(write_config(tmp_path))
    assert result.selected_pair_count == 3
    assert result.possible_pair_count == 3
    assert result.eligible_pair_count == 3
    assert result.coordinate_methods == ("linear", "spline_30", "native")
    assert result.maximum_observations == 5
    assert hashes(source) == before

    catalog = pd.read_csv(
        result.catalog_path, dtype={"platform_code_1": str, "platform_code_2": str},
    )
    assert list(zip(catalog.platform_code_1, catalog.platform_code_2)) == [
        ("100", "200"), ("100", "300"), ("200", "300"),
    ]
    assert catalog.loc[0, "encounter_time_utc"] == "2025-01-01T00:05:00Z"
    assert catalog.loc[0, "encounter_observation"] == 1
    assert catalog.loc[0, "encounter_distance_linear_m"] < 200
    assert "encounter_distance_spline_30_m" in catalog
    assert "encounter_distance_native_m" in catalog
    assert catalog.array_id_1.tolist() == [1, 1, 1]
    assert catalog.array_id_2.tolist() == [1, 2, 2]
    assert catalog.cluster_id_1.tolist() == ["A01_C01", "A01_C01", "A01_C01"]
    assert catalog.member_id_2.tolist() == [2, 1, 1]
    assert catalog.same_array.tolist() == [True, False, False]
    assert catalog.same_cluster.tolist() == [True, False, False]
    assert catalog.same_array_filter.tolist() == [False, False, False]
    assert catalog.same_cluster_filter.tolist() == [False, False, False]
    assert catalog.loc[0, "observed_start_lon_2"] == pytest.approx(0.020)
    assert catalog.common_overlap_observations.tolist() == [5, 3, 3]

    with xr.open_zarr(result.zarr_path, consolidated=True, chunks=None) as dataset:
        assert dict(dataset.sizes) == {"trajectory": 3, "obs": 5}
        assert dataset.attrs["schema_version"] == "1.2"
        assert dataset.attrs["same_array_filter"] is False
        assert dataset.attrs["same_cluster_filter"] is False
        assert dataset.attrs["canonical_coordinate_method"] == "spline_30"
        assert dataset.attrs["available_coordinate_methods"] == [
            "linear", "spline_30", "native",
        ]
        assert dataset.group_size.values.tolist() == [2, 2, 2]
        assert dataset.array_id_1.values.tolist() == [1, 1, 1]
        assert dataset.array_id_2.values.tolist() == [1, 2, 2]
        assert dataset.cluster_id_1.values.astype(str).tolist() == [
            "A01_C01", "A01_C01", "A01_C01",
        ]
        assert dataset.member_id_2.values.tolist() == [2, 1, 1]
        assert dataset.cluster_size_2.values.tolist() == [2, 1, 1]
        assert dataset.cluster_assignment_status_2.values.astype(str).tolist() == [
            "inferred", "algorithmic_singleton", "algorithmic_singleton",
        ]
        assert dataset.same_array.values.tolist() == [True, False, False]
        assert dataset.same_cluster.values.tolist() == [True, False, False]
        np.testing.assert_allclose(dataset.observed_start_lon_2, [0.020, 0.004, 0.004])
        np.testing.assert_allclose(dataset.lon_1, dataset.lon_spline_30_1, equal_nan=True)
        np.testing.assert_allclose(dataset.lon, dataset.center_lon, equal_nan=True)
        np.testing.assert_allclose(dataset.lon_native_1, dataset.lon_linear_1, equal_nan=True)
        assert dataset.lon_linear_2.values[0, 0] != dataset.lon_2.values[0, 0]
        assert dataset.time.values[0, 0] == np.datetime64("2025-01-01T00:00:00")
        assert dataset.encounter_observation.values.tolist() == [1, 0, 1]
        assert dataset.time.values[0, 1] == dataset.encounter_time.values[0]
        assert dataset.common_overlap_observations.values.tolist() == [5, 3, 3]
        assert dataset.post_encounter_observations.values.tolist() == [4, 3, 2]
        assert np.isnat(dataset.time.values[1, 3])
        assert np.isnan(dataset.lon.values[1, 3])
        # Platform 100 participates in two independent pairs.
        assert dataset.platform_code_1.values.astype(str).tolist().count("100") == 2


def test_finite_window_and_null_chance_encounter_modes(tmp_path):
    write_zarr(tmp_path / "trajectories.zarr")
    limited = run_pair_workflow(write_config(tmp_path, window=0, output="limited"))
    limited_catalog = pd.read_csv(limited.catalog_path, dtype=str)
    assert list(zip(limited_catalog.platform_code_1, limited_catalog.platform_code_2)) == [
        ("100", "300"),
    ]
    chance = run_pair_workflow(write_config(tmp_path, window=None, output="chance"))
    assert chance.selected_pair_count == 3


def test_array_and_cluster_filters_limit_pair_eligibility(tmp_path):
    write_zarr(tmp_path / "trajectories.zarr")
    same_array = run_pair_workflow(
        write_config(tmp_path, same_array=True, output="same_array"),
    )
    assert same_array.possible_pair_count == 3
    assert same_array.eligible_pair_count == 1
    assert same_array.overlapping_pair_count == 1
    assert same_array.selected_pair_count == 1
    array_catalog = pd.read_csv(same_array.catalog_path, dtype=str)
    assert list(zip(array_catalog.platform_code_1, array_catalog.platform_code_2)) == [
        ("100", "200"),
    ]

    same_cluster = run_pair_workflow(
        write_config(tmp_path, same_cluster=True, output="same_cluster"),
    )
    assert same_cluster.possible_pair_count == 3
    assert same_cluster.eligible_pair_count == 1
    assert same_cluster.selected_pair_count == 1


def test_config_validation_atomic_existing_output_and_cli(tmp_path, capsys):
    write_zarr(tmp_path / "trajectories.zarr")
    config = write_config(tmp_path)
    values = config_values(distance=None)
    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="required and cannot be null"):
        load_pair_config(config)

    values = config_values()
    del values["filters"]["same_array"]
    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="filters.same_array is required"):
        load_pair_config(config)

    values = config_values()
    values["filters"]["same_cluster"] = "false"
    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="filters.same_cluster must be Boolean"):
        load_pair_config(config)

    config.write_text(yaml.safe_dump(config_values()), encoding="utf-8")
    assert main([str(config)]) == 0
    assert "Candidate pairs: 3 selected" in capsys.readouterr().out
    with pytest.raises(FileExistsError, match="use --overwrite"):
        run_pair_workflow(config)
    replacement = run_pair_workflow(config, overwrite=True)
    assert (replacement.output_directory / ZARR_NAME).is_dir()
    assert (replacement.output_directory / CATALOG_NAME).is_file()
