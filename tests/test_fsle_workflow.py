"""Configuration, publication, and rendering tests for the FSLE workflow."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from pyproj import Geod

from drifterlab.cli.fsle import main
from drifterlab.workflows.fsle import load_fsle_config, run_fsle_workflow

WGS84 = Geod(ellps="WGS84")


def _longitude_at_distance(distance_km: float) -> float:
    longitude, _, _ = WGS84.fwd(0, 0, 90, distance_km * 1000)
    return longitude


def _write_pair_zarr(path: Path) -> None:
    distances = np.asarray(
        [
            [0.8, 1.2, 2.5, 2.6],
            [0.9, 1.1, 1.5, 2.2],
            [1.0, 1.3, 2.4, 2.7],
            [0.8, 1.2, 2.3, 2.8],
        ]
    )
    lon_2 = np.vectorize(_longitude_at_distance)(distances)
    time = np.asarray(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
        ],
        dtype="datetime64[ns]",
    )
    dataset = xr.Dataset(
        {
            "group_id": ("trajectory", ["pair_a", "pair_b", "between", "cross"]),
            "group_size": ("trajectory", np.full(4, 2, dtype=np.int16)),
            "platform_code_1": ("trajectory", ["a1", "b1", "a1", "a1"]),
            "platform_code_2": ("trajectory", ["a2", "b2", "b1", "c1"]),
            "array_id_1": ("trajectory", np.asarray([1, 1, 1, 1])),
            "array_id_2": ("trajectory", np.asarray([1, 1, 1, 2])),
            "cluster_id_1": ("trajectory", ["A", "B", "A", "A"]),
            "cluster_id_2": ("trajectory", ["A", "B", "B", "C"]),
            "cluster_size_1": ("trajectory", np.asarray([2, 2, 2, 2])),
            "cluster_size_2": ("trajectory", np.asarray([2, 2, 2, 1])),
            "same_array": ("trajectory", [True, True, True, False]),
            "same_cluster": ("trajectory", [True, True, False, False]),
            "encounter_observation": ("trajectory", np.zeros(4, dtype=np.int64)),
            "common_overlap_observations": (
                "trajectory",
                np.full(4, 4, dtype=np.int64),
            ),
            "time": (("trajectory", "obs"), np.tile(time, (4, 1))),
            "lon_linear_1": (("trajectory", "obs"), np.zeros((4, 4))),
            "lat_linear_1": (("trajectory", "obs"), np.zeros((4, 4))),
            "lon_linear_2": (("trajectory", "obs"), lon_2),
            "lat_linear_2": (("trajectory", "obs"), np.zeros((4, 4))),
        },
        coords={"trajectory": np.arange(4), "obs": np.arange(4)},
        attrs={
            "schema_version": "1.2",
            "algorithm_version": "test-pairs",
            "product_status": "candidate_pairs",
            "canonical_coordinate_method": "linear",
            "available_coordinate_methods": ["linear"],
            "maximum_distance_m": 25000.0,
            "possible_pair_count": 6,
            "overlapping_pair_count": 4,
            "selected_pair_count": 4,
        },
    )
    dataset.to_zarr(path, mode="w", consolidated=True, zarr_format=2)


def _values() -> dict:
    return {
        "input": {"pairs_zarr": "pairs.zarr", "coordinate_method": "linear"},
        "analysis": {
            "minimum_scale_km": 1,
            "maximum_scale_km": 2,
            "rho": 2,
            "anchor_scale_km": 1,
            "expected_interval_minutes": 1440,
            "minimum_reached_pairs_per_scale": 1,
        },
        "plotting": {
            "dpi": 30,
            "x_range_km": None,
            "y_range_day_inverse": None,
            "standard_error_bars": True,
            "reference_slope": {
                "enabled": True,
                "exponent": -2 / 3,
                "label": "delta^-2/3",
            },
        },
        "output": {"directory": "fsle"},
    }


def _write_config(tmp_path: Path, values: dict | None = None) -> Path:
    path = tmp_path / "fsle.yml"
    path.write_text(
        yaml.safe_dump(values or _values(), sort_keys=False), encoding="utf-8"
    )
    return path


def _hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256(item.read_bytes()).hexdigest()
        for item in path.rglob("*")
        if item.is_file()
    }


def test_config_is_strict_and_scale_anchor_must_be_in_range(tmp_path):
    config = load_fsle_config(_write_config(tmp_path))
    assert config.coordinate_method == "linear"
    assert config.anchor_scale_km == 1
    assert config.plotting.standard_error_bars is True
    values = _values()
    values["analysis"]["anchor_scale_km"] = 0.5
    with pytest.raises(ValueError, match="anchor scale must lie"):
        load_fsle_config(_write_config(tmp_path, values))
    values = _values()
    values["analysis"]["unknown"] = True
    with pytest.raises(ValueError, match="Unknown analysis keys"):
        load_fsle_config(_write_config(tmp_path, values))


def test_workflow_filters_pairs_writes_two_figures_and_is_atomic(tmp_path, capsys):
    pair_zarr = tmp_path / "pairs.zarr"
    _write_pair_zarr(pair_zarr)
    before = _hashes(pair_zarr)
    config_path = _write_config(tmp_path)

    assert main([str(config_path)]) == 0
    assert "2 same-cluster pairs" in capsys.readouterr().out
    output = tmp_path / "fsle"
    assert before == _hashes(pair_zarr)
    assert (output / "fsle_spectra.csv").is_file()
    assert (output / "fsle_first_passages.parquet").is_file()
    figures = sorted((output / "array_001").glob("*.png"))
    assert [path.name for path in figures] == [
        "fsle_individual_clusters.png",
        "fsle_same_cluster_pooled.png",
    ]
    assert all(path.stat().st_size > 0 for path in figures)

    passages = pd.read_parquet(output / "fsle_first_passages.parquet")
    assert len(passages) == 2
    assert set(passages.pair_id) == {"pair_a", "pair_b"}
    spectrum = pd.read_csv(output / "fsle_spectra.csv")
    assert len(spectrum) == 3
    assert spectrum.scope.value_counts().to_dict() == {
        "individual_cluster": 2,
        "same_cluster_pooled": 1,
    }
    pooled = spectrum[spectrum.scope == "same_cluster_pooled"].iloc[0]
    expected = (math.log(2.5 / 1.2) + math.log(2.2 / 1.1)) / 2
    expected /= (1 + 2) / 2
    assert pooled.fsle_day_inverse == pytest.approx(expected)
    assert math.isfinite(pooled.fsle_standard_error_day_inverse)

    manifest = json.loads((output / "fsle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis"]["analyzed_same_cluster_pair_count"] == 2
    assert manifest["analysis"]["ignored_between_cluster_pair_count"] == 1
    assert manifest["analysis"]["ignored_cross_array_pair_count"] == 1
    assert manifest["analysis"]["arrays"][0]["reference_anchor_available"] is True
    assert manifest["analysis"]["uncertainty"]["column"] == (
        "fsle_standard_error_day_inverse"
    )
    assert (
        manifest["effective_configuration"]["plotting"]["standard_error_bars"] is True
    )

    with pytest.raises(FileExistsError, match="use --overwrite"):
        run_fsle_workflow(config_path)
    replacement = run_fsle_workflow(config_path, overwrite=True)
    assert replacement.array_count == 1
    assert replacement.analyzed_pair_count == 2
