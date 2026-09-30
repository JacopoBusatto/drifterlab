"""Read-only inspection, review, publication, and plotting tests."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from drifterlab.cli.clusters import main
from drifterlab.workflows.clusters import (
    CANDIDATE_DIRECTORY, INSPECTION_DIRECTORY, MEMBERS_NAME, NEIGHBOR_NAME,
    REVIEW_NAME, SUMMARY_NAME, load_cluster_config, run_cluster_workflow,
)


def write_zarr(path: Path):
    platform = ["1001", "1002", "1003", "1004"]
    times = np.asarray([
        "2025-01-01T00:00:00", "2025-01-01T00:05:00",
        "2025-01-01T00:10:00", "2025-01-03T00:00:00",
    ], dtype="datetime64[ns]")
    dataset = xr.Dataset(
        {
            "platform_id": ("platform", platform),
            "start_time": ("platform", times),
            "start_lon": ("platform", [0., .0005, .1, 5.]),
            "start_lat": ("platform", [70., 70., 70., 71.]),
            "longitude_linear": (
                ("platform", "time"),
                np.asarray([[0., 1.], [.0005, 1.], [.1, 1.], [5., 6.]]),
            ),
            "latitude_linear": (
                ("platform", "time"),
                np.asarray([[70., 70.1], [70., 70.1], [70., 70.1], [71., 71.1]]),
            ),
        },
        coords={"platform": np.arange(4), "time": np.asarray([
            "2025-01-01", "2025-01-04",
        ], dtype="datetime64[ns]")},
        attrs={
            "schema_version": "1.1", "algorithm_version": "test-reconstruction",
            "product_status": "candidate_pending_gap_review", "build_report_sha256": "test",
        },
    )
    dataset.to_zarr(path, mode="w", consolidated=True, zarr_format=2)


def config_values(*, plotting=False, reviewed=None, diameter=None, spread=None):
    return {
        "input": {"zarr": "trajectories.zarr"},
        "cohorts": {
            "maximum_adjacent_start_gap_hours": 24,
            "reviewed_assignments": reviewed,
        },
        "grouping": {
            "maximum_cluster_diameter_m": diameter,
            "maximum_members_per_cluster": 5,
            "maximum_observed_start_spread_minutes": spread,
            "cohort_overrides": {},
        },
        "diagnostics": {"nearest_neighbor_ranks": 5},
        "output": {"directory": "clusters"},
        "plotting": {
            "enabled": plotting, "projection": "PlateCarree", "central_longitude": 0,
            "extent": None, "figsize": [6, 5], "dpi": 80,
            "land": False, "coastlines": False, "gridlines": False,
        },
    }


def write_config(tmp_path: Path, **kwargs) -> Path:
    path = tmp_path / "clusters.yml"
    path.write_text(yaml.safe_dump(config_values(**kwargs)), encoding="utf-8")
    return path


def zarr_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): sha256(item.read_bytes()).hexdigest()
        for item in path.rglob("*") if item.is_file()
    }


def test_inspection_is_compact_and_read_only(tmp_path):
    zarr = tmp_path / "trajectories.zarr"
    write_zarr(zarr)
    before = zarr_hashes(zarr)
    result = run_cluster_workflow(write_config(tmp_path), inspection=True)
    assert result.output_directory.name == INSPECTION_DIRECTORY
    assert sorted(path.name for path in result.output_directory.iterdir()) == [
        "cohort_review.csv", "manifest.json", "neighbor_diagnostics.csv",
    ]
    review = pd.read_csv(result.output_directory / REVIEW_NAME, dtype={"platform_id": str})
    assert review.proposed_cohort_id.tolist() == [
        "cohort_001", "cohort_001", "cohort_001", "cohort_002",
    ]
    neighbors = pd.read_csv(result.output_directory / NEIGHBOR_NAME)
    assert set(neighbors.neighbor_rank) == {1, 2}
    assert zarr_hashes(zarr) == before


def test_build_requires_review_then_enforces_space_and_time(tmp_path):
    write_zarr(tmp_path / "trajectories.zarr")
    config = write_config(tmp_path, diameter=100, spread=30)
    with pytest.raises(ValueError, match="reviewed_assignments"):
        run_cluster_workflow(config)
    inspection = run_cluster_workflow(config, inspection=True)
    reviewed = inspection.output_directory / REVIEW_NAME
    values = config_values(
        reviewed="clusters/inspection/cohort_review.csv", diameter=100, spread=30,
    )
    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    result = run_cluster_workflow(config)
    assert result.output_directory.name == CANDIDATE_DIRECTORY
    members = pd.read_csv(result.output_directory / MEMBERS_NAME, dtype={"platform_id": str})
    summary = pd.read_csv(result.output_directory / SUMMARY_NAME)
    cluster_1001 = members.loc[members.platform_id.eq("1001"), "candidate_cluster_id"].item()
    cluster_1002 = members.loc[members.platform_id.eq("1002"), "candidate_cluster_id"].item()
    cluster_1003 = members.loc[members.platform_id.eq("1003"), "candidate_cluster_id"].item()
    assert cluster_1001 == cluster_1002
    assert cluster_1001 != cluster_1003
    assert (summary.diameter_m <= 100 + 1e-9).all()
    assert (summary.observed_start_spread_minutes <= 30 + 1e-9).all()


def test_stale_review_and_unknown_config_are_rejected(tmp_path):
    write_zarr(tmp_path / "trajectories.zarr")
    config = write_config(tmp_path)
    inspection = run_cluster_workflow(config, inspection=True)
    review_path = inspection.output_directory / REVIEW_NAME
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    review.loc[0, "observed_start_sha256"] = "stale"
    review.to_csv(review_path, index=False)
    config.write_text(yaml.safe_dump(config_values(
        reviewed="clusters/inspection/cohort_review.csv", diameter=100, spread=30,
    )), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        run_cluster_workflow(config)
    values = config_values()
    values["grouping"]["forced_cluster_count"] = 2
    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown grouping keys"):
        load_cluster_config(config)


def test_inspection_and_candidate_png_pdf_per_cohort_and_cli(tmp_path, capsys):
    write_zarr(tmp_path / "trajectories.zarr")
    config = write_config(tmp_path, plotting=True, diameter=100, spread=30)
    assert main([str(config), "--inspection"]) == 0
    assert "read only" in capsys.readouterr().out
    inspection_figures = sorted((tmp_path / "clusters/inspection/figures").glob("*"))
    assert len(inspection_figures) == 4
    assert {path.suffix for path in inspection_figures} == {".png", ".pdf"}
    assert all(path.stat().st_size > 1000 for path in inspection_figures)

    values = config_values(
        plotting=True, reviewed="clusters/inspection/cohort_review.csv",
        diameter=100, spread=30,
    )
    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    assert main([str(config)]) == 0
    candidate_figures = sorted((tmp_path / "clusters/candidate/figures").glob("*"))
    assert len(candidate_figures) == 4
    assert all(path.stat().st_size > 1000 for path in candidate_figures)
