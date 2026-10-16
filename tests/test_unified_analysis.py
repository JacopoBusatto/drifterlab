from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
import zarr

from drifterlab.analysis import load_analysis_config, run_analysis
from drifterlab.reconstruction.product_store import (
    PlatformAssignmentMetadata,
    PlatformProductMetadata,
    create_product_store,
    write_platform_assignment,
    write_platform_metadata,
    write_platform_products,
)
from drifterlab.reconstruction.products import (
    TrajectorySource,
    build_platform_products,
    build_regular_grids,
)
from drifterlab.trajectories.product_store import load_trajectory_product


def _times(minutes: np.ndarray | list[int]) -> np.ndarray:
    return (
        np.datetime64("2025-02-06T00:00:00", "ns")
        + np.asarray(minutes).astype("timedelta64[m]")
    )


def _write_v2_store(path: Path, *, status: str = "final_complete_qc") -> Path:
    minutes = np.arange(0, 125, 5)
    times = _times(minutes)
    grids = build_regular_grids(times[0], times[0], times[-1])
    platforms = ["p1", "p2", "p3"]
    root = create_product_store(path, platforms, grids, chunk_platform=1, chunk_time=32)
    for index, platform in enumerate(platforms):
        elapsed = minutes.astype(float)
        longitude = -150.0 + 0.001 * index + elapsed * (0.00002 + index * 0.000005)
        latitude = 70.0 + 0.0005 * index + elapsed * (0.00001 + index * 0.000002)
        source = TrajectorySource(times, longitude, latitude)
        products = build_platform_products(
            source,
            source,
            grids,
            coverage_start=times[0],
            coverage_end=times[-1],
        )
        write_platform_products(root, index, products)
        write_platform_assignment(
            root,
            index,
            assignment=PlatformAssignmentMetadata(
                array_id=3,
                gap_from_previous_start_hours=None,
                gap_to_next_start_hours=None,
                cluster_id="array_003__cluster_001",
                member_id=index + 1,
                cluster_size=3,
                cluster_assignment_status="candidate_cluster",
                cluster_start_spread_seconds=0.0,
                cluster_diameter_m=200.0,
                cluster_maximum_available_pair_distance_m=200.0,
                cluster_observed_start_diameter_m=200.0,
                cluster_first_common_grid_diameter_m=200.0,
                cluster_first_common_grid_unavailable_pair_count=0,
                cluster_candidate_link_count=3,
                cluster_unavailable_pair_count=0,
            ),
        )
        write_platform_metadata(
            root,
            index,
            PlatformProductMetadata(
                deployment_start_time=times[0],
                deployment_start_longitude=float(longitude[0]),
                deployment_start_latitude=float(latitude[0]),
                trajectory_end_time=times[-1],
                review_end_time=times[-1],
                requested_review_span_days=1.0,
                accepted_review_span_days=1.0,
                review_end_reason="trajectory_end",
                accepted_fix_count=len(times),
                raw_fix_count=len(times),
                source_sha256=f"source-{platform}",
                qc_input_sha256=f"qc-{platform}",
            ),
        )
    root.attrs.update({
        "algorithm_version": "synthetic-v2",
        "product_status": status,
        "array_assignment_sha256": "array-hash",
        "initial_cluster_assignment_sha256": "cluster-hash",
        "build_manifest_sha256": "manifest-hash",
    })
    zarr.consolidate_metadata(str(path))
    return path


def _config(store: Path, output: Path, *, method: str = "centered_fixed_span") -> dict:
    kinematics = {
        "method": method,
        "maximum_source_gap_minutes": 30,
        "require_full_stencil_support": True,
        "speed_mask_m_s": None,
    }
    if method == "centered_fixed_span":
        kinematics["total_span_minutes"] = 30
    return {
        "schema_version": 1,
        "input": {
            "trajectory_zarr": str(store),
            "trajectory_product": "qc_5min",
            "dataset_label": "synthetic",
            "allow_provisional": False,
        },
        "selection": {
            "arrays": [3],
            "clusters": ["array_003__cluster_001"],
        },
        "kinematics": kinematics,
        "analyses": {
            "trajectory_plots": {
                "enabled": True,
                "color_by": "cluster_id",
                "color_mode": "categorical",
                "cmap": "tab10",
                "time": {"start": None, "end": None},
                "movie": {"enabled": False},
                "map": {
                    "land": False,
                    "coastlines": False,
                    "gridlines": False,
                    "scale_bar": False,
                    "dpi": 30,
                    "figsize": [4, 3],
                },
                "array_overrides": {},
            },
            "trajectory_statistics": {
                "enabled": True,
                "time": {"start": None, "end": None},
                "stop_on_member_loss": False,
                "percentiles": [0, 50, 100],
                "velocity": {
                    "histogram_bins": 5,
                    "speed_range_m_s": None,
                    "share_probability_density_y_axis": False,
                },
                "plotting": {"dpi": 30, "relative_dispersion_yscale": "linear"},
                "array_overrides": {},
            },
            "epsilon": {
                "enabled": True,
                "pair_scope": "within_cluster",
                "array_anchor": None,
                "minimum_active_members": 2,
                "sample_cadence_minutes": 5,
                "sample_phase_reference": "activation_anchor",
                "windows": [{"id": "initial_0_1h", "start_hour": 0, "end_hour": 1}],
                "minimum_reliable_separation_m": None,
                "separation_bins": {
                    "mode": "explicit",
                    "edges_m": [1, 100, 1000, 10000],
                    "axis_scale": "log",
                },
                "support": {
                    "observations": 1,
                    "unique_pairs": 1,
                    "platforms": 2,
                    "unique_utc": 1,
                },
                "uncertainty": {
                    "enabled": True,
                    "method": "synchronized_utc_block_bootstrap",
                    "confidence_level": 0.95,
                    "replicates": 10,
                    "random_seed": 7,
                    "primary_block_minutes": 15,
                    "sensitivity_block_minutes": [30],
                    "minimum_contributing_blocks": 1,
                    "minimum_successful_replicate_fraction": 0.5,
                },
                "q_distribution": {
                    "enabled": True,
                    "maximum_absolute_m2_s3": 0.01,
                    "linear_threshold_m2_s3": 0.00000001,
                    "bins_per_decade": 2,
                },
                "figures": {
                    "enabled": True,
                    "dpi": 30,
                    "scatter_maximum_points": 100,
                    "epsilon_linear_threshold_m2_s3": 0.00000001,
                    "heatmap_probability_maximum": 0.5,
                },
            },
            "epsilon_diagnostics": {
                "enabled": True,
                "time": {
                    "window": "maximum_configured",
                    "mark_elapsed_hours": [0, 1],
                },
                "trajectory_windows": {
                    "enabled": True,
                    "coordinates": "local_projected_xy",
                    "equal_aspect": True,
                    "show_analysis_valid": True,
                    "show_invalid_reconstructed_positions": True,
                    "label_platforms": True,
                    "mark_activation": True,
                    "mark_window_boundaries": True,
                    "dpi": 30,
                },
                "velocity_availability": {"enabled": True, "dpi": 30},
                "pair_relative_motion": {
                    "enabled": True,
                    "derivative_span": "source_analysis_velocity",
                    "dpi": 30,
                },
            },
        },
        "output": {"root": str(output)},
    }


def _write_config(path: Path, values: dict) -> Path:
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_grouped_product_reader_and_unified_numbered_outputs(tmp_path: Path) -> None:
    store = _write_v2_store(tmp_path / "trajectories.zarr")
    metadata_before = _sha256(store / ".zmetadata")
    dataset, info = load_trajectory_product(store, "qc_5min")
    assert info.cadence_minutes == 5
    assert info.source_kind == "qc"
    assert dataset.start_time.dims == ("platform",)
    assert dataset.longitude.dims == ("platform", "time")

    output = tmp_path / "analysis"
    config_path = _write_config(tmp_path / "analysis.yml", _config(store, output))
    config = load_analysis_config(config_path)
    assert config.trajectory_product == "qc_5min"

    result = run_analysis(config_path)

    assert result.array_count == 1
    assert result.cluster_count == 1
    assert (output / "01_trajectory_plots/array_003/trajectories.png").is_file()
    assert (
        output / "02_trajectory_statistics/array_003/cluster_timeseries.csv"
    ).is_file()
    assert (output / "02_trajectory_statistics/array_003/figures").is_dir()
    assert (output / "03_epsilon/array_003/tables/epsilon_by_scale.csv").is_file()
    assert (output / "03_epsilon/array_003/data/q_histograms.parquet").is_file()
    assert not (
        output / "03_epsilon/array_003/data/leave_one_platform_out.parquet"
    ).exists()
    assert (
        output
        / "04_epsilon_diagnostics/array_003/trajectory_window/array_003__cluster_001.png"
    ).is_file()
    assert not list(output.rglob("time_scale_contributions*"))
    assert not list(output.rglob("influence_audit*"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input"]["trajectory_product"] == "qc_5min"
    assert manifest["kinematics"]["method"] == "centered_fixed_span"
    assert manifest["output_layout"] == [
        "01_trajectory_plots",
        "02_trajectory_statistics",
        "03_epsilon",
        "04_epsilon_diagnostics",
    ]
    assert _sha256(store / ".zmetadata") == metadata_before
    with pytest.raises(FileExistsError, match="already exists"):
        run_analysis(config_path)


def test_provisional_input_requires_explicit_opt_in(tmp_path: Path) -> None:
    store = _write_v2_store(
        tmp_path / "trajectories.zarr", status="provisional_partial_qc",
    )
    values = _config(store, tmp_path / "analysis")
    values["analyses"]["trajectory_plots"]["enabled"] = False
    values["analyses"]["trajectory_statistics"]["enabled"] = False
    values["analyses"]["epsilon_diagnostics"]["enabled"] = False
    for section in ("uncertainty", "q_distribution", "figures"):
        values["analyses"]["epsilon"][section]["enabled"] = False
    config_path = _write_config(tmp_path / "analysis.yml", values)
    with pytest.raises(ValueError, match="allow_provisional"):
        run_analysis(config_path)


def test_stored_forward_is_recorded_as_earlier_endpoint_velocity(tmp_path: Path) -> None:
    store = _write_v2_store(tmp_path / "trajectories.zarr")
    output = tmp_path / "analysis"
    values = _config(store, output, method="stored_forward")
    values["analyses"]["trajectory_plots"]["enabled"] = False
    values["analyses"]["trajectory_statistics"]["enabled"] = False
    for section in ("uncertainty", "q_distribution", "figures"):
        values["analyses"]["epsilon"][section]["enabled"] = False
    result = run_analysis(_write_config(tmp_path / "analysis.yml", values))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["kinematics"] == {
        "effective_span_minutes": 5.0,
        "maximum_source_gap_minutes": 30.0,
        "method": "stored_forward",
        "time_assignment": "earlier_endpoint",
    }
    observations = pd.read_parquet(
        output / "03_epsilon/array_003/data/pair_observations.parquet"
    )
    assert len(observations) > 0
    assert (
        output
        / "04_epsilon_diagnostics/array_003/pair_relative_motion/array_003__cluster_001.png"
    ).is_file()


def test_centered_adjacent_uses_immediate_grid_neighbours(tmp_path: Path) -> None:
    store = _write_v2_store(tmp_path / "trajectories.zarr")
    output = tmp_path / "analysis"
    values = _config(store, output, method="centered_adjacent")
    values["analyses"]["trajectory_plots"]["enabled"] = False
    values["analyses"]["trajectory_statistics"]["enabled"] = False
    values["analyses"]["epsilon_diagnostics"]["enabled"] = False
    for section in ("uncertainty", "q_distribution", "figures"):
        values["analyses"]["epsilon"][section]["enabled"] = False

    result = run_analysis(_write_config(tmp_path / "analysis.yml", values))

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["kinematics"] == {
        "effective_span_minutes": 10.0,
        "maximum_source_gap_minutes": 30.0,
        "method": "centered_adjacent",
        "time_assignment": "center_timestamp",
    }


@pytest.mark.parametrize("method", ["centered_fixed_span", "centered_adjacent", "stored_forward"])
def test_spatial_maps_share_utc_windows_and_publish_auditable_products(tmp_path: Path, method: str) -> None:
    store = _write_v2_store(tmp_path / "trajectories.zarr")
    root = zarr.open_group(str(store), mode="a")
    root["cluster_id"][2] = "array_003__cluster_002"
    root["cluster_size"][:] = [2, 2, 1]
    zarr.consolidate_metadata(str(store))
    metadata_before = _sha256(store / ".zmetadata")
    output = tmp_path / "analysis"
    values = _config(store, output, method=method)
    values["selection"]["clusters"] = "all"
    for module in ("trajectory_plots", "trajectory_statistics", "epsilon_diagnostics"):
        values["analyses"][module]["enabled"] = False
    epsilon = values["analyses"]["epsilon"]
    for section in ("uncertainty", "q_distribution", "figures"):
        epsilon[section]["enabled"] = False
    epsilon["spatial_maps"] = {
        "enabled": True,
        "couples": {"between": "array", "max_initial_distance_m": 100},
        "resample_stride": 3,
        "color": {"maximum_absolute_m2_s3": 1e-10, "linear_threshold_m2_s3": 1e-12},
        "windows": [
            "initial_0_1h",
            {"id": "empty_late", "start_hour": 20, "end_hour": 21},
        ],
        "dpi": 30,
    }

    result = run_analysis(_write_config(tmp_path / "analysis.yml", values))

    array = output / "03_epsilon/array_003"
    observations = pd.read_parquet(array / "data/spatial_pair_observations.parquet")
    assert observations.pair_id.nunique() == 2
    assert set(observations.pair_cluster_relation) == {"within_cluster", "between_clusters"}
    assert observations.array_age_hours.max() < 1
    assert not observations.duplicated(["pair_id", "time_utc"]).any()
    inventory = pd.read_csv(array / "tables/spatial_pair_selection.csv")
    assert inventory.selected.sum() == 2
    figures = pd.read_csv(array / "tables/spatial_figures.csv")
    assert len(figures) == 6
    assert set(figures.reference_time_utc) == {"2025-02-06T00:00:00+00:00"}
    assert set(figures[figures.window_id == "initial_0_1h"].window_end_utc) == {"2025-02-06T01:00:00+00:00"}
    assert figures[figures.window_id == "empty_late"].displayed_observations.sum() == 0
    whole = figures[(figures.scope == "array") & (figures.window_id == "initial_0_1h")].iloc[0]
    assert whole.available_observations == len(observations)
    assert whole.displayed_observations == int((observations.schedule_index % 3 == 0).sum())
    assert whole.displayed_observations < whole.available_observations
    assert whole.positive_saturated_count + whole.negative_saturated_count > 0
    singleton = figures[(figures.population_id == "array_003__cluster_002")]
    assert singleton.available_observations.sum() == 0
    assert all((array / path).is_file() for path in figures.figure)
    metadata = json.loads((array / "data/spatial_metadata.json").read_text())
    assert metadata["settings"]["color"]["maximum_absolute_m2_s3"] == 1e-10
    assert metadata["cluster_maps"] == "within_cluster_pairs_only"
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["modules"]["epsilon"]["spatial_maps"]["figure_count"] == 6
    products = str(manifest["products"])
    assert "spatial_pair_observations.parquet" in products
    assert "spatial_metadata.json" in products
    assert _sha256(store / ".zmetadata") == metadata_before
