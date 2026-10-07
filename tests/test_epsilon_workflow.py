from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from drifterlab.workflows.epsilon import run_epsilon_workflow


def _write_trajectory_store(path: Path) -> None:
    count = 160
    times = np.datetime64("2025-01-01T00:00:00", "ns") + np.arange(count) * np.timedelta64(5, "m")
    platform = np.arange(4)
    seconds = np.arange(count) * 300.0
    metres_per_degree = 111_000.0
    east_m = np.vstack((
        0.10 * seconds,
        100.0 + 0.15 * seconds,
        250.0 + 0.05 * seconds,
        1000.0 + 0.02 * seconds,
    ))
    north_m = np.vstack((
        0.02 * seconds,
        50.0 + 0.01 * seconds,
        -80.0 + 0.03 * seconds,
        500.0 + 0.01 * seconds,
    ))
    lon = -150.0 + east_m / (metres_per_degree * np.cos(np.deg2rad(70.0)))
    lat = 70.0 + north_m / metres_per_degree
    # The third member is recorded later. The singleton never forms a pair.
    lon[2, :6] = np.nan
    lat[2, :6] = np.nan
    gaps = np.full((4, count), 5.0)
    gaps[~np.isfinite(lon)] = np.nan
    starts = np.array([
        times[0], times[0], times[6], times[0],
    ], dtype="datetime64[ns]")
    dataset = xr.Dataset(
        data_vars={
            "platform_id": (("platform",), np.array(["a", "b", "c", "singleton"])),
            "start_time": (("platform",), starts),
            "array_id": (("platform",), np.ones(4, dtype=np.int32)),
            "cluster_id": (("platform",), np.array(["cluster_a", "cluster_a", "cluster_a", "cluster_b"])),
            "cluster_size": (("platform",), np.array([3, 3, 3, 1], dtype=np.int32)),
            "cluster_assignment_status": (("platform",), np.array(["candidate_cluster"] * 3 + ["algorithmic_singleton"])),
            "source_gap_minutes": (("platform", "time"), gaps),
            "longitude_linear": (("platform", "time"), lon),
            "latitude_linear": (("platform", "time"), lat),
            "longitude_spline_15": (("platform", "time"), lon),
            "latitude_spline_15": (("platform", "time"), lat),
            "longitude_spline_30": (("platform", "time"), lon),
            "latitude_spline_30": (("platform", "time"), lat),
        },
        coords={"platform": platform, "time": times},
        attrs={
            "schema_version": "1.4",
            "algorithm_version": "synthetic-test",
            "product_status": "candidate_pending_gap_review",
            "initial_cluster_assignment_sha256": "synthetic",
        },
    )
    dataset.to_zarr(path, mode="w", consolidated=True)


def test_stage_2_workflow_keeps_singletons_and_uses_full_pair_coverage_denominator(
    tmp_path: Path,
) -> None:
    store = tmp_path / "trajectories.zarr"
    _write_trajectory_store(store)
    output = tmp_path / "epsilon"
    config = tmp_path / "epsilon.yml"
    config.write_text(
        f"""
input:
  trajectories: {store.as_posix()}
  dataset_label: synthetic
  allow_candidate_input: true
selection:
  arrays: [1]
  clusters: all
  pair_scope: within_cluster
  array_anchor: null
activation:
  minimum_active_members: 2
  reference:
    position_method: linear
    velocity_method: centered_difference
    total_span_minutes: 30
    maximum_source_gap_minutes: 30
    require_full_stencil_support: true
    speed_mask_m_s: null
analysis:
  position_method: linear
  velocity_method: centered_difference
  total_span_minutes: 30
  maximum_source_gap_minutes: 30
  require_full_stencil_support: true
  speed_mask_m_s: null
  windows:
    - {{id: initial_0_6h, start_hour: 0, end_hour: 6}}
    - {{id: initial_0_12h, start_hour: 0, end_hour: 12}}
  minimum_reliable_separation_m: null
separation_bins:
  mode: explicit
  edges_m: [1, 100, 1000, 10000]
  axis_scale: log
support:
  observations: 12
  unique_pairs: 3
  platforms: 3
  unique_utc: 12
stage3:
  enabled: true
  uncertainty:
    method: synchronized_utc_block_bootstrap
    confidence_level: 0.95
    replicates: 100
    random_seed: 7
    primary_block_minutes: 30
    sensitivity_block_minutes: [60]
    minimum_contributing_blocks: 2
    minimum_successful_replicate_fraction: 0.9
  q_histogram:
    maximum_absolute_m2_s3: 0.01
    linear_threshold_m2_s3: 0.00000001
    bins_per_decade: 2
  figures:
    dpi: 72
    scatter_maximum_points: 100
    epsilon_linear_threshold_m2_s3: 0.00000001
    heatmap_probability_maximum: 0.5
output:
  root: {output.as_posix()}
""",
        encoding="utf-8",
    )
    result = run_epsilon_workflow(config)
    assert result.activated_cluster_count == 1
    array_directory = result.run_directory / "array_001"
    assert result.clusters_paths == (array_directory / "clusters.csv",)
    inventory = pd.read_csv(array_directory / "clusters.csv")
    assert inventory.columns[:4].tolist() == [
        "array_id", "cluster_id", "analysis_type", "window_id",
    ]
    assert set(inventory.analysis_type) == {"inventory"}
    assert set(inventory.cluster_id) == {"cluster_a", "cluster_b"}
    singleton = inventory[inventory.cluster_id == "cluster_b"].iloc[0]
    assert singleton.snapshot_status == "singleton_no_internal_pair"

    observations = pd.read_parquet(array_directory / "data" / "pair_observations.parquet")
    assert observations.is_snapshot.sum() == 1
    assert observations.pair_id.nunique() == 3
    assert not observations.duplicated(
        ["array_id", "cluster_id", "pair_id", "time_utc"]
    ).any()
    coverage = pd.read_csv(array_directory / "coverage.csv")
    six = coverage[
        (coverage.coverage_level == "cluster")
        & (coverage.window_id == "initial_0_6h")
    ].iloc[0]
    assert six.scheduled_utc_count == 72
    assert six.expected_pair_observation_count == 3 * 72
    assert six.valid_pair_observation_count < six.expected_pair_observation_count

    summaries = pd.read_csv(array_directory / "epsilon_by_scale.csv")
    summaries = summaries[summaries.analysis_type == "window"]
    assert not summaries.duplicated(
        ["array_id", "cluster_id", "analysis_type", "window_id", "separation_bin_id"]
    ).any()
    finite = summaries[np.isfinite(summaries.epsilon_eff_m2_s3)]
    assert len(finite)
    assert np.allclose(
        finite.epsilon_eff_m2_s3,
        finite.separation_weighted_mean_q_m2_s3,
        rtol=1e-12,
        atol=1e-18,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "3.0"
    assert manifest["input"]["product_status"] == "candidate_pending_gap_review"
    assert manifest["scientific_conventions"]["velocity_stencil_point_count"] == 7
    assert manifest["scientific_conventions"]["analysis_required_validity_point_count"] == 7
    assert manifest["analysis"]["figures"] == "stage3_cluster_products_and_comparisons"
    assert set(manifest["output_layout"]["readable_tables"]) == {
        "clusters.csv", "epsilon_by_scale.csv", "statistics.csv", "coverage.csv",
    }
    assert (array_directory / "figures").is_dir()
    assert (array_directory / "figures" / "cluster_a" / "snapshot.png").is_file()
    assert (
        array_directory / "figures" / "cluster_a" / "initial_0_6h.png"
    ).is_file()
    assert (array_directory / "data" / "q_histograms.parquet").is_file()
    assert (array_directory / "data" / "bootstrap_sensitivity.parquet").is_file()
    assert (array_directory / "data" / "leave_one_platform_out.parquet").is_file()
    assert {"epsilon_ci_lower_m2_s3", "maximum_lopo_absolute_change_m2_s3"} <= set(
        pd.read_csv(array_directory / "epsilon_by_scale.csv").columns
    )
    assert not any(array_directory.glob("cluster_*"))
    statistics = pd.read_csv(array_directory / "statistics.csv")
    assert set(statistics.population_level) == {"pair", "platform"}
    assert {"snapshot", "window"} <= set(statistics.analysis_type)
    assert set(coverage.coverage_level) == {
        "cluster", "pair", "exclusion", "separation_range",
    }
    diagnostics = pd.read_parquet(
        array_directory / "data" / "platform_diagnostics.parquet"
    )
    assert {
        "reference_valid", "reference_invalid_reason", "analysis_valid",
        "analysis_invalid_reason", "analysis_u_m_s", "analysis_v_m_s",
    } <= set(diagnostics.columns)


def test_analysis_sensitivity_reuses_reference_activation_anchor(tmp_path: Path) -> None:
    store = tmp_path / "trajectories.zarr"
    _write_trajectory_store(store)
    output = tmp_path / "epsilon"
    config = tmp_path / "epsilon.yml"
    config.write_text(
        f"""
input:
  trajectories: {store.as_posix()}
  dataset_label: synthetic
  allow_candidate_input: true
selection:
  arrays: [1]
  clusters: all
  pair_scope: within_cluster
  array_anchor: null
activation:
  minimum_active_members: 2
  reference:
    position_method: linear
    velocity_method: centered_difference
    total_span_minutes: 30
    maximum_source_gap_minutes: 30
    require_full_stencil_support: true
    speed_mask_m_s: null
analysis:
  position_method: spline_30
  velocity_method: centered_difference
  total_span_minutes: 30
  maximum_source_gap_minutes: 4
  require_full_stencil_support: true
  speed_mask_m_s: null
  windows:
    - {{id: initial_0_6h, start_hour: 0, end_hour: 6}}
  minimum_reliable_separation_m: null
separation_bins:
  mode: explicit
  edges_m: [1, 100, 1000, 10000]
  axis_scale: log
support:
  observations: 12
  unique_pairs: 3
  platforms: 3
  unique_utc: 12
output:
  root: {output.as_posix()}
""",
        encoding="utf-8",
    )
    result = run_epsilon_workflow(config)
    array_directory = result.run_directory / "array_001"
    inventory = pd.read_csv(array_directory / "clusters.csv")
    cluster = inventory[inventory.cluster_id == "cluster_a"].iloc[0]
    # The reference method anchors at 00:15 (the first complete seven-point stencil).
    assert cluster.activation_time_utc == "2025-01-01T00:15:00Z"
    # The stricter analysis source-gap mask invalidates that fixed snapshot; it is not moved.
    assert cluster.snapshot_status == "unavailable_analysis_has_fewer_than_two_members"
    observations = pd.read_parquet(array_directory / "data" / "pair_observations.parquet")
    assert not observations.is_snapshot.any()
    diagnostics = pd.read_parquet(
        array_directory / "data" / "platform_diagnostics.parquet"
    )
    at_anchor = diagnostics[
        (diagnostics.cluster_id == "cluster_a") & diagnostics.is_snapshot
    ]
    assert len(at_anchor) == 3
    assert at_anchor.reference_valid.sum() == 2
    assert at_anchor.analysis_valid.sum() == 0

    trusted_config = tmp_path / "epsilon_trusted.yml"
    trusted_output = tmp_path / "epsilon_trusted"
    trusted_config.write_text(
        config.read_text(encoding="utf-8").replace(
            "maximum_source_gap_minutes: 4\n  require_full_stencil_support: true",
            "maximum_source_gap_minutes: 4\n  require_full_stencil_support: false",
        ).replace(output.as_posix(), trusted_output.as_posix()),
        encoding="utf-8",
    )
    trusted_result = run_epsilon_workflow(trusted_config)
    trusted_array = trusted_result.run_directory / "array_001"
    trusted_inventory = pd.read_csv(trusted_array / "clusters.csv")
    trusted_cluster = trusted_inventory[
        trusted_inventory.cluster_id == "cluster_a"
    ].iloc[0]
    assert trusted_cluster.activation_time_utc == "2025-01-01T00:15:00Z"
    assert trusted_cluster.snapshot_status == "available_partial_assigned_members"
    trusted_observations = pd.read_parquet(
        trusted_array / "data" / "pair_observations.parquet"
    )
    assert trusted_observations.is_snapshot.sum() == 1
    trusted_manifest = json.loads(
        trusted_result.manifest_path.read_text(encoding="utf-8")
    )
    assert trusted_manifest["scientific_conventions"][
        "analysis_required_validity_point_count"
    ] == 2
    assert not trusted_manifest["effective_configuration"]["analysis"][
        "maximum_source_gap_applied"
    ]


def test_all_array_population_uses_common_anchor_and_fifteen_minute_sampling(
    tmp_path: Path,
) -> None:
    store = tmp_path / "trajectories.zarr"
    _write_trajectory_store(store)
    output = tmp_path / "epsilon_poje"
    config = tmp_path / "epsilon_poje.yml"
    config.write_text(
        f"""
input:
  trajectories: {store.as_posix()}
  dataset_label: synthetic_poje
  allow_candidate_input: true
selection:
  arrays: [1]
  clusters: all
  pair_scope: all_array
  array_anchor:
    method: first_time_all_selected_platforms_reference_valid
activation:
  minimum_active_members: all
  reference:
    position_method: spline_15
    velocity_method: centered_difference
    total_span_minutes: 30
    maximum_source_gap_minutes: 30
    require_full_stencil_support: true
    speed_mask_m_s: null
analysis:
  position_method: spline_15
  velocity_method: centered_difference
  total_span_minutes: 30
  sample_cadence_minutes: 15
  sample_phase_reference: array_anchor
  maximum_source_gap_minutes: 30
  require_full_stencil_support: true
  speed_mask_m_s: null
  windows:
    - {{id: initial_0_6h, start_hour: 0, end_hour: 6}}
  minimum_reliable_separation_m: null
separation_bins:
  mode: explicit
  edges_m: [1, 100, 1000, 10000, 100000]
  axis_scale: log
support:
  observations: 12
  unique_pairs: 3
  platforms: 3
  unique_utc: 12
output:
  root: {output.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_epsilon_workflow(config)
    assert result.activated_cluster_count == 1
    assert result.run_directory.parent.name == "centered_difference_30min_on_15min_grid"
    array_directory = result.run_directory / "array_001"

    inventory = pd.read_csv(array_directory / "clusters.csv")
    assert len(inventory) == 1
    population = inventory.iloc[0]
    assert population.cluster_id == "array_001__all_array"
    assert population.pair_scope == "all_array"
    assert population.assigned_member_count == 4
    assert population.source_cluster_count == 2
    assert population.possible_internal_pair_count == 6
    assert population.reference_active_member_count_at_activation == 4
    assert population.activation_time_utc == "2025-01-01T00:45:00Z"
    assert population.snapshot_pair_observation_count == 6

    observations = pd.read_parquet(
        array_directory / "data" / "pair_observations.parquet"
    )
    assert observations.pair_id.nunique() == 6
    assert observations.loc[observations.is_snapshot, "pair_id"].nunique() == 6
    assert observations.loc[
        observations.pair_cluster_relation == "between_clusters", "pair_id"
    ].nunique() == 3
    observed_times = np.sort(observations.time_utc.unique())
    assert np.all(np.diff(observed_times) == np.timedelta64(15, "m"))

    coverage = pd.read_csv(array_directory / "coverage.csv")
    aggregate = coverage[coverage.coverage_level == "cluster"].iloc[0]
    assert aggregate.scheduled_utc_count == 24
    assert aggregate.possible_pair_count == 6
    assert aggregate.expected_pair_observation_count == 6 * 24
    pair_coverage = coverage[coverage.coverage_level == "pair"]
    assert pair_coverage.loc[
        pair_coverage.pair_cluster_relation == "between_clusters", "pair_id"
    ].nunique() == 3

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    effective = manifest["effective_configuration"]
    assert effective["selection"]["pair_scope"] == "all_array"
    assert effective["analysis"]["sample_cadence_minutes"] == 15
    assert effective["analysis"]["sample_phase_reference"] == "array_anchor"
