from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from drifterlab.poje import (
    IncrementDistributionScale,
    SeparationMemoryGroup,
    calculate_figure4_dispersion,
    calculate_figure5_self_similarity,
    calculate_figure6_increment_distributions,
    calculate_figure7_structure_functions,
)
from drifterlab.workflows.poje import run_poje_workflow


def _platform_diagnostics() -> pd.DataFrame:
    times = pd.to_datetime([
        "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z",
        "2025-01-03T00:00:00Z", "2025-01-09T00:00:00Z",
    ])
    positions = {
        "p1": [0.0, 0.0, 0.0, 0.0],
        "p2": [100.0, 200.0, 400.0, 900.0],
        "p3": [1000.0, 1000.0, 1000.0, 1000.0],
    }
    velocities = {
        "p1": [0.0, 0.0, 0.0, 0.0],
        "p2": [0.0, 1.0, 2.0, 3.0],
        "p3": [0.0, 2.0, 4.0, 6.0],
    }
    rows = []
    for platform_id, x_values in positions.items():
        for time, age, x, u in zip(
            times, (0.0, 24.0, 48.0, 192.0), x_values, velocities[platform_id],
        ):
            rows.append({
                "array_id": 1,
                "cluster_id": "array_001__all_array",
                "source_cluster_id": f"cluster_{platform_id}",
                "platform_id": platform_id,
                "time_utc": time,
                "cluster_age_hours": age,
                "x_m": x,
                "y_m": 0.0,
                "analysis_valid": True,
                "analysis_u_m_s": u,
                "analysis_v_m_s": 0.0,
            })
    return pd.DataFrame(rows)


def _figure6_pair_observations() -> pd.DataFrame:
    rows = []
    values = (-4.0, 0.0, 1.0, 2.0, 2.0)
    for scale_index, separation in enumerate((250, 500, 1000, 5000, 7000, 10000)):
        for observation_index, increment in enumerate(values):
            rows.append({
                "array_id": 1,
                "pair_id": f"pair_{scale_index}_{observation_index}",
                "platform_id_1": f"p{observation_index}",
                "platform_id_2": f"p{observation_index + 1}",
                "time_utc": pd.Timestamp("2025-01-01T00:00:00Z")
                + pd.Timedelta(hours=observation_index),
                "cluster_age_hours": float(observation_index),
                "separation_m": float(separation),
                "delta_u_l_m_s": increment,
                "delta_u_l_delta_u_t_squared_m3_s3": increment * 0.25,
            })
    return pd.DataFrame(rows)


def _figure7_epsilon_by_scale() -> pd.DataFrame:
    observations = _figure6_pair_observations()
    edges = (0, 300, 600, 1200, 6000, 8000, 12000)
    centers = (250, 500, 1000, 5000, 7000, 10000)
    rows = []
    for index, center in enumerate(centers):
        selected = observations[observations.separation_m == center]
        values = selected.delta_u_l_m_s.to_numpy(dtype=float)
        mean_separation = float(selected.separation_m.mean())
        second = float(np.mean(values ** 2))
        third = float(np.mean(values ** 3))
        epsilon = -1.25 * third / mean_separation
        coriolis = 5e-5 + index * 1e-7
        rows.append({
            "analysis_type": "window",
            "window_id": "initial_0_12d",
            "separation_bin_index": index,
            "separation_bin_lower_m": edges[index],
            "separation_bin_upper_m": edges[index + 1],
            "separation_bin_nominal_center_m": center,
            "mean_separation_m": mean_separation,
            "raw_delta_u_l_m_s_second": second,
            "raw_delta_u_l_m_s_third": third,
            "epsilon_eff_m2_s3": epsilon,
            "observation_count": len(selected),
            "unique_pair_count": selected.pair_id.nunique(),
            "platform_count": len(
                set(selected.platform_id_1) | set(selected.platform_id_2)
            ),
            "unique_utc_count": selected.time_utc.nunique(),
            "support_ok": True,
            "support_status": "supported",
            "mean_bin_coriolis_s_inverse": coriolis,
            "poje_longitudinal_rossby_number": np.sqrt(second) / (
                coriolis * center
            ),
            "epsilon_ci_lower_m2_s3": epsilon * 0.9,
            "epsilon_ci_upper_m2_s3": epsilon * 1.1,
            "confidence_interval_status": "available",
        })
    return pd.DataFrame(rows)


def test_figure4_dispersion_uses_poije_scalar_separation_change() -> None:
    result = calculate_figure4_dispersion(
        _platform_diagnostics(),
        array_id=1,
        window_id="initial_0_12d",
        window_end_hours=288,
        absolute_plot_end_days=8,
        initial_pair_separation_maximum_m=300,
        ballistic_fit_end_hours=24,
        richardson_fit_start_days=1,
        richardson_fit_end_days=8,
        bootstrap_replicates=20,
        bootstrap_confidence_level=0.95,
        bootstrap_random_seed=7,
    )
    assert len(result.pair_inventory) == 1
    assert result.pair_inventory.initial_separation_m.iloc[0] == pytest.approx(100)
    np.testing.assert_allclose(
        result.timeseries.adjusted_radial_relative_dispersion_m2,
        [0, 100 ** 2, 300 ** 2, 800 ** 2],
    )
    np.testing.assert_allclose(
        result.timeseries.relative_ci_lower_m2,
        result.timeseries.adjusted_radial_relative_dispersion_m2,
    )
    np.testing.assert_allclose(
        result.timeseries.absolute_mean_squared_displacement_m2,
        [0, 100 ** 2 / 3, 300 ** 2 / 3, 800 ** 2 / 3],
    )
    assert result.summary["initial_pair_count"] == 1


def test_figure5_uses_complete_pair_matrix_and_anchor_velocity_change() -> None:
    result = calculate_figure5_self_similarity(
        _platform_diagnostics(),
        array_id=1,
        window_id="initial_0_12d",
        window_end_hours=288,
        memory_groups=(
            SeparationMemoryGroup("days_1_3", 1, 3),
            SeparationMemoryGroup("days_7_9", 7, 9),
        ),
        similarity_bin_count=10,
        kolmogorov_constant=6.5,
    )
    assert result.summary["figure5_complete_pair_count"] == 3
    zero_lag = result.separation_memory_samples[
        np.isclose(result.separation_memory_samples.normalized_lag_tau_over_t, 0)
    ]
    np.testing.assert_allclose(zero_lag.normalized_separation_correlation, 1)
    assert set(zero_lag.contributing_pair_count) == {3}
    day_one = result.lagrangian_structure.query("elapsed_days == 1").iloc[0]
    assert day_one.lagrangian_velocity_structure_second_m2_s2 == pytest.approx(5 / 3)
    assert day_one.epsilon_proxy_m2_s3 == pytest.approx((5 / 3) / (6.5 * 86400))
    assert "similarity_bin_index" in result.separation_memory_samples


def test_figure6_normalizes_raw_increments_and_preserves_histogram_mass() -> None:
    scales = tuple(
        IncrementDistributionScale(
            f"scale_{center}", "small" if center <= 1000 else "large", center,
        )
        for center in (250, 500, 1000, 5000, 7000, 10000)
    )
    result = calculate_figure6_increment_distributions(
        _figure6_pair_observations(),
        array_id=1,
        window_id="initial_0_12d",
        window_end_hours=288,
        source_edges_m=np.array([0, 300, 600, 1200, 6000, 8000, 12000]),
        source_centers_m=np.array([250, 500, 1000, 5000, 7000, 10000]),
        scales=scales,
        maximum_center_relative_difference=0.1,
        normalized_increment_limit=5,
        histogram_bin_count=100,
        gaussian_core_maximum_absolute_normalized_increment=1.5,
    )
    assert len(result.statistics) == 6
    assert (result.statistics.observation_count == 5).all()
    assert (result.statistics.central_skewness < 0).all()
    assert not np.allclose(
        result.observations.normalized_delta_u_l_by_sigma.mean(), 0,
    )
    mass = result.distributions.groupby("figure6_scale_id").probability_mass.sum()
    np.testing.assert_allclose(mass, 1)
    assert {
        "unit_variance_gaussian_density", "fitted_core_gaussian_density",
    }.issubset(result.distributions.columns)
    assert (result.statistics.gaussian_core_fit_status == "success").all()
    assert (result.statistics.gaussian_core_fit_width > 0).all()
    normalized = result.observations.query("figure6_scale_id == 'scale_250'")[
        "normalized_delta_u_l_by_sigma"
    ].to_numpy()
    assert np.mean((normalized - normalized.mean()) ** 2) == pytest.approx(1)


def test_figure7_preserves_four_fifths_identity_and_mixed_relation() -> None:
    result = calculate_figure7_structure_functions(
        _figure6_pair_observations(),
        _figure7_epsilon_by_scale(),
        array_id=1,
        window_id="initial_0_12d",
        window_end_hours=288,
        source_edges_m=np.array([0, 300, 600, 1200, 6000, 8000, 12000]),
        source_centers_m=np.array([250, 500, 1000, 5000, 7000, 10000]),
        fit_minimum_separation_m=100,
        fit_maximum_separation_m=10000,
        panel_a_maximum_separation_m=10000,
        panel_b_maximum_separation_m=10000,
    )
    frame = result.structure_functions
    np.testing.assert_allclose(
        frame.epsilon_four_fifths_m2_s3,
        frame.epsilon_four_fifths_recomputed_m2_s3,
    )
    np.testing.assert_allclose(
        frame.epsilon_four_fifths_identity_absolute_error_m2_s3, 0,
    )
    first = frame.iloc[0]
    assert first.epsilon_mixed_relation_m2_s3 == pytest.approx(9.35 / (2 * 250))
    assert result.summary["figure7_second_order_free_fit_exponent"] == pytest.approx(0)
    np.testing.assert_allclose(
        frame.latitude_dependent_rossby_identity_absolute_error, 0,
    )


def test_poje_workflow_writes_figure4_bundle(tmp_path: Path) -> None:
    source = tmp_path / "epsilon-v1-source"
    array = source / "array_001"
    data = array / "data"
    data.mkdir(parents=True)
    pd.DataFrame([{
        "array_id": 1,
        "cluster_id": "array_001__all_array",
        "pair_scope": "all_array",
        "assigned_member_count": 3,
        "activation_time_utc": "2025-01-01T00:00:00Z",
    }]).to_csv(array / "clusters.csv", index=False)
    _platform_diagnostics().to_parquet(data / "platform_diagnostics.parquet", index=False)
    _figure6_pair_observations().to_parquet(
        data / "pair_observations.parquet", index=False,
    )
    _figure7_epsilon_by_scale().to_csv(array / "epsilon_by_scale.csv", index=False)
    (source / "config_resolved.yaml").write_text(
        yaml.safe_dump({
            "input": {"allow_candidate_input": True},
            "selection": {"pair_scope": "all_array"},
            "analysis": {
                "sample_cadence_minutes": 15,
                "windows": [{
                    "id": "initial_0_12d", "start_hour": 0, "end_hour": 288,
                }],
            },
            "separation_bins": {
                "resolved_edges_m": [0, 300, 600, 1200, 6000, 8000, 12000],
                "resolved_nominal_centers_m": [250, 500, 1000, 5000, 7000, 10000],
            },
        }),
        encoding="utf-8",
    )
    (source / "manifest.json").write_text(
        json.dumps({"run_id": "epsilon-v1-source", "products": {}}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "poje"
    config = tmp_path / "poje.yml"
    config.write_text(
        f"""
input:
  epsilon_run: {source.as_posix()}
selection:
  arrays: [1]
time:
  window: maximum_configured
figure4:
  initial_pair_separation_maximum_m: 300
  absolute_plot_end_days: 8
  ballistic_fit_end_hours: 24
  richardson_fit_start_days: 1
  richardson_fit_end_days: 8
  bootstrap:
    replicates: 20
    confidence_level: 0.95
    random_seed: 7
    unit: fixed_initial_pair_identity_synchronized_across_times
  figure:
    dpi: 72
figure5:
  separation_memory:
    groups:
      - {{id: days_1_3, start_day: 1, end_day: 3}}
      - {{id: days_7_9, start_day: 7, end_day: 9}}
    similarity_bin_count: 10
    correlation_time_fraction: 0.3333333333333333
  lagrangian:
    kolmogorov_constant: 6.5
  figure:
    dpi: 72
figure6:
  distributions:
    scales:
      - {{id: small_0250m, panel_group: small, target_center_m: 250}}
      - {{id: small_0500m, panel_group: small, target_center_m: 500}}
      - {{id: small_1000m, panel_group: small, target_center_m: 1000}}
      - {{id: large_5000m, panel_group: large, target_center_m: 5000}}
      - {{id: large_7000m, panel_group: large, target_center_m: 7000}}
      - {{id: large_10000m, panel_group: large, target_center_m: 10000}}
    maximum_center_relative_difference: 0.1
    normalized_increment_limit: 5
    histogram_bin_count: 100
    gaussian_core_maximum_absolute_normalized_increment: 1.5
  figure:
    dpi: 72
figure7:
  structure_functions:
    fit_minimum_separation_m: 100
    fit_maximum_separation_m: 10000
    panel_a_maximum_separation_m: 10000
    panel_b_maximum_separation_m: 10000
    rossby_reference_values: [1.0, 0.1]
  figure:
    epsilon_linear_threshold_m2_s3: 0.00000001
    dpi: 72
output:
  root: {output.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_poje_workflow(config)
    assert result.array_count == 1
    assert result.figure_count == 4
    array_output = result.run_directory / "array_001"
    assert (array_output / "dispersion.csv").is_file()
    assert (array_output / "pairs.csv").is_file()
    assert (array_output / "data" / "pair_dispersion.parquet").is_file()
    figure = array_output / "figures" / "figure_04_dispersion.png"
    assert figure.is_file() and figure.stat().st_size > 0
    assert (array_output / "separation_memory.csv").is_file()
    assert (array_output / "lagrangian_structure.csv").is_file()
    assert (array_output / "data" / "separation_memory_samples.parquet").is_file()
    figure5 = array_output / "figures" / "figure_05_self_similarity.png"
    assert figure5.is_file() and figure5.stat().st_size > 0
    assert (array_output / "increment_distributions.csv").is_file()
    assert (array_output / "increment_statistics.csv").is_file()
    assert (array_output / "data" / "figure_06_pair_observations.parquet").is_file()
    figure6 = array_output / "figures" / "figure_06_increment_distributions.png"
    assert figure6.is_file() and figure6.stat().st_size > 0
    assert (array_output / "structure_functions.csv").is_file()
    figure7 = array_output / "figures" / "figure_07_structure_functions.png"
    assert figure7.is_file() and figure7.stat().st_size > 0
    summary = pd.read_csv(result.run_directory / "summary.csv")
    assert summary.initial_pair_count.iloc[0] == 1
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["analysis"]["implemented_figures"] == [
        "figure_04", "figure_05", "figure_06", "figure_07",
    ]
    assert manifest["source"]["candidate_input_allowed"] is True
