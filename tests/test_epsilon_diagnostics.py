from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from drifterlab.workflows.epsilon_diagnostics import run_epsilon_diagnostics


def _write_source_run(path: Path) -> None:
    array = path / "array_001"
    data = array / "data"
    data.mkdir(parents=True)
    pd.DataFrame((
        {
            "array_id": 1,
            "cluster_id": "cluster_a",
            "analysis_type": "inventory",
            "window_id": "all",
            "assigned_member_count": 2,
            "activation_time_utc": "2025-01-01T00:00:00Z",
        },
        {
            "array_id": 1,
            "cluster_id": "cluster_b",
            "analysis_type": "inventory",
            "window_id": "all",
            "assigned_member_count": 1,
            "activation_time_utc": None,
        },
    )).to_csv(array / "clusters.csv", index=False)
    times = pd.to_datetime([
        "2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z",
        "2025-01-01T00:30:00Z",
    ])
    rows = []
    for platform_index, platform_id in enumerate(("a", "b")):
        for time_index, (time, age) in enumerate(zip(times, (0.0, 0.25, 0.5))):
            rows.append({
                "array_id": 1,
                "cluster_id": "cluster_a",
                "platform_id": platform_id,
                "time_utc": time,
                "cluster_age_hours": age,
                "x_m": 100 * platform_index + 10 * time_index,
                "y_m": 50 * platform_index + 5 * time_index,
                "reference_valid": True,
                "analysis_valid": not (platform_id == "b" and time_index == 1),
                "analysis_u_m_s": 0.1 + 0.01 * platform_index + 0.001 * time_index,
                "analysis_v_m_s": 0.02 - 0.005 * platform_index,
            })
    pd.DataFrame(rows).to_parquet(data / "platform_diagnostics.parquet", index=False)
    pd.DataFrame([
        {
            "array_id": 1,
            "cluster_id": "cluster_a",
            "pair_id": "pair_a_b",
            "platform_id_1": "a",
            "platform_id_2": "b",
            "time_utc": time,
            "cluster_age_hours": age,
            "separation_m": 100.0 + 10.0 * index,
            "delta_u_l_m_s": value,
            "delta_u_t_m_s": 0.5 * value,
        }
        for index, (time, age, value) in enumerate(zip(
            times, (0.0, 0.25, 0.5), (-0.01, 0.02, 0.03),
        ))
    ]).to_parquet(data / "pair_observations.parquet", index=False)
    expected_epsilon = -1.25 * ((-0.01) ** 3 + 0.02 ** 3 + 0.03 ** 3) / 330.0
    pd.DataFrame([{
        "array_id": 1,
        "cluster_id": "cluster_a",
        "analysis_type": "window",
        "window_id": "initial_0_12h",
        "separation_bin_id": "bin_000",
        "separation_bin_index": 0,
        "separation_bin_lower_m": 50.0,
        "separation_bin_upper_m": 150.0,
        "separation_bin_nominal_center_m": (50.0 * 150.0) ** 0.5,
        "mean_separation_m": 110.0,
        "epsilon_eff_m2_s3": expected_epsilon,
        "epsilon_without_most_extreme_m2_s3": abs(expected_epsilon),
        "support_ok": True,
        "support_status": "supported",
        "observation_count": 3,
        "unique_pair_count": 1,
        "platform_count": 2,
        "unique_utc_count": 3,
        "maximum_absolute_cube_fraction": 0.75,
        "maximum_lopo_relative_change": 2.0,
    }]).to_csv(array / "epsilon_by_scale.csv", index=False)
    pd.DataFrame([
        {
            "array_id": 1,
            "cluster_id": "cluster_a",
            "analysis_type": "window",
            "window_id": "initial_0_12h",
            "separation_bin_id": "bin_000",
            "removed_platform_id": platform_id,
            "epsilon_after_removal_m2_s3": value,
            "sign_reversal": value > 0,
            "support_ok_after_removal": False,
        }
        for platform_id, value in (("a", abs(expected_epsilon)), ("b", expected_epsilon))
    ]).to_parquet(data / "leave_one_platform_out.parquet", index=False)
    (path / "config_resolved.yaml").write_text(
        yaml.safe_dump({
            "activation": {"minimum_active_members": 2},
            "analysis": {
                "total_span_minutes": 30,
                "four_fifths_coefficient": -1.25,
                "windows": [
                    {"id": "initial_0_6h", "start_hour": 0, "end_hour": 6},
                    {"id": "initial_0_12h", "start_hour": 0, "end_hour": 12},
                ]
            },
            "separation_bins": {
                "resolved_edges_m": [50.0, 150.0, 300.0],
                "resolved_nominal_centers_m": [
                    (50.0 * 150.0) ** 0.5,
                    (150.0 * 300.0) ** 0.5,
                ],
            },
        }),
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps({"run_id": "epsilon-v1-synthetic", "products": {}}) + "\n",
        encoding="utf-8",
    )


def test_trajectory_diagnostics_render_from_completed_epsilon_run(tmp_path: Path) -> None:
    source = tmp_path / "epsilon-v1-synthetic"
    _write_source_run(source)
    output = tmp_path / "diagnostics"
    config = tmp_path / "diagnostics.yml"
    config.write_text(
        f"""
input:
  epsilon_run: {source.as_posix()}
selection:
  arrays: [1]
  clusters: all
time:
  window: maximum_configured
  mark_elapsed_hours: [0, 6, 12]
figures:
  trajectory_window:
    enabled: true
    coordinates: local_projected_xy
    equal_aspect: true
    show_analysis_valid: true
    show_invalid_reconstructed_positions: true
    label_platforms: true
    mark_activation: true
    mark_window_boundaries: true
    dpi: 72
  velocity_availability:
    enabled: true
    dpi: 72
  pair_relative_motion:
    enabled: true
    derivative_span: source_analysis_velocity
    dpi: 72
  time_scale_contributions:
    enabled: true
    dpi: 72
  influence_audit:
    enabled: true
    small_scale_maximum_m: 1000
    dominant_observation_fraction: 0.5
    dominant_platform_relative_change: 0.5
    dpi: 72
output:
  root: {output.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_epsilon_diagnostics(config)
    assert result.figure_count == 5
    assert result.cluster_count == 2
    assert result.run_directory.parent.name == "epsilon-v1-synthetic"
    summary = pd.read_csv(result.run_directory / "diagnostic_figures.csv")
    available = summary[summary.cluster_id == "cluster_a"].iloc[0]
    skipped = summary[summary.cluster_id == "cluster_b"].iloc[0]
    assert available.status == "available"
    assert available.window_end_hour == 12
    assert available.retained_platform_count == 2
    assert available.retained_row_count == 6
    assert skipped.status == "skipped_no_activation"
    figure = result.run_directory / str(available.figure)
    assert figure.is_file()
    assert figure.stat().st_size > 0
    velocity_figure = result.run_directory / str(available.velocity_availability_figure)
    assert velocity_figure.is_file()
    assert velocity_figure.stat().st_size > 0
    pair_figure = result.run_directory / str(available.pair_relative_motion_figure)
    assert pair_figure.is_file()
    assert pair_figure.stat().st_size > 0
    assert available.pair_observation_count == 3
    assert available.pair_count == 1
    time_scale_figure = result.run_directory / str(
        available.time_scale_contributions_figure
    )
    assert time_scale_figure.is_file()
    assert time_scale_figure.stat().st_size > 0
    contribution_table = pd.read_parquet(
        result.run_directory / "data" / "time_scale_contributions.parquet"
    )
    assert len(contribution_table) == 3
    expected_epsilon = -1.25 * ((-0.01) ** 3 + 0.02 ** 3 + 0.03 ** 3) / 330.0
    assert abs(
        contribution_table.epsilon_additive_contribution_m2_s3.sum()
        - expected_epsilon
    ) < 1e-15
    influence_figure = result.run_directory / str(available.influence_audit_figure)
    assert influence_figure.is_file()
    assert influence_figure.stat().st_size > 0
    influence = pd.read_csv(result.run_directory / "influence_audit.csv")
    assert len(influence) == 1
    assert influence.iloc[0].audit_classification == "event_and_platform_sign_sensitive"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["run_id"] == "epsilon-v1-synthetic"
    assert manifest["analysis"]["window_end_hour"] == 12
    assert manifest["analysis"]["available_cluster_count"] == 1
    assert manifest["analysis"]["trajectory_figure_count"] == 1
    assert manifest["analysis"]["velocity_availability_figure_count"] == 1
    assert manifest["analysis"]["pair_relative_motion_figure_count"] == 1
    assert manifest["analysis"]["time_scale_contributions_figure_count"] == 1
    assert manifest["analysis"]["time_scale_contribution_row_count"] == 3
    assert manifest["analysis"]["influence_audit_figure_count"] == 1
    assert manifest["analysis"]["influence_audit_row_count"] == 1
    assert manifest["analysis"]["available_figure_count"] == 5
