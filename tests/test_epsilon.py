from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from drifterlab.epsilon import (
    BinSupportRequirements,
    bin_indices,
    epsilon_from_samples,
    first_cluster_activation,
    numeric_distribution_summary,
    pair_kinematics,
    summarize_separation_bins,
)
from drifterlab.epsilon_stage3 import (
    conditional_q_histogram,
    leave_one_platform_out,
    symmetric_log_edges,
    synchronized_block_bootstrap,
)
from drifterlab.trajectories.kinematics import centered_supported_velocity


def _five_minute_times(count: int) -> np.ndarray:
    return np.datetime64("2025-01-01T00:00:00", "ns") + np.arange(count) * np.timedelta64(5, "m")


def test_centered_velocity_uses_two_endpoints_and_seven_point_validity() -> None:
    times = _five_minute_times(13)
    seconds = np.arange(13) * 300.0
    x = np.vstack((2.0 * seconds, -0.5 * seconds))
    y = np.vstack((0.25 * seconds, 3.0 * seconds))
    gaps = np.full_like(x, 5.0)
    result = centered_supported_velocity(
        x, y, times, gaps,
        total_span_minutes=30,
        maximum_source_gap_minutes=30,
    )
    assert result.half_span_steps == 3
    assert result.stencil_point_count == 7
    assert result.required_validity_point_count == 7
    assert result.full_stencil_support_required
    assert result.total_span_seconds == 1800
    assert result.u_m_s[:, 6] == pytest.approx([2.0, -0.5])
    assert result.v_m_s[:, 6] == pytest.approx([0.25, 3.0])

    # A bad intervening point invalidates t=30 min even though the two derivative
    # endpoints at t-15 and t+15 remain finite and supported.
    x_bad = x.copy()
    x_bad[0, 5] = np.nan
    invalid_position = centered_supported_velocity(
        x_bad, y, times, gaps,
        total_span_minutes=30,
        maximum_source_gap_minutes=30,
    )
    assert not invalid_position.valid[0, 6]
    assert invalid_position.invalid_reason[0, 6] == "nonfinite_position_in_stencil"

    gaps_bad = gaps.copy()
    gaps_bad[0, 5] = 31
    invalid_support = centered_supported_velocity(
        x, y, times, gaps_bad,
        total_span_minutes=30,
        maximum_source_gap_minutes=30,
    )
    assert not invalid_support.valid[0, 6]
    assert invalid_support.invalid_reason[0, 6] == "source_gap_exceeded_in_stencil"

    trusted_reconstruction = centered_supported_velocity(
        x_bad, y, times, gaps_bad,
        total_span_minutes=30,
        maximum_source_gap_minutes=30,
        require_full_stencil_support=False,
    )
    assert trusted_reconstruction.valid[0, 6]
    assert trusted_reconstruction.u_m_s[0, 6] == pytest.approx(2.0)
    assert trusted_reconstruction.required_validity_point_count == 2
    assert not trusted_reconstruction.full_stencil_support_required

    endpoint_bad = x.copy()
    endpoint_bad[0, 3] = np.nan
    missing_endpoint = centered_supported_velocity(
        endpoint_bad, y, times, gaps,
        total_span_minutes=30,
        maximum_source_gap_minutes=30,
        require_full_stencil_support=False,
    )
    assert not missing_endpoint.valid[0, 6]


def test_uniform_translation_has_zero_increment_and_swapping_labels_is_invariant() -> None:
    values = pair_kinematics(
        np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([-2.0]),
        np.array([3.0]), np.array([4.0]), np.array([1.0]), np.array([-2.0]),
    )
    assert values["valid"].tolist() == [True]
    assert values["delta_u_l_m_s"][0] == pytest.approx(0)
    assert values["delta_u_t_m_s"][0] == pytest.approx(0)
    assert values["q_m2_s3"][0] == pytest.approx(0)

    forward = pair_kinematics(
        np.array([0.0]), np.array([1.0]), np.array([0.2]), np.array([0.1]),
        np.array([4.0]), np.array([3.0]), np.array([-0.4]), np.array([0.7]),
    )
    reverse = pair_kinematics(
        np.array([4.0]), np.array([3.0]), np.array([-0.4]), np.array([0.7]),
        np.array([0.0]), np.array([1.0]), np.array([0.2]), np.array([0.1]),
    )
    for key in ("separation_m", "delta_u_l_m_s", "delta_u_t_m_s", "q_m2_s3"):
        assert reverse[key] == pytest.approx(forward[key])


def test_approaching_and_separating_pairs_have_expected_q_signs() -> None:
    approaching = pair_kinematics(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]),
        np.array([10.0]), np.array([0.0]), np.array([-1.0]), np.array([0.0]),
    )
    separating = pair_kinematics(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]),
        np.array([10.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]),
    )
    assert approaching["delta_u_l_m_s"][0] < 0
    assert approaching["q_m2_s3"][0] > 0
    assert separating["delta_u_l_m_s"][0] > 0
    assert separating["q_m2_s3"][0] < 0


def test_zero_separation_is_invalid() -> None:
    values = pair_kinematics(*[np.array([0.0])] * 8)
    assert values["valid"].tolist() == [False]
    assert np.isnan(values["delta_u_l_m_s"][0])
    assert np.isnan(values["q_m2_s3"][0])


def test_activation_uses_cluster_clock_and_does_not_restart_after_gap() -> None:
    usable = np.array([
        [False, True, True, False, True],
        [False, False, True, False, True],
        [False, False, False, False, True],
    ])
    assert first_cluster_activation(usable, 2) == 2
    assert first_cluster_activation(usable, "all") == 4
    # A later gap does not alter the already selected activation index.
    assert not usable[:, 3].any()


def test_instantaneous_bins_and_epsilon_weighted_q_identity() -> None:
    edges = np.array([0.0, 10.0, 20.0])
    assert bin_indices(np.array([1.0, 15.0, 20.0, 25.0]), edges).tolist() == [0, 1, 1, 2]
    separation = np.array([2.0, 8.0])
    longitudinal = np.array([1.0, 2.0])
    result = epsilon_from_samples(separation, longitudinal)
    q = -1.25 * longitudinal ** 3 / separation
    assert result["epsilon_eff_m2_s3"] == pytest.approx(
        np.sum(separation * q) / np.sum(separation)
    )
    assert result["mean_q_m2_s3"] != pytest.approx(result["epsilon_eff_m2_s3"])
    assert result["identity_absolute_error_m2_s3"] < 1e-15


def test_raw_structure_ratios_are_distinct_from_central_descriptive_moments() -> None:
    frame = pd.DataFrame({"increment": [1.0, 2.0, 4.0, 8.0]})
    summary = numeric_distribution_summary(frame, ("increment",)).iloc[0]
    assert summary.raw_value_third_ratio != pytest.approx(summary.central_value_skewness)
    assert summary.raw_value_fourth_ratio != pytest.approx(summary.central_value_flatness)


def test_snapshot_and_temporal_support_are_labelled_per_bin() -> None:
    observations = pd.DataFrame({
        "time": np.array(["2025-01-01T00:00", "2025-01-01T00:05"], dtype="datetime64[ns]"),
        "cluster_age_hours": [0.0, 1 / 12],
        "pair_id": ["p", "p"],
        "platform_id_1": ["a", "a"],
        "platform_id_2": ["b", "b"],
        "separation_m": [100.0, 110.0],
        "delta_u_l_m_s": [-0.1, -0.2],
        "delta_u_t_m_s": [0.05, 0.06],
        "q_m2_s3": [1.25e-5, 9.09e-5],
        "midpoint_latitude": [70.0, 70.0],
    })
    requirements = BinSupportRequirements(12, 3, 3, 12)
    snapshot = summarize_separation_bins(
        observations.iloc[:1], np.array([50.0, 200.0]), np.array([100.0]),
        product_kind="snapshot", scheduled_utc_count=1, requirements=requirements,
    ).iloc[0]
    temporal = summarize_separation_bins(
        observations, np.array([50.0, 200.0]), np.array([100.0]),
        product_kind="window", scheduled_utc_count=72, requirements=requirements,
    ).iloc[0]
    assert snapshot.support_status == "descriptive_snapshot_single_pair"
    assert temporal.support_status.startswith("low_support_")
    assert temporal.unique_utc_count == 2
    expected_f = 2 * 7.292115e-5 * math.sin(math.radians(70))
    expected_ro = math.sqrt(np.mean(np.array([-0.1, -0.2]) ** 2)) / (
        expected_f * np.mean([100.0, 110.0])
    )
    assert temporal.poje_longitudinal_rossby_number == pytest.approx(expected_ro)


def test_conditional_histogram_columns_are_probability_mass_or_missing() -> None:
    observations = pd.DataFrame({
        "separation_m": [2.0, 2.0, 20.0],
        "q_m2_s3": [-1e-5, 1e-5, 2e-5],
    })
    q_edges = symmetric_log_edges(1e-3, 1e-6, 2)
    result = conditional_q_histogram(
        observations, np.array([1.0, 10.0, 100.0, 1000.0]), q_edges,
    )
    totals = result.groupby("separation_bin_id").conditional_probability_mass.sum(
        min_count=1
    )
    assert totals.loc["bin_000"] == pytest.approx(1)
    assert totals.loc["bin_001"] == pytest.approx(1)
    assert np.isnan(totals.loc["bin_002"])


def test_synchronized_bootstrap_is_deterministic_and_preserves_empty_blocks() -> None:
    times = _five_minute_times(12)
    observations = pd.DataFrame({
        "cluster_age_hours": np.arange(12) / 12,
        "separation_m": np.full(12, 100.0),
        "delta_u_l_m_s": np.linspace(-0.2, 0.2, 12),
    })
    arguments = dict(
        separation_edges_m=np.array([50.0, 200.0]),
        window_start_hour=0.0,
        window_end_hour=1.0,
        block_minutes=30,
        confidence_level=0.95,
        replicates=200,
        random_seed=7,
        identity="array|cluster|window",
        minimum_contributing_blocks=2,
        minimum_successful_replicate_fraction=0.95,
        support_by_bin={"bin_000": True},
    )
    first = synchronized_block_bootstrap(observations, **arguments)
    second = synchronized_block_bootstrap(observations, **arguments)
    pd.testing.assert_frame_equal(first, second)
    row = first.iloc[0]
    assert row.scheduled_block_count == 2
    assert row.contributing_block_count == 2
    assert row.confidence_interval_status == "available"
    assert row.epsilon_ci_lower_m2_s3 <= row.epsilon_ci_upper_m2_s3


def test_leave_one_platform_out_removes_every_pair_containing_platform() -> None:
    observations = pd.DataFrame({
        "time_utc": np.array(["2025-01-01T00:00"] * 3, dtype="datetime64[ns]"),
        "pair_id": ["ab", "ac", "bc"],
        "platform_id_1": ["a", "a", "b"],
        "platform_id_2": ["b", "c", "c"],
        "separation_m": [100.0, 100.0, 100.0],
        "delta_u_l_m_s": [0.1, 0.2, 0.3],
    })
    result = leave_one_platform_out(
        observations, np.array([50.0, 200.0]),
        support_minimum_observations=1,
        support_minimum_pairs=1,
        support_minimum_platforms=2,
        support_minimum_utc=1,
    )
    removed_a = result[result.removed_platform_id == "a"].iloc[0]
    assert removed_a.remaining_observation_count == 1
    assert removed_a.remaining_unique_pair_count == 1
    assert removed_a.remaining_platform_count == 2
