"""Numerical tests for candidate encounter pairs."""

import numpy as np

from drifterlab.pairs import circular_mean_longitude, find_candidate_pairs


def test_circular_mean_crosses_antimeridian():
    result = circular_mean_longitude(np.asarray([179.0]), np.asarray([-179.0]))
    assert abs(abs(float(result[0])) - 180.0) < 1e-12


def test_first_crossing_and_window_is_measured_from_both_starts():
    time = np.asarray([
        "2025-01-01T00:00:00", "2025-01-01T00:01:00",
        "2025-01-01T00:02:00", "2025-01-01T00:03:00",
    ], dtype="datetime64[ns]")
    starts = np.asarray([
        "2025-01-01T00:00:00", "2025-01-01T00:01:00",
    ], dtype="datetime64[ns]")
    longitude = np.asarray([
        [0.0, 0.0, 0.0, 0.0],
        [0.02, 0.01, 0.004, 0.001],
    ])
    latitude = np.zeros_like(longitude)

    limited = find_candidate_pairs(
        ["2", "1"], time, starts, longitude, latitude,
        maximum_distance_m=500,
        maximum_seconds_from_each_observed_start=90,
    )
    assert limited.candidates == ()
    assert limited.overlapping_pair_count == 1

    chance = find_candidate_pairs(
        ["2", "1"], time, starts, longitude, latitude,
        maximum_distance_m=500,
        maximum_seconds_from_each_observed_start=None,
    )
    candidate = chance.candidates[0]
    assert (candidate.platform_id_1, candidate.platform_id_2) == ("1", "2")
    assert candidate.encounter_index == 2
    assert candidate.encounter_delay_seconds_1 == 60
    assert candidate.encounter_delay_seconds_2 == 120
    assert candidate.encounter_distance_m < 500
    # The later, closer point must not replace the first threshold crossing.
    assert candidate.post_encounter_observations == 2


def test_a_platform_can_be_selected_in_multiple_pairs():
    time = np.asarray(["2025-01-01"], dtype="datetime64[ns]")
    starts = np.repeat(time, 3)
    longitude = np.asarray([[0.0], [0.001], [0.002]])
    latitude = np.zeros_like(longitude)
    result = find_candidate_pairs(
        ["a", "b", "c"], time, starts, longitude, latitude,
        maximum_distance_m=250,
        maximum_seconds_from_each_observed_start=0,
    )
    assert result.possible_pair_count == 3
    assert [(item.platform_id_1, item.platform_id_2) for item in result.candidates] == [
        ("a", "b"), ("a", "c"), ("b", "c"),
    ]
