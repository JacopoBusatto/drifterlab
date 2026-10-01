"""Focused tests for reconstructed-grid initial deployment clustering."""

import numpy as np

from drifterlab.reconstruction import InitialClusterRule, assign_initial_clusters


BASE = np.datetime64("2025-01-01T00:00:00", "ns")


def times(minutes):
    return BASE + np.asarray(minutes, dtype="timedelta64[m]")


def rule(*, pair=100.0, diameter=200.0, seconds=600.0, members=5, assignment="infer"):
    return InitialClusterRule(assignment, seconds, pair, diameter, members)


def test_asynchronous_pair_uses_first_common_reconstructed_instant():
    grid = times([0, 5, 10])
    longitude = np.asarray([
        [0.0, 0.01, 0.02],
        [np.nan, 0.01, 0.02],
    ])
    latitude = np.asarray([
        [0.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
    ])
    original = longitude.copy()
    result = assign_initial_clusters(
        ["A", "B"], [1, 1], times([0, 5]), grid, longitude, latitude,
        rule(pair=10, diameter=10),
    )
    assert {item.cluster_id for item in result.assignments} == {
        "array_001__cluster_001"
    }
    diagnostic = result.pair_diagnostics[0]
    assert diagnostic.first_common_time == grid[1]
    assert diagnostic.distance_m == 0
    assert diagnostic.candidate_link is True
    np.testing.assert_array_equal(longitude, original)


def test_observed_start_reference_uses_start_coordinates_and_keeps_common_diagnostic():
    grid = times([0, 5, 10])
    longitude = np.asarray([
        [0.0, 0.01, 0.02],
        [np.nan, 0.0001, 0.0002],
    ])
    latitude = np.asarray([
        [0.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
    ])
    result = assign_initial_clusters(
        ["A", "B"], [1, 1], times([0, 5]), grid, longitude, latitude,
        rule(pair=100, diameter=100), distance_reference="observed_starts",
        start_longitude=[0.0, 0.0001], start_latitude=[0.0, 0.0],
    )
    assert result.assignments[0].cluster_id == result.assignments[1].cluster_id
    diagnostic = result.pair_diagnostics[0]
    assert diagnostic.distance_reference == "observed_starts"
    assert diagnostic.distance_m == diagnostic.observed_start_distance_m
    assert diagnostic.observed_start_distance_m < 100
    assert diagnostic.first_common_grid_distance_m > 1000
    assert result.assignments[0].cluster_observed_start_diameter_m < 100
    assert result.assignments[0].cluster_first_common_grid_diameter_m > 1000


def test_complete_diameter_prevents_nearest_neighbor_chaining():
    grid = times([0, 5])
    longitude = np.asarray([[0.0, 0.0], [0.0005, 0.0005], [0.001, 0.001]])
    latitude = np.zeros_like(longitude)
    result = assign_initial_clusters(
        ["A", "B", "C"], [1, 1, 1], times([0, 0, 0]), grid,
        longitude, latitude, rule(pair=70, diameter=90, members=None),
    )
    assert sorted({item.cluster_size for item in result.assignments}) == [1, 2]
    assert sum(item.candidate_link for item in result.pair_diagnostics) == 2


def test_member_cap_is_a_cluster_constraint_not_platform_exclusivity():
    grid = times([0, 5])
    longitude = np.asarray([[0.0, 0.0], [0.0001, 0.0001], [0.0002, 0.0002]])
    latitude = np.zeros_like(longitude)
    result = assign_initial_clusters(
        ["A", "B", "C"], [1, 1, 1], times([0, 0, 0]), grid,
        longitude, latitude, rule(pair=100, diameter=100, members=2),
    )
    assert sorted({item.cluster_size for item in result.assignments}) == [1, 2]


def test_single_cluster_override_bypasses_assignment_thresholds_but_reports_them():
    grid = times([0, 5])
    longitude = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    latitude = np.zeros_like(longitude)
    result = assign_initial_clusters(
        ["A", "B"], [4, 4], times([0, 5]), grid, longitude, latitude,
        rule(), {4: rule(pair=10, diameter=20, seconds=10, assignment="single_cluster")},
    )
    assert {item.cluster_id for item in result.assignments} == {
        "array_004__cluster_001"
    }
    assert {item.cluster_assignment_status for item in result.assignments} == {
        "configured_single_cluster"
    }
    assert result.pair_diagnostics[0].candidate_link is False
    assert result.assignments[0].cluster_diameter_m > 100_000


def test_single_cluster_records_unavailable_distance_without_trimming():
    grid = times([0, 5, 10])
    longitude = np.asarray([[0.0, np.nan, np.nan], [np.nan, np.nan, 0.0]])
    latitude = longitude.copy()
    result = assign_initial_clusters(
        ["A", "B"], [1, 1], times([0, 10]), grid, longitude, latitude,
        rule(assignment="single_cluster"),
    )
    assert result.assignments[0].cluster_diameter_m is None
    assert result.assignments[0].cluster_unavailable_pair_count == 1
    assert result.pair_diagnostics[0].first_common_time is None


def test_numbering_is_deterministic_under_platform_input_reordering():
    grid = times([0, 5])
    platforms = ["C", "A", "B"]
    starts = times([5, 0, 0])
    longitude = np.asarray([[1.0, 1.0], [0.0, 0.0], [0.0001, 0.0001]])
    latitude = np.zeros_like(longitude)
    first = assign_initial_clusters(
        platforms, [1, 1, 1], starts, grid, longitude, latitude, rule(),
    )
    order = [1, 2, 0]
    second = assign_initial_clusters(
        [platforms[index] for index in order], [1, 1, 1], starts[order], grid,
        longitude[order], latitude[order], rule(),
    )
    fields = lambda result: {
        item.platform_id: (item.cluster_id, item.member_id, item.cluster_size)
        for item in result.assignments
    }
    assert fields(first) == fields(second)
