"""Numerical tests for observed-start diagnostics and candidate grouping."""

import numpy as np
import pandas as pd

from drifterlab.clustering import (
    candidate_clusters,
    infer_start_cohorts,
    nearest_neighbor_diagnostics,
    pairwise_start_metrics,
)


def starts(platforms, minutes, longitude, latitude=None):
    return pd.DataFrame({
        "platform_id": platforms,
        "observed_start_time": (
            pd.Timestamp("2025-01-01T00:00:00Z") + pd.to_timedelta(minutes, unit="m")
        ),
        "observed_start_lon": longitude,
        "observed_start_lat": latitude if latitude is not None else np.zeros(len(platforms)),
    })


def test_wgs84_pairwise_distance_and_dateline_are_geodesic():
    frame = starts(["a", "b", "c"], [0, 5, 10], [0, .001, 179.999])
    frame.loc[0, "observed_start_lon"] = -179.999
    distance, gap = pairwise_start_metrics(frame)
    assert 222 < distance[0, 2] < 223
    assert gap[0, 2] == 10
    np.testing.assert_allclose(distance, distance.T)
    np.testing.assert_allclose(np.diag(distance), 0)


def test_inferred_cohorts_split_only_above_configured_adjacent_gap():
    frame = starts(["a", "b", "c", "d"], [0, 60, 1500, 1560], [0, 1, 2, 3])
    result = infer_start_cohorts(frame, maximum_adjacent_gap_hours=24)
    assert result.proposed_cohort_id.tolist() == [
        "cohort_001", "cohort_001", "cohort_001", "cohort_001",
    ]
    result = infer_start_cohorts(frame, maximum_adjacent_gap_hours=23)
    assert result.proposed_cohort_id.tolist() == [
        "cohort_001", "cohort_001", "cohort_002", "cohort_002",
    ]


def test_neighbor_ranks_carry_corresponding_time_gap_and_reciprocal_rank():
    frame = starts(["a", "b", "c"], [0, 40, 10], [0, .001, .003])
    frame["cohort_id"] = "array"
    result = nearest_neighbor_diagnostics(frame, ranks=5)
    rows = result.loc[result.platform_id.eq("a")].sort_values("neighbor_rank")
    assert rows.neighbor_platform_id.tolist() == ["b", "c"]
    assert rows.start_time_gap_minutes.tolist() == [40, 10]
    assert rows.reciprocal_neighbor_rank.tolist() == [1, 2]


def test_joint_complete_linkage_enforces_diameter_time_and_capacity_without_chaining():
    frame = starts(
        ["a", "b", "c", "d", "e"],
        [0, 5, 10, 120, 4],
        [0, .001, .002, 0, .0011],
    )
    members, summary, _merges = candidate_clusters(
        frame, maximum_diameter_m=150, maximum_start_spread_minutes=30,
        maximum_members=2, cohort_id="array",
    )
    groups = {
        frozenset(group.platform_id)
        for _cluster, group in members.groupby("candidate_cluster_id")
    }
    assert frozenset({"a", "b"}) in groups or frozenset({"b", "e"}) in groups
    assert all(len(group) <= 2 for group in groups)
    assert not any({"a", "b", "c"} <= group for group in groups)
    assert members.loc[members.platform_id.eq("d"), "candidate_status"].item() == "singleton"
    assert (summary.diameter_m <= 150 + 1e-9).all()
    assert (summary.observed_start_spread_minutes <= 30 + 1e-9).all()


def test_candidate_grouping_is_deterministic_under_input_reordering():
    frame = starts(["a", "b", "c", "d"], [0, 1, 2, 3], [0, .001, .01, .011])
    first = candidate_clusters(
        frame, maximum_diameter_m=200, maximum_start_spread_minutes=10,
        maximum_members=5, cohort_id="array",
    )[0]
    second = candidate_clusters(
        frame.sample(frac=1, random_state=5), maximum_diameter_m=200,
        maximum_start_spread_minutes=10, maximum_members=5, cohort_id="array",
    )[0]
    columns = ["candidate_cluster_id", "platform_id"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values(columns).reset_index(drop=True),
        second[columns].sort_values(columns).reset_index(drop=True),
    )
