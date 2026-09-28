"""Focused scientific tests for generic native-position QC v2."""

import json

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drifterlab.io.position import NativeTrajectory
from drifterlab.qc.drogue import ResolvedDrogueDecision
from drifterlab.qc.native_position import (
    DeploymentBoundary, NativePositionConfig, TemporalSegmentConfig,
    resolved_position_trajectory, run_native_position_qc,
)


def trajectory(lon, *, seconds=None, platform="p", source_indices=None):
    lon = np.asarray(lon, dtype=float)
    n = len(lon)
    seconds = np.arange(n) * 300 if seconds is None else np.asarray(seconds)
    time = np.datetime64("2025-01-01", "ns") + seconds.astype("timedelta64[s]")
    return NativeTrajectory(
        platform, time, lon, np.zeros(n),
        np.arange(n) if source_indices is None else np.asarray(source_indices),
        Path(f"{platform}.mat"), f"hash-{platform}",
    )


def trajectory_xy(lon, lat, seconds, source_indices, *, platform="p"):
    lon, lat = np.asarray(lon), np.asarray(lat)
    time = np.datetime64("2025-01-01", "ns") + np.asarray(seconds).astype("timedelta64[s]")
    return NativeTrajectory(
        platform, time, lon, lat, np.asarray(source_indices),
        Path(f"{platform}.mat"), f"hash-{platform}",
    )


def drogue(status="not_lost", cutoff="NaT"):
    cutoff = np.datetime64(cutoff, "ns")
    return ResolvedDrogueDecision(
        status, np.datetime64("NaT", "ns"), 24, cutoff, "test",
    )


def one_sided_spike_track(kind="outgoing", *, n=45, center=22):
    """A smooth track with only one of the displaced point's edges over 3 m/s."""
    lon = np.arange(n, dtype=float) * .001
    if kind == "outgoing":
        # Previous -> candidate remains plausible; candidate -> next is anomalous.
        lon[center] = lon[center - 1] - .0079
    elif kind == "incoming":
        # Previous -> candidate is anomalous; candidate -> next remains plausible.
        lon[center] = lon[center + 1] + .0079
    else:
        raise ValueError(kind)
    return trajectory(lon)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_bidirectional_blocks_within_elapsed_time_limit_are_rejected(count):
    values = [0, .001, .002, *[.2 + i * .001 for i in range(count)], .003, .004, .005]
    result = run_native_position_qc(trajectory(values), drogue())
    rejected = result[result.point_auto_status == "high_confidence_reject"]
    assert rejected.source_obs_index.tolist() == list(range(3, 3 + count))
    assert rejected.final_position_status.eq("rejected").all()
    assert rejected.candidate_bridge_speed_m_s.le(3).all()


def test_block_beyond_elapsed_time_limit_and_uncured_single_edge_remain_unresolved():
    result = run_native_position_qc(
        trajectory([0, .001, .002, .20, .21, .22, .23, .24, .25, .003, .004, .005]),
        drogue(),
    )
    assert not result.point_auto_status.eq("high_confidence_reject").any()
    assert result.final_position_status.eq("unresolved").any()
    assert result.local_event_type.eq("persistent_excursion").any()
    single = run_native_position_qc(trajectory([0, .001, .002, .03, .031, .032]), drogue())
    assert single.local_event_type.eq("single_edge_ambiguous").any()
    assert not single.final_position_status.eq("rejected").any()


def test_aggressive_solver_expands_only_to_configured_removal_limit():
    values = [0, .001, .002, .20, .201, .202, .203, .204, .205, .003, .004, .005]
    limited = run_native_position_qc(
        trajectory(values), drogue(),
        position_config=NativePositionConfig(
            speed_threshold_m_s=2, max_local_gap_seconds=3000,
            max_automatic_removal_points=5,
        ),
        resolution_policy="aggressive",
    )
    assert not limited.final_position_status.eq("rejected").any()
    assert limited.point_auto_reason.eq("automatic_keep_removal_limit_reached").any()
    assert not limited.final_position_status.isin(["unresolved", "uncertain"]).any()

    cured = run_native_position_qc(
        trajectory(values), drogue(),
        position_config=NativePositionConfig(
            speed_threshold_m_s=2, max_local_gap_seconds=3000,
            max_automatic_removal_points=6,
        ),
        resolution_policy="aggressive",
    )
    assert cured.loc[cured.final_position_status.eq("rejected"), "source_obs_index"].tolist() == list(
        range(3, 9)
    )


def test_aggressive_equal_size_cures_use_vector_acceleration_and_are_deterministic():
    result = run_native_position_qc(
        trajectory([-.004, 0, .004, .010, .014, .018]), drogue(),
        position_config=NativePositionConfig(speed_threshold_m_s=2),
        resolution_policy="aggressive",
    )
    rejected = result.loc[result.final_position_status.eq("rejected")]
    assert len(rejected) == 1
    evidence = json.loads(rejected.iloc[0].forward_result_json)
    cures = [item for item in evidence["tested_hypotheses"] if item["cures"]]
    assert len(cures) == 2
    selected_acceleration = evidence["maximum_boundary_acceleration_m_s2"]
    assert selected_acceleration == pytest.approx(min(
        item["maximum_boundary_acceleration_m_s2"] for item in cures
    ))
    rerun = run_native_position_qc(
        trajectory([-.004, 0, .004, .010, .014, .018]), drogue(),
        position_config=NativePositionConfig(speed_threshold_m_s=2),
        resolution_policy="aggressive",
    )
    assert rerun.loc[rerun.final_position_status.eq("rejected"), "source_obs_index"].tolist() == (
        rejected.source_obs_index.tolist()
    )


def test_aggressive_duplicate_time_keeps_smoothest_representative():
    result = run_native_position_qc(
        trajectory([0, .001, .002, .2, .003, .004], seconds=[0, 300, 600, 600, 900, 1200]),
        drogue(), resolution_policy="aggressive",
    )
    kept = result.loc[result.source_obs_index == 2].iloc[0]
    rejected = result.loc[result.source_obs_index == 3].iloc[0]
    assert kept.point_auto_status == "duplicate_timestamp_representative"
    assert kept.final_position_status == "valid"
    assert rejected.final_position_status == "rejected"
    assert rejected.position_decision_source == "automatic_duplicate_time"
    assert not result.final_position_status.isin(["unresolved", "uncertain"]).any()


def test_conservative_policy_preserves_ambiguous_review_event():
    result = run_native_position_qc(
        trajectory([0, .001, .002, .03, .031, .032]), drogue(),
        position_config=NativePositionConfig(speed_threshold_m_s=2),
        resolution_policy="conservative",
    )
    assert result.local_event_type.eq("single_edge_ambiguous").any()
    assert result.final_position_status.eq("unresolved").any()


def test_arcterx_18554_excerpt_has_one_unique_speed_cure():
    source = np.arange(18564, 18543, -1)
    seconds = [
        0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700,
        3120, 3480, 3600, 3900, 4200, 4500, 4800, 5100, 5400, 5700, 6000,
    ]
    lon = [
        129.127481, 129.126118, 129.124968, 129.123938, 129.122861,
        129.121877, 129.120822, 129.119878, 129.119637, 129.120636,
        129.131478, 129.111731, 129.111393, 129.109829, 129.108249,
        129.106626, 129.105062, 129.103488, 129.101806, 129.100091, 129.098367,
    ]
    lat = [
        23.812313, 23.811619, 23.810991, 23.810335, 23.809727,
        23.809070, 23.808471, 23.807835, 23.807092, 23.806210,
        23.803745, 23.804743, 23.804590, 23.804041, 23.803475,
        23.802893, 23.802427, 23.801911, 23.801238, 23.800643, 23.800008,
    ]
    result = run_native_position_qc(
        trajectory_xy(lon[::-1], lat[::-1], seconds[::-1], source[::-1]), drogue(),
        position_config=NativePositionConfig(minimum_local_dt_seconds=120),
    )
    rejected = result.loc[result.point_auto_status == "high_confidence_reject"]
    assert rejected.source_obs_index.tolist() == [18554]
    row = rejected.iloc[0]
    assert row.point_auto_reason == "unique_endpoint_speed_cure"
    assert row.candidate_bridge_speed_m_s == pytest.approx(1.18018, rel=1e-4)
    chosen = json.loads(row.forward_result_json)
    competitor = json.loads(row.backward_result_json)
    assert chosen["bridge_dt_seconds"] == 780
    assert chosen["bridge_atypical_warning"] is True
    assert competitor["bridge_speed"] == pytest.approx(4.26152, rel=1e-4)
    assert competitor["cures"] is False


def test_arcterx_29530_to_29526_excerpt_is_first_time_bounded_cure():
    source = np.arange(29538, 29517, -1)
    seconds = np.arange(21) * 300
    lon = [
        130.722420, 130.722652, 130.722777, 130.722909, 130.722932,
        130.722979, 130.722670, 130.722146, 130.796745, 130.797996,
        130.798668, 130.798759, 130.798323, 130.718927, 130.718542,
        130.718100, 130.717640, 130.717296, 130.716939, 130.716526, 130.715996,
    ]
    lat = [
        20.662810, 20.663175, 20.663660, 20.664036, 20.664498,
        20.665119, 20.665899, 20.666787, 20.707522, 20.705116,
        20.702173, 20.698838, 20.695353, 20.671204, 20.671782,
        20.672372, 20.672980, 20.673627, 20.674312, 20.674947, 20.675583,
    ]
    result = run_native_position_qc(
        trajectory_xy(lon[::-1], lat[::-1], seconds[::-1], source[::-1]), drogue(),
        position_config=NativePositionConfig(minimum_local_dt_seconds=120),
    )
    rejected = result.loc[result.point_auto_status == "high_confidence_reject"]
    assert rejected.source_obs_index.tolist() == [29526, 29527, 29528, 29529, 29530]
    assert rejected.point_auto_reason.eq(
        "bidirectionally_confirmed_time_bounded_excursion_cure"
    ).all()
    evidence = json.loads(rejected.iloc[0].forward_result_json)
    assert evidence["bridge_dt_seconds"] == 1800
    assert evidence["bridge_speed"] == pytest.approx(.330238, rel=1e-4)
    trials = evidence["tested_reconnections"]
    assert [trial["cures"] for trial in trials] == [False, False, False, False, True]
    assert trials[-2]["bridge_speed"] == pytest.approx(5.69171, rel=1e-4)


@pytest.mark.parametrize("kind", ["incoming", "outgoing"])
def test_unique_one_sided_spike_rejects_the_interpolation_outlier(kind):
    result = run_native_position_qc(one_sided_spike_track(kind), drogue())
    rejected = result[result.point_auto_status == "high_confidence_reject"]

    assert rejected.source_obs_index.tolist() == [22]
    row = rejected.iloc[0]
    assert row.final_position_status == "rejected"
    assert row.point_auto_reason == "unique_endpoint_speed_cure"
    assert row.candidate_bridge_speed_m_s <= 3

    chosen = json.loads(row.forward_result_json)
    competitor = json.loads(row.backward_result_json)
    assert chosen["candidate"] == 22 and chosen["selected"] is True
    assert chosen["cures"] is True
    assert chosen["baseline"]["sample_count"] >= 8
    assert chosen["baseline"]["speed_scale_m_s"] >= .05
    assert chosen["smoothness_score"] < competitor["smoothness_score"]
    assert np.isfinite(chosen["residual_m"])


def test_one_sided_spike_needs_local_samples_and_respects_human_protection():
    sparse = run_native_position_qc(
        one_sided_spike_track(), drogue(),
        position_config=NativePositionConfig(one_sided_spike_min_samples=100),
    )
    assert not sparse.point_auto_reason.eq(
        "unique_endpoint_speed_cure"
    ).any()
    assert sparse.final_position_status.eq("unresolved").any()

    reviews = pd.DataFrame([{
        "platform_code": "p", "target_type": "observation",
        "target_id": "observation:22", "source_obs_index": 22,
        "decision": "keep", "decision_source": "manual", "event_id": "prior",
    }])
    protected = run_native_position_qc(one_sided_spike_track(), drogue(), reviews=reviews)
    point = protected.loc[protected.source_obs_index == 22].iloc[0]
    assert point.final_position_status == "valid"
    assert point.position_decision_source == "human"
    assert point.point_auto_status != "high_confidence_reject"


def test_one_sided_spike_does_not_cross_a_timing_boundary():
    track = one_sided_spike_track()
    seconds = np.arange(len(track.time)) * 300
    seconds[24:] += 2100
    bounded = trajectory(track.lon, seconds=seconds)
    result = run_native_position_qc(bounded, drogue())
    assert not result.point_auto_reason.eq(
        "unique_endpoint_speed_cure"
    ).any()


def test_endpoint_speed_cure_requires_a_clear_smoothness_winner():
    lon = np.arange(45, dtype=float) * .001
    lon[24:] += .0012
    lon[22], lon[23] = .0281, .0181
    tied = run_native_position_qc(trajectory(lon), drogue())
    assert not tied.point_auto_reason.eq("unique_endpoint_speed_cure").any()
    assert tied.local_event_type.eq("single_edge_ambiguous").any()

    winner = run_native_position_qc(one_sided_spike_track(), drogue())
    row = winner.loc[winner.point_auto_reason == "unique_endpoint_speed_cure"].iloc[0]
    chosen = json.loads(row.forward_result_json)
    competitor = json.loads(row.backward_result_json)
    assert chosen["cures"] and competitor["cures"]
    assert competitor["smoothness_score"] - chosen["smoothness_score"] >= 2


def test_one_sided_recovery_runs_to_a_fixed_point_after_other_rejections():
    lon = np.arange(60, dtype=float) * .001
    lon[15] = .2
    lon[40] = lon[39] - .0079
    result = run_native_position_qc(trajectory(lon), drogue())
    rejected = result[result.point_auto_status == "high_confidence_reject"]

    assert rejected.source_obs_index.tolist() == [15, 40]
    one_sided = rejected.loc[rejected.source_obs_index == 40].iloc[0]
    assert one_sided.point_auto_reason == "unique_endpoint_speed_cure"
    assert one_sided.auto_iteration == 2


def test_short_edges_are_automatic_but_long_edges_remain_diagnostics():
    result = run_native_position_qc(
        trajectory([0, .02, .021, .022, .023], seconds=[0, 60, 360, 660, 960]), drogue(),
    )
    short = result.loc[result.point_auto_status == "short_interval_reject"]
    assert short.source_obs_index.tolist() == [1]
    assert short.short_dt_flag.all()
    assert short.final_position_status.eq("rejected").all()
    assert short.position_decision_source.eq("automatic_short_interval").all()
    long = run_native_position_qc(
        trajectory([0, .001, .2, .201, .202], seconds=[0, 300, 2401, 2701, 3001]), drogue(),
    )
    assert long.large_gap_flag.any()
    assert not long.final_position_status.eq("rejected").any()


def test_short_interval_thinning_is_iterative_and_threshold_equality_survives():
    result = run_native_position_qc(
        trajectory([0, .0001, .0002, .0003], seconds=[0, 60, 120, 180]),
        drogue(), position_config=NativePositionConfig(minimum_local_dt_seconds=120),
    )
    rejected = result.loc[result.point_auto_status == "short_interval_reject"]
    assert rejected.source_obs_index.tolist() == [1, 3]
    assert result.loc[result.source_obs_index.isin([0, 2]), "final_position_status"].eq("valid").all()
    row = result.loc[result.source_obs_index == 2].iloc[0]
    assert row.surviving_predecessor_source_obs_index == 0
    assert row.surviving_dt_seconds == 120
    evidence = [json.loads(value) for value in rejected.forward_result_json]
    assert [item["predecessor_source_obs_index"] for item in evidence] == [0, 2]
    assert [item["dt_seconds"] for item in evidence] == [60, 60]
    assert all(item["minimum_local_dt_seconds"] == 120 for item in evidence)


def test_short_interval_thinning_maps_back_from_reverse_source_order():
    result = run_native_position_qc(
        trajectory(
            [.0003, .0002, .0001, 0], seconds=[180, 120, 60, 0],
            source_indices=[40, 30, 20, 10],
        ),
        drogue(), position_config=NativePositionConfig(minimum_local_dt_seconds=120),
    )
    rejected = result.loc[result.point_auto_status == "short_interval_reject"]
    assert rejected.source_obs_index.tolist() == [40, 20]
    assert result.loc[result.source_obs_index.isin([10, 30]), "final_position_status"].eq("valid").all()


@pytest.mark.parametrize("decision", ["keep", "reject", "uncertain"])
def test_short_interval_rejection_overrides_human_point_decisions(decision):
    track = trajectory([0, .0001, .0002], seconds=[0, 60, 120])
    reviews = pd.DataFrame([{
        "platform_code": "p", "target_type": "observation",
        "target_id": "observation:1", "source_obs_index": 1,
        "decision": decision, "decision_source": "manual", "event_id": "old-short-review",
    }])
    result = run_native_position_qc(
        track, drogue(), reviews=reviews,
        position_config=NativePositionConfig(minimum_local_dt_seconds=120),
    )
    point = result.loc[result.source_obs_index == 1].iloc[0]
    assert point.final_position_status == "rejected"
    assert point.position_decision_source == "automatic_short_interval"
    assert point.human_position_decision == decision
    assert result.platform_qc_complete.all()


def test_short_interval_scope_boundaries_and_repeat_precedence():
    invalid = run_native_position_qc(
        trajectory([0, np.nan, .0002], seconds=[0, 60, 120]), drogue(),
        position_config=NativePositionConfig(minimum_local_dt_seconds=150),
    )
    assert not invalid.point_auto_status.eq("short_interval_reject").any()

    cutoff = run_native_position_qc(
        trajectory([0, .0001, .0002], seconds=[0, 60, 120]),
        drogue("lost", "2025-01-01T00:01:30"),
        position_config=NativePositionConfig(minimum_local_dt_seconds=120),
    )
    assert cutoff.loc[cutoff.source_obs_index == 1, "point_auto_status"].iloc[0] == "short_interval_reject"
    assert cutoff.loc[cutoff.source_obs_index == 2, "final_position_status"].iloc[0] == "outside_drogue_window"

    repeated = run_native_position_qc(
        trajectory([0, .0001, .0001, .0002], seconds=[0, 60, 120, 180]), drogue(),
        position_config=NativePositionConfig(minimum_local_dt_seconds=150),
    )
    group = repeated.loc[repeated.exact_repeat_flag]
    assert group.point_auto_status.eq("repeated_position").all()
    assert group.position_decision_source.eq("automatic_repeat").all()


def test_pending_segments_are_not_short_interval_thinned():
    seconds = [0, 60, 120, 86400, 86460, 86520]
    result = run_native_position_qc(
        trajectory(np.arange(6) * .0001, seconds=seconds), drogue(),
        position_config=NativePositionConfig(minimum_local_dt_seconds=120),
    )
    assert result.segment_status.eq("pending").all()
    assert not result.point_auto_status.eq("short_interval_reject").any()


def test_short_interval_thinning_keeps_retained_segments_independent():
    seconds = [0, 60, 120, 86400, 86460, 86520]
    track = trajectory(np.arange(6) * .0001, seconds=seconds)
    pending = run_native_position_qc(
        track, drogue(),
        position_config=NativePositionConfig(minimum_local_dt_seconds=120),
    )
    fingerprints = pending.groupby("segment_id").segment_fingerprint.first().tolist()
    reviews = pd.DataFrame([{
        "platform_code": "p", "target_type": "segment", "target_id": fingerprint,
        "source_obs_index": "", "decision": "retain",
    } for fingerprint in fingerprints])

    result = run_native_position_qc(
        track, drogue(), reviews=reviews,
        position_config=NativePositionConfig(minimum_local_dt_seconds=120),
    )
    rejected = result.loc[
        result.point_auto_status == "short_interval_reject", "source_obs_index",
    ].astype(int).tolist()
    assert rejected == [1, 4]
    assert result.loc[result.source_obs_index == 2, "surviving_predecessor_source_obs_index"].iloc[0] == 0
    assert result.loc[result.source_obs_index == 5, "surviving_predecessor_source_obs_index"].iloc[0] == 3


def test_exact_repeats_reject_every_occurrence_and_recalculate_adjacency():
    result = run_native_position_qc(trajectory([0, .001, .2, .002, .2, .003, .004]), drogue())
    repeated = result[result.exact_repeat_flag]
    assert repeated.source_obs_index.tolist() == [2, 4]
    assert repeated.repeated_coordinate_count.eq(2).all()
    assert repeated.final_position_status.eq("rejected").all()
    assert repeated.repeated_coordinate_group_id.nunique() == 1
    row = result[result.source_obs_index == 3].iloc[0]
    assert row.surviving_predecessor_source_obs_index == 1


@pytest.mark.parametrize("decision", ["keep", "reject", "uncertain"])
def test_exact_repeats_override_every_human_point_decision(decision):
    track = trajectory([0, .001, .2, .002, .2, .003, .004])
    baseline = run_native_position_qc(track, drogue())
    members = baseline.loc[baseline.exact_repeat_flag, "source_obs_index"].astype(int).tolist()
    reviews = pd.DataFrame([{
        "platform_code": "p", "target_type": "observation",
        "target_id": f"observation:{source}", "source_obs_index": source,
        "decision": decision, "decision_source": "manual", "event_id": "old-repeat-review",
    } for source in members])

    result = run_native_position_qc(track, drogue(), reviews=reviews)
    repeated = result.loc[result.source_obs_index.isin(members)]
    assert repeated.final_position_status.eq("rejected").all()
    assert repeated.position_decision_source.eq("automatic_repeat").all()
    assert repeated.human_position_decision.eq(decision).all()
    assert result.platform_qc_complete.all()
    row = result.loc[result.source_obs_index == 3].iloc[0]
    assert row.surviving_predecessor_source_obs_index == 1


def test_repeat_scope_excludes_post_cutoff_and_pending_segments():
    track = trajectory([0, .001, .2, .2], seconds=[0, 300, 600, 900])
    result = run_native_position_qc(track, drogue("lost", "2025-01-01T00:12:00"))
    assert not result.exact_repeat_flag.any()
    assert result.final_position_status.tolist()[-1] == "outside_drogue_window"


def test_detached_fragment_and_source_rows_are_preserved():
    seconds = [0, 300, 600] + list(3 * 86400 + np.arange(60) * 3600)
    lon = np.arange(len(seconds)) * .0001
    result = run_native_position_qc(trajectory(lon, seconds=seconds), drogue())
    early = result.iloc[:3]
    assert len(result) == len(seconds)
    assert early.segment_status.eq("excluded").all()
    assert early.temporal_event_type.eq("pre_main_fragment").all()
    assert early.final_position_status.eq("outside_operational_segment").all()
    assert result.iloc[3:].segment_status.eq("retained").all()


def test_post_fragment_and_subthreshold_gap_classification():
    main = list(np.arange(60) * 3600)
    late = list(5 * 86400 + np.arange(3) * 300)
    result = run_native_position_qc(
        trajectory(np.arange(63) * .0001, seconds=main + late), drogue(),
    )
    assert result.iloc[-3:].temporal_event_type.eq("post_main_fragment").all()
    assert result.iloc[-3:].segment_status.eq("excluded").all()
    connected = run_native_position_qc(
        trajectory(np.arange(8) * .0001, seconds=[0, 300, 600, 7800, 8100, 8400, 8700, 9000]),
        drogue(),
    )
    assert connected.segment_id.nunique() == 1


def test_multiple_substantial_segments_pending_while_tiny_is_excluded():
    a = list(np.arange(60) * 300)
    tiny = list(2 * 86400 + np.arange(2) * 300)
    b = list(4 * 86400 + np.arange(55) * 300)
    seconds = a + tiny + b
    lon = np.arange(len(seconds)) * .00001
    result = run_native_position_qc(
        trajectory(lon, seconds=seconds), drogue(),
        temporal_config=TemporalSegmentConfig(minimum_main_duration_hours=4),
    )
    statuses = result.groupby("segment_id").segment_status.first().tolist()
    assert statuses == ["pending", "excluded", "pending"]
    assert result[result.segment_status == "pending"].final_position_status.eq("unresolved").all()
    assert not result.point_auto_status.eq("high_confidence_reject").any()


def test_pending_segments_keep_diagnostics_but_defer_point_rejection():
    first = [0, .001, .002, .2, .003, .004]
    second = [1, 1.001, 1.002, 1.2, 1.003, 1.004]
    seconds = list(np.arange(6) * 300) + list(86400 + np.arange(6) * 300)
    result = run_native_position_qc(trajectory(first + second, seconds=seconds), drogue())
    assert result.segment_status.eq("pending").all()
    assert result.high_speed_flag.any()
    assert result.local_review_required.any()
    assert not result.point_auto_status.eq("high_confidence_reject").any()
    assert result.final_position_status.eq("unresolved").all()


def test_segment_review_can_retain_multiple_independent_segments_or_exclude_all():
    seconds = list(np.arange(6) * 300) + list(86400 + np.arange(6) * 300)
    track = trajectory(np.arange(12) * .0001, seconds=seconds)
    pending = run_native_position_qc(track, drogue())
    fingerprints = pending.groupby("segment_id").segment_fingerprint.first().tolist()

    def decisions(value):
        return pd.DataFrame([{
            "platform_code": "p", "target_type": "segment", "target_id": fingerprint,
            "source_obs_index": "", "decision": value,
        } for fingerprint in fingerprints])

    retained = run_native_position_qc(track, drogue(), reviews=decisions("retain"))
    assert retained.segment_status.eq("retained").all()
    second_start = retained[retained.segment_id == "s0002"].sort_values("time").iloc[0]
    assert pd.isna(second_start.surviving_predecessor_source_obs_index)
    excluded = run_native_position_qc(track, drogue(), reviews=decisions("exclude"))
    assert excluded.final_position_status.eq("outside_operational_segment").all()
    assert excluded.platform_qc_complete.all()
    assert not excluded.reconstruction_available.any()


def test_continuous_early_data_are_not_called_predeployment_without_metadata():
    result = run_native_position_qc(trajectory(np.arange(60) * .0001), drogue())
    assert result.segment_id.nunique() == 1
    assert not result.temporal_event_type.eq("pre_main_fragment").any()


def test_deployment_exact_and_window_boundaries():
    track = trajectory(np.arange(8) * .0001)
    exact = DeploymentBoundary(deployment_time=np.datetime64("2025-01-01T00:10:00", "ns"))
    result = run_native_position_qc(track, drogue(), deployment=exact)
    assert result.final_position_status.iloc[:2].eq("outside_operational_segment").all()
    assert result.final_position_status.iloc[2:].eq("valid").all()
    window = DeploymentBoundary(
        window_start=np.datetime64("2025-01-01T00:05:00", "ns"),
        window_end=np.datetime64("2025-01-01T00:15:00", "ns"),
    )
    result = run_native_position_qc(track, drogue(), deployment=window)
    assert result.final_position_status.iloc[0] == "outside_operational_segment"
    assert result.final_position_status.iloc[1:3].eq("unresolved").all()
    assert result.final_position_status.iloc[3:].eq("valid").all()


def test_strict_drogue_cutoff_and_uncertain_blocking():
    track = trajectory(np.arange(5) * .0001)
    result = run_native_position_qc(track, drogue("lost", "2025-01-01T00:10:00"))
    assert result.final_position_status.tolist() == [
        "valid", "valid", "outside_drogue_window", "outside_drogue_window", "outside_drogue_window",
    ]
    blocked = run_native_position_qc(track, drogue("uncertain"))
    assert blocked.final_position_status.eq("unresolved").all()
    assert blocked.point_auto_status.eq("blocked_by_drogue").all()


def test_duplicate_timestamp_is_review_required():
    result = run_native_position_qc(
        trajectory([0, .001, .002, .003, .004], seconds=[0, 300, 300, 600, 900]), drogue(),
    )
    duplicate = result[result.time.duplicated(keep=False)]
    assert duplicate.local_event_type.eq("boundary_or_insufficient_context").all()
    assert duplicate.final_position_status.eq("unresolved").all()
    assert duplicate.nonpositive_dt_flag.all()
    assert not result.point_auto_status.eq("short_interval_reject").any()


def test_alternating_spikes_and_directional_or_boundary_ambiguity():
    alternating = run_native_position_qc(
        trajectory([0, .001, .002, .2, .003, .25, .004, .005, .006]), drogue(),
    )
    rejected = alternating[alternating.point_auto_status == "high_confidence_reject"]
    assert rejected.source_obs_index.tolist() == [3, 5]
    assert rejected.local_event_id.nunique() == 2

    disagreement = run_native_position_qc(
        trajectory([0, .001, .002, .02, .021, .022, .004, .005, .006]), drogue(),
    )
    assert not disagreement.point_auto_status.eq("high_confidence_reject").any()
    assert disagreement.local_review_required.any()
    boundary = run_native_position_qc(
        trajectory([0, .001, .002, .2, .003, .004]), drogue(),
    )
    assert not boundary.point_auto_status.eq("high_confidence_reject").any()
    assert boundary.final_position_status.eq("unresolved").any()


def test_human_point_decisions_override_geometry_only_inside_eligible_scope():
    track = trajectory([0, .001, .002, .2, .003, .004, .005])
    reviews = pd.DataFrame([{
        "platform_code": "p", "target_type": "observation", "target_id": "observation:3",
        "source_obs_index": 3, "decision": "keep", "decision_source": "manual",
        "event_id": "prior",
    }])
    result = run_native_position_qc(track, drogue(), reviews=reviews)
    point = result[result.source_obs_index == 3].iloc[0]
    assert point.final_position_status == "valid"
    assert point.position_decision_source == "human"
    assert point.point_auto_status != "high_confidence_reject"
    outside = run_native_position_qc(track, drogue("lost", "2025-01-01T00:10:00"), reviews=reviews)
    assert outside[outside.source_obs_index == 3].final_position_status.iloc[0] == "outside_drogue_window"


def test_resolver_validates_identity_and_blocks_unresolved():
    track = trajectory([0, .001, .002, .03, .031, .032])
    result = run_native_position_qc(track, drogue())
    with pytest.raises(ValueError, match="unresolved"):
        resolved_position_trajectory(track, result)
    _, resolution = resolved_position_trajectory(track, result, allow_unresolved=True)
    assert not resolution.platform_qc_complete
    changed = result.copy()
    changed.loc[0, "source_sha256"] = "other"
    with pytest.raises(ValueError, match="identity"):
        resolved_position_trajectory(track, changed, allow_unresolved=True)


def test_resolver_returns_masks_in_native_array_order():
    track = trajectory([0, .001, .002], source_indices=[20, 5, 11])
    result = run_native_position_qc(track, drogue())
    shuffled = result.sample(frac=1, random_state=2)
    _, resolution = resolved_position_trajectory(track, shuffled)
    expected = result.set_index("source_obs_index").loc[[20, 5, 11], "final_position_status"]
    assert resolution.final_position_status.tolist() == expected.tolist()
