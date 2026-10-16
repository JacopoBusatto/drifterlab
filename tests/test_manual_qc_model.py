from pathlib import Path

import numpy as np
import pytest

from drifterlab.io.position import NativeTrajectory
from drifterlab.manual_qc.model import (
    ManualTrajectoryModel,
    adjacent_kinematics,
    speed_warning_masks,
)
from drifterlab.manual_qc.state import ManualQCState


def trajectory(
    longitude=(0.0, 0.001, 0.002, 0.003, 0.004),
    *,
    seconds=(0, 300, 600, 900, 1200),
    platform="p",
    sst=None,
):
    longitude = np.asarray(longitude, dtype=float)
    seconds = np.asarray(seconds, dtype=np.int64)
    if sst is None:
        sst = 20.0 + np.arange(len(longitude), dtype=float) * 0.1
    return NativeTrajectory(
        platform,
        np.datetime64("2025-01-01", "ns") + seconds.astype("timedelta64[s]"),
        longitude,
        np.zeros(len(longitude)),
        np.arange(len(longitude), dtype=np.int64),
        Path(f"{platform}.mat"),
        f"sha-{platform}",
        np.asarray(sst, dtype=float),
        1.0 + np.arange(len(longitude), dtype=float),
        10.0 - np.arange(len(longitude), dtype=float),
    )


def test_adjacent_kinematics_uses_actual_irregular_elapsed_time():
    item = trajectory(
        longitude=(0.0, 0.001, 0.003),
        seconds=(0, 300, 900),
    )
    values = adjacent_kinematics(
        item.time, item.lon, item.lat, item.source_obs_index,
    )
    assert np.isnan(values.speed_m_s[0])
    assert values.edge_dt_seconds[1:].tolist() == [300.0, 600.0]
    # Twice the distance over twice the elapsed time is approximately equal.
    assert values.speed_m_s[1] == pytest.approx(values.speed_m_s[2], rel=1e-6)
    assert np.isfinite(values.acceleration_m_s2[1])
    assert values.acceleration_m_s2[1] == pytest.approx(0.0, abs=1e-10)


def test_speed_warnings_are_advisory_threshold_masks():
    high, zero = speed_warning_masks(
        np.asarray([np.nan, 0.0, 0.01, 2.0, 2.01]),
        high_speed_m_s=2.0,
        zero_speed_tolerance_m_s=0.0,
    )
    assert high.tolist() == [False, False, False, False, True]
    assert zero.tolist() == [False, True, False, False, False]


def test_rejection_bridges_raw_kinematics_and_restore_reverses_it():
    model = ManualTrajectoryModel(trajectory())
    original = model.raw_kinematics()
    model.set_rejected([2], True)
    bridged = model.raw_kinematics()
    assert bridged.source_indices.tolist() == [0, 1, 3, 4]
    assert bridged.edge_dt_seconds[2] == 600
    assert bridged.speed_m_s[2] == pytest.approx(original.speed_m_s[2], rel=1e-6)
    model.set_rejected([2], False)
    assert model.raw_kinematics().source_indices.tolist() == [0, 1, 2, 3, 4]


def test_cursor_crossing_changes_chunk_and_drogue_stop_limits_scope():
    item = trajectory(
        longitude=np.arange(8) * 0.001,
        seconds=np.arange(8) * 900,
    )
    stop = np.datetime64("2025-01-01T01:15:00", "ns")
    model = ManualTrajectoryModel(item, chunk_seconds=3600, stop_time=stop)
    assert len(model.chunks) == 2
    assert model.chunks[0].source_indices == (0, 1, 2, 3)
    assert model.chunks[1].source_indices == (4, 5)
    model.set_selected_source(3)
    assert model.current_chunk_index == 0
    model.move_cursor(1)
    assert model.selected_source == 4
    assert model.current_chunk_index == 1
    assert 6 not in model.review_order and 7 not in model.review_order


def test_accepted_span_starts_at_deployment_and_extends_after_rejections():
    item = trajectory(
        longitude=np.arange(21, dtype=float) * 0.001,
        seconds=np.arange(21, dtype=np.int64) * 86_400,
    )
    model = ManualTrajectoryModel(
        item, chunk_seconds=86_400, maximum_accepted_span_days=15,
    )
    assert model.deployment_start_time == item.time[0]
    assert model.review_end_time == item.time[15]
    assert model.review_end_reason == "maximum_span"
    assert model.review_order == list(range(16))
    assert model.accepted_review_span_days == pytest.approx(15.0)

    # The rejected laboratory point remains reviewable, but deployment and the
    # tail both move one day later so the accepted record still spans 15 days.
    model.set_rejected([0], True)
    assert model.deployment_start_time == item.time[1]
    assert model.review_end_time == item.time[16]
    assert model.review_order == list(range(17))
    assert model.accepted_review_span_days == pytest.approx(15.0)

    # Rejecting the point that supplied the duration extends to the next one.
    model.set_rejected([16], True)
    assert model.review_end_time == item.time[17]
    assert model.review_order == list(range(18))

    # Restoring the first point contracts the active tail again.
    model.set_rejected([0], False)
    assert model.deployment_start_time == item.time[0]
    assert model.review_end_time == item.time[15]
    assert 16 not in model.review_position and 17 not in model.review_position


def test_short_record_uses_hard_endpoint_before_requested_accepted_span():
    item = trajectory(
        longitude=np.arange(11, dtype=float) * 0.001,
        seconds=np.arange(11, dtype=np.int64) * 86_400,
    )
    model = ManualTrajectoryModel(
        item,
        stop_time=item.time[10],
        hard_stop_reason="physical_loss",
        maximum_accepted_span_days=15,
    )
    assert model.review_end_time == item.time[10]
    assert model.review_end_reason == "physical_loss"
    assert model.accepted_review_span_days == pytest.approx(10.0)


def test_resample_is_anchored_at_first_accepted_point():
    model = ManualTrajectoryModel(trajectory())
    first = model.resample(300)
    assert first.time[0] == model.time[0]
    model.set_rejected([0], True)
    second = model.resample(300)
    assert second.time[0] == model.time[1]
    assert len(second.time) == len(first.time) - 1


def test_sst_rejection_is_independent_and_limits_interpolation():
    model = ManualTrajectoryModel(trajectory(sst=(20.0, 21.0, 50.0, 23.0, 24.0)))
    model.set_sst_rejected([2], True)
    assert 2 in model.rejected_sst_sources
    assert 2 not in model.rejected_sources
    result = model.resample(300, sst_maximum_gap_minutes=15)
    assert result.sst[2] == pytest.approx(22.0)
    model.set_rejected([1], True)
    assert 1 in model.rejected_sources
    assert 1 not in model.rejected_sst_sources


def test_resample_requires_manual_resolution_of_duplicate_time():
    model = ManualTrajectoryModel(trajectory(seconds=(0, 300, 300, 900, 1200)))
    with pytest.raises(ValueError, match="strictly increasing"):
        model.resample(300)
    model.set_rejected([2], True)
    model.set_sst_rejected([2], True)
    assert len(model.resample(300).time) == 5


def test_invalid_timestamp_has_explicit_review_chunk_and_blocks_completion():
    item = trajectory()
    item.time[2] = np.datetime64("NaT", "ns")
    model = ManualTrajectoryModel(item)
    assert model.chunks[-1].kind == "invalid_time"
    reviewed = {chunk.key for chunk in model.chunks}
    assert any("invalid timestamps" in value for value in model.completion_issues(reviewed))
    model.set_rejected([2], True)
    assert model.completion_issues(reviewed) == []


def test_sqlite_state_autosaves_decisions_progress_cursor_and_resample(tmp_path):
    path = tmp_path / "manual.sqlite"
    with ManualQCState(path) as state:
        assert not state.register_source("p", "sha-p", "not_lost")
        changed = state.set_rejected(
            "p", "sha-p", [2], rejected=True, dirty_chunk_keys=["c0"],
        )
        assert changed == {2}
        assert state.set_sst_rejected(
            "p", "sha-p", [3], rejected=True, dirty_chunk_keys=["c0"],
        ) == {3}
        state.mark_chunk_reviewed(
            "p", "sha-p", "c0", start_time_ns=0, end_time_ns=10,
        )
        state.mark_resampled("p", 300)
        state.save_cursor("p", "sha-p", "c0", 2)
    with ManualQCState(path) as state:
        assert state.rejected_sources("p", "sha-p") == {2}
        assert state.rejected_sst_sources("p", "sha-p") == {3}
        assert state.chunk_states("p", "sha-p") == {"c0": "reviewed"}
        assert state.load_cursor()["selected_source_obs_index"] == 2
        assert state.resample_state("p")["stale"] is False
        state.set_rejected("p", "sha-p", [4], rejected=True)
        assert state.resample_state("p")["stale"] is True
        assert state.undo_last("p", "sha-p") == {4}
        assert state.rejected_sources("p", "sha-p") == {2}


def test_source_hash_change_is_rejected_and_drogue_change_dirties_progress(tmp_path):
    with ManualQCState(tmp_path / "manual.sqlite") as state:
        state.register_source("p", "sha-p", "lost:1")
        state.mark_chunk_reviewed(
            "p", "sha-p", "c0", start_time_ns=0, end_time_ns=10,
        )
        assert state.register_source("p", "sha-p", "lost:2")
        assert state.chunk_states("p", "sha-p") == {"c0": "dirty"}
        with pytest.raises(ValueError, match="Raw source changed"):
            state.register_source("p", "other", "lost:2")
