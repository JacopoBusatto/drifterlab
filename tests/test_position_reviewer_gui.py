"""Agg-backed tests for the lazy, staged native-position reviewer."""

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from drifterlab.io.position import NativeTrajectory
from drifterlab.qc.drogue import ResolvedDrogueDecision
from drifterlab.qc.native_position import NativePositionConfig, TemporalSegmentConfig, run_native_position_qc
from drifterlab.review.position import PositionReviewer, PositionReviews, position_event_catalog


def track(lon, *, seconds=None, platform="p"):
    lon = np.asarray(lon, dtype=float)
    seconds = np.arange(len(lon)) * 300 if seconds is None else np.asarray(seconds)
    return NativeTrajectory(
        platform, np.datetime64("2025-01-01", "ns") + seconds.astype("timedelta64[s]"),
        lon, np.zeros(len(lon)), np.arange(len(lon)), Path(f"{platform}.mat"), f"source-{platform}",
    )


def drogue():
    return ResolvedDrogueDecision(
        "not_lost", np.datetime64("NaT", "ns"), 24, np.datetime64("NaT", "ns"), "test",
    )


def one_sided_case(kind="outgoing"):
    lon = np.arange(45, dtype=float) * .001
    center = 22
    lon[center] = (
        lon[center - 1] - .0079 if kind == "outgoing"
        else lon[center + 1] + .0079
    )
    trajectory = track(lon)
    return trajectory, run_native_position_qc(trajectory, drogue())


def completed_geometry_case(platform="z"):
    trajectory = track([0, .001, .002, .2, .003, .004, .005], platform=platform)
    baseline = run_native_position_qc(trajectory, drogue())
    point = baseline.loc[baseline.point_auto_status == "high_confidence_reject"].iloc[0]
    reviews = pd.DataFrame([{
        "platform_code": platform,
        "target_type": "observation",
        "target_id": f"observation:{int(point.source_obs_index)}",
        "source_obs_index": int(point.source_obs_index),
        "decision": "reject",
        "decision_source": "manual",
        "event_id": point.local_event_id,
        "event_type": point.local_event_type,
        "auto_status_snapshot": point.point_auto_status,
        "auto_decision_snapshot": point.point_auto_decision,
        "auto_reason_snapshot": point.point_auto_reason,
        "qc_config_sha256": "config-current",
    }])
    return trajectory, run_native_position_qc(trajectory, drogue(), reviews=reviews)


def two_ambiguous_case():
    trajectory = track([0, .001, .002, .03, .031, .032, .033, .034, .035, .065, .066, .067])
    return trajectory, run_native_position_qc(trajectory, drogue())


def make_reviewer(tmp_path, trajectory, frame, *, reviews=None, recompute=None,
                  mode="manual", session_path=None, temporal=None):
    reviews = reviews or PositionReviews(tmp_path / "position_review.csv")
    temporal = temporal or TemporalSegmentConfig()
    recompute = recompute or (lambda _code: run_native_position_qc(
        trajectory, drogue(), reviews=reviews.table(), temporal_config=temporal,
    ))
    return PositionReviewer(
        frame, reviews, mode=mode, config_sha256="config-current", context_points=3,
        trajectory_loader=lambda _code: trajectory, recompute_platform=recompute,
        session_path=session_path, position_config=NativePositionConfig(),
    )


def make_multi_reviewer(tmp_path, trajectories, frames, *, reviews=None,
                        recompute=None, session_path=None):
    reviews = reviews or PositionReviews(tmp_path / "position_review.csv")
    catalog = pd.concat([
        position_event_catalog(frames[code], "config-current")
        for code in sorted(frames)
    ], ignore_index=True)
    recompute = recompute or (lambda code: frames[code])
    return PositionReviewer(
        None, reviews, mode="manual", config_sha256="config-current", context_points=3,
        event_catalog=catalog, platform_loader=lambda code: frames[code],
        trajectory_loader=lambda code: trajectories[code],
        recompute_platform=recompute, session_path=session_path,
        position_config=NativePositionConfig(),
    )


def session_payload(platform, event_id, *, source=None, source_hash=None):
    return {
        "schema_version": 2, "mode": "manual", "platform_code": platform,
        "event_id": event_id, "event_queue_index": 0,
        "selected_source_obs_index": source, "selected_time_utc": "",
        "config_sha256": "config-current",
        "source_sha256": source_hash or f"source-{platform}",
        "review_sha256": "", "draft_note": "", "saved_at_utc": "",
    }


def select_event_type(reviewer, event_type):
    for position in range(len(reviewer.events)):
        event = reviewer.events.iloc[position]
        first = reviewer._group_for(event).iloc[0]
        value = (first.temporal_event_type if event.kind == "temporal" else
                 first.human_review_event_type if event.kind == "reviewed_local" else
                 first.local_event_type)
        if value == event_type:
            reviewer._change_event(position)
            return
    raise AssertionError(f"No {event_type!r} event")


def representative_case(name):
    temporal = TemporalSegmentConfig()
    if name == "ambiguous":
        trajectory, expected = track([0, .001, .002, .03, .031, .032]), "single_edge_ambiguous"
    elif name == "isolated_spike":
        trajectory, expected = track([0, .001, .002, .2, .003, .004, .005]), "local_spike_or_short_block"
    elif name == "persistent":
        trajectory, expected = track(
            [0, .001, .002, .20, .21, .22, .23, .24, .25, .003, .004, .005],
        ), "persistent_excursion"
    elif name == "tiny_fragment":
        seconds = [0, 300, 600] + list(3 * 86400 + np.arange(60) * 3600)
        trajectory, expected = track(np.arange(len(seconds)) * .0001, seconds=seconds), "pre_main_fragment"
    elif name == "multiple_segments":
        seconds = list(np.arange(6) * 300) + list(86400 + np.arange(6) * 300)
        trajectory, expected = track(np.arange(12) * .0001, seconds=seconds), "multiple_substantial_segments"
    elif name == "duplicate_time":
        trajectory, expected = track([0, .001, .002, .003, .004], seconds=[0, 300, 300, 600, 900]), "boundary_or_insufficient_context"
    else:
        raise AssertionError(name)
    return trajectory, run_native_position_qc(trajectory, drogue(), temporal_config=temporal), temporal, expected


@pytest.mark.parametrize("name", [
    "ambiguous", "isolated_spike", "persistent", "tiny_fragment",
    "multiple_segments", "duplicate_time",
])
def test_three_panels_seven_controls_and_unmistakable_selection(tmp_path, name):
    trajectory, frame, temporal, expected = representative_case(name)
    reviewer = make_reviewer(tmp_path / name, trajectory, frame, temporal=temporal)
    select_event_type(reviewer, expected)
    assert reviewer.overview_ax.get_title() == "Full native trajectory"
    assert reviewer.local_ax.get_title() == "Local event map"
    assert reviewer.velocity_ax.get_title() == "Surviving velocity"
    assert set(reviewer.buttons) == {
        "previous_point", "next_point", "keep_point", "reject_point",
        "previous_event", "next_event", "quit",
    }
    assert not hasattr(reviewer, "note_box") and not hasattr(reviewer, "sequence_ax")
    for artist in (
        reviewer._selected_overview, reviewer._selected_local,
        reviewer._selected_overview_ring, reviewer._selected_local_ring,
        reviewer._selected_overview_label, reviewer._selected_label,
    ):
        assert artist.get_visible()
    assert reviewer._selected_label.get_text().startswith("SELECTED ")
    assert reviewer.point_info_text.get_text().startswith("SELECTED ")
    if expected in {"pre_main_fragment", "multiple_substantial_segments"}:
        assert not reviewer.buttons["previous_point"].active
        assert not reviewer.buttons["next_point"].active
        assert reviewer.buttons["keep_point"].label.get_text() == "K  Retain segment"
        assert reviewer.buttons["reject_point"].label.get_text() == "R  Exclude segment"
    plt.close(reviewer.figure)


def test_point_navigation_is_fast_eligible_nonwrapping_and_unsaved(tmp_path, monkeypatch):
    trajectory, frame, _temporal, _event_type = representative_case("ambiguous")
    frame.loc[frame.source_obs_index == 0, "source_position_valid"] = False
    frame.loc[frame.source_obs_index == 1, "drogue_eligible"] = False
    frame.loc[frame.source_obs_index == 5, "segment_status"] = "pending"
    calls = []
    reviewer = make_reviewer(tmp_path, trajectory, frame, recompute=lambda code: calls.append(code) or frame)
    assert reviewer.selectable_source_indices == [2, 3, 4]
    static = {axis: (tuple(map(id, axis.lines)), tuple(map(id, axis.collections)))
              for axis in (reviewer.overview_ax, reviewer.local_ax, reviewer.velocity_ax)}
    monkeypatch.setattr(reviewer, "_presentation", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("rebuilt")))
    reviewer.select_source(2)
    reviewer.navigate_point(-1)
    assert reviewer.selected_source_obs_index == 2
    reviewer.navigate_point(1); reviewer.navigate_point(1); reviewer.navigate_point(1)
    assert reviewer.selected_source_obs_index == 4
    assert calls == [] and not reviewer.session_path.exists()
    assert static == {axis: (tuple(map(id, axis.lines)), tuple(map(id, axis.collections)))
                      for axis in (reviewer.overview_ax, reviewer.local_ax, reviewer.velocity_ax)}
    assert "SELECTED 4" in reviewer.point_info_text.get_text()
    plt.close(reviewer.figure)


def test_reviewer_disconnects_conflicting_matplotlib_shortcuts(tmp_path):
    trajectory, frame, temporal, _event_type = representative_case("isolated_spike")
    reviewer = make_reviewer(tmp_path, trajectory, frame, temporal=temporal)
    handler_id = getattr(reviewer.figure.canvas.manager, "key_press_handler_id", None)
    registered = reviewer.figure.canvas.callbacks.callbacks.get("key_press_event", {})

    assert handler_id is None or handler_id not in registered
    assert reviewer.velocity_ax.get_xscale() == "linear"
    reviewer.on_key(SimpleNamespace(key="k"))
    assert reviewer.velocity_ax.get_xscale() == "linear"
    plt.close(reviewer.figure)


def test_point_navigation_is_chronological_for_reverse_order_source(tmp_path):
    trajectory = track(
        [.032, .031, .03, .002, .001, 0],
        seconds=[1500, 1200, 900, 600, 300, 0],
    )
    frame = run_native_position_qc(trajectory, drogue())
    reviewer = make_reviewer(tmp_path, trajectory, frame)
    ordered = reviewer.selectable_source_indices
    time_by_source = dict(zip(
        trajectory.source_obs_index.astype(int), trajectory.time,
    ))

    assert ordered == [5, 4, 3, 2, 1, 0]
    reviewer.select_source(3)
    reviewer.navigate_point(1)
    assert reviewer.selected_source_obs_index == 2
    assert time_by_source[2] > time_by_source[3]
    plt.close(reviewer.figure)


def test_trigger_evidence_does_not_cover_surviving_blue_line(tmp_path):
    trajectory, frame, temporal, event_type = representative_case("ambiguous")
    reviewer = make_reviewer(tmp_path, trajectory, frame, temporal=temporal)
    select_event_type(reviewer, event_type)

    red_edges = [line for line in reviewer.local_ax.lines if line.get_color() == "red"]
    blue_edges = [line for line in reviewer.local_ax.lines if line.get_color() == "tab:blue"]
    assert red_edges and blue_edges
    assert all(line.get_linestyle() != "-" and line.get_alpha() < 1 for line in red_edges)
    assert max(line.get_zorder() for line in red_edges) < min(
        line.get_zorder() for line in blue_edges
    )
    trigger_markers = [
        collection for collection in reviewer.velocity_ax.collections
        if collection.get_label() == "frozen trigger edge"
    ]
    assert trigger_markers
    assert all(len(collection.get_facecolors()) > 0 for collection in trigger_markers)
    plt.close(reviewer.figure)


def test_map_click_selects_and_recenters_without_qc(tmp_path):
    trajectory = track(np.arange(30) * .0001)
    frame = run_native_position_qc(trajectory, drogue())
    frame.loc[10, ["local_event_id", "local_event_type", "local_review_required"]] = ["event", "single_edge_ambiguous", True]
    calls = []
    reviewer = make_reviewer(tmp_path, trajectory, frame, recompute=lambda code: calls.append(code) or frame)
    source = 29
    pixel = reviewer.overview_ax.transData.transform([trajectory.lon[source], trajectory.lat[source]])
    reviewer.on_map_click(SimpleNamespace(inaxes=reviewer.overview_ax, button=1, x=pixel[0], y=pixel[1]))
    assert reviewer.selected_source_obs_index == source and source in reviewer.display_source_indices
    assert calls == []
    plt.close(reviewer.figure)


def test_r_stages_and_n_saves_recomputes_once_and_writes_cursor(tmp_path):
    trajectory, frame, temporal, event_type = representative_case("ambiguous")
    reviews = PositionReviews(tmp_path / "review.csv")
    calls = []
    def recompute(code):
        calls.append(code)
        return run_native_position_qc(trajectory, drogue(), reviews=reviews.table(), temporal_config=temporal)
    reviewer = make_reviewer(tmp_path, trajectory, frame, reviews=reviews, recompute=recompute)
    select_event_type(reviewer, event_type)
    selected = int(reviewer.selected_source_obs_index)
    reviewer.action("reject_point")
    assert reviews.table().empty and calls == []
    assert reviewer._staged_point_decision("p", selected) == "reject"
    assert "staged: reject" in reviewer.point_info_text.get_text()
    reviewer.navigate_event(1)
    assert calls == ["p"]
    saved = reviews.table()
    assert saved.source_obs_index.astype(int).tolist() == [selected]
    assert saved.decision.tolist() == ["reject"] and reviewer.session_path.exists()
    plt.close(reviewer.figure)


def test_p_keeps_staged_edits_in_memory_without_saving(tmp_path):
    trajectory, frame = two_ambiguous_case()
    calls = []
    reviewer = make_reviewer(tmp_path, trajectory, frame, recompute=lambda code: calls.append(code) or frame)
    assert len(reviewer.events) >= 2
    event_id = str(reviewer.events.iloc[reviewer.index].event_id)
    reviewer.action("reject_point")
    reviewer.navigate_event(-1)
    assert reviewer.reviews.table().empty and calls == [] and not reviewer.session_path.exists()
    assert any(item["event_id"] == event_id for item in reviewer._staged.values())
    plt.close(reviewer.figure)


def test_q_commits_all_staged_events_and_recomputes_platform_once(tmp_path):
    trajectory, frame = two_ambiguous_case()
    reviews = PositionReviews(tmp_path / "review.csv")
    calls = []
    def recompute(code):
        calls.append(code)
        return run_native_position_qc(trajectory, drogue(), reviews=reviews.table())
    reviewer = make_reviewer(tmp_path, trajectory, frame, reviews=reviews, recompute=recompute)
    first = int(reviewer.selected_source_obs_index)
    reviewer.action("reject_point")
    reviewer._change_event(1)
    second = int(reviewer.selected_source_obs_index)
    reviewer.action("reject_point")
    reviewer.clean_quit()
    assert calls == ["p"]
    assert set(reviews.table().source_obs_index.astype(int)) == {first, second}
    assert json.loads(reviewer.session_path.read_text(encoding="utf-8"))["draft_note"] == ""
    assert reviewer.close_handled


def test_n_without_edits_only_saves_cursor_and_leaves_event_unresolved(tmp_path):
    trajectory, frame, _temporal, _event_type = representative_case("ambiguous")
    calls = []
    reviewer = make_reviewer(tmp_path, trajectory, frame, recompute=lambda code: calls.append(code) or frame)
    reviewer.navigate_event(1)
    assert calls == [] and reviewer.reviews.table().empty and reviewer.session_path.exists()
    assert frame.final_position_status.isin(["unresolved", "uncertain"]).any()
    plt.close(reviewer.figure)


def test_k_handles_automatic_rejection_and_temporal_controls_are_contextual(tmp_path):
    trajectory, frame, temporal, event_type = representative_case("isolated_spike")
    reviewer = make_reviewer(tmp_path / "point", trajectory, frame, temporal=temporal)
    select_event_type(reviewer, event_type)
    rejected = int(frame.loc[frame.final_position_status == "rejected", "source_obs_index"].iloc[0])
    reviewer.select_source(rejected)
    assert reviewer.buttons["keep_point"].active
    assert reviewer._staged_reject_local.get_visible()
    reviewer.action("keep_point")
    assert reviewer._staged_point_decision("p", rejected) == "keep"
    assert reviewer._provisional_local_line.get_visible()
    assert reviewer._live_velocity_line.get_visible()
    assert not reviewer._staged_reject_local.get_visible()
    plt.close(reviewer.figure)
    trajectory, frame, temporal, event_type = representative_case("multiple_segments")
    reviewer = make_reviewer(tmp_path / "segment", trajectory, frame, temporal=temporal)
    select_event_type(reviewer, event_type)
    reviewer.action("reject_point")
    assert any(key[1] == "segment" and item["decision"] == "exclude" for key, item in reviewer._staged.items())
    plt.close(reviewer.figure)


def test_atomic_review_failure_on_n_keeps_window_and_staging(tmp_path, monkeypatch):
    trajectory, frame, _temporal, event_type = representative_case("ambiguous")
    reviewer = make_reviewer(tmp_path, trajectory, frame)
    select_event_type(reviewer, event_type)
    reviewer.action("reject_point")
    monkeypatch.setattr(reviewer.reviews, "save", lambda: (_ for _ in ()).throw(OSError("locked")))
    reviewer.navigate_event(1)
    assert reviewer._staged and reviewer.reviews.table().empty
    assert "not saved" in reviewer.status_text.get_text() and plt.fignum_exists(reviewer.figure.number)
    plt.close(reviewer.figure)


def test_staged_rejection_marks_speed_and_recalculates_provisional_adjacency(tmp_path):
    trajectory, frame, _temporal, event_type = representative_case("ambiguous")
    calls = []
    reviewer = make_reviewer(
        tmp_path, trajectory, frame, recompute=lambda code: calls.append(code) or frame,
    )
    select_event_type(reviewer, event_type)
    selected = int(reviewer.selected_source_obs_index)
    eligible = reviewer.selectable_source_indices
    selected_position = eligible.index(selected)
    assert 0 < selected_position < len(eligible) - 1
    previous_source, next_source = eligible[selected_position - 1], eligible[selected_position + 1]
    live_artist = reviewer._live_velocity_line
    frozen = next(
        collection for collection in reviewer.velocity_ax.collections
        if collection.get_label() == "frozen trigger edge"
    )
    frozen_before = np.asarray(frozen.get_offsets()).copy()

    reviewer.action("reject_point")
    provisional = reviewer._provisional_speed_model("p")
    assert calls == []
    assert provisional["incoming"][next_source][2] == previous_source
    assert selected not in provisional["incoming"] and selected not in provisional["outgoing"]
    assert reviewer._provisional_local_line.get_visible()
    assert reviewer._live_velocity_line is live_artist
    assert reviewer._live_velocity_line.get_visible()
    assert reviewer._original_local_overlay.get_visible()
    for removed in (
        "_provisional_velocity_line", "_original_velocity_overlay",
        "_provisional_bridge_velocity", "_staged_velocity_reject",
        "_rejected_velocity_no_speed", "_staged_velocity_keep",
    ):
        assert not hasattr(reviewer, removed)
    assert np.array_equal(np.asarray(frozen.get_offsets()), frozen_before)
    preview_speed = np.asarray(reviewer._live_velocity_line.get_ydata(), dtype=float)
    displayed = set(reviewer.display_source_indices)
    expected_preview_count = sum(
        int(trajectory.source_obs_index[predecessor]) in displayed
        and int(source) in displayed and np.isfinite(speed)
        for predecessor, source, speed in zip(
            provisional["predecessor_positions"],
            provisional["sources"], provisional["speed"],
        )
    )
    assert np.isfinite(preview_speed).sum() == expected_preview_count
    assert expected_preview_count > 1
    local_segments = np.column_stack(reviewer._provisional_local_line.get_data()).reshape(-1, 3, 2)
    expected_bridge = np.asarray([
        [trajectory.lon[previous_source], trajectory.lat[previous_source]],
        [trajectory.lon[next_source], trajectory.lat[next_source]],
    ])
    assert any(np.allclose(segment[:2], expected_bridge) for segment in local_segments)
    selected_position_native = reviewer._native("p")["source_to_native"][selected]
    next_position_native = reviewer._native("p")["source_to_native"][next_source]
    native_times = reviewer._native("p")["times"]
    live_times = pd.to_datetime(reviewer._live_velocity_line.get_xdata(), utc=True)
    finite = np.isfinite(preview_speed) & ~pd.isna(live_times)
    live_by_time = dict(zip(live_times[finite], preview_speed[finite]))
    assert native_times[next_position_native] in live_by_time
    assert live_by_time[native_times[next_position_native]] == pytest.approx(
        provisional["incoming"][next_source][0],
    )
    assert native_times[selected_position_native] not in live_by_time

    reviewer.action("keep_point")
    restored = reviewer._provisional_speed_model("p")
    assert restored["incoming"][selected][2] == previous_source
    assert restored["incoming"][next_source][2] == selected
    restored_speed = np.asarray(reviewer._live_velocity_line.get_ydata(), dtype=float)
    restored_times = pd.to_datetime(reviewer._live_velocity_line.get_xdata(), utc=True)
    restored_by_time = dict(zip(
        restored_times[np.isfinite(restored_speed)], restored_speed[np.isfinite(restored_speed)],
    ))
    assert native_times[selected_position_native] in restored_by_time
    assert restored_by_time[native_times[next_position_native]] == pytest.approx(
        restored["incoming"][next_source][0],
    )
    plt.close(reviewer.figure)


def test_velocity_preview_separates_chains_and_expands_visible_range(tmp_path):
    times, speed = PositionReviewer._segmented_velocity_data(
        np.asarray([0, 1, 3]), np.asarray([1, 2, 4]),
        pd.to_datetime(["2025-01-01T00:05Z", "2025-01-01T00:10Z", "2025-01-01T00:20Z"]),
        np.asarray([1.0, 2.0, 9.0]), np.asarray([True, True, True]),
    )
    assert pd.isna(times[2]) and np.isnan(speed[2])
    assert speed[[0, 1, 3]].tolist() == [1.0, 2.0, 9.0]

    trajectory, frame, _temporal, event_type = representative_case("ambiguous")
    reviewer = make_reviewer(tmp_path, trajectory, frame)
    select_event_type(reviewer, event_type)
    reviewer.velocity_ax.set_ylim(0, .1)
    reviewer.action("reject_point")
    preview = np.asarray(reviewer._live_velocity_line.get_ydata(), dtype=float)
    assert np.nanmax(preview) > .1
    assert reviewer.velocity_ax.get_ylim()[1] > np.nanmax(preview)
    plt.close(reviewer.figure)


def test_velocity_xlim_comes_from_displayed_raw_times_and_contains_live_line(tmp_path):
    trajectory, frame = one_sided_case()
    reviewer = make_reviewer(tmp_path, trajectory, frame)

    displayed_positions = [
        reviewer._native("p")["source_to_native"][source]
        for source in reviewer.display_source_indices
    ]
    displayed_times = reviewer._native("p")["times"][displayed_positions]
    expected = reviewer.velocity_ax.convert_xunits(displayed_times)
    lower, upper = reviewer.velocity_ax.get_xlim()
    assert lower < np.nanmin(expected) < np.nanmax(expected) < upper
    assert upper - lower <= 1.06 * (np.nanmax(expected) - np.nanmin(expected))

    live_times = pd.to_datetime(reviewer._live_velocity_line.get_xdata(), utc=True)
    finite_speed = np.isfinite(np.asarray(reviewer._live_velocity_line.get_ydata(), dtype=float))
    live_x = reviewer.velocity_ax.convert_xunits(live_times[finite_speed])
    assert len(live_x) and np.all((live_x >= lower) & (live_x <= upper))
    assert not reviewer._empty_velocity_text.get_visible()

    reviewer.action("keep_point")
    lower, upper = reviewer.velocity_ax.get_xlim()
    live_times = pd.to_datetime(reviewer._live_velocity_line.get_xdata(), utc=True)
    finite_speed = np.isfinite(np.asarray(reviewer._live_velocity_line.get_ydata(), dtype=float))
    live_x = reviewer.velocity_ax.convert_xunits(live_times[finite_speed])
    assert len(live_x) and np.all((live_x >= lower) & (live_x <= upper))
    plt.close(reviewer.figure)


def test_empty_velocity_series_is_explained_instead_of_looking_broken(tmp_path):
    trajectory, frame, _temporal, event_type = representative_case("ambiguous")
    frame = frame.copy()
    frame["surviving_predecessor_source_obs_index"] = np.nan
    frame["surviving_dt_seconds"] = np.nan
    frame["surviving_speed_m_s"] = np.nan
    reviewer = make_reviewer(tmp_path, trajectory, frame)
    select_event_type(reviewer, event_type)

    assert not reviewer._live_velocity_line.get_visible()
    assert reviewer._empty_velocity_text.get_visible()
    assert reviewer._empty_velocity_text.get_text() == "No surviving velocity in this window"
    plt.close(reviewer.figure)


def test_one_sided_spike_is_manual_confirmation_but_not_semiautomatic_work(tmp_path):
    trajectory, frame = one_sided_case()
    catalog = position_event_catalog(frame, "config-current")
    event = catalog.iloc[0]
    assert not bool(event.pending)
    assert bool(event.manual_confirmation_required)

    manual = make_reviewer(tmp_path / "manual", trajectory, frame, mode="manual")
    assert len(manual.events) == 1 and bool(manual.events.iloc[0].pending)
    assert manual.selected_source_obs_index == 22
    assert "confirmation required" in manual.presentation.title.lower()
    assert "competing endpoint" in manual.presentation.explanation.lower()
    assert "clean local speeds" in manual.presentation.explanation.lower()
    plt.close(manual.figure)

    semiautomatic = make_reviewer(
        tmp_path / "semiautomatic", trajectory, frame, mode="semiautomatic",
    )
    assert semiautomatic.events.empty
    plt.close(semiautomatic.figure)


def test_time_bounded_excursion_is_point_by_point_manual_confirmation(tmp_path):
    trajectory = track([0, .001, .002, .20, .21, .22, .23, .24, .003, .004, .005])
    frame = run_native_position_qc(trajectory, drogue())
    proposed = frame.loc[
        frame.point_auto_reason == "bidirectionally_confirmed_time_bounded_excursion_cure",
        "source_obs_index",
    ].astype(int).tolist()
    catalog = position_event_catalog(frame, "config-current")

    assert proposed == [3, 4, 5, 6, 7]
    assert bool(catalog.iloc[0].manual_confirmation_required)
    manual = make_reviewer(tmp_path / "manual-block", trajectory, frame, mode="manual")
    assert manual.presentation.proposed_sources == tuple(proposed)
    assert manual.selected_source_obs_index == proposed[0]
    assert "elapsed time" in manual.presentation.explanation.lower()
    assert manual._action_target() == f"Action target: selected observation [{proposed[0]}]"
    semiautomatic = make_reviewer(
        tmp_path / "semi-block", trajectory, frame, mode="semiautomatic",
    )
    assert semiautomatic.events.empty
    plt.close(manual.figure)
    plt.close(semiautomatic.figure)


def test_partially_confirmed_excursion_remains_pending_after_recalculation(tmp_path):
    trajectory = track([0, .001, .002, .20, .21, .22, .23, .24, .003, .004, .005])
    reviews = PositionReviews(tmp_path / "review.csv")

    def recompute(_code):
        return run_native_position_qc(trajectory, drogue(), reviews=reviews.table())

    frame = recompute("p")
    reviewer = make_reviewer(
        tmp_path, trajectory, frame, reviews=reviews, recompute=recompute,
    )
    event_id = str(reviewer.events.iloc[reviewer.index].event_id)
    reviewer.action("reject_point")
    assert reviewer._commit_events({event_id})

    assert reviews.table().source_obs_index.astype(int).tolist() == [3]
    assert reviewer.events.pending.astype(bool).any()
    remaining = reviewer._frame("p")
    proposed = remaining.loc[
        remaining.point_auto_reason == "bidirectionally_confirmed_time_bounded_excursion_cure",
        "source_obs_index",
    ].astype(int).tolist()
    assert proposed == [4, 5, 6, 7]
    plt.close(reviewer.figure)


@pytest.mark.parametrize("decision", ["reject", "keep"])
def test_manual_endpoint_r_or_k_recomputes_confirmation(tmp_path, decision):
    trajectory, frame = one_sided_case()
    reviews = PositionReviews(tmp_path / decision / "review.csv")

    def recompute(_code):
        return run_native_position_qc(trajectory, drogue(), reviews=reviews.table())

    reviewer = make_reviewer(
        tmp_path / decision, trajectory, frame, reviews=reviews, recompute=recompute,
    )
    event_id = str(reviewer.events.iloc[reviewer.index].event_id)
    reviewer.action("reject_point" if decision == "reject" else "keep_point")
    assert reviewer._commit_events({event_id})
    saved = reviews.table()
    assert saved.source_obs_index.astype(int).tolist() == [22]
    assert saved.decision.tolist() == [decision]
    if decision == "reject":
        assert not reviewer.events.manual_confirmation_required.astype(bool).any()
        assert not reviewer.events.pending.astype(bool).any()
    else:
        updated = reviewer._frame("p")
        point = updated.loc[updated.source_obs_index == 22].iloc[0]
        assert point.final_position_status == "valid"
        assert point.position_decision_source == "human"
        assert reviewer.events.manual_confirmation_required.astype(bool).any()
    plt.close(reviewer.figure)


def test_n_from_last_pending_event_opens_completion_screen(tmp_path):
    trajectory, frame = one_sided_case()
    reviews = PositionReviews(tmp_path / "review.csv")

    def recompute(_code):
        return run_native_position_qc(trajectory, drogue(), reviews=reviews.table())

    reviewer = make_reviewer(
        tmp_path, trajectory, frame, reviews=reviews, recompute=recompute,
    )
    reviewer.action("reject_point")
    reviewer.navigate_event(1)

    assert reviewer._completion_screen
    assert "REVIEW COMPLETE" in reviewer.header_text.get_text()
    assert "Pending events remaining: 0" in reviewer.header_text.get_text()
    assert reviewer.session_path.exists()
    reviewer.navigate_event(-1)
    assert "COMPLETED HISTORY" in reviewer.header_text.get_text()
    plt.close(reviewer.figure)


def test_velocity_timing_diagnostics_use_distinct_destination_markers(tmp_path):
    cases = [
        (
            track([0, .02, .021, .022, .023], seconds=[0, 60, 360, 660, 960]),
            "short_dt_flag", "short interval",
        ),
        (
            track([0, .001, .2, .201, .202], seconds=[0, 300, 2401, 2701, 3001]),
            "large_gap_flag", "large gap",
        ),
        (
            track([0, .001, .002, .003, .004], seconds=[0, 300, 300, 600, 900]),
            "nonpositive_dt_flag", "duplicate/nonpositive interval",
        ),
    ]
    marker_signatures = set()
    for number, (trajectory, flag, label) in enumerate(cases):
        frame = run_native_position_qc(trajectory, drogue())
        reviewer = make_reviewer(tmp_path / str(number), trajectory, frame)
        marker = next(
            collection for collection in reviewer.velocity_ax.collections
            if collection.get_label() == label
        )
        flagged_time = pd.Timestamp(frame.loc[frame[flag], "time"].iloc[0])
        expected_x = float(reviewer.velocity_ax.convert_xunits(flagged_time))
        actual_x = np.asarray(marker.get_offsets())[:, 0]
        assert np.any(np.abs(actual_x - expected_x) < 1e-8)
        path = marker.get_paths()[0]
        marker_signatures.add(tuple(np.round(path.vertices.ravel(), 6)))
        plt.close(reviewer.figure)
    assert len(marker_signatures) == 3


def test_repeat_groups_are_hidden_and_skipped_but_remain_faint_raw_context(tmp_path):
    trajectory = track([0, .001, .2, .002, .2, .003, .004, .034, .035, .036])
    frame = run_native_position_qc(trajectory, drogue())
    reviewer = make_reviewer(tmp_path, trajectory, frame)
    members = frame.loc[frame.exact_repeat_flag, "source_obs_index"].astype(int).tolist()
    assert len(members) == 2
    assert not reviewer.events.event_type.astype(str).eq("repeated_position").any()
    assert not set(members) & set(reviewer.selectable_source_indices)
    assert all(reviewer._scope_colors("p", np.asarray([source]))[0][-1] == .12 for source in members)
    surviving_pairs = reviewer._surviving_edges_cache["p"]
    assert not any(source in pair for source in members for pair in surviving_pairs)
    live_times = set(pd.to_datetime(reviewer._live_velocity_line.get_xdata(), utc=True).dropna())
    native = reviewer._native("p")
    assert not any(native["times"][native["source_to_native"][source]] in live_times for source in members)
    plt.close(reviewer.figure)


def test_short_interval_removals_are_hidden_skipped_and_fixed_timing_context(tmp_path):
    trajectory = track(
        [0, .0001, .0002, .0003, .0303, .0313, .0323],
        seconds=[0, 60, 300, 600, 900, 1200, 1500],
    )
    frame = run_native_position_qc(trajectory, drogue())
    reviewer = make_reviewer(tmp_path, trajectory, frame)
    removed = frame.loc[
        frame.point_auto_status == "short_interval_reject", "source_obs_index",
    ].astype(int).tolist()

    assert removed == [1]
    assert not reviewer.events.event_type.astype(str).eq("short_interval").any()
    assert not set(removed) & set(reviewer.selectable_source_indices)
    assert reviewer._scope_colors("p", np.asarray(removed))[0][-1] == .12
    assert not any(
        source in pair for source in removed
        for pair in reviewer._surviving_edges_cache["p"]
    )
    live_times = set(pd.to_datetime(reviewer._live_velocity_line.get_xdata(), utc=True).dropna())
    native = reviewer._native("p")
    assert not any(
        native["times"][native["source_to_native"][source]] in live_times
        for source in removed
    )
    timing_marker = next(
        collection for collection in reviewer.velocity_ax.collections
        if collection.get_label() == "short interval"
    )
    expected_x = float(reviewer.velocity_ax.convert_xunits(native["times"][1]))
    assert np.any(np.abs(np.asarray(timing_marker.get_offsets())[:, 0] - expected_x) < 1e-8)

    selected = reviewer.selected_source_obs_index
    reviewer.select_source(removed[0])
    assert reviewer.selected_source_obs_index == selected
    reviewer.selected_source_obs_index = removed[0]
    reviewer.draw()
    assert reviewer.selected_source_obs_index != removed[0]
    reviewer.action("reject_point")
    assert not any(key[2] == "observation:1" for key in reviewer._staged)
    plt.close(reviewer.figure)


def test_large_stagnant_group_is_hidden_without_building_a_presentation(tmp_path):
    repeated_count = 1200
    trajectory = track(np.r_[0, .001, np.full(repeated_count, .2), .002, .003, .004])
    frame = run_native_position_qc(trajectory, drogue())
    reviewer = make_reviewer(tmp_path, trajectory, frame)
    members = frame.loc[frame.exact_repeat_flag, "source_obs_index"].astype(int).tolist()

    assert len(members) == repeated_count
    assert reviewer.events.empty
    assert reviewer._completion_screen
    assert reviewer.presentation is None
    assert reviewer._staged == {}
    assert "REVIEW COMPLETE" in reviewer.header_text.get_text()
    plt.close(reviewer.figure)


def test_point_action_render_failure_rolls_back_without_escaping(tmp_path, monkeypatch):
    trajectory, frame, _temporal, event_type = representative_case("isolated_spike")
    reviewer = make_reviewer(tmp_path, trajectory, frame)
    select_event_type(reviewer, event_type)
    rejected = int(frame.loc[frame.final_position_status == "rejected", "source_obs_index"].iloc[0])
    reviewer.select_source(rejected)
    monkeypatch.setattr(
        reviewer, "_update_selection_artists",
        lambda: (_ for _ in ()).throw(RuntimeError("renderer failed")),
    )

    reviewer.action("keep_point")

    assert reviewer._staged == {}
    assert "rolled back safely" in reviewer.status_text.get_text()
    assert "renderer failed" in reviewer.status_text.get_text()
    assert plt.fignum_exists(reviewer.figure.number)
    plt.close(reviewer.figure)


def test_lazy_catalog_loads_only_visible_platform(tmp_path):
    first, first_frame, _, _ = representative_case("ambiguous")
    second, second_frame = completed_geometry_case("q")
    frames, tracks = {"p": first_frame, "q": second_frame}, {"p": first, "q": second}
    catalog = pd.concat([
        position_event_catalog(first_frame, "config-current"),
        position_event_catalog(second_frame, "config-current"),
    ], ignore_index=True)
    loaded_frames, loaded_tracks = [], []
    reviewer = PositionReviewer(
        None, PositionReviews(tmp_path / "review.csv"), mode="manual", config_sha256="config-current",
        event_catalog=catalog, platform_loader=lambda code: loaded_frames.append(code) or frames[code],
        trajectory_loader=lambda code: loaded_tracks.append(code) or tracks[code],
        recompute_platform=lambda code: frames[code],
    )
    assert loaded_frames == ["p"] and loaded_tracks == ["p"]
    reviewer.navigate_event(-1)
    assert loaded_frames == ["p", "q"] and loaded_tracks == ["p", "q"]
    assert len(reviewer.frames) <= 2 and len(reviewer._trajectory_cache) <= 2
    plt.close(reviewer.figure)


def test_completed_session_resumes_pending_and_p_reaches_history(tmp_path):
    pending_track = track([0, .001, .002, .03, .031, .032], platform="a")
    completed_track, completed_frame = completed_geometry_case("z")
    tracks = {"a": pending_track, "z": completed_track}
    frames = {
        "a": run_native_position_qc(pending_track, drogue()),
        "z": completed_frame,
    }
    completed_catalog = position_event_catalog(frames["z"], "config-current")
    completed_event = str(completed_catalog.loc[~completed_catalog.pending, "event_id"].iloc[0])
    session = tmp_path / "cursor.json"
    session.write_text(json.dumps(session_payload("z", completed_event)), encoding="utf-8")

    reviewer = make_multi_reviewer(
        tmp_path, tracks, frames, session_path=session,
    )

    current = reviewer.events.iloc[reviewer.index]
    assert bool(current.pending) and current.platform_code == "a"
    assert "Saved event is complete" in reviewer.status_text.get_text()
    assert "Pending events remaining: 1" in reviewer.header_text.get_text()
    reviewer.navigate_event(-1)
    assert not bool(reviewer.events.iloc[reviewer.index].pending)
    assert reviewer.events.iloc[reviewer.index].platform_code == "z"
    plt.close(reviewer.figure)


def test_pending_and_completed_only_sessions_restore_exact_cursor(tmp_path):
    pending_track = track([0, .001, .002, .03, .031, .032], platform="a")
    pending_frame = run_native_position_qc(pending_track, drogue())
    pending_catalog = position_event_catalog(pending_frame, "config-current")
    pending_event = str(pending_catalog.loc[pending_catalog.pending, "event_id"].iloc[0])
    selected = int(pending_frame.source_obs_index.iloc[2])
    pending_session = tmp_path / "pending.json"
    pending_session.write_text(
        json.dumps(session_payload("a", pending_event, source=selected)), encoding="utf-8",
    )
    pending_reviewer = make_multi_reviewer(
        tmp_path / "pending", {"a": pending_track}, {"a": pending_frame},
        session_path=pending_session,
    )
    assert str(pending_reviewer.events.iloc[pending_reviewer.index].event_id) == pending_event
    assert pending_reviewer.selected_source_obs_index == selected
    plt.close(pending_reviewer.figure)

    completed_track, completed_frame = completed_geometry_case("z")
    completed_catalog = position_event_catalog(completed_frame, "config-current")
    completed_event = str(completed_catalog.loc[~completed_catalog.pending, "event_id"].iloc[0])
    completed_session = tmp_path / "completed.json"
    completed_session.write_text(
        json.dumps(session_payload("z", completed_event)), encoding="utf-8",
    )
    completed_reviewer = make_multi_reviewer(
        tmp_path / "completed", {"z": completed_track}, {"z": completed_frame},
        session_path=completed_session,
    )
    assert str(completed_reviewer.events.iloc[completed_reviewer.index].event_id) == completed_event
    assert not bool(completed_reviewer.events.iloc[completed_reviewer.index].pending)
    assert completed_reviewer._completion_screen
    assert "REVIEW COMPLETE" in completed_reviewer.header_text.get_text()
    completed_reviewer.navigate_event(-1)
    assert not completed_reviewer._completion_screen
    assert "COMPLETED HISTORY" in completed_reviewer.header_text.get_text()
    completed_reviewer.navigate_event(1)
    assert completed_reviewer._completion_screen
    plt.close(completed_reviewer.figure)


def test_n_skips_completed_events_while_pending_review_is_active(tmp_path):
    tracks = {
        "a": track([0, .001, .002, .03, .031, .032], platform="a"),
        "b": track([0, .001, .002, .04, .041, .042], platform="b"),
    }
    completed_track, completed_frame = completed_geometry_case("z")
    tracks["z"] = completed_track
    frames = {code: run_native_position_qc(value, drogue()) for code, value in tracks.items() if code != "z"}
    frames["z"] = completed_frame
    reviewer = make_multi_reviewer(tmp_path, tracks, frames)

    assert reviewer.events.iloc[reviewer.index].platform_code == "a"
    assert "Pending events remaining: 2" in reviewer.header_text.get_text()
    assert "event 1/" not in reviewer.header_text.get_text().lower()
    reviewer.navigate_event(1)
    assert reviewer.events.iloc[reviewer.index].platform_code == "b"
    assert bool(reviewer.events.iloc[reviewer.index].pending)
    reviewer.navigate_event(1)
    assert reviewer.events.iloc[reviewer.index].platform_code == "a"
    assert bool(reviewer.events.iloc[reviewer.index].pending)
    reviewer.navigate_event(-1)
    assert reviewer.events.iloc[reviewer.index].platform_code == "z"
    assert not bool(reviewer.events.iloc[reviewer.index].pending)
    reviewer.navigate_event(1)
    assert reviewer.events.iloc[reviewer.index].platform_code == "a"
    plt.close(reviewer.figure)


def test_n_uses_chronological_anchor_when_extra_edits_regroup_future_event(tmp_path):
    trajectory, frame = two_ambiguous_case()
    reviews = PositionReviews(tmp_path / "review.csv")

    def recompute(_code):
        return run_native_position_qc(trajectory, drogue(), reviews=reviews.table())

    reviewer = make_reviewer(
        tmp_path, trajectory, frame, reviews=reviews, recompute=recompute,
    )
    initial_ids = reviewer.events.event_id.astype(str).tolist()
    reviewer.select_source(2)
    reviewer.action("reject_point")
    # Clean a point belonging to the next event while still viewing the first.
    reviewer.select_source(8)
    reviewer.action("reject_point")
    reviewer.navigate_event(1)

    current = reviewer.events.iloc[reviewer.index]
    assert str(current.event_id) not in initial_ids
    assert bool(current.pending)
    assert int(current.event_start_source_obs_index) == 7
    assert "Pending events remaining: 2" in reviewer.header_text.get_text()
    assert "COMPLETED HISTORY" not in reviewer.header_text.get_text()
    plt.close(reviewer.figure)


def test_old_footer_catalog_without_anchors_remains_readable(tmp_path):
    trajectory, frame = two_ambiguous_case()
    old_catalog = position_event_catalog(frame, "config-current").drop(columns=[
        "event_type", "event_start_time_utc", "event_start_source_obs_index",
    ])
    reviewer = PositionReviewer(
        None, PositionReviews(tmp_path / "review.csv"), mode="manual",
        config_sha256="config-current", event_catalog=old_catalog,
        platform_loader=lambda _code: frame,
        trajectory_loader=lambda _code: trajectory,
        recompute_platform=lambda _code: frame,
    )
    assert len(reviewer.events) == 2
    assert reviewer.events.event_start_time_utc.fillna("").eq("").all()
    reviewer.navigate_event(1)
    assert bool(reviewer.events.iloc[reviewer.index].pending)
    plt.close(reviewer.figure)


def test_q_then_restart_cannot_resume_in_completed_block(tmp_path):
    tracks = {
        "a": track([0, .001, .002, .03, .031, .032], platform="a"),
        "b": track([0, .001, .002, .04, .041, .042], platform="b"),
    }
    frames = {code: run_native_position_qc(value, drogue()) for code, value in tracks.items()}
    reviews = PositionReviews(tmp_path / "review.csv")

    def recompute(code):
        frames[code] = run_native_position_qc(
            tracks[code], drogue(), reviews=reviews.table(),
        )
        return frames[code]

    reviewer = make_multi_reviewer(
        tmp_path, tracks, frames, reviews=reviews, recompute=recompute,
    )
    reviewer.action("reject_point")
    reviewer.clean_quit()

    resumed = make_multi_reviewer(
        tmp_path, tracks, frames, reviews=reviews, recompute=recompute,
        session_path=reviewer.session_path,
    )
    assert bool(resumed.events.iloc[resumed.index].pending)
    assert int(resumed.events.pending.astype(bool).sum()) >= 1
    assert "completed history" not in resumed.status_text.get_text().lower()
    plt.close(resumed.figure)


def test_unsupported_corrupt_and_source_mismatched_sessions_fall_back(tmp_path):
    trajectory, frame, _temporal, _event_type = representative_case("persistent")
    event_id = frame.loc[frame.local_event_id.astype(str).ne(""), "local_event_id"].iloc[0]
    session = tmp_path / "cursor.json"
    session.write_text(f'{{"event_id": "{event_id}", "point_index": 1}}', encoding="utf-8")
    unsupported = make_reviewer(tmp_path, trajectory, frame, session_path=session)
    assert "unsupported schema" in unsupported.status_text.get_text().lower()
    plt.close(unsupported.figure)
    session.write_text("not-json", encoding="utf-8")
    corrupt = make_reviewer(tmp_path, trajectory, frame, session_path=session)
    assert "unreadable" in corrupt.status_text.get_text()
    plt.close(corrupt.figure)
    payload = {
        "schema_version": 2, "mode": "manual", "platform_code": "p", "event_id": event_id,
        "event_queue_index": 0, "selected_source_obs_index": None, "selected_time_utc": "",
        "config_sha256": "config-current", "source_sha256": "different", "review_sha256": "",
        "draft_note": "ignored", "saved_at_utc": "",
    }
    session.write_text(json.dumps(payload), encoding="utf-8")
    mismatch = make_reviewer(tmp_path, trajectory, frame, session_path=session)
    assert "source hash changed" in mismatch.status_text.get_text()
    plt.close(mismatch.figure)
