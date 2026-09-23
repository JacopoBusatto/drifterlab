"""Focused recovery geometry, audit persistence and explicit UI acceptance."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from drifterlab.cli.review_arcterx_positions import main
from drifterlab.experiments.arcterx import preprocess
from drifterlab.experiments.arcterx.jump_recovery import recover_region, suggest_regions
from drifterlab.experiments.arcterx.position_review import (
    Decisions, apply_decisions, build_iteration, candidates, load_queue, load_suggestions, native_frame,
)
from drifterlab.qc.position import position_valid


def synthetic(lon, seconds=None):
    lon = np.asarray(lon, dtype=float)
    n = len(lon)
    times = np.datetime64("2025-02-06T00:00:00.000004471", "ns") + np.asarray(
        np.arange(n) * 300 if seconds is None else seconds, dtype="timedelta64[s]")
    return pd.DataFrame({"platform_code": "0001", "time_qc": times, "time": pd.to_datetime(times, utc=True),
                         "lon_qc": lon, "lat_qc": np.zeros(n), "position_valid": position_valid(lon, np.zeros(n)),
                         "source_obs_index": np.arange(n)[::-1], "source_sha256": "source-hash",
                         "row_index": np.arange(n), "manual_review_decision": "",
                         "reviewed_position_valid": position_valid(lon, np.zeros(n))})


def proposals(data, adjacency="native"):
    queue, edges = candidates(data, 1, adjacency=adjacency)
    enhanced, suggestions = suggest_regions(data, queue, edges)
    return enhanced, suggestions


@pytest.mark.parametrize("count", [1, 2, 3])
def test_exact_blocks_up_to_three_have_two_anchors_and_continuations(count):
    data = synthetic([0, .001, .002, *([.2] * count), .003, .004, .005])
    before = data.copy(deep=True)
    queue, suggestions = proposals(data)
    high = suggestions[suggestions.auto_suggested_reject]
    assert high.row_index.tolist() == list(range(3, 3 + count))
    assert high.auto_skip_count.eq(count).all()
    assert high.auto_forward_backward_agree.all()
    assert high.auto_bridge_speed_m_s.le(3).all()
    assert high.auto_continuation_speed_m_s.le(3).all()
    assert high.auto_backward_continuation_speed_m_s.le(3).all()
    assert high.auto_block_id.nunique() == 1
    assert queue.auto_region_status.tolist() == ["high"]
    assert high.time.str.endswith("000004471+00:00").all()
    pd.testing.assert_frame_equal(data, before)


def test_alternating_spikes_reconnect_recursively_as_separate_blocks():
    queue, suggestions = proposals(synthetic([0, .001, .002, .2, .003, .2, .004, .005, .006]))
    high = suggestions[suggestions.auto_suggested_reject]
    assert len(queue) == 1
    assert high.row_index.tolist() == [3, 5]
    assert high.auto_skip_count.tolist() == [1, 1]
    assert high.auto_block_id.nunique() == 2
    # First continuation crosses the next spike, not the immediate bad fix.
    assert high.auto_reconnection_row.tolist() == [4, 6]
    assert high.auto_continuation_row.tolist() == [6, 7]


def test_four_point_block_cannot_be_skipped():
    _, suggestions = proposals(synthetic([0, .001, .002, .2, .2, .2, .2, .003, .004, .005]))
    assert not suggestions.auto_suggested_reject.any()
    assert suggestions.auto_confidence.eq("unresolved").all()
    assert suggestions.auto_skip_count.le(3).all()


@pytest.mark.parametrize("lon", [[.2, 0, .001, .002, .003], [0, .2, .001, .2, .002]])
def test_start_without_coherent_run_stays_manual(lon):
    _, suggestions = proposals(synthetic(lon))
    assert len(suggestions)
    assert not suggestions.auto_suggested_reject.any()
    assert suggestions.auto_confidence.eq("unresolved").all()


@pytest.mark.parametrize("suffix", [[], [.004]])
def test_forward_only_recovery_is_medium_without_continuation_or_backward_anchor(suffix):
    _, suggestions = proposals(synthetic([0, .001, .002, .2, .003, *suffix]))
    row = suggestions[suggestions.row_index == 3].iloc[0]
    assert row.auto_confidence == "medium" and not row.auto_suggested_reject
    assert row.auto_bridge_speed_m_s < 3
    assert row.auto_continuation_speed_m_s < 3 if suffix else np.isnan(row.auto_continuation_speed_m_s)
    assert not row.auto_forward_backward_agree


def test_forward_backward_disagreement_is_unresolved():
    _, suggestions = proposals(synthetic([0, .001, .002, .02, .021, .022, .004, .005, .006]))
    assert not suggestions.auto_suggested_reject.any()
    assert suggestions.auto_reason.str.contains("disagree").any()


@pytest.mark.parametrize("decision", ["keep", "uncertain"])
def test_manual_choices_cannot_be_suggested_for_rejection(decision):
    data = synthetic([0, .001, .002, .2, .003, .004, .005])
    data.loc[3, "manual_review_decision"] = decision
    _, suggestions = proposals(data)
    assert not suggestions.auto_suggested_reject.any()
    assert suggestions.loc[suggestions.row_index == 3, "auto_reason"].iloc[0] == "existing_manual_decision_is_authoritative"


def test_duplicate_times_and_long_bridges_do_not_gain_confidence():
    for seconds in ([0, 300, 600, 600, 900, 1200, 1500], [0, 300, 600, 900, 3000, 3300, 3600]):
        _, suggestions = proposals(synthetic([0, .001, .002, .2, .003, .004, .005], seconds))
        assert not suggestions.auto_suggested_reject.any()


def test_native_missing_fix_is_not_skipped_but_surviving_adjacency_is_supported():
    data = synthetic([0, .001, .002, .2, np.nan, .003, .004, .005])
    _, native = proposals(data)
    assert not native.auto_suggested_reject.any()
    _, surviving = proposals(data, "surviving")
    assert surviving.loc[surviving.auto_suggested_reject, "row_index"].tolist() == [3]
    assert not (surviving.row_index == 4).any()


@pytest.fixture
def recovery_project(mat_factory, config_factory, tmp_path, request):
    lon = getattr(request, "param", [0, .001, .002, .2, .003, .2, .004, .005, .006])
    n = len(lon)
    changes = {"time": 719529 + np.arange(n) * 300 / 86400,
               "longitude": lon, "latitude": np.zeros(n),
               **{name: np.ones(n) for name in ("SST", "SLP", "battery", "drogue", "speed")}}
    source, _ = mat_factory(native_changes=changes, no_interp=True)
    preprocess(config_factory())
    return (tmp_path / "out/master.zarr", tmp_path / "review/decisions.csv",
            tmp_path / "review/queues", tmp_path / "processed/reviewed.parquet", source)


def test_suggest_cli_writes_audited_proposals_not_decisions_or_parquet(recovery_project):
    master, review, directory, output, source = recovery_project
    protected = [source, master.parent / "inventory.parquet", *[p for p in master.rglob("*") if p.is_file()]]
    before = {p: sha256(p.read_bytes()).hexdigest() for p in protected}
    assert main(["suggest", "--master", str(master), "--review-file", str(review),
                 "--queue-directory", str(directory), "--output", str(output)]) == 0
    assert not review.exists() and not output.exists()
    state, queue, edges = load_queue(directory)
    saved = load_suggestions(state)
    assert saved.loc[saved.auto_suggested_reject, "row_index"].tolist() == [3, 5]
    assert state["high_confidence_suggestion_points"] == 2
    assert len(state["recovery_code_sha256"]) == 64
    assert state["recovery_configuration"]["max_skip_points"] == 3
    assert queue.auto_high_confidence_points.tolist() == [2]
    assert len(edges) == 4
    assert all(sha256(p.read_bytes()).hexdigest() == checksum for p, checksum in before.items())
    # Applying still does not accept proposals on its own.
    build_iteration(master, review, directory, output, export=True)
    product = pd.read_parquet(output)
    assert not product.manual_reject.any() and product.reviewed_position_valid.all()
    original = output.read_bytes()
    build_iteration(master, review, directory, output, export=False)
    assert output.read_bytes() == original


def test_tampered_suggestions_fail_and_legacy_queue_remains_readable(recovery_project):
    master, review, directory, output, _ = recovery_project
    state = build_iteration(master, review, directory, output, export=False)
    path = Path(state["iteration_directory"]) / "recovery_suggestions.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="suggestions changed"):
        load_suggestions(state)
    legacy = dict(state)
    del legacy["suggestions_sha256"]
    assert load_suggestions(legacy).empty


def test_gui_bulk_requires_user_action_and_allows_overrides_and_iteration(recovery_project):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from drifterlab.experiments.arcterx.position_reviewer import PositionReviewer

    master, review, directory, output, _ = recovery_project
    build_iteration(master, review, directory, output, export=False)
    state, queue, edges = load_queue(directory)
    ui = PositionReviewer(master, review, directory, state, queue, edges)
    assert not review.exists()
    ui.selected = 3
    ui.draw()
    assert "HIGH" in ui.recovery_info.get_text()
    assert "continuation" in ui.recovery_info.get_text()
    ui.figure.savefig(directory / "recursive_recovery_smoke.png")
    ui.action("a")
    saved = Decisions(review).table()
    assert len(saved) == 2 and saved.decision.eq("reject").all()
    assert saved.reason.str.startswith("user_accepted_high_confidence_recovery").all()
    resolved, _ = candidates(apply_decisions(ui.native, Decisions(review)), 2, adjacency="surviving")
    assert resolved.empty
    ui.action("a")  # Repeating bulk acceptance must not create duplicates.
    assert len(Decisions(review).rows) == 2
    ui.action("k")  # Explicitly reverse one accepted suggestion.
    assert Decisions(review).table().decision.eq("keep").sum() == 1
    ui.action("a")
    assert Decisions(review).table().decision.eq("keep").sum() == 1
    ui.on_close(None)
    ui.plt.close(ui.figure)
    applied = build_iteration(master, review, directory, output, export=True)
    product = pd.read_parquet(output)
    assert product.manual_reject.sum() == 1
    assert product.reviewed_position_valid.iloc[3]
    assert applied["flagged_edges"] > 0
    next_suggestions = load_suggestions(applied)
    assert not next_suggestions.loc[next_suggestions.row_index == 3, "auto_suggested_reject"].any()


def test_gui_manual_keep_is_not_bulk_accepted(recovery_project):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from drifterlab.experiments.arcterx.position_reviewer import PositionReviewer

    master, review, directory, output, _ = recovery_project
    build_iteration(master, review, directory, output, export=False)
    state, queue, edges = load_queue(directory)
    ui = PositionReviewer(master, review, directory, state, queue, edges)
    ui.selected = 3
    ui.draw()
    ui.action("k")
    ui.action("a")
    assert Decisions(review).table().decision.tolist() == ["keep"]
    ui.on_close(None)
    ui.plt.close(ui.figure)


@pytest.mark.parametrize("recovery_project", [
    [0, .001, .002, .2, .003, .004],  # Medium: forward bridge/continuation, no backward seed.
    [.2, 0, .001, .002, .003],       # Unresolved: beginning lacks a trusted anchor.
], indirect=True)
def test_gui_medium_and_unresolved_cannot_be_bulk_accepted(recovery_project):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from drifterlab.experiments.arcterx.position_reviewer import PositionReviewer

    master, review, directory, output, _ = recovery_project
    build_iteration(master, review, directory, output, export=False)
    state, queue, edges = load_queue(directory)
    ui = PositionReviewer(master, review, directory, state, queue, edges)
    assert not ui.suggestions.empty and not ui.suggestions.auto_suggested_reject.any()
    ui.action("a")
    assert not Decisions(review).rows
    assert "Accepted 0" in ui.status.get_text()
    ui.on_close(None)
    ui.plt.close(ui.figure)
