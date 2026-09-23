"""Focused synthetic coverage of review identity, iteration and UI controls."""

from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from drifterlab.experiments.arcterx import preprocess
from drifterlab.experiments.arcterx.position_review import (
    Decisions, apply_decisions, build_iteration, candidates, edge_arrays, load_queue,
    native_frame, validate_paths,
)
from drifterlab.qc.position import position_valid


def frame(lon, seconds=None, platform="0001"):
    lon = np.array(lon, dtype=float)
    n = len(lon)
    times = np.datetime64("2025-02-06T00:00:00.000004471", "ns") + np.array(
        np.arange(n) * 300 if seconds is None else seconds, dtype="timedelta64[s]")
    return pd.DataFrame({"platform_code": platform, "time_qc": times,
                         "time": pd.to_datetime(times, utc=True), "lon_qc": lon, "lat_qc": np.zeros(n),
                         "position_valid": position_valid(lon, np.zeros(n)),
                         "source_obs_index": np.arange(n)[::-1], "source_sha256": "test-source-hash",
                         "row_index": np.arange(n)})


def decide(decisions, data, row, decision, iteration=1, note=""):
    decisions.set(data.iloc[row], decision, iteration=iteration, reason="manual local geometry",
                  candidate_id=f"i{iteration}:test", note=note)


def test_roundtrip_latest_decision_precision_and_unchanged_rows(tmp_path):
    data = frame([0, .02, .001])
    path = tmp_path / "review.csv"
    decisions = Decisions(path)
    decide(decisions, data, 1, "reject", note="out and back")
    decisions.save()
    loaded = Decisions(path)
    assert list(loaded.rows)[0] == ("0001", 1)
    assert loaded.table().time.iloc[0].endswith("000004471+00:00")
    before = loaded.table().copy()
    decide(loaded, data, 1, "reject", note="out and back")
    pd.testing.assert_frame_equal(before, loaded.table())
    decide(loaded, data, 1, "keep", iteration=2)
    loaded.save()
    latest = Decisions(path).table()
    assert len(latest) == 1 and latest.decision.iloc[0] == "keep"
    assert latest.review_iteration.iloc[0] == 2


def test_uncertain_validity_invalid_keep_and_source_identity(tmp_path):
    data = frame([0, .02, np.nan, .001])
    original = data.copy(deep=True)
    decisions = Decisions(tmp_path / "review.csv")
    decide(decisions, data, 0, "reject")
    decide(decisions, data, 1, "uncertain")
    decide(decisions, data, 2, "keep")
    result = apply_decisions(data, decisions)
    assert result.reviewed_position_valid.tolist() == [False, True, False, True]
    assert result.manual_uncertain.tolist() == [False, True, False, False]
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_series_equal(result.position_valid, data.position_valid)
    altered = data.copy()
    altered.loc[0, "time"] += pd.Timedelta(1, unit="ns")
    with pytest.raises(ValueError, match="identity/time"):
        apply_decisions(altered, decisions)
    altered = data.copy()
    altered["source_sha256"] = "another-source"
    with pytest.raises(ValueError, match="identity/time"):
        apply_decisions(altered, decisions)
    altered = data.copy()
    altered.loc[0, "source_obs_index"] = 99
    with pytest.raises(ValueError, match="Unknown source"):
        apply_decisions(altered, decisions)


def test_duplicate_invalid_and_concurrent_reviews_fail(tmp_path):
    path = tmp_path / "review.csv"
    data = frame([0, .02, 0])
    a, b = Decisions(path), Decisions(path)
    decide(a, data, 1, "reject")
    a.save()
    with pytest.raises(ValueError, match="another session"):
        b.save()
    duplicate = pd.concat([a.table(), a.table()])
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        Decisions(path)
    bad = a.table()
    bad.loc[0, "decision"] = "delete"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Invalid"):
        Decisions(path)


@pytest.mark.parametrize("lon,category", [
    ([0, .001, .02, .002, .003], "isolated_spike_candidate"),
    ([0, .02, 0, .02], "persistent_high_speed_run"),
    ([0, .001, .03, .031], "single_flagged_edge"),
    ([0, .03], "boundary_or_insufficient_context"),
])
def test_candidate_run_classification(tmp_path, lon, category):
    data = apply_decisions(frame(lon), Decisions(tmp_path / "reviews.csv"))
    queue, edges = candidates(data, 1, adjacency="native")
    assert len(queue) == 1
    assert queue.classification.iloc[0] == category
    assert len(edges) == int(queue.n_flagged_edges.iloc[0])


def test_neighboring_runs_group_and_context_counts_valid_points(tmp_path):
    data = frame([0, np.nan, .001, .02, .021, .04, .041, np.nan, .042])
    data = apply_decisions(data, Decisions(tmp_path / "review.csv"))
    queue, edges = candidates(data, 1, adjacency="native", context_points=2)
    assert len(queue) == 1 and len(edges) == 2
    assert queue.classification.iloc[0] == "neighboring_excursion_region"
    assert queue.context_start.iloc[0] == 0
    assert queue.context_end.iloc[0] == 8
    split, _ = candidates(data, 1, adjacency="native", merge_gap_points=0)
    assert len(split) == 2


def test_surviving_adjacency_exposes_new_edge_without_automatic_rejection(tmp_path):
    data = frame([0, .001, .03, .031, .032])
    decisions = Decisions(tmp_path / "reviews.csv")
    decide(decisions, data, 2, "reject")
    reviewed = apply_decisions(data, decisions)
    queue, edges = candidates(reviewed, 2, adjacency="surviving")
    assert edges[["row_index_1", "row_index_2"]].values.tolist() == [[1, 3]]
    assert edges.n_rows_between.iloc[0] == 1
    assert reviewed.manual_reject.sum() == 1
    assert reviewed.reviewed_position_valid.iloc[3]
    assert not queue.empty
    spike = frame([0, .001, .03, .002, .003])
    reviewed_spike = apply_decisions(spike, decisions)
    queue, edges = candidates(reviewed_spike, 2, adjacency="surviving")
    assert queue.empty and edges.empty


def test_invalid_rows_and_nonpositive_intervals(tmp_path):
    data = frame([0, .03, np.nan, .06, .09], seconds=[0, 0, 300, 600, 600])
    reviewed = apply_decisions(data, Decisions(tmp_path / "review.csv"))
    native_queue, _ = candidates(reviewed, 1, adjacency="native")
    assert native_queue.empty
    indices, speeds, dt = edge_arrays(reviewed, "surviving")
    assert indices.tolist() == [0, 1, 3, 4]
    assert np.isnan(speeds[[0, 1, 3]]).all()
    assert dt[2] == 600 and speeds[2] > 3


@pytest.fixture
def review_project(mat_factory, config_factory, tmp_path):
    n = 6
    native = {"time": 719529 + np.arange(n)[::-1] * 300 / 86400,
              "longitude": [0.004, 0.003, 0.002, .03, .001, 0], "latitude": np.zeros(n),
              **{key: np.ones(n) for key in ("SST", "SLP", "battery", "drogue", "speed")}}
    references = {"time": 719529 + np.arange(5) * 300 / 86400,
                  "longitude_30min": np.arange(5) * .001,
                  "longitude_60min": np.arange(5) * .0012,
                  "latitude_30min": np.arange(5) * .0002,
                  "latitude_60min": np.arange(5) * .0003}
    source, _ = mat_factory(native_changes=native, interp_changes=references)
    mat_factory("1002")  # Mixed lengths, real padding, different source hashes.
    preprocess(config_factory())
    return (tmp_path / "out/master.zarr", tmp_path / "review/decisions.csv",
            tmp_path / "review/queues", tmp_path / "processed/reviewed.parquet", source)


def test_export_iteration_provenance_and_immutability(review_project):
    master, review, directory, output, source = review_project
    protected = [source, master.parent / "inventory.parquet", *[p for p in master.rglob("*") if p.is_file()]]
    before = {p: sha256(p.read_bytes()).hexdigest() for p in protected}
    first = build_iteration(master, review, directory, output, export=False)
    assert first["flagged_edges"] == 2 and not output.exists()
    with xr.open_zarr(master, chunks=None) as ds:
        data = native_frame(ds, 0)
    decisions = Decisions(review)
    decide(decisions, data, 2, "reject")
    decide(decisions, data, 3, "uncertain")
    decisions.save()
    second = build_iteration(master, review, directory, output, export=True)
    assert second["review_iteration"] == 2 and second["flagged_edges"] == 0
    result = pd.read_parquet(output)
    assert len(result) == 9
    native = result[result.platform_code == "1001"].reset_index(drop=True)
    for name in data.columns:
        pd.testing.assert_series_equal(native[name], data[name])
    assert native.manual_reject.tolist() == [False, False, True, False, False, False]
    assert native.manual_uncertain.iloc[3] and native.reviewed_position_valid.iloc[3]
    assert native.reviewed_audit_speed_m_s.iloc[3] < 3
    assert np.isnan(native.reviewed_audit_speed_m_s.iloc[2])
    assert all(sha256(p.read_bytes()).hexdigest() == value for p, value in before.items())
    assert second["reviewed_output_sha256"] == sha256(output.read_bytes()).hexdigest()
    assert (directory / "iteration_0001/candidates.csv").exists()
    assert len(pd.read_csv(directory / "iteration_0002/applied_decisions.csv")) == 2
    _, queue, edges = load_queue(directory)
    assert queue.empty and edges.empty


def test_invalid_reviews_do_not_replace_export(review_project):
    master, review, directory, output, _ = review_project
    build_iteration(master, review, directory, output, export=True)
    before = output.read_bytes()
    state_before = (directory / "latest.json").read_bytes()
    decisions = Decisions(review)
    decide(decisions, frame([0, .02, 0], platform="nonexistent"), 1, "reject")
    decisions.save()
    with pytest.raises(ValueError, match="unknown platforms"):
        build_iteration(master, review, directory, output, export=True)
    assert output.read_bytes() == before
    assert (directory / "latest.json").read_bytes() == state_before


def test_path_guards_and_queue_tampering(review_project):
    master, review, directory, output, _ = review_project
    with pytest.raises(ValueError, match="outside"):
        validate_paths(master, review, master / "review", output)
    with pytest.raises(ValueError, match="separate"):
        validate_paths(master, review, directory, review)
    build_iteration(master, review, directory, output, export=False)
    path = directory / "iteration_0001/candidates.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="queue changed"):
        load_queue(directory)


def test_gui_headless_decisions_selection_resume_and_references(review_project, monkeypatch):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from drifterlab.experiments.arcterx.position_reviewer import PositionReviewer

    master, review, directory, output, _ = review_project
    build_iteration(master, review, directory, output, export=False)
    state, queue, edges = load_queue(directory)
    ui = PositionReviewer(master, review, directory, state, queue, edges)
    assert ui.selected == 2
    assert len(ui.references) == 2
    reference_lines = [line for line in ui.map_ax.lines if "reference only" in line.get_label()]
    assert len(reference_lines) == 2 and all(len(line.get_xdata()) > 0 for line in reference_lines)
    ui.on_key(SimpleNamespace(key="r"))
    assert Decisions(review).table().decision.tolist() == ["reject"]
    ui.on_key(SimpleNamespace(key="right"))
    assert ui.selected == 3
    ui.note_box.set_val("needs another look")
    ui.action("u")
    ui.action("s")
    ui.figure.savefig(directory / "headless_smoke.png")
    ui.on_close(None)
    ui.plt.close(ui.figure)
    resumed = PositionReviewer(master, review, directory, state, queue, edges)
    assert resumed.selected == 3
    assert resumed.frame.manual_uncertain.iloc[3]
    assert resumed.note_box.text == "needs another look"
    resumed.action("left")
    resumed.action("k")
    assert Decisions(review).table().decision.tolist().count("keep") == 1
    resumed.on_close(None)
    resumed.plt.close(resumed.figure)
