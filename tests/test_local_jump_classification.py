from hashlib import sha256

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from drifterlab.experiments.arcterx import preprocess
from drifterlab.qc.flags import consecutive_speed
from drifterlab.qc.position import position_valid
from scripts.diagnostics.residual_jumps import Track, flagged_pairs, load_track
from scripts.diagnostics.local_jump_classification import (
    CLASSES, classify_track, precision_sensitivity, run, turning_angle,
)


def make_track(lon, seconds=None):
    lon = np.asarray(lon, dtype=float)
    time = np.datetime64("2025-02-06T00:00:00", "ns") + np.asarray(
        np.arange(len(lon)) * 300 if seconds is None else seconds, dtype="timedelta64[s]")
    lat = np.zeros(len(lon))
    speed = consecutive_speed(time, lon, lat)
    return Track("0001", "source.mat", "hash", time, lon, lat, position_valid(lon, lat),
                 np.arange(len(lon))[::-1], speed, speed > 3, np.ones(len(lon)))


def test_isolated_middle_point_uses_actual_bridge_time_and_preserves_edges():
    track = make_track([0, .001, .02, .002, .003], [0, 300, 420, 900, 1200])
    edges, runs, points = classify_track(track)
    assert len(edges) == 2 and len(runs) == len(points) == 1
    assert runs.classification.iloc[0] == CLASSES[0]
    assert runs.suspected_point_index.iloc[0] == 2
    assert points.source_obs_index_B.iloc[0] == 2
    assert points.dt_seconds_AB.iloc[0] == 120
    assert points.dt_seconds_BC.iloc[0] == 480
    assert points.dt_seconds_AC.iloc[0] == 600
    assert points.speed_m_s_AC.iloc[0] <= 3
    assert points.turning_angle_at_B_degrees.iloc[0] == pytest.approx(180)
    assert edges.explained_by_isolated_run_candidate.all()
    assert edges.row_index_2.tolist() == [2, 3]
    pd.testing.assert_frame_equal(edges[flagged_pairs(track)[0].columns], flagged_pairs(track)[0])


def test_overlapping_local_candidates_do_not_resolve_longer_run():
    edges, runs, points = classify_track(make_track([0, .02, 0, .02]))
    assert runs.n_flagged_edges.iloc[0] == 3
    assert runs.classification.iloc[0] == CLASSES[2]
    assert pd.isna(runs.suspected_point_index.iloc[0])
    assert points.passes_local_skip_one_test.tolist() == [True, True]
    assert not points.isolated_run_candidate.any()
    assert edges.has_local_skip_one_support.all()
    assert edges.n_local_candidates_supporting_edge.tolist() == [1, 2, 1]
    assert not edges.explained_by_isolated_run_candidate.any()


def test_persistent_forward_run_has_no_reasonable_skip_one_bridge():
    track = make_track([0, .02, .04])
    _, runs, points = classify_track(track)
    assert runs.classification.iloc[0] == CLASSES[2]
    assert not points.passes_local_skip_one_test.any()
    assert points.speed_m_s_AC.iloc[0] > 3
    assert turning_angle(track, 0, 1, 2) == pytest.approx(0, abs=1e-5)


def test_single_edge_and_insufficient_context_do_not_guess_endpoint():
    _, runs, points = classify_track(make_track([0, .0001, .02, .0201]))
    assert runs.classification.iloc[0] == CLASSES[1]
    assert pd.isna(runs.suspected_point_index.iloc[0]) and points.empty
    _, runs, _ = classify_track(make_track([0, .02, .0201]))
    assert runs.classification.iloc[0] == CLASSES[3]
    _, runs, _ = classify_track(make_track([0, .0001, .02, .0201], [0, 300, 600, 3000]))
    assert runs.classification.iloc[0] == CLASSES[3]


def test_masked_rows_separate_runs_and_context_crossings_are_explicit():
    track = make_track([0, .02, np.nan, 0, .02, 0, .0001])
    edges, runs, _ = classify_track(track)
    assert runs.n_flagged_edges.tolist() == [1, 2]
    assert edges.row_index_2.tolist() == [1, 4, 5]
    assert runs.classification.iloc[1] == CLASSES[0]
    assert runs.n_rows_crossed_before_context.iloc[1] == 1
    assert runs.context_before_index.iloc[1] == 1
    assert edges.n_rows_between.eq(0).all()


def test_long_bridge_fails_local_gap_requirement():
    track = make_track([0, 1, 0], [0, 1200, 2400])
    _, runs, points = classify_track(track)
    assert points.speed_m_s_AC.iloc[0] == 0
    assert not points.triplet_context_usable.iloc[0]
    assert runs.classification.iloc[0] == CLASSES[3]


def test_precision_experiments_preserve_clear_case_and_detect_threshold_case():
    track = make_track([0, .001, .02, .002, .003])
    track.time[2] += np.timedelta64(4500, "ns")
    track.speed = consecutive_speed(track.time, track.lon, track.lat)
    track.flags = track.speed > 3
    results = precision_sensitivity(track, classify_track(track))
    assert all(row["changed_flags"] == row["changed_edge_classes"] == row["changed_candidate_points"] == 0 for row in results)
    assert results[0]["max_clock_shift_microseconds"] == pytest.approx(4.5)
    assert results[1]["max_clock_shift_microseconds"] == pytest.approx(10)
    close = make_track([0, .0081])
    distance = close.speed[1] * 300
    close.time[1] = close.time[0] + np.timedelta64(int(round(distance / 3 * 1e9)), "ns")
    close.speed = consecutive_speed(close.time, close.lon, close.lat)
    close.flags = close.speed > 3
    results = precision_sensitivity(close, classify_track(close))
    assert any(row["changed_flags"] for row in results[1:])


def test_run_outputs_preserve_master_and_previous_diagnostic(mat_factory, config_factory, tmp_path):
    n = 5
    changes = {"time": 719529 + np.arange(n) * 300 / 86400,
               "longitude": [0, .001, .02, .002, .003], "latitude": np.zeros(n),
               **{key: np.ones(n) for key in ["SST", "SLP", "battery", "drogue", "speed"]}}
    source, _ = mat_factory(native_changes=changes, no_interp=True)
    preprocess(config_factory())
    master = tmp_path / "out/master.zarr"
    previous = tmp_path / "previous.parquet"
    with xr.open_zarr(master, chunks=None) as ds:
        flagged_pairs(load_track(ds, 0))[0].to_parquet(previous, index=False)
    protected = [source, previous, tmp_path / "out/inventory.parquet", *[p for p in master.rglob("*") if p.is_file()]]
    before = {p: sha256(p.read_bytes()).hexdigest() for p in protected}
    output = tmp_path / "diagnostic"
    summary = run(master, output, flag_table=previous, figures=False)
    assert summary["total_flagged_edges"] == 2
    assert summary["isolated_spike_candidate_points"] == 1
    assert summary["isolated_run_edge_fraction"] == 1
    result = pd.read_parquet(output / "classified_flagged_edges.parquet")
    original = pd.read_parquet(previous)
    pd.testing.assert_frame_equal(result[original.columns], original)
    csv = pd.read_csv(output / "classified_flagged_edges.csv", dtype={"platform_code": str})
    np.testing.assert_array_equal(pd.to_datetime(csv.time_2, utc=True, format="ISO8601").astype("datetime64[ns, UTC]").array.asi8,
                                  result.time_2.array.asi8)
    assert all(sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    with pytest.raises(ValueError, match="outside"):
        run(master, master / "diagnostics", figures=False)


def test_no_flags_produces_empty_tables(mat_factory, config_factory, tmp_path):
    mat_factory(no_interp=True)
    preprocess(config_factory())
    summary = run(tmp_path / "out/master.zarr", tmp_path / "diagnostic", figures=False)
    assert summary["total_flagged_edges"] == summary["total_flagged_runs"] == 0
    assert summary["isolated_run_edge_fraction"] == 0
