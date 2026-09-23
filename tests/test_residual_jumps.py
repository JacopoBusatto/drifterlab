from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from drifterlab.experiments.arcterx import preprocess
from drifterlab.qc.flags import consecutive_speed
from drifterlab.qc.position import position_valid
from scripts.diagnostics.residual_jumps import Track, duration_counts, flagged_pairs, run, summarize


def make_track(seconds, lon, lat=None):
    time = np.datetime64("2025-02-06T00:00:00", "ns") + np.asarray(seconds, dtype="timedelta64[s]")
    lon = np.asarray(lon, dtype=float)
    lat = np.zeros(len(time)) if lat is None else np.asarray(lat, dtype=float)
    speed = consecutive_speed(time, lon, lat)
    return Track("0001", "original.mat", "source-hash", time, lon, lat, position_valid(lon, lat),
                 np.arange(len(time))[::-1], speed, speed > 3, np.ones(len(time)))


def test_pairs_do_not_bridge_missing_and_preserve_original_row_indices():
    track = make_track([0, 300, 600, 660, 3000], [0, np.nan, 1, 1.01, 1.08])
    frame, stats = flagged_pairs(track)
    assert frame.row_index_1.tolist() == [2, 3]
    assert frame.row_index_2.tolist() == [3, 4]
    assert frame.source_obs_index_1.tolist() == [2, 1]
    assert frame.source_obs_index_2.tolist() == [1, 0]
    assert frame.n_rows_between.eq(0).all()
    assert frame.n_missing_positions_between.eq(0).all()
    assert frame.n_invalid_positions_between.eq(0).all()
    assert frame.position_1_valid.all() and frame.position_2_valid.all()
    assert frame.invalid_position_immediately_before_pair.tolist() == [True, False]
    assert frame.dt_minutes.tolist() == [1, 39]
    np.testing.assert_allclose(frame.distance_m, frame.speed_m_s * frame.dt_minutes * 60)
    assert stats["n_pairs_touching_invalid_position_or_time"] == 2
    assert stats["n_finite_speeds_on_blocked_pairs"] == 0


def test_invalid_coordinate_and_duplicate_time_are_not_speed_pairs():
    track = make_track([0, 300, 300, 600], [0, 181, 1, 1.02])
    frame, stats = flagged_pairs(track)
    assert frame.row_index_2.tolist() == [3]
    assert stats["n_finite_speeds_on_blocked_pairs"] == 0
    duplicate = make_track([0, 0, 300], [0, 1, 1.02])
    frame, _ = flagged_pairs(duplicate)
    assert frame.row_index_2.tolist() == [2]


def test_mismatch_is_an_error_instead_of_redefining_flags():
    track = make_track([0, 300], [0, .02])
    track.flags[1] = False
    with pytest.raises(ValueError, match="residual flags"):
        flagged_pairs(track)
    track = make_track([0, 300], [0, .02])
    track.speed[1] *= 2
    with pytest.raises(ValueError, match="audit speeds"):
        flagged_pairs(track)


def test_cadence_tolerance_is_only_a_reporting_choice():
    minutes = pd.Series([6 + 5e-6 / 60, 10 + 5e-6 / 60, 10 + 2 / 60, 31])
    assert duration_counts(minutes, 0) == {"dt_le_6_min": 0, "dt_le_10_min": 1, "dt_gt_10_min": 3, "dt_gt_30_min": 1}
    assert duration_counts(minutes, 1) == {"dt_le_6_min": 1, "dt_le_10_min": 2, "dt_gt_10_min": 2, "dt_gt_30_min": 1}


def test_empty_flag_table_has_a_valid_summary():
    frame, stats = flagged_pairs(make_track([0, 300], [0, .0001]))
    summary = summarize(frame, pd.DataFrame([stats]))
    assert summary["total_flagged_jumps"] == 0
    assert summary["speed_quantiles_m_s"]["max"] is None
    assert summary["affected_drifters"] == 0


def test_complete_diagnostic_preserves_inputs_and_verifies_source(mat_factory, config_factory, tmp_path):
    seconds = np.array([0, 300, 600, 660, 3000])
    changes = {"time": (719529 + seconds / 86400)[::-1],
               "longitude": np.array([0, -999, 1, 1.01, 1.08])[::-1], "latitude": np.zeros(5),
               **{name: np.ones(5) for name in ["SST", "SLP", "battery", "drogue", "speed"]}}
    source, _ = mat_factory(native_changes=changes, no_interp=True)
    preprocess(config_factory())
    master = tmp_path / "out/master.zarr"
    protected = [source, tmp_path / "out/inventory.parquet", *[p for p in master.rglob("*") if p.is_file()]]
    before = {p: sha256(p.read_bytes()).hexdigest() for p in protected}
    output = tmp_path / "jump_diagnostic"
    summary = run(master, output, qc_directory=source.parent, figures=False)
    assert summary["total_flagged_jumps"] == 2
    assert summary["source_verification"]["status"] == "passed"
    assert summary["duration_counts_with_1s_comparison_tolerance"]["dt_gt_30_min"] == 1
    assert summary["finite_speeds_on_blocked_pairs"] == 0
    result = pd.read_parquet(output / "flagged_jumps.parquet")
    assert result.platform_code.tolist() == ["1001", "1001"]
    assert result.source_obs_index_2.tolist() == [1, 0]
    assert result.position_1_valid.all()
    csv = pd.read_csv(output / "flagged_jumps.csv", dtype={"platform_code": str})
    np.testing.assert_array_equal(pd.to_datetime(csv.time_2, utc=True, format="ISO8601").astype("datetime64[ns, UTC]").array.asi8,
                                  result.time_2.array.asi8)
    assert all(sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    assert (output / "report.md").exists()
    with pytest.raises(ValueError, match="outside"):
        run(master, master / "diagnostics", figures=False)
