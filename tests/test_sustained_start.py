from hashlib import sha256

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.diagnostics.sustained_start import (
    Settings, ValidTrack, gap_distribution, populations, read_native_tracks,
    run, segment_track, select_start,
)


BASE = np.datetime64("2025-02-06T00:00:00", "ns")


def track(minutes, platform="001"):
    times = BASE + np.asarray(np.asarray(minutes) * 60 * 1e9, dtype="timedelta64[ns]")
    return ValidTrack(platform, times, np.arange(len(times)))


def test_isolated_and_multiple_fragments_are_skipped_without_fallback():
    data = track([0, 120, 125, *range(300, 361, 5)])
    segments = segment_track(data, Settings())
    assert segments.n_valid_positions.tolist() == [1, 2, 13]
    result = select_start(data.platform_code, segments, Settings())
    assert result["inferred_start_time"] == pd.Timestamp("2025-02-06T05:00:00Z")
    assert result["n_segments_before_inferred_start"] == 2
    assert result["n_valid_positions_before_inferred_start"] == 3
    assert result["start_status"] == "multiple_early_segments"
    assert result["delta_first_valid_to_inferred_hours"] == 5
    single = segment_track(track([0, *range(120, 181, 5)]), Settings())
    assert select_start("001", single, Settings())["start_status"] == "earlier_isolated_fragment_removed"
    short = select_start("001", segments.iloc[:2], Settings())
    assert pd.isna(short["inferred_start_time"])
    assert short["start_status"] == "no_sustained_segment_found"
    assert short["n_segments_before_inferred_start"] is None


def test_chronological_sort_is_stable_and_duplicate_support_is_explicit():
    data = track([60, 0, 0, 5, 10])
    segments = segment_track(data, Settings())
    assert segments.start_source_obs_index.tolist() == [1, 0]
    assert segments.n_valid_positions.tolist() == [4, 1]
    assert segments.n_distinct_times.tolist() == [3, 1]
    assert segments.median_dt_seconds.iloc[0] == 300
    # Thirteen rows spanning one hour do not imply thirteen distinct fixes.
    duplicates = segment_track(track([0] * 10 + [20, 40, 60]), Settings())
    assert select_start("001", duplicates, Settings())["start_status"] == "no_sustained_segment_found"


def test_both_duration_and_observation_count_are_required():
    dense_short = segment_track(track(range(13)), Settings())
    sparse_long = segment_track(track(range(0, 121, 20)), Settings())
    assert pd.isna(select_start("001", dense_short, Settings())["inferred_start_time"])
    assert pd.isna(select_start("001", sparse_long, Settings())["inferred_start_time"])
    regular = segment_track(track(range(0, 61, 5)), Settings())
    assert select_start("001", regular, Settings())["inferred_start_segment_n_obs"] == 13


def test_gap_and_duration_tolerance_preserve_timestamp_precision():
    data = track([0, 30, 60])
    data.time[1] += np.timedelta64(5000, "ns")
    data.time[2] += np.timedelta64(2, "s")
    parts = segment_track(data, Settings())
    assert parts.n_valid_positions.tolist() == [2, 1]
    assert parts.segment_end.iloc[0].nanosecond == 0
    assert parts.segment_end.iloc[0].microsecond == 5
    regular = track(range(0, 61, 5))
    regular.time[0] += np.timedelta64(1234, "ns")
    chosen = select_start("001", segment_track(regular, Settings()), Settings())
    assert chosen["inferred_start_time"].nanosecond == 234
    assert chosen["inferred_start_segment_duration_hours"] < 1
    assert chosen["start_status"] == "same_or_nearly_same"
    distribution = gap_distribution([data])
    thirty = next(row for row in distribution["exceedances"] if row["threshold_minutes"] == 30)
    assert thirty["strict_count"] == 2 and thirty["tolerance_adjusted_count"] == 1


def test_empty_track_and_invalid_settings():
    data = track([])
    segments = segment_track(data, Settings())
    assert segments.empty
    assert pd.isna(select_start("001", segments, Settings())["inferred_start_time"])
    assert gap_distribution([data])["quantiles_seconds"]["median"] is None
    for values in ({"continuity_gap_threshold_minutes": 0}, {"minimum_segment_n_obs": 1},
                   {"minimum_segment_n_obs": 2.5}, {"comparison_tolerance_seconds": -1}):
        with pytest.raises(ValueError):
            Settings(**values)


def write_inputs(tmp_path):
    # Unsorted native rows, a missing-position early date, NaT, padding and a
    # real valid isolated fix. Drogue/analysis validity is deliberately all false.
    time = np.r_[BASE + np.arange(120, 181, 5).astype("timedelta64[m]"),
                 BASE - np.timedelta64(1, "D"), BASE, np.datetime64("NaT", "ns"),
                 BASE - np.timedelta64(12, "h")]
    time[0] += np.timedelta64(1234, "ns")
    time = np.r_[time, BASE + np.timedelta64(30, "D")][None, :]
    lon, lat = np.full(time.shape, 130.), np.full(time.shape, 20.)
    lon[0, 13] = np.nan
    lat[0, 16] = 91
    position = np.isfinite(lon) & (lat <= 90)
    master = tmp_path / "master.zarr"
    ds = xr.Dataset({
        "platform_code": ("trajectory", ["001"]), "source_sha256": ("trajectory", ["source-hash"]),
        "n_obs_qc": ("trajectory", [17]), "time_qc": (("trajectory", "obs_qc"), time),
        "lon_qc": (("trajectory", "obs_qc"), lon), "lat_qc": (("trajectory", "obs_qc"), lat),
        "position_valid_qc": (("trajectory", "obs_qc"), position),
        "analysis_valid_qc": (("trajectory", "obs_qc"), np.zeros_like(position)),
        "source_obs_index_qc": (("trajectory", "obs_qc"), np.arange(time.size)[None, :]),
    })
    encoding = {col: {"_FillValue": None} for col in ds if col != "time_qc"}
    ds.to_zarr(master, zarr_format=2, consolidated=True, encoding=encoding)
    timing = pd.DataFrame({
        "platform_code": ["001"], "raw_first_time": pd.to_datetime(["2025-02-05T00:00:00Z"]),
        "qc_first_time": pd.to_datetime(["2025-02-05T00:00:00Z"]),
        "qc_first_valid_position_time": pd.to_datetime(["2025-02-06T00:00:00Z"]),
        "qc_source_sha256": ["source-hash"], "match_status": ["matched"],
        "first_valid_qc_time_cohort": ["2025-02-06"], "drifter_index": [1],
    })
    table = tmp_path / "timing.parquet"
    timing.to_parquet(table, index=False)
    return master, table, timing


def test_native_filtering_uses_positions_and_time_not_drogue_or_padding(tmp_path):
    master, _, timing = write_inputs(tmp_path)
    loaded = read_native_tracks(master, timing)
    assert len(loaded[0].time) == 14
    assert loaded[0].source_index[0] == 14
    assert loaded[0].time[0] == BASE
    assert loaded[0].time[-1] == BASE + np.timedelta64(180, "m")
    timing.loc[0, "qc_first_valid_position_time"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="disagrees"):
        read_native_tracks(master, timing)


def test_run_round_trip_sensitivity_and_input_integrity(tmp_path):
    master, table, timing = write_inputs(tmp_path)
    before = {p: sha256(p.read_bytes()).hexdigest() for p in [table, *master.rglob("*")] if p.is_file()}
    output = tmp_path / "diagnostic"
    summary = run(master, table, output, figures=False)
    assert summary["counts"]["one_or_more_early_fragments_skipped"] == 1
    assert summary["duration_sensitivity"][-1]["n_without_sustained_start"] == 1
    assert summary["manual_review_platforms"] == ["001"]
    result = pd.read_parquet(output / "microsvp_sustained_start_diagnostic.parquet")
    pd.testing.assert_frame_equal(result[timing.columns], timing)
    assert result.platform_code.iloc[0] == "001"
    assert result.inferred_start_time.iloc[0].nanosecond == 234
    assert result.inferred_start_source_obs_index.iloc[0] == 0
    csv = pd.read_csv(output / "microsvp_sustained_start_diagnostic.csv", dtype={"platform_code": str})
    assert pd.Timestamp(csv.inferred_start_time.iloc[0]) == result.inferred_start_time.iloc[0]
    assert csv.platform_code.iloc[0] == "001"
    assert all(sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    assert populations(result.inferred_start_time)[0]["n"] == 1
    with pytest.raises(ValueError, match="outside the master"):
        run(master, table, master / "diagnostic", figures=False)


def test_run_with_no_qualifying_starts_records_nulls(tmp_path):
    master, table, _ = write_inputs(tmp_path)
    output = tmp_path / "no_sustained"
    summary = run(master, table, output, settings=Settings(minimum_segment_duration_minutes=120,
                                                         minimum_segment_n_obs=25), figures=False)
    assert summary["counts"]["no_sustained_segment_found"] == 1
    result = pd.read_parquet(output / "microsvp_sustained_start_diagnostic.parquet")
    assert result.inferred_start_time.isna().all()
    assert result.n_segments_before_inferred_start.isna().all()
    assert result.manual_review_recommended.all()
    assert not result.start_nearly_same_as_first_valid.any()
    assert "no_sustained_segment_found" in result.manual_review_reason.iloc[0]
