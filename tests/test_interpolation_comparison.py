from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from drifterlab.experiments.arcterx import preprocess
from drifterlab.qc.position import position_valid
from scripts.diagnostics.interpolation_comparison import (
    ReconstructionTrack,
    compare_native_edges,
    match_stored_time,
    run,
    scan_reconstructed_flags,
)
from scripts.diagnostics.local_jump_classification import run as classify_local_jumps


def reconstruction(lon30, lon60=None, *, microseconds=None, valid30=None, valid60=None):
    lon30 = np.asarray(lon30, dtype=float)
    lon60 = np.asarray(lon30 if lon60 is None else lon60, dtype=float)
    offsets = np.arange(len(lon30)) * 300_000_000
    if microseconds is not None:
        offsets += np.asarray(microseconds, dtype=np.int64)
    time = np.datetime64("2025-02-06T00:00:00", "ns") + offsets.astype("timedelta64[us]")
    lat = np.zeros(len(lon30))
    return ReconstructionTrack(
        "0001", time, np.arange(len(time)), lon30, lat,
        position_valid(lon30, lat) if valid30 is None else np.asarray(valid30, bool),
        lon60, lat, position_valid(lon60, lat) if valid60 is None else np.asarray(valid60, bool),
    )


def native_edge(time_1, time_2, *, row_1=0, row_2=1):
    return pd.DataFrame([{
        "platform_code": "0001", "time_1": pd.Timestamp(time_1, tz="UTC"),
        "time_2": pd.Timestamp(time_2, tz="UTC"), "dt_minutes": 5.0,
        "distance_m": 1200.0, "speed_m_s": 4.0,
        "row_index_1": row_1, "row_index_2": row_2,
        "run_id": "0001:r0001", "classification": "single_flagged_edge",
        "explained_by_isolated_run_candidate": False,
    }])


def test_timestamp_match_uses_stored_nearby_sample_and_rejects_duplicates():
    track = reconstruction([0, .001, .002], microseconds=[0, 5, 0])
    target = np.datetime64("2025-02-06T00:05:00", "ns")
    match = match_stored_time(track.time, target)
    assert match["status"] == "matched" and match["index"] == 1
    assert match["offset_microseconds"] == pytest.approx(5)
    assert not match["exact"]
    duplicate = np.array([target, target], dtype="datetime64[ns]")
    assert match_stored_time(duplicate, target)["status"] == "ambiguous_duplicate_or_equidistant_timestamp"


def test_native_interval_uses_actual_reconstruction_time_and_classifies_products():
    track = reconstruction([0, .001], [0, .02], microseconds=[-4, 6])
    start = np.datetime64("2025-02-06T00:00:00", "ns")
    end = np.datetime64("2025-02-06T00:05:00", "ns")
    result = compare_native_edges(native_edge(start, end), track).iloc[0]
    assert result.reconstruction_behavior == "resolved_in_30_only"
    assert result.interp30_speed_m_s <= 3 < result.interp60_speed_m_s
    assert result.interp30_dt_minutes == pytest.approx((300.000010) / 60)
    assert result.interp30_time_1_offset_microseconds == pytest.approx(-4)
    assert result.interp30_time_2_offset_microseconds == pytest.approx(6)


def test_missing_invalid_reconstructed_endpoint_is_explicit():
    track = reconstruction([0, .001], [0, np.nan])
    start, end = track.time
    result = compare_native_edges(native_edge(start, end), track).iloc[0]
    assert result.reconstruction_behavior == "missing_reconstructed_context"
    assert result.interp30_available
    assert not result.interp60_available
    assert result.interp60_unavailable_reason == "one_or_both_endpoint_positions_invalid"
    assert np.isnan(result.interp60_speed_m_s)


def test_reconstruction_scan_uses_adjacent_rows_and_separates_new_events():
    track = reconstruction([0, .02, .0201, .0401])
    native = native_edge(track.time[0], track.time[1])
    flags = scan_reconstructed_flags(track, "interp30", native)
    assert flags.row_index_2.tolist() == [1, 3]
    assert flags.associated_with_native_flag.tolist() == [True, False]
    assert flags.reconstruction_only_event.tolist() == [False, True]
    # A masked row blocks both neighboring pairs; no valid-position bridge is made.
    masked = reconstruction([0, np.nan, .04])
    assert scan_reconstructed_flags(masked, "interp30", native).empty


def test_focused_synthetic_run_writes_outputs_without_modifying_master(mat_factory, config_factory, tmp_path):
    n = 5
    matlab_time = 719529 + np.arange(n) * 300 / 86400
    native = {
        "time": matlab_time, "longitude": np.array([0, .001, .02, .002, .003]),
        "latitude": np.zeros(n), "SST": np.ones(n), "SLP": np.ones(n),
        "battery": np.ones(n), "drogue": np.ones(n), "speed": np.ones(n),
    }
    interp = {
        "time": matlab_time, "longitude_30min": np.array([0, .001, .002, .003, .004]),
        "latitude_30min": np.zeros(n), "longitude_60min": np.array([0, .02, 0, .02, 0]),
        "latitude_60min": np.zeros(n), "SST": np.ones(n),
        "speed_30min": np.ones(n), "speed_60min": np.ones(n),
    }
    mat_factory(native_changes=native, interp_changes=interp)
    preprocess(config_factory())
    master = tmp_path / "out/master.zarr"
    local = tmp_path / "local"
    classify_local_jumps(master, local, figures=False)
    protected = [p for p in master.rglob("*") if p.is_file()]
    before = {p: sha256(p.read_bytes()).hexdigest() for p in protected}
    output = tmp_path / "comparison"
    summary = run(master, local / "classified_flagged_edges.parquet", output,
                  platform_codes=["1001"], figures=False)
    assert summary["native_flagged_edges"] == 2
    assert summary["behavior_counts"]["resolved_in_30_only"] == 2
    assert summary["reconstructed_track_scans"]["interp60"]["reconstruction_only_events"] == 2
    assert summary["scope"]["complete_master_scope"]  # one selected of one is complete
    assert (output / "jump_interpolation_comparison.csv").exists()
    assert (output / "reconstruction_only_flags.csv").exists()
    assert (output / "report.md").exists()
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())
