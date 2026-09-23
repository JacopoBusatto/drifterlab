from hashlib import sha256

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from scripts.diagnostics.timing_diagnostic import (
    TimingSeries, assign_cohorts, build_table, catalog, compare_series,
    matching_report, read_raw_timing, run,
)


def series(times, missing=()):
    time = np.asarray(times, dtype="datetime64[ns]")
    lon, lat = np.full(len(time), 130.), np.full(len(time), 20.)
    lon[list(missing)] = np.nan
    return TimingSeries(time, lon, lat, np.arange(len(time)), {})


@pytest.mark.parametrize("raw_times,qc_times,missing,case,omitted,masked", [
    (["2025-02-06T02:00", "2025-02-06T02:05"], ["2025-02-06T02:00", "2025-02-06T02:05"], [0], "A", 0, 1),
    (["2025-02-03T01:45", "2025-02-06T02:00"], ["2025-02-06T02:00"], [], "B", 1, 0),
    (["2025-02-03T02:00", "2025-02-06T02:00"], ["2025-02-03T02:00", "2025-02-06T02:00"], [0], "C", 0, 1),
    (["2025-01-28T02:00", "2025-02-06T02:00"], ["2025-01-28T02:00", "2025-02-06T02:00"], [0], "D", 0, 1),
])
def test_cases_distinguish_omitted_from_masked(raw_times, qc_times, missing, case, omitted, masked):
    result = compare_series(series(raw_times), series(qc_times, missing))
    assert result["feb06_case"] == case
    assert result["raw_early_rows_omitted_from_qc"] == omitted
    assert result["raw_early_rows_retained_masked_in_qc"] == masked
    assert result["raw_n_obs"] == len(raw_times)


def test_time_precision_is_tolerated_without_rounding_values():
    raw = series(["2025-02-06T02:00:00", "2025-02-06T02:05:00"])
    qc = series(["2025-02-06T01:59:59.999995", "2025-02-06T02:05:00.000004"], [0])
    result = compare_series(raw, qc)
    assert result["raw_early_rows_omitted_from_qc"] == 0
    assert result["raw_early_rows_retained_masked_in_qc"] == 1
    assert result["qc_first_time"].nanosecond == 0
    assert result["qc_first_time"].microsecond == 999995
    assert result["delta_raw_to_qc_first_time_hours"] != 0


def test_first_valid_raw_position_is_not_first_raw_timestamp():
    raw = series(["2025-02-03", "2025-02-06T02:00"], [0])
    qc = series(["2025-02-06T02:05"])
    result = compare_series(raw, qc)
    assert result["delta_raw_to_qc_first_valid_position_hours"] == pytest.approx(5/60)
    assert result["delta_raw_first_time_to_qc_first_valid_position_hours"] > 72


def test_mapping_reports_duplicates_and_unmatched_without_pairing():
    raw = {"1": [{"path": "a"}, {"path": "b"}], "2": [{"path": "c"}], "4": [{"path": "d"}]}
    qc = {"1": [{"path": "z"}], "3": [{"path": "y"}], "4": [{"path": "x"}]}
    result = matching_report(raw, qc, [], [])
    assert result["matched_pairs"] == 1
    assert result["ambiguous_platform_ids"] == ["1"]
    assert result["qc_without_raw"] == ["3"]
    assert result["raw_without_qc"] == ["2"]


def test_raw_calendar_strings_and_embedded_ids(tmp_path):
    # Deliberately misleading filename: the embedded identifier must win.
    path = tmp_path / "999.mat"
    payload = {"dataset": {"drifter_1001": {"PlatformId": 1001,
               "ObsTimestamp": np.array(["2025-02-06 02:05:00", "2025-02-06 02:00:00"]),
               "GpsLongitude": [131., 130.], "GpsLatitude": [21., 20.]},
               "meta_1001": {"PlatformID": 1001}}}
    savemat(path, payload)
    found, errors = catalog(tmp_path, "raw")
    assert list(found) == ["1001"] and not errors
    raw = read_raw_timing(found["1001"][0])
    assert raw.time[0] == np.datetime64("2025-02-06T02:00:00", "ns")
    np.testing.assert_array_equal(raw.source_index, [1, 0])
    np.testing.assert_array_equal(raw.lon, [130., 131.])
    payload["dataset"]["meta_1001"]["PlatformID"] = 1002
    savemat(path, payload)
    _, errors = catalog(tmp_path, "raw")
    assert "Conflicting raw identifier" in errors[0]["error"]


def test_cohorts_are_derived_from_observed_consecutive_dates():
    frame = pd.DataFrame({"qc_first_valid_position_time": pd.to_datetime([
        "2025-01-12T23:00:00Z", "2025-01-13T00:05:00Z", "2025-02-01T23:00:00Z", "2025-02-06T03:00:00Z", None
    ])})
    assert assign_cohorts(frame).tolist() == ["2025-01-12 to 2025-01-13", "2025-01-12 to 2025-01-13", "2025-02-01", "2025-02-06", "no_valid_qc_time"]


def test_inventory_first_valid_definition_recomputed(mat_factory):
    path, _ = mat_factory()
    qc, errors = catalog(path.parent, "qc")
    assert not errors
    inventory = pd.DataFrame({"platform_code": ["1001"], "first_time_qc": [pd.Timestamp("1970-01-02")],
                              "first_valid_position_time_qc": [pd.Timestamp("1970-01-02")],
                              "source_sha256": [sha256(path.read_bytes()).hexdigest()]})
    table = build_table({}, qc, inventory)
    assert table.match_status.iloc[0] == "qc_only"
    assert table.inventory_first_valid_time_diff_seconds.iloc[0] == 0
    assert table.inventory_source_hash_matches.iloc[0]
    assert pd.isna(table.raw_first_time.iloc[0])
    assert table.raw_n_rows_before_qc_first_time.dtype.kind == "f"
    assert table.raw_n_rows_before_qc_first_valid_date.dtype.kind == "f"


def test_complete_diagnostic_writes_report_without_changing_inventory(mat_factory, config_factory, tmp_path):
    path, payload = mat_factory()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = {"PlatformId": 1001, "ObsTimestamp": ["1970-01-04 00:00:00", "1970-01-03 00:00:00", "1970-01-02 00:00:00"],
           "GpsLongitude": payload["drifter"]["longitude"], "GpsLatitude": payload["drifter"]["latitude"]}
    savemat(raw_dir / "unrelated_filename.mat", {"dataset": {"drifter_1001": raw}})
    inventory = pd.DataFrame({"platform_code": ["1001"], "first_time_qc": [pd.Timestamp("1970-01-02")],
                              "first_valid_position_time_qc": [pd.Timestamp("1970-01-02")],
                              "source_sha256": [sha256(path.read_bytes()).hexdigest()]})
    inv_path = tmp_path / "out/inventory.parquet"
    inv_path.parent.mkdir()
    inventory.to_parquet(inv_path)
    before = sha256(inv_path.read_bytes()).hexdigest()
    config = config_factory()
    summary = run(config, raw_dir, tmp_path / "diagnostic", figures=False)
    assert summary["matching"]["matched_pairs"] == 1
    assert summary["inventory_check"]["first_valid_time_mismatches"] == 0
    assert sha256(inv_path.read_bytes()).hexdigest() == before
    table = pd.read_parquet(tmp_path / "diagnostic/microsvp_timing_diagnostic.parquet")
    assert table.raw_n_rows_before_qc_first_valid_date.iloc[0] == 0
    assert (tmp_path / "diagnostic/microsvp_timing_report.md").exists()
    assert not (tmp_path / "out/master.zarr").exists()
