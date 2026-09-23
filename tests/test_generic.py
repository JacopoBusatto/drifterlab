from datetime import date

import numpy as np
import pytest

from drifterlab.io.matlab import matlab_datenum_to_datetime64, normalize_missing
from drifterlab.qc.drogue import drogue_cutoff, drogue_valid
from drifterlab.qc.flags import AuditConfig, EARTH_RADIUS_M, audit_time, consecutive_speed, interval_seconds
from drifterlab.qc.position import analysis_valid, position_valid
from drifterlab.trajectories.records import ObservationSeries


def test_matlab_dates_precision_and_invalid_values():
    dates = matlab_datenum_to_datetime64([[719529., 719529.5], [-999, np.nan]])
    assert dates.shape == (2, 2)
    assert dates.dtype == np.dtype("datetime64[ns]")
    assert dates[0, 0] == np.datetime64("1970-01-01T00:00:00", "ns")
    assert dates[0, 1] == np.datetime64("1970-01-01T12:00:00", "ns")
    assert np.isnat(dates[1]).all()
    leap = date(2000, 2, 29).toordinal() + 366
    assert matlab_datenum_to_datetime64(leap) == np.datetime64("2000-02-29")
    fractional = 739629 + 0.123456 / 86400
    result = matlab_datenum_to_datetime64(fractional)[()]
    expected = np.datetime64("2025-01-12T00:00:00.123456", "ns")
    assert abs(int((result - expected) / np.timedelta64(1, "ns"))) < 6000
    assert np.isnat(matlab_datenum_to_datetime64([np.inf, -np.inf, 1e20, 0])).all()


def test_integer_sentinel_does_not_mutate_input():
    source = np.array([-999, 0, 4], dtype=np.int16)
    result = normalize_missing(source)
    assert result.dtype == np.float64
    assert np.isnan(result[0]) and result[1] == 0
    assert source[0] == -999


def test_stable_sort_duplicates_and_nat_preserve_rows():
    series = ObservationSeries.from_matlab(np.array([719531., 719530., 719530., np.nan]), {"value": np.array([20., 10., 11., 99.])})
    np.testing.assert_array_equal(series.source_index, [1, 2, 0, 3])
    np.testing.assert_array_equal(series.variables["value"], [10, 11, 20, 99])
    audit = audit_time(series.time, AuditConfig())
    np.testing.assert_array_equal(audit["duplicate_time_flag"], [True, True, False, False])
    np.testing.assert_array_equal(audit["invalid_time_flag"], [False, False, False, True])


def test_speed_haversine_missing_duplicate_and_dateline():
    t = np.array(["2025-01-01T00:00", "2025-01-01T00:05"], dtype="datetime64[ns]")
    speed = consecutive_speed(t, np.array([0., 1.]), np.array([0., 0.]))
    assert np.isnan(speed[0])
    assert speed[1] == pytest.approx(EARTH_RADIUS_M * np.pi / 180 / 300)
    across_date_line = consecutive_speed(t, np.array([179.999, -179.999]), np.array([0., 0.]))
    assert across_date_line[1] == pytest.approx(EARTH_RADIUS_M * np.pi / 180 * .002 / 300)
    assert np.isnan(consecutive_speed(t, np.array([np.nan, 1.]), np.array([0., 0.]))).all()
    assert np.isnan(consecutive_speed(t[[0, 0]], np.array([0., 1.]), np.array([0., 0.]))).all()
    assert np.isnan(consecutive_speed(t, np.array([0., 1.]), np.array([0., 100.]))).all()


def test_time_audit_tolerance_and_no_missing_position_bridge():
    base = np.datetime64("2025-01-01", "ns")
    t = base + np.array([0, 300, 600, 2402]) * np.timedelta64(1, "s")
    flags = audit_time(t, AuditConfig())
    np.testing.assert_array_equal(flags["sampling_interval_flag"], [False, False, False, True])
    np.testing.assert_array_equal(flags["large_gap_flag"], [False, False, False, True])
    v = consecutive_speed(t, np.array([0., np.nan, 1., 2.]), np.zeros(4))
    assert np.isnan(v[:3]).all() and np.isfinite(v[3])
    subsecond = base + np.array([0, 300_000_000_017]) * np.timedelta64(1, "ns")
    assert interval_seconds(subsecond)[1] == pytest.approx(300.000000017, abs=1e-12)


def test_drogue_cutoff_unknown_and_retained():
    loss = np.datetime64("2025-01-03", "ns")
    cutoff = drogue_cutoff(loss)
    assert cutoff == np.datetime64("2025-01-02")
    times = np.array(["2025-01-01", "2025-01-02", "2025-01-03", "NaT"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(drogue_valid(times, "lost", cutoff), [True, False, False, False])
    assert not drogue_valid(times, "unknown", np.datetime64("NaT", "ns")).any()
    np.testing.assert_array_equal(drogue_valid(times, "retained", np.datetime64("NaT", "ns")), [True, True, True, False])
    with pytest.raises(ValueError):
        drogue_cutoff(loss, -1)


def test_position_and_analysis_masks_are_separate():
    good = position_valid(np.array([0., 181., np.nan, 180.]), np.array([90., 0., 0., -90.]))
    np.testing.assert_array_equal(good, [True, False, False, True])
    times = np.array(["2025-01-01"] * 3 + ["NaT"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(analysis_valid(good, np.ones(4, bool), times), [True, False, False, False])
