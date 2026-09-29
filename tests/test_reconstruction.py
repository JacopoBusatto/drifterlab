"""Focused numerical tests for common-grid trajectory reconstruction."""

import numpy as np

from drifterlab.reconstruction.core import reconstruct_platform


def times(minutes):
    return np.datetime64("2025-01-12T00:00:00", "ns") + np.asarray(minutes).astype("timedelta64[m]")


def reconstruct(source_minutes, lon, lat, *, grid_minutes=range(0, 61, 5), periods=(15,), thresholds=None):
    grid = times(list(grid_minutes))
    indices = np.asarray(list(grid_minutes), dtype=np.int64) // 5
    return reconstruct_platform(
        times(source_minutes), lon, lat, grid, indices, dt_minutes=5,
        periods_minutes=periods,
        long_gap_threshold_minutes=thresholds or {period: period for period in periods},
    )


def test_linear_grid_source_gap_and_no_extrapolation():
    result = reconstruct([7, 17, 32], [0, 1, 4], [10, 11, 14])
    assert np.isnan(result.longitude_linear[:2]).all()
    assert np.isnan(result.longitude_linear[7:]).all()
    np.testing.assert_allclose(result.longitude_linear[2:7], [.3, .8, 1.6, 2.6, 3.6])
    np.testing.assert_allclose(result.source_gap_minutes[2:4], [10, 10])
    np.testing.assert_allclose(result.source_gap_minutes[4:7], [15, 15, 15])
    assert result.exact_grid_points == 0 and result.filled_grid_points == 5


def test_every_phase_contributes_and_output_stays_at_dt():
    source = list(range(0, 61, 5))
    x = np.asarray(source, dtype=float) / 60
    result = reconstruct(source, x, x ** 2)
    assert len(result.latitude_spline[15]) == 13
    assert np.isfinite(result.latitude_spline[15]).all()
    assert result.fallback[15].total_fallback_points == 0
    assert not np.array_equal(result.latitude_spline[15], result.latitude_linear)


def test_long_gap_is_linear_and_reported():
    source = [0, 5, 10, 15, 45, 50, 55, 60]
    lon = np.asarray(source, dtype=float) / 100
    result = reconstruct(source, lon, np.zeros(len(source)), thresholds={15: 15})
    interval = slice(3, 10)
    np.testing.assert_allclose(result.longitude_spline[15][interval], result.longitude_linear[interval])
    assert result.fallback[15].long_gap_portions == 1
    assert result.fallback[15].long_gap_points == 7


def test_insufficient_phase_support_falls_back_to_linear():
    result = reconstruct([0, 10, 20], [0, 1, 2], [0, .1, .2], grid_minutes=range(0, 21, 5))
    np.testing.assert_allclose(result.longitude_spline[15], result.longitude_linear)
    assert result.fallback[15].insufficient_support_portions == 1
    assert result.fallback[15].insufficient_support_points == 5


def test_dateline_is_unwrapped_before_interpolation_and_wrapped_afterward():
    result = reconstruct([0, 10], [179, -179], [0, 0], grid_minutes=range(0, 11, 5))
    assert abs(abs(result.longitude_linear[1]) - 180) < 1e-12
    assert not np.isclose(result.longitude_linear[1], 0)
    assert result.longitude_linear.tolist() == [179.0, -180.0, -179.0]


def test_exact_fixes_have_zero_source_gap():
    result = reconstruct([0, 10, 20], [0, 1, 2], [0, 0, 0], grid_minutes=range(0, 21, 5))
    np.testing.assert_array_equal(result.source_gap_minutes, [0, 10, 0, 10, 0])


def test_phase_overshoot_falls_back_for_the_complete_portion():
    source = list(range(0, 61, 5))
    latitude = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    result = reconstruct(source, np.asarray(source) / 100, latitude)
    np.testing.assert_allclose(result.latitude_spline[15], result.latitude_linear)
    assert result.fallback[15].overshoot_portions == 1
    assert result.fallback[15].overshoot_points == 13
