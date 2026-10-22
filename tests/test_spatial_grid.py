from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from drifterlab.analysis.spatial_grid import (
    SpatialBand,
    daily_windows,
    estimate_spatial_cells,
    fixed_geographic_extent,
    resolve_bands,
    resolve_daily_maps,
)
from drifterlab.epsilon import BinSupportRequirements


class _IdentityTransform:
    def transform(self, x, y):
        return x, y


def _samples():
    frame = pd.DataFrame({
        "array_age_hours": [0, 1, 2, 24, 25, 48],
        "midpoint_x_m": [50, 50, 1150, 50, 50, 50],
        "midpoint_y_m": [50] * 6,
        "separation_m": [100, 200, 500, 100, 200, 9000],
        "delta_u_l_cube_m3_s3": [-8, 8, -10, 4, 4, 100],
        "pair_id": ["ab", "ac", "ab", "ab", "ac", "ab"],
        "platform_id_1": ["a"] * 6,
        "platform_id_2": ["b", "c", "b", "b", "c", "b"],
        "source_cluster_id_1": ["A"] * 6,
        "source_cluster_id_2": ["A", "B", "A", "A", "B", "A"],
        "time_utc": pd.to_datetime("2025-01-01", utc=True) + pd.to_timedelta([0, 1, 2, 24, 25, 48], unit="h"),
    })
    frame["q_m2_s3"] = -1.25 * frame.delta_u_l_cube_m3_s3 / frame.separation_m
    return frame


def test_daily_windows_are_nonoverlapping_and_keep_final_partial_window():
    windows = daily_windows([
        {"start_hour": 0, "end_hour": 24}, {"start_hour": 12, "end_hour": 360},
    ], 24)
    assert len(windows) == 15
    assert windows[0] == {"id": "day_001", "start_hour": 0, "end_hour": 24}
    assert windows[-1] == {"id": "day_015", "start_hour": 336, "end_hour": 360}
    assert all(left["end_hour"] == right["start_hour"] for left, right in pairwise(windows))
    partial = daily_windows([{"start_hour": 12, "end_hour": 61}], 24)
    assert [(w["start_hour"], w["end_hour"]) for w in partial] == [(12, 36), (36, 60), (60, 61)]


def test_spatial_epsilon_is_ratio_of_sums_with_zero_distinct_from_missing():
    windows = daily_windows([{"start_hour": 0, "end_hour": 48}], 24)
    cells, contributions = estimate_spatial_cells(
        _samples(), windows=windows, bands=[SpatialBand(0, 0, None, 1000, None)],
        support=BinSupportRequirements(2, 2, 3, 2),
        projection=SimpleNamespace(inverse=_IdentityTransform()),
    )
    assert len(cells) == 3  # Only occupied cells are saved; day-2 upper bound is excluded.
    assert cells.observation_count.sum() == 5
    zero = cells[(cells.window_id == "day_001") & (cells.cell_x_index == 0)].iloc[0]
    assert zero.support_ok
    assert zero.epsilon_eff_m2_s3 == 0
    assert _samples().iloc[:2].q_m2_s3.mean() != 0
    assert zero.dominant_cluster_fraction == 0.75
    unsupported = cells[cells.cell_x_index == 1].iloc[0]
    assert not unsupported.support_ok
    assert np.isnan(unsupported.epsilon_eff_m2_s3)
    assert "unique_pairs" in unsupported.support_status
    second_day = cells[cells.window_id == "day_002"].iloc[0]
    assert second_day.epsilon_eff_m2_s3 == pytest.approx(-1.25 * 8 / 300)
    assert zero.x_lower_m == second_day.x_lower_m == 0
    weights = contributions.groupby(["window_id", "separation_bin_index", "cell_x_index", "cell_y_index"])
    np.testing.assert_allclose(weights.observation_fraction.sum(), 1)
    assert contributions.observation_weight.sum() == 5


def test_outer_bins_include_exact_boundaries_and_auto_sizes_use_lower_edges():
    settings = resolve_daily_maps({"separation_edges_m": [250, 1000], "cell_size_factor": 2})
    samples = _samples()
    samples.loc[1, "separation_m"] = 250
    samples.loc[2, "separation_m"] = 1000
    bands = resolve_bands(samples, settings)
    assert [(b.lower_m, b.upper_m) for b in bands] == [(0, 250), (250, 1000), (1000, None)]
    assert [b.cell_size_m for b in bands] == [500, 500, 2000]
    changed = samples.copy()
    changed.loc[5, "separation_m"] = 1e9
    assert resolve_bands(changed, settings) == bands
    cells, _ = estimate_spatial_cells(
        samples, windows=daily_windows([{"start_hour": 0, "end_hour": 72}], 24), bands=bands,
        support=BinSupportRequirements(1, 1, 2, 1), projection=SimpleNamespace(inverse=_IdentityTransform()),
    )
    assert cells.groupby("separation_bin_index").observation_count.sum().to_dict() == {0: 3, 1: 1, 2: 2}
    assert cells[cells.separation_bin_index == 2].cell_size_m.nunique() == 1


def test_comprehensive_manual_and_empty_automatic_band_sizes():
    samples = _samples()
    comprehensive = resolve_bands(samples, resolve_daily_maps({"manual_cell_size_m": 1000}))
    assert len(comprehensive) == 1
    assert comprehensive[0].cell_size_m == 1000
    scalar = resolve_bands(samples, resolve_daily_maps({"separation_edges_m": [250, 1000], "manual_cell_size_m": 500}))
    assert [b.cell_size_m for b in scalar] == [500] * 3
    per_bin = resolve_daily_maps({"separation_edges_m": [250, 1000], "manual_cell_size_m": [100, 500, 2000]})
    assert [b.cell_size_m for b in resolve_bands(samples, per_bin)] == [100, 500, 2000]
    assert resolve_daily_maps(per_bin) == per_bin  # Resolved configs can be loaded again.
    empty = resolve_bands(samples.iloc[:0], resolve_daily_maps({"separation_edges_m": [250]}))
    assert empty[0].cell_size_m == 250
    assert empty[1].cell_size_m == 250


def test_grid_cells_have_stable_half_open_boundaries_including_negative_coordinates():
    samples = _samples().iloc[:3].copy()
    samples.midpoint_x_m = [-100, -0.01, 0]
    cells, _ = estimate_spatial_cells(
        samples, windows=daily_windows([{"start_hour": 0, "end_hour": 24}], 24),
        bands=[SpatialBand(0, 0, None, 100, None)], support=BinSupportRequirements(1, 1, 2, 1),
        projection=SimpleNamespace(inverse=_IdentityTransform()),
    )
    assert cells.cell_x_index.tolist() == [-1, 0]
    assert cells.observation_count.tolist() == [2, 1]


def test_fixed_map_region_does_not_expand_to_giant_overflow_cells():
    projection = SimpleNamespace(inverse=_IdentityTransform(), origin_longitude=10)
    fallback = (np.array([10, 11]), np.array([20, 21]))
    small = pd.DataFrame({"x_lower_m": [10], "x_upper_m": [11], "y_lower_m": [20], "y_upper_m": [21]})
    giant = pd.DataFrame({"x_lower_m": [-100], "x_upper_m": [100], "y_lower_m": [-100], "y_upper_m": [100]})
    assert fixed_geographic_extent(small, projection, fallback) == fixed_geographic_extent(giant, projection, fallback)
    np.testing.assert_allclose(fixed_geographic_extent(giant, projection, fallback), [9.95, 11.05, 19.95, 21.05])


@pytest.mark.parametrize("values,match", [
    ({"window_hours": 0, "manual_cell_size_m": 100}, "positive"),
    ({"window_hours": True, "manual_cell_size_m": 100}, "numeric"),
    ({"separation_edges_m": [1000, 250]}, "increase strictly"),
    ({"separation_edges_m": [250, 250]}, "increase strictly"),
    ({"separation_edges_m": [0, 250]}, "positive"),
    ({"separation_edges_m": [float("inf")]}, "finite"),
    ({"separation_edges_m": 250}, "list"),
    ({"manual_cell_size_m": 100, "cell_size_factor": 1}, "mutually exclusive"),
    ({"manual_cell_size_m": 100, "minimum_cell_size_m": 50}, "no longer used"),
    ({"manual_cell_size_m": [100, 200]}, "one size per"),
    ({"manual_cell_size_m": 0}, "positive"),
    ({"manual_cell_size_m": True}, "numeric"),
    ({"cell_size_factor": -1, "separation_edges_m": [250]}, "positive"),
    ({"minimum_cell_size_m": float("nan")}, "no longer used"),
    ({}, "required without separation edges"),
    ({"manual_cell_size_m": 100, "enforce_support": "false"}, "true or false"),
    ({"smoothing": True}, "Unknown"),
])
def test_invalid_daily_map_settings_are_rejected(values, match):
    with pytest.raises(ValueError, match=match):
        resolve_daily_maps(values)


def test_skipping_support_preserves_counts_and_single_observation_estimate():
    cells, _ = estimate_spatial_cells(
        _samples(), windows=daily_windows([{"start_hour": 0, "end_hour": 48}], 24),
        bands=[SpatialBand(0, 0, None, 1000, None)], support=BinSupportRequirements(12, 3, 3, 12),
        projection=SimpleNamespace(inverse=_IdentityTransform()), enforce_support=False,
    )
    assert not cells.support_ok.any()
    assert cells.display_ok.all()
    assert np.isfinite(cells.epsilon_eff_m2_s3).all()
    single = cells[cells.cell_x_index == 1].iloc[0]
    assert single.observation_count == single.unique_pair_count == single.unique_utc_count == 1
    assert single.epsilon_eff_m2_s3 == pytest.approx(_samples().iloc[2].q_m2_s3)
    assert len(cells) == 3  # No empty cells are filled when support is bypassed.


def test_bypassing_support_still_excludes_nonfinite_estimates():
    samples = _samples().iloc[2:3].copy()
    samples["delta_u_l_cube_m3_s3"] = np.inf
    cells, _ = estimate_spatial_cells(
        samples, windows=daily_windows([{"start_hour": 0, "end_hour": 24}], 24),
        bands=[SpatialBand(0, 0, None, 1000, None)], support=BinSupportRequirements(1, 1, 2, 1),
        projection=SimpleNamespace(inverse=_IdentityTransform()), enforce_support=False,
    )
    assert not cells.iloc[0].display_ok
    assert np.isnan(cells.iloc[0].epsilon_eff_m2_s3)
