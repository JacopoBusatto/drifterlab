from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from drifterlab.qc.strain_two_regime import (
    StrainTwoRegimeAmbiguityConfig,
    StrainTwoRegimeConfig,
    aggregate_strain_blocks,
    robust_two_regime_strain_change,
)


def hourly(length: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=length, freq="h", tz="UTC")


def clear_step(length: int = 480, event: int = 240) -> tuple[pd.DatetimeIndex, np.ndarray]:
    return hourly(length), np.r_[np.full(event, 20.0), np.full(length - event, 10.0)]


def config(**changes) -> StrainTwoRegimeConfig:
    return replace(StrainTwoRegimeConfig(), **changes)


def test_clean_downward_step_has_exact_boundary_and_levels():
    time, strain = clear_step()
    detection = robust_two_regime_strain_change(time, strain)
    result = detection.result
    assert result.status == "clear"
    assert result.change_time == time[240].to_datetime64()
    assert result.level_before == 20
    assert result.level_after == 10
    assert result.absolute_drop == 10
    assert result.relative_drop == pytest.approx(.5)
    assert result.fit_improvement == pytest.approx(1)
    assert result.n_blocks_before == result.n_blocks_after == 40


def test_large_isolated_raw_spikes_do_not_move_block_median_change_point():
    time, strain = clear_step()
    strain = strain.copy()
    strain[np.arange(2, len(strain), 18)] = 1000
    strain[np.arange(8, len(strain), 24)] = -500
    detection = robust_two_regime_strain_change(time, strain)
    assert detection.result.status == "clear"
    assert detection.result.change_time == time[240].to_datetime64()
    assert detection.result.level_before == 20
    assert detection.result.level_after == 10


def test_constant_signal_has_no_downward_change_and_zero_cost_is_safe():
    detection = robust_two_regime_strain_change(hourly(300), np.full(300, 12.0))
    assert detection.result.status == "no_change"
    assert detection.result.j0 == 0
    assert np.isnat(detection.result.change_time)


def test_upward_change_is_not_accepted():
    strain = np.r_[np.full(240, 10.0), np.full(240, 20.0)]
    detection = robust_two_regime_strain_change(hourly(len(strain)), strain)
    assert detection.result.status == "no_change"
    assert np.isnat(detection.result.change_time)


def test_gradual_drift_does_not_look_like_a_unique_clear_step():
    strain = np.linspace(30.0, 10.0, 720)
    detection = robust_two_regime_strain_change(hourly(len(strain)), strain)
    assert detection.result.status != "clear"


def test_two_comparable_downward_steps_are_ambiguous():
    strain = np.r_[
        np.full(240, 30.0), np.full(240, 20.0), np.full(240, 10.0),
    ]
    detection = robust_two_regime_strain_change(hourly(len(strain)), strain)
    assert detection.result.status == "ambiguous"
    assert np.isnat(detection.result.change_time)
    assert not np.isnat(detection.result.candidate_time)
    assert detection.result.n_comparable_alternatives >= 1
    assert not np.isnat(detection.result.comparable_alternative_time)


def test_dominant_step_wins_over_a_smaller_temporary_fluctuation():
    time, strain = clear_step(600, 300)
    strain[120:144] = 17.0
    detection = robust_two_regime_strain_change(time, strain)
    assert detection.result.status == "clear"
    assert detection.result.change_time == time[300].to_datetime64()


def test_missing_blocks_irregular_sampling_and_sentinel_are_not_filled():
    base_time, base_strain = clear_step(600, 300)
    keep = np.arange(len(base_time)) % 2 == 0
    keep[120:138] = False
    time = base_time[keep]
    strain = base_strain[keep].copy()
    strain[[5, 80, 160]] = [np.nan, -999, np.nan]
    detection = robust_two_regime_strain_change(
        time, strain, missing_values=(-999,),
    )
    assert detection.result.status == "clear"
    assert abs(
        pd.Timestamp(detection.result.change_time).tz_localize("UTC") - base_time[300]
    ) <= pd.Timedelta(hours=6)
    assert detection.blocks.strain_median.isna().any()
    missing = detection.blocks[detection.blocks.strain_median.isna()]
    assert (missing.n_valid < StrainTwoRegimeConfig().min_samples_per_block).all()


def test_short_record_is_insufficient_for_both_minimum_sides():
    time, strain = clear_step(120, 60)
    detection = robust_two_regime_strain_change(time, strain)
    assert detection.result.status == "insufficient_data"
    assert detection.candidates.empty


def test_change_too_near_record_end_does_not_force_a_public_date():
    strain = np.r_[np.full(450, 20.0), np.full(30, 10.0)]
    result = robust_two_regime_strain_change(hourly(len(strain)), strain).result
    assert result.status != "clear"
    assert np.isnat(result.change_time)


def test_clear_step_is_stable_under_three_six_and_twelve_hour_aggregation():
    time, strain = clear_step(720, 360)
    dates = []
    for hours in (3, 6, 12):
        detection = robust_two_regime_strain_change(
            time, strain, config(aggregation_hours=hours),
        )
        assert detection.result.status == "clear"
        dates.append(pd.Timestamp(detection.result.change_time))
    assert max(dates) - min(dates) <= pd.Timedelta(hours=12)


def test_exact_l1_costs_and_fit_improvement():
    time = hourly(4)
    strain = np.array([10.0, 10.0, 5.0, 5.0])
    exact = StrainTwoRegimeConfig(
        aggregation_hours=1,
        min_samples_per_block=1,
        minimum_side_duration_hours=2,
        minimum_drop_absolute=0,
        minimum_drop_relative=0,
        minimum_fit_improvement=0,
        ambiguity=StrainTwoRegimeAmbiguityConfig(
            comparable_cost_fraction=.02,
            minimum_separation_hours=1,
        ),
    )
    result = robust_two_regime_strain_change(time, strain, exact).result
    assert result.j0 == pytest.approx(10)
    assert result.j1_best == pytest.approx(0)
    assert result.fit_improvement == pytest.approx(1)
    assert result.level_before == 10
    assert result.level_after == 5


def test_block_counts_and_centers_are_explicit_without_interpolation():
    time = pd.to_datetime([
        "2025-01-01T00:00Z", "2025-01-01T01:00Z", "2025-01-01T02:00Z",
        "2025-01-01T06:00Z", "2025-01-01T07:00Z",
        "2025-01-01T12:00Z", "2025-01-01T13:00Z", "2025-01-01T14:00Z",
    ])
    blocks = aggregate_strain_blocks(time, np.arange(8.0), StrainTwoRegimeConfig())
    assert blocks.n_valid.tolist() == [3, 2, 3]
    assert blocks.strain_median.iloc[0] == 1
    assert np.isnan(blocks.strain_median.iloc[1])
    assert blocks.time_center.iloc[0] == pd.Timestamp("2025-01-01T03:00Z")


def test_configuration_rejects_unknown_or_invalid_values():
    with pytest.raises(ValueError, match="Unknown strain_two_regime keys"):
        StrainTwoRegimeConfig.from_dict({"rolling_window": "12h"})
    with pytest.raises(ValueError, match="positive integer"):
        StrainTwoRegimeConfig(min_samples_per_block=0)
