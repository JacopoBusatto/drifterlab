from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from drifterlab.qc.drogue import (
    DrogueDetectionConfig,
    StrainStepConfig,
    TTFFCessationComparisonConfig,
    TTFFCessationConfig,
    ValueBinningConfig,
    analysis_cutoff_time,
    detect_drogue_loss,
    detect_strain_step,
    detect_ttff_cessation,
    histogram_counts,
    make_value_bins,
    temporal_event_counts,
)


def hourly(length: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=length, freq="h", tz="UTC")


def ttff_config(**comparison_changes) -> TTFFCessationConfig:
    comparison = replace(
        TTFFCessationComparisonConfig(),
        **comparison_changes,
    )
    return TTFFCessationConfig(
        ValueBinningConfig(
            "linear", 100, None, 50, 300, 12,
            include_lower_than_first_bin=False,
        ),
        comparison,
    )


BIN_VALUES = (120.0, 170.0, 220.0, 270.0, 350.0)


def binned_track(
    length: int,
    drop_hours: list[int | None],
    *,
    cadence_hours: int = 1,
    extra_events: list[tuple[int, int]] | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Build all-valid TTFF telemetry with per-bin event cessation times."""
    times: list[pd.Timestamp] = []
    values: list[float] = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for hour in range(0, length, cadence_hours):
        stamp = start + pd.Timedelta(hours=hour)
        times.append(stamp)
        values.append(20.0)  # valid availability below the excluded first edge
        for bin_index, drop_hour in enumerate(drop_hours):
            if drop_hour is None or hour < drop_hour:
                times.append(stamp)
                values.append(BIN_VALUES[bin_index])
    for hour, bin_index in extra_events or []:
        times.append(start + pd.Timedelta(hours=hour))
        values.append(BIN_VALUES[bin_index])
    order = np.argsort(pd.DatetimeIndex(times).asi8, kind="stable")
    return pd.DatetimeIndex(np.asarray(times)[order]), np.asarray(values)[order]


def test_linear_and_log_bins_cover_excluded_lower_values_and_open_overflow():
    log_bins = make_value_bins(ValueBinningConfig("log", 10, 2, None, 100, 12))
    assert log_bins.physical_edges.tolist() == [0, 10, 20, 40, 80, 100]
    counts, n_valid = histogram_counts(
        [-2, 0, 9.9, 10, 100, 1e9, np.nan, -999],
        log_bins,
        missing_values=(-999,),
    )
    assert n_valid == 6
    assert counts.tolist() == [1, 2, 1, 0, 0, 0, 2]

    linear_bins = make_value_bins(ValueBinningConfig(
        "linear", 100, None, 50, 300, 12,
        include_lower_than_first_bin=False,
    ))
    counts, n_valid = histogram_counts(
        [0, 99, 100, 149, 300, 1e9, np.nan, -999],
        linear_bins,
        missing_values=(-999,),
    )
    assert n_valid == 4
    assert counts.tolist() == [2, 0, 0, 0, 2]
    assert linear_bins.labels[-1] == ">=300"


def test_temporal_counts_are_raw_integers_and_n_valid_uses_every_valid_value():
    time = hourly(24)
    values = np.zeros(24)
    values[[1, 2, 13]] = [120, 170, 350]
    values[5] = np.nan
    values[6] = -999
    table, _ = temporal_event_counts(
        time, values, ttff_config(), missing_values=(-999,),
    )
    assert table.iloc[0].n_valid == 10
    assert table.iloc[1].n_valid == 12
    assert np.issubdtype(np.asarray(table.iloc[0].bin_counts).dtype, np.integer)
    assert table.iloc[0].bin_counts.tolist() == [1, 1, 0, 0, 0]
    assert table.iloc[1].bin_counts.tolist() == [0, 0, 0, 0, 1]


def test_shared_multibin_cessation_is_clear_despite_dominant_excluded_values():
    time, values = binned_track(600, [240] * len(BIN_VALUES))
    detection = detect_ttff_cessation(time, values, ttff_config())
    assert detection.status == "clear"
    assert detection.selected.time == time[0] + pd.Timedelta(hours=240)
    assert detection.selected.agreeing_bin_count == len(BIN_VALUES)
    assert (detection.bin_results.status == "stable_drop").all()
    assert detection.temporal_counts.n_valid.sum() == len(values)


def test_logarithmic_value_bins_use_the_same_raw_cessation_rule():
    config = TTFFCessationConfig(
        ValueBinningConfig(
            "log", 100, 2, None, 800, 12,
            include_lower_than_first_bin=False,
        ),
        TTFFCessationComparisonConfig(),
    )
    time, values = binned_track(600, [240] * len(BIN_VALUES))
    detection = detect_ttff_cessation(time, values, config)
    assert detection.status == "clear"
    assert detection.selected.time == time[0] + pd.Timedelta(hours=240)
    assert detection.selected.agreeing_bin_count == 2


def test_last_drop_is_selected_after_an_early_lull_and_reactivation():
    time, values = binned_track(900, [None] * len(BIN_VALUES))
    hours = ((time - time[0]) / pd.Timedelta(hours=1)).astype(int)
    analyzed = values >= 100
    keep = ~(analyzed & (hours >= 240) & (hours < 400))
    keep &= ~(analyzed & (hours >= 648))
    detection = detect_ttff_cessation(time[keep], values[keep], ttff_config())
    assert detection.status == "clear"
    assert detection.selected.time == time[0] + pd.Timedelta(hours=648)


def test_one_isolated_late_event_per_window_is_tolerated():
    extras = [(400, bin_index) for bin_index in range(len(BIN_VALUES))]
    time, values = binned_track(650, [240] * len(BIN_VALUES), extra_events=extras)
    detection = detect_ttff_cessation(time, values, ttff_config())
    assert detection.status == "clear"
    assert detection.selected.time == time[0] + pd.Timedelta(hours=240)
    assert detection.selected.confirmed_until == time[0] + pd.Timedelta(hours=648)


def test_two_late_events_reactivate_when_the_rate_threshold_also_fails():
    config = ttff_config(quiet_fraction_of_pre_rate=.01)
    extras = [
        (400, bin_index)
        for bin_index in range(len(BIN_VALUES))
    ] + [
        (412, bin_index)
        for bin_index in range(len(BIN_VALUES))
    ]
    time, values = binned_track(650, [240] * len(BIN_VALUES), extra_events=extras)
    detection = detect_ttff_cessation(time, values, config)
    assert detection.status == "none"
    assert (detection.bin_results.status == "reactivated").all()


def test_active_and_never_eligible_bins_do_not_block_an_agreeing_group():
    time, values = binned_track(650, [None, 240, 240, 240, 2])
    detection = detect_ttff_cessation(time, values, ttff_config())
    assert detection.status == "clear"
    assert detection.selected.agreeing_bin_count == 3
    statuses = detection.bin_results.set_index("bin_index").status
    assert statuses[0] == "no_stable_drop"
    assert statuses[4] == "insufficient_pre_activity"


def test_short_followup_and_telemetry_outage_do_not_produce_clear_detection():
    time, values = binned_track(330, [240] * len(BIN_VALUES))
    short = detect_ttff_cessation(time, values, ttff_config())
    assert short.status == "insufficient_followup"

    time, values = binned_track(650, [None] * len(BIN_VALUES))
    hours = ((time - time[0]) / pd.Timedelta(hours=1)).astype(int)
    outage = (hours < 240) | (hours >= 500)
    detection = detect_ttff_cessation(time[outage], values[outage], ttff_config())
    assert detection.status != "clear"


def test_irregular_two_hour_cadence_can_detect_a_stable_drop():
    time, values = binned_track(
        700, [240] * len(BIN_VALUES), cadence_hours=2,
    )
    detection = detect_ttff_cessation(time, values, ttff_config())
    assert detection.status == "clear"
    assert detection.selected.time == time[0] + pd.Timedelta(hours=240)


def test_consensus_keeps_a_singleton_outlier_visible_without_invalidating_clear():
    time, values = binned_track(850, [240, 264, 264, 420, 240])
    detection = detect_ttff_cessation(time, values, ttff_config())
    assert detection.status == "clear"
    assert detection.selected.agreeing_bin_count == 4
    assert detection.selected.consensus_span_hours == 24
    assert detection.selected.time == time[0] + pd.Timedelta(hours=252)
    assert not detection.bin_results.set_index("bin_index").loc[3, "in_consensus"]


def test_disjoint_agreeing_groups_are_ambiguous():
    time, values = binned_track(850, [240, 240, 420, 420, None])
    detection = detect_ttff_cessation(time, values, ttff_config())
    assert detection.status == "ambiguous"


def test_rolling_median_strain_detector_regression_for_persistent_step():
    time = hourly(420)
    strain = np.r_[np.full(240, 20.0), np.full(180, 10.0)]
    detection = detect_strain_step(time, strain, StrainStepConfig())
    assert detection.status == "clear"
    assert detection.selected.time == time[240]
    assert detection.selected.drop_absolute == pytest.approx(10.0)
    assert detection.selected.drop_relative == pytest.approx(.5)
    assert detection.selected.normalized_drop == pytest.approx(10.0)


def test_isolated_strain_spike_is_suppressed_by_rolling_median():
    strain = np.full(360, 20.0)
    strain[120] = 2.0
    detection = detect_strain_step(hourly(len(strain)), strain, StrainStepConfig())
    assert detection.status == "none"
    assert detection.selected is None


def test_temporary_strain_dip_that_recovers_is_not_a_loss_event():
    strain = np.r_[np.full(120, 20.0), np.full(36, 10.0), np.full(204, 20.0)]
    detection = detect_strain_step(hourly(len(strain)), strain, StrainStepConfig())
    assert detection.status == "none"
    assert detection.selected is None
    assert "transient" in set(detection.candidates.candidate_level)


def test_stable_strain_has_no_detectable_drop():
    detection = detect_strain_step(
        hourly(360), np.full(360, 20.0), StrainStepConfig(),
    )
    assert detection.status == "none"
    assert detection.selected is None


def test_strain_step_handles_irregular_times_missing_values_and_sentinel():
    removed = [3, 4, 8, 35, 121, 122, 180, 181, 250]
    time = hourly(360).delete(removed)
    strain = np.delete(np.r_[np.full(120, 20.0), np.full(240, 10.0)], removed)
    strain[[5, 70, 160]] = [np.nan, -999, np.nan]
    detection = detect_strain_step(
        time, strain, StrainStepConfig(), missing_values=(-999,),
    )
    assert detection.status == "clear"
    assert abs(detection.selected.time - hourly(360)[120]) <= pd.Timedelta("6h")
    assert detection.selected.pre_n_valid > 0
    assert detection.selected.post_n_valid > 0


def test_multiple_persistent_strain_steps_are_ambiguous():
    strain = np.r_[np.full(120, 30.0), np.full(180, 20.0), np.full(250, 10.0)]
    detection = detect_strain_step(hourly(len(strain)), strain, StrainStepConfig())
    assert detection.status == "ambiguous"
    assert detection.selected is not None


@pytest.mark.parametrize("start, stop", [(240, 246), (240, 264)])
def test_rolling_median_strain_detector_rejects_transient_spikes(start, stop):
    time = hourly(420)
    strain = np.full(420, 20.0)
    strain[start:stop] = 5.0
    detection = detect_strain_step(time, strain, StrainStepConfig())
    assert detection.status == "none"


def test_strain_remains_primary_and_ttff_only_corroborates():
    time, ttff = binned_track(700, [240] * len(BIN_VALUES))
    hours = ((time - time[0]) / pd.Timedelta(hours=1)).astype(int)
    strain = np.where(hours < 300, 20.0, 10.0)
    result = detect_drogue_loss(
        "test", time, ttff, strain=strain,
        config=DrogueDetectionConfig(ttff=ttff_config()),
    ).result
    assert result.ttff_change_time == np.datetime64("2025-01-11T00:00:00")
    assert result.strain_change_time == np.datetime64("2025-01-13T12:00:00")
    assert result.auto_drogue_loss_time == result.strain_change_time
    assert result.auto_status == "detected_strain_primary_corroborated"
    assert result.ttff_strain_offset_hours == 60


def test_ttff_is_provisional_when_strain_is_unavailable():
    time, ttff = binned_track(650, [240] * len(BIN_VALUES))
    result = detect_drogue_loss(
        "test", time, ttff,
        config=DrogueDetectionConfig(ttff=ttff_config()),
    ).result
    assert result.auto_status == "detected_ttff_provisional"
    assert result.auto_drogue_loss_time == result.ttff_change_time
    assert result.strain_change_status == "unavailable"


def test_no_valid_ttff_is_unavailable_and_keeps_public_counts_zero():
    time = hourly(400)
    detection = detect_drogue_loss(
        "test", time, np.full(400, -999.0), strain=np.full(400, 20.0),
        config=DrogueDetectionConfig(ttff=ttff_config()),
    )
    assert detection.result.ttff_change_status == "unavailable"
    assert detection.result.ttff_eligible_bin_count == 0
    assert detection.result.ttff_stable_drop_bin_count == 0


def test_ttff_configuration_validation_and_obsolete_sections():
    with pytest.raises(ValueError, match="multiple"):
        TTFFCessationConfig(
            ValueBinningConfig("linear", 100, None, 50, 300, 10),
            TTFFCessationComparisonConfig(),
        )
    with pytest.raises(ValueError, match="Unknown detection.ttff keys"):
        DrogueDetectionConfig.from_dict({"ttff": {"tail": {}}})
    with pytest.raises(ValueError, match="Unknown detection.ttff.comparison"):
        DrogueDetectionConfig.from_dict({
            "ttff": {"comparison": {"wasserstein_threshold": .2}},
        })


def test_analysis_cutoff_remains_distinct_from_the_detected_event_time():
    event = np.datetime64("2025-01-10T12:00:00")
    assert analysis_cutoff_time(event, 24) == np.datetime64("2025-01-09T12:00:00")
