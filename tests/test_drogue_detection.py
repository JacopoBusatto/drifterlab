from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from drifterlab.qc.drogue import (
    DistributionComparisonConfig,
    DrogueDetectionConfig,
    SignalDistributionConfig,
    StrainStepConfig,
    TailSupportConfig,
    ValueBinningConfig,
    analysis_cutoff_time,
    bounded_tail_area,
    detect_distribution_change,
    detect_drogue_loss,
    detect_strain_step,
    directional_wasserstein,
    histogram_counts,
    make_value_bins,
    temporal_histograms,
)


def repeated(values, length):
    return np.resize(np.asarray(values, dtype=float), length)


def hourly(length):
    return pd.date_range("2025-01-01", periods=length, freq="h", tz="UTC")


def test_log_and_linear_bins_include_zero_underflow_overflow_and_missing_values():
    log_bins = make_value_bins(ValueBinningConfig("log", 10, 2, None, 100, 12))
    assert log_bins.physical_edges.tolist() == [0, 10, 20, 40, 80, 100]
    counts, n_valid = histogram_counts(
        [-2, 0, 9.9, 10, 100, 1e9, np.nan, -999], log_bins,
        missing_values=(-999,),
    )
    assert n_valid == 6
    assert counts[0] == 1
    assert counts[1] == 2
    assert counts[2] == 1
    assert counts[-1] == 2

    linear_bins = make_value_bins(ValueBinningConfig("linear", 0, None, 2, 6, 12))
    assert linear_bins.physical_edges.tolist() == [0, 2, 4, 6]
    assert linear_bins.labels == ("<0", "[0, 2)", "[2, 4)", "[4, 6)", ">=6")


def test_lower_than_first_bin_can_be_excluded_from_counts_and_normalization():
    config = ValueBinningConfig(
        "log", 10, 2, None, 100, 12,
        include_lower_than_first_bin=False,
    )
    bins = make_value_bins(config)
    assert bins.physical_edges.tolist() == [10, 20, 40, 80, 100]
    assert bins.labels == ("[10, 20)", "[20, 40)", "[40, 80)", "[80, 100)", ">=100")
    counts, n_valid = histogram_counts(
        [-2, 0, 9.9, 10, 100, 1e9, np.nan, -999], bins,
        missing_values=(-999,),
    )
    assert n_valid == 3
    assert counts.sum() == 3
    assert counts[0] == 1
    assert counts[-1] == 2


def test_excluded_lower_values_do_not_contribute_temporal_coverage():
    time = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    config = ValueBinningConfig(
        "linear", 10, None, 2, 20, 12,
        include_lower_than_first_bin=False,
    )
    table, _ = temporal_histograms(time, [1, 5, 12], config)
    first = table.iloc[0]
    assert first.n_valid == 1
    assert first.bin_counts.sum() == 1
    assert first.fractions.sum() == pytest.approx(1)
    assert first.coverage_fraction == pytest.approx(1 / 12)
    assert np.isnan(first.underflow_fraction)


def test_time_histograms_record_counts_probabilities_and_irregular_coverage():
    time = pd.to_datetime([
        "2025-01-01T00:00Z", "2025-01-01T01:00Z", "2025-01-01T11:00Z",
        "2025-01-01T13:00Z",
    ])
    values = [0, 2, 8, np.nan]
    config = ValueBinningConfig("linear", 0, None, 2, 6, 12)
    table, _ = temporal_histograms(time, values, config)
    first = table.iloc[0]
    assert first.n_valid == 3
    assert first.bin_counts.sum() == 3
    assert first.fractions.sum() == pytest.approx(1)
    assert 0 < first.coverage_fraction < .8
    assert table.iloc[1].n_valid == 0


def test_directional_wasserstein_identical_downward_upward_and_identity():
    bins = make_value_bins(ValueBinningConfig("linear", 0, None, 1, 10, 12))
    low, _ = histogram_counts(np.full(20, 2.), bins)
    high, _ = histogram_counts(np.full(20, 8.), bins)
    identical = directional_wasserstein(high, high, bins)
    downward = directional_wasserstein(high, low, bins)
    upward = directional_wasserstein(low, high, bins)
    assert identical.w1 == pytest.approx(0)
    assert np.isnan(identical.downward_fraction)
    assert downward.d_down > 0 and downward.d_up == pytest.approx(0)
    assert upward.d_up > 0 and upward.d_down == pytest.approx(0)
    for result in (downward, upward):
        assert result.w1 == pytest.approx(result.d_down + result.d_up)

    no_lower = make_value_bins(
        ValueBinningConfig(
            "linear", 0, None, 1, 10, 12,
            include_lower_than_first_bin=False,
        )
    )
    low, _ = histogram_counts(np.full(20, 2.), no_lower)
    high, _ = histogram_counts(np.full(20, 8.), no_lower)
    assert directional_wasserstein(high, low, no_lower).d_down > 0


def test_ttff_extremes_are_capped_but_open_bin_and_tail_depletion_remain_visible():
    config = ValueBinningConfig("log", 10, 1.5, None, 1000, 12)
    bins = make_value_bins(config)
    before_a, _ = histogram_counts([80, 150, 300, 1e9], bins)
    before_b, _ = histogram_counts([80, 150, 300, 1e15], bins)
    after, _ = histogram_counts([5, 10, 20, 1e9], bins)
    shift_a = directional_wasserstein(before_a, after, bins)
    shift_b = directional_wasserstein(before_b, after, bins)
    assert shift_a == shift_b
    assert before_a[-1] == 1
    tail = TailSupportConfig(50, 1000)
    assert bounded_tail_area(before_a, bins, tail) > bounded_tail_area(after, bins, tail)


def test_temporary_ttff_quiet_period_with_reactivation_is_not_clear():
    ttff = np.r_[
        repeated([10, 30, 80, 150, 400], 120),
        repeated([2, 4, 6, 8], 90),
        repeated([10, 30, 80, 150, 400], 190),
    ]
    result = detect_drogue_loss("001", hourly(len(ttff)), ttff).result
    assert result.ttff_change_status != "clear"
    assert np.isnat(result.auto_drogue_loss_time)


def test_persistent_ttff_cloud_loss_survives_one_late_extreme_spike():
    ttff = np.r_[
        repeated([10, 30, 80, 150, 400], 120),
        repeated([2, 4, 6, 8], 280),
    ]
    ttff[270] = 1e12
    result = detect_drogue_loss("001", hourly(len(ttff)), ttff).result
    assert result.ttff_change_status == "clear"
    assert result.ttff_d_down > result.ttff_d_up
    assert result.ttff_tail_area_drop > 0
    assert result.auto_status == "detected_ttff_provisional"


def test_persistent_rolling_median_strain_step_has_correct_date_and_metrics():
    n, event = 360, 120
    strain = np.r_[np.full(event, 20.), np.full(n - event, 10.)]
    result = detect_drogue_loss(
        "001", hourly(n), np.full(n, 10.), strain=strain
    ).result
    assert result.strain_change_status == "clear"
    assert result.strain_change_time == hourly(n)[event].to_datetime64()
    assert result.strain_drop_absolute == pytest.approx(10)
    assert result.strain_drop_relative == pytest.approx(.5)
    assert result.strain_normalized_drop == pytest.approx(10)
    assert result.auto_drogue_loss_time == result.strain_change_time
    assert result.auto_status == "detected_strain_primary"


def test_isolated_strain_spike_is_suppressed_by_rolling_median():
    strain = np.full(360, 20.)
    strain[120] = 2
    detected = detect_strain_step(hourly(len(strain)), strain, StrainStepConfig())
    assert detected.status == "none"
    assert detected.selected is None


def test_temporary_strain_dip_that_recovers_is_not_a_loss_event():
    strain = np.r_[np.full(120, 20.), np.full(36, 10.), np.full(204, 20.)]
    detected = detect_strain_step(hourly(len(strain)), strain, StrainStepConfig())
    assert detected.status == "none"
    assert detected.selected is None
    assert "transient" in set(detected.candidates.candidate_level)


def test_stable_strain_has_no_detectable_drop():
    detected = detect_strain_step(hourly(360), np.full(360, 20.), StrainStepConfig())
    assert detected.status == "none"
    assert detected.selected is None


def test_strain_step_handles_irregular_times_missing_values_and_sentinel():
    removed = [3, 4, 8, 35, 121, 122, 180, 181, 250]
    time = hourly(360).delete(removed)
    strain = np.delete(np.r_[np.full(120, 20.), np.full(240, 10.)], removed)
    strain[[5, 70, 160]] = [np.nan, -999, np.nan]
    detected = detect_strain_step(
        time, strain, StrainStepConfig(), missing_values=(-999,),
    )
    assert detected.status == "clear"
    assert abs(detected.selected.time - hourly(360)[120]) <= pd.Timedelta("6h")
    assert detected.selected.pre_n_valid > 0
    assert detected.selected.post_n_valid > 0


def test_multiple_persistent_strain_steps_are_ambiguous():
    strain = np.r_[np.full(120, 30.), np.full(180, 20.), np.full(250, 10.)]
    detected = detect_strain_step(hourly(len(strain)), strain, StrainStepConfig())
    assert detected.status == "ambiguous"
    assert detected.selected is not None


@pytest.mark.parametrize("after", [([0, 40], "variance"), ([30, 40, 50], "upward")])
def test_variance_increase_or_upward_shift_does_not_trigger(after):
    values, _ = after
    ttff = np.r_[np.full(120, 20.), repeated(values, 280)]
    result = detect_drogue_loss("001", hourly(len(ttff)), ttff).result
    assert result.ttff_change_status != "clear"
    assert np.isnat(result.auto_drogue_loss_time)


def test_no_transition_and_insufficient_followup_are_distinct():
    stable = repeated([10, 30, 80, 150, 400], 360)
    none = detect_drogue_loss("001", hourly(360), stable).result
    assert none.ttff_change_status in {"none", "weak"}
    assert np.isnat(none.auto_drogue_loss_time)

    late = np.r_[repeated([10, 30, 80, 150, 400], 300), repeated([2, 4, 6, 8], 60)]
    insufficient = detect_drogue_loss("001", hourly(360), late).result
    assert insufficient.ttff_change_status == "insufficient_followup"
    assert insufficient.auto_status == "insufficient_followup"
    assert np.isnat(insufficient.auto_drogue_loss_time)


def test_multiple_comparable_persistent_changes_are_ambiguous():
    ttff = np.r_[
        repeated([10, 30, 80, 150, 400], 100), repeated([2, 4, 6, 8], 140),
        repeated([10, 30, 80, 150, 400], 100), repeated([2, 4, 6, 8], 260),
    ]
    result = detect_drogue_loss("001", hourly(len(ttff)), ttff).result
    assert result.ttff_change_status == "ambiguous"
    assert result.auto_status == "ambiguous_component_changes"
    assert np.isnat(result.auto_drogue_loss_time)


def test_clear_event_is_stable_to_modest_bin_and_window_changes():
    n, event = 420, 144
    ttff = np.r_[
        repeated([10, 30, 80, 150, 400], event),
        repeated([2, 4, 6, 8], n - event),
    ]
    base = DrogueDetectionConfig().ttff
    alternative = replace(
        base,
        binning=replace(base.binning, factor=1.7),
        comparison=replace(base.comparison, pre_window_hours=60, post_window_hours=60),
    )
    first = detect_distribution_change(hourly(n), ttff, base)
    second = detect_distribution_change(hourly(n), ttff, alternative)
    assert first.status == second.status == "clear"
    assert abs(pd.Timestamp(first.selected.time) - pd.Timestamp(second.selected.time)) <= pd.Timedelta("12h")


def test_strain_is_primary_and_earlier_sustained_ttff_corroborates_without_date_averaging():
    n, ttff_event, strain_event = 420, 96, 132
    ttff = np.r_[
        repeated([10, 30, 80, 150, 400], ttff_event),
        repeated([2, 4, 6, 8], n - ttff_event),
    ]
    strain = np.r_[
        repeated([18, 20, 22, 24], strain_event),
        repeated([8, 10, 12, 14], n - strain_event),
    ]
    result = detect_drogue_loss("001", hourly(n), ttff, strain=strain).result
    assert result.ttff_change_status == result.strain_change_status == "clear"
    assert result.ttff_change_time < result.strain_change_time
    assert result.ttff_strain_relation == "corroborating"
    assert result.auto_drogue_loss_time == result.strain_change_time
    assert result.auto_status == "detected_strain_primary_corroborated"


def test_missing_strain_uses_medium_confidence_ttff_fallback():
    n, event = 360, 120
    ttff = np.r_[
        repeated([10, 30, 80, 150, 400], event),
        repeated([2, 4, 6, 8], n - event),
    ]
    result = detect_drogue_loss("001", hourly(n), ttff, strain=None).result
    assert result.strain_change_status == "unavailable"
    assert result.auto_drogue_loss_time == result.ttff_change_time
    assert result.auto_confidence == "medium"


def test_strain_step_detector_does_not_change_ttff_component_result():
    n, event = 360, 120
    ttff = np.r_[
        repeated([10, 30, 80, 150, 400], event),
        repeated([2, 4, 6, 8], n - event),
    ]
    direct = detect_distribution_change(hourly(n), ttff, DrogueDetectionConfig().ttff)
    combined = detect_drogue_loss(
        "001", hourly(n), ttff, strain=np.r_[
            np.full(event, 20.), np.full(n - event, 10.),
        ],
    ).result
    assert combined.ttff_change_status == direct.status
    assert combined.ttff_change_time == direct.selected.time.to_datetime64()
    assert combined.ttff_d_down == pytest.approx(direct.selected.d_down)
    assert combined.ttff_d_up == pytest.approx(direct.selected.d_up)


def test_configuration_validation_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="factor"):
        ValueBinningConfig("log", 10, 1, None, 1000, 12)
    with pytest.raises(ValueError, match="width"):
        ValueBinningConfig("linear", 0, None, None, 60, 12)
    with pytest.raises(ValueError, match="include_lower_than_first_bin"):
        ValueBinningConfig("linear", 0, None, 2, 60, 12, "false")
    with pytest.raises(ValueError, match="downward_fraction"):
        DistributionComparisonConfig(48, 48, 72, 24, .75, .3, .5, .2, 48, .8)
    with pytest.raises(ValueError, match="minimum_drop_absolute"):
        StrainStepConfig(minimum_drop_absolute=.5, weak_drop_absolute=1)
    with pytest.raises(ValueError, match="persistence_window"):
        StrainStepConfig(persistence_window="0h")
    with pytest.raises(ValueError, match="tail.upper_edge"):
        SignalDistributionConfig(
            ValueBinningConfig("log", 10, 1.5, None, 100, 12),
            DrogueDetectionConfig().ttff.comparison,
            TailSupportConfig(50, 1000),
        )


def test_analysis_cutoff_is_separate_from_physical_event():
    event = np.datetime64("2025-02-03T18:00:00", "ns")
    cutoff = analysis_cutoff_time(event, 24)
    assert event == np.datetime64("2025-02-03T18:00:00", "ns")
    assert cutoff == np.datetime64("2025-02-02T18:00:00", "ns")
