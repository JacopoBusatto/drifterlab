from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from drifterlab.qc.drogue import (
    ComponentDetectionResult,
    DrogueDetectionConfig,
    DrogueCombinationConfig,
    TTFFCessationComparisonConfig,
    TTFFCessationConfig,
    ValueBinningConfig,
    analysis_cutoff_time,
    combine_drogue_detections,
    detect_drogue_loss,
    detect_ttff_cessation,
    histogram_counts,
    make_value_bins,
    resolve_drogue_decision,
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


def component(hour: int | None, status: str) -> ComponentDetectionResult:
    value = (
        np.datetime64("NaT", "ns") if hour is None
        else np.datetime64("2025-01-01") + np.timedelta64(hour, "h")
    )
    return ComponentDetectionResult(value, status)


@pytest.mark.parametrize("offset", [24, 48])
def test_clear_component_agreement_chooses_earlier_date(offset):
    combined = combine_drogue_detections(
        component(240, "clear"), component(240 + offset, "clear"),
        DrogueCombinationConfig(agreement_tolerance_hours=48),
    )
    assert combined.status == "clear_agreement"
    assert combined.source == "ttff+strain"
    assert combined.change_time == component(240, "clear").change_time


def test_clear_component_dates_outside_tolerance_are_a_conflict():
    combined = combine_drogue_detections(
        component(240, "clear"), component(289, "clear"),
    )
    assert combined.status == "signal_conflict"
    assert combined.source == "none"
    assert np.isnat(combined.change_time)


@pytest.mark.parametrize(
    "ttff_status,strain_status,expected_status,expected_source,expected_hour",
    [
        ("clear", "no_change", "clear_ttff_only", "ttff", 240),
        ("clear", "insufficient_data", "clear_ttff_only", "ttff", 240),
        ("no_change", "clear", "clear_strain_only", "strain", 300),
        ("insufficient_data", "clear", "clear_strain_only", "strain", 300),
        ("weak", "no_change", "unresolved", "none", None),
        ("ambiguous", "clear", "unresolved", "none", None),
        ("no_change", "insufficient_data", "unresolved", "none", None),
    ],
)
def test_combination_of_single_clear_and_unresolved_components(
    ttff_status, strain_status, expected_status, expected_source, expected_hour,
):
    combined = combine_drogue_detections(
        component(240 if ttff_status in {"clear", "weak", "ambiguous"} else None,
                  ttff_status),
        component(300 if strain_status in {"clear", "weak", "ambiguous"} else None,
                  strain_status),
    )
    assert combined.status == expected_status
    assert combined.source == expected_source
    if expected_hour is None:
        assert np.isnat(combined.change_time)
    else:
        assert combined.change_time == component(expected_hour, "clear").change_time


def test_production_integration_keeps_conflicting_component_dates_separate():
    time, ttff = binned_track(700, [240] * len(BIN_VALUES))
    hours = ((time - time[0]) / pd.Timedelta(hours=1)).astype(int)
    strain = np.where(hours < 300, 20.0, 10.0)
    result = detect_drogue_loss(
        "test", time, ttff, strain=strain,
        config=DrogueDetectionConfig(ttff=ttff_config()),
    ).result
    assert result.ttff_change_time == np.datetime64("2025-01-11T00:00:00")
    assert result.strain_change_time == np.datetime64("2025-01-13T12:00:00")
    assert np.isnat(result.auto_drogue_loss_time)
    assert result.auto_status == "signal_conflict"
    assert result.auto_source == "none"
    assert result.ttff_strain_offset_hours == 60


def test_ttff_is_automatic_when_strain_is_insufficient():
    time, ttff = binned_track(650, [240] * len(BIN_VALUES))
    result = detect_drogue_loss(
        "test", time, ttff,
        config=DrogueDetectionConfig(ttff=ttff_config()),
    ).result
    assert result.auto_status == "clear_ttff_only"
    assert result.auto_source == "ttff"
    assert result.auto_drogue_loss_time == result.ttff_change_time
    assert result.strain_status == "insufficient_data"


def test_no_valid_ttff_is_unavailable_and_keeps_public_counts_zero():
    time = hourly(400)
    detection = detect_drogue_loss(
        "test", time, np.full(400, -999.0), strain=np.full(400, 20.0),
        config=DrogueDetectionConfig(ttff=ttff_config()),
    )
    assert detection.result.ttff_status == "insufficient_data"
    assert detection.result.ttff_detail_status == "unavailable"
    assert detection.result.ttff_eligible_bin_count == 0
    assert detection.result.ttff_stable_drop_bin_count == 0


def test_ttff_configuration_validation_and_unknown_sections():
    with pytest.raises(ValueError, match="multiple"):
        TTFFCessationConfig(
            ValueBinningConfig("linear", 100, None, 50, 300, 10),
            TTFFCessationComparisonConfig(),
        )
    with pytest.raises(ValueError, match="Unknown detection.ttff keys"):
        DrogueDetectionConfig.from_dict({"ttff": {"unknown": {}}})


def test_analysis_cutoff_remains_distinct_from_the_detected_event_time():
    event = np.datetime64("2025-01-10T12:00:00")
    assert analysis_cutoff_time(event, 24) == np.datetime64("2025-01-09T12:00:00")


def test_resolver_uses_automatic_when_no_human_review_exists():
    automatic = {"auto_drogue_loss_time": "2025-01-10T12:00:00Z"}
    resolved = resolve_drogue_decision(automatic, default_margin_hours=24)
    assert resolved.final_drogue_status == "lost"
    assert resolved.final_drogue_loss_time == np.datetime64("2025-01-10T12:00:00")
    assert resolved.analysis_cutoff_time == np.datetime64("2025-01-09T12:00:00")
    assert resolved.decision_source == "automatic"


def test_resolver_human_decision_overrides_automatic_and_uses_custom_margin():
    automatic = {"auto_drogue_loss_time": "2025-01-10T12:00:00Z"}
    review = {
        "review_status": "manual_date",
        "reviewed_drogue_loss_time": "2025-01-12T06:00:00Z",
        "analysis_cutoff_margin_hours": 30,
    }
    resolved = resolve_drogue_decision(automatic, review)
    assert resolved.final_drogue_loss_time == np.datetime64("2025-01-12T06:00:00")
    assert resolved.analysis_cutoff_time == np.datetime64("2025-01-11T00:00:00")
    assert resolved.decision_source == "manual_date"


@pytest.mark.parametrize("status", ["not_lost", "uncertain"])
def test_resolver_keeps_no_loss_and_uncertain_distinct(status):
    resolved = resolve_drogue_decision(
        {"auto_drogue_loss_time": "2025-01-10T12:00:00Z"},
        {
            "review_status": status,
            "reviewed_drogue_loss_time": "",
            "analysis_cutoff_margin_hours": 18,
        },
    )
    assert resolved.final_drogue_status == status
    assert np.isnat(resolved.final_drogue_loss_time)
    assert np.isnat(resolved.analysis_cutoff_time)


def test_resolver_leaves_unresolved_automatic_case_uncertain():
    resolved = resolve_drogue_decision({"auto_drogue_loss_time": pd.NaT})
    assert resolved.final_drogue_status == "uncertain"
    assert resolved.decision_source == "automatic"
