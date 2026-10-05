"""Analytical tests for overshoot-aware FSLE calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from drifterlab.fsle import (
    REACHED,
    RIGHT_CENSORED_GAP,
    aggregate_fsle_spectra,
    build_scale_thresholds,
    collect_pair_first_passages,
)


def _passages(
    pair_id: str,
    cluster_id: str,
    distances: list[float],
    days: list[int],
) -> pd.DataFrame:
    times = np.asarray(
        [np.datetime64("2026-01-01", "D") + np.timedelta64(day, "D") for day in days],
        dtype="datetime64[ns]",
    )
    return collect_pair_first_passages(
        pair_id=pair_id,
        array_id=1,
        cluster_id=cluster_id,
        platform_code_1=f"{pair_id}_1",
        platform_code_2=f"{pair_id}_2",
        times=times,
        distance_km=np.asarray(distances),
        thresholds_km=np.asarray([1.0, 2.0]),
        expected_interval_seconds=86400,
    )


def test_scale_thresholds_are_anchored_exactly_at_one_km():
    thresholds = build_scale_thresholds(0.2, 5, math.sqrt(2), 1)
    assert np.count_nonzero(thresholds == 1.0) == 1
    np.testing.assert_allclose(thresholds[1:] / thresholds[:-1], math.sqrt(2))
    assert thresholds[0] >= 0.2
    assert thresholds[-1] <= 5


def test_overshoot_passage_uses_observed_entry_exit_and_actual_time():
    row = _passages("p", "A", [0.8, 1.2, 2.5], [0, 1, 2]).iloc[0]
    assert row.status == REACHED
    assert row.entry_distance_km == pytest.approx(1.2)
    assert row.exit_distance_km == pytest.approx(2.5)
    assert row.passage_time_days == pytest.approx(1)
    assert row.log_growth == pytest.approx(math.log(2.5 / 1.2))


def test_upper_crossing_must_follow_shell_entry():
    row = _passages("p", "A", [2.5, 1.5, 1.4, 2.4], [0, 1, 2, 3]).iloc[0]
    assert row.status == REACHED
    assert row.shell_entry_time == np.datetime64("2026-01-02")
    assert row.shell_exit_time == np.datetime64("2026-01-04")
    assert row.passage_time_days == pytest.approx(2)


def test_missing_coordinate_censors_passage_and_prevents_restart():
    row = _passages("p", "A", [1.2, np.nan, 2.5], [0, 1, 2]).iloc[0]
    assert row.status == RIGHT_CENSORED_GAP
    assert pd.isna(row.shell_exit_time)
    assert pd.isna(row.passage_time_days)
    assert row.termination_reason == "missing_coordinate_or_time"


def test_pooled_spectrum_combines_pair_events_not_cluster_spectra():
    pair_a = _passages("a", "A", [1.2, 2.4], [0, 1])
    pair_b = _passages("b", "B", [1.1, 1.5, 1.7, 2.5], [0, 1, 2, 3])
    passages = pd.concat([pair_a, pair_b], ignore_index=True)
    spectrum = aggregate_fsle_spectra(
        passages,
        np.asarray([1.0, 2.0]),
        array_ids=[1],
        minimum_reached_pairs=1,
    )
    pooled = spectrum[spectrum.scope == "same_cluster_pooled"].iloc[0]
    expected = (math.log(2) + math.log(2.5 / 1.1)) / 2
    expected /= (1 + 3) / 2
    assert pooled.fsle_day_inverse == pytest.approx(expected)
    log_growth = np.asarray([math.log(2), math.log(2.5 / 1.1)])
    passage_time = np.asarray([1.0, 3.0])
    variance = (
        np.mean(np.square(log_growth) / passage_time) / passage_time.mean()
        - expected**2
    )
    assert pooled.fsle_standard_error_day_inverse == pytest.approx(
        math.sqrt(variance / 2)
    )
    assert pooled.total_pair_count == 2
    assert pooled.reached_pair_count == 2
    assert set(spectrum[spectrum.scope == "individual_cluster"].cluster_id) == {
        "A",
        "B",
    }


def test_standard_error_reduces_to_pdf_formula_for_fixed_shell_growth():
    pair_a = _passages("a", "A", [1.2, 2.4], [0, 1])
    pair_b = _passages("b", "A", [1.1, 1.5, 1.7, 2.2], [0, 1, 2, 3])
    spectrum = aggregate_fsle_spectra(
        pd.concat([pair_a, pair_b], ignore_index=True),
        np.asarray([1.0, 2.0]),
        array_ids=[1],
        minimum_reached_pairs=1,
    )
    pooled = spectrum[spectrum.scope == "same_cluster_pooled"].iloc[0]
    passage_time = np.asarray([1.0, 3.0])
    mean_time = passage_time.mean()
    variance_inverse_time = (np.mean(1 / passage_time) * mean_time - 1) / mean_time**2
    expected = math.log(2) * math.sqrt(variance_inverse_time / len(passage_time))
    assert pooled.fsle_standard_error_day_inverse == pytest.approx(expected)


def test_standard_error_is_missing_for_one_reached_passage():
    spectrum = aggregate_fsle_spectra(
        _passages("a", "A", [1.2, 2.4], [0, 1]),
        np.asarray([1.0, 2.0]),
        array_ids=[1],
        minimum_reached_pairs=1,
    )
    assert spectrum.fsle_standard_error_day_inverse.isna().all()
