"""Scientific contract tests for the direct v2 FSLE analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Geod

from drifterlab.analysis.fsle import resolve_fsle, run_fsle_analysis
from drifterlab.fsle import (
    aggregate_fsle_populations,
    collect_observed_first_passages,
    fit_fsle_power_laws,
)


def _times(count: int) -> np.ndarray:
    return np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(count) * np.timedelta64(5, "m")


def _passages(distances, *, policy="strict_observed", thresholds=None, **kwargs):
    return collect_observed_first_passages(
        times=_times(len(distances)), distance_km=np.asarray(distances),
        thresholds_km=np.asarray(thresholds if thresholds is not None else [1, 2, 4, 8, 16, 32, 64, 128]),
        expected_interval_seconds=300, policy=policy,
        pair_id="a-b", array_id=1, population="within_cluster", cluster_id="A",
        platform_code_1="a", platform_code_2="b", **kwargs,
    )


@pytest.mark.parametrize("policy,expected", [
    ("strict_observed", [1, 32]), ("bridge_missing", [1, 2, 32]),
])
def test_discussed_gap_example_only_contributes_witnessed_shells(policy, expected):
    distances = [0.9, 1.1, 1.2, 1.4, 2.6, np.nan, np.nan, np.nan, 19.5, 25, 31.5, 32.1, 45, 64.5]
    rows = _passages(distances, policy=policy).set_index("scale_lower_km")
    assert rows.index[rows.status == "reached"].tolist() == expected
    assert rows.loc[1, "entry_distance_km"] == 1.1
    assert rows.loc[1, "exit_distance_km"] == 2.6
    assert rows.loc[1, "log_growth"] == pytest.approx(np.log(2.6 / 1.1))
    assert rows.loc[1, "passage_time_days"] == pytest.approx(15 / 1440)
    assert rows.loc[16, "status"] == "not_observed_entry"
    assert rows.loc[32, "entry_distance_km"] == 32.1
    if policy == "bridge_missing":
        assert rows.loc[2, "log_growth"] == pytest.approx(np.log(19.5 / 2.6))
        assert rows.loc[2, "passage_time_days"] == pytest.approx(20 / 1440)
        assert rows.loc[2, "missing_observation_count"] == 3
    else:
        assert rows.loc[2, "status"] == "right_censored_gap"
        assert pd.isna(rows.loc[2, "log_growth"])


def test_overshoot_exit_is_next_entry_and_exact_upper_reaches():
    rows = _passages([0.9, 1.1, 2.6, 4, 8], thresholds=[1, 2, 4, 8]).set_index("scale_lower_km")
    assert (rows.status == "reached").all()
    assert rows.loc[1, "shell_exit_time"] == rows.loc[2, "shell_entry_time"]
    assert rows.loc[1, "exit_distance_km"] == rows.loc[2, "entry_distance_km"]
    assert rows.loc[2, "shell_exit_time"] == rows.loc[4, "shell_entry_time"]


def test_initial_partial_shell_skipped_shell_and_incomplete_passage_do_not_contribute():
    rows = _passages([1.1, 8.5, 9, 17], thresholds=[1, 2, 4, 8, 16, 32]).set_index("scale_lower_km")
    assert rows.loc[1, "status"] == "not_observed_entry"
    assert rows.loc[2, "status"] == "not_observed_entry"
    assert rows.loc[4, "status"] == "not_observed_entry"
    assert rows.loc[8, "status"] == "reached"
    assert rows.loc[16, "status"] == "right_censored_track_end"
    assert pd.isna(rows.loc[16, "passage_time_days"])


@pytest.mark.parametrize("policy", ["strict_observed", "bridge_missing"])
def test_null_gap_has_no_extra_source_limit_but_finite_limit_is_hard(policy):
    distances = [0.9, 1.1, 1.5, 2.2]
    unrestricted = _passages(distances, policy=policy, thresholds=[1, 2], source_gap_minutes=np.array([0, 0, 60, 0]))
    assert unrestricted.iloc[0].status == "reached"
    restricted = _passages(
        distances, policy=policy, thresholds=[1, 2],
        source_gap_minutes=np.array([0, 0, 60, 0]), maximum_source_gap_minutes=30,
    )
    assert restricted.iloc[0].status == "right_censored_gap"
    assert restricted.iloc[0].termination_reason == "source_gap_limit"


def test_bridge_elapsed_gap_limit_and_strict_no_retry_of_interrupted_shell():
    rows = _passages(
        [0.9, 1.1, np.nan, np.nan, 2.2], policy="bridge_missing", thresholds=[1, 2],
        source_gap_minutes=np.zeros(5), maximum_source_gap_minutes=10,
    )
    assert rows.iloc[0].termination_reason == "elapsed_gap_limit"
    rows = _passages([0.9, 1.1, np.nan, 0.9, 1.2, 2.3], thresholds=[1, 2])
    assert rows.iloc[0].status == "right_censored_gap"


def test_bridge_accepts_short_missing_gap_with_nan_source_metadata():
    rows = _passages(
        [0.9, 1.1, np.nan, 2.2], policy="bridge_missing", thresholds=[1, 2],
        source_gap_minutes=np.array([0, 0, np.nan, 0]), maximum_source_gap_minutes=10,
    )
    assert rows.iloc[0].status == "reached"
    assert rows.iloc[0].maximum_observation_gap_minutes == 10


def test_pooled_estimator_uses_passages_and_keeps_populations_separate():
    first = _passages([0.9, 1.1, 2.4], thresholds=[1, 2])
    second = _passages([0.9, 1.2, 1.4, 1.7, 2.5], thresholds=[1, 2])
    second["pair_id"], second["cluster_id"] = "c-d", "B"
    chance = first.copy()
    chance["population"], chance["pair_id"], chance["cluster_id"] = "cross_cluster", "b-c", None
    table = aggregate_fsle_populations(
        pd.concat([first, second, chance]), np.array([1, 2]),
        array_id=1, cluster_ids=["A", "B", "singleton"], minimum_reached_pairs=1,
    )
    pooled = table[table.scope == "within_cluster_pooled"].iloc[0]
    expected = (np.log(2.4 / 1.1) + np.log(2.5 / 1.2)) / ((5 + 15) / 1440)
    assert pooled.fsle_day_inverse == pytest.approx(expected)
    assert pooled.reached_pair_count == 2
    assert table[table.scope == "cross_cluster"].iloc[0].reached_pair_count == 1
    assert table[table.cluster_id == "singleton"].iloc[0].total_pair_count == 0


def test_fixed_power_fits_use_entire_supported_shells_and_si_coefficients():
    lower = np.array([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
    power, coefficient = -2 / 3, 0.002
    spectra = pd.DataFrame({
        "array_id": 1, "scope": "within_cluster_pooled", "cluster_id": None,
        "scale_lower_km": lower, "scale_upper_km": lower * 2,
        "fsle_day_inverse": coefficient * (lower * 1000) ** power * 86400,
        "plot_included": True,
    })
    # A corrupted but unsupported scale must not affect the fitted coefficient.
    spectra.loc[2, "fsle_day_inverse"] *= 100
    spectra.loc[2, "plot_included"] = False
    fits = [
        {"id": "small", "power": power, "scale_min_km": 0.1, "scale_max_km": 1},
        {"id": "large", "power": power, "scale_min_km": 10, "scale_max_km": 64},
        {"id": "shear", "power": -1.0, "scale_min_km": 0.1, "scale_max_km": 1},
        {"id": "empty", "power": -1.0, "scale_min_km": 100, "scale_max_km": 200},
    ]
    table = fit_fsle_power_laws(spectra, fits).set_index("fit_id")
    assert table.loc["small", "used_scale_lower_km"] == [0.125, 0.5]
    assert table.loc["large", "used_scale_lower_km"] == [16, 32]
    assert table.loc["small", "coefficient_si"] == pytest.approx(coefficient)
    assert table.loc["large", "coefficient_si"] == pytest.approx(coefficient)
    assert table.loc["small", "coefficient_cubed_m2_s3"] == pytest.approx(coefficient**3)
    assert table.loc["small", "log_rmse"] == pytest.approx(0, abs=1e-14)
    assert table.loc["shear", "log_rmse"] > 0
    assert pd.isna(table.loc["shear", "coefficient_cubed_m2_s3"])
    assert table.loc["empty", "status"] == "insufficient_shells"


@pytest.mark.parametrize("raw,match", [
    ({"chance_pair_threshold_km": None}, "positive|numeric"),
    ({"maximum_source_gap_minutes": 0}, "positive"),
    ({"passage_policy": "guess"}, "passage_policy"),
    ({"fits": [{"id": "test", "power": "-2/3", "scale_min_km": 1, "scale_max_km": 10}]}, "numeric"),
    ({"fits": [{"id": "test", "power": -1, "scale_min_km": 10, "scale_max_km": 1}]}, "exceed"),
    ({"plotting": {"y_range_day_inverse": [1, 0.1]}}, "exceed"),
    ({"time": {"start": "2026-02-01", "end": "2026-01-01"}}, "follow"),
    ({"unknown": True}, "Unknown"),
])
def test_invalid_fsle_configuration_is_rejected(raw, match):
    with pytest.raises(ValueError, match=match):
        resolve_fsle(raw)


def _dataset() -> xr.Dataset:
    distances = np.array([0.5, 0.8, 1.1, 1.5, 2.2, 3, 4.5, 6, 8.5]) * 1000
    geod = Geod(ellps="WGS84")
    lon, lat, _ = geod.fwd(np.zeros(len(distances)), np.zeros(len(distances)), np.full(len(distances), 90.0), distances)
    return xr.Dataset({
        "platform_id": ("platform", ["a", "b", "c", "far"]),
        "array_id": ("platform", [1, 1, 1, 2]),
        "cluster_id": ("platform", ["A", "A", "B", "outside"]),
        "longitude": (("platform", "time"), np.array([np.zeros(len(lon)), lon, -lon, np.ones(len(lon))])),
        "latitude": (("platform", "time"), np.array([np.zeros(len(lat)), lat, lat, np.zeros(len(lat))])),
        "source_gap_minutes": (("platform", "time"), np.zeros((4, len(lon)))),
    }, coords={"time": _times(len(lon))}, attrs={"selected_trajectory_product": "qc_5min"})


def test_direct_analysis_separates_populations_selects_chance_minimum_and_shares_axes(tmp_path: Path):
    settings = resolve_fsle({
        "enabled": True, "rho": 2, "minimum_scale_km": 0.5, "maximum_scale_km": 16,
        "chance_pair_threshold_km": 0.75, "minimum_reached_pairs_per_scale": 1,
        "fits": [
            {"id": "richardson", "power": -2 / 3, "scale_min_km": 1, "scale_max_km": 8},
            {"id": "shear", "power": -1, "scale_min_km": 1, "scale_max_km": 8},
        ], "plotting": {"dpi": 30},
    })
    dataset = _dataset()
    before = dataset.copy(deep=True)
    result = run_fsle_analysis(dataset, settings, tmp_path, cadence_minutes=5)
    xr.testing.assert_identical(dataset, before)
    assert result["selected_pair_counts"] == {"within_cluster": 1, "cross_cluster": 1}
    selection = pd.read_csv(tmp_path / "array_001/tables/pair_selection.csv")
    cross = selection[selection.population == "cross_cluster"]
    assert cross.selected.sum() == 1
    assert cross[cross.selected].iloc[0].minimum_distance_km == pytest.approx(0.5)
    assert not (tmp_path / "pairs.zarr").exists()
    assert not list(tmp_path.rglob("*.zarr"))
    spectra = pd.read_csv(tmp_path / "array_001/tables/fsle_spectra.csv")
    assert set(spectra.scope) == {"within_cluster_pooled", "cross_cluster", "individual_cluster"}
    assert spectra[spectra.cluster_id == "B"].total_pair_count.eq(0).all()
    metadata = json.loads((tmp_path / "fsle_metadata.json").read_text())
    records = metadata["figures"]["records"]
    # Four spectra in array 1, three in singleton array 2, each in three forms.
    assert len(records) == 21
    assert len({tuple(row["x_range_km"]) for row in records}) == 1
    for power in (None, -2 / 3, -1):
        assert len({tuple(row["y_range"]) for row in records if row["power"] == power}) == 1
    assert len(list(tmp_path.rglob("*.png"))) == 21


def test_chance_pairs_start_at_earliest_global_minimum_in_selected_interval(tmp_path: Path):
    dataset = _dataset().isel(platform=[0, 2], time=[0, 1, 2, 3, 4, 5, 6, 7, 8]).copy(deep=True)
    dataset["longitude"].values[1] = [0.03, 0.02, 0.01, 0.005, 0.005, 0.01, 0.02, 0.03, 0.04]
    settings = resolve_fsle({
        "enabled": True, "chance_pair_threshold_km": 1,
        "time": {"start": "2026-01-01T00:05:00Z", "end": "2026-01-01T00:40:00Z"},
        "plotting": {"enabled": False},
    })
    run_fsle_analysis(dataset, settings, tmp_path, cadence_minutes=5)
    selection = pd.read_csv(tmp_path / "array_001/tables/pair_selection.csv")
    assert len(selection) == 1 and selection.iloc[0].selected
    assert pd.Timestamp(selection.iloc[0].analysis_start_time) == pd.Timestamp("2026-01-01T00:15:00")
    metadata = json.loads((tmp_path / "fsle_metadata.json").read_text())
    assert pd.Timestamp(metadata["search_interval"]["last_sample_utc"]) == pd.Timestamp("2026-01-01T00:35:00")


def test_empty_time_interval_fails_before_publishing(tmp_path: Path):
    settings = resolve_fsle({"time": {"start": "2030-01-01"}})
    with pytest.raises(ValueError, match="no trajectory samples"):
        run_fsle_analysis(_dataset(), settings, tmp_path, cadence_minutes=5)


def test_pair_track_end_is_distinguished_from_internal_gap(tmp_path: Path):
    dataset = _dataset().isel(platform=[0, 1]).copy(deep=True)
    # The pair enters [1,2), but its common lifetime ends before reaching 2 km.
    dataset["longitude"].values[1, 4:] = np.nan
    dataset["latitude"].values[1, 4:] = np.nan
    settings = resolve_fsle({
        "enabled": True, "rho": 2, "minimum_scale_km": 1, "maximum_scale_km": 4,
        "plotting": {"enabled": False},
    })
    run_fsle_analysis(dataset, settings, tmp_path, cadence_minutes=5)
    passages = pd.read_parquet(tmp_path / "array_001/data/first_passages.parquet")
    row = passages[passages.scale_lower_km == 1].iloc[0]
    assert row.status == "right_censored_track_end"
    assert row.missing_observation_count == 0
    assert row.censor_time == pd.Timestamp(_times(4)[-1])
