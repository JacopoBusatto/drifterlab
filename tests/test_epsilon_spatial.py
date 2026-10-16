from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from drifterlab.analysis.spatial import (
    resolve_spatial_maps,
    run_spatial_maps,
    spatial_pair_observations,
)
from drifterlab.workflows.epsilon import VelocitySpec


class _IdentityTransform:
    def transform(self, x, y):
        return x, y


def _observations(*, between="array", maximum=150, reverse=False, sample_steps=1):
    # a/b are a cluster. c first appears at t=1; it starts near b but far
    # from a, then moves much further away. The initial-distance decision
    # must remain fixed despite the requested window starting at t=2.
    x = np.array([
        [0, 0, 0, 0, 0, 0],
        [1000, 1000, 1000, 1000, 1000, 1000],
        [np.nan, 1100, 1300, 1500, 1700, 1900],
    ], dtype=float)
    u = np.array([[0] * 6, [1] * 6, [-1] * 6], dtype=float)
    valid = np.isfinite(x)
    valid[1, 2] = False  # Missing samples must not shift the UTC stride.
    ids = np.array(["a", "b", "c"])
    clusters = np.array(["cluster_1", "cluster_1", "cluster_2"])
    if reverse:
        x, u, valid, ids, clusters = x[::-1], u[::-1], valid[::-1], ids[::-1], clusters[::-1]
    return spatial_pair_observations(
        platform_ids=ids, cluster_ids=clusters,
        times=np.datetime64("2025-01-01") + np.arange(6) * np.timedelta64(1, "h"),
        projection=SimpleNamespace(x_m=x, y_m=x * 0, inverse=_IdentityTransform()),
        velocity=SimpleNamespace(u_m_s=u, v_m_s=u * 0, valid=valid),
        reference_index=0, sample_steps=sample_steps,
        windows=[{"id": "later", "start_hour": 2, "end_hour": 5}],
        between=between, max_initial_distance_m=maximum,
        minimum_separation_m=1, maximum_separation_m=10000,
    )


def test_initial_distance_filters_only_cross_cluster_pairs_and_remains_fixed():
    observations, inventory = _observations()
    selected = inventory[inventory.selected]
    assert set(zip(selected.platform_id_1, selected.platform_id_2)) == {("a", "b"), ("b", "c")}
    rejected = inventory[~inventory.selected].iloc[0]
    assert (rejected.platform_id_1, rejected.platform_id_2) == ("a", "c")
    assert rejected.selection_status == "above_maximum_initial_distance"
    cross = inventory[inventory.platform_id_1 == "b"].iloc[0]
    assert cross.initial_separation_m == 100
    assert cross.initial_time_utc == "2025-01-01T01:00:00Z"
    evolving = observations[observations.platform_id_2 == "c"]
    assert set(evolving.separation_m) == {500, 700}
    assert (evolving.separation_m > 150).all()
    assert (evolving.q_m2_s3 > 0).all()
    assert (observations[observations.platform_id_2 == "b"].q_m2_s3 < 0).all()
    assert set(observations.array_age_hours) == {3, 4}
    np.testing.assert_allclose(evolving.midpoint_x_m, [1250, 1350])


def test_cluster_mode_ignores_initial_distance_and_array_null_includes_all_pairs():
    clustered, clustered_inventory = _observations(between="cluster", maximum=1)
    assert len(clustered_inventory) == 1
    assert clustered_inventory.selected.all()
    assert set(clustered.platform_id_2) == {"b"}
    all_pairs, all_inventory = _observations(maximum=None)
    assert len(all_inventory) == 3
    assert all_inventory.selected.all()
    assert all_pairs.pair_id.nunique() == 3


def test_schedule_stride_is_shared_and_does_not_reset_after_gaps_or_at_window_start():
    observations, _ = _observations(maximum=None)
    displayed = observations[observations.schedule_index % 2 == 0]
    assert set(displayed.array_age_hours) == {2, 4}
    assert len(displayed[displayed.array_age_hours == 2]) == 1
    assert len(displayed[displayed.array_age_hours == 4]) == 3
    # Analysis cadence is separate from display stride and has the same origin.
    sampled, _ = _observations(maximum=None, sample_steps=2)
    assert set(sampled.array_age_hours) == {2, 4}
    assert set(sampled.schedule_index) == {1, 2}


def test_pair_population_and_values_do_not_depend_on_platform_order():
    forward, _ = _observations(maximum=None)
    backward, _ = _observations(maximum=None, reverse=True)
    assert forward.equals(backward)


def test_spatial_config_resolves_windows_and_fixed_color_limits():
    epsilon = {
        "windows": [{"id": "later", "start_hour": 2, "end_hour": 5}],
        "q_distribution": {"maximum_absolute_m2_s3": 0.2, "linear_threshold_m2_s3": 1e-6},
    }
    resolved = resolve_spatial_maps({"enabled": True, "windows": ["later"]}, epsilon)
    assert resolved["windows"] == epsilon["windows"]
    assert resolved["color"] == epsilon["q_distribution"]
    assert resolved["couples"]["between"] == "cluster"
    custom = resolve_spatial_maps({"windows": [{"id": "custom", "start_hour": 0, "end_hour": 2}]}, epsilon)
    assert custom["windows"][0]["end_hour"] == 2


@pytest.mark.parametrize("settings,match", [
    ({"resample_stride": 0}, "resample_stride"),
    ({"resample_stride": True}, "resample_stride"),
    ({"resample_stride": 1.5}, "resample_stride"),
    ({"couples": {"between": "unknown"}}, "between"),
    ({"couples": {"max_initial_distance_m": -1}}, "max_initial_distance"),
    ({"color": {"linear_threshold_m2_s3": 1}}, "less than"),
    ({"color": {"maximum_absolute_m2_s3": float("nan")}}, "finite"),
    ({"windows": ["missing"]}, "Unknown"),
    ({"windows": [{"id": "../escape", "start_hour": 0, "end_hour": 1}]}, "path-safe"),
    ({"windows": [{"id": "zero", "start_hour": 1, "end_hour": 1}]}, "exceed"),
    ({"windows": [{"id": "bool", "start_hour": False, "end_hour": 1}]}, "numeric"),
    ({"windows": []}, "nonempty"),
    ({"unknown_setting": True}, "Unknown"),
])
def test_spatial_config_rejects_invalid_controls(settings, match):
    with pytest.raises(ValueError, match=match):
        resolve_spatial_maps(settings, {})


def test_empty_spatial_run_anchors_to_first_array_entry_before_cluster_selection(tmp_path):
    times = np.datetime64("2025-01-01") + np.arange(6) * np.timedelta64(1, "h")
    # The unselected cluster has the first finite entry at 01:00 UTC.
    # The selected singleton starts at 02:00 UTC and has no internal pairs.
    longitude = np.array([[np.nan, 10, 10.1, 10.2, 10.3, 10.4],
                          [np.nan, np.nan, 10.5, 10.6, 10.7, 10.8]])
    zeros = np.zeros_like(longitude)
    dataset = xr.Dataset({
        "array_id": ("platform", [3, 3]),
        "cluster_id": ("platform", ["early", "late"]),
        "platform_id": ("platform", ["a", "b"]),
        "longitude": (("platform", "time"), longitude),
        "latitude": (("platform", "time"), longitude * 0 + 50),
        "source_gap_minutes": (("platform", "time"), zeros + 60),
        "eastward_velocity": (("platform", "time"), zeros),
        "northward_velocity": (("platform", "time"), zeros),
        "velocity_dt_seconds": (("platform", "time"), zeros + 3600),
    }, coords={"time": times})
    epsilon = SimpleNamespace(
        arrays=(3,), clusters=("late",), sample_cadence_minutes=60,
        analysis=VelocitySpec("product", "centered_difference", 120, 120, True, None),
        minimum_reliable_separation_m=None, bins=SimpleNamespace(edges_m=(1, 10000)),
        dataset_label="Empty diagnostic",
    )
    settings = resolve_spatial_maps({
        "enabled": True, "dpi": 30,
        "windows": [{"id": "first_hour", "start_hour": 0, "end_hour": 1}],
    }, {})
    summary = run_spatial_maps(dataset, epsilon, settings, tmp_path, progress=lambda _: None)
    assert summary == {"enabled": True, "figure_count": 2, "observation_count": 0}
    figures = pd.read_csv(tmp_path / "array_003/spatial_figures.csv")
    assert set(figures.reference_time_utc) == {"2025-01-01T01:00:00+00:00"}
    assert set(figures.window_end_utc) == {"2025-01-01T02:00:00+00:00"}
    assert len(pd.read_parquet(tmp_path / "array_003/data/spatial_pair_observations.parquet")) == 0
    assert pd.read_csv(tmp_path / "array_003/spatial_pair_selection.csv").empty
