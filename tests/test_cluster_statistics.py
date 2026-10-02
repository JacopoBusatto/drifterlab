"""Analytical tests for within-array cluster statistics."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import CRS, Transformer

from drifterlab.postprocessing.cluster_statistics import (
    calculate_array_cluster_statistics,
)
from drifterlab.postprocessing.config import (
    ClusterStatisticsConfig,
    ClusterStatisticsPlottingConfig,
    ClusterVelocityConfig,
    TimeWindowConfig,
)
from drifterlab.postprocessing.trajectories import ClusterArrayTrajectoryData


def _config(*, stop: bool = False, interval: float = 20) -> ClusterStatisticsConfig:
    return ClusterStatisticsConfig(
        enabled=True,
        time=TimeWindowConfig(np.datetime64("NaT", "ns"), np.datetime64("NaT", "ns")),
        stop_on_member_loss=stop,
        percentiles=(0.0, 25.0, 50.0, 75.0, 100.0),
        velocity=ClusterVelocityConfig(interval, 10, None),
        plotting=ClusterStatisticsPlottingConfig(50, "linear"),
        array_overrides={},
    )


def _data(
    x: np.ndarray, y: np.ndarray, cluster_ids: list[str], cluster_sizes: list[int],
    *, array_id: int = 1,
) -> ClusterArrayTrajectoryData:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape
    times = (
        np.datetime64("2025-01-01T00:00", "m")
        + np.arange(x.shape[1]) * np.timedelta64(10, "m")
    ).astype("datetime64[ns]")
    definition = (
        "+proj=aeqd +lat_0=20 +lon_0=130 +datum=WGS84 "
        "+units=m +no_defs +type=crs"
    )
    inverse = Transformer.from_crs(
        CRS.from_proj4(definition), CRS.from_epsg(4326), always_xy=True,
    )
    longitude = np.full(x.shape, np.nan)
    latitude = np.full(y.shape, np.nan)
    finite = np.isfinite(x) & np.isfinite(y)
    longitude[finite], latitude[finite] = inverse.transform(x[finite], y[finite])
    return ClusterArrayTrajectoryData(
        array_id=array_id,
        dataset_label="Synthetic",
        coordinate_method="linear",
        platform_ids=tuple(f"p{index + 1}" for index in range(x.shape[0])),
        cluster_ids=tuple(cluster_ids),
        cluster_sizes=np.asarray(cluster_sizes, dtype=np.int64),
        times=times,
        longitude=longitude,
        latitude=latitude,
        projection_origin_longitude=130.0,
        projection_origin_latitude=20.0,
    )


def test_velocity_displacement_and_extreme_percentiles_are_analytical():
    result = calculate_array_cluster_statistics(
        _data([[0, 10, 20]], [[0, 0, 0]], ["singleton"], [1]),
        _config(interval=20),
    )
    frame = result.timeseries
    np.testing.assert_allclose(frame.mean_velocity_u_m_s, 1 / 60, rtol=1e-8)
    np.testing.assert_allclose(frame.mean_velocity_v_m_s, 0, atol=1e-10)
    np.testing.assert_allclose(frame.mean_absolute_displacement_m, [0, 10, 20], atol=1e-7)
    np.testing.assert_allclose(frame.speed_p000_m_s, frame.speed_p100_m_s)
    assert frame.valid_velocity_sample_count.tolist() == [1, 1, 1]
    assert frame.covariance_xx_m2.isna().all()


def test_velocity_does_not_bridge_an_internal_missing_coordinate():
    result = calculate_array_cluster_statistics(
        _data([[0, np.nan, 20]], [[0, np.nan, 0]], ["singleton"], [1]),
        _config(interval=20),
    )
    assert result.timeseries.valid_velocity_sample_count.tolist() == [0, 0, 0]
    assert result.timeseries.mean_platform_speed_m_s.isna().all()


def test_relative_dispersion_uses_vector_change_not_radial_change():
    result = calculate_array_cluster_statistics(
        _data(
            [[0, 0], [1, 0]],
            [[0, 0], [0, 1]],
            ["pair", "pair"],
            [2, 2],
        ),
        _config(),
    )
    frame = result.timeseries
    np.testing.assert_allclose(frame.mean_pair_separation_m, [1, 1], atol=1e-7)
    assert frame.relative_dispersion_D2_m2.iloc[0] == pytest.approx(0, abs=1e-12)
    assert frame.relative_dispersion_D2_m2.iloc[1] == pytest.approx(2, rel=1e-7)
    assert frame.relative_displacement_q_p000_m2.iloc[1] == pytest.approx(2, rel=1e-7)
    assert frame.valid_pair_count.tolist() == [1, 1]


def test_population_covariance_collinear_and_noncollinear_hulls():
    x = np.asarray([
        [-1], [0], [1],
        [0], [2], [0],
    ])
    y = np.asarray([
        [0], [0], [0],
        [0], [0], [2],
    ])
    result = calculate_array_cluster_statistics(
        _data(
            x, y,
            ["line", "line", "line", "triangle", "triangle", "triangle"],
            [3, 3, 3, 3, 3, 3],
        ),
        _config(),
    )
    line = result.timeseries[result.timeseries.cluster_id == "line"].iloc[0]
    triangle = result.timeseries[result.timeseries.cluster_id == "triangle"].iloc[0]
    assert line.covariance_xx_m2 == pytest.approx(2 / 3, rel=1e-7)
    assert line.lambda_major_m2 == pytest.approx(2 / 3, rel=1e-7)
    assert line.lambda_minor_m2 == pytest.approx(0, abs=1e-12)
    assert line.radius_of_gyration_m ** 2 == pytest.approx(
        line.lambda_major_m2 + line.lambda_minor_m2, rel=1e-12,
    )
    assert line.aspect_ratio == pytest.approx(0, abs=1e-8)
    assert line.convex_hull_area_m2 == pytest.approx(0, abs=1e-8)
    assert triangle.lambda_minor_m2 > 0
    assert triangle.convex_hull_area_m2 == pytest.approx(2, rel=1e-7)


def test_convex_hull_area_ratio_uses_first_finite_admitted_hull():
    result = calculate_array_cluster_statistics(
        _data(
            [[0, 0], [1, 2], [0, 0]],
            [[0, 0], [0, 0], [1, 2]],
            ["triangle"] * 3,
            [3] * 3,
        ),
        _config(),
    )
    frame = result.timeseries
    np.testing.assert_allclose(frame.convex_hull_area_m2, [.5, 2], rtol=1e-7)
    np.testing.assert_allclose(frame.convex_hull_area_ratio, [1, 4], rtol=1e-7)
    summary = result.summary.iloc[0]
    assert summary.convex_hull_reference_time.endswith("00:00:00Z")
    assert summary.convex_hull_reference_area_m2 == pytest.approx(.5, rel=1e-7)
    assert summary.mean_convex_hull_area_ratio == pytest.approx(2.5, rel=1e-7)


def test_convex_hull_area_ratio_is_undefined_for_zero_reference_area():
    result = calculate_array_cluster_statistics(
        _data(
            [[0, 0], [1, 1], [2, 0]],
            [[0, 0], [0, 0], [0, 1]],
            ["cluster"] * 3,
            [3] * 3,
        ),
        _config(),
    )
    frame = result.timeseries
    assert frame.convex_hull_area_m2.iloc[0] == pytest.approx(0, abs=1e-8)
    assert frame.convex_hull_area_m2.iloc[1] == pytest.approx(.5, rel=1e-7)
    assert frame.convex_hull_area_ratio.isna().all()
    assert result.summary.iloc[0].convex_hull_reference_area_m2 == pytest.approx(
        0, abs=1e-8,
    )


def test_equal_extent_does_not_imply_equal_shape_and_isotropic_orientation_is_nan():
    aligned_x = [-1, -1 / 3, 1 / 3, 1]
    radius_component = np.sqrt(5 / 18)
    square_x = [-radius_component, -radius_component, radius_component, radius_component]
    square_y = [-radius_component, radius_component, -radius_component, radius_component]
    result = calculate_array_cluster_statistics(
        _data(
            np.asarray([[value] for value in aligned_x + square_x]),
            np.asarray([[0] for _ in aligned_x] + [[value] for value in square_y]),
            ["aligned"] * 4 + ["isotropic"] * 4,
            [4] * 8,
        ),
        _config(),
    )
    rows = result.timeseries.set_index("cluster_id")
    assert rows.loc["aligned", "radius_of_gyration_m"] == pytest.approx(
        rows.loc["isotropic", "radius_of_gyration_m"], rel=1e-7,
    )
    assert rows.loc["aligned", "aspect_ratio"] == pytest.approx(0, abs=1e-8)
    assert rows.loc["isotropic", "aspect_ratio"] == pytest.approx(1, rel=1e-7)
    assert np.isnan(rows.loc["isotropic", "major_axis_orientation_deg"])


def test_dynamic_and_stopped_membership_counts_and_no_restart():
    x = np.asarray([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [np.nan, 2, 2, np.nan, 2],
    ])
    y = np.zeros_like(x)
    dynamic = calculate_array_cluster_statistics(
        _data(x, y, ["cluster"] * 3, [3] * 3), _config(stop=False),
    )
    assert dynamic.timeseries.active_member_count.tolist() == [2, 3, 3, 2, 3]
    assert dynamic.timeseries.valid_pair_count.tolist() == [1, 3, 3, 1, 3]
    assert dynamic.timeseries.analysis_included.tolist() == [True] * 5

    stopped = calculate_array_cluster_statistics(
        _data(x, y, ["cluster"] * 3, [3] * 3), _config(stop=True),
    )
    frame = stopped.timeseries
    assert frame.analysis_included.tolist() == [False, True, True, False, False]
    assert frame.loc[~frame.analysis_included, "centroid_x_m"].isna().all()
    summary = stopped.summary.iloc[0]
    assert summary.analysis_truncated
    assert summary.termination_reason == "member_loss"
    assert summary.analysis_start_time.endswith("00:10:00Z")
    assert summary.analysis_end_time.endswith("00:20:00Z")


def test_never_complete_cluster_is_auditable_and_warns():
    x = np.asarray([[0, np.nan], [np.nan, 1]])
    y = np.asarray([[0, np.nan], [np.nan, 0]])
    with pytest.warns(RuntimeWarning, match="never simultaneously available"):
        result = calculate_array_cluster_statistics(
            _data(x, y, ["pair"] * 2, [2, 2]), _config(stop=True),
        )
    assert not result.timeseries.analysis_included.any()
    assert result.timeseries.active_member_count.tolist() == [1, 1]
    assert result.summary.iloc[0].termination_reason == "complete_cluster_never_available"
    assert not result.summary.iloc[0].analysis_truncated
