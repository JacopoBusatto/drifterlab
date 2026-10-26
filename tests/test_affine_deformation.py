from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from drifterlab.analysis.affine import (
    calculate_array,
    endpoint_positions,
    fit_affine,
    resolve_affine,
    run_affine_analysis,
    schedule,
)
from drifterlab.analysis.affine_sst import PreparedSST, resolve_sst
from drifterlab.trajectories.kinematics import project_array_positions


def settings(tmp_path: Path, **overrides) -> dict:
    return resolve_affine({"enabled": True, **overrides}, tmp_path)


INITIAL = np.array([[-100.0, -80.0], [120.0, -70.0], [110.0, 100.0], [-90.0, 95.0]])


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(2),
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.diag([2.0, 0.5]),
        np.array([[1.0, 0.7], [0.0, 1.0]]),
    ],
)
def test_known_maps_translation_axes_and_coordinate_rotation(tmp_path, matrix):
    q = settings(tmp_path)["quality"]
    final = INITIAL @ matrix.T + [1600, -2500]
    result = fit_affine(INITIAL, final, 6, q)
    assert result["numerically_valid"]
    f = np.array([[result[f"f_{i}{j}"] for j in (1, 2)] for i in (1, 2)])
    c = np.array([[result[f"cauchy_green_{i}{j}"] for j in (1, 2)] for i in (1, 2)])
    np.testing.assert_allclose(f, matrix, atol=1e-14)
    np.testing.assert_allclose(c, matrix.T @ matrix, atol=1e-14)
    np.testing.assert_allclose(
        [result[f"stretch_{i}"] for i in (1, 2)],
        np.linalg.svd(matrix, compute_uv=False),
        atol=1e-14,
    )
    for i in (1, 2):
        e = np.array([result[f"initial_axis_{i}_x"], result[f"initial_axis_{i}_y"]])
        np.testing.assert_allclose(c @ e, result[f"cauchy_green_eigenvalue_{i}"] * e, atol=1e-14)
        assert result[f"rate_h_inverse_{i}"] == pytest.approx(
            np.log(result[f"stretch_{i}"]) / 6, abs=1e-14
        )
    shifted = fit_affine(INITIAL + [500, 300], final - [100, 900], 6, q)
    assert shifted["f_12"] == pytest.approx(result["f_12"], abs=1e-14)
    angle = 0.61
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = fit_affine(INITIAL @ rotation.T, final @ rotation.T, 6, q)
    rotated_c = np.array([[rotated[f"cauchy_green_{i}{j}"] for j in (1, 2)] for i in (1, 2)])
    np.testing.assert_allclose(rotated_c, rotation @ c @ rotation.T, atol=1e-13)
    if np.allclose(c, np.eye(2)):
        assert result["isotropic"] and not result["axis_displayable"]
        assert result["rate_h_inverse_1"] == pytest.approx(0, abs=1e-14)


def test_geometry_residuals_reflection_collapse_and_three_members(tmp_path):
    q = settings(tmp_path)["quality"]
    collinear = np.column_stack((np.arange(4), np.zeros(4)))
    assert fit_affine(collinear, collinear, 6, q)["reason"] == "rank_deficient_initial_geometry"
    slender = INITIAL * [1, 0.01]
    poor = fit_affine(slender, slender @ np.diag([2, 0.5]), 6, q)
    assert poor["numerically_valid"] and not poor["axis_displayable"]
    assert "poor_initial_geometry" in poor["quality_flags"]
    reflected = fit_affine(INITIAL, INITIAL * [-1, 1], 6, q)
    assert reflected["determinant"] < 0 and not reflected["axis_displayable"]
    assert "orientation_reversal" in reflected["quality_flags"]
    collapse = fit_affine(INITIAL, np.zeros_like(INITIAL), 6, q)
    assert collapse["numerically_valid"] and not collapse["axis_displayable"]
    assert collapse["rate_h_inverse_1"] == -np.inf
    assert np.isnan(collapse["normalized_residual"])
    three = fit_affine(INITIAL[:3], INITIAL[:3] * [2, 0.5], 6, q)
    assert "three_members_no_residual_degrees_of_freedom" in three["quality_flags"]
    assert three["rmse_m"] < 1e-12
    final = INITIAL * [2, 0.5]
    final[0] += [20, 50]
    residual = fit_affine(INITIAL, final, 6, q)
    assert residual["rmse_m"] > 0
    assert residual["rmse_m"] ** 2 * 4 == pytest.approx(residual["residual_sum_squares_m2"])
    assert residual["normalized_residual"] == pytest.approx(
        residual["rmse_m"] / residual["final_rms_radius_m"]
    )


def test_schedule_analysis_bounds_midnight_and_cutoff(tmp_path):
    table = schedule("2025-02-01T02:00Z", "2025-02-02T10:00Z", 6)
    assert pd.Timestamp(table.fit_start_utc.iloc[0]).hour == 6
    assert pd.Timestamp(table.fit_end_utc.iloc[-1]).hour == 6
    assert len(table) == 4
    assert table.map_start_utc.iloc[-1] == "2025-02-02T00:00:00+00:00"
    assert schedule("2025-02-01T02:00Z", "2025-02-01T07:00Z", 6).empty
    for value in (0, -1, 7, 25):
        with pytest.raises(ValueError):
            settings(tmp_path, deformation_window_hours=value)
    assert settings(tmp_path, deformation_window_hours=12)["deformation_window_hours"] == 12
    assert settings(tmp_path, deformation_window_hours=24)["deformation_window_hours"] == 24


def test_endpoint_interpolation_source_provenance_no_extrapolation_and_cutoff():
    times = pd.date_range("2025-02-01", periods=4, freq="10min").values
    lon = np.array([[179.8, 179.9, -180.0, -179.9]])
    lat = np.ones_like(lon) * 70
    gaps = np.array([[0, 30, 30, 0.0]])
    targets = pd.to_datetime(["2025-02-01T00:05", "2025-02-01T00:15", "2025-02-01T00:40"]).values
    out, _, support, _ = endpoint_positions(times, lon, lat, gaps, targets, 30)
    np.testing.assert_allclose(out[0, :2], [179.85, 179.95])
    assert np.isnan(out[0, 2]) and support[0, 0] == 30
    limited, _, _, _ = endpoint_positions(times, lon, lat, gaps, targets, 15)
    assert np.isnan(
        limited
    ).all()  # Cannot treat already interpolated product positions as exact fixes.
    cutoff, _, _, _ = endpoint_positions(
        times, lon, lat, gaps, targets[:2], 30, cutoff="2025-02-01T00:15Z"
    )
    assert np.isfinite(cutoff[0, 0]) and np.isnan(cutoff[0, 1])


def array_fixture() -> xr.Dataset:
    times = pd.date_range("2025-02-01", periods=49, freq="h").values
    projection = project_array_positions(np.array([[-150.0]]), np.array([[70.0]]))
    offsets = np.vstack([INITIAL, INITIAL + [2500, 1500]])
    elapsed = np.arange(len(times))
    x = offsets[:, 0, None] * np.exp(elapsed[None, :] * 0.015) + 40 * elapsed
    y = offsets[:, 1, None] * np.exp(-elapsed[None, :] * 0.01) + 20 * elapsed
    lon, lat = projection.inverse.transform(x, y)
    return xr.Dataset(
        {
            "longitude": (("platform", "time"), lon),
            "latitude": (("platform", "time"), lat),
            "source_gap_minutes": (("platform", "time"), np.zeros_like(lon)),
            "array_id": ("platform", [1] * 8),
            "cluster_id": ("platform", ["cluster_001"] * 4 + ["cluster_002"] * 4),
            "cluster_size": ("platform", [4] * 8),
            "platform_id": ("platform", [f"p{i}" for i in range(8)]),
        },
        coords={"time": times},
    )


def test_lifecycle_endpoint_only_gaps_later_activation_and_original_member_end(tmp_path):
    array = array_fixture()
    array.longitude.values[0, [3, 6]] = np.nan
    array.latitude.values[0, [3, 6]] = np.nan
    array.longitude.values[0, 31:] = np.nan
    array.latitude.values[0, 31:] = np.nan
    array.longitude.values[4:, :9] = np.nan
    array.latitude.values[4:, :9] = np.nan
    rows, _members, meta, _ = calculate_array(array, settings(tmp_path), 60)
    first = rows[rows.cluster_id == "cluster_001"].reset_index(drop=True)
    assert not first.numerically_valid.iloc[0]  # Missing 06:00 endpoint.
    assert not first.numerically_valid.iloc[1]
    assert first.numerically_valid.iloc[2]  # Restarts after temporary gap.
    assert first.numerically_valid.iloc[4]  # Ends exactly at earliest member end.
    assert first.reason.iloc[5] == "original_member_usable_trajectory_ended"
    second = rows[rows.cluster_id == "cluster_002"].reset_index(drop=True)
    assert second.reason.iloc[1] == "before_cluster_activation"
    assert second.numerically_valid.iloc[2]
    assert meta["analysis_start_utc"] == "2025-02-01T00:00:00+00:00"
    selected, _, selected_meta, _ = calculate_array(array, settings(tmp_path), 60, ("cluster_002",))
    assert selected_meta["analysis_start_utc"] == meta["analysis_start_utc"]
    assert set(selected.cluster_id) == {"cluster_002"}
    limited, _, limited_meta, _ = calculate_array(array, settings(tmp_path), 60, ("cluster_001",))
    assert limited_meta["analysis_end_utc"] == "2025-02-02T06:00:00+00:00"
    assert len(limited) == 5  # Null Tmax follows this cluster's usable end, not product padding.
    array.longitude.values[0, 6], array.latitude.values[0, 6] = (
        array_fixture().longitude.values[0, 6],
        array_fixture().latitude.values[0, 6],
    )
    rows, _, _, _ = calculate_array(array, settings(tmp_path), 60)
    assert rows.numerically_valid.iloc[0]  # Interior 03:00 gap does not invalidate endpoint fit.


def write_sst(path: Path, day="2025-02-01", *, units="degree_C", zero=False) -> Path:
    values = np.full((1, 7, 9), 20.0, dtype=np.float32)
    for i in range(4):
        values[0, i, i] = np.nan
    if zero:
        values[:] = np.nan
    start = pd.Timestamp(day).value // 1_000_000_000
    ds = xr.Dataset(
        {
            "sea_surface_temperature": (("time", "lat", "lon"), values),
            "time_bounds": (("time", "nv"), [[start, start + 86400]]),
            "l2p_flags": (("time", "lat", "lon"), np.full(values.shape, -1, dtype=np.int16)),
        },
        coords={
            "time": [start + 43200],
            "lat": np.linspace(70.03, 69.97, 7),
            "lon": np.linspace(209.96, 210.04, 9),
        },
        attrs={"title": "Already quality-masked daily SST"},
    )
    ds.time.attrs.update(
        units="seconds since 1970-01-01 00:00:00 UTC", calendar="standard", bounds="time_bounds"
    )
    ds.sea_surface_temperature.attrs["units"] = units
    ds.to_netcdf(path)
    return path


def test_prepared_sst_units_coordinates_zero_coverage_and_errors(tmp_path):
    path = write_sst(tmp_path / "daily.nc")
    config = resolve_sst({"enabled": True, "file": str(path)}, tmp_path)
    reader = PreparedSST(config)
    projection = project_array_positions(np.array([[-150.0]]), np.array([[70.0]]))
    product, meta = reader.load(
        pd.Timestamp("2025-02-01", tz="UTC"), [-4000, 4000, -4000, 4000], projection
    )
    assert meta["status"] == "sst_available"
    assert np.isnan(product.sst_celsius.values).sum() == 4
    np.testing.assert_allclose(
        product.sst_celsius.values[np.isfinite(product.sst_celsius.values)], 20
    )
    assert np.all(np.diff(product.latitude) > 0) and np.all(product.longitude < 0)
    assert meta["coverage_fraction"] == pytest.approx(59 / 63)
    assert "contributing_bin_count" in product
    assert meta["aggregation"] == "supplied_bin_unchanged"
    missing, meta = reader.load(
        pd.Timestamp("2025-02-02", tz="UTC"), [-4000, 4000, -4000, 4000], projection
    )
    assert missing is None and meta["status"] == "sst_unavailable"
    zero = write_sst(tmp_path / "zero.nc", zero=True)
    product, meta = PreparedSST(resolve_sst({"enabled": True, "file": str(zero)}, tmp_path)).load(
        pd.Timestamp("2025-02-01", tz="UTC"), [-4000, 4000, -4000, 4000], projection
    )
    assert meta["coverage_fraction"] == 0 and not np.isfinite(product.sst_celsius.values).any()
    bad = write_sst(tmp_path / "bad.nc", units="unknown")
    with pytest.raises(ValueError, match="Celsius"):
        PreparedSST(resolve_sst({"enabled": True, "file": str(bad)}, tmp_path))
    with pytest.raises(ValueError, match="Unknown"):
        PreparedSST(
            resolve_sst(
                {"enabled": True, "file": str(path), "variables": {"sst": "absent"}}, tmp_path
            )
        )


def test_end_to_end_optional_sst_independence_and_replot(tmp_path):
    from drifterlab.plotting.affine_deformation import replot_array

    array = array_fixture()
    config = settings(tmp_path, plotting={"dpi": 45, "fixed_extent": False})
    result = run_affine_analysis(
        array, config, tmp_path / "without", selected_clusters="all", product_maximum_gap_minutes=60
    )
    assert result["numerically_valid_fit_count"] == 16 and result["figure_count"] == 2
    sst = write_sst(tmp_path / "sst.nc")
    config = settings(
        tmp_path,
        plotting={"dpi": 45, "fixed_extent": False},
        sst={"enabled": True, "file": str(sst)},
    )
    run_affine_analysis(
        array, config, tmp_path / "with", selected_clusters="all", product_maximum_gap_minutes=60
    )
    frames = [
        pd.read_parquet(tmp_path / name / "array_001/data/affine_intervals.parquet")
        for name in ("with", "without")
    ]
    pd.testing.assert_frame_equal(
        frames[0].drop(columns="sst_status"), frames[1].drop(columns="sst_status")
    )
    assert set(frames[0].sst_status) == {"sst_available", "sst_unavailable"}
    metadata = json.loads((tmp_path / "with/array_001/metadata.json").read_text())
    assert metadata["maximum_source_gap_minutes"] == 60
    assert metadata["windows"][0]["extent_m"] != metadata["extent_m"]
    assert metadata["windows"][0]["extent_m"] != metadata["windows"][1]["extent_m"]
    original_figure = (tmp_path / "with/array_001/figures/20250201.png").read_bytes()
    assert replot_array(tmp_path / "with/array_001") == 2
    assert (tmp_path / "with/array_001/figures/20250201.png").read_bytes() == original_figure
    assert (tmp_path / "with/array_001/replots/20250201.png").exists()


def test_unified_affine_only_ignores_velocity_grid(tmp_path):
    from test_unified_analysis import _write_config, _write_v2_store

    from drifterlab.analysis import load_analysis_config, run_analysis

    store = _write_v2_store(tmp_path / "input.zarr")
    raw = {
        "input": {"trajectory_zarr": str(store), "trajectory_product": "qc_5min"},
        "selection": {"arrays": [3]},
        "kinematics": {"total_span_minutes": 1},
        "analyses": {
            "affine_deformation": {
                "enabled": True,
                "deformation_window_hours": 0.5,
                "plotting": {"enabled": False},
            }
        },
        "output": {"root": str(tmp_path / "output")},
    }
    config = _write_config(tmp_path / "config.yml", raw)
    assert load_analysis_config(config).affine_deformation["enabled"]
    result = run_analysis(config)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["modules"]["affine_deformation"]["attempted_fit_count"] == 4
    assert "06_affine_deformation/array_003/data/affine_intervals.parquet" in manifest["products"]


def test_off_grid_full_pipeline_and_empty_or_ineligible_clusters(tmp_path):
    array = array_fixture().isel(platform=slice(0, 4))
    array = array.assign_coords(time=array.time.values + np.timedelta64(30, "m"))
    config = settings(tmp_path, maximum_elapsed_hours=19, plotting={"enabled": False})
    rows, _, meta, _ = calculate_array(array, config, 60)
    assert meta["analysis_start_utc"] == "2025-02-01T00:30:00+00:00"
    assert len(rows) == 2 and rows.numerically_valid.all()  # 06–12 and 12–18, both off-grid.
    pair = array.isel(platform=slice(0, 2)).copy()
    pair.cluster_size.values[:] = 2
    run_affine_analysis(
        pair, config, tmp_path / "pair", selected_clusters="all", product_maximum_gap_minutes=60
    )
    membership = pd.read_csv(tmp_path / "pair/array_001/data/original_membership.csv")
    assert membership.eligibility.iloc[0] == "fewer_than_three_original_members"
    explicit = settings(
        tmp_path,
        reference_time_utc="2025-02-01T00:30Z",
        maximum_elapsed_hours=19,
        plotting={"enabled": False},
    )
    result = run_affine_analysis(
        pair,
        explicit,
        tmp_path / "explicit",
        selected_clusters="all",
        product_maximum_gap_minutes=60,
    )
    assert result["attempted_fit_count"] == 2
    invalid = pd.read_parquet(tmp_path / "explicit/array_001/data/affine_intervals.parquet")
    assert invalid.f_11.isna().all() and invalid.cauchy_green_11.isna().all()
    with pytest.raises(ValueError, match="membership"):
        calculate_array(array.assign(cluster_size=("platform", [3, 3, 3, 3])), config, 60)


def test_fixed_and_adaptive_views_leave_affine_results_identical(tmp_path):
    from drifterlab.plotting.affine_deformation import map_extent

    array = array_fixture()
    frames, manifests = [], []
    for fixed in (True, False):
        root = tmp_path / str(fixed)
        config = settings(tmp_path, plotting={"enabled": False, "fixed_extent": fixed})
        run_affine_analysis(
            array, config, root, selected_clusters="all", product_maximum_gap_minutes=60
        )
        frames.append(pd.read_parquet(root / "array_001/data/affine_intervals.parquet"))
        manifests.append(json.loads((root / "array_001/metadata.json").read_text()))
    pd.testing.assert_frame_equal(*frames)
    fixed, adaptive = manifests
    assert all(window["extent_m"] == fixed["extent_m"] for window in fixed["windows"])
    assert adaptive["windows"][0]["extent_m"] != adaptive["windows"][1]["extent_m"]
    assert map_extent(pd.DataFrame(), adaptive, {"fixed_extent": False}) == adaptive["extent_m"]
    isolated = frames[0].iloc[:1]
    view = map_extent(isolated, adaptive, {"fixed_extent": False, "axis_length_km": 10})
    assert view[1] - view[0] >= 11000 and view[3] - view[2] >= 11000
    with pytest.raises(ValueError, match="true or false"):
        settings(tmp_path, plotting={"fixed_extent": "false"})


@pytest.mark.parametrize(
    "plot",
    [
        {"longitude_bounds": [135, 125]},
        {"longitude_bounds": [-181, 135]},
        {"longitude_bounds": [125, 181]},
        {"longitude_bounds": [125, float("nan")]},
        {"longitude_bounds": [125, float("inf")]},
        {"longitude_bounds": [True, 135]},
        {"longitude_bounds": ["125", 135]},
        {"longitude_bounds": [125]},
        {"latitude_bounds": [30, 30]},
        {"latitude_bounds": [-91, 30]},
        {"latitude_bounds": [15, 91]},
        {"fixed_extent": False, "longitude_bounds": [125, 135]},
        {"fixed_extent": False, "latitude_bounds": [15, 30]},
    ],
)
def test_invalid_explicit_map_bounds(tmp_path, plot):
    with pytest.raises(ValueError, match="bounds"):
        settings(tmp_path, plotting=plot)


def test_explicit_bounds_maps_sst_and_replot_preserve_diagnostics(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from drifterlab.plotting.affine_deformation import geographic_extent, map_extent, replot_array

    array = array_fixture()
    baseline = settings(tmp_path, plotting={"enabled": False})
    expected, _, meta, _ = calculate_array(array, baseline, 60)
    automatic = geographic_extent(meta["extent_m"], meta, baseline["plotting"])
    for key, pair, positions in (
        ("longitude_bounds", [-150.02, -149.98], slice(0, 2)),
        ("latitude_bounds", [69.98, 70.02], slice(2, 4)),
    ):
        plot = settings(tmp_path, plotting={key: pair})["plotting"]
        extent = map_extent(pd.DataFrame(), meta, plot)
        bounds = geographic_extent(extent, meta, plot)
        wanted = automatic.copy()
        wanted[positions] = pair
        np.testing.assert_allclose(bounds, wanted, atol=1e-12)

    captured = []
    savefig = Figure.savefig

    def capture(fig, *args, **kwargs):
        captured.extend([list(ax.get_xlim()) + list(ax.get_ylim()) for ax in fig.axes[:2]])
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    config = settings(
        tmp_path,
        plotting={
            "dpi": 35, "fixed_extent": True,
            "longitude_bounds": [-150.02, -149.98], "latitude_bounds": [69.98, 70.02],
        },
        sst={"enabled": True, "file": str(write_sst(tmp_path / "sst.nc"))},
    )
    run_affine_analysis(
        array, config, tmp_path / "custom", selected_clusters="all", product_maximum_gap_minutes=60
    )
    root = tmp_path / "custom/array_001"
    actual = pd.read_parquet(root / "data/affine_intervals.parquet")
    pd.testing.assert_frame_equal(
        expected.drop(columns="sst_status"), actual.drop(columns="sst_status")
    )
    metadata = json.loads((root / "metadata.json").read_text())
    for window in metadata["windows"]:
        assert window["geographic_extent_deg"] == [-150.02, -149.98, 69.98, 70.02]
    with xr.open_dataset(root / "sst/20250201.nc") as subset:
        assert subset.longitude.min() >= -150.02 - 0.011
        assert subset.longitude.max() <= -149.98 + 0.011
        assert subset.latitude.min() >= 69.98 - 0.011
        assert subset.latitude.max() <= 70.02 + 0.011
        assert subset.sizes["longitude"] < 9
    assert replot_array(root) == 2
    assert len(captured) == 8  # Two panels, two days, original and replot.
    np.testing.assert_allclose(captured, [[-150.02, -149.98, 69.98, 70.02]] * 8, atol=1e-12)
