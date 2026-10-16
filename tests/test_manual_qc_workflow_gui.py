import importlib.util
import os
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import xarray as xr
import zarr
from scipy.io import savemat

import drifterlab.workflows.reconstruction as reconstruction_workflow
from drifterlab.io.position import read_microsvp_position_mat
from drifterlab.manual_qc.workflow import ManualQCWorkflow
from drifterlab.reconstruction import build_regular_grids
from drifterlab.workflows.drogue import AUTOMATIC_REQUIRED_COLUMNS
from drifterlab.workflows.manual_qc_zarr import (
    write_manual_qc_product_store,
)

GUI_TEST_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("PySide6", "pyqtgraph", "pytestqt")
)
if GUI_TEST_AVAILABLE:
    import pyqtgraph as pg

    from drifterlab.manual_qc.gui import ManualQCWindow, _set_finite_scatter_data


def manual_config(
    tmp_path: Path,
    *,
    final_zarr: bool = False,
    platform_codes: tuple[str, ...] = ("p",),
    selected_platforms: tuple[str, ...] | None = None,
) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    timestamps = pd.date_range("2025-01-01", periods=16, freq="5min", tz="UTC")
    rows = []
    for platform_index, code in enumerate(platform_codes):
        savemat(raw / f"{code}.mat", {
            "dataset": {
                f"drifter_{code}": {
                    "PlatformId": code,
                    "ObsTimestamp": np.asarray(
                        [value.isoformat() for value in timestamps], dtype=object,
                    ),
                    "GpsLongitude": (
                        platform_index * 0.1
                        + np.arange(len(timestamps), dtype=float) * 0.001
                    ),
                    "GpsLatitude": np.zeros(len(timestamps)),
                    "SST": 20.0 + np.arange(len(timestamps), dtype=float) * 0.1,
                    "GpsPosQual": 1.0 + np.arange(len(timestamps), dtype=float) % 5,
                    "GpsNumSat": 12.0 - np.arange(len(timestamps), dtype=float) % 6,
                },
            },
        })
        loaded = read_microsvp_position_mat(raw / f"{code}.mat")
        row = {name: np.nan for name in AUTOMATIC_REQUIRED_COLUMNS}
        row.update({
            "platform_code": code,
            "source_path": str(raw / f"{code}.mat"),
            "source_filename": f"{code}.mat",
            "source_sha256": loaded.source_sha256,
            "auto_drogue_loss_time": pd.Timestamp("2025-01-01T01:00:00Z"),
            "auto_status": "clear_agreement",
            "auto_source": "test",
            "default_analysis_cutoff_time": pd.Timestamp("2025-01-01T00:00:00Z"),
            "default_analysis_cutoff_margin_hours": 1.0,
        })
        rows.append(row)
    pd.DataFrame(rows).to_parquet(tmp_path / "drogue.parquet", index=False)
    config = tmp_path / "manual.yml"
    lines = [
            "schema_version: 2",
            "input:",
            "  reader: microsvp_mat",
            "  directory: raw",
            '  pattern: "*.mat"',
            "  options: {}",
            "drogue:",
            "  automatic: drogue.parquet",
            "  review: drogue_review.csv",
            "  stop_at: physical_loss",
            "output:",
            "  state: state/manual.sqlite",
            "  directory: output",
            "review:",
            "  chunk_hours: 0.5",
            "  maximum_accepted_span_days: 15",
            "  resample_frequency_seconds: 300",
            "  warnings:",
            "    high_speed_m_s: 0.2",
            "    zero_speed_tolerance_m_s: 0.0",
        "  cache_size: 1",
    ]
    if final_zarr:
        lines.extend([
            "final_zarr:",
            "  enabled: true",
            "  build_when_all_platforms_reviewed: true",
            "  output:",
            "    directory: final",
            "    chunks: {platform: 1, time: 8}",
            "  grid:",
            '    anchor_time: "2025-01-01T00:00:00Z"',
            "  products:",
            "    - {name: raw_5min, source: raw, cadence_minutes: 5, maximum_gap_minutes: 30}",
            "    - {name: qc_5min, source: qc, cadence_minutes: 5, maximum_gap_minutes: 30}",
            "    - {name: qc_15min, source: qc, cadence_minutes: 15, maximum_gap_minutes: 60}",
            "    - {name: qc_30min, source: qc, cadence_minutes: 30, maximum_gap_minutes: 120}",
            "  arrays: {maximum_adjacent_start_gap_hours: 24}",
            "  initial_clustering:",
            "    distance_reference: observed_starts",
            "    defaults:",
            "      maximum_start_time_difference_seconds: 3600",
            "      maximum_pair_distance_m: 1000",
            "      maximum_cluster_diameter_m: 2000",
            "      maximum_members: 5",
            "    array_overrides: {}",
        ])
    if selected_platforms is not None:
        selection_lines = [
            "selection:",
            "  platforms:",
            *(f'    - "{code}"' for code in selected_platforms),
        ]
        insertion = lines.index("drogue:")
        lines[insertion:insertion] = selection_lines
    config.write_text("\n".join(lines), encoding="utf-8")
    return config


def finalize_platform(workflow: ManualQCWorkflow, platform_code: str) -> Path:
    model = workflow.load_model(platform_code)
    for chunk in model.chunks:
        workflow.state.mark_chunk_reviewed(
            platform_code,
            model.source_sha256,
            chunk.key,
            start_time_ns=int(chunk.start_time.astype(np.int64)),
            end_time_ns=int(chunk.end_time.astype(np.int64)),
        )
    return workflow.publish(model)


def test_manual_workflow_selects_explicit_platform_subset(tmp_path):
    config = manual_config(
        tmp_path,
        platform_codes=("p", "q", "r"),
        selected_platforms=("q", "r"),
    )

    workflow = ManualQCWorkflow(config)
    try:
        assert tuple(
            platform.platform_code for platform in workflow.platforms
        ) == ("q", "r")
        assert tuple(workflow.platform_by_code) == ("q", "r")

        with pytest.raises(
            ValueError,
            match="Unknown manual-QC platform: p",
        ):
            workflow.load_model("p")
    finally:
        workflow.close()


@pytest.mark.parametrize(
    ("selected_platforms", "expected_message"),
    [
        ((), "selection.platforms must be 'all' or a nonempty list"),
        (("q", "q"), "selection.platforms contains duplicates"),
    ],
)
def test_manual_workflow_rejects_invalid_platform_selection(
    tmp_path,
    selected_platforms,
    expected_message,
):
    config = manual_config(
        tmp_path,
        platform_codes=("p", "q"),
        selected_platforms=selected_platforms,
    )

    with pytest.raises(ValueError, match=expected_message):
        ManualQCWorkflow(config)


def test_manual_workflow_rejects_unknown_selected_platform(tmp_path):
    config = manual_config(
        tmp_path,
        platform_codes=("p", "q"),
        selected_platforms=("missing",),
    )

    with pytest.raises(
        ValueError,
        match="Selected platforms are absent from the drogue product",
    ):
        ManualQCWorkflow(config)


def test_manual_workflow_requires_quoted_platform_ids(tmp_path):
    config = manual_config(
        tmp_path,
        platform_codes=("p", "q"),
        selected_platforms=("q",),
    )
    contents = config.read_text(encoding="utf-8")
    config.write_text(
        contents.replace('    - "q"', "    - 123"),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="selection.platforms must contain nonempty strings",
    ):
        ManualQCWorkflow(config)

@pytest.mark.skipif(not GUI_TEST_AVAILABLE, reason="manual-QC Qt test dependencies are absent")
def test_scatter_data_drops_nonfinite_coordinates(qtbot):
    item = pg.ScatterPlotItem()

    _set_finite_scatter_data(
        item,
        np.asarray([1.0, np.nan, 3.0, 4.0]),
        np.asarray([10.0, 20.0, np.inf, np.nan]),
    )
    x, y = item.getData()
    np.testing.assert_array_equal(x, [1.0])
    np.testing.assert_array_equal(y, [10.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _set_finite_scatter_data(item, [np.nan], [np.nan])
        assert item.dataBounds(0) == (None, None)
        assert item.dataBounds(1) == (None, None)


def test_independent_workflow_stops_at_physical_loss_and_publishes_audit(tmp_path):
    workflow = ManualQCWorkflow(manual_config(tmp_path))
    try:
        model = workflow.load_model("p")
        assert model.review_order == list(range(13))
        assert len(model.chunks) == 3
        workflow.state.set_rejected(
            "p", model.source_sha256, [4], rejected=True,
            dirty_chunk_keys=model.affected_chunk_keys({4}),
        )
        model.set_rejected([4], True)
        for chunk in model.chunks:
            workflow.state.mark_chunk_reviewed(
                "p", model.source_sha256, chunk.key,
                start_time_ns=None if np.isnat(chunk.start_time) else int(chunk.start_time.astype(np.int64)),
                end_time_ns=None if np.isnat(chunk.end_time) else int(chunk.end_time.astype(np.int64)),
            )
        destination = workflow.publish(model)
        track = reconstruction_workflow._load_track(destination)

        assert track.native_source_index.tolist() == list(range(13))

        rejected_native_index = int(np.flatnonzero(
            track.native_source_index == 4
        )[0])
        assert not track.native_position_accepted[rejected_native_index]
        assert (
            track.native_position_status[rejected_native_index]
            == "rejected"
        )

        accepted_native_index = int(np.flatnonzero(
            track.native_source_index == 3
        )[0])
        assert track.native_position_accepted[accepted_native_index]
        assert (
            track.native_position_status[accepted_native_index]
            == "accepted"
        )
        
        target = np.datetime64("2025-01-01T00:20:00", "ns")
        grids = build_regular_grids(
            track.time[0],
            track.time[0],
            track.time[-1],
        )
        products = reconstruction_workflow._build_track_products(
            track,
            grids,
        )

        raw_index = int(np.flatnonzero(
            products["raw_5min"].time == target
        )[0])
        qc_index = int(np.flatnonzero(
            products["qc_5min"].time == target
        )[0])

        # Source observation 4 exists exactly in raw but was manually rejected.
        assert (
            products["raw_5min"]
            .position.source_gap_minutes[raw_index]
            == 0
        )

        # QC reconstructs that time between observations 3 and 5.
        assert (
            products["qc_5min"]
            .position.source_gap_minutes[qc_index]
            == 10
        )
        # Then test that the same results survive the Zarr round trip.
        store_path = write_manual_qc_product_store(
            [track],
            tmp_path / "v2-products.zarr",
            anchor=np.datetime64("2025-01-01T00:00:00", "ns"),
            chunk_platform=1,
            chunk_time=8,
        )
        with xr.open_zarr(
            store_path,
            group="products/raw_5min",
            consolidated=True,
            chunks=None,
        ) as raw_product:
            raw_store_index = int(np.flatnonzero(
                raw_product.time.values == target
            )[0])
            assert (
                raw_product.source_gap_minutes
                .isel(platform=0, time=raw_store_index)
                .item()
                == 0
            )

        with xr.open_zarr(
            store_path,
            group="products/qc_5min",
            consolidated=True,
            chunks=None,
        ) as qc_product:
            qc_store_index = int(np.flatnonzero(
                qc_product.time.values == target
            )[0])
            assert (
                qc_product.source_gap_minutes
                .isel(platform=0, time=qc_store_index)
                .item()
                == 10
            )

        with xr.open_zarr(
            store_path,
            consolidated=True,
            chunks=None,
        ) as metadata_dataset:
            assert metadata_dataset.platform_id.values.tolist() == ["p"]
            assert metadata_dataset.deployment_start_time.values[0] == (
                np.datetime64("2025-01-01T00:00:00", "ns")
            )
            assert metadata_dataset.trajectory_end_time.values[0] == (
                np.datetime64("2025-01-01T01:00:00", "ns")
            )
            assert metadata_dataset.review_end_time.values[0] == (
                np.datetime64("2025-01-01T01:00:00", "ns")
            )
            assert metadata_dataset.review_end_reason.item() == "physical_loss"
            assert metadata_dataset.accepted_fix_count.item() == 12
            assert metadata_dataset.raw_fix_count.item() == 13
            assert metadata_dataset.requested_review_span_days.item() == 15
            assert metadata_dataset.accepted_review_span_days.item() == (
                pytest.approx(1 / 24)
            )

        table = pd.read_parquet(destination)
        assert len(table) == 16
        assert table.loc[table.source_obs_index == 4, "manual_qc_status"].item() == "rejected"
        assert "manual_sst_qc_status" in table
        assert "source_sst" in table
        assert "source_gps_pos_qual" in table
        assert "source_gps_num_sat" in table
        assert table.loc[
            table.source_obs_index == 4, "source_gps_pos_qual"
        ].item() == 5
        assert table.loc[
            table.source_obs_index == 4, "source_gps_num_sat"
        ].item() == 8
        assert table.loc[table.source_obs_index == 15, "manual_qc_status"].item() == "outside_review_scope"
        assert table.deployment_start_time.dropna().nunique() == 1
        assert table.review_end_time.dropna().nunique() == 1
        assert table.review_end_reason.unique().tolist() == ["physical_loss"]
        assert table.requested_review_span_days.unique().tolist() == [15.0]
        assert table.accepted_review_span_days.unique().item() == pytest.approx(1 / 24)
        metadata = pq.read_metadata(destination).metadata
        assert b"drifterlab_manual_qc" in metadata
    finally:
        workflow.close()


def test_final_zarr_uses_grouped_products_and_complete_assignments(tmp_path):
    workflow = ManualQCWorkflow(manual_config(tmp_path, final_zarr=True))
    try:
        model = workflow.load_model("p")
        workflow.state.set_rejected(
            "p", model.source_sha256, [4], rejected=True,
            dirty_chunk_keys=model.affected_chunk_keys({4}),
        )
        model.set_rejected([4], True)
        workflow.state.set_sst_rejected(
            "p", model.source_sha256, [6], rejected=True,
            dirty_chunk_keys=model.affected_sst_chunk_keys({6}),
        )
        model.set_sst_rejected([6], True)
        for chunk in model.chunks:
            workflow.state.mark_chunk_reviewed(
                "p", model.source_sha256, chunk.key,
                start_time_ns=int(chunk.start_time.astype(np.int64)),
                end_time_ns=int(chunk.end_time.astype(np.int64)),
            )
        workflow.publish(model)
        readiness = workflow.final_zarr_readiness()
        assert readiness.ready
        assert readiness.complete
        result = workflow.build_final_zarr()

        assert result.product_status == "final_complete_qc"
        assert result.platform_count == result.configured_platform_count == 1
        assert result.array_count == result.cluster_count == 1

        with xr.open_zarr(
            result.zarr_path,
            group="products/raw_5min",
            consolidated=True,
            chunks=None,
        ) as raw_product:
            assert raw_product.source_gap_minutes.sel(
                platform=0, time="2025-01-01T00:20:00",
            ).item() == 0
        with xr.open_zarr(
            result.zarr_path,
            group="products/qc_5min",
            consolidated=True,
            chunks=None,
        ) as qc_product:
            assert qc_product.source_gap_minutes.sel(
                platform=0, time="2025-01-01T00:20:00",
            ).item() == 10

        root = zarr.open_group(result.zarr_path, mode="r")
        assert set(root["products"].group_keys()) == {
            "raw_5min", "qc_5min", "qc_15min", "qc_30min",
        }
        assert root.attrs["product_status"] == "final_complete_qc"
        assert root.attrs["included_platform_ids"] == ["p"]
        assert int(root["array_id"][0]) == 1
        assert str(root["cluster_id"][0]) == "array_001__cluster_001"
        assert str(root["cluster_assignment_status"][0]) == "algorithmic_singleton"
        assert not any("spline" in name for name in root.array_keys())
        assert (result.zarr_path / "build_manifest.json").is_file()
        assert (result.zarr_path / "build_report.csv").is_file()
        assert (result.zarr_path / "cluster_pair_diagnostics.csv").is_file()
    finally:
        workflow.close()


def test_provisional_zarr_is_replaced_by_complete_build(tmp_path):
    workflow = ManualQCWorkflow(manual_config(
        tmp_path,
        final_zarr=True,
        platform_codes=("p", "q"),
    ))
    try:
        finalize_platform(workflow, "p")
        readiness = workflow.final_zarr_readiness()
        assert readiness.ready
        assert not readiness.complete
        assert (readiness.current_platforms, readiness.total_platforms) == (1, 2)

        provisional_plan = workflow.prepare_final_zarr_build()
        assert provisional_plan.product_status == "provisional_partial_qc"
        assert [item.platform_code for item in provisional_plan.inputs] == ["p"]
        provisional = workflow.build_final_zarr(plan=provisional_plan)
        assert provisional.product_status == "provisional_partial_qc"

        provisional_root = zarr.open_group(provisional.zarr_path, mode="r")
        assert provisional_root.attrs["included_platform_count"] == 1
        assert provisional_root.attrs["configured_platform_count"] == 2
        assert provisional_root.attrs["assignment_scope_status"] == "provisional"
        assert np.asarray(provisional_root["platform_id"][:]).astype(str).tolist() == [
            "p"
        ]

        finalize_platform(workflow, "q")
        readiness = workflow.final_zarr_readiness()
        assert readiness.ready and readiness.complete
        complete_plan = workflow.prepare_final_zarr_build()
        assert complete_plan.product_status == "final_complete_qc"
        complete = workflow.build_final_zarr(plan=complete_plan)
        assert complete.zarr_path == provisional.zarr_path
        assert complete.product_status == "final_complete_qc"

        complete_root = zarr.open_group(complete.zarr_path, mode="r")
        assert complete_root.attrs["included_platform_count"] == 2
        assert complete_root.attrs["configured_platform_count"] == 2
        assert complete_root.attrs["assignment_scope_status"] == "final"
        assert np.asarray(complete_root["platform_id"][:]).astype(str).tolist() == [
            "p", "q",
        ]
        assert np.all(np.asarray(complete_root["array_id"][:]) >= 1)
        assert all(
            np.asarray(complete_root["cluster_id"][:]).astype(str).tolist()
        )
        assert not list(tmp_path.glob(".final.*.tmp"))
        assert not list(tmp_path.glob(".final.*.backup"))
    finally:
        workflow.close()


@pytest.mark.skipif(not GUI_TEST_AVAILABLE, reason="manual-QC Qt test dependencies are absent")
def test_gui_offers_partial_build_and_auto_starts_complete_build(tmp_path, qtbot):
    workflow = ManualQCWorkflow(manual_config(
        tmp_path,
        final_zarr=True,
        platform_codes=("p", "q"),
    ))
    finalize_platform(workflow, "p")
    window = ManualQCWindow(workflow)
    qtbot.addWidget(window)

    window._refresh_final_zarr_status(auto_build=False)
    assert window.final_zarr_button.isEnabled()
    assert window.final_zarr_button.text() == "Build provisional Zarr"
    assert "1/2 drifters finalized" in window.final_zarr_label.text()
    assert "1 remain unreviewed" in window.final_zarr_label.text()

    finalize_platform(workflow, "q")
    starts = []
    window.start_final_zarr_build = lambda: starts.append("started")
    window._refresh_final_zarr_status(auto_build=True)
    assert window.final_zarr_button.text() == "Build final Zarr"
    assert "All 2 drifters reviewed" in window.final_zarr_label.text()
    assert starts == ["started"]
    window.close()


@pytest.mark.skipif(not GUI_TEST_AVAILABLE, reason="manual-QC Qt test dependencies are absent")
def test_gui_reject_restore_resample_staleness_and_resume(tmp_path, qtbot):
    config = manual_config(tmp_path)
    workflow = ManualQCWorkflow(config)
    window = ManualQCWindow(workflow)
    qtbot.addWidget(window)
    window.show()
    assert window.model.platform_code == "p"
    assert window.model.current_chunk_index == 0
    assert window.model.gps_pos_qual[4] == 5
    assert window.model.gps_num_sat[4] == 8
    assert "GPS PosQual" in window.point_label.text()

    window.resample_now()
    assert window.resampled is not None and not window.resample_stale
    window.auto_fit_visible_y.setChecked(False)
    chunk = window.model.current_chunk
    start = int(chunk.start_time.astype(np.int64)) / 1e9
    window.speed_plot.setXRange(start + 60, start + 1_200, padding=0)
    qtbot.wait(10)
    expected_time_range = window.speed_viewbox.viewRange()[0]
    np.testing.assert_allclose(
        window.acceleration_viewbox.viewRange()[0], expected_time_range,
    )
    source = int(window.model.selected_source)
    window.set_rejected(True)
    assert source in window.model.rejected_sources
    assert window.resample_stale
    assert "STALE" in window.resample_status.text()
    np.testing.assert_allclose(
        window.speed_viewbox.viewRange()[0], expected_time_range,
    )
    window.speed_plot.setYRange(0, 100, padding=0)
    window.fit_visible_y()
    np.testing.assert_allclose(
        window.speed_viewbox.viewRange()[0], expected_time_range,
    )
    assert np.diff(window.speed_viewbox.viewRange()[1]).item() < 100
    window.set_rejected(False)
    assert source not in window.model.rejected_sources

    window.model.set_selected_source(window.model.chunks[0].source_indices[-1])
    window.move_cursor(1)
    assert window.model.current_chunk_index == 1
    window.previous_chunk()
    assert window.model.current_chunk_index == 0
    window.save_cursor()
    selected = window.model.selected_source
    window.close()

    resumed_workflow = ManualQCWorkflow(config)
    resumed = ManualQCWindow(resumed_workflow)
    qtbot.addWidget(resumed)
    assert resumed.model.selected_source == selected
    resumed.close()


@pytest.mark.skipif(not GUI_TEST_AVAILABLE, reason="manual-QC Qt test dependencies are absent")
def test_gui_goes_to_first_unreviewed_or_dirty_chunk(tmp_path, qtbot):
    workflow = ManualQCWorkflow(manual_config(tmp_path))
    window = ManualQCWindow(workflow)
    qtbot.addWidget(window)

    window.model.set_chunk(2)
    window.go_to_first_unreviewed()
    assert window.model.current_chunk_index == 0
    assert window.model.selected_source == window.model.chunks[0].source_indices[0]

    first_key = window.model.chunks[0].key
    window.review_and_next()
    assert window.model.current_chunk_index == 1
    window.model.set_chunk(2)
    window.go_to_first_unreviewed()
    assert window.model.current_chunk_index == 1

    workflow.state.mark_chunks_dirty("p", [first_key])
    window.model.set_chunk(2)
    window.go_to_first_unreviewed()
    assert window.model.current_chunk_index == 0
    assert "changed after review" in window.statusBar().currentMessage()
    window.close()


@pytest.mark.skipif(not GUI_TEST_AVAILABLE, reason="manual-QC Qt test dependencies are absent")
def test_gui_marks_speed_warnings_without_changing_manual_qc(tmp_path, qtbot):
    workflow = ManualQCWorkflow(manual_config(tmp_path))
    window = ManualQCWindow(workflow)
    qtbot.addWidget(window)

    _original, accepted = window._get_kinematics()
    expected_high = int(np.sum(
        np.isfinite(accepted.speed_m_s)
        & (accepted.speed_m_s > workflow.config.high_speed_warning_m_s)
    ))
    warning_x, warning_y = window.speed_items["high_speed_warning"].getData()

    assert len(warning_x) == expected_high
    assert len(warning_y) == expected_high
    assert not window.model.rejected_sources
    assert "Visual warnings only" in window.speed_warning_label.text()
    window.close()


@pytest.mark.skipif(not GUI_TEST_AVAILABLE, reason="manual-QC Qt test dependencies are absent")
def test_gui_sst_selection_rejects_only_sst(tmp_path, qtbot):
    workflow = ManualQCWorkflow(manual_config(tmp_path))
    window = ManualQCWindow(workflow)
    qtbot.addWidget(window)
    source = 2
    position = window.model.source_to_position[source]
    timestamp = int(window.model.time[position].astype(np.int64)) / 1e9
    value = float(window.model.sst[position])

    window._rectangle_selected(
        "sst", timestamp - 1, timestamp + 1, value - 0.01, value + 0.01,
    )
    assert window.selection_domain == "sst"
    assert window.selected_sources == {source}
    window.set_rejected(True)
    assert source in window.model.rejected_sst_sources
    assert source not in window.model.rejected_sources
    assert "SST" in window.reject_button.text()

    projection = window.model.projected_positions()
    x = float(projection.x_m[position])
    y = float(projection.y_m[position])
    window._rectangle_selected("map", x - 0.1, x + 0.1, y - 0.1, y + 0.1)
    assert window.selection_domain == "position"
    window.set_rejected(True)
    assert source in window.model.rejected_sources
    assert source in window.model.rejected_sst_sources
    window.close()
