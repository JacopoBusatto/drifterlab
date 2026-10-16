"""PySide6/pyqtgraph desktop interface for independent manual QC."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Import PySide6 first so pyqtgraph cannot select a different installed Qt
# binding and create an unsafe mixed-Qt process.
# isort: off
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
# isort: on

from .model import (
    KinematicSeries,
    ManualTrajectoryModel,
    ResampledTrajectory,
    speed_warning_masks,
)
from .workflow import FinalZarrBuildPlan, ManualQCWorkflow


def _epoch_seconds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="datetime64[ns]")
    result = np.full(len(values), np.nan, dtype=float)
    valid = ~np.isnat(values)
    result[valid] = values[valid].astype(np.int64) / 1e9
    return result


def _set_finite_scatter_data(
    item: pg.ScatterPlotItem,
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[float] | tuple[float, ...],
) -> None:
    """Set scatter coordinates after removing points that cannot be drawn."""
    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if x_values.shape != y_values.shape:
        raise ValueError("Scatter x and y coordinates must have the same shape")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    item.setData(x_values[finite], y_values[finite])


def _ns_or_none(value: np.datetime64) -> int | None:
    return None if np.isnat(value) else int(np.datetime64(value, "ns").astype(np.int64))


class SelectionViewBox(pg.ViewBox):
    """A normal pan/zoom ViewBox with an explicit rectangle-selection mode."""

    selection_finished = QtCore.Signal(float, float, float, float)

    def __init__(self, *args, time_series: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_series = bool(time_series)
        self.selection_enabled = False
        self._rubber = QtWidgets.QGraphicsRectItem()
        self._rubber.setPen(pg.mkPen("#f59e0b", width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self._rubber.setBrush(pg.mkBrush(245, 158, 11, 35))
        self._rubber.setVisible(False)
        self.addItem(self._rubber, ignoreBounds=True)

    def set_selection_enabled(self, enabled: bool) -> None:
        self.selection_enabled = bool(enabled)
        if not enabled:
            self._rubber.setVisible(False)

    def mouseDragEvent(self, event, axis=None):
        if (
            self.selection_enabled
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            event.accept()
            start = self.mapSceneToView(event.buttonDownScenePos())
            finish = self.mapSceneToView(event.scenePos())
            rectangle = QtCore.QRectF(start, finish).normalized()
            self._rubber.setRect(rectangle)
            self._rubber.setVisible(True)
            if event.isFinish():
                self._rubber.setVisible(False)
                self.selection_finished.emit(
                    rectangle.left(), rectangle.right(),
                    rectangle.top(), rectangle.bottom(),
                )
            return
        super().mouseDragEvent(event, axis=axis)

    def wheelEvent(self, event, axis=None):
        """Keep time and value zoom independent in time-series plot areas."""
        if self.time_series and axis is None:
            shift = bool(
                event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
            )
            axis = 1 if shift else 0
        super().wheelEvent(event, axis=axis)


class FinalZarrWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self, workflow: ManualQCWorkflow, plan: FinalZarrBuildPlan,
    ):
        super().__init__()
        self.workflow = workflow
        self.plan = plan

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self.workflow.build_final_zarr(
                plan=self.plan, progress=self.progress.emit,
            )
        except Exception as exc:  # noqa: BLE001 - surface worker failures in the GUI
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class ManualQCWindow(QtWidgets.QMainWindow):
    """Fast, resumable reviewer for raw positions up to the drogue endpoint."""

    def __init__(self, workflow: ManualQCWorkflow):
        super().__init__()
        self.workflow = workflow
        self.model: ManualTrajectoryModel | None = None
        self.platform_code = ""
        self.selected_sources: set[int] = set()
        self.selection_domain = "position"
        self.resampled: ResampledTrajectory | None = None
        self.previous_resampled: ResampledTrajectory | None = None
        self.resample_stale = False
        self._loading = False
        self._closed = False
        self._map_fit = "trajectory"
        self._kinematics_platform = ""
        self._kinematics_rejected: frozenset[int] = frozenset()
        self._original_kinematics: KinematicSeries | None = None
        self._accepted_kinematics: KinematicSeries | None = None
        self._fitting_time_y = False
        self._build_thread: QtCore.QThread | None = None
        self._build_worker: FinalZarrWorker | None = None
        self.setWindowTitle("drifterlab — independent manual trajectory QC")
        self.resize(1600, 1050)
        self._build_ui()
        self._make_shortcuts()
        cursor = self.workflow.state.load_cursor()
        initial = (
            str(cursor["platform_code"])
            if cursor and str(cursor["platform_code"]) in self.workflow.platform_by_code
            else self.workflow.platforms[0].platform_code
        )
        self.load_platform(initial, cursor=cursor)
        QtCore.QTimer.singleShot(0, self._refresh_final_zarr_status)

    # ---- construction -------------------------------------------------

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Platform"))
        self.platform_combo = QtWidgets.QComboBox()
        for platform in self.workflow.platforms:
            self.platform_combo.addItem(platform.platform_code, platform.platform_code)
        self.platform_combo.currentIndexChanged.connect(self._platform_changed)
        header.addWidget(self.platform_combo, 1)
        self.drogue_label = QtWidgets.QLabel()
        self.drogue_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self.drogue_label, 2)
        self.scope_label = QtWidgets.QLabel()
        self.scope_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        header.addWidget(self.scope_label, 2)
        self.progress_label = QtWidgets.QLabel()
        header.addWidget(self.progress_label)
        root.addLayout(header)

        chunks = QtWidgets.QHBoxLayout()
        chunks.addWidget(QtWidgets.QLabel("Chunk"))
        self.chunk_combo = QtWidgets.QComboBox()
        self.chunk_combo.currentIndexChanged.connect(self._chunk_changed)
        chunks.addWidget(self.chunk_combo, 1)
        self.first_unreviewed_button = QtWidgets.QPushButton("U  First unreviewed")
        self.first_unreviewed_button.setToolTip(
            "Go to the earliest never-reviewed or changed-after-review chunk"
        )
        self.first_unreviewed_button.clicked.connect(self.go_to_first_unreviewed)
        chunks.addWidget(self.first_unreviewed_button)
        self.previous_chunk_button = QtWidgets.QPushButton("P  Previous chunk")
        self.previous_chunk_button.clicked.connect(self.previous_chunk)
        chunks.addWidget(self.previous_chunk_button)
        self.next_chunk_button = QtWidgets.QPushButton("N  Review + next")
        self.next_chunk_button.clicked.connect(self.review_and_next)
        chunks.addWidget(self.next_chunk_button)
        root.addLayout(chunks)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.map_viewbox = SelectionViewBox()
        self.map_plot = pg.PlotWidget(viewBox=self.map_viewbox)
        self.map_plot.setBackground("w")
        self.map_plot.setLabel("bottom", "Local east", units="m")
        self.map_plot.setLabel("left", "Local north", units="m")
        self.map_plot.setTitle("Trajectory")
        self.map_plot.setAspectLocked(True)
        self.map_viewbox.selection_finished.connect(
            lambda x0, x1, y0, y1: self._rectangle_selected("map", x0, x1, y0, y1)
        )
        self.map_plot.scene().sigMouseClicked.connect(self._map_clicked)
        splitter.addWidget(self.map_plot)

        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.speed_viewbox = SelectionViewBox(time_series=True)
        self.speed_plot = pg.PlotWidget(
            viewBox=self.speed_viewbox,
            axisItems={"bottom": pg.DateAxisItem(orientation="bottom", utcOffset=0)},
        )
        self.speed_plot.setBackground("w")
        self.speed_plot.setLabel("left", "Speed", units="m/s")
        self.speed_plot.setLabel("bottom", "UTC time")
        self.speed_plot.setTitle("Adjacent-point speed")
        self.speed_plot.getAxis("left").setWidth(85)
        self.speed_viewbox.selection_finished.connect(
            lambda x0, x1, y0, y1: self._rectangle_selected("speed", x0, x1, y0, y1)
        )
        right.addWidget(self.speed_plot)

        self.acceleration_viewbox = SelectionViewBox(time_series=True)
        self.acceleration_plot = pg.PlotWidget(
            viewBox=self.acceleration_viewbox,
            axisItems={"bottom": pg.DateAxisItem(orientation="bottom", utcOffset=0)},
        )
        self.acceleration_plot.setBackground("w")
        self.acceleration_plot.setLabel("left", "Vector acceleration", units="m/s²")
        self.acceleration_plot.setLabel("bottom", "UTC time")
        self.acceleration_plot.setTitle("Adjacent-velocity acceleration")
        self.acceleration_plot.getAxis("left").setWidth(85)
        self.acceleration_viewbox.selection_finished.connect(
            lambda x0, x1, y0, y1: self._rectangle_selected(
                "acceleration", x0, x1, y0, y1,
            )
        )
        self.acceleration_viewbox.setXLink(self.speed_viewbox)
        right.addWidget(self.acceleration_plot)

        self.sst_viewbox = SelectionViewBox(time_series=True)
        self.sst_plot = pg.PlotWidget(
            viewBox=self.sst_viewbox,
            axisItems={"bottom": pg.DateAxisItem(orientation="bottom", utcOffset=0)},
        )
        self.sst_plot.setBackground("w")
        self.sst_plot.setLabel("left", "SST", units="°C")
        self.sst_plot.setLabel("bottom", "UTC time")
        self.sst_plot.setTitle("SST — independent manual QC")
        self.sst_plot.getAxis("left").setWidth(85)
        self.sst_viewbox.selection_finished.connect(
            lambda x0, x1, y0, y1: self._rectangle_selected(
                "sst", x0, x1, y0, y1,
            )
        )
        self.sst_viewbox.setXLink(self.speed_viewbox)
        self.sst_plot.scene().sigMouseClicked.connect(self._sst_clicked)
        right.addWidget(self.sst_plot)

        self.gps_pos_qual_viewbox = SelectionViewBox(time_series=True)
        self.gps_pos_qual_plot = pg.PlotWidget(
            viewBox=self.gps_pos_qual_viewbox,
            axisItems={"bottom": pg.DateAxisItem(orientation="bottom", utcOffset=0)},
        )
        self.gps_pos_qual_plot.setBackground("w")
        self.gps_pos_qual_plot.setLabel("left", "GPS PosQual")
        self.gps_pos_qual_plot.setLabel("bottom", "UTC time")
        self.gps_pos_qual_plot.setTitle("GPS position quality — source reference")
        self.gps_pos_qual_plot.getAxis("left").setWidth(85)
        self.gps_pos_qual_viewbox.selection_finished.connect(
            lambda x0, x1, y0, y1: self._rectangle_selected(
                "gps_pos_qual", x0, x1, y0, y1,
            )
        )
        self.gps_pos_qual_viewbox.setXLink(self.speed_viewbox)
        self.gps_pos_qual_plot.scene().sigMouseClicked.connect(
            lambda event: self._reference_clicked(
                event, "gps_pos_qual", self.gps_pos_qual_plot,
                self.gps_pos_qual_viewbox,
            )
        )
        right.addWidget(self.gps_pos_qual_plot)

        self.gps_num_sat_viewbox = SelectionViewBox(time_series=True)
        self.gps_num_sat_plot = pg.PlotWidget(
            viewBox=self.gps_num_sat_viewbox,
            axisItems={"bottom": pg.DateAxisItem(orientation="bottom", utcOffset=0)},
        )
        self.gps_num_sat_plot.setBackground("w")
        self.gps_num_sat_plot.setLabel("left", "Satellites", units="count")
        self.gps_num_sat_plot.setLabel("bottom", "UTC time")
        self.gps_num_sat_plot.setTitle("GPS satellite count — source reference")
        self.gps_num_sat_plot.getAxis("left").setWidth(85)
        self.gps_num_sat_viewbox.selection_finished.connect(
            lambda x0, x1, y0, y1: self._rectangle_selected(
                "gps_num_sat", x0, x1, y0, y1,
            )
        )
        self.gps_num_sat_viewbox.setXLink(self.speed_viewbox)
        self.gps_num_sat_plot.scene().sigMouseClicked.connect(
            lambda event: self._reference_clicked(
                event, "gps_num_sat", self.gps_num_sat_plot,
                self.gps_num_sat_viewbox,
            )
        )
        right.addWidget(self.gps_num_sat_plot)
        right.setSizes([240, 240, 220, 140, 140])
        splitter.addWidget(right)
        splitter.setSizes([800, 800])
        root.addWidget(splitter, 1)

        time_display = QtWidgets.QHBoxLayout()
        time_help = QtWidgets.QLabel(
            "Time plots: wheel = time zoom  \u00b7  Shift+wheel = Y zoom"
        )
        time_display.addWidget(time_help)
        self.fit_visible_y_button = QtWidgets.QPushButton("Fit visible Y")
        self.fit_visible_y_button.setToolTip(
            "Fit every time-series Y axis independently without changing time"
        )
        self.fit_visible_y_button.clicked.connect(self.fit_visible_y)
        time_display.addWidget(self.fit_visible_y_button)
        self.auto_fit_visible_y = QtWidgets.QCheckBox("Auto-fit Y after time zoom")
        self.auto_fit_visible_y.setChecked(True)
        self.auto_fit_visible_y.setToolTip(
            "Recalculate each vertical range when the shared time window changes"
        )
        self.auto_fit_visible_y.toggled.connect(self._auto_fit_visible_y_changed)
        time_display.addWidget(self.auto_fit_visible_y)
        time_display.addStretch(1)
        root.addLayout(time_display)

        actions = QtWidgets.QHBoxLayout()
        self.target_label = QtWidgets.QLabel()
        actions.addWidget(self.target_label)
        self.previous_point_button = QtWidgets.QPushButton("←  Previous point")
        self.previous_point_button.clicked.connect(lambda: self.move_cursor(-1))
        actions.addWidget(self.previous_point_button)
        self.next_point_button = QtWidgets.QPushButton("Next point  →")
        self.next_point_button.clicked.connect(lambda: self.move_cursor(1))
        actions.addWidget(self.next_point_button)
        self.restore_button = QtWidgets.QPushButton("K  Restore")
        self.restore_button.clicked.connect(lambda: self.set_rejected(False))
        actions.addWidget(self.restore_button)
        self.reject_button = QtWidgets.QPushButton("R  Reject")
        self.reject_button.clicked.connect(lambda: self.set_rejected(True))
        actions.addWidget(self.reject_button)
        self.undo_button = QtWidgets.QPushButton("Ctrl+Z  Undo")
        self.undo_button.clicked.connect(self.undo)
        actions.addWidget(self.undo_button)
        self.selection_button = QtWidgets.QPushButton("Rectangle select")
        self.selection_button.setCheckable(True)
        self.selection_button.toggled.connect(self._selection_mode_changed)
        actions.addWidget(self.selection_button)
        self.clear_selection_button = QtWidgets.QPushButton("Clear selection")
        self.clear_selection_button.clicked.connect(self.clear_selection)
        actions.addWidget(self.clear_selection_button)
        root.addLayout(actions)

        display = QtWidgets.QHBoxLayout()
        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.zoom_in_button.setToolTip("Zoom map in")
        self.zoom_in_button.clicked.connect(lambda: self._zoom_map(0.8))
        display.addWidget(self.zoom_in_button)
        self.zoom_out_button = QtWidgets.QPushButton("−")
        self.zoom_out_button.setToolTip("Zoom map out")
        self.zoom_out_button.clicked.connect(lambda: self._zoom_map(1.25))
        display.addWidget(self.zoom_out_button)
        self.fit_chunk_button = QtWidgets.QPushButton("Fit chunk")
        self.fit_chunk_button.clicked.connect(self.fit_chunk)
        display.addWidget(self.fit_chunk_button)
        self.fit_trajectory_button = QtWidgets.QPushButton("Fit trajectory")
        self.fit_trajectory_button.clicked.connect(self.fit_trajectory)
        display.addWidget(self.fit_trajectory_button)
        display.addSpacing(18)
        display.addWidget(QtWidgets.QLabel("Regular interval"))
        self.frequency_spin = QtWidgets.QDoubleSpinBox()
        self.frequency_spin.setRange(1.0, 86400.0)
        self.frequency_spin.setDecimals(1)
        self.frequency_spin.setSuffix(" s")
        self.frequency_spin.setValue(self.workflow.config.resample_frequency_seconds)
        self.frequency_spin.valueChanged.connect(self._frequency_changed)
        display.addWidget(self.frequency_spin)
        self.resample_button = QtWidgets.QPushButton("Resample now")
        self.resample_button.clicked.connect(self.resample_now)
        display.addWidget(self.resample_button)
        self.show_resampled = QtWidgets.QCheckBox("Show resampled")
        self.show_resampled.toggled.connect(self._show_resampled_changed)
        display.addWidget(self.show_resampled)
        self.resample_status = QtWidgets.QLabel("Not resampled")
        display.addWidget(self.resample_status, 1)
        self.save_button = QtWidgets.QPushButton("Ctrl+S  Save state")
        self.save_button.clicked.connect(self.save_cursor)
        display.addWidget(self.save_button)
        self.publish_button = QtWidgets.QPushButton("Finalize platform")
        self.publish_button.clicked.connect(self.publish_current)
        display.addWidget(self.publish_button)
        self.final_zarr_button = QtWidgets.QPushButton("Build Zarr")
        self.final_zarr_button.clicked.connect(self.start_final_zarr_build)
        display.addWidget(self.final_zarr_button)
        root.addLayout(display)

        self.final_zarr_label = QtWidgets.QLabel()
        self.final_zarr_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.final_zarr_label)

        self.speed_warning_label = QtWidgets.QLabel()
        self.speed_warning_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.speed_warning_label)

        self.point_label = QtWidgets.QLabel()
        self.point_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.point_label)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self.map_items = self._make_map_items()
        self.speed_items = self._make_time_items(self.speed_plot)
        self.speed_items["high_speed_warning"] = pg.ScatterPlotItem(
            size=14, symbol="t", pen=pg.mkPen("#a21caf", width=2), brush=None,
        )
        self.speed_items["zero_speed_warning"] = pg.ScatterPlotItem(
            size=12, symbol="s", pen=pg.mkPen("#b45309", width=2), brush=None,
        )
        self.speed_plot.addItem(self.speed_items["high_speed_warning"])
        self.speed_plot.addItem(self.speed_items["zero_speed_warning"])
        self.acceleration_items = self._make_time_items(self.acceleration_plot)
        self.sst_items = self._make_time_items(self.sst_plot)
        self.sst_items["accepted"].setPen(pg.mkPen("#0f766e", width=2))
        self.sst_items["accepted_points"].setBrush(pg.mkBrush("#0f766e"))
        self.sst_items["chunk"].setPen(pg.mkPen("#0f766e"))
        self.sst_items["chunk"].setBrush(pg.mkBrush("#2dd4bf"))
        self.sst_items["position_rejected"] = pg.ScatterPlotItem(
            size=10, symbol="o", pen=pg.mkPen("#ea580c", width=2), brush=None,
        )
        self.sst_plot.addItem(self.sst_items["position_rejected"])
        self.gps_pos_qual_items = self._make_reference_items(
            self.gps_pos_qual_plot, "#9333ea",
        )
        self.gps_num_sat_items = self._make_reference_items(
            self.gps_num_sat_plot, "#0891b2",
        )
        self.speed_viewbox.sigXRangeChanged.connect(self._time_x_range_changed)
        self._set_selection_domain("position")

    def _make_map_items(self) -> dict[str, pg.GraphicsObject]:
        items = {
            "raw_line": self.map_plot.plot(pen=pg.mkPen(155, 160, 168, 120, width=1)),
            "accepted_line": self.map_plot.plot(pen=pg.mkPen("#2563eb", width=2)),
            "chunk_line": self.map_plot.plot(pen=pg.mkPen("#06b6d4", width=4)),
            "raw_points": pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(120, 125, 132, 120)),
            "chunk_points": pg.ScatterPlotItem(size=11, pen=pg.mkPen("#0369a1"), brush=pg.mkBrush("#38bdf8")),
            "rejected": pg.ScatterPlotItem(size=12, symbol="x", pen=pg.mkPen("#dc2626", width=2), brush=None),
            "selected": pg.ScatterPlotItem(size=15, symbol="o", pen=pg.mkPen("#f59e0b", width=2), brush=None),
            "cursor": pg.ScatterPlotItem(size=17, symbol="o", pen=pg.mkPen("#111827", width=2), brush=pg.mkBrush("#facc15")),
            "high_speed_warning": pg.ScatterPlotItem(
                size=16, symbol="t", pen=pg.mkPen("#a21caf", width=2), brush=None,
            ),
            "zero_speed_warning": pg.ScatterPlotItem(
                size=14, symbol="s", pen=pg.mkPen("#b45309", width=2), brush=None,
            ),
            "resampled": self.map_plot.plot(pen=pg.mkPen("#7c3aed", width=2, style=QtCore.Qt.PenStyle.DashLine)),
            "resampled_points": pg.ScatterPlotItem(size=10, symbol="d", pen=pg.mkPen("#7c3aed"), brush=None),
            "previous_resampled": self.map_plot.plot(pen=pg.mkPen(234, 88, 12, 150, width=2, style=QtCore.Qt.PenStyle.DashLine)),
            "drogue_end": pg.ScatterPlotItem(size=16, symbol="t", pen=pg.mkPen("#991b1b", width=2), brush=pg.mkBrush("#ef4444")),
        }
        for name in (
            "raw_points", "chunk_points", "rejected", "selected", "cursor",
            "high_speed_warning", "zero_speed_warning",
            "resampled_points", "drogue_end",
        ):
            self.map_plot.addItem(items[name])
        return items

    def _make_time_items(self, plot: pg.PlotWidget) -> dict[str, pg.GraphicsObject]:
        items = {
            "raw": plot.plot(
                pen=pg.mkPen(150, 155, 163, 105, width=1), connect="finite",
            ),
            "raw_points": pg.ScatterPlotItem(
                size=4, pen=None, brush=pg.mkBrush(150, 155, 163, 90),
            ),
            "accepted": plot.plot(
                pen=pg.mkPen("#2563eb", width=2), connect="finite",
            ),
            "accepted_points": pg.ScatterPlotItem(
                size=5, pen=None, brush=pg.mkBrush("#2563eb"),
            ),
            "chunk": pg.ScatterPlotItem(size=8, pen=pg.mkPen("#0369a1"), brush=pg.mkBrush("#38bdf8")),
            "rejected": pg.ScatterPlotItem(size=11, symbol="x", pen=pg.mkPen("#dc2626", width=2), brush=None),
            "selected": pg.ScatterPlotItem(size=14, symbol="o", pen=pg.mkPen("#f59e0b", width=2), brush=None),
            "cursor": pg.ScatterPlotItem(size=16, symbol="o", pen=pg.mkPen("#111827", width=2), brush=pg.mkBrush("#facc15")),
            "resampled": plot.plot(pen=pg.mkPen("#7c3aed", width=2, style=QtCore.Qt.PenStyle.DashLine), connect="finite"),
            "resampled_points": pg.ScatterPlotItem(size=6, symbol="d", pen=pg.mkPen("#7c3aed"), brush=None),
            "previous_resampled": plot.plot(pen=pg.mkPen(234, 88, 12, 140, width=2, style=QtCore.Qt.PenStyle.DashLine), connect="finite"),
            "drogue_line": pg.InfiniteLine(angle=90, pen=pg.mkPen("#b91c1c", width=2, style=QtCore.Qt.PenStyle.DotLine)),
        }
        for name in (
            "raw_points", "accepted_points", "chunk", "rejected", "selected",
            "cursor", "resampled_points", "drogue_line",
        ):
            plot.addItem(items[name])
        items["raw"].setClipToView(True)
        items["accepted"].setClipToView(True)
        return items

    def _make_reference_items(
        self, plot: pg.PlotWidget, color: str,
    ) -> dict[str, pg.GraphicsObject]:
        items = {
            "raw": plot.plot(
                pen=pg.mkPen(color, width=1.5), connect="finite",
            ),
            "raw_points": pg.ScatterPlotItem(
                size=5, pen=None, brush=pg.mkBrush(color),
            ),
            "chunk": pg.ScatterPlotItem(
                size=8, pen=pg.mkPen("#0369a1"), brush=pg.mkBrush("#38bdf8"),
            ),
            "rejected": pg.ScatterPlotItem(
                size=11, symbol="x", pen=pg.mkPen("#dc2626", width=2), brush=None,
            ),
            "selected": pg.ScatterPlotItem(
                size=14, symbol="o", pen=pg.mkPen("#f59e0b", width=2), brush=None,
            ),
            "cursor": pg.ScatterPlotItem(
                size=16, symbol="o", pen=pg.mkPen("#111827", width=2),
                brush=pg.mkBrush("#facc15"),
            ),
            "drogue_line": pg.InfiniteLine(
                angle=90,
                pen=pg.mkPen("#b91c1c", width=2, style=QtCore.Qt.PenStyle.DotLine),
            ),
        }
        for name in (
            "raw_points", "chunk", "rejected", "selected", "cursor", "drogue_line",
        ):
            plot.addItem(items[name])
        items["raw"].setClipToView(True)
        return items

    def _make_shortcuts(self) -> None:
        definitions: tuple[tuple[str, Callable[[], None]], ...] = (
            ("Left", lambda: self.move_cursor(-1)),
            ("Right", lambda: self.move_cursor(1)),
            ("K", lambda: self.set_rejected(False)),
            ("R", lambda: self.set_rejected(True)),
            ("Delete", lambda: self.set_rejected(True)),
            ("N", self.review_and_next),
            ("P", self.previous_chunk),
            ("U", self.go_to_first_unreviewed),
            ("Ctrl+Z", self.undo),
            ("Ctrl+S", self.save_cursor),
            ("Escape", self.clear_selection),
        )
        self.shortcuts: list[QtGui.QShortcut] = []
        for key, callback in definitions:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self, callback)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            self.shortcuts.append(shortcut)

    # ---- loading and persistence -------------------------------------

    def load_platform(self, platform_code: str, *, cursor: dict[str, object] | None = None) -> None:
        if self.model is not None:
            self.save_cursor(quiet=True)
        self._loading = True
        try:
            model = self.workflow.load_model(platform_code)
            self.model = model
            self.platform_code = model.platform_code
            self.selected_sources.clear()
            self._set_selection_domain("position")
            self.resampled = None
            self.previous_resampled = None
            self.resample_stale = False
            self._kinematics_platform = ""
            self._original_kinematics = None
            self._accepted_kinematics = None
            platform_index = self.platform_combo.findData(model.platform_code)
            self.platform_combo.setCurrentIndex(platform_index)
            item = self.workflow.platform_by_code[model.platform_code]
            if item.drogue.final_drogue_status == "lost":
                stop = np.datetime_as_string(item.stop_time, unit="s", timezone="UTC")
                self.drogue_label.setText(
                    f"Drogue lost — review stops at {stop} ({self.workflow.config.stop_at})"
                )
                self.drogue_label.setStyleSheet("")
            elif item.blocked_reason:
                self.drogue_label.setText(item.blocked_reason)
                self.drogue_label.setStyleSheet("color: #b91c1c; font-weight: bold")
            else:
                self.drogue_label.setText("Drogue retained — review continues to final observation")
                self.drogue_label.setStyleSheet("")

            saved = cursor if cursor and str(cursor.get("platform_code")) == model.platform_code else None
            if saved and str(saved.get("source_sha256")) == model.source_sha256:
                source = saved.get("selected_source_obs_index")
                if source is not None and int(source) in model.review_position:
                    model.set_selected_source(int(source))
                elif saved.get("chunk_key"):
                    match = next(
                        (index for index, chunk in enumerate(model.chunks)
                         if chunk.key == str(saved["chunk_key"])), None,
                    )
                    if match is not None:
                        model.set_chunk(match)
            self._refresh_chunk_combo()
            sample_state = self.workflow.state.resample_state(model.platform_code)
            frequency = sample_state["frequency_seconds"]
            if frequency is not None:
                self.frequency_spin.setValue(float(frequency))
            self.show_resampled.blockSignals(True)
            self.show_resampled.setChecked(bool(sample_state["shown"]))
            self.show_resampled.blockSignals(False)
            self.resample_stale = bool(sample_state["stale"])
            if (
                bool(sample_state["shown"]) and not self.resample_stale
                and frequency is not None and sample_state["resample_revision"] is not None
            ):
                try:
                    self.resampled = model.resample(
                        float(frequency),
                        sst_maximum_gap_minutes=self._sst_maximum_gap_minutes(),
                    )
                except ValueError as exc:
                    self.resample_stale = True
                    self._status(str(exc), error=True)
            self._render()
            self.fit_trajectory()
            self.save_cursor(quiet=True)
            if model.platform_code in self.workflow.review_scope_changed:
                self._status(
                    "Review endpoint or duration policy changed; existing chunk "
                    "progress was marked for review.",
                    error=True,
                )
            elif item.blocked_reason:
                self._status(item.blocked_reason, error=True)
            else:
                self._status(f"Loaded {model.platform_code}: {len(model.review_order):,} reviewable raw points")
        finally:
            self._loading = False

    def save_cursor(self, *, quiet: bool = False) -> None:
        if self.model is None:
            return
        chunk = self.model.current_chunk
        self.workflow.state.save_cursor(
            self.model.platform_code, self.model.source_sha256,
            None if chunk is None else chunk.key,
            self.model.selected_source,
        )
        if not quiet:
            self._status(f"State saved to {self.workflow.state.path}")

    def publish_current(self) -> None:
        if self.model is None:
            return
        try:
            destination = self.workflow.publish(self.model)
        except (ValueError, OSError) as exc:
            self._status(str(exc), error=True)
            return
        self._status(f"Finalized manual-QC product: {destination}")
        self._refresh_final_zarr_status(auto_build=True)

    def _sst_maximum_gap_minutes(self) -> float | None:
        # SST is intentionally outside the v2 grouped trajectory products.
        return None

    def _refresh_final_zarr_status(self, *, auto_build: bool = True) -> None:
        if self._build_thread is not None:
            return
        final = self.workflow.config.final_zarr
        if not final.enabled:
            self.final_zarr_label.setText("Grouped Zarr: disabled in configuration")
            self.final_zarr_button.setEnabled(False)
            return
        readiness = self.workflow.final_zarr_readiness()
        self.final_zarr_button.setEnabled(readiness.ready)
        if readiness.complete:
            self.final_zarr_button.setText("Build final Zarr")
            self.final_zarr_label.setText(
                f"All {readiness.total_platforms} drifters reviewed; "
                "ready to build the final Zarr."
            )
            self.final_zarr_button.setToolTip(
                f"Build or validate {final.output_directory}"
            )
            if auto_build and final.build_when_all_platforms_reviewed:
                self.start_final_zarr_build()
        elif readiness.ready:
            remaining = readiness.total_platforms - readiness.current_platforms
            self.final_zarr_button.setText("Build provisional Zarr")
            self.final_zarr_label.setText(
                f"{readiness.current_platforms}/{readiness.total_platforms} drifters "
                f"finalized; {remaining} remain unreviewed. A manual build will be "
                "marked provisional."
            )
            self.final_zarr_button.setToolTip(
                f"Build a provisional Zarr at {final.output_directory}"
            )
        else:
            self.final_zarr_button.setText("Build Zarr")
            detail = readiness.problems[0] if readiness.problems else "review incomplete"
            self.final_zarr_label.setText(
                "Grouped Zarr: no current finalized drifters are available; "
                f"{detail}"
            )
            self.final_zarr_button.setToolTip(
                "Finalize at least one drifter before building"
            )

    def start_final_zarr_build(self) -> None:
        if self._build_thread is not None:
            return
        try:
            plan = self.workflow.prepare_final_zarr_build()
        except (ValueError, OSError) as exc:
            self._refresh_final_zarr_status(auto_build=False)
            self._status(str(exc), error=True)
            return
        included = len(plan.inputs)
        remaining = plan.configured_platform_count - included
        self.centralWidget().setEnabled(False)
        if plan.complete:
            message = (
                f"All {included} drifters reviewed; building the final Zarr in "
                "the background..."
            )
        else:
            message = (
                f"Building the provisional Zarr from {included}/"
                f"{plan.configured_platform_count} finalized drifters; "
                f"{remaining} remain unreviewed..."
            )
        self.final_zarr_label.setText(message)
        self._status(message)
        thread = QtCore.QThread(self)
        worker = FinalZarrWorker(self.workflow, plan)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda message: self.final_zarr_label.setText(f"Grouped Zarr: {message}")
        )
        worker.finished.connect(self._final_zarr_finished)
        worker.failed.connect(self._final_zarr_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._final_zarr_thread_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._build_thread = thread
        self._build_worker = worker
        thread.start()

    @QtCore.Slot(object)
    def _final_zarr_finished(self, result: object) -> None:
        self.centralWidget().setEnabled(True)
        path = getattr(result, "zarr_path", self.workflow.config.final_zarr.output_directory)
        action = getattr(result, "action", "built")
        product_status = getattr(result, "product_status", "unknown status")
        self.final_zarr_label.setText(
            f"Grouped Zarr: {action} ({product_status}) — {path}"
        )
        self._status(f"Grouped Zarr {action} ({product_status}): {path}")
        self.final_zarr_button.setEnabled(True)

    @QtCore.Slot(str)
    def _final_zarr_failed(self, message: str) -> None:
        self.centralWidget().setEnabled(True)
        self.final_zarr_label.setText(f"Grouped Zarr failed: {message}")
        self.final_zarr_button.setEnabled(True)
        self._status(f"Grouped Zarr failed: {message}", error=True)

    @QtCore.Slot()
    def _final_zarr_thread_finished(self) -> None:
        self._build_thread = None
        self._build_worker = None

    # ---- navigation ---------------------------------------------------

    def _platform_changed(self, index: int) -> None:
        if self._loading or index < 0:
            return
        previous = self.platform_code
        try:
            self.load_platform(str(self.platform_combo.itemData(index)))
        except (ValueError, OSError) as exc:
            self._status(str(exc), error=True)
            restore = self.platform_combo.findData(previous)
            self.platform_combo.blockSignals(True)
            self.platform_combo.setCurrentIndex(restore)
            self.platform_combo.blockSignals(False)

    def _chunk_changed(self, index: int) -> None:
        if self._loading or self.model is None or index < 0:
            return
        self.model.set_chunk(index)
        self.selected_sources.clear()
        self._render()
        self._follow_cursor()
        self.save_cursor(quiet=True)

    def move_cursor(self, step: int) -> None:
        if self.model is None:
            return
        old_chunk = self.model.current_chunk_index
        if self.selection_domain == "sst":
            self.model.move_sst_cursor(step)
        else:
            self.model.move_cursor(step)
        self.selected_sources.clear()
        if old_chunk != self.model.current_chunk_index:
            self._refresh_chunk_combo()
            self._render()
        else:
            self._update_selection_items()
        self._follow_cursor()
        self.save_cursor(quiet=True)

    def previous_chunk(self) -> None:
        if self.model is None:
            return
        self.model.move_chunk(-1)
        self.selected_sources.clear()
        self._refresh_chunk_combo()
        self._render()
        self.fit_chunk()
        self.save_cursor(quiet=True)

    def go_to_first_unreviewed(self) -> None:
        if self.model is None:
            return
        states = self.workflow.state.chunk_states(
            self.model.platform_code, self.model.source_sha256,
        )
        index = next(
            (
                index for index, chunk in enumerate(self.model.chunks)
                if states.get(chunk.key, "unreviewed") != "reviewed"
            ),
            None,
        )
        if index is None:
            self._status("All chunks on this platform are reviewed")
            return
        chunk = self.model.set_chunk(index)
        self.model.set_selected_source(chunk.source_indices[0])
        self.selected_sources.clear()
        self._refresh_chunk_combo()
        self._render()
        self.fit_chunk()
        self.save_cursor(quiet=True)
        state = states.get(chunk.key, "unreviewed")
        description = "changed after review" if state == "dirty" else "unreviewed"
        self._status(
            f"Opened first {description} chunk: {index + 1}/{len(self.model.chunks)}"
        )

    def review_and_next(self) -> None:
        if self.model is None or self.model.current_chunk is None:
            return
        chunk = self.model.current_chunk
        self.workflow.state.mark_chunk_reviewed(
            self.model.platform_code, self.model.source_sha256, chunk.key,
            start_time_ns=_ns_or_none(chunk.start_time),
            end_time_ns=_ns_or_none(chunk.end_time),
        )
        if self.model.current_chunk_index < len(self.model.chunks) - 1:
            self.model.move_chunk(1)
            self.selected_sources.clear()
            self._refresh_chunk_combo()
            self._render()
            self.fit_chunk()
            self.save_cursor(quiet=True)
            return
        reviewed = self.workflow.state.reviewed_chunk_keys(
            self.model.platform_code, self.model.source_sha256,
        )
        issues = self.model.completion_issues(reviewed)
        blocked = self.workflow.platform_by_code[self.model.platform_code].blocked_reason
        if blocked:
            issues.append(blocked)
        self._refresh_chunk_combo()
        if issues:
            self._status("Cannot finalize: " + "; ".join(issues), error=True)
            return
        try:
            destination = self.workflow.publish(self.model)
        except (ValueError, OSError) as exc:
            self._status(str(exc), error=True)
            return
        index = self.platform_combo.currentIndex()
        if index + 1 < self.platform_combo.count():
            next_code = str(self.platform_combo.itemData(index + 1))
            self.load_platform(next_code)
            self._status(f"Published {destination.name}; opened {next_code}")
        else:
            self._status(f"Published {destination.name}; all configured platforms reached")
        self._refresh_final_zarr_status(auto_build=True)

    # ---- decisions and resampling ------------------------------------

    def _targets(self) -> set[int]:
        if self.model is None or self.model.selected_source is None:
            return set()
        return set(self.selected_sources) or {int(self.model.selected_source)}

    def set_rejected(self, rejected: bool) -> None:
        if self.model is None:
            return
        targets = self._targets()
        if self.selection_domain == "sst":
            targets = {
                source for source in targets
                if np.isfinite(self.model.sst[self.model.source_to_position[source]])
            }
            desired = {
                source for source in targets
                if (source not in self.model.rejected_sst_sources) == rejected
            }
            if not desired:
                self._status("No selected SST observations changed")
                return
            dirty = self.model.affected_sst_chunk_keys(desired)
            try:
                changed = self.workflow.state.set_sst_rejected(
                    self.model.platform_code, self.model.source_sha256, desired,
                    rejected=rejected, dirty_chunk_keys=dirty,
                )
            except (ValueError, OSError) as exc:
                self._status(str(exc), error=True)
                return
            self.model.set_sst_rejected(changed, rejected)
            if self.resampled is not None:
                self.resample_stale = True
                self.previous_resampled = None
            self._refresh_chunk_combo()
            self._render(reset_time_range=False)
            action = "Rejected" if rejected else "Restored"
            self._status(
                f"{action} {len(changed)} SST observation(s); position QC unchanged"
            )
            return
        desired = {
            source for source in targets
            if (source not in self.model.rejected_sources) == rejected
        }
        if not desired:
            self._status("No selected raw observations changed")
            return
        dirty = self.model.affected_chunk_keys(desired)
        previous_start = self.model.deployment_start_time
        previous_end = self.model.review_end_time
        try:
            changed = self.workflow.state.set_rejected(
                self.model.platform_code, self.model.source_sha256, desired,
                rejected=rejected, dirty_chunk_keys=dirty,
            )
        except (ValueError, OSError) as exc:
            self._status(str(exc), error=True)
            return
        self.model.set_rejected(changed, rejected)
        self._accepted_kinematics = None
        self._kinematics_platform = ""
        if self.resampled is not None:
            self.resample_stale = True
            self.previous_resampled = None
        self._refresh_chunk_combo()
        self._render(reset_time_range=False)
        action = "Rejected" if rejected else "Restored"
        scope_changed = (
            previous_start != self.model.deployment_start_time
            or previous_end != self.model.review_end_time
        )
        suffix = "; deployment review span recalculated" if scope_changed else ""
        self._status(
            f"{action} {len(changed)} raw observation(s); "
            f"raw bridges recalculated{suffix}"
        )

    def undo(self) -> None:
        if self.model is None:
            return
        try:
            domain, sources = self.workflow.state.undo_last_with_domain(
                self.model.platform_code, self.model.source_sha256,
            )
        except (ValueError, OSError) as exc:
            self._status(str(exc), error=True)
            return
        if not sources:
            self._status("Nothing to undo")
            return
        if domain == "sst":
            self.model.rejected_sst_sources = self.workflow.state.rejected_sst_sources(
                self.model.platform_code, self.model.source_sha256,
            )
            dirty = self.model.affected_sst_chunk_keys(sources)
        else:
            self.model.rejected_sources = self.workflow.state.rejected_sources(
                self.model.platform_code, self.model.source_sha256,
            )
            self.model.refresh_review_scope()
            self._accepted_kinematics = None
            self._kinematics_platform = ""
            dirty = self.model.affected_chunk_keys(sources)
        self.workflow.state.mark_chunks_dirty(self.model.platform_code, dirty)
        if self.resampled is not None:
            self.resample_stale = True
            self.previous_resampled = None
        self._refresh_chunk_combo()
        self._render(reset_time_range=False)
        self._set_selection_domain(domain, clear=True)
        self._status(
            f"Undid the last {domain} decision affecting {len(sources)} observation(s)"
        )

    def resample_now(self) -> None:
        if self.model is None:
            return
        frequency = float(self.frequency_spin.value())
        try:
            result = self.model.resample(
                frequency,
                sst_maximum_gap_minutes=self._sst_maximum_gap_minutes(),
            )
        except ValueError as exc:
            self._status(str(exc), error=True)
            return
        self.previous_resampled = self.resampled if self.resample_stale else None
        self.resampled = result
        self.resample_stale = False
        self.show_resampled.blockSignals(True)
        self.show_resampled.setChecked(True)
        self.show_resampled.blockSignals(False)
        self.workflow.state.mark_resampled(self.model.platform_code, frequency, shown=True)
        self._render(reset_time_range=False)
        self._status(
            f"Resampled {len(result.time):,} regular points every {frequency:g} seconds"
        )

    def _frequency_changed(self, value: float) -> None:
        if self.resampled is not None:
            actual = (
                (int(self.resampled.time[1].astype(np.int64)) - int(self.resampled.time[0].astype(np.int64))) / 1e9
                if len(self.resampled.time) > 1 else float(value)
            )
            if not np.isclose(actual, float(value)):
                self.resample_stale = True
                self._update_resample_status()
                self._render_resampled()

    def _show_resampled_changed(self, shown: bool) -> None:
        if self._loading or self.model is None:
            return
        self.workflow.state.set_resample_shown(self.model.platform_code, shown)
        if shown and self.resampled is None:
            state = self.workflow.state.resample_state(self.model.platform_code)
            frequency = state["frequency_seconds"]
            if not bool(state["stale"]) and frequency is not None:
                try:
                    self.resampled = self.model.resample(
                        float(frequency),
                        sst_maximum_gap_minutes=self._sst_maximum_gap_minutes(),
                    )
                except ValueError as exc:
                    self.resample_stale = True
                    self._status(str(exc), error=True)
        self._render_resampled()

    # ---- selection ----------------------------------------------------

    def _selection_mode_changed(self, enabled: bool) -> None:
        for viewbox in (
            self.map_viewbox, self.speed_viewbox,
            self.acceleration_viewbox, self.sst_viewbox,
            self.gps_pos_qual_viewbox, self.gps_num_sat_viewbox,
        ):
            viewbox.set_selection_enabled(enabled)
        self.selection_button.setText("Selecting…" if enabled else "Rectangle select")

    def _set_selection_domain(self, domain: str, *, clear: bool = False) -> None:
        if domain not in {"position", "sst"}:
            raise ValueError(domain)
        changed = self.selection_domain != domain
        self.selection_domain = domain
        if clear and changed:
            self.selected_sources.clear()
        if not hasattr(self, "target_label"):
            return
        if domain == "sst":
            self.target_label.setText("Editing: SST")
            self.target_label.setStyleSheet(
                "background: #ccfbf1; color: #115e59; font-weight: bold; padding: 4px"
            )
            self.reject_button.setText("R  Reject SST")
            self.restore_button.setText("K  Restore SST")
        else:
            self.target_label.setText("Editing: POSITION")
            self.target_label.setStyleSheet(
                "background: #dbeafe; color: #1e40af; font-weight: bold; padding: 4px"
            )
            self.reject_button.setText("R  Reject position")
            self.restore_button.setText("K  Restore position")
        if clear and changed and self.model is not None:
            self._update_selection_items()

    def clear_selection(self) -> None:
        self.selected_sources.clear()
        self._update_selection_items()

    def _rectangle_selected(
        self, panel: str, x0: float, x1: float, y0: float, y1: float,
    ) -> None:
        if self.model is None:
            return
        domain = "sst" if panel == "sst" else "position"
        self._set_selection_domain(domain, clear=True)
        low_x, high_x = sorted((float(x0), float(x1)))
        low_y, high_y = sorted((float(y0), float(y1)))
        if panel == "map":
            projection = self.model.projected_positions()
            positions = np.asarray([
                self.model.source_to_position[source] for source in self.model.review_order
            ], dtype=np.int64)
            x, y = projection.x_m[positions], projection.y_m[positions]
            inside = (
                np.isfinite(x) & np.isfinite(y)
                & (x >= low_x) & (x <= high_x) & (y >= low_y) & (y <= high_y)
            )
            sources = self.model.sources[positions[inside]].astype(int)
        else:
            values = (
                {
                    source: float(self.model.sst[self.model.source_to_position[source]])
                    for source in self.model.review_order
                }
                if panel == "sst" else self._raw_metric_lookup(panel)
            )
            sources_list: list[int] = []
            for source, metric in values.items():
                position = self.model.source_to_position[source]
                timestamp = self.model.time[position]
                if np.isnat(timestamp) or not np.isfinite(metric):
                    continue
                x = int(timestamp.astype(np.int64)) / 1e9
                if low_x <= x <= high_x and low_y <= metric <= high_y:
                    sources_list.append(source)
            sources = np.asarray(sources_list, dtype=np.int64)
        self.selected_sources = set(map(int, sources))
        if self.selected_sources:
            primary = min(
                self.selected_sources,
                key=lambda source: self.model.review_position[source],
            )
            old_chunk = self.model.current_chunk_index
            self.model.set_selected_source(primary)
            self._refresh_chunk_combo()
            self.save_cursor(quiet=True)
            if old_chunk != self.model.current_chunk_index:
                self._render()
            else:
                self._update_selection_items()
        else:
            self._update_selection_items()
        self._status(
            f"Selected {len(self.selected_sources)} {domain} observation(s)"
        )

    def _map_clicked(self, event) -> None:
        if self.model is None or self.selection_button.isChecked():
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if not self.map_plot.sceneBoundingRect().contains(event.scenePos()):
            return
        self._set_selection_domain("position", clear=True)
        point = self.map_viewbox.mapSceneToView(event.scenePos())
        projection = self.model.projected_positions()
        positions = np.asarray([
            self.model.source_to_position[source] for source in self.model.review_order
        ], dtype=np.int64)
        x, y = projection.x_m[positions], projection.y_m[positions]
        visible = np.isfinite(x) & np.isfinite(y)
        if not visible.any():
            return
        x_range, y_range = self.map_viewbox.viewRange()
        x_span = max(abs(x_range[1] - x_range[0]), 1.0)
        y_span = max(abs(y_range[1] - y_range[0]), 1.0)
        distance = ((x - point.x()) / x_span) ** 2 + ((y - point.y()) / y_span) ** 2
        distance[~visible] = np.inf
        nearest = int(np.argmin(distance))
        if distance[nearest] > 0.0025:
            return
        source = int(self.model.sources[positions[nearest]])
        old_chunk = self.model.current_chunk_index
        self.model.set_selected_source(source)
        self.selected_sources.clear()
        self._refresh_chunk_combo()
        if old_chunk != self.model.current_chunk_index:
            self._render()
        else:
            self._update_selection_items()
        self.save_cursor(quiet=True)

    def _sst_clicked(self, event) -> None:
        if self.model is None or self.selection_button.isChecked():
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if not self.sst_plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.sst_viewbox.mapSceneToView(event.scenePos())
        positions = self.model._ordered_sst_positions(accepted_only=False)
        if not len(positions):
            return
        x = _epoch_seconds(self.model.time[positions])
        y = self.model.sst[positions]
        visible = np.isfinite(x) & np.isfinite(y)
        if not visible.any():
            return
        x_range, y_range = self.sst_viewbox.viewRange()
        x_span = max(abs(x_range[1] - x_range[0]), 1.0)
        y_span = max(abs(y_range[1] - y_range[0]), 1e-9)
        distance = ((x - point.x()) / x_span) ** 2 + ((y - point.y()) / y_span) ** 2
        distance[~visible] = np.inf
        nearest = int(np.argmin(distance))
        if distance[nearest] > 0.0025:
            return
        self._set_selection_domain("sst", clear=True)
        source = int(self.model.sources[positions[nearest]])
        old_chunk = self.model.current_chunk_index
        self.model.set_selected_source(source)
        self._refresh_chunk_combo()
        if old_chunk != self.model.current_chunk_index:
            self._render()
        else:
            self._update_selection_items()
        self.save_cursor(quiet=True)

    def _reference_clicked(
        self, event, panel: str, plot: pg.PlotWidget, viewbox: SelectionViewBox,
    ) -> None:
        if self.model is None or self.selection_button.isChecked():
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if not plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = viewbox.mapSceneToView(event.scenePos())
        positions = self.model._ordered_positions(accepted_only=False)
        if not len(positions):
            return
        values = (
            self.model.gps_pos_qual if panel == "gps_pos_qual"
            else self.model.gps_num_sat
        )
        x = _epoch_seconds(self.model.time[positions])
        y = values[positions]
        visible = np.isfinite(x) & np.isfinite(y)
        if not visible.any():
            return
        x_range, y_range = viewbox.viewRange()
        x_span = max(abs(x_range[1] - x_range[0]), 1.0)
        y_span = max(abs(y_range[1] - y_range[0]), 1e-9)
        distance = ((x - point.x()) / x_span) ** 2 + ((y - point.y()) / y_span) ** 2
        distance[~visible] = np.inf
        nearest = int(np.argmin(distance))
        if distance[nearest] > 0.0025:
            return
        self._set_selection_domain("position", clear=True)
        source = int(self.model.sources[positions[nearest]])
        old_chunk = self.model.current_chunk_index
        self.model.set_selected_source(source)
        self._refresh_chunk_combo()
        if old_chunk != self.model.current_chunk_index:
            self._render()
        else:
            self._update_selection_items()
        self.save_cursor(quiet=True)

    # ---- rendering ----------------------------------------------------

    def _refresh_chunk_combo(self) -> None:
        if self.model is None:
            return
        states = self.workflow.state.chunk_states(
            self.model.platform_code, self.model.source_sha256,
        )
        self.chunk_combo.blockSignals(True)
        self.chunk_combo.clear()
        for index, chunk in enumerate(self.model.chunks):
            state = states.get(chunk.key, "unreviewed")
            symbol = {"reviewed": "✓", "dirty": "!", "unreviewed": "○"}[state]
            current = "▶" if index == self.model.current_chunk_index else " "
            self.chunk_combo.addItem(
                f"{current} {symbol} {index + 1}/{len(self.model.chunks)}  {chunk.label}", index,
            )
        self.chunk_combo.setCurrentIndex(self.model.current_chunk_index)
        self.chunk_combo.blockSignals(False)
        active_keys = {chunk.key for chunk in self.model.chunks}
        reviewed = sum(states.get(key) == "reviewed" for key in active_keys)
        dirty = sum(states.get(key) == "dirty" for key in active_keys)
        self.progress_label.setText(
            f"Reviewed {reviewed}/{len(self.model.chunks)}" + (f" · {dirty} changed" if dirty else "")
        )
        self._update_scope_label()

    def _update_scope_label(self) -> None:
        if self.model is None:
            self.scope_label.clear()
            return
        model = self.model
        if np.isnat(model.deployment_start_time):
            self.scope_label.setText("Deployment start unresolved")
            self.scope_label.setStyleSheet("color: #b91c1c; font-weight: bold")
            return
        start = np.datetime_as_string(
            model.deployment_start_time, unit="m", timezone="UTC",
        )
        end = np.datetime_as_string(model.review_end_time, unit="m", timezone="UTC")
        if model.maximum_accepted_span_days is None:
            text = f"Deployment {start} · review through {end}"
        else:
            text = (
                f"Deployment {start} · accepted span "
                f"{model.accepted_review_span_days:.2f}/"
                f"{model.maximum_accepted_span_days:g} d · through {end}"
            )
        self.scope_label.setText(text)
        self.scope_label.setStyleSheet("")

    def _render(self, *, reset_time_range: bool = True) -> None:
        if self.model is None:
            return
        model = self.model
        projection = model.projected_positions()
        raw_positions = model._ordered_positions(accepted_only=False)
        accepted_positions = model._ordered_positions(accepted_only=True)
        chunk = model.current_chunk
        chunk_sources = set() if chunk is None else set(chunk.source_indices)
        chunk_positions = np.asarray([
            model.source_to_position[source] for source in chunk_sources
            if source not in model.rejected_sources
        ], dtype=np.int64)
        if len(chunk_positions):
            chunk_positions = chunk_positions[np.lexsort((
                model.sources[chunk_positions],
                np.where(
                    np.isnat(model.time[chunk_positions]), np.iinfo(np.int64).max,
                    model.time[chunk_positions].astype(np.int64),
                ),
            ))]
        review_positions = np.asarray([
            model.source_to_position[source] for source in model.review_order
        ], dtype=np.int64)
        rejected_positions = np.asarray([
            model.source_to_position[source] for source in model.review_order
            if source in model.rejected_sources
        ], dtype=np.int64)

        self.map_items["raw_line"].setData(
            projection.x_m[raw_positions], projection.y_m[raw_positions], connect="finite",
        )
        self.map_items["accepted_line"].setData(
            projection.x_m[accepted_positions], projection.y_m[accepted_positions], connect="finite",
        )
        self.map_items["chunk_line"].setData(
            projection.x_m[chunk_positions], projection.y_m[chunk_positions], connect="finite",
        )
        _set_finite_scatter_data(
            self.map_items["raw_points"],
            projection.x_m[review_positions], projection.y_m[review_positions],
        )
        _set_finite_scatter_data(
            self.map_items["chunk_points"],
            projection.x_m[chunk_positions], projection.y_m[chunk_positions],
        )
        _set_finite_scatter_data(
            self.map_items["rejected"],
            projection.x_m[rejected_positions], projection.y_m[rejected_positions],
        )

        original, accepted = self._get_kinematics()
        self._set_time_data(self.speed_items, original, accepted, "speed_m_s", chunk_sources)
        self._render_speed_warnings(accepted, projection)
        self._set_time_data(
            self.acceleration_items, original, accepted, "acceleration_m_s2", chunk_sources,
        )
        self._set_sst_data(chunk_sources)
        self._set_reference_data(
            self.gps_pos_qual_items, model.gps_pos_qual, chunk_sources,
        )
        self._set_reference_data(
            self.gps_num_sat_items, model.gps_num_sat, chunk_sources,
        )
        self._render_drogue_endpoint()
        self._render_resampled()
        self._update_selection_items()
        if reset_time_range:
            self._set_chunk_time_range()
        elif self.auto_fit_visible_y.isChecked():
            self.fit_visible_y()
        self._update_resample_status()
        self._update_point_label()

    def _set_time_data(
        self,
        items: dict[str, pg.GraphicsObject],
        original: KinematicSeries,
        accepted: KinematicSeries,
        field: str,
        chunk_sources: set[int],
    ) -> None:
        raw_y = np.asarray(getattr(original, field), dtype=float)
        accepted_y = np.asarray(getattr(accepted, field), dtype=float)
        raw_x = _epoch_seconds(original.time)
        accepted_x = _epoch_seconds(accepted.time)
        items["raw"].setData(raw_x, raw_y)
        items["accepted"].setData(accepted_x, accepted_y)
        _set_finite_scatter_data(items["raw_points"], raw_x, raw_y)
        _set_finite_scatter_data(items["accepted_points"], accepted_x, accepted_y)
        chunk = np.asarray([
            int(source) in chunk_sources for source in accepted.source_indices
        ], dtype=bool)
        _set_finite_scatter_data(
            items["chunk"], accepted_x[chunk], accepted_y[chunk],
        )
        original_lookup = {
            int(source): (x, y)
            for source, x, y in zip(original.source_indices, raw_x, raw_y)
        }
        rejected_values = [
            original_lookup[source] for source in sorted(self.model.rejected_sources)
            if source in original_lookup and np.isfinite(original_lookup[source][1])
        ]
        if rejected_values:
            rejected_x, rejected_y = zip(*rejected_values)
            _set_finite_scatter_data(items["rejected"], rejected_x, rejected_y)
        else:
            items["rejected"].setData([], [])

    def _render_speed_warnings(
        self,
        accepted: KinematicSeries,
        projection,
    ) -> None:
        assert self.model is not None
        high, zero = speed_warning_masks(
            accepted.speed_m_s,
            high_speed_m_s=self.workflow.config.high_speed_warning_m_s,
            zero_speed_tolerance_m_s=(
                self.workflow.config.zero_speed_tolerance_m_s
            ),
        )
        time = _epoch_seconds(accepted.time)
        positions = np.asarray([
            self.model.source_to_position[int(source)]
            for source in accepted.source_indices
        ], dtype=np.int64)

        _set_finite_scatter_data(
            self.speed_items["high_speed_warning"],
            time[high], accepted.speed_m_s[high],
        )
        _set_finite_scatter_data(
            self.speed_items["zero_speed_warning"],
            time[zero], accepted.speed_m_s[zero],
        )
        _set_finite_scatter_data(
            self.map_items["high_speed_warning"],
            projection.x_m[positions[high]], projection.y_m[positions[high]],
        )
        _set_finite_scatter_data(
            self.map_items["zero_speed_warning"],
            projection.x_m[positions[zero]], projection.y_m[positions[zero]],
        )

        high_setting = self.workflow.config.high_speed_warning_m_s
        zero_setting = self.workflow.config.zero_speed_tolerance_m_s
        high_text = (
            "disabled" if high_setting is None
            else f"{int(high.sum())} speed > {high_setting:g} m/s"
        )
        zero_text = (
            "disabled" if zero_setting is None
            else f"{int(zero.sum())} speed <= {zero_setting:g} m/s"
        )
        self.speed_warning_label.setText(
            "Visual warnings only (accepted adjacent edges; marker at later point): "
            f"magenta triangle = {high_text}; brown square = {zero_text}."
        )

    def _set_sst_data(self, chunk_sources: set[int]) -> None:
        assert self.model is not None
        raw_positions = self.model._ordered_sst_positions(accepted_only=False)
        accepted_positions = self.model._ordered_sst_positions(accepted_only=True)
        raw_x = _epoch_seconds(self.model.time[raw_positions])
        accepted_x = _epoch_seconds(self.model.time[accepted_positions])
        self.sst_items["raw"].setData(raw_x, self.model.sst[raw_positions])
        self.sst_items["accepted"].setData(
            accepted_x, self.model.sst[accepted_positions],
        )
        _set_finite_scatter_data(
            self.sst_items["raw_points"], raw_x, self.model.sst[raw_positions],
        )
        _set_finite_scatter_data(
            self.sst_items["accepted_points"],
            accepted_x, self.model.sst[accepted_positions],
        )
        chunk = np.asarray([
            int(source) in chunk_sources
            for source in self.model.sources[accepted_positions]
        ], dtype=bool)
        _set_finite_scatter_data(
            self.sst_items["chunk"],
            accepted_x[chunk], self.model.sst[accepted_positions][chunk],
        )
        rejected_positions = np.asarray([
            self.model.source_to_position[source]
            for source in self.model.rejected_sst_sources
            if source in self.model.review_position
            and np.isfinite(self.model.sst[self.model.source_to_position[source]])
            and not np.isnat(self.model.time[self.model.source_to_position[source]])
        ], dtype=np.int64)
        _set_finite_scatter_data(
            self.sst_items["rejected"],
            _epoch_seconds(self.model.time[rejected_positions]),
            self.model.sst[rejected_positions],
        )
        position_rejected = np.asarray([
            self.model.source_to_position[source]
            for source in self.model.rejected_sources
            if source in self.model.review_position
            and source not in self.model.rejected_sst_sources
            and np.isfinite(self.model.sst[self.model.source_to_position[source]])
            and not np.isnat(self.model.time[self.model.source_to_position[source]])
        ], dtype=np.int64)
        _set_finite_scatter_data(
            self.sst_items["position_rejected"],
            _epoch_seconds(self.model.time[position_rejected]),
            self.model.sst[position_rejected],
        )

    def _set_reference_data(
        self,
        items: dict[str, pg.GraphicsObject],
        values: np.ndarray,
        chunk_sources: set[int],
    ) -> None:
        assert self.model is not None
        positions = self.model._ordered_positions(accepted_only=False)
        x = _epoch_seconds(self.model.time[positions])
        y = np.asarray(values[positions], dtype=float)
        items["raw"].setData(x, y)
        _set_finite_scatter_data(items["raw_points"], x, y)
        source_values = self.model.sources[positions].astype(int)
        chunk = np.asarray(
            [source in chunk_sources for source in source_values], dtype=bool,
        )
        _set_finite_scatter_data(items["chunk"], x[chunk], y[chunk])
        rejected = np.asarray(
            [source in self.model.rejected_sources for source in source_values],
            dtype=bool,
        )
        _set_finite_scatter_data(items["rejected"], x[rejected], y[rejected])

    def _render_drogue_endpoint(self) -> None:
        assert self.model is not None
        item = self.workflow.platform_by_code[self.model.platform_code]
        visible = not np.isnat(item.stop_time)
        for values in (
            self.speed_items, self.acceleration_items, self.sst_items,
            self.gps_pos_qual_items, self.gps_num_sat_items,
        ):
            values["drogue_line"].setVisible(visible)
            if visible:
                values["drogue_line"].setPos(int(item.stop_time.astype(np.int64)) / 1e9)
        self.map_items["drogue_end"].setVisible(visible)
        if not visible:
            return
        valid = (
            self.model.hard_time_scope_mask
            & np.isfinite(self.model.longitude)
            & np.isfinite(self.model.latitude)
        )
        positions = np.flatnonzero(valid)
        if len(positions):
            last = positions[np.argmax(self.model.time[positions])]
            projection = self.model.projected_positions()
            _set_finite_scatter_data(
                self.map_items["drogue_end"],
                [projection.x_m[last]], [projection.y_m[last]],
            )

    def _render_resampled(self) -> None:
        if self.model is None:
            return
        shown = self.show_resampled.isChecked()
        current_visible = shown and self.resampled is not None
        previous_visible = shown and self.previous_resampled is not None
        self.map_items["resampled"].setVisible(current_visible)
        self.map_items["resampled_points"].setVisible(current_visible)
        self.map_items["previous_resampled"].setVisible(previous_visible)
        for items in (self.speed_items, self.acceleration_items, self.sst_items):
            items["resampled"].setVisible(current_visible)
            items["resampled_points"].setVisible(current_visible)
            items["previous_resampled"].setVisible(previous_visible)
        if current_visible:
            self._set_resampled_data(self.resampled, current=True)
        if previous_visible:
            self._set_resampled_data(self.previous_resampled, current=False)
        color = "#ea580c" if self.resample_stale else "#7c3aed"
        self.map_items["resampled"].setPen(
            pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.DashLine)
        )
        for items in (self.speed_items, self.acceleration_items, self.sst_items):
            items["resampled"].setPen(
                pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.DashLine)
            )
            items["resampled_points"].setPen(pg.mkPen(color))
        self.map_items["resampled_points"].setPen(pg.mkPen(color))

    def _set_resampled_data(self, values: ResampledTrajectory, *, current: bool) -> None:
        assert self.model is not None
        key = "resampled" if current else "previous_resampled"
        projection = self.model.projected_positions()
        x, y = projection.forward.transform(values.longitude, values.latitude)
        self.map_items[key].setData(np.asarray(x), np.asarray(y), connect="finite")
        time = _epoch_seconds(values.time)
        self.speed_items[key].setData(time, values.kinematics.speed_m_s)
        self.acceleration_items[key].setData(time, values.kinematics.acceleration_m_s2)
        self.sst_items[key].setData(time, values.sst)
        if current and self.model.current_chunk is not None:
            chunk = self.model.current_chunk
            if np.isnat(chunk.start_time):
                inside = np.zeros(len(values.time), dtype=bool)
            else:
                inside = (values.time >= chunk.start_time) & (values.time < chunk.end_time)
            _set_finite_scatter_data(
                self.map_items["resampled_points"],
                np.asarray(x)[inside], np.asarray(y)[inside],
            )
            _set_finite_scatter_data(
                self.speed_items["resampled_points"],
                time[inside], values.kinematics.speed_m_s[inside],
            )
            _set_finite_scatter_data(
                self.acceleration_items["resampled_points"],
                time[inside], values.kinematics.acceleration_m_s2[inside],
            )
            _set_finite_scatter_data(
                self.sst_items["resampled_points"],
                time[inside], values.sst[inside],
            )

    def _raw_metric_lookup(self, panel: str) -> dict[int, float]:
        assert self.model is not None
        if panel in {"gps_pos_qual", "gps_num_sat"}:
            values = (
                self.model.gps_pos_qual if panel == "gps_pos_qual"
                else self.model.gps_num_sat
            )
            return {
                source: float(values[self.model.source_to_position[source]])
                for source in self.model.review_order
            }
        original, accepted = self._get_kinematics()
        field = "speed_m_s" if panel == "speed" else "acceleration_m_s2"
        original_values = {
            int(source): float(value)
            for source, value in zip(original.source_indices, getattr(original, field))
        }
        accepted_values = {
            int(source): float(value)
            for source, value in zip(accepted.source_indices, getattr(accepted, field))
        }
        return {
            source: (
                original_values.get(source, np.nan)
                if source in self.model.rejected_sources
                else accepted_values.get(source, np.nan)
            )
            for source in self.model.review_order
        }

    def _get_kinematics(self) -> tuple[KinematicSeries, KinematicSeries]:
        assert self.model is not None
        rejected = frozenset(self.model.rejected_sources)
        if self._kinematics_platform != self.model.platform_code:
            self._kinematics_platform = self.model.platform_code
            self._original_kinematics = None
            self._accepted_kinematics = None
        if self._original_kinematics is None:
            self._original_kinematics = self.model.raw_kinematics(accepted_only=False)
        if self._accepted_kinematics is None or self._kinematics_rejected != rejected:
            self._accepted_kinematics = self.model.raw_kinematics(accepted_only=True)
            self._kinematics_rejected = rejected
        return self._original_kinematics, self._accepted_kinematics

    def _update_selection_items(self) -> None:
        if self.model is None:
            return
        projection = self.model.projected_positions()
        selected = sorted(self.selected_sources)
        position_selected = selected if self.selection_domain == "position" else []
        selected_positions = np.asarray([
            self.model.source_to_position[source] for source in position_selected
        ], dtype=np.int64)
        _set_finite_scatter_data(
            self.map_items["selected"],
            projection.x_m[selected_positions], projection.y_m[selected_positions],
        )
        cursor_position = (
            None if self.model.selected_source is None
            else self.model.source_to_position[int(self.model.selected_source)]
        )
        if cursor_position is None:
            self.map_items["cursor"].setData([], [])
        else:
            _set_finite_scatter_data(
                self.map_items["cursor"],
                [projection.x_m[cursor_position]], [projection.y_m[cursor_position]],
            )
        for panel, items in (
            ("speed", self.speed_items), ("acceleration", self.acceleration_items),
        ):
            values = self._raw_metric_lookup(panel)
            coordinates = []
            for source in position_selected:
                position = self.model.source_to_position[source]
                if not np.isnat(self.model.time[position]) and np.isfinite(values[source]):
                    coordinates.append((
                        int(self.model.time[position].astype(np.int64)) / 1e9,
                        values[source],
                    ))
            if coordinates:
                x, y = zip(*coordinates)
                _set_finite_scatter_data(items["selected"], x, y)
            else:
                items["selected"].setData([], [])
            if cursor_position is not None:
                source = int(self.model.selected_source)
                timestamp = self.model.time[cursor_position]
                metric = values.get(source, np.nan)
                if not np.isnat(timestamp) and np.isfinite(metric):
                    _set_finite_scatter_data(
                        items["cursor"],
                        [int(timestamp.astype(np.int64)) / 1e9], [metric],
                    )
                else:
                    items["cursor"].setData([], [])
            else:
                items["cursor"].setData([], [])
        sst_selected = selected if self.selection_domain == "sst" else []
        sst_coordinates = []
        for source in sst_selected:
            position = self.model.source_to_position[source]
            timestamp = self.model.time[position]
            value = self.model.sst[position]
            if not np.isnat(timestamp) and np.isfinite(value):
                sst_coordinates.append((
                    int(timestamp.astype(np.int64)) / 1e9, float(value),
                ))
        if sst_coordinates:
            x, y = zip(*sst_coordinates)
            _set_finite_scatter_data(self.sst_items["selected"], x, y)
        else:
            self.sst_items["selected"].setData([], [])
        if cursor_position is not None:
            timestamp = self.model.time[cursor_position]
            value = self.model.sst[cursor_position]
            if not np.isnat(timestamp) and np.isfinite(value):
                _set_finite_scatter_data(
                    self.sst_items["cursor"],
                    [int(timestamp.astype(np.int64)) / 1e9], [float(value)],
                )
            else:
                self.sst_items["cursor"].setData([], [])
        else:
            self.sst_items["cursor"].setData([], [])
        for panel, items in (
            ("gps_pos_qual", self.gps_pos_qual_items),
            ("gps_num_sat", self.gps_num_sat_items),
        ):
            values = self._raw_metric_lookup(panel)
            coordinates = []
            for source in position_selected:
                position = self.model.source_to_position[source]
                timestamp = self.model.time[position]
                value = values[source]
                if not np.isnat(timestamp) and np.isfinite(value):
                    coordinates.append((
                        int(timestamp.astype(np.int64)) / 1e9, value,
                    ))
            if coordinates:
                x, y = zip(*coordinates)
                _set_finite_scatter_data(items["selected"], x, y)
            else:
                items["selected"].setData([], [])
            if cursor_position is not None:
                timestamp = self.model.time[cursor_position]
                value = values[int(self.model.selected_source)]
                if not np.isnat(timestamp) and np.isfinite(value):
                    _set_finite_scatter_data(
                        items["cursor"],
                        [int(timestamp.astype(np.int64)) / 1e9], [value],
                    )
                else:
                    items["cursor"].setData([], [])
            else:
                items["cursor"].setData([], [])
        self._update_point_label()

    def _set_chunk_time_range(self) -> None:
        if self.model is None or self.model.current_chunk is None:
            return
        chunk = self.model.current_chunk
        if np.isnat(chunk.start_time):
            self.speed_plot.enableAutoRange(axis="x")
            self.fit_visible_y()
            return
        start = int(chunk.start_time.astype(np.int64)) / 1e9
        end = int(chunk.end_time.astype(np.int64)) / 1e9
        padding = max((end - start) * 0.04, 1.0)
        self.speed_plot.setXRange(start - padding, end + padding, padding=0)
        self.fit_visible_y()

    def _time_x_range_changed(self, *_args) -> None:
        if self.auto_fit_visible_y.isChecked():
            self.fit_visible_y()

    def _auto_fit_visible_y_changed(self, enabled: bool) -> None:
        if enabled:
            self.fit_visible_y()

    def fit_visible_y(self) -> None:
        """Fit each metric to the shared visible time span without changing it."""
        if self.model is None or self._fitting_time_y:
            return
        self._fitting_time_y = True
        try:
            x_range = self.speed_viewbox.viewRange()[0]
            for plot, items, nonnegative, item_names in (
                (self.speed_plot, self.speed_items, True, ("accepted", "resampled")),
                (
                    self.acceleration_plot, self.acceleration_items, True,
                    ("accepted", "resampled"),
                ),
                (self.sst_plot, self.sst_items, False, ("accepted", "resampled")),
                (self.gps_pos_qual_plot, self.gps_pos_qual_items, True, ("raw",)),
                (self.gps_num_sat_plot, self.gps_num_sat_items, True, ("raw",)),
            ):
                visible_values: list[np.ndarray] = []
                for name in item_names:
                    item = items[name]
                    if not item.isVisible():
                        continue
                    x, y = item.getData()
                    if x is None or y is None:
                        continue
                    x_values = np.asarray(x, dtype=float)
                    y_values = np.asarray(y, dtype=float)
                    inside = (
                        np.isfinite(x_values) & np.isfinite(y_values)
                        & (x_values >= x_range[0]) & (x_values <= x_range[1])
                    )
                    if inside.any():
                        visible_values.append(y_values[inside])
                if not visible_values:
                    continue
                values = np.concatenate(visible_values)
                low, high = float(np.min(values)), float(np.max(values))
                span = high - low
                if not np.isfinite(span) or span <= 0:
                    span = max(abs(high) * 0.1, 1e-6)
                margin = span * 0.08
                bottom = max(0.0, low - margin) if nonnegative else low - margin
                plot.setYRange(bottom, high + margin, padding=0)
        finally:
            self._fitting_time_y = False

    def _update_point_label(self) -> None:
        if self.model is None or self.model.selected_source is None:
            self.point_label.setText("No raw observation selected")
            return
        source = int(self.model.selected_source)
        position = self.model.source_to_position[source]
        time = (
            "invalid timestamp" if np.isnat(self.model.time[position])
            else np.datetime_as_string(self.model.time[position], unit="s", timezone="UTC")
        )
        position_status = (
            "REJECTED" if source in self.model.rejected_sources else "accepted"
        )
        if not np.isfinite(self.model.sst[position]):
            sst_text = "missing"
        else:
            sst_status = (
                "REJECTED" if source in self.model.rejected_sst_sources else "accepted"
            )
            sst_text = f"{self.model.sst[position]:.5g} °C ({sst_status})"
        selection = f" · {len(self.selected_sources)} selected" if self.selected_sources else ""
        gps_pos_qual = (
            "missing" if not np.isfinite(self.model.gps_pos_qual[position])
            else f"{self.model.gps_pos_qual[position]:g}"
        )
        gps_num_sat = (
            "missing" if not np.isfinite(self.model.gps_num_sat[position])
            else f"{self.model.gps_num_sat[position]:g}"
        )
        self.point_label.setText(
            f"Raw observation {source} · {time} · lon {self.model.longitude[position]:.7g} · "
            f"lat {self.model.latitude[position]:.7g} · position {position_status} · "
            f"SST {sst_text} · GPS PosQual {gps_pos_qual} · satellites {gps_num_sat} · "
            f"target {self.selection_domain.upper()}{selection}"
        )

    def _update_resample_status(self) -> None:
        if self.resampled is None:
            text, style = ("Resample is stale; press Resample now", "color: #c2410c; font-weight: bold") if self.resample_stale else ("Not resampled", "")
        elif self.resample_stale:
            text, style = "STALE — orange result predates raw edits", "color: #c2410c; font-weight: bold"
        else:
            text, style = f"Current — {len(self.resampled.time):,} regular points", "color: #6d28d9"
        self.resample_status.setText(text)
        self.resample_status.setStyleSheet(style)

    # ---- map view and status -----------------------------------------

    def fit_chunk(self) -> None:
        if self.model is None or self.model.current_chunk is None:
            return
        projection = self.model.projected_positions()
        positions = np.asarray([
            self.model.source_to_position[source]
            for source in self.model.current_chunk.source_indices
        ], dtype=np.int64)
        self._fit_positions(projection.x_m[positions], projection.y_m[positions])
        self._map_fit = "chunk"

    def fit_trajectory(self) -> None:
        if self.model is None:
            return
        projection = self.model.projected_positions()
        positions = np.asarray([
            self.model.source_to_position[source] for source in self.model.review_order
        ], dtype=np.int64)
        self._fit_positions(projection.x_m[positions], projection.y_m[positions])
        self._map_fit = "trajectory"

    def _fit_positions(self, x: np.ndarray, y: np.ndarray) -> None:
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            return
        low_x, high_x = float(np.min(x[finite])), float(np.max(x[finite]))
        low_y, high_y = float(np.min(y[finite])), float(np.max(y[finite]))
        if low_x == high_x:
            low_x, high_x = low_x - 1, high_x + 1
        if low_y == high_y:
            low_y, high_y = low_y - 1, high_y + 1
        self.map_viewbox.setRange(
            xRange=(low_x, high_x), yRange=(low_y, high_y), padding=0.08,
        )

    def _zoom_map(self, factor: float) -> None:
        self.map_viewbox.scaleBy((factor, factor))
        self._map_fit = "manual"

    def _follow_cursor(self) -> None:
        if self.model is None or self.model.selected_source is None:
            return
        position = self.model.source_to_position[int(self.model.selected_source)]
        projection = self.model.projected_positions()
        x, y = projection.x_m[position], projection.y_m[position]
        if not np.isfinite(x) or not np.isfinite(y):
            return
        (x0, x1), (y0, y1) = self.map_viewbox.viewRange()
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            self.map_viewbox.setRange(
                xRange=(x - (x1 - x0) / 2, x + (x1 - x0) / 2),
                yRange=(y - (y1 - y0) / 2, y + (y1 - y0) / 2),
                padding=0,
            )

    def _status(self, message: str, *, error: bool = False) -> None:
        self.statusBar().setStyleSheet("color: #b91c1c" if error else "")
        self.statusBar().showMessage(message)

    def closeEvent(self, event):
        if self._build_thread is not None:
            self._status("Wait for the final Zarr build to finish before closing", error=True)
            event.ignore()
            return
        if self._closed:
            event.accept()
            return
        try:
            self.save_cursor(quiet=True)
        finally:
            self.workflow.close()
            self._closed = True
            event.accept()


def run_manual_qc_gui(config_path: str) -> int:
    """Start the independent manual-QC Qt application."""
    pg.setConfigOptions(antialias=False)
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    application.setApplicationName("drifterlab manual QC")
    workflow = ManualQCWorkflow(config_path)
    window = ManualQCWindow(workflow)
    window.show()
    return int(application.exec())
