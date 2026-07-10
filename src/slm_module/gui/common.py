"""Shared GUI plumbing: thread-pool workers + the live progress dialog.

Extracted from app.py so page modules (e.g. pipeline_page) can use them
without importing the (huge) main-window module and creating an import cycle.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from ..calibration.calibration_new import CalibrationProgress


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "—"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)


class FunctionWorker(QtCore.QRunnable):
    def __init__(self, func: Callable[[], Any]):
        super().__init__()
        self.func = func
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(self.func())
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class CalibrationProgressDialog(QtWidgets.QDialog):
    """Live view of an OSA calibration run: phase, progress bar, log and plot.

    update_progress() is called on the GUI thread for every measured step; the
    plot itself is redrawn on a timer so a fast stream of points cannot flood
    the event loop. finish() freezes the view and enables Close. Unknown phase
    names fall back to a generic title/labels, so new stages can never KeyError.
    """

    _PHASES = {
        "min_max": ("Step 1 / 3 · Min/Max level sweep", "Level", "Output power (W)"),
        "wavelength": (
            "Step 2 / 3 · Wavelength mapping",
            "x coordinate (px)",
            "Wavelength (nm)",
        ),
        "intensity": (
            "Step 3 / 3 · Intensity vs level",
            "Level",
            "Normalized intensity",
        ),
        "fast_center": (
            "Fast channel setup - 778 nm center",
            "x coordinate (px)",
            "Wavelength (nm)",
        ),
        "batch_intensity": (
            "Fast channel calibration",
            "Level",
            "Mean normalized intensity",
        ),
        "tpa_center": (
            "TPA centre scan",
            "Centre wavelength (nm)",
            "Net signal (V)",
        ),
        "pair_eta": (
            "Step 6 · TPA pair efficiency",
            "Grid point",
            "Signal (V)",
        ),
        "comb_phase": (
            "Step 7 · Comb phase sweep",
            "Sweep point",
            "Signal (V)",
        ),
    }

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        on_stop: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration Progress")
        self.setModal(False)
        self.resize(760, 600)
        self._on_stop = on_stop
        self._running = True
        self._phase: str | None = None
        self._phase_start: float | None = None
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._dirty = False

        layout = QtWidgets.QVBoxLayout(self)

        self.phase_label = QtWidgets.QLabel("Preparing…")
        self.phase_label.setObjectName("PageSubtitle")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m  (%p%)")
        self.eta_label = QtWidgets.QLabel("Elapsed 0:00 · ETA —")
        self.status_label = QtWidgets.QLabel("\N{EN DASH}")
        self.status_label.setWordWrap(True)

        self.figure = Figure(figsize=(6, 3.2), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self._style_axes()

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("LogBox")
        self.log.setMaximumHeight(140)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setProperty("variant", "danger")
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.setEnabled(False)
        self.stop_button.clicked.connect(self._handle_stop)
        self.close_button.clicked.connect(self.close)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.close_button)

        layout.addWidget(self.phase_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.eta_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.log)
        layout.addLayout(buttons)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._redraw)
        self._timer.start(120)

    def _style_axes(self) -> None:
        self.figure.patch.set_facecolor("#101820")
        axes = self.axes
        axes.set_facecolor("#101820")
        axes.grid(True, color="#2b3a42", linewidth=0.7)
        axes.tick_params(colors="#d8dee9")
        axes.xaxis.label.set_color("#d8dee9")
        axes.yaxis.label.set_color("#d8dee9")
        for spine in axes.spines.values():
            spine.set_color("#41515c")

    def update_progress(self, progress: CalibrationProgress) -> None:
        if progress.phase != self._phase:
            self._enter_phase(progress.phase)
        total = max(int(progress.total), 1)
        done = min(int(progress.step) + 1, total)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        self.status_label.setText(progress.message)
        self._update_eta(done, total)
        if progress.x is not None and progress.y is not None:
            self._xs.append(float(progress.x))
            self._ys.append(float(progress.y))
            self._dirty = True

    def _enter_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_start = time.perf_counter()
        self._xs.clear()
        self._ys.clear()
        title, _xlabel, _ylabel = self._PHASES.get(phase, (phase, "x", "y"))
        self.phase_label.setText(title)
        self.log.appendPlainText(f"\N{BLACK RIGHT-POINTING TRIANGLE} {title}")
        self.eta_label.setText("Elapsed 0:00 · ETA —")
        self._dirty = True

    def _update_eta(self, done: int, total: int) -> None:
        """Estimate time remaining from the average pace of this phase so far."""
        if self._phase_start is None:
            return
        elapsed = time.perf_counter() - self._phase_start
        if done <= 0:
            self.eta_label.setText(f"Elapsed {_format_duration(elapsed)} · ETA —")
            return
        remaining = (elapsed / done) * max(total - done, 0)
        self.eta_label.setText(
            f"Elapsed {_format_duration(elapsed)} · ETA {_format_duration(remaining)}"
        )

    def _redraw(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self.axes.clear()
        self._style_axes()
        phase = self._phase or ""
        _title, xlabel, ylabel = self._PHASES.get(phase, (phase, "x", "y"))
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if self._xs:
            self.axes.plot(
                self._xs,
                self._ys,
                color="#47b8e0",
                marker="o",
                markersize=3,
                linewidth=1.0,
            )
        self.canvas.draw_idle()

    def finish(self, success: bool, message: str) -> None:
        self._running = False
        self._timer.stop()
        self._dirty = True
        self._redraw()
        self.status_label.setText(message)
        self.log.appendPlainText(message)
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)
        if success:
            self.progress_bar.setValue(self.progress_bar.maximum())

    def _handle_stop(self) -> None:
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping…")
        if self._on_stop is not None:
            self._on_stop()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # closing mid-run requests a stop but still lets the window close
        if self._running and self._on_stop is not None:
            self._on_stop()
        self._timer.stop()
        super().closeEvent(event)


__all__ = [
    "CalibrationProgressDialog",
    "FunctionWorker",
    "WorkerSignals",
    "_format_duration",
]
