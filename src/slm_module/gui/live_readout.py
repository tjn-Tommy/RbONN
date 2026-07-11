"""Live readout panel: sample-listener bridge + rolling strip chart, docked.

Every TPA readout in the codebase funnels through ``monitor_cycle()`` on
whichever monitor instrument is connected (``ScopeController`` or
``DAQController``); both notify their registered sample listeners on the
*calling worker thread*. :class:`MonitorSampleBridge` is such a listener: it
only emits a queued Qt signal, so the sample crosses safely onto the GUI
thread where a :class:`LiveSampleView` appends it to a rolling mean ± std
strip chart -- TPA centre scans, step 6/7, pipeline stages and encoder-page
reads all stream here, no matter which module triggered them.

This panel is a *passive mirror* of readings other modules trigger. It is
deliberately separate from the DAQ Monitor page (``_build_daq_monitor_page``
in app.py), which is an *active* live voltmeter driving its own continuous
acquisition loop; when that loop runs, its samples stream here too, since
they go through the same ``monitor_cycle()`` funnel.

:class:`LiveReadoutDock` bundles the bridge and the view into one
self-contained dock widget, so the main window only creates it and calls
``watch(controller)`` / ``unwatch(controller)`` on connect/disconnect.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets


class MonitorSampleBridge(QtCore.QObject):
    """Thread-safe hand-off of MonitorSamples from worker threads to the GUI.

    ``on_sample`` is registered as a scope/DAQ sample listener; the ``emit``
    is queued by Qt, so slots always run on the GUI thread.
    """

    sample_ready = QtCore.pyqtSignal(object)         # MonitorSample

    def on_sample(self, sample) -> None:
        self.sample_ready.emit(sample)


def _format_volts(value: float) -> str:
    """Human scale: mV below 1 V, µV below 1 mV."""
    v = abs(float(value))
    if not np.isfinite(v):
        return f"{value:g} V"
    if v >= 1.0 or v == 0.0:
        return f"{value:.4g} V"
    if v >= 1e-3:
        return f"{value * 1e3:.4g} mV"
    return f"{value * 1e6:.4g} \N{MICRO SIGN}V"


class LiveSampleView(QtWidgets.QWidget):
    """Rolling mean ± std strip chart of monitor readings.

    Keeps the last ``max_points`` samples and repaints at most every
    ``interval_ms`` (readings may arrive faster than the canvas can draw).
    A Pause box freezes the display without unhooking the listener; Clear
    restarts the record.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        interval_ms: int = 200,
        max_points: int = 500,
    ) -> None:
        super().__init__(parent)
        self._samples: deque[tuple[int, float, float, float]] = deque(
            maxlen=int(max_points)
        )
        self._dirty = False
        self.sample_count = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        bar = QtWidgets.QHBoxLayout()
        self.pause_check = QtWidgets.QCheckBox("Pause")
        self.pause_check.setToolTip(
            "Freeze the display; readings keep streaming and resume on uncheck"
        )
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        self.status_label = QtWidgets.QLabel("waiting for a reading…")
        bar.addWidget(self.pause_check)
        bar.addWidget(self.clear_button)
        bar.addWidget(self.status_label, 1)
        layout.addLayout(bar)

        self.figure = Figure(figsize=(6, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(180)
        layout.addWidget(self.canvas, 1)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._redraw_if_dirty)
        self._timer.start()

        self._draw_placeholder()

    # ---- GUI-thread slots --------------------------------------------------
    def add_sample(self, sample) -> None:
        """Append one MonitorSample; the timer repaints on its own cadence."""
        value = float(sample.value)
        std = float(sample.std) if sample.std is not None else float("nan")
        sem = getattr(sample, "sem", None)
        sem = float(sem) if sem is not None else float("nan")
        self.sample_count += 1
        self._samples.append((self.sample_count, value, std, sem))
        self._dirty = True

    def clear(self) -> None:
        self._samples.clear()
        self.sample_count = 0
        self._dirty = False
        self.status_label.setText("waiting for a reading…")
        self._draw_placeholder()

    # ---- painting ----------------------------------------------------------
    def _redraw_if_dirty(self) -> None:
        if not self._dirty or self.pause_check.isChecked():
            return
        self._dirty = False
        self._draw_samples()

    def _draw_placeholder(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "waiting for a scope/DAQ reading",
                ha="center", va="center", color="#d8dee9", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        self._style_axes(ax)
        self.canvas.draw_idle()

    def _draw_samples(self) -> None:
        if not self._samples:
            self._draw_placeholder()
            return
        n = np.array([s[0] for s in self._samples], dtype=float)
        value = np.array([s[1] for s in self._samples], dtype=float)
        std = np.array([s[2] for s in self._samples], dtype=float)
        sem = np.array([s[3] for s in self._samples], dtype=float)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        band = np.where(np.isfinite(std), std, 0.0)
        ax.fill_between(n, value - band, value + band,
                        color="#88c0d0", alpha=0.25, linewidth=0)
        ax.plot(n, value, color="#88c0d0", linewidth=1.0,
                marker="o", markersize=2.5)
        ax.plot(n[-1], value[-1], "o", color="#ebcb8b", markersize=5)
        ax.set_xlabel("reading #", color="#d8dee9", fontsize=8)
        ax.set_ylabel("mean (V)", color="#d8dee9", fontsize=8)
        ax.grid(True, color="#2a3540", linewidth=0.5)
        self._style_axes(ax)
        self.canvas.draw_idle()

        last = f"last {_format_volts(value[-1])}"
        if np.isfinite(std[-1]):
            last += f" \N{PLUS-MINUS SIGN} {_format_volts(std[-1])}"
        if np.isfinite(sem[-1]):
            last += f"  ·  SEM {_format_volts(sem[-1])}"
        self.status_label.setText(f"reading {self.sample_count}  ·  {last}")

    @staticmethod
    def _style_axes(ax) -> None:
        ax.set_facecolor("#101821")
        ax.tick_params(colors="#d8dee9", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#2a3540")
        ax.figure.set_facecolor("#0b1118")


class LiveReadoutDock(QtWidgets.QDockWidget):
    """Self-contained dockable live-readout panel (bridge + strip chart).

    Owns its worker→GUI bridge and knows how to hook itself onto a monitor
    controller, so the main window only creates and docks it, then calls
    ``watch()`` / ``unwatch()`` when the scope or DAQ connects/disconnects.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Live Readout (scope/DAQ)", parent)
        self.setObjectName("LiveReadoutDock")
        self.view = LiveSampleView(self)
        self.setWidget(self.view)
        self.bridge = MonitorSampleBridge(self)
        self.bridge.sample_ready.connect(self.view.add_sample)

    def watch(self, controller) -> None:
        """Start mirroring a controller's monitor_cycle() samples."""
        controller.add_sample_listener(self.bridge.on_sample)

    def unwatch(self, controller) -> None:
        """Stop mirroring (safe to call even if watch() never ran)."""
        controller.remove_sample_listener(self.bridge.on_sample)


__all__ = ["MonitorSampleBridge", "LiveSampleView", "LiveReadoutDock"]
