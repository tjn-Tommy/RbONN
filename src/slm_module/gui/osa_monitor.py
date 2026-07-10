"""Live OSA spectrum monitor: trace-listener bridge + throttled plot widget.

Every OSA sweep in the codebase funnels through ``OSAController.measure()``,
which notifies its registered trace listeners on the *calling worker thread*.
:class:`OSATraceBridge` is such a listener: it only emits a queued Qt signal,
so the trace crosses safely onto the GUI thread where a
:class:`LiveSpectrumView` (or the OSA Viewer page) repaints it.

Wiring (done by the main window)::

    bridge = OSATraceBridge(parent)
    osa.add_trace_listener(bridge.on_trace)      # on connect
    bridge.trace_ready.connect(view.set_trace)   # GUI-thread slot
    osa.remove_trace_listener(bridge.on_trace)   # on disconnect
"""
from __future__ import annotations

import time

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets


class OSATraceBridge(QtCore.QObject):
    """Thread-safe hand-off of TraceData from worker threads to the GUI.

    ``on_trace`` is registered as an ``OSAController`` trace listener; the
    ``emit`` is queued by Qt, so slots always run on the GUI thread.
    """

    trace_ready = QtCore.pyqtSignal(object)      # TraceData

    def on_trace(self, trace) -> None:
        self.trace_ready.emit(trace)


class LiveSpectrumView(QtWidgets.QWidget):
    """Throttled live spectrum plot: repaints at most every ``interval_ms``.

    Traces may arrive faster than the canvas can draw (short sweeps in a tight
    loop); only the latest trace is kept and stale ones are dropped. A Pause
    box freezes the display without unhooking the listener.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        interval_ms: int = 200,
    ) -> None:
        super().__init__(parent)
        self._trace = None
        self._dirty = False
        self._sweep_count = 0
        self._last_stamp = 0.0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        bar = QtWidgets.QHBoxLayout()
        self.pause_check = QtWidgets.QCheckBox("Pause")
        self.pause_check.setToolTip(
            "Freeze the display; traces keep streaming and resume on uncheck"
        )
        self.status_label = QtWidgets.QLabel("waiting for a sweep…")
        bar.addWidget(self.pause_check)
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

    # ---- GUI-thread slot -------------------------------------------------
    def set_trace(self, trace) -> None:
        """Store the newest trace; the timer repaints it (stale ones drop)."""
        self._trace = trace
        self._sweep_count += 1
        self._last_stamp = time.monotonic()
        self._dirty = True

    # ---- painting ----------------------------------------------------------
    def _redraw_if_dirty(self) -> None:
        if not self._dirty or self.pause_check.isChecked():
            return
        self._dirty = False
        self._draw_trace(self._trace)

    def _draw_placeholder(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "waiting for an OSA sweep",
                ha="center", va="center", color="#d8dee9", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        self._style_axes(ax)
        self.canvas.draw_idle()

    def _draw_trace(self, trace) -> None:
        if trace is None:
            self._draw_placeholder()
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        wl = np.asarray(trace.wavelengths_nm, dtype=float)
        is_log = trace.power_label == "power_dBm"
        if is_log:
            y = np.asarray(trace.powers, dtype=float)
            ylabel = "power (dBm)"
            unit = "dBm"
        else:
            y = np.asarray(trace.powers, dtype=float) * 1e6
            ylabel = "power (µW)"
            unit = "µW"
        ax.plot(wl, y, color="#88c0d0", linewidth=1.0)
        if wl.size and np.any(np.isfinite(y)):
            peak = int(np.nanargmax(y))
            ax.plot(wl[peak], y[peak], "o", color="#ebcb8b", markersize=4)
            ax.annotate(
                f"{wl[peak]:.4f} nm\n{y[peak]:.3g} {unit}",
                (wl[peak], y[peak]), color="#ebcb8b", fontsize=7,
                xytext=(6, -2), textcoords="offset points",
            )
            avg = f" · avg {trace.averages}" if trace.averages > 1 else ""
            self.status_label.setText(
                f"sweep {self._sweep_count}  ·  peak {y[peak]:.3g} {unit} @ "
                f"{wl[peak]:.4f} nm  ·  {wl.size} pts{avg}"
            )
        ax.set_xlabel("wavelength (nm)", color="#d8dee9", fontsize=8)
        ax.set_ylabel(ylabel, color="#d8dee9", fontsize=8)
        ax.grid(True, color="#2a3540", linewidth=0.5)
        self._style_axes(ax)
        self.canvas.draw_idle()

    @staticmethod
    def _style_axes(ax) -> None:
        ax.set_facecolor("#101821")
        ax.tick_params(colors="#d8dee9", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#2a3540")
        ax.figure.set_facecolor("#0b1118")


__all__ = ["OSATraceBridge", "LiveSpectrumView"]
