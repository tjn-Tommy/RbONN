"""Live plots for the OSA encoding-shape optimisation.

:class:`LiveLossCanvas` streams :class:`slm_module.optimization.EvaluationSample`
points (attached to ``OptimizationProgress``) into two axes:

* left  -- per-evaluation loss (scatter) + running best (line), with the flat
  (all-ones) profile's loss as a horizontal reference line;
* right -- the candidate's main-anchor ``eta`` and ``c_total`` traces with the
  flat profile's values as reference lines: the live current-vs-flat view.

Repaints are timer-throttled so a fast evaluation stream cannot flood the GUI.
"""
from __future__ import annotations

import itertools
import math

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

_FG = "#d8dee9"
_GRID = "#2b3a42"
_BG = "#101820"


class LiveLossCanvas(QtWidgets.QWidget):
    """Loss / eta / crosstalk vs evaluation index, with flat reference lines."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        interval_ms: int = 300,
    ) -> None:
        super().__init__(parent)
        self._counter = itertools.count(1)
        self._index: list[int] = []
        self._loss: list[float] = []
        self._best: list[float] = []
        self._eta: list[float] = []
        self._c_total: list[float] = []
        self._flat: dict[str, float] | None = None
        self._dirty = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8, 2.6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(170)
        layout.addWidget(self.canvas)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._redraw_if_dirty)
        self._timer.start()
        self._draw()

    # ------------------------------------------------------------------ data
    def reset(self) -> None:
        self._counter = itertools.count(1)
        self._index.clear()
        self._loss.clear()
        self._best.clear()
        self._eta.clear()
        self._c_total.clear()
        self._flat = None
        self._dirty = True

    def add_sample(self, sample) -> None:
        """Append one EvaluationSample (any stage; a global index is kept)."""
        if sample is None or not math.isfinite(sample.loss):
            return
        self._index.append(next(self._counter))
        self._loss.append(float(sample.loss))
        best = self._best[-1] if self._best else math.inf
        self._best.append(min(best, float(sample.loss)))
        self._eta.append(
            float(sample.eta) if sample.eta is not None else math.nan
        )
        self._c_total.append(
            float(sample.c_total) if sample.c_total is not None else math.nan
        )
        self._dirty = True

    def set_flat_reference(self, reference: dict[str, float] | None) -> None:
        if reference:
            self._flat = dict(reference)
            self._dirty = True

    def on_progress(self, progress) -> None:
        """Convenience slot: feed an OptimizationProgress directly."""
        self.add_sample(getattr(progress, "sample", None))
        self.set_flat_reference(getattr(progress, "flat_reference", None))

    # -------------------------------------------------------------- painting
    def _redraw_if_dirty(self) -> None:
        if self._dirty:
            self._dirty = False
            self._draw()

    def _draw(self) -> None:
        self.figure.clear()
        ax1, ax2 = self.figure.subplots(1, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor(_BG)
            ax.grid(True, color=_GRID, linewidth=0.5)
            ax.tick_params(colors=_FG, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(_GRID)
        self.figure.patch.set_facecolor(_BG)

        if self._index:
            ax1.scatter(self._index, self._loss, s=10, color="#81a1c1",
                        alpha=0.7, label="loss")
            ax1.plot(self._index, self._best, color="#a3be8c", lw=1.4,
                     label="best")
        if self._flat is not None and math.isfinite(self._flat.get("loss", math.nan)):
            ax1.axhline(self._flat["loss"], color="#ebcb8b", ls="--", lw=1.0,
                        label="flat")
        ax1.set_xlabel("evaluation", color=_FG, fontsize=8)
        ax1.set_ylabel("loss", color=_FG, fontsize=8)
        if self._index or self._flat:
            ax1.legend(loc="upper right", fontsize=7, facecolor=_BG,
                       labelcolor=_FG, edgecolor=_GRID)

        if self._index:
            ax2.plot(self._index, self._eta, color="#a3be8c", lw=1.0,
                     label="eta")
            ax2.plot(self._index, self._c_total, color="#bf616a", lw=1.0,
                     label="c_total")
        if self._flat is not None:
            if math.isfinite(self._flat.get("eta", math.nan)):
                ax2.axhline(self._flat["eta"], color="#a3be8c", ls="--", lw=0.9)
            if math.isfinite(self._flat.get("c_total", math.nan)):
                ax2.axhline(self._flat["c_total"], color="#bf616a", ls="--",
                            lw=0.9)
        ax2.set_xlabel("evaluation", color=_FG, fontsize=8)
        ax2.set_ylabel("eta / crosstalk", color=_FG, fontsize=8)
        if self._index:
            ax2.legend(loc="center right", fontsize=7, facecolor=_BG,
                       labelcolor=_FG, edgecolor=_GRID)
        self.canvas.draw_idle()


class BatchResultsTable(QtWidgets.QTableWidget):
    """Per-variant summary of a batch optimisation run."""

    _HEADERS = (
        "label", "sampling", "sensitivity", "eta", "c_total",
        "accepted", "run dir",
    )

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(self._HEADERS), parent)
        self.setHorizontalHeaderLabels(list(self._HEADERS))
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(True)
        self.setMaximumHeight(140)

    def show_outcomes(self, outcomes) -> None:
        self.setRowCount(len(outcomes))
        for row, outcome in enumerate(outcomes):
            result = outcome.result
            if outcome.stopped:
                verdict, eta, c_total = "stopped", "", ""
            elif outcome.error:
                verdict, eta, c_total = f"error: {outcome.error}", "", ""
            elif result is not None:
                metric = result.final_metrics.get(0)
                eta = f"{metric.eta:.4f}" if metric is not None else ""
                c_total = f"{metric.c_total:.4g}" if metric is not None else ""
                verdict = "yes" if result.accepted else "no"
            else:
                verdict, eta, c_total = "?", "", ""
            values = (
                outcome.variant.label,
                outcome.variant.settings.sampling_points,
                outcome.variant.settings.sensitivity,
                eta,
                c_total,
                verdict,
                outcome.run_dir or "",
            )
            for col, value in enumerate(values):
                self.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))


__all__ = ["BatchResultsTable", "LiveLossCanvas"]
