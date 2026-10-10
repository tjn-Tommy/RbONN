"""The PyQt5 control suite: one window over the SLM and every instrument.

    app      MainWindow -- the nav, every page, and the worker-thread wrapper
             each instrument call goes through.  Its docstring maps the pages.
    common   the widgets pages share: labelled fields, spin rows, status pills.
    live_plots    BatchResultsTable and LiveLossCanvas, for sweeps in progress.
    osa_monitor   the dockable live spectrum view and its OSA trace bridge.
    style    DARK_STYLESHEET, the one Qt stylesheet.

The physics lives in ``calibration_module``; this package only drives it.
"""
from .app import MainWindow, main

__all__ = ["MainWindow", "main"]
