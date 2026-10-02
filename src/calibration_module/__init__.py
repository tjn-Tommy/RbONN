"""Calibration: drive the bench, collect the rows, fit the physics.

The one line that matters is fitting vs acquisition, and it is drawn by TEST
COST, not subject matter.  Acquisition is thin and boring -- a handful of SLM
calls and one DAQ read across the repo -- and it fails loudly, when a cable is
out.  The fits are thousands of lines of physics and they fail quietly and
wrongly.  So the fits stay importable with no driver behind them, which is what
lets them be exercised against saved CSVs with the laser off.  Testing a sweep
instead needs a FakeSLM returning plausible traces, and a fake returning
PLAUSIBLE data is where a test quietly stops testing the physics.

    fit/      arrays of commanded levels and measured volts in, fitted
              parameters out.  Imports no driver, and should not start.
    measure/  drives the SLM, the DAQ and the OSA and returns rows; no fitting.
    steps/    the runnable scripts.  They ask ``measure`` for rows, hand them to
              ``fit``, write the JSON/CSV and render the report.  The GUI calls
              ``measure`` and ``fit`` directly and never these scripts.

Each subpackage's own docstring maps its modules.  The JSON and CSV readers live
beside the fit that DEFINES each format, because a file layout is part of that
fit's contract with the next step, not a separate concern.
"""
