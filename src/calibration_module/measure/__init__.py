"""Acquisition: drive the SLM, the DAQ and the OSA, return the rows.

Thin on purpose.  A sweep here decides what to display and when to read, and
hands back arrays -- it does not fit them; ``calibration_module.fit`` does that,
and keeping the two apart is what lets the fits be tested without hardware.

    bench     connect/read primitives shared by the step scripts
    slm_response  steps 1-3 -- the OSA/DAQ sweeps, and one ``run_step*`` per
              step that the GUI and ``steps/calib_step{1,2,3b,3c}.py`` share
    pair_v2   step 6 -- one pair's interleaved schedule, with the near-rail
              range escalation and the TIA sign convention

``center`` and ``phase`` are LEGACY sweep drivers with one GUI page each.
They predate the v2 step scripts and are scheduled to die with the GUI v2
rebuild; do not import them from new code.  ``pair`` was the third and is
gone -- GUI Step 6 runs ``pair_v2`` now, so nothing was left importing it.

A v2 driver takes the monitor as an argument rather than opening one, and
accepts ``progress_callback``/``stop_event``, so the same call serves the
offline script (a printing callback) and the GUI (a progress bar and a Stop
button).  That is the shape the remaining two should be rebuilt to.
"""
