"""Acquisition: drive the SLM, the DAQ and the OSA, return the rows.

Thin on purpose.  A sweep here decides what to display and when to read, and
hands back arrays; it does not fit them -- :mod:`calibration_module.fit` does,
and keeping the two apart is what lets the fits be tested without hardware.

    bench     connect/read primitives shared by the step scripts
    slm_response  steps 1-3 -- the OSA/DAQ sweeps, and one ``run_step*`` per
              step that the GUI and ``steps/calib_step{1,2,3b,3c}.py`` share
    pair_v2   step 6 -- one pair's interleaved schedule, and the ONE bench
              acquisition (window, range, settle, TIA sign) every step imports
    phase_v2  step 7 -- one target pair swept against the reference
    patterns  step 8 and the ONN loop -- drive a list of multi-pair patterns
    pmt_temp  the PMT thermistor, as a per-read row hook

``center`` is a LEGACY sweep driver with one GUI page; it predates the v2 step
scripts and is scheduled to die with the GUI v2 rebuild, so do not import it
from new code.

A v2 driver takes the monitor as an argument rather than opening one, and
accepts ``progress_callback``/``stop_event``, so one call serves the offline
script (a printing callback) and the GUI (a progress bar and a Stop button).
That is the shape ``center`` should be rebuilt to.
"""
