"""Calibration: drive the bench, collect the rows, fit the physics.

The line that matters here is fitting vs. acquisition, and it is drawn by test
cost rather than by subject matter.  Acquisition is thin and boring -- seven
SLM calls and one DAQ read across the whole repo -- and it fails loudly, when a
cable is out.  The fits are thousands of lines of physics and they fail quietly
and wrongly.  So the fits are kept importable with no driver behind them, which
is what lets them be exercised against saved CSVs with the laser off:
``tests/test_slm_transfer_fit.py`` gets 54 tests out of that with no fake
instrument at all, where testing a sweep first needs a FakeSLM that returns
plausible traces -- and a fake returning *plausible* data is where a test
quietly stops testing the physics.

    fit/      arrays of commanded levels and measured volts in, fitted
              parameters out.  Imports no driver, and should not start.

        slm_response  steps 1-3 -- CalibrationResult and its JSON/CSV, the
                  step-3 channel grid, peak centroids, the usable-band clip
        transfer  the per-channel sin^2 transfer curve the encoder inverts
        outliers  post-sweep outlier flags for the step 2/3 re-measure loops
        legacy_intensity  the old sin^2 fit; only outliers and one GUI CSV
                  loader still reach it
        pair_v2   step 6 -- per-pair TPA efficiency eta from the difference
                  estimator; carries the PairV2Config it was fitted under
        pair      step 6 v1 -- the joint grid fit pair_v2 replaced; only the
                  v1 scripts still reach it
        phase     steps 7 + 8 -- comb phase dPhi_comb, and the forward model
                  built from the step-6 pair fits
        center    centre-wavelength scan -- weighted quadratic vertex fit
        sigma     the shared systematic std floor
        report    fringe / residual plots over the fits above

    measure/  drives the SLM, the DAQ and the OSA and returns rows; does no
              fitting.  ``bench`` holds the connect/read primitives;
              ``slm_response`` the step 1-3 sweeps and their ``run_step*``
              runners.  Its ``center``, ``pair`` and ``phase`` are legacy GUI
              sweep drivers, scheduled to die with the GUI v2 rebuild.

    steps/    the runnable scripts (1, 2, 3b, 3c, 6, 7, 8 and the v1
              predecessors): ask ``measure`` for rows, hand them to ``fit``,
              write the JSON/CSV, render the report.  The GUI calls ``measure``
              and ``fit`` directly and never these scripts.

The JSON and CSV readers live beside the fit that defines each format
(``fit.phase.save_comb_phase_json``, ``fit.center.save_tpa_center_json``,
``fit.pair.load_tpa_pair_csv``): a file layout is part of that fit's contract
with the next step, not a separate concern.
"""
