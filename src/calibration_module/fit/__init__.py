"""The calibration physics: arrays in, fitted parameters out.

Nothing in here imports a driver, and nothing should start.  That is the point
of the subpackage: these fits are the part that fails quietly and wrongly, so
they have to stay testable against saved CSVs with the bench powered down.
Anything that must ask an instrument for a number belongs in
:mod:`calibration_module.measure`; call in here afterwards with the rows.

    slm_response  steps 1-3 -- CalibrationResult and its JSON/CSV, the step-3
              channel grid, peak centroids, the usable-band clip
    transfer  the per-channel sin^2 transfer curve the encoder inverts
    outliers  post-sweep outlier flags, and the step 6-8 burst guards
    pair_v2   step 6 -- per-pair eta from the difference estimator
    phase     step 7 -- comb phase dPhi_comb, and its covariance
    cascade   step 7 -- reaching every pair from one root when a neighbour
              cannot be measured directly
    verify    step 8 -- the n-pair forward model, each block's full scale, the
              Y_sat fit and the verify CSV
    levels    drives that land on a chosen fraction of full scale
    sigma     the shared systematic std floor, imported by all three v2 steps
    center    centre-wavelength scan -- weighted quadratic vertex fit
    report / levels_report / step8_report   plots over the fits above

    pair      step 6 v1, the joint 6-parameter grid fit pair_v2 replaced, and
              legacy_intensity the old sin^2 fit.  Kept so historical CSVs can
              be re-fit both ways; reached only by the v1 scripts.

The JSON and CSV readers live beside the fit that defines each format, a file
layout being part of that fit's contract with the next step.

Plot helpers take a Matplotlib ``Figure`` the caller owns rather than making
one, so a GUI canvas and a headless PNG render from one implementation;
``save_plot`` is the PNG wrapper.
"""
