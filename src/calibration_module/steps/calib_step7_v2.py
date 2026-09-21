"""Step 7 v2 -- comb phase dPhi_comb per pair, ONE free parameter.

Needs real hardware for the sweep; the refit path is offline.  Either way
``--step6``, ``--targets`` and ``--ref`` are required and have no default::

    S="--step6 src/calib_data/archive/2026-09/run_0907_1724/calib_step6v2_result_0907_1757.json --targets 2,3,4,5,6 --ref 2"
    python src/calibration_module/steps/calib_step7_v2.py $S
    python src/calibration_module/steps/calib_step7_v2.py $S --out DIR   # CSV, JSON, PNGs into DIR
    python src/calibration_module/steps/calib_step7_v2.py $S --meas
    python src/calibration_module/steps/calib_step7_v2.py $S some.csv

A sweep fits the CSV it just wrote (``--meas`` stops after the CSV).  Either way
the fit writes a combined ``calib_step7_result_*.json`` -- the step-3 + step-6
payloads carried over from ``--step6`` plus the fitted ``{Phi_k}`` spectrum --
the single input downstream consumers read.

There is deliberately no default step-6 path, for the same reason step 6 has no
default ``--step3``: it used to be a constant edited by hand, so the file it
named drifted out from under the script every time a run was filed away.  That
one file decides both what is driven (its embedded step-3 calibration IS the
channel layout) and what the fringe amplitudes are pinned to, so running against
the wrong one does not fail -- it reports a phase measured under a different
aperture.  The targets and the reference are required for the same reason: as
constants they silently measured whichever pairs the last edit left in them.

What changed vs :mod:`calib_step7_v1`
-------------------------------------
1. **Sign convention.**  The fringe is ``cos(dPhi_comb - dPhi_SLM)``: the panel
   phase SUBTRACTS from the comb phase (it compensates the comb's dispersion,
   so the fringe peaks where the two match), where v1 had
   ``cos(dPhi_SLM + dPhi_comb)``.  Since ``cos`` is even the two differ only in
   the SIGN of the recovered ``dPhi_comb`` -- a v2 ``Phi_k`` is the negative of
   a v1 one, so do NOT feed a v2 JSON to a consumer written against v1's
   forward model without flipping the sign.
2. **Nothing floats but the phase.**  ``a = eta_ref*g_ref``, ``b = eta_tgt``,
   the single-beam response and the dark are ALL taken from step 6 / the
   measured all-off read, so ``dPhi_comb`` is the ONLY free parameter.
3. **Both arms stop below 1.0** (0904 on).  See :class:`PhaseV2Config`.
4. **The reads are inverted at the source.**  The transimpedance amplifier
   outputs a NEGATIVE voltage for positive light, so ``ACQ.invert`` records a
   positive light signal and the CSV on disk already holds one.  v1's ``--flip``
   post-processing is gone with it.

What it measures.  Each target pair carries a fixed comb-phase offset
``dPhi_comb`` relative to a common reference pair (the reference defines
``Phi = 0``).  Driving the two pairs at once makes them interfere; the fringe
encodes ``dPhi_comb``.  Running every target builds the phase spectrum
``{Phi_k}``.

The drive.  The reference pair is held at ``x_r = w_r = CONFIG.ref_level``; the
target's TWO channels are swept TOGETHER (``x_t = w_t = v``) over the ramp.  A
channel at intensity ``v`` sits at panel phase ``theta = 2*asin(sqrt(v))`` with
field ``sqrt(v)*exp(i theta/2)``, so the target field amplitude is
``g = sqrt(x_t w_t) = sin^2(theta/2)``, the reference's is ``g_ref``, and
``dPhi_SLM = theta - theta_ref``::

    Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_comb - dPhi_SLM)
        + step-6 single-beam background

Sweeping ``v`` 0.1 -> 0.9 sweeps ``theta`` over ~37..143 deg, tracing most of
the half fringe.

The fit lives in :mod:`calibration_module.fit.phase`
(:func:`~calibration_module.fit.phase.fit_phase_fixed`): weighted nonlinear
least squares in the single parameter ``dPhi_comb``, errors from the weighted
Jacobian as-is (no chi2/dof, no Birge rescaling).  The drive and the reads live
in :mod:`calibration_module.measure.phase_v2`.  This file is the runner:
it loads the step-6 JSON, drives the sweep, then fits, reports, plots and saves.
Everything tunable is the block of constants below -- the drive's half in
``CONFIG``, the acquisition's in ``ACQ``.  Both are plain values the GUI builds
too, so the two front ends run the same fit over the same sweep.

Each point is one fixed-duration ``daq_module`` acquisition like step 6:
``t_single_s`` for the all-off dark (near-zero signal needs the averaging) and
``t_both_s`` for the sweep points (the reference is on, so they are bright),
low-passed at the ``DAQMonitorSettings`` bandwidth.  Every CSV row records the
mean, its trace std and the std ratio (std/|mean|).

PMT temperature
---------------
Every row also carries the PMT thermistor (Keithley 2400, ohms and TH10K
degrees) in ``pmt_ohm_Ohm`` and ``pmt_c``, and the CSV brackets itself with a
start and end reading in its ``# key,value`` meta.  A step-7 run is ~20 minutes
of bench time in which the reference amplitude is assumed constant, so what the
detector's temperature did while it ran is worth having on the same rows.  ON by
default (``--no-pmt`` turns it off), re-read at most every ``--pmt-period``
seconds (default 60) with the rows in between repeating the last reading.  A
missing or failing 2400 blanks the two columns rather than ending the run.  One
GPIB, one 2400: do NOT run ``drafts/daq_pmt_monitor.py`` alongside a sweep.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.phase import (  # noqa: E402
    PhaseFit,
    PhaseResult,
    load_pair_models,
    load_phase_csv,
    references_in_csv,
    save_comb_phase_json,
    targets_in_csv,
    write_meas_csv,
)
from calibration_module.fit.cascade import (  # noqa: E402
    MIN_PAIR_SEP,
    Stage,
    chain_fits,
    plan_cascade,
    plan_lines,
)
from calibration_module.fit.report import plot_fringe  # noqa: E402
from calibration_module.fit.sigma import STD_FLOOR_V  # noqa: E402
from calibration_module.measure.bench import connect_daq, connect_slm  # noqa: E402
from calibration_module.measure.phase_v2 import (  # noqa: E402
    PhaseV2Acq,
    PhaseV2Config,
    build_xw_sweep,
    measure_target,
    run_seconds,
)
from calibration_module.measure.pmt_temp import (  # noqa: E402
    PMT_PERIOD_S,
    PmtTemperature,
    open_pmt,
)
from calibration_module.fit.slm_response import (  # noqa: E402
    calibration_result_from_dict,
)
from slm_module.encoding import channel_layout_from_calibration  # noqa: E402

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"   # output directory; --out overrides it

# Neither the step-6 result nor the pairs are here -- they are the required
# --step6, --targets and --ref arguments.  See the module docstring for why they
# have no default.  Left empty, so code that imports this module and sweeps
# without setting the list fails before touching the hardware.
TGT_INDICES: list[int] = []

# The drive.  Every field has the validated default (see PhaseV2Config); name
# one here only to change it.  Keep pair_index_base in step with
# calib_step6_v2.CONFIG -- the pair labels come out of that script's JSON.
CONFIG = PhaseV2Config(
    pair_index_base=1,                      # pairs are numbered 1..N
    ref_index=1,                            # the common reference (Phi = 0); --ref sets it
    ref_level=0.9,
    sweep_min=0.1,
    sweep_max=0.9,
    n_points=10,
)

# Acquisition timing and input range.  See PhaseV2Acq for what each one costs.
ACQ = PhaseV2Acq(
    t_single_s=10.0,
    t_both_s=10.0,
    settle_s=0.25,
    range_v=0.2,
    range_wide_v=0.5,
    autorange=True,
    invert=True,
)

# Fold in the step-6 single-beam response as a FIXED background.  The reference
# is held at a constant level -> its single-beam is a constant; only the swept
# target ramps.  Keeps the fringe from having to absorb the single-beam ramp.
# Together with the pinned etas and the measured dark this leaves dPhi_comb as
# the sole free parameter (v2's whole point) -- turn it off only to diagnose
# step 6.
SINGLE_BEAM_BG = True

# Method label stored in the output JSON (v2 has exactly one fit method).
METHOD = "fixed_comb_only"

SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"


# ======================================================================
# inputs  (layout + step-6 models, both out of the one combined JSON)
# ======================================================================

def _load_layout(step6: Path):
    """Channel layout from the Step-3 calibration EMBEDDED in the step-6 JSON.

    ``save_combined_json`` stores the raw step-3 payload under ``"step3"``, so
    the layout this run drives is guaranteed to be the one the step-6 etas were
    calibrated under.  The encoding convention is read back the same way: step 6
    recorded which mapping it drove with, and reading it back is what keeps a
    chain internally consistent.  A file written before the fitted encoding
    existed carries no marker and was measured under ``"interp"``.
    """
    step6 = Path(step6)
    if not step6.is_file():
        raise FileNotFoundError(f"Step-6 result not found: {step6}")
    import json

    payload = json.loads(step6.read_text(encoding="utf-8"))
    step3 = payload.get("step3")
    if step3 is None:
        raise ValueError(
            f"{step6} has no embedded 'step3' calibration; point --step6 at a "
            f"combined step-6 result (calib_step6_v2 save_combined_json)"
        )
    enc_method = (payload.get("encoding") or {}).get("method", "interp")
    layout = channel_layout_from_calibration(
        calibration_result_from_dict(step3), method=enc_method
    )
    for name, idx in ([("ref_index", CONFIG.ref_index)]
                      + [("TGT_INDICES", k) for k in TGT_INDICES]):
        if not (0 <= CONFIG.slot(idx) < layout.n_channels):
            raise ValueError(
                f"{name} entry {idx} out of range (layout has "
                f"{layout.n_channels} pairs, numbered from "
                f"{CONFIG.pair_index_base})"
            )
    return layout


def _load_models(step6: Path, targets, *, ref: int | None = None):
    """Step-6 pair models; require the reference and every requested target.

    ``ref`` defaults to ``CONFIG.ref_index`` -- a sweep is about to drive it --
    but a re-fit passes the reference the CSV itself records instead.
    """
    ref = CONFIG.ref_index if ref is None else ref
    models = load_pair_models([Path(step6)])
    needed = [("reference", ref)] + [("target", k) for k in targets]
    for role, idx in needed:
        if idx not in models:
            raise ValueError(
                f"no step-6 model for {role} pair index {idx}; found "
                f"{sorted(models)} in {step6}"
            )
    print(f"Step 6: eta[ref {ref}] = {models[ref].eta:.4g} ; "
          + " ".join(f"eta[{k}]={models[k].eta:.4g}" for k in targets))
    return models


# ======================================================================
# report + plot
# ======================================================================

def report(fit: PhaseFit, tgt: int, ref: int) -> None:
    """Print dPhi_comb (rad + deg) and the fit quality of the one-parameter fit."""
    print("Model:  Y = a^2 + b^2 sin^4(theta/2) "
          "+ 2ab sin^2(theta/2) cos(dPhi_comb + pi - theta) + step6 single-beam")
    print("            [both target channels swept together; theta the shared panel phase]")
    print("            a = eta_ref*sqrt(x_r w_r), b = eta_tgt, background and dark ALL fixed from step 6")
    print("            -> dPhi_comb is the ONLY free parameter")
    print(f"Pair {tgt} vs reference {ref}  (value +/- error):")
    # The quoted error is the TOTAL: fringe noise and the pinned step-6 eta in
    # quadrature.  a and b do not float, so the fitter's own dphi_comb_err
    # cannot see the eta error -- it is invisible there, not absent.
    err = fit.dphi_comb_err_total
    print(f"  dPhi_comb = {fit.dphi_comb:+.4f} +/- {err:.4f} rad"
          f"   ( {fit.dphi_comb_deg:+.2f} +/- {np.degrees(err):.2f} deg )")
    print(f"     of which fringe noise {np.degrees(fit.dphi_comb_err):.3f} deg, "
          f"step-6 eta {np.degrees(fit.dphi_comb_err_eta):.3f} deg"
          f"   [d(dPhi)/d(eta): ref {fit.dphi_deta_ref:+.2f}, "
          f"tgt {fit.dphi_deta_tgt:+.2f} rad per unit eta]")
    g_ref = float(np.median(np.sqrt(fit.x_r * fit.w_r))) if fit.x_r is not None else 1.0
    print(f"  a (ref arm)      = {fit.a*1e3:.4f} mV^0.5   "
          f"(pinned: eta_ref {fit.eta_ref*1e3:.4f} x g_ref {g_ref:.4g})")
    print(f"  b (tgt eta CxCw) = {fit.b*1e3:.4f} mV^0.5   (pinned to step-6 eta)")
    print(f"  fringe amp 2ab   = {fit.amp*1e3:.4f} mV   (pinned)")
    # Not fitted here (v1 floated it as `d`), so it is a pure check on step 6:
    # a big mean residual means the fixed amplitudes/background are off.
    print(f"  mean residual    = {float(np.mean(fit.residuals))*1e3:+.4f} mV   "
          f"(NOT fitted -- should be ~0 if step 6 is right)")
    # Pull denominator is std_total, not the DAQ trace std: with a and b pinned
    # the CURVE carries step 6's eta error, and the fit had no freedom to absorb
    # it, so charging the residual against the point spread alone reads high.
    eta_v = fit.std_model_eta
    print(f"  max |pull|       = {float(np.max(np.abs(fit.pulls))):.2f}   "
          f"[pull = residual / std_total]")
    print(f"     std_total     = {float(np.median(fit.std))*1e3:.4f} (measurement) "
          f"(+) {float(np.median(eta_v))*1e3:.4f} (step-6 eta) mV, median over points"
          f"   -> {float(np.median(fit.std_total))*1e3:.4f} mV")
    print(f"  max |pull| on std alone = "
          f"{float(np.max(np.abs(fit.residuals / fit.std))):.2f}   "
          f"(measurement only -- overstates, the eta error is not in it; "
          f"std includes the {STD_FLOOR_V*1e3:.3f} mV systematic floor)")
    print(f"  R^2 = {fit.r2:.4f}")


def save_plot(fit: PhaseFit, tgt: int, path) -> None:
    """Measured Y(dPhi_SLM) with the fitted model curve + pulls, as a PNG.

    The figure itself is :func:`calibration_module.fit.report.plot_fringe`, the
    same renderer the GUI draws into, so a fringe reviewed on screen and one
    archived beside the JSON are the same picture.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))
    plot_fringe(fig, fit, tgt)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ======================================================================
# the run
# ======================================================================

def _run_sweep(step6: Path, fit_after: bool, *,
               pmt: "PmtTemperature | None" = None) -> None:
    """Sweep every target pair back to the reference; write one raw CSV.

    Not necessarily against it.  A pair adjacent to the reference cannot be
    measured against it at all, so :func:`~calibration_module.fit.cascade.plan_cascade`
    splits the run into stages, each with its own reference, and the fit chains
    the far ones back.  One reference and no neighbours is the one-stage case,
    which is what this always did.
    """
    if not TGT_INDICES:
        raise ValueError("no target pairs -- pass --targets, e.g. --targets 2,3,4,5,6")
    layout = _load_layout(step6)
    print(f"Step 6 in: {step6}")
    drive = build_xw_sweep(CONFIG)
    secs = run_seconds(drive, ACQ)
    values = [x for x, _, _, _ in drive]
    stages = plan_cascade(TGT_INDICES, CONFIG.ref_index,
                          n_pairs=layout.n_channels)
    n_targets = sum(len(s) for s in stages)
    print(f"Step 7 v2: {len(drive)} points + 1 dark per target "
          f"(~{secs/60:.1f} min/target)")
    for line in plan_lines(stages, CONFIG.ref_index, seconds_per_target=secs):
        print(f"  {line}")
    print(f"Reference held at {CONFIG.ref_level:g}")
    print(f"Ramp: x_t = w_t over {values}")
    print(f"Layout: {layout.channel_width_px} px window + "
          f"{layout.pitch_px - layout.channel_width_px} px pad "
          f"(pitch {layout.pitch_px} px)")

    meta: dict[str, object] = {}
    if pmt is not None:
        first = pmt.read()          # seeds the cache, so row 1 is not blank
        meta["pmt_resource"] = pmt.resource
        meta["pmt_period_s"] = f"{pmt.period_s:g}"
        meta.update({f"{k}_start": v for k, v in first.items()})
        print(f"PMT       : {pmt.resource}  "
              + (f"{first['pmt_c']} C ({first['pmt_ohm_Ohm']} Ohm)"
                 if first["pmt_c"] else "unavailable")
              + f", resampled every {pmt.period_s:g} s")

    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=ACQ.t_both_s, t_single=ACQ.t_single_s,
                      min_val=-ACQ.range_v, max_val=ACQ.range_v)
    results: list[PhaseResult] = []
    per_target = len(drive) + 1
    total = per_target * n_targets
    done = 0
    try:
        for stage in stages:
            stage_cfg = replace(CONFIG, ref_index=stage.ref)
            for k in stage.targets:
                print(f"\n=== Sweep: pair {k} vs reference {stage.ref} ===")
                results.append(measure_target(
                    daq, slm, layout, k, drive,
                    cfg=stage_cfg, acq=ACQ,
                    step0=done * per_target, total=total,
                    progress_callback=lambda p: print(p.line()),
                    log=print,
                    row_hook=None if pmt is None else pmt.sample,
                ))
                done += 1
    finally:
        slm.close_slm()
        daq.disconnect()
        if pmt is not None:
            meta.update({f"{k}_end": v for k, v in pmt.read().items()})

    stamp = time.strftime("%m%d_%H%M")
    csv_path = write_meas_csv(results, CALIB_PATH / f"calib_step7_meas_{stamp}.csv",
                              meta=meta)
    n_rows = sum(r.trial.size for r in results)
    print(f"\nSaved {n_rows} rows to {csv_path}")  # raw rows on disk BEFORE fitting
    if not fit_after:
        print(f"Fit with:  python {Path(__file__).name} --step6 {step6} {csv_path}")
        return
    fit_csv(csv_path, step6)


def fit_csv(path: str | Path, step6: Path) -> None:
    """Re-fit an already-recorded CSV offline (no hardware).

    Needs two inputs: this CSV plus the combined step-6 JSON for each pair's
    eta + single-beam background.  The CSV may carry several target pairs -- a
    collected file records every target vs the shared reference -- so every
    target present that has a step-6 model is fit separately against the
    reference and gets its own PNG.

    Everything is persisted into ONE combined ``calib_step7_result_*.json``
    (:func:`~calibration_module.fit.phase.save_comb_phase_json`), under a fresh
    timestamp so a refit never clobbers an earlier result.
    """
    targets = targets_in_csv(path, TGT_INDICES)
    # The CSV knows its own reference(s) -- every row records one -- so it wins
    # over CONFIG, which describes the NEXT sweep and has no business
    # deciding how an existing one is read.  Without this, re-fitting a file
    # recorded against a different reference asks step 6 for a pair that was
    # never the reference, and every target fails for a reason that looks
    # like arithmetic.  A cascaded run records TWO, one per stage; the root is
    # the one nothing else measured (self-rows do not count as measuring).
    recorded = references_in_csv(path)
    measured = {k for k, r in recorded.items() if k != r}
    roots = sorted(set(recorded.values()) - measured)
    if len(roots) > 1:
        raise ValueError(
            f"{Path(path).name} has no single root: references {roots} are all "
            f"unmeasured in it, which means two runs were concatenated"
        )
    ref = CONFIG.ref_index if not roots else roots[0]
    if roots and roots[0] != CONFIG.ref_index:
        print(f"Reference: {Path(path).name} was swept against pair "
              f"{roots[0]}; using that, not CONFIG's {CONFIG.ref_index}")
    ref_of = {k: recorded.get(k, ref) for k in targets}
    models = _load_models(
        step6, sorted({k for k in targets if k != ref}
                      | {r for k, r in ref_of.items() if k != ref}),
        ref=ref)
    fittable = [k for k in targets
                if k in models and k != ref and ref_of[k] in models]
    if not fittable:
        raise ValueError(
            f"no fittable target in {path}: found targets {targets}, but have "
            f"step-6 models only for {sorted(models)} "
            f"(reference is pair {ref})"
        )
    stages = _stages_from_refs(ref_of, fittable, ref)
    print(f"Fitting pair(s) {fittable} vs reference {ref} from "
          f"{path} (one free parameter: dPhi_comb)")
    if any(s.depth for s in stages):
        for line in plan_lines(stages, ref):
            print(f"  as recorded: {line}")

    stamp = time.strftime("%m%d_%H%M")
    fits: dict[int, PhaseFit] = {}
    for k in fittable:
        print(f"\n=== pair {k} vs reference {ref_of[k]} ===")
        result = load_phase_csv(path, models[k], models[ref_of[k]],
                                comb_only=True, single_beam_bg=SINGLE_BEAM_BG,
                                only_tgt=k)
        dts = result.per_trial_darks()
        drift = f" +/- {dts.std(ddof=1)*1e3:.4f} drift" if dts.size > 1 else ""
        print(f"Loaded {result.trial.size} rows, "
              f"dark = {result.dark*1e3:.4f}{drift} mV")
        report(result.fit, result.tgt_index, result.ref_index)
        plot_path = CALIB_PATH / f"calib_step7v2_pair{k}_{stamp}.png"
        save_plot(result.fit, result.tgt_index, plot_path)
        print(f"Plot saved to {plot_path}")
        fits[k] = result.fit

    # Refer every fit back to the ROOT reference.  A one-stage run is the
    # identity here; a cascaded one sums each pair's hops and propagates the
    # error with the hub's eta counted ONCE, coherently across the two hops it
    # appears in (see cascade.ChainedPhase).
    chains = chain_fits(fits, stages, ref)
    chained = {k: c for k, c in chains.items() if c.chained}
    if chained:
        print(f"\nChained to reference {ref}:")
        for k, c in sorted(chained.items()):
            print(f"  pair {k} via {c.via}: "
                  f"{np.degrees(fits[k].dphi_comb):+.2f} deg (measured) "
                  f"{np.degrees(c.dphi_raw - fits[k].dphi_comb):+.2f} deg (hub) "
                  f"-> {c.dphi_deg:+.2f} +/- {np.degrees(c.err_total):.2f} deg")

    # Persist the fitted spectrum {Phi_k} as ONE combined JSON (step3 + step6
    # carried over verbatim from --step6) -- the single input for downstream
    # consumers.  NOTE the "comb-slm" convention recorded per fit: these phases
    # are the NEGATIVE of v1's (see the module docstring).
    out_json = CALIB_PATH / f"calib_step7_result_{stamp}.json"
    save_comb_phase_json({(k, METHOD): f for k, f in fits.items()},
                         step6, out_json, ref_index=ref,
                         csv_path=str(Path(path).resolve()),
                         single_beam_bg=SINGLE_BEAM_BG,
                         chains=chains,
                         extra={"cascade": {
                             "min_sep": MIN_PAIR_SEP,
                             "stages": [{"ref": s.ref, "targets": list(s.targets),
                                         "depth": s.depth} for s in stages],
                         }} if chained else None)
    print(f"\nCombined step-7 result (step3 + step6 + step7) saved to {out_json}")


def _stages_from_refs(ref_of: dict[int, int], fittable, root: int) -> list[Stage]:
    """Rebuild the stage list from what the CSV recorded, root-first.

    A re-fit does not re-plan: the file says which reference each target was
    swept against, and that is the truth about it however the run was planned.
    Ordering by depth is what lets :func:`chain_fits` resolve a hub before
    whatever hangs off it.
    """
    by_ref: dict[int, list[int]] = {}
    for k in sorted(fittable):
        by_ref.setdefault(int(ref_of[k]), []).append(int(k))
    depth = {int(root): 0}
    stages: list[Stage] = []
    pending = dict(by_ref)
    while pending:
        ready = [r for r in sorted(pending) if r in depth] or sorted(pending)
        for r in ready:
            targets = pending.pop(r)
            stages.append(Stage(ref=r, targets=tuple(targets),
                                depth=depth.get(r, 0)))
            for k in targets:
                depth.setdefault(k, depth.get(r, 0) + 1)
    return stages


def _index_list(text: str) -> list[int]:
    """``"2,3,4"`` -> ``[2, 3, 4]``: the argparse type of a pair list."""
    try:
        out = [int(tok) for tok in text.replace(" ", "").split(",") if tok]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated pair labels, got {text!r}") from None
    if not out:
        raise argparse.ArgumentTypeError("empty pair list")
    return out


def main(argv: list[str] | None = None) -> int:
    global TGT_INDICES, CONFIG, CALIB_PATH
    parser = argparse.ArgumentParser(
        prog="calib_step7_v2.py",
        description="Step 7 v2 -- comb phase dPhi_comb per pair (one free parameter).",
        epilog="--step6, --targets and --ref have no default.  --step6 decides both what is "
               "driven (its embedded step-3 calibration IS the layout) and what "
               "the fringe amplitudes are pinned to, so running against the "
               "wrong one reports a phase measured under a different aperture.",
    )
    parser.add_argument(
        "--step6", required=True, type=Path, metavar="JSON",
        help="combined step-6 result JSON: layout + pinned etas (required)",
    )
    parser.add_argument(
        "--targets", required=True, type=_index_list, metavar="2,3,4",
        help="target pair labels measured against the reference, comma-separated (required)",
    )
    parser.add_argument(
        "--ref", required=True, type=int, metavar="K",
        help="reference pair label, Phi = 0 (required)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, metavar="DIR",
        help=f"directory for the CSV, JSON and PNGs, created if missing "
             f"(default {CALIB_PATH})",
    )
    parser.add_argument(
        "csv", nargs="?", type=Path,
        help="re-fit this recorded measurement CSV offline instead of sweeping",
    )
    parser.add_argument(
        "--meas", "-m", action="store_true",
        help="sweep and write the raw CSV only; do not fit",
    )
    parser.add_argument(
        "--no-pmt", dest="pmt", action="store_false",
        help="do not log the PMT thermistor (on by default; a missing 2400 is "
             "not an error either way, it just blanks the two columns)",
    )
    parser.add_argument(
        "--pmt-resource", default=None, metavar="VISA",
        help="VISA address of the 2400 (default: daq_pmt_monitor.K2400_RESOURCE)",
    )
    parser.add_argument(
        "--pmt-period", type=float, default=PMT_PERIOD_S, metavar="S",
        help=f"seconds between 2400 readings; rows in between repeat the last "
             f"one (default {PMT_PERIOD_S:g})",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    TGT_INDICES = args.targets
    CONFIG = replace(CONFIG, ref_index=args.ref)
    if args.out is not None:
        CALIB_PATH = args.out
    Path(CALIB_PATH).mkdir(parents=True, exist_ok=True)

    if args.csv is not None:            # offline re-fit, no hardware
        fit_csv(args.csv, args.step6)
        return 0
    _run_sweep(args.step6, fit_after=not args.meas,
               pmt=open_pmt(args.pmt_resource, args.pmt_period,
                            enabled=args.pmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
