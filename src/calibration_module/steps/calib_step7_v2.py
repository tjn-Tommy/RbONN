"""Step 7 v2 -- comb phase dPhi_comb per pair, ONE free parameter.

Needs real hardware for the sweep; the refit path is offline.  ``--calib``,
``--targets`` and ``--ref`` are required and have no default::

    S="--calib src/calib_data/archive/2026-09/run_0930/calib.json --targets 2,3,4,5,6 --ref 2"
    python src/calibration_module/steps/calib_step7_v2.py $S
    python src/calibration_module/steps/calib_step7_v2.py $S --meas
    python src/calibration_module/steps/calib_step7_v2.py $S some.csv

In:  one ``calib.json`` -- its ``step3`` section IS the channel layout driven,
     its ``step6`` section is what the fringe amplitudes are pinned to -- plus
     the target list and the reference.  A positional CSV re-fits that sweep
     offline; ``--meas`` stops after writing the CSV.  Nothing has a default:
     running against the wrong file does not fail, it reports a phase measured
     under a different aperture.
Out: ``calib_step7v2_meas_<MMDD_HHMM>.csv`` in that file's folder, and the
     file's ``step7`` section -- the ``{Phi_k}`` spectrum, beside the step-3
     layout and step-6 etas it was measured against.

Each target pair carries a fixed comb-phase offset relative to a common
reference pair, which defines ``Phi = 0``.  Driving the two at once makes them
interfere, and the fringe encodes that offset; every target gives the spectrum.

The drive.  The reference is held at ``x_r = w_r = CONFIG.ref_level``; the
target's TWO channels are swept TOGETHER (``x_t = w_t = v``) over the ramp.  A
channel at intensity ``v`` sits at panel phase ``theta = 2 asin(sqrt(v))`` with
field ``sqrt(v) exp(i theta/2)``, so ``g = sqrt(x_t w_t) = sin^2(theta/2)`` and
``dPhi_SLM = theta - theta_ref``::

    Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_SLM + dPhi_comb)
        + step-6 single-beam background

Sweeping ``v`` 0.1 -> 0.9 traces most of the half fringe (~37..143 deg).

What changed vs :mod:`calib_step7_v1`
-------------------------------------
1. **Nothing floats but the phase.**  ``a = eta_ref*g_ref``, ``b = eta_tgt``,
   the single-beam response and the dark are ALL taken from step 6 and the
   measured all-off read, so ``dPhi_comb`` is the only free parameter.
2. **Both arms stop below 1.0.**  See :class:`PhaseV2Config`.
3. **The reads are inverted at the source.**  The transimpedance amplifier
   outputs a NEGATIVE voltage for positive light, so ``ACQ.invert`` records a
   positive signal and the CSV on disk already holds one; v1's ``--flip``
   post-processing is gone with it.

The fit is :func:`~calibration_module.fit.phase.fit_phase_fixed` -- weighted
nonlinear least squares in the one parameter, errors from the weighted Jacobian
as-is -- and the drive and reads are in
:mod:`calibration_module.measure.phase_v2`.  This file is the runner.
Everything tunable is the constants below: the drive in ``CONFIG``, the
acquisition in ``ACQ``, both plain values the GUI builds too.

Each point is one fixed-duration acquisition, as in step 6: ``t_meas_s`` for
every read including the all-off dark, since one window is what makes a sweep
point comparable to the step-6 eta it is pinned against.  Every row records the
mean, its trace std and the ratio of the two.

``--pmt`` (on by default) logs the PMT thermistor beside every row and brackets
the CSV with a start and end reading.  A step-7 run is ~20 minutes in which the
reference amplitude is ASSUMED CONSTANT, so what the detector's temperature did
meanwhile is worth having on the same rows.  Re-read at most every
``--pmt-period`` seconds; a failing 2400 blanks the columns rather than ending
the run.  One GPIB, one 2400: do NOT run ``drafts/daq_pmt_monitor.py``
alongside a sweep.
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
    darks_in_csv,
    load_pair_models,
    load_phase_csv,
    median_dark_without,
    references_in_csv,
    save_step7_section,
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
from calibration_module.fit.outliers import BURST_K  # noqa: E402
from calibration_module.fit.pair_v2 import step6_floor  # noqa: E402
from calibration_module.measure.bench import connect_daq, connect_slm  # noqa: E402
from calibration_module.measure.phase_v2 import (  # noqa: E402
    DEFAULT_ACQ,
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
from calibration_module.calib_file import calib_path, load_calib  # noqa: E402

# ---- Edit these to match your setup ----
# Neither the calib file nor the pairs are here -- they are the required
# --calib, --targets and --ref arguments.  See the module docstring for why they
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

# Acquisition timing and input range: the bench's, shared with steps 6 and 8
# and the ONN loop.  This step used to run its own 10 s / +/-0.2 V because it
# is the brightest of the three; the range is now the amplifier's own 1 V for
# everyone, and one window is what makes a step-7 point comparable to the
# step-6 eta it is pinned against.
ACQ = DEFAULT_ACQ

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

def _load_layout(calib_file: Path):
    """Channel layout from the file's step-3 section.

    The layout this run drives sits beside the step-6 etas in the same file, and
    writing a new step 3 drops the step 6, so the two cannot describe different
    apertures.
    """
    layout = load_calib(calib_file).layout()
    for name, idx in ([("ref_index", CONFIG.ref_index)]
                      + [("TGT_INDICES", k) for k in TGT_INDICES]):
        if not (0 <= CONFIG.slot(idx) < layout.n_channels):
            raise ValueError(
                f"{name} entry {idx} out of range (layout has "
                f"{layout.n_channels} pairs, numbered from "
                f"{CONFIG.pair_index_base})"
            )
    return layout


def _load_models(calib_file: Path, targets, *, ref: int | None = None):
    """Step-6 pair models; require the reference and every requested target.

    ``ref`` defaults to ``CONFIG.ref_index`` -- a sweep is about to drive it --
    but a re-fit passes the reference the CSV itself records instead.
    """
    ref = CONFIG.ref_index if ref is None else ref
    models = load_pair_models([load_calib(calib_file).path])
    needed = [("reference", ref)] + [("target", k) for k in targets]
    for role, idx in needed:
        if idx not in models:
            raise ValueError(
                f"no step-6 model for {role} pair index {idx}; found "
                f"{sorted(models)} in {load_calib(calib_file).path}"
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
          f"std includes the run's sigma floor, inherited from step 6)")
    print(f"  R^2 = {fit.r2:.4f}")


# ======================================================================
# the run
# ======================================================================

def _run_sweep(calib_file: Path, fit_after: bool, *,
               pmt: "PmtTemperature | None" = None) -> None:
    """Sweep every target pair back to the reference; write one raw CSV.

    Back to, not necessarily against: a pair adjacent to the reference cannot be
    measured against it at all, so
    :func:`~calibration_module.fit.cascade.plan_cascade` splits the run into
    stages with their own references and the fit chains the far ones back.  One
    reference and no neighbours is the one-stage case.
    """
    if not TGT_INDICES:
        raise ValueError("no target pairs -- pass --targets, e.g. --targets 2,3,4,5,6")
    layout = _load_layout(calib_file)
    # The whole chain, stamps and all, before the laser is asked for anything.
    load_calib(calib_file).print_chain()
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
                      t_both=ACQ.t_meas_s, t_single=ACQ.t_meas_s,
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
    csv_path = write_meas_csv(
        results, calib_file.parent / f"calib_step7_meas_{stamp}.csv", meta=meta)
    n_rows = sum(r.trial.size for r in results)
    print(f"\nSaved {n_rows} rows to {csv_path}")  # raw rows on disk BEFORE fitting
    if not fit_after:
        print(f"Fit with:  python {Path(__file__).name} "
              f"--calib {calib_file} {csv_path}")
        return
    fit_csv(csv_path, calib_file)


def fit_csv(path: str | Path, calib_file: Path, *,
            dark: dict[int, float | None] | None = None) -> None:
    """Re-fit an already-recorded CSV offline (no hardware).

    In:  this CSV plus the calib file, whose step-6 section gives each pair's
         eta and single-beam background.  One collected CSV carries several
         targets against the shared reference, and every one with a step-6 model
         is fitted separately.  ``dark`` repairs a glitched all-off read:
         ``{pair: volts}`` pins that pair's dark, ``{pair: None}`` takes the
         other targets' median
         (:func:`~calibration_module.fit.phase.median_dark_without`).
    Out: the file's step7 section, REPLACING what was there -- to keep the old
         one, re-fit against a copy, the file being the calibration.  No
         pictures: the GUI's Load draws the fringe from these same rows.

    A dark override leaves the drive rows untouched -- it exists because the
    dark is ONE unguarded acquisition and the sweep around it can be perfectly
    good -- and is recorded in the section, so a repaired run never reads as a
    clean one.
    """
    targets = targets_in_csv(path, TGT_INDICES)
    # The CSV knows its own reference(s), so it wins over CONFIG, which
    # describes the NEXT sweep: otherwise re-fitting a file recorded against a
    # different reference asks step 6 for a pair that was never the reference,
    # and every target fails for a reason that looks like arithmetic.  A
    # cascaded run records one per stage, and the root is the one nothing else
    # measured (self-rows do not count).
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
        calib_file, sorted({k for k in targets if k != ref}
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

    fits: dict[int, PhaseFit] = {}
    bursts: dict[int, list[dict]] = {}
    floor = _step6_floor(calib_file)
    print(f"Sigma floor: {floor[0]*1e6:.1f} uV [{floor[1]}]")
    darks = dict(dark or {})
    if darks and not set(darks) <= set(fittable):
        raise ValueError(
            f"--dark names pair(s) {sorted(set(darks) - set(fittable))}, which "
            f"this CSV does not fit (fittable: {fittable})"
        )
    repaired: dict[str, dict] = {}
    for k in fittable:
        print(f"\n=== pair {k} vs reference {ref_of[k]} ===")
        dark_k = darks[k] if k in darks else None
        if k in darks:
            was = darks_in_csv(path).get(k)
            if dark_k is None:
                dark_k = median_dark_without(path, k)
            # Loud for the same reason a burst is: the bench misbehaved, and a
            # repair that only lives in a shell history is a repair nobody can
            # audit six weeks from now.
            print(f"!! DARK OVERRIDE: recorded {was*1e3:+.4f} mV -> "
                  f"{dark_k*1e3:+.4f} mV (drive rows untouched)")
            repaired[str(k)] = {"recorded_v": was, "used_v": float(dark_k)}
        result = load_phase_csv(path, models[k], models[ref_of[k]],
                                comb_only=True, single_beam_bg=SINGLE_BEAM_BG,
                                only_tgt=k, dark=dark_k, floor=floor)
        for key, entry in result.dark_overrides.items():
            # The automatic ceiling fired; --dark did not have to be asked for.
            print(f"!! DARK OVER CEILING: pair {key} recorded "
                  f"{entry['recorded_v']*1e3:+.4f} mV -> "
                  f"{entry['used_v']*1e3:+.4f} mV (drive rows untouched)")
            repaired[key] = entry
        dts = result.per_trial_darks()
        drift = f" +/- {dts.std(ddof=1)*1e3:.4f} drift" if dts.size > 1 else ""
        print(f"Loaded {result.trial.size} rows, "
              f"dark = {result.dark*1e3:.4f}{drift} mV")
        # Loud, because a burst is a BENCH fault and not a statistical
        # nuisance.  The 1/std^2 weighting means dropping it barely moves
        # dPhi_comb, so this print is the only way it reaches a human.
        if result.dropped_bursts:
            bursts[k] = result.dropped_bursts
            print(f"!! {len(result.dropped_bursts)} BURST acquisition(s) dropped "
                  f"(trace std >{BURST_K:g}x the sweep median) -- "
                  f"the bench glitched, investigate:")
            for b in result.dropped_bursts:
                print(f"     x_t=w_t={b['x_t']:g}: "
                      f"{b['voltage_mean_v']*1e3:+.4f} mV with a "
                      f"{b['voltage_std_v']*1e3:.4f} mV trace std")
        report(result.fit, result.tgt_index, result.ref_index)
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

    # A stored dphi_comb_rad IS Phi_k in the forward model
    # E = sum_k eta_k sqrt(x_k w_k) exp(i[u_k + v_k + Phi_k]) -- nothing negates
    # it either way.  Bursts are recorded per target so a result traces back to
    # the rows that made it, cascaded or not.
    extra: dict = {}
    if chained:
        extra["cascade"] = {
            "min_sep": MIN_PAIR_SEP,
            "stages": [{"ref": s.ref, "targets": list(s.targets),
                        "depth": s.depth} for s in stages],
        }
    if bursts:
        extra["dropped_bursts"] = {str(k): v for k, v in bursts.items()}
    if repaired:
        extra["dark_overrides"] = repaired
    extra["sigma_floor_v"] = float(floor[0])
    extra["sigma_floor_source"] = str(floor[1])
    calib = save_step7_section({(k, METHOD): f for k, f in fits.items()},
                               calib_file, ref_index=ref, csv=Path(path),
                               single_beam_bg=SINGLE_BEAM_BG,
                               chains=chains,
                               extra=extra or None)
    print(f"\nStep-7 spectrum -> {calib.path}")


def _step6_floor(calib_file: Path) -> tuple[float, str]:
    """The sigma floor to weight this step-7 fit by, from the file's step 6.

    Step 7 has no dark level of its own -- one all-off read per target and ten
    drive rows is too few to take a quiet-point median from -- while the step 6
    in the same file measured it properly, on a commanded dark level with
    repeats across every pair, through the same detector on the same night.  So
    it is inherited, and marked inherited.  A section predating the recorded
    floor has it measured from its CSV, never a constant
    (:func:`~calibration_module.fit.pair_v2.step6_floor`).
    """
    return step6_floor(calib_file)


def _stages_from_refs(ref_of: dict[int, int], fittable, root: int) -> list[Stage]:
    """Rebuild the stage list from what the CSV recorded, root-first.

    A re-fit does not re-plan: the file says which reference each target was
    swept against, whatever the run intended.  Root-first is what lets
    :func:`chain_fits` resolve a hub before what hangs off it.
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


def _dark_override(text: str) -> tuple[int, float | None]:
    """``"4"`` -> ``(4, None)`` (the other pairs' median); ``"4=-6e-5"`` -> ``(4, -6e-5)``.

    Pair-scoped on purpose.  A glitched all-off read belongs to one target, and
    a blanket scalar would quietly move every other pair's phase to fix one.
    """
    pair, _, volts = text.partition("=")
    try:
        k = int(pair)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected PAIR or PAIR=VOLTS, got {text!r}") from None
    if not volts:
        return k, None
    try:
        return k, float(volts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{volts!r} is not a voltage (pair {k})") from None


def main(argv: list[str] | None = None) -> int:
    global TGT_INDICES, CONFIG
    parser = argparse.ArgumentParser(
        prog="calib_step7_v2.py",
        description="Step 7 v2 -- comb phase dPhi_comb per pair (one free parameter).",
        epilog="--calib, --targets and --ref have no default.  --calib decides "
               "both what is driven (its step-3 section IS the layout) and what "
               "the fringe amplitudes are pinned to, so running against the "
               "wrong one reports a phase measured under a different aperture.",
    )
    parser.add_argument(
        "--calib", required=True, type=Path, metavar="FILE",
        help="calib.json whose step3 section is the layout and whose step6 "
             "section is the pinned etas; the CSV lands beside it and the "
             "step7 section goes in it; a directory means its calib.json "
             "(required)",
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
        "csv", nargs="?", type=Path,
        help="re-fit this recorded measurement CSV offline instead of sweeping",
    )
    parser.add_argument(
        "--dark", action="append", type=_dark_override, default=None,
        metavar="PAIR[=VOLTS]",
        help="re-fit PAIR with a repaired all-off dark: bare PAIR takes the "
             "other targets' median, PAIR=VOLTS pins it.  Offline re-fit only; "
             "the override is recorded in the step-7 section",
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
    calib_file = calib_path(args.calib)

    if args.csv is not None:            # offline re-fit, no hardware
        fit_csv(args.csv, calib_file, dark=dict(args.dark or []))
        return 0
    if args.dark:
        parser.error("--dark repairs a recorded CSV; pass one to re-fit")
    _run_sweep(calib_file, fit_after=not args.meas,
               pmt=open_pmt(args.pmt_resource, args.pmt_period,
                            enabled=args.pmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
