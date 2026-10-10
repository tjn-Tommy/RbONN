"""Step 8 -- the two-photon saturation constant, and the gain anchor.

    python src/calibration_module/steps/calib_step8.py --calib FILE --dry-run
    python src/calibration_module/steps/calib_step8.py --calib FILE
    python src/calibration_module/steps/calib_step8.py --calib FILE some.csv

In:  ``--calib FILE``, whose ``step6`` and ``step7`` sections are the forward
     model every drive is solved through.  A positional CSV re-fits that sweep
     offline instead of driving the bench.
Out: into that file's folder, and its ``step8`` section --

     ``calib_step8_patterns_<MMDD_HHMM>.csv``  every pattern, written BEFORE the
         hardware is touched, so an interrupted run is still reproducible.  The
         scalars it was solved under ride in ``#`` meta lines.
     ``calib_step8_<MMDD_HHMM>.csv``           one row per read, partial if
         stopped.
     ``calib_step8_<MMDD_HHMM>_ysat.png``      the three-panel report.

Steps 6 and 7 read at most ~66 mV and are both fitted linear.  Once a real
computation drives the full scale the readout is ``|E|^2 / (1 + |E|^2/Y_sat)``,
not ``|E|^2``; this step measures that one constant and nothing else.

The sweep
---------
ONE band: ``n`` rungs stratified over ``alpha_lo``..``alpha_hi``, where alpha is
the fraction of ``Y_max``, the reachable ceiling of ``|E|^2``.  Each rung is a
fresh random drive found by least-move projection onto its level set
(:mod:`calibration_module.fit.levels`), and the band is shuffled before it is
driven, so level order cannot alias slow drift onto the level axis.  30 rungs is
about five minutes.

The band is a per-run setting (``--alpha-lo``, ``--alpha-hi``, ``--n``, or the
GUI's two alpha boxes).  ``--max-mv`` spells ``--alpha-hi`` in MILLIVOLTS,
converted through this model's ``Y_max`` by :func:`alpha_for_volts`.  Alpha is
what gets stored either way, together with the millivolts it meant and the
``Y_max`` connecting them -- alpha follows the PMT gain and the amplifier does
not, so a bare alpha is not interpretable after a gain change.

PLACE THE BAND HIGH.  A rung's information about ``1/Y_sat`` goes as
``(Y^2/sigma)^2``, so thirty rungs over the bright end return ``Y_sat`` to ~4%.
But not to full scale: :data:`LADDER_CEILING_V` is 450 mV against a ``Y_max`` of
~1534 mV, because the amplifier rails at :data:`AMP_LIMIT_V` and COMPRESSES
below it, which bends the top of the ladder down the way saturation does and
which the one-parameter fit cannot tell apart.  The fitted form
``1/y = 1/Y + 1/Y_sat`` is exact, not a local expansion, so what matters is that
the ladder BRACKETS the range the machine uses -- an inference run's brightest
read is ~400 mV.

Every prediction here -- ``pred_v``, the input range, the read window -- is the
LINEAR model: fitting a saturation constant from a sweep that already assumed
one would be circular.  The linear prediction is the larger of the two, so the
range it picks is always wide enough.

The gain anchor
---------------
The fit has one free parameter: the detector gain is pinned to ``g = 1``, which
is a DEFINITION.  Any gain error is absorbed into ``Y_sat``, levered by
``Y_sat / Y_max``, about 12x here, so this run's ``Y_sat`` and this run's reads
are ONE OBJECT.  Downstream, a computation re-drives a handful of this sweep's
own ladder patterns -- row by row in the CSV, seed in the JSON -- and the ratio
of its reads to this run's on the same drives IS ``g``, free of ``Y_sat``,
``eta`` and the phases, the SLM frame being identical.  The prediction is then::

    Y_pred = g * Y / (1 + Y / Y_sat)

Gain outside the saturation: the atoms saturate on the light they receive and
the PMT amplifies afterwards.

RUN THIS IMMEDIATELY AFTER STEP 7, in the same sitting.  ``pred_v`` is step
6/7's ``eta`` in step 6/7's VOLTS and nothing here floats a scale, so any gain
drift between the two goes entirely into ``Y_sat``, levered ~15x: a 1% drift
moves it 28%, a 2% drift 72%, and the rmse barely moves either way.  Only past
~5% does ``unbounded`` fire.  The gain wanders ~1%/h.

One range and one window
------------------------
:data:`~calibration_module.measure.pair_v2.RANGE_V` and
:data:`~calibration_module.measure.pair_v2.T_MEAS_S`, for every read including
the darks -- the same ones steps 6 and 7 use.  A per-read range would put the
board's per-range offset difference into every dark subtraction that crosses a
boundary, and a window that varied along the level axis would put a
window-dependent systematic on the axis
:func:`~calibration_module.fit.verify.fit_y_sat` reads.  ``range_v`` records the
one range and ``peak_v`` the raw trace peak, so a flat-top is visible after the
fact.

Alpha is coherent, and single_beam is subtracted
------------------------------------------------
``FS = Y_max`` excludes the single-beam background, so every level is on
``|E|^2`` alone and the fit subtracts the model's background from each read.  It
is small but it RAMPS with level, and a ramp coherent with level is what a
saturation fit looks for.  Raw ``voltage_mean_v`` and ``dark_v`` are never
touched: ``sb_pred_v`` is a column and the subtraction happens in the fit.

The probe
---------
:func:`probe_verdict` and :func:`print_probe` display ONE level, read it once
and write nothing, reporting it against the linear model and against
:data:`AMP_LIMIT_V`.  :func:`display_alpha` is the same level with the read left
off -- SLM, no DAQ.

That is the tool for the question :data:`LADDER_CEILING_V` answers: walk alpha
up and watch for the read falling behind the linear prediction faster than one
constant can explain, or the raw peak approaching the rail.  Re-run it after any
change in optical power or PMT gain, ``Y_max`` scaling with both while the
amplifier does not.

``--pmt`` logs the PMT thermistor (Keithley 2400, ohms and TH10K degrees) beside
every read, cached for ``--pmt-period`` seconds.  A 2400 that fails blanks the
columns rather than ending the run.

Not a unittest (it needs real hardware) -- run it directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.levels import (  # noqa: E402
    Pattern,
    full_scale_pattern,
    level_patterns,
)
from calibration_module.fit.verify import (  # noqa: E402
    full_scale,
    load_forward_model,
    predict,
    write_plan_csv,
    write_verify_csv,
)
from calibration_module.measure.bench import connect_daq, connect_slm  # noqa: E402
from calibration_module.measure.patterns import (  # noqa: E402
    DARK_EVERY_S,
    DEFAULT_ACQ,
    PairV2Acq,
    load_layout,
    range_for,
    run_patterns,
    select_pairs,
)
from calibration_module.measure.pmt_temp import (  # noqa: E402,F401
    PMT_PERIOD_S,
    PmtTemperature,
)
from calibration_module.calib_file import (  # noqa: E402
    calib_path,
    load_calib,
    write_section,
)

# ---- Edit these to match your setup ----
# The calib file is NOT here -- it is the required --calib argument, as it is
# for steps 6 and 7.  Everything this step writes lands in its folder.

PAIRS = None                # None -> every pair the calib.json calibrates; --p overrides
PHASE_METHOD = None         # stored step-7 fit; None -> the single one a v2 run has

SLM_DISPLAY_NO = None       # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1              # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

#: The bench acquisition, unchanged -- same window, range and settle as steps 6
#: and 7.  This SCRIPT declares no acquisition of its own; the GUI page does,
#: off the same defaults, so an untouched page runs what this runs.
ACQ = DEFAULT_ACQ

# ---- The run ----
LADDER = 30                     # rungs in the band, one per stratum
SEED = None                     # None -> a fresh seed, printed and written to the plan CSV

# ---- Where the band goes ----
#: The default brightest read, as a VOLTAGE -- what ``--max-mv`` starts at and
#: what :func:`alpha_for_volts` converts into an ``alpha_hi``.  A voltage
#: because every argument that sets it is about voltage: the amplifier, the
#: noise floor, and how bright an inference run reads.
#:
#: AN UNVERIFIED WORKING FIGURE, not a measurement.  Where this amplifier starts
#: to COMPRESS has never been measured; only its hard rail is known
#: (:data:`AMP_LIMIT_V`).  Use the probe (:func:`probe_verdict`) to replace the
#: guess.  Meanwhile the errors are asymmetric -- too low costs precision only
#: (700 -> 450 mV moved sigma(Y_sat) 3.0% -> 3.9%, against the ~11% where it
#: starts to matter), too high is silent -- and 450 is near a hard floor, since
#: a ladder stopping below an inference run's ~400 mV stops covering the range
#: the machine computes in.  Usable range roughly 400..700 mV.
LADDER_CEILING_V = 0.450
#: The default dimmest rung, as a fraction of ``Y_max``.  An ALPHA, not a
#: voltage: nothing about the amplifier or the noise floor sets it, and where
#: the sweep starts is a choice about what you want covered.
ALPHA_LO = 0.15

# ---- The amplifier ceiling ----
#: Where the transimpedance amplifier saturates, at every PMT gain and on every
#: board range.  A property of the electronics, and the one hard limit in the
#: chain: a read flat-topped here measures nothing.  It matters most to this
#: step, the only one that deliberately drives the top of the scale.
AMP_LIMIT_V = 1.0

#: Raw trace peak over the reported mean, p90 across the 0929 sweep.  The 20 Hz
#: low-pass averages a flat top away, so a RAILED read still reports a
#: reasonable mean and the mean cannot be compared to :data:`AMP_LIMIT_V`
#: directly.  This is what turns a predicted mean into the peak that actually
#: reaches the rail -- worth saying out loud before a sweep, and the reason
#: ``peak_v`` is a column in every step-8 CSV.
PEAK_OVER_MEAN = 1.31


# ======================================================================
# building the run
# ======================================================================

def run_seconds(n_reads: int, acq: PairV2Acq = ACQ) -> float:
    """Estimated wall time: every window and settle, plus the periodic darks.

    ``acq`` because the window is a control on the GUI page, as for steps 6 and
    7 -- an estimate quoted under a constant the page is not using is a wrong
    number.
    """
    secs = n_reads * (acq.t_meas_s + acq.settle_s)
    return secs + acq.t_meas_s * (1 + secs // DARK_EVERY_S)


def alpha_for_volts(y_max_v: float, volts: float = LADDER_CEILING_V) -> float:
    """``volts`` as a fraction of ``Y_max`` -- the mV spelling of an alpha.

    Converted against the forward model in hand, the voltages that decide a band
    (the rail, the compression point, how bright an inference run reads) not
    moving with the PMG gain while alpha does.  Clamped to 1: a model dimmer
    than ``volts`` cannot reach it.
    """
    fs = float(y_max_v)
    if not fs > 0:
        return 1.0
    return min(1.0, float(volts) / fs)


def pattern_for_alpha(models, phases, pairs, alpha: float, *, seed: int,
                      label: str = "level") -> Pattern:
    """One solved drive at ``alpha`` -- a rung, or the frame to display.

    At ``alpha == 1`` the level set is a single point, which
    :func:`full_scale_pattern` has in closed form; below that it is an ordinary
    solved rung, so what :func:`display_alpha` and the probe put on the panel is
    a drive the sweep could have chosen.
    """
    if alpha >= 1.0 - 1e-9:
        return replace(full_scale_pattern(models, phases, pairs, label=label),
                       kind="A")
    rung, = level_patterns(models, phases, pairs, 1, seed=seed,
                           alpha_lo=alpha, alpha_hi=alpha)
    return replace(rung, label=label)


def shuffle(rungs: Sequence[Pattern], *, seed: int) -> list[Pattern]:
    """The band in random order.

    Level order would put slow drift on the level axis, the axis the fit reads.
    """
    rng = np.random.default_rng(seed)
    return [rungs[i] for i in rng.permutation(len(rungs))]


def build(models, phases, pairs, *, seed: int, n: int = LADDER,
          alpha_lo: float = ALPHA_LO, alpha_hi: float | None = None):
    """``(scale, patterns, preds)`` -- the whole run, solved, in read order.

    ONE band: ``n`` rungs stratified over ``alpha_lo``..``alpha_hi``, shuffled.
    ``alpha_hi`` defaults to :data:`LADDER_CEILING_V` converted against this
    model's own ``Y_max`` (:func:`alpha_for_volts`), which is the only place a
    voltage still decides anything here.

    ``preds`` are LINEAR: this run fits ``Y_sat``, so it must not assume one.
    """
    scale = full_scale(models, phases, pairs)
    hi = alpha_for_volts(scale.fs) if alpha_hi is None else float(alpha_hi)
    lo = float(alpha_lo)
    if not 0.0 < lo < hi <= 1.0:
        raise ValueError(
            f"band must satisfy 0 < alpha_lo < alpha_hi <= 1, got {lo} .. {hi}")
    rungs = list(level_patterns(models, phases, pairs, int(n), seed=seed,
                                alpha_lo=lo, alpha_hi=hi))
    patterns = shuffle(rungs, seed=seed + 2)
    return scale, patterns, [float(predict(models, phases, pairs, p.x, p.w))
                             for p in patterns]


# ======================================================================
# collect
# ======================================================================

def collect(calib_file: Path, *, pairs=None, method=None, out_dir=None,
            n: int = LADDER, alpha_lo: float = ALPHA_LO,
            alpha_hi: float | None = None, seed: int | None = None,
            dry_run: bool = False, pmt: "PmtTemperature | None" = None) -> str | None:
    """Solve the sweep, write its plan CSV, drive it, write the reads CSV."""
    calib_file = calib_path(calib_file)
    out = Path(calib_file.parent if out_dir is None else out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = select_pairs(PAIRS if pairs is None else pairs)
    models, phases, ref = load_forward_model(calib_file, pairs=wanted,
                                             method=method)
    pairs = sorted(models)
    if seed is None:
        seed = SEED if SEED is not None else int(np.random.SeedSequence().entropy % 2**32)

    scale, patterns, preds = build(models, phases, pairs, seed=seed, n=n,
                                   alpha_lo=alpha_lo, alpha_hi=alpha_hi)
    hi = alpha_for_volts(scale.fs) if alpha_hi is None else float(alpha_hi)
    coh = [float(predict(models, phases, pairs, p.x, p.w, coherent=True))
           for p in patterns]
    secs = run_seconds(len(patterns))

    load_calib(calib_file).print_chain()
    print(f"Pairs     : {pairs}  (N = {len(pairs)}, reference {ref})")
    print(f"Full scale: Y_max {scale.fs * 1e3:.4f} mV   S^2 {scale.bound * 1e3:.4f} mV "
          f"({100 * scale.headroom:.1f} %)   psi* {np.degrees(scale.psi):.2f} deg")
    print(f"            4-bit level step = {scale.fs / 15 * 1e3:.3f} mV (linear)")
    print(f"Model     : LINEAR -- this run fits Y_sat, so it assumes none")
    print(f"Read      : {ACQ.t_meas_s:g} s every read, dark included, at "
          f"+/-{ACQ.range_v:g} V  (dimmest {min(preds) * 1e3:.2f} mV, "
          f"brightest {max(preds) * 1e3:.2f} mV)")
    # Both spellings of the band, because the sweep is specified in one and
    # argued about in the other, and Y_max is what connects them.
    print(f"Band      : {n} rungs over alpha {alpha_lo:.4f}..{hi:.4f} "
          f"= {alpha_lo * scale.fs * 1e3:.1f}..{hi * scale.fs * 1e3:.1f} mV"
          + ("   [alpha_hi is 1: Y_max is under the ceiling]"
             if hi >= 1.0 - 1e-9 else ""))
    peak = hi * scale.fs * PEAK_OVER_MEAN
    if peak > 0.85 * AMP_LIMIT_V:
        print(f"            the top rung's raw trace peak runs ~{peak * 1e3:.0f} mV "
              f"against the {AMP_LIMIT_V:g} V amplifier limit "
              f"({100 * peak / AMP_LIMIT_V:.0f} %) -- compression bends the band "
              f"down the way saturation does, and peak_v is what will show it")
    print(f"Patterns  : {len(patterns)} reads, ~{secs / 60:.0f} min")
    print(f"Seed      : {seed}")
    fell_back = [p.label for p in patterns if not p.exact]
    if fell_back:
        print(f"            {len(fell_back)} pattern(s) fell back to the "
              f"deterministic drive: {', '.join(fell_back[:6])}"
              + (" ..." if len(fell_back) > 6 else ""))

    stamp = time.strftime("%m%d_%H%M")
    plan_path = write_plan_csv(
        out / f"calib_step8_patterns_{stamp}.csv", patterns, preds, coh,
        pairs=pairs, scale=scale,
        meta={"step7_ran": load_calib(calib_file).ran(7),
              "reference": ref, "method": method, "seed": seed,
              "Y_max_v": scale.fs, "S2_v": scale.bound, "psi_rad": scale.psi,
              "level_step_v": scale.fs / 15.0,
              # Alpha is the band; the mV are what it meant at THIS Y_max.  Both,
              # because neither alone survives a gain change.
              "n": n, "alpha_lo": alpha_lo, "alpha_hi": hi,
              "alpha_lo_mv": 1e3 * alpha_lo * scale.fs,
              "alpha_hi_mv": 1e3 * hi * scale.fs,
              "t_meas_s": ACQ.t_meas_s, "range_v": ACQ.range_v})
    print(f"\nPatterns written to {plan_path}")

    if dry_run:
        report_plan(patterns, preds, coh, scale.fs, alpha_hi=hi)
        return None

    return _drive(calib_file, pairs, patterns, preds, coh, models, phases,
                  out=out, stamp=stamp, plan_path=plan_path,
                  meta={"n_pairs": len(pairs), "seed": seed,
                        "y_max_v": scale.fs, "n": n,
                        "alpha_lo": alpha_lo, "alpha_hi": hi},
                  pmt=pmt)


def _drive(calib_file, pairs, patterns, preds, coh, models, phases, *,
           out: Path, stamp: str, plan_path: Path, meta: dict,
           columns: Sequence[str] = (), extra_of=None,
           pmt: "PmtTemperature | None" = None):
    """Display every pattern in read order and write the CSV, partial or not.

    ``extra_of(i, pattern)`` adds per-read columns beyond the standard ones and
    ``columns`` names them in CSV order.  Every read takes ``ACQ.t_meas_s``.
    ``pmt`` adds its two columns and brackets the run with a reading in the meta.
    """
    x = np.stack([p.x for p in patterns])[None]      # (1 block, P, n)
    w = np.stack([p.w for p in patterns])[None]
    extra = [[{"kind": p.kind, "label": p.label, "seq": i,
               "target_alpha": f"{p.target_alpha:.6f}",
               "alpha": f"{p.alpha:.6f}",
               "coherent_v": f"{c:.9g}", "sb_pred_v": f"{q - c:.9g}",
               "t_meas_s": f"{ACQ.t_meas_s:.3f}",
               **(extra_of(i, p) if extra_of else {})}
              for i, (p, c, q) in enumerate(zip(patterns, coh, preds))]]

    meta = dict(meta)
    if pmt is not None:
        columns = [*columns, *PmtTemperature.COLUMNS]
        first = pmt.read()          # also seeds the cache for the first patterns
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
    layout = load_layout(calib_file, pairs)
    rows: list[dict] = []
    status = "complete"
    csv_path = None
    try:
        run_patterns(daq, slm, layout, [tuple(pairs)], x, w, models, phases,
                     rows=rows, extra=extra, y_sat=None,
                     row_hook=pmt)
    except BaseException as exc:
        status = (f"partial ({type(exc).__name__} after {len(rows)} of "
                  f"{len(patterns)} reads)")
        raise
    finally:
        try:
            if pmt is not None:
                # The 2400 is independent of the SLM and DAQ, so this last
                # reading is worth having even when the run died mid-way.
                meta.update({f"{k}_end": v for k, v in pmt.read().items()})
            if rows:            # on disk before anything else can fail
                csv_path = write_verify_csv(
                    out / f"calib_step8_{stamp}.csv", pairs, rows,
                    meta={"step7_ran": load_calib(calib_file).ran(7),
                          "patterns_csv": Path(plan_path).name,
                          "status": status, **meta},
                    extra=["kind", "label", "seq", "target_alpha", "alpha",
                           "coherent_v", "sb_pred_v", "t_meas_s", "peak_v",
                           *columns])
                print(f"\nCSV ({len(rows)} reads, {status}) written to {csv_path}")
        finally:
            slm.close_slm()
            daq.disconnect()
    return csv_path


# ======================================================================
# display one level -- no read
# ======================================================================

def display_alpha(slm, layout, models, phases, pairs, alpha: float, *,
                  seed: int | None = None, log=print) -> Pattern:
    """Solve the drive for one ``alpha`` and put it on the panel.  Nothing else.

    No DAQ, no read, nothing written -- the whole difference from
    :func:`probe_verdict`.

    Out: the pattern, so a caller can report what was ACHIEVED; the projection
    lands near the requested alpha, not exactly on it.
    """
    from slm_module.encoding import encode_to_pattern

    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**32)
    pattern = pattern_for_alpha(models, phases, pairs, float(alpha), seed=seed)
    pred = float(predict(models, phases, pairs, pattern.x, pattern.w))
    slm_width, slm_height = slm.get_slm_info()
    slm.display_array(encode_to_pattern(pattern.x, pattern.w, layout,
                                        slm_width, slm_height))
    log(f"alpha {pattern.alpha:.4f} (asked {alpha:.4f}) displayed: "
        f"predicts {pred * 1e3:.3f} mV, LINEAR"
        + ("" if pattern.exact else "   [fell back to the deterministic drive]"))
    return pattern


# ======================================================================
# the probe -- display one level, read it once
# ======================================================================
#
# One drive, one read, nothing written.  This is how the bench finds out where
# the readout stops being linear -- a question about the ELECTRONICS, not the
# atoms.  Raise the optical power or the PMT gain and the brightest reads clip,
# or merely COMPRESS well before the rail, and either bends the top of the
# ladder down exactly the way saturation does; the one-parameter fit cannot tell
# them apart, so what comes out looks clean and is wrong.  Walking alpha up
# until the read falls behind the linear model is the measurement that decides
# :data:`LADDER_CEILING_V`.

def probe_verdict(*, alpha: float, y_max_v: float, coherent_v: float,
                  sb_pred_v: float, mean_v: float, dark_v: float, std_v: float,
                  peak_v: float = float("nan"), range_v: float = float("nan"),
                  limit_v: float = AMP_LIMIT_V,
                  alpha_hi: float | None = None) -> dict:
    """One displayed level and one read: what the bench did with it.

    In:  ``coherent_v`` and ``sb_pred_v``, the LINEAR predictions for the drive
         displayed (:func:`~calibration_module.fit.verify.predict` with
         ``coherent=True``, and the single-beam background it leaves out), plus
         the read itself.
    Out: a dict of verdicts, including an indicative ``Y_sat`` from inverting
         the one read.  Saturation acts on the coherent part alone::

             Y_sat = coh * m / (coh - m),    m = (mean - dark) - sb

    A preview of what the sweep will fit, not a replacement, and meaningless if
    the read was clipped.  Clipping is called on the RAW trace peak, never the
    mean: the 20 Hz low-pass pulls a flat top back under the rail.
    """
    coh = float(coherent_v)
    meas = float(mean_v) - float(dark_v)
    m = meas - float(sb_pred_v)

    if 0.0 < m < coh:
        y_sat = coh * m / (coh - m)
    elif m >= coh:
        y_sat = float("inf")            # read met or beat the linear model
    else:
        y_sat = float("nan")            # nothing was there

    limit = float(limit_v)
    peak = float(peak_v)
    near = max(peak if np.isfinite(peak) else 0.0, meas)
    pred = coh + float(sb_pred_v)
    # The band's own top, so the probe can say whether this read is inside the
    # range a sweep would cover or out past it.  Defaults to what the ceiling
    # voltage converts to at this Y_max, which is where a sweep would stop.
    cap = alpha_for_volts(y_max_v) if alpha_hi is None else float(alpha_hi)
    return {
        "alpha": float(alpha),
        "y_max_v": float(y_max_v),
        "limit_v": limit,
        "alpha_hi": cap,
        "coherent_pred_v": coh,
        "sb_pred_v": float(sb_pred_v),
        "pred_v": pred,
        "mean_v": float(mean_v), "dark_v": float(dark_v), "std_v": float(std_v),
        "measured_v": meas,
        "measured_coherent_v": m,
        "peak_v": peak,
        "peak_fraction": (peak / limit) if limit else float("nan"),
        "read_range_v": float(range_v),
        "shortfall": (1.0 - meas / pred) if pred else float("nan"),
        "y_sat_implied_v": y_sat,
        "over_band": bool(float(alpha) > cap),
        "clipped": bool(near >= 0.98 * limit),
        # TIGHT is about the raw peak, not the mean: compression starts before
        # the rail does, and the peak is what reaches it first.
        "verdict": ("CLIPPED" if near >= 0.98 * limit
                    else "TIGHT" if near >= 0.85 * limit
                    else "OK"),
    }


def print_probe(d: dict, log=print) -> None:
    """The probe block: what was asked for, what came back, and what it implies."""
    fs = d["y_max_v"]
    log(f"alpha {d['alpha']:.4f} of Y_max {fs * 1e3:.1f} mV   "
        f"(a sweep's top would be {d['alpha_hi']:.3f} "
        f"= {d['alpha_hi'] * fs * 1e3:.0f} mV)")
    log(f"predicted         {d['pred_v'] * 1e3:9.3f} mV  (LINEAR, of which "
        f"{d['sb_pred_v'] * 1e3:.3f} mV is single beam)")
    log(f"measured          {d['measured_v'] * 1e3:9.3f} mV  "
        f"+/-{d['std_v'] * 1e3:.3f}  (dark {d['dark_v'] * 1e3:+.4f} mV, "
        f"range {d['read_range_v']:g} V)")
    log(f"vs prediction     {100 * d['shortfall']:+9.2f} %")
    log(f"raw trace peak    {d['peak_v'] * 1e3:9.3f} mV  = "
        f"{100 * d['peak_fraction']:.1f} % of the {d['limit_v']:g} V amplifier "
        f"limit")
    y = d["y_sat_implied_v"]
    if np.isfinite(y):
        log(f"implied Y_sat     {y:9.3f} V   (ONE read -- a preview of the fit, "
            f"and meaningless if clipped)")
    else:
        log("implied Y_sat         none   (the read met or beat the linear model)")
    log(f"verdict           {d['verdict']:>9}"
        + ("   the raw trace reached the rail: this read is a flat-top"
           if d["clipped"] else
           "   the raw peak is close to the rail" if d["verdict"] == "TIGHT"
           else ""))
    if d["over_band"]:
        log(f"                            this alpha is above the band's top "
            f"({d['alpha_hi'] * fs * 1e3:.0f} mV) -- brighter than the sweep "
            f"goes, which is what walking it up is for")


# ======================================================================
# the fit
# ======================================================================

def fit_and_write(csv_path, calib_file, *, pairs=None, method=None,
                  out_dir=None, log=print) -> dict:
    """Fit ``Y_sat`` on a step-8 CSV, write the step8 section and the PNG.

    The section records the CSV it came from: a ``Y_sat`` and the reads that
    defined ``g = 1`` are one object, and a later run measures its gain against
    THOSE reads.
    """
    from calibration_module.fit.step8_report import report_csv

    csv_path = Path(csv_path)
    calib_file = calib_path(calib_file)
    out = Path(calib_file.parent if out_dir is None else out_dir)
    result = report_csv(csv_path, calib_file, pairs=pairs, method=method,
                        out_dir=out, log=log)

    section = {
        "y_sat_v": result["y_sat_v"],
        "y_sat_sigma_v": result["sigma_v"],
        "y_sat_lo_v": result["lo_v"], "y_sat_hi_v": result["hi_v"],
        "unbounded": result["unbounded"],
        "gain_lever": result["gain_lever"],
        "rmse_v": result["rmse_v"], "rmse_linear_v": result["rmse_linear_v"],
        "n_reads": result["n"], "n_ladder": result["n_ladder"],
        "png": None if not result.get("png") else Path(result["png"]).name,
        "y_max_pred_v": result["y_max_pred_v"],
        # g = 1 is DEFINED by the reads in this section's csv.  A later run
        # divides its reads on a subset of those same drives by them to get its
        # gain, so the convention is recorded with the constant it qualifies.
        "gain_reference": {"convention": "g = 1 at this sweep"},
    }
    calib = write_section(calib_file, 8, section, csv=csv_path, log=None)
    # Same reason the PNG name is in the section: the caller that asked for a
    # different out_dir should not have to re-derive it from the CSV.
    result["result_json"] = str(calib.path)
    log(f"Y_sat written to {calib.path}")
    return result


def report_plan(patterns, preds, coh, fs: float, *,
                alpha_hi: float | None = None) -> None:
    """The dry-run table: what will be driven, in read order, and how bright.

    ``wants`` is the range :func:`range_for` WOULD pick per read; every read is
    actually taken at the fixed ``ACQ.range_v``, so a row wanting more is a row
    the amplifier cannot hold either -- they are the same 1 V.
    """
    step = fs / 15.0
    print(f"\n  {'seq':>4} {'label':<22} {'alpha':>7} "
          f"{'coh (mV)':>10} {'pred (mV)':>10} {'sb (mV)':>8} {'wants':>7}"
          f"   (every read +/-{ACQ.range_v:g} V, {ACQ.t_meas_s:g} s)")
    for i, (p, c, q) in enumerate(zip(patterns, coh, preds)):
        rng = range_for(q)[0]
        print(f"  {i:4d} {p.label:<22} {p.alpha:7.4f} "
              f"{c * 1e3:10.4f} {q * 1e3:10.4f} {(q - c) * 1e3:8.4f} "
              f"{rng:6.1f}V" + ("" if p.exact else "  [fallback]"))

    a = np.array([p.alpha for p in patterns])
    used = np.array([range_for(q)[0] for q in preds])
    sb = np.array([q - c for q, c in zip(preds, coh)])
    # Binned over the BAND's own top, not over [0, Y_max].  The sweep stops short
    # of full scale on purpose, so a histogram against Y_max would report the
    # empty levels above the band as a gap rather than as the design.
    cap = max(float(a.max()) if alpha_hi is None else float(alpha_hi), 1e-9)
    hist = np.bincount(np.clip((a / cap * 15 + 1e-9).astype(int), 0, 15),
                       minlength=16)
    print(f"\n  alpha        : min {a.min():.4f}  median {np.median(a):.4f}  "
          f"max {a.max():.4f}   (band top {cap:.4f} = {cap * fs * 1e3:.0f} mV)")
    print(f"  coverage     : {' '.join(str(c) for c in hist)}   "
          f"({int((hist > 0).sum())} of 16 bins of [0, band top] occupied)")
    print(f"  single_beam  : {sb.min() * 1e3:.3f} .. {sb.max() * 1e3:.3f} mV  "
          f"({100 * (sb.max() - sb.min()) / step:.1f} % of a level step)")
    print("  DAQ ranges   : " + ", ".join(
        f"{r:g}V x{n}" for r, n in zip(*np.unique(used, return_counts=True))))
    top, bot = max(preds), min(preds)
    print(f"\n  the fit lives at the top: at the brightest rung "
          f"({top * 1e3:.1f} mV) a Y_sat of 5 V bends\n  the reading down by "
          f"{100 * top / (5.0 + top):.1f} %, and at the dimmest "
          f"({bot * 1e3:.1f} mV) by {100 * bot / (5.0 + bot):.2f} % -- which is "
          f"why\n  a band placed low measures almost nothing")


# ======================================================================
# CLI
# ======================================================================

def _alpha_hi(args, calib_file: Path, wanted) -> float | None:
    """``--alpha-hi`` if given, else ``--max-mv`` as an alpha, else the default.

    ``None`` means neither was named, which lets :func:`collect` convert against
    the ``Y_max`` it is about to compute -- one place, not two.  A typed
    ``--max-mv`` must be converted here instead: it needs the forward model, and
    ``collect`` would never see the number.
    """
    if args.alpha_hi is not None:
        return float(args.alpha_hi)
    if abs(args.max_mv - 1e3 * LADDER_CEILING_V) < 1e-12:
        return None                     # untouched default: let collect convert
    models, phases, _ref = load_forward_model(calib_file, pairs=wanted,
                                              method=args.method)
    return alpha_for_volts(full_scale(models, phases, sorted(models)).fs,
                           args.max_mv / 1e3)


def _one_level(calib_file: Path, alpha: float, *, pairs, method,
               read: bool) -> int:
    """``--alpha``: display one level, and optionally read it once.

    Writes nothing either way.  Without ``--read`` the DAQ is never opened and
    the frame stays up until interrupted, which is the point of it.
    """
    models, phases, _ref = load_forward_model(calib_file, pairs=pairs,
                                              method=method)
    driven = sorted(models)
    scale = full_scale(models, phases, driven)
    layout = load_layout(calib_file, driven)
    load_calib(calib_file).print_chain()
    print(f"Pairs     : {driven}   Y_max {scale.fs * 1e3:.4f} mV")

    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = None
    try:
        pattern = display_alpha(slm, layout, models, phases, driven, alpha)
        if not read:
            print("Holding the frame.  Ctrl-C to clear it and exit.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\nClearing.")
            return 0

        daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                          t_both=ACQ.t_meas_s, t_single=ACQ.t_meas_s,
                          min_val=-ACQ.range_v, max_val=ACQ.range_v)
        rows: list[dict] = []
        run_patterns(daq, slm, layout, [tuple(driven)],
                     pattern.x[None, None], pattern.w[None, None],
                     models, phases, rows=rows, acq=ACQ, y_sat=None)
        if not rows:
            print("the probe read produced no row")
            return 1
        r = rows[0]
        coh = float(predict(models, phases, driven, pattern.x, pattern.w,
                            coherent=True))
        tot = float(predict(models, phases, driven, pattern.x, pattern.w))
        print()
        print_probe(probe_verdict(
            alpha=pattern.alpha, y_max_v=scale.fs, coherent_v=coh,
            sb_pred_v=tot - coh, mean_v=r["mean_v"], dark_v=r["dark_v"],
            std_v=r["std_v"], peak_v=r.get("peak_v", float("nan")),
            range_v=r["range_v"]))
        return 0
    finally:
        slm.close_slm()
        if daq is not None:
            daq.disconnect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calib_step8.py",
        description="Step 8: fit the two-photon saturation constant Y_sat from "
                    "one level sweep, and anchor g = 1 for everything "
                    "downstream.",
        epilog="--calib has no default: that one file is the whole forward "
               "model (step 3 the layout, step 6 the etas, step 7 the phases), "
               "so running against the wrong one reports a saturation constant "
               "that is really a file mix-up.",
    )
    parser.add_argument("--calib", required=True, type=Path, metavar="FILE",
                        help="calib.json carrying the forward model; the CSVs "
                             "land beside it and the Y_sat goes back into it as "
                             "its step8 section; a directory means its "
                             "calib.json (required)")
    parser.add_argument("csv", nargs="?", type=Path,
                        help="re-fit this recorded CSV offline instead of collecting")
    parser.add_argument("--p", nargs="+", type=int, default=None, metavar="PAIR",
                        help="only drive these pairs (default: every calibrated pair)")
    parser.add_argument("--n", type=int, default=LADDER, metavar="N",
                        help=f"rungs in the band, one per stratum (default "
                             f"{LADDER})")
    parser.add_argument("--alpha-lo", type=float, default=ALPHA_LO, metavar="A",
                        help=f"dimmest rung, as a fraction of Y_max (default "
                             f"{ALPHA_LO:g}).  A rung's information about "
                             f"1/Y_sat goes as (Y^2/sigma)^2, so the bottom of "
                             f"the band weighs almost nothing in the fit -- set "
                             f"this by what you want COVERED, not by what you "
                             f"want measured")
    parser.add_argument("--alpha-hi", type=float, default=None, metavar="A",
                        help="brightest rung, as a fraction of Y_max.  Default: "
                             "whatever --max-mv converts to at this forward "
                             "model's Y_max")
    parser.add_argument("--max-mv", type=float, metavar="MV",
                        default=1e3 * LADDER_CEILING_V,
                        help=f"--alpha-hi spelled in MILLIVOLTS, converted "
                             f"against this model's Y_max (default "
                             f"{1e3 * LADDER_CEILING_V:g}).  This is an "
                             f"unverified figure for where the amplifier starts "
                             f"to compress, not a measurement -- see "
                             f"LADDER_CEILING_V.  Ignored if --alpha-hi is given")
    parser.add_argument("--alpha", type=float, default=None, metavar="A",
                        help="display ONE level at this alpha and hold it until "
                             "interrupted; drive nothing else, read nothing, "
                             "write nothing.  Needs the SLM only.  Add --read to "
                             "take one probe read of it instead")
    parser.add_argument("--read", action="store_true",
                        help="--alpha only: read the displayed level once and "
                             "report it against the linear model (the probe). "
                             "Needs the DAQ too; still writes nothing")
    parser.add_argument("--method", default=PHASE_METHOD,
                        help="stored step-7 fit to predict from")
    parser.add_argument("--seed", type=int, default=None,
                        help="reproduce a previous run's patterns")
    parser.add_argument("--pmt", nargs="?", const="", default=None, metavar="RESOURCE",
                        help="log the PMT thermistor (Keithley 2400) beside every "
                             "read: adds pmt_ohm_Ohm and pmt_c columns, sampled "
                             f"every {PMT_PERIOD_S:g} s.  The instrument settings "
                             "come from drafts/daq_pmt_monitor.py; give a VISA "
                             "resource to override its address")
    parser.add_argument("--pmt-period", type=float, default=PMT_PERIOD_S, metavar="S",
                        help=f"--pmt only: seconds between 2400 readings "
                             f"(default {PMT_PERIOD_S:g})")
    parser.add_argument("--dry-run", action="store_true",
                        help="solve the patterns, write the plan CSV, print it; "
                             "touch no hardware")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        wanted = select_pairs(args.p)
    except ValueError as exc:
        parser.error(str(exc))

    calib_file = calib_path(args.calib)
    if args.csv is not None:
        fit_and_write(args.csv, calib_file, pairs=wanted, method=args.method)
        return 0
    if args.alpha is not None:
        return _one_level(calib_file, args.alpha, pairs=wanted,
                          method=args.method, read=args.read)
    if args.read:
        parser.error("--read applies to --alpha; pass one")

    pmt = None
    if args.pmt is not None and not args.dry_run:
        try:
            pmt = PmtTemperature(args.pmt or None, period_s=args.pmt_period)
        except Exception as exc:        # noqa: BLE001
            # A missing VISA stack must not cost the run the sweep.
            print(f"--pmt ignored ({type(exc).__name__}: {exc})")

    csv_path = collect(calib_file, pairs=wanted, method=args.method,
                       n=args.n, alpha_lo=args.alpha_lo,
                       alpha_hi=_alpha_hi(args, calib_file, wanted),
                       seed=args.seed, dry_run=args.dry_run, pmt=pmt)
    if csv_path is not None:
        fit_and_write(csv_path, calib_file, method=args.method)
        print(f"\nRe-fit with:  python {Path(__file__).name} "
              f"--calib {calib_file} {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
