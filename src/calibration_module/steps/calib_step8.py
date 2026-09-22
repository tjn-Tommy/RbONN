"""Step 8 -- the two-photon saturation constant, and the gain anchor.

    python src/calibration_module/steps/calib_step8.py --step7 RESULT.json --dry-run
    python src/calibration_module/steps/calib_step8.py --step7 RESULT.json
    python src/calibration_module/steps/calib_step8.py --step7 RESULT.json some.csv

Steps 6 and 7 fit the forward model -- the per-pair ``eta`` and single-beam
response, then the relative phases ``Phi_k``.  Both read at most ~66 mV, where
the atoms are linear to ~1 %, so both are fitted linear and neither can see the
one thing that matters once a real computation drives the full scale: the
readout is ``|E|^2 / (1 + |E|^2 / Y_sat)``, not ``|E|^2``.  On the 0916 ten-pair
run, which reached 500 mV, the linear model read 8 % low at the top and the
whole-run rms error fell from 26 mV to 4 mV once ``Y_sat`` was fitted.

This step measures that one constant, and nothing else.

The sweep
---------
One family: the ladder.  ``LADDER`` patterns, one per stratum of ``[0, Y_max]``,
each a fresh random drive found by least-move projection onto its level set
(:mod:`calibration_module.fit.levels`), plus the closed-form full-scale drive
at exactly ``alpha = 1`` displayed at the start and again at the end.

There used to be two more families -- repeats at fixed drive, and a psi scan
for measured full scale -- and they are gone on purpose.  The repeats measured
readout reproducibility, which is a question about the machine rather than
about ``Y_sat``; the psi scan measured the ceiling, which the ladder now
reaches by itself because ``LADDER_ALPHA_MAX`` is 1 and the ``alpha = 1`` drive
has a closed form.  What is left runs in about a quarter of an hour instead of
half of one, and every read in it constrains the fit.

The ladder is shuffled before it is driven, because running it in level order
would alias slow drift straight onto the level axis -- the axis being measured.

Why the top of the sweep is the whole measurement
-------------------------------------------------
The curvature a saturation fit sees grows with ``|E|^2``: at 10 % of full scale
the saturated and linear models differ by under 1 %, and no amount of averaging
down there will separate them.  That is why the ladder now runs to 1 rather
than stopping at 0.995, and why the closed-form ceiling drive is read twice.  A
sweep that never reaches the top returns a ``Y_sat`` whose error bar has run
into the search grid, and :func:`~calibration_module.fit.verify.fit_y_sat` says
so rather than quoting a number.

The run assumes no Y_sat
------------------------
Every prediction driven here -- ``pred_v``, the input range, the read window --
is the LINEAR model.  Fitting a saturation constant from a sweep that already
assumed one would be circular.  The linear prediction is the larger of the two,
so the input range it picks is always at least wide enough and a read can never
clip because of the choice.

The gain anchor, and what Y_sat actually means
----------------------------------------------
The fit has one free parameter.  The detector gain is pinned to ``g = 1``, and
that is a DEFINITION, not an assumption that the gain is 1.  With ``g`` pinned,
any gain error is absorbed into ``Y_sat`` and levered by roughly
``Y_sat / Y_max`` -- about 12x here.  Measured on the 0917 bench: over one night
of hourly runs a ``g = 1`` ``Y_sat`` swung 4.39..5.44 V while tracking the
independently measured gain at r = 0.98, and with the gain left free it sat at
6.84 +- 0.20 V with no trend at all.

So this run's ``Y_sat`` and this run's reads are one object.  Downstream, a
computation re-drives a handful of this sweep's own ladder patterns -- they are
in the CSV row by row, and the seed is in the JSON -- and takes the ratio of its
reads to this run's reads on the same drives.  That ratio is ``g``, free of
``Y_sat``, of ``eta`` and of the phases, because the SLM frame is identical.
The prediction downstream is then::

    Y_pred = g * Y / (1 + Y / Y_sat)

with ``Y`` the linear model and ``Y_sat`` fixed at this run's value.  Gain
outside the saturation, on physical grounds: the atoms saturate on the light
they receive, and the PMT amplifies afterwards.  The 0917 data cannot tell that
apart from ``g * Y / (1 + g Y / Y_sat)`` -- the two differ by 0.45 mV at worst
against a 4 mV floor -- so the choice is made on physics, not on fit quality.

What Y_sat cannot fix.  About 3.9 mV of the residual on the 0917 runs is
per-pattern model error that neither ``g`` nor ``Y_sat`` touches, and the gain
wanders ~1 %/h bidirectionally, worth ~1.4 mV per 1 %.  Refreshing ``g`` every
15 minutes keeps that second term at ~12 % of the first.

Input range
-----------
Every read gets the range its prediction calls for -- the smallest board range
holding it under ``RANGE_FILL`` of full -- so a dim pattern keeps the sensitive
range and a bright one does not clip.  If a trace comes back near the rail
anyway, step 6's escalation remeasures it once, wider, and keeps the wide
reading; ``range_v`` in the CSV is the range the kept read was taken at.

Alpha is coherent, and single_beam is subtracted
------------------------------------------------
``FS = Y_max`` excludes the single-beam background, so every level here is on
``|E|^2`` alone and the fit subtracts the model's background from each read.
It is small but it RAMPS -- 0.39 -> 1.13 mV bottom to top on the 0911 run -- and
a ramp coherent with level is exactly what a saturation fit is looking for, so
leaving it in would bias ``Y_sat``.  The raw ``voltage_mean_v`` and ``dark_v``
are never touched; ``sb_pred_v`` is a column and the subtraction happens in the
fit.  Input range still comes from the FULL prediction, background included:
the ADC sees the total.

Read window
-----------
A dim read has the worst signal to noise, so it gets the longest window:
``T_both`` runs linearly from ``T_BOTH_DIM_S`` at the run's dimmest prediction
to ``T_BOTH_BRIGHT_S`` at its brightest.  The dark keeps ``T_single``.

PMT temperature
---------------
``--pmt`` logs the PMT's thermistor (Keithley 2400, ohms and TH10K degrees)
next to every read.  The 2400 settings are imported from
``drafts/daq_pmt_monitor.py``, which owns them; the reading is cached for
``--pmt-period`` seconds (default 60), since the bead's time constant is
minutes and a per-read GPIB round trip would add minutes to a run.  The run
brackets itself with a reading in the CSV meta, and a 2400 that fails blanks
the two columns rather than ending the run.

Output
------
``calib_step8_patterns_<MMDD_HHMM>.json``  every pattern, written BEFORE the
hardware is touched, so an interrupted run is still reproducible.
``calib_step8_<MMDD_HHMM>.csv``            one row per read, partial if stopped.
``calib_step8_result_<MMDD_HHMM>.json``    the fitted ``Y_sat`` -- what steps
                                           downstream read.
``calib_step8_<MMDD_HHMM>_ysat.png``       the three-panel report.

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
    write_verify_csv,
)
from calibration_module.measure.bench import connect_daq, connect_slm  # noqa: E402
from calibration_module.measure.patterns import (  # noqa: E402
    DARK_EVERY_S,
    DEFAULT_ACQ,
    load_layout,
    range_for,
    run_patterns,
    select_pairs,
)
from calibration_module.measure.pmt_temp import (  # noqa: E402,F401
    PMT_PERIOD_S,
    PmtTemperature,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"   # output directory; --out overrides it

# The step-7 result is NOT here -- it is the required --step7 argument, as
# step 6's --step3 and step 7's --step6 are.

PAIRS = None                # None -> every pair the step-7 result calibrates; --p overrides
PHASE_METHOD = None         # stored step-7 fit; None -> the single one a v2 JSON has

SLM_DISPLAY_NO = None       # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1              # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

#: Windows come from T_BOTH_DIM_S / T_BOTH_BRIGHT_S per read, not from ``t_both_s``.
ACQ = DEFAULT_ACQ

# ---- The run ----
LADDER = 90                     # patterns across the scale, one per stratum
LADDER_ALPHA_MAX = 1.0          # the ladder owns the ceiling; nothing else measures it
SEED = None                     # None -> a fresh seed, printed and written to the JSON

# ---- The read ----
T_BOTH_DIM_S = 10.0             # window at the run's dimmest prediction
T_BOTH_BRIGHT_S = 5.0           # ... linear down to this at its brightest

OUT_DIR = CALIB_PATH


# ======================================================================
# building the run
# ======================================================================

def read_window(preds: Sequence[float]):
    """``t(pred)`` -- linear from T_BOTH_DIM_S at ``min(preds)`` to T_BOTH_BRIGHT_S at ``max``."""
    lo, hi = float(min(preds)), float(max(preds))

    def t_both(pred: float) -> float:
        f = 0.0 if hi <= lo else min(max((float(pred) - lo) / (hi - lo), 0.0), 1.0)
        return T_BOTH_DIM_S + f * (T_BOTH_BRIGHT_S - T_BOTH_DIM_S)
    return t_both


def run_seconds(preds: Sequence[float], t_both) -> float:
    """Estimated wall time: every window and settle, plus the periodic darks."""
    secs = sum(t_both(q) + ACQ.settle_s for q in preds)
    return secs + ACQ.t_single_s * (1 + secs // DARK_EVERY_S)


def interleave(ladder: Sequence[Pattern], ceiling: Pattern, *, seed: int) -> list[Pattern]:
    """Shuffle the ladder; bracket it with the ceiling drive at both ends.

    Shuffled, because driving the ladder in level order would put slow drift on
    the level axis -- the axis the fit reads.  Bracketed, because the same frame
    at both ends of the run is the only read here that measures drift rather
    than model error: the two differ by the gain, and by nothing else.
    """
    rng = np.random.default_rng(seed)
    middle = [ladder[i] for i in rng.permutation(len(ladder))]
    return ([replace(ceiling, label=ceiling.label + "@start")]
            + middle
            + [replace(ceiling, label=ceiling.label + "@end")])


def build(models, phases, pairs, *, seed: int, ladder: int = LADDER):
    """``(scale, patterns, preds)`` -- the whole run, solved, in read order.

    ``preds`` are LINEAR: this run fits ``Y_sat``, so it must not assume one.
    """
    scale = full_scale(models, phases, pairs)
    rungs = level_patterns(models, phases, pairs, ladder, seed=seed,
                           alpha_hi=LADDER_ALPHA_MAX)
    patterns = interleave(rungs, full_scale_pattern(models, phases, pairs),
                          seed=seed + 2)
    return scale, patterns, [float(predict(models, phases, pairs, p.x, p.w))
                             for p in patterns]


# ======================================================================
# collect
# ======================================================================

def collect(step7: Path, *, pairs=None, method=None, out_dir=None,
            ladder: int = LADDER, seed: int | None = None,
            dry_run: bool = False, pmt: "PmtTemperature | None" = None) -> str | None:
    """Solve the sweep, write its JSON, drive it, write the CSV."""
    out = Path(OUT_DIR if out_dir is None else out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = select_pairs(PAIRS if pairs is None else pairs)
    models, phases, ref = load_forward_model(step7, pairs=wanted, method=method)
    pairs = sorted(models)
    if seed is None:
        seed = SEED if SEED is not None else int(np.random.SeedSequence().entropy % 2**32)

    scale, patterns, preds = build(models, phases, pairs, seed=seed, ladder=ladder)
    coh = [float(predict(models, phases, pairs, p.x, p.w, coherent=True))
           for p in patterns]
    t_both = read_window(preds)
    secs = run_seconds(preds, t_both)

    print(f"Step 7 in : {step7}")
    print(f"Pairs     : {pairs}  (N = {len(pairs)}, reference {ref})")
    print(f"Full scale: Y_max {scale.fs * 1e3:.4f} mV   S^2 {scale.bound * 1e3:.4f} mV "
          f"({100 * scale.headroom:.1f} %)   psi* {np.degrees(scale.psi):.2f} deg")
    print(f"            4-bit level step = {scale.fs / 15 * 1e3:.3f} mV (linear)")
    print(f"Model     : LINEAR -- this run fits Y_sat, so it assumes none")
    print(f"Read      : T_both {T_BOTH_DIM_S:g} s (dimmest, {min(preds) * 1e3:.2f} mV) "
          f"-> {T_BOTH_BRIGHT_S:g} s (brightest, {max(preds) * 1e3:.2f} mV)")
    print(f"Patterns  : {len(patterns)} reads  (ladder {ladder}, ceiling 2), "
          f"~{secs / 60:.0f} min")
    print(f"Seed      : {seed}")
    fell_back = [p.label for p in patterns if not p.exact]
    if fell_back:
        print(f"            {len(fell_back)} pattern(s) fell back to the "
              f"deterministic drive: {', '.join(fell_back[:6])}"
              + (" ..." if len(fell_back) > 6 else ""))

    stamp = time.strftime("%m%d_%H%M")
    json_path = out / f"calib_step8_patterns_{stamp}.json"
    json_path.write_text(json.dumps({
        "step7_json": str(Path(step7).resolve()),
        "pairs": pairs, "reference": ref, "method": method, "seed": seed,
        "Y_max_v": scale.fs, "S2_v": scale.bound, "psi_rad": scale.psi,
        "level_step_v": scale.fs / 15.0,
        "eta": dict(zip(map(str, pairs), scale.amplitudes)),
        "phi_rad": dict(zip(map(str, pairs), scale.phases)),
        "config": {"ladder": ladder, "ladder_alpha_max": LADDER_ALPHA_MAX,
                   "t_both_dim_s": T_BOTH_DIM_S, "t_both_bright_s": T_BOTH_BRIGHT_S},
        "patterns": [
            {"seq": i, "kind": p.kind, "label": p.label,
             "target_alpha": p.target_alpha, "alpha": p.alpha,
             "coherent_v": c, "pred_v": q, "t_both_s": t_both(q), "exact": p.exact,
             "x": dict(zip(map(str, pairs), map(float, p.x))),
             "w": dict(zip(map(str, pairs), map(float, p.w)))}
            for i, (p, c, q) in enumerate(zip(patterns, coh, preds))
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nPatterns written to {json_path}")

    if dry_run:
        report_plan(patterns, preds, coh, scale.fs, t_both)
        return None

    return _drive(step7, pairs, patterns, preds, coh, models, phases,
                  out=out, stamp=stamp, json_path=json_path,
                  meta={"n_pairs": len(pairs), "seed": seed, "y_max_v": scale.fs},
                  pmt=pmt)


def _drive(step7, pairs, patterns, preds, coh, models, phases, *,
           out: Path, stamp: str, json_path: Path, meta: dict,
           columns: Sequence[str] = (), extra_of=None,
           pmt: "PmtTemperature | None" = None):
    """Display every pattern in read order and write the CSV, partial or not.

    ``extra_of(i, pattern)`` adds per-read columns beyond the standard ones;
    ``columns`` names them, in CSV order.  Each read's window comes from
    :func:`read_window` over ``preds``.

    ``pmt``, if given, adds its two columns to every read and brackets the run
    with a reading in the meta.
    """
    x = np.stack([p.x for p in patterns])[None]      # (1 block, P, n)
    w = np.stack([p.w for p in patterns])[None]
    t_both = read_window(preds)
    extra = [[{"kind": p.kind, "label": p.label, "seq": i,
               "target_alpha": f"{p.target_alpha:.6f}",
               "alpha": f"{p.alpha:.6f}",
               "coherent_v": f"{c:.9g}", "sb_pred_v": f"{q - c:.9g}",
               "t_both_s": f"{t_both(q):.3f}",
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
                      t_both=ACQ.t_both_s, t_single=ACQ.t_single_s,
                      min_val=-ACQ.range_v, max_val=ACQ.range_v)
    layout = load_layout(step7, pairs)
    rows: list[dict] = []
    status = "complete"
    csv_path = None
    try:
        run_patterns(daq, slm, layout, [tuple(pairs)], x, w, models, phases,
                     rows=rows, extra=extra, y_sat=None, t_both_of=t_both,
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
                    meta={"step7_json": Path(step7).resolve(),
                          "patterns_json": json_path, "status": status, **meta},
                    extra=["kind", "label", "seq", "target_alpha", "alpha",
                           "coherent_v", "sb_pred_v", "t_both_s", *columns])
                print(f"\nCSV ({len(rows)} reads, {status}) written to {csv_path}")
        finally:
            slm.close_slm()
            daq.disconnect()
    return csv_path


# ======================================================================
# the fit
# ======================================================================

def fit_and_write(csv_path, step7, *, pairs=None, method=None, out_dir=None,
                  log=print) -> dict:
    """Fit ``Y_sat`` on a step-8 CSV, write the result JSON and the PNG.

    The result JSON is what everything downstream reads.  It carries the CSV it
    came from, because a ``Y_sat`` and the reads that defined ``g = 1`` are one
    object: a later run measures its gain against THESE reads, so quoting the
    constant without saying which sweep produced it is how the two get mixed.
    """
    from calibration_module.fit.step8_report import report_csv

    csv_path = Path(csv_path)
    out = Path(csv_path.parent if out_dir is None else out_dir)
    result = report_csv(csv_path, step7, pairs=pairs, method=method,
                        out_dir=out, log=log)

    stamp = csv_path.stem.replace("calib_step8_", "")
    result_path = out / f"calib_step8_result_{stamp}.json"
    result_path.write_text(json.dumps({
        "step7_json": str(Path(step7).resolve()),
        "csv": str(csv_path.resolve()),
        "y_sat_v": result["y_sat_v"],
        "y_sat_sigma_v": result["sigma_v"],
        "y_sat_lo_v": result["lo_v"], "y_sat_hi_v": result["hi_v"],
        "unbounded": result["unbounded"],
        "gain_lever": result["gain_lever"],
        "rmse_v": result["rmse_v"], "rmse_linear_v": result["rmse_linear_v"],
        "n_reads": result["n"], "n_ladder": result["n_ladder"],
        "png": result.get("png"),
        "full_scale": result.get("full_scale", {}),
        # g = 1 is DEFINED by the reads in the CSV above.  A later run divides
        # its reads on a subset of those same drives by them to get its gain.
        "gain_reference": {"convention": "g = 1 at this sweep",
                           "csv": str(csv_path.resolve())},
    }, indent=2), encoding="utf-8")
    log(f"Result written to {result_path}")
    return result


def report_plan(patterns, preds, coh, fs: float, t_both) -> None:
    """The dry-run table: what will be driven, in read order, on which range, how long."""
    step = fs / 15.0
    print(f"\n  {'seq':>4} {'kind':>4} {'label':<22} {'alpha':>7} "
          f"{'coh (mV)':>10} {'pred (mV)':>10} {'sb (mV)':>8} {'range':>7} {'T (s)':>6}")
    for i, (p, c, q) in enumerate(zip(patterns, coh, preds)):
        rng = range_for(q)[0]
        print(f"  {i:4d} {p.kind:>4} {p.label:<22} {p.alpha:7.4f} "
              f"{c * 1e3:10.4f} {q * 1e3:10.4f} {(q - c) * 1e3:8.4f} "
              f"{rng:6.1f}V {t_both(q):6.2f}" + ("" if p.exact else "  [fallback]"))

    a = np.array([p.alpha for p in patterns if p.kind == "A"])
    used = np.array([range_for(q)[0] for p, q in zip(patterns, preds)
                     if p.kind == "A"])
    sb = np.array([q - c for p, q, c in zip(patterns, preds, coh) if p.kind == "A"])
    # every read, not just the ladder: the ceiling is what occupies level 15
    every = np.array([p.alpha for p in patterns])
    # +eps so the ceiling's alpha, 1 to within rounding, lands in level 15
    hist = np.bincount(np.clip((every * 15 + 1e-9).astype(int), 0, 15), minlength=16)
    print(f"\n  ladder alpha : min {a.min():.4f}  median {np.median(a):.4f}  "
          f"max {a.max():.4f}   (the ceiling reads sit at exactly 1)")
    print(f"  4-bit levels : {' '.join(str(c) for c in hist)}   "
          f"({int((hist > 0).sum())} of 16 occupied)")
    print(f"  single_beam  : {sb.min() * 1e3:.3f} .. {sb.max() * 1e3:.3f} mV  "
          f"({100 * (sb.max() - sb.min()) / step:.1f} % of a level step)")
    print("  DAQ ranges   : " + ", ".join(
        f"{r:g}V x{n}" for r, n in zip(*np.unique(used, return_counts=True))))
    top = max(preds)
    print(f"\n  the fit lives at the top: at alpha = 1 ({top * 1e3:.1f} mV) a "
          f"Y_sat of 5 V bends\n  the reading down by "
          f"{100 * top / (5.0 + top):.1f} %, at alpha = 0.1 by only "
          f"{100 * 0.1 * top / (5.0 + 0.1 * top):.1f} %")


# ======================================================================
# CLI
# ======================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calib_step8.py",
        description="Step 8: fit the two-photon saturation constant Y_sat from "
                    "one level sweep, and anchor g = 1 for everything "
                    "downstream.",
        epilog="--step7 has no default: that one file is the whole forward model "
               "(its embedded step 3 is the layout, step 6 the etas, step 7 the "
               "phases), so running against the wrong one reports a saturation "
               "constant that is really a file mix-up.",
    )
    parser.add_argument("--step7", required=True, type=Path, metavar="JSON",
                        help="combined step-7 result JSON (required)")
    parser.add_argument("csv", nargs="?", type=Path,
                        help="re-fit this recorded CSV offline instead of collecting")
    parser.add_argument("--p", nargs="+", type=int, default=None, metavar="PAIR",
                        help="only drive these pairs (default: every calibrated pair)")
    parser.add_argument("--ladder", type=int, default=LADDER, metavar="N",
                        help=f"patterns across the scale (default {LADDER})")
    parser.add_argument("--method", default=PHASE_METHOD,
                        help="stored step-7 fit to predict from")
    parser.add_argument("--seed", type=int, default=None,
                        help="reproduce a previous run's patterns")
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help=f"directory for the JSON, CSV and PNGs, created if "
                             f"missing (default {OUT_DIR})")
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
                        help="solve the patterns, write the JSON, print the plan; "
                             "touch no hardware")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        wanted = select_pairs(args.p)
    except ValueError as exc:
        parser.error(str(exc))

    if args.csv is not None:
        fit_and_write(args.csv, args.step7, pairs=wanted, method=args.method,
                      out_dir=args.out)
        return 0

    pmt = None
    if args.pmt is not None and not args.dry_run:
        try:
            pmt = PmtTemperature(args.pmt or None, period_s=args.pmt_period)
        except Exception as exc:        # noqa: BLE001
            # A missing VISA stack must not cost the run the sweep.
            print(f"--pmt ignored ({type(exc).__name__}: {exc})")

    csv_path = collect(args.step7, pairs=wanted, method=args.method,
                       ladder=args.ladder, out_dir=args.out, seed=args.seed,
                       dry_run=args.dry_run, pmt=pmt)
    if csv_path is not None:
        fit_and_write(csv_path, args.step7, method=args.method, out_dir=args.out)
        print(f"\nRe-fit with:  python {Path(__file__).name} "
              f"--step7 {args.step7} {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
