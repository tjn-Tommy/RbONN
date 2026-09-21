"""System accuracy and stability -- what the whole machine can actually compute.

    python src/calibration_module/steps/system_accuracy.py --step7 RESULT.json --dry-run
    python src/calibration_module/steps/system_accuracy.py --step7 RESULT.json
    python src/calibration_module/steps/system_accuracy.py --step7 RESULT.json some.csv
    python src/calibration_module/steps/system_accuracy.py --step7 RESULT.json --loo

Step 8 answers "does the model predict this pattern".  This answers the question
a 4-bit computation actually asks: **over the whole output scale, can the machine
put a value where it is told, and read back the same value twice?**  It fits
nothing.

Why step 8 cannot answer it.  Step 8 drives ``x_k, w_k`` uniform in a box; with
ten pairs that is a random phasor sum, so ``|E|^2`` concentrates near ``Y_max/10``.
The 0911 ten-pair run covered 4..24 % of full scale, seven of its eight patterns
under 10 % -- levels 0..3 of 15, essentially all on level 1.  Uniform in the
drive is nowhere near uniform in the output.

The three families
------------------
``A``  the ladder.  One pattern per stratum of ``[0, Y_max]``, each a fresh random
       drive found by least-move projection onto that level set (see
       :mod:`calibration_module.fit.levels`).  Every drive differs, every
       prediction differs.  Measures MODEL error against level -- and binning A
       by 4-bit level afterwards also answers "do different drives at one level
       read the same", at no extra cost.

``B``  the repeats.  ONE drive per level, displayed and read several times,
       spread through the run.  Measures READOUT reproducibility against level,
       which is the 4-bit noise budget.  A changes the drive every time and tests
       the model; B never changes it and tests the machine.  From A alone the two
       are not separable -- a 10 mV error could be either, and the fix differs.

``C``  the psi scan.  The support-optimal drive for a fan of projection axes
       around ``psi*``.  Its peak is full scale MEASURED rather than predicted,
       and the peak's displacement from offset 0 is the model's global phase
       error.  Read at the start and again at the end, so it doubles as a drift
       bracket.

``--loo``: which pair is dragging the machine down
--------------------------------------------------
Family A alone, driven on the full pool AND on the pool minus each pair in
turn, all shuffled into ONE interleaved run.  Three things make it an answer
rather than a pile of numbers:

* **Every subset is driven, not just the suspect one.**  The nine other
  drop-one subsets are the null distribution for the tenth.  A lone
  drop-pair-1 run has nothing to be an outlier against, and its own 5 %
  improvement is what removing any one of ten error sources buys.
* **Interleaved.**  Eleven ladders back to back is two hours end to end, and
  any drift over that window would land on the subset axis -- the axis being
  measured.  Shuffled, drift is common mode; the C bracket still measures it.
* **Scored in level steps, never mV.**  Each subset has its own ``Y_max``, so
  mV are on a different scale in every row.  Dividing by the subset's own step
  fixes that but introduces the opposite bias -- dropping a bright pair shrinks
  the step -- which is why the report plots rms against ``Y_max`` too: if the
  ranking follows the ``1/Y_max`` curve it is full scale, not calibration.

One block of ``len(pool)`` slots throughout: an excluded pair is driven at
``x = w = 0``, which contributes no field and no background and displays the
same frame as omitting it.

Input range
-----------
Every read gets the range its PREDICTION calls for -- step 8's ``range_for``,
the smallest board range holding the prediction under ``RANGE_FILL`` of it --
so a dim pattern keeps the sensitive range and a bright one does not clip.  The
ladder spans 5 to 320 mV, so it uses 0.1, 0.2 and 0.5 V.  If the trace comes
back near the rail anyway, step 6's escalation remeasures it once, wider, and
keeps the wide reading; ``range_v`` in the CSV is the range the kept read was
taken at.

Alpha is coherent, and single_beam is subtracted
------------------------------------------------
``FS = Y_max`` excludes the single-beam background, so every level here is on
``|E|^2`` alone and the report subtracts the model's background from each read.
It is small but it RAMPS -- 0.39 -> 1.13 mV bottom to top on the 0911 run, 3.4 %
of a 4-bit level step -- and a ramp coherent with level is exactly what A is
looking for, so leaving it in would tilt the answer.  The raw ``voltage_mean_v``
and ``dark_v`` are never touched; ``sb_pred_v`` is a column and the subtraction
happens in the report.

Input range still comes from the FULL prediction, background included: the ADC
sees the total.

Saturation
----------
The atoms saturate: the readout is ``|E|^2 / (1 + |E|^2 / Y_sat)``, not
``|E|^2`` (:func:`verify.saturate`).  Steps 6 and 7 never read above ~66 mV, so
they cannot see it; the 0916 ten-pair run, which went to 500 mV, read 8 % low at
the top and fitted ``Y_sat = 4.82 +- 0.07 V`` (all 168 reads, no free gain),
cutting the rms error from 26 to 4 mV.  Every prediction here -- ``pred_v``, the
range choice, and the report's levels and full scale -- uses ``Y_SAT_V``
(``--y-sat``, ``0`` for the linear model).  The patterns are still SOLVED on
the coherent ``|E|^2``: saturation is monotonic, so it moves no drive, and A's
strata shift by under 0.5 % of scale.

Read window
-----------
A dim read has the worst signal to noise, so it gets the longest window:
``T_both`` runs linearly from ``T_BOTH_DIM_S`` at the run's dimmest prediction
to ``T_BOTH_BRIGHT_S`` at its brightest.  The dark keeps ``T_single``.

Order.  Reads are shuffled and B is spread through the run, because running the
ladder in level order would alias slow drift straight onto the level axis -- the
one axis this is trying to measure.

PMT temperature
---------------
``--pmt`` logs the PMT's thermistor (Keithley 2400, ohms and TH10K degrees)
next to every read, so a night of hourly runs can be plotted against the
detector's own temperature rather than against the room's.  The 2400 settings
are imported from ``drafts/daq_pmt_monitor.py``, which owns them; the reading
is cached for ``--pmt-period`` seconds (default 60), since the bead's time
constant is minutes and a per-read GPIB round trip would add minutes to a run.
The run brackets itself with a reading in the CSV meta, and a 2400 that fails
blanks the two columns rather than ending the run.

Output: ``system_accuracy_<MMDD_HHMM>.json`` (every pattern, written BEFORE the
hardware is touched, so an interrupted run is still reproducible) and
``.csv`` beside it, partial if the run is stopped.

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
    PSI_OFFSETS_DEG,
    Pattern,
    level_patterns,
    loo_patterns,
    psi_scan_patterns,
    repeat_patterns,
    subset_tag,
)
from calibration_module.fit.verify import (  # noqa: E402
    full_scale,
    load_forward_model,
    pairs_tag,
    predict,
    saturate,
    write_verify_csv,
)
from calibration_module.measure.pmt_temp import (  # noqa: E402,F401
    PMT_PERIOD_S,
    PmtTemperature,
)
from calibration_module.steps import calib_npair_verify as S8  # noqa: E402

# ---- The run ----
LADDER = 120                    # A: patterns across the scale, one per stratum
LADDER_ALPHA_MAX = 0.995        # A stops short of 1; the top belongs to C
REPEAT_ALPHAS = (0.1, 0.3, 0.5, 0.75, 1.0)      # B: levels to repeat
REPEATS = 6                     # B: reads per level
LOO_LADDER = 60                 # --loo: A patterns PER SUBSET (11 subsets on a 10 pool)
SEED = None                     # None -> a fresh seed, printed and written to the JSON

# ---- The model ----
Y_SAT_V = 4.82                  # two-photon saturation, fitted on the 0916 10-pair run

# ---- The read ----
T_BOTH_DIM_S = 10.0             # window at the run's dimmest prediction
T_BOTH_BRIGHT_S = 5.0           # ... linear down to this at its brightest

# ---- The PMT thermistor (--pmt) ----
# Both live in calibration_module.measure.pmt_temp so steps 6 and 7 stamp their
# rows from the same implementation; re-exported here because --pmt, the tests
# and every caller that says SA.PmtTemperature predate the move.

OUT_DIR = S8.OUT_DIR


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
    secs = sum(t_both(q) + S8.ACQ.settle_s for q in preds)
    return secs + S8.ACQ.t_single_s * (1 + secs // S8.DARK_EVERY_S)


def interleave(ladder, repeats, scan, *, seed: int) -> list[Pattern]:
    """Order the reads so drift cannot masquerade as a level effect.

    The scan brackets the run at both ends, contiguous -- it is a scan, so it
    wants to be fast relative to drift.  The ladder is shuffled, so level and
    read order are independent.  Each repeat of a B level is placed at an even fraction of the run,
    with the levels offset slightly from each other so they do not all land on the
    same read.
    """
    rng = np.random.default_rng(seed)
    middle = list(ladder)
    order = rng.permutation(len(middle))
    timed = [(float(pos) / len(middle), middle[i]) for pos, i in enumerate(order)]

    levels = sorted({p.target_alpha for p in repeats})
    for p in repeats:
        # (repeat_id - 0.5)/REPEATS walks 0..1 through the run; the per-level
        # nudge keeps five levels from stacking on the same instant
        spread = (levels.index(p.target_alpha) - (len(levels) - 1) / 2) * 0.01
        timed.append(((p.repeat_id - 0.5) / REPEATS + spread, p))

    timed.sort(key=lambda t: t[0])
    head = [Pattern(**{**p.__dict__, "label": p.label + "@start"}) for p in scan]
    tail = [Pattern(**{**p.__dict__, "label": p.label + "@end"}) for p in scan]
    return head + [p for _, p in timed] + tail


def build(models, phases, pairs, *, seed: int, ladder: int = LADDER,
          y_sat: float | None = Y_SAT_V):
    """``(scale, patterns, preds)`` -- the whole run, solved, in read order."""
    scale = full_scale(models, phases, pairs)
    rungs = level_patterns(models, phases, pairs, ladder, seed=seed,
                           alpha_hi=LADDER_ALPHA_MAX)
    patterns = interleave(rungs,
                          repeat_patterns(models, phases, pairs, REPEAT_ALPHAS,
                                          REPEATS, seed=seed + 1),
                          psi_scan_patterns(models, phases, pairs),
                          seed=seed + 2)
    return scale, patterns, [float(predict(models, phases, pairs, p.x, p.w, y_sat=y_sat))
                             for p in patterns]


# ======================================================================
# collect
# ======================================================================

def collect(step7: Path, *, pairs=None, method=None, out_dir=None, ladder: int = LADDER,
            seed: int | None = None, y_sat: float | None = Y_SAT_V,
            dry_run: bool = False, pmt: "PmtTemperature | None" = None) -> str | None:
    """Solve the run, write its JSON, drive it, write the CSV."""
    out = Path(OUT_DIR if out_dir is None else out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = S8.select_pairs(S8.PAIRS if pairs is None else pairs)
    models, phases, ref = load_forward_model(step7, pairs=wanted, method=method)
    pairs = sorted(models)
    if seed is None:
        seed = SEED if SEED is not None else int(np.random.SeedSequence().entropy % 2**32)

    scale, patterns, preds = build(models, phases, pairs, seed=seed, ladder=ladder,
                                   y_sat=y_sat)
    coh = [float(predict(models, phases, pairs, p.x, p.w, coherent=True, y_sat=y_sat))
           for p in patterns]
    counts = {k: sum(1 for p in patterns if p.kind == k) for k in "ABCD"}
    t_both = read_window(preds)
    secs = run_seconds(preds, t_both)
    fs_sat = float(saturate(scale.fs, y_sat))

    print(f"Step 7 in : {step7}")
    print(f"Pairs     : {pairs}  (N = {len(pairs)}, reference {ref})")
    print(f"Full scale: Y_max {scale.fs * 1e3:.4f} mV   S^2 {scale.bound * 1e3:.4f} mV "
          f"({100 * scale.headroom:.1f} %)   psi* {np.degrees(scale.psi):.2f} deg")
    print(f"Saturation: Y_sat {y_sat or float('inf'):g} V -> readout full scale "
          f"{fs_sat * 1e3:.4f} mV ({100 * (fs_sat / scale.fs - 1):+.1f} %)")
    print(f"            4-bit level step = {fs_sat / 15 * 1e3:.3f} mV")
    print(f"Read      : T_both {T_BOTH_DIM_S:g} s (dimmest, {min(preds) * 1e3:.2f} mV) "
          f"-> {T_BOTH_BRIGHT_S:g} s (brightest, {max(preds) * 1e3:.2f} mV)")
    if scale.headroom > 0.9:
        print("            NOTE: headroom near 1 -- the phases are clustered, so the "
              "C scan will be one-sided and cannot locate a peak")
    print(f"Patterns  : {len(patterns)} reads  "
          f"(A ladder {counts['A']}, B repeats {counts['B']}, "
          f"C scan {counts['C']}), "
          f"~{secs / 60:.0f} min")
    print(f"Seed      : {seed}")
    fell_back = [p.label for p in patterns if not p.exact]
    if fell_back:
        print(f"            {len(fell_back)} pattern(s) fell back to the "
              f"deterministic drive: {', '.join(fell_back[:6])}"
              + (" ..." if len(fell_back) > 6 else ""))

    stamp = time.strftime("%m%d_%H%M")
    json_path = out / f"system_accuracy_{stamp}.json"
    json_path.write_text(json.dumps({
        "step7_json": str(Path(step7).resolve()),
        "pairs": pairs, "reference": ref, "method": method, "seed": seed,
        "Y_max_v": scale.fs, "S2_v": scale.bound, "psi_rad": scale.psi,
        "y_sat_v": y_sat, "Y_max_sat_v": fs_sat,
        "level_step_v": fs_sat / 15.0,
        "eta": dict(zip(map(str, pairs), scale.amplitudes)),
        "phi_rad": dict(zip(map(str, pairs), scale.phases)),
        "config": {"ladder": ladder, "ladder_alpha_max": LADDER_ALPHA_MAX,
                   "repeat_alphas": list(REPEAT_ALPHAS), "repeats": REPEATS,
                   "psi_offsets_deg": list(PSI_OFFSETS_DEG),
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
        report_plan(patterns, preds, coh, fs_sat, t_both)
        return None

    return _drive(step7, pairs, patterns, preds, coh, models, phases,
                  out=out, stamp=stamp, json_path=json_path, y_sat=y_sat,
                  meta={"n_pairs": len(pairs), "seed": seed,
                        "y_max_v": scale.fs, "y_sat_v": y_sat},
                  pmt=pmt)


def _drive(step7, pairs, patterns, preds, coh, models, phases, *,
           out: Path, stamp: str, json_path: Path, meta: dict,
           y_sat: float | None, columns: Sequence[str] = (), extra_of=None,
           pmt: "PmtTemperature | None" = None):
    """Display every pattern in read order and write the CSV, partial or not.

    ``extra_of(i, pattern)`` adds per-read columns beyond the standard ones;
    ``columns`` names them, in CSV order.  Each read's window comes from
    :func:`read_window` over ``preds``.

    ``pmt``, if given, adds its two columns to every read and brackets the run
    with a reading in the meta, so a night of runs can be plotted against PMT
    temperature.
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

    slm = S8.connect_slm(S8.SLM_DISPLAY_NO, S8.USB_SLM_NO)
    daq = S8.connect_daq(device=S8.DAQ_DEVICE, channel=S8.DAQ_CHANNEL,
                         t_both=S8.ACQ.t_both_s, t_single=S8.ACQ.t_single_s,
                         min_val=-S8.ACQ.range_v, max_val=S8.ACQ.range_v)
    layout = S8.load_layout(step7, pairs)
    rows: list[dict] = []
    status = "complete"
    csv_path = None
    try:
        S8.run_patterns(daq, slm, layout, [tuple(pairs)], x, w, models, phases,
                        rows=rows, extra=extra, y_sat=y_sat, t_both_of=t_both,
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
                    out / f"system_accuracy_{stamp}.csv", pairs, rows,
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
# collect: the leave-one-out sweep
# ======================================================================

def collect_loo(step7: Path, *, pairs=None, ladder: int = LOO_LADDER,
                method=None, out_dir=None, seed: int | None = None,
                bracket: bool = True, y_sat: float | None = Y_SAT_V,
                dry_run: bool = False,
                pmt: "PmtTemperature | None" = None) -> str | None:
    """Family A on the full pool and on the pool minus each pair, interleaved.

    One block, one CSV: every read drives all ``len(pool)`` slots, the excluded
    pair at ``x = w = 0`` (see :func:`levels.pad_to_pool`), so the subsets can be
    shuffled into a single read order.  That is the point -- eleven ladders run
    back to back would put two hours between the first subset and the last, and
    any drift over that window would land on the subset axis, which is the axis
    being measured.

    ``bracket`` keeps the C scan on the FULL pool at both ends, 18 reads, as the
    drift measurement the interleave is protecting against.
    """
    out = Path(OUT_DIR if out_dir is None else out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = S8.select_pairs(S8.PAIRS if pairs is None else pairs)
    models, phases, ref = load_forward_model(step7, pairs=wanted, method=method)
    pool = sorted(models)
    if seed is None:
        seed = SEED if SEED is not None else int(np.random.SeedSequence().entropy % 2**32)

    subs, scales, flat = loo_patterns(models, phases, pool, int(ladder), seed=seed,
                                      alpha_hi=LADDER_ALPHA_MAX)
    rng = np.random.default_rng(seed + 3)
    patterns = [flat[i] for i in rng.permutation(len(flat))]
    if bracket:
        scan = psi_scan_patterns(models, phases, pool)
        patterns = ([replace(p, label=p.label + "@start") for p in scan]
                    + patterns
                    + [replace(p, label=p.label + "@end") for p in scan])
    tag_of = {id(p): p.label.split("/")[1] if p.kind == "A" else "all"
              for p in patterns}

    preds = [float(predict(models, phases, pool, p.x, p.w, y_sat=y_sat)) for p in patterns]
    coh = [float(predict(models, phases, pool, p.x, p.w, coherent=True, y_sat=y_sat))
           for p in patterns]
    t_both = read_window(preds)
    secs = run_seconds(preds, t_both)

    full = scales["all"]
    print(f"Step 7 in : {step7}")
    print(f"Pool      : {pool}  (N = {len(pool)}, reference {ref})")
    print(f"Subsets   : {len(subs)}  x {ladder} A patterns"
          + (f" + {2 * len(PSI_OFFSETS_DEG)} C bracket" if bracket else ""))
    print(f"Patterns  : {len(patterns)} reads, ~{secs / 60:.0f} min")
    print(f"Seed      : {seed}")
    print(f"\n  {'subset':<10} {'N':>3} {'Y_max (mV)':>11} {'step (mV)':>10} "
          f"{'vs all':>8}")
    for sub in subs:
        tag = subset_tag(sub, pool)
        sc = scales[tag]
        print(f"  {tag:<10} {len(sub):3d} {sc.fs * 1e3:11.4f} "
              f"{sc.fs / 15 * 1e3:10.4f} {100 * (sc.fs / full.fs - 1):+7.1f}%")
    print("  a subset's residuals are only comparable divided by ITS level step; "
          "dropping\n  a bright pair shrinks the step, which inflates the error "
          "in level units on its own")
    fell_back = [p.label for p in patterns if not p.exact]
    if fell_back:
        print(f"\n            {len(fell_back)} pattern(s) fell back to the "
              f"deterministic drive")

    stamp = time.strftime("%m%d_%H%M")
    json_path = out / f"system_accuracy_loo_{stamp}.json"
    json_path.write_text(json.dumps({
        "mode": "loo", "step7_json": str(Path(step7).resolve()),
        "pairs": pool, "reference": ref, "method": method, "seed": seed,
        "y_sat_v": y_sat,
        "config": {"ladder": ladder, "ladder_alpha_max": LADDER_ALPHA_MAX,
                   "bracket": bracket,
                   "psi_offsets_deg": list(PSI_OFFSETS_DEG)},
        "subsets": {subset_tag(s, pool): {
            "pairs": list(s), "Y_max_v": scales[subset_tag(s, pool)].fs,
            "level_step_v": scales[subset_tag(s, pool)].fs / 15.0,
            "S2_v": scales[subset_tag(s, pool)].bound,
            "psi_rad": scales[subset_tag(s, pool)].psi} for s in subs},
        "patterns": [
            {"seq": i, "kind": p.kind, "label": p.label,
             "subset": tag_of[id(p)],
             "target_alpha": p.target_alpha, "alpha": p.alpha,
             "coherent_v": c, "pred_v": q, "t_both_s": t_both(q), "exact": p.exact,
             "x": dict(zip(map(str, pool), map(float, p.x))),
             "w": dict(zip(map(str, pool), map(float, p.w)))}
            for i, (p, c, q) in enumerate(zip(patterns, coh, preds))
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nPatterns written to {json_path}")

    if dry_run:
        report_plan(patterns, preds, coh, float(saturate(full.fs, y_sat)), t_both)
        return None

    return _drive(step7, pool, patterns, preds, coh, models, phases,
                  out=out, stamp=f"loo_{stamp}", json_path=json_path, y_sat=y_sat,
                  meta={"n_pairs": len(pool), "seed": seed, "mode": "loo",
                        "ladder": ladder, "y_max_v": full.fs, "y_sat_v": y_sat},
                  columns=["subset"],
                  extra_of=lambda i, p: {"subset": tag_of[id(p)]},
                  pmt=pmt)


def report_plan(patterns, preds, coh, fs: float, t_both) -> None:
    """The dry-run table: what will be driven, in read order, on which range, how long."""
    step = fs / 15.0
    print(f"\n  {'seq':>4} {'kind':>4} {'label':<22} {'alpha':>7} "
          f"{'coh (mV)':>10} {'pred (mV)':>10} {'sb (mV)':>8} {'range':>7} {'T (s)':>6}")
    for i, (p, c, q) in enumerate(zip(patterns, coh, preds)):
        rng = S8.range_for(q)[0]
        print(f"  {i:4d} {p.kind:>4} {p.label:<22} {p.alpha:7.4f} "
              f"{c * 1e3:10.4f} {q * 1e3:10.4f} {(q - c) * 1e3:8.4f} "
              f"{rng:6.1f}V {t_both(q):6.2f}" + ("" if p.exact else "  [fallback]"))

    a = np.array([p.alpha for p in patterns if p.kind == "A"])
    used = np.array([S8.range_for(q)[0] for p, q in zip(patterns, preds)
                     if p.kind == "A"])
    sb = np.array([q - c for p, q, c in zip(patterns, preds, coh) if p.kind == "A"])
    hist = np.bincount(np.clip((a * 15).astype(int), 0, 15), minlength=16)
    print(f"\n  ladder alpha : min {a.min():.4f}  median {np.median(a):.4f}  "
          f"max {a.max():.4f}")
    print(f"  4-bit levels : {' '.join(str(c) for c in hist)}   "
          f"({int((hist > 0).sum())} of 16 occupied)")
    print(f"  single_beam  : {sb.min() * 1e3:.3f} .. {sb.max() * 1e3:.3f} mV  "
          f"({100 * (sb.max() - sb.min()) / step:.1f} % of a level step)")
    print("  DAQ ranges   : " + ", ".join(
        f"{r:g}V x{n}" for r, n in zip(*np.unique(used, return_counts=True))))


# ======================================================================
# CLI
# ======================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="system_accuracy.py",
        description="System accuracy and stability: model error, readout "
                    "reproducibility and measured full scale, across the whole "
                    "output range.",
        epilog="--step7 has no default: that one file is the whole forward model "
               "(its embedded step 3 is the layout, step 6 the etas, step 7 the "
               "phases), so running against the wrong one reports a model error "
               "that is really a file mix-up.",
    )
    parser.add_argument("--step7", required=True, type=Path, metavar="JSON",
                        help="combined step-7 result JSON (required)")
    parser.add_argument("csv", nargs="?", type=Path,
                        help="re-report this recorded CSV offline instead of collecting")
    parser.add_argument("--p", nargs="+", type=int, default=None, metavar="PAIR",
                        help="only drive these pairs (default: every calibrated pair)")
    parser.add_argument("--loo", action="store_true",
                        help="leave-one-out sweep: family A on the full pool AND "
                             "on the pool minus each pair in turn, all shuffled "
                             "into one interleaved run, so drift cannot land on "
                             "the subset axis")
    parser.add_argument("--ladder", type=int, default=None, metavar="N",
                        help=f"A patterns per subset (default {LADDER}, or "
                             f"{LOO_LADDER} with --loo)")
    parser.add_argument("--no-bracket", dest="bracket", action="store_false",
                        help="--loo only: skip the C scan at both ends")
    parser.add_argument("--method", default=S8.PHASE_METHOD,
                        help="stored step-7 fit to predict from")
    parser.add_argument("--seed", type=int, default=None,
                        help="reproduce a previous run's patterns")
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help=f"directory for the JSON, CSV and PNGs, created if "
                             f"missing (default {OUT_DIR})")
    parser.add_argument("--y-sat", type=float, default=Y_SAT_V, metavar="V",
                        help=f"two-photon saturation Y_sat in volts (default "
                             f"{Y_SAT_V:g}; 0 for the linear model)")
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
        wanted = S8.select_pairs(args.p)
    except ValueError as exc:
        parser.error(str(exc))

    pmt = None
    if args.pmt is not None and not args.dry_run:
        try:
            pmt = PmtTemperature(args.pmt or None, period_s=args.pmt_period)
        except Exception as exc:        # noqa: BLE001
            # A missing VISA stack must not cost the run the accuracy data.
            print(f"--pmt ignored ({type(exc).__name__}: {exc})")

    loo = args.loo or (args.csv is not None and "_loo_" in Path(args.csv).name)
    if args.csv is not None:
        from calibration_module.fit.levels_report import report_csv, report_loo_csv
        (report_loo_csv if loo else report_csv)(
            args.csv, args.step7, pairs=wanted, method=args.method,
            out_dir=args.out, y_sat=args.y_sat)
        return 0

    if loo:
        csv_path = collect_loo(args.step7, pairs=wanted,
                               ladder=LOO_LADDER if args.ladder is None else args.ladder,
                               method=args.method, out_dir=args.out,
                               seed=args.seed, bracket=args.bracket,
                               y_sat=args.y_sat, dry_run=args.dry_run, pmt=pmt)
    else:
        csv_path = collect(args.step7, pairs=wanted, method=args.method,
                           ladder=LADDER if args.ladder is None else args.ladder,
                           out_dir=args.out, seed=args.seed, y_sat=args.y_sat,
                           dry_run=args.dry_run, pmt=pmt)
    if csv_path is not None:
        print(f"Re-report with:  python {Path(__file__).name} "
              f"--step7 {args.step7} {csv_path}")
        from calibration_module.fit.levels_report import report_csv, report_loo_csv
        (report_loo_csv if loo else report_csv)(
            csv_path, args.step7, method=args.method, out_dir=args.out,
            y_sat=args.y_sat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
