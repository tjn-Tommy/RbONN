"""Draft: acquire the ladder that measures how many bits the bench really has.

    python src/drafts/bit_acc_chara.py --calib RUN --dry-run   # solve it, no hardware
    python src/drafts/bit_acc_chara.py --calib RUN             # the ~45 min acquisition

``--calib`` is the run whose forward model the drives are solved from, either
its ``calib.json`` or the directory holding one, and it is REQUIRED: the
resolution being measured is the resolution of ONE calibration, so which one
has to be said out loud rather than left to whichever run was current when
this file was last edited.

This script MEASURES and writes a CSV to ``runs/4bit_acc_char``.  It does no
fitting, no reporting and no plotting -- the analysis happens separately, off
the CSV.

What is on the panel
--------------------
``LEVELS`` drive levels evenly spaced over ``(0, hi]`` -- a uniform 4-bit grid,
with the dark standing in for code 0.  Each level is solved ONCE and then
re-measured ``REPEATS`` times: repeats of the same x/w, so pattern-to-pattern
differences stay out of the spread the analysis will measure.  One seed for
every level, so the levels are as close to one drive at different amplitudes
as the solver allows.

Why alpha stops at 0.25, not 1
------------------------------
The 1027 ONN run never drove past ``alpha = 0.235`` -- median read 0.028, max
0.235 of the model's ``Y_max``.  Resolution measured over 0..1 would answer a
question about a range the machine never uses; the codes that matter are the
ones inside the operating band.  ``--hi`` moves it if that changes.

The order is SHUFFLED across the whole (level x repeat) list.  Drift over 45
minutes would otherwise land on the level axis, which is the axis the analysis
reads.

Unlike ``alpha_sweep.py`` this run is deliberately much longer than
``DARK_EVERY_S``, so :func:`run_patterns` re-reads the dark throughout.  That is
wanted here: the 1027 ONN run held one dark for 40 h and a 14.7 mV stray-light
pedestal sat undetected in it for 7 h.

``--t-meas`` defaults to 4 s, the window the 1027 ONN run used, because the
resolution being measured should be the resolution that run had.  Longer
windows buy a lower sigma and a different answer.

The forward model is asked for one thing only: which drive is ``alpha = 0.1``.
Its gain never enters, because nothing here is compared against a prediction.

Needs real hardware (DAQ Dev1, SLM on its own display), and the GUI cannot be
holding them at the same time.  The DAQ is opened first, so a machine that is
still busy stops the run before the panel is touched.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.measure.patterns import (               # noqa: E402
    DARK_EVERY_S, run_patterns,
)
from calibration_module.steps import calib_step8 as STEP8       # noqa: E402
from calibration_module.calib_file import calib_path            # noqa: E402

OUT_DIR = REPO_ROOT / "runs/4bit_acc_char"

#: 16 codes = 4 bits.  The dark is code 0, so the levels are 1..LEVELS.
LEVELS = 16
ALPHA_HI = 0.25
REPEATS = 40

#: The 1027 ONN run's window, not the bench default of 8 s -- see the docstring.
T_MEAS_S = 4.0

#: Stop the run if a raw trace peak reaches this.  The mean can look sane while
#: the peak flat-tops; step 8 measured peak/mean at 1.22 typical and 1.9 at the
#: p99, so a mean well under the rail is not on its own a promise.
PEAK_ABORT_V = 0.90


# ======================================================================
# the ladder
# ======================================================================

def solve(step7: Path, hi: float, levels: int, seed: int):
    """``(models, phases, pairs, scale, patterns)`` -- no hardware.

    One pattern per level, ``alpha_k = k * hi / levels`` for ``k = 1..levels``.
    """
    models, phases, _ref = STEP8.load_forward_model(step7)
    pairs = sorted(models)
    scale = STEP8.full_scale(models, phases, pairs)
    patterns = [
        STEP8.pattern_for_alpha(models, phases, pairs,
                                float(k * hi / levels),
                                seed=seed, label=f"L{k:02d}")
        for k in range(1, levels + 1)
    ]
    return models, phases, pairs, scale, patterns


def build_reads(patterns, repeats: int, seed: int):
    """``entries`` -- every level ``repeats`` times, shuffled.

    Each entry is ``(level_index, repeat_index, pattern)``, in read order.
    """
    plan = [(k, r, p) for k, p in enumerate(patterns, start=1)
            for r in range(repeats)]
    order = np.random.default_rng(seed).permutation(len(plan))
    return [plan[i] for i in order]


def report_plan(scale, patterns, entries, acq, out: Path) -> None:
    """What the run will cost, before anything is opened."""
    secs = (acq.t_meas_s + acq.settle_s) * len(entries)
    darks = int(secs // DARK_EVERY_S) + 1
    secs += darks * acq.t_meas_s
    print(f"\n{len(patterns)} levels, alpha {patterns[0].alpha:.4f} .. "
          f"{patterns[-1].alpha:.4f}   (asked {patterns[0].target_alpha:.4f} "
          f".. {patterns[-1].target_alpha:.4f})")
    print(f"{len(entries) // len(patterns)} repeats each -> "
          f"{len(entries)} reads of {acq.t_meas_s:g} s")
    print(f"Y_max of the model  {scale.fs * 1e3:9.1f} mV  -- in the MODEL's "
          f"volts, which may not be the bench's")
    print(f"top level predicts  {patterns[-1].alpha * scale.fs * 1e3:9.1f} mV "
          f"of the {STEP8.AMP_LIMIT_V:g} V rail, IF the model still has the "
          f"bench's gain")
    print(f"~{darks} darks (every {DARK_EVERY_S:g} s)")
    print(f"Estimate  {secs / 60:.1f} min ({secs / 3600:.2f} h)")
    print(f"CSV ->    {out}")


def measure(step7: Path, patterns, entries, models, phases, pairs,
            acq, out: Path) -> Path:
    """Drive the shuffled list and write one row per read."""
    from calibration_module.measure.bench import connect_daq, connect_slm
    from calibration_module.fit.verify import write_verify_csv

    # DAQ first: if the GUI still holds it, nothing has been displayed yet.
    daq = connect_daq(device="Dev1", channel="ai0",
                      t_both=acq.t_meas_s, t_single=acq.t_meas_s,
                      min_val=-acq.range_v, max_val=acq.range_v)
    slm = connect_slm()
    layout = STEP8.load_layout(step7, pairs)

    x = np.stack([p.x for _k, _r, p in entries])[None]
    w = np.stack([p.w for _k, _r, p in entries])[None]
    extra = [[{"kind": p.kind, "label": p.label, "seq": i,
               "level": k, "rep": r,
               "target_alpha": f"{p.target_alpha:.6f}",
               "alpha": f"{p.alpha:.6f}",
               "t_meas_s": f"{acq.t_meas_s:.3f}"}
              for i, (k, r, p) in enumerate(entries)]]

    rows: list[dict] = []
    import threading
    stop = threading.Event()

    def watch(row: dict) -> None:
        """Peak guard.  Cannot raise, so it stops the loop between reads."""
        if float(row.get("peak_v", 0.0)) >= PEAK_ABORT_V:
            print(f"  PEAK {row['peak_v'] * 1e3:.0f} mV -- at the rail, stopping")
            stop.set()

    try:
        run_patterns(daq, slm, layout, [tuple(pairs)], x, w, models, phases,
                     rows=rows, acq=acq, extra=extra, y_sat=None,
                     row_hook=watch, stop_event=stop, log=print)
    finally:
        try:
            wpx, hpx = slm.get_slm_info()
            slm.display_array(np.zeros((hpx, wpx), dtype=np.uint8))
        except Exception:                                    # noqa: BLE001
            pass
        for shut, name in ((getattr(slm, "close_slm", None), "SLM"),
                           (getattr(daq, "disconnect", None), "DAQ")):
            try:
                if shut is not None:
                    shut()
            except Exception as exc:                         # noqa: BLE001
                print(f"{name} did not close cleanly: {exc}")

    stamp = time.strftime("%m%d_%H%M")
    path = write_verify_csv(
        out / f"bit_acc_chara_{stamp}.csv", pairs, rows,
        meta={"step7_json": Path(step7).resolve(),
              "levels": len(patterns),
              "repeats": len(entries) // len(patterns),
              "alpha_hi": patterns[-1].target_alpha,
              "t_meas_s": f"{acq.t_meas_s:g}",
              "y_max_model_v": f"{STEP8.full_scale(models, phases, pairs).fs:.9g}"},
        extra=["kind", "label", "seq", "level", "rep", "target_alpha",
               "alpha", "t_meas_s", "peak_v"])
    print(f"\nCSV written to {path}")
    return Path(path)


# ======================================================================
# cli
# ======================================================================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--calib", required=True, type=Path, metavar="RUN",
                   help="the run to solve the drives from: its calib.json, or "
                        "the directory holding one.  Only its step-6 etas and "
                        "step-7 phases matter here -- they decide which drive "
                        "is alpha = 0.1")
    p.add_argument("--hi", type=float, default=ALPHA_HI,
                   help="top of the band; the 1027 ONN run reached 0.235")
    p.add_argument("--levels", type=int, default=LEVELS)
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--t-meas", type=float, default=T_MEAS_S,
                   help="DAQ window per read; the 1027 ONN run used 4 s")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--dry-run", action="store_true",
                   help="solve the drives and stop; no instrument is opened")
    args = p.parse_args(argv)

    if not 0.0 < args.hi <= 1.0:
        p.error(f"--hi {args.hi}: expected 0 < hi <= 1")

    # resolve a directory to the calib.json inside it once, here, so the file
    # the drives came from is what the CSV's metadata records
    step7 = calib_path(args.calib)
    if not step7.is_file():
        p.error(f"--calib {args.calib}: no calib.json at {step7}")

    acq = replace(STEP8.ACQ, t_meas_s=args.t_meas)
    models, phases, pairs, scale, patterns = solve(
        step7, args.hi, args.levels, args.seed)
    entries = build_reads(patterns, args.repeats, args.shuffle_seed)
    print(f"step 7: {step7.parent.name}/{step7.name}   pairs {pairs}")
    report_plan(scale, patterns, entries, acq, args.out)
    if args.dry_run:
        print("\ndry run: nothing was opened")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    measure(step7, patterns, entries, models, phases, pairs, acq, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
