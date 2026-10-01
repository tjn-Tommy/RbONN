"""Step 6 v2 -- per-pair TPA efficiency eta from the DIFFERENCE estimator.

Needs real hardware for the sweep; the refit path is offline.  Either way
``--run`` and ``--pairs`` are required and have no default::

    S="--run src/calib_data/run_0930 --pairs 2,3,4,5,6"
    python src/calibration_module/steps/calib_step6_v2.py $S
    python src/calibration_module/steps/calib_step6_v2.py $S --meas
    python src/calibration_module/steps/calib_step6_v2.py $S some.csv
    python src/calibration_module/steps/calib_step6_v2.py $S some.csv --anchor

``--run`` is a run DIRECTORY holding one ``calib.json``.  Its ``step3`` section
is the calibration every level here is *encoded through*, and the CSV and PNGs
land beside it; the fit is written back into the same file as its ``step6``
section, replacing whatever was there.  There is deliberately no default: the
path used to be a constant edited by hand, which meant the file it named drifted
out from under the script every time a run was filed away, and a run against the
wrong calibration does not fail, it silently calibrates a different aperture.
The pair list is required for the same reason -- as a constant it silently
calibrated whichever pairs the last edit left in it.

The underlying physics is unchanged from v1::

    Y(x, w) = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d

What changes is how eta is *extracted*.

The estimator itself -- how the difference is formed, the GLS covariance that
ties the two fits together, the four verification checks and where the
uncertainties come from -- lives with the code it describes, in
:mod:`calibration_module.fit.pair_v2`.  This file is the runner: it loads the
Step-3 calibration, drives the schedule through
:mod:`calibration_module.measure.pair_v2`, then fits, reports, plots and saves.
Everything tunable is the block of constants below -- the fit's half in
``CONFIG``, the acquisition's in ``ACQ``.  Both are plain values the GUI builds
too, so the two front ends run the same estimator over the same schedule.

Acquisition order
-----------------
Repeats are interleaved: the schedule runs round-robin passes over the grid
rather than n back-to-back reads of one level, so a slow drift cannot
masquerade as a slope.  Within each pass levels run brightest first, so
``(1, 1.0)`` is the very first acquisition and a dead or blocked beam shows
immediately.

The CSV keeps v1's column layout, so the same data can be re-fit with v1's
joint 6-parameter estimator for comparison::

    python src/calibration_module/steps/calib_step6_v1.py <a_v2_meas.csv>

That is the cross-check that justifies the change: same rows, two estimators.

PMT temperature
---------------
Every row carries the PMT thermistor (Keithley 2400, ohms and TH10K degrees) in
``pmt_ohm_Ohm`` and ``pmt_c``, and the CSV brackets itself with a start and end
reading in its ``# key,value`` meta -- so a drifting eta can be read against the
detector's own temperature rather than against the room's.  It is ON by default
(``--no-pmt`` turns it off); the 2400 is re-read at most every ``--pmt-period``
seconds (default 60) and the rows in between repeat the last reading, since the
bead's time constant is minutes and a per-row GPIB round trip would add ~90 s to
a sweep and measure nothing extra.  A missing or failing 2400 blanks the two
columns rather than ending the run.  One GPIB, one 2400: do NOT run
``drafts/daq_pmt_monitor.py`` alongside a sweep.

Output is one ``calib_step6v2_result_MMDD_HHMM.json`` embedding the input
Step-3 calibration, and carrying ``a_x, q_x, a_w, q_w, d`` and ``eta`` per pair
(a q that was not fitted is written out as an exact zero)
in the schema step 7 already reads (``PairModel.from_json_channel``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.pair_v2 import (  # noqa: E402
    PairV2Config,
    PairV2Fit,
    average_levels,
    burst_rows,
    fit_pair,
    load_meas_csv,
    report,
    save_plot,
    save_step6_section,
    write_meas_csv,
)
from calibration_module.fit.sigma import STD_FLOOR_V, floor_std  # noqa: E402,F401
from calibration_module.measure.bench import connect_daq, connect_slm  # noqa: E402
from calibration_module.measure.pair_v2 import (  # noqa: E402
    DEFAULT_ACQ,
    build_schedule,
    measure_pair,
    run_seconds,
)
from calibration_module.measure.pmt_temp import (  # noqa: E402
    PMT_PERIOD_S,
    PmtTemperature,
    open_pmt,
)
from calibration_module.calib_file import CalibFileError, load_calib  # noqa: E402
from calibration_module.fit.outliers import BURST_K_REPEAT  # noqa: E402

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"   # the run directory; --run sets it

# Neither the run directory nor the pair list is here -- they are the required
# --run and --pairs arguments.  See the module docstring for why they have no
# default.  Left empty, so code that imports this module and sweeps without
# setting the list fails before touching the hardware.
PAIR_INDICES: list[int] = []

# The estimator's setup.  Every field has the validated default (see
# PairV2Config); name one here only to change it.  Keep pair_index_base in step
# with calib_step7_v2.PAIR_INDEX_BASE -- step 7 reads this script's pair labels
# straight out of the step6 section.
CONFIG = PairV2Config(
    pair_index_base=1,                      # pairs are numbered 1..N
    fit_q=False,
    verify_enabled=True,
)

# Acquisition timing and input range: the bench's, shared with steps 7 and 8
# and the ONN loop, so their numbers are comparable without a caveat.  See
# PairV2Acq / T_MEAS_S / RANGE_V for why each value is what it is.
ACQ = DEFAULT_ACQ

SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"


# ======================================================================
# inputs
# ======================================================================

def _load_layout(run: Path):
    """The run's step-3 section -> channel layout, validating PAIR_INDICES.

    The Step-3b/3c rows ARE the channels, so the layout is loaded verbatim (the
    same ``channel_layout_from_calibration`` the GUI encoding page uses) -- no
    re-tiling, so pair indices here mean the same thing as in the UI.

    Step 3 records only a wavelength per channel, never an index -- a channel's
    identity IS its position in that list -- so the pair label is positional
    too: pair ``i`` is the ``CONFIG.slot(i)``-th channel of the Step-3
    calibration.
    """
    layout = load_calib(run).layout()
    for pi in PAIR_INDICES:
        if not (0 <= CONFIG.slot(pi) < layout.n_channels):
            raise ValueError(
                f"pair index {pi} out of range (layout has {layout.n_channels} "
                f"pairs, numbered from {CONFIG.pair_index_base})"
            )
    return layout


def _pair_wavelengths(layout, index: int) -> tuple[float, float, float]:
    """Wavelengths of a pair's two channels, or NaN if the layout lacks that slot.

    ``index`` is a pair LABEL; ``CONFIG.slot`` maps it into the Step-3 layout.
    A CSV may carry pair labels the loaded layout does not cover (a synthetic
    file, or a layout from a different run); the wavelengths are cosmetic --
    only the JSON/plot labels use them -- so fall back to NaN rather than
    killing the fit.
    """
    nan = float("nan")
    slot = CONFIG.slot(index)
    if not (0 <= slot < min(len(layout.x_channels), len(layout.w_channels))):
        return nan, nan, nan
    x_ch = layout.x_channels[slot]
    w_ch = layout.w_channels[slot]
    return (float(x_ch.wavelength_nm), float(w_ch.wavelength_nm),
            0.5 * (float(x_ch.wavelength_nm) + float(w_ch.wavelength_nm)))


# ======================================================================
# the run
# ======================================================================

def _run_sweep(run: Path, fit_after: bool, *, include_anchor: bool = False,
               pmt: "PmtTemperature | None" = None) -> None:
    """Drive every pair's interleaved schedule; optionally fit, plot and save."""
    if not PAIR_INDICES:
        raise ValueError("no pairs to calibrate -- pass --pairs, e.g. --pairs 2,3,4,5,6")
    layout = _load_layout(run)
    # The whole chain in the file, stamps and all, before the laser is asked for
    # anything: a step 3 older than you think is the one failure that does not
    # announce itself later.
    load_calib(run).print_chain()
    schedule = build_schedule(CONFIG)
    secs = run_seconds(schedule, ACQ)
    n_verify = len(CONFIG.verify_grid) if CONFIG.verify_enabled else 0
    print(f"Step 6 v2: {len(schedule)} acquisitions/pair over "
          f"{len(CONFIG.full_grid())} levels ({len(CONFIG.grid)} estimator + "
          f"{n_verify} verification), interleaved, brightest first "
          f"(~{secs/60:.1f} min/pair)")
    print(f"Pairs: {list(PAIR_INDICES)}  "
          f"(~{secs * len(PAIR_INDICES) / 60:.0f} min total)")
    # The aperture is not a knob here -- it comes from the Step-3 file, which
    # records the window it swept -- but it belongs in run.log all the same, so
    # a result can be read back knowing what was lit.
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
    rows_by_pair: dict[int, list] = {}
    total = len(schedule) * len(PAIR_INDICES)
    try:
        for n, index in enumerate(PAIR_INDICES):
            print(f"\n=== Sweep: pair {index} ===")
            rows_by_pair[index] = measure_pair(
                daq, slm, layout, index, schedule,
                cfg=CONFIG, acq=ACQ,
                step0=n * len(schedule), total=total,
                progress_callback=lambda p: print(p.line()),
                log=print,
                row_hook=None if pmt is None else pmt.sample,
            )
    finally:
        slm.close_slm()
        daq.disconnect()
        if pmt is not None:
            meta.update({f"{k}_end": v for k, v in pmt.read().items()})

    stamp = time.strftime("%m%d_%H%M")
    csv_path = write_meas_csv(rows_by_pair,
                              CALIB_PATH / f"calib_step6v2_meas_{stamp}.csv",
                              meta=meta)
    n_rows = sum(len(v) for v in rows_by_pair.values())
    print(f"\nSaved {n_rows} rows to {csv_path}")  # raw rows on disk BEFORE fitting
    if not fit_after:
        return
    _fit_and_save(rows_by_pair, stamp, run, csv_path, layout=layout,
                  include_anchor=include_anchor)


def _fit_and_save(rows_by_pair: dict[int, list], stamp: str, run: Path,
                  csv_path: Path, *, layout=None,
                  include_anchor: bool = False) -> None:
    """Fit every pair, print the report, write the step6 section and the plots."""
    fits: list[PairV2Fit] = []
    bursts: dict[int, list[dict]] = {}
    for index in sorted(rows_by_pair):
        rows = list(rows_by_pair[index])
        # Loud, because a burst is a BENCH fault, not a statistical nuisance: the
        # trace jumped mid-acquisition.  Dropping it quietly would turn a run
        # that should be investigated into one that merely looks clean.
        mask = burst_rows(rows)
        if mask.any():
            bursts[index] = [
                {"x": float(r[1]), "w": float(r[2]), "trial": int(r[0]),
                 "voltage_mean_v": float(r[3]), "voltage_std_v": float(r[4])}
                for r, bad in zip(rows, mask) if bad
            ]
            print(f"\n!! pair {index}: {int(mask.sum())} BURST acquisition(s) "
                  f"dropped (trace std >{BURST_K_REPEAT:g}x the quietest repeat "
                  f"of the same level) -- the bench glitched, investigate:")
            for b in bursts[index]:
                sib = [float(r[4]) for r in rows
                       if (round(float(r[1]), 6), round(float(r[2]), 6))
                       == (round(b["x"], 6), round(b["w"], 6))]
                quiet = min(sib) if sib else float("nan")
                print(f"     x={b['x']:g} w={b['w']:g} trial {b['trial']}: "
                      f"{b['voltage_mean_v']*1e3:+.4f} mV with a "
                      f"{b['voltage_std_v']*1e3:.4f} mV trace std "
                      f"({b['voltage_std_v']/quiet:.0f}x the quietest repeat)")
        levels = average_levels(rows)
        try:
            fit = fit_pair(index, levels, CONFIG, include_anchor=include_anchor)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"\n=== pair {index} ===\n  fit FAILED: {exc}")
            continue
        if layout is not None:
            fit.wl_x_nm, fit.wl_w_nm, fit.nominal_wl_nm = _pair_wavelengths(layout, index)
        fits.append(fit)
        print(f"\n=== pair {index} ===")
        report(fit)

    if not fits:
        print("\nNo pair fitted -- nothing saved.")
        return

    center_wl = float(getattr(layout, "center_wl", 0.0)) if layout is not None else 0.0
    calib = save_step6_section(fits, run, center_wl=center_wl,
                               dropped_bursts=bursts, csv=csv_path)
    print(f"\nStep-6 v2 fits -> {calib.path}")
    for fit in fits:
        plot_path = Path(run) / f"step6_pair{fit.index}_{stamp}.png"
        save_plot(fit, plot_path)
        print(f"Plot saved to {plot_path}")


def fit_csv(path: str | Path, run: Path, *, include_anchor: bool = False) -> None:
    """Re-fit an already-recorded v2 CSV offline (no hardware).

    Writes the step6 section and the per-pair PNGs the hardware run would,
    REPLACING whatever step 6 the run already held -- one directory, one
    calibration, and the CSV this came from is recorded in the section.  ``run``
    is required even here: the fit itself needs only the CSV, but the wavelength
    labels come from the layout, and a step-3 section that fails to load still
    leaves the fit runnable with NaN wavelengths.
    """
    rows_by_pair = load_meas_csv(path)
    n = sum(len(v) for v in rows_by_pair.values())
    print(f"Loaded {path}: {len(rows_by_pair)} pair(s), {n} acquisitions")
    if include_anchor:
        print("Anchor: D(0) included as a fitted point (intercept pinned on data).")
    try:
        layout = _load_layout(run)
    except (CalibFileError, FileNotFoundError, ValueError) as exc:
        print(f"(layout unavailable, wavelengths left as NaN: {exc})")
        layout = None
    _fit_and_save(rows_by_pair, time.strftime("%m%d_%H%M"), run, Path(path),
                  layout=layout, include_anchor=include_anchor)


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
    global PAIR_INDICES, CALIB_PATH
    parser = argparse.ArgumentParser(
        prog="calib_step6_v2.py",
        description="Step 6 v2 -- per-pair TPA efficiency eta (difference estimator).",
        epilog="--run and --pairs have no default: every level is encoded "
               "through that run's step-3 section, so running against the wrong "
               "one calibrates a different aperture without failing.",
    )
    parser.add_argument(
        "--run", required=True, type=Path, metavar="DIR",
        help="run directory holding calib.json; its step3 section is the "
             "layout, and the CSV, PNGs and step6 section land there (required)",
    )
    parser.add_argument(
        "--pairs", required=True, type=_index_list, metavar="2,3,4",
        help="pair labels to calibrate, comma-separated (required)",
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
        "--anchor", action="store_true",
        help="put the measured D(0) into the fit, pinning the intercept on data",
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

    PAIR_INDICES = args.pairs
    CALIB_PATH = args.run
    Path(CALIB_PATH).mkdir(parents=True, exist_ok=True)

    if args.csv is not None:            # offline re-fit, no hardware
        fit_csv(args.csv, args.run, include_anchor=args.anchor)
        return 0
    _run_sweep(args.run, fit_after=not args.meas, include_anchor=args.anchor,
               pmt=open_pmt(args.pmt_resource, args.pmt_period,
                            enabled=args.pmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
