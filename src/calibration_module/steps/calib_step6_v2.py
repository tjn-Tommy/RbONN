"""Step 6 v2 -- per-pair TPA efficiency eta from the DIFFERENCE estimator.

Needs real hardware for the sweep; the refit path is offline.  ``--calib`` and
``--pairs`` are required and have no default::

    S="--calib src/calib_data/run_0930/calib.json --pairs 2,3,4,5,6"
    python src/calibration_module/steps/calib_step6_v2.py $S
    python src/calibration_module/steps/calib_step6_v2.py $S --meas
    python src/calibration_module/steps/calib_step6_v2.py $S some.csv

In:  one ``calib.json``, whose ``step3`` section is the calibration every level
     here is ENCODED THROUGH, plus the pair list.  A positional CSV re-fits that
     sweep offline instead of driving the bench.  Neither argument has a
     default: as constants they drifted out from under the script whenever a run
     was filed away, and a sweep against the wrong calibration does not fail --
     it silently calibrates a different aperture.
Out: ``calib_step6v2_meas_<MMDD_HHMM>.csv`` in that file's folder, and the
     file's own ``step6`` section: ``a_x, q_x, a_w, q_w, d`` and ``eta`` per
     pair (a q that was not fitted written as an exact zero), in the schema
     step 7 reads through ``PairModel.from_json_channel``.

The physics is unchanged from v1::

    Y(x, w) = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d

What changes is how eta is EXTRACTED.  The estimator -- the difference, the GLS
covariance that ties the two fits together, the verification -- lives with its
code in :mod:`calibration_module.fit.pair_v2`; this file is the runner.
Everything tunable is the constants below: the sweep's levels in ``CONFIG``, the
acquisition's in ``ACQ``, both plain values the GUI builds too, so the two front
ends run the same estimator over the same schedule.  The fit takes nothing from
either -- it reads the levels off the rows, so a re-fit of a recorded CSV is
fitted exactly as it was swept.

Acquisition order.  Repeats are interleaved, round-robin passes over the grid
rather than n back-to-back reads of one level, so a slow drift cannot
masquerade as a slope.  Within a pass levels run brightest first, so a dead or
blocked beam shows on the very first acquisition.

The CSV keeps v1's column layout, so the same rows can be re-fit with v1's joint
6-parameter estimator -- the cross-check that justifies the change.

``--pmt`` (on by default) logs the PMT thermistor (Keithley 2400, ohms and TH10K
degrees) beside every row and brackets the CSV with a start and end reading, so
a drifting eta can be read against the detector's own temperature rather than
the room's.  Re-read at most every ``--pmt-period`` seconds, the bead's time
constant being minutes; a missing or failing 2400 blanks the two columns rather
than ending the run.  One GPIB, one 2400: do NOT run
``drafts/daq_pmt_monitor.py`` alongside a sweep.
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
    sweep_floor,
    burst_rows,
    fit_pair,
    load_meas_csv,
    report,
    save_step6_section,
    write_meas_csv,
)
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
from calibration_module.calib_file import (  # noqa: E402
    CalibFileError,
    calib_path,
    load_calib,
)
from calibration_module.fit.outliers import BURST_K_REPEAT  # noqa: E402

# ---- Edit these to match your setup ----

# Neither the calib file nor the pair list is here -- they are the required
# --calib and --pairs arguments.  See the module docstring for why they have no
# default.  Left empty, so code that imports this module and sweeps without
# setting the list fails before touching the hardware.
PAIR_INDICES: list[int] = []

# The sweep's levels.  Every field has the validated default (see
# PairV2Config); name one here only to change it.  The fit does not read it.  Keep pair_index_base in step
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

def _load_layout(calib_file: Path):
    """The run's step-3 section -> channel layout, validating PAIR_INDICES.

    The step-3 rows ARE the channels, so the layout is loaded verbatim -- no
    re-tiling, and a pair index here means what it means in the GUI.  Step 3
    records a wavelength per channel and never an index, a channel's identity
    being its POSITION in that list, so the pair label is positional too: pair
    ``i`` is the ``CONFIG.slot(i)``-th channel.
    """
    layout = load_calib(calib_file).layout()
    for pi in PAIR_INDICES:
        if not (0 <= CONFIG.slot(pi) < layout.n_channels):
            raise ValueError(
                f"pair index {pi} out of range (layout has {layout.n_channels} "
                f"pairs, numbered from {CONFIG.pair_index_base})"
            )
    return layout


def _pair_wavelengths(layout, index: int) -> tuple[float, float, float]:
    """Wavelengths of a pair's two channels, or NaN if the layout lacks that slot.

    ``index`` is a pair LABEL, which ``CONFIG.slot`` maps into the layout.  A
    CSV may carry labels the loaded layout does not cover (a synthetic file, or
    another run's layout); these are cosmetic -- only the JSON and plot labels
    use them -- so NaN rather than killing the fit.
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

def _run_sweep(calib_file: Path, fit_after: bool, *,
               pmt: "PmtTemperature | None" = None) -> None:
    """Drive every pair's interleaved schedule; optionally fit, plot and save."""
    if not PAIR_INDICES:
        raise ValueError("no pairs to calibrate -- pass --pairs, e.g. --pairs 2,3,4,5,6")
    layout = _load_layout(calib_file)
    # The whole chain in the file, stamps and all, before the laser is asked
    # for anything: a step 3 older than you think is the one failure that does
    # not announce itself later.
    load_calib(calib_file).print_chain()
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
                              calib_file.parent / f"calib_step6v2_meas_{stamp}.csv",
                              meta=meta)
    n_rows = sum(len(v) for v in rows_by_pair.values())
    print(f"\nSaved {n_rows} rows to {csv_path}")  # raw rows on disk BEFORE fitting
    if not fit_after:
        return
    _fit_and_save(rows_by_pair, calib_file, csv_path, layout=layout)


def _fit_and_save(rows_by_pair: dict[int, list], calib_file: Path,
                  csv_path: Path, *, layout=None) -> None:
    """Fit every pair, print the report and write the step6 section.

    No pictures: the fit is in the section and the rows are in the CSV it points
    at, so the GUI's Load redraws any of it on demand, which a PNG cannot.

    The fit reads only the rows -- which levels are the cross line, and so what
    range eta is the slope over, comes from the file and never from ``CONFIG``,
    which describes the NEXT sweep.
    """
    fits: list[PairV2Fit] = []
    bursts: dict[int, list[dict]] = {}
    # Once, over the whole sweep: the floor is a property of the detector, and
    # two dark acquisitions are an anecdote where twenty are not.
    floor = sweep_floor(rows_by_pair)
    print(f"\nSigma floor: {floor[0]*1e6:.1f} uV [{floor[1]}]")
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
        levels = average_levels(rows, floor=floor)
        try:
            fit = fit_pair(index, levels)
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
    calib = save_step6_section(fits, calib_file, center_wl=center_wl,
                               dropped_bursts=bursts, csv=csv_path)
    print(f"\nStep-6 v2 fits -> {calib.path}")


def fit_csv(path: str | Path, calib_file: Path) -> None:
    """Re-fit an already-recorded v2 CSV offline (no hardware).

    Writes the step6 section the hardware run would, REPLACING whatever that
    file held, and records the CSV it came from.  To keep the old fit, re-fit
    against a COPY of the calib.json: the file is the calibration, so copying it
    is how you get a second opinion without losing the first.

    ``--calib`` is required even here.  The fit needs only the CSV, but the
    wavelength labels come from the layout -- and a step-3 section that fails to
    load still leaves the fit runnable, with NaN wavelengths.
    """
    rows_by_pair = load_meas_csv(path)
    n = sum(len(v) for v in rows_by_pair.values())
    print(f"Loaded {path}: {len(rows_by_pair)} pair(s), {n} acquisitions")
    try:
        layout = _load_layout(calib_file)
    except (CalibFileError, FileNotFoundError, ValueError) as exc:
        print(f"(layout unavailable, wavelengths left as NaN: {exc})")
        layout = None
    _fit_and_save(rows_by_pair, calib_file, Path(path), layout=layout)


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
    global PAIR_INDICES
    parser = argparse.ArgumentParser(
        prog="calib_step6_v2.py",
        description="Step 6 v2 -- per-pair TPA efficiency eta (difference estimator).",
        epilog="--calib and --pairs have no default: every level is encoded "
               "through that file's step-3 section, so running against the wrong "
               "one calibrates a different aperture without failing.",
    )
    parser.add_argument(
        "--calib", required=True, type=Path, metavar="FILE",
        help="calib.json whose step3 section is the layout; the CSV and PNGs "
             "land beside it and the step6 section goes in it; a directory "
             "means its calib.json (required)",
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
    calib_file = calib_path(args.calib)

    if args.csv is not None:            # offline re-fit, no hardware
        fit_csv(args.csv, calib_file)
        return 0
    _run_sweep(calib_file, fit_after=not args.meas,
               pmt=open_pmt(args.pmt_resource, args.pmt_period,
                            enabled=args.pmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
