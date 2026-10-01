"""Step 3 -- per-channel transfer curves on the DAQ, one channel window at a time.

Needs the DAQ and the SLM.  ``--run`` is required: a run directory holding one
``calib.json`` whose ``step2`` section gives the px -> nm line::

    python src/calibration_module/steps/calib_step3.py --run src/calib_data/run_0930
    python src/calibration_module/steps/calib_step3.py --run DIR --width 20 --gap 30

Tiles ``CHANNELS_PER_SIDE`` mirror-symmetric channels around ``TARGET_NM`` on
the step-2 line, skipping any that overlaps ``GUARD_BANDS_NM``, then lights each
channel at each of ``LEVELS`` and reads the bucket detector against one all-dark
frame.  Writes ``step3_MMDD_HHMM.csv`` with the sweep and the ``step3`` section
of ``calib.json`` with the FITTED transfer curves -- which is what
``calib_step6_v2.py`` encodes every level through, and it records the
``--width`` / ``--gap`` it swept.

Writing this section DROPS the run's step 6, 7 and 8: a new channel grid is a
new aperture, and etas measured through the old one describe a bench that no
longer exists.

This was ``calib_step3c.py``; the OSA-based 3b is legacy (see the README).  The
sweep itself is :func:`calibration_module.measure.slm_response.run_step3c`, the
same call the GUI's Step 3 tab makes.  The constants below are that tab's
defaults.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.calib_file import load_calib, write_section  # noqa: E402
from calibration_module.fit.slm_response import (  # noqa: E402
    CalibrationResult,
    load_step2_map,
    step3_section,
    write_intensity_calibration_csv,
)
from calibration_module.measure.bench import connect_slm, print_progress  # noqa: E402
from calibration_module.measure.slm_response import run_step3c  # noqa: E402
from daq_module import DAQController, DAQMonitorSettings  # noqa: E402

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"   # output directory; --out overrides it

LEVELS = [*range(0, 1024, 10), 1023]
CSV_MIN_MAX = (0, 1023)          # off/on levels for a --step2 CSV, which has none

TARGET_NM = 778.04
CHANNELS_PER_SIDE = 20
WIDTH_PX = 15                    # --width overrides
GAP_PX = 5                       # --gap overrides
GUARD_BANDS_NM = [(780.14, 0.06), (775.94, 0.06)]   # (centre, half-width); [] = none

# Raw volts, dark-subtracted, no sign flip: the sweep reads the DAQ exactly as
# the GUI's Step 3c page does.
DAQ_DEVICE = "Dev1"
DAQ = DAQMonitorSettings(
    channel="ai1",
    sample_rate=100_000.0,
    duration=1.0,                # averaging window per reading, s
    hold=0.150,                  # settle after each SLM frame, s
    min_val=-0.1,
    max_val=0.1,
)

SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch


def load_step2(run: Path) -> CalibrationResult:
    """The run's step-2 map -- see :func:`fit.slm_response.load_step2_map`."""
    return load_step2_map(run, fallback_min_max=CSV_MIN_MAX)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, metavar="DIR",
                        help="run directory holding calib.json; its step2 "
                             "section is the wavelength map, and the CSV and "
                             "step3 section land there (required)")
    parser.add_argument("--width", type=int, default=WIDTH_PX,
                        help=f"channel window, px (default {WIDTH_PX})")
    parser.add_argument("--gap", type=int, default=GAP_PX,
                        help=f"dark pad between channels, px (default {GAP_PX})")
    args = parser.parse_args(argv)
    args.run.mkdir(parents=True, exist_ok=True)
    step2 = load_step2(args.run)
    stamp = time.strftime("%m%d_%H%M")

    load_calib(args.run).print_chain()
    print(f"Step 3: {2 * CHANNELS_PER_SIDE} channels around {TARGET_NM:g} nm, "
          f"width {args.width} px, gap {args.gap} px, "
          f"{len(LEVELS)} levels {LEVELS[0]}..{LEVELS[-1]}")
    if GUARD_BANDS_NM:
        print("Guard bands: " + ", ".join(f"{c:g}+/-{h:g} nm" for c, h in GUARD_BANDS_NM))

    daq = DAQController(device=DAQ_DEVICE)
    daq.connect()
    try:
        daq.configure_monitor(DAQ)
        print(f"DAQ: {daq.identify()} {DAQ.channel} @ {DAQ.sample_rate:g} S/s, "
              f"avg {DAQ.duration:g} s, hold {DAQ.hold * 1e3:g} ms, +/-{DAQ.max_val:g} V")
        slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
        try:
            sweep = run_step3c(
                daq, slm, LEVELS, step2,
                target_wavelength_nm=TARGET_NM,
                n_channels_per_side=CHANNELS_PER_SIDE,
                channel_width_px=args.width,
                gap_px=args.gap,
                guard_bands_nm=GUARD_BANDS_NM,
                read_timeout=max(30.0, DAQ.duration * 3.0 + 10.0),
                progress_callback=print_progress,
            )
        finally:
            slm.close_slm()
    finally:
        daq.disconnect()

    result = sweep.result
    # CSV first: the sweep is on disk before a fit can fail on it.
    csv_path = write_intensity_calibration_csv(
        result, args.run / f"step3_{stamp}.csv")
    calib = write_section(args.run, 3, step3_section(result), csv=csv_path)
    print(f"\n{result.coordinates.size} channels, pitch {args.width + args.gap} px, "
          f"centre x={sweep.center_coordinate:.3f} px")
    print(f"Saved -> {calib.path}\n         {csv_path}")


if __name__ == "__main__":
    main()
