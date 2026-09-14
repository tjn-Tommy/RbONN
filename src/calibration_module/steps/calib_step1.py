"""Step 1 -- the SLM's off/on grayscale levels, from a full-screen OSA sweep.

Needs the OSA and the SLM::

    python src/calibration_module/steps/calib_step1.py
    python src/calibration_module/steps/calib_step1.py --out DIR

Lights the whole panel at each of ``LEVELS`` in turn, reads the OSA, and keeps
the darkest and the brightest level.  Writes ``calib_step1_MMDD_HHMM.json``,
the seed ``calib_step2.py --step1`` reads.

This is the runner only: the sweep is
:func:`calibration_module.measure.slm_response.run_step1`, the same call the
GUI's Step 1 tab makes.  The constants below are that tab's defaults.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.slm_response import save_calibration_result  # noqa: E402
from calibration_module.measure.bench import (  # noqa: E402
    connect_osa,
    connect_slm,
    print_progress,
)
from calibration_module.measure.slm_response import run_step1  # noqa: E402
from osa_module.controller import MeasurementSettings  # noqa: E402

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"   # output directory; --out overrides it

MEASURE = MeasurementSettings(
    center_wl="778nm",
    span="8nm",
    sensitivity="HIGH2",
    sampling_points="AUTO",
    reference_level="10uW",
    y_unit="LINear",
)
LEVELS = [*range(0, 1024, 64), 1023]

OSA_HOST = "192.168.1.11"
OSA_PORT = 10001
SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=CALIB_PATH,
                        help=f"output directory (default {CALIB_PATH})")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M")

    print(f"Step 1: {len(LEVELS)} levels, {LEVELS[0]}..{LEVELS[-1]}")
    osa = connect_osa(OSA_HOST, OSA_PORT)
    try:
        slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
        try:
            result = run_step1(osa, slm, LEVELS, MEASURE,
                               progress_callback=print_progress)
        finally:
            slm.close_slm()
    finally:
        osa.disconnect()

    path = save_calibration_result(result, args.out / f"calib_step1_{stamp}.json")
    print(f"\nmin level {result.min_level}, max level {result.max_level}")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
