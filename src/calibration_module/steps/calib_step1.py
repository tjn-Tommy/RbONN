"""Step 1 -- the SLM's off/on grayscale levels, from a full-screen OSA sweep.

Needs the OSA and the SLM::

    python src/calibration_module/steps/calib_step1.py --calib DIR/calib.json

In:  ``--calib``, the file to write into; created with its folder if it is not
     there, this being the step that starts a calibration.
Out: its ``step1`` section -- the darkest and brightest of ``LEVELS``, found by
     lighting the whole panel at each in turn and reading the OSA.

Nothing downstream reads that section.  It is the seed ``calib_step2.py`` takes
its off/on levels from, and step 2 records them in its own section, so a
calibration whose levels were given to step 2 by hand (``--min-max``) is
complete with no step 1 at all.  What a step 1 ADDS is the record of how the
levels were found.

Running it against a file that already has sections DROPS step 2 onwards -- new
off/on levels are a new panel response, and everything was measured through the
old one.  What goes is named as it goes.

The runner only: the sweep is
:func:`calibration_module.measure.slm_response.run_step1`, the same call the
GUI's Step 1 tab makes, and the constants below are that tab's defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.calib_file import calib_path, write_section  # noqa: E402
from calibration_module.fit.slm_response import step1_section  # noqa: E402
from calibration_module.measure.bench import (  # noqa: E402
    connect_osa,
    connect_slm,
    print_progress,
)
from calibration_module.measure.slm_response import run_step1  # noqa: E402
from osa_module.controller import MeasurementSettings  # noqa: E402

# ---- Edit these to match your setup ----
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
    parser.add_argument("--calib", required=True, type=Path, metavar="FILE",
                        help="calib.json to write the off/on levels into, "
                             "created if it is not there; a directory means "
                             "its calib.json (required)")
    args = parser.parse_args(argv)

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

    calib = write_section(calib_path(args.calib), 1, step1_section(result))
    print(f"\nmin level {result.min_level}, max level {result.max_level}")
    print(f"Saved -> {calib.path}")


if __name__ == "__main__":
    main()
