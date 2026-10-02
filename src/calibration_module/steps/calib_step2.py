"""Step 2 -- the SLM column -> wavelength map, from a bright-window OSA sweep.

Needs the OSA and the SLM::

    python src/calibration_module/steps/calib_step2.py --calib DIR/calib.json
    python src/calibration_module/steps/calib_step2.py --calib DIR/calib.json --min-max 400,900

In:  ``--calib``, whose ``step1`` section gives the off/on levels, or
     ``--min-max`` to give them by hand.
Out: ``step2_MMDD_HHMM.csv`` beside that file, and the ``step2`` section
     pointing at it -- the wavelength map step 3 reads.

Walks a ``WINDOW_PX``-wide bright window across the panel, centroids the peak it
puts on the OSA, and fits one line through every peak.  ``USABLE_BAND_NM`` clips
the finished map to the band steps 3-8 can use, never the peak search (see
:func:`calibration_module.measure.slm_response.run_step2`).

``--min-max`` is not a fallback.  This is the only step that ever reads step 1,
and the section written here carries the levels onward, so a calibration whose
levels were given by hand needs no step 1 at all.

The runner only: the sweep is ``run_step2``, the same call the GUI's Step 2 tab
makes, and the constants below are that tab's defaults.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.calib_file import (  # noqa: E402
    calib_path,
    load_calib,
    write_section,
)
from calibration_module.fit.slm_response import (  # noqa: E402
    CalibrationResult,
    step2_section,
    write_wavelength_map_csv,
)
from calibration_module.measure.bench import (  # noqa: E402
    connect_osa,
    connect_slm,
    print_progress,
)
from calibration_module.measure.slm_response import run_step2  # noqa: E402
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
WINDOW_PX = 30
PEAK_HALF_WINDOW_NM = 0.2        # centroid half-window around each peak
STRIDE = 15                      # measure every Nth column; the fit fills the rest
SWEEP_SPAN_NM: float | None = 1.0    # narrow re-centred span; None = wide span everywhere
USABLE_BAND_NM: tuple[float | None, float | None] = (775.0, 781.0)   # None = open
REGION: tuple[int, int] | None = None     # (x_start, x_end) columns; None = full width

OSA_HOST = "192.168.1.11"
OSA_PORT = 10001
SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch


def _seed(calib_file: Path, min_max: str | None) -> CalibrationResult:
    """The off/on levels to sweep with: ``--min-max`` if given, else step 1.

    Only the two levels are taken; the sweep reads nothing else off the seed
    (see :func:`~calibration_module.measure.slm_response.wavelength_calibration`),
    which is why a hand-given pair is as good as a measured one.
    """
    if min_max is not None:
        try:
            low, high = (int(part) for part in min_max.split(","))
        except ValueError as exc:
            raise SystemExit(
                f"--min-max wants LOW,HIGH levels, got {min_max!r}") from exc
        if high < low:
            raise SystemExit("--min-max: HIGH must be >= LOW")
    else:
        section = load_calib(calib_file, missing_ok=True).get(1)
        if section is None:
            raise SystemExit(
                f"{calib_file} has no step1 section -- run step 1 against it, "
                f"or give the levels here with --min-max LOW,HIGH"
            )
        low = int(section["min_level"])
        high = int(section["max_level"])
    return CalibrationResult(
        wavelength=np.asarray([]),
        coordinates=np.asarray([]),
        max_level=high,
        min_level=low,
        level_range=np.asarray([], dtype=int),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calib", required=True, type=Path, metavar="FILE",
                        help="calib.json to read the off/on levels from and to "
                             "write the map into; a directory means its "
                             "calib.json (required)")
    parser.add_argument("--min-max", metavar="LOW,HIGH",
                        help="off/on levels by hand, e.g. 400,900 -- instead of "
                             "the file's step1 section, and the way a "
                             "calibration without a step 1 gets its levels")
    args = parser.parse_args(argv)
    calib_file = calib_path(args.calib)
    seed = _seed(calib_file, args.min_max)
    calib_file.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M")

    lo, hi = USABLE_BAND_NM
    print(f"Step 2: window {WINDOW_PX} px, stride {STRIDE}, levels "
          f"{seed.min_level}/{seed.max_level}, usable band "
          f"{'open' if lo is None else lo}-{'open' if hi is None else hi} nm")
    osa = connect_osa(OSA_HOST, OSA_PORT)
    try:
        slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
        try:
            clipped, swept = run_step2(
                osa, slm, MEASURE, seed,
                window_size=WINDOW_PX,
                peak_half_window_nm=PEAK_HALF_WINDOW_NM,
                region=REGION,
                coordinate_stride=STRIDE,
                sweep_span_nm=SWEEP_SPAN_NM,
                min_wavelength_nm=lo,
                max_wavelength_nm=hi,
                progress_callback=print_progress,
            )
        finally:
            slm.close_slm()
    finally:
        osa.disconnect()

    # CSV first: the map is on disk before anything else can fail.
    csv_path = write_wavelength_map_csv(
        clipped, calib_file.parent / f"step2_{stamp}.csv")
    calib = write_section(calib_file, 2, step2_section(clipped), csv=csv_path)
    print(f"\n{clipped.coordinates.size} of {swept.coordinates.size} swept coordinates "
          f"kept ({swept.wavelength.min():.3f}-{swept.wavelength.max():.3f} nm swept)")
    print(f"Saved -> {calib.path}\n         {csv_path}")


if __name__ == "__main__":
    main()
