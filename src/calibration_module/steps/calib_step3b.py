"""Step 3b -- per-channel transfer curves on the OSA, channels swept in groups.

Needs the OSA and the SLM.  ``--step2`` is required: a step-2 JSON, or a
coordinate -> wavelength CSV (its min/max levels are then ``CSV_MIN_MAX``)::

    python src/calibration_module/steps/calib_step3b.py --step2 calib_step2_0910_1737.json
    python src/calibration_module/steps/calib_step3b.py --step2 FILE --width 20 --gap 30
    python src/calibration_module/steps/calib_step3b.py --step2 FILE --out DIR

Measures the target window once to find the column the OSA puts
``TARGET_NM`` at (``FINE_TUNE_CENTER``), tiles ``CHANNELS_PER_SIDE`` mirror-
symmetric channels around it skipping ``GUARD_BANDS_NM``, then sweeps
``LEVELS`` on several non-neighbouring channels per OSA trace.  Writes
``calib_step3b_MMDD_HHMM.json`` and ``.csv``.

This is the runner only: the sweep is
:func:`calibration_module.measure.slm_response.run_step3b`, the same call the
GUI's Step 3b tab makes.  The constants below are that tab's defaults.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.slm_response import (  # noqa: E402
    CalibrationResult,
    load_calibration_result,
    load_wavelength_map_csv,
    save_calibration_result,
    write_intensity_calibration_csv,
)
from calibration_module.measure.bench import (  # noqa: E402
    connect_osa,
    connect_slm,
    print_progress,
)
from calibration_module.measure.slm_response import run_step3b  # noqa: E402
from osa_module.controller import MeasurementSettings  # noqa: E402

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"   # output directory; --out overrides it

MEASURE = MeasurementSettings(
    center_wl="778.04nm",
    span="8nm",
    sensitivity="HIGH3",
    sampling_points="AUTO",
    reference_level="10uW",
    y_unit="LINear",
)
LEVELS = list(range(350, 951, 15))
CSV_MIN_MAX = (0, 1023)          # off/on levels for a --step2 CSV, which has none

TARGET_NM = 778.04
CHANNELS_PER_SIDE = 20
WIDTH_PX = 30                    # --width overrides
GAP_PX = 15                      # --gap overrides
SKIP_BETWEEN_ACTIVE = 3          # channel pitches left dark between lit channels
FINE_TUNE_CENTER = True
CENTER_PEAK_HALF_WINDOW_NM = 0.1
AVG_WINDOW_NM: float | None = 0.1    # intensity average per channel; None = nearest samples
REFINE_WAVELENGTH = True
REFINE_HALF_WINDOW_NM = 0.2
GUARD_BANDS_NM = [(780.14, 0.06), (775.94, 0.06)]   # (centre, half-width); [] = none

OSA_HOST = "192.168.1.11"
OSA_PORT = 10001
SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch


def load_step2(path: Path) -> CalibrationResult:
    """A step-2 JSON, or a coordinate -> wavelength CSV given ``CSV_MIN_MAX``."""
    if path.suffix.lower() == ".csv":
        low, high = CSV_MIN_MAX
        return load_wavelength_map_csv(path, min_level=low, max_level=high)
    return load_calibration_result(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step2", type=Path, required=True,
                        help="step-2 JSON or coordinate/wavelength CSV")
    parser.add_argument("--width", type=int, default=WIDTH_PX,
                        help=f"channel window, px (default {WIDTH_PX})")
    parser.add_argument("--gap", type=int, default=GAP_PX,
                        help=f"dark pad between channels, px (default {GAP_PX})")
    parser.add_argument("--out", type=Path, default=CALIB_PATH,
                        help=f"output directory (default {CALIB_PATH})")
    args = parser.parse_args(argv)
    step2 = load_step2(args.step2)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M")

    print(f"Step 2 in: {args.step2}")
    print(f"Step 3b: {2 * CHANNELS_PER_SIDE} channels around {TARGET_NM:g} nm, "
          f"width {args.width} px, gap {args.gap} px, skip {SKIP_BETWEEN_ACTIVE}, "
          f"{len(LEVELS)} levels {LEVELS[0]}..{LEVELS[-1]}")
    if GUARD_BANDS_NM:
        print("Guard bands: " + ", ".join(f"{c:g}+/-{h:g} nm" for c, h in GUARD_BANDS_NM))
    osa = connect_osa(OSA_HOST, OSA_PORT)
    try:
        slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
        try:
            sweep = run_step3b(
                osa, slm, LEVELS, MEASURE, step2,
                target_wavelength_nm=TARGET_NM,
                n_channels_per_side=CHANNELS_PER_SIDE,
                channel_width_px=args.width,
                gap_px=args.gap,
                group_skip_channels=SKIP_BETWEEN_ACTIVE,
                fine_tune_center=FINE_TUNE_CENTER,
                peak_half_window_nm=CENTER_PEAK_HALF_WINDOW_NM,
                wavelength_window_nm=AVG_WINDOW_NM,
                refine_wavelength=REFINE_WAVELENGTH,
                refine_half_window_nm=REFINE_HALF_WINDOW_NM if REFINE_WAVELENGTH else None,
                guard_bands_nm=GUARD_BANDS_NM,
                progress_callback=print_progress,
            )
        finally:
            slm.close_slm()
    finally:
        osa.disconnect()

    result = sweep.result
    json_path = save_calibration_result(result, args.out / f"calib_step3b_{stamp}.json")
    csv_path = write_intensity_calibration_csv(result, args.out / f"calib_step3b_{stamp}.csv")
    print(f"\n{result.coordinates.size} channels, pitch {args.width + args.gap} px, "
          f"centre x={sweep.center_coordinate:.3f} px")
    print(f"Saved -> {json_path}\n         {csv_path}")


if __name__ == "__main__":
    main()
