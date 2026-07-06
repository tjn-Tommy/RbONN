"""Quick Step-3 intensity calibration for exactly TWO pairs (4 channels).

Most of Step 3 sweeps every coordinate in the Step-2 map.  Here we only want the
four SLM columns that make up two TPA pairs, so we hand ``intensity_calibration``
a mapping that contains *only those four columns* (built from the Step-2
wavelength <-> pixel map in ``calib_step222.json``).  Step 3 then measures the
level -> intensity transfer curve at exactly those 4 columns -- fast.

Geometry (defaults match ``build_channel_layout``: center 778 nm, pitch 20 px =
15 px channel + 5 px gap):

    * Pair A ("near"):  layout pair index 0  -> cols center_x +/- 10
        - straddles 778.0000 nm but each 15 px window is kept CLEAR of the
          exact 778.0000 nm column (asserted below).
    * Pair B ("far"):   layout pair index 3  -> cols center_x +/- 70
        - every channel is >= MIN_SEPARATION_PX from Pair A's channels, so the
          bright bars never overlap (no crosstalk).

Because both pairs land on the default layout grid, a later
``build_channel_layout(calib)`` snaps pair 0 and pair 3 straight onto the
measured curves -- use ``REF_INDEX = 0`` and ``TGT_INDICES = [3]`` in the TPA
steps.

Run::

    python tests/step3_two_pairs_calibration.py            # measure (needs OSA + SLM)
    python tests/step3_two_pairs_calibration.py --dry-run  # print geometry only, no hardware
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from osa_module.controller import MeasurementSettings, OSAController  # noqa: E402
from slm_module.calibration.calibration_new import (  # noqa: E402
    CalibrationResult,
    intensity_calibration,
    load_calibration_result,
    save_calibration_result,
    write_intensity_calibration_csv,
)
from slm_module.controller import SLMController  # noqa: E402

# ---------------------------------------------------------------- edit me ----
STEP2_PATH = REPO_ROOT / "calib_step222.json"   # Step-2 wl<->px map (the source)

# --- geometry (defaults == build_channel_layout defaults) ---
CENTER_WL = 778.0          # nm; the column the pairs straddle
WINDOW_PX = 15             # bright bar width per channel (== channel_width_px)
PITCH_PX = 20              # channel_width_px (15) + gap_px (5)
NEAR_PAIR_INDEX = 0        # Pair A: closest pair to CENTER_WL
FAR_PAIR_INDEX = 3         # Pair B: first grid pair that clears MIN_SEPARATION_PX
MIN_SEPARATION_PX = 45     # required gap (center-to-center) between the two pairs

# --- OSA settings (requirement 1) ---
OSA_HOST = "192.168.1.11"
OSA_PORT = 10001
OSA_SENSITIVITY = "HIGH2"
OSA_REF_LEVEL = "4uW"
OSA_CENTER_WL = "778nm"    # wide dark/bright reference traces are taken here...
OSA_SPAN = "4nm"           # ...spanning all four channel wavelengths

# --- level sweep (the transfer curve) ---
LEVEL_START, LEVEL_STOP, LEVEL_STEP = 400, 900, 5

# --- narrow per-coordinate acquisition (keeps it quick) ---
SWEEP_SPAN_NM = 0.5        # narrow OSA span re-centered on each channel's wl
AVG_NM = 0.1              # nm averaging window around each wavelength
REFINE_WAVELENGTH = False  # keep the Step-2 linear map (only 4 non-uniform cols)

# --- SLM ---
SLM_DISPLAY_NO = None      # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1             # SLM_Ctrl_* device index for the DVI-mode switch

# --- outputs ---
OUT_JSON = REPO_ROOT / "calib_step3_2pairs.json"
OUT_CSV = REPO_ROOT / "calibration_2pairs.csv"
# -----------------------------------------------------------------------------


def channel_plan() -> dict:
    """Compute the 4 channel columns + wavelengths from the Step-2 map and check
    all three constraints.  Raises AssertionError if any constraint is violated.
    """
    step2 = load_calibration_result(STEP2_PATH)
    coords = np.asarray(step2.coordinates, dtype=float)
    wls = np.asarray(step2.wavelength, dtype=float)
    a, b = np.polyfit(coords, wls, 1)                 # wl = a*col + b   (a < 0)
    center_x = (CENTER_WL - b) / a                    # column of exactly 778.0000

    half = WINDOW_PX // 2                             # window = [c-half, c-half+WINDOW-1]

    def channels_of(pair_index: int) -> tuple[int, int]:
        off = (pair_index + 0.5) * PITCH_PX
        x_col = int(round(center_x - off))            # wl > center (lower column)
        w_col = int(round(center_x + off))            # wl < center (higher column)
        return x_col, w_col

    near = channels_of(NEAR_PAIR_INDEX)
    far = channels_of(FAR_PAIR_INDEX)

    def window(col: int) -> tuple[int, int]:
        start = col - half
        return start, start + WINDOW_PX - 1           # inclusive [start, end]

    # --- constraint 2: neither NEAR window may contain the 778.0000 column ---
    # (wl is monotonic in column, so "window excludes 778.0000 nm" <=>
    #  "window excludes the center_x pixel".)
    for col in near:
        lo, hi = window(col)
        assert not (lo <= center_x <= hi), (
            f"near channel col {col} window [{lo},{hi}] contains the "
            f"{CENTER_WL:.4f} nm column ({center_x:.2f}); move the pair out."
        )

    # --- constraint 3: FAR channels >= MIN_SEPARATION_PX from every NEAR one ---
    for fc in far:
        for nc in near:
            assert abs(fc - nc) >= MIN_SEPARATION_PX, (
                f"far channel col {fc} is only {abs(fc - nc)} px from near "
                f"channel col {nc} (need >= {MIN_SEPARATION_PX})."
            )

    cols = sorted([*near, *far])                      # measure ascending by column
    plan_wls = [float(a * c + b) for c in cols]
    return {
        "a": float(a), "b": float(b), "center_x": float(center_x),
        "near": near, "far": far, "cols": cols, "wls": plan_wls,
        "window": window, "half": half,
        "min_level": step2.min_level, "max_level": step2.max_level,
    }


def print_plan(plan: dict) -> None:
    print(f"Step-2 fit: wl = {plan['a']:.6g}*col + {plan['b']:.5f}")
    print(f"{CENTER_WL:.4f} nm sits at column {plan['center_x']:.2f}")
    print(f"min/max level (from Step 2): {plan['min_level']} / {plan['max_level']}")
    labels = {plan["near"][0]: "A x", plan["near"][1]: "A w",
              plan["far"][0]: "B x", plan["far"][1]: "B w"}
    print(f"\n{'chan':<6}{'col':>6}{'window(px)':>14}{'wl (nm)':>12}")
    for col, wl in zip(plan["cols"], plan["wls"]):
        lo, hi = plan["window"](col)
        print(f"pair {labels[col]:<3}{col:>6}{f'[{lo},{hi}]':>14}{wl:>12.4f}")
    ax, aw = plan["near"]
    print(f"\nPair A width across channels : {abs(aw - ax)} px")
    print(f"Pair A<->B nearest gap       : "
          f"{min(abs(f - n) for f in plan['far'] for n in plan['near'])} px "
          f"(need >= {MIN_SEPARATION_PX})")
    print("All constraints satisfied.")


def build_mapping(plan: dict) -> CalibrationResult:
    """A CalibrationResult holding ONLY the 4 channel columns to measure."""
    return CalibrationResult(
        wavelength=np.asarray(plan["wls"], dtype=float),
        coordinates=np.asarray(plan["cols"], dtype=float),
        max_level=plan["max_level"],
        min_level=plan["min_level"],
        level_range=np.asarray([], dtype=int),        # unused as Step-3 input
        wavelength_fit_coefficients=np.asarray([plan["a"], plan["b"]], dtype=float),
    )


def levels() -> list[int]:
    vals = list(range(LEVEL_START, LEVEL_STOP + 1, LEVEL_STEP))
    if not vals:
        vals = [LEVEL_START]
    if vals[-1] != LEVEL_STOP:
        vals.append(LEVEL_STOP)
    return vals


def detect_slm_display() -> int:
    probe = SLMController(display_no=1)
    for display_no, width, height, name in probe.detect_displays():
        print(f"  display {display_no}: {width}x{height} ({name})")
        if name.startswith("LCOS-SLM"):
            return display_no
    raise RuntimeError("No LCOS-SLM display found; set SLM_DISPLAY_NO manually.")


def connect_slm() -> SLMController:
    display_no = SLM_DISPLAY_NO if SLM_DISPLAY_NO is not None else detect_slm_display()
    slm = SLMController(display_no=display_no)
    slm.open_slm()
    width, height = slm.get_slm_info()
    print(f"SLM: display {display_no} ({width}x{height})")
    slm.set_dvi_mode(USB_SLM_NO)
    return slm


def connect_osa() -> OSAController:
    osa = OSAController(host=OSA_HOST, port=OSA_PORT)
    osa.connect()
    print(f"OSA: connected ({osa.identify().strip()})")
    return osa


def run() -> None:
    plan = channel_plan()
    print_plan(plan)

    settings = MeasurementSettings(
        center_wl=OSA_CENTER_WL, span=OSA_SPAN,
        sensitivity=OSA_SENSITIVITY, reference_level=OSA_REF_LEVEL,
        y_unit="LINear",
    )
    mapping = build_mapping(plan)
    lv = levels()
    print(f"\nMeasuring {len(plan['cols'])} channels x {len(lv)} levels "
          f"({LEVEL_START}..{LEVEL_STOP} step {LEVEL_STEP})\n")

    def report(p) -> None:  # p: CalibrationProgress
        print(f"[{p.step + 1}/{p.total}] {p.message}")

    slm = connect_slm()
    osa = connect_osa()
    stop_event = threading.Event()
    try:
        result = intensity_calibration(
            osa, slm, lv, settings, mapping,
            window_size=WINDOW_PX,
            wavelength_window_nm=AVG_NM,
            sweep_span_nm=SWEEP_SPAN_NM,
            coordinate_stride=1,
            refine_wavelength=REFINE_WAVELENGTH,
            region=None,
            stop_event=stop_event,
            progress_callback=report,
        )
    finally:
        osa.disconnect()
        slm.close_slm()

    save_calibration_result(result, OUT_JSON)
    csv_path = write_intensity_calibration_csv(result, OUT_CSV)
    print(f"\nSaved JSON -> {OUT_JSON}")
    print(f"Saved CSV  -> {csv_path}")
    print("\nUse in the TPA steps:  CALIB_PATH = calib_step3_2pairs.json, "
          "REF_INDEX = 0, TGT_INDICES = [3]")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("--dry-run", "-n"):
        print_plan(channel_plan())
        return 0
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
