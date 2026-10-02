"""Step 7 v1 -- each pair's comb phase (dPhi_comb) vs a reference.

The multi-parameter fit, superseded by :mod:`calib_step7_v2`'s one-parameter
form and kept because either version re-fits either version's CSV.

Not a unittest (needs real hardware) -- run it directly::

    python src/calibration_module/steps/calib_step7_v1.py           # sweep, then fit
    python src/calibration_module/steps/calib_step7_v1.py some.csv  # re-fit offline

In:  ``IN_STEP6``, one combined step-6 result JSON, which is the ONLY input: it
     embeds the step-3 calibration (-> channel layout) and every fitted pair
     (-> eta plus the single-beam/dark background), so the reference and all
     targets come from one file.  A positional CSV re-fits offline.
Out: one combined ``calib_step7_result_*.json`` carrying step 3 and step 6
     forward plus the fitted ``{Phi_k}`` spectrum -- what downstream consumers
     read -- and a PNG per fitted (pair, method).

A sweep fits the CSV it just wrote, so the normal run needs no second command
(``--no-fit`` stops after the CSV).

``--bounded`` / ``--fix`` pick how the step-6 amplitudes enter, on either path;
give both to run both and print a comparison table.  ``--bounded`` (the default)
locks ``a:b`` to the step-6 ``eta_ref:eta_tgt`` ratio and floats one shared
scale boxed to ``+/-BOUND_FRAC`` about 1; ``--fix`` pins ``a``/``b`` at the etas
exactly, leaving only ``dPhi_comb`` and the residual dark ``d``.

``--flip`` is for a detector that reads inverted (more light -> more negative
volts).  It negates ``voltage_mean_v`` and the per-row ``dark_v`` -- the same
channel, so the fit's ``y = mean - dark`` becomes the positive light signal.  On
a refit it writes a sibling ``*_flipped.csv`` and fits that; on a sweep the
values are negated as they are READ, so the automatic fit does not negate them
twice.  ``voltage_std_v`` and ``std_ratio`` are sign-independent and untouched.

What it measures.  Each target pair carries a fixed comb-phase offset relative
to a common reference pair, which defines ``Phi = 0``.  Driving the two at once
makes them interfere, and the fringe encodes that offset.

The drive.  The reference is held FULLY ON (``x_r = w_r = 1``); the target's two
channels are swept together (``x_t = w_t = v``) over
``SWEEP_MIN..SWEEP_MAX``.  A channel at intensity ``v`` sits at panel phase
``theta = 2 asin(sqrt(v))`` with field ``sqrt(v) exp(i theta/2)``, so
``g = sqrt(x_t w_t) = sin^2(theta/2)`` and ``dPhi_SLM = theta - pi``.  Sweeping
``v`` 0.1 -> 1.0 traces the half fringe (~37..180 deg).

The fit is in :mod:`calibration_module.fit.phase`.  Every point is reduced to
``(g, dPhi_SLM)`` from its commanded intensities, and ``dPhi_comb`` floats in::

    Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_SLM + dPhi_comb)
        + step-6 single-beam background + d

Both pairs' step-6 single-beam response is folded in as a FIXED background
either way, so the fringe need not absorb the single-beam ramp.

Each point is one fixed-duration acquisition as in step 6: ``T_SINGLE_S`` for
the all-off dark, ``T_BOTH_S`` for the sweep points, which are bright because
the reference is fully on.  Every row records the mean, its trace std and the
ratio of the two.

This file only wires up hardware and prints/plots; the model, the background
removal, the weighted fit and the persistence are all in ``fit.phase``.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.measure.bench import connect_daq, connect_slm, read_point  # noqa: E402
from calibration_module.fit.slm_response import calibration_result_from_dict  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration  # noqa: E402
from calibration_module.fit.phase import (  # noqa: E402
    PhaseFit,
    PhaseResult,
    load_pair_models,
    load_phase_csv,
    phi_half,
    save_comb_phase_json,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"          # data directory: inputs + outputs live here
REF_INDEX = 0                                      # common reference pair (Phi = 0)
TGT_INDICES = [1, 3]                            # target pairs measured vs the reference

# The ONE input: a combined step-6 result JSON (save_combined_json).  It embeds
# the raw Step-3 calibration under "step3" (-> channel layout) and every fitted
# pair under "step6" (-> eta + single-beam background), so the reference + all
# targets come from this single file -- no separate step-3 import.
IN_STEP6 = CALIB_PATH / "calib_step6_result_0829_0033.json"  # pairs 1 (ref) + 3,4,5 (targets)

# ---- The target sweep ----
# Reference fully on (x_r = w_r = 1); the target's two channels swept TOGETHER
# (x_t = w_t) over this ramp.
SWEEP_MIN = 0.1                  # min per-side target intensity in the ramp (0..1)
SWEEP_MAX = 1.0                  # max per-side target intensity in the ramp (0..1)
N_SWEEP_POINTS = 10              # points in the ramp

OUT_DIR = CALIB_PATH             # all step-7 outputs live in the data directory

SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display (like the GUI's Detect)
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch (USB link)

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

# ---- Fixed per-point acquisition (daq_module) ----
# Sample rate / range / low-pass bandwidth are the DAQMonitorSettings defaults
# (1 kS/s, +/-0.1 V DIFF, 20 Hz).  The all-off dark sits at zero signal, so it
# gets the longer T_single window; sweep points always have the reference
# fully on (bright, both pairs driven) and read T_both.  Every CSV row records
# the per-point std (voltage_std_v) and std_ratio.
T_SINGLE_S = 10.0                 # all-off dark (at most one beam on) (s)
T_BOTH_S = 10.0                   # sweep points: reference + target on (s)

SETTLE_S = 0.25                  # wait after each SLM pattern change, before reading

# Amplitude handling for the --bounded dPhi_comb fit: LOCK the ratio a:b
# (= eta_ref:eta_tgt) from step 6 and float a single shared scale s boxed to
# +/- this fraction about 1 (1.0 == s in [0, 2]), so a and b cannot diverge --
# only a common gain drift between step 6 and 7 is allowed.  report()/make_plot()
# flag when s hits its box.  The --fix method ignores this (s pinned at 1).
BOUND_FRAC = 1.0

# Fold in the step-6 single-beam response as a FIXED background.  The reference is
# held fully on -> its single-beam is a constant; only the swept target ramps.
# Keeps the fringe from having to absorb the single-beam ramp.
SINGLE_BEAM_BG = True


# ======================================================================
# input loading  (layout + step-6 models from the combined JSON)
# ======================================================================

def load_layout():
    """Channel layout from the Step-3 calibration EMBEDDED in the step-6 JSON.

    ``save_combined_json`` stores the raw step-3 payload under ``"step3"``, so
    the layout the hardware run drives is guaranteed to be the one the step-6
    etas were calibrated under.
    """
    payload = json.loads(IN_STEP6.read_text(encoding="utf-8"))
    step3 = payload.get("step3")
    if step3 is None:
        raise ValueError(
            f"{IN_STEP6} has no embedded 'step3' calibration; point IN_STEP6 at "
            f"a combined step-6 result (calib_step6_test.save_combined_json)"
        )
    layout = channel_layout_from_calibration(calibration_result_from_dict(step3))
    for name, idx in [("REF_INDEX", REF_INDEX)] + [("TGT_INDICES", k) for k in TGT_INDICES]:
        if not (0 <= idx < layout.n_channels):
            raise ValueError(
                f"{name} entry {idx} out of range (layout has {layout.n_channels} pairs)"
            )
    return layout


def load_models():
    """Load the step-6 pair models; require REF_INDEX and every TGT_INDICES entry."""
    models = load_pair_models([IN_STEP6])
    needed = [("reference", REF_INDEX)] + [("target", k) for k in TGT_INDICES]
    for role, idx in needed:
        if idx not in models:
            raise ValueError(
                f"no step-6 model for {role} pair index {idx}; found "
                f"{sorted(models)} in {IN_STEP6}"
            )
    print(f"Step 6: eta[ref {REF_INDEX}] = {models[REF_INDEX].eta:.4g} ; "
          + " ".join(f"eta[{k}]={models[k].eta:.4g}" for k in TGT_INDICES))
    return models


# ======================================================================
# report + plot
# ======================================================================

def _sigma(value: float, err: float) -> float:
    return abs(value) / err if err else float("nan")


def _bound_note(value: float, eta: float, frac: float, at_bound: bool) -> str:
    """Deviation from the step-6 eta plus an '[AT +/-100% BOUND]' warning tag."""
    if frac == 0:
        return "  (pinned to eta)"
    dev = (value / eta - 1.0) * 100.0 if eta else float("nan")
    tag = f"  [AT +/-{frac*100:.0f}% BOUND]" if at_bound else ""
    return f"  ({dev:+.0f}% vs eta {eta*1e3:.4f}){tag}"


def report(fit: PhaseFit, tgt: int, ref: int) -> None:
    """Print dPhi_comb (rad + deg), the a/b handling and fit quality."""
    print("Model:  Y = s^2 (a^2 + b^2 sin^4(theta/2) + 2ab sin^2(theta/2) cos(dPhi_comb - pi + theta))")
    print("            [both target channels swept together; theta the shared panel phase]")
    if fit.bound_frac == 0:
        print("            + step6 single-beam + d     "
              "(a, b PINNED to the step-6 etas; s = 1 fixed)")
    else:
        print("            + step6 single-beam + d     "
              f"(a:b locked to step-6 eta ratio; scale s boxed +/-{fit.bound_frac*100:.0f}%)")
    print(f"Pair {tgt} vs reference {ref}  (value +/- error):")
    print(f"  dPhi_comb = {fit.dphi_comb:+.4f} +/- {fit.dphi_comb_err:.4f} rad"
          f"   ( {fit.dphi_comb_deg:+.2f} +/- {np.degrees(fit.dphi_comb_err):.2f} deg )")
    print(f"  a (ref R_1)      = {fit.a*1e3:.4f} +/- {fit.a_err*1e3:.4f} mV^0.5"
          + _bound_note(fit.a, fit.eta_ref, fit.bound_frac, fit.a_at_bound))
    print(f"  b (tgt eta CxCw) = {fit.b*1e3:.4f} +/- {fit.b_err*1e3:.4f} mV^0.5"
          + _bound_note(fit.b, fit.eta_tgt, fit.bound_frac, fit.b_at_bound))
    print(f"  fringe amp 2ab   = {fit.amp*1e3:.4f} +/- {fit.amp_err*1e3:.4f} mV")
    print(f"  residual dark d  = {fit.offset*1e3:+.4f} +/- {fit.offset_err*1e3:.4f} mV"
          f"   (should be ~0 after per-row dark subtraction)")
    print(f"  R^2 = {fit.r2:.4f}")


def make_plot(fit: PhaseFit, tgt: int, path) -> None:
    """Measured Y(dPhi_SLM) with the fitted a/b/dPhi_comb model curve + pulls, PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    dphi = np.degrees(fit.dphi_slm)             # dPhi_SLM at the measured points
    pulls = fit.residuals / fit.std

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Smooth model over the swept geometry.  Rebuild g/dPhi_SLM the same way the
    # fit did (from the per-point commanded intensities) so the curve tracks the
    # data: sweep the shared phase over the full 0..180 deg half turn
    # (x_t = w_t = sin^2(theta/2)).
    wt_s = xt_s = np.sin(np.radians(np.linspace(0.0, 180.0, 400)) / 2.0) ** 2
    xr_c = float(np.median(fit.x_r)) if fit.x_r is not None else 1.0
    wr_c = float(np.median(fit.w_r)) if fit.w_r is not None else 1.0
    g_s = np.sqrt(np.clip(xt_s * wt_s, 0.0, None))
    dslm = phi_half(xt_s) + phi_half(wt_s) - phi_half(xr_c) - phi_half(wr_c)

    # Fixed step-6 single-beam background = fit.known - a^2 - b^2 g^2 at each
    # fitted point; interpolate it onto the smooth grid (exact at the points, and
    # geometry-agnostic, so no assumption about how the background splits in g).
    bg_pts = fit.known - fit.a**2 - fit.b**2 * fit.g**2
    order = np.argsort(fit.dphi_slm)
    bg_s = np.interp(dslm, fit.dphi_slm[order], bg_pts[order])

    model = (fit.a**2 + fit.b**2 * g_s**2
             + 2.0 * fit.a * fit.b * g_s * np.cos(dslm + fit.dphi_comb)
             + bg_s + fit.offset)
    label = r"fit: $a^2+b^2\sin^4+2ab\sin^2\cos$"
    ax1.plot(np.degrees(dslm), model * 1e3, "-", color="tab:blue", lw=1.6, label=label)
    ax1.errorbar(dphi, fit.y * 1e3, yerr=fit.std * 1e3, fmt="o", ms=5, color="tab:orange",
                 ecolor="lightgray", elinewidth=1, capsize=2, zorder=3,
                 label="measured (dark-subtracted)")
    ax1.set_xlabel(r"$\Delta\Phi_{SLM}$  (deg)")
    ax1.set_ylabel(r"$Y$, dark-subtracted  (mV)")
    ax1.set_title(f"Pair {tgt} interference  (both channels, half fringe)")
    ax1.legend(loc="best", fontsize=8)

    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.scatter(dphi, pulls, c="tab:red", s=40, edgecolor="k", lw=0.4)
    ax2.set_xlabel(r"$\Delta\Phi_{SLM}$  (deg)")
    ax2.set_ylabel("Pull = residual / std")
    ax2.set_title("Pulls")
    ax2.legend(loc="upper right", fontsize=8)

    bflag = ("  [a@bound]" if fit.a_at_bound else "") + ("  [b@bound]" if fit.b_at_bound else "")
    mode = ("a,b pinned (s=1)" if fit.bound_frac == 0
            else f"s boxed $\\pm${fit.bound_frac*100:.0f}%")
    txt = (
        f"$\\Delta\\Phi_{{comb}}$ = {fit.dphi_comb_deg:+.2f} $\\pm$ "
        f"{np.degrees(fit.dphi_comb_err):.2f} deg  "
        f"({_sigma(fit.dphi_comb, fit.dphi_comb_err):.0f}$\\sigma$)\n"
        f"a = {fit.a*1e3:.3f} ($\\eta$ {fit.eta_ref*1e3:.3f}), "
        f"b = {fit.b*1e3:.3f} ($\\eta$ {fit.eta_tgt*1e3:.3f}) mV$^{{1/2}}${bflag}\n"
        f"d = {fit.offset*1e3:+.3f} mV  (should be $\\approx$0)\n"
        f"R$^2$ = {fit.r2:.4f}  [{mode}]"
    )
    ax1.text(0.05, 0.95, txt, transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ======================================================================
# offline refit  (python calib_step7_test.py some.csv)
# ======================================================================

def _targets_in_csv(path, default) -> list[int]:
    """Distinct target-pair indices recorded in a CSV (sorted).

    A collected CSV (:func:`measure_only`) stacks every TGT_INDICES entry vs the
    shared reference in one file, so its ``tgt_index`` column lists several pairs.
    Falls back to ``default`` if the column is missing (an old single-target CSV).
    """
    seen: list[int] = []
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            t = row.get("tgt_index")
            if t in (None, ""):
                continue
            k = int(float(t))
            if k not in seen:
                seen.append(k)
    return sorted(seen) if seen else list(default)


def _flip_meas_csv(path) -> Path:
    """Write a sign-flipped copy of a raw step-7 CSV and return its path.

    ``voltage_mean_v`` and its per-row ``dark_v`` are negated -- both the same
    channel, so the fit's ``y = mean - dark`` becomes the positive light signal
    with the residual dark still near zero.  Every other column copies through:
    the spread and ``std_ratio`` are sign-independent.  Lands beside the source
    as ``<stem>_flipped.csv``.
    """
    src = Path(path)
    with open(src, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    comments = [ln for ln in lines if ln.lstrip().startswith("#")]
    data_lines = [ln for ln in lines if not ln.lstrip().startswith("#")]

    reader = csv.DictReader(data_lines)
    fields = reader.fieldnames or []
    rows = []
    for row in reader:
        for col in ("voltage_mean_v", "dark_v"):
            val = row.get(col)
            if val not in (None, ""):
                row[col] = f"{-float(val):.9g}"
        rows.append(row)

    dst = src.with_name(f"{src.stem}_flipped.csv")
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        for ln in comments:                              # carry trailing comments over
            parts = ln.lstrip("#").strip().split(",")    # negate a dark_mean_v comment too
            if len(parts) == 2 and parts[0].strip() == "dark_mean_v":
                f.write(f"# dark_mean_v,{-float(parts[1]):.9g}\n")
            else:
                f.write(ln if ln.endswith("\n") else ln + "\n")
    return dst


# Refit method -> the `frac` handed to load_phase_csv (flags --bounded / --fix)
METHODS = {
    "bounded": BOUND_FRAC,   # a:b ratio locked, shared scale s boxed +/-BOUND_FRAC
    "fix": 0.0,              # a, b pinned to the step-6 etas exactly (s = 1 fixed)
}


def _compare_methods(fits: dict[tuple[int, str], PhaseFit], pairs, methods) -> None:
    """Side-by-side dPhi_comb per method (same CSV, same step-6 models)."""
    print("\n=== Method comparison ===")
    for k in pairs:
        for m in methods:
            f = fits[(k, m)]
            s = f.a / f.eta_ref if f.eta_ref else float("nan")
            stxt = "s=1 (pinned)" if f.bound_frac == 0 else f"s={s:.3f}"
            print(f"pair {k}  {m:<7}: dPhi_comb = {f.dphi_comb_deg:+7.2f} +/- "
                  f"{np.degrees(f.dphi_comb_err):5.2f} deg   {stxt:<12}  "
                  f"R^2={f.r2:.4f}")


def fit_csv(path, *, flip: bool = False, methods: tuple[str, ...] = ("bounded",)) -> None:
    """Re-fit an already-recorded CSV offline (no hardware).

    In:  this CSV plus the combined step-6 JSON (``IN_STEP6``) for each pair's
         eta and single-beam background.  One collected file carries several
         targets against the shared reference, and every one with a step-6 model
         is fitted separately.  ``methods`` picks how the amplitudes enter
         (:data:`METHODS`); each runs on every target, and more than one prints
         a comparison table.  ``flip`` re-fits a sign-flipped sibling CSV
         (:func:`_flip_meas_csv`) instead.
    Out: one combined ``calib_step7_result_*.json`` holding every fitted
         (pair, method), plus a PNG each.
    """
    if flip:
        path = _flip_meas_csv(path)
        print(f"Flip: negated voltage_mean_v + dark_v -> re-fitting {path}")
    models = load_models()
    targets = _targets_in_csv(path, TGT_INDICES)
    fittable = [k for k in targets if k in models and k != REF_INDEX]
    if not fittable:
        raise ValueError(
            f"no fittable target in {path}: found targets {targets}, but have "
            f"step-6 models only for {sorted(models)} (reference is pair {REF_INDEX})"
        )
    print(f"Fitting pair(s) {fittable} vs reference {REF_INDEX} from {path} "
          f"(method(s): {', '.join(methods)})")
    fits: dict[tuple[int, str], PhaseFit] = {}
    for k in fittable:
        for method in methods:
            print(f"\n=== Re-fit [{method}]: pair {k} vs reference {REF_INDEX} ===")
            result = load_phase_csv(path, models[k], models[REF_INDEX],
                                    frac=METHODS[method], single_beam_bg=SINGLE_BEAM_BG,
                                    only_tgt=k)
            dts = result.per_trial_darks()
            drift = f" +/- {dts.std(ddof=1)*1e3:.4f} drift" if dts.size > 1 else ""
            print(f"Loaded {result.trial.size} rows, "
                  f"dark = {result.dark*1e3:.4f}{drift} mV")
            report(result.fit, result.tgt_index, result.ref_index)
            plot_path = OUT_DIR / f"calib_step7_pair{k}_refit_{method}.png"
            make_plot(result.fit, result.tgt_index, plot_path)
            print(f"Plot saved to {plot_path}")
            fits[(k, method)] = result.fit
    if len(methods) > 1:
        _compare_methods(fits, fittable, methods)
    # Persist the fitted spectrum {Phi_k} as ONE combined JSON (step3 + step6
    # carried over verbatim from IN_STEP6) -- the single input for downstream
    # consumers (e.g. the step-8 random-input forward-model check).
    out_json = OUT_DIR / f"calib_step7_result_{time.strftime('%m%d_%H%M')}.json"
    save_comb_phase_json(fits, IN_STEP6, out_json, ref_index=REF_INDEX,
                         csv_path=str(Path(path).resolve()),
                         single_beam_bg=SINGLE_BEAM_BG)
    print(f"\nCombined step-7 result (step3 + step6 + step7) saved to {out_json}")


# ======================================================================
# collect  (python calib_step7_test.py  ->  drive SLM, record raw CSV, no fit)
# ======================================================================

def build_xw_sweep() -> list[tuple[float, float, float, float]]:
    """Drive tuples: reference fully on, the target's two channels swept together.

    Returns target-first commanded-intensity tuples ``(x_t, w_t, x_r, w_r)`` with
    ``x_r = w_r = 1`` and ``x_t = w_t`` stepping over the
    ``SWEEP_MIN..SWEEP_MAX`` ramp.  Raw data only -- the fit is done later,
    offline.
    """
    values = np.round(np.linspace(SWEEP_MIN, SWEEP_MAX, N_SWEEP_POINTS), 6)
    return [(float(v), float(v), 1.0, 1.0) for v in values]


_MEAS_CSV_HEADER = [
    "trial", "tgt_index", "ref_index",
    "phi_xt_deg", "phi_wt_deg", "x_t", "w_t", "x_r", "w_r",
    "dark_v", "voltage_mean_v", "voltage_std_v", "std_ratio",
]


def write_meas_csv(results, path) -> str:
    """Write raw rows for one or more target pairs into a single CSV.

    Same column layout as :func:`tpa_phase.write_phase_csv` plus a trailing
    ``std_ratio`` (std/|mean|) column, and concatenates several
    :class:`PhaseResult` objects so every row carries its own ``tgt_index`` (and
    the shared ``ref_index``) -- i.e. REF_INDEX and every TGT_INDICES entry are
    recorded in the file, per row.  Round-trips via
    :func:`tpa_phase.load_phase_csv` (used by the offline refit).
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_MEAS_CSV_HEADER)
        for result in results:
            for t, x_t, w_t, x_r, w_r, dark_v, mean_v, std_v in zip(
                result.trial, result.x_t, result.w_t, result.x_r, result.w_r,
                result.dark_v, result.voltage_mean_v, result.voltage_std_v,
            ):
                phi_xt = np.degrees(2.0 * float(phi_half(x_t)))
                phi_wt = np.degrees(2.0 * float(phi_half(w_t)))
                ratio = abs(std_v / mean_v) if mean_v else float("inf")
                writer.writerow(
                    [int(t), result.tgt_index, result.ref_index,
                     f"{phi_xt:.4g}", f"{phi_wt:.4g}",
                     f"{x_t:.6g}", f"{w_t:.6g}", f"{x_r:.6g}", f"{w_r:.6g}",
                     f"{dark_v:.9g}", f"{mean_v:.9g}", f"{std_v:.9g}",
                     f"{ratio:.6g}"]
                )
    return str(out)


def _read_point(daq, x_t: float, w_t: float, x_r: float, w_r: float) -> tuple[float, float, float]:
    """One fixed-duration DAQ read for a drive point -> ``(mean, std, duration)``.

    Any channel on reads ``T_BOTH_S`` -- sweep points are bright, the reference
    being fully on -- and the all-off dark reads ``T_SINGLE_S``.  ``std`` is the
    spread of the low-passed trace, recorded verbatim as the per-point sigma the
    fit weights by.
    """
    single = not any(v > 0.0 for v in (x_t, w_t, x_r, w_r))
    mean_v, std_v = read_point(daq, single=single)
    return mean_v, std_v, (T_SINGLE_S if single else T_BOTH_S)


def _measure_target(slm, daq, layout, k: int, drive, *, flip: bool = False) -> PhaseResult:
    """Drive pair ``k`` vs REF_INDEX over ``drive`` and read Y.  No fit.

    Only ``k`` and ``REF_INDEX`` are driven, everything else held off.  The
    all-off dark is read once at the start and stored per row, for per-row
    subtraction.  Needs no step-6 model -- raw data only.

    ``flip`` negates the raw mean and dark as read; the std is a magnitude and
    stays as it was.
    """
    from slm_module.encoding import encode_to_pattern

    n = layout.n_channels
    zeros = np.zeros(n)
    slm_width, slm_height = slm.get_slm_info()

    def _display(x_t, w_t, x_r, w_r) -> None:
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        x_vals[k], w_vals[k] = x_t, w_t
        x_vals[REF_INDEX], w_vals[REF_INDEX] = x_r, w_r
        slm.display_array(encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height))
        if SETTLE_S:
            time.sleep(SETTLE_S)

    total = len(drive) + 1
    step = 0
    rows: list[tuple] = []
    _display(0.0, 0.0, 0.0, 0.0)                     # all-off dark, once
    dark_v, _, dur = _read_point(daq, 0.0, 0.0, 0.0, 0.0)
    if flip:
        dark_v = -dark_v                             # inverted DAQ sign (same channel)
    step += 1
    print(f"[{step}/{total}] pair {k} dark (all off, {dur:.0f}s) "
          f"= {dark_v*1000:.4f} mV")
    for x_t, w_t, x_r, w_r in drive:
        _display(x_t, w_t, x_r, w_r)
        mean_v, std_v, dur = _read_point(daq, x_t, w_t, x_r, w_r)
        if flip:
            mean_v = -mean_v
        rows.append((0, x_t, w_t, x_r, w_r, mean_v, std_v, dark_v))
        step += 1
        ratio = abs(std_v / mean_v) if mean_v else float("inf")
        print(f"[{step}/{total}] pair {k} x=w={x_t:.3f} ({dur:.0f}s) "
              f"-> {mean_v*1000:.4f} mV std ratio {ratio*100:.2f}%")

    return PhaseResult(
        tgt_index=k, ref_index=REF_INDEX,
        trial=np.array([r[0] for r in rows], dtype=int),
        x_t=np.array([r[1] for r in rows], dtype=float),
        w_t=np.array([r[2] for r in rows], dtype=float),
        x_r=np.array([r[3] for r in rows], dtype=float),
        w_r=np.array([r[4] for r in rows], dtype=float),
        voltage_mean_v=np.array([r[5] for r in rows], dtype=float),
        voltage_std_v=np.array([r[6] for r in rows], dtype=float),
        dark_v=np.array([r[7] for r in rows], dtype=float),
        n_trials=1,
    )


def measure_only(*, flip: bool = False) -> Path:
    """Sweep every target pair vs the shared reference; write one raw CSV.

    Every TGT_INDICES entry against REF_INDEX, the reference fully on and the
    target's two channels swept together over the ramp.  All rows go to one
    timestamped CSV, each tagged with its ``tgt_index`` and ``ref_index``.  Raw
    data only -- no step-6 models, no fit.

    Out: the CSV path, which :func:`main` fits straight away unless ``--no-fit``.

    ``flip`` negates each raw mean/dark as it is read, so the written CSV
    already holds the positive light signal and no later refit needs the flag.
    """
    layout = load_layout()
    if flip:
        print("Flip: negating voltage_mean_v + dark_v as read (inverted DAQ sign).")

    drive = build_xw_sweep()
    values = [x_t for x_t, _, _, _ in drive]
    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=T_BOTH_S, t_single=T_SINGLE_S)
    results = []
    try:
        for k in TGT_INDICES:
            print(f"\n=== Sweep: pair {k} vs reference {REF_INDEX}  "
                  f"(x{REF_INDEX}=w{REF_INDEX}=1, sweep x{k}=w{k} over "
                  f"{values}) ===")
            results.append(_measure_target(slm, daq, layout, k, drive, flip=flip))
    finally:
        slm.close_slm()
        daq.disconnect()

    csv_path = OUT_DIR / f"calib_step7_meas_{time.strftime('%m%d_%H%M')}.csv"
    write_meas_csv(results, csv_path)
    print(f"\nCSV (ref {REF_INDEX}, targets {TGT_INDICES}) written to {csv_path}")
    return Path(csv_path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flip = "--flip" in argv     # inverted DAQ read -> negate voltage_mean_v + dark_v
    # fit method flag(s): --bounded / --fix (both -> run both + comparison table)
    methods = tuple(m for m in METHODS if f"--{m}" in argv) or ("bounded",)
    positional = [a for a in argv if not a.startswith("-")]
    if positional:              # a CSV path -> offline refit, no hardware
        fit_csv(positional[0], flip=flip, methods=methods)
        return 0

    csv_path = measure_only(flip=flip)   # no arg -> fresh sweep (drives the SLM/DAQ)
    if "--no-fit" in argv:               # raw data only; fit by hand later
        print(f"Fit with:  python {Path(__file__).name} {csv_path}")
        return 0
    # Fit what was just collected, so no second command is needed.  flip=False:
    # a --flip COLLECT already negated the reads before writing the CSV.
    try:
        fit_csv(csv_path, flip=False, methods=methods)
    except Exception as exc:             # the raw CSV is on disk either way
        print(f"\nAuto-fit FAILED ({type(exc).__name__}: {exc})")
        print(f"Data is safe -- refit with:  python {Path(__file__).name} {csv_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
