"""Draft: PMT relative sensitivity vs its gain-control voltage.

    S="--step3 src/calib_data/calib_step3c_0928_1823.json"
    python src/drafts/pmt_gain_sweep.py $S                 # the ~43 min sweep
    python src/drafts/pmt_gain_sweep.py $S --pair 10       # which pair drives it
    python src/drafts/pmt_gain_sweep.py $S --dry-run       # resolve everything, touch no hardware
    python src/drafts/pmt_gain_sweep.py some.csv           # re-plot an earlier run offline

Three instruments, no fitting: the ``psu_module`` walks the PMT's gain-control
input from :data:`V_START` to :data:`V_STOP` in :data:`N_POINTS` steps, the
``slm_module`` holds one channel pair at full drive, and the ``daq_module``
reads the TIA at each step.  The product is a curve of relative sensitivity --
signal normalised to its value at :data:`V_REF` -- against control voltage.

Why the SLM is here at all
--------------------------
The control voltage does not only scale the light signal: PMT dark current is
amplified by the same gain, so the dark level climbs with V right alongside the
signal.  A raw-mean ratio would therefore overstate the low-gain end by
whatever the dark happens to be there.  So every control voltage is measured
twice -- once with the SLM blanked (dark) and once with pair :data:`PAIR_INDEX`
at ``(x, w) = (DRIVE_X, DRIVE_W)`` -- and the recorded signal is the
difference.  That is the same all-off dark point the step-6 grid carries, taken
at each gain rather than once per run.

Per control voltage, in order::

    VSET -> V        SLM all-off   0.25 s   read T_MEAS_S   (10 s)
                     SLM pair on   0.25 s   read T_MEAS_S   (10 s)
    net = signal - dark

The dark read is itself 15 s of settling at the new gain before the signal
point is taken, so :data:`SETTLE_S` only ever has to cover the SLM frame
change -- the same 0.25 s the step-6 runs validated.

Input range
-----------
One range for the whole sweep, :data:`RANGE_V`, dark and signal alike.

This used to escalate to +/-0.2 V on a near-rail raw peak and then latch there
for the rest of the run.  It was never about the signal: the 0928 sweep fired
it at point 43 with a 0.236 mV mean and a 0.056 mV spread, against a 95 mV
threshold -- 400x the mean and 1700x the std.  Nothing was near the rail; the
raw trace took a single excursion three orders above its own spread, i.e. a
spike, and the sweep then ran its remaining 58 points on a range it had no
reason to be on.  Spikes are not signal and there is nothing here to protect
against clipping: this drive reads single-digit millivolts at the top of the
gain curve.

``measure.pair_v2.read_point`` still checks the raw peak and still says so in
the log -- it just does not act on it.

A single pass, brightest last
-----------------------------
No reference revisits and no descending return pass: any laser or gain drift
over the ~43 min run lands in this curve's slope with nothing to separate it
from real sensitivity.  That is a deliberate choice for a first look, not an
oversight -- if the curve matters quantitatively, run it twice and compare.

Needs real hardware (PSU on COM4, DAQ Dev1, SLM on its own display).  Ctrl+C
writes out whatever points are already measured.
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.pair_v2 import PairV2Config  # noqa: E402
from calibration_module.fit.sigma import STD_FLOOR_V  # noqa: E402
from calibration_module.fit.slm_response import load_calibration_result  # noqa: E402
from calibration_module.measure.bench import connect_daq, connect_slm  # noqa: E402
from calibration_module.measure.pair_v2 import (  # noqa: E402
    PairV2Acq,
    read_point,
)
from psu_module import PMT_CONTROL_I_MAX, PMT_CONTROL_V_MAX, PSUController  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration, encode_to_pattern  # noqa: E402

OUT_DIR = REPO_ROOT / "src/calib_data"

# ---- the sweep ----
V_START = 0.5                    # gain-control voltage at the first point (V)
V_STOP = PMT_CONTROL_V_MAX       # ... and at the last one; the module's 1.1 V ceiling
N_POINTS = 100
V_REF = 1.0                      # sensitivity is reported relative to the point
                                 # nearest this voltage (the usual operating gain)

# ---- what drives the signal ----
# Pair numbering is step-6 v2's: 1..N, so "pair 10" is the tenth pair of the
# Step-3 calibration.  The pattern is encoded exactly as steps 6/7/8 encode it,
# so this curve is measured through the aperture those use.
PAIR_INDEX = 10
DRIVE_X = 1.0
DRIVE_W = 1.0
CONFIG = PairV2Config(pair_index_base=1)

# ---- acquisition ----
# One window for both reads of a point, so the dark and the signal it is
# subtracted from average the same noise -- the calibration steps' rule.  This
# draft keeps its own +/-0.1 V rather than the bench's 1 V: it deliberately
# measures down into the tens of microvolts at the bottom of the gain curve,
# where it wants every bit of the board's absolute accuracy, and its brightest
# read is single-digit millivolts, so nothing can clip either way.
T_MEAS_S = 10.0
SETTLE_S = 0.25
RANGE_V = 0.1
ACQ = PairV2Acq(
    t_meas_s=T_MEAS_S,
    settle_s=SETTLE_S,
    range_v=RANGE_V,
    invert=True,
)

PSU_PORT = "COM4"
DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"
SLM_DISPLAY_NO = None             # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1                    # SLM_Ctrl_* device index for the DVI-mode switch

COLUMNS = [
    "index", "v_control_v", "range_v",
    "dark_mean_v", "dark_std_v",
    "sig_mean_v", "sig_std_v",
    "net_v", "net_std_v",
    "elapsed_s",
]


# ======================================================================
# inputs
# ======================================================================

def load_layout(step3: Path):
    """Step-3 calibration -> channel layout, with PAIR_INDEX validated.

    Loaded verbatim through the same ``channel_layout_from_calibration`` the GUI
    encoding page uses, so the pair number here means what it means in the UI.
    """
    step3 = Path(step3)
    if not step3.is_file():
        raise FileNotFoundError(f"Step-3 calibration not found: {step3}")
    layout = channel_layout_from_calibration(
        load_calibration_result(step3)
    )
    slot = CONFIG.slot(PAIR_INDEX)
    if not 0 <= slot < layout.n_channels:
        raise ValueError(
            f"pair {PAIR_INDEX} is out of range: the layout has "
            f"{layout.n_channels} pairs, numbered "
            f"{CONFIG.pair_index_base}..{layout.n_channels - 1 + CONFIG.pair_index_base}"
        )
    return layout


def describe_pair(layout) -> str:
    """One line naming the pair actually driven -- a wrong --pair shows here."""
    slot = CONFIG.slot(PAIR_INDEX)
    x_ch, w_ch = layout.x_channels[slot], layout.w_channels[slot]
    return (f"pair {PAIR_INDEX} (slot {slot}): "
            f"x {x_ch.wavelength_nm:.3f} nm @ col {int(x_ch.x_center)}, "
            f"w {w_ch.wavelength_nm:.3f} nm @ col {int(w_ch.x_center)}")


def control_voltages() -> np.ndarray:
    """The sweep, rounded to the supply's 1 mV setpoint resolution.

    Rounding is not cosmetic: ``linspace`` lands the last point a float hair
    above 1.1 V, and ``PSUController`` raises on a request over its cap rather
    than clamping it -- so an unrounded sweep dies on its final point after
    40 minutes.
    """
    return np.round(np.linspace(V_START, V_STOP, N_POINTS), 3)


def run_seconds() -> float:
    return N_POINTS * 2 * (SETTLE_S + T_MEAS_S)


# ======================================================================
# hardware
# ======================================================================

def connect_psu() -> PSUController:
    """Open the supply under the PMT caps, and report what it is already doing."""
    psu = PSUController(port=PSU_PORT, max_voltage=PMT_CONTROL_V_MAX,
                        max_current=PMT_CONTROL_I_MAX)
    psu.connect()
    state = psu.read_state()
    print(f"PSU: {psu.identify()} on {PSU_PORT} -- "
          f"VSET {state.vset:.3f} V, ISET {state.iset:.3f} A, "
          f"OVP {state.ovp:.3f} V, output {'ON' if state.output_on else 'off'}")
    if state.ovp <= V_STOP:
        raise RuntimeError(
            f"OVP is {state.ovp:.3f} V but the sweep ends at {V_STOP:.3f} V -- "
            f"the supply would trip mid-run. Raise it (1.15 V is the bench "
            f"setting) from the GUI's PMT Gain page or with set_ovp()."
        )
    return psu


def arm_output(psu: PSUController) -> None:
    """Bring the supply to V_START and make sure the output is live.

    V_START goes in *before* the output is enabled, so the PMT never sees
    whatever the front panel was left at.  Enabling energises the detector, so
    an output found off is confirmed at the keyboard -- the script-side version
    of the GUI page's confirm dialog.
    """
    psu.set_current(PMT_CONTROL_I_MAX)
    psu.set_voltage(V_START)
    if psu.is_output_on():
        print(f"PSU: output already on, now at {V_START:.3f} V")
        return
    print(f"\nPSU: output is OFF. Enabling it feeds {V_START:.3f} V to the PMT "
          f"gain input.")
    input("      press Enter to enable, or Ctrl+C to abort: ")
    psu.enable()
    print(f"PSU: output ON at {V_START:.3f} V")


def restore(psu: PSUController, vset: float, iset: float, output_on: bool) -> None:
    """Put the supply back exactly as it was found, output state included."""
    try:
        psu.set_voltage(vset)
        psu.set_current(iset)
        if psu.is_output_on() != output_on:
            psu.set_output(output_on)
        print(f"PSU: restored VSET {vset:.3f} V, ISET {iset:.3f} A, "
              f"output {'ON' if output_on else 'off'}")
    except Exception as exc:          # never let cleanup mask the real error
        print(f"PSU: ** restore failed: {exc} -- check the front panel **")


def show(slm, layout, pattern_x: float, pattern_w: float) -> None:
    """Drive pair PAIR_INDEX at (x, w); every other channel off."""
    slot = CONFIG.slot(PAIR_INDEX)
    zeros = np.zeros(layout.n_channels)
    x_vals, w_vals = zeros.copy(), zeros.copy()
    x_vals[slot] = pattern_x
    w_vals[slot] = pattern_w
    width, height = slm.get_slm_info()
    slm.display_array(encode_to_pattern(x_vals, w_vals, layout, width, height))


# ======================================================================
# the sweep
# ======================================================================

def sweep(psu: PSUController, daq, slm, layout) -> list[dict]:
    """Walk the control voltage, returning one raw row per point.

    Rows are returned even on Ctrl+C, so a 40-minute run interrupted at point
    80 still yields 80 usable points.
    """
    voltages = control_voltages()
    acq = ACQ
    t0 = time.time()
    rows: list[dict] = []

    for i, v in enumerate(voltages, start=1):
        psu.set_voltage(float(v))          # verified readback; raises if it did not take

        show(slm, layout, 0.0, 0.0)        # dark first: also settles the new gain
        time.sleep(acq.settle_s)
        dark, dark_std, dark_rng = read_point(
            daq, single=True, acq=acq, log=print)

        show(slm, layout, DRIVE_X, DRIVE_W)
        time.sleep(acq.settle_s)
        sig, sig_std, rng = read_point(
            daq, single=False, acq=acq, log=print)

        net = sig - dark
        net_std = float(np.hypot(sig_std, dark_std))
        rows.append({
            "index": i, "v_control_v": float(v), "range_v": max(rng, dark_rng),
            "dark_mean_v": dark, "dark_std_v": dark_std,
            "sig_mean_v": sig, "sig_std_v": sig_std,
            "net_v": net, "net_std_v": net_std,
            "elapsed_s": time.time() - t0,
        })
        ratio = abs(net_std / net) * 100.0 if net else float("inf")
        print(f"[{i}/{len(voltages)}] {v:.3f} V -> net {net*1e3:8.4f} mV "
              f"(dark {dark*1e3:7.4f}, sig {sig*1e3:8.4f}) "
              f"+/-{net_std*1e3:.4f} mV  {ratio:5.2f}%"
              )

    return rows


# ======================================================================
# output
# ======================================================================

def write_csv(rows: list[dict], path: Path, meta: dict) -> None:
    """Raw rows plus a ``# key,value`` provenance header.

    Only measured quantities go in: relative sensitivity is derived at plot
    time, so re-normalising to a different V_REF never means re-measuring.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        for key, value in meta.items():
            fh.write(f"# {key},{value}\n")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in COLUMNS})


def read_csv(path: Path) -> tuple[list[dict], dict]:
    """Load a sweep CSV back, meta included, for an offline re-plot."""
    meta: dict[str, str] = {}
    data_lines: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].strip().partition(",")
            meta[key] = value
        elif line.strip():
            data_lines.append(line)
    rows = [
        {k: (int(v) if k == "index" else float(v)) for k, v in row.items()}
        for row in csv.DictReader(data_lines)
    ]
    return rows, meta


def relative(rows: list[dict], v_ref: float = V_REF) -> tuple[np.ndarray, ...]:
    """(V, net, net_std, rel, rel_std, v_at_ref) -- sensitivity vs the ref point.

    The reference is the measured point nearest ``v_ref``, not an interpolation:
    normalising by a number that was actually read keeps the ratio's error
    honest.  Its own error propagates into every other point, which is why
    ``rel_std`` at the reference is not zero.
    """
    v = np.array([r["v_control_v"] for r in rows], dtype=float)
    net = np.array([r["net_v"] for r in rows], dtype=float)
    std = np.array([r["net_std_v"] for r in rows], dtype=float)
    k = int(np.argmin(np.abs(v - v_ref)))
    ref, ref_std = net[k], std[k]
    if ref == 0.0:
        raise ValueError(f"the reference point at {v[k]:.3f} V measured 0 V")
    rel = net / ref
    # ratio of two measured levels: fractional errors in quadrature.  A point
    # that measured exactly 0 V has no fractional error to speak of, so its own
    # term drops out and only the reference's remains.
    denom = np.where(net != 0.0, np.abs(net), np.inf)
    rel_std = np.abs(rel) * np.hypot(std / denom, ref_std / abs(ref))
    return v, net, std, rel, rel_std, v[k]


def report(rows: list[dict]) -> None:
    v, net, std, rel, _, v_ref = relative(rows)
    dark = np.array([r["dark_mean_v"] for r in rows], dtype=float)
    positive = net > 0
    ref_level = float(net[int(np.argmin(np.abs(v - v_ref)))])
    print(f"\nreference   : {v_ref:.3f} V -> {ref_level*1e3:.4f} mV")
    print(f"net range   : {net.min()*1e3:.4f} .. {net.max()*1e3:.4f} mV")
    if positive.sum() >= 2:
        print(f"rel. span   : {rel[positive].max() / rel[positive].min():.1f}x "
              f"over {v.min():.3f}-{v.max():.3f} V")
    # "Near the floor" is about the SIGNAL, not its trace spread: a point whose
    # net sits within a few multiples of the systematic floor cannot be trusted
    # to better than tens of percent however quiet its own trace was.
    weak = np.abs(net) < 5.0 * STD_FLOOR_V
    print(f"near floor  : {int(weak.sum())}/{len(v)} points with |net| under 5x "
          f"the {STD_FLOOR_V*1e3:.3f} mV systematic floor")
    n_dark = int(np.sum(dark >= net))
    if n_dark:
        print(f"** {n_dark} point(s) where the dark equals or exceeds the signal -- "
              f"the low-gain end is not resolved **")


def make_plot(rows: list[dict], path: Path, meta: dict | None = None) -> None:
    """Two stacked panels: the measured levels, then relative sensitivity.

    Both are log-y because PMT gain runs over decades in this voltage span.  The
    dark trace shares the top panel on purpose: where it meets the net signal is
    exactly where the bottom panel stops meaning anything.
    """
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg",):
        matplotlib.use("Agg")     # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    v, net, std, rel, rel_std, v_ref = relative(rows)
    dark = np.array([r["dark_mean_v"] for r in rows], dtype=float)

    fig, (ax_lvl, ax_rel) = plt.subplots(
        2, 1, figsize=(8.5, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]})

    ok = net > 0
    ax_lvl.errorbar(v[ok], net[ok] * 1e3, yerr=std[ok] * 1e3, fmt="o-", ms=3,
                    lw=1, color="tab:blue", label="net (signal - dark)")
    if (~ok).any():
        # Mark them just under the lowest real level rather than at a fixed
        # value: parking them at 1e-3 mV stretches a log axis over two empty
        # decades and squashes the data that matters into the top strip.
        floor_mark = 0.4 * min(net[ok].min() if ok.any() else STD_FLOOR_V,
                               STD_FLOOR_V)
        ax_lvl.plot(v[~ok], np.full((~ok).sum(), floor_mark * 1e3), "x",
                    color="tab:red", label="net <= 0 (off the log scale)")
    ax_lvl.plot(v, np.abs(dark) * 1e3, "s-", ms=3, lw=1, color="tab:grey",
                alpha=0.7, label="|dark| (SLM blanked)")
    ax_lvl.axhline(STD_FLOOR_V * 1e3, ls=":", color="k", lw=1,
                   label=f"{STD_FLOOR_V*1e3:.2f} mV systematic floor")
    ax_lvl.set_yscale("log")
    ax_lvl.set_ylabel("DAQ level (mV)")
    ax_lvl.legend(fontsize=8)
    ax_lvl.grid(alpha=0.3, which="both")
    title = "PMT sensitivity vs gain-control voltage"
    if meta:
        title += f"\npair {meta.get('pair', PAIR_INDEX)}, " \
                 f"(x,w)=({meta.get('drive_x', DRIVE_X)},{meta.get('drive_w', DRIVE_W)}), " \
                 f"{meta.get('t_meas_s', T_MEAS_S)} s per read"
    ax_lvl.set_title(title, fontsize=10)

    ax_rel.errorbar(v[ok], rel[ok], yerr=rel_std[ok], fmt="o-", ms=3, lw=1,
                    color="tab:green")
    ax_rel.axhline(1.0, ls="--", color="k", lw=1)
    ax_rel.axvline(v_ref, ls="--", color="k", lw=1)
    ax_rel.annotate(f"ref {v_ref:.3f} V", (v_ref, 1.0), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax_rel.set_yscale("log")
    ax_rel.set_xlabel("PMT gain-control voltage (V)")
    ax_rel.set_ylabel(f"relative sensitivity (= 1 at {v_ref:.3f} V)")
    ax_rel.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ======================================================================
# entry points
# ======================================================================

def dry_run(step3: Path) -> None:
    """Resolve the calibration, pair and schedule; touch no instrument."""
    layout = load_layout(step3)
    voltages = control_voltages()
    print(f"Step 3      : {step3}")
    print(f"Layout      : {layout.n_channels} pairs -- {describe_pair(layout)}")
    print(f"Drive       : (x, w) = ({DRIVE_X}, {DRIVE_W}), every other channel off")
    print(f"Sweep       : {len(voltages)} points {voltages[0]:.3f} -> {voltages[-1]:.3f} V, "
          f"step {voltages[1] - voltages[0]:.4f} V")
    print(f"Per point   : {SETTLE_S:g} s + {T_MEAS_S:g} s dark, "
          f"{SETTLE_S:g} s + {T_MEAS_S:g} s signal")
    print(f"Range       : +/-{ACQ.range_v:g} V, fixed, dark and signal alike")
    print(f"Runtime     : {run_seconds()/60.0:.1f} min ({run_seconds():.0f} s) plus PSU/SLM overhead")
    print(f"Reference   : nearest point to {V_REF:.3f} V")
    print("Dry run: no hardware touched.")


def measure(step3: Path) -> None:
    layout = load_layout(step3)
    print(f"Layout      : {layout.n_channels} pairs -- {describe_pair(layout)}")
    print(f"Sweep       : {N_POINTS} points {V_START:.3f} -> {V_STOP:.3f} V, "
          f"~{run_seconds()/60.0:.1f} min\n")

    psu = connect_psu()
    before = psu.read_state()
    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=T_MEAS_S, t_single=T_MEAS_S,
                      min_val=-ACQ.range_v, max_val=ACQ.range_v)

    rows: list[dict] = []
    interrupted = False
    started = time.time()
    try:
        arm_output(psu)
        rows = sweep(psu, daq, slm, layout)
    except KeyboardInterrupt:
        interrupted = True
        print("\n** Ctrl+C: stopping after the point in hand **")
    finally:
        try:
            show(slm, layout, 0.0, 0.0)   # leave the panel dark, not illuminating
        except Exception:
            pass
        restore(psu, before.vset, before.iset, before.output_on)
        psu.disconnect()
        slm.close_slm()
        daq.disconnect()

    if not rows:
        print("No points measured; nothing written.")
        return

    stamp = time.strftime("%m%d_%H%M", time.localtime(started))
    csv_path = OUT_DIR / f"pmt_gain_sweep_{stamp}.csv"
    meta = {
        "script": "pmt_gain_sweep",
        "step3": step3,
        "pair": PAIR_INDEX,
        "pair_index_base": CONFIG.pair_index_base,
        "drive_x": DRIVE_X,
        "drive_w": DRIVE_W,
        "v_start_v": V_START,
        "v_stop_v": V_STOP,
        "n_points": N_POINTS,
        "n_measured": len(rows),
        "t_meas_s": T_MEAS_S,
        "settle_s": SETTLE_S,
        "v_ref_v": V_REF,
        "invert": ACQ.invert,
        "psu_port": PSU_PORT,
        "daq": f"{DAQ_DEVICE}/{DAQ_CHANNEL}",
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "interrupted": interrupted,
    }
    write_csv(rows, csv_path, meta)
    print(f"\nCSV ({len(rows)} points) -> {csv_path}")
    report(rows)
    png_path = csv_path.with_suffix(".png")
    make_plot(rows, png_path, meta)
    print(f"Plot -> {png_path}")


def replot(csv_path: Path) -> None:
    rows, meta = read_csv(csv_path)
    print(f"Loaded {csv_path}: {len(rows)} points"
          + (f", pair {meta.get('pair')}" if meta.get("pair") else ""))
    report(rows)
    png_path = Path(csv_path).with_suffix(".png")
    make_plot(rows, png_path, meta)
    print(f"Plot -> {png_path}")


def main(argv: list[str] | None = None) -> int:
    global PAIR_INDEX
    argv = sys.argv[1:] if argv is None else argv

    def take(flag: str) -> str | None:
        if flag not in argv:
            return None
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        return argv[i + 1]

    step3 = take("--step3")
    pair = take("--pair")
    if pair is not None:
        PAIR_INDEX = int(pair)
    consumed = {"--step3", step3, "--pair", pair, "--dry-run"}
    positional = [a for a in argv if a not in consumed]

    if positional:                     # a CSV path -> offline re-plot, no hardware
        replot(Path(positional[0]))
        return 0
    if step3 is None:
        raise SystemExit(
            "--step3 <calib_step3c_*.json> is required. There is no default on "
            "purpose: every level here is encoded through that calibration, so "
            "the wrong one does not fail, it silently measures a different "
            "aperture.\n"
            "  python src/drafts/pmt_gain_sweep.py --step3 src/calib_data/calib_step3c_0928_1823.json"
        )
    if "--dry-run" in argv:
        dry_run(Path(step3))
        return 0
    measure(Path(step3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
