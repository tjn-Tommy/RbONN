"""Draft: measured signal against drive fraction, alpha = 0.1 .. 1.

    python src/drafts/alpha_sweep.py --dry-run     # solve it all, touch no hardware
    python src/drafts/alpha_sweep.py               # the ~3 min sweep
    python src/drafts/alpha_sweep.py some.csv      # re-plot an earlier run offline

One question: does the read bend as the drive goes up, and is the bend
saturation?  Step 8 asks the same thing through the forward model's predicted
volts, which only works while the model and the bench share a gain.  This does
not: ``alpha`` is a fraction of full drive, fixed by the pattern that was
displayed, and the plot is measured volts against it.  A model that is a
factor out therefore cannot move a point on this curve -- it can only change
which patterns were solved.

Two fits, both in measured volts, both with the gain free::

    linear        y = A alpha
    saturating    y = A alpha / (1 + B alpha)

``A`` is the read at full drive the bench would give with no saturation, ``B``
the shortfall parameter: at ``alpha = 1`` the curve falls ``B / (1 + B)`` short
of the line, and ``A / B`` is ``Y_sat`` expressed in this run's own volts.
Whether saturation is there at all is ``delta chi^2`` between the two.

Why the whole range, and why it fits under the rail
---------------------------------------------------
Step 8's ladder stops at :data:`~calibration_module.steps.calib_step8.LADDER_CEILING_V`
of READ volts, so on 0930 it reached alpha 0.29 and saw nothing bend -- at
that drive a 5 V ``Y_sat`` hides inside a few noisy percent.  Curvature grows
with the drive, so the measurement wants alpha near 1; what stopped it was the
1 V amplifier.  With the TIA gain lowered on 0930 the whole range fits: full
drive predicts a couple of hundred millivolts, a fifth of the rail.

ONE dark for the run, not one per read: :func:`run_patterns` reads its opening
dark and refreshes every ``DARK_EVERY_S``, which this run is deliberately
shorter than.

The order is SHUFFLED.  Drift over three minutes would otherwise land in the
curve's shape, which is the one thing being measured; shuffled, it lands in the
scatter instead.

Needs real hardware (DAQ Dev1, SLM on its own display), and the GUI cannot be
holding them at the same time.  The DAQ is opened first, so a machine that is
still busy stops the run before the panel is touched.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.sigma import floor_std              # noqa: E402
from calibration_module.measure.patterns import (               # noqa: E402
    DARK_EVERY_S, run_patterns,
)
from calibration_module.steps import calib_step8 as STEP8       # noqa: E402

#: The forward model the patterns are solved from.  Its GAIN may be stale --
#: see the module docstring -- but its etas and phases are what decide which
#: drive is alpha = 0.3, and that is all it is asked for here.
STEP7 = REPO_ROOT / "src/calib_data/archive/2026-09/run_0930/calib_step7_result_0930_1353.json"

OUT_DIR = REPO_ROOT / "src/calib_data"
ALPHA_LO, ALPHA_HI, N_POINTS = 0.1, 1.0, 20

#: Stop the run if a raw trace peak reaches this.  The mean can look sane
#: while the peak flat-tops; step 8 measured peak/mean at 1.22 typical and
#: 1.9 at the p99, so a mean well under the rail is not on its own a promise.
PEAK_ABORT_V = 0.90


# ======================================================================
# the sweep
# ======================================================================

def solve(step7: Path, lo: float, hi: float, n: int, seed: int):
    """``(models, phases, pairs, scale, patterns)`` -- no hardware."""
    models, phases, _ref = STEP8.load_forward_model(step7)
    pairs = sorted(models)
    scale = STEP8.full_scale(models, phases, pairs)
    seeds = np.random.SeedSequence(seed).generate_state(n)
    patterns = [
        STEP8.pattern_for_alpha(models, phases, pairs, float(a),
                                seed=int(s), label=f"a{a:.3f}")
        for a, s in zip(np.linspace(lo, hi, n), seeds)
    ]
    return models, phases, pairs, scale, patterns


def report_plan(scale, patterns, secs: float) -> None:
    print(f"\n{len(patterns)} drives, alpha "
          f"{patterns[0].alpha:.3f} .. {patterns[-1].alpha:.3f}"
          f"   (asked {patterns[0].target_alpha:.3f} .. "
          f"{patterns[-1].target_alpha:.3f})")
    print(f"Y_max of the model  {scale.fs * 1e3:9.1f} mV  -- in the MODEL's "
          f"volts, which may not be the bench's")
    print(f"one dark, then {len(patterns)} reads of "
          f"{STEP8.ACQ.t_meas_s:g} s     ~{secs / 60:.1f} min")
    if secs >= DARK_EVERY_S:
        print(f"WARNING: the run is longer than DARK_EVERY_S "
              f"({DARK_EVERY_S:g} s), so it will take more than one dark")
    worst = max(p.alpha for p in patterns) * scale.fs
    print(f"if the model still has the bench's gain, the top read is "
          f"{worst * 1e3:.0f} mV of the {STEP8.AMP_LIMIT_V:g} V rail; the "
          f"0930 sweep measured the model {1 / 0.123:.1f}x high, which would "
          f"make it {worst * 0.123 * 1e3:.0f} mV")


def sweep(step7: Path, patterns, models, phases, pairs, out: Path) -> Path:
    """Drive the lot in shuffled order and write one row per read."""
    from calibration_module.measure.bench import connect_daq, connect_slm
    from calibration_module.fit.verify import write_verify_csv

    acq = STEP8.ACQ
    order = np.random.default_rng(0).permutation(len(patterns))
    shuffled = [patterns[i] for i in order]

    # DAQ first: if the GUI still holds it, nothing has been displayed yet.
    daq = connect_daq(device="Dev1", channel="ai0",
                      t_both=acq.t_meas_s, t_single=acq.t_meas_s,
                      min_val=-acq.range_v, max_val=acq.range_v)
    slm = connect_slm()
    layout = STEP8.load_layout(step7, pairs)

    x = np.stack([p.x for p in shuffled])[None]
    w = np.stack([p.w for p in shuffled])[None]
    extra = [[{"kind": p.kind, "label": p.label, "seq": i,
               "target_alpha": f"{p.target_alpha:.6f}",
               "alpha": f"{p.alpha:.6f}",
               "t_meas_s": f"{acq.t_meas_s:.3f}"}
              for i, p in enumerate(shuffled)]]

    rows: list[dict] = []
    import threading
    stop = threading.Event()

    def watch(row: dict) -> None:
        """Peak guard.  Cannot raise, so it stops the loop between reads."""
        if float(row.get("peak_v", 0.0)) >= PEAK_ABORT_V:
            print(f"  PEAK {row['peak_v'] * 1e3:.0f} mV -- at the rail, stopping")
            stop.set()

    try:
        run_patterns(daq, slm, layout, [tuple(pairs)], x, w, models, phases,
                     rows=rows, acq=acq, extra=extra, y_sat=None,
                     row_hook=watch, stop_event=stop, log=print)
    finally:
        try:
            wpx, hpx = slm.get_slm_info()
            slm.display_array(np.zeros((hpx, wpx), dtype=np.uint8))
        except Exception:                                    # noqa: BLE001
            pass
        for shut, name in ((getattr(slm, "close_slm", None), "SLM"),
                           (getattr(daq, "disconnect", None), "DAQ")):
            try:
                if shut is not None:
                    shut()
            except Exception as exc:                         # noqa: BLE001
                print(f"{name} did not close cleanly: {exc}")

    stamp = time.strftime("%m%d_%H%M")
    path = write_verify_csv(
        out / f"alpha_sweep_{stamp}.csv", pairs, rows,
        meta={"step7_json": Path(step7).resolve(),
              "alpha_lo": patterns[0].target_alpha,
              "alpha_hi": patterns[-1].target_alpha,
              "n_points": len(patterns),
              "y_max_model_v": f"{STEP8.full_scale(models, phases, pairs).fs:.9g}",
              "one_dark": True},
        extra=["kind", "label", "seq", "target_alpha", "alpha", "t_meas_s",
               "peak_v"])
    print(f"\nCSV written to {path}")
    return Path(path)


# ======================================================================
# the fits -- measured volts against alpha, nothing predicted
# ======================================================================

def fit_linear(alpha, y, sigma) -> dict:
    """``y = A alpha``."""
    w = 1.0 / sigma ** 2
    A = float((w * y * alpha).sum() / (w * alpha ** 2).sum())
    chi2 = float((w * (y - A * alpha) ** 2).sum())
    return {"A": A, "chi2": chi2, "dof": y.size - 1}


def fit_saturating(alpha, y, sigma) -> dict:
    """``y = A alpha / (1 + B alpha)``, profiled over ``B`` with ``A`` exact."""
    w = 1.0 / sigma ** 2
    grid = np.linspace(0.0, 20.0, 400001)                 # B >= 0
    m = alpha[None, :] / (1.0 + grid[:, None] * alpha[None, :])
    A = (w * y * m).sum(1) / (w * m * m).sum(1)
    chi2 = (w * (y[None, :] - A[:, None] * m) ** 2).sum(1)
    i = int(np.argmin(chi2))
    out = {"A": float(A[i]), "B": float(grid[i]), "chi2": float(chi2[i]),
           "dof": y.size - 2}
    scale = max(out["chi2"] / max(out["dof"], 1), 1.0)
    ok = chi2 <= chi2[i] + scale
    out["B_lo"], out["B_hi"] = float(grid[ok].min()), float(grid[ok].max())
    return out


def load_csv(path: Path):
    """``(alpha, y, sigma, meta)`` from a sweep CSV -- dark already removed."""
    meta: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        body = []
        for line in fh:
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition(",")
                meta[k.strip()] = v.strip()
            else:
                body.append(line)
    rows = list(csv.DictReader(body))
    alpha = np.array([float(r["alpha"]) for r in rows])
    y = np.array([float(r["voltage_mean_v"]) - float(r["dark_v"]) for r in rows])
    sigma = floor_std(np.array([float(r["voltage_std_v"]) for r in rows]))
    order = np.argsort(alpha)
    return alpha[order], y[order], sigma[order], meta


def plot(path: Path) -> Path:
    """Two panels: the curve with both fits, and what each one leaves over."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha, y, sigma, meta = load_csv(path)
    lin, sat = fit_linear(alpha, y, sigma), fit_saturating(alpha, y, sigma)
    d_chi2 = lin["chi2"] - sat["chi2"]
    grid = np.linspace(0, 1.03 * alpha.max(), 400)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    fig.suptitle(f"{path.stem}   --   measured signal vs drive fraction, "
                 f"gain free, nothing predicted", fontsize=12,
                 fontweight="bold")

    ax = axes[0]
    ax.errorbar(alpha, y * 1e3, yerr=sigma * 1e3, fmt="o", ms=5,
                color="tab:blue", lw=0, elinewidth=0.9, capsize=2,
                label=f"{alpha.size} reads", zorder=3)
    ax.plot(grid, lin["A"] * grid * 1e3, "--", color="0.45", lw=1.6,
            label=f"linear   A = {lin['A'] * 1e3:.1f} mV"
                  f"   chi2/dof {lin['chi2'] / lin['dof']:.2f}")
    ax.plot(grid, sat["A"] * grid / (1 + sat["B"] * grid) * 1e3, "-",
            color="tab:red", lw=2.0,
            label=f"saturating   B = {sat['B']:.3f}"
                  f"   chi2/dof {sat['chi2'] / sat['dof']:.2f}")
    ax.set_xlabel("alpha   (fraction of full drive)")
    ax.set_ylabel("measured, dark subtracted   [mV]")
    ax.legend(fontsize=9, loc="upper left")
    short = 100 * sat["B"] / (1 + sat["B"])
    ax.set_title(f"at full drive the saturating fit sits {short:.1f} % under "
                 f"the line\n"
                 f"Y_sat = A/B = "
                 + (f"{sat['A'] / sat['B'] * 1e3:.0f} mV of this run's volts"
                    if sat["B"] > 0 else "unbounded (B = 0)"),
                 fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.axhline(0, color="0.5", lw=1.0)
    ax.errorbar(alpha, (y - lin["A"] * alpha) * 1e3, yerr=sigma * 1e3,
                fmt="s", ms=5, color="0.45", lw=0, elinewidth=0.9, capsize=2,
                label="residual, linear")
    ax.errorbar(alpha, (y - sat["A"] * alpha / (1 + sat["B"] * alpha)) * 1e3,
                yerr=sigma * 1e3, fmt="o", ms=5, color="tab:red", lw=0,
                elinewidth=0.9, capsize=2, label="residual, saturating")
    ax.set_xlabel("alpha   (fraction of full drive)")
    ax.set_ylabel("measured - fit   [mV]")
    ax.legend(fontsize=9)
    verdict = ("saturation is not needed" if d_chi2 < 1.0 else
               "saturation is needed" if d_chi2 > 9.0 else "marginal")
    ax.set_title(f"delta chi2 = {d_chi2:.1f} for the one extra parameter"
                 f"   --   {verdict}\n"
                 f"B = {sat['B']:.3f} "
                 f"[{sat['B_lo']:.3f}, {sat['B_hi']:.3f}] at 1 sigma",
                 fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=150)
    plt.close(fig)

    print(f"\nlinear      A {lin['A'] * 1e3:8.2f} mV        "
          f"chi2/dof {lin['chi2'] / lin['dof']:.2f}")
    print(f"saturating  A {sat['A'] * 1e3:8.2f} mV  B {sat['B']:.4f} "
          f"[{sat['B_lo']:.4f}, {sat['B_hi']:.4f}]  "
          f"chi2/dof {sat['chi2'] / sat['dof']:.2f}")
    print(f"delta chi2  {d_chi2:.2f}   ->   {verdict}")
    if sat["B_hi"] > 0:
        print(f"Y_sat > {sat['A'] / sat['B_hi'] * 1e3:.0f} mV of this run's "
              f"volts at 1 sigma (= {1 / sat['B_hi']:.1f} x full drive)")
    print(f"Plot saved to {png}")
    return png


# ======================================================================
# cli
# ======================================================================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv", nargs="?", type=Path,
                   help="re-plot this sweep CSV instead of driving anything")
    p.add_argument("--step7", type=Path, default=STEP7)
    p.add_argument("--lo", type=float, default=ALPHA_LO)
    p.add_argument("--hi", type=float, default=ALPHA_HI)
    p.add_argument("--points", type=int, default=N_POINTS)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--dry-run", action="store_true",
                   help="solve the drives and stop; no instrument is opened")
    args = p.parse_args(argv)

    if args.csv is not None:
        plot(args.csv)
        return 0

    models, phases, pairs, scale, patterns = solve(
        args.step7, args.lo, args.hi, args.points, args.seed)
    secs = STEP8.ACQ.t_meas_s * (len(patterns) + 1) + \
        STEP8.ACQ.settle_s * len(patterns)
    print(f"step 7: {Path(args.step7).name}   pairs {pairs}")
    report_plan(scale, patterns, secs)
    if args.dry_run:
        print("\ndry run: nothing was opened")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    plot(sweep(args.step7, patterns, models, phases, pairs, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
