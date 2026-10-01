"""Step 8's fit and its report: one saturation constant out of one level sweep.

Deliberately thin, and separate from :mod:`~calibration_module.fit.levels_report`.
That module reports a *system accuracy* run -- families A, B and C, model error
against readout reproducibility against measured full scale -- and old runs on
disk still carry all three.  Step 8 asks one question with one free parameter,
so it gets its own three panels rather than a flag threaded through the other
report.

Importable without touching hardware, so the GUI tab and the CLI show the same
numbers from the same code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from calibration_module.fit.levels_report import load_reads
from calibration_module.fit.verify import fit_y_sat, note_step7_moved, saturate

#: A 4-bit computation divides full scale into this many levels.
LEVELS = 16


def _unsaturate(y, y_sat: float):
    """Invert :func:`saturate` -- the ``|E|^2`` a read of ``y`` implies."""
    y = np.asarray(y, dtype=float)
    head = 1.0 - y / float(y_sat)
    return np.where(head > 0, y / np.where(head > 0, head, np.nan), np.inf)


def fit_from_csv(path, run, *, pairs=None, method=None) -> tuple:
    """``(reads, result)`` -- fit ``Y_sat`` to a step-8 CSV.

    Reads are loaded with ``y_sat=None`` on purpose: the fit needs the LINEAR
    ``|E|^2`` to fit against, and a sweep that assumed a saturation constant to
    measure one would be circular.  The ``F`` reads -- the closed-form
    full-scale drive, displayed at the start and the end -- are kept in the fit
    (they are the most informative points on the curve) and also reported
    separately as measured-vs-predicted full scale.
    """
    reads, meta = load_reads(path, run, pairs=pairs, method=method, y_sat=None)
    result = fit_y_sat(reads.y, reads.pred, reads.sigma)

    f = reads.of("F")
    result["full_scale"] = _full_scale(f, reads.ymax_pred, result["y_sat_v"])
    result["csv"] = str(Path(path).resolve())
    result["n_ladder"] = int(len(reads.of("A")))
    result["n_full_scale"] = int(len(f))
    return reads, result, meta


def _full_scale(f, ymax_pred: float, y_sat: float) -> dict:
    """Predicted vs measured full scale, from the ``F`` reads.

    Two comparisons, because they answer different questions.  The readout one
    is direct -- what the ADC saw against what the saturated model said it
    would see.  The ``|E|^2`` one inverts the saturation to recover the
    coherent ceiling, which is what a computation divides into levels; it
    is the more useful number and the more fragile one, since near the ceiling
    the inversion amplifies any error in ``Y_sat``.
    """
    out = {"y_max_pred_v": float(ymax_pred),
           "y_max_pred_sat_v": float(saturate(ymax_pred, y_sat))}
    if not len(f):
        return out
    start = [y for y, lab in zip(f.y, f.label) if lab.endswith("@start")]
    end = [y for y, lab in zip(f.y, f.label) if lab.endswith("@end")]
    meas = float(np.mean(f.y))
    out.update({
        "y_max_meas_sat_v": meas,
        "y_max_meas_v": float(_unsaturate(meas, y_sat)),
        "readout_ratio": meas / out["y_max_pred_sat_v"],
        "start_v": float(np.mean(start)) if start else float("nan"),
        "end_v": float(np.mean(end)) if end else float("nan"),
    })
    if start and end:
        # The same frame at both ends of the run: any difference is drift,
        # not model error, and it bounds what a single gain number can be
        # worth for the rest of the sweep.
        out["drift_frac"] = out["end_v"] / out["start_v"] - 1.0
    return out


def print_result(result: dict, log=print) -> None:
    """The fit, in the order it should be read."""
    fs = result.get("full_scale", {})
    log(f"\nY_sat     : {result['y_sat_v']:.4f} "
        f"+{result['hi_v'] - result['y_sat_v']:.4f}"
        f"/-{result['y_sat_v'] - result['lo_v']:.4f} V   "
        f"({result['n']} reads, ladder {result.get('n_ladder', 0)})")
    if result.get("unbounded"):
        log("            WARNING: the interval ran into the search grid -- the "
            "sweep never got\n            bright enough to bound Y_sat.  Treat "
            "this as a limit, not a measurement.")
    log(f"Residual  : {result['rmse_v'] * 1e3:.4f} mV rms  "
        f"(linear model: {result['rmse_linear_v'] * 1e3:.3f} mV, "
        f"{result['rmse_linear_v'] / max(result['rmse_v'], 1e-12):.1f}x worse)")
    log(f"Gain lever: {result['gain_lever']:.1f}x -- this fit pins g = 1, so a "
        f"1 % gain error\n            moves Y_sat by "
        f"{result['gain_lever']:.0f} %.  It is the g = 1 anchor, not a constant "
        f"of the atoms.")
    if "y_max_meas_v" in fs:
        log(f"Full scale: readout  {fs['y_max_meas_sat_v'] * 1e3:.4f} mV measured "
            f"vs {fs['y_max_pred_sat_v'] * 1e3:.4f} predicted "
            f"({100 * (fs['readout_ratio'] - 1):+.2f} %)")
        log(f"            |E|^2    {fs['y_max_meas_v'] * 1e3:.4f} mV implied "
            f"vs {fs['y_max_pred_v'] * 1e3:.4f} predicted "
            f"({100 * (fs['y_max_meas_v'] / fs['y_max_pred_v'] - 1):+.2f} %)")
        log(f"            4-bit level step = "
            f"{fs['y_max_pred_sat_v'] / (LEVELS - 1) * 1e3:.3f} mV")
    if "drift_frac" in fs:
        log(f"Drift     : the full-scale frame moved "
            f"{100 * fs['drift_frac']:+.3f} % from the start of the run to the "
            f"end")


def make_plot(reads, result: dict, out_path: Path) -> Path:
    """Three panels: the curve, the residuals, and how well Y_sat is pinned."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_sat = result["y_sat_v"]
    a, f = reads.of("A"), reads.of("F")
    fs = result.get("full_scale", {})

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
    fig.suptitle(f"Step 8 - two-photon saturation     "
                 f"Y_sat = {y_sat:.3f} +{result['hi_v'] - y_sat:.3f}"
                 f"/-{y_sat - result['lo_v']:.3f} V     "
                 f"{result['n']} reads, gain pinned to 1",
                 fontsize=12, fontweight="bold")

    # ---- the curve ----
    ax = axes[0]
    grid = np.linspace(0, reads.pred.max() * 1.03, 400)
    ax.plot(grid * 1e3, grid * 1e3, "--", color="0.6", lw=1.3, label="linear |E|^2")
    ax.plot(grid * 1e3, saturate(grid, y_sat) * 1e3, "-", color="tab:red", lw=2.0,
            label=f"Y/(1 + Y/{y_sat:.2f})")
    ax.errorbar(a.pred * 1e3, a.y * 1e3, yerr=a.sigma * 1e3, fmt="o", ms=4,
                color="tab:blue", lw=0, elinewidth=0.8, capsize=2, label="ladder")
    if len(f):
        ax.plot(f.pred * 1e3, f.y * 1e3, "*", ms=15, color="tab:orange",
                markeredgecolor="0.2", lw=0, label="full scale (closed form)")
    ax.set_xlabel("predicted |E|^2, unsaturated   [mV]")
    ax.set_ylabel("measured   [mV]")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title(f"the top of the sweep is the whole measurement:\n"
                 f"the linear model reads "
                 f"{100 * (1 - saturate(reads.pred.max(), y_sat) / reads.pred.max()):.1f} "
                 f"% high at full scale", fontsize=10)

    # ---- residuals ----
    ax = axes[1]
    resid = (reads.y - saturate(reads.pred, y_sat)) * 1e3
    ax.axhline(0, color="0.5", lw=1.0)
    ax.errorbar(reads.alpha, resid, yerr=reads.sigma * 1e3, fmt="o", ms=4,
                color="tab:blue", lw=0, elinewidth=0.8, capsize=2)
    if len(f):
        m = reads.kind == "F"
        ax.plot(reads.alpha[m], resid[m], "*", ms=15, color="tab:orange",
                markeredgecolor="0.2", lw=0)
    rms = result["rmse_v"] * 1e3
    ax.axhspan(-rms, rms, color="tab:gray", alpha=0.16, label=f"+/-{rms:.2f} mV rms")
    if "y_max_pred_sat_v" in fs:
        step = fs["y_max_pred_sat_v"] / (LEVELS - 1) * 1e3
        for s in (-1, 1):
            ax.axhline(s * step / 2, color="tab:green", ls=":", lw=1.3)
        ax.plot([], [], ":", color="tab:green", lw=1.3,
                label=f"half a 4-bit level ({step / 2:.1f} mV)")
    ax.set_xlabel("alpha   (predicted level, fraction of full scale)")
    ax.set_ylabel("measured - saturated model   [mV]")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("a tilt left to right here means Y_sat is wrong;\n"
                 "a constant offset means the gain is", fontsize=10)

    # ---- how well is it pinned ----
    ax = axes[2]
    span = max(result["hi_v"] - y_sat, y_sat - result["lo_v"]) * 6 + 1e-9
    trial = np.linspace(max(y_sat - span, 0.2), y_sat + span, 300)
    w = 1.0 / reads.sigma
    chi2 = np.array([np.sum(((reads.y - saturate(reads.pred, q)) * w) ** 2)
                     for q in trial])
    ax.plot(trial, chi2 - chi2.min(), "-", color="tab:purple", lw=2.0)
    ax.axhline(1.0, color="0.5", ls="--", lw=1.2, label="delta chi^2 = 1")
    ax.axvline(y_sat, color="tab:red", lw=1.5, label=f"{y_sat:.3f} V")
    ax.axvspan(result["lo_v"], result["hi_v"], color="tab:red", alpha=0.14)
    ax.set_ylim(0, 25)
    ax.set_xlabel("trial Y_sat   [V]")
    ax.set_ylabel("chi^2 - min")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(f"the fit is {100 * result['sigma_v'] / y_sat:.1f} % tight -- but a "
                 f"1 % gain error\nwould move it {result['gain_lever']:.0f} %, "
                 f"which this cannot see", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def report_csv(path, run, *, pairs=None, method=None, out_dir=None,
               log=print) -> dict:
    """Fit a recorded step-8 CSV, print the result, write the PNG.  No hardware."""
    path = Path(path)
    out_dir = Path(path.parent if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reads, result, meta = fit_from_csv(path, run, pairs=pairs, method=method)
    note_step7_moved(run, meta, log=log)
    print_result(result, log=log)
    png = make_plot(reads, result, out_dir / f"{path.stem}_ysat.png")
    # The report knows where it put the PNG; nothing downstream should have to
    # re-derive it from the CSV name and an --out it was not told about.
    result["png"] = str(png)
    log(f"\nPlot saved to {png}")
    return result
