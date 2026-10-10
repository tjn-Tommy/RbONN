"""Read back a system-accuracy run: model error, reproducibility, measured full scale.

Fits nothing.  Every prediction is recomputed from the step-7 JSON, so this runs
against a saved CSV with no hardware.

Everything here is in COHERENT volts.  A read's measurement is::

    y = (voltage_mean_v - dark_v) - sb_pred_v

-- the raw dark-subtracted signal with the model's single-beam background taken
back off, because ``FS = Y_max`` excludes that background and the levels are
defined on it.  With ``y_sat`` the prediction, the levels and ``Y_max`` are all
on the SATURATED readout (:func:`verify.saturate`); ``y_sat=None`` is linear.  The raw columns are never modified; the subtraction happens here
and only here, so the same CSV can always be re-read the other way.

The four questions, one per family:

``A``  is the model biased, and does the bias grow with level?  Read the
       residual-vs-level panel, not the RMSE: a bias that grows is a calibration
       problem, while a flat scatter is noise.
``B``  can one read tell two 4-bit levels apart?  Read the repeat spread against
       the level step.  This is the noise budget A's errors have to be judged
       against, and it cannot be got from A.
``C``  where is full scale really?  The peak of the psi scan, with the start and
       end copies bracketing the run's drift.
``D``  is any of A's level-dependence just the board changing range?  Each pair of
       reads differs only in the input range, so their difference is the gain step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sigma import floor_std
from .verify import (full_scale, load_forward_model, load_verify_csv,
                     note_step7_moved, predict, saturate)

__all__ = ["Reads", "load_reads", "report_csv"]

#: A 4-bit computation divides full scale into this many levels.
LEVELS = 16


@dataclass
class Reads:
    """One run's reads, already in coherent volts, with the family labels."""

    kind: np.ndarray            # "A" / "B" / "C"
    label: np.ndarray
    seq: np.ndarray
    alpha: np.ndarray           # predicted level, fraction of the PREDICTED Y_max
    y: np.ndarray               # measured, dark-subtracted, single-beam removed
    pred: np.ndarray            # predicted coherent |E|^2
    sigma: np.ndarray           # per-read sigma, trace std with the systematic floor
    range_v: np.ndarray
    ymax_pred: float

    def of(self, kind: str) -> "Reads":
        return self[self.kind == kind]

    def __getitem__(self, m) -> "Reads":
        return Reads(self.kind[m], self.label[m], self.seq[m], self.alpha[m],
                     self.y[m], self.pred[m], self.sigma[m], self.range_v[m],
                     self.ymax_pred)

    def __len__(self) -> int:
        return int(self.y.size)

    @property
    def resid(self) -> np.ndarray:
        return self.y - self.pred

    @property
    def level(self) -> np.ndarray:
        """Which 4-bit level each read's PREDICTION belongs to, 0 .. LEVELS-1."""
        return np.clip((self.alpha * (LEVELS - 1)).astype(int), 0, LEVELS - 1)


def load_reads(path, run, *, pairs=None, method=None,
               y_sat: float | None = None) -> tuple[Reads, dict]:
    """``(reads, meta)`` for a system-accuracy CSV, predictions recomputed."""
    _, blocks, meta = load_verify_csv(path)
    if len(blocks) != 1:
        raise ValueError(f"{Path(path).name} holds {len(blocks)} blocks; a "
                         f"system-accuracy run is one block of every pair")
    block = blocks[0]
    if "kind" not in block.extra:
        raise ValueError(f"{Path(path).name} has no 'kind' column, so it is not "
                         f"a system-accuracy CSV: every row needs the "
                         f"family/level label the sweep writes)")
    driven = list(block.driven)
    models, phases, _ = load_forward_model(run, pairs=pairs or driven, method=method)
    scale = full_scale(models, phases, driven)

    # Predict here rather than trust the stored column: a CSV re-read against a
    # different step-7 file has to be recomputed, and that is the whole point of
    # keeping --step7 required.
    pred = predict(models, phases, driven, block.x, block.w, coherent=True, y_sat=y_sat)
    sb = np.asarray(block.extra.get("sb_pred_v",
                                    predict(models, phases, driven, block.x, block.w,
                                            y_sat=y_sat)
                                    - pred), dtype=float)
    ymax = float(saturate(scale.fs, y_sat))
    reads = Reads(
        kind=np.asarray(block.extra["kind"], dtype=str),
        label=np.asarray(block.extra.get("label", block.extra["kind"]), dtype=str),
        seq=np.asarray(block.extra.get("seq", np.arange(block.y.size)), dtype=float),
        alpha=pred / ymax,
        y=block.y - sb,
        pred=np.asarray(pred, dtype=float),
        sigma=floor_std(block.std_v),
        range_v=block.range_v,
        ymax_pred=ymax,
    )
    return reads, dict(meta)


# ======================================================================
# the four questions
# ======================================================================

def ladder_bias(reads: Reads) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(level, mean resid, sem)`` per 4-bit level over the A reads."""
    a = reads.of("A")
    out = []
    for lv in range(LEVELS):
        here = a.resid[a.level == lv]
        if here.size:
            out.append((lv, here.mean(),
                        here.std(ddof=1) / np.sqrt(here.size) if here.size > 1 else np.nan))
    lv, mean, sem = (np.array(c) for c in zip(*out))
    return lv, mean, sem


def repeat_spread(reads: Reads) -> list[tuple[float, np.ndarray]]:
    """``[(alpha, the reads of that one drive), ...]`` for B, grouped by level.

    Grouped on the drive itself (via the label's level tag), not on the measured
    value: the point is that the SLM pattern never moved.
    """
    b = reads.of("B")
    tags = np.array([lab.split("r")[0] for lab in b.label])
    out = []
    for tag in sorted(set(tags.tolist()), key=lambda t: float(t[1:])):
        m = tags == tag
        out.append((float(tag[1:]), b.y[m]))
    return out


#: Shifts tried when matching the measured scan to the predicted one, in degrees.
PSI_SHIFT_GRID = np.arange(-25.0, 25.0 + 1e-9, 0.05)


def psi_peak(reads: Reads) -> dict:
    """Measured full scale from the C scan, by sliding the PREDICTED curve onto it.

    Not a parabola.  The fan is asymmetric -- a pair's phase can only be advanced,
    so one side of ``psi*`` costs more amplitude than the other -- and a parabola
    through the top of it puts the vertex tens of degrees out even on a perfect
    bench.  The predicted curve already has the right shape, so the fit is two
    parameters with a physical meaning each::

        y(offset) ~ scale * pred(offset - shift)

    ``shift`` is the model's global phase error and ``scale`` its amplitude error,
    so ``FS = scale * Y_max``.  ``scale`` is closed form at each trial shift, which
    leaves a 1-D grid search.
    """
    c = reads.of("C")
    if not len(c):
        return {}
    # the scan's own offsets, recovered from the labels ("C+20deg@start")
    off = np.array([float(lab[1:lab.index("deg")]) for lab in c.label])
    when = np.array(["end" if lab.endswith("@end") else "start" for lab in c.label])
    out: dict = {"offsets": off, "when": when, "y": c.y, "pred": c.pred}
    for tag in ("start", "end"):
        m = when == tag
        if m.sum() < 3:
            continue
        order = np.argsort(off[m])
        o, y, p = off[m][order], c.y[m][order], c.pred[m][order]
        best = None
        for shift in PSI_SHIFT_GRID:
            model = np.interp(o - shift, o, p, left=np.nan, right=np.nan)
            ok = np.isfinite(model)
            if ok.sum() < 3 or not np.any(model[ok]):
                continue                    # the shift ran off the end of the fan
            scale = float(model[ok] @ y[ok] / (model[ok] @ model[ok]))
            cost = float(np.sum((y[ok] - scale * model[ok]) ** 2))
            if best is None or cost < best[0]:
                best = (cost, float(shift), scale)
        if best is None:
            continue
        _, shift, scale = best
        out[tag] = {
            "fs_meas": scale * reads.ymax_pred,
            "peak_deg": shift,
            "scale": scale,
            "at_zero": float(y[np.argmin(np.abs(o))]),
        }
    return out


# ======================================================================
# report
# ======================================================================

def _note_y_sat(path, meta: dict, y_sat: float | None) -> None:
    """Say which saturation the report uses, and if it differs from the run's."""
    recorded = meta.get("y_sat_v")
    print(f"Saturation: Y_sat = {f'{y_sat:g} V' if y_sat else 'none (linear)'}")
    if recorded not in (None, "", "None") and float(recorded) != float(y_sat or 0):
        print(f"NOTE: {Path(path).name} was collected with Y_sat = {recorded} V")

def print_report(reads: Reads, peak: dict, meta: dict) -> float:
    """Print every panel's numbers; return the full scale the NRMSE is quoted on."""
    step_pred = reads.ymax_pred / (LEVELS - 1)
    fs = reads.ymax_pred
    print(f"\n=== system accuracy: {len(reads)} reads "
          f"(A {len(reads.of('A'))}, B {len(reads.of('B'))}, "
          f"C {len(reads.of('C'))}) ===")
    if meta.get("status", "complete") != "complete":
        print(f"  NOTE: this run is {meta['status']}")
    print(f"  Y_max predicted {reads.ymax_pred * 1e3:.4f} mV   "
          f"4-bit level step {step_pred * 1e3:.4f} mV")

    # --- C first: it sets the full scale everything else is quoted against
    if peak:
        for tag in ("start", "end"):
            if tag in peak:
                p = peak[tag]
                print(f"  C scan {tag:<5s}: FS {p['fs_meas'] * 1e3:9.4f} mV   "
                      f"phase error {p['peak_deg']:+6.2f} deg   "
                      f"amplitude {p['scale']:.4f}x   "
                      f"(read at 0 deg: {p['at_zero'] * 1e3:.4f} mV)")
        if "start" in peak and "end" in peak:
            drift = peak["end"]["at_zero"] - peak["start"]["at_zero"]
            fs = float(np.mean([peak["start"]["fs_meas"], peak["end"]["fs_meas"]]))
            print(f"  C drift over the run at 0 deg: {drift * 1e3:+.4f} mV "
                  f"({100 * drift / fs:+.2f} % of measured FS)")
            print(f"  FS measured {fs * 1e3:.4f} mV = "
                  f"{100 * fs / reads.ymax_pred:.2f} % of predicted   "
                  f"(a LOWER bound -- the scan is one direction of a 20-D space)")

    # --- A
    a = reads.of("A")
    if len(a):
        print(f"\n  A ladder ({len(a)} reads), residual = measured - predicted:")
        print(f"    RMSE {np.sqrt(np.mean(a.resid ** 2)) * 1e3:.4f} mV "
              f"= {100 * np.sqrt(np.mean(a.resid ** 2)) / fs:.3f} % of FS "
              f"= {np.sqrt(np.mean(a.resid ** 2)) / (fs / (LEVELS - 1)):.3f} level steps")
        print(f"    pull mean {np.mean(a.resid / a.sigma):+.3f}, "
              f"std {np.std(a.resid / a.sigma):.3f}  (1.0 = the quoted sigma is honest)")
        lv, mean, sem = ladder_bias(a)
        slope = np.polyfit(lv, mean, 1)[0]
        print(f"    bias per level (mV): "
              + " ".join(f"{m * 1e3:+.2f}" for m in mean))
        print(f"    bias trend {slope * 1e3:+.4f} mV per level "
              f"({100 * slope * (LEVELS - 1) / fs:+.2f} % of FS across the scale)")

    # --- B
    groups = repeat_spread(reads)
    if groups:
        print(f"\n  B repeats -- same drive, {len(groups[0][1])} reads each:")
        print(f"    {'alpha':>6} {'mean (mV)':>10} {'spread (mV)':>12} "
              f"{'vs level step':>14} {'vs quoted sigma':>16}")
        worst = 0.0
        for alpha, ys in groups:
            spread = ys.std(ddof=1) if ys.size > 1 else np.nan
            m = reads.label == f"B{alpha:.2f}r1"
            quoted = float(reads.sigma[m][0]) if m.any() else np.nan
            worst = max(worst, spread / (fs / (LEVELS - 1)))
            print(f"    {alpha:6.2f} {ys.mean() * 1e3:10.4f} {spread * 1e3:12.4f} "
                  f"{spread / (fs / (LEVELS - 1)):13.3f}x {spread / quoted:15.2f}x")
        # The verdict is computed, not asserted: +/-2 sigma has to stay inside
        # half a level for a single read to place a value on the right level.
        verdict = ("one read resolves 4 bits" if worst <= 0.25 else
                   "4 bits is MARGINAL -- average, or accept confusion between "
                   "adjacent levels")
        print(f"    worst spread {worst:.3f} level steps (need <= 0.25): {verdict}")

    return fs


def make_plot(reads: Reads, peak: dict, fs: float, path) -> None:
    """Six panels: the ladder, its bias, its pulls, the repeats, the scan, the levels."""
    import matplotlib

    matplotlib.use("Agg")       # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    step = fs / (LEVELS - 1)
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 9.5))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()
    a = reads.of("A")

    # 1. measured vs predicted, coloured by level
    sc = ax1.scatter(a.pred * 1e3, a.y * 1e3, c=a.alpha, cmap="viridis", s=26,
                     vmin=0, vmax=1, zorder=3)
    top = 1e3 * max(a.pred.max(), a.y.max())
    ax1.plot([0, top], [0, top], "--", color="gray", lw=1, zorder=1)
    fig.colorbar(sc, ax=ax1, label=r"level  $\alpha = Y / Y_{max}$")
    ax1.set_xlabel("predicted coherent Y  (mV)")
    ax1.set_ylabel("measured, dark and single-beam removed  (mV)")
    ax1.set_title(f"A ladder: {len(a)} patterns across the full scale")

    # 2. the headline -- residual against level
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.axhspan(-step / 4, step / 4, color="tab:green", alpha=0.10,
                label="+/- 1/4 level step")
    ax2.scatter(a.alpha, a.resid * 1e3, c=a.alpha, cmap="viridis", s=26,
                vmin=0, vmax=1, zorder=3)
    if len(a) > 2:
        lv, mean, sem = ladder_bias(a)
        ax2.errorbar(lv / (LEVELS - 1), mean * 1e3, yerr=sem * 1e3, fmt="s-",
                     color="crimson", ms=6, lw=1.6, capsize=3, zorder=4,
                     label="mean per 4-bit level")
    ax2.set_xlabel(r"predicted level  $\alpha$")
    ax2.set_ylabel("residual = measured - predicted  (mV)")
    ax2.set_title("A: is the bias growing with level?")
    ax2.legend(fontsize=8, loc="best")

    # 3. pulls
    ax3.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax3.axhline(0, color="gray", ls="--", lw=1)
    ax3.scatter(a.alpha, a.resid / a.sigma, c=a.alpha, cmap="viridis", s=26,
                vmin=0, vmax=1, zorder=3)
    ax3.set_xlabel(r"predicted level  $\alpha$")
    ax3.set_ylabel(r"pull = (meas - pred) / $\sigma$")
    ax3.set_title(f"A: pulls   (std {np.std(a.resid / a.sigma):.2f})")
    ax3.legend(fontsize=8)

    # 4. B -- the 4-bit noise budget
    groups = repeat_spread(reads)
    if groups:
        alphas = [g[0] for g in groups]
        spread = [g[1].std(ddof=1) if g[1].size > 1 else np.nan for g in groups]
        ax4.bar(range(len(groups)), np.array(spread) * 1e3, width=0.6,
                color="tab:purple", alpha=0.85)
        ax4.axhline(step / 4 * 1e3, color="crimson", ls="--", lw=1.6,
                    label="1/4 level step -- the 4-bit budget")
        ax4.axhline(step * 1e3, color="black", ls=":", lw=1.4, label="one level step")
        for i, (alpha, ys) in enumerate(groups):
            ax4.text(i, (spread[i] or 0) * 1e3, f"{(spread[i] or 0) * 1e3:.2f}",
                     ha="center", va="bottom", fontsize=8)
        ax4.set_xticks(range(len(groups)))
        ax4.set_xticklabels([f"{alpha:.2f}" for alpha in alphas])
        ax4.set_yscale("log")
        ax4.set_xlabel(r"level  $\alpha$  (same drive, repeated)")
        ax4.set_ylabel("spread of the repeats, std  (mV)")
        ax4.set_title("B: can one read resolve a 4-bit level?")
        ax4.legend(fontsize=8, loc="best")

    # 5. C -- the scan, and where full scale really is
    if peak:
        for tag, marker in (("start", "o"), ("end", "s")):
            m = peak["when"] == tag
            if not m.any():
                continue
            order = np.argsort(peak["offsets"][m])
            ax5.plot(peak["offsets"][m][order], peak["y"][m][order] * 1e3,
                     marker + "-", ms=6, lw=1.4, label=f"measured ({tag})")
            if tag in peak and np.isfinite(peak[tag]["peak_deg"]):
                ax5.axvline(peak[tag]["peak_deg"], ls=":", lw=1.2, color="0.4")
        order = np.argsort(peak["offsets"])
        ax5.plot(peak["offsets"][order], peak["pred"][order] * 1e3, "--",
                 color="gray", lw=1.4, label="predicted")
        ax5.axvline(0, color="crimson", ls="--", lw=1,
                    label=r"$\psi^*$ from the model")
        ax5.set_xlabel(r"projection axis offset from $\psi^*$  (deg)")
        ax5.set_ylabel("measured Y  (mV)")
        ax5.set_title("C: where is full scale, really?")
        ax5.legend(fontsize=8, loc="lower center")

    # 6. error per 4-bit level, against the measured full scale
    if len(a) > 2:
        lv = np.arange(LEVELS)
        rms = np.array([np.sqrt(np.mean(a.resid[a.level == k] ** 2))
                        if np.any(a.level == k) else np.nan for k in lv])
        ax6.bar(lv, 100 * rms / fs, width=0.7, color="tab:blue", alpha=0.85)
        ax6.axhline(100 * (step / 4) / fs, color="crimson", ls="--", lw=1.6,
                    label="1/4 level step")
        ok = np.isfinite(rms)
        ax6.axhline(100 * np.sqrt(np.nanmean(rms[ok] ** 2)) / fs, color="black",
                    ls=":", lw=1.4, label="mean over levels")
        ax6.set_xticks(lv)
        ax6.set_xlabel("4-bit level of the prediction")
        ax6.set_ylabel("RMSE / FS  (%)")
        ax6.set_title(f"Error per level, on measured FS = {fs * 1e3:.2f} mV")
        ax6.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_csv(path, run, *, pairs=None, method=None, out_dir=None,
               y_sat: float | None = None) -> Reads:
    """Recompute a recorded run and write its report; no hardware."""
    path = Path(path)
    out_dir = Path(path.parent if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reads, meta = load_reads(path, run, pairs=pairs, method=method, y_sat=y_sat)
    _note_y_sat(path, meta, y_sat)
    note_step7_moved(run, meta)

    peak = psi_peak(reads)
    fs = print_report(reads, peak, meta)
    png = out_dir / f"{path.stem}_report.png"
    make_plot(reads, peak, fs, png)
    print(f"\n  Plot saved to {png}")
    return reads
