"""Measure how many bits the bench really resolves, off one ladder CSV.

    python report.py bit_acc_chara_1029_2033.csv

The other end of ``src/drafts/bit_acc_chara.py``, which drives a uniform ladder
and writes one row per read.  Everything this needs is in that CSV, so the
analysis can be re-run, re-cut and re-plotted without the bench and without
importing anything from ``src``.

Undoing the instrument response
-------------------------------
The detector sees, for gain ``g``, the single-beam response ``sb`` and the
coherent signal ``Y``::

    V = g * [ Y + sb(x, w) ] + dark        so   Y = (V - dark) / g - sb

in that order -- ``sb`` is light that went through the same gain.  ``sb`` is
NOT a column, but it is exactly recoverable from two that are::

    sb = pred_v - alpha * Y_max

because ``pred_v`` is the model's ``coherent + sb`` while ``alpha`` is defined
on the coherent part alone.  On the 1029 ladder that pedestal runs 0.28 mV at
the bottom level to 0.37 mV at the top: 3.2% of the reading at level 1 and
0.27% at level 16.  It is a CURVED pedestal, so leaving it in makes a straight
ladder look bent.

``g`` is fitted on the FIRST REPEAT alone -- the 16 ``rep == 0`` rows, one per
level -- as the origin slope of ``(V - dark)`` against ``pred_v``, the same
form :mod:`onn_module.onn_infer` fits its gain anchors with.  Every other
repeat is then a holdout: the gain never saw the data whose error this quotes,
so the ideal line is a prediction rather than a fit to the points under it.

Note that :func:`bit_acc_chara.build_reads` shuffles the whole (level x repeat)
list, so ``rep == 0`` is scattered across the whole run rather than sitting at
the start.  ``g`` is therefore a run-average, not a start-of-run calibration.

What the numbers mean
---------------------
``Y_ideal = alpha * Y_max`` is what the drive asked for.  ``LSB`` is the ideal
step of the ladder, ``Y_ideal`` at the top divided by the number of levels.

``INL`` is each level's distance from ``Y_ideal`` -- measured against the
identity, not against a line refitted to the data, so a gain error cannot hide
in it.  ``DNL`` is how far each step between adjacent levels is from one LSB;
a DNL at or below ``-1`` is a missing code.

``bits`` is ``log2(range / error)``: how many levels the ladder separates at
one sigma of the quoted error.  It is a separation count, not an ADC's SINAD
ENOB, and the error it divides by is named on each line.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


# ======================================================================
# the input
# ======================================================================

def read_rows(path: Path):
    """``(rows, meta)`` -- one dict per read, plus the trailing ``# key,value``.

    The metadata lines are split off before the reader sees them, so a row that
    still fails to parse is a real problem and is dropped loudly.
    """
    body, meta_lines = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for line in f.read().splitlines():
            (meta_lines if line.startswith("#") else body).append(line)

    meta = {}
    for line in meta_lines:
        key, _, value = line.lstrip("#").partition(",")
        meta[key.strip()] = value.strip()

    rows, bad = [], 0
    for row in csv.DictReader(body):
        try:
            rows.append({
                "level": int(row["level"]),
                "rep": int(row["rep"]),
                "seq": int(row["seq"]),
                "alpha": float(row["alpha"]),
                "pred": float(row["pred_v"]),
                "v": float(row["voltage_mean_v"]),
                "dark": float(row["dark_v"]),
                "trace_std": float(row["voltage_std_v"]),
                "range": float(row["range_v"]),
                "peak": float(row.get("peak_v") or "nan"),
            })
        except (KeyError, TypeError, ValueError):
            bad += 1
    if not rows:
        raise SystemExit(f"{path}: no readable ladder rows -- is this a "
                         f"bit_acc_chara CSV?")
    if bad:
        print(f"warning: {bad} row(s) could not be parsed and were dropped")
    return rows, meta


def model_scale(meta: dict) -> float:
    """``Y_max`` from the meta -- without it neither ``sb`` nor the ideal exists."""
    if "y_max_model_v" not in meta:
        raise SystemExit("the CSV has no '# y_max_model_v' line, so alpha "
                         "cannot be turned into a volts target and the "
                         "single-beam term cannot be separated from pred_v")
    y_max = float(meta["y_max_model_v"])
    if y_max <= 0.0:
        raise SystemExit(f"y_max_model_v is {y_max}, expected a positive volts")
    return y_max


# ======================================================================
# undoing the instrument response
# ======================================================================

def split_prediction(rows, y_max: float) -> None:
    """Give every row its ``y_ideal`` and its single-beam pedestal ``sb``."""
    for r in rows:
        r["y_ideal"] = r["alpha"] * y_max
        r["sb"] = r["pred"] - r["y_ideal"]


def fit_gain(rows):
    """``(g, calibration_rows)`` from the first repeat alone.

    The origin slope of ``(V - dark)`` against the model's ``pred_v``: the
    forward direction, which is linear in ``g`` and so has a closed form, and
    the same fit ``onn_infer`` uses for its gain anchors.
    """
    cal = [r for r in rows if r["rep"] == 0]
    if len(cal) < 2:
        raise SystemExit(f"only {len(cal)} row(s) with rep == 0; the gain is "
                         f"fitted on the first repeat and needs the ladder")
    den = sum(r["pred"] ** 2 for r in cal)
    if den <= 0.0:
        raise SystemExit("every predicted level is zero -- nothing to fit g on")
    g = sum((r["v"] - r["dark"]) * r["pred"] for r in cal) / den
    if g <= 0.0:
        raise SystemExit(f"fitted g = {g:.4g}, not positive: the bright "
                         f"patterns read below the dark")
    return g, cal


def recover(rows, g: float) -> None:
    """``Y = (V - dark) / g - sb`` for every read -- divide, then subtract."""
    for r in rows:
        r["y"] = (r["v"] - r["dark"]) / g - r["sb"]


def per_level(rows):
    """One entry per level: the target, the mean of its repeats, the spread."""
    out = []
    for k in sorted({r["level"] for r in rows}):
        group = [r for r in rows if r["level"] == k]
        y = np.array([r["y"] for r in group])
        out.append({
            "k": k,
            "n": len(group),
            "alpha": float(np.mean([r["alpha"] for r in group])),
            "sb": float(np.mean([r["sb"] for r in group])),
            "ideal": float(np.mean([r["y_ideal"] for r in group])),
            "mean": float(y.mean()),
            "std": float(y.std(ddof=1)) if len(y) > 1 else float("nan"),
        })
    return out


def correlate(a, b) -> float:
    """Pearson r, or nan when either side is flat."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ======================================================================
# the report
# ======================================================================

def show(levels, lsb: float) -> None:
    """The per-level table, in recovered Y: target, response, distance."""
    print(f"\n{'lvl':>4}{'alpha':>9}{'sb mV':>8}{'ideal mV':>10}{'meas mV':>10}"
          f"{'sigma mV':>10}{'SNR':>8}{'resid mV':>10}{'resid LSB':>11}{'n':>5}")
    print("-" * 85)
    for L in levels:
        resid = L["mean"] - L["ideal"]
        snr = L["mean"] / L["std"] if L["std"] > 0 else float("inf")
        print(f"{L['k']:>4}{L['alpha']:>9.4f}{L['sb'] * 1e3:>8.3f}"
              f"{L['ideal'] * 1e3:>10.3f}{L['mean'] * 1e3:>10.3f}"
              f"{L['std'] * 1e3:>10.4f}{snr:>8.1f}"
              f"{resid * 1e3:>+10.3f}{resid / lsb:>+11.3f}{L['n']:>5}")
    print("-" * 85)


def analyse(levels, rows, cal, meta, g, y_max) -> dict:
    """Print the error analysis; return the pieces the figure also needs."""
    top = max(L["ideal"] for L in levels)
    lsb = top / len(levels)
    ideal = np.array([L["ideal"] for L in levels])
    means = np.array([L["mean"] for L in levels])
    stds = np.array([L["std"] for L in levels])
    resid = means - ideal
    dnl = np.diff(np.concatenate([[0.0], means])) / lsb - 1.0   # code 0 is dark

    all_r = np.array([r["y"] - r["y_ideal"] for r in rows])
    rms = float(np.sqrt((all_r ** 2).mean()))

    show(levels, lsb)
    print(f"ideal range (dark to top level) {top * 1e3:>8.3f} mV      "
          f"1 LSB = {lsb * 1e3:.3f} mV over {len(levels)} levels")

    cal_r = np.array([(r["v"] - r["dark"]) / g - r["sb"] - r["y_ideal"]
                      for r in cal])
    print(f"\nGAIN   fitted on the first repeat, {len(cal)} rungs, then held")
    print(f"  g                      {g:>10.4f}   bench volts per model volt")
    print(f"  calibration residual   {cal_r.mean() * 1e3:>+10.4f} mV mean,"
          f" {np.abs(cal_r).max() * 1e3:>7.4f} mV worst   (rep 0, {len(cal)} reads)")
    print(f"  holdout residual       {all_r.mean() * 1e3:>+10.4f} mV mean,"
          f" {np.abs(all_r).max() * 1e3:>7.4f} mV worst   "
          f"(reps 1+, {len(rows)} reads)")
    print(f"  single-beam removed    {min(L['sb'] for L in levels) * 1e3:>10.3f}"
          f" .. {max(L['sb'] for L in levels) * 1e3:.3f} mV across the band")

    worst_inl = int(np.argmax(np.abs(resid)))
    worst_dnl = int(np.argmax(np.abs(dnl)))
    print("\nLINEARITY   against the ideal Y = alpha * Y_max, not a refitted line")
    print(f"  INL  max |meas - ideal|{np.abs(resid).max() * 1e3:>10.3f} mV"
          f"   {np.abs(resid).max() / lsb:>6.3f} LSB"
          f"   {100 * np.abs(resid).max() / top:>5.2f}% of range"
          f"   at level {levels[worst_inl]['k']}")
    print(f"  DNL  max |step - LSB|  {np.abs(dnl).max() * lsb * 1e3:>10.3f} mV"
          f"   {np.abs(dnl).max():>6.3f} LSB"
          f"   at step {worst_dnl} -> {worst_dnl + 1}")
    print(f"  missing codes          {int((dnl <= -1.0).sum()):>10}"
          f"   step(s) with DNL <= -1 LSB")

    sig_med, sig_max = float(np.median(stds)), float(np.nanmax(stds))
    print(f"\nNOISE  one read, {meta.get('t_meas_s', '?')} s window, over "
          f"{levels[0]['n']} holdout repeats")
    print(f"  median sigma           {sig_med * 1e3:>10.4f} mV"
          f"   {sig_med / lsb:>6.3f} LSB")
    print(f"  worst sigma            {sig_max * 1e3:>10.4f} mV"
          f"   {sig_max / lsb:>6.3f} LSB   at level "
          f"{levels[int(np.nanargmax(stds))]['k']}")
    print(f"  rms error, all reads   {rms * 1e3:>10.4f} mV"
          f"   {rms / lsb:>6.3f} LSB   noise and bend together")

    print("\nRESOLUTION   log2(range / error) -- levels separated at one sigma")
    for name, err in (("noise only ", sig_med),
                      ("bend only  ", float(np.abs(resid).max())),
                      ("both (rms) ", rms)):
        if err > 0:
            print(f"  {name}            {top / err:>10.1f} levels"
                  f"   {math.log2(top / err):>6.2f} bits")

    print("\nCHECKS")
    print(f"  residual vs read order   r = "
          f"{correlate([r['seq'] for r in rows], all_r):>+.3f}"
          f"   -- the shuffle is there to keep this near 0")
    peaks = np.array([r["peak"] for r in rows])
    if np.isfinite(peaks).any():
        rails = np.array([r["range"] for r in rows])
        print(f"  worst trace peak         {np.nanmax(peaks) * 1e3:>8.3f} mV"
              f"   {100 * float(np.nanmax(peaks / rails)):>5.1f}% of the "
              f"+/-{rails[0]:g} V range")
    return {"top": top, "lsb": lsb}


# ======================================================================
# the figure
# ======================================================================

def plot(levels, rows, top: float, lsb: float, out: Path) -> Path:
    """The normalized transfer function, with the residual underneath.

    Both axes run 0 to 1 against the ladder's own full scale, so the ideal is
    the dashed identity and nothing on the plot has been fitted to the points
    on it -- ``g`` came from the held-out first repeat.  Two panels because a
    sub-percent bend is invisible on an axis that spans the whole ladder: the
    top one says the response is linear, the lower one says how linear.
    """
    import matplotlib
    matplotlib.use("Agg")                     # no window, no display needed
    import matplotlib.pyplot as plt

    xs = np.array([r["y_ideal"] for r in rows]) / top
    ys = np.array([r["y"] for r in rows]) / top
    lx = np.array([L["ideal"] for L in levels]) / top
    ly = np.array([L["mean"] for L in levels]) / top
    le = np.array([L["std"] for L in levels]) / top

    # constrained, not tight: tight_layout reserves no room for a suptitle.
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.0, 8.2), sharex=True,
                                 height_ratios=[2.2, 1.0],
                                 layout="constrained")

    ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="#d62728", lw=1.4, zorder=1,
            label="ideal, y = x")
    ax.scatter(xs, ys, s=10, alpha=0.25, color="#1f77b4", linewidths=0, zorder=2,
               label=f"measured, {len(rows)} reads")
    ax.errorbar(lx, ly, yerr=le, fmt="o", ms=5, color="#0b3c5d", capsize=3,
                lw=1.0, zorder=3, label="level mean +/- 1 sigma")
    ax.set_ylabel("measured Y / full scale")
    ax.set_title("normalized transfer function")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    # The y range follows the residual, not the LSB: a ladder comfortably
    # better than its own code size would otherwise be one grey rectangle with
    # the shape squashed flat along the middle.
    rl = (ys - xs) * top / lsb
    span = max(float(np.abs(rl).max()) * 1.3, 1e-9)
    if 0.5 <= span:
        bx.axhspan(-0.5, 0.5, color="0.88", zorder=0, label="+/- 1/2 LSB")
    else:
        bx.plot([], [], " ", label=f"+/- 1/2 LSB off scale ({lsb * 1e3:.2f} mV)")
    bx.set_ylim(-span, span)
    bx.axhline(0.0, ls="--", color="#d62728", lw=1.2)
    bx.scatter(xs, rl, s=10, alpha=0.25, color="#1f77b4", linewidths=0)
    bx.errorbar(lx, (ly - lx) * top / lsb, yerr=le * top / lsb, fmt="o", ms=5,
                color="#0b3c5d", capsize=3, lw=1.0)
    bx.set_xlabel("commanded alpha / full scale")
    bx.set_ylabel("residual (LSB)")
    bx.set_title("distance from the ideal")
    bx.legend(fontsize=9, loc="upper left")
    bx.grid(alpha=0.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path, help="a bit_acc_chara ladder CSV")
    ap.add_argument("--png", type=Path, default=None, metavar="PATH",
                    help="where to write the figure (default: beside the CSV, "
                         "<csv stem>_ladder.png)")
    ap.add_argument("--no-png", action="store_true",
                    help="print the analysis and skip the figure")
    args = ap.parse_args(argv)

    every, meta = read_rows(args.csv)
    y_max = model_scale(meta)
    split_prediction(every, y_max)
    g, cal = fit_gain(every)
    recover(every, g)

    # The gain was fitted on rep 0, so rep 0 is not scored: the error quoted
    # below is the error on reads the calibration never saw.
    rows = [r for r in every if r["rep"] != 0]
    if not rows:
        raise SystemExit("every read is rep 0, so the gain fit and the "
                         "measurement would be the same data; nothing to score")
    levels = per_level(rows)
    if len(levels) < 2:
        raise SystemExit(f"{args.csv}: only {len(levels)} level(s) in the "
                         f"holdout -- a ladder needs at least two")

    print(f"csv       : {args.csv}")
    print(f"calib     : {meta.get('step7_json', '?')}")
    print(f"ladder    : {len(levels)} levels, alpha {levels[0]['alpha']:.4f} .. "
          f"{levels[-1]['alpha']:.4f}, {len(every)} reads of "
          f"{meta.get('t_meas_s', '?')} s")
    print(f"model     : Y_max {y_max * 1e3:.1f} mV   -- the MODEL's volts; the "
          f"bench's gain against them is fitted below")
    print(f"scored    : {len(rows)} reads, reps 1+; the {len(cal)} rep-0 reads "
          f"fitted g and are held out")

    out = analyse(levels, rows, cal, meta, g, y_max)

    if not args.no_png:
        png = args.png or args.csv.with_name(f"{args.csv.stem}_ladder.png")
        print(f"\nfigure -> {plot(levels, rows, out['top'], out['lsb'], png)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
