"""Re-fit the 0904 run against what the SLM was ACTUALLY driven to.

The 0904 run encoded with ``method="interp"``: a requested value ``v`` was
turned into a grayscale level by interpolating the measured step-3b sweep.  The
fitted sin^2 transfer model says that level did not deliver ``v`` -- it
delivered ``sin^2(Delta(L)/2)`` for the level's fitted retardance.  On pair 4's
x channel the sweep has a noise spike at level 770 that ``argmax`` took as full
scale, so a commanded ``x = 1.0`` wrote level 770 and delivered 0.928.

This harness replays that.  For every commanded value in the 0904 measurement
CSVs it recovers the integer level the run wrote and reads the delivered value
off the fitted curve::

    v_cmd --level_for_interp--> L --fit.retardance--> Delta --sin^2(Delta/2)--> v_driven

The conversion is exact, not approximate: none of steps 6/7/8 passes a
``col_ratio``, so a channel's whole band sat at ONE integer level, and
``level_for_interp`` replays bit-for-bit from the step-3b JSON.

WHAT THIS IS AND IS NOT
-----------------------
This is a systematic-error correction of an existing run: the voltages are
untouched, only the x/w axis is relabelled.  It is NOT a re-run.  A genuinely
fit-encoded run would place its points at DIFFERENT levels -- pair 4's x = 1.0
would go to level 850, which that sweep never sampled on that channel -- so
this says nothing about what the panel does there.  Only hardware does.

It also cannot, on its own, show that the fitted encoding is better: it uses
the fit as truth to relabel, so "the refit gives a nicer eta" would be
circular.  The falsifiable part is the INTERNAL CONSISTENCY checks, which
compare independent measurements against each other and not against the fit:

  * step 6's product check -- (1, 0.25) vs (0.5, 0.5), commanded identical,
    delivered differing by up to 3.8%
  * step 6's intercept check -- the difference line's beta0 vs the
    independently measured a_x
  * step 7's per-point pulls
  * step 8's prediction residual, the strongest, since it predicts from the
    step-6 + step-7 parameters against measurements that fed neither

A wrong relabelling makes those WORSE.  That is what to read the outputs on.

THE STEP MODULES ARE NOT MODIFIED
---------------------------------
Everything here either reuses a step function unchanged, subclasses one of its
types, or rebinds a module-level constant for the duration of the run.  Two
pieces of step 6 cannot be reused as-is and are reimplemented below:

  * ``Level``'s block classification tests literal values (``x == 1.0``,
    ``x == 0.0``).  Delivered values are 0.928 and 0.0007, so every level would
    reclassify and the fit would collapse.  :class:`DrivenLevel` keeps the
    COMMANDED pair as the level's identity -- block, cross line, fit window,
    verification flag -- while ``x``/``w`` carry what was delivered.  The fit
    window in particular stays keyed on the commanded w: which levels to trust
    is a design choice about the grid, not a property of where they landed.

  * ``fit_difference`` builds its design as ``[w, 1]``, i.e. it hardcodes
    ``x == 1`` on the cross line, so on pair 4 it would return ``eta^2 * 0.928``.
    :func:`fit_difference_driven` uses ``[x*w, x]``, which is the same model
    with that assumption lifted and reduces to ``[w, 1]`` exactly when x = 1.

Everything else -- ``fit_background`` (already general in x and w),
``design_bg``, ``grad_bg_w``, ``PairV2Fit``, ``report``, ``save_combined_json``,
all of step 7's fit, all of step 8's ``compare`` -- is the step module's own.

Run:  python refit_as_driven.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.slm_response import load_calibration_result  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration            # noqa: E402
from calibration_module.steps import calib_step6_v2 as s6                  # noqa: E402
from calibration_module.steps import calib_step7_v2 as s7                  # noqa: E402
from calibration_module.steps import calib_step8_v2 as s8                  # noqa: E402
from calibration_module.phase import phi_half                              # noqa: E402

# ---- inputs: the 0904 run and the step-3b sweep it encoded against ----------
SRC_RUN = REPO_ROOT / "src/calib_data/archive/2026-09/run_0904_0117"
IN_STEP3 = REPO_ROOT / "src/calib_data/archive/2026-09/run_0903/calib_step3b_0903_1517.json"

CSV6 = HERE / "calib_step6v2_meas_0904_0140.csv"
CSV7 = HERE / "calib_step7_meas_0904_0146.csv"
CSV8 = HERE / "calib_step8_simple_0904_0150.csv"

OUT6 = HERE / "calib_step6v2_meas_asdriven.csv"
OUT7 = HERE / "calib_step7_meas_asdriven.csv"
OUT8 = HERE / "calib_step8_simple_asdriven.csv"

# the 0904 run's configuration (step 6 pairs, step 7 ref, step 8 blocks)
PAIRS = (2, 4, 5)
REF_INDEX = 2
STAMP = time.strftime("%m%d_%H%M")


# ======================================================================
# the conversion
# ======================================================================

class Driven:
    """commanded v -> level the 0904 run wrote -> value that level delivered."""

    def __init__(self, step3_path: Path):
        calib = load_calibration_result(step3_path)
        # what the run drove with, and what the fitted model says it delivered
        self.lay_cmd = channel_layout_from_calibration(calib, method="interp",
                                                       warn=False)
        self.lay_fit = channel_layout_from_calibration(calib, method="fit",
                                                       warn=False)
        self._memo: dict[tuple[int, str, float], tuple[int, float, float]] = {}

    def _channels(self, pair: int, side: str):
        slot = pair - s6.PAIR_INDEX_BASE
        chans = (self.lay_cmd.x_channels if side == "x" else self.lay_cmd.w_channels)
        fits = (self.lay_fit.x_channels if side == "x" else self.lay_fit.w_channels)
        return chans[slot], fits[slot]

    def __call__(self, pair: int, side: str, v_cmd: float):
        """-> (level written, value delivered, retardance delivered [rad])."""
        key = (pair, side, round(float(v_cmd), 9))
        hit = self._memo.get(key)
        if hit is None:
            ch_cmd, ch_fit = self._channels(pair, side)
            level = int(ch_cmd.level_for_interp(float(v_cmd)))
            delta = float(ch_fit.transfer_fit.retardance(float(level)))
            hit = (level, float(np.sin(delta / 2.0) ** 2), delta)
            self._memo[key] = hit
        return hit

    def value(self, pair: int, side: str, v_cmd: float) -> float:
        return self(pair, side, v_cmd)[1]


DRIVEN = Driven(IN_STEP3)


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def _banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ======================================================================
# CSV rewriting
# ======================================================================
# Every rewritten file keeps the COMMANDED values in `*_cmd` columns beside the
# corrected ones.  The corrected columns keep their original names, so the
# stock step loaders (all csv.DictReader) read these files unchanged and the
# extra columns are simply ignored -- the file stays a complete record of both
# what was asked for and what was delivered.

def rewrite_step6() -> Path:
    """x, w, product -> delivered.  Adds x_cmd/w_cmd/level_x/level_w."""
    rows = _rows(CSV6)
    header = ["trial", "pair_index", "x", "w", "product",
              "voltage_mean_v", "voltage_std_v", "std_ratio",
              "x_cmd", "w_cmd", "level_x", "level_w"]
    out = []
    for r in rows:
        pair = int(float(r["pair_index"]))
        xc, wc = float(r["x"]), float(r["w"])
        lx, xd, _ = DRIVEN(pair, "x", xc)
        lw, wd, _ = DRIVEN(pair, "w", wc)
        out.append({
            "trial": r["trial"], "pair_index": r["pair_index"],
            "x": f"{xd:.9g}", "w": f"{wd:.9g}", "product": f"{xd * wd:.9g}",
            "voltage_mean_v": r["voltage_mean_v"],
            "voltage_std_v": r["voltage_std_v"],
            "std_ratio": r["std_ratio"],
            "x_cmd": f"{xc:.6g}", "w_cmd": f"{wc:.6g}",
            "level_x": lx, "level_w": lw,
        })
    _write(OUT6, header, out, "step-6 v2 measurement, x/w replaced by delivered")
    return OUT6


def rewrite_step7() -> Path:
    """phi_xt_deg, phi_wt_deg, x_t, w_t, x_r, w_r -> delivered."""
    rows = _rows(CSV7)
    header = ["trial", "tgt_index", "ref_index", "phi_xt_deg", "phi_wt_deg",
              "x_t", "w_t", "x_r", "w_r", "dark_v",
              "voltage_mean_v", "voltage_std_v", "std_ratio",
              "x_t_cmd", "w_t_cmd", "x_r_cmd", "w_r_cmd",
              "level_x_t", "level_w_t", "level_x_r", "level_w_r"]
    out = []
    for r in rows:
        tgt, ref = int(float(r["tgt_index"])), int(float(r["ref_index"]))
        cmds = {"x_t": (tgt, "x", float(r["x_t"])),
                "w_t": (tgt, "w", float(r["w_t"])),
                "x_r": (ref, "x", float(r["x_r"])),
                "w_r": (ref, "w", float(r["w_r"]))}
        got = {k: DRIVEN(p, s, v) for k, (p, s, v) in cmds.items()}
        row = {"trial": r["trial"], "tgt_index": r["tgt_index"],
               "ref_index": r["ref_index"]}
        # phi_*_deg is the full retardance the phase model assigns the target's
        # channels.  Under the fit it IS the delivered retardance -- that is the
        # identity the fitted encoding exists to make exact.
        row["phi_xt_deg"] = f"{np.degrees(got['x_t'][2]):.6g}"
        row["phi_wt_deg"] = f"{np.degrees(got['w_t'][2]):.6g}"
        for k in ("x_t", "w_t", "x_r", "w_r"):
            row[k] = f"{got[k][1]:.9g}"
        row["dark_v"] = r["dark_v"]
        row["voltage_mean_v"] = r["voltage_mean_v"]
        row["voltage_std_v"] = r["voltage_std_v"]
        row["std_ratio"] = r["std_ratio"]
        for k in ("x_t", "w_t", "x_r", "w_r"):
            row[f"{k}_cmd"] = f"{cmds[k][2]:.6g}"
            row[f"level_{k}"] = got[k][0]
        out.append(row)
    _write(OUT7, header, out, "step-7 v2 measurement, x/w replaced by delivered")
    return OUT7


def rewrite_step8() -> Path:
    """x_<k>, w_<k> -> delivered, per pair.

    ``v`` is left at the COMMANDED value on purpose: it is the block's sweep
    abscissa, one number shared by every driven pair in the row, and the three
    pairs delivered three different values for it.  Replacing it with any one
    of them would make the plot's x-axis mean nothing.  The per-pair columns
    carry the delivered values, and :func:`predict_curve_driven` reads those.
    """
    rows = _rows(CSV8)
    cols = [f"{s}_{k}" for k in PAIRS for s in ("x", "w")]
    header = (["block", "pairs", "v"] + cols
              + ["dark_v", "voltage_mean_v", "voltage_std_v", "std_ratio",
                 "pred_v"]
              + [f"{c}_cmd" for c in cols]
              + [f"level_{c}" for c in cols])
    out = []
    for r in rows:
        driven = tuple(int(t) for t in r["pairs"].split("+"))
        row = {"block": r["block"], "pairs": r["pairs"], "v": r["v"]}
        for k in PAIRS:
            for side in ("x", "w"):
                c = f"{side}_{k}"
                cmd = float(r[c])
                if k in driven:
                    lvl, val, _ = DRIVEN(k, side, cmd)
                else:
                    lvl, val = DRIVEN(k, side, 0.0)[0], 0.0   # not driven
                row[c] = f"{val:.9g}"
                row[f"{c}_cmd"] = f"{cmd:.6g}"
                row[f"level_{c}"] = lvl
        for c in ("dark_v", "voltage_mean_v", "voltage_std_v", "std_ratio"):
            row[c] = r[c]
        row["pred_v"] = ""      # recomputed by the comparison below
        out.append(row)
    _write(OUT8, header, out,
           "step-8 simple measurement, per-pair x/w replaced by delivered")
    return OUT8


def _write(path: Path, header: list[str], rows: list[dict], what: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# {what}\n")
        f.write(f"# source: {CSV6.name if 'step-6' in what else ''}"
                f"{CSV7.name if 'step-7' in what else ''}"
                f"{CSV8.name if 'step-8' in what else ''}"
                f"  (0904 run, encoded with method=interp)\n")
        f.write(f"# delivered = sin^2(Delta(L)/2) from the fitted transfer model "
                f"in {IN_STEP3.name}\n")
        f.write("# *_cmd columns are the values the run asked for; level_* is "
                "the grayscale it actually wrote\n")
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.name}  ({len(rows)} rows)")


# ======================================================================
# step 6:  the two pieces that cannot be reused verbatim
# ======================================================================

@dataclass
class DrivenLevel(s6.Level):
    """A step-6 level whose identity is commanded but whose value is delivered.

    ``x``/``w`` (inherited) are what the panel actually delivered, so every
    model evaluation -- ``design_bg``, the difference design, the product check
    -- sees the truth.  ``x_cmd``/``w_cmd`` are what the grid asked for, and all
    the BOOKKEEPING keys off those: which block a level belongs to, whether it
    is on the cross line, whether it is inside the slope-fit window, whether it
    is a verification point.  That split is the whole trick, and it is also the
    physically right one -- the fit window is a statement about which grid
    points the estimator trusts, not about where they happened to land.
    """

    x_cmd: float = 0.0
    w_cmd: float = 0.0

    @property
    def block(self) -> str:
        return s6.block_of(self.x_cmd, self.w_cmd)

    @property
    def is_verification(self) -> bool:
        return (round(self.x_cmd, 6),
                round(self.w_cmd, 6)) in s6.verification_points()

    def on_cross_line(self) -> bool:
        return (self.block == "cross" and self.x_cmd == 1.0
                and not self.is_verification)

    def in_fit_window(self) -> bool:
        lo, hi = s6.FIT_W_RANGE
        return self.on_cross_line() and lo - 1e-9 <= self.w_cmd <= hi + 1e-9


def average_levels_driven(rows) -> list[DrivenLevel]:
    """Group by the COMMANDED setting; every repeat of one wrote the same level.

    Grouping cannot fragment: a commanded value maps deterministically to one
    integer level and therefore to one delivered value, so the repeats of a
    grid point all carry identical x/w.
    """
    grouped: dict[tuple[float, float], list] = {}
    for _rep, xc, wc, xd, wd, mean_v, std_v in rows:
        grouped.setdefault((round(xc, 6), round(wc, 6)), []).append(
            (xd, wd, float(mean_v), float(std_v)))
    levels = []
    for (xc, wc), vals in grouped.items():
        levels.append(DrivenLevel(
            x=float(vals[0][0]), w=float(vals[0][1]),
            means=np.array([m for _, _, m, _ in vals], dtype=float),
            trace_stds=np.array([s for _, _, _, s in vals], dtype=float),
            x_cmd=xc, w_cmd=wc,
        ))
    levels.sort(key=lambda L: (s6._BLOCK_ORDER[L.block], L.x_cmd, L.w_cmd))
    return levels


def _level_at_cmd(levels, x_cmd: float, w_cmd: float):
    for L in levels:
        if abs(L.x_cmd - x_cmd) < 1e-9 and abs(L.w_cmd - w_cmd) < 1e-9:
            return L
    return None


def fit_difference_driven(levels, bg, bg_cov) -> dict:
    """GLS ``D = eta^2*(x*w) + a_x*x`` over the in-window cross levels.

    Step 6's own ``fit_difference`` writes this as ``D(w) = eta^2*w + beta0``,
    which is the same line under ``x == 1``.  The 0904 run never reached x = 1
    on pair 4 -- a commanded 1.0 wrote level 770 and delivered 0.928 -- so the
    design carries x explicitly.  With x = 1 the two are identical, column for
    column; with x < 1 step 6's version would return ``eta^2 * x``, low by 7.2%
    on that pair.

    The covariance construction is step 6's, unchanged: D is formed against the
    FITTED background, so its points share the background parameters and are
    correlated, and the slope is a GLS solution against the full matrix.
    """
    used = [L for L in levels if L.in_fit_window()]
    if len(used) < 2:
        raise ValueError(f"need >= 2 cross levels in {s6.FIT_W_RANGE}, got {len(used)}")

    x = np.array([L.x for L in used], dtype=float)
    w = np.array([L.w for L in used], dtype=float)
    y = np.array([L.mean for L in used], dtype=float)
    y_sig = np.array([L.sigma for L in used], dtype=float)

    g = s6.grad_bg_w(w)                      # step 6's own Bhat gradient
    d = y - (g @ np.array([bg[n][0] for n in s6.PARAMS_BG]))
    cov_d = np.diag(y_sig ** 2) + g @ bg_cov @ g.T

    a = np.column_stack([x * w, x])          # <- the generalization
    cov_inv = np.linalg.pinv(cov_d)
    m = np.linalg.pinv(a.T @ cov_inv @ a)
    beta = m @ (a.T @ cov_inv @ d)
    pred = a @ beta

    resid = d - pred
    chi2 = float(resid @ cov_inv @ resid)
    one = np.ones_like(d)
    d_bar = float((one @ cov_inv @ d) / (one @ cov_inv @ one))
    total = float((d - d_bar) @ cov_inv @ (d - d_bar))
    r2 = 1.0 - chi2 / total if total > 0 else float("nan")

    return {"x": x, "w": w, "prod": x * w, "d": d,
            "sigma": np.sqrt(np.clip(np.diag(cov_d), 0.0, None)),
            "pred": pred, "beta": beta, "cov": m, "r2": r2, "used": used}


def verify_product_driven(fit) -> list[dict]:
    """Step 6's product check, looked up by commanded and evaluated on delivered.

    This is the falsifiable one.  ``PRODUCT_CHECKS`` groups levels the grid
    COMMANDED to the same x*w, which is why the lookup stays on the commanded
    pair -- but what the panel delivered are two different products (up to 3.8%
    apart on the 0904 run), and the model's prediction ``eta^2 * (x*w)`` has to
    use those.  Correcting the labels should bring the two residues together;
    if it pushes them apart, the transfer fit is wrong.
    """
    out = []
    p = np.array([fit.bg[n][0] for n in s6.PARAMS_BG])
    for group in s6.PRODUCT_CHECKS:
        pts = []
        for x_cmd, w_cmd in group:
            L = _level_at_cmd(fit.levels, x_cmd, w_cmd)
            if L is None:
                continue
            g = s6.design_bg([L.x], [L.w])        # delivered
            pts.append({"x": L.x, "w": L.w, "n": L.n,
                        "tpa": L.mean - float((g @ p)[0]),
                        "tpa_err": float(np.sqrt(L.sigma ** 2
                                                 + float((g @ fit.bg_cov @ g.T)[0, 0]))),
                        "product": L.x * L.w, "_g": g[0], "_sigma": L.sigma})
        if len(pts) < 2:
            continue
        rec = {"product": float(np.mean([q["product"] for q in pts])),
               "expected": fit.b * float(np.mean([q["product"] for q in pts])),
               "points": pts, "pairs": []}
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                dg = a["_g"] - b["_g"]
                err = float(np.sqrt(max(a["_sigma"] ** 2 + b["_sigma"] ** 2
                                        + float(dg @ fit.bg_cov @ dg), 0.0)))
                # the model no longer says these are equal: it says they differ
                # by eta^2 * (product_a - product_b), which is now nonzero
                expect = fit.b * (a["product"] - b["product"])
                diff = (a["tpa"] - b["tpa"]) - expect
                mid = 0.5 * (a["tpa"] + b["tpa"])
                rec["pairs"].append({
                    "a": (a["x"], a["w"]), "b": (b["x"], b["w"]),
                    "diff": diff, "err": err, "expected_diff": expect,
                    "pull": diff / err if err else float("nan"),
                    "frac": diff / mid if mid else float("nan")})
        out.append(rec)
    return out


def fit_pair_driven(index: int, levels) -> s6.PairV2Fit:
    """Step 6's ``fit_pair`` with the two generalized pieces swapped in."""
    bg, bg_cov = s6.fit_background(levels)          # step 6's own, already general
    df = fit_difference_driven(levels, bg, bg_cov)

    b, beta0 = float(df["beta"][0]), float(df["beta"][1])
    b_err = float(np.sqrt(max(df["cov"][0, 0], 0.0)))
    beta0_err = float(np.sqrt(max(df["cov"][1, 1], 0.0)))
    if b > 0:
        eta = float(np.sqrt(b))
        eta_err = b_err / (2.0 * eta)
    else:
        eta, eta_err = float("nan"), float("nan")

    dark = _level_at_cmd(levels, 0.0, 0.0)
    x_on = _level_at_cmd(levels, 1.0, 0.0)
    anchor = ((x_on.mean - dark.mean, float(np.hypot(x_on.sigma, dark.sigma)))
              if dark is not None and x_on is not None
              else (float("nan"), float("nan")))

    excluded = []
    for L in levels:
        if not L.on_cross_line() or L.in_fit_window():
            continue
        g = s6.grad_bg_w([L.w])
        d_meas = L.mean - float((g @ np.array([bg[n][0] for n in s6.PARAMS_BG]))[0])
        d_err = float(np.sqrt(L.sigma ** 2 + float((g @ bg_cov @ g.T)[0, 0])))
        a_row = np.array([L.x * L.w, L.x])
        excluded.append((L.w, d_meas, d_err,
                         float(a_row @ df["beta"]),
                         float(np.sqrt(max(a_row @ df["cov"] @ a_row, 0.0)))))

    fit = s6.PairV2Fit(
        index=index, levels=levels, bg=bg, bg_cov=bg_cov,
        # the difference panel's abscissa is now the delivered PRODUCT x*w,
        # which is the variable the line is straight in
        fit_w=df["prod"], fit_d=df["d"], fit_sigma=df["sigma"], fit_pred=df["pred"],
        b=b, b_err=b_err, beta0=beta0, beta0_err=beta0_err,
        eta=eta, eta_err=eta_err, r2=float(df["r2"]),
        anchor=anchor, excluded=excluded)
    # beta0 is now the coefficient of the x column, i.e. a_x itself, so step 6's
    # intercept check compares two unbiased numbers and keeps its meaning
    fit.checks = {"intercept": s6.verify_intercept(fit),
                  "product": verify_product_driven(fit)}
    fit._driven_x = float(df["x"][0])       # cross-line drive, for the plot/report
    return fit


def plot_pair(fit, path: Path) -> None:
    """The difference panel, drawn in the delivered-product coordinate.

    Step 6's ``make_plot`` draws the line as ``b*w + beta0``, which is only the
    fitted line when x = 1; here the line is ``eta^2*(x*w) + a_x*x``.  Rather
    than hand it a coordinate it would mis-draw, this plots the same content in
    the coordinate the generalized fit is straight in.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xc = fit._driven_x
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.suptitle(f"pair {fit.index} -- re-fit as driven "
                 f"(cross line delivered x = {xc:.4f})",
                 fontsize=13, fontweight="bold")

    p_fine = np.linspace(0.0, float(fit.fit_w.max()) * 1.08, 200)
    ax1.plot(p_fine, (fit.b * p_fine + fit.beta0 * xc) * 1e3, "-",
             color="tab:green", lw=1.5,
             label="GLS  $D = \\eta^2 (xw) + a_x x$")
    order = np.argsort(fit.fit_w)
    ax1.errorbar(fit.fit_w[order], fit.fit_d[order] * 1e3,
                 yerr=fit.fit_sigma[order] * 1e3, fmt="o", ms=5,
                 color="tab:blue", capsize=3, label="cross levels (delivered)")
    ax1.set_xlabel("delivered product  $x\\,w$")
    ax1.set_ylabel("$D$ (mV)")
    ax1.grid(alpha=.25)
    ax1.legend(fontsize=9)
    ax1.set_title(f"$\\eta$ = {fit.eta:.5g} $\\pm$ {fit.eta_err:.2g}   "
                  f"$R^2$ = {fit.r2:.5f}", fontsize=10)

    ax2.axhline(0, color="k", lw=.8)
    ax2.plot(fit.fit_w[order], fit.pulls[order], "o-", ms=5, color="tab:red")
    for lim in (-1, 1):
        ax2.axhline(lim, color="gray", ls=":", lw=1)
    ax2.set_xlabel("delivered product  $x\\,w$")
    ax2.set_ylabel("pull  (resid / $\\sigma$)")
    ax2.set_title(f"max |pull| = {np.abs(fit.pulls).max():.2f}", fontsize=10)
    ax2.grid(alpha=.25)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_step6() -> Path:
    _banner("STEP 6  -- re-fit as driven")
    rows_by_pair: dict[int, list] = {}
    for r in _rows(OUT6):
        rows_by_pair.setdefault(int(float(r["pair_index"])), []).append((
            int(float(r["trial"])),
            float(r["x_cmd"]), float(r["w_cmd"]),
            float(r["x"]), float(r["w"]),
            float(r["voltage_mean_v"]), float(r["voltage_std_v"])))

    layout = channel_layout_from_calibration(
        load_calibration_result(IN_STEP3), method="fit", warn=False)

    fits = []
    for index in sorted(rows_by_pair):
        levels = average_levels_driven(rows_by_pair[index])
        fit = fit_pair_driven(index, levels)
        fit.wl_x_nm, fit.wl_w_nm, fit.nominal_wl_nm = s6._pair_wavelengths(layout, index)
        fits.append(fit)
        print(f"\n=== pair {index} "
              f"(cross line delivered x = {fit._driven_x:.4f}) ===")
        s6.report(fit)
        png = HERE / f"calib_step6v2_pair{index}_asdriven.png"
        plot_pair(fit, png)
        print(f"Plot saved to {png}")

    out_json = HERE / f"calib_step6v2_result_asdriven_{STAMP}.json"
    old_calib, old_in3 = s6.CALIB_PATH, s6.IN_STEP3
    try:
        s6.CALIB_PATH, s6.IN_STEP3 = HERE, IN_STEP3
        s6.save_combined_json(fits, out_json,
                              center_wl=float(getattr(layout, "center_wl", 0.0)))
    finally:
        s6.CALIB_PATH, s6.IN_STEP3 = old_calib, old_in3
    print(f"\nSaved -> {out_json}")
    return out_json


# ======================================================================
# step 7:  stock fit_csv, redirected
# ======================================================================

def run_step7(step6_json: Path) -> Path:
    _banner("STEP 7  -- re-fit as driven (step 7's own fit, unmodified)")
    saved = (s7.IN_STEP6, s7.OUT_DIR, s7.REF_INDEX, s7.TGT_INDICES)
    try:
        s7.IN_STEP6 = step6_json
        s7.OUT_DIR = HERE
        s7.REF_INDEX = REF_INDEX
        s7.TGT_INDICES = list(PAIRS)
        s7.fit_csv(OUT7)
    finally:
        s7.IN_STEP6, s7.OUT_DIR, s7.REF_INDEX, s7.TGT_INDICES = saved
    newest = max(HERE.glob("calib_step7_result_*.json"),
                 key=lambda p: p.stat().st_mtime)
    return newest


# ======================================================================
# step 8:  stock compare, with a per-pair forward model
# ======================================================================

def _step8_driven_lookup() -> dict:
    """{(driven, v_cmd): {pair: (x_delivered, w_delivered)}} from the CSV."""
    table: dict = {}
    for r in _rows(OUT8):
        driven = tuple(int(t) for t in r["pairs"].split("+"))
        key = (driven, round(float(r["v"]), 6))
        table[key] = {k: (float(r[f"x_{k}"]), float(r[f"w_{k}"])) for k in driven}
    return table


def make_predict_curve_driven(table):
    """Step 8's forward model with x and w free per pair.

    Step 8's own ``predict`` takes the block's scalar ``v`` and uses it for
    every driven pair: ``eta_k * v * exp(i*2*asin(sqrt(v)))``.  That is exactly
    right when the panel delivers what it was asked for -- and it is what the
    0904 run assumed.  As driven, each pair got its own x and w, so the field
    is ``eta_k * sqrt(x_k w_k) * exp(i[phi_half(x_k) + phi_half(w_k)])``, which
    is the same expression with x = w = v substituted back out.
    """
    def predict_curve_driven(models, phases, driven, levels):
        out = []
        for v in levels:
            got = table.get((tuple(driven), round(float(v), 6)))
            field, bg = 0.0 + 0.0j, 0.0
            for k in driven:
                x, w = got[k]
                m = models[k]
                ph = float(phi_half(x)) + float(phi_half(w))
                field += m.eta * np.sqrt(max(x * w, 0.0)) * np.exp(1j * (ph + phases[k]))
                bg += float(m.single_beam(x, w))
            out.append(float(np.abs(field) ** 2) + bg)
        return np.array(out)
    return predict_curve_driven


def run_step8(step7_json: Path) -> Path:
    _banner("STEP 8  -- compare as driven (step 8's own compare, unmodified)")
    png = HERE / f"calib_step8_compare_asdriven_{STAMP}.png"
    saved = (s8.IN_STEP7, s8.OUT_DIR, s8.PAIRS, s8.predict_curve)
    try:
        s8.IN_STEP7, s8.OUT_DIR, s8.PAIRS = step7_json, HERE, list(PAIRS)
        s8.predict_curve = make_predict_curve_driven(_step8_driven_lookup())
        pairs, blocks = s8.load_csv(OUT8)
        # method=None -> "the single stored fit", which is step 8's own "auto"
        _layout, models, phases = s8.load_inputs(None, pairs)
        s8.compare(blocks, models, phases, method=s8._method_tag(None),
                   png_path=png)
    finally:
        s8.IN_STEP7, s8.OUT_DIR, s8.PAIRS, s8.predict_curve = saved
    return png


# ======================================================================

def main() -> int:
    _banner("CONVERSION  -- commanded -> level written -> delivered")
    print(f"  step-3b sweep : {IN_STEP3}")
    print(f"  0904 run      : {SRC_RUN}")
    print("  the 0904 run encoded with method=interp; delivered values come "
          "from the fitted sin^2 model\n")
    for p in PAIRS:
        for side in ("x", "w"):
            cells = []
            for v in (0.0, 0.25, 0.5, 0.9, 1.0):
                lvl, val, _ = DRIVEN(p, side, v)
                cells.append(f"{v:.2f}->{val:.4f}(L{lvl})")
            print(f"  pair {p} {side}: " + "  ".join(cells))

    _banner("REWRITING THE MEASUREMENT CSVs")
    rewrite_step6()
    rewrite_step7()
    rewrite_step8()

    j6 = run_step6()
    j7 = run_step7(j6)
    png = run_step8(j7)

    _banner("OUTPUTS")
    for p in sorted(HERE.iterdir()):
        if p.name != Path(__file__).name:
            print(f"  {p.name}")
    print(f"\n  step-8 comparison PNG -> {png.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
