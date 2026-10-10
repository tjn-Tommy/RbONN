"""Why block 4+5 misses by 7 %FS when every other block sits near its noise floor.

Four things this establishes, in order of how much model they assume:

1. It is not the railed point.  4+5's v=1 read 0.10546 V with std 4e-6 against a
   +/-0.1 V DAQ range -- a flat-topped trace, i.e. the ADC rail.  Dropping it
   makes the block WORSE (7.21 vs 6.95 %FS), so the rail is a separate defect.

2. It is not a drift.  Blocks are time-ordered in the CSV; the excess over the
   predicted TPA runs -0.5, -2.4, -3.1, -2.7, +1.8, +28.5 % -- a step at the last
   block, not a ramp.  One global gain(t) makes all five other blocks worse.

3. It is not eta.  ``Y_jk - Y_j - Y_k = 2 R_j R_k cos(dPhi)`` with R_j, R_k taken
   from the one-pair blocks at the SAME drive, so step 6's amplitudes and
   backgrounds cancel.  Model-free, 2+4 and 2+5 land on their step-7 phases and
   4+5 does not.

4. The phase triangle does not close.  Phi is meant to be a per-pair constant, so
   dPhi_45 is fixed once dPhi_24 and dPhi_25 are known.  It is not: the two
   reference blocks force |dPhi_45| = 64 +/- 3 deg and the block itself reads
   ~39 deg.  No (Phi_4, Phi_5) -- sign flips included -- satisfies all three.

What survives is a degeneracy this run cannot break: either dPhi_45 really is
~40 deg (and Phi is not a per-pair constant), or the amplitudes R_4, R_5 are
~14 % larger when both pairs are driven together than when each is driven alone,
which inflates the extracted cos by exactly the observed amount.  The second is
suspicious because 4+5 is the ONLY block whose driven channels are SLM
neighbours: the column order is x5 x4 x3 x2 x1 w1 w2 w3 w4 w5, so pairs 4 and 5
touch on both the x and the w side, while 2+4 and 2+5 are 2 and 3 columns apart.
Steps 6 and 7 only ever drove one pair, or pair 2 plus one target -- never two
neighbours -- so every calibrated parameter was measured with neighbours off.

The experiment that separates them is in the printed report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from calibration_module.steps import calib_step8_v2 as s8      # noqa: E402
from calibration_module.sigma import floor_std                 # noqa: E402
from calibration_module.phase import phi_half                  # noqa: E402
from calibration_module.fit.slm_response import calibration_result_from_dict  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration  # noqa: E402
import refit_as_driven as rad                                  # noqa: E402

STEP7 = max(HERE.glob("calib_step7_result_*.json"), key=lambda p: p.stat().st_mtime)
OUT_PNG = HERE / "why_block_4plus5.png"
RAIL_V = 0.1          # DAQ input range, daq_module/controller.py

s8.IN_STEP7 = STEP7
table = rad._step8_driven_lookup()
pairs, raw = s8.load_csv(rad.OUT8)
_layout, models, phases = s8.load_inputs(None, pairs)

B = {}
for order, (driven, v, dark, mean, std) in enumerate(raw):
    dk = np.broadcast_to(np.asarray(dark, float), np.shape(mean))
    bg = np.zeros(len(v))
    th, amp = {}, {}
    for k in driven:
        t, a = [], []
        for i, vv in enumerate(v):
            x, w = table[(tuple(driven), round(float(vv), 6))][k]
            bg[i] += float(models[k].single_beam(x, w))
            t.append(float(phi_half(x)) + float(phi_half(w)))
            a.append(float(models[k].eta * np.sqrt(max(x * w, 0.0))))
        th[k], amp[k] = np.array(t), np.array(a)
    B[s8._pairs_tag(driven)] = dict(order=order, driven=driven, v=v, y=mean - dk,
                                    raw=mean, std=std, sd=floor_std(std),
                                    bg=bg, th=th, amp=amp)

TWO = ("2+4", "2+5", "4+5")
LIVE = {t: (slice(0, 5) if t == "4+5" else slice(None)) for t in TWO}
COL = {"2+4": "#0077b6", "2+5": "#2a9d8f", "4+5": "#c1121f"}


def model_y(tag, dphi_extra=0.0, gain=1.0):
    """The step-8 forward model for one block, with optional extra phase / gain."""
    b = B[tag]
    f = 0j
    for n, k in enumerate(b["driven"]):
        ph = b["th"][k] + phases[k] + (dphi_extra if n == 1 else 0.0)
        f = f + gain * b["amp"][k] * np.exp(1j * ph)
    return np.abs(f) ** 2 + b["bg"]


def measured_cos(tag):
    """(cos, sigma, model_cos, model_dphi_deg), step-6 amplitudes cancelled.

    Y_jk - Y_j - Y_k = 2 R_j R_k cos(dPhi) exactly, and R_j, R_k come from the
    one-pair blocks at the SAME drive -- so eta never enters and the single-beam
    backgrounds subtract off.
    """
    b = B[tag]
    j, k = b["driven"]
    Bj, Bk = B[str(j)], B[str(k)]
    fringe = b["y"] - Bj["y"] - Bk["y"]
    cap = 2 * np.sqrt(np.clip(Bj["y"] - Bj["bg"], 1e-12, None)
                      * np.clip(Bk["y"] - Bk["bg"], 1e-12, None))
    sig = np.sqrt(b["sd"] ** 2 + Bj["sd"] ** 2 + Bk["sd"] ** 2) / cap
    dphi = (b["th"][j] + phases[j]) - (b["th"][k] + phases[k])
    return fringe / cap, sig, np.cos(dphi), np.degrees(dphi)


# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(19.5, 10.6))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.24)
fig.suptitle("Block 4+5: the phase triangle does not close, and 4+5 is the only "
             "block whose driven channels are SLM neighbours",
             fontsize=15, fontweight="bold")

# 1 -- model-free cos
ax = fig.add_subplot(gs[0, 0])
for t in TWO:
    c, s, cm, _ = measured_cos(t)
    k = LIVE[t]
    ax.errorbar(B[t]["v"][k], c[k], yerr=s[k], fmt="o", ms=6, capsize=3,
                color=COL[t], label=t + "  measured", zorder=4)
    ax.plot(B[t]["v"], cm, "--", lw=1.8, color=COL[t], alpha=.75,
            label=t + "  from step-7 $\\Phi$")
ax.axhspan(1, 1.6, color="0.85", zorder=0)
ax.axhline(1, color="k", lw=.8)
ax.annotate("unphysical  $|\\cos| > 1$", (0.06, 1.06), fontsize=8, color="0.35")
ax.set_title("model-free fringe:  "
             "$\\cos\\Delta\\Phi = (Y_{jk} - Y_j - Y_k)\\,/\\,2R_jR_k$\n"
             "$R_j, R_k$ from the one-pair blocks -- $\\eta$ and background cancel",
             fontsize=10)
ax.set_xlabel("commanded $v$")
ax.set_ylabel("$\\cos\\Delta\\Phi$")
ax.set_ylim(-1.15, 1.5)
ax.grid(alpha=.25)
ax.legend(fontsize=7.5, ncol=2, loc="lower left")

# 2 -- closure
ax = fig.add_subplot(gs[0, 1])
meas = {}
for t in TWO:
    c, s, _, _ = measured_cos(t)
    k = LIVE[t]
    w = 1 / s[k] ** 2
    cb = float(np.sum(w * c[k]) / np.sum(w))
    ce = float(1 / np.sqrt(np.sum(w)))
    cc = float(np.clip(cb, -0.999, 0.999))
    meas[t] = (np.degrees(np.arccos(cc)), np.degrees(ce / np.sqrt(1 - cc ** 2)))
pred45 = abs(meas["2+4"][0] - meas["2+5"][0])
pred45e = float(np.hypot(meas["2+4"][1], meas["2+5"][1]))
sig = abs(pred45 - meas["4+5"][0]) / np.hypot(pred45e, meas["4+5"][1])
xs = [0, 1, 2, 3]
lab = ["$|\\Delta\\Phi_{24}|$", "$|\\Delta\\Phi_{25}|$",
       "$|\\Delta\\Phi_{45}|$\nrequired by\nthe other two",
       "$|\\Delta\\Phi_{45}|$\nmeasured"]
vals = [meas["2+4"][0], meas["2+5"][0], pred45, meas["4+5"][0]]
errs = [meas["2+4"][1], meas["2+5"][1], pred45e, meas["4+5"][1]]
ax.bar(xs, vals, 0.6, yerr=errs, capsize=5,
       color=[COL["2+4"], COL["2+5"], "0.45", COL["4+5"]])
for x, vl, e in zip(xs, vals, errs):
    ax.annotate("%.1f$\\pm$%.1f" % (vl, e), (x, vl + e + 4), ha="center",
                fontsize=9, fontweight="bold")
ax.plot([2, 3], [max(pred45, meas["4+5"][0]) + 26] * 2, "k-", lw=1.2)
ax.annotate("misclosure  %.1f$^\\circ$  =  %.1f$\\sigma$"
            % (abs(pred45 - meas["4+5"][0]), sig),
            (2.5, max(pred45, meas["4+5"][0]) + 31), ha="center", fontsize=10,
            fontweight="bold", color=COL["4+5"])
ax.set_xticks(xs)
ax.set_xticklabels(lab, fontsize=8.5)
ax.set_ylabel("$|\\Delta\\Phi|$   (deg)")
ax.set_title("closure test.  $\\Phi$ is a per-pair constant, so $\\Delta\\Phi_{45}$\n"
             "is fixed once $\\Delta\\Phi_{24}$ and $\\Delta\\Phi_{25}$ are known",
             fontsize=10)
ax.set_ylim(0, 200)
ax.grid(alpha=.25, axis="y")

# 3 -- drift ruled out
ax = fig.add_subplot(gs[0, 2])
tags = sorted(B, key=lambda t: B[t]["order"])
exc = []
for t in tags:
    b = B[t]
    k = LIVE.get(t, slice(None))
    p = (model_y(t) if len(b["driven"]) == 2
         else b["amp"][b["driven"][0]] ** 2 + b["bg"])
    exc.append(100 * float(np.sum((b["y"] - p)[k]) / np.sum((p - b["bg"])[k])))
ax.bar(range(len(tags)), exc, color=[COL.get(t, "0.6") for t in tags])
for i, e in enumerate(exc):
    ax.annotate("%+.1f" % e, (i, e + (1.4 if e > 0 else -3.0)), ha="center",
                fontsize=9, fontweight="bold")
ax.axhline(0, color="k", lw=.8)
ax.axhspan(-4, 3, color="#2a9d8f", alpha=.12)
ax.annotate("every other block within $-3 .. +2$ %", (0.04, 0.06),
            xycoords="axes fraction", fontsize=9, color="#1b5e50",
            fontweight="bold")
ax.set_xticks(range(len(tags)))
ax.set_xticklabels(["%d\n%s" % (B[t]["order"], t) for t in tags], fontsize=9)
ax.set_xlabel("block, in the order it was measured")
ax.set_ylabel("(meas $-$ pred) / predicted TPA   (%)")
ax.set_title("not a drift: a step at the last block, not a ramp\n"
             "(one global gain$(t)$ makes the other five worse)", fontsize=10)
ax.grid(alpha=.25, axis="y")

# 4 -- the rail
ax = fig.add_subplot(gs[1, 0])
for t in tags:
    ax.plot(B[t]["raw"], B[t]["std"], "o", ms=6, color=COL.get(t, "0.6"),
            label=t, zorder=3)
b = B["4+5"]
ax.annotate("4+5 at $v$=1\n0.1055 V, std 4e-6\nflat trace = ADC rail",
            xy=(b["raw"][-1], b["std"][-1]), xytext=(0.045, 3.5e-4),
            fontsize=9, fontweight="bold", color=COL["4+5"],
            arrowprops=dict(arrowstyle="->", color=COL["4+5"], lw=1.6))
ax.axvline(RAIL_V, color="k", ls="--", lw=1.4)
ax.annotate("DAQ range $\\pm$0.1 V", (RAIL_V, 2.4e-3), rotation=90, ha="right",
            va="top", fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("measured mean (V)")
ax.set_ylabel("trace std (V)")
ax.set_title("a separate defect: one point is railed\n"
             "(dropping it makes 4+5 worse, 6.95 $\\to$ 7.21 %FS)", fontsize=10)
ax.grid(alpha=.25, which="both")
ax.legend(fontsize=8, ncol=2, loc="lower right")

# 5 -- adjacency map
ax = fig.add_subplot(gs[1, 1])
lay = channel_layout_from_calibration(
    calibration_result_from_dict(
        json.loads(STEP7.read_text(encoding="utf-8"))["step3"]),
    method="interp", warn=False)
chans = []
for s in range(lay.n_channels):
    for side, ch in (("x", lay.x_channels[s]), ("w", lay.w_channels[s])):
        chans.append((float(ch.x_center), s + 1, side))
chans.sort()
for row, t in enumerate(tags):
    dr = B[t]["driven"]
    for i, (_col, p, _side) in enumerate(chans):
        ax.add_patch(plt.Rectangle(
            (i - .42, -row - .38), .84, .76,
            fc=(COL.get(t, "0.6") if p in dr else "0.92"), ec="0.5", lw=.6))
    for i in range(len(chans) - 1):
        if chans[i][1] in dr and chans[i + 1][1] in dr:
            ax.plot([i, i + 1], [-row] * 2, "k-", lw=3, zorder=5)
            ax.plot(i + .5, -row, "k*", ms=14, zorder=6)
    ax.annotate(t, (-1.3, -row), va="center", ha="right", fontsize=10,
                fontweight="bold")
ax.set_xticks(range(len(chans)))
ax.set_xticklabels(["%s%d" % (s, p) for _c, p, s in chans], fontsize=9)
ax.set_xlim(-2.5, len(chans) - .3)
ax.set_ylim(-len(tags) + .4, .8)
ax.set_yticks([])
ax.set_xlabel("SLM column order   (52 px / 0.29 nm pitch)")
ax.set_title("who touches whom.  $\\bigstar$ = two driven channels in\n"
             "adjacent columns -- only ever in block 4+5", fontsize=10)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)

# 6 -- the two survivors
ax = fig.add_subplot(gs[1, 2])
b = B["4+5"]
k = LIVE["4+5"]
gr = np.linspace(-np.pi, np.pi, 40001)
dbest = gr[int(np.argmin([float(np.sum((((b["y"] - model_y("4+5", dphi_extra=d))
                                         / b["sd"])[k]) ** 2)) for d in gr]))]
gg = np.linspace(0.9, 1.4, 20001)
gbest = gg[int(np.argmin([float(np.sum((((b["y"] - model_y("4+5", gain=g))
                                         / b["sd"])[k]) ** 2)) for g in gg]))]


def nr(p):
    return 100 * float(np.sqrt(np.mean(((b["y"] - p)[k]) ** 2))) / s8.full_scale(
        models, b["driven"], p)


ax.errorbar(b["v"][k], 1e3 * b["y"][k], yerr=1e3 * b["sd"][k], fmt="ko", ms=7,
            capsize=3, label="measured", zorder=5)
ax.plot(b["v"][-1], 1e3 * b["y"][-1], "x", ms=12, mew=2.5, color="0.5",
        label="railed, excluded", zorder=5)
p0, p1, p2 = (model_y("4+5"), model_y("4+5", dphi_extra=dbest),
              model_y("4+5", gain=gbest))
ax.plot(b["v"], 1e3 * p0, "-", lw=2.2, color=COL["4+5"],
        label="as calibrated  (%.2f %%FS)" % nr(p0))
ax.plot(b["v"], 1e3 * p1, "--", lw=2, color="#7b2cbf",
        label=("free phase %+.1f$^\\circ$  (%.2f %%FS)\n   -- breaks closure"
               % (np.degrees(dbest), nr(p1))))
ax.plot(b["v"], 1e3 * p2, ":", lw=2.6, color="#f77f00",
        label=("free gain $\\times$%.3f on $R$  (%.2f %%FS)\n"
               "   -- breaks blocks 4 and 5" % (gbest, nr(p2))))
ax.set_xlabel("commanded $v$")
ax.set_ylabel("dark-free signal (mV)")
ax.set_title("two one-parameter fixes, each excluded by other data\n"
             "-- this run cannot tell them apart", fontsize=10)
ax.grid(alpha=.25)
ax.legend(fontsize=8, loc="upper left")

fig.savefig(OUT_PNG, dpi=125, bbox_inches="tight")
print("wrote", OUT_PNG)

# ------------------------------------------------------------------ report
print()
print("=" * 88)
print("BLOCK 4+5 -- WHAT IS AND IS NOT ESTABLISHED")
print("=" * 88)
print("model-free |dPhi| per block (step-6 amplitudes cancelled):")
for t in TWO:
    _c, _s, _cm, dm = measured_cos(t)
    print("   %-5s  measured %6.2f +/- %4.2f deg      step-7 model %6.2f deg"
          % (t, meas[t][0], meas[t][1],
             abs(np.degrees(np.angle(np.exp(1j * np.radians(dm[2])))))))
print("   required of 4+5 by the other two: %.2f +/- %.2f deg"
      % (pred45, pred45e))
print("   -> misclosure %.1f deg = %.1f sigma"
      % (abs(pred45 - meas["4+5"][0]), sig))
print()
print("ruled out:")
print("   the ADC rail          -- dropping that point makes 4+5 worse")
print("   a run-long drift      -- step at block 5, not a ramp")
print("   eta / step-6 amplitude-- cancels in the model-free fringe")
print("   detector saturation   -- the anomaly is already +13 % at v=0.1,")
print("                            where 4+5 reads 1.4 mV")
print()
print("still open, and degenerate in this run:")
print("   (a) dPhi_45 really is ~40 deg  -> Phi is not a per-pair constant")
print("   (b) R_4, R_5 are ~14 % larger when both pairs are driven together,")
print("       which inflates the extracted cos by exactly the amount seen")
print()
print("THE MEASUREMENT THAT SEPARATES THEM (no new code paths):")
print("   Drive pair 4's x and w at v, set pair 5's x channel ON and its w OFF.")
print("   Pair 5 then makes NO TPA -- it needs both beams -- so any change from")
print("   'pair 4 alone' is pure neighbour leakage, i.e. hypothesis (b) measured")
print("   directly.  If that block reproduces block 4, (b) is dead and the")
print("   closure failure is real physics.")
print()
print("   Fuller version: a step-7 phase ramp of pair 4 against pair 5 instead of")
print("   against the reference.  A fringe scan separates the pedestal from the AC")
print("   amplitude, so a gain cancels in the ratio and dPhi_45 comes from the")
print("   fringe position -- the number step 7 never measured.")
print()
print("FIX REGARDLESS:")
print("   - 4+5 at v=1 railed the DAQ (0.1055 V into +/-0.1 V).  Widen the range")
print("     or attenuate before re-running; that point carries no information.")
print("   - three channels warn v=1 is an encoding extrapolation (778.183 /")
print("     777.304 / 777.016 nm); widen level_range in step 3b.")
