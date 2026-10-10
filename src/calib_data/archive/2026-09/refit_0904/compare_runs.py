"""0904 as-run vs 0906 as-driven: what correcting the x/w labels did to step 8.

Both sides use step 8's own forward model and its own full-scale ruler.  The
only difference is which (x, w) each point is credited with, and which step-6 /
step-7 parameters were fitted from those labels:

  0904 as-run    -- x = w = v for every driven pair, the run's own assumption
  0906 as-driven -- the per-pair values the fitted transfer model says the
                    levels actually delivered

FS is 4*R_j*R_k at v = 1 in both cases, so the percentages are on the same
ruler and the two runs are directly comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.steps import calib_step8_v2 as s8   # noqa: E402
from calibration_module.sigma import floor_std              # noqa: E402
import refit_as_driven as rad                               # noqa: E402

OLD_DIR = REPO_ROOT / "src/calib_data/archive/2026-09/run_0904_0117"
OLD_CSV = OLD_DIR / "calib_step8_simple_0904_0150.csv"
OLD_JSON = OLD_DIR / "calib_step7_result_0904_0147.json"
NEW_CSV = rad.OUT8
NEW_JSON = max(HERE.glob("calib_step7_result_*.json"), key=lambda p: p.stat().st_mtime)
OUT_PNG = HERE / "step8_compare_0904_vs_asdriven.png"


def evaluate(csv_path, step7_json, predict_curve):
    """-> {block tag: dict of arrays}, using step 8's own model and ruler."""
    saved = (s8.IN_STEP7, s8.predict_curve)
    try:
        s8.IN_STEP7 = step7_json
        s8.predict_curve = predict_curve
        pairs, blocks = s8.load_csv(csv_path)
        _layout, models, phases = s8.load_inputs(None, pairs)
        out = {}
        for driven, v, dark, mean, std in blocks:
            y = mean - dark
            std = floor_std(std)
            pred = s8.predict_curve(models, phases, driven, v)
            resid = y - pred
            fs = s8.full_scale(models, driven, pred)
            out[s8._pairs_tag(driven)] = {
                "driven": driven, "v": v, "y": y, "pred": pred, "resid": resid,
                "pull": resid / std, "fs": fs,
                "nrmse": 100.0 * float(np.sqrt(np.mean(resid ** 2))) / fs,
                "floor": 100.0 * float(np.sqrt(np.mean(std ** 2))) / fs,
                "pct": 100.0 * resid / fs,
            }
        return out, models, phases
    finally:
        s8.IN_STEP7, s8.predict_curve = saved


print("--- 0904 as-run ---")
old, m_old, p_old = evaluate(OLD_CSV, OLD_JSON, s8.predict_curve)
print("\n--- 0906 as-driven ---")
new, m_new, p_new = evaluate(
    NEW_CSV, NEW_JSON, rad.make_predict_curve_driven(rad._step8_driven_lookup()))

tags = list(old)
C_OLD, C_NEW = "#0077b6", "#c1121f"

fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8))
fig.suptitle("Step 8 forward-model error: 0904 as-run vs 0906 re-fit as driven",
             fontsize=14, fontweight="bold")

# ---- 1. NRMSE per block -------------------------------------------------
ax = axes[0]
i = np.arange(len(tags))
ax.bar(i - 0.2, [old[t]["nrmse"] for t in tags], 0.4, color=C_OLD,
       label="0904 as-run")
ax.bar(i + 0.2, [new[t]["nrmse"] for t in tags], 0.4, color=C_NEW,
       label="0906 as-driven")
for k, t in enumerate(tags):
    ax.plot([k - 0.4, k + 0.4], [old[t]["floor"]] * 2, "-", color="0.35", lw=2,
            zorder=5)
    ax.annotate(f"{old[t]['nrmse']:.2f}", (k - 0.2, old[t]["nrmse"]),
                ha="center", va="bottom", fontsize=8, color=C_OLD)
    ax.annotate(f"{new[t]['nrmse']:.2f}", (k + 0.2, new[t]["nrmse"]),
                ha="center", va="bottom", fontsize=8, color=C_NEW,
                fontweight="bold")
ax.set_xticks(i)
ax.set_xticklabels(tags)
ax.set_xlabel("driven block")
ax.set_ylabel("NRMSE  (% of full scale)")
ax.set_title("rms(meas $-$ pred) per block\n(grey bar = that block's noise floor)",
             fontsize=10)
ax.grid(alpha=.25, axis="y")
ax.legend(fontsize=9)

# ---- 2. residual vs drive ----------------------------------------------
ax = axes[1]
cmap = plt.get_cmap("tab10")
for k, t in enumerate(tags):
    c = cmap(k)
    ax.plot(old[t]["v"], old[t]["pct"], "o--", ms=4, lw=1.2, color=c, alpha=.55)
    ax.plot(new[t]["v"], new[t]["pct"], "o-", ms=5, lw=2.0, color=c, label=t)
ax.axhline(0, color="k", lw=.8)
ax.set_xlabel("commanded $v$")
ax.set_ylabel("(meas $-$ pred) / FS   (%)")
ax.set_title("per-point residual\ndashed = 0904 as-run,  solid = as-driven",
             fontsize=10)
ax.grid(alpha=.25)
ax.legend(fontsize=8, ncol=2)

# ---- 3. what moved upstream --------------------------------------------
ax = axes[2]
ax.axis("off")
lines = [
    "STEP 6   eta  (difference-fit slope)",
    "  pair    0904       as-driven     change",
]
for k in rad.PAIRS:
    lines.append(f"    {k}     {m_old[k].eta:.5f}     {m_new[k].eta:.5f}      "
                 f"{100 * (m_new[k].eta - m_old[k].eta) / m_old[k].eta:+6.2f} %")
lines += ["", "STEP 7   Phi  (comb phase vs pair 2)",
          "  pair    0904       as-driven     change"]
for k in rad.PAIRS:
    a, b = np.degrees(p_old[k]), np.degrees(p_new[k])
    lines.append(f"    {k}   {a:+8.2f}    {b:+8.2f}     {b - a:+6.2f} deg")
lines += ["", "STEP 8   mean NRMSE  (% FS)",
          "                0904    as-driven"]
for n, kind in ((1, "one pair "), (2, "two pairs")):
    o = np.mean([old[t]["nrmse"] for t in tags if len(old[t]["driven"]) == n])
    v = np.mean([new[t]["nrmse"] for t in tags if len(new[t]["driven"]) == n])
    lines.append(f"  {kind}    {o:6.3f}     {v:6.3f}    ({100 * (v - o) / o:+.0f} %)")
ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
        fontsize=10.5, transform=ax.transAxes)

fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT_PNG, dpi=130)
print(f"\nwrote {OUT_PNG}")

print("\n block    NRMSE 0904   as-driven    change     floor")
for t in tags:
    o, n = old[t]["nrmse"], new[t]["nrmse"]
    print(f"  {t:<6s}   {o:8.3f}   {n:9.3f}   {100 * (n - o) / o:+7.1f} %   "
          f"{old[t]['floor']:6.3f}")
