"""Draft: step 6 v2's PRODUCT-only check, on its own short grid.

    python src/drafts/calib_step6_verify.py --step3 S.json             # measure, then check
    python src/drafts/calib_step6_verify.py --step3 S.json --meas      # raw CSV only, no check
    python src/drafts/calib_step6_verify.py --step3 S.json some.csv    # re-check a CSV offline

Why this exists
---------------
:mod:`calib_step6_v2` prints its verification with every fit, but getting it
costs the whole 37-acquisition grid (~5.6 min/pair) because the checks ride
along with the eta fit.  When the thing under investigation IS a check -- the
product split -- that is the wrong ratio: 31 of those acquisitions serve only an
eta the check does not need.

This draft keeps the check and drops the estimator.  What it measures:

* the **background block** -- dark, x-only, w-only: the five levels
  ``fit_background`` needs, at step 6's own repeat counts.  These are neither
  optional nor reusable from an earlier run.  The check subtracts a *fitted*
  single-beam background from each point, so a background carried over from
  another day would put that day's drift straight into the residue difference,
  which is precisely the quantity being measured.
* the **check levels**, derived from :data:`PRODUCT_CHECKS` -- add a group and
  its levels are measured automatically; there is no second list to keep in step
  with it.

What it does NOT do
-------------------
Step 6's other verification -- the intercept identity ``beta0 = a_x (+ q_x)`` --
needs ``beta0``, which exists only once the cross line has been fitted.  It is
inseparable from the estimator, as is the top-drive compression diagnostic, so
neither is here.  Run :mod:`calib_step6_v2` for those.

The report -- printed and drawn
-------------------------------
One PNG per pair, and it shows one quantity: the **TPA residue**
``Y(x, w) - Bhat(x, w)``, i.e. what is left of each check level once the fitted
single-beam background is subtracted.  Every member of a group shares one
``x*w``, so the model says those residues are the same number measured several
ways; the left panel plots them against the drive split and the right panel
grades every pairwise difference in sigma.  Nothing else is drawn -- the
background block, the level table and the repeat scatter are printed, because
they are inputs to the subtraction rather than the thing under test.

How it borrows
--------------
The sigma model, ``fit_background`` and ``verify_product`` come from
``calibration_module.fit.pair_v2`` and the measurement path from
``calibration_module.measure.pair_v2``; the step script itself is still imported
by path (the way ``src/drafts/calib_step6-8_v2.py`` imports the steps) for its
layout loader and its ``CONFIG``/``ACQ``.  Nothing is reimplemented, so a split
this script reports is a split step 6 would report on the same rows, which is
the only thing that makes a shortened grid worth trusting.

``fit_q`` is inherited rather than redeclared, for the same reason: the
background subtracted here has to be the background subtracted there.
:func:`_config` gets that by starting from step 6's own ``CONFIG`` and
``dataclasses.replace``-ing only the few fields this draft owns -- so what it
does not name cannot disagree.  It used to be done by assigning into step 6's
module globals, which worked but left this file able to change step 6's
behaviour for anything else importing it in the same process.

The CSV keeps step 6's column layout, so ``load_meas_csv`` reads it back.  It
will NOT re-fit under ``calib_step6_v2 <csv>``: there is no cross line in it and
the slope fit correctly refuses it.
"""
from __future__ import annotations

from dataclasses import replace
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

STEP6_PATH = REPO_ROOT / "src" / "calibration_module" / "steps" / "calib_step6_v2.py"

Group = tuple[tuple[float, float], ...]
Grid = tuple[tuple[float, float, int], ...]

# Decimals a drive level survives a round trip with.  ``write_meas_csv`` formats
# x and w as "%.6g" and ``_level_at`` matches to 1e-9, so a level carried at full
# float precision is written short, read back different, and then simply not
# found -- ``verify_product`` skips a point it cannot locate, so the group would
# quietly lose members on an offline re-check while the live run kept them.
# 6 dp is also what ``average_levels`` and :func:`_key` index on.
LEVEL_DP = 6


from calibration_module.fit.pair_v2 import (  # noqa: E402
    PairV2Fit,
    _background_dict,  # noqa: PLC2701 -- same family, see "How it borrows"
    average_levels,
    fit_background,
    load_meas_csv,
    verify_product,
    write_meas_csv,
)
from calibration_module.fit.sigma import STD_FLOOR_V  # noqa: E402
from calibration_module.measure.pair_v2 import (  # noqa: E402
    build_schedule,
    measure_pair,
)


def _load_step6():
    """Import ``calib_step6_v2`` by path -- the step scripts are not a package."""
    name = "calib_step6_v2"
    if name in sys.modules:
        return sys.modules[name]
    if not STEP6_PATH.is_file():
        raise FileNotFoundError(f"step 6 v2 not found: {STEP6_PATH}")
    spec = importlib.util.spec_from_file_location(name, STEP6_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # register before exec, as import does
    spec.loader.exec_module(mod)
    return mod


s6 = _load_step6()


def group_at(product: float, *xs: float) -> Group:
    """A check group at one ``product``, with ``w = product / x`` computed exactly.

    Write the drive splits you want to compare and let the arithmetic follow.
    Hand-rounding ``w`` is the one way to break a group silently: a member whose
    product is off by a fraction f carries a residue off by f, and that lands in
    the report as an f-sized split with nothing to distinguish it from a real
    one.  At the percent level -- which ``0.25/0.6 = 0.4167`` written as
    ``0.425`` reaches -- it is the same size as the effect being hunted.

    ``w`` is quantised to :data:`LEVEL_DP` because that is the precision the CSV
    and the level index carry; the rounding costs ~1e-6 of the product, six
    orders below :data:`PRODUCT_TOL_FRAC`.
    """
    if not (product > 0.0):
        raise ValueError(f"product must be positive, got {product}")
    group = []
    for x in xs:
        x = round(float(x), LEVEL_DP)
        w = round(product / x, LEVEL_DP) if x else float("inf")
        if not (0.0 < x <= 1.0 and 0.0 < w <= 1.0):
            raise ValueError(
                f"x = {x:g} at product {product:g} needs w = {w:.4f}, which is "
                "not a drive level in (0, 1]; x must lie in [product, 1]"
            )
        group.append((x, w))
    return tuple(group)


# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src" / "calib_data"
# The Step-3 calibration is the required --step3 argument, not a constant.  It
# used to be one, and this draft's copy was independent of step 6's, so the two
# drifted apart quietly -- which is exactly the failure this file exists to
# catch: a check run against the wrong step-3 file looks like a check that
# passed.  Naming it per run is the only version that cannot go wrong silently.

PAIR_INDEX_BASE = 1
PAIR_INDICES = [1, 2, 3, 4, 5, 6]

# Optional: a previous step-6 v2 result, read ONLY to print the eta it fitted
# beside each measured residue (the model's expected eta^2*(x*w)).  It is never
# subtracted and never enters the split, so a stale file cannot bias the check
# -- leave it None and that reference is simply not drawn.
REF_STEP6: Path | None = None

SLM_DISPLAY_NO = None
USB_SLM_NO = 1

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

T_SINGLE_S = 10.0
T_BOTH_S = 8.0
SETTLE_S = 0.25

DAQ_RANGE_V = 0.1

# ---- The grid ----
# The background block, verbatim from step 6's GRID including its repeat counts,
# so the background fitted here is weighted exactly as step 6 weights it and the
# two runs' parameters are directly comparable.  (1, 0) keeps n = 3 even though
# nothing here uses it as the D(0) anchor -- matching step 6 is the point.
BG_GRID: Grid = (
    (0.0, 0.00, 2),   # dark    -> d
    (0.5, 0.00, 2),   # x-only  -> a_x
    (1.0, 0.00, 3),   # x-only  -> a_x
    (0.0, 0.50, 2),   # w-only  -> a_w
    (0.0, 1.00, 2),   # w-only  -> a_w
)

# Repeats per check level.  3 is what step 6's VERIFY_GRID uses; dropping the
# estimator frees ~20 acquisitions per pair, so raising this is the cheapest
# available way to tighten a marginal pull.
CHECK_REPEATS = 3

# Groups of levels sharing one product x*w.  The model says Y depends on the two
# drives ONLY through that product, so once the single-beam background is
# subtracted the members of a group must agree.  They sit at different x, which
# is the entire point: (1, 0.25) drives the x side hard and the w side softly,
# (0.5, 0.5) splits the drive evenly.
#
# Use :func:`group_at` rather than writing (x, w) pairs by hand -- see its
# docstring.  Each extra MEMBER costs CHECK_REPEATS * T_BOTH_S ~ 25 s/pair, and
# the comparisons grow as n(n-1)/2, so a group of 8 grades 28 splits.
PRODUCT_CHECKS: tuple[Group, ...] = (
    group_at(0.25, 0.277, 0.357, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# A group's members must share one product.  Enforced as a fraction of that
# product because the injected split scales that way: at 1e-3 a bad member fakes
# a 0.1% split, an order below anything worth reporting.  Reach for
# :func:`group_at` before reaching for this number.
PRODUCT_TOL_FRAC = 1e-3

PULL_LIMIT = 3.0                 # |pull| at or above this prints ** CHECK **


# ======================================================================
# grid assembly
# ======================================================================

def _key(x: float, w: float) -> tuple[float, float]:
    """The (x, w) identity the whole pipeline agrees on (step 6 rounds to 6 dp)."""
    return round(float(x), 6), round(float(w), 6)


def _dedupe(grid) -> Grid:
    """Collapse repeated levels, keeping the largest repeat count asked for."""
    seen: dict[tuple[float, float], int] = {}
    for x, w, n in grid:
        key = _key(x, w)
        seen[key] = max(seen.get(key, 0), int(n))
    return tuple((x, w, n) for (x, w), n in seen.items())


def check_grid() -> Grid:
    """The levels :data:`PRODUCT_CHECKS` asks for, deduped, at CHECK_REPEATS."""
    return _dedupe((x, w, CHECK_REPEATS)
                   for group in PRODUCT_CHECKS for x, w in group)


def measure_grid() -> Grid:
    """Background block + check levels; a level named twice keeps the larger n."""
    return _dedupe(tuple(BG_GRID) + check_grid())


def group_product(group: Group) -> tuple[float, float]:
    """``(mean product, worst relative deviation from it)`` for one group."""
    products = np.array([float(x) * float(w) for x, w in group])
    mean = float(products.mean())
    if not mean:
        return mean, float("inf")
    return mean, float(np.max(np.abs(products / mean - 1.0)))


def validate_checks() -> None:
    """Fail before touching hardware on a group that cannot mean what it says."""
    if not PRODUCT_CHECKS:
        raise ValueError("PRODUCT_CHECKS is empty -- there is nothing to verify")
    for group in PRODUCT_CHECKS:
        if len(group) < 2:
            raise ValueError(f"check group {group} has fewer than 2 levels")
        for x, w in group:
            if not (0.0 < x <= 1.0 and 0.0 < w <= 1.0):
                raise ValueError(
                    f"check level ({x}, {w}) is not a cross level with both drives "
                    "in (0, 1]; a level with a beam off belongs to the background "
                    "block and would be fitted rather than checked"
                )
            if (x, w) != _key(x, w):
                raise ValueError(
                    f"check level ({x!r}, {w!r}) carries more than {LEVEL_DP} "
                    "decimals, which the measurement CSV does not; re-reading that "
                    "file would fail to match the level and drop it from the group "
                    "without saying so.  Round it, or build the group with group_at()"
                )
        product, dev = group_product(group)
        if dev > PRODUCT_TOL_FRAC:
            worst = max(group, key=lambda pt: abs(pt[0] * pt[1] / product - 1.0))
            raise ValueError(
                f"check group does not share one product: mean {product:.6g}, "
                f"worst member {worst} at {worst[0] * worst[1]:.6g} "
                f"({(worst[0] * worst[1] / product - 1.0) * 100:+.3f}%, tolerance "
                f"{PRODUCT_TOL_FRAC * 100:g}%).  That member's residue is off by the "
                f"same fraction and reports as a split of that size.  Build the "
                f"group with group_at({product:.4g}, ...) instead of rounding w."
            )


def _config():
    """The estimator config this check runs under: step 6's, with our overrides.

    ``replace`` rather than a fresh :class:`PairV2Config` so everything this
    draft does not own -- ``fit_q`` and the ``params_bg`` derived from it above
    all -- is *inherited* from step 6 and cannot silently disagree with it.
    That was the point of the old rebinding dance; a frozen config does it
    without mutating another module.

    ``verify_enabled`` / ``verify_grid`` matter only to the level table's tag --
    nothing here calls ``on_cross_line`` -- but a check level should still print
    as "verify+" and not as "cross".
    """
    return replace(
        s6.CONFIG,
        pair_index_base=PAIR_INDEX_BASE,
        product_checks=PRODUCT_CHECKS,
        verify_enabled=True,
        verify_grid=check_grid(),
    )


def _acq():
    """Acquisition timing: step 6's, with this draft's windows and range."""
    return replace(
        s6.ACQ,
        t_meas_s=T_BOTH_S, settle_s=SETTLE_S,
        range_v=DAQ_RANGE_V,
    )


def _push_config() -> None:
    """Rebind the step-6 script's remaining module constants to this draft's.

    Only the *script-level* ones are left: paths, pair list and device numbers,
    which ``s6._load_layout`` reads at call time.  The estimator's parameters no
    longer live on the module at all -- they are in :func:`_config` -- so this
    can no longer reach into the fit by accident.

    The step-3 calibration is no longer among them: it is passed explicitly to
    every function that needs it, so this draft and step 6 cannot end up naming
    two different files.
    """
    for name in ("CALIB_PATH", "PAIR_INDICES",
                 "SLM_DISPLAY_NO", "USB_SLM_NO", "DAQ_DEVICE", "DAQ_CHANNEL"):
        setattr(s6, name, globals()[name])
    s6.CONFIG = _config()
    s6.ACQ = _acq()


# ======================================================================
# the check
# ======================================================================

def _shell_fit(index: int, levels, bg, bg_cov, b: float):
    """A :class:`PairV2Fit` carrying only what the borrowed code reads.

    ``verify_product`` and ``bg_residuals`` touch ``levels``, ``bg``, ``bg_cov``
    and ``b``, nothing else.  The estimator's fields do not exist on this run, so
    they are left empty / NaN rather than invented -- anything that reaches for
    them gets a NaN and not a plausible-looking number.
    """
    nan = float("nan")
    eta = float(np.sqrt(b)) if np.isfinite(b) and b > 0 else nan
    return PairV2Fit(
        index=index, cfg=_config(), levels=levels, bg=bg, bg_cov=bg_cov,
        fit_w=np.zeros(0), fit_d=np.zeros(0),
        fit_sigma=np.zeros(0), fit_pred=np.zeros(0),
        b=b, b_err=nan, beta0=nan, beta0_err=nan,
        eta=eta, eta_err=nan, r2=nan,
        anchor=(nan, nan), excluded=[],
    )


def check_pair(index: int, levels, *, b: float = float("nan")):
    """Fit the background block, then run :func:`verify_product` against it."""
    bg, bg_cov = fit_background(levels, _config())
    fit = _shell_fit(index, levels, bg, bg_cov, b)
    fit.checks = {"product": verify_product(fit)}
    return fit


def product_records(fit) -> list[dict]:
    """The check records on a fit, or an empty list when it has none."""
    return (fit.checks or {}).get("product") or []


def _eta_eff(tpa: float, tpa_err: float, product: float) -> tuple[float, float]:
    """The eta one level implies on its own: ``residue = eta^2 (x w)``.

    Two group members disagreeing here IS the split, restated in the units eta is
    quoted in -- a 6% residue split is a 3% eta split, and it is eta that steps 7
    and 8 consume.
    """
    if not (product > 0.0) or not np.isfinite(tpa) or tpa <= 0.0:
        return float("nan"), float("nan")
    eta = float(np.sqrt(tpa / product))
    return eta, float(tpa_err / (2.0 * eta * product))


def group_mean(rec: dict) -> tuple[float, float]:
    """Inverse-variance weighted mean residue of a group, and its error.

    The reference the plot draws the split against.  Weighted by each level's OWN
    sigma, not by ``tpa_err``: the members share one fitted background, and that
    shared part moves them all together, so it cannot produce a split and does
    not belong in the spread they are judged against.  Deciding whether a split
    is real is the pairwise pulls' job -- those propagate the background
    covariance properly -- so this is a guide to the eye, not a test.
    """
    vals = np.array([pt["tpa"] for pt in rec["points"]], dtype=float)
    errs = np.array([pt["_sigma"] for pt in rec["points"]], dtype=float)
    ok = np.isfinite(vals) & np.isfinite(errs) & (errs > 0.0)
    if not ok.any():
        return float("nan"), float("nan")
    wts = 1.0 / errs[ok] ** 2
    return (float(np.sum(wts * vals[ok]) / np.sum(wts)),
            float(1.0 / np.sqrt(np.sum(wts))))


def pull_matrix(rec: dict) -> np.ndarray:
    """``n x n`` of pairwise pulls, NaN on and above the diagonal.

    Cell ``(i, j)`` is the pull of ``points[i] - points[j]``, so it reads
    "row minus column" -- the same orientation, and the same sign, as the
    printed comparison lines.
    """
    order = {_key(pt["x"], pt["w"]): i for i, pt in enumerate(rec["points"])}
    out = np.full((len(order), len(order)), np.nan)
    for pr in rec["pairs"]:
        i, j = order[_key(*pr["a"])], order[_key(*pr["b"])]
        out[max(i, j), min(i, j)] = pr["pull"] if i > j else -pr["pull"]
    return out


def _ref_etas() -> dict[int, float]:
    """``{pair: eta}`` from :data:`REF_STEP6`, or empty when it is not set."""
    if REF_STEP6 is None:
        return {}
    path = Path(REF_STEP6)
    if not path.is_file():
        print(f"(REF_STEP6 not found, expected value left blank: {path})")
        return {}
    try:
        from calibration_module.fit.phase import load_pair_models
        return {k: float(m.eta) for k, m in load_pair_models(path).items()}
    except Exception as exc:                 # noqa: BLE001 -- a cosmetic input only
        print(f"(REF_STEP6 unreadable, expected value left blank: {exc})")
        return {}


# ======================================================================
# reporting
# ======================================================================

def _report_levels(fit) -> None:
    """The measured level means and the two spreads sigma is built from."""
    print(f"  sigma = hypot(max(rep_std, trace_std)/sqrt(n), "
          f"{STD_FLOOR_V*1e3:.3f} mV systematic floor)")
    print("    block     x     w   n   mean(mV)  rep_std(mV)  trace_std(mV)  sigma(mV)")
    for L in fit.levels:
        tag = "verify+" if L.is_verification(fit.cfg) else L.block
        rep = f"{L.rep_std*1e3:11.4f}" if np.isfinite(L.rep_std) else f"{'--':>11}"
        print(f"    {tag:<7} {L.x:5.2f} {L.w:5.2f} {L.n:3d} {L.mean*1e3:10.4f} "
              f"{rep}  {L.trace_std*1e3:13.4f} {L.sigma*1e3:10.4f}")
    print("    (+ a check level: never fitted, only compared)")


def _report_background(fit) -> None:
    """The single-beam fit that gets subtracted, and whether it fits."""
    n_bg = sum(1 for L in fit.levels if L.block != "cross")
    params_bg = fit.cfg.params_bg
    dof = n_bg - len(params_bg)
    shape = "saturated" if dof == 0 else f"{dof} dof"
    print(f"\n  Background block ({n_bg} levels, {len(params_bg)} parameters "
          f"-> {shape}) -- this is what gets subtracted:")
    for name in params_bg:
        v, e = fit.bg[name]
        scale, unit = (1e3, "mV") if name == "d" else (1.0, "")
        print(f"    {name:<3} = {v*scale:.4e} +/- {e*scale:.3e} {unit}".rstrip())

    used_bg, resid, pull = fit.bg_residuals()
    if not resid.size:
        return
    print("      block     x     w   resid(mV)   pull")
    for L, r, pl in zip(used_bg, resid, pull):
        print(f"      {L.block:<7} {L.x:5.2f} {L.w:5.2f} {r*1e3:10.4f} {pl:6.2f}")
    mx = float(np.max(np.abs(pull)))
    verdict = ("OK" if mx < PULL_LIMIT
               else "** CHECK: the arm is curved, so the subtraction is suspect **")
    print(f"      residual RMS = {float(np.sqrt(np.mean(resid**2)))*1e3:.4f} mV   "
          f"max |pull| = {mx:.2f}   {verdict}")


def _report_product(fit) -> None:
    """The check itself: residues at one product, then every pairwise split."""
    prods = product_records(fit)
    print("\n  Product-only dependence   Y depends on (x, w) only via x*w")
    if not prods:
        print("    NOT MEASURED -- no check group had two levels in the data")
        return
    for rec in prods:
        expected = (f"{rec['expected']*1e3:.4f} mV" if np.isfinite(rec["expected"])
                    else "-- (set REF_STEP6 for the fitted eta)")
        print(f"    x*w = {rec['product']:.3f}   model expects eta^2*(x*w) = {expected}")
        for pt in rec["points"]:
            eta, eta_err = _eta_eff(pt["tpa"], pt["tpa_err"], rec["product"])
            col = (f"   eta_eff = {eta:.5f} +/- {eta_err:.5f}"
                   if np.isfinite(eta) else "   eta_eff = --")
            print(f"      ({pt['x']:.3f}, {pt['w']:.3f}) n={pt['n']}  "
                  f"TPA residue = {pt['tpa']*1e3:9.4f} +/- {pt['tpa_err']*1e3:.4f} mV{col}")
        mean, mean_err = group_mean(rec)
        if np.isfinite(mean):
            print(f"      group weighted mean = {mean*1e3:.4f} +/- {mean_err*1e3:.4f} mV"
                  "   (uncorrelated part only -- a reference, not a test)")
        for pr in rec["pairs"]:
            verdict = "OK" if abs(pr["pull"]) < PULL_LIMIT else "** CHECK **"
            print(f"      {pr['a']} - {pr['b']} = {pr['diff']*1e3:+.4f} +/- "
                  f"{pr['err']*1e3:.4f} mV  ({pr['frac']*100:+.2f}%)  "
                  f"pull = {pr['pull']:+.2f}   {verdict}")
    print("    (a split here does not make step 6's estimator wrong -- it means the")
    print("     pair has no single eta at all, which is what steps 7/8 assume)")


def report(fit) -> None:
    """Level table, the background block it subtracts, then the check itself."""
    _report_levels(fit)
    _report_background(fit)
    _report_product(fit)


# ======================================================================
# the plot  (the check, and only the check)
# ======================================================================

def _plot_residues(ax, rec: dict) -> None:
    """Residues against the drive split, with the group mean as the reference.

    The right-hand axis restates the same points as a percentage of that mean:
    a split is quoted as a fraction, and this is the left axis rescaled rather
    than a second quantity.
    """
    xs = np.array([pt["x"] for pt in rec["points"]], dtype=float)
    vals = np.array([pt["tpa"] for pt in rec["points"]], dtype=float) * 1e3
    errs = np.array([pt["tpa_err"] for pt in rec["points"]], dtype=float) * 1e3
    order = np.argsort(xs)
    mean, mean_err = group_mean(rec)

    if np.isfinite(mean):
        ax.axhspan((mean - mean_err) * 1e3, (mean + mean_err) * 1e3,
                   color="tab:purple", alpha=0.15, zorder=0)
        ax.axhline(mean * 1e3, color="tab:purple", lw=1.3, zorder=1,
                   label=f"group weighted mean = {mean*1e3:.4f} mV")
    if np.isfinite(rec["expected"]):
        ax.axhline(rec["expected"] * 1e3, color="tab:green", ls="--", lw=1.3,
                   zorder=1, label="$\\eta^2(x\\,w)$ from REF_STEP6")

    ax.errorbar(xs[order], vals[order], yerr=errs[order], fmt="o", ms=7,
                capsize=3, lw=1.0, color="tab:purple", mec="k", mew=0.4, zorder=3)
    # Beside the marker rather than above it -- the error bar owns the vertical --
    # and alternating up/down along x, because the case worth reading at a glance
    # is the one where every point sits at the same height and one column of
    # labels would pile up on the next.
    for rank, i in enumerate(order):
        pt = rec["points"][i]
        label = f"w={pt['w']:.3f}"
        if np.isfinite(mean) and mean:
            label += f"\n{(pt['tpa'] / mean - 1.0) * 100:+.2f}%"
        dy = 9 if rank % 2 == 0 else -9
        ax.annotate(label, (pt["x"], pt["tpa"] * 1e3), textcoords="offset points",
                    xytext=(8, dy), ha="left", fontsize=7, color="0.25",
                    va="bottom" if dy > 0 else "top")

    ax.set_xlabel("x   (the drive split, at fixed $x\\,w$)")
    ax.set_ylabel("TPA residue  $Y - \\hat{B}(x, w)$   (mV)")
    ax.set_title(f"Background subtracted, $x\\,w$ = {rec['product']:g}"
                 f"   ({len(rec['points'])} levels)")
    ax.margins(x=0.16, y=0.16)      # right margin holds the last point's label
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=8)

    if np.isfinite(mean) and mean:
        lo, hi = ax.get_ylim()
        twin = ax.twinx()
        twin.set_ylim((lo / (mean * 1e3) - 1.0) * 100,
                      (hi / (mean * 1e3) - 1.0) * 100)
        twin.set_ylabel("deviation from the group mean (%)", fontsize=9)


def _plot_pulls(ax, rec: dict) -> None:
    """Every pairwise split, in sigma -- the quantity :data:`PULL_LIMIT` grades.

    A matrix rather than a list: n members make n(n-1)/2 comparisons, which at
    n = 8 is 28 lines of text and one glance as a grid.  The colour scale
    saturates at PULL_LIMIT, so a cell at full colour is exactly a flagged one.
    """
    mat = pull_matrix(rec)
    labels = [f"({pt['x']:.3g}, {pt['w']:.3g})" for pt in rec["points"]]
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-PULL_LIMIT, vmax=PULL_LIMIT)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isfinite(mat[i, j]):
                continue
            flagged = abs(mat[i, j]) >= PULL_LIMIT
            ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold" if flagged else "normal",
                    color="w" if abs(mat[i, j]) > 0.6 * PULL_LIMIT else "0.15")

    ax.set_xticks(range(len(labels)), labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="w", lw=1.0)
    ax.tick_params(which="minor", length=0)

    worst = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else float("nan")
    verdict = "OK" if worst < PULL_LIMIT else "** CHECK **"
    ax.set_title("Pairwise split (row $-$ column), in $\\sigma$\n"
                 f"max $|$pull$|$ = {worst:.2f}   {verdict}", fontsize=10)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"pull   (clipped at $\\pm${PULL_LIMIT:g})", fontsize=8)
    cb.ax.tick_params(labelsize=7)


def make_plot(fit, path: str | Path | None = None) -> Path | None:
    """One row per check group: the residues, and every pairwise split in sigma.

    Deliberately narrow.  The background block, the level table and the repeat
    scatter are all printed by :func:`report` and none of them are drawn -- they
    are what goes INTO the subtraction, and the figure is about what comes out.
    Returns the path written, or None when there was no check to draw.
    """
    prods = product_records(fit)
    if not prods:
        return None

    import matplotlib
    if path is not None:
        matplotlib.use("Agg")     # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    n_max = max(len(rec["points"]) for rec in prods)
    fig, axes = plt.subplots(
        len(prods), 2, squeeze=False,
        figsize=(13.5, max(5.0, 0.55 * n_max + 2.2) * len(prods)),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    for row, rec in zip(axes, prods):
        _plot_residues(row[0], rec)
        _plot_pulls(row[1], rec)

    eta = (f"$\\eta_{{ref}}$ = {fit.eta:.4g}" if np.isfinite(fit.eta)
           else "no REF_STEP6")
    fig.suptitle(f"Step 6 product check -- pair {fit.index}   ({eta})   "
                 "residue = $Y(x, w) - \\hat{B}(x, w)$, which the model says "
                 "is one number", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    if path is None:
        plt.show()
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ======================================================================
# persistence
# ======================================================================

def _points_dict(rec: dict) -> list[dict]:
    """A group's points with the eta each implies alone, minus the scratch keys."""
    out = []
    for pt in rec["points"]:
        eta, eta_err = _eta_eff(pt["tpa"], pt["tpa_err"], rec["product"])
        out.append({
            **{k: v for k, v in pt.items() if not k.startswith("_")},
            "eta_eff": None if not np.isfinite(eta) else eta,
            "eta_eff_err": None if not np.isfinite(eta_err) else eta_err,
        })
    return out


def _check_dict(fit) -> dict:
    """The product records, JSON-shaped."""
    out = []
    for rec in product_records(fit):
        mean, mean_err = group_mean(rec)
        out.append({
            "product": rec["product"],
            "expected": None if not np.isfinite(rec["expected"]) else rec["expected"],
            "weighted_mean": None if not np.isfinite(mean) else mean,
            "weighted_mean_err": None if not np.isfinite(mean_err) else mean_err,
            "points": _points_dict(rec),
            "pairs": rec["pairs"],
        })
    return {"product": out}


def _channel_dict(fit, plot: Path | None) -> dict:
    """One pair's entry: the background it subtracted, and what was left."""
    return {
        "index": fit.index,
        "wl_x_nm": fit.wl_x_nm,
        "wl_w_nm": fit.wl_w_nm,
        "nominal_wl_nm": fit.nominal_wl_nm,
        "eta_ref": None if not np.isfinite(fit.eta) else fit.eta,
        "plot": None if plot is None else str(plot),
        # "params" mirrors step 6's channels[].fit.params, so the two files'
        # background parameters read the same way; step 6's own block summary is
        # NESTED rather than merged -- it carries a "params" key of its own (the
        # parameter NAMES) that would otherwise silently replace this one.
        "background": {
            "params": {n: {"value": v, "err": e} for n, (v, e) in fit.bg.items()},
            "block": _background_dict(fit),     # noqa: SLF001 -- same family
        },
        "check": _check_dict(fit),
        "levels": [
            {"x": L.x, "w": L.w, "n": L.n, "mean_v": L.mean,
             "rep_std_v": None if not np.isfinite(L.rep_std) else L.rep_std,
             "trace_std_v": L.trace_std, "sigma_v": L.sigma,
             "is_check": L.is_verification(fit.cfg)}
            for L in fit.levels
        ],
    }


def save_json(fits: list, out_path: str | Path, step3: Path,
              plots: dict[int, Path] | None = None) -> Path:
    """Background parameters and the check records, per pair.

    Deliberately NOT step 6's schema.  There is no eta in here, so this file must
    not be loadable where a step-6 result is expected: steps 7 and 8 read
    ``channels[].fit.eta``, this writes ``channels[].check``, and they will refuse
    it rather than silently run on a missing number.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plots = plots or {}
    payload = {
        "version": 2,
        "kind": "step6-verify",
        "borrows": "calib_step6_v2.fit_background + calib_step6_v2.verify_product",
        "fit_q": _config().fit_q,
        "params_bg": list(_config().params_bg),
        "bg_grid": [{"x": x, "w": w, "n": n} for x, w, n in BG_GRID],
        "check_grid": [{"x": x, "w": w, "n": n} for x, w, n in check_grid()],
        "product_checks": [[list(pt) for pt in group] for group in PRODUCT_CHECKS],
        "pull_limit": PULL_LIMIT,
        "ref_step6": None if REF_STEP6 is None else str(REF_STEP6),
        "step3_path": str(step3),
        "channels": [_channel_dict(fit, plots.get(fit.index)) for fit in fits],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


# ======================================================================
# drive
# ======================================================================

def _flagged(fits: list) -> list[tuple[int, dict]]:
    """Every pairwise split at or over :data:`PULL_LIMIT`, with its pair."""
    return [(fit.index, pr)
            for fit in fits
            for rec in product_records(fit)
            for pr in rec["pairs"] if abs(pr["pull"]) >= PULL_LIMIT]


def _plot_pair(fit, stamp: str) -> Path | None:
    """Draw one pair's report PNG; a drawing failure never costs the numbers."""
    png = CALIB_PATH / f"calib_step6verify_pair{fit.index}_{stamp}.png"
    try:
        written = make_plot(fit, png)
    except Exception as exc:              # noqa: BLE001 -- the report above stands
        print(f"  (plot failed, the numbers above stand: {exc})")
        return None
    if written is None:
        print("  (no plot: nothing was checked)")
    else:
        print(f"  Plot -> {written}")
    return written


def _check_and_save(rows_by_pair: dict[int, list], stamp: str, step3: Path, *,
                    layout=None) -> None:
    """Check every pair, print the report, draw its PNG, write the JSON."""
    etas = _ref_etas()
    fits: list = []
    plots: dict[int, Path] = {}
    for index in sorted(rows_by_pair):
        levels = average_levels(rows_by_pair[index])
        eta = etas.get(index)
        b = float(eta) ** 2 if eta is not None else float("nan")
        try:
            fit = check_pair(index, levels, b=b)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"\n=== pair {index} ===\n  check FAILED: {exc}")
            continue
        if layout is not None:
            fit.wl_x_nm, fit.wl_w_nm, fit.nominal_wl_nm = (
                s6._pair_wavelengths(layout, index))       # noqa: SLF001
        fits.append(fit)
        print(f"\n=== pair {index} ===")
        report(fit)
        written = _plot_pair(fit, stamp)
        if written is not None:
            plots[index] = written

    if not fits:
        print("\nNo pair checked -- nothing saved.")
        return
    path = save_json(fits, CALIB_PATH / f"calib_step6verify_result_{stamp}.json",
                     step3, plots)
    print(f"\nSaved background + product checks -> {path}")

    flagged = _flagged(fits)
    if flagged:
        print(f"\n{len(flagged)} check(s) at or over |pull| = {PULL_LIMIT:g}:")
        for index, pr in flagged:
            print(f"  pair {index}: {pr['a']} - {pr['b']}  "
                  f"{pr['frac']*100:+.2f}%  pull {pr['pull']:+.2f}")
    else:
        print(f"\nEvery product check inside |pull| = {PULL_LIMIT:g}.")


def _rel(path: Path) -> str:
    """Repo-relative display path, falling back to absolute for off-tree files."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _print_inputs(step3: Path) -> None:
    """Name the step-3 calibration every level here is encoded through.

    Both entry points print it -- a sweep and an offline re-check -- because
    both encode through it, and a check run against the wrong step-3 file looks
    exactly like a check that passed.  It arrives as ``--step3`` now rather than
    as a hand-edited constant, which is what stopped this draft's copy and step
    6's from drifting apart; printing it is still worth the line, since the
    argument can name the wrong file just as easily as a constant could.

    Printed before the layout is loaded, so a missing or misnamed file names
    itself on the way to the exception rather than after it; ``check_csv``
    swallows that exception and carries on with NaN wavelengths, which is what
    the ``(MISSING)`` mark explains.
    """
    mark = "" if Path(step3).is_file() else "   (MISSING)"
    print(f"Step 3 in : {_rel(step3)}{mark}")


def _print_plan(grid: Grid, schedule) -> None:
    """What is about to be measured, and roughly how long it will take."""
    n_single = sum(1 for _, x, w in schedule if x == 0.0 or w == 0.0)
    secs = (n_single * T_SINGLE_S + (len(schedule) - n_single) * T_BOTH_S
            + len(schedule) * SETTLE_S)
    n_check = sum(n for _, _, n in check_grid())
    print(f"Step 6 verify: {len(schedule)} acquisitions/pair over {len(grid)} levels "
          f"({len(schedule) - n_check} background + {n_check} check), interleaved, "
          f"brightest first (~{secs/60:.1f} min/pair)")
    for group in PRODUCT_CHECKS:
        product, dev = group_product(group)
        n_pairs = len(group) * (len(group) - 1) // 2
        print(f"Group:  x*w = {product:.6g} (+/-{dev*100:.3f}%), {len(group)} levels "
              f"-> {n_pairs} comparisons: "
              f"{', '.join(f'({x:g}, {w:.4g})' for x, w in group)}")
    print(f"Pairs:  {list(PAIR_INDICES)}")


def _run_sweep(step3: Path, check_after: bool) -> None:
    """Drive every pair's interleaved schedule; optionally check and save."""
    validate_checks()
    _push_config()
    _print_inputs(step3)
    grid = measure_grid()
    layout = s6._load_layout(step3)                        # noqa: SLF001
    schedule = build_schedule(_config(), grid)
    _print_plan(grid, schedule)

    slm = s6.connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = s6.connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                         t_both=T_BOTH_S, t_single=T_SINGLE_S)
    rows_by_pair: dict[int, list] = {}
    try:
        for index in PAIR_INDICES:
            print(f"\n=== Sweep: pair {index} ===")
            rows_by_pair[index] = measure_pair(
                daq, slm, layout, index, schedule,
                cfg=_config(), acq=_acq(),
                progress_callback=lambda p: print(p.line()), log=print)
    finally:
        slm.close_slm()
        daq.disconnect()

    stamp = time.strftime("%m%d_%H%M")
    csv_path = write_meas_csv(
        rows_by_pair, CALIB_PATH / f"calib_step6verify_meas_{stamp}.csv")
    total = sum(len(v) for v in rows_by_pair.values())
    print(f"\nSaved {total} rows to {csv_path}")   # raw rows on disk BEFORE checking
    if check_after:
        _check_and_save(rows_by_pair, stamp, step3, layout=layout)


def check_csv(path: str | Path, step3: Path) -> None:
    """Re-run the check on an already-recorded CSV (no hardware).

    Reads step 6's own column layout, so a full ``calib_step6v2_meas_*.csv`` works
    here too: the background block and the check levels are a subset of step 6's
    grid, and the extra cross levels are simply never referenced.
    """
    validate_checks()
    _push_config()
    _print_inputs(step3)
    rows_by_pair = load_meas_csv(path)
    n = sum(len(v) for v in rows_by_pair.values())
    print(f"Loaded {path}: {len(rows_by_pair)} pair(s), {n} acquisitions")
    try:
        layout = s6._load_layout(step3)                    # noqa: SLF001
    except (FileNotFoundError, ValueError) as exc:
        print(f"(layout unavailable, wavelengths left as NaN: {exc})")
        layout = None
    _check_and_save(rows_by_pair, time.strftime("%m%d_%H%M"), step3,
                    layout=layout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calib_step6_verify.py",
        description="Step 6 v2's product-only check, on its own short grid.",
        epilog="--step3 has no default: a check run against the wrong step-3 "
               "calibration looks exactly like a check that passed.",
    )
    parser.add_argument(
        "--step3", required=True, type=Path, metavar="JSON",
        help="Step-3b/3c calibration the channel layout is built from (required)",
    )
    parser.add_argument(
        "csv", nargs="?", type=Path,
        help="re-check this recorded CSV offline instead of measuring",
    )
    parser.add_argument(
        "--meas", "-m", action="store_true",
        help="measure and write the raw CSV only; do not check",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.csv is not None:            # offline re-check, no hardware
        check_csv(args.csv, args.step3)
        return 0
    _run_sweep(args.step3, check_after=not args.meas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
