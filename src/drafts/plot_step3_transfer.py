"""Fit and plot the sin^2 transfer curve of every channel in a Step-3 CSV.

Reads a ``calib_step3{a,b,c}_<stamp>.csv`` (the long
``coordinate_px,wavelength_nm,level,intensity,raw_intensity_w`` format written
by ``write_intensity_calibration_csv``) and fits each channel's level sweep
twice with :func:`calibration_module.fit.transfer.fit_transfer_curve` -- once
with ``curvature=False`` (the original 4-parameter linear-``Delta`` model) and
once with it on (the 5-parameter model, now the default everywhere)::

    I(L) = floor + contrast * sin(Delta(L) / 2)^2
    Delta(L) = phase_curv * L^2 + phase_slope * L + phase_offset

so the panel that gets drawn is a direct comparison of what the encoder used to
do against what it does now.  Both models come from the production fitter --
this script has no model of its own to drift out of step with it.

Draws one PNG next to the CSV: a panel per channel with both curves and both
residuals, four trend axes across the array, and a table.

Spiked cells are dropped by the production fitter, which does this on its own
now -- ``--clip`` is passed straight through to ``fit_transfer_curve`` and the
markers on the plot are recovered from the fit it returns, so the red points are
the ones the stored JSON was actually built without.  ``--exclude`` drops points
by hand on top of that, and ``--no-clip`` turns the automatic pass off::

    python src/drafts/plot_step3_transfer.py
    python src/drafts/plot_step3_transfer.py src/calib_data/run_x/calib_step3c_0907_1358.csv
    python src/drafts/plot_step3_transfer.py --exclude 1428:845,860,875,890 --no-clip
    python src/drafts/plot_step3_transfer.py --linear-only

That clipping is not cosmetic.  A channel with a 25x spike in it fits to
nonsense -- on the 0907 Step-3c run, channel 11's stored fit has an rms of 824%
of its contrast and an encoding window of 417..573 instead of 399..887 -- and
because ``save_calibration_result`` fits at write time, that nonsense is what
lands in the JSON steps 6/7/8 read.  Files written before the fitter learned to
clip still carry it, so comparing this report's table against the JSON's stored
fits is how you find out whether a given file needs regenerating.

Note the units.  On the DAQ path (Step 3a/3c) ``intensity`` is a dark-subtracted
*volt*, not a normalized power, and the CSV's ``raw_intensity_w`` column is a
copy of it -- only the OSA path (Step 3b) fills those two columns differently.
The sin^2 model carries a free amplitude, so neither fit cares; the axis labels
and the ``contrast`` column just mean volts here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from calibration_module.fit.transfer import (  # noqa: E402
    TransferFit,
    TransferFitError,
    fit_transfer_curve,
)

CALIB_DIR = REPO_SRC / "calib_data"

C_USED = "#1f77b4"
C_LIN = "#ff7f0e"
C_QUAD = "#2ca02c"
C_DROP = "#d62728"


# --------------------------------------------------------------------- load


def load_step3_csv(path: Path):
    """-> (coordinates, wavelengths, levels, curves[channel, level]).

    Channels come back in coordinate order, levels ascending, and a
    (coordinate, level) pair the file never recorded becomes NaN -- which is
    what the fitter already treats as "not measured".
    """
    values: dict[float, dict[int, float]] = defaultdict(dict)
    wavelength: dict[float, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            coord = float(row["coordinate_px"])
            values[coord][int(row["level"])] = float(row["intensity"])
            wavelength[coord] = float(row["wavelength_nm"])

    if not values:
        raise SystemExit(f"{path.name}: no rows")

    coords = np.array(sorted(values), dtype=float)
    levels = np.array(sorted({lv for d in values.values() for lv in d}), dtype=float)
    curves = np.full((coords.size, levels.size), np.nan)
    for i, coord in enumerate(coords):
        for j, level in enumerate(levels):
            curves[i, j] = values[coord].get(int(level), np.nan)
    waves = np.array([wavelength[c] for c in coords], dtype=float)
    return coords, waves, levels, curves


def parse_exclusions(specs: list[str], coords: np.ndarray) -> dict[int, set[int]]:
    """``--exclude 1428:845,860`` -> {channel_index: {levels}}.

    The coordinate is matched to the nearest channel, so the integer pixel read
    off a CSV line is enough -- no need to retype the full float.
    """
    out: dict[int, set[int]] = defaultdict(set)
    for spec in specs:
        coord_text, _, level_text = spec.partition(":")
        if not level_text:
            raise SystemExit(f"--exclude {spec!r}: expected <coord>:<level>[,<level>...]")
        idx = int(np.argmin(np.abs(coords - float(coord_text))))
        for token in level_text.split(","):
            out[idx].add(int(token))
    return out


# ------------------------------------------------------------- fit compare


def encoding_shift(lin: TransferFit, quad: TransferFit) -> float:
    """Largest ``level_for(v)`` disagreement between the models, over v in [0,1].

    The two models can differ in rms and still command the same grayscale, so
    this says whether the extra term reaches the encoder at all.  Levels are
    left unrounded, so a shift smaller than one level still reads as what it is.
    """
    theta = 2.0 * np.arcsin(np.sqrt(np.linspace(0.0, 1.0, 201)))
    diff = np.array([quad.level_at(t) - lin.level_at(t) for t in theta])
    return float(np.nanmax(np.abs(diff)))


def command_error(lin: TransferFit, quad: TransferFit):
    """-> (worst dv, the v it happens at, rms dv) for the linear model.

    Levels are only the middle of the story -- a level shift where the curve is
    flat costs nothing, and a small one on the steep flank costs a lot.  So ask
    the question in the units the encoder is actually asked for: command ``v``,
    take the level the *linear* model picks for it, and evaluate what the
    curved model says the panel really delivers there.  The gap is the encoding
    error the linear retardance was carrying, in normalized units.
    """
    v = np.linspace(0.0, 1.0, 401)
    theta = 2.0 * np.arcsin(np.sqrt(v))
    lin_levels = np.array([lin.level_at(t) for t in theta])
    delivered = np.sin(quad.retardance(lin_levels) / 2.0) ** 2
    diff = delivered - v
    worst = int(np.nanargmax(np.abs(diff)))
    return float(diff[worst]), float(v[worst]), float(np.sqrt(np.nanmean(diff ** 2)))


def dropped_by(pre: TransferFit, levels, curve, used, clip: float, expect: int):
    """-> boolean mask (over all levels) of the points the clip took.

    ``fit_transfer_curve`` reports *how many* points it dropped, not which.
    Recover them by re-running the decision it made: residual against the
    UNCLIPPED fit ``pre``, MAD for the scale, floored at 0.1% of contrast.

    It has to be the unclipped fit.  Judging against the clipped one gets the
    obvious cases right and the marginal ones wrong, because once a point is
    out of the fit its residual and the MAD both move -- 0907 channel 0's
    dropped endpoint reads 6.06 sigma against the fit that dropped it and 5.01
    against the fit that resulted, so the plot would have silently disagreed
    with the ``n_clipped`` beside it.

    Exact whenever the clip converged in one pass, which is every channel seen
    so far; ``expect`` cross-checks that and an empty mask is returned if it
    does not hold, since a marker that contradicts ``n_clipped`` is worse than
    no marker.
    """
    out = np.zeros(levels.shape, dtype=bool)
    if not expect or clip <= 0:
        return out
    resid = curve[used] - pre.intensity(levels[used])
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    sigma = max(sigma, 1e-3 * pre.contrast)
    bad = np.abs(resid) > clip * sigma
    if int(bad.sum()) != expect:
        return out
    out[np.flatnonzero(used)[bad]] = True
    return out


def fit_channel(levels, curve, *, clip: float, forced: set[int], quadratic: bool):
    """-> (linear fit, curved fit or None, mask of points used, note).

    Outlier removal is the production fitter's now, so this passes ``clip``
    through and asks afterwards which points it took.  Both models are then
    refitted on that surviving set with clipping off, so the linear-vs-curved
    comparison in the table is over identical points -- otherwise each model
    would clip its own and the rms columns would not be comparable.

    ``forced`` levels are dropped by hand first, before the fitter sees them.

    The curved fit comes back None when the production fitter declined the
    curvature term -- it reports that as ``phase_curv == 0.0``.
    """
    used = np.isfinite(curve)
    for level in forced:
        used &= levels != float(level)

    note = ""
    try:
        pre = fit_transfer_curve(
            levels[used], curve[used], clip=0, curvature=quadratic
        )
        best = fit_transfer_curve(
            levels[used], curve[used], clip=clip, curvature=quadratic
        )
    except TransferFitError as exc:
        return None, None, used, f"FIT FAILED: {exc}"

    if best.n_clipped:
        bad = dropped_by(pre, levels, curve, used, clip, best.n_clipped)
        if bad.any():
            note = "clipped " + ",".join(str(int(v)) for v in levels[bad])
            used &= ~bad
        else:
            note = f"clipped {best.n_clipped} (levels not identified)"

    try:
        lin = fit_transfer_curve(levels[used], curve[used], curvature=False, clip=0)
    except TransferFitError as exc:
        return None, None, used, f"FIT FAILED: {exc}"

    quad = None
    if quadratic:
        try:
            candidate = fit_transfer_curve(levels[used], curve[used], clip=0)
            quad = candidate if candidate.phase_curv != 0.0 else None
            if quad is None:
                note = (note + "; " if note else "") + "curvature declined"
        except TransferFitError as exc:
            note = (note + "; " if note else "") + f"curved fit failed: {exc}"
    return lin, quad, used, note


# --------------------------------------------------------------------- plot


def mark_at_edge(ax, x, y, *, ms: float = 8.0, mew: float = 2.0) -> None:
    """Draw excluded points, pinning any that fall outside the current ylim.

    An excluded point sits where it does *because* it is wild, so letting it
    set the frame hides the curve it was excluded from.  Points inside the
    range keep their cross; points outside become a triangle on the frame edge
    at the level they were measured at, so the panel still says where the
    exclusion happened rather than dropping it silently.
    """
    lo, hi = ax.get_ylim()
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    inside = (y >= lo) & (y <= hi)
    if inside.any():
        ax.plot(x[inside], y[inside], "x", ms=ms, mew=mew, color=C_DROP,
                zorder=5, label="excluded")
    for outside, marker, edge in ((y > hi, "^", hi), (y < lo, "v", lo)):
        if outside.any():
            ax.plot(x[outside], np.full(int(outside.sum()), edge), marker,
                    ms=ms * 0.8, color=C_DROP, zorder=5, clip_on=False,
                    label=None if inside.any() else "excluded")
    ax.set_ylim(lo, hi)


def edge_curv(fit: TransferFit, levels: np.ndarray) -> float:
    """``phase_curv`` restated as radians of departure from linear at the sweep edge.

    ``phase_curv`` itself is ~1e-6 rad/level^2, which reads as nothing.  Scaled
    by the half-span it becomes the number worth quoting: how far the curvature
    bends ``Delta`` by the end of the sweep.
    """
    half = 0.5 * (float(levels.max()) - float(levels.min()))
    return fit.phase_curv * half * half


def draw(path, coords, waves, levels, curves, lins, quads, masks, notes, out_png) -> None:
    n = coords.size
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    any_quad = any(q is not None for q in quads)

    fig = plt.figure(figsize=(15.5, 3.6 * nrow + 8.5))
    outer = GridSpec(
        nrow + 2, ncol, figure=fig,
        height_ratios=[3.0] * nrow + [2.2, 2.8],
        hspace=0.45, wspace=0.22,
        left=0.06, right=0.985, top=0.955, bottom=0.03,
    )

    dense = np.linspace(levels.min(), levels.max(), 600)

    for i in range(n):
        cell = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[i // ncol, i % ncol],
            height_ratios=[3, 1], hspace=0.08,
        )
        ax = fig.add_subplot(cell[0])
        axr = fig.add_subplot(cell[1], sharex=ax)

        lin, quad, used = lins[i], quads[i], masks[i]
        drop = np.isfinite(curves[i]) & ~used

        ax.plot(levels[used], curves[i][used] * 1e3, "o", ms=3.4,
                color=C_USED, zorder=3, label="used")
        # Scale to the points the fit actually saw: a spike big enough to be
        # worth excluding is also big enough to flatten the curve it was
        # excluded from, so it gets pinned to the frame instead of setting it.
        kept = curves[i][used] * 1e3
        lo, hi = float(np.min(kept)), float(np.max(kept))
        pad = 0.12 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
        if drop.any():
            mark_at_edge(ax, levels[drop], curves[i][drop] * 1e3)

        if lin is not None:
            ax.plot(dense, lin.intensity(dense) * 1e3, "--", lw=1.3,
                    color=C_LIN, zorder=2, label=r"linear $\Delta$")
            for lvl, style in ((lin.off_level_exact, ":"), (lin.on_level_exact, "--")):
                if levels.min() - 60 <= lvl <= levels.max() + 60:
                    ax.axvline(lvl, ls=style, lw=0.9, color="#888888", zorder=1)
            rl = (curves[i] - lin.intensity(levels)) * 1e3
            axr.plot(levels[used], rl[used], "o", ms=2.6, color=C_LIN, label="linear")
            span = float(np.max(np.abs(rl[used])))

            if quad is not None:
                ax.plot(dense, quad.intensity(dense) * 1e3, "-", lw=1.4,
                        color=C_QUAD, zorder=2.5, label=r"quadratic $\Delta$")
                rq = (curves[i] - quad.intensity(levels)) * 1e3
                axr.axhspan(-quad.rms * 1e3, quad.rms * 1e3, color=C_QUAD, alpha=0.16, lw=0)
                axr.plot(levels[used], rq[used], "o", ms=2.6, color=C_QUAD, label="quadratic")
            else:
                axr.axhspan(-lin.rms * 1e3, lin.rms * 1e3, color=C_LIN, alpha=0.16, lw=0)

            axr.set_ylim(-1.25 * span or -1.0, 1.25 * span or 1.0)
            if drop.any():
                mark_at_edge(axr, levels[drop], rl[drop], ms=6, mew=1.6)

            pct = 100.0 * lin.rms / lin.contrast if lin.contrast else float("nan")
            text = [
                f"contrast {lin.contrast * 1e3:.3f} mV",
                f"floor    {lin.floor * 1e3:.3f} mV",
                f"rate     {float(lin.phase_rate(0)) * 1e3:.3f} mrad/lvl",
                f"off {lin.off_level:d} -> on {lin.on_level:d}",
                f"rms lin  {lin.rms * 1e3:.4f} mV ({pct:.2f}%)",
            ]
            if quad is not None:
                qpct = 100.0 * quad.rms / quad.contrast if quad.contrast else float("nan")
                gain = lin.rms / quad.rms if quad.rms > 0 else float("inf")
                dv, at_v, _ = command_error(lin, quad)
                text += [
                    f"rms quad {quad.rms * 1e3:.4f} mV ({qpct:.2f}%)  {gain:.1f}x",
                    f"curv {edge_curv(quad, levels):+.4f} rad @ edge",
                    f"rate {float(quad.phase_rate(quad.off_level_exact)) * 1e3:.2f}"
                    f" -> {float(quad.phase_rate(quad.on_level_exact)) * 1e3:.2f}",
                    f"dL max {encoding_shift(lin, quad):.1f} level",
                    f"dv max {dv * 100:+.2f}% at v={at_v:.2f}",
                ]
            flags = []
            if drop.any():
                worst = float(np.max(np.abs(curves[i][drop]))) * 1e3
                flags.append(f"{int(drop.sum())} pt excluded (to {worst:.1f} mV)")
            if lin.extrapolated:
                flags.append("extrapolated")
            if lin.clipped:
                flags.append("clipped")
            if quad is not None and quad.turns_in_panel:
                flags.append(f"turns at {quad.turning_level:.0f}")
            ax.text(
                0.03, 0.97, "\n".join(text + flags),
                transform=ax.transAxes, va="top", ha="left", fontsize=7.0,
                family="monospace",
                bbox=dict(fc="white", ec="#cccccc", alpha=0.85, pad=2.5),
            )
        else:
            ax.text(0.5, 0.5, notes[i], transform=ax.transAxes,
                    ha="center", va="center", color=C_DROP, fontsize=9, wrap=True)

        axr.axhline(0.0, lw=0.8, color="#444444")
        ax.set_title(f"ch {i}   x = {coords[i]:.1f} px   {waves[i]:.3f} nm", fontsize=9.5)
        ax.set_ylabel("intensity [mV]", fontsize=8)
        axr.set_ylabel("resid", fontsize=7.5)
        axr.set_xlabel("grayscale level", fontsize=8)
        ax.tick_params(labelsize=7.5, labelbottom=False)
        axr.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        axr.grid(alpha=0.25)
        if i == 0 or drop.any():
            ax.legend(fontsize=6.8, loc="lower right", framealpha=0.85)

    # --- trends across the array ----------------------------------------
    keep = [i for i, f in enumerate(lins) if f is not None]
    qk = [i for i in keep if quads[i] is not None]
    ok_x = coords[keep]
    row = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[nrow, :], wspace=0.32)

    # The linear model has one rate; the curved model has a range, so show the
    # band it sweeps rather than a single misleading number.
    ax = fig.add_subplot(row[0])
    ax.plot(ok_x, [float(lins[i].phase_rate(0)) * 1e3 for i in keep], "o-", ms=4, lw=1.2,
            color=C_LIN, label="linear")
    if any_quad:
        at_off = [float(quads[i].phase_rate(quads[i].off_level_exact)) * 1e3 for i in qk]
        at_on = [float(quads[i].phase_rate(quads[i].on_level_exact)) * 1e3 for i in qk]
        ax.fill_between(coords[qk], at_on, at_off, color=C_QUAD, alpha=0.22,
                        label="quadratic, off..on")
        ax.plot(coords[qk], at_off, "-", lw=1.0, color=C_QUAD)
        ax.plot(coords[qk], at_on, "-", lw=1.0, color=C_QUAD)
        ax.legend(fontsize=6.8)
    ax.set_xlabel("coordinate [px]", fontsize=8)
    ax.set_ylabel("phase rate [mrad/level]", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(row[1])
    if any_quad:
        ax.plot(coords[qk], [edge_curv(quads[i], levels) for i in qk], "o-",
                ms=4, lw=1.2, color=C_QUAD)
        ax.axhline(0.0, lw=0.9, ls="--", color="#888888")
        ax.set_ylabel("curvature [rad at sweep edge]", fontsize=8)
    else:
        ax.plot(ok_x, [lins[i].contrast * 1e3 for i in keep], "o-", ms=4, lw=1.2, color=C_QUAD)
        ax.set_ylabel("contrast [mV]", fontsize=8)
    ax.set_xlabel("coordinate [px]", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.3)

    # What the linear model cost where it was used: command v, take the level
    # the linear model picks, ask the curved model what arrives.
    ax = fig.add_subplot(row[2])
    if any_quad:
        vv = np.linspace(0.0, 1.0, 401)
        theta = 2.0 * np.arcsin(np.sqrt(vv))
        cmap = plt.get_cmap("viridis")
        for j, i in enumerate(qk):
            lvl = np.array([lins[i].level_at(t) for t in theta])
            delivered = np.sin(quads[i].retardance(lvl) / 2.0) ** 2
            ax.plot(vv, (delivered - vv) * 100, lw=1.1,
                    color=cmap(j / max(len(qk) - 1, 1)))
        ax.axhline(0.0, lw=0.9, ls="--", color="#888888")
        ax.set_xlabel("commanded v", fontsize=8)
        ax.set_ylabel("delivered - commanded [% FS]", fontsize=8)
        ax.set_title("what the linear model cost", fontsize=8.5)
    else:
        ax.axis("off")
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(row[3])
    ax.plot(ok_x, [100.0 * lins[i].rms / lins[i].contrast for i in keep], "o-",
            ms=4, lw=1.2, color=C_LIN, label="linear")
    if any_quad:
        ax.plot(coords[qk], [100.0 * quads[i].rms / quads[i].contrast for i in qk], "s-",
                ms=4, lw=1.2, color=C_QUAD, label="quadratic")
        ax.set_yscale("log")
        ax.legend(fontsize=6.8)
    ax.set_xlabel("coordinate [px]", fontsize=8)
    ax.set_ylabel("rms residual [% of contrast]", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.3, which="both")

    # --- parameter table -------------------------------------------------
    ax = fig.add_subplot(outer[nrow + 1, :])
    ax.axis("off")
    head = (f"{'ch':>3} {'x_px':>8} {'nm':>9} {'contr_mV':>9} {'rate_off':>9} "
            f"{'rate_on':>8} {'off':>5} {'on':>5} {'rms_lin%':>9} {'rms_qua%':>9} "
            f"{'gain':>6} {'curv_rad':>9} {'dL':>6} {'dv_max%':>8} {'n':>4}  notes")
    lines = [head, "-" * len(head)]
    for i, lin in enumerate(lins):
        if lin is None:
            lines.append(f"{i:>3} {coords[i]:>8.1f} {waves[i]:>9.3f}   {notes[i]}")
            continue
        pct = 100.0 * lin.rms / lin.contrast if lin.contrast else float("nan")
        quad = quads[i]
        shown = quad or lin
        if quad is not None:
            qpct = 100.0 * quad.rms / quad.contrast if quad.contrast else float("nan")
            gain = lin.rms / quad.rms if quad.rms > 0 else float("inf")
            dv, _, _ = command_error(lin, quad)
            qcols = (f"{qpct:>9.3f} {gain:>6.1f} {edge_curv(quad, levels):>+9.4f} "
                     f"{encoding_shift(lin, quad):>6.1f} {dv * 100:>+8.2f}")
        else:
            qcols = f"{'-':>9} {'-':>6} {'-':>9} {'-':>6} {'-':>8}"
        extra = [notes[i]] if notes[i] else []
        if lin.extrapolated:
            extra.append("extrapolated")
        if lin.clipped:
            extra.append("clipped")
        lines.append(
            f"{i:>3} {coords[i]:>8.1f} {waves[i]:>9.3f} {lin.contrast * 1e3:>9.4f} "
            f"{float(shown.phase_rate(shown.off_level_exact)) * 1e3:>9.4f} "
            f"{float(shown.phase_rate(shown.on_level_exact)) * 1e3:>8.4f} "
            f"{shown.off_level:>5d} {shown.on_level:>5d} "
            f"{pct:>9.3f} {qcols} {lin.n_used:>4d}  {'; '.join(extra)}"
        )

    ok_lin = [f for f in lins if f is not None]
    ok_quad = [q for q in quads if q is not None]
    if ok_lin and ok_quad:
        idx = [i for i in range(len(lins)) if lins[i] is not None and quads[i] is not None]
        lines.append("-" * len(head))
        lines.append(
            f"{'mean':>3} {'':>8} {'':>9} {np.mean([f.contrast for f in ok_lin]) * 1e3:>9.4f} "
            f"{np.mean([float(quads[i].phase_rate(quads[i].off_level_exact)) for i in idx]) * 1e3:>9.4f} "
            f"{np.mean([float(quads[i].phase_rate(quads[i].on_level_exact)) for i in idx]) * 1e3:>8.4f} "
            f"{'':>5} {'':>5} "
            f"{np.mean([100 * f.rms / f.contrast for f in ok_lin]):>9.3f} "
            f"{np.mean([100 * q.rms / q.contrast for q in ok_quad]):>9.3f} "
            f"{np.mean([lins[i].rms / quads[i].rms for i in idx]):>6.1f} "
            f"{np.mean([edge_curv(quads[i], levels) for i in idx]):>+9.4f} "
            f"{np.mean([encoding_shift(lins[i], quads[i]) for i in idx]):>6.1f} "
            f"{np.mean([command_error(lins[i], quads[i])[0] for i in idx]) * 100:>+8.2f}"
        )
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=7.4)

    dropped_total = sum(int((np.isfinite(curves[i]) & ~masks[i]).sum()) for i in range(n))
    model = r"linear vs quadratic $\Delta(L)$" if any_quad else r"linear $\Delta(L)$"
    fig.suptitle(
        f"{path.name} - sin$^2$ transfer fit, {model}, {n} channels x {levels.size} levels"
        f"   ({dropped_total} point(s) excluded)",
        fontsize=13, y=0.985,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print("\n".join(lines))
    print(f"\nwrote {out_png}")


# --------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", type=Path,
                    help="Step-3 CSV (default: newest calib_step3*.csv under src/calib_data)")
    ap.add_argument("-o", "--out", type=Path, help="output PNG (default: <stem>_fit.png)")
    ap.add_argument("--exclude", action="append", default=[], metavar="COORD:L1,L2",
                    help="drop these levels on the channel nearest COORD; repeatable")
    ap.add_argument("--clip", type=float, default=6.0,
                    help="robust-sigma threshold for automatic spike rejection (default 6)")
    ap.add_argument("--no-clip", dest="clip", action="store_const", const=0.0,
                    help="keep every point except those named by --exclude")
    ap.add_argument("--linear-only", dest="quadratic", action="store_false",
                    help="fit only the 4-parameter linear-retardance model")
    args = ap.parse_args(argv)

    path = args.csv
    if path is None:
        found = sorted(CALIB_DIR.rglob("calib_step3*.csv"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit(f"no calib_step3*.csv under {CALIB_DIR}")
        path = found[-1]
    if not path.exists():
        raise SystemExit(f"{path}: not found")

    coords, waves, levels, curves = load_step3_csv(path)
    forced = parse_exclusions(args.exclude, coords)

    lins: list[TransferFit | None] = []
    quads: list[TransferFit | None] = []
    masks: list[np.ndarray] = []
    notes: list[str] = []
    for i in range(coords.size):
        lin, quad, used, note = fit_channel(
            levels, curves[i], clip=args.clip, forced=forced.get(i, set()),
            quadratic=args.quadratic,
        )
        if forced.get(i):
            hand = ",".join(str(v) for v in sorted(forced[i]))
            note = f"excluded {hand}" + (f"; {note}" if note else "")
        lins.append(lin)
        quads.append(quad)
        masks.append(used)
        notes.append(note)

    out_png = args.out or path.with_name(path.stem + "_fit.png")
    draw(path, coords, waves, levels, curves, lins, quads, masks, notes, out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
