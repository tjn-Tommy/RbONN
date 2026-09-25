"""Matplotlib renderers for step-7 (comb phase) results, shared GUI/CLI.

Factored out of ``src/calibration_module/steps/calib_step7_v1.py`` so the GUI's
pipeline page and the step scripts draw byte-identical figures.  Every function
renders into a caller-supplied :class:`matplotlib.figure.Figure` (works with any
backend -- Agg for PNGs, the Qt canvas in the GUI) and never calls ``savefig``
itself.

:func:`plot_fringe` serves both estimators.  Every fit in the project fits the
same cosine argument, ``cos(dPhi_SLM + dPhi_comb)``
(:func:`~calibration_module.fit.phase.fringe_arg`), so the curve is drawn the
way it was fitted; the error bars come from ``std_total``, which collapses to
``std`` for a fit that floated a/b.
"""
from __future__ import annotations

import numpy as np

from .phase import phi_half


def _sigma(value: float, err: float) -> float:
    return abs(value) / err if err else float("nan")


def plot_fringe(fig, fit, tgt: int) -> None:
    """Measured Y(dPhi_SLM) with the fitted a/b/dPhi_comb model curve + pulls.

    Left: the dark-subtracted measurements over the half fringe with the full
    fitted model (interference + the pinned step-6 single-beam background).
    Right: the pulls.

    The cosine argument comes from
    :meth:`~calibration_module.fit.phase.PhaseFit.fringe_arg`
    (``dPhi_SLM + dPhi_comb``, the project's one convention), so the curve is
    drawn the way it was fitted.

    Error bars and pulls are taken against ``std_total`` -- the point spread
    with the pinned step-6 eta's model error folded in.  For a fit that floated
    a/b that term is zero and ``std_total`` is exactly ``std``, so this is
    unchanged for those; for a pinned fit it is the sigma the residual is
    actually owed, and using ``std`` alone would draw the bars several times too
    small.
    """
    fig.clear()
    ax1, ax2 = fig.subplots(1, 2)

    dphi = np.degrees(fit.dphi_slm)             # dPhi_SLM at the measured points
    pulls = fit.pulls
    std_tot = fit.std_total
    # a and b are pinned (and carry a step-6 error) only for fit_phase_fixed.
    pinned = fit.dm_deta_ref is not None

    # Smooth model over the swept geometry.  Rebuild g / dPhi_SLM the way the
    # fit did -- from the commanded intensities -- so the curve tracks the data:
    # sweep the shared phase over the full 0..180 deg half turn, at the actual
    # reference drive rather than assuming a fully-on one.
    wt_s = xt_s = np.sin(np.radians(np.linspace(0.0, 180.0, 400)) / 2.0) ** 2
    xr_c = float(np.median(fit.x_r)) if fit.x_r is not None else 1.0
    wr_c = float(np.median(fit.w_r)) if fit.w_r is not None else 1.0
    g_s = np.sqrt(np.clip(xt_s * wt_s, 0.0, None))
    dslm = phi_half(xt_s) + phi_half(wt_s) - phi_half(xr_c) - phi_half(wr_c)

    # Fixed step-6 single-beam background = fit.known - a^2 - b^2 g^2 at each
    # fitted point; interpolate it onto the smooth grid (exact at the points, and
    # geometry-agnostic, so no assumption about how the background splits in g).
    bg_pts = fit.known - fit.a**2 - fit.b**2 * fit.g**2
    order = np.argsort(fit.dphi_slm)
    bg_s = np.interp(dslm, fit.dphi_slm[order], bg_pts[order])

    model = (fit.a**2 + fit.b**2 * g_s**2
             + 2.0 * fit.a * fit.b * g_s * np.cos(fit.fringe_arg(dslm))
             + bg_s + fit.offset)
    arg = r"\Delta\Phi_{SLM}+\Delta\Phi_{comb}"
    ax1.plot(np.degrees(dslm), model * 1e3, "-", color="tab:blue", lw=1.6,
             label=rf"fit: $a^2+b^2\sin^4+2ab\sin^2\cos({arg})$")
    ax1.errorbar(dphi, fit.y * 1e3, yerr=std_tot * 1e3, fmt="o", ms=5,
                 color="tab:orange", ecolor="lightgray", elinewidth=1,
                 capsize=2, zorder=3,
                 label=r"measured (dark-subtracted), $\sigma_{tot}$")
    ax1.set_xlabel(r"$\Delta\Phi_{SLM}$  (deg)")
    ax1.set_ylabel(r"$Y$, dark-subtracted  (mV)")
    ax1.set_title(f"Pair {tgt} interference (half fringe)")
    ax1.legend(loc="best", fontsize=8)

    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.scatter(dphi, pulls, c="tab:red", s=40, edgecolor="k", lw=0.4)
    ax2.set_xlabel(r"$\Delta\Phi_{SLM}$  (deg)")
    ax2.set_ylabel(r"Pull = residual / $\sigma_{tot}$")
    if pinned:
        ax2.set_title(r"Pulls   [$\sigma_{tot}^2$ = DAQ trace std$^2$ + "
                      r"(pinned step-6 $\eta$)$^2$]", fontsize=10)
    else:
        ax2.set_title("Pulls")
    ax2.legend(loc="upper right", fontsize=8)

    # With a and b pinned, dphi_comb_err is the fringe noise alone and would
    # quote a bar several times too small; dphi_comb_err_total adds the eta term
    # and collapses back to dphi_comb_err when a/b floated.
    err = fit.dphi_comb_err_total
    txt = (
        f"$\\Delta\\Phi_{{comb}}$ = {fit.dphi_comb_deg:+.2f} $\\pm$ "
        f"{np.degrees(err):.2f} deg  "
        f"({_sigma(fit.dphi_comb, err):.0f}$\\sigma$)\n"
    )
    if pinned:
        txt += (
            f"  = {np.degrees(fit.dphi_comb_err):.2f} (fringe) $\\oplus$ "
            f"{np.degrees(fit.dphi_comb_err_eta):.2f} (step-6 $\\eta$) deg\n"
            f"a = {fit.a*1e3:.3f}, b = {fit.b*1e3:.3f} mV$^{{1/2}}$ "
            f"(both pinned to step 6)\n"
            f"mean resid = {float(np.mean(fit.residuals))*1e3:+.3f} mV (not fitted)\n"
            f"R$^2$ = {fit.r2:.4f}  [1 free param: $\\Delta\\Phi_{{comb}}$]"
        )
    else:
        bflag = (("  [a@bound]" if fit.a_at_bound else "")
                 + ("  [b@bound]" if fit.b_at_bound else ""))
        txt += (
            f"a = {fit.a*1e3:.3f} ($\\eta$ {fit.eta_ref*1e3:.3f}), "
            f"b = {fit.b*1e3:.3f} ($\\eta$ {fit.eta_tgt*1e3:.3f}) mV$^{{1/2}}${bflag}\n"
            f"d = {fit.offset*1e3:+.3f} mV  (should be $\\approx$0)\n"
            f"R$^2$ = {fit.r2:.4f}"
        )
    ax1.text(0.05, 0.95, txt, transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontsize=8)

    fig.tight_layout()


def plot_report(fig, result, tgt: int, ref: int, *, subtitle: str = "") -> None:
    """Ch-efficiency-style report: measured-vs-predicted full voltage + pulls.

    Diagonal (phi^x = phi^w, swap-trivial) cells are squares, off-diagonal
    circles, coloured by x*w so a symmetry breakdown is visible at a glance.
    Port of the draft's ``make_report``.
    """
    from .phase import _average_points  # same cell averaging as the fit

    fit = result.fit
    if fit is None:
        raise ValueError("result has no fit attached; run the fit first")

    fig.clear()
    ax1, ax2 = fig.subplots(1, 2)

    x_t, w_t, _x_r, _w_r, _, _ = _average_points(result)
    y_meas = fit.y
    y_pred = fit.y_pred
    std = fit.std
    pulls = fit.residuals / std
    diag = np.abs(x_t - w_t) < 1e-6                     # phi^x = phi^w
    off = ~diag
    xw = x_t * w_t
    vmin, vmax = float(np.min(xw)), float(np.max(xw))

    # ---- left: measured vs predicted --------------------------------------
    lims = [min(y_meas.min(), y_pred.min()) * 1e3,
            max(y_meas.max(), y_pred.max()) * 1e3]
    pad = 0.03 * ((lims[1] - lims[0]) or 1.0)
    lims = [lims[0] - pad, lims[1] + pad]
    ax1.plot(lims, lims, "--", color="gray", lw=1, label="ideal")
    ax1.errorbar(y_meas * 1e3, y_pred * 1e3, xerr=std * 1e3, fmt="none",
                 ecolor="lightgray", elinewidth=1, zorder=1)
    sc = None
    if off.any():
        sc = ax1.scatter(y_meas[off] * 1e3, y_pred[off] * 1e3, c=xw[off],
                         cmap="viridis", vmin=vmin, vmax=vmax, marker="o",
                         s=55, edgecolor="k", lw=0.4, zorder=2,
                         label=r"off-diagonal ($\phi^x\neq\phi^w$)")
    if diag.any():
        sc_d = ax1.scatter(y_meas[diag] * 1e3, y_pred[diag] * 1e3, c=xw[diag],
                           cmap="viridis", vmin=vmin, vmax=vmax, marker="s",
                           s=75, edgecolor="k", lw=0.6, zorder=3,
                           label=r"diagonal ($\phi^x=\phi^w$)")
        sc = sc if sc is not None else sc_d
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("Measured voltage, trial-averaged (mV)")
    ax1.set_ylabel("Predicted voltage, full model (mV)")
    ax1.set_title(f"Joint fit  (R$^2$ = {fit.r2:.3f})")
    ax1.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=ax1).set_label(r"$x\cdot w$")

    bflag = (("  [a@bound]" if fit.a_at_bound else "")
             + ("  [b@bound]" if fit.b_at_bound else ""))
    txt = (
        f"$\\Delta\\Phi_{{comb}}$ = {fit.dphi_comb_deg:+.1f} $\\pm$ "
        f"{np.degrees(fit.dphi_comb_err):.1f} deg\n"
        f"a = {fit.a*1e3:.3f} $\\pm$ {fit.a_err*1e3:.3f},  "
        f"b = {fit.b*1e3:.3f} $\\pm$ {fit.b_err*1e3:.3f} mV$^{{1/2}}${bflag}\n"
        f"(boxed to $\\pm${fit.bound_frac*100:.0f}% of $\\eta$; "
        f"$\\eta_{{ref}}$={fit.eta_ref*1e3:.3f}, $\\eta_{{tgt}}$={fit.eta_tgt*1e3:.3f})\n"
        f"d = {fit.offset*1e3:+.2f} mV  (should be $\\approx$0)"
    )
    ax1.text(0.05, 0.95, txt, transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontsize=8)

    # ---- right: pulls ------------------------------------------------------
    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    if off.any():
        ax2.scatter(y_pred[off] * 1e3, pulls[off], c="tab:red", marker="o",
                    s=50, edgecolor="k", lw=0.4,
                    label=r"off-diagonal ($\phi^x\neq\phi^w$)")
    if diag.any():
        ax2.scatter(y_pred[diag] * 1e3, pulls[diag], marker="s", s=70,
                    facecolor="none", edgecolor="tab:orange", lw=1.6,
                    label=r"diagonal ($\phi^x=\phi^w$)")
    ax2.set_xlabel("Predicted voltage, full model (mV)")
    ax2.set_ylabel("Pull = residual / std")
    ax2.set_title("Pulls")
    ax2.legend(loc="upper left", fontsize=8)

    # chi2/dof is gone with the SEM weighting; R^2 is the surviving
    # goodness-of-fit number, gated loosely so only clearly bad fits reject.
    ok = (fit.r2 > 0.9 and np.isfinite(fit.a) and fit.b > 0
          and abs(fit.offset) < 0.5 * fit.a**2)
    verdict = "model OK" if ok else "model REJECTED"
    head = f"TPA comb-phase fit: pair {tgt} vs pair {ref}"
    if subtitle:
        head += f" -- {subtitle}"
    fig.suptitle(f"{head}  [{verdict}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


__all__ = ["plot_fringe", "plot_report"]
