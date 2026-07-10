"""Matplotlib renderers for step-7 (comb phase) results, shared GUI/CLI.

Factored out of ``src/drafts/calib_step7_test.py`` so the GUI's pipeline page
and the draft script draw byte-identical figures.  Every function renders into
a caller-supplied :class:`matplotlib.figure.Figure` (works with any backend --
Agg for PNGs, the Qt canvas in the GUI) and never calls ``savefig`` itself.
"""
from __future__ import annotations

import numpy as np


def _sigma(value: float, err: float) -> float:
    return abs(value) / err if err else float("nan")


def plot_fringe(fig, fit, tgt: int) -> None:
    """Measured Y(theta2) with the fitted a/b/dPhi_comb model curve + pulls.

    Left: the dark-subtracted measurements over the half fringe with the full
    fitted model (interference + pinned step-6 single-beam background).
    Right: the pulls. Port of the draft's ``make_plot``.
    """
    fig.clear()
    ax1, ax2 = fig.subplots(1, 2)

    dphi = np.degrees(fit.dphi_slm)             # theta2 - 180 deg
    pulls = fit.residuals / fit.sem

    # smooth model over the reachable half turn theta2 in [0, 180] deg
    th = np.radians(np.linspace(0.0, 180.0, 400))
    g = np.sin(th / 2.0) ** 2                    # sin^2(theta2/2)
    dslm = th - np.pi
    model = (fit.a**2 + fit.b**2 * g**2
             + 2.0 * fit.a * fit.b * g * np.cos(dslm + fit.dphi_comb)
             + fit.bg0 + fit.bg1 * g + fit.bg2 * g**2 + fit.offset)
    ax1.plot(np.degrees(dslm), model * 1e3, "-", color="tab:blue", lw=1.6,
             label=r"fit: $a^2+b^2\sin^4+2ab\sin^2\cos+\mathrm{sb}(\theta_2)$")
    ax1.errorbar(dphi, fit.y * 1e3, yerr=fit.sem * 1e3, fmt="o", ms=5,
                 color="tab:orange", ecolor="lightgray", elinewidth=1,
                 capsize=2, zorder=3, label="measured (dark-subtracted)")
    ax1.set_xlabel(r"$\Delta\Phi_{SLM} = \theta_2 - 180^\circ$  (deg)")
    ax1.set_ylabel(r"$Y$, dark-subtracted  (mV)")
    ax1.set_title(f"Pair {tgt} interference (half fringe)")
    ax1.legend(loc="best", fontsize=8)

    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.scatter(dphi, pulls, c="tab:red", s=40, edgecolor="k", lw=0.4)
    ax2.set_xlabel(r"$\Delta\Phi_{SLM}$  (deg)")
    ax2.set_ylabel("Pull = residual / SEM")
    ax2.set_title(f"Pulls  ($\\chi^2$/dof = {fit.chi2_red:.2f})")
    ax2.legend(loc="upper right", fontsize=8)

    bflag = (("  [a@bound]" if fit.a_at_bound else "")
             + ("  [b@bound]" if fit.b_at_bound else ""))
    txt = (
        f"$\\Delta\\Phi_{{comb}}$ = {fit.dphi_comb_deg:+.2f} $\\pm$ "
        f"{np.degrees(fit.dphi_comb_err):.2f} deg  "
        f"({_sigma(fit.dphi_comb, fit.dphi_comb_err):.0f}$\\sigma$)\n"
        f"a = {fit.a*1e3:.3f} ($\\eta$ {fit.eta_ref*1e3:.3f}), "
        f"b = {fit.b*1e3:.3f} ($\\eta$ {fit.eta_tgt*1e3:.3f}) mV$^{{1/2}}${bflag}\n"
        f"d = {fit.offset*1e3:+.3f} mV  (should be $\\approx$0)\n"
        f"$\\chi^2$/dof = {fit.chi2_red:.2f} (Birge x{fit.birge:.2f})"
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
    from .tpa_phase import _average_points  # same cell averaging as the fit

    fit = result.fit
    if fit is None:
        raise ValueError("result has no fit attached; run the fit first")

    fig.clear()
    ax1, ax2 = fig.subplots(1, 2)

    x_t, w_t, _x_r, _w_r, _, _ = _average_points(result)
    y_meas = fit.y
    y_pred = fit.y_pred
    sem = fit.sem
    pulls = fit.residuals / sem
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
    ax1.errorbar(y_meas * 1e3, y_pred * 1e3, xerr=sem * 1e3, fmt="none",
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
        f"d = {fit.offset*1e3:+.2f} mV  (should be $\\approx$0)\n"
        f"$\\chi^2$/dof = {fit.chi2_red:.1f} (Birge x{fit.birge:.2f})"
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
    ax2.set_ylabel("Pull = residual / SEM")
    ax2.set_title(f"Pulls  ($\\chi^2$/dof = {fit.chi2_red:.1f})")
    ax2.legend(loc="upper left", fontsize=8)

    ok = (fit.chi2_red < 3.0 and np.isfinite(fit.a) and fit.b > 0
          and abs(fit.offset) < 0.5 * fit.a**2)
    verdict = "model OK" if ok else "model REJECTED"
    head = f"TPA comb-phase fit: pair {tgt} vs pair {ref}"
    if subtitle:
        head += f" -- {subtitle}"
    fig.suptitle(f"{head}  [{verdict}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


__all__ = ["plot_fringe", "plot_report"]
