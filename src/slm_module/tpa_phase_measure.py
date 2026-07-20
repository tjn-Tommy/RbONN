"""Step-7 measurement driver: two-pair phase sweep against a scope/DAQ monitor.

The fitting/IO half of step 7 lives in :mod:`.tpa_phase` and is geometry
general -- it recovers ``dPhi_comb`` from whatever drive produced the raw
rows.  This module owns the instrument-facing half the unified pipeline
needs: building a drive table, walking it on the SLM while reading the
monitor, and handing the collected rows to :func:`.tpa_phase.fit_result`.
It is kept OUT of ``tpa_phase.py`` so that module stays in lock-step with
the offline driver ``src/drafts/calib_step7_test.py`` (which does its own
SLM/DAQ wiring).

Readings follow the step-6 convention: one fixed-window monitor read per
point (the DAQ's T_both/T_single windows already average enough); ``std``
is the raw low-passed trace spread (diagnostic) and ``sem`` the
instrument-reported standard error of the mean when the sample carries one
(DAQ), falling back to the trace std -- ``sem`` is what the fit weights by
(see :func:`.tpa_phase._average_points`).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .tpa_phase import (
    PairModel,
    PhaseResult,
    fit_result,
    intensity_for_phase,
    phi_half,
    slm_phase_diff,
)


class TPAPhaseAborted(Exception):
    """Raised when a stop event interrupts the phase sweep."""


# ======================================================================
# drive builders
# ======================================================================

def build_phase_sweep(
    *,
    n_points: int = 15,
    phi_start_deg: float = 0.0,
    phi_stop_deg: float = 180.0,
    ref_phase_deg: float = 180.0,
) -> list[tuple[float, float, float, float]]:
    """Symmetric target phase sweep vs a fixed reference (half fringe).

    The target pair is driven symmetrically ``phi^x = phi^w = phi`` over
    ``[phi_start_deg, phi_stop_deg]`` (default 0..180 deg -- the full reachable
    half turn), the reference pair fixed at ``ref_phase_deg`` on both channels
    (default 180 deg == intensity 1, fully on).  Returns target-first commanded
    intensity tuples ``(x_t, w_t, x_r, w_r)`` with ``x = sin(phi/2)^2``, so
    ``dPhi_SLM = phi - ref_phase`` sweeps the fringe.
    """
    phis = np.radians(np.linspace(phi_start_deg, phi_stop_deg, int(n_points)))
    x_r = float(intensity_for_phase(np.radians(ref_phase_deg)))
    x_t = intensity_for_phase(phis)
    return [(float(v), float(v), x_r, x_r) for v in x_t]


def build_symmetry_grid(
    *,
    phi_values_deg: Sequence[float] = (90.0, 135.0, 180.0),
    ref_phase_deg: float = 180.0,
) -> list[tuple[float, float, float, float]]:
    """3x3 grid on the target's individual channel phases (symmetry check).

    Sweeps ``phi^x`` and ``phi^w`` of the target *independently* over
    ``phi_values_deg`` with the reference fixed, so swapped cells and equal-sum
    cells can be compared (see :func:`.tpa_phase.swap_invariance`).  Returns
    target-first commanded intensity tuples.
    """
    x_r = float(intensity_for_phase(np.radians(ref_phase_deg)))
    out: list[tuple[float, float, float, float]] = []
    for px in phi_values_deg:
        xt = float(intensity_for_phase(np.radians(px)))
        for pw in phi_values_deg:
            wt = float(intensity_for_phase(np.radians(pw)))
            out.append((xt, wt, x_r, x_r))
    return out


# ======================================================================
# measurement  (instrument-agnostic two-pair sweep)
# ======================================================================

@dataclass
class TPAPhaseProgress:
    step: int
    total: int
    message: str
    dphi_comb: float | None = None


ProgressCallback = Callable[["TPAPhaseProgress"], None]


def _read_mean_std(
    monitor, timeout: float = 30.0, single: bool = False
) -> tuple[float, float, float]:
    """One averaged reading, its raw trace std, and the per-point SEM of the mean.

    Raises :class:`TPAPhaseAborted` when the monitor read is stopped.  ``sem``
    is the instrument-reported standard error (low-passed, effective-N) when
    the sample carries one, else the trace std -- matching the draft's
    ``_read_daq`` so pipeline and offline runs weight their fits the same way.
    ``single`` marks a weak point (here only the all-off dark): the DAQ reads
    it over its longer T_single window (``single_duration``); the scope
    ignores the flag.
    """
    sample = monitor.monitor_cycle(timeout=timeout, single=single)
    if sample is None:
        raise TPAPhaseAborted("monitor read aborted")
    mean_v = float(sample.value)
    std = getattr(sample, "std", None)
    waveform = getattr(monitor, "last_values", None)
    std_v = (
        float(std) if std is not None and np.isfinite(std)
        else (
            float(np.std(waveform))
            if waveform is not None and np.size(waveform) > 1
            else 0.0
        )
    )
    sem = getattr(sample, "sem", None)
    if sem is not None and np.isfinite(sem):
        sem_v = float(sem)                                     # SEM of the mean (DAQ)
    else:
        sem_v = std_v                                          # scope: no effective-N -> raw std
    return mean_v, std_v, sem_v


def measure_phase_sweep(
    monitor,
    slm,
    layout,
    *,
    tgt_index: int,
    ref_index: int,
    drive: Sequence[tuple[float, float, float, float]],
    tgt_model: PairModel,
    ref_model: PairModel,
    settle: float = 0.15,
    read_timeout: float = 30.0,
    measure_dark: bool = True,
    col_ratio: np.ndarray | None = None,
    frac: float | None = None,
    single_beam_bg: bool = False,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PhaseResult:
    """Drive target + reference over ``drive``, read Y at each point, fit dPhi_comb.

    ``monitor`` must already be configured (caller runs ``configure_monitor``);
    this only calls ``monitor_cycle``.  Only channels ``tgt_index`` and
    ``ref_index`` are driven; all others held off.  ``drive`` tuples are
    target-first ``(x_t, w_t, x_r, w_r)`` intensities.  Each point is read
    once over the DAQ's fixed window (T_both for the sweep points -- the
    reference is fully on, so they are bright).

    Dark handling: with ``measure_dark`` (default) a single all-off reading is
    taken at the start, over the DAQ's longer T_single window, and stored per
    row for per-row subtraction.  If ``measure_dark`` is False the mean of the
    two step-6 darks is used for every row.  Raises :class:`TPAPhaseAborted`
    if ``stop_event`` is set.

    ``frac``/``single_beam_bg`` are forwarded to :func:`.tpa_phase.fit_result`:
    ``frac=None`` (default) keeps the unconstrained closed-form fit; a number
    locks ``a:b`` to the step-6 ``eta_ref:eta_tgt`` ratio and floats a shared
    scale boxed to ``+/- frac`` (``frac=0`` pins ``a``/``b`` to the step-6 etas
    exactly).  ``single_beam_bg`` additionally folds in both pairs' step-6
    single-beam response as a fixed background.
    """
    n = layout.n_channels
    for name, idx in (("tgt_index", tgt_index), ("ref_index", ref_index)):
        if not (0 <= idx < n):
            raise ValueError(f"{name}={idx} out of range (layout has {n} pairs)")
    if tgt_index == ref_index:
        raise ValueError("tgt_index and ref_index must differ")

    zeros = np.zeros(n)
    slm_width, slm_height = slm.get_slm_info()
    from .encoding import encode_to_pattern

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise TPAPhaseAborted("phase sweep stopped by request")

    def _display(x_t, w_t, x_r, w_r) -> None:
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        x_vals[tgt_index], w_vals[tgt_index] = x_t, w_t
        x_vals[ref_index], w_vals[ref_index] = x_r, w_r
        pattern = encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height,
                                    col_ratio=col_ratio)
        slm.display_array(pattern)
        if settle:
            time.sleep(settle)

    # dark handling: step-6 mean is the fallback; a measured all-off reading
    # (taken once at the start, over the T_single window) overrides it
    fallback_dark = 0.5 * (tgt_model.d + ref_model.d)

    drive = list(drive)
    total = max(len(drive) + (1 if measure_dark else 0), 1)

    step = 0
    dark = fallback_dark
    if measure_dark:
        _check_stop()
        _display(0.0, 0.0, 0.0, 0.0)
        dark, _, _ = _read_mean_std(monitor, read_timeout, single=True)
        step += 1
        if progress_callback is not None:
            progress_callback(TPAPhaseProgress(
                step=step, total=total,
                message=f"dark (all off) = {dark*1000:.4f} mV"))

    rows: list[tuple] = []
    for x_t, w_t, x_r, w_r in drive:
        _check_stop()
        _display(x_t, w_t, x_r, w_r)
        mean_v, std_v, sem_v = _read_mean_std(monitor, read_timeout)
        rows.append((0, x_t, w_t, x_r, w_r, mean_v, std_v, sem_v, dark))
        step += 1
        if progress_callback is not None:
            dphi_slm = float(slm_phase_diff(x_t, w_t, x_r, w_r))
            phi_t = float(np.degrees(2.0 * phi_half(x_t)))
            progress_callback(
                TPAPhaseProgress(
                    step=step, total=total,
                    message=(
                        f"phi_t={phi_t:.1f}deg "
                        f"dPhi_SLM={np.degrees(dphi_slm):+.1f}deg "
                        f"-> {mean_v*1000:.4f} mV (dark {dark*1000:.4f})"
                    ),
                )
            )

    result = PhaseResult(
        tgt_index=tgt_index, ref_index=ref_index,
        trial=np.array([r[0] for r in rows], dtype=int),
        x_t=np.array([r[1] for r in rows], dtype=float),
        w_t=np.array([r[2] for r in rows], dtype=float),
        x_r=np.array([r[3] for r in rows], dtype=float),
        w_r=np.array([r[4] for r in rows], dtype=float),
        voltage_mean_v=np.array([r[5] for r in rows], dtype=float),
        voltage_std_v=np.array([r[6] for r in rows], dtype=float),
        voltage_sem_v=np.array([r[7] for r in rows], dtype=float),
        dark_v=np.array([r[8] for r in rows], dtype=float),
        n_trials=1,
    )
    fit_result(result, tgt_model, ref_model, frac=frac, single_beam_bg=single_beam_bg)
    if progress_callback is not None and result.fit is not None:
        progress_callback(
            TPAPhaseProgress(
                step=total, total=total,
                message=(
                    f"fit: dPhi_comb = {np.degrees(result.fit.dphi_comb):+.2f} deg "
                    f"(a = {result.fit.a:.4g}, b = {result.fit.b:.4g}"
                    + (", a@bound" if result.fit.a_at_bound else "")
                    + (", b@bound" if result.fit.b_at_bound else "") + ")"
                ),
                dphi_comb=result.fit.dphi_comb,
            )
        )
    return result


__all__ = [
    "TPAPhaseAborted",
    "TPAPhaseProgress",
    "ProgressCallback",
    "build_phase_sweep",
    "build_symmetry_grid",
    "measure_phase_sweep",
]
