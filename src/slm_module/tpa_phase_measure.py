"""Step-7 measurement driver: two-pair phase sweep against a scope/DAQ monitor.

The fitting/IO half of step 7 lives in :mod:`.tpa_phase` and is geometry
general -- it recovers ``dPhi_comb`` from whatever drive produced the raw
rows.  This module owns the instrument-facing half the unified pipeline
needs: building a drive table, walking it on the SLM while reading the
monitor, and handing the collected rows to :func:`.tpa_phase.fit_result`.
It is kept OUT of ``tpa_phase.py`` so that module stays in lock-step with
the offline driver ``src/drafts/calib_step7_test.py`` (which does its own
SLM/DAQ wiring).

Readings follow the step-6 convention: per point the monitor is read
``repeats`` times; ``std`` is the raw low-passed trace spread (diagnostic)
and ``sem`` the instrument-reported standard error of the mean when the
sample carries one (DAQ), falling back to the trace std -- ``sem`` is what
the fit weights by (see :func:`.tpa_phase._average_points`).
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
    ``dPhi_SLM = phi - ref_phase`` sweeps the fringe.  The step-7 fit is
    geometry-general, so this drive and the draft's w-only drive
    (:func:`build_w_only_sweep`) fit identically.
    """
    phis = np.radians(np.linspace(phi_start_deg, phi_stop_deg, int(n_points)))
    x_r = float(intensity_for_phase(np.radians(ref_phase_deg)))
    x_t = intensity_for_phase(phis)
    return [(float(v), float(v), x_r, x_r) for v in x_t]


def build_w_only_sweep(
    w_values: Sequence[float],
) -> list[tuple[float, float, float, float]]:
    """The draft's drive: reference + target-x fully on, only ``w_t`` swept.

    Mirrors ``build_w2_sweep`` in ``src/drafts/calib_step7_test.py`` so a
    pipeline run can reproduce the draft's sweep exactly: target-first tuples
    ``(1, w_t, 1, 1)`` with ``w_t`` stepping over ``w_values``.
    """
    return [(1.0, float(w), 1.0, 1.0) for w in w_values]


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
    monitor, repeats: int, timeout: float
) -> tuple[float, float, float]:
    """Averaged reading, its raw trace std, and the per-point SEM of the mean.

    Raises :class:`TPAPhaseAborted` when the monitor read is stopped.  ``sem``
    is the instrument-reported standard error (low-passed, effective-N) when
    the sample carries one, else the trace std -- matching the draft's
    ``_read_daq`` so pipeline and offline runs weight their fits the same way.
    """
    means: list[float] = []
    std_vars: list[float] = []
    sem_vars: list[float] = []
    for _ in range(max(1, repeats)):
        sample = monitor.monitor_cycle(timeout=timeout)
        if sample is None:
            raise TPAPhaseAborted("monitor read aborted")
        means.append(float(sample.value))
        std = getattr(sample, "std", None)
        waveform = getattr(monitor, "last_values", None)
        raw_std = (
            float(std) if std is not None and np.isfinite(std)
            else (
                float(np.std(waveform))
                if waveform is not None and np.size(waveform) > 1
                else 0.0
            )
        )
        std_vars.append(raw_std ** 2)
        sem = getattr(sample, "sem", None)
        sem_vars.append(
            float(sem) ** 2 if sem is not None and np.isfinite(sem)
            else raw_std ** 2
        )
    mean_v = float(np.mean(means))
    std_v = float(np.sqrt(np.mean(std_vars))) if std_vars else 0.0
    sem_v = float(np.sqrt(np.mean(sem_vars))) if sem_vars else 0.0
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
    n_trials: int = 1,
    repeats: int = 1,
    settle: float = 0.15,
    read_timeout: float = 30.0,
    measure_dark: bool = True,
    dark_per_trial: bool = True,
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
    target-first ``(x_t, w_t, x_r, w_r)`` intensities.

    Dark handling: with ``measure_dark`` an all-off reading is taken and stored
    per row for per-row subtraction (drift removal).  ``dark_per_trial`` (default)
    takes a fresh all-off reading at the START OF EACH TRIAL, so slow dark drift
    over the run is tracked; set it False to take a single all-off reading once at
    the start.  If ``measure_dark`` is False the mean of the two step-6 darks is
    used for every row.  Raises :class:`TPAPhaseAborted` if ``stop_event`` is set.

    ``frac``/``single_beam_bg`` are forwarded to :func:`.tpa_phase.fit_result`:
    ``frac=None`` (default) keeps the unconstrained closed-form fit; a number
    locks ``a:b`` to the step-6 ``eta_ref:eta_tgt`` ratio and floats a shared
    scale boxed to ``+/- frac``.  ``single_beam_bg`` additionally folds in both
    pairs' step-6 single-beam response as a fixed background.
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
    # overrides it (once, or per trial for drift tracking)
    fallback_dark = 0.5 * (tgt_model.d + ref_model.d)

    def _read_dark(trial: int, step: int, total: int) -> float:
        _check_stop()
        _display(0.0, 0.0, 0.0, 0.0)
        d, _, _ = _read_mean_std(monitor, repeats, read_timeout)
        if progress_callback is not None:
            progress_callback(TPAPhaseProgress(
                step=step, total=total,
                message=f"trial {trial} dark (all off) = {d*1000:.4f} mV"))
        return d

    drive = list(drive)
    reads_per_trial = len(drive) + (1 if measure_dark and dark_per_trial else 0)
    total = max(
        n_trials * reads_per_trial
        + (1 if measure_dark and not dark_per_trial else 0),
        1,
    )

    start_dark = fallback_dark
    step = 0
    if measure_dark and not dark_per_trial:
        step += 1
        start_dark = _read_dark(0, step, total)

    rows: list[tuple] = []
    for trial in range(n_trials):
        if measure_dark and dark_per_trial:
            step += 1
            trial_dark = _read_dark(trial, step, total)
        elif measure_dark:
            trial_dark = start_dark
        else:
            trial_dark = fallback_dark
        for x_t, w_t, x_r, w_r in drive:
            _check_stop()
            _display(x_t, w_t, x_r, w_r)
            mean_v, std_v, sem_v = _read_mean_std(monitor, repeats, read_timeout)
            rows.append(
                (trial, x_t, w_t, x_r, w_r, mean_v, std_v, sem_v, trial_dark)
            )
            step += 1
            if progress_callback is not None:
                dphi_slm = float(slm_phase_diff(x_t, w_t, x_r, w_r))
                phi_t = float(np.degrees(2.0 * phi_half(x_t)))
                progress_callback(
                    TPAPhaseProgress(
                        step=step, total=total,
                        message=(
                            f"trial {trial} phi_t={phi_t:.1f}deg "
                            f"dPhi_SLM={np.degrees(dphi_slm):+.1f}deg "
                            f"-> {mean_v*1000:.4f} mV (dark {trial_dark*1000:.4f})"
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
        n_trials=n_trials,
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
    "build_w_only_sweep",
    "measure_phase_sweep",
]
