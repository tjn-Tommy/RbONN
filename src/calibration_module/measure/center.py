"""Centre-wavelength scan driver (SLM + monitor).

LEGACY -- kept only so the current GUI page and pipeline stage keep running.

This is the in-app sweep driver: it talks to BOTH the SLM and the monitor, so
it is orchestration, not physics.  The offline step scripts under ``calibration_module/steps`` do
not use it -- they compose the SLM and DAQ calls themselves, which is the
pattern the rebuilt calibration pipeline should follow.  Nothing new should
import from here; delete this module once the GUI centre-scan page / ``pipeline._run_tpa_center`` is rebuilt.

The physics lives in :mod:`calibration_module.fit.center`.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from calibration_module.fit.slm_response import CalibrationResult

from ..fit.center import (
    TPACenterResult,
    average_trace_points,
    fit_center_trace,
)


class TPACenterAborted(Exception):
    """Raised when a centre scan is interrupted by a stop request."""


@dataclass
class TPACenterProgress:
    step: int
    total: int
    message: str
    center_wl_nm: float | None = None
    signal_v: float | None = None


ProgressCallback = Callable[[TPACenterProgress], None]


def _read_mean_std(monitor, repeats: int, timeout: float) -> tuple[float, float]:
    means: list[float] = []
    variances: list[float] = []
    for _ in range(max(1, int(repeats))):
        sample = monitor.monitor_cycle(timeout=timeout)
        if sample is None:
            raise TPACenterAborted("monitor read aborted")
        means.append(float(sample.value))
        waveform = getattr(monitor, "last_values", None)
        if waveform is not None and np.size(waveform) > 1:
            variances.append(float(np.var(waveform)))
        elif getattr(sample, "std", None) is not None:
            variances.append(float(sample.std) ** 2)
    mean_v = float(np.mean(means))
    std_v = float(np.sqrt(np.mean(variances))) if variances else 0.0
    return mean_v, std_v


def measure_center_scan(
    monitor,
    slm,
    calibration: CalibrationResult,
    *,
    center_wavelengths_nm: Sequence[float],
    n_channels: int,
    channel_width_px: int,
    gap_px: int,
    center_gap_px: int | None = None,
    pair_index: int = 0,
    drive_level: float = 1.0,
    n_trials: int = 1,
    repeats: int = 1,
    settle: float = 0.15,
    read_timeout: float = 30.0,
    col_ratio: np.ndarray | None = None,
    subtract_background: bool = True,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TPACenterResult:
    """Scan centre wavelength and fit the monitor brightness peak."""
    if calibration.intensity_levels is None:
        raise ValueError("centre scan requires a Step 3 intensity calibration")
    centers = np.asarray(list(center_wavelengths_nm), dtype=float)
    if centers.ndim != 1 or centers.size < 3:
        raise ValueError("centre scan needs at least three wavelength points")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centre wavelength list contains NaN or infinity")
    if float(np.min(centers)) == float(np.max(centers)):
        raise ValueError("centre wavelength range must not collapse to one value")
    if pair_index < 0:
        raise ValueError("pair_index must be non-negative")

    from slm_module.encoding import build_channel_layout, encode_to_pattern

    slm_width, slm_height = slm.get_slm_info()

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise TPACenterAborted("centre scan stopped by request")

    total_reads = max(
        int(n_trials) * int(centers.size) * (2 if subtract_background else 1),
        1,
    )
    step = 0

    rows: list[tuple[int, float, float, float, float, float, float, float]] = []
    for trial in range(int(n_trials)):
        for center_wl in centers:
            _check_stop()
            layout = build_channel_layout(
                calibration,
                n_channels=int(n_channels),
                channel_width_px=int(channel_width_px),
                gap_px=int(gap_px),
                center_gap_px=center_gap_px,
                center_wl=float(center_wl),
            )
            if pair_index >= layout.n_channels:
                raise ValueError(
                    f"pair {pair_index} is out of range for centre {center_wl:.4f} nm "
                    f"(layout has {layout.n_channels} pairs)"
                )

            zeros = np.zeros(layout.n_channels, dtype=float)
            x_vals = zeros.copy()
            w_vals = zeros.copy()
            x_vals[pair_index] = float(drive_level)
            w_vals[pair_index] = float(drive_level)

            bg_mean = float("nan")
            bg_std = float("nan")
            if subtract_background:
                bg_pattern = encode_to_pattern(
                    zeros, zeros, layout, slm_width, slm_height, col_ratio=col_ratio
                )
                slm.display_array(bg_pattern)
                if settle:
                    time.sleep(settle)
                bg_mean, bg_std = _read_mean_std(monitor, repeats, read_timeout)
                step += 1
                if progress_callback is not None:
                    progress_callback(
                        TPACenterProgress(
                            step=step,
                            total=total_reads,
                            message=(
                                f"trial {trial} centre {center_wl:.4f} nm background "
                                f"-> {bg_mean * 1000:.4f} mV"
                            ),
                            center_wl_nm=float(center_wl),
                            signal_v=bg_mean,
                        )
                    )

            pattern = encode_to_pattern(
                x_vals, w_vals, layout, slm_width, slm_height, col_ratio=col_ratio
            )
            slm.display_array(pattern)
            if settle:
                time.sleep(settle)
            signal_mean, signal_std = _read_mean_std(monitor, repeats, read_timeout)
            net_signal = float(signal_mean - bg_mean) if subtract_background else float(signal_mean)
            rows.append(
                (
                    int(trial),
                    float(center_wl),
                    float(layout.center_x),
                    float(signal_mean),
                    float(signal_std),
                    float(bg_mean),
                    float(bg_std),
                    net_signal,
                )
            )
            step += 1
            if progress_callback is not None:
                progress_callback(
                    TPACenterProgress(
                        step=step,
                        total=total_reads,
                        message=(
                            f"trial {trial} centre {center_wl:.4f} nm "
                            f"pair[{pair_index}] -> {net_signal * 1000:.4f} mV"
                        ),
                        center_wl_nm=float(center_wl),
                        signal_v=net_signal,
                    )
                )

    result = TPACenterResult(
        center_wl_nm=np.array([row[1] for row in rows], dtype=float),
        center_x_px=np.array([row[2] for row in rows], dtype=float),
        trial=np.array([row[0] for row in rows], dtype=int),
        signal_v=np.array([row[3] for row in rows], dtype=float),
        signal_std_v=np.array([row[4] for row in rows], dtype=float),
        background_v=np.array([row[5] for row in rows], dtype=float),
        background_std_v=np.array([row[6] for row in rows], dtype=float),
        net_signal_v=np.array([row[7] for row in rows], dtype=float),
        pair_index=int(pair_index),
        drive_level=float(drive_level),
        n_trials=int(n_trials),
        repeats=int(repeats),
        subtract_background=bool(subtract_background),
    )
    fit_wl, fit_signal, fit_sem = average_trace_points(result.center_wl_nm, result.net_signal_v)
    result.fit = fit_center_trace(fit_wl, fit_signal, fit_sem)
    return result


__all__ = [
    "TPACenterAborted",
    "TPACenterProgress",
    "ProgressCallback",
    "measure_center_scan",
]
