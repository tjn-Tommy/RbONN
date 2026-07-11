"""TPA centre calibration by scanning the layout centre wavelength.

The TPA pair-grid sweep in :mod:`slm_module.tpa_pair` assumes the encoder
layout is already centred on the true two-photon resonance. This module adds a
lighter-weight 1-D scan: rebuild the symmetric x/w layout at a list of centre
wavelengths, turn on one pair at a fixed drive level, read the fluorescence
brightness from the active monitor, then fit the resulting peak with a weighted
quadratic.
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration.calibration_new import CalibrationResult


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


@dataclass
class TPACenterFit:
    center_wl_nm: float
    center_wl_err_nm: float
    peak_signal_v: float
    peak_signal_err_v: float
    coeffs: tuple[float, float, float]
    coeff_errs: tuple[float, float, float]
    chi2_red: float
    dof: int
    birge: float
    best_sample_center_wl_nm: float
    best_sample_signal_v: float
    valid: bool
    message: str
    center_wl: np.ndarray = field(repr=False)
    signal_v: np.ndarray = field(repr=False)
    sem_v: np.ndarray = field(repr=False)
    signal_pred_v: np.ndarray = field(repr=False)
    residuals_v: np.ndarray = field(repr=False)


@dataclass
class TPACenterResult:
    center_wl_nm: np.ndarray = field(repr=False)
    center_x_px: np.ndarray = field(repr=False)
    trial: np.ndarray = field(repr=False)
    signal_v: np.ndarray = field(repr=False)
    signal_std_v: np.ndarray = field(repr=False)
    background_v: np.ndarray = field(repr=False)
    background_std_v: np.ndarray = field(repr=False)
    net_signal_v: np.ndarray = field(repr=False)
    fit: TPACenterFit | None = None
    pair_index: int = 0
    drive_level: float = 1.0
    n_trials: int = 1
    repeats: int = 1
    subtract_background: bool = False


def average_trace_points(
    center_wl_nm: np.ndarray,
    signal_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average repeated readings at each scanned centre wavelength."""
    grouped: dict[float, list[float]] = defaultdict(list)
    for wl, signal in zip(np.asarray(center_wl_nm, dtype=float), np.asarray(signal_v, dtype=float)):
        grouped[float(wl)].append(float(signal))

    wl_out: list[float] = []
    mean_out: list[float] = []
    sem_out: list[float] = []
    for wl in sorted(grouped):
        arr = np.asarray(grouped[wl], dtype=float)
        wl_out.append(wl)
        mean_out.append(float(arr.mean()))
        sem_out.append(
            float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
        )

    wl_arr = np.asarray(wl_out, dtype=float)
    mean_arr = np.asarray(mean_out, dtype=float)
    sem_arr = np.asarray(sem_out, dtype=float)
    finite = sem_arr[np.isfinite(sem_arr) & (sem_arr > 0.0)]
    if finite.size:
        floor = float(np.median(finite))
    else:
        floor = float(np.std(mean_arr, ddof=1)) if mean_arr.size > 1 else abs(float(mean_arr[0]))
        floor = max(floor, 1e-12)
    sem_arr = np.where(np.isfinite(sem_arr) & (sem_arr > 0.0), sem_arr, floor)
    return wl_arr, mean_arr, sem_arr


def fit_center_trace(
    center_wl_nm: np.ndarray,
    signal_v: np.ndarray,
    sem_v: np.ndarray,
) -> TPACenterFit:
    """Weighted quadratic fit of brightness vs centre wavelength."""
    wl = np.asarray(center_wl_nm, dtype=float)
    signal = np.asarray(signal_v, dtype=float)
    sem = np.asarray(sem_v, dtype=float)
    if wl.ndim != 1 or signal.ndim != 1 or sem.ndim != 1:
        raise ValueError("centre-trace arrays must be 1-D")
    if wl.size != signal.size or wl.size != sem.size:
        raise ValueError("centre-trace arrays must have matching lengths")
    if wl.size < 3:
        raise ValueError("need at least three centre points for a quadratic fit")

    A = np.column_stack([wl**2, wl, np.ones_like(wl)])
    Aw = A / sem[:, None]
    coeffs, *_ = np.linalg.lstsq(Aw, signal / sem, rcond=None)
    cov = np.linalg.pinv(Aw.T @ Aw)

    pred = A @ coeffs
    residuals = signal - pred
    dof = max(len(signal) - 3, 1)
    chi2_red = float(np.sum((residuals / sem) ** 2) / dof)
    birge = max(1.0, float(np.sqrt(chi2_red)))
    cov_scaled = cov * (birge**2)
    coeff_errs = tuple(float(v) for v in np.sqrt(np.diag(cov_scaled)))

    a, b, c = (float(v) for v in coeffs)
    best_idx = int(np.nanargmax(signal))
    best_sample_center = float(wl[best_idx])
    best_sample_signal = float(signal[best_idx])

    center = float("nan")
    center_err = float("nan")
    peak = float("nan")
    peak_err = float("nan")
    valid = False
    message = "quadratic fit is invalid"
    if np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and abs(a) > 0.0:
        center = float(-b / (2.0 * a))
        peak = float(c - (b * b) / (4.0 * a))
        grad_center = np.array([b / (2.0 * a * a), -1.0 / (2.0 * a), 0.0], dtype=float)
        grad_peak = np.array([b * b / (4.0 * a * a), -b / (2.0 * a), 1.0], dtype=float)
        center_var = float(grad_center @ cov_scaled @ grad_center)
        peak_var = float(grad_peak @ cov_scaled @ grad_peak)
        center_err = float(np.sqrt(center_var)) if center_var >= 0.0 else float("nan")
        peak_err = float(np.sqrt(peak_var)) if peak_var >= 0.0 else float("nan")
        if a >= 0.0:
            message = "fit is convex; no local maximum in the scanned window"
        elif center < float(np.min(wl)) or center > float(np.max(wl)):
            message = "fit peak lies outside the scanned wavelength range"
        else:
            valid = True
            message = "ok"

    return TPACenterFit(
        center_wl_nm=center,
        center_wl_err_nm=center_err,
        peak_signal_v=peak,
        peak_signal_err_v=peak_err,
        coeffs=(a, b, c),
        coeff_errs=coeff_errs,
        chi2_red=chi2_red,
        dof=dof,
        birge=birge,
        best_sample_center_wl_nm=best_sample_center,
        best_sample_signal_v=best_sample_signal,
        valid=valid,
        message=message,
        center_wl=wl,
        signal_v=signal,
        sem_v=sem,
        signal_pred_v=pred,
        residuals_v=residuals,
    )


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

    from .encoding import build_channel_layout, encode_to_pattern

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


_SCHEMA = "tpa_center_result_v1"


def save_tpa_center_json(result: TPACenterResult, path: str | Path) -> str:
    """Persist a centre scan (raw rows + fit + scan config) as JSON."""
    fit = result.fit
    fit_payload = None
    if fit is not None:
        fit_payload = {
            "center_wl_nm": fit.center_wl_nm,
            "center_wl_err_nm": fit.center_wl_err_nm,
            "peak_signal_v": fit.peak_signal_v,
            "peak_signal_err_v": fit.peak_signal_err_v,
            "coeffs": list(fit.coeffs),
            "coeff_errs": list(fit.coeff_errs),
            "chi2_red": fit.chi2_red,
            "dof": fit.dof,
            "birge": fit.birge,
            "best_sample_center_wl_nm": fit.best_sample_center_wl_nm,
            "best_sample_signal_v": fit.best_sample_signal_v,
            "valid": fit.valid,
            "message": fit.message,
            "center_wl": fit.center_wl.tolist(),
            "signal_v": fit.signal_v.tolist(),
            "sem_v": fit.sem_v.tolist(),
            "signal_pred_v": fit.signal_pred_v.tolist(),
            "residuals_v": fit.residuals_v.tolist(),
        }
    payload = {
        "schema": _SCHEMA,
        "pair_index": result.pair_index,
        "drive_level": result.drive_level,
        "n_trials": result.n_trials,
        "repeats": result.repeats,
        "subtract_background": result.subtract_background,
        "center_wl_nm": result.center_wl_nm.tolist(),
        "center_x_px": result.center_x_px.tolist(),
        "trial": result.trial.tolist(),
        "signal_v": result.signal_v.tolist(),
        "signal_std_v": result.signal_std_v.tolist(),
        "background_v": result.background_v.tolist(),
        "background_std_v": result.background_std_v.tolist(),
        "net_signal_v": result.net_signal_v.tolist(),
        "fit": fit_payload,
    }
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


def load_tpa_center_json(path: str | Path) -> TPACenterResult:
    """Rebuild a :class:`TPACenterResult` saved by :func:`save_tpa_center_json`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != _SCHEMA:
        raise ValueError(
            f"{path}: expected schema {_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    fit = None
    fp = payload.get("fit")
    if fp is not None:
        fit = TPACenterFit(
            center_wl_nm=float(fp["center_wl_nm"]),
            center_wl_err_nm=float(fp["center_wl_err_nm"]),
            peak_signal_v=float(fp["peak_signal_v"]),
            peak_signal_err_v=float(fp["peak_signal_err_v"]),
            coeffs=tuple(float(v) for v in fp["coeffs"]),
            coeff_errs=tuple(float(v) for v in fp["coeff_errs"]),
            chi2_red=float(fp["chi2_red"]),
            dof=int(fp["dof"]),
            birge=float(fp["birge"]),
            best_sample_center_wl_nm=float(fp["best_sample_center_wl_nm"]),
            best_sample_signal_v=float(fp["best_sample_signal_v"]),
            valid=bool(fp["valid"]),
            message=str(fp["message"]),
            center_wl=np.asarray(fp["center_wl"], dtype=float),
            signal_v=np.asarray(fp["signal_v"], dtype=float),
            sem_v=np.asarray(fp["sem_v"], dtype=float),
            signal_pred_v=np.asarray(fp["signal_pred_v"], dtype=float),
            residuals_v=np.asarray(fp["residuals_v"], dtype=float),
        )
    return TPACenterResult(
        center_wl_nm=np.asarray(payload["center_wl_nm"], dtype=float),
        center_x_px=np.asarray(payload["center_x_px"], dtype=float),
        trial=np.asarray(payload["trial"], dtype=int),
        signal_v=np.asarray(payload["signal_v"], dtype=float),
        signal_std_v=np.asarray(payload["signal_std_v"], dtype=float),
        background_v=np.asarray(payload["background_v"], dtype=float),
        background_std_v=np.asarray(payload["background_std_v"], dtype=float),
        net_signal_v=np.asarray(payload["net_signal_v"], dtype=float),
        fit=fit,
        pair_index=int(payload["pair_index"]),
        drive_level=float(payload["drive_level"]),
        n_trials=int(payload["n_trials"]),
        repeats=int(payload["repeats"]),
        subtract_background=bool(payload["subtract_background"]),
    )


__all__ = [
    "TPACenterAborted",
    "TPACenterProgress",
    "TPACenterFit",
    "TPACenterResult",
    "average_trace_points",
    "fit_center_trace",
    "load_tpa_center_json",
    "measure_center_scan",
    "save_tpa_center_json",
]
