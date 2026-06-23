"""Modulation error analysis (B1): isolated single-channel spectral shape.

For each encoding channel on the (778 nm-centred) grid, turn that channel on
and all others off, sweep the OSA across the whole spectrum, and quantify how
far the measured single-channel lineshape is from an ideal rectangular passband.

Metrics per channel:
  - peak_wl_nm       : wavelength of maximum measured power
  - fwhm_nm          : full width at half maximum of the lineshape
  - in_band_fraction : power inside the nominal band / total channel power
  - neighbor_leakage : power inside the two adjacent nominal bands / total
                       (an upper-bound estimate of nearest-neighbour crosstalk;
                        see notes on OSA resolution in the module discussion)

The raw spectra are stored so deeper, data-driven analysis (e.g. per-channel
bandwidth, deconvolution against the OSA slit) can be done later.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from osa_module.controller import MeasurementSettings, OSAController, TraceData

from .controller import SLMController
from .encoding import ChannelLayout, encode_to_pattern

# numpy 2.0 renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", None) or np.trapz


class AnalysisAborted(Exception):
    """Raised when a stop_event interrupts a modulation-error sweep."""


@dataclass
class AnalysisProgress:
    step: int
    total: int
    message: str
    wl: float | None = None        # x for a live plot (channel wavelength)
    metric: float | None = None    # y for a live plot (in-band fraction)


ProgressCallback = Callable[["AnalysisProgress"], None]


@dataclass
class ChannelSpectrum:
    index: int
    side: str                       # 'x' or 'w'
    x_center: int
    nominal_wl_nm: float
    nominal_bw_nm: float
    wavelengths_nm: np.ndarray = field(repr=False)
    signal_w: np.ndarray = field(repr=False)
    peak_wl_nm: float = 0.0
    fwhm_nm: float = 0.0
    total_power_w: float = 0.0
    in_band_fraction: float = 0.0
    neighbor_leakage: float = 0.0    # sum of +/-1 neighbour bands / total
    # fraction of this channel's power landing in offset neighbour bands
    crosstalk: dict[int, float] = field(default_factory=dict)


@dataclass
class ModulationErrorResult:
    channels: list[ChannelSpectrum]
    center_wl: float
    channel_width_px: int
    pitch_px: int
    nm_per_px: float
    raw_npz_path: str | None = None    # consolidated raw spectra written after the sweep


def _trace_power_w(trace: TraceData) -> np.ndarray:
    powers = np.asarray(trace.powers, dtype=float)
    if trace.power_label == "power_dBm":
        powers = 1e-3 * (10.0 ** (powers / 10.0))
    return np.nan_to_num(powers, nan=0.0, posinf=0.0, neginf=0.0)


def _band_power(wl: np.ndarray, sig: np.ndarray, lo: float, hi: float) -> float:
    """Trapezoidal integral of sig over the wavelength band [lo, hi]."""
    mask = (wl >= lo) & (wl <= hi)
    if mask.sum() < 2:
        return 0.0
    return float(_trapz(sig[mask], wl[mask]))


def _fwhm(wl: np.ndarray, sig: np.ndarray, peak_idx: int) -> float:
    """Full width at half maximum around peak_idx, by linear interpolation."""
    half = sig[peak_idx] / 2.0
    if half <= 0:
        return 0.0

    # walk left to the first crossing below half
    left = peak_idx
    while left > 0 and sig[left] > half:
        left -= 1
    if sig[left] >= half:
        wl_left = wl[left]
    else:
        f = (half - sig[left]) / (sig[left + 1] - sig[left])
        wl_left = wl[left] + f * (wl[left + 1] - wl[left])

    right = peak_idx
    n = sig.size
    while right < n - 1 and sig[right] > half:
        right += 1
    if sig[right] >= half:
        wl_right = wl[right]
    else:
        f = (half - sig[right]) / (sig[right - 1] - sig[right])
        wl_right = wl[right] + f * (wl[right - 1] - wl[right])

    return abs(float(wl_right - wl_left))


# neighbour band offsets (in channel pitches) recorded for crosstalk
_NEIGHBOR_OFFSETS = (-2, -1, 1, 2)


def _channel_metrics(
    wl: np.ndarray,
    sig: np.ndarray,
    nominal_center: float,
    nominal_bw: float,
    pitch_nm: float,
) -> tuple[float, float, float, float, dict[int, float]]:
    """Return (peak_wl, fwhm_nm, total_power, in_band_fraction, crosstalk).

    crosstalk maps neighbour offset (+/-1, +/-2 pitches) -> fraction of this
    channel's total power that falls within that neighbour's nominal band.
    """
    empty = (0.0, 0.0, 0.0, 0.0, {o: 0.0 for o in _NEIGHBOR_OFFSETS})
    if wl.size == 0 or sig.size == 0:
        return empty

    # locate the peak within +/- one pitch of the nominal centre
    local = (wl >= nominal_center - pitch_nm) & (wl <= nominal_center + pitch_nm)
    if local.any():
        local_idx = np.where(local)[0]
        peak_idx = int(local_idx[np.argmax(sig[local_idx])])
    else:
        peak_idx = int(np.argmax(sig))
    peak_wl = float(wl[peak_idx])
    fwhm = _fwhm(wl, sig, peak_idx)

    half_bw = nominal_bw / 2.0
    total = float(_trapz(np.clip(sig, 0.0, None), wl))
    if total <= 0:
        return peak_wl, fwhm, 0.0, 0.0, {o: 0.0 for o in _NEIGHBOR_OFFSETS}

    in_band = _band_power(wl, sig, nominal_center - half_bw, nominal_center + half_bw)
    crosstalk = {}
    for offset in _NEIGHBOR_OFFSETS:
        c = nominal_center + offset * pitch_nm
        crosstalk[offset] = _band_power(wl, sig, c - half_bw, c + half_bw) / total
    return peak_wl, fwhm, total, in_band / total, crosstalk


def measure_channel_spectra(
    osa: OSAController,
    slm: SLMController,
    layout: ChannelLayout,
    settings: MeasurementSettings,
    *,
    averages: int = 1,
    stride: int = 1,
    subtract_background: bool = True,
    capture_dir: str | Path | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ModulationErrorResult:
    """Sweep every (strided) channel on the grid, isolated, and measure the OSA.

    For each channel the OSA is re-centred on that channel's estimated
    wavelength (the ``settings`` span is kept, e.g. 0.8 nm) so the channel sits
    in the middle of a narrow, high-resolution window. When
    ``subtract_background`` is set, an all-off trace is taken at the same centre
    just before the channel-on trace and subtracted. ``stride`` measures only
    every Nth channel index per side.

    When ``capture_dir`` is given, each channel's raw spectrum is written there
    immediately after measurement (crash-safe), and a consolidated ``spectra.npz``
    is built in that directory once the sweep finishes.
    """
    capture_path = Path(capture_dir) if capture_dir is not None else None
    if capture_path is not None:
        capture_path.mkdir(parents=True, exist_ok=True)

    slm_width, slm_height = slm.get_slm_info()
    n = layout.n_channels
    nominal_bw = layout.channel_width_px * layout.nm_per_px
    pitch_nm = layout.pitch_px * layout.nm_per_px
    zeros = np.zeros(n)

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise AnalysisAborted("analysis stopped by request")

    bg_pattern = encode_to_pattern(zeros, zeros, layout, slm_width, slm_height)

    indices = list(range(0, n, max(1, stride)))
    targets: list[tuple[int, str]] = [(i, "x") for i in indices] + [
        (i, "w") for i in indices
    ]
    total = len(targets)

    spectra: list[ChannelSpectrum] = []
    for step, (i, side) in enumerate(targets):
        _check_stop()
        channel = (layout.x_channels if side == "x" else layout.w_channels)[i]

        # re-centre the OSA on this channel; keep the (narrow) span from settings
        ch_settings = replace(settings, center_wl=f"{channel.wavelength_nm:.4f}nm")

        # per-window background (all channels off) at this OSA centre
        bg_power = None
        if subtract_background:
            slm.display_array(bg_pattern)
            bg_trace = osa.measure(ch_settings, averages=averages, stop_event=stop_event)
            bg_power = _trace_power_w(bg_trace)

        x_vals = zeros.copy()
        w_vals = zeros.copy()
        if side == "x":
            x_vals[i] = 1.0
        else:
            w_vals[i] = 1.0
        pattern = encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height)
        slm.display_array(pattern)

        trace = osa.measure(ch_settings, averages=averages, stop_event=stop_event)
        power = _trace_power_w(trace)
        wl = trace.wavelengths_nm
        if bg_power is not None:
            count = min(wl.size, power.size, bg_power.size)
            signal = np.clip(power[:count] - bg_power[:count], 0.0, None)
            wl = wl[:count]
        else:
            signal = np.clip(power, 0.0, None)

        peak_wl, fwhm, total_p, in_band, crosstalk = _channel_metrics(
            wl, signal, channel.wavelength_nm, nominal_bw, pitch_nm
        )
        leak = crosstalk.get(-1, 0.0) + crosstalk.get(1, 0.0)
        spectrum = ChannelSpectrum(
            index=i,
            side=side,
            x_center=channel.x_center,
            nominal_wl_nm=channel.wavelength_nm,
            nominal_bw_nm=nominal_bw,
            wavelengths_nm=wl,
            signal_w=signal,
            peak_wl_nm=peak_wl,
            fwhm_nm=fwhm,
            total_power_w=total_p,
            in_band_fraction=in_band,
            neighbor_leakage=leak,
            crosstalk=crosstalk,
        )
        spectra.append(spectrum)

        # crash-safe incremental save of this capture's raw spectrum
        if capture_path is not None:
            np.savez(
                capture_path / f"{side}{i:03d}.npz",
                wavelengths_nm=wl,
                signal_w=signal,
                nominal_wl_nm=channel.wavelength_nm,
                x_center=channel.x_center,
            )

        if progress_callback is not None:
            progress_callback(
                AnalysisProgress(
                    step=step,
                    total=total,
                    message=(
                        f"{side}[{i}] @ {channel.wavelength_nm:.3f} nm  "
                        f"FWHM {fwhm:.4f} nm  in-band {in_band*100:.1f}%  "
                        f"leak {leak*100:.1f}%"
                    ),
                    wl=channel.wavelength_nm,
                    metric=in_band,
                )
            )

    result = ModulationErrorResult(
        channels=spectra,
        center_wl=layout.center_wl,
        channel_width_px=layout.channel_width_px,
        pitch_px=layout.pitch_px,
        nm_per_px=layout.nm_per_px,
    )

    # consolidate the per-capture files into one NPZ once the sweep is done
    if capture_path is not None and spectra:
        result.raw_npz_path = build_spectra_npz(result, capture_path / "spectra.npz")

    return result


def build_spectra_npz(result: ModulationErrorResult, path: str | Path) -> str:
    """Consolidate every channel's raw spectrum + metrics into one NPZ.

    Per-channel arrays are stored under keys ``<side><index>_wl`` /
    ``<side><index>_sig``; aligned metadata arrays (one entry per channel) are
    stored under ``meta_*`` keys so the file is self-describing.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, np.ndarray] = {}
    for ch in result.channels:
        key = f"{ch.side}{ch.index}"
        data[f"{key}_wl"] = np.asarray(ch.wavelengths_nm, dtype=float)
        data[f"{key}_sig"] = np.asarray(ch.signal_w, dtype=float)

    data["meta_key"] = np.array([f"{c.side}{c.index}" for c in result.channels])
    data["meta_side"] = np.array([c.side for c in result.channels])
    data["meta_index"] = np.array([c.index for c in result.channels])
    data["meta_x_center"] = np.array([c.x_center for c in result.channels])
    data["meta_nominal_wl_nm"] = np.array([c.nominal_wl_nm for c in result.channels])
    data["meta_peak_wl_nm"] = np.array([c.peak_wl_nm for c in result.channels])
    data["meta_nominal_bw_nm"] = np.array([c.nominal_bw_nm for c in result.channels])
    data["meta_fwhm_nm"] = np.array([c.fwhm_nm for c in result.channels])
    data["meta_total_power_w"] = np.array([c.total_power_w for c in result.channels])
    data["meta_in_band_fraction"] = np.array([c.in_band_fraction for c in result.channels])
    data["meta_neighbor_leakage"] = np.array([c.neighbor_leakage for c in result.channels])
    for offset in _NEIGHBOR_OFFSETS:
        data[f"meta_xtalk_{offset}"] = np.array(
            [c.crosstalk.get(offset, 0.0) for c in result.channels]
        )
    data["center_wl"] = np.array(result.center_wl)
    data["channel_width_px"] = np.array(result.channel_width_px)
    data["pitch_px"] = np.array(result.pitch_px)
    data["nm_per_px"] = np.array(result.nm_per_px)

    np.savez_compressed(out, **data)
    return str(out)


def write_analysis_csv(result: ModulationErrorResult, path: str) -> str:
    """Write the per-channel metrics table to CSV."""
    import csv
    from pathlib import Path

    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["side", "index", "x_center", "nominal_wl_nm", "peak_wl_nm",
             "nominal_bw_nm", "fwhm_nm", "total_power_w", "in_band_fraction",
             "neighbor_leakage", "xtalk_-2", "xtalk_-1", "xtalk_+1", "xtalk_+2"]
        )
        for ch in result.channels:
            writer.writerow([
                ch.side, ch.index, ch.x_center,
                f"{ch.nominal_wl_nm:.5f}", f"{ch.peak_wl_nm:.5f}",
                f"{ch.nominal_bw_nm:.5f}", f"{ch.fwhm_nm:.5f}",
                f"{ch.total_power_w:.6e}", f"{ch.in_band_fraction:.5f}",
                f"{ch.neighbor_leakage:.5f}",
                f"{ch.crosstalk.get(-2, 0.0):.5f}", f"{ch.crosstalk.get(-1, 0.0):.5f}",
                f"{ch.crosstalk.get(1, 0.0):.5f}", f"{ch.crosstalk.get(2, 0.0):.5f}",
            ])
    return str(out)
