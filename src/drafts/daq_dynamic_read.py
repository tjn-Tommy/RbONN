"""Dynamic-duration NI-DAQ read: probe once, predict the time needed, read the rest.

Runnable script *and* module, built on ``daq_read_waveform``.  A test-bed for
*dynamic reading time*: instead of a fixed window, probe for ``T_MIN`` seconds and,
if the low-passed ``sem / |mean|`` is above ``TARGET_SEM_RATIO`` (2%), predict how
long is actually needed and take one more read to get there.

Because the SEM ratio scales as ``1 / sqrt(T)`` (the low-passed std barely changes
with window length), a probe fixes the whole curve::

    ratio(T) = ratio_probe * sqrt(T_MIN / T)   ->   T = T_MIN * (ratio_probe / target)**2

e.g. a 4% probe over T_MIN needs 4x the time (2x the sqrt) to reach 2%, so we read
``3 * T_MIN`` more, concatenate the two waveforms, and report the pooled mean/SEM.
The prediction is clamped to ``T_MAX``; a settle delay ``T_HOLD`` runs before every
read.

* Run it directly -- probe, extend, print the numbers, and plot both waveforms::

      python src/drafts/daq_dynamic_read.py

* Import it from another draft (e.g. calib_step6) and read one dynamic point::

      from daq_dynamic_read import measure_adaptive
      mean_v, sem_v, duration_s = measure_adaptive()   # aims for sem ratio <= 2%

This is a smoke test to see whether the dynamic-timing model actually lands near the
target on real hardware -- the plot overlays the measured probe/final points on the
predicted 1/sqrt(T) curve so you can see the estimate work (or not).
"""
from __future__ import annotations

import os
import sys
import time

# Make ``import daq_read_waveform`` resolve whether we are run directly (src/drafts
# is already sys.path[0]) or imported with only ``src`` on the path (calib_step6).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from daq_read_waveform import (  # noqa: E402
    CHANNEL,
    DEVICE,
    F_CUT,
    FILTER_ORDER,
    MAX_VAL_V,
    MIN_VAL_V,
    SAMPLE_RATE_HZ,
    lowpass,
    read_waveform,
    stats,
)

# Dynamic-duration parameters -- these replace the fixed DURATION_S in daq_read_waveform.
T_MIN = 1               # probe / minimum acquisition window, seconds
T_HOLD = 0.25             # settle delay BEFORE every read, seconds
T_MAX = 5.0               # hard cap on the total pooled acquisition, seconds
TARGET_SEM_RATIO = 0.02   # aim for sem / |mean| <= this (2%)


def _mean_sem_ratio(
    v: np.ndarray, fs: float, f_cut: float, order: int
) -> tuple[float, float, float]:
    """Low-pass ``v`` then return ``(mean, sem, sem_ratio)`` over n_eff = 2*T*f_cut."""
    filtered = lowpass(v, fs, f_cut, order)
    duration = v.size / fs if fs else 0.0
    n_eff = max(2.0 * duration * f_cut, 1.0)
    mean, sem = stats(filtered, n_eff)
    ratio = abs(sem / mean) if mean else float("inf")
    return mean, sem, ratio


def adaptive_acquire(
    *,
    target_sem_ratio: float = TARGET_SEM_RATIO,
    t_min: float = T_MIN,
    t_hold: float = T_HOLD,
    t_max: float = T_MAX,
    f_cut: float = F_CUT,
    filter_order: int = FILTER_ORDER,
    device: str = DEVICE,
    channel: str = CHANNEL,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    min_val_v: float = MIN_VAL_V,
    max_val_v: float = MAX_VAL_V,
    verbose: bool = False,
) -> tuple[np.ndarray, list[dict], dict]:
    """Probe for ``t_min``, predict the time needed, then read the remainder once.

    Settle ``t_hold`` seconds, read ``t_min``, and measure the low-passed SEM ratio.
    If it already meets ``target_sem_ratio`` we stop with one waveform.  Otherwise
    predict the total time from ``T = t_min * (ratio_probe / target)**2`` (the
    1/sqrt(T) SEM scaling), clamp it to ``[t_min, t_max]``, settle again, read the
    remainder, and concatenate.  The final mean/SEM are recomputed on the *pooled*
    trace, so the reported uncertainty is the truly achieved one.

    Returns ``(voltages, segments, summary)``:

    * ``voltages`` -- the pooled trace (one or two reads concatenated),
    * ``segments`` -- one dict per hardware read (``voltages``, ``duration``,
      ``label``) so the caller can plot the probe and the extension separately,
    * ``summary`` -- ``mean``, ``sem``, ``sem_ratio``, ``duration``, ``probe_ratio``,
      ``t_predicted`` and ``target_met``.
    """
    segments: list[dict] = []

    def read(duration: float, label: str) -> np.ndarray:
        if t_hold:
            time.sleep(t_hold)  # settle before every measurement
        _, v = read_waveform(
            device=device,
            channel=channel,
            sample_rate_hz=sample_rate_hz,
            duration_s=duration,
            min_val_v=min_val_v,
            max_val_v=max_val_v,
            verbose=False,
        )
        segments.append({"voltages": v, "duration": v.size / sample_rate_hz, "label": label})
        return v

    # --- probe ---
    v1 = read(t_min, f"probe {t_min:.2f}s")
    _, _, probe_ratio = _mean_sem_ratio(v1, sample_rate_hz, f_cut, filter_order)

    # --- predict the total time needed (sem ratio ~ 1/sqrt(T)) ---
    if probe_ratio <= target_sem_ratio:
        t_predicted = t_min
    elif np.isfinite(probe_ratio):
        t_predicted = t_min * (probe_ratio / target_sem_ratio) ** 2
    else:
        t_predicted = t_max  # near-zero mean: can't predict, take the max
    t_total = min(max(t_predicted, t_min), t_max)
    t_extra = t_total - t_min

    if verbose:
        print(
            f"  probe  t={t_min:.2f}s  sem ratio={probe_ratio*100:.4f}%  "
            f"-> predict {t_predicted:.2f}s, extend {max(t_extra, 0.0):.2f}s "
            f"(capped at {t_max:.2f}s)"
        )

    # --- extend once, then pool ---
    voltages = v1
    if t_extra > 1e-3:
        v2 = read(t_extra, f"extend {t_extra:.2f}s")
        voltages = np.concatenate([v1, v2])

    duration = voltages.size / sample_rate_hz
    mean, sem, ratio = _mean_sem_ratio(voltages, sample_rate_hz, f_cut, filter_order)
    summary = {
        "mean": mean,
        "sem": sem,
        "sem_ratio": ratio,
        "duration": duration,
        "probe_ratio": probe_ratio,
        "t_predicted": t_predicted,
        "target_met": ratio <= target_sem_ratio,
    }
    if verbose:
        print(
            f"  final  t={duration:.2f}s  mean={abs(mean)*1000:.4f} mV  "
            f"sem={sem*1000:.4f} mV  sem ratio={ratio*100:.4f}%"
        )
    return voltages, segments, summary


def measure_adaptive(
    *,
    target_sem_ratio: float = TARGET_SEM_RATIO,
    t_min: float = T_MIN,
    t_hold: float = T_HOLD,
    t_max: float = T_MAX,
    f_cut: float = F_CUT,
    filter_order: int = FILTER_ORDER,
    device: str = DEVICE,
    channel: str = CHANNEL,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    min_val_v: float = MIN_VAL_V,
    max_val_v: float = MAX_VAL_V,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """Read one dynamic point; return ``(mean_v, sem_v, duration_s)``.

    The one-call entry point for importers (e.g. calib_step6 reading Y after each
    SLM pattern).  The final SEM ratio is ``sem_v / |mean_v|``; compare it to
    ``target_sem_ratio`` to know whether the ``t_max`` cap was hit before convergence.
    """
    _, _, summary = adaptive_acquire(
        target_sem_ratio=target_sem_ratio,
        t_min=t_min,
        t_hold=t_hold,
        t_max=t_max,
        f_cut=f_cut,
        filter_order=filter_order,
        device=device,
        channel=channel,
        sample_rate_hz=sample_rate_hz,
        min_val_v=min_val_v,
        max_val_v=max_val_v,
        verbose=verbose,
    )
    return summary["mean"], summary["sem"], summary["duration"]


__all__ = [
    "adaptive_acquire",
    "measure_adaptive",
    "T_MIN",
    "T_HOLD",
    "T_MAX",
    "TARGET_SEM_RATIO",
]


def main() -> None:
    import matplotlib.pyplot as plt

    print(
        f"Dynamic read (target sem ratio {TARGET_SEM_RATIO*100:.1f}%, "
        f"probe {T_MIN:.2f}s, cap {T_MAX:.2f}s, settle {T_HOLD:.2f}s):"
    )
    voltages, segments, summary = adaptive_acquire(verbose=True)

    mean, sem, ratio = summary["mean"], summary["sem"], summary["sem_ratio"]
    duration, probe_ratio = summary["duration"], summary["probe_ratio"]
    print(
        f"\nFinal: mean={abs(mean)*1000:.4f} mV, sem={sem*1000:.4f} mV, "
        f"sem ratio={ratio*100:.4f}%  over {duration:.2f}s "
        f"({'target met' if summary['target_met'] else 'hit T_MAX cap'})"
    )

    filtered_full = lowpass(voltages, SAMPLE_RATE_HZ, F_CUT)
    times_full = np.arange(voltages.size) / SAMPLE_RATE_HZ

    fig, (ax_t, ax_c) = plt.subplots(2, 1, figsize=(10, 8))

    # --- time domain: the two recorded waveforms (raw, colored per read) + pooled low-pass ---
    start = 0
    for seg in segments:
        n = seg["voltages"].size
        sl = slice(start, start + n)
        ax_t.plot(times_full[sl], voltages[sl] * 1000.0, linewidth=0.8, alpha=0.6,
                  label=f"raw: {seg['label']}")
        start += n
    ax_t.plot(times_full, filtered_full * 1000.0, "k", linewidth=1.5,
              label=f"low-pass {F_CUT:.1f} Hz (pooled)")
    if len(segments) > 1:
        seam = segments[0]["voltages"].size / SAMPLE_RATE_HZ
        ax_t.axvline(seam, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
                     label="probe | extend seam")
    ax_t.set_title(
        f"{DEVICE}/{CHANNEL}  dynamic read ({SAMPLE_RATE_HZ/1000:.0f} kS/s, {duration:.2f} s pooled)"
    )
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Voltage (mV)")
    ax_t.legend()
    ax_t.grid(True)

    # --- did the 1/sqrt(T) prediction land on target? probe/final points vs the model ---
    t_curve = np.linspace(T_MIN, max(duration, T_MIN * 1.01), 100)
    pred_curve = probe_ratio * np.sqrt(T_MIN / t_curve) * 100.0
    ax_c.plot(t_curve, pred_curve, linewidth=1.5,
              label="predicted  ratio_probe * sqrt(T_MIN/T)")
    ax_c.plot(T_MIN, probe_ratio * 100.0, "o", markersize=9, label="probe measured")
    ax_c.plot(duration, ratio * 100.0, "s", markersize=9, label="final measured")
    ax_c.axhline(TARGET_SEM_RATIO * 100.0, color="k", linestyle="--", linewidth=1.0,
                 alpha=0.6, label=f"{TARGET_SEM_RATIO*100:.1f}% target")
    ax_c.set_title("Dynamic-timing model: predicted vs achieved SEM ratio")
    ax_c.set_xlabel("Acquisition time T (s)")
    ax_c.set_ylabel("SEM ratio (%)")
    ax_c.legend()
    ax_c.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
