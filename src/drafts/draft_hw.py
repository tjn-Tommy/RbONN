"""Shared hardware wiring for the calibration drafts (SLM + DAQ).

The calib_step6/7 drafts drive the same instruments the GUI does, but from
plain scripts.  This module holds the wiring that used to be copy-pasted into
each draft:

* :func:`connect_slm` -- LCOS-SLM display auto-detection + DVI-mode switch.
* :func:`connect_daq` -- a :class:`daq_module.DAQController` configured for the
  fixed-window read scheme (``t_both`` when both beams are on, the longer
  ``t_single`` when at most one beam is on -- incl. all-off darks).  All other
  acquisition parameters (1 kS/s, +/-0.1 V DIFF, 20 Hz low-pass, SEM over
  ``n_eff = 2*T*f_cut``) are the :class:`~daq_module.DAQMonitorSettings`
  defaults -- the values these drafts validated on hardware, now owned by
  ``daq_module``.
* :func:`read_point` -- one fixed-window averaged read -> ``(mean, std, sem)``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from daq_module import DAQController, DAQMonitorSettings  # noqa: E402
from slm_module.controller import SLMController  # noqa: E402

# Our transimpedance amplifier outputs a NEGATIVE voltage for positive light, so
# every read is sign-inverted here to record a positive light signal (more light
# -> more positive volts).  This is the single read chokepoint the step-6/7
# calibrations share (via read_point -> monitor_cycle), so inverting here flips
# both without touching the shared daq_module.
INVERT = True


def detect_slm_display() -> int:
    """Find the LCOS-SLM display number (the GUI's Detect step).

    Probing needs the DLL, so use a throwaway controller on display 1 just to
    run detect_displays(); the real controller is then built on the found no.
    Hardcoding display 1 is what dumped the pattern onto the main monitor.
    """
    probe = SLMController(display_no=1)
    for display_no, width, height, name in probe.detect_displays():
        print(f"  display {display_no}: {width}x{height} ({name})")
        if name.startswith("LCOS-SLM"):
            return display_no
    raise RuntimeError(
        "No LCOS-SLM display found. Check the SLM is connected as an extended "
        "display, or set SLM_DISPLAY_NO manually."
    )


def connect_slm(display_no: int | None = None, usb_slm_no: int = 1) -> SLMController:
    """Open the SLM (auto-detecting the display when ``display_no`` is None).

    ``display_array()`` only writes the DVI-mode frame buffer; if the panel's
    video interface is still set to Memory mode over USB, that write is
    silently ignored by the hardware.  Force DVI mode so what we send is
    actually what the panel shows (mirrors the GUI's "Switch to DVI mode").
    """
    if display_no is None:
        display_no = detect_slm_display()
    slm = SLMController(display_no=display_no)
    slm.open_slm()
    width, height = slm.get_slm_info()
    print(f"SLM: connected on display {display_no} ({width}x{height})")
    slm.set_dvi_mode(usb_slm_no)
    print(f"SLM: DVI mode set (USB device {usb_slm_no})")
    return slm


def connect_daq(
    *,
    device: str = "Dev1",
    channel: str = "ai0",
    t_both: float | None = None,
    t_single: float | None = None,
) -> DAQController:
    """Connect a DAQController configured for the fixed T_both/T_single windows.

    ``t_both`` / ``t_single`` default to the :class:`DAQMonitorSettings` values
    (3 s / 5 s).  ``hold=0``: the drafts own their settle waits, matching how
    the pipeline configures its monitor.
    """
    defaults = DAQMonitorSettings()
    settings = DAQMonitorSettings(
        channel=channel,
        duration=defaults.duration if t_both is None else float(t_both),
        single_duration=(
            defaults.single_duration if t_single is None else float(t_single)
        ),
        hold=0.0,
    )
    daq = DAQController(device=device)
    daq.connect()
    daq.configure_monitor(settings)
    print(f"DAQ: {daq.identify()} {settings.channel} @ {settings.sample_rate:g} S/s, "
          f"+/-{settings.max_val:g} V, low-pass {settings.f_cut:g} Hz -- "
          f"T_both {settings.duration:g} s, T_single {settings.single_duration:g} s")
    return daq


def read_point(
    daq: DAQController, *, single: bool = False, invert: bool = INVERT,
    timeout: float = 30.0,
) -> tuple[float, float, float]:
    """One fixed-window averaged read; return ``(mean, std, sem)`` in volts.

    ``single=True`` reads the longer T_single window -- at most one beam on
    (``x == 0 or w == 0``, incl. all-off darks; see ``DAQMonitorSettings``).
    ``std`` is the low-passed trace spread, ``sem`` the standard error of the
    mean over the effective independent-sample count ``2 * T * f_cut``.

    With ``invert`` (default ``INVERT``) the mean is negated so the TIA's
    negative-for-light output records as a positive light signal.  ``std`` and
    ``sem`` are spreads and stay non-negative -- negating the trace leaves them
    unchanged.
    """
    sample = daq.monitor_cycle(timeout=timeout, single=single)
    if sample is None:
        raise RuntimeError("DAQ read aborted")
    mean = -float(sample.value) if invert else float(sample.value)
    return mean, float(sample.std), float(sample.sem)


__all__ = [
    "detect_slm_display",
    "connect_slm",
    "connect_daq",
    "read_point",
]
