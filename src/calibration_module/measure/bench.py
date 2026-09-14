"""Shared bench wiring for the calibration steps (SLM + DAQ + OSA).

The step scripts drive the same instruments the GUI does, but from plain
scripts.  This module holds the wiring that used to be copy-pasted into each
one, and is the single chokepoint every step-6/7/8 read passes through:

* :func:`connect_slm` -- LCOS-SLM display auto-detection + DVI-mode switch.
* :func:`connect_daq` -- a :class:`daq_module.DAQController` configured for the
  fixed-window read scheme (``t_both`` when both beams are on, the longer
  ``t_single`` when at most one beam is on -- incl. all-off darks).  All other
  acquisition parameters (1 kS/s, +/-0.1 V DIFF, 20 Hz low-pass) are the
  :class:`~daq_module.DAQMonitorSettings` defaults -- the values these steps
  validated on hardware, now owned by ``daq_module``.
* :func:`read_point` -- one fixed-window averaged read -> ``(mean, std)``.
* :func:`connect_osa` -- the AQ637X over LAN, for the step 1-3 OSA sweeps.
* :func:`print_progress` -- a ``progress_callback`` that prints each update.

It was ``steps/draft_hw.py`` until the fit/measure split; nothing about it was
ever a draft, and it sits here now because it is the acquisition layer itself,
not one of the scripts that calls it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from daq_module import DAQController, DAQMonitorSettings  # noqa: E402
from osa_module.controller import OSAController  # noqa: E402
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
    min_val: float | None = None,
    max_val: float | None = None,
) -> DAQController:
    """Connect a DAQController configured for the fixed T_both/T_single windows.

    ``t_both`` / ``t_single`` default to the :class:`DAQMonitorSettings` values
    (3 s / 5 s).  ``hold=0``: the drafts own their settle waits, matching how
    the pipeline configures its monitor.

    ``min_val`` / ``max_val`` override the input range (default +/-0.1 V, the
    most sensitive the board offers).  Raise it when a step drives enough light
    to clip: step 7 holds the reference on while sweeping a second pair, so its
    bright end can run past 0.1 V, and a clipped read is a silently wrong mean,
    not an error.  The board quantizes the range -- it only offers +/-0.1, 0.2,
    0.5, 1, 2, 5, 10 V and rounds a request UP -- so ask for one of those.
    """
    defaults = DAQMonitorSettings()
    settings = DAQMonitorSettings(
        channel=channel,
        duration=defaults.duration if t_both is None else float(t_both),
        single_duration=(
            defaults.single_duration if t_single is None else float(t_single)
        ),
        hold=0.0,
        min_val=defaults.min_val if min_val is None else float(min_val),
        max_val=defaults.max_val if max_val is None else float(max_val),
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
) -> tuple[float, float]:
    """One fixed-window averaged read; return ``(mean, std)`` in volts.

    ``single=True`` reads the longer T_single window -- at most one beam on
    (``x == 0 or w == 0``, incl. all-off darks; see ``DAQMonitorSettings``).
    ``std`` is the spread of the low-passed trace -- the one uncertainty these
    drafts carry, undivided by any effective-N.

    With ``invert`` (default ``INVERT``) the mean is negated so the TIA's
    negative-for-light output records as a positive light signal.  ``std`` is a
    spread and stays non-negative -- negating the trace leaves it unchanged.
    """
    sample = daq.monitor_cycle(timeout=timeout, single=single)
    if sample is None:
        raise RuntimeError("DAQ read aborted")
    mean = -float(sample.value) if invert else float(sample.value)
    return mean, float(sample.std)


def connect_osa(host: str = "192.168.1.11", port: int = 10001) -> OSAController:
    """Open the OSA over LAN (the GUI's Connections-page defaults)."""
    osa = OSAController(host=host, port=port)
    osa.connect()
    print(f"OSA: {osa.identify()} at {host}:{port}")
    return osa


def print_progress(progress) -> None:
    """A ``progress_callback`` for the step 1-3 sweeps: one line per update.

    ``progress`` is a ``measure.slm_response.CalibrationProgress``.
    """
    print(f"  [{progress.phase} {progress.step + 1}/{progress.total}] "
          f"{progress.message}")


__all__ = [
    "detect_slm_display",
    "connect_slm",
    "connect_daq",
    "connect_osa",
    "read_point",
    "print_progress",
]
