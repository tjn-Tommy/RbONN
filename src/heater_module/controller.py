"""High-level Thorlabs TC300B orchestration for the GUI.

This mirrors ``src/drafts/heat_controller.py`` -- the standalone staircase heater
driver -- as a reusable controller the SLM control suite can drive, in the same
shape as ``DAQController`` / ``ScopeController`` (connect / read / a background
loop that emits samples). The staircase, landing/rest logic, adaptive step and
emergency rail-rest are a faithful port of that script; only the I/O has moved
behind :class:`heater_module.driver.TC300Driver` and the once-per-loop status
print is now a ``sample_cb`` callback so the GUI can stream it to a chart.

Firmware facts baked in (FV 4.04): PID params are integers x100 (KP=0.5 ->
``KP1=50``); ``TSETx=n`` -> n/1000 degC; ``VMAXx=n`` -> n/10 V. Heater mode is
``MOD 0`` (temp PID off TSET, sensor live). See ``tc300-device-quirks`` notes.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .driver import HeaterConnectionError, HeaterError, TC300Driver

# Tuned heater-hold PID, per channel -- kept in lock-step with heat_controller.py
# PID_DEFAULTS. KP=0.5/TI=20/TD=2 holds std ~0.006 C / pp ~0.02-0.03 C at 79.5 C
# on both channels, but only with a constant-voltage DC base heater carrying the
# baseline power (else the trim heater rails and limit-cycles).
PID_DEFAULTS: dict[int, dict[str, float]] = {
    1: {"kp": 0.5, "ti": 20.0, "td": 2.0},
    2: {"kp": 0.5, "ti": 20.0, "td": 2.0},
}


def to_float(s: Any) -> float | None:
    """First float token in a TC300 reply, or None (mirrors heat_controller)."""
    for tok in str(s).replace(",", " ").split():
        try:
            return float(tok)
        except ValueError:
            continue
    return None


def pid_int(val: float) -> int:
    """PID params are set as integers x100 on this firmware (KP=0.5 -> 50)."""
    return int(round(val * 100))


@dataclass
class HeaterSample:
    """One channel's live state at a single loop cycle."""

    channel: int
    temp: float | None = None       # TACT, real degC
    volt: float = 0.0               # VOLT, real V
    curr: float | None = None       # CURR, mA
    setpoint: float | None = None   # current staircase step target (hold), degC
    railed_s: float = 0.0           # continuous seconds at/above the rail


@dataclass
class HeaterCycle:
    """All channels' state for one loop iteration, plus the global error flag."""

    elapsed: float
    channels: dict[int, HeaterSample] = field(default_factory=dict)
    err: str = ""


@dataclass(frozen=True)
class StaircaseSettings:
    """Parameters for a staircase ramp-and-hold (defaults match heat_controller)."""

    target: float = 79.5        # final target, real degC
    step: float = 2.0           # max staircase step size, degC
    rest: float = 2.0           # calm seconds at a landing before stepping again
    railmax: float = 14.0       # wall-clock railed seconds -> emergency EN=0 rest
    vmax: float | None = None   # optional voltage cap, real V (None = device max)
    period: float = 0.5         # loop period, s
    rail_frac: float = 0.85     # VOLT >= rail_frac * VMAX counts as railed


class TC300Controller:
    """Connect / read / drive wrapper around :class:`TC300Driver` (injectable)."""

    def __init__(self, port: str | None = None, *, driver: Any | None = None):
        if driver is not None:
            self.driver = driver
        elif port is not None:
            self.driver = TC300Driver(port=port)
        else:
            raise ValueError("either port or an explicit driver is required")

    # ------------------------------------------------------------------ conn
    @property
    def is_connected(self) -> bool:
        return self.driver.is_connected

    def connect(self) -> None:
        self.driver.connect()

    def disconnect(self) -> None:
        self.driver.disconnect()

    def identify(self) -> str:
        return self.driver.identify()

    def read_error(self) -> str:
        """Global controller error flag ('0' when healthy)."""
        return self.driver.command("ERR?")

    # ------------------------------------------------------------- primitives
    def _query_float(self, cmd: str) -> float | None:
        return to_float(self.driver.command(cmd))

    def set_mode_heater(self, ch: int) -> None:
        """Force Heater mode (MOD 0) so TSET is honoured (never trust prior mode)."""
        if self.driver.command(f"MOD{ch}?") != "0":
            self.driver.command(f"MOD{ch}=0")

    def set_vmax(self, ch: int, volts: float) -> None:
        self.driver.command(f"VMAX{ch}={int(round(volts * 10))}")  # n/10 V

    def get_vmax(self, ch: int) -> float | None:
        return self._query_float(f"VMAX{ch}?")

    def set_target(self, ch: int, deg_c: float) -> None:
        self.driver.command(f"TSET{ch}={int(round(deg_c * 1000))}")  # n/1000 degC

    def set_pid(self, ch: int, kp: float | None = None, ti: float | None = None,
                td: float | None = None) -> None:
        if kp is not None:
            self.driver.command(f"KP{ch}={pid_int(kp)}")
        if ti is not None:
            self.driver.command(f"TI{ch}={pid_int(ti)}")
        if td is not None:
            self.driver.command(f"TD{ch}={pid_int(td)}")

    def read_pid(self, ch: int) -> str:
        return self.driver.command(f"PID{ch}?")

    def enable(self, ch: int) -> None:
        self.driver.command(f"EN{ch}=1")

    def disable(self, ch: int) -> None:
        self.driver.command(f"EN{ch}=0")

    def read_temp(self, ch: int) -> float | None:
        return self._query_float(f"TACT{ch}?")

    def read_channel(self, ch: int) -> HeaterSample:
        """One TACT/VOLT/CURR snapshot for a channel."""
        return HeaterSample(
            channel=ch,
            temp=self._query_float(f"TACT{ch}?"),
            volt=self._query_float(f"VOLT{ch}?") or 0.0,
            curr=self._query_float(f"CURR{ch}?"),
        )

    def hold(self, ch: int, target: float, *, kp: float | None = None,
             ti: float | None = None, td: float | None = None,
             vmax: float | None = None) -> None:
        """One-shot: Heater mode + tuned PID + target + enable (no staircase).

        Use when the block is already near the target and a direct hold is safe;
        for a cold start use :meth:`run_staircase` so the ~23 s no-load watchdog
        never trips on a long full-rail climb.
        """
        self.set_mode_heater(ch)
        if vmax is not None:
            self.set_vmax(ch, vmax)
        d = PID_DEFAULTS.get(ch, {})
        self.set_pid(
            ch,
            kp if kp is not None else d.get("kp"),
            ti if ti is not None else d.get("ti"),
            td if td is not None else d.get("td"),
        )
        self.set_target(ch, target)
        self.enable(ch)

    # ------------------------------------------------------------------ loops
    def monitor(
        self,
        channels: list[int],
        *,
        sample_cb: Callable[[HeaterCycle], None],
        stop_event: threading.Event,
        period: float = 0.5,
    ) -> None:
        """Read-only live loop: poll TACT/VOLT/CURR + ERR until ``stop_event``.

        Touches no EN / TSET / PID, so it never disturbs a running hold -- safe
        to watch a block the standalone driver (or a prior ramp) is holding.
        """
        t0 = time.time()
        while not stop_event.is_set():
            err = self.read_error()
            chans = {ch: self.read_channel(ch) for ch in channels}
            sample_cb(HeaterCycle(elapsed=time.time() - t0, channels=chans, err=err))
            stop_event.wait(period)

    def run_staircase(
        self,
        channels: list[int],
        settings: StaircaseSettings,
        *,
        pid_map: dict[int, dict[str, float]] | None = None,
        sample_cb: Callable[[HeaterCycle], None] | None = None,
        stop_event: threading.Event | None = None,
        disable_on_exit: bool = True,
    ) -> dict[int, dict[str, Any]]:
        """Staircase ramp to ``settings.target`` then hold -- faithful port of
        ``heat_controller.main``.

        Never asks for more than ``settings.step`` above the current temperature:
        the loop rails briefly, lands on the step target, rests (which resets the
        ~23 s no-load watchdog), then steps again. A wall-clock rail timer
        (``settings.railmax``) forces an emergency EN=0 rest if a step ever rails
        too long. After reaching the target it keeps looping (holding + emitting
        samples) until ``stop_event`` is set; on exit it disables the channels
        unless ``disable_on_exit`` is False (the TC300 holds autonomously after
        the link closes, so leaving it enabled keeps the block hot).

        Returns the per-channel staircase state (steps/rests counters, etc.).
        """
        channels = sorted(set(channels))
        pid_map = pid_map or {}
        stop_event = stop_event or threading.Event()

        # Per-channel setup: Heater mode, optional VMAX cap, tuned PID, first
        # landing (<= step above current temp), enable.
        st: dict[int, dict[str, Any]] = {}
        for ch in channels:
            self.set_mode_heater(ch)
            if settings.vmax is not None:
                self.set_vmax(ch, settings.vmax)
            vmax = self.get_vmax(ch) or 24.0
            pid = pid_map.get(ch) or PID_DEFAULTS.get(ch, {})
            self.set_pid(ch, pid.get("kp"), pid.get("ti"), pid.get("td"))
            t_now = self.read_temp(ch) or 25.0
            hold = min(settings.target, t_now + settings.step)
            self.set_target(ch, hold)
            self.enable(ch)
            st[ch] = {
                "rail_v": settings.rail_frac * vmax,  # VOLT >= this counts as railed
                "hold": hold,                          # current step target
                "rail_start": None,                    # ts when railing began
                "calm_since": None,                    # ts when landing dwell began
                "last_t": t_now,                       # temp at previous sample
                "steps": 0,
                "rests": 0,
            }

        t0 = time.time()
        last_time = t0
        try:
            while not stop_event.is_set():
                now = time.time()
                el = now - t0
                err = self.read_error()          # ERR? is global to the controller
                chans: dict[int, HeaterSample] = {}

                for ch in channels:
                    s = st[ch]
                    temp = self.read_temp(ch)
                    volt = self._query_float(f"VOLT{ch}?") or 0.0
                    curr = self._query_float(f"CURR{ch}?")

                    railed = abs(volt) >= s["rail_v"]
                    s["rail_start"] = (s["rail_start"] or now) if railed else None
                    railed_s = (now - s["rail_start"]) if s["rail_start"] else 0.0
                    chans[ch] = HeaterSample(
                        channel=ch, temp=temp, volt=volt, curr=curr,
                        setpoint=s["hold"], railed_s=railed_s,
                    )
                    if temp is None:
                        continue

                    # Emergency rest: a HARD EN=0 is the only thing that drops the
                    # output to 0 and resets the ~23 s watchdog; merely lowering
                    # TSET does not (a wound-up integral keeps the output pinned).
                    if railed_s >= settings.railmax:
                        s["rests"] += 1
                        self.disable(ch)
                        r0 = time.time()
                        while time.time() - r0 < 2.5:
                            v = self._query_float(f"VOLT{ch}?") or 0.0
                            if v < 1.0 and (time.time() - r0) >= 1.0:
                                break
                            time.sleep(0.3)
                        self.enable(ch)
                        s["rail_start"] = None
                        s["calm_since"] = None

                    # Landing: advance once the TEMPERATURE reaches the step target
                    # and dwells --rest seconds -- do NOT require electrical calm (a
                    # channel that overshoots instead of desaturating never goes
                    # calm; railmax keeps a still-railed channel watchdog-safe).
                    if temp >= s["hold"] - 0.25:
                        s["calm_since"] = s["calm_since"] or now
                        if (now - s["calm_since"]) >= settings.rest and s["hold"] < settings.target:
                            dt = max(now - last_time, 1e-3)
                            rate = max((temp - s["last_t"]) / dt, 0.0)      # degC/s
                            adaptive = max(0.3, min(settings.step, rate * 10))  # ~10 s climb
                            s["hold"] = min(settings.target, temp + adaptive)
                            self.set_target(ch, s["hold"])
                            s["steps"] += 1
                            s["calm_since"] = None
                    else:
                        s["calm_since"] = None       # fell back below landing; re-arm
                    s["last_t"] = temp

                if sample_cb is not None:
                    sample_cb(HeaterCycle(elapsed=el, channels=chans, err=err))

                if err not in ("", "0"):
                    # Latched despite pacing -- caller must power-cycle. Stop here;
                    # the finally still disables the channels.
                    break

                last_time = now
                stop_event.wait(settings.period)
        finally:
            if disable_on_exit:
                for ch in channels:
                    try:
                        self.disable(ch)
                    except HeaterError:
                        pass
        return st

    def __enter__(self) -> "TC300Controller":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


__all__ = [
    "TC300Controller",
    "HeaterSample",
    "HeaterCycle",
    "StaircaseSettings",
    "PID_DEFAULTS",
    "to_float",
    "pid_int",
    "HeaterError",
    "HeaterConnectionError",
]
