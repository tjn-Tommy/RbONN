#!/usr/bin/env python3
"""
TC300B staircase heater driver.

Strategy (user-designed): never ask for more than a small step above the
current temperature. The loop rails briefly, lands on the step target,
desaturates and rests -- which resets the controller's ~29 s
continuous-drive "NO LOAD" watchdog -- then we step again. Repeat to the
final target. A wall-clock rail timer provides an emergency rest in case
a step ever takes too long (e.g. near the power-limited plateau).

One or both channels are driven SIMULTANEOUSLY (default: both), each with its
own independent staircase, landing/rest logic and rail timer, sharing the one
serial link. All run defaults live in the *_DEFAULT constants below.

Scaling on FV 4.04:  TSETx=n -> n/1000 degC     VMAXx=n -> n/10 V

The tuned hold PID (KP=0.5, TI=20, TD=2) is baked in as PID_DEFAULTS and applied
before enabling, so the final landing holds flat instead of inheriting the stock
bang-bang tuning (KP=1.5/TI=0.01/TD=0) that the controller reverts to after a
power-cycle. The same defaults are used on both channels (swept independently,
both optimal at 0.5/20/2). Tuned at the 79.5 C target with a constant-voltage DC
base heater carrying the baseline power, so the trim heater sits in the linear
regime (~6-8 V, no rail) and holds std ~0.006 C / pp ~0.02-0.03 C.

Usage:
  python heat_controller.py                         # both channels -> 79.5 C (defaults)
  python heat_controller.py --ch 1 --tset 60 --log stairs60.csv
  python heat_controller.py --ch 1 2 --tset 65 --step 2 --rest 4 --log both.csv

Requires:  pip install pyserial
"""

import argparse
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed. Run:  pip install pyserial")

BAUD = 115200

# Tuned heater-hold PID, per channel. On this firmware the params are set as
# integers x100 (KP=0.5 -> "KP1=50"); TI is an integral TIME (larger = gentler),
# TD is derivative time. KP=0.5/TI=20/TD=2 holds std ~0.006 C / pp ~0.02-0.03 C
# at 79.5 C on BOTH channels (swept independently via pid_sweep.py; both agreed
# on this optimum). Needs a constant-voltage DC base heater carrying the baseline
# power so the trim heater stays in the linear ~6-8 V regime -- without it the
# heater rails at 79.5 C and limit-cycles (~pp 0.4 C). TD has a sharp optimum at
# 2: TD=5 is looser, TD=0 breaks into a limit cycle. CH2 is less tolerant of a
# short TI (TI=10 went unstable), so keep TI=20. Override per run with --kp/--ti/--td.
PID_DEFAULTS = {
    1: {"kp": 0.5, "ti": 20.0, "td": 2.0},
    2: {"kp": 0.5, "ti": 20.0, "td": 2.0},
}

# Run defaults -- edit here in one place; any --flag overrides its default.
PORT_DEFAULT = "COM3"
CH_DEFAULT = [1, 2]        # channel(s) driven SIMULTANEOUSLY (default: both)
TSET_DEFAULT = 79.5        # final target, REAL degC
STEP_DEFAULT = 2.0         # max staircase step size, degC
REST_DEFAULT = 2.0         # calm seconds at a landing before stepping again
RAILMAX_DEFAULT = 14.0     # wall-clock railed seconds before an emergency EN=0 rest
                           # (< ~23 s watchdog, with margin -- CH2 rails flat-out)
VMAX_DEFAULT = None        # optional voltage cap, REAL volts (None = device max)


def send(ser, cmd, term, quiet=True):
    ser.reset_input_buffer()
    ser.write((cmd + term).encode("ascii"))
    time.sleep(0.08)
    raw = ser.read(256).decode("ascii", errors="replace")
    resp = raw.replace(cmd, "").replace(">", "").replace("\r", " ").replace("\n", " ").strip()
    if not quiet:
        print(f"  {cmd:<14} -> {resp}")
    return resp


def probe_terminator(ser):
    for term in ("\r", "\r\n", "\n"):
        resp = send(ser, "IDN?", term)
        if "TC300" in resp.upper() or "THORLABS" in resp.upper():
            print(f"Connected: {resp}  (terminator={term!r})")
            return term
    return None


def to_float(s):
    for tok in s.replace(",", " ").split():
        try:
            return float(tok)
        except ValueError:
            continue
    return None


def pid_int(val):
    """PID params are set as integers x100 on this firmware (KP=0.5 -> 50)."""
    return int(round(val * 100))


def set_target(ser, ch, term, deg_c):
    send(ser, f"TSET{ch}={int(round(deg_c * 1000))}", term)   # n/1000 degC


def resolve_pid(args, ch):
    """Per-channel PID: a CLI override if given, else the tuned PID_DEFAULTS[ch]."""
    d = PID_DEFAULTS.get(ch, {})
    return (
        args.kp if args.kp is not None else d.get("kp"),
        args.ti if args.ti is not None else d.get("ti"),
        args.td if args.td is not None else d.get("td"),
    )


def main():
    ap = argparse.ArgumentParser(description="TC300B staircase ramp, one or both channels (watchdog-safe).")
    ap.add_argument("--port", default=PORT_DEFAULT)
    ap.add_argument("--ch", type=int, nargs="+", choices=[1, 2], default=list(CH_DEFAULT),
                    help="channel(s) to drive SIMULTANEOUSLY (default: both)")
    ap.add_argument("--tset", type=float, default=TSET_DEFAULT, help="final target, REAL degC")
    ap.add_argument("--step", type=float, default=STEP_DEFAULT, help="max step size, degC")
    ap.add_argument("--rest", type=float, default=REST_DEFAULT,
                    help="seconds of calm (unrailed, near step target) before stepping again")
    ap.add_argument("--railmax", type=float, default=RAILMAX_DEFAULT,
                    help="wall-clock seconds of continuous rail before emergency rest")
    ap.add_argument("--vmax", type=float, default=VMAX_DEFAULT, help="optional voltage cap, REAL volts")
    ap.add_argument("--kp", type=float, default=None, help="proportional gain (default: PID_DEFAULTS[ch])")
    ap.add_argument("--ti", type=float, default=None, help="integral time, s (larger = gentler)")
    ap.add_argument("--td", type=float, default=None, help="derivative time, s")
    ap.add_argument("--secs", type=float, default=None,
                    help="optional run-duration cap; exits cleanly (disables channels) after this many s")
    ap.add_argument("--log", default=None, help="CSV log file")
    args = ap.parse_args()
    channels = sorted(set(args.ch))

    try:
        ser = serial.Serial(args.port, BAUD, bytesize=8, parity="N", stopbits=1, timeout=0.2)
    except serial.SerialException as e:
        sys.exit(f"Could not open {args.port}: {e}\n-> Close the Thorlabs GUI first.")

    term = probe_terminator(ser)
    if term is None:
        sys.exit("No response to IDN?.")

    err = send(ser, "ERR?", term, quiet=False)
    if err not in ("", "0"):
        sys.exit("ERR latched -- power-cycle the controller, then rerun immediately.")

    # Per-channel setup: VMAX cap, tuned PID, first landing, enable. Each channel
    # keeps its own independent staircase/rail state in st[ch].
    st = {}
    for ch in channels:
        # Force Heater mode (MOD 0) so TSET is actually honored. A channel left
        # in constant-voltage/current mode IGNORES TSET and drives open-loop --
        # the temperature then runs away past the setpoint (exactly what CH2 did:
        # full rail at 10 C above its 64.9 C target). Never trust the prior mode.
        if send(ser, f"MOD{ch}?", term) != "0":
            send(ser, f"MOD{ch}=0", term)
        send(ser, f"MOD{ch}?", term, quiet=False)

        if args.vmax is not None:
            send(ser, f"VMAX{ch}={int(round(args.vmax * 10))}", term)  # n/10 V
        vmax = to_float(send(ser, f"VMAX{ch}?", term, quiet=False)) or 24.0

        # Apply the tuned hold PID before enabling (integers x100 on this firmware).
        kp, ti, td = resolve_pid(args, ch)
        if kp is not None:
            send(ser, f"KP{ch}={pid_int(kp)}", term, quiet=False)
        if ti is not None:
            send(ser, f"TI{ch}={pid_int(ti)}", term, quiet=False)
        if td is not None:
            send(ser, f"TD{ch}={pid_int(td)}", term, quiet=False)
        send(ser, f"PID{ch}?", term, quiet=False)

        t_now = to_float(send(ser, f"TACT{ch}?", term)) or 25.0
        hold = min(args.tset, t_now + args.step)
        set_target(ser, ch, term, hold)
        send(ser, f"EN{ch}=1", term)
        st[ch] = {
            "rail_v": 0.85 * vmax,   # VOLT >= this counts as railed
            "hold": hold,            # current step target
            "rail_start": None,      # wall-clock ts when railing began
            "calm_since": None,      # wall-clock ts when calm-at-landing began
            "last_t": t_now,         # temp at previous sample (for climb rate)
            "steps": 0,
            "rests": 0,
        }
        print(f"CH{ch}: staircase to {args.tset:.2f} C in <= {args.step:.1f} C steps; "
              f"first landing {hold:.2f} C.")

    chlbl = "+".join(f"CH{c}" for c in channels)
    print(f"\nDriving {chlbl} simultaneously. Emergency rest if railed > "
          f"{args.railmax:.0f}s (wall clock). Ctrl+C stops and disables.\n")

    logf = open(args.log, "w") if args.log else None
    if logf:
        cols = ",".join(
            f"temp{ch}_C,volt{ch}_V,curr{ch}_mA,hold{ch}_C,railed{ch}_s,steps{ch},rests{ch}"
            for ch in channels
        )
        logf.write(f"elapsed_s,{cols},err\n")

    t0 = time.time()
    last_time = t0
    try:
        while True:
            now = time.time()
            el = now - t0
            err = send(ser, "ERR?", term)      # ERR? is global to the controller
            row = {}                            # ch -> (temp, volt, curr, railed_s)

            for ch in channels:
                s = st[ch]
                temp = to_float(send(ser, f"TACT{ch}?", term))
                volt = to_float(send(ser, f"VOLT{ch}?", term)) or 0.0
                curr = to_float(send(ser, f"CURR{ch}?", term))

                railed = abs(volt) >= s["rail_v"]
                if railed:
                    s["rail_start"] = s["rail_start"] or now
                else:
                    s["rail_start"] = None
                # NOTE: do NOT touch calm_since here. A channel that heats at full
                # power stays railed the whole climb; clearing calm_since while
                # railed reset the landing dwell every loop, so the staircase never
                # advanced (frozen hold, steps=0 -- exactly CH2's runaway). The
                # landing block below owns calm_since entirely (temperature-based).
                railed_s = (now - s["rail_start"]) if s["rail_start"] else 0.0
                row[ch] = (temp, volt, curr, railed_s)

                if temp is None:
                    continue

                # Emergency rest: railed too long mid-step (e.g. near power plateau).
                # A HARD EN=0 is the ONLY thing that actually drops the output to 0
                # and resets the ~23 s no-load watchdog. Merely lowering TSET does
                # NOT: after a long rail the integral is wound up and keeps the
                # output pinned, so the old rest fired but V never left 23.3 V and
                # the watchdog kept counting until it tripped (ERR). Disable, wait
                # for V to fall, then re-enable and resume the same setpoint.
                if railed_s >= args.railmax:
                    s["rests"] += 1
                    send(ser, f"EN{ch}=0", term)
                    r0 = time.time()
                    while time.time() - r0 < 2.5:
                        v = to_float(send(ser, f"VOLT{ch}?", term)) or 0.0
                        if v < 1.0 and (time.time() - r0) >= 1.0:
                            break
                        time.sleep(0.3)
                    send(ser, f"EN{ch}=1", term)
                    s["rail_start"] = None
                    s["calm_since"] = None

                # Landing: advance the staircase once the TEMPERATURE reaches the
                # step target and dwells there for --rest seconds -- do NOT require
                # the output to be electrically calm. A channel that overshoots
                # instead of desaturating (CH2) never goes "calm", so gating on
                # `not railed` froze its hold at the first landing while the temp
                # ran past it. --railmax's emergency rest keeps a still-railed
                # channel watchdog-safe, so stepping while railed is fine.
                if temp >= s["hold"] - 0.25:
                    s["calm_since"] = s["calm_since"] or now
                    if (now - s["calm_since"]) >= args.rest and s["hold"] < args.tset:
                        # Adapt step to recent climb rate so late steps stay short.
                        dt = max(now - last_time, 1e-3)
                        rate = max((temp - s["last_t"]) / dt, 0.0)      # degC/s
                        adaptive = max(0.3, min(args.step, rate * 10))  # ~10 s of climb
                        s["hold"] = min(args.tset, temp + adaptive)
                        set_target(ser, ch, term, s["hold"])
                        s["steps"] += 1
                        s["calm_since"] = None
                else:
                    s["calm_since"] = None       # fell back below the landing; re-arm
                s["last_t"] = temp

            # One status line + one CSV row covering every channel.
            parts, cells = [], []
            for ch in channels:
                temp, volt, curr, railed_s = row[ch]
                parts.append(f"CH{ch} T={temp} V={volt} I={curr} hold={st[ch]['hold']:.2f} "
                             f"rail={railed_s:4.1f}s st={st[ch]['steps']} rs={st[ch]['rests']}")
                cells.append(f"{temp},{volt},{curr},{st[ch]['hold']:.2f},{railed_s:.1f},"
                             f"{st[ch]['steps']},{st[ch]['rests']}")
            print(f"[{el:7.1f}s] " + " | ".join(parts) + f"  ERR={err}")
            if logf:
                logf.write(f"{el:.1f}," + ",".join(cells) + f",{err}\n")
                logf.flush()

            if err not in ("", "0"):
                print("ERR latched despite pacing -- power-cycle; try smaller --step / --railmax.")
                break

            if args.secs is not None and el >= args.secs:
                print(f"Reached --secs {args.secs:.0f}s cap; stopping cleanly.")
                break

            last_time = now
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping: disabling channels.")
    finally:
        for ch in channels:
            send(ser, f"EN{ch}=0", term)
        print("Channels disabled (safe).")
        if logf:
            logf.close()
        ser.close()


if __name__ == "__main__":
    main()