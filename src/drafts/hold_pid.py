#!/usr/bin/env python3
"""
TC300B: heat CH to a target and HOLD it steady with tunable PID (Heater mode).

Replaces the bang-bang limit cycle (KP too small + TI far too short -> integral
windup) with a calm, steady hold. KP/TI/TD and a capped VMAX are CLI args so we
can iterate and watch each run's steadiness summary.

TRIP-PROOF heat-up: the "no load" watchdog latches on ~23 s of output pinned
near rail (>= ~0.9*VMAX) CONTINUOUSLY. This script tracks continuous railed
time and, well before that (--railmax, default 10 s), forces a hard EN=0 rest
(--rest-secs) that drops the output to 0 and resets the watchdog timer, then
resumes. Any dip below rail also resets it. So even the aggressive initial
climb cannot trip the latch. The old target-backoff couldn't do this: during
heat-up temp is far below target, so the PID keeps railing no matter what the
target is nudged to.

Scaling on FV 4.04:
  TSETx=n -> n/1000 degC      VMAXx=n -> n/10 V
  PID params are INTEGERS x100:  KP1=400 -> 4.00, TI1=300 -> 3.00, TD1=50 -> 0.50
  (decimals return PARAMETER_ERR; there is no PID1= setter). PIDx? -> "KP TI TD n".
  TI is an integral TIME: larger TI = gentler integral. The stock TI=0.01 is
  absurdly short -> windup -> oscillation; raise it.

Usage:
  python hold_pid.py --port COM3 --ch 1 --tset 25 --kp 4 --ti 3 --td 0 \
      --vmax 24 --secs 180 --log hold_a.csv
"""

import argparse
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed. Run:  pip install pyserial")

BAUD = 115200


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
    """PID params are integers x100 on this firmware (KP1=400 -> 4.00)."""
    return int(round(val * 100))


def set_target(ser, ch, term, deg_c):
    send(ser, f"TSET{ch}={int(round(deg_c * 1000))}", term)   # n/1000 degC


def main():
    ap = argparse.ArgumentParser(description="TC300B heat-and-hold with tunable PID (trip-proof).")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--ch", type=int, choices=[1, 2], default=1)
    ap.add_argument("--tset", type=float, default=25.0, help="target, REAL degC")
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--ti", type=float, default=None)
    ap.add_argument("--td", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=24.0, help="cap max voltage, REAL volts")
    ap.add_argument("--cmax", type=float, default=None,
                    help="cap max current, mA. Limits power and keeps VOLT low (I*R), so the "
                         "output never nears the VMAX rail -> trip-proof by physics, not just by rests.")
    ap.add_argument("--railfrac", type=float, default=0.88,
                    help="VOLT >= railfrac*VMAX counts as 'railed'")
    ap.add_argument("--railmax", type=float, default=10.0,
                    help="continuous railed seconds before a forced EN=0 rest (< ~23 s watchdog)")
    ap.add_argument("--rest-secs", type=float, default=2.0,
                    help="EN=0 rest length that resets the watchdog")
    ap.add_argument("--secs", type=float, default=180.0, help="run duration")
    ap.add_argument("--window", type=float, default=60.0,
                    help="trailing seconds used for the steadiness summary")
    ap.add_argument("--keep", action="store_true", help="leave channel ENABLED on exit")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    ch = args.ch

    try:
        ser = serial.Serial(args.port, BAUD, bytesize=8, parity="N", stopbits=1, timeout=0.2)
    except serial.SerialException as e:
        sys.exit(f"Could not open {args.port}: {e}\n-> Close the Thorlabs GUI first.")

    term = probe_terminator(ser)
    if term is None:
        sys.exit("No response to IDN?.")

    # Heater mode: temperature PID off TSET, sensor stays live.
    if send(ser, f"MOD{ch}?", term) != "0":
        send(ser, f"MOD{ch}=0", term)
    send(ser, f"MOD{ch}?", term, quiet=False)

    err = send(ser, "ERR?", term, quiet=False)
    if err not in ("", "0"):
        sys.exit("ERR latched -- power-cycle the controller, then rerun.")

    # Apply caps + PID BEFORE enabling.
    send(ser, f"VMAX{ch}={int(round(args.vmax * 10))}", term)   # n/10 V
    if args.kp is not None:
        send(ser, f"KP{ch}={pid_int(args.kp)}", term, quiet=False)
    if args.ti is not None:
        send(ser, f"TI{ch}={pid_int(args.ti)}", term, quiet=False)
    if args.td is not None:
        send(ser, f"TD{ch}={pid_int(args.td)}", term, quiet=False)

    vmax = to_float(send(ser, f"VMAX{ch}?", term, quiet=False)) or 24.0
    pid = send(ser, f"PID{ch}?", term, quiet=False)
    rail_v = args.railfrac * vmax
    set_target(ser, ch, term, args.tset)
    send(ser, f"TSET{ch}?", term, quiet=False)
    send(ser, f"EN{ch}=1", term)
    print(f"\nHold {args.tset:.2f} C  (VMAX={vmax:.1f}V, rail>={rail_v:.1f}V, PID={pid}).")
    print(f"Forced EN=0 rest {args.rest_secs:.0f}s after {args.railmax:.0f}s continuous "
          f"rail (trip-proof). Ctrl+C stops.\n")

    logf = open(args.log, "w") if args.log else None
    if logf:
        logf.write("elapsed_s,temp_C,volt_V,curr_mA,railed_s,err\n")

    def rest(t0):
        """Hard EN=0 rest to reset the watchdog; keeps logging while resting."""
        send(ser, f"EN{ch}=0", term)
        r0 = time.time()
        while time.time() - r0 < args.rest_secs:
            v = to_float(send(ser, f"VOLT{ch}?", term)) or 0.0
            t = to_float(send(ser, f"TACT{ch}?", term))
            el = time.time() - t0
            print(f"[{el:6.1f}s]   (rest) V={v} V  T={t} C")
            if logf:
                logf.write(f"{el:.1f},{t},{v},0,0.0,rest\n")
                logf.flush()
            time.sleep(0.4)
        send(ser, f"EN{ch}=1", term)

    t0 = time.time()
    rail_start = None
    samples = []          # (elapsed, temp, volt) for the summary
    try:
        while True:
            temp = to_float(send(ser, f"TACT{ch}?", term))
            volt = to_float(send(ser, f"VOLT{ch}?", term)) or 0.0
            curr = to_float(send(ser, f"CURR{ch}?", term))
            err = send(ser, "ERR?", term)
            now = time.time()
            el = now - t0

            railed = abs(volt) >= rail_v
            rail_start = (rail_start or now) if railed else None
            railed_s = (now - rail_start) if rail_start else 0.0

            print(f"[{el:6.1f}s] T={temp} C  V={volt} V  I={curr} mA  "
                  f"railed={railed_s:4.1f}s  ERR={err}")
            if logf:
                logf.write(f"{el:.1f},{temp},{volt},{curr},{railed_s:.1f},{err}\n")
                logf.flush()
            if temp is not None:
                samples.append((el, temp, volt))

            if err not in ("", "0"):
                print("ERR latched -- disabling and stopping.")
                send(ser, f"EN{ch}=0", term)
                break

            # TRIP-PROOF: reset the watchdog well before it can latch.
            if railed_s >= args.railmax:
                print(f"  -> {railed_s:.0f}s continuous rail; forced EN=0 rest to reset watchdog.")
                rest(t0)
                rail_start = None

            if el >= args.secs:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C.")
    finally:
        win = [s for s in samples if s[0] >= max(0.0, (samples[-1][0] - args.window))] if samples else []
        if len(win) >= 3:
            temps = [t for _, t, _ in win]
            volts = [v for _, _, v in win]
            n = len(temps)
            mean = sum(temps) / n
            var = sum((t - mean) ** 2 for t in temps) / n
            std = var ** 0.5
            print("\n---- steadiness over last "
                  f"{args.window:.0f}s ({n} samples) ----")
            print(f"  target      = {args.tset:.3f} C")
            print(f"  temp mean   = {mean:.3f} C   (offset {mean - args.tset:+.3f})")
            print(f"  temp min/max= {min(temps):.3f} / {max(temps):.3f} C")
            print(f"  peak-peak   = {max(temps) - min(temps):.3f} C")
            print(f"  temp std    = {std:.3f} C")
            print(f"  volt mean   = {sum(volts)/n:.2f} V   "
                  f"min/max {min(volts):.2f}/{max(volts):.2f}")
        if not args.keep:
            send(ser, f"EN{ch}=0", term)
            print("Channel disabled (safe).")
        else:
            print("Channel left ENABLED (--keep).")
        if logf:
            logf.close()
        ser.close()


if __name__ == "__main__":
    main()
