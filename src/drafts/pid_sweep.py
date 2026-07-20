#!/usr/bin/env python3
"""
TC300B PID sweep on an ALREADY-HOT channel.

Context: an external constant-voltage DC base heater now carries most of the
hold power, so the TC300 heater only trims -- it sits at ~6-7 V instead of
railing near 23 V. That is the linear regime where a flat hold is actually
achievable, so it is finally worth sweeping KP/TI/TD.

The channel stays ENABLED for the WHOLE sweep so the temperature never drops
between combos. For each (KP, TI, TD): live-update the gains (they take effect
immediately on the running loop), wait --settle for it to re-settle, then
collect --measure seconds and print a steadiness summary (temp std / peak-peak,
volt mean / rail%). At the end the channel is LEFT ENABLED holding the BEST
gains found (--disable-at-end to stop instead), so the block stays hot.

Safety: with the base heater the output stays far below rail, but if a bad gain
combo winds up and rails for --railmax continuous seconds the combo is aborted
as UNSTABLE (a 1.2 s EN=0 blip resets the watchdog; the base heater holds the
block), then the sweep continues.

Scaling (FV 4.04): TSETx=n -> n/1000 C; VMAXx=n -> n/10 V; PID params are
integers x100 (KP=0.5 -> KP1=50). PIDx? -> "KP TI TD n".
"""
import argparse
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed. Run:  pip install pyserial")

BAUD = 115200

# Sweep grid -- one variable at a time around the 25 C-tuned baseline (0.5,20,5).
# Edit freely. Cost per combo ~ (settle + measure) seconds.
# CH2 grid: re-confirm the CH1 finding (TD=2 sweet spot, smooth-V wins) on the
# second channel while zooming the good region. Includes old default (TD=5) and
# CH1's winner (0.5/20/2) as anchors, plus a KP/TI/TD spread.
GRID = [
    (0.5, 20, 5),    # old default -- CH2 baseline for comparison
    (0.5, 20, 2),    # CH1 winner -- prime candidate
    (0.5, 20, 1),    # TD lower
    (0.25, 20, 2),   # low KP + TD=2
    (0.5, 10, 2),    # stronger integral + TD=2
    (0.75, 20, 2),   # higher KP (in case CH2 wants more gain)
    (0.35, 12, 2),   # CH1 near-tie
    (0.5, 30, 5),    # gentler integral (control probe)
]


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


def pid_int(v):
    return int(round(v * 100))


def stats(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return dict(n=n, mean=mean, std=var ** 0.5, mn=min(vals), mx=max(vals), pp=max(vals) - min(vals))


def main():
    ap = argparse.ArgumentParser(description="TC300B PID sweep (channel stays enabled the whole time).")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--ch", type=int, choices=[1, 2], default=1)
    ap.add_argument("--tset", type=float, default=79.5, help="target, REAL degC")
    ap.add_argument("--vmax", type=float, default=24.0, help="cap max voltage, REAL volts")
    ap.add_argument("--settle", type=float, default=25.0,
                    help="s to wait after a gain change before measuring")
    ap.add_argument("--measure", type=float, default=60.0,
                    help="s of samples per combo used for the steadiness stat")
    ap.add_argument("--period", type=float, default=0.5, help="sample loop period, s")
    ap.add_argument("--warmup-max", type=float, default=90.0,
                    help="max s to wait for temp to reach target before combo 1")
    ap.add_argument("--railfrac", type=float, default=0.88, help="VOLT >= railfrac*VMAX counts as railed")
    ap.add_argument("--railmax", type=float, default=15.0,
                    help="continuous railed s -> abort combo as UNSTABLE")
    ap.add_argument("--disable-at-end", action="store_true",
                    help="disable the channel when the sweep finishes (default: leave enabled on best gains)")
    ap.add_argument("--log", default=None, help="raw per-sample CSV")
    args = ap.parse_args()
    ch = args.ch

    try:
        ser = serial.Serial(args.port, BAUD, bytesize=8, parity="N", stopbits=1, timeout=0.2)
    except serial.SerialException as e:
        sys.exit(f"Could not open {args.port}: {e}\n-> Close the Thorlabs GUI / other scripts first.")

    term = probe_terminator(ser)
    if term is None:
        sys.exit("No response to IDN?.")

    # Heater mode (temp PID off TSET, sensor live).
    if send(ser, f"MOD{ch}?", term) != "0":
        send(ser, f"MOD{ch}=0", term)
    send(ser, f"MOD{ch}?", term, quiet=False)

    err = send(ser, "ERR?", term, quiet=False)
    if err not in ("", "0"):
        sys.exit("ERR latched -- power-cycle the controller, then rerun.")

    send(ser, f"VMAX{ch}={int(round(args.vmax * 10))}", term)
    vmax = to_float(send(ser, f"VMAX{ch}?", term, quiet=False)) or 24.0
    rail_v = args.railfrac * vmax
    send(ser, f"TSET{ch}={int(round(args.tset * 1000))}", term)
    send(ser, f"EN{ch}=1", term)

    print(f"\nSweeping CH{ch} at {args.tset:.2f} C  (VMAX={vmax:.1f} V, rail>={rail_v:.1f} V).")
    print(f"{len(GRID)} combos x (settle {args.settle:.0f}s + measure {args.measure:.0f}s). "
          f"Channel stays ENABLED between combos. Ctrl+C stops.")
    print("Grid (KP, TI, TD): " + ", ".join(str(g) for g in GRID) + "\n")

    logf = open(args.log, "w") if args.log else None
    if logf:
        logf.write("combo,kp,ti,td,phase,elapsed_s,temp_C,volt_V,curr_mA,railed_s,err\n")

    # Warm to target before combo 1 so the first measurement is a settled hold,
    # not the initial rise from the base-heater floor.
    print("Warming to target before sweep...")
    w0 = time.time()
    while time.time() - w0 < args.warmup_max:
        temp = to_float(send(ser, f"TACT{ch}?", term))
        if temp is not None and temp >= args.tset - 0.3:
            print(f"  reached {temp:.3f} C after {time.time()-w0:.0f}s.\n")
            break
        time.sleep(0.5)

    results = []
    t_start = time.time()
    try:
        for idx, (kp, ti, td) in enumerate(GRID):
            send(ser, f"KP{ch}={pid_int(kp)}", term)
            send(ser, f"TI{ch}={pid_int(ti)}", term)
            send(ser, f"TD{ch}={pid_int(td)}", term)
            pid = send(ser, f"PID{ch}?", term)
            print(f"[combo {idx+1}/{len(GRID)}] KP={kp} TI={ti} TD={td}  (PID?={pid})")

            temps, volts = [], []
            rail_start = None
            unstable = False
            measure_start = time.time() + args.settle
            phase_end = measure_start + args.measure
            while time.time() < phase_end:
                temp = to_float(send(ser, f"TACT{ch}?", term))
                volt = to_float(send(ser, f"VOLT{ch}?", term)) or 0.0
                curr = to_float(send(ser, f"CURR{ch}?", term))
                err = send(ser, "ERR?", term)
                now = time.time()
                railed = abs(volt) >= rail_v
                rail_start = (rail_start or now) if railed else None
                railed_s = (now - rail_start) if rail_start else 0.0
                phase = "measure" if now >= measure_start else "settle"
                el = now - t_start
                if phase == "measure" and temp is not None:
                    temps.append(temp)
                    volts.append(volt)
                if logf:
                    logf.write(f"{idx+1},{kp},{ti},{td},{phase},{el:.1f},{temp},{volt},{curr},{railed_s:.1f},{err}\n")
                    logf.flush()
                if err not in ("", "0"):
                    print("  ERR latched during combo -- disabling and stopping sweep.")
                    send(ser, f"EN{ch}=0", term)
                    raise SystemExit("ERR latched (power-cycle needed).")
                if railed_s >= args.railmax:
                    print(f"  -> {railed_s:.0f}s continuous rail; UNSTABLE combo -- blip EN=0 and skip.")
                    unstable = True
                    send(ser, f"EN{ch}=0", term)
                    time.sleep(1.2)
                    send(ser, f"EN{ch}=1", term)
                    rail_start = None
                    break
                time.sleep(args.period)

            ts = stats(temps)
            vs = stats(volts)
            railpct = (100.0 * sum(1 for v in volts if v >= rail_v) / len(volts)) if volts else 0.0
            results.append(dict(i=idx + 1, kp=kp, ti=ti, td=td, ts=ts, vs=vs,
                                railpct=railpct, unstable=unstable))
            if ts and vs:
                tag = "  [UNSTABLE]" if unstable else ""
                print(f"    -> temp mean={ts['mean']:.3f} std={ts['std']:.3f} pp={ts['pp']:.3f} C  "
                      f"| volt mean={vs['mean']:.1f} min/max={vs['mn']:.1f}/{vs['mx']:.1f} rail={railpct:.0f}%{tag}\n")
            else:
                print("    -> (insufficient samples){}\n".format("  [UNSTABLE]" if unstable else ""))
    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping sweep.")
    finally:
        ranked = sorted((r for r in results if r["ts"]), key=lambda r: r["ts"]["std"])
        print("\n==== SWEEP SUMMARY (best temp std first) ====")
        print(f"{'#':>2} {'KP':>5} {'TI':>5} {'TD':>4} {'std_C':>8} {'pp_C':>7} {'Vmean':>7} {'rail%':>6}  flag")
        for r in ranked:
            flag = "UNSTABLE" if r["unstable"] else ""
            print(f"{r['i']:>2} {r['kp']:>5} {r['ti']:>5} {r['td']:>4} "
                  f"{r['ts']['std']:>8.3f} {r['ts']['pp']:>7.3f} {r['vs']['mean']:>7.1f} "
                  f"{r['railpct']:>5.0f}%  {flag}")

        if args.disable_at_end:
            send(ser, f"EN{ch}=0", term)
            print("\nChannel DISABLED (--disable-at-end).")
        elif ranked:
            b = ranked[0]
            send(ser, f"KP{ch}={pid_int(b['kp'])}", term)
            send(ser, f"TI{ch}={pid_int(b['ti'])}", term)
            send(ser, f"TD{ch}={pid_int(b['td'])}", term)
            print(f"\nChannel LEFT ENABLED holding BEST gains: "
                  f"KP={b['kp']} TI={b['ti']} TD={b['td']} (combo {b['i']}, std={b['ts']['std']:.3f} C).")
        else:
            print("\nChannel LEFT ENABLED (no ranked results).")

        if logf:
            logf.close()
        ser.close()


if __name__ == "__main__":
    main()
