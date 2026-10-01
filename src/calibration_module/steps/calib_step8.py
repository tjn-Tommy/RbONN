"""Step 8 -- the two-photon saturation constant, and the gain anchor.

    python src/calibration_module/steps/calib_step8.py --run RUNDIR --dry-run
    python src/calibration_module/steps/calib_step8.py --run RUNDIR
    python src/calibration_module/steps/calib_step8.py --run RUNDIR some.csv

Steps 6 and 7 fit the forward model -- the per-pair ``eta`` and single-beam
response, then the relative phases ``Phi_k``.  Both read at most ~66 mV, where
the atoms are linear to ~1 %, so both are fitted linear and neither can see the
one thing that matters once a real computation drives the full scale: the
readout is ``|E|^2 / (1 + |E|^2 / Y_sat)``, not ``|E|^2``.  On the 0916 ten-pair
run, which reached 500 mV, the linear model read 8 % low at the top and the
whole-run rms error fell from 26 mV to 4 mV once ``Y_sat`` was fitted.

This step measures that one constant, and nothing else.

The sweep
---------
One family, the ladder, in two bands.  ``LADDER`` rungs stratify the BRIGHT
band, :data:`LADDER_FLOOR_V` to :data:`LADDER_CEILING_V`; ``LADDER_LOW`` rungs
stratify everything below it, down to :data:`LADDER_LOW_V`.  Each is a fresh
random drive found by least-move projection onto its level set
(:mod:`calibration_module.fit.levels`), and one more rung solved at the very top
brackets the run, displayed first and last.

Those three ends are per-run settings, not fixed constants -- ``--ceiling-mv``,
``--floor-mv`` and ``--low-mv``, and three boxes on the GUI's Step 8 page
(Sweep max, Band floor, Sweep min).  The module constants are their defaults,
and one of them is a guess the bench has not checked yet: see below.  They are
MILLIVOLTS at both boundaries, because :func:`ladder_levels` is what turns them
into alphas -- an alpha typed by hand would silently follow the PMT gain, which
is the one thing the volts exist to prevent.

38 reads, about five minutes.  There used to be 92 over a quarter of an hour,
spread uniformly from zero to full scale, and two more families besides -- see
"Where the reads go" for why almost all of that was spent on nothing.

The ladder is shuffled before it is driven, because running it in level order
would alias slow drift straight onto the level axis -- the axis being measured.

Where the reads go, and why so few
----------------------------------
The curvature a saturation fit sees grows with ``|E|^2``, and the information a
rung carries about ``1/Y_sat`` goes as ``(Y^2/sigma)^2``.  That is a very steep
weighting: with ``sigma`` set by the ~1 % per-pattern model error it is ``Y^2``,
and with the 0.2 mV systematic floor it is ``Y^4``.  Either way the bottom of
the scale carries essentially none of it -- the old 90-rung uniform ladder spent
40 % of its reads below ``alpha = 0.4``, which between them held about 1 % of the
leverage.

So the bright band gets 30 rungs and everything below it gets 6.  Simulated
through :func:`~calibration_module.fit.verify.fit_y_sat` at the present gain,
that returns ``Y_sat`` to ~4 %, against the ~11 % it would have to be wrong by
before it cost as much as the model error it sits next to.  The old 90 reads
bought 2 % -- five times the precision that is usable, for two and a half times
the run and two and a half times the drift exposure.

The 6 low rungs are not leverage and are not pretending to be.  They are there
because an inference run's MEDIAN read is ~40 mV: without them nothing in this
sweep would touch the decade the machine spends most of its time in, and a
step-6/7 error down there would leave no trace in any file.

Why the ladder stops short of full scale
----------------------------------------
``LADDER_CEILING_V`` is 450 mV and ``Y_max`` is presently ~1534 mV, so the top
of the ladder is ``alpha ~ 0.29``, not 1.  Two separate reasons, neither of them
about the atoms:

* The amplifier rails at :data:`AMP_LIMIT_V`, and its transimpedance stage
  begins to COMPRESS somewhere below that -- where, exactly, is not known.  Both
  bend the top of the ladder down in the same way saturation does, and
  :func:`~calibration_module.fit.verify.fit_y_sat` would absorb either into
  ``Y_sat`` and report a confident wrong number.  The raw trace peak runs ~1.2x
  the mean, so 450 mV keeps the peak near 59 % of the rail.
* Nothing downstream ever goes there.  An inference run's brightest read is
  ~400 mV, ``alpha ~ 0.26``.  Driving the ladder to full scale would measure the
  saturation of a regime the computation never enters, at the price of running
  the amplifier where it is least trustworthy.

The model being fitted is ``1/y = 1/Y + 1/Y_sat`` -- a fixed functional form, not
a local expansion -- so a ``Y_sat`` measured with its leverage at 450 mV is
exactly as valid at 260 mV.  What matters is that the ladder BRACKETS the range
the machine uses, and 450 mV clears 400 mV.  That also makes 450 mV close to a
floor rather than a safe default: much lower and the ladder stops covering the
computation at all.

Use the probe (see below) to find out where the electronics actually stop being
linear; that measurement is what ``LADDER_CEILING_V`` is supposed to encode.
When it is made, it goes in the Sweep max box (or ``--ceiling-mv``) -- the
constant is only the default. Past ~770 mV the GUI asks first, since 1.31x is
the measured p90 peak/mean and that is where the raw PEAK starts touching the
rail -- on every bright rung, not on one probe read.

The run assumes no Y_sat
------------------------
Every prediction driven here -- ``pred_v``, the input range, the read window --
is the LINEAR model.  Fitting a saturation constant from a sweep that already
assumed one would be circular.  The linear prediction is the larger of the two,
so the input range it picks is always at least wide enough and a read can never
clip because of the choice.

The gain anchor, and what Y_sat actually means
----------------------------------------------
The fit has one free parameter.  The detector gain is pinned to ``g = 1``, and
that is a DEFINITION, not an assumption that the gain is 1.  With ``g`` pinned,
any gain error is absorbed into ``Y_sat`` and levered by roughly
``Y_sat / Y_max`` -- about 12x here.  Measured on the 0917 bench: over one night
of hourly runs a ``g = 1`` ``Y_sat`` swung 4.39..5.44 V while tracking the
independently measured gain at r = 0.98, and with the gain left free it sat at
6.84 +- 0.20 V with no trend at all.

So this run's ``Y_sat`` and this run's reads are one object.  Downstream, a
computation re-drives a handful of this sweep's own ladder patterns -- they are
in the CSV row by row, and the seed is in the JSON -- and takes the ratio of its
reads to this run's reads on the same drives.  That ratio is ``g``, free of
``Y_sat``, of ``eta`` and of the phases, because the SLM frame is identical.
The prediction downstream is then::

    Y_pred = g * Y / (1 + Y / Y_sat)

with ``Y`` the linear model and ``Y_sat`` fixed at this run's value.  Gain
outside the saturation, on physical grounds: the atoms saturate on the light
they receive, and the PMT amplifies afterwards.  The 0917 data cannot tell that
apart from ``g * Y / (1 + g Y / Y_sat)`` -- the two differ by 0.45 mV at worst
against a 4 mV floor -- so the choice is made on physics, not on fit quality.

What Y_sat cannot fix.  About 3.9 mV of the residual on the 0917 runs is
per-pattern model error that neither ``g`` nor ``Y_sat`` touches, and the gain
wanders ~1 %/h bidirectionally, worth ~1.4 mV per 1 %.  Refreshing ``g`` every
15 minutes keeps that second term at ~12 % of the first.

Input range
-----------
One range, :data:`~calibration_module.measure.pair_v2.RANGE_V`, for every read
including the darks -- the same one steps 6 and 7 use.  Reads used to get the
range their prediction called for, which put the board's per-range offset
difference into every dark subtraction that crossed a boundary, and a near-rail
trace used to trigger a re-read one range wider.  Both are gone; ``range_v`` in
the CSV records the one range, and ``peak_v`` still records the raw trace peak
so a flat-top is visible after the fact.

Alpha is coherent, and single_beam is subtracted
------------------------------------------------
``FS = Y_max`` excludes the single-beam background, so every level here is on
``|E|^2`` alone and the fit subtracts the model's background from each read.
It is small but it RAMPS -- 0.39 -> 1.13 mV bottom to top on the 0911 run -- and
a ramp coherent with level is exactly what a saturation fit is looking for, so
leaving it in would bias ``Y_sat``.  The raw ``voltage_mean_v`` and ``dark_v``
are never touched; ``sb_pred_v`` is a column and the subtraction happens in the
fit.

The probe
---------
:func:`probe_verdict` and :func:`print_probe` report ONE displayed level read
once -- nothing written -- against the linear model and against
:data:`AMP_LIMIT_V`.  The GUI's Step 8 page drives it: pick an alpha, see the
predicted millivolts before committing, display and read.

That is the tool for the question ``LADDER_CEILING_V`` answers: walk alpha up
and watch when the read starts falling behind the linear prediction faster than
one constant can explain, or when the raw peak approaches the rail.  Run it
after any change in optical power or PMT gain, because ``Y_max`` scales with
both and the amplifier does not.

It replaces a check that could only drive ``alpha = 1``.  That was safe while
``Y_max`` fitted under the amplifier, and it no longer does: at the present gain
the check would have clipped by construction, then inverted the clipped read to
report a confident ceiling.  A caller-chosen alpha cannot do that.

Timing matters more than the cap
--------------------------------
``pred_v`` here is step 6/7's ``eta`` IN STEP 6/7's VOLTS, and
:func:`~calibration_module.fit.verify.fit_y_sat` pins ``g = 1`` with no free
scale.  So any gain difference between when ``eta`` was measured and when this
sweep reads is absorbed entirely into ``Y_sat``, levered by ``Y_sat / top read``
-- about 15x at the present ceiling.  A 1 % gain drift between step 7 and step 8
moves ``Y_sat`` by 28 %, a 2 % drift by 72 %, and the rmse barely moves in
either case, so the fit looks clean while it is wrong.  Only past ~5 % does
``unbounded`` fire.

The gain wanders ~1 %/h, so every hour between step 7 and step 8 costs ~15 % of
``Y_sat``.  RUN THIS IMMEDIATELY AFTER STEP 7, in the same sitting.  The 0917
observation quoted below -- a ``g = 1`` ``Y_sat`` swinging 4.39..5.44 V overnight
while tracking gain at r = 0.98 -- is this effect, measuring how stale step 7 had
become, and not a property of the atoms.

Read window
-----------
One window, :data:`~calibration_module.measure.pair_v2.T_MEAS_S`, for every read
including the darks -- the same one steps 6 and 7 use.  This step used to ramp
it from 10 s at the dimmest prediction down to 5 s at the brightest, on the
argument that a dim read needs the averaging.  It does, but a LADDER is read
point against point: a window that varies along the level axis puts a
window-dependent systematic on exactly the axis :func:`fit_y_sat` reads.

PMT temperature
---------------
``--pmt`` logs the PMT's thermistor (Keithley 2400, ohms and TH10K degrees)
next to every read.  The 2400 settings are imported from
``drafts/daq_pmt_monitor.py``, which owns them; the reading is cached for
``--pmt-period`` seconds (default 60), since the bead's time constant is
minutes and a per-read GPIB round trip would add minutes to a run.  The run
brackets itself with a reading in the CSV meta, and a 2400 that fails blanks
the two columns rather than ending the run.

Output
------
``calib_step8_patterns_<MMDD_HHMM>.csv``   every pattern, written BEFORE the
                                           hardware is touched, so an
                                           interrupted run is still
                                           reproducible.  A CSV and not a JSON
                                           because a run directory holds exactly
                                           one JSON, and a pattern list is a
                                           table anyway; the scalars it was
                                           solved under ride in ``#`` meta lines.
``calib_step8_<MMDD_HHMM>.csv``            one row per read, partial if stopped.
``calib_step8_<MMDD_HHMM>_ysat.png``       the three-panel report.

The fitted ``Y_sat`` goes into the run's ``calib.json`` as its ``step8``
section -- what everything downstream reads.

All of them go into ``--run`` as soon as they exist: a script run is a run
somebody asked for.  The GUI is the other way round -- it stages a sweep in a
temp directory and writes into the run directory only when Save is pressed, so
an abandoned run does not have to be recognised and deleted later.

Not a unittest (it needs real hardware) -- run it directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.levels import (  # noqa: E402
    Pattern,
    full_scale_pattern,
    level_patterns,
)
from calibration_module.fit.verify import (  # noqa: E402
    full_scale,
    load_forward_model,
    predict,
    write_plan_csv,
    write_verify_csv,
)
from calibration_module.measure.bench import connect_daq, connect_slm  # noqa: E402
from calibration_module.measure.patterns import (  # noqa: E402
    DARK_EVERY_S,
    DEFAULT_ACQ,
    load_layout,
    range_for,
    run_patterns,
    select_pairs,
)
from calibration_module.measure.pmt_temp import (  # noqa: E402,F401
    PMT_PERIOD_S,
    PmtTemperature,
)
from calibration_module.calib_file import load_calib, write_section  # noqa: E402

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"   # the run directory; --run sets it

# The run directory is NOT here -- it is the required --run argument, as it is
# for steps 6 and 7.

PAIRS = None                # None -> every pair the run's calib.json calibrates; --p overrides
PHASE_METHOD = None         # stored step-7 fit; None -> the single one a v2 run has

SLM_DISPLAY_NO = None       # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1              # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

#: The bench acquisition, unchanged -- same window, range and settle as steps 6
#: and 7.  This step declares no acquisition of its own.
ACQ = DEFAULT_ACQ

# ---- The run ----
LADDER = 30                     # rungs in the bright band, one per stratum
LADDER_LOW = 6                  # rungs below it -- forward-model checks, not leverage
SEED = None                     # None -> a fresh seed, printed and written to the plan CSV

# ---- Where the ladder starts and stops ----
#
# All three are VOLTS, not fractions of Y_max, because every argument that set
# them is about volts: the amplifier, the noise floor, and how bright an
# inference run actually reads.  :func:`ladder_levels` turns them into the alphas
# a given forward model needs, so the geometry survives a change of PMT gain.
#
#: The brightest read the ladder is allowed to take, and the only thing that
#: decides where the sweep stops -- :func:`ladder_levels` divides it by ``Y_max``
#: and :func:`build` uses the result as ``alpha_hi``.
#:
#: 450 mV IS AN UNVERIFIED WORKING FIGURE, not a measurement.  Nobody has
#: measured where this transimpedance amplifier starts to COMPRESS; only its hard
#: rail is known, :data:`AMP_LIMIT_V` = 1 V.  That matters because compression
#: bends the top of the ladder down in exactly the way saturation does, the fit
#: has one free parameter, and it cannot tell them apart -- so it would absorb
#: the amplifier into ``Y_sat`` and quote a confident wrong number.  Use the
#: probe (:func:`probe_verdict`) to replace the guess: walk alpha up and watch
#: for the read falling behind the linear model faster than one constant can
#: explain, and for the raw peak approaching the rail.
#:
#: Two reasons 450 is the right guess to make MEANWHILE.  The errors are
#: asymmetric: too low costs precision only -- 700 -> 450 mV moved sigma(Y_sat)
#: from 3.0 % to 3.9 %, against the ~11 % where it would start to matter at all
#: -- while too high is silent, for the reason above.  And it is near a hard
#: floor: an inference run's brightest read is ~400 mV, and a ladder stopping
#: below that stops covering the range the machine computes in.  So the usable
#: range is roughly 400..700 mV and this is the bottom of it.
#:
#: For scale, 450 mV keeps the raw trace peak near 590 mV (the 0929 sweep's p90
#: peak/mean is 1.31) = 59 % of the rail.
LADDER_CEILING_V = 0.450
#: The bright band's floor.  A rung's information about ``1/Y_sat`` goes as
#: ``(Y^2/sigma)^2``, so the bottom of the scale carries almost none of it: the
#: 90-rung ladder this replaces spent 40 % of its reads on 1 % of its leverage.
LADDER_FLOOR_V = 0.230
#: Below :data:`LADDER_FLOOR_V` the rungs are FORWARD-MODEL CHECKS rather than
#: ``Y_sat`` leverage -- they carry no curvature worth fitting, and they are here
#: because an inference run's MEDIAN read is ~40 mV.  Without them nothing in
#: this sweep would touch the decade the machine spends most of its time in, and
#: a step-6/7 error down there would leave no trace.
LADDER_LOW_V = 0.031

# ---- The amplifier ceiling ----
#: The transimpedance amplifier saturates here, at EVERY PMT gain and on every
#: board range.  It is a property of the electronics, not of the atoms, and it is
#: the one hard limit in the chain: a read flat-topped at this value measures
#: nothing.  It matters most to this step, because step 8 is the only step that
#: deliberately drives the top of the scale -- and because a clipped ceiling read
#: bends the top of the ladder down in exactly the way real saturation does, so
#: :func:`~calibration_module.fit.verify.fit_y_sat` would absorb it into
#: ``Y_sat`` and report a confident wrong number.
AMP_LIMIT_V = 1.0

OUT_DIR = CALIB_PATH


# ======================================================================
# building the run
# ======================================================================

def run_seconds(n_reads: int) -> float:
    """Estimated wall time: every window and settle, plus the periodic darks."""
    secs = n_reads * (ACQ.t_meas_s + ACQ.settle_s)
    return secs + ACQ.t_meas_s * (1 + secs // DARK_EVERY_S)


def ladder_levels(y_max_v: float, *, ceiling_v: float = LADDER_CEILING_V,
                  floor_v: float = LADDER_FLOOR_V,
                  low_v: float = LADDER_LOW_V) -> dict:
    """The ladder's three voltages, as alphas for this particular forward model.

    ``alpha`` is a fraction of ``Y_max``, which moves with the PMT gain and with
    the optical power; the voltages do not.  Converting here rather than storing
    alphas is what lets one set of constants describe the run at any gain: turn
    the PMT down and the alphas rise to keep the ladder on the same volts.

    The three are ARGUMENTS, defaulting to the module constants, because the
    ceiling is a guess about the amplifier that the probe is meant to replace
    (see :data:`LADDER_CEILING_V`) -- so the GUI and the CLI can move the band
    without editing the module, and the run records what it actually swept.
    They stay VOLTS at the boundary for the same reason they are volts here: an
    alpha typed by hand would silently follow the PMT gain.

    Clamped so the bands stay ordered and non-empty even for a forward model
    dimmer than ``ceiling_v``, where the cap saturates at 1.
    """
    fs = float(y_max_v)
    ceiling, band, check = float(ceiling_v), float(floor_v), float(low_v)
    if not fs > 0:
        return {"cap": 1.0, "floor": 0.5, "low": 0.1, "capped": False}
    cap = min(1.0, ceiling / fs)
    floor = min(band / fs, 0.9 * cap)
    low = min(check / fs, 0.9 * floor)
    return {"cap": cap, "floor": floor, "low": low,
            # True -> the amplifier, not the atoms, decides where this run stops
            "capped": bool(ceiling < fs)}


def ceiling_pattern(models, phases, pairs, cap: float, *, seed: int) -> Pattern:
    """The single rung the run is bracketed with, at the top of the ladder.

    ONE pattern, displayed at both ends: the bracket measures drift, and two
    different drives at the same level would measure model error instead.

    At ``cap == 1`` the level set is a single point and :func:`full_scale_pattern`
    has it in closed form.  Below that the set has diversity, so the bracket is an
    ordinary solved rung -- it keeps ``kind="F"`` so the CSV still says which read
    is the bracket, and so the ONN's anchor picker still recognises it.
    """
    if cap >= 1.0 - 1e-9:
        return full_scale_pattern(models, phases, pairs, label="ceiling")
    rung, = level_patterns(models, phases, pairs, 1, seed=seed,
                           alpha_lo=cap, alpha_hi=cap)
    return replace(rung, kind="F", label="ceiling")


def interleave(ladder: Sequence[Pattern], ceiling: Pattern, *, seed: int) -> list[Pattern]:
    """Shuffle the ladder; bracket it with the ceiling drive at both ends.

    Shuffled, because driving the ladder in level order would put slow drift on
    the level axis -- the axis the fit reads.  Bracketed, because the same frame
    at both ends of the run is the only read here that measures drift rather
    than model error: the two differ by the gain, and by nothing else.
    """
    rng = np.random.default_rng(seed)
    middle = [ladder[i] for i in rng.permutation(len(ladder))]
    return ([replace(ceiling, label=ceiling.label + "@start")]
            + middle
            + [replace(ceiling, label=ceiling.label + "@end")])


def build(models, phases, pairs, *, seed: int, ladder: int = LADDER,
          low: int = LADDER_LOW, ceiling_v: float = LADDER_CEILING_V,
          floor_v: float = LADDER_FLOOR_V, low_v: float = LADDER_LOW_V):
    """``(scale, patterns, preds)`` -- the whole run, solved, in read order.

    Two bands, one ladder.  ``ladder`` rungs stratify the bright band, which is
    where all of the ``Y_sat`` leverage is, and ``low`` rungs stratify everything
    below it, where there is none -- those exist to catch a step-6/7 error in the
    decade an inference run actually computes in.  Both are shuffled together by
    :func:`interleave`, so neither band is a block in time.

    ``ceiling_v``, ``floor_v`` and ``low_v`` are where the two bands end, in
    VOLTS -- :func:`ladder_levels` turns them into this model's alphas.

    ``preds`` are LINEAR: this run fits ``Y_sat``, so it must not assume one.
    """
    scale = full_scale(models, phases, pairs)
    lv = ladder_levels(scale.fs, ceiling_v=ceiling_v, floor_v=floor_v,
                       low_v=low_v)
    rungs = list(level_patterns(models, phases, pairs, ladder, seed=seed,
                                alpha_lo=lv["floor"], alpha_hi=lv["cap"]))
    if low:
        rungs += list(level_patterns(models, phases, pairs, low, seed=seed + 1,
                                     alpha_lo=lv["low"], alpha_hi=lv["floor"]))
    patterns = interleave(
        rungs, ceiling_pattern(models, phases, pairs, lv["cap"], seed=seed + 3),
        seed=seed + 2)
    return scale, patterns, [float(predict(models, phases, pairs, p.x, p.w))
                             for p in patterns]


# ======================================================================
# collect
# ======================================================================

def collect(run: Path, *, pairs=None, method=None, out_dir=None,
            ladder: int = LADDER, low: int = LADDER_LOW,
            ceiling_v: float = LADDER_CEILING_V,
            floor_v: float = LADDER_FLOOR_V, low_v: float = LADDER_LOW_V,
            seed: int | None = None,
            dry_run: bool = False, pmt: "PmtTemperature | None" = None) -> str | None:
    """Solve the sweep, write its JSON, drive it, write the CSV."""
    out = Path(OUT_DIR if out_dir is None else out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = select_pairs(PAIRS if pairs is None else pairs)
    models, phases, ref = load_forward_model(run, pairs=wanted, method=method)
    pairs = sorted(models)
    if seed is None:
        seed = SEED if SEED is not None else int(np.random.SeedSequence().entropy % 2**32)

    scale, patterns, preds = build(models, phases, pairs, seed=seed,
                                   ladder=ladder, low=low, ceiling_v=ceiling_v,
                                   floor_v=floor_v, low_v=low_v)
    lv = ladder_levels(scale.fs, ceiling_v=ceiling_v, floor_v=floor_v,
                       low_v=low_v)
    coh = [float(predict(models, phases, pairs, p.x, p.w, coherent=True))
           for p in patterns]
    secs = run_seconds(len(patterns))

    load_calib(run).print_chain()
    print(f"Pairs     : {pairs}  (N = {len(pairs)}, reference {ref})")
    print(f"Full scale: Y_max {scale.fs * 1e3:.4f} mV   S^2 {scale.bound * 1e3:.4f} mV "
          f"({100 * scale.headroom:.1f} %)   psi* {np.degrees(scale.psi):.2f} deg")
    print(f"            4-bit level step = {scale.fs / 15 * 1e3:.3f} mV (linear)")
    print(f"Model     : LINEAR -- this run fits Y_sat, so it assumes none")
    print(f"Read      : {ACQ.t_meas_s:g} s every read, dark included, at "
          f"+/-{ACQ.range_v:g} V  (dimmest {min(preds) * 1e3:.2f} mV, "
          f"brightest {max(preds) * 1e3:.2f} mV)")
    print(f"Ladder    : {ladder} rungs over "
          f"{lv['floor'] * scale.fs * 1e3:.0f}..{lv['cap'] * scale.fs * 1e3:.0f} mV "
          f"(alpha {lv['floor']:.3f}..{lv['cap']:.3f}), plus {low} model checks "
          f"down to {lv['low'] * scale.fs * 1e3:.0f} mV"
          + ("" if lv["capped"] else "   [Y_max is under the ceiling: no cap]"))
    print(f"Patterns  : {len(patterns)} reads  (ladder {ladder + low}, "
          f"ceiling 2), ~{secs / 60:.0f} min")
    print(f"Seed      : {seed}")
    fell_back = [p.label for p in patterns if not p.exact]
    if fell_back:
        print(f"            {len(fell_back)} pattern(s) fell back to the "
              f"deterministic drive: {', '.join(fell_back[:6])}"
              + (" ..." if len(fell_back) > 6 else ""))

    stamp = time.strftime("%m%d_%H%M")
    plan_path = write_plan_csv(
        out / f"calib_step8_patterns_{stamp}.csv", patterns, preds, coh,
        pairs=pairs, scale=scale,
        meta={"step7_ran": load_calib(run).ran(7),
              "reference": ref, "method": method, "seed": seed,
              "Y_max_v": scale.fs, "S2_v": scale.bound, "psi_rad": scale.psi,
              "level_step_v": scale.fs / 15.0,
              "ladder": ladder, "ladder_low": low,
              "ceiling_v": ceiling_v, "floor_v": floor_v, "low_v": low_v,
              "alpha_floor": lv["floor"], "alpha_cap": lv["cap"],
              "alpha_low": lv["low"], "alpha_capped": lv["capped"],
              "t_meas_s": ACQ.t_meas_s, "range_v": ACQ.range_v})
    print(f"\nPatterns written to {plan_path}")

    if dry_run:
        report_plan(patterns, preds, coh, scale.fs, ceiling_v=ceiling_v,
                    floor_v=floor_v, low_v=low_v)
        return None

    return _drive(run, pairs, patterns, preds, coh, models, phases,
                  out=out, stamp=stamp, plan_path=plan_path,
                  meta={"n_pairs": len(pairs), "seed": seed, "y_max_v": scale.fs},
                  pmt=pmt)


def _drive(run, pairs, patterns, preds, coh, models, phases, *,
           out: Path, stamp: str, plan_path: Path, meta: dict,
           columns: Sequence[str] = (), extra_of=None,
           pmt: "PmtTemperature | None" = None):
    """Display every pattern in read order and write the CSV, partial or not.

    ``extra_of(i, pattern)`` adds per-read columns beyond the standard ones;
    ``columns`` names them, in CSV order.  Every read takes ``ACQ.t_meas_s``.

    ``pmt``, if given, adds its two columns to every read and brackets the run
    with a reading in the meta.
    """
    x = np.stack([p.x for p in patterns])[None]      # (1 block, P, n)
    w = np.stack([p.w for p in patterns])[None]
    extra = [[{"kind": p.kind, "label": p.label, "seq": i,
               "target_alpha": f"{p.target_alpha:.6f}",
               "alpha": f"{p.alpha:.6f}",
               "coherent_v": f"{c:.9g}", "sb_pred_v": f"{q - c:.9g}",
               "t_meas_s": f"{ACQ.t_meas_s:.3f}",
               **(extra_of(i, p) if extra_of else {})}
              for i, (p, c, q) in enumerate(zip(patterns, coh, preds))]]

    meta = dict(meta)
    if pmt is not None:
        columns = [*columns, *PmtTemperature.COLUMNS]
        first = pmt.read()          # also seeds the cache for the first patterns
        meta["pmt_resource"] = pmt.resource
        meta["pmt_period_s"] = f"{pmt.period_s:g}"
        meta.update({f"{k}_start": v for k, v in first.items()})
        print(f"PMT       : {pmt.resource}  "
              + (f"{first['pmt_c']} C ({first['pmt_ohm_Ohm']} Ohm)"
                 if first["pmt_c"] else "unavailable")
              + f", resampled every {pmt.period_s:g} s")

    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=ACQ.t_meas_s, t_single=ACQ.t_meas_s,
                      min_val=-ACQ.range_v, max_val=ACQ.range_v)
    layout = load_layout(run, pairs)
    rows: list[dict] = []
    status = "complete"
    csv_path = None
    try:
        run_patterns(daq, slm, layout, [tuple(pairs)], x, w, models, phases,
                     rows=rows, extra=extra, y_sat=None,
                     row_hook=pmt)
    except BaseException as exc:
        status = (f"partial ({type(exc).__name__} after {len(rows)} of "
                  f"{len(patterns)} reads)")
        raise
    finally:
        try:
            if pmt is not None:
                # The 2400 is independent of the SLM and DAQ, so this last
                # reading is worth having even when the run died mid-way.
                meta.update({f"{k}_end": v for k, v in pmt.read().items()})
            if rows:            # on disk before anything else can fail
                csv_path = write_verify_csv(
                    out / f"calib_step8_{stamp}.csv", pairs, rows,
                    meta={"step7_ran": load_calib(run).ran(7),
                          "patterns_csv": Path(plan_path).name,
                          "status": status, **meta},
                    extra=["kind", "label", "seq", "target_alpha", "alpha",
                           "coherent_v", "sb_pred_v", "t_meas_s", "peak_v",
                           *columns])
                print(f"\nCSV ({len(rows)} reads, {status}) written to {csv_path}")
        finally:
            slm.close_slm()
            daq.disconnect()
    return csv_path


# ======================================================================
# the probe -- display one level, read it once
# ======================================================================
#
# One drive, one read, nothing written.  This is how the bench finds out where
# the readout stops being linear, which is a question about the ELECTRONICS and
# not about the atoms.  Raise the optical power or the PMT gain and the brightest
# reads clip -- or merely COMPRESS, well before the rail -- and either bends the
# top of the ladder down in exactly the way saturation does.  The fit has one
# free parameter and no way to tell them apart, so what comes out is a ``Y_sat``
# that looks clean and is wrong.
#
# This replaces a check that could only ever drive ``alpha = 1``.  That was fine
# while ``Y_max`` fitted under the amplifier, and it does not: at the present
# gain ``Y_max`` is ~1534 mV against a 1 V rail, so the old check clipped by
# construction and then reported a confident ceiling computed FROM the clipped
# read.  An alpha the caller picks cannot do that, and it does strictly more --
# walking alpha up until the read falls behind the linear model is the
# measurement that decides :data:`LADDER_CEILING_V` in the first place.

def probe_verdict(*, alpha: float, y_max_v: float, coherent_v: float,
                  sb_pred_v: float, mean_v: float, dark_v: float, std_v: float,
                  peak_v: float = float("nan"), range_v: float = float("nan"),
                  limit_v: float = AMP_LIMIT_V,
                  ceiling_v: float = LADDER_CEILING_V) -> dict:
    """One displayed level and one read: what the bench did with it.

    ``coherent_v`` and ``sb_pred_v`` are the LINEAR predictions for the drive
    that was displayed -- :func:`~calibration_module.fit.verify.predict` with
    ``coherent=True``, and the single-beam background it leaves out.  Saturation
    acts on the coherent part alone (the background is detector response to each
    beam separately, not TPA field), so inverting the one read gives an
    indicative ``Y_sat``::

        Y_sat = coh * m / (coh - m),    m = (mean - dark) - sb

    ONE read, so that is a preview of what the sweep will fit rather than a
    replacement for it -- and it means nothing at all if the read was clipped.

    Clipping is called on the RAW trace peak, never on the mean: the 20 Hz
    low-pass pulls a flat-top back under the rail, so a clipped read still
    reports a perfectly reasonable mean.  ``peak_v`` is the only number here that
    sees it, which is why it is a column in every step-8 CSV.
    """
    coh = float(coherent_v)
    meas = float(mean_v) - float(dark_v)
    m = meas - float(sb_pred_v)

    if 0.0 < m < coh:
        y_sat = coh * m / (coh - m)
    elif m >= coh:
        y_sat = float("inf")            # read met or beat the linear model
    else:
        y_sat = float("nan")            # nothing was there

    limit = float(limit_v)
    ceiling = float(ceiling_v)
    peak = float(peak_v)
    near = max(peak if np.isfinite(peak) else 0.0, meas)
    pred = coh + float(sb_pred_v)
    lv = ladder_levels(y_max_v, ceiling_v=ceiling)
    return {
        "alpha": float(alpha),
        "y_max_v": float(y_max_v),
        "limit_v": limit,
        "ceiling_v": ceiling,
        "alpha_cap": lv["cap"],
        "coherent_pred_v": coh,
        "sb_pred_v": float(sb_pred_v),
        "pred_v": pred,
        "mean_v": float(mean_v), "dark_v": float(dark_v), "std_v": float(std_v),
        "measured_v": meas,
        "measured_coherent_v": m,
        "peak_v": peak,
        "peak_fraction": (peak / limit) if limit else float("nan"),
        "read_range_v": float(range_v),
        "shortfall": (1.0 - meas / pred) if pred else float("nan"),
        "y_sat_implied_v": y_sat,
        "over_ceiling": bool(meas > ceiling),
        "clipped": bool(near >= 0.98 * limit),
        # TIGHT is about the raw peak, not the mean: compression starts before
        # the rail does, and the peak is what reaches it first.
        "verdict": ("CLIPPED" if near >= 0.98 * limit
                    else "TIGHT" if near >= 0.85 * limit
                    else "OK"),
    }


def print_probe(d: dict, log=print) -> None:
    """The probe block: what was asked for, what came back, and what it implies."""
    fs = d["y_max_v"]
    log(f"alpha {d['alpha']:.4f} of Y_max {fs * 1e3:.1f} mV   "
        f"(ladder cap {d['alpha_cap']:.3f} = {d['ceiling_v'] * 1e3:.0f} mV)")
    log(f"predicted         {d['pred_v'] * 1e3:9.3f} mV  (LINEAR, of which "
        f"{d['sb_pred_v'] * 1e3:.3f} mV is single beam)")
    log(f"measured          {d['measured_v'] * 1e3:9.3f} mV  "
        f"+/-{d['std_v'] * 1e3:.3f}  (dark {d['dark_v'] * 1e3:+.4f} mV, "
        f"range {d['read_range_v']:g} V)")
    log(f"vs prediction     {100 * d['shortfall']:+9.2f} %")
    log(f"raw trace peak    {d['peak_v'] * 1e3:9.3f} mV  = "
        f"{100 * d['peak_fraction']:.1f} % of the {d['limit_v']:g} V amplifier "
        f"limit")
    y = d["y_sat_implied_v"]
    if np.isfinite(y):
        log(f"implied Y_sat     {y:9.3f} V   (ONE read -- a preview of the fit, "
            f"and meaningless if clipped)")
    else:
        log("implied Y_sat         none   (the read met or beat the linear model)")
    log(f"verdict           {d['verdict']:>9}"
        + ("   the raw trace reached the rail: this read is a flat-top"
           if d["clipped"] else
           "   the raw peak is close to the rail" if d["verdict"] == "TIGHT"
           else ""))
    if d["over_ceiling"]:
        log(f"                            this read is above the ladder's "
            f"ceiling ({d['ceiling_v'] * 1e3:.0f} mV) -- brighter than the "
            f"sweep goes")


# ======================================================================
# the fit
# ======================================================================

def fit_and_write(csv_path, run, *, pairs=None, method=None, out_dir=None,
                  log=print) -> dict:
    """Fit ``Y_sat`` on a step-8 CSV, write the step8 section and the PNG.

    The section records the CSV it came from, because a ``Y_sat`` and the reads
    that defined ``g = 1`` are one object: a later run measures its gain against
    THOSE reads, so quoting the constant without saying which sweep produced it
    is how the two get mixed.
    """
    from calibration_module.fit.step8_report import report_csv

    csv_path = Path(csv_path)
    out = Path(run if out_dir is None else out_dir)
    result = report_csv(csv_path, run, pairs=pairs, method=method,
                        out_dir=out, log=log)

    section = {
        "y_sat_v": result["y_sat_v"],
        "y_sat_sigma_v": result["sigma_v"],
        "y_sat_lo_v": result["lo_v"], "y_sat_hi_v": result["hi_v"],
        "unbounded": result["unbounded"],
        "gain_lever": result["gain_lever"],
        "rmse_v": result["rmse_v"], "rmse_linear_v": result["rmse_linear_v"],
        "n_reads": result["n"], "n_ladder": result["n_ladder"],
        "png": None if not result.get("png") else Path(result["png"]).name,
        "full_scale": result.get("full_scale", {}),
        # g = 1 is DEFINED by the reads in this section's csv.  A later run
        # divides its reads on a subset of those same drives by them to get its
        # gain, so the convention is recorded with the constant it qualifies.
        "gain_reference": {"convention": "g = 1 at this sweep"},
    }
    calib = write_section(run, 8, section, csv=csv_path, log=None)
    # Same reason the PNG name is in the section: the caller that asked for a
    # different out_dir should not have to re-derive it from the CSV.
    result["result_json"] = str(calib.path)
    log(f"Y_sat written to {calib.path}")
    return result


def report_plan(patterns, preds, coh, fs: float, *,
                ceiling_v: float = LADDER_CEILING_V,
                floor_v: float = LADDER_FLOOR_V,
                low_v: float = LADDER_LOW_V) -> None:
    """The dry-run table: what will be driven, in read order, and how bright.

    ``wants`` is the range :func:`range_for` WOULD pick per read.  Every read is
    actually taken at the fixed ``ACQ.range_v``, so a row wanting more than that
    is a row the amplifier cannot hold either -- they are the same 1 V.
    """
    step = fs / 15.0
    print(f"\n  {'seq':>4} {'kind':>4} {'label':<22} {'alpha':>7} "
          f"{'coh (mV)':>10} {'pred (mV)':>10} {'sb (mV)':>8} {'wants':>7}"
          f"   (every read +/-{ACQ.range_v:g} V, {ACQ.t_meas_s:g} s)")
    for i, (p, c, q) in enumerate(zip(patterns, coh, preds)):
        rng = range_for(q)[0]
        print(f"  {i:4d} {p.kind:>4} {p.label:<22} {p.alpha:7.4f} "
              f"{c * 1e3:10.4f} {q * 1e3:10.4f} {(q - c) * 1e3:8.4f} "
              f"{rng:6.1f}V" + ("" if p.exact else "  [fallback]"))

    a = np.array([p.alpha for p in patterns if p.kind == "A"])
    used = np.array([range_for(q)[0] for p, q in zip(patterns, preds)
                     if p.kind == "A"])
    sb = np.array([q - c for p, q, c in zip(patterns, preds, coh) if p.kind == "A"])
    lv = ladder_levels(fs, ceiling_v=ceiling_v, floor_v=floor_v, low_v=low_v)
    bright = a[a >= lv["floor"] - 1e-9]
    # Binned over the LADDER's own span, not over [0, Y_max].  The sweep stops at
    # LADDER_CEILING_V on purpose, so a histogram against full scale would report
    # eleven empty levels as if they were a gap rather than the design.
    cap = max(lv["cap"], 1e-9)
    every = np.array([p.alpha for p in patterns])
    hist = np.bincount(np.clip((every / cap * 15 + 1e-9).astype(int), 0, 15),
                       minlength=16)
    print(f"\n  ladder alpha : min {a.min():.4f}  median {np.median(a):.4f}  "
          f"max {a.max():.4f}   (the two ceiling reads sit at the cap, "
          f"{lv['cap']:.4f})")
    print(f"  bright band  : {len(bright)} rungs >= {lv['floor']:.4f} "
          f"({lv['floor'] * fs * 1e3:.0f} mV) -- the leverage;  "
          f"{len(a) - len(bright)} below it are model checks")
    print(f"  coverage     : {' '.join(str(c) for c in hist)}   "
          f"({int((hist > 0).sum())} of 16 bins of [0, cap] occupied)")
    print(f"  single_beam  : {sb.min() * 1e3:.3f} .. {sb.max() * 1e3:.3f} mV  "
          f"({100 * (sb.max() - sb.min()) / step:.1f} % of a level step)")
    print("  DAQ ranges   : " + ", ".join(
        f"{r:g}V x{n}" for r, n in zip(*np.unique(used, return_counts=True))))
    top, bot = max(preds), lv["floor"] * fs
    print(f"\n  the fit lives at the top: at the cap ({top * 1e3:.1f} mV) a "
          f"Y_sat of 5 V bends\n  the reading down by "
          f"{100 * top / (5.0 + top):.1f} %, at the band floor "
          f"({bot * 1e3:.1f} mV) by {100 * bot / (5.0 + bot):.1f} %, and at the "
          f"dimmest\n  model check ({min(preds) * 1e3:.1f} mV) by "
          f"{100 * min(preds) / (5.0 + min(preds)):.2f} % -- which is why those "
          f"are checks and not leverage")


# ======================================================================
# CLI
# ======================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calib_step8.py",
        description="Step 8: fit the two-photon saturation constant Y_sat from "
                    "one level sweep, and anchor g = 1 for everything "
                    "downstream.",
        epilog="--run has no default: that one directory's calib.json is the "
               "whole forward model (step 3 the layout, step 6 the etas, step 7 "
               "the phases), so running against the wrong one reports a "
               "saturation constant that is really a file mix-up.",
    )
    parser.add_argument("--run", required=True, type=Path, metavar="DIR",
                        help="run directory holding calib.json; the Y_sat is "
                             "written back into it as its step8 section (required)")
    parser.add_argument("csv", nargs="?", type=Path,
                        help="re-fit this recorded CSV offline instead of collecting")
    parser.add_argument("--p", nargs="+", type=int, default=None, metavar="PAIR",
                        help="only drive these pairs (default: every calibrated pair)")
    parser.add_argument("--ladder", type=int, default=LADDER, metavar="N",
                        help=f"rungs in the bright band, where the Y_sat "
                             f"leverage is (default {LADDER})")
    parser.add_argument("--low", type=int, default=LADDER_LOW, metavar="N",
                        help=f"rungs BELOW the bright band (default "
                             f"{LADDER_LOW}).  They carry no usable curvature; "
                             f"they are there so a step-6/7 error in the decade "
                             f"an inference run computes in cannot pass "
                             f"unnoticed.  0 drops them")
    parser.add_argument("--ceiling-mv", type=float, metavar="MV",
                        default=1e3 * LADDER_CEILING_V,
                        help=f"brightest read the ladder may take, in "
                             f"MILLIVOLTS (default "
                             f"{1e3 * LADDER_CEILING_V:g}).  This is an "
                             f"unverified figure for where the amplifier starts "
                             f"to compress, not a measurement -- see "
                             f"LADDER_CEILING_V.  Volts, not alpha, so it stays "
                             f"put when the PMT gain moves")
    parser.add_argument("--floor-mv", type=float, metavar="MV",
                        default=1e3 * LADDER_FLOOR_V,
                        help=f"bottom of the bright band, in MILLIVOLTS "
                             f"(default {1e3 * LADDER_FLOOR_V:g}).  --ladder "
                             f"rungs stratify from here to --ceiling-mv")
    parser.add_argument("--low-mv", type=float, metavar="MV",
                        default=1e3 * LADDER_LOW_V,
                        help=f"dimmest read of all, in MILLIVOLTS (default "
                             f"{1e3 * LADDER_LOW_V:g}).  --low model checks "
                             f"stratify from here to --floor-mv")
    parser.add_argument("--method", default=PHASE_METHOD,
                        help="stored step-7 fit to predict from")
    parser.add_argument("--seed", type=int, default=None,
                        help="reproduce a previous run's patterns")
    parser.add_argument("--pmt", nargs="?", const="", default=None, metavar="RESOURCE",
                        help="log the PMT thermistor (Keithley 2400) beside every "
                             "read: adds pmt_ohm_Ohm and pmt_c columns, sampled "
                             f"every {PMT_PERIOD_S:g} s.  The instrument settings "
                             "come from drafts/daq_pmt_monitor.py; give a VISA "
                             "resource to override its address")
    parser.add_argument("--pmt-period", type=float, default=PMT_PERIOD_S, metavar="S",
                        help=f"--pmt only: seconds between 2400 readings "
                             f"(default {PMT_PERIOD_S:g})")
    parser.add_argument("--dry-run", action="store_true",
                        help="solve the patterns, write the plan CSV, print it; "
                             "touch no hardware")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        wanted = select_pairs(args.p)
    except ValueError as exc:
        parser.error(str(exc))

    if args.csv is not None:
        fit_and_write(args.csv, args.run, pairs=wanted, method=args.method)
        return 0

    pmt = None
    if args.pmt is not None and not args.dry_run:
        try:
            pmt = PmtTemperature(args.pmt or None, period_s=args.pmt_period)
        except Exception as exc:        # noqa: BLE001
            # A missing VISA stack must not cost the run the sweep.
            print(f"--pmt ignored ({type(exc).__name__}: {exc})")

    csv_path = collect(args.run, pairs=wanted, method=args.method,
                       ladder=args.ladder, low=args.low,
                       ceiling_v=args.ceiling_mv / 1e3,
                       floor_v=args.floor_mv / 1e3, low_v=args.low_mv / 1e3,
                       out_dir=args.run,
                       seed=args.seed, dry_run=args.dry_run, pmt=pmt)
    if csv_path is not None:
        fit_and_write(csv_path, args.run, method=args.method)
        print(f"\nRe-fit with:  python {Path(__file__).name} "
              f"--run {args.run} {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
