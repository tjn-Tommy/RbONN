"""Step-6 v2 estimator, exercised against a recorded run with no hardware.

The point of moving this estimator out of the step script and into
:mod:`calibration_module.fit.pair_v2` was that it becomes testable: the whole
suite below runs off one committed measurement CSV with the bench powered down
and no fake instrument anywhere, which is exactly the property
``calibration_module.fit`` exists to protect.

The eta values asserted here are the ones the 0907 run published.  They are
pinned to 12 significant figures deliberately -- this file's job is to notice
that a refactor moved an answer, and a loose tolerance would let a real change
in the weighting or the covariance slip through as "close enough".
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from calibration_module.fit.pair_v2 import (  # noqa: E402
    DEFAULT_CONFIG,
    PairV2Config,
    average_levels,
    burst_rows,
    fit_pair,
    load_meas_csv,
)
from calibration_module.fit.pair_v2 import product_groups, sweep_floor  # noqa: E402

MEAS_CSV = REPO / "src/calib_data/archive/2026-09/run_0907_1724/calib_step6v2_meas_0907_1757.csv"

#: What the 0907 CSV fits to, pair label -> eta.  Re-baselined 2026-10-01 when
#: the fit became "every cross level, floor measured from the sweep's own dark
#: reads"; the 0907 run was published at 0.1776 / 0.1795 / 0.1921 / 0.2142 /
#: 0.2550 under the old 0.2-0.9 window and the 200 uV constant.
RECORDED_ETA = {
    2: 0.17723508222881532,
    3: 0.17949741792999532,
    4: 0.19180164107087086,
    5: 0.21419773519244129,
    6: 0.2549718270732447,
}


def _levels(pair: int):
    rows = load_meas_csv(MEAS_CSV)
    return average_levels(rows[pair], floor=sweep_floor(rows))


class RecordedRunTests(unittest.TestCase):
    """The estimator still returns the numbers that run published."""

    def test_csv_present(self) -> None:
        self.assertTrue(MEAS_CSV.is_file(), f"missing fixture: {MEAS_CSV}")

    def test_every_pair_refits_to_its_published_eta(self) -> None:
        rows = load_meas_csv(MEAS_CSV)
        floor = sweep_floor(rows)
        self.assertEqual(sorted(rows), sorted(RECORDED_ETA))
        for pair, expected in RECORDED_ETA.items():
            with self.subTest(pair=pair):
                fit = fit_pair(pair, average_levels(rows[pair], floor=floor))
                self.assertAlmostEqual(fit.eta, expected, places=12)
                self.assertGreater(fit.r2, 0.999)

    def test_every_cross_level_is_fitted(self) -> None:
        """The window is the sweep: no level on x = 1 is held back.

        The 0907 file was swept 0.2..0.9 with (1, 1.0) and (1, 0.25) alongside;
        all six are on the line, so all six are in the slope, and the fitted
        range is read off them.
        """
        fit = fit_pair(5, _levels(5))
        self.assertEqual(sorted(round(w, 6) for w in fit.fit_w),
                         [0.2, 0.25, 0.45, 0.7, 0.9, 1.0])
        self.assertEqual(fit.fit_w_range, (0.2, 1.0))

    def test_the_product_check_is_found_in_the_data(self) -> None:
        """(1, 0.25) and (0.5, 0.5) share x*w = 0.25: that pairing IS the check.

        Nothing declares it -- an older file with the check level on the cross
        line and a new one with it at x = 0.95 both carry their own check.
        """
        fit = fit_pair(5, _levels(5))
        groups = [[(L.x, L.w) for L in g] for g in product_groups(fit.levels)]
        self.assertEqual(groups, [[(1.0, 0.25), (0.5, 0.5)]])
        (rec,) = fit.checks["product"]
        self.assertAlmostEqual(rec["product"], 0.25)
        self.assertEqual(len(rec["pairs"]), 1)


class ConfigTests(unittest.TestCase):
    """The config replaced module globals; these are the properties that bought."""

    def test_frozen(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_CONFIG.fit_q = True          # type: ignore[misc]

    def test_params_follow_fit_q(self) -> None:
        self.assertEqual(PairV2Config(fit_q=False).params_bg, ("a_x", "a_w", "d"))
        self.assertEqual(PairV2Config(fit_q=True).params_bg,
                         ("a_x", "q_x", "a_w", "q_w", "d"))
        self.assertEqual(PairV2Config(fit_q=False).x_side, ("a_x",))
        self.assertEqual(PairV2Config(fit_q=True).x_side, ("a_x", "q_x"))

    def test_slot_converts_the_pair_label(self) -> None:
        self.assertEqual(PairV2Config(pair_index_base=1).slot(1), 0)
        self.assertEqual(PairV2Config(pair_index_base=0).slot(1), 1)

    def test_two_configs_coexist_in_one_process(self) -> None:
        """The whole reason for the dataclass: module globals could not do this.

        A GUI that fits one pair with the q columns and another without must get
        two different answers back, not whichever setting was assigned last.
        """
        levels = _levels(5)
        plain = fit_pair(5, levels, PairV2Config(fit_q=False))
        with_q = fit_pair(5, levels, PairV2Config(fit_q=True))
        self.assertEqual(set(plain.bg), {"a_x", "a_w", "d"})
        self.assertEqual(set(with_q.bg), {"a_x", "q_x", "a_w", "q_w", "d"})
        self.assertEqual(plain.cfg.fit_q, False)
        self.assertEqual(with_q.cfg.fit_q, True)

    def test_from_ramp_is_the_ramp_and_nothing_else(self) -> None:
        """No appended (1, 1.0): the fit uses every cross level, so a level
        measured only to be left out of it would simply be fitted."""
        cfg = PairV2Config.from_ramp(0.2, 0.9, 4)
        cross = [(w, n) for x, w, n in cfg.grid if x and w]
        self.assertEqual(cross, [(0.2, 4), (0.433333, 4), (0.666667, 4), (0.9, 6)])

    def test_the_verification_level_is_off_the_cross_line(self) -> None:
        """x = 0.95: no ramp point can coincide with it; same x*w as (0.5, 0.5)."""
        (x, w, n), (x2, w2, n2) = DEFAULT_CONFIG.verify_grid
        self.assertNotEqual(x, 1.0)
        self.assertAlmostEqual(x * w, x2 * w2, places=5)
        self.assertEqual((n, n2), (3, 3))


class SigmaTests(unittest.TestCase):
    def test_no_level_reads_as_better_known_than_the_floor(self) -> None:
        """The floor goes on after ``/sqrt(n)``: a systematic does not average away."""
        for level in _levels(5):
            with self.subTest(x=level.x, w=level.w):
                self.assertGreaterEqual(level.sigma, level.floor)

    def test_repeats_are_averaged_not_concatenated(self) -> None:
        """Each measured level collapses to one Level carrying all of its repeats."""
        rows = load_meas_csv(MEAS_CSV)[5]
        cells: dict[tuple[float, float], int] = {}
        for _rep, x, w, *_ in rows:
            key = (round(x, 6), round(w, 6))
            cells[key] = cells.get(key, 0) + 1
        levels = average_levels(rows, floor=sweep_floor({5: rows}))
        self.assertEqual({(round(v.x, 6), round(v.w, 6)): v.n for v in levels}, cells)


class ScheduleTests(unittest.TestCase):
    """Acquisition order is a fit-relevant choice, so it is pinned here too."""

    def setUp(self) -> None:
        from calibration_module.measure.pair_v2 import build_schedule
        self.schedule = build_schedule(DEFAULT_CONFIG)

    def test_length_matches_the_grid(self) -> None:
        self.assertEqual(len(self.schedule),
                         sum(n for _, _, n in DEFAULT_CONFIG.full_grid()))

    def test_brightest_point_goes_first(self) -> None:
        """A dead or blocked beam should be visible on acquisition one."""
        _, x, w = self.schedule[0]
        bx, bw, _n = max(DEFAULT_CONFIG.full_grid(), key=lambda lv: lv[0] * lv[1])
        self.assertEqual((x, w), (bx, bw))

    def test_repeats_run_as_round_robin_passes(self) -> None:
        """One pass per repeat, each level at most once in it.

        This is what stops a slow drift masquerading as a slope: a level's
        repeats are separated by every other level still owing one.  It does not
        promise no two repeats are ever adjacent -- once the low-count levels are
        satisfied, the highest-repeat level is the only one left and its last
        passes necessarily run together.
        """
        reps = [rep for rep, _, _ in self.schedule]
        self.assertEqual(reps, sorted(reps), "passes are not in repeat order")

        per_pass: dict[int, list[tuple[float, float]]] = {}
        for rep, x, w in self.schedule:
            per_pass.setdefault(rep, []).append((x, w))
        for rep, levels in per_pass.items():
            self.assertEqual(len(levels), len(set(levels)),
                             f"pass {rep} reads a level twice")

        for x, w, n in DEFAULT_CONFIG.full_grid():
            with self.subTest(x=x, w=w):
                got = [rep for rep, sx, sw in self.schedule if (sx, sw) == (x, w)]
                self.assertEqual(got, list(range(n)))


class NoDriverTests(unittest.TestCase):
    def test_importing_the_fit_starts_no_instrument_module(self) -> None:
        """The subpackage rule, enforced rather than documented.

        Run in a subprocess so an instrument module another test already
        imported cannot make this pass by accident.
        """
        import subprocess

        code = (
            "import sys;"
            "import calibration_module.fit.pair_v2;"
            "bad=[m for m in sys.modules if m.startswith(('slm_module.controller',"
            "'daq_module','pyvisa','nidaqmx','PyQt5'))];"
            "print(','.join(sorted(bad)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=str(REPO / "src"),
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "", f"fit imported a driver: {out.stdout}")


class BurstRowTests(unittest.TestCase):
    """average_levels drops burst acquisitions before they reach Level.mean."""

    #: run_0929 pair 10 x=1 w=0.9: one burst repeat among six.
    BURST_CSV = REPO / "src/calib_data/run_0929/calib_step6v2_meas_0930_0215.csv"

    def _rows(self):
        self.assertTrue(self.BURST_CSV.is_file(), f"missing: {self.BURST_CSV}")
        return load_meas_csv(self.BURST_CSV)[10]

    def test_finds_the_one_known_burst(self) -> None:
        mask = burst_rows(self._rows())
        self.assertEqual(int(mask.sum()), 1)
        row = [r for r, bad in zip(self._rows(), mask) if bad][0]
        self.assertAlmostEqual(row[1], 1.0)      # x
        self.assertAlmostEqual(row[2], 0.9)      # w
        self.assertGreater(row[4], 0.1)          # a 111 mV trace std

    def test_the_clean_0907_run_loses_nothing(self) -> None:
        for pair, rows in load_meas_csv(MEAS_CSV).items():
            self.assertFalse(burst_rows(rows).any(), f"false positive on pair {pair}")

    def test_dropping_restores_the_cell_mean(self) -> None:
        # Level.mean is a PLAIN mean over repeats, so the burst dragged the cell
        # from its five honest repeats at ~1.70 mV up to 4.09 mV.
        cell = (1.0, 0.9)
        keep = [L for L in average_levels(self._rows()) if (L.x, L.w) == cell][0]
        raw = [L for L in average_levels(self._rows(), drop_bursts=False)
               if (L.x, L.w) == cell][0]
        self.assertEqual(keep.n, 5)
        self.assertEqual(raw.n, 6)
        self.assertAlmostEqual(keep.mean * 1e3, 1.70, places=1)
        self.assertAlmostEqual(raw.mean * 1e3, 4.09, places=1)

    def test_dropping_tightens_eta(self) -> None:
        clean = fit_pair(10, average_levels(self._rows()))
        raw = fit_pair(10, average_levels(self._rows(), drop_bursts=False))
        self.assertLess(clean.eta_err, raw.eta_err)
        self.assertGreater(clean.r2, raw.r2)
        # the burst inflated the quoted error by ~a third without moving eta
        # more than a twelfth of a sigma -- it was noise, not signal
        self.assertLess(abs(clean.eta - raw.eta), 0.2 * clean.eta_err)

    def test_never_empties_a_cell(self) -> None:
        # every repeat of one cell flagged -> all kept, so the level survives and
        # the design matrix still describes the model that was fitted
        rows = [(0, 1.0, 0.9, 0.05, 1.0), (1, 1.0, 0.9, 0.05, 1.0),
                (0, 0.0, 0.0, 0.0, 0.001), (1, 0.0, 0.0, 0.0, 0.001),
                (2, 0.0, 0.0, 0.0, 0.001)]
        self.assertFalse(burst_rows(rows).any())
        cells = {(L.x, L.w) for L in average_levels(rows)}
        self.assertIn((1.0, 0.9), cells)

    def test_empty_input(self) -> None:
        self.assertEqual(burst_rows([]).size, 0)
        self.assertEqual(average_levels([]), [])


if __name__ == "__main__":
    unittest.main()
