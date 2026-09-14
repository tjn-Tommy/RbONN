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
    fit_pair,
    load_meas_csv,
)
from calibration_module.fit.sigma import STD_FLOOR_V  # noqa: E402

MEAS_CSV = REPO / "src/calib_data/archive/2026-09/run_0907_1724/calib_step6v2_meas_0907_1757.csv"

#: What the 0907 run published, pair label -> eta.
RECORDED_ETA = {
    2: 0.17760529803227112,
    3: 0.17954179839571333,
    4: 0.1920940148157429,
    5: 0.2142402286038278,
    6: 0.25500301123102215,
}


def _levels(pair: int):
    return average_levels(load_meas_csv(MEAS_CSV)[pair])


class RecordedRunTests(unittest.TestCase):
    """The estimator still returns the numbers that run published."""

    def test_csv_present(self) -> None:
        self.assertTrue(MEAS_CSV.is_file(), f"missing fixture: {MEAS_CSV}")

    def test_every_pair_refits_to_its_published_eta(self) -> None:
        rows = load_meas_csv(MEAS_CSV)
        self.assertEqual(sorted(rows), sorted(RECORDED_ETA))
        for pair, expected in RECORDED_ETA.items():
            with self.subTest(pair=pair):
                fit = fit_pair(pair, average_levels(rows[pair]))
                self.assertAlmostEqual(fit.eta, expected, places=12)
                self.assertGreater(fit.r2, 0.999)

    def test_verification_levels_stay_out_of_the_slope_fit(self) -> None:
        """(1, 0.25) is a legitimate cross point but must not reach the fit.

        It is measured to *check* product-only dependence, so letting it into
        the slope would make the check a test of its own input.
        """
        fit = fit_pair(5, _levels(5))
        self.assertNotIn(0.25, [round(w, 6) for w in fit.fit_w])
        self.assertIn("product", fit.checks)

    def test_top_drive_level_is_excluded_but_reported(self) -> None:
        """w = 1.0 is dropped from the fit and kept as the compression diagnostic."""
        fit = fit_pair(5, _levels(5))
        self.assertNotIn(1.0, [round(w, 6) for w in fit.fit_w])
        self.assertEqual([round(w, 6) for w, *_ in fit.excluded], [1.0])


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

    def test_narrower_fit_window_drops_levels(self) -> None:
        levels = _levels(5)
        wide = fit_pair(5, levels, PairV2Config(fit_w_range=(0.2, 0.9)))
        narrow = fit_pair(5, levels, PairV2Config(fit_w_range=(0.4, 0.9)))
        self.assertLess(narrow.fit_w.size, wide.fit_w.size)
        self.assertTrue(np.all(narrow.fit_w >= 0.4 - 1e-9))


class SigmaTests(unittest.TestCase):
    def test_no_level_reads_as_better_known_than_the_floor(self) -> None:
        """The floor goes on after ``/sqrt(n)``: a systematic does not average away."""
        for level in _levels(5):
            with self.subTest(x=level.x, w=level.w):
                self.assertGreaterEqual(level.sigma, STD_FLOOR_V)

    def test_repeats_are_averaged_not_concatenated(self) -> None:
        """Each grid level collapses to one Level carrying all of its repeats."""
        levels = _levels(5)
        self.assertEqual(len(levels), len(DEFAULT_CONFIG.full_grid()))
        by_xw = {(round(v.x, 6), round(v.w, 6)): v for v in levels}
        for x, w, n in DEFAULT_CONFIG.full_grid():
            self.assertEqual(by_xw[(round(x, 6), round(w, 6))].n, n)


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
        self.assertEqual((x, w), (1.0, 1.0))

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


if __name__ == "__main__":
    unittest.main()
