"""The sigma floor is measured per sweep, and the read ceiling is absolute.

Both exist because of one night.  On 2026-09-30 the detection gain was cut ~8x
between the 13:25 and 20:41 step-6 sweeps.  ``STD_FLOOR_V`` is a constant in
VOLTS, so it did not follow: it went from 2.8x the bench's all-off noise to 8x
it, stopped rebalancing the fit weights and started owning them.  Step-6
``eta_err/eta`` on pair 1 went 2.5% -> 8.4% and on pair 10 15% -> 105%, and
step-7 statistical errors roughly doubled, on fringes that were just as clean.

The same night, one step-7 all-off dark read +345.7 mV where its eight
neighbours read -60 uV.  Nothing caught it: both burst guards judge an
acquisition by its TRACE SPREAD, and the dark is one acquisition per target
with no repeat and no sweep median to sit against.  It was subtracted from all
ten of that pair's drive rows and put the fit at R^2 = -11438 with ten good
rows underneath it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from calibration_module.fit.outliers import (  # noqa: E402
    BURST_MAX_V,
    flag_over_range,
)
from calibration_module.fit.pair_v2 import (  # noqa: E402
    PairV2Config,
    average_levels,
    burst_rows,
    fit_pair,
    load_meas_csv,
    sweep_floor,
)
from calibration_module.fit.phase import (  # noqa: E402
    PhaseResult,
    guard_darks,
    load_pair_models,
    load_phase_csv,
)
from calibration_module.fit.sigma import (  # noqa: E402
    STD_FLOOR_V,
    FloorSource,
    floor_std,
    measured_floor,
)

NOYSAT = REPO / "src/calib_data/run_0930_noYsat"
BRIGHT = REPO / "src/calib_data/run_0930"
S6_DIM = NOYSAT / "calib_step6v2_meas_0930_2041.csv"
S6_BRIGHT = BRIGHT / "calib_step6v2_meas_0930_1325.csv"
S7_DIM = NOYSAT / "calib_step7_meas_0930_2112.csv"


class MeasuredFloorTests(unittest.TestCase):
    def test_a_dark_sweep_measures_its_own_floor(self) -> None:
        floor, source = sweep_floor(load_meas_csv(S6_DIM))
        self.assertEqual(source, FloorSource.MEASURED.value)
        # ~25 uV: the electronics, with no light on the detector at all.
        self.assertLess(floor, 50e-6, f"{floor*1e6:.1f} uV")
        self.assertGreater(floor, 5e-6)

    def test_the_floor_follows_the_gain_and_the_constant_does_not(self) -> None:
        """The whole point: 8x less signal, 8x less noise, same old constant."""
        dim, _ = sweep_floor(load_meas_csv(S6_DIM))
        bright, _ = sweep_floor(load_meas_csv(S6_BRIGHT))
        self.assertGreater(bright / dim, 5.0,
                           "these two sweeps are 8x apart in gain")
        self.assertAlmostEqual(STD_FLOOR_V, 2.0e-4, places=9)

    def test_the_dark_level_is_chosen_by_drive_not_by_voltage(self) -> None:
        """13:25 read 21 mV at its all-off level: stray light, still the dark.

        A voltage-based test would call that 'not near zero', throw the level
        away and fall back to a constant measured on another night -- when the
        900 uV it fluctuated by is a real noise every point of that sweep
        carries.
        """
        floor, source = sweep_floor(load_meas_csv(S6_BRIGHT))
        self.assertEqual(source, FloorSource.MEASURED.value)
        self.assertGreater(floor, STD_FLOOR_V,
                           "that sweep was noisier than the constant assumed")

    def test_too_few_quiet_points_falls_back_and_says_so(self) -> None:
        floor, source = measured_floor([1.0, 2.0], [0.1, 0.2])
        self.assertEqual(source, FloorSource.FALLBACK)
        self.assertEqual(floor, STD_FLOOR_V)

    def test_a_sweep_with_no_dark_level_cannot_be_fitted(self) -> None:
        """No constant stands in: a file with no (0, 0) rows has no floor."""
        rows = {1: [(0, 1.0, 1.0, 0.05, 1e-4) for _ in range(20)]}
        with self.assertRaisesRegex(ValueError, "no dark level"):
            sweep_floor(rows)

    def test_one_pair_s_two_dark_repeats_are_enough(self) -> None:
        """Selected by commanded drive, so even two reads are the floor."""
        rows = {1: [(0, 0.0, 0.0, 1e-5, 3e-5), (1, 0.0, 0.0, 2e-5, 5e-5),
                    (0, 1.0, 0.9, 0.05, 1e-3)]}
        floor, source = sweep_floor(rows)
        self.assertEqual(source, FloorSource.MEASURED.value)
        self.assertAlmostEqual(floor, 4e-5, places=12)

    def test_floor_std_takes_the_sweep_s_value_over_the_constant(self) -> None:
        self.assertAlmostEqual(float(floor_std(0.0, 3e-5)), 3e-5, places=12)
        self.assertAlmostEqual(float(floor_std(0.0)), STD_FLOOR_V, places=12)


class FloorChangesTheErrorBarTests(unittest.TestCase):
    """What it was for: an eta_err that describes the bench, not 0903."""

    def test_the_dim_run_s_eta_precision_is_recovered(self) -> None:
        rows = load_meas_csv(S6_DIM)
        cfg = PairV2Config()
        floor = sweep_floor(rows)
        for pair, ceiling in ((1, 0.04), (10, 0.30)):
            with self.subTest(pair=pair):
                old = fit_pair(pair, average_levels(
                    rows[pair], floor=(STD_FLOOR_V, "const")), cfg)
                new = fit_pair(pair, average_levels(rows[pair], floor=floor), cfg)
                self.assertGreater(old.eta_err / old.eta, ceiling,
                                   "the constant inflated this")
                self.assertLess(new.eta_err / new.eta, ceiling)
                # The VALUE barely moves; only how well it is claimed to be known.
                self.assertAlmostEqual(new.eta / old.eta, 1.0, places=2)

    def test_the_level_carries_the_floor_it_was_weighted_by(self) -> None:
        rows = load_meas_csv(S6_DIM)
        floor = sweep_floor(rows)
        levels = average_levels(rows[1], floor=floor)
        self.assertAlmostEqual(levels[0].floor, floor[0], places=12)
        self.assertEqual(levels[0].floor_source, FloorSource.MEASURED.value)


class ReadCeilingTests(unittest.TestCase):
    def test_the_ceiling_is_absolute_and_needs_no_neighbours(self) -> None:
        flags = flag_over_range([-6e-5, 0.3457, 0.058, np.nan, -0.4])
        self.assertEqual(list(flags), [False, True, False, True, True])

    def test_it_is_300_mv(self) -> None:
        self.assertAlmostEqual(BURST_MAX_V, 0.300, places=9)

    def test_a_railed_read_is_dropped_even_with_a_calm_trace(self) -> None:
        """The trace-spread guards cannot see this: the rail latched first."""
        rows = [(0, 1.0, 0.9, 0.025, 8e-4), (1, 1.0, 0.9, 0.025, 8e-4),
                (2, 1.0, 0.9, 0.900, 8e-4)]          # calm trace, railed mean
        mask = burst_rows(rows)
        self.assertEqual(list(mask), [False, False, True])

    def test_an_honest_full_scale_read_survives(self) -> None:
        rows = [(0, 1.0, 1.0, 0.163, 2e-3)] * 4      # the 13:25 sweep's peak
        self.assertFalse(burst_rows(rows).any())


class DarkGuardTests(unittest.TestCase):
    """The pair-4 story, both paths: from a CSV and from a live sweep."""

    @staticmethod
    def _result(tgt: int, dark: float) -> PhaseResult:
        n = 4
        return PhaseResult(
            tgt_index=tgt, ref_index=2,
            trial=np.zeros(n, dtype=int),
            x_t=np.full(n, 0.5), w_t=np.full(n, 0.5),
            x_r=np.full(n, 0.9), w_r=np.full(n, 0.9),
            voltage_mean_v=np.full(n, 0.004),
            voltage_std_v=np.full(n, 2e-4),
            dark_v=np.full(n, dark),
        )

    def test_a_railed_dark_is_replaced_by_the_other_targets(self) -> None:
        results = {1: self._result(1, -6e-5), 3: self._result(3, -7e-5),
                   4: self._result(4, 0.3457), 5: self._result(5, -5e-5)}
        moved = guard_darks(results)
        self.assertEqual(list(moved), ["4"])
        self.assertAlmostEqual(moved["4"]["recorded_v"], 0.3457, places=9)
        self.assertAlmostEqual(float(results[4].dark_v[0]), -6e-5, places=7)
        self.assertEqual(results[1].dark_overrides, {})

    def test_a_sweep_that_is_all_over_the_ceiling_is_left_alone(self) -> None:
        """Lights on is not a glitched read, and must reach the operator."""
        results = {k: self._result(k, 0.5) for k in (1, 3, 4)}
        self.assertEqual(guard_darks(results), {})
        self.assertAlmostEqual(float(results[4].dark_v[0]), 0.5, places=9)

    def test_the_recorded_sweep_repairs_pair_4_on_load(self) -> None:
        """R^2 = -11438 with ten good drive rows under it; now 0.99."""
        models = load_pair_models(str(NOYSAT / "calib.json"))
        r = load_phase_csv(S7_DIM, models[4], models[2], comb_only=True,
                           single_beam_bg=True, only_tgt=4)
        self.assertIn("4", r.dark_overrides)
        self.assertGreater(r.fit.r2, 0.99)
        self.assertLess(abs(np.degrees(r.fit.dphi_comb) + 84.0), 5.0)

    def test_an_explicit_dark_still_wins(self) -> None:
        models = load_pair_models(str(NOYSAT / "calib.json"))
        r = load_phase_csv(S7_DIM, models[4], models[2], comb_only=True,
                           single_beam_bg=True, only_tgt=4, dark=-6e-5)
        self.assertEqual(r.dark_overrides, {})
        self.assertGreater(r.fit.r2, 0.99)


class RecordedInTheSectionTests(unittest.TestCase):
    def test_both_runs_record_the_floor_they_used(self) -> None:
        import json

        payload = json.loads((NOYSAT / "calib.json").read_text(encoding="utf-8"))
        for step in ("step6", "step7"):
            with self.subTest(step=step):
                section = payload[step]
                self.assertIn("sigma_floor_v", section)
                self.assertIn("sigma_floor_source", section)
                self.assertLess(section["sigma_floor_v"], STD_FLOOR_V)
        self.assertEqual(payload["step6"]["sigma_floor_source"],
                         FloorSource.MEASURED.value)
        self.assertEqual(payload["step7"]["sigma_floor_source"],
                         FloorSource.INHERITED.value)

    def test_the_repaired_dark_is_on_the_record(self) -> None:
        """A repaired run must never read as a clean one."""
        import json

        payload = json.loads((NOYSAT / "calib.json").read_text(encoding="utf-8"))
        overrides = payload["step7"]["dark_overrides"]
        self.assertIn("4", overrides)
        self.assertAlmostEqual(overrides["4"]["recorded_v"], 0.345741872, places=9)
        self.assertIn("ceiling", overrides["4"]["reason"])


if __name__ == "__main__":
    unittest.main()
