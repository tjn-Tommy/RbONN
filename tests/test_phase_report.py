"""Phase-fit rendering and comb-phase JSON round trip (no hardware).

Both suites moved here verbatim from ``tests/test_pipeline.py`` when the unified
pipeline was deleted: neither ever touched the pipeline -- they cover
:mod:`calibration_module.fit.report` and :mod:`calibration_module.fit.phase`, which are
very much alive, and would have been lost with the file that happened to host
them.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class PhaseReportRenderTests(unittest.TestCase):
    def test_plot_fringe_renders_on_agg(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from calibration_module.fit.phase import PhaseFit
        from calibration_module.fit.report import plot_fringe

        n = 15
        theta = np.linspace(0.0, np.pi, n)
        g = np.sin(theta / 2.0) ** 2
        dphi_slm = theta - np.pi
        a, b, dphi_comb = 0.03, 0.02, 0.4
        y = a**2 + b**2 * g**2 + 2 * a * b * g * np.cos(dphi_slm + dphi_comb)
        std = np.full(n, 1e-5)
        fit = PhaseFit(
            dphi_comb=dphi_comb, dphi_comb_err=0.02,
            a=a, a_err=1e-3, b=b, b_err=1e-3,
            amp=2 * a * b, amp_err=1e-4,
            offset=0.0, offset_err=1e-5,
            r2=0.99,
            eta_ref=a, eta_tgt=b, bound_frac=1.0,
            a_at_bound=False, b_at_bound=False,
            bg0=0.0, bg1=0.0, bg2=0.0,
            dphi_slm=dphi_slm, g=g, y=y, std=std,
            known=a**2 + b**2 * g**2, y_pred=y, residuals=np.zeros(n),
        )
        fig = Figure(figsize=(8, 4))
        plot_fringe(fig, fit, tgt=3)      # must render without raising
        self.assertEqual(len(fig.axes), 2)


class CombPhaseJsonTests(unittest.TestCase):
    """save_step7_section / load_comb_phase_json round trip (no hardware)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    @staticmethod
    def _fit(dphi: float, frac: float):
        from calibration_module.fit.phase import PhaseFit

        n = 5
        arr = np.zeros(n)
        return PhaseFit(
            dphi_comb=dphi, dphi_comb_err=0.02,
            a=0.05, a_err=1e-3, b=0.048, b_err=1e-3,
            amp=2 * 0.05 * 0.048, amp_err=1e-4,
            offset=0.0, offset_err=1e-5,
            r2=0.99,
            eta_ref=0.05, eta_tgt=0.048, bound_frac=frac,
            a_at_bound=False, b_at_bound=False,
            bg0=0.0, bg1=0.0, bg2=0.0,
            dphi_slm=arr, g=arr, y=arr, std=np.ones(n),
            known=arr, y_pred=arr, residuals=arr,
        )

    def test_round_trip_and_method_selection(self) -> None:
        import json

        from calibration_module.calib_file import write_section
        from calibration_module.fit.phase import (
            load_comb_phase_json,
            save_step7_section,
        )

        # The sections step 7 is written on top of; it refuses without them.
        write_section(self.out, 3, {"probe": 1}, log=None)
        write_section(self.out, 6, {"channels": []}, log=None)
        fits = {
            (3, "bounded"): self._fit(0.30, 1.0),
            (3, "fix"): self._fit(0.29, 0.0),
            (5, "bounded"): self._fit(-0.23, 1.0),
            (5, "fix"): self._fit(-0.51, 0.0),
        }
        calib = save_step7_section(fits, self.out, ref_index=1,
                                   csv=self.out / "meas.csv",
                                   single_beam_bg=True, log=None)
        out = calib.path

        payload = json.loads(out.read_text(encoding="utf-8"))
        # Not copied forward any more: they are sections of this same file.
        self.assertEqual(payload["step3"]["probe"], 1)
        self.assertEqual(payload["step6"]["channels"], [])
        self.assertEqual(payload["step7"]["ref_index"], 1)
        self.assertEqual(payload["step7"]["csv"], "meas.csv")
        self.assertNotIn("step6_json", payload["step7"])

        ref, entries = load_comb_phase_json(out, method="bounded")
        self.assertEqual(ref, 1)
        self.assertEqual(sorted(entries), [3, 5])
        self.assertAlmostEqual(entries[3]["fit"]["dphi_comb_rad"], 0.30)
        self.assertAlmostEqual(entries[5]["fit"]["dphi_comb_rad"], -0.23)

        _, fixed = load_comb_phase_json(out, method="fix")
        self.assertAlmostEqual(fixed[5]["fit"]["dphi_comb_rad"], -0.51)
        self.assertEqual(fixed[3]["fit"]["bound_frac"], 0.0)

        with self.assertRaisesRegex(ValueError, "several stored fits"):
            load_comb_phase_json(out)                      # ambiguous without method
        with self.assertRaisesRegex(ValueError, "no 'other' fit"):
            load_comb_phase_json(out, method="other")


class Step7BurstTests(unittest.TestCase):
    """load_phase_csv drops burst acquisitions and says which ones."""

    REPO = Path(__file__).resolve().parents[1]
    #: run_0929 step 7: five bursts across targets 1, 6 and 10.
    CSV = REPO / "src/calib_data/run_0929/calib_step7_meas_0930_0236.csv"
    S6 = REPO / "src/calib_data/run_0929/calib.json"

    @classmethod
    def setUpClass(cls) -> None:
        from calibration_module.fit.phase import load_pair_models
        if not (cls.CSV.is_file() and cls.S6.is_file()):
            raise unittest.SkipTest(f"missing run_0929 fixture: {cls.CSV}")
        cls.models = load_pair_models(cls.S6)

    def _load(self, tgt: int, ref: int, **kw):
        from calibration_module.fit.phase import load_phase_csv
        return load_phase_csv(self.CSV, self.models[tgt], self.models[ref],
                              comb_only=True, single_beam_bg=True,
                              only_tgt=tgt, **kw)

    def test_reports_what_it_dropped(self) -> None:
        r = self._load(10, 2)
        self.assertEqual(len(r.dropped_bursts), 2)
        worst = max(r.dropped_bursts, key=lambda b: b["voltage_mean_v"])
        # 0.910 V against a step-6 full scale of 0.122 V, 91% of the +/-1 V range
        self.assertAlmostEqual(worst["voltage_mean_v"], 0.910, places=2)
        # both carry a ~0.35 V trace std: the burst covered part of the window,
        # which is why the mean is wrong rather than merely noisy
        self.assertTrue(all(b["voltage_std_v"] > 0.3 for b in r.dropped_bursts))

    def test_a_clean_target_reports_nothing(self) -> None:
        for tgt in (3, 4, 5, 7, 8, 9):
            self.assertEqual(self._load(tgt, 2 if tgt != 3 else 5).dropped_bursts,
                             [], f"false positive on target {tgt}")

    def test_dropping_rescues_r2_without_moving_the_phase(self) -> None:
        """The whole point: the FIT was always fine, only r2 read as a disaster.

        1/std^2 weighting already starves a burst (a 0.42 V trace std against a
        1.8 mV one is a weight ratio of ~54000), but r2 is unweighted, so these
        three pairs were stamped "model REJECTED" on good fits.
        """
        for tgt, ref in ((1, 5), (6, 2), (10, 2)):
            clean, raw = self._load(tgt, ref), self._load(tgt, ref, drop_bursts=False)
            self.assertLess(raw.fit.r2, 0.95, f"target {tgt} was meant to look bad")
            self.assertGreater(clean.fit.r2, 0.95)
            self.assertAlmostEqual(
                np.degrees(clean.fit.dphi_comb), np.degrees(raw.fit.dphi_comb),
                places=2, msg=f"target {tgt}: dropping a burst moved dPhi")

    def test_row_count_drops_by_exactly_the_bursts(self) -> None:
        clean, raw = self._load(1, 5), self._load(1, 5, drop_bursts=False)
        self.assertEqual(raw.trial.size - clean.trial.size, len(clean.dropped_bursts))

    def test_off_switch_keeps_the_old_behaviour(self) -> None:
        self.assertEqual(self._load(10, 2, drop_bursts=False).dropped_bursts, [])


class LiveSweepBurstTests(unittest.TestCase):
    """A swept PhaseResult must fit the same as its own CSV refit.

    The CLI sweep refits the CSV it just wrote, so it goes through
    load_phase_csv; a live GUI sweep fits the in-memory result instead.  Without
    drop_bursts on that path the two would disagree on the same sweep.
    """

    REPO = Path(__file__).resolve().parents[1]
    CSV = REPO / "src/calib_data/run_0929/calib_step7_meas_0930_0236.csv"
    S6 = REPO / "src/calib_data/run_0929/calib.json"

    @classmethod
    def setUpClass(cls) -> None:
        from calibration_module.fit.phase import load_pair_models
        if not (cls.CSV.is_file() and cls.S6.is_file()):
            raise unittest.SkipTest(f"missing run_0929 fixture: {cls.CSV}")
        cls.models = load_pair_models(cls.S6)

    def _raw(self, tgt: int, ref: int):
        """The sweep as measure_target would hand it over: nothing dropped."""
        from calibration_module.fit.phase import load_phase_csv
        return load_phase_csv(self.CSV, self.models[tgt], self.models[ref],
                              comb_only=True, single_beam_bg=True,
                              only_tgt=tgt, drop_bursts=False)

    def test_in_place_drop_matches_the_csv_path(self) -> None:
        from calibration_module.fit.phase import (drop_bursts, fit_result,
                                                  load_phase_csv)
        for tgt, ref in ((1, 5), (6, 2), (10, 2)):
            live = self._raw(tgt, ref)
            gone = drop_bursts(live)
            fit_result(live, self.models[tgt], self.models[ref],
                       comb_only=True, single_beam_bg=True)
            viacsv = load_phase_csv(self.CSV, self.models[tgt], self.models[ref],
                                    comb_only=True, single_beam_bg=True,
                                    only_tgt=tgt)
            self.assertEqual(len(gone), len(viacsv.dropped_bursts))
            self.assertEqual(live.trial.size, viacsv.trial.size)
            self.assertAlmostEqual(live.fit.dphi_comb, viacsv.fit.dphi_comb,
                                   places=9, msg=f"target {tgt}")
            self.assertAlmostEqual(live.fit.r2, viacsv.fit.r2, places=9)

    def test_every_per_row_array_is_filtered_together(self) -> None:
        from calibration_module.fit.phase import drop_bursts
        live = self._raw(10, 2)
        n = live.trial.size
        live.pmt_ohm = np.array([f"{9000 + i}" for i in range(n)], dtype=object)
        live.pmt_c = np.array([f"{27.0 + i:.2f}" for i in range(n)], dtype=object)
        drop_bursts(live)
        kept = live.trial.size
        self.assertEqual(kept, n - 2)
        for name in ("x_t", "w_t", "x_r", "w_r", "voltage_mean_v",
                     "voltage_std_v", "dark_v", "pmt_ohm", "pmt_c"):
            self.assertEqual(len(getattr(live, name)), kept, name)

    def test_unlogged_thermistor_stays_none(self) -> None:
        from calibration_module.fit.phase import drop_bursts
        live = self._raw(10, 2)
        live.pmt_ohm = live.pmt_c = None
        drop_bursts(live)
        self.assertIsNone(live.pmt_ohm)
        self.assertIsNone(live.pmt_c)

    def test_a_clean_sweep_is_untouched(self) -> None:
        from calibration_module.fit.phase import drop_bursts
        live = self._raw(4, 2)
        n = live.trial.size
        self.assertEqual(drop_bursts(live), [])
        self.assertEqual(live.trial.size, n)

    def test_calling_twice_keeps_the_first_record(self) -> None:
        from calibration_module.fit.phase import drop_bursts
        live = self._raw(10, 2)
        first = drop_bursts(live)
        self.assertEqual(drop_bursts(live), [])          # nothing left to drop
        self.assertEqual(len(live.dropped_bursts), len(first))


if __name__ == "__main__":
    unittest.main()
