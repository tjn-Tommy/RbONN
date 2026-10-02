"""The system-accuracy pattern families -- all of it offline, no hardware.

The one bug that would silently poison every level is a convention drift between
``levels.y_coherent_uv`` (which solves in ``u, v``) and ``verify.predict`` (which
drives in ``x, w``): the solver would hit a target that the report then predicts
differently, and every residual would be wrong by an amount that looks exactly
like model error.  :class:`ParametrisationTests` pins the two together.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibration_module.fit import levels  # noqa: E402
from calibration_module.fit.phase import PairModel  # noqa: E402
from calibration_module.fit.verify import full_scale, predict  # noqa: E402

RUN = ROOT / "src" / "calib_data" / "run_0911_10pairs_w20g30"
STEP7 = RUN / "calib.json"


def toy(etas, phases, *, single_beam: float = 0.0):
    """A models/phases pair straight from numbers -- no JSON, no fit."""
    models = {i + 1: PairModel(index=i + 1, eta=float(e), a_x=single_beam, q_x=0.0,
                               a_w=single_beam, q_w=0.0, d=0.0)
              for i, e in enumerate(etas)}
    return models, {i + 1: float(p) for i, p in enumerate(phases)}, sorted(models)


def real():
    """The 0911 w20g30 forward model, or a skip if the run folder is gone.

    The stored phases are negated on the way in: that JSON predates the
    project's single sign convention (2026-09-25) and holds ``-Phi_k``, while
    ``load_forward_model`` now reads ``dphi_comb_rad`` as ``Phi_k`` verbatim.
    The geometry tests below are about THIS run's real spectrum -- negating
    ``Phi`` is not a symmetry of the reachable set, because ``t = u + v`` can
    only advance a phasor -- so the sign has to be right here.
    """
    if not STEP7.exists():
        raise unittest.SkipTest(f"{STEP7} not in the tree")
    from calibration_module.fit.verify import load_forward_model
    models, phases, _ = load_forward_model(STEP7)
    return models, {k: -v for k, v in phases.items()}, sorted(models)


class ParametrisationTests(unittest.TestCase):
    """(u, v) and (x, w) must describe the same field."""

    def test_xw_uv_round_trip(self):
        x = np.array([0.0, 0.02, 0.37, 0.5, 0.99, 1.0])
        w = np.array([1.0, 0.99, 0.5, 0.37, 0.02, 0.0])
        back_x, back_w = levels.xw_from_uv(levels.uv_from_xw(x, w))
        np.testing.assert_allclose(back_x, x, atol=1e-12)
        np.testing.assert_allclose(back_w, w, atol=1e-12)

    def test_y_coherent_matches_predict(self):
        """The solver's objective IS verify.predict(coherent=True)."""
        models, phases, pairs = real()
        scale = full_scale(models, phases, pairs)
        rng = np.random.default_rng(0)
        for _ in range(25):
            x, w = rng.uniform(0.0, 1.0, len(pairs)), rng.uniform(0.0, 1.0, len(pairs))
            mine = levels.y_coherent_uv(levels.uv_from_xw(x, w),
                                        scale.amplitudes, scale.phases)
            theirs = float(predict(models, phases, pairs, x, w, coherent=True))
            self.assertAlmostEqual(mine, theirs, places=12)

    def test_coherent_flag_drops_only_the_single_beam(self):
        models, phases, pairs = toy([0.1, 0.2], [0.0, 1.0], single_beam=0.003)
        x = np.array([0.4, 0.9])
        w = np.array([0.7, 0.2])
        full = float(predict(models, phases, pairs, x, w))
        bare = float(predict(models, phases, pairs, x, w, coherent=True))
        bg = sum(models[k].single_beam(x[i], w[i]) for i, k in enumerate(pairs))
        self.assertAlmostEqual(full - bare, bg, places=12)
        self.assertGreater(bg, 0.0)          # the test would pass trivially at 0

    def test_gradient_matches_finite_difference(self):
        models, phases, pairs = real()
        scale = full_scale(models, phases, pairs)
        rng = np.random.default_rng(1)
        uv = rng.uniform(0.05, levels.U_MAX - 0.05, 2 * len(pairs))
        got = levels.grad_y_coherent_uv(uv, scale.amplitudes, scale.phases)
        h = 1e-6
        want = np.empty_like(got)
        for i in range(uv.size):
            step = np.zeros_like(uv)
            step[i] = h
            want[i] = (levels.y_coherent_uv(uv + step, scale.amplitudes, scale.phases)
                       - levels.y_coherent_uv(uv - step, scale.amplitudes, scale.phases)) / (2 * h)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-12)


class ProjectionTests(unittest.TestCase):

    def test_hits_every_level_across_the_scale(self):
        models, phases, pairs = real()
        scale = full_scale(models, phases, pairs)
        rng = np.random.default_rng(2)
        for alpha in (0.02, 0.1, 0.25, 0.5, 0.75, 0.9):
            hit = False
            for _ in range(20):
                uv0 = levels.uv_from_xw(*np.split(rng.uniform(0.02, 1.0, 2 * len(pairs)), 2))
                _, err = levels.project_to_level(alpha, uv0, scale.amplitudes,
                                                 scale.phases, ymax=scale.fs)
                if err < 1e-4:
                    hit = True
                    break
            self.assertTrue(hit, f"no drive found for alpha = {alpha}")

    def test_the_solution_is_a_real_drive(self):
        """Whatever comes back must be displayable: x, w inside [0, 1]."""
        models, phases, pairs = real()
        scale = full_scale(models, phases, pairs)
        rng = np.random.default_rng(3)
        uv0 = levels.uv_from_xw(*np.split(rng.uniform(0.02, 1.0, 2 * len(pairs)), 2))
        uv, err = levels.project_to_level(0.6, uv0, scale.amplitudes,
                                          scale.phases, ymax=scale.fs)
        self.assertLess(err, 1e-4)
        x, w = levels.xw_from_uv(uv)
        self.assertTrue(np.all((x >= 0.0) & (x <= 1.0)))
        self.assertTrue(np.all((w >= 0.0) & (w <= 1.0)))

    def test_different_starts_give_different_drives(self):
        """The least-move constraint is what keeps A from being one drive repeated."""
        models, phases, pairs = real()
        scale = full_scale(models, phases, pairs)
        rng = np.random.default_rng(4)
        found = []
        while len(found) < 6:
            uv0 = levels.uv_from_xw(*np.split(rng.uniform(0.02, 1.0, 2 * len(pairs)), 2))
            uv, err = levels.project_to_level(0.3, uv0, scale.amplitudes,
                                              scale.phases, ymax=scale.fs)
            if err < 1e-4:
                found.append(levels.xw_from_uv(uv)[0])
        spread = np.std(np.array(found), axis=0).mean()
        self.assertGreater(spread, 0.1, "solutions collapsed onto one drive")


class AttenuatedDriveTests(unittest.TestCase):
    """The deterministic fallback, which must never fail even when SLSQP does."""

    def test_reaches_the_level_it_is_asked_for(self):
        models, phases, pairs = real()
        scale = full_scale(models, phases, pairs)
        for alpha in (0.05, 0.4, 0.8, 0.99):
            drive = levels.attenuated_drive(scale, alpha)
            got = float(predict(models, phases, pairs, drive, drive, coherent=True))
            self.assertAlmostEqual(got / scale.fs, alpha, places=6)

    def test_alpha_one_is_the_ymax_drive(self):
        models, phases, pairs = real()
        scale = full_scale(models, phases, pairs)
        np.testing.assert_allclose(levels.attenuated_drive(scale, 1.0),
                                   np.asarray(scale.drive), atol=1e-9)


class LadderTests(unittest.TestCase):
    """Block A.

    Built once for the class: ``level_patterns`` solves 32 strata and takes
    seconds, and nothing below mutates it.
    """

    @classmethod
    def setUpClass(cls):
        cls.models, cls.phases, cls.pairs = real()
        cls.scale = full_scale(cls.models, cls.phases, cls.pairs)
        cls.pats = levels.level_patterns(cls.models, cls.phases, cls.pairs,
                                         32, seed=11)
        # Seeding is a property of the RNG, not of the stratum count, so the two
        # seed tests below share one small plan rather than re-solving 32.
        cls.few = levels.level_patterns(cls.models, cls.phases, cls.pairs,
                                        8, seed=11)

    def test_one_pattern_per_stratum(self):
        width = 0.995 / 32
        for i, p in enumerate(self.pats):
            self.assertGreaterEqual(p.target_alpha, i * width - 1e-12)
            self.assertLess(p.target_alpha, (i + 1) * width + 1e-12)

    def test_it_covers_the_whole_scale(self):
        alpha = np.array([p.alpha for p in self.pats])
        self.assertLess(alpha.min(), 0.05)
        self.assertGreater(alpha.max(), 0.90)
        # the failure this whole module exists to prevent: everything at the bottom
        self.assertGreater(np.median(alpha), 0.35)

    def test_the_achieved_level_is_what_predict_says(self):
        """alpha is recomputed AFTER rounding, so the CSV and the JSON agree."""
        for p in self.pats:
            got = float(predict(self.models, self.phases, self.pairs, p.x, p.w,
                                coherent=True)) / self.scale.fs
            self.assertAlmostEqual(got, p.alpha, places=9)

    def test_rounding_does_not_move_the_level(self):
        for p in self.pats:
            self.assertLess(abs(p.alpha - p.target_alpha), 2e-4)

    def test_every_drive_is_different(self):
        x = np.array([p.x for p in self.pats])
        self.assertEqual(len({tuple(row) for row in x}), len(self.pats))

    def test_seed_reproduces_the_run(self):
        again = levels.level_patterns(self.models, self.phases, self.pairs, 8, seed=11)
        for a, b in zip(self.few, again):
            np.testing.assert_array_equal(a.x, b.x)
            np.testing.assert_array_equal(a.w, b.w)

    def test_a_different_seed_does_not(self):
        other = levels.level_patterns(self.models, self.phases, self.pairs, 8, seed=12)
        same = sum(np.array_equal(a.x, b.x) for a, b in zip(self.few, other))
        self.assertEqual(same, 0)

    def test_drives_stay_in_range(self):
        for p in self.pats:
            self.assertTrue(np.all((p.x >= 0.0) & (p.x <= 1.0)))
            self.assertTrue(np.all((p.w >= 0.0) & (p.w <= 1.0)))


class RepeatTests(unittest.TestCase):
    """Block B."""

    @classmethod
    def setUpClass(cls):
        cls.models, cls.phases, cls.pairs = real()
        cls.scale = full_scale(cls.models, cls.phases, cls.pairs)
        cls.alphas = (0.1, 0.5, 1.0)
        cls.pats = levels.repeat_patterns(cls.models, cls.phases, cls.pairs,
                                          cls.alphas, 4, seed=5)

    def test_one_drive_per_level_repeated(self):
        self.assertEqual(len(self.pats), len(self.alphas) * 4)
        for i, _ in enumerate(self.alphas):
            group = self.pats[i * 4:(i + 1) * 4]
            for p in group[1:]:
                # byte-identical: B measures the machine, so the SLM pattern
                # must not move between repeats
                np.testing.assert_array_equal(p.x, group[0].x)
                np.testing.assert_array_equal(p.w, group[0].w)
            self.assertEqual([p.repeat_id for p in group], [1, 2, 3, 4])

    def test_levels_differ_from_each_other(self):
        first = [self.pats[i * 4] for i in range(len(self.alphas))]
        for a, b in zip(first, first[1:]):
            self.assertFalse(np.array_equal(a.x, b.x))

    def test_each_level_lands_where_it_was_asked(self):
        for p in self.pats:
            self.assertAlmostEqual(p.alpha, p.target_alpha, places=3)

    def test_alpha_one_uses_the_exact_ymax_drive(self):
        top = self.pats[-1]
        np.testing.assert_allclose(top.x, np.round(np.asarray(self.scale.drive), 6),
                                   atol=1e-9)
        np.testing.assert_array_equal(top.x, top.w)


class PsiScanTests(unittest.TestCase):
    """Block C."""

    def test_offset_zero_is_the_ymax_drive(self):
        models, phases, pairs = real()
        scale = full_scale(models, phases, pairs)
        pats = levels.psi_scan_patterns(models, phases, pairs)
        centre = [p for p in pats if abs(p.psi_deg - np.degrees(scale.psi)) < 1e-9]
        self.assertEqual(len(centre), 1)
        self.assertAlmostEqual(centre[0].alpha, 1.0, places=6)

    def test_the_scan_peaks_in_the_middle(self):
        models, phases, pairs = real()
        pats = levels.psi_scan_patterns(models, phases, pairs)
        alpha = [p.alpha for p in pats]
        self.assertEqual(int(np.argmax(alpha)), len(pats) // 2)
        self.assertLess(alpha[0], 1.0)
        self.assertLess(alpha[-1], 1.0)

    def test_rotating_off_axis_costs_when_the_phases_are_spread(self):
        models, phases, pairs = toy([0.1, 0.1, 0.1], [-2.0, 0.0, 2.0])
        pats = levels.psi_scan_patterns(models, phases, pairs, (-20, 0, 20))
        self.assertAlmostEqual(pats[1].alpha, 1.0, places=6)
        self.assertLess(pats[0].alpha, 1.0)
        self.assertLess(pats[2].alpha, 1.0)

    def test_clustered_phases_make_the_scan_one_sided(self):
        """The documented blind spot -- C cannot bracket a peak that is pinned.

        A pair's phase is Phi + t with t in [0, pi], so it can only be advanced.
        With every Phi equal, psi* sits at the edge of that reachable half-plane:
        one side of the fan moves the optimum interior and falls off, the other
        cannot rotate at all and pins every pair at x = w = 1, still at Y_max.
        A run whose phases are that well aligned gets no peak position out of C.
        """
        models, phases, pairs = toy([0.1, 0.1, 0.1], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(full_scale(models, phases, pairs).headroom, 1.0, places=9)
        alpha = [p.alpha for p in
                 levels.psi_scan_patterns(models, phases, pairs, (-20, 0, 20))]
        self.assertAlmostEqual(alpha[1], 1.0, places=9)
        pinned = [a for a in (alpha[0], alpha[2]) if a >= 1.0 - 1e-9]
        self.assertEqual(len(pinned), 1, f"expected exactly one pinned side, got {alpha}")

    def test_spread_phases_bracket_the_peak_on_both_sides(self):
        """The real runs: psi* interior, so the fan falls off either way."""
        models, phases, pairs = real()
        self.assertLess(full_scale(models, phases, pairs).headroom, 0.9)
        alpha = [p.alpha for p in
                 levels.psi_scan_patterns(models, phases, pairs, (-20, 0, 20))]
        self.assertAlmostEqual(alpha[1], 1.0, places=6)
        self.assertLess(alpha[0], 1.0)
        self.assertLess(alpha[2], 1.0)

    def test_the_default_fan_beats_the_read_noise(self):
        """+/-10 deg alone is under 1 sigma, so the fan has to reach further."""
        models, phases, pairs = real()
        scale = full_scale(models, phases, pairs)
        sigma = 0.264e-3 * np.sqrt(scale.fs * 1e3)     # the sigma.py law at the peak
        pats = {p.label: p for p in levels.psi_scan_patterns(models, phases, pairs)}
        widest = (1.0 - pats["C-45deg"].alpha) * scale.fs
        self.assertGreater(widest, 5.0 * sigma)

    def test_drives_x_equals_w(self):
        models, phases, pairs = real()
        for p in levels.psi_scan_patterns(models, phases, pairs):
            np.testing.assert_array_equal(p.x, p.w)


if __name__ == "__main__":
    unittest.main()
