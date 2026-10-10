"""Step-7 cascade: the plan, the chained phase, and its error bars.

The reference the arithmetic is checked against is the 0916 ten-pair run, which
was measured in two sweeps (4-10 vs reference 2, then 1 and 3 vs reference 5)
and merged BY HAND into ``calib_step7_result_0916_merged_ref2.json``.  That file
is the specification: anything this module computes automatically has to come
out of it unchanged, correlation coefficient included.
"""
from __future__ import annotations

import json
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.cascade import (  # noqa: E402
    CascadeError,
    Stage,
    cascade_covariance,
    chain_fits,
    chained_payload,
    eligible,
    plan_cascade,
    plan_lines,
    wrap_pi,
)

RUN = REPO_ROOT / "src/calib_data/archive/2026-09/run_0916_10pairs_w15g35"
# The two stages of one cascaded run.  A run directory holds ONE
# calib.json, whose step7 section is the merged spectrum, so the
# per-stage fits live here as fixtures trimmed to what this test reads.
STAGE1 = REPO_ROOT / "tests/fixtures/cascade_0916/stage1_ref2.json"   # 4-10 vs ref 2
STAGE2 = REPO_ROOT / "tests/fixtures/cascade_0916/stage2_ref5.json"   # 1,3  vs ref 5
MERGED = RUN / "calib.json"


def _fits(path: Path) -> dict:
    """The stored fits as the duck type ``chain_fits`` actually consumes."""
    out = {}
    for ch in json.loads(path.read_text(encoding="utf-8"))["step7"]["channels"]:
        d = ch["fit"]
        out[int(ch["tgt_index"])] = types.SimpleNamespace(
            dphi_comb=d["dphi_comb_rad"],
            dphi_comb_err=d["dphi_comb_err_rad"],
            dphi_deta_tgt=d["dphi_deta_tgt"],
            dphi_deta_ref=d["dphi_deta_ref"],
            eta_tgt_err=d["eta_tgt_err"],
            eta_ref_err=d["eta_ref_err"],
            # not used by chain_fits; phase_covariance reads it to tell a
            # single-reference set of fits from a cascaded one
            eta_ref=d["eta_ref"],
            eta_tgt=d["eta_tgt"],
        )
    return out


class PlanTests(unittest.TestCase):
    """What gets swept against what, decided before anything is driven."""

    def test_the_0916_run_is_what_the_planner_would_have_chosen(self) -> None:
        stages = plan_cascade(range(1, 11), 2)
        self.assertEqual([(s.ref, s.targets) for s in stages],
                         [(2, (4, 5, 6, 7, 8, 9, 10)), (5, (1, 3))])
        self.assertEqual([s.depth for s in stages], [0, 1])

    def test_no_neighbour_is_ever_measured_against_its_reference(self) -> None:
        for root in range(1, 11):
            for stage in plan_cascade(range(1, 11), root):
                for k in stage.targets:
                    self.assertGreaterEqual(
                        abs(k - stage.ref), 2,
                        f"root {root}: pair {k} against neighbour {stage.ref}",
                    )

    def test_every_requested_pair_is_reached_exactly_once(self) -> None:
        for root in range(1, 11):
            measured = [k for s in plan_cascade(range(1, 11), root)
                        for k in s.targets]
            self.assertEqual(sorted(measured),
                             [k for k in range(1, 11) if k != root])
            self.assertEqual(len(measured), len(set(measured)))

    def test_a_hub_is_always_a_pair_measured_earlier(self) -> None:
        known = {2}
        for stage in plan_cascade(range(1, 11), 2):
            self.assertIn(stage.ref, known)
            known.update(stage.targets)

    def test_no_neighbours_means_no_second_stage(self) -> None:
        """The old single-reference run is still a one-stage plan."""
        stages = plan_cascade([4, 5, 6], 2)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0], Stage(ref=2, targets=(4, 5, 6), depth=0))

    def test_the_hub_is_the_lowest_that_does_the_job(self) -> None:
        """5..10 all reach {1,3} in one stage; the tie breaks on the label."""
        for hub in range(5, 11):
            self.assertTrue(eligible(1, hub) and eligible(3, hub))
        self.assertEqual(plan_cascade(range(1, 11), 2)[1].ref, 5)

    def test_a_hub_that_covers_more_beats_a_lower_one_that_covers_less(self) -> None:
        """4 reaches only pair 1; 5 reaches both, so coverage wins over label."""
        self.assertTrue(eligible(1, 4))
        self.assertFalse(eligible(3, 4))
        self.assertEqual(len(plan_cascade(range(1, 11), 2)), 2)

    def test_an_unreachable_pair_is_refused_before_the_bench(self) -> None:
        with self.assertRaises(CascadeError) as caught:
            plan_cascade([1, 2, 3], 2, n_pairs=10)
        message = str(caught.exception)
        self.assertIn("1, 3", message)
        self.assertIn("reference 2", message)
        self.assertIn("5-10", message)         # what would rescue them

    def test_a_reachable_neighbour_is_not_refused(self) -> None:
        """[1,2,5] works: 5 is direct, and 1 hops through it."""
        stages = plan_cascade([1, 2, 5], 2)
        self.assertEqual([(s.ref, s.targets) for s in stages],
                         [(2, (5,)), (5, (1,))])

    def test_a_layout_too_small_to_bridge_says_so(self) -> None:
        with self.assertRaises(CascadeError) as caught:
            plan_cascade([1, 2, 3], 2, n_pairs=3)
        self.assertIn("No pair in the layout", str(caught.exception))

    def test_a_cascade_error_reaches_a_value_error_handler(self) -> None:
        """The GUI catches ValueError; a new exception type must not escape."""
        self.assertTrue(issubclass(CascadeError, ValueError))

    def test_the_plan_text_names_every_stage_and_the_hop(self) -> None:
        lines = plan_lines(plan_cascade(range(1, 11), 2), 2,
                           seconds_per_target=600.0)
        self.assertEqual(len(lines), 3)
        self.assertIn("ref 2 -> 4-10", lines[0])
        self.assertIn("ref 5 -> 1,3", lines[1])
        self.assertIn("chained via 5", lines[1])
        self.assertIn("~90 min", lines[2])

    def test_the_plan_text_survives_a_windows_console(self) -> None:
        """The runner prints these to stdout, which is cp1252 on this bench."""
        for line in plan_lines(plan_cascade(range(1, 11), 2), 2):
            line.encode("cp1252")              # must not raise


class ChainTests(unittest.TestCase):
    """Phi_k|root = Phi_k|hub + Phi_hub|root, against the hand merge."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fits = {**_fits(STAGE1), **_fits(STAGE2)}
        cls.stages = plan_cascade(range(1, 11), 2)
        cls.chains = chain_fits(cls.fits, cls.stages, 2)
        cls.merged = {int(c["tgt_index"]): c["fit"] for c in
                      json.loads(MERGED.read_text(encoding="utf-8"))["step7"]["channels"]}

    def test_every_pair_reproduces_the_hand_merged_phase(self) -> None:
        for k, chain in sorted(self.chains.items()):
            self.assertAlmostEqual(chain.dphi, self.merged[k]["dphi_comb_rad"],
                                   places=12, msg=f"pair {k}")

    def test_the_fringe_error_gains_the_hub_hop_in_quadrature(self) -> None:
        for k, chain in sorted(self.chains.items()):
            self.assertAlmostEqual(chain.err_fringe,
                                   self.merged[k]["dphi_comb_err_rad"],
                                   places=12, msg=f"pair {k}")
        # explicitly, for the one that hopped
        self.assertAlmostEqual(
            self.chains[1].err_fringe,
            math.hypot(self.fits[1].dphi_comb_err, self.fits[5].dphi_comb_err),
            places=12)

    def test_the_hub_eta_is_summed_coherently_not_in_quadrature(self) -> None:
        """The hub's two roles partially cancel; treating them as independent
        would overstate pair 1's eta error roughly threefold."""
        chain = self.chains[1]
        coherent = self.fits[5].dphi_deta_tgt + self.fits[1].dphi_deta_ref
        self.assertAlmostEqual(chain.deta[5], coherent, places=12)
        self.assertAlmostEqual(chain.deta[5], -3.4106539, places=6)
        independent = math.hypot(self.fits[5].dphi_deta_tgt,
                                 self.fits[1].dphi_deta_ref)
        self.assertLess(abs(coherent), 0.4 * independent)
        self.assertAlmostEqual(chain.err_eta,
                               self.merged[1]["dphi_comb_err_eta_rad"],
                               places=12)

    def test_the_total_is_the_two_parts_in_quadrature(self) -> None:
        for k, chain in sorted(self.chains.items()):
            self.assertAlmostEqual(
                chain.err_total, math.hypot(chain.err_fringe, chain.err_eta),
                places=12, msg=f"pair {k}")
            self.assertAlmostEqual(chain.err_total,
                                   self.merged[k]["dphi_comb_err_total_rad"],
                                   places=12, msg=f"pair {k}")

    def test_a_direct_pair_is_left_exactly_as_fitted(self) -> None:
        for k in (4, 5, 6, 7, 8, 9, 10):
            chain, fit = self.chains[k], self.fits[k]
            self.assertFalse(chain.chained)
            self.assertEqual(chain.via, 2)
            self.assertAlmostEqual(chain.dphi, fit.dphi_comb, places=12)
            self.assertAlmostEqual(chain.err_fringe, fit.dphi_comb_err,
                                   places=12)

    def test_only_the_hopped_pairs_are_marked_chained(self) -> None:
        self.assertEqual(sorted(k for k, c in self.chains.items() if c.chained),
                         [1, 3])
        self.assertEqual(self.chains[1].via, 5)
        self.assertEqual(self.chains[1].path, (5, 2))

    def test_the_root_is_not_a_result(self) -> None:
        self.assertNotIn(2, self.chains)

    def test_a_pair_whose_hub_failed_is_dropped_not_chained_to_nothing(self) -> None:
        fits = {k: v for k, v in self.fits.items() if k != 5}
        chains = chain_fits(fits, self.stages, 2)
        self.assertNotIn(1, chains)            # measured, but no route home
        self.assertNotIn(3, chains)
        self.assertIn(4, chains)

    def test_the_sum_is_wrapped_and_the_raw_sum_kept(self) -> None:
        raw = self.fits[1].dphi_comb + self.fits[5].dphi_comb
        self.assertAlmostEqual(self.chains[1].dphi_raw, raw, places=12)
        self.assertAlmostEqual(self.chains[1].dphi, wrap_pi(raw), places=12)
        self.assertLessEqual(abs(self.chains[1].dphi), math.pi)

    def test_wrapping_takes_a_sum_past_pi_back_into_range(self) -> None:
        self.assertAlmostEqual(wrap_pi(3.5), 3.5 - 2 * math.pi, places=12)
        self.assertAlmostEqual(wrap_pi(math.pi), math.pi, places=12)

    def test_the_chained_block_can_be_taken_apart_again(self) -> None:
        block = chained_payload(self.chains[1], self.fits[1])
        self.assertEqual(block["via"], 5)
        self.assertAlmostEqual(block["dphi_edge_rad"], self.fits[1].dphi_comb,
                               places=12)
        self.assertAlmostEqual(block["dphi_hub_rad"], self.fits[5].dphi_comb,
                               places=12)
        self.assertAlmostEqual(
            block["dphi_edge_rad"] + block["dphi_hub_rad"],
            self.chains[1].dphi_raw, places=12)


class CovarianceTests(unittest.TestCase):
    """Pairs sharing a hub are correlated, and by how much the merge said."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fits = {**_fits(STAGE1), **_fits(STAGE2)}
        cls.chains = chain_fits(cls.fits, plan_cascade(range(1, 11), 2), 2)
        cls.pairs = sorted(cls.chains)
        cls.cov = cascade_covariance(cls.chains, cls.pairs)

    def test_the_diagonal_is_the_error_bar_the_table_quotes(self) -> None:
        for i, k in enumerate(self.pairs):
            self.assertAlmostEqual(math.sqrt(self.cov[i, i]),
                                   self.chains[k].err_total,
                                   places=12, msg=f"pair {k}")

    def test_two_pairs_through_one_hub_carry_the_recorded_correlation(self) -> None:
        i, j = self.pairs.index(1), self.pairs.index(3)
        r = self.cov[i, j] / math.sqrt(self.cov[i, i] * self.cov[j, j])
        self.assertAlmostEqual(r, 0.52718056850019, places=12)

    def test_the_matrix_is_symmetric_and_positive_definite(self) -> None:
        np.testing.assert_allclose(self.cov, self.cov.T, rtol=0, atol=1e-18)
        self.assertGreater(np.min(np.linalg.eigvalsh(self.cov)), 0.0)

    def test_direct_pairs_are_still_correlated_through_the_root_eta(self) -> None:
        i, j = self.pairs.index(4), self.pairs.index(6)
        self.assertNotEqual(self.cov[i, j], 0.0)

    def test_a_missing_pair_is_a_named_error(self) -> None:
        with self.assertRaises(KeyError):
            cascade_covariance(self.chains, [1, 99])

    def test_a_flat_plan_agrees_with_the_single_reference_matrix(self) -> None:
        """One stage: this must equal what phase_covariance builds."""
        from calibration_module.fit.phase import phase_covariance
        fits = _fits(STAGE1)
        pairs = sorted(fits)
        chains = chain_fits(fits, plan_cascade(pairs, 2), 2)
        np.testing.assert_allclose(
            cascade_covariance(chains, pairs),
            phase_covariance(fits, pairs), rtol=1e-12, atol=0.0,
        )

    def test_the_flat_matrix_refuses_a_cascade_rather_than_mis_weighting(self) -> None:
        from calibration_module.fit.phase import phase_covariance
        with self.assertRaises(ValueError) as caught:
            phase_covariance(self.fits, self.pairs)
        self.assertIn("cascade_covariance", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
