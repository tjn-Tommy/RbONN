"""Cascaded step-7 references: reach every pair from ONE root, two hops if needed.

In:  the target pairs and a root reference.
Out: :func:`plan_cascade` lays out the stages before anything is driven;
     :func:`chain_fits` sums the hops into one spectrum against that root, and
     :func:`cascade_covariance` its covariance.  So the GUI and the offline
     runner both take one target list and one root and get one spectrum.

Step 7 measures a pair's comb phase by interfering it with a reference pair, so
one sweep yields ``Phi_k`` relative to whatever it was driven against.  The
catch is that a pair CANNOT be measured against its SLM neighbour: the column
order is ``x_n .. x_1 w_1 .. w_n``, so pair ``k`` sits next to ``k-1`` and
``k+1`` on BOTH sides, and the leakage between adjacent windows is not in the
model.  A run with reference 2 therefore cannot report pairs 1 and 3 at all.
The fix is to sweep them against a hub that is not their neighbour and merge
with ``Phi_k|2 = Phi_k|5 + Phi_5|2``.

Two questions, deliberately kept apart:

* FEASIBILITY is decided here, before anything is driven, from adjacency alone.
  ``pairs=[1,2,3], root=2`` has no answer -- 1 and 3 are both neighbours of 2
  and nothing else was measured to hop through -- and that is worth saying in
  the status line rather than an hour in.
* WHICH HUB is a tie among everything eligible, broken on the lowest index.
  Deliberately not scored on fit quality: the spread between the best and worst
  ELIGIBLE hub is well inside the accuracy this stage needs, and a runtime
  choice would mean the plan shown at the start is not the plan that runs.

The chained error is NOT two error bars in quadrature.  The hub's eta appears in
both hops -- as the target amplitude in ``Phi_h|root`` and as the reference
amplitude in ``Phi_k|h`` -- so its error moves them COHERENTLY and the
sensitivities are summed before squaring.  Measured on one ten-pair run that is
``+6.13 + (-9.54) = -3.41`` rad per unit eta rather than ``hypot(6.13, 9.54)``:
0.49 deg instead of 1.63, so treating the hops as independent overstates the
chained error roughly threefold.  :func:`cascade_covariance` does the same
bookkeeping across targets, where a shared hub correlates every pair that hopped
through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

# Minimum label separation between a target and the reference it is measured
# against.  2 == "not an immediate neighbour", which is the panel geometry
# talking, not a preference -- see the module docstring.
MIN_PAIR_SEP = 2


class CascadeError(ValueError):
    """No sequence of stages reaches every requested pair from the root."""


# ======================================================================
# planning
# ======================================================================

@dataclass(frozen=True)
class Stage:
    """One sweep: every pair in ``targets`` driven against the single ``ref``.

    ``depth`` is how many hops ``ref`` itself is from the root (0 for the root),
    so a stage's targets end up ``depth + 1`` hops out.
    """

    ref: int
    targets: tuple[int, ...]
    depth: int = 0

    def __len__(self) -> int:
        return len(self.targets)


def eligible(target: int, ref: int, *, min_sep: int = MIN_PAIR_SEP) -> bool:
    """Can ``target`` be measured against ``ref``?  Non-neighbours only."""
    return target != ref and abs(int(target) - int(ref)) >= int(min_sep)


def plan_cascade(
    pairs: Iterable[int],
    root: int,
    *,
    min_sep: int = MIN_PAIR_SEP,
    n_pairs: int | None = None,
) -> list[Stage]:
    """Stages that measure every pair in ``pairs`` back to ``root``.

    In:  ``pairs``, what you want phases FOR, and ``root``, which defines
         ``Phi = 0`` and is dropped from the list if it appears there.
         ``n_pairs`` only makes the failure message name real labels.
    Out: the stages.  Stage 1 is the root against everything it reaches
         directly; each later stage picks a hub from the pairs already measured,
         the one covering the most of what is left, ties on the lowest label.
         So the common case is two stages, known before the first pattern is
         written.  Raises :class:`CascadeError` if a pair can never be reached.

    >>> [(s.ref, s.targets) for s in plan_cascade(range(1, 11), 2)]
    [(2, (4, 5, 6, 7, 8, 9, 10)), (5, (1, 3))]
    """
    root = int(root)
    wanted = sorted({int(k) for k in pairs} - {root})
    if not wanted:
        raise CascadeError(
            "Need at least one target pair that is not the reference."
        )

    depth = {root: 0}
    remaining = list(wanted)
    stages: list[Stage] = []
    while remaining:
        best: tuple[int, list[int]] | None = None
        # sorted() is the tie-break: among hubs reaching the same number of
        # pairs the lowest label wins, and `>` (not `>=`) keeps it that way.
        for hub in sorted(depth):
            covered = [k for k in remaining
                       if eligible(k, hub, min_sep=min_sep)]
            if covered and (best is None or len(covered) > len(best[1])):
                best = (hub, covered)
        if best is None:
            raise CascadeError(_stranded_message(
                remaining, root, min_sep=min_sep, n_pairs=n_pairs,
                measured=sorted(depth),
            ))
        hub, covered = best
        stages.append(Stage(ref=hub, targets=tuple(covered), depth=depth[hub]))
        for k in covered:
            depth[k] = depth[hub] + 1
        remaining = [k for k in remaining if k not in depth]
    return stages


def _stranded_message(
    remaining: Sequence[int],
    root: int,
    *,
    min_sep: int,
    n_pairs: int | None,
    measured: Sequence[int],
) -> str:
    """Name the unreachable pairs AND what would rescue them.

    A bare "cannot reach pairs 1,3" sends you back to guess; the fix is always
    "put a far enough pair in the list", so say which ones those are.  Without
    ``n_pairs`` there is no honest upper bound to suggest from, so the message
    stops after the diagnosis.
    """
    stranded = ", ".join(str(k) for k in sorted(remaining))
    head = (
        f"Pair(s) {stranded} cannot be measured against reference {root} or "
        f"anything else in the list: a reference must be at least {min_sep} "
        f"labels away, and {', '.join(str(m) for m in measured)} "
        f"{'is' if len(measured) == 1 else 'are'} all this run would measure."
    )
    if n_pairs is None:
        return head
    rescuers = [h for h in range(1, int(n_pairs) + 1)
                if h not in measured
                and all(eligible(k, h, min_sep=min_sep) for k in remaining)]
    if not rescuers:
        return head + "  No pair in the layout could bridge them."
    return head + (
        f"  Add {_compact(rescuers)} to the target list to bridge them, "
        f"or choose another reference."
    )


def _compact(values: Sequence[int]) -> str:
    """[5,6,7,9] -> "5-7,9" -- a label list that stays one line at ten pairs."""
    out: list[str] = []
    for v in sorted(values):
        if out and v == int(out[-1].split("-")[-1]) + 1:
            out[-1] = f"{out[-1].split('-')[0]}-{v}"
        else:
            out.append(str(v))
    return ",".join(out)


def plan_lines(stages: Sequence[Stage], root: int,
               seconds_per_target: float | None = None) -> list[str]:
    """The plan as text, for the progress dialog's log and the runner's stdout.

    One line per stage plus a total::

        Stage 1/2 · ref 2 -> 4-10  (7 targets, ~72 min)
        Stage 2/2 · ref 5 -> 1,3   (2 targets, ~21 min, chained via 5)
        Total 9 targets, ~93 min

    ASCII apart from the separator, which is cp1252-safe: the same lines go to a
    Windows console, where a nicer arrow is a UnicodeEncodeError.
    """
    lines: list[str] = []
    n_total = 0
    for i, stage in enumerate(stages, start=1):
        n_total += len(stage)
        bits = [f"{len(stage)} target{'s' if len(stage) != 1 else ''}"]
        if seconds_per_target:
            bits.append(f"~{seconds_per_target * len(stage) / 60:.0f} min")
        if stage.depth:
            bits.append(f"chained via {stage.ref}")
        lines.append(
            f"Stage {i}/{len(stages)} \N{MIDDLE DOT} ref {stage.ref} "
            f"-> {_compact(stage.targets)}  ({', '.join(bits)})"
        )
    total = f"Total {n_total} target{'s' if n_total != 1 else ''}"
    if seconds_per_target:
        total += f", ~{seconds_per_target * n_total / 60:.0f} min"
    lines.append(total)
    return lines


# ======================================================================
# chaining
# ======================================================================

def wrap_pi(angle: float) -> float:
    """Wrap to (-pi, pi], the interval every stored ``dphi_comb`` lives on."""
    return float(np.angle(np.exp(1j * float(angle))))


@dataclass
class ChainedPhase:
    """One pair's comb phase referred to the ROOT, however many hops it took.

    ``dphi`` is the wrapped sum along the path; ``dphi_raw`` is the unwrapped
    one, kept because a chain landing near +/-pi is exactly where the wrap could
    have gone the other way.

    The error is carried in the two pieces it is made of, because they combine
    differently ACROSS pairs (see :func:`cascade_covariance`):

    * ``edge_var`` -- fringe-noise variance per measured hop, keyed by
      ``(target, reference)``.  Independent between hops, SHARED between two
      pairs whose paths cross the same hop.
    * ``deta`` -- ``d(dphi)/d(eta_p)`` per pair, already summed over the hops
      touching it.  That sum is the point: the hub's eta enters one hop as the
      target amplitude and the next as the reference, so the two sensitivities
      partially cancel.
    """

    tgt: int
    root: int
    path: tuple[int, ...]              # references from the near hop out to root
    dphi: float
    dphi_raw: float
    edge_var: dict[tuple[int, int], float] = field(default_factory=dict)
    deta: dict[int, float] = field(default_factory=dict)
    eta_err: dict[int, float] = field(default_factory=dict)

    @property
    def via(self) -> int | None:
        """The pair this one was actually swept against -- the root, if directly."""
        return self.path[0] if self.path else None

    @property
    def chained(self) -> bool:
        """Did this pair need a hub?  A direct sweep has the root as its only hop."""
        return len(self.path) > 1

    @property
    def dphi_deg(self) -> float:
        return float(np.degrees(self.dphi))

    @property
    def err_fringe(self) -> float:
        """Fringe noise along the whole path (rad): hops are independent."""
        return float(np.sqrt(sum(self.edge_var.values())))

    @property
    def err_eta(self) -> float:
        """Pinned-step-6-eta term (rad), sensitivities summed per pair first."""
        return float(np.sqrt(sum(
            (c * self.eta_err.get(p, 0.0)) ** 2 for p, c in self.deta.items()
        )))

    @property
    def err_total(self) -> float:
        return float(np.hypot(self.err_fringe, self.err_eta))


def chain_fits(
    fits: Mapping[int, object],
    stages: Sequence[Stage],
    root: int,
) -> dict[int, ChainedPhase]:
    """Refer every fitted pair back to ``root``: ``{tgt: ChainedPhase}``.

    In:  ``fits``, each target pair against the
         :class:`~calibration_module.fit.phase.PhaseFit` of the hop that
         MEASURED it, one entry per pair whichever stage it came from.  Only
         :func:`~calibration_module.fit.phase.fit_phase_fixed` fills the
         ``dphi_deta_*`` sensitivities, so only its results carry an
         ``err_eta``; anything else chains with fringe noise alone.
    Out: one :class:`ChainedPhase` per target.  Stages are walked in order, so a
         hub resolves before what hangs off it, and a target whose hub has no
         fit is left out rather than chained onto nothing.
    """
    root = int(root)
    chains: dict[int, ChainedPhase] = {
        root: ChainedPhase(tgt=root, root=root, path=(), dphi=0.0, dphi_raw=0.0)
    }
    for stage in stages:
        base = chains.get(stage.ref)
        if base is None:
            continue                       # hub never fitted -> nothing hangs off it
        for k in stage.targets:
            fit = fits.get(k)
            if fit is None:
                continue
            edge = (int(k), int(stage.ref))
            edge_var = dict(base.edge_var)
            edge_var[edge] = float(fit.dphi_comb_err) ** 2
            deta = dict(base.deta)
            eta_err = dict(base.eta_err)
            # Coherent accumulation: += , never a fresh assignment.  When the
            # hub appears again here as this hop's REFERENCE, its two
            # sensitivities have to land in the same bucket to cancel.
            deta[k] = deta.get(k, 0.0) + float(fit.dphi_deta_tgt)
            deta[stage.ref] = deta.get(stage.ref, 0.0) + float(fit.dphi_deta_ref)
            eta_err.setdefault(k, float(fit.eta_tgt_err))
            eta_err.setdefault(stage.ref, float(fit.eta_ref_err))
            raw = float(fit.dphi_comb) + base.dphi_raw
            chains[k] = ChainedPhase(
                tgt=int(k), root=root,
                path=(int(stage.ref),) + base.path,
                dphi=wrap_pi(raw), dphi_raw=raw,
                edge_var=edge_var, deta=deta, eta_err=eta_err,
            )
    chains.pop(root, None)                 # the root is not a measured target
    return chains


def cascade_covariance(
    chains: Mapping[int, ChainedPhase], pairs: Sequence[int]
) -> np.ndarray:
    """Covariance of the root-referred phases, in the order ``pairs`` gives.

    Two pairs are correlated wherever their paths overlap -- a shared hop, or
    any pair whose eta both are pinned to::

        cov_ij = sum_{hops on both paths} sigma_hop^2
               + sum_p  deta_i[p] * deta_j[p] * eta_err_p^2

    With a flat plan every path is one hop and the only shared pair is the root,
    so this collapses to
    :func:`~calibration_module.fit.phase.phase_covariance`'s rank-1 term.  Pass
    it to :func:`~calibration_module.fit.phase.fit_beta2` for a cascaded
    spectrum: the flat version reads one common ``eta_ref_err`` off the first
    fit and would mis-weight a run with two references.

    The diagonal is :attr:`ChainedPhase.err_total` squared, so ``sqrt(diag)``
    agrees with what the table quotes per pair.
    """
    pairs = [int(k) for k in pairs]
    if not pairs:
        raise ValueError("no target pairs to build a covariance over")
    missing = [k for k in pairs if k not in chains]
    if missing:
        raise KeyError(f"no chained phase for pair(s) {missing}")
    n = len(pairs)
    cov = np.zeros((n, n), dtype=float)
    for i, ki in enumerate(pairs):
        ci = chains[ki]
        for j, kj in enumerate(pairs):
            cj = chains[kj]
            shared = set(ci.edge_var) & set(cj.edge_var)
            total = sum(ci.edge_var[e] for e in shared)
            for p, a in ci.deta.items():
                b = cj.deta.get(p)
                if b:
                    s = ci.eta_err.get(p, 0.0)
                    total += a * b * s * s
            cov[i, j] = total
    return cov


def chained_payload(chain: ChainedPhase, fit) -> dict:
    """The ``"chained"`` block stored beside a chained pair's fit in the JSON.

    Only written for a pair that actually hopped.  It records what the stored
    (root-referred) phase was built FROM -- the hop as measured and the hub's
    own phase -- so a merged spectrum can still be taken apart, which is the one
    thing the 0916 hand merge got right and worth keeping.
    """
    return {
        "via": chain.via,
        "path": list(chain.path),
        "dphi_edge_rad": float(fit.dphi_comb),
        "dphi_edge_deg": float(np.degrees(fit.dphi_comb)),
        "dphi_hub_rad": float(chain.dphi_raw - fit.dphi_comb),
        "dphi_hub_deg": float(np.degrees(chain.dphi_raw - fit.dphi_comb)),
        "dphi_unwrapped_rad": float(chain.dphi_raw),
        "edges": [list(e) for e in sorted(chain.edge_var)],
        "deta": {str(p): float(c) for p, c in sorted(chain.deta.items())},
        "eta_err": {str(p): float(s) for p, s in sorted(chain.eta_err.items())},
    }


__all__ = [
    "MIN_PAIR_SEP",
    "CascadeError",
    "Stage",
    "ChainedPhase",
    "eligible",
    "plan_cascade",
    "plan_lines",
    "wrap_pi",
    "chain_fits",
    "cascade_covariance",
    "chained_payload",
]
