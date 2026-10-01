"""Bench calibration in the form :mod:`onn_module.layer` wants.

This module is the bridge between the calibration chain and the torch layer,
and it is deliberately the only place the two meet.  It parses no calibration
JSON of its own: ``eta`` and ``Phi`` come from
:func:`calibration_module.fit.verify.load_forward_model`, the one reader that
defines this project's phase-sign convention.  There is therefore exactly one
answer to "what is Phi_k" in the repo, and re-deriving it here -- which is what
an independent JSON walk amounts to -- is how the two would drift apart.

``phase_sign``
--------------
The project has ONE sign convention (see the README): a stored
``dphi_comb_rad`` IS ``Phi_k`` in

    E = sum_k eta_k sqrt(x_k w_k) exp(i [u_k + v_k + Phi_k])

so every loader here sets ``phase_sign = +1`` and that is not a choice.  The
field survives on :class:`ONNParams` for two reasons only: the flat template
schema can express the ``exp(i[Phi - u - v])`` convention of
``rbonn_3pair_sweep.py`` for a side-by-side comparison, and the layer's
selftest exercises both branches.  Nothing that reads a real calibration
should set it.

``y_sat``
---------
Step 8 fits the saturation level and writes ``y_sat_v`` beside the
``step7_json`` it resolved its forward model from, so
:meth:`ONNParams.from_step8` takes ``y_sat`` from the step-8 result and
``eta``/``phi_rad`` from the step-7 file it names -- one calibration, reached
through the artifact that measured the saturation against it.  (A step-8 result
that carries ``eta``/``phi_rad`` inline is read directly; none has ever been
written that way.)  There is no default saturation level in this module on
purpose: a hardcoded 5 V that silently disagrees with the run it is quoted
against is worse than an explicit ``y_sat=None`` (no saturation).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor

from calibration_module.fit.verify import load_forward_model

__all__ = ["ONNParams", "write_template"]


def _as_vec(value, k: int, name: str) -> Tensor:
    """Scalar or length-k sequence -> float tensor of shape (k,)."""
    if value is None:
        value = 0.0
    if isinstance(value, (int, float)):
        return torch.full((k,), float(value))
    t = torch.as_tensor(value, dtype=torch.get_default_dtype()).flatten()
    if t.numel() == 1:
        return t.expand(k).clone()
    if t.numel() != k:
        raise ValueError(f"{name}: expected 1 or {k} values, got {t.numel()}")
    return t.clone()


def _y_sat(value) -> float:
    """File/CLI saturation level -> the layer's float.  None or <= 0 = linear."""
    if value is None:
        return math.inf
    value = float(value)
    return math.inf if value <= 0 else value


@dataclass
class ONNParams:
    """Fitted hardware calibration for K channels.

    ``channels`` holds the bench pair indices in the same order as ``eta`` and
    ``phi``, so a fit can always be traced back to the pair it was measured on.
    ``reference`` is the step-7 reference pair, the one with ``Phi = 0``.
    """

    eta: Tensor                       # (K,) field coupling, |E| scale
    phi: Tensor                       # (K,) static phase [rad], Phi_k
    y_sat: float = math.inf           # saturation scale; inf = no saturation
    phase_sign: float = 1.0           # +1 = exp(i(u+v+Phi)); see the module docstring
    channels: Optional[list] = None   # bench pair indices, for provenance
    source: str = "<none>"
    eta_err: Optional[Tensor] = None  # (K,) step-6 sigma, carried for propagation
    reference: Optional[int] = None   # step-7 reference pair (Phi = 0)

    @property
    def n_channels(self) -> int:
        return int(self.eta.numel())

    # -- constructors ----------------------------------------------------
    @staticmethod
    def ideal(n_channels: int, y_sat: float | None = None) -> "ONNParams":
        """Lossless, phase-flat reference optics."""
        k = n_channels
        return ONNParams(
            eta=torch.ones(k), phi=torch.zeros(k), y_sat=_y_sat(y_sat),
            channels=list(range(k)), source="ideal",
        )

    @staticmethod
    def from_step7(path: str | Path, *, pairs: Sequence[int] | None = None,
                   y_sat: float | None = None,
                   method: str | None = None) -> "ONNParams":
        """Load a combined step-7 JSON through ``verify.load_forward_model``.

        ``pairs=None`` takes every pair the file calibrates end to end (a step-6
        eta and, unless it is the reference, a step-7 phase).  A step-7 file
        carries no saturation level, so ``y_sat`` has to be passed or the layer
        runs linear -- use :meth:`from_step8` to get it from a measurement.
        """
        models, phases, ref = load_forward_model(path, pairs=pairs, method=method)
        idx = sorted(models)
        return ONNParams(
            eta=torch.tensor([float(models[k].eta) for k in idx]),
            phi=torch.tensor([float(phases[k]) for k in idx]),
            eta_err=torch.tensor([float(models[k].eta_err) for k in idx]),
            y_sat=_y_sat(y_sat), phase_sign=1.0,
            channels=idx, reference=int(ref), source=str(path),
        )

    @staticmethod
    def from_step8(path: str | Path, *, pairs: Sequence[int] | None = None,
                   y_sat: float | None = None) -> "ONNParams":
        """Load a step-8 / system-accuracy JSON: eta, Phi and Y_sat in one file.

        This is the preferred loader.  Step 8 resolves the forward model once
        and stores what it ran with -- ``eta``, ``phi_rad``, ``y_sat_v`` and the
        ``step7_json`` it came from -- so the layer is guaranteed to be the model
        the bench was last measured against.  An explicit ``y_sat`` overrides the
        file's.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if "eta" not in d or "phi_rad" not in d:
            return ONNParams._step8_via_step7(path, d, pairs=pairs, y_sat=y_sat)
        eta_d = {int(k): float(v) for k, v in d["eta"].items()}
        phi_d = {int(k): float(v) for k, v in d["phi_rad"].items()}
        idx = sorted(eta_d) if pairs is None else sorted(int(p) for p in pairs)
        missing = [k for k in idx if k not in eta_d or k not in phi_d]
        if missing:
            raise KeyError(f"pair(s) {missing} are not in {path.name} "
                           f"(it covers {sorted(eta_d)})")
        return ONNParams(
            eta=torch.tensor([eta_d[k] for k in idx]),
            phi=torch.tensor([phi_d[k] for k in idx]),
            y_sat=_y_sat(d.get("y_sat_v") if y_sat is None else y_sat),
            phase_sign=1.0, channels=idx,
            reference=None if d.get("reference") is None else int(d["reference"]),
            source=str(path),
        )

    @staticmethod
    def _step8_via_step7(path: Path, d: dict, *, pairs, y_sat) -> "ONNParams":
        """A step-8 result that stores only ``y_sat_v`` and a path to its step 7.

        Step 8 has never written ``eta``/``phi_rad`` of its own -- it records
        ``step7_json``, the file it resolved its forward model from.  Following
        that path is better than copying the numbers in: one source, and a
        step-8 result cannot drift from the calibration it was measured under.

        The recorded path is absolute, so a moved calib_data directory breaks
        it; the step-7 file is then looked for beside the step-8 file, which is
        where a run directory keeps them both.
        """
        if "y_sat_v" not in d:
            raise KeyError(f"{path.name} has neither 'eta'/'phi_rad' nor 'y_sat_v': "
                           f"not a step-8 result (keys: {sorted(d)})")
        recorded = d.get("step7_json")
        if not recorded:
            raise KeyError(f"{path.name} has no 'eta'/'phi_rad' and no 'step7_json' "
                           f"to resolve them from")
        for cand in (Path(recorded), path.parent / Path(recorded).name):
            if cand.is_file():
                break
        else:
            raise FileNotFoundError(
                f"{path.name} points at {recorded}, which is not there, and no "
                f"{Path(recorded).name} sits beside it -- pass the step-7 file "
                f"directly with an explicit y_sat")
        p = ONNParams.from_step7(
            cand, pairs=pairs,
            y_sat=d["y_sat_v"] if y_sat is None else y_sat)
        p.source = f"{path} (y_sat) + {cand} (eta, phi)"
        return p

    @staticmethod
    def from_json(path: str | Path, channels: Sequence[int] | None = None,
                  y_sat: float | None = None) -> "ONNParams":
        """Dispatch on what the file is: step-8 result, step-7 result, or template.

        ``y_sat`` overrides whatever the file says, and sets the units of ``Y``:
        pass the saturation level in detector volts and ``Y`` comes out in volts.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if "phi_rad" in d and "eta" in d and isinstance(d["eta"], dict):
            return ONNParams.from_step8(path, pairs=channels, y_sat=y_sat)
        if "y_sat_v" in d and "step7_json" in d:      # a step-8 result, lean form
            return ONNParams.from_step8(path, pairs=channels, y_sat=y_sat)
        if "step7" in d or "step6" in d:
            return ONNParams.from_step7(path, pairs=channels, y_sat=y_sat)
        p = ONNParams._from_flat(d, channels)
        if y_sat is not None:
            p.y_sat = _y_sat(y_sat)
        p.source = str(path)
        return p

    @staticmethod
    def _from_flat(d: dict, channels: Sequence[int] | None) -> "ONNParams":
        """The hand-written ``onn_layer_v1`` schema -- see :func:`write_template`."""
        k = int(d.get("n_channels") or len(d["eta"]))
        deg = str(d.get("phase_units", "rad")).startswith("deg")
        p = ONNParams(
            eta=_as_vec(d.get("eta", 1.0), k, "eta"),
            phi=_as_vec(d.get("phi", 0.0), k, "phi") * (math.pi / 180.0 if deg else 1.0),
            y_sat=_y_sat(d.get("y_sat")),
            phase_sign=float(d.get("phase_sign", 1.0)),
            channels=list(d.get("channels", range(k))),
        )
        return p.select(channels) if channels is not None else p

    # -- reshaping -------------------------------------------------------
    def select(self, channels: Sequence[int]) -> "ONNParams":
        pos = {c: j for j, c in enumerate(self.channels or range(self.n_channels))}
        missing = [c for c in channels if c not in pos]
        if missing:
            raise KeyError(f"channel(s) {missing} not in {sorted(pos)}")
        j = torch.tensor([pos[c] for c in channels])
        return ONNParams(
            self.eta[j], self.phi[j], self.y_sat, self.phase_sign, list(channels),
            self.source, None if self.eta_err is None else self.eta_err[j],
            self.reference,
        )

    def tile_to(self, k: int) -> "ONNParams":
        """Repeat the calibration cyclically to fill k channels.

        For trying a layer wider than the bench (8 fitted pairs -> 784 inputs).
        Physically this pretends the same K pairs exist many times over; it is
        a simulation convenience, not a measured device.
        """
        n = self.n_channels
        if k == n:
            return self
        rep = -(-k // n)
        j = torch.arange(n).repeat(rep)[:k]
        return ONNParams(
            self.eta[j], self.phi[j], self.y_sat, self.phase_sign,
            [(self.channels or list(range(n)))[i] for i in j.tolist()],
            self.source + f" (tiled {n}->{k})",
            None if self.eta_err is None else self.eta_err[j], self.reference,
        )

    def summary(self) -> str:
        f = lambda t: f"[{t.min():.4g}, {t.max():.4g}] mean {t.mean():.4g}"
        # The free-phase bound, for scale only: it is NOT the reachable maximum,
        # because a pair only reaches |z| = eta with its phase pinned to Phi.
        # fit.verify.reachable_max is the real ceiling.
        i_free = float(self.eta.sum()) ** 2
        return (f"ONNParams from {self.source}\n"
                f"  channels   {self.n_channels}: {self.channels}"
                f"{'' if self.reference is None else f'   ref {self.reference}'}\n"
                f"  eta        {f(self.eta)}\n"
                f"  phi [deg]  {f(torch.rad2deg(self.phi))}\n"
                f"  y_sat      {self.y_sat}   phase_sign {self.phase_sign:+.0f}\n"
                f"  free-phase (sum eta)^2 = {i_free:.6g} V  "
                f"(an over-estimate of Y_max; see fit.verify.reachable_max)")


def write_template(path: str | Path, n_channels: int = 8) -> None:
    """Emit the flat JSON schema, filled with an ideal device."""
    tpl = {
        "schema": "onn_layer_v1",
        "n_channels": n_channels,
        "channels": list(range(n_channels)),
        "phase_units": "rad",
        "phase_sign": 1,
        "_comment": "eta/phi: one value per channel, or a single scalar to "
                    "broadcast. y_sat: null disables saturation. phase_sign is "
                    "+1 for this project's convention -- see onn_module.params.",
        "eta": [1.0] * n_channels,
        "phi": [0.0] * n_channels,
        "y_sat": None,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(tpl, fh, indent=2)
