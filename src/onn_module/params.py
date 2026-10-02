"""Bench calibration in the form :mod:`onn_module.layer` wants.

In:  a ``calib.json``, or the directory holding one, or a flat template.
Out: :class:`ONNParams` -- ``eta``, ``Phi`` and ``Y_sat`` as torch tensors.

The bridge between the calibration chain and the torch layer, and deliberately
the only place the two meet.  It parses no calibration JSON of its own: ``eta``
and ``Phi`` come from
:func:`calibration_module.fit.verify.load_forward_model`, the one reader that
defines this project's phase-sign convention, so there is exactly one answer in
the repo to "what is Phi_k".

That convention: a stored ``dphi_comb_rad`` IS ``Phi_k`` in

    E = sum_k eta_k sqrt(x_k w_k) exp(i [u_k + v_k + Phi_k])

so every loader here sets ``phase_sign = +1`` and that is not a choice.  The
field survives on :class:`ONNParams` only so the flat template can express
``rbonn_3pair_sweep.py``'s ``exp(i[Phi - u - v])`` for a side-by-side
comparison and the layer's selftest can exercise both branches; nothing reading
a real calibration should set it.

There is no default saturation level, on purpose: an explicit ``y_sat=None`` (no
saturation) is better than a hardcoded 5 V that silently disagrees with the run
it is quoted against.
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
    def from_run(run: str | Path, *, pairs: Sequence[int] | None = None,
                 y_sat: float | None = None,
                 method: str | None = None) -> "ONNParams":
        """Load one calib.json -- eta, Phi and Y_sat from the same file.

        In:  ``run``, a ``calib.json`` or the directory holding one.  ``pairs``
             selects channels; ``None`` takes every pair the run calibrates end
             to end (a step-6 eta and, unless it is the reference, a step-7
             phase).  An explicit ``y_sat`` overrides the file's.
        Out: :class:`ONNParams` -- etas from the step-6 section, phases from
             step 7, the saturation level from step 8.  A file with no step 8
             runs linear unless a ``y_sat`` is passed.
        """
        from calibration_module.calib_file import load_calib

        calib = load_calib(run)
        models, phases, ref = load_forward_model(
            calib.path, pairs=pairs, method=method)
        idx = sorted(models)
        if y_sat is None and 8 in calib:
            y_sat = calib.section(8).get("y_sat_v")
        return ONNParams(
            eta=torch.tensor([float(models[k].eta) for k in idx]),
            phi=torch.tensor([float(phases[k]) for k in idx]),
            eta_err=torch.tensor([float(models[k].eta_err) for k in idx]),
            y_sat=_y_sat(y_sat), phase_sign=1.0,
            channels=idx, reference=int(ref), source=str(calib.path),
        )

    @staticmethod
    def from_template(path: str | Path, channels: Sequence[int] | None = None,
                      y_sat: float | None = None) -> "ONNParams":
        """Load the hand-written ``onn_layer_v1`` schema -- see :func:`write_template`.

        For a model somebody typed, not one the bench measured; a measured one
        comes from :meth:`from_run`.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        p = ONNParams._from_flat(d, channels)
        if y_sat is not None:
            p.y_sat = _y_sat(y_sat)
        p.source = str(path)
        return p

    @staticmethod
    def from_json(path: str | Path, channels: Sequence[int] | None = None,
                  y_sat: float | None = None) -> "ONNParams":
        """A ``calib.json``, or a hand-written template.

        Kept as the one entry point that does not make the caller know which it
        has.  The test is the ``schema`` key, not the shape of the payload: a v3
        calib.json says what it is.
        """
        from calibration_module.calib_file import CalibFileError

        try:
            return ONNParams.from_run(path, pairs=channels, y_sat=y_sat)
        except CalibFileError:
            return ONNParams.from_template(path, channels, y_sat)

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
