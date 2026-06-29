"""Differentiable digital twin of the rubidium optical neural network.

Hardware physics
----------------
SLM E-field:   E(phi) = sin(phi/2) * exp(j*phi/2)
               Equivalent form (valid for all real phi):
               E(phi) = sin(phi)/2  +  j*(1 - cos(phi))/2

Input encoding (two modes):
  PCA mode (legacy):
    PCA amplitude a in [0,1]  ->  phi_x = 2*arcsin(a)
    ->  E_x = a * exp(j*arcsin(a))   (|E_x| = a, phase tied to amplitude)

  Patch-scan mode (default):
    Raw 28x28 pixels split into N patches of patch_size consecutive pixels.
    Each pixel value p in [0,1] encoded as E = amplitudes_to_efield(p).
    Patches are coherently accumulated:
        x_sum = sum_patches  E(pixels_in_patch)   shape: (batch, n_in) complex
    This uses all 784 pixels without PCA compression.
    The imaginary part of x_sum contains sum(p^2) per channel -- nonlinear
    in the pixel values -- giving extra discriminative power over pure PCA.

Weight parameterization (10 x 20 matrix):
  Learnable phi_w in R^{10 x 20}  ->  W = E(phi_w)
  The SLM cycles through the 10 rows one at a time (time-multiplexed).

Forward model (10 neurons, one per class):
  S_k = sum_i W_ki * x_i    coherent sum for class k
  I_k = |S_k|^2             square-law detection (420 nm intensity, row k)
  y_k = I_k / (I_sat + I_k) saturable absorption, each in [0, 1)

The 10 y_k values are logits for softmax classification.
On hardware: 10 sequential SLM patterns -> 10 scope readings -> argmax.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def phases_to_efield(phi: torch.Tensor) -> torch.Tensor:
    """E(phi) = sin(phi/2)*exp(j*phi/2), computed as sin(phi)/2 + j*(1-cos(phi))/2."""
    return torch.complex(0.5 * torch.sin(phi), 0.5 * (1.0 - torch.cos(phi)))


def amplitudes_to_efield(a: torch.Tensor) -> torch.Tensor:
    """Map pixel/amplitude values a in [0,1] to complex E-fields.

    phi = 2*arcsin(a)  ->  E = a*exp(j*arcsin(a)),  so |E| = a.
    Real part: a (linear in pixel).  Imag part: a^2 (quadratic in pixel).

    The clamp upper bound is 1 - 1e-4 (not 1.0): arcsin'(a) = 1/sqrt(1-a^2)
    diverges at a = 1, so an amplitude pinned exactly at 1.0 injects an
    infinite gradient that turns the loss into NaN. Capping just below 1
    keeps the gradient finite (<= ~70) while leaving the physics unchanged.
    """
    return phases_to_efield(2.0 * torch.arcsin(a.clamp(0.0, 1.0 - 1e-4)))


DETECTION_MODES = ("intensity", "amplitude")


class RbONNTwin(nn.Module):
    """10-class optical classifier with selectable detection mode and input encoding.

    Detection modes
    ---------------
    'intensity' (physical default):
        I_k = |S_k|^2   square-law detection -- matches Rb 420 nm photodetector
    'amplitude':
        I_k = |S_k|      linear amplitude detection

    Input encoding
    --------------
    patch_size > 0  (patch-scan mode, default):
        a is (batch, n_pixels) raw pixel values in [0, 1].
        Padded to a multiple of patch_size, split into non-overlapping patches of
        patch_size pixels each (raster order), E-field encoded, then coherently
        accumulated over patches to give x of shape (batch, n_in).
        patch_size must equal n_in (both default to 20).
        For 28x28 MNIST: 784 pixels -> 40 patches of 20 (last patch zero-padded).

    patch_size == 0  (direct / PCA mode):
        a is (batch, n_in) pre-processed amplitudes (e.g., PCA output) in [0, 1].

    Parameters
    ----------
    n_in           : SLM spatial channels per exposure (default 20)
    n_out          : weight rows = digit classes (default 10)
    detection_mode : 'intensity' or 'amplitude'
    patch_size     : pixels per patch for raw image scanning (0 = direct amplitude input)

    Trainable parameters
    --------------------
    phi_w    : (n_out, n_in) real  -- weight phases, one per (class, channel)
    log_I_sat: scalar              -- log of saturation scale (clamped >= log(0.05))
    """

    def __init__(
        self,
        n_in: int = 20,
        n_out: int = 10,
        detection_mode: str = "intensity",
        patch_size: int = 20,
    ):
        super().__init__()
        if detection_mode not in DETECTION_MODES:
            raise ValueError(f"detection_mode must be one of {DETECTION_MODES}")
        if patch_size > 0 and patch_size != n_in:
            raise ValueError(f"patch_size ({patch_size}) must equal n_in ({n_in})")
        self.n_in = n_in
        self.n_out = n_out
        self.detection_mode = detection_mode
        self.patch_size = patch_size
        self.phi_w = nn.Parameter(torch.rand(n_out, n_in) * 2.0 * math.pi)
        self.log_I_sat = nn.Parameter(torch.tensor(0.0))

    @property
    def W(self) -> torch.Tensor:
        """Complex weight matrix (n_out, n_in) on the E-field manifold."""
        return phases_to_efield(self.phi_w)

    def forward(self, a: torch.Tensor, phi_override: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        a            : (batch, n_pixels) raw pixels in [0,1] when patch_size > 0
                       (batch, n_in) amplitude values in [0,1] when patch_size == 0
        phi_override : (n_out, n_in) optional -- replaces phi_w (used for noise injection)

        Returns
        -------
        (batch, n_out) real in [0, 1) -- per-class detector signals (logits)
        """
        phi = phi_override if phi_override is not None else self.phi_w
        W = phases_to_efield(phi)  # (n_out, n_in) complex

        if self.patch_size > 0:
            batch, n_px = a.shape
            n_pad = (-n_px) % self.patch_size
            if n_pad > 0:
                a = F.pad(a, (0, n_pad))
            a_patches = a.reshape(batch, -1, self.patch_size)      # (batch, n_patches, n_in)
            x_patches = amplitudes_to_efield(a_patches)            # (batch, n_patches, n_in) complex
            S_patches = x_patches @ W.T                            # (batch, n_patches, n_out) complex
            if self.detection_mode == "intensity":
                I_patches = S_patches.abs().pow(2)
            else:
                I_patches = S_patches.abs()
            logits = I_patches.sum(dim=1)                          # (batch, n_out)
        else:
            x = amplitudes_to_efield(a)                            # (batch, n_in) complex
            S = x @ W.T                                            # (batch, n_out) complex
            if self.detection_mode == "intensity":
                logits = S.abs().pow(2)
            else:
                logits = S.abs()

        # I_sat saturation commented out -- training on raw intensities / summed votes
        # I_sat = self.log_I_sat.clamp(min=math.log(0.05)).exp()
        # logits = logits / (I_sat + logits)

        return logits

    def weight_phases(self) -> torch.Tensor:
        """Current weight phases wrapped to [0, 2*pi], detached. (n_out, n_in)"""
        return self.phi_w.detach() % (2.0 * math.pi)

    def I_sat_value(self) -> float:
        if self.patch_size > 0:
            return float("nan")  # not used in patch mode
        return float(self.log_I_sat.clamp(min=math.log(0.05)).exp().detach())


# ---------------------------------------------------------------------------
# 3-layer deep RbONN
# ---------------------------------------------------------------------------

# Measured 420 nm saturable-absorption transfer function (net detector volts vs
# input power in mW), fit to  V(p) = a/(1+exp(-b*(p-c))) + d.  R^2 = 0.9995.
SAT_A, SAT_B, SAT_C, SAT_D = 1.1056, 0.2956, 10.1150, -0.0920


class RbONNDeep(nn.Module):
    """3-layer hierarchical optical neural network on raw 28x28 MNIST pixels.

    Architecture (per class, 10 independent classifiers):

        Layer 1  — image patches
            28x28 pixels -> 7x7 = 49 non-overlapping 4x4 patches (16 px each)
            E-field encode each patch -> coherent sum with W1[k,p] -> |S|^2
            Saturable absorption per-class across 49 values -> 49 latent amps

        Layer 2  — latent patches  (WIDENED bottleneck)
            49 latent amps -> 7 non-overlapping patches of 7  (no values dropped)
            E-field encode each group -> coherent sum with W2[k,p] -> |S|^2
            Saturable absorption per-class across 7 values   -> 7 latent amps

        Layer 3  — final projection
            7 latent amps -> E-field encode -> coherent sum with W3[k] -> |S|^2
            Raw intensity = class score (logit)

        argmax over 10 class scores -> predicted digit

    Inter-layer nonlinearity = measured Rb 420 nm sigmoid
    ----------------------------------------------------
        The latent intensities are first sum-normalized to mean ~1 (keeps the
        re-encoding numerically stable and gradients alive), then scaled by a
        learnable per-layer optical gain g (init so mean lands on the sigmoid's
        steep region, center c=10.1), then passed through the *measured* 420 nm
        transfer function V(p)=a/(1+exp(-b(p-c)))+d and clamped to [0,1] for
        re-encoding as the next layer's E-field amplitudes.  The learnable gain
        physically corresponds to tuning the input laser power / detector gain.

    Trainable parameters
    --------------------
    phi_w1 : (n_out, 49, 16)  -- per-class, per-patch-position L1 weights
    phi_w2 : (n_out,  7,  7)  -- per-class, per-patch-position L2 weights
    phi_w3 : (n_out,  7)      -- per-class L3 weights
    log_g1, log_g2 : scalars  -- learnable optical gain placing the operating point
    Total  : 10*(49*16 + 7*7 + 7) + 2 = 8,402 params
    """

    N_PATCHES_1 = 49   # 7x7 grid of 4x4 patches on 28x28
    PATCH_PX_1  = 16   # 4x4 = 16 pixels per L1 patch

    def __init__(self, n_out: int = 10, detection_mode: str = "intensity",
                 l2_patch: int = 7):
        super().__init__()
        if detection_mode not in DETECTION_MODES:
            raise ValueError(f"detection_mode must be one of {DETECTION_MODES}")
        if self.N_PATCHES_1 % l2_patch != 0:
            raise ValueError(f"l2_patch ({l2_patch}) must divide {self.N_PATCHES_1}")
        self.n_out = n_out
        self.detection_mode = detection_mode
        self.l2_patch = l2_patch
        self.n_patches_2 = self.N_PATCHES_1 // l2_patch    # 49 // 7 = 7

        self.phi_w1 = nn.Parameter(
            torch.rand(n_out, self.N_PATCHES_1, self.PATCH_PX_1) * 2.0 * math.pi
        )
        self.phi_w2 = nn.Parameter(
            torch.rand(n_out, self.n_patches_2, self.l2_patch) * 2.0 * math.pi
        )
        self.phi_w3 = nn.Parameter(
            torch.rand(n_out, self.n_patches_2) * 2.0 * math.pi
        )
        # Optical gain: init so a unit-mean latent maps onto the sigmoid centre.
        self.log_g1 = nn.Parameter(torch.tensor(math.log(SAT_C)))
        self.log_g2 = nn.Parameter(torch.tensor(math.log(SAT_C)))

    def _detect(self, S: torch.Tensor) -> torch.Tensor:
        return S.abs().pow(2) if self.detection_mode == "intensity" else S.abs()

    @staticmethod
    def _norm(I: torch.Tensor) -> torch.Tensor:
        """Sum-normalize last dim to mean ~1 (every element gets gradient)."""
        n = I.shape[-1]
        return I / (I.sum(dim=-1, keepdim=True) + 1e-8) * n

    def _saturate(self, I: torch.Tensor, log_g: torch.Tensor) -> torch.Tensor:
        """Measured 420 nm saturable absorption applied as the inter-layer activation.

        I -> normalize (mean~1) -> optical power p = g*I -> V(p) sigmoid -> [0,1].
        """
        p = self._norm(I) * log_g.exp()
        v = SAT_A / (1.0 + torch.exp(-SAT_B * (p - SAT_C))) + SAT_D
        return v.clamp(0.0, 1.0)

    @staticmethod
    def _image_to_patches(a: torch.Tensor) -> torch.Tensor:
        """Extract 49 non-overlapping 4x4 spatial patches from flat 784-px images.

        Returns (batch, 49, 16) — true 4x4 blocks, not raster slices.
        """
        img = a.reshape(-1, 28, 28)                         # (batch, 28, 28)
        p = img.unfold(1, 4, 4).unfold(2, 4, 4)            # (batch, 7, 7, 4, 4)
        return p.contiguous().reshape(-1, 49, 16)           # (batch, 49, 16)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        a : (batch, 784) raw pixel amplitudes in [0, 1]

        Returns
        -------
        (batch, n_out) class logits (raw summed intensities)
        """
        b2, p2 = self.n_patches_2, self.l2_patch

        # ── Layer 1: 4x4 image patches -> 49 latent values ──────────────
        patches1 = self._image_to_patches(a)                # (batch, 49, 16)
        x1 = amplitudes_to_efield(patches1)                 # (batch, 49, 16) complex
        W1 = phases_to_efield(self.phi_w1)                  # (n_out, 49, 16) complex

        # S1[b,k,p] = sum_i W1[k,p,i] * x1[b,p,i]
        S1 = (x1.unsqueeze(1) * W1.unsqueeze(0)).sum(-1)   # (batch, n_out, 49) complex
        I1 = self._detect(S1)                               # (batch, n_out, 49)
        y1 = self._saturate(I1, self.log_g1)               # (batch, n_out, 49) in [0,1]

        # ── Layer 2: 49 latent -> 7 patches of 7 (widened) ──────────────
        y1_t = y1.reshape(-1, self.n_out, b2, p2)          # (batch, n_out, 7, 7)
        x2 = amplitudes_to_efield(y1_t)                    # (batch, n_out, 7, 7) complex
        W2 = phases_to_efield(self.phi_w2)                 # (n_out, 7, 7) complex

        # S2[b,k,p] = sum_i W2[k,p,i] * x2[b,k,p,i]
        S2 = (x2 * W2.unsqueeze(0)).sum(-1)                # (batch, n_out, 7) complex
        I2 = self._detect(S2)                              # (batch, n_out, 7)
        y2 = self._saturate(I2, self.log_g2)               # (batch, n_out, 7) in [0,1]

        # ── Layer 3: 7 latent -> 1 scalar per class ─────────────────────
        x3 = amplitudes_to_efield(y2)                      # (batch, n_out, 7) complex
        W3 = phases_to_efield(self.phi_w3)                 # (n_out, 7) complex

        # S3[b,k] = sum_i W3[k,i] * x3[b,k,i]
        S3 = (x3 * W3.unsqueeze(0)).sum(-1)               # (batch, n_out) complex
        return self._detect(S3)                            # (batch, n_out) logits

    def I_sat_value(self) -> float:
        return float("nan")  # no saturation in deep mode
