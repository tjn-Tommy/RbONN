"""Train the universal ``nn`` encoding (requires PyTorch).

A single MLP shared by every channel produces per-pixel weights ``w(a)`` and the
encoded profile is ``p(a) = a * w(a)``. Starting from ``w == 1`` (the flat
encoding) the net learns beneficial edge tapering. Each step sends the target
through the encoder's own LUT calibration first, then computes the post-LUT
readout error; an intensity penalty measured at maximum drive (``a = 1``) stops
the net from tapering optical power away.

Forward chain (torch port of ``optics.propagate_phase``):
    p -> phi = 2*arccos(p) -> (*)k_fringe -> 0.5(e^{i phi}+1) -> (*)k_spot -> |.|
    -> group-average readout.

Run:    python bit_depth/train_nn.py
Saves:  bit_depth/outputs/nn_encoder.npz  (loaded torch-free by nn_encoder.py)
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# torch and matplotlib each bundle their own OpenMP runtime; allow both to load
# in one process (we only do small CPU convs, so the duplicate-runtime risk is moot).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import bit_depth as bd
from bit_depth import nn_encoder
from bit_depth.optics import kernel_grid, k_spot, k_fringe

# float32 is ~2x faster than float64 on CPU and easily within the model's
# accuracy needs (the torch/numpy sanity check stays < 1e-4).
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
PX_PER_CH = bd.operating_geometry()['px_per_ch']     # 15
GUARD = bd.operating_geometry()['guard']             # 2.5
N_CH = 3                                              # center + 2 nearest neighbors
CENTER = N_CH // 2
HIDDEN = (32, 32)
SYMMETRIC = True
ACTIVATION = 'tanh'

EPOCHS = 200
BATCH = 512
LR = 5e-3
W_FID = 1.0
W_XTALK = 10.0         # neighbor-leakage (crosstalk) penalty weight
# intensity-loss penalty weight -- the accuracy<->intensity trade-off knob.
# Frontier at 15px/2.5px (ENOB @ max-intensity efficiency): 3e-4: 6.30@0.72,
# 5e-4: 6.27@0.77, 1e-3: 6.00@0.87, 2e-3: 5.69@0.97. Default 1e-3 keeps ~87%
# peak intensity while beating flat (5.02) and taper-1px (5.55) and matching taper-2px.
W_INT = 1e-3
XT_POINTS = 33         # isolated-drive command grid used for the crosstalk term
USE_LUT = True
LUT_POINTS = 61
LUT_REFRESH = 5        # rebuild the (slowly drifting) LUT every N epochs, not every step
SEED = 0


def _config():
    return bd.cfg_from_calibration()


class Encoder(nn.Module):
    """Universal MLP a -> per-pixel weights w(a) in [0,1]^PX_PER_CH.

    The encoded profile is a * w(a); the last bias is initialized high so w ~ 1
    at the start (i.e. the flat encoding), giving a well-conditioned transfer
    curve for the LUT from epoch 0.
    """

    def __init__(self):
        super().__init__()
        self.symmetric = SYMMETRIC
        out_dim = (PX_PER_CH + 1) // 2 if SYMMETRIC else PX_PER_CH
        dims = (1,) + tuple(HIDDEN) + (out_dim,)
        self.linears = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.act = torch.tanh if ACTIVATION == 'tanh' else F.relu
        with torch.no_grad():
            self.linears[-1].weight.mul_(0.01)
            self.linears[-1].bias.fill_(2.5)        # sigmoid(2.5) ~ 0.92 -> near flat

    def weights(self, a):              # a: (N, 1) in [0,1] -> (N, PX_PER_CH) in [0,1]
        h = a
        for i, lin in enumerate(self.linears):
            z = lin(h)
            h = torch.sigmoid(z) if i == len(self.linears) - 1 else self.act(z)
        if self.symmetric:
            half = h.shape[1]
            full = torch.empty((h.shape[0], PX_PER_CH), dtype=h.dtype)
            full[:, :half] = h
            full[:, PX_PER_CH - half:] = torch.flip(h, dims=[1])
            h = full
        return h

    def profile(self, a):              # a: (N, 1) -> (N, PX_PER_CH); p(a) = a * w(a)
        return a * self.weights(a)


def _phi(p):
    return 2.0 * torch.arccos(torch.clamp(p, 0.0, 1.0))


class Propagator:
    """Torch port of the numpy optical forward model, evaluated only over the
    center channel.

    The center readout depends on the phase grid only within +/- two kernel
    half-widths of the center group (two sequential convolutions). Cropping to
    that window instead of convolving all N_CH groups, plus running the two
    valid convolutions with no padding, makes a step ~2x cheaper with no change
    to the center result (neighbors beyond the halo cannot reach it).
    """

    def __init__(self, c):
        x = kernel_grid(c)
        self.H = (len(x) - 1) // 2
        self.k_fringe = torch.tensor(k_fringe(c, x), dtype=DTYPE).reshape(1, 1, -1)
        self.k_spot = torch.tensor(k_spot(c, x), dtype=DTYPE).reshape(1, 1, -1)
        self.ovs = c.ovs
        self.guard_samples = int(round(GUARD * c.ovs))
        self.active = c.ovs * PX_PER_CH
        self.group = int(round((PX_PER_CH + 2 * GUARD) * c.ovs))
        center_lo = CENTER * self.group
        self.crop_lo = center_lo - 2 * self.H
        self.crop_hi = center_lo + self.group + 2 * self.H
        if self.crop_lo < 0 or self.crop_hi > N_CH * self.group:
            raise ValueError('center halo exceeds the N_CH window; increase N_CH')

    def raw_center(self, profiles):    # profiles: (B, N_CH, PX_PER_CH) -> (B,)
        B = profiles.shape[0]
        phi = _phi(profiles).repeat_interleave(self.ovs, dim=2)        # (B,N_CH,active)
        grid = torch.full((B, N_CH, self.group), math.pi)
        grid[:, :, self.guard_samples:self.guard_samples + self.active] = phi
        grid = grid.reshape(B, N_CH * self.group)[:, self.crop_lo:self.crop_hi]
        grid = grid.unsqueeze(1)                                       # (B,1,crop)

        phase_eff = F.conv1d(grid, self.k_fringe)                      # valid
        re = 0.5 * (torch.cos(phase_eff) + 1.0)
        im = 0.5 * torch.sin(phase_eff)
        re_c = F.conv1d(re, self.k_spot)                              # valid -> center group
        im_c = F.conv1d(im, self.k_spot)
        amp = torch.sqrt(re_c ** 2 + im_c ** 2 + 1e-15)               # (B,1,group)
        return amp.mean(dim=2).squeeze(1)

    def raw_groups(self, profiles):    # profiles: (B, N_CH, PX_PER_CH) -> (B, N_CH)
        """Per-group readout over the full window (needed for neighbor leakage)."""
        B = profiles.shape[0]
        phi = _phi(profiles).repeat_interleave(self.ovs, dim=2)
        grid = torch.full((B, N_CH, self.group), math.pi)
        grid[:, :, self.guard_samples:self.guard_samples + self.active] = phi
        grid = grid.reshape(B, 1, N_CH * self.group)
        phase_eff = F.conv1d(F.pad(grid, (self.H, self.H), value=math.pi), self.k_fringe)
        re = 0.5 * (torch.cos(phase_eff) + 1.0)
        im = 0.5 * torch.sin(phase_eff)
        re_c = F.conv1d(F.pad(re, (self.H, self.H), value=0.0), self.k_spot)
        im_c = F.conv1d(F.pad(im, (self.H, self.H), value=0.0), self.k_spot)
        amp = torch.sqrt(re_c ** 2 + im_c ** 2 + 1e-15).reshape(B, N_CH, self.group)
        return amp.mean(dim=2)


def _profiles(encoder, center_cmd, neigh_cmd):
    """Per-channel profiles p = cmd * w(cmd) for center cmd among neighbor cmds."""
    cmds = neigh_cmd.clone()
    cmds[:, CENTER] = center_cmd
    w = encoder.weights(cmds.reshape(-1, 1)).reshape(cmds.shape[0], N_CH, PX_PER_CH)
    return cmds.unsqueeze(-1) * w


def _isolated_raw(encoder, prop, center_cmd):
    neigh = torch.zeros(center_cmd.shape[0], N_CH)
    return prop.raw_center(_profiles(encoder, center_cmd, neigh))


def _sanity_check(prop, c):
    """The torch forward must match the numpy model on an isolated flat channel."""
    a = 0.6
    prof = torch.zeros(1, N_CH, PX_PER_CH)
    prof[:, CENTER, :] = a
    raw_torch = float(prop.raw_center(prof)[0])
    amp_np = bd.amplitude_from_targets(c, np.array([0.0, a, 0.0]), px_per_ch=PX_PER_CH, guard=GUARD)
    raw_np = float(bd.channel_readout(c, amp_np, 3, PX_PER_CH, GUARD)[CENTER])
    diff = abs(raw_torch - raw_np)
    print(f'sanity torch-vs-numpy raw center: torch={raw_torch:.6f} numpy={raw_np:.6f} diff={diff:.2e}')
    if diff > 1e-4:
        raise RuntimeError('torch forward model diverges from numpy model')


def _build_lut(encoder, prop):
    """Epoch-fixed (blank, diag, target->command LUT) from the encoder's transfer."""
    with torch.no_grad():
        blank = float(_isolated_raw(encoder, prop, torch.zeros(1)))
        diag = float(_isolated_raw(encoder, prop, torch.ones(1))) - blank
        cmd = torch.linspace(0.0, 1.0, LUT_POINTS)
        actual = ((_isolated_raw(encoder, prop, cmd) - blank) / diag).numpy()
    cmd = cmd.numpy()
    actual = np.maximum.accumulate(actual)
    y, idx = np.unique(actual, return_index=True)
    x = cmd[idx]
    if y[0] > 0:
        y = np.r_[0.0, y]; x = np.r_[0.0, x]
    if y[-1] < 1:
        y = np.r_[y, 1.0]; x = np.r_[x, 1.0]
    return blank, diag, y, x


def _apply_lut(values, lut_y, lut_x):
    if not USE_LUT:
        return values
    arr = np.atleast_1d(values.detach().numpy())
    return torch.tensor(np.interp(arr, lut_y, lut_x).reshape(values.shape), dtype=DTYPE)


def run_training(w_int=W_INT, w_xtalk=W_XTALK, epochs=EPOCHS, lr=LR,
                 verbose=True, sanity=True):
    """Train one encoder; returns (encoder, final_fid_rmse, max_intensity_eff)."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    c = _config()
    encoder = Encoder()
    prop = Propagator(c)
    if sanity:
        _sanity_check(prop, c)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)

    xt_cmd = torch.linspace(0.0, 1.0, XT_POINTS).reshape(-1, 1)
    xt_neigh = torch.zeros(XT_POINTS, N_CH)

    fid_rmse = eff = float('nan')
    blank, diag, lut_y, lut_x = 0.0, 1.0, None, None
    start = time.perf_counter()
    for epoch in range(epochs):
        if USE_LUT and epoch % LUT_REFRESH == 0:
            blank, diag, lut_y, lut_x = _build_lut(encoder, prop)
        elif not USE_LUT and epoch == 0:
            with torch.no_grad():
                blank = float(_isolated_raw(encoder, prop, torch.zeros(1)))
                diag = float(_isolated_raw(encoder, prop, torch.ones(1))) - blank

        a = torch.tensor(rng.uniform(0, 1, BATCH), dtype=DTYPE)
        neigh = torch.tensor(rng.uniform(0, 1, (BATCH, N_CH)), dtype=DTYPE)
        cmd_center = _apply_lut(a, lut_y, lut_x)
        cmd_neigh = _apply_lut(neigh.reshape(-1), lut_y, lut_x).reshape(BATCH, N_CH)

        raw = prop.raw_center(_profiles(encoder, cmd_center, cmd_neigh))
        cal = (raw - blank) / diag
        loss_fid = torch.mean((cal - a) ** 2)

        # crosstalk: drive only the center channel and penalize the power that
        # leaks into the two neighbor readout windows (drives edge tapering).
        leak = prop.raw_groups(_profiles(encoder, xt_cmd.squeeze(1), xt_neigh))
        loss_xtalk = torch.mean(leak[:, CENTER - 1] ** 2 + leak[:, CENTER + 1] ** 2)

        cmd_max = _apply_lut(torch.ones(1), lut_y, lut_x)
        p_max = encoder.profile(cmd_max.reshape(1, 1))[0]
        loss_int = 1.0 - torch.mean(p_max ** 2)

        loss = W_FID * loss_fid + w_xtalk * loss_xtalk + w_int * loss_int
        opt.zero_grad(); loss.backward(); opt.step()

        fid_rmse = loss_fid.item() ** 0.5
        eff = 1 - loss_int.item()
        if verbose and (epoch % 25 == 0 or epoch == epochs - 1):
            elapsed = time.perf_counter() - start
            eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
            print(f'epoch {epoch:4d}/{epochs} | {elapsed:5.1f}s elapsed | ETA {eta:5.1f}s | '
                  f'loss={loss.item():.5f} fid_rmse={fid_rmse:.4f} eff={eff:.4f}')
    return encoder, fid_rmse, eff


def train():
    c = _config()
    encoder, _, _ = run_training()
    save_checkpoint(encoder)
    evaluate(c)


def save_checkpoint(encoder, out=None, verbose=True):
    payload = dict(px_per_ch=PX_PER_CH, symmetric=bool(SYMMETRIC),
                   activation=ACTIVATION, n_layers=len(encoder.linears),
                   multiplicative=True)
    for i, lin in enumerate(encoder.linears):
        payload[f'W{i}'] = lin.weight.detach().numpy().T.astype(np.float64)
        payload[f'b{i}'] = lin.bias.detach().numpy().astype(np.float64)
    table_a = np.linspace(0.0, 1.0, 256)
    with torch.no_grad():
        table_profile = encoder.profile(
            torch.tensor(table_a, dtype=DTYPE).reshape(-1, 1)).numpy().astype(np.float64)
    payload['table_a'] = table_a
    payload['table_profile'] = table_profile

    out = bd.OUTPUT_DIR / 'nn_encoder.npz' if out is None else Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    nn_encoder.clear_cache()
    if verbose:
        print(f'\nsaved {out}')
        print(f'profile(a=1) max-intensity efficiency mean(p^2) = {np.mean(table_profile[-1] ** 2):.4f}')
        print(f'profile(a=1) = {np.round(table_profile[-1], 3)}')
    return out


def evaluate(c):
    """Compare nn vs flat/taper at 15px/2.5px guard, all under LUT correction."""
    print('\n=== 15px window / 2.5px guard, correction=lut (n_ch=21, n_trials=200) ===')
    print('  encoding         cal_ENOB cal_RMSE nearest_xtalk mean_group_power max_intensity_eff')
    for enc in ('flat', 'edge_taper_1px', 'edge_taper_2px', 'nn'):
        row = bd.monte_carlo_geometry(
            c, px_per_ch=PX_PER_CH, guard=GUARD, n_ch=21, n_trials=200, seed=1,
            correction='lut', encoding=enc,
        )
        print(f'  {enc:<15s} {row["cal_enob"]:8.2f} {row["cal_rmse"]:8.4f} '
              f'{row["nearest_xtalk"]:13.4f} {row["mean_group_power"]:16.4f} '
              f'{_max_intensity_eff(enc):17.4f}')


def _max_intensity_eff(encoding):
    if encoding == 'nn':
        return float(np.mean(nn_encoder.nn_profile_single(1.0, PX_PER_CH) ** 2))
    return bd.taper_active_power_factor(PX_PER_CH, encoding)


if __name__ == '__main__':
    train()
