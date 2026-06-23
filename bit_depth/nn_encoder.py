"""Torch-free inference for the learned (``nn``) encoding.

``train_nn.py`` (which needs PyTorch) writes ``outputs/nn_encoder.npz`` holding
the universal MLP weights *and* a dense ``target -> 15-px profile`` table. This
module loads that checkpoint with numpy only, so the Monte-Carlo / figure code
can score the learned encoding without importing torch.

Checkpoint keys (see train_nn.save_checkpoint):
    px_per_ch   int      active pixels per window the profile was trained for
    symmetric   bool     MLP emits ceil(px/2) values, mirrored to px_per_ch
    activation  str      hidden activation ('tanh' or 'relu')
    n_layers    int      number of weight matrices L
    W0..W{L-1}, b0..b{L-1}   layer weights / biases
    multiplicative bool  if set, per-pixel weights w(a) are multiplied by the
                         scalar target a, so profile(a) = a * w(a) -- this lets
                         the encoder start at the flat encoding (w == 1)
    table_a       (G,)        input grid in [0, 1]
    table_profile (G, px)     profile sampled on table_a (used for inference)
"""
from pathlib import Path

import numpy as np

from .paths import OUTPUT_DIR

DEFAULT_CHECKPOINT = OUTPUT_DIR / 'nn_encoder.npz'
_CACHE: dict[str, dict] = {}


def _load(path=None) -> dict:
    path = Path(path) if path is not None else DEFAULT_CHECKPOINT
    key = str(path.resolve())
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    if not path.is_file():
        raise FileNotFoundError(
            f"Learned encoder checkpoint not found: {path}. "
            f"Run `python bit_depth/train_nn.py` to train and save it first."
        )
    with np.load(path, allow_pickle=False) as data:
        ckpt = {k: data[k] for k in data.files}
    _CACHE[key] = ckpt
    return ckpt


def clear_cache():
    _CACHE.clear()


def _act(name: str):
    if name == 'tanh':
        return np.tanh
    if name == 'relu':
        return lambda z: np.maximum(z, 0.0)
    raise ValueError(f'unknown activation {name!r}')


def _sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def mlp_forward(amplitudes, path=None) -> np.ndarray:
    """Run the saved MLP on targets in [0, 1]; returns (n, px_per_ch) in [0, 1].

    This mirrors the torch model exactly and is used to verify the numpy and
    torch forwards agree; inference for the simulation uses :func:`nn_profile`.
    """
    ckpt = _load(path)
    a = np.atleast_1d(np.asarray(amplitudes, dtype=float)).reshape(-1, 1)
    hidden = _act(str(ckpt['activation']))
    n_layers = int(ckpt['n_layers'])
    h = a
    for i in range(n_layers):
        z = h @ ckpt[f'W{i}'] + ckpt[f'b{i}']
        h = _sigmoid(z) if i == n_layers - 1 else hidden(z)
    out = h
    if bool(ckpt['symmetric']):
        px = int(ckpt['px_per_ch'])
        half = out.shape[1]
        full = np.empty((out.shape[0], px), dtype=float)
        full[:, :half] = out
        full[:, px - half:] = out[:, ::-1]
        out = full
    if bool(ckpt.get('multiplicative', False)):
        out = out*a
    return np.clip(out, 0.0, 1.0)


def nn_profile(amplitudes, path=None) -> np.ndarray:
    """Per-pixel profile for each target via the saved dense table; (n, px)."""
    ckpt = _load(path)
    table_a = ckpt['table_a']
    table_profile = ckpt['table_profile']
    a = np.atleast_1d(np.asarray(amplitudes, dtype=float))
    out = np.empty((a.size, table_profile.shape[1]), dtype=float)
    for j in range(table_profile.shape[1]):
        out[:, j] = np.interp(a, table_a, table_profile[:, j])
    return np.clip(out, 0.0, 1.0)


def nn_profile_single(amplitude: float, px_per_ch: int, path=None) -> np.ndarray:
    """Profile for one target; validates the trained window size matches."""
    ckpt = _load(path)
    trained_px = int(ckpt['px_per_ch'])
    if int(px_per_ch) != trained_px:
        raise ValueError(
            f"'nn' encoding was trained for px_per_ch={trained_px}, "
            f"but px_per_ch={px_per_ch} was requested"
        )
    return nn_profile(np.array([amplitude]), path)[0]
