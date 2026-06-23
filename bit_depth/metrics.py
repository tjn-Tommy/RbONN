"""Error metrics, crosstalk summaries, and the model self-check."""
import numpy as np

from .config import Cfg
from .optics import kernel_grid, k_spot, k_fringe, amplitude_of_phi, phi_for_amplitude
from .geometry import channel_group_px, phase_from_channel_phases
from .encoding import phase_from_targets, edge_taper_weights


def normalized_crosstalk_matrix(response: np.ndarray):
    diag = np.diag(response)
    if np.any(diag == 0):
        raise ValueError('zero diagonal response in crosstalk matrix')
    return response/diag[:, None]


def enob_from_rmse(rmse: float):
    return np.log2(1/(rmse*np.sqrt(12))) if rmse > 0 else np.inf


def error_metrics(readout: np.ndarray, target: np.ndarray):
    abs_error = readout - target
    rel_mask = target > 0.05
    rel_error = np.abs(abs_error[rel_mask])/(target[rel_mask] + 1e-6)
    rmse_abs = np.sqrt(np.mean(abs_error**2))
    return dict(
        enob=enob_from_rmse(rmse_abs),
        rmse=rmse_abs,
        bias=np.mean(abs_error),
        mae=np.mean(np.abs(abs_error)),
        p95_abs=np.percentile(np.abs(abs_error), 95),
        max_abs=np.max(np.abs(abs_error)),
        mean_rel_signal=np.mean(rel_error) if rel_error.size else np.nan,
    )


def crosstalk_summary(response: np.ndarray, eval_idx: np.ndarray):
    C = normalized_crosstalk_matrix(response)
    sub = C[np.ix_(eval_idx, np.arange(C.shape[1]))].copy()
    for row, ch in enumerate(eval_idx):
        sub[row, ch] = 0.0
    mid = C.shape[0]//2
    nearest = np.nan
    if 0 < mid < C.shape[0] - 1:
        nearest = 0.5*(C[mid, mid - 1] + C[mid, mid + 1])
    return dict(
        nearest_xtalk=nearest,
        mean_row_xtalk=np.mean(np.sum(np.abs(sub), axis=1)),
        max_offdiag=np.max(np.abs(sub)),
    )


def validate_active_model(c: Cfg):
    checks = []
    amplitude = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    checks.append(('A(phi_for_amplitude(A))',
                   np.allclose(amplitude_of_phi(phi_for_amplitude(amplitude)),
                               amplitude, atol=1e-12)))
    xk = kernel_grid(c)
    checks.append(('k_spot normalization', np.isclose(k_spot(c, xk).sum(), 1.0)))
    checks.append(('k_fringe normalization', np.isclose(k_fringe(c, xk).sum(), 1.0)))
    test_phase = phase_from_targets(c, np.array([0.0, 1.0]), px_per_ch=3, guard=1)
    checks.append(('phase length', len(test_phase) == 2*channel_group_px(3, 1)))
    checks.append(('blank phase is pi', np.all(test_phase[:1] == np.pi)))
    flat_phase = phase_from_targets(c, amplitude, px_per_ch=5, guard=2, encoding='flat')
    old_phase = phase_from_channel_phases(phi_for_amplitude(amplitude), px_per_ch=5, guard=2)
    checks.append(('flat encoding preserves phase grid', np.array_equal(flat_phase, old_phase)))
    for encoding in ('edge_taper_1px', 'edge_taper_2px'):
        weights = edge_taper_weights(5, encoding)
        checks.append((f'{encoding} weights symmetric', np.allclose(weights, weights[::-1])))
        checks.append((f'{encoding} weights bounded', np.all((weights >= 0) & (weights <= 1))))
    taper_phase = phase_from_targets(c, np.array([1.0]), px_per_ch=5, guard=2,
                                     encoding='edge_taper_2px')
    checks.append(('taper guard pixels remain blank',
                   np.all(taper_phase[:2] == np.pi) and np.all(taper_phase[-2:] == np.pi)))
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise AssertionError('Active simulation checks failed: ' + ', '.join(failed))
    return checks
