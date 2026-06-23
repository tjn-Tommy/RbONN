"""Optical forward model: kernels, the parallel-analyzer amplitude mask, and the
nonlinear propagation chain.

Unified 1D propagation:
    commanded phase phi_cmd(x)
      --(LCOS fringe field k_fringe)--> phi_eff = phi_cmd (*) k_fringe
      --(parallel analyzer)--> complex amplitude mask M_eff(x) = 1/2 (exp(i phi_eff) + 1)
      --(continuous spectrum + finite spot k_spot)--> shaper transfer H = M_eff (*) k_spot
Because k_fringe acts in the phase domain and the analyzer is a nonlinear
amplitude operation, the channel response must be computed by the full chain
(it cannot be reduced to one linear kernel).
"""
import numpy as np
from scipy.ndimage import convolve

from .config import Cfg


# =============================================================================
# 2. Kernels
# =============================================================================
def _norm(k: np.ndarray):
    s = k.sum(); return k/s if s != 0 else k


def kernel_grid(c: Cfg, half_px=6):
    n = int(half_px*c.ovs)
    x = (np.arange(-n, n+1))*c.dx
    return x


def k_spot(c: Cfg, x: np.ndarray):
    return _norm(np.exp(-2*x**2/c.w**2))


def k_fringe(c: Cfg, x: np.ndarray):
    s = c.sig_fringe_px*c.pitch
    if s <= 0:
        k = np.zeros_like(x)
        k[len(k)//2] = 1.0
        return k
    return _norm(np.exp(-0.5*(x/s)**2))


def k_total(c: Cfg):
    """Linear reference kernel only; the true channel response uses the full chain."""
    x = kernel_grid(c)
    kt = np.convolve(k_fringe(c, x), k_spot(c, x), mode='same')
    return x, _norm(kt)


# =============================================================================
# 3. Amplitude response
# =============================================================================
def amp_M(phi: np.ndarray):
    """Complex amplitude mask after the parallel analyzer, |M|=cos(phi/2)."""
    return 0.5*(np.exp(1j*phi) + 1.0)


def amplitude_of_phi(phi: np.ndarray):
    return np.abs(amp_M(phi))


def phi_for_amplitude(amplitude: np.ndarray):
    amplitude = np.clip(amplitude, 0, 1)
    return 2*np.arccos(amplitude)


# Backward-compatible aliases for old commented/reference code.
T_of_phi = amplitude_of_phi
phi_for_T = phi_for_amplitude


# =============================================================================
# 4. Convolution / propagation
# =============================================================================
def conv_slm(c: Cfg, phase_grid: np.ndarray):
    xk = kernel_grid(c)
    kf = k_fringe(c, xk)
    return convolve(phase_grid, kf, mode='constant', cval=np.pi)


def conv_amp(c: Cfg, phase_grid: np.ndarray):
    xk = kernel_grid(c)
    kf = k_spot(c, xk)
    amp_grid = amp_M(phase_grid)
    return convolve(amp_grid, kf, mode='constant', cval=0.0)


def conv_total(c: Cfg, phase: np.ndarray):
    phase_grid = np.repeat(phase, c.ovs)
    return propagate_phase(c, phase_grid)


def propagate_phase(c: Cfg, phase_grid: np.ndarray):
    phase_eff = conv_slm(c, phase_grid)
    return conv_amp(c, phase_eff)


def amplitude_from_phase(c: Cfg, phase_grid: np.ndarray):
    return np.abs(propagate_phase(c, phase_grid))


intensity_from_phase = amplitude_from_phase
