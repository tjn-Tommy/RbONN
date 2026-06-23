"""Configuration and physical dispersion conversions.

The fs pulse shaper is a Santec SLM-200 in a 2D panel; the column direction is
held phase-uniform so the problem reduces to one dimension (the horizontal /
dispersion axis).
"""
from dataclasses import dataclass

import numpy as np


# =============================================================================
# 0. Parameters
# =============================================================================
@dataclass
class Cfg:
    pitch: float = 8.0e-6          # pixel pitch [m]
    w_px:  float = 1.5*2.5         # single-frequency spot 1/e^2 radius [pixels]
    group: int   = 5               # pixels per spectral channel
    guard: float = 0               # blank guard pixels on each group edge (T=0); may be fractional
    n_levels: int = 1024           # 10-bit phase
    sig_flicker: float = 0.001*np.pi   # phase flicker RMS [rad] (SLM-200 spec)
    sig_fringe_px: float = 1.0     # fringe-field smoothing kernel sigma [pixels] (measure to calibrate)
    # ---- dispersion (physical conversions; does not affect crosstalk geometry) ----
    f: float = 1.000; G: float = 1200e3; theta_i_deg: float = 30.95; lam0: float = 778e-9
    ovs: int = 40                  # grid samples per pixel (grid resolution)

    @property
    def w(self):     return self.w_px*self.pitch
    @property
    def gp(self):    return self.group*self.pitch          # group spacing [m]
    @property
    def dx(self):    return self.pitch/self.ovs            # grid step


# =============================================================================
# 1. Dispersion / spectral resolution / channel count (physical conversions)
# =============================================================================
def dispersion(c: Cfg):
    thi = np.deg2rad(c.theta_i_deg)
    thd = np.arcsin(c.lam0*c.G - np.sin(thi))
    dxdlam = c.f*c.G/np.cos(thd)                # m_x per m_lambda
    out = dict(theta_d_deg=np.rad2deg(thd),
               mm_per_nm=dxdlam*1e-9*1e3,
               nm_per_px=c.pitch/dxdlam*1e9,
               nm_per_group=c.gp/dxdlam*1e9,
               spec_res_nm=c.w/dxdlam*1e9,                 # 1/e^2 spectral resolution (radius)
               n_groups=1920//c.group,
               band_nm=1920*c.pitch/dxdlam*1e9)
    # time domain: frequency/pixel -> replica; spectral resolution -> time window
    cc = 3e8
    dnu_px  = cc*(c.pitch/dxdlam)/c.lam0**2
    dnu_res = cc*(c.w/dxdlam)/c.lam0**2
    out.update(replica_pixel_ps=1/dnu_px*1e12,        # pixel-grid time-domain replica (first)
               Twindow_ps=1/dnu_res*1e12)             # spot-limited time window (resolution limit)
    return out
