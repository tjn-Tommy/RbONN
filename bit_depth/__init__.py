"""fs pulse shaper: amplitude-modulation precision & crosstalk model (Santec SLM-200).

The former single-file ``bit_depth.py`` is now a package. The public API is
re-exported here so existing call sites keep working as ``from bit_depth import X``.
"""
from .config import Cfg, dispersion
from .optics import (
    kernel_grid, k_spot, k_fringe, k_total,
    amp_M, amplitude_of_phi, phi_for_amplitude, T_of_phi, phi_for_T,
    conv_slm, conv_amp, conv_total, propagate_phase,
    amplitude_from_phase, intensity_from_phase,
)
from .geometry import (
    channel_group_px, group_samples, guard_samples, active_samples,
    active_slice, quantize_channel_phases, phase_from_channel_phases,
)
from .encoding import (
    ENCODING_ORDER, ENCODING_LABELS, ENCODING_STYLES, ENCODING_MARKERS,
    ENCODING_TAPER_PX, ordered_encodings, resolve_edge_taper_px,
    edge_taper_weights, taper_active_power_factor,
    phase_from_targets, phase_grid_from_targets, encoding_to_phase,
    amplitude_from_targets, intensity_from_targets, single_simulation,
)
from .readout import (
    CORRECTION_ORDER, CORRECTION_LABELS, CORRECTION_STYLES, CORRECTION_MARKERS,
    ordered_corrections, preferred_correction,
    channel_readout, guard_readout, group_power_readout, guard_power_readout,
    readout_calibration, calibrated_channel_readout,
    build_single_channel_lut, build_background_lut, apply_lut,
    build_lut_for_correction, recovered_for_target,
    single_channel_transfer_curve, lut_corrected_transfer_curve,
)
from .metrics import (
    normalized_crosstalk_matrix, enob_from_rmse, error_metrics,
    crosstalk_summary, validate_active_model,
)
from .montecarlo import (
    monte_carlo_geometry, resolve_worker_count, monte_carlo_sweep,
    add_power_loss_references, csv_ready_rows, save_monte_carlo_rows,
    print_monte_carlo_rows, simulation_grid,
)
from .calibration_data import (
    cfg_from_calibration, measured_nm_per_px, operating_geometry,
    summarize_calibration,
)
from .paths import OUTPUT_DIR, write_csv
from . import nn_encoder

__all__ = [
    'Cfg', 'dispersion',
    'kernel_grid', 'k_spot', 'k_fringe', 'k_total', 'amp_M', 'amplitude_of_phi',
    'phi_for_amplitude', 'T_of_phi', 'phi_for_T', 'conv_slm', 'conv_amp',
    'conv_total', 'propagate_phase', 'amplitude_from_phase', 'intensity_from_phase',
    'channel_group_px', 'group_samples', 'guard_samples', 'active_samples',
    'active_slice', 'quantize_channel_phases', 'phase_from_channel_phases',
    'ENCODING_ORDER', 'ENCODING_LABELS', 'ENCODING_STYLES', 'ENCODING_MARKERS',
    'ENCODING_TAPER_PX', 'ordered_encodings', 'resolve_edge_taper_px',
    'edge_taper_weights', 'taper_active_power_factor', 'phase_from_targets',
    'phase_grid_from_targets', 'encoding_to_phase', 'amplitude_from_targets',
    'intensity_from_targets', 'single_simulation',
    'CORRECTION_ORDER', 'CORRECTION_LABELS', 'CORRECTION_STYLES', 'CORRECTION_MARKERS',
    'ordered_corrections', 'preferred_correction', 'channel_readout', 'guard_readout',
    'group_power_readout', 'guard_power_readout', 'readout_calibration',
    'calibrated_channel_readout', 'build_single_channel_lut', 'build_background_lut',
    'apply_lut', 'build_lut_for_correction', 'recovered_for_target',
    'single_channel_transfer_curve', 'lut_corrected_transfer_curve',
    'normalized_crosstalk_matrix', 'enob_from_rmse', 'error_metrics',
    'crosstalk_summary', 'validate_active_model',
    'monte_carlo_geometry', 'resolve_worker_count', 'monte_carlo_sweep',
    'add_power_loss_references', 'csv_ready_rows', 'save_monte_carlo_rows',
    'print_monte_carlo_rows', 'simulation_grid',
    'cfg_from_calibration', 'measured_nm_per_px', 'operating_geometry',
    'summarize_calibration',
    'OUTPUT_DIR', 'write_csv',
]
