from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_module.encoding import (
    ChannelLayout,
    EncodingChannel,
    build_channel_layout,
    compute_channel_geometry,
    encode_to_pattern,
    optimize_from_osa,
)
from slm_module.calibration.calibration_new import CalibrationResult
import slm_module.analysis as analysis
from slm_module.analysis import (
    ChannelSpectrum,
    ModulationErrorResult,
    encoding_gain,
    measure_channel_spectra,
    measure_one_channel,
    write_gain_csv,
)
from slm_module.optimization import (
    FLAT_ENCODING_SHAPE,
    OPTIMIZED_ENCODING_SHAPE,
    mirror_intensity_profile,
    round_encoding_profile,
)
from osa_module.controller import MeasurementSettings, TraceData


def _make_layout(width: int = 10) -> ChannelLayout:
    """Two-channel layout (one x, one w) with a simple linear transfer curve.

    levels 0..1023 map linearly to normalised power 0..1, so level_for(v) picks
    the swept level nearest ``v`` * 1023: level_for(0) -> 0 (measured off),
    level_for(0.5) -> 512, level_for(1.0) -> 1023.
    """
    levels = np.array([0, 256, 512, 768, 1023], dtype=int)
    curve = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

    def _ch(index: int, x_start: int) -> EncodingChannel:
        return EncodingChannel(
            index=index, side="x" if index == 0 else "w",
            x_center=x_start + width // 2, x_start=x_start, x_end=x_start + width,
            wavelength_nm=780.0, levels=levels.copy(), intensity_curve=curve.copy(),
        )

    return ChannelLayout(
        x_channels=[_ch(0, 45)],
        w_channels=[_ch(1, 145)],
        center_wl=778.0, center_x=100.0,
        channel_width_px=width, pitch_px=width + 5, nm_per_px=0.05,
        calib_coords=np.array([0, 50, 150, 200], dtype=float),
        calib_off_levels=np.zeros(4, dtype=int),
    )


class EncodeToPatternTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = _make_layout(width=10)
        self.slm_w, self.slm_h = 200, 4

    def test_col_ratio_none_matches_ones(self) -> None:
        flat = encode_to_pattern([0.5], [0.5], self.layout, self.slm_w, self.slm_h)
        ones = encode_to_pattern([0.5], [0.5], self.layout, self.slm_w, self.slm_h,
                                 col_ratio=np.ones(10))
        self.assertTrue(np.array_equal(flat, ones))
        # flat band: whole x-channel width sits at level_for(0.5) == 512
        self.assertTrue(np.all(flat[:, 45:55] == 512))

    def test_taper_lowers_edges_and_zero_hits_off_level(self) -> None:
        ratios = np.ones(10)
        ratios[0] = 0.0     # edge column -> channel's measured off level
        ratios[1] = 0.5
        ratios[-1] = 0.0
        pat = encode_to_pattern([1.0], [1.0], self.layout, self.slm_w, self.slm_h,
                                col_ratio=ratios)
        off_level = self.layout.x_channels[0].off_level
        self.assertEqual(pat[0, 45], off_level)          # ratio 0.0 -> off (0)
        self.assertEqual(pat[0, 54], off_level)
        self.assertEqual(pat[0, 46], 512)                # ratio 0.5 -> 512
        self.assertTrue(np.all(pat[:, 47:54] == 1023))   # interior -> full
        self.assertLess(int(pat[0, 45]), int(pat[0, 50]))

    def test_level_trim_applied_and_clipped(self) -> None:
        trim = lambda lv: lv.astype(np.int32) + 100
        # value 0.5 -> every column at 512, trimmed to 612
        pat = encode_to_pattern([0.5], [0.5], self.layout, self.slm_w, self.slm_h,
                                level_trim=trim)
        self.assertTrue(np.all(pat[:, 45:55] == 612))
        # value 1.0 -> 1023, trimmed to 1123, clipped back to 1023
        pat_hi = encode_to_pattern([1.0], [1.0], self.layout, self.slm_w, self.slm_h,
                                   level_trim=trim)
        self.assertEqual(int(pat_hi[:, 45:55].max()), 1023)

    def test_wrong_ratio_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            encode_to_pattern([0.5], [0.5], self.layout, self.slm_w, self.slm_h,
                              col_ratio=np.ones(9))


class OptimizeFromOsaTests(unittest.TestCase):
    def test_live_interface_requires_controllers(self) -> None:
        layout = _make_layout()
        with self.assertRaisesRegex(ValueError, "controllers"):
            optimize_from_osa(layout, None, col_ratio=np.ones(10))


def _spectrum(side: str, index: int, leak: float, in_band: float,
              window: float = 0.0, channel: float = 0.0) -> ChannelSpectrum:
    return ChannelSpectrum(
        index=index, side=side, x_center=0, nominal_wl_nm=780.0, nominal_bw_nm=0.5,
        wavelengths_nm=np.array([]), signal_w=np.array([]),
        window_power_w=window, channel_power_w=channel,
        neighbor_leakage=leak, in_band_fraction=in_band,
    )


def _result(specs: list[ChannelSpectrum]) -> ModulationErrorResult:
    return ModulationErrorResult(
        channels=specs, center_wl=778.0, channel_width_px=10, pitch_px=15, nm_per_px=0.05,
    )


class EncodingGainTests(unittest.TestCase):
    def test_per_channel_and_mean_deltas(self) -> None:
        baseline = _result([_spectrum("x", 0, 0.10, 0.80), _spectrum("w", 0, 0.06, 0.90)])
        tuned = _result([_spectrum("x", 0, 0.04, 0.88), _spectrum("w", 0, 0.03, 0.93)])
        gain = encoding_gain(baseline, tuned)
        self.assertEqual(gain.n, 2)
        self.assertAlmostEqual(gain.channels[0].d_leak, -0.06)
        self.assertAlmostEqual(gain.channels[0].d_in_band, 0.08)
        self.assertAlmostEqual(gain.mean_d_leak, -0.045)
        self.assertAlmostEqual(gain.mean_d_in_band, 0.055)

    def test_intensity_loss_vs_flat_baseline(self) -> None:
        # before = flat/rectangular baseline, after = taper (lower throughput)
        baseline = _result([_spectrum("x", 0, 0.10, 0.80, window=1.0, channel=1.2)])
        tuned = _result([_spectrum("x", 0, 0.04, 0.85, window=0.7, channel=1.0)])
        gain = encoding_gain(baseline, tuned)
        c = gain.channels[0]
        self.assertAlmostEqual(c.loss_window, 0.30)              # (1.0-0.7)/1.0
        self.assertAlmostEqual(c.loss_total, (1.2 - 1.0) / 1.2)
        self.assertAlmostEqual(gain.mean_loss_window, 0.30)

    def test_only_channels_present_in_both_count(self) -> None:
        baseline = _result([_spectrum("x", 0, 0.10, 0.80), _spectrum("w", 0, 0.06, 0.90)])
        tuned = _result([_spectrum("x", 0, 0.04, 0.88)])  # w0 missing
        gain = encoding_gain(baseline, tuned)
        self.assertEqual(gain.n, 1)
        self.assertEqual(gain.channels[0].side, "x")

    def test_write_gain_csv_has_mean_row(self) -> None:
        baseline = _result([_spectrum("x", 0, 0.10, 0.80)])
        tuned = _result([_spectrum("x", 0, 0.04, 0.88)])
        gain = encoding_gain(baseline, tuned)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "gain.csv")
            write_gain_csv(gain, path)
            lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
        self.assertTrue(lines[0].startswith("side,index,nominal_wl_nm"))
        self.assertTrue(lines[-1].startswith("mean,"))


class _FakeSLM:
    def get_slm_info(self):
        return (200, 4)

    def display_array(self, pattern):
        self.last = np.asarray(pattern)


class _FakeOSA:
    def measure(self, settings, averages=1, stop_event=None):
        wl = np.linspace(779.5e-9, 780.5e-9, 21)
        powers = np.ones(21) * 1e-6
        return TraceData(wavelengths=wl, powers=powers, trace_id="TRA", y_unit="LINear")


class ChannelMetricsTests(unittest.TestCase):
    def test_window_and_channel_integrals_peak_centered(self) -> None:
        wl = np.linspace(779.0, 781.0, 401)
        peak = 780.1  # offset from the nominal 780.0 -> exercises re-location
        sig = np.exp(-((wl - peak) ** 2) / (2 * 0.15 ** 2))
        (peak_wl, fwhm, total, window, channel, in_band,
         xt) = analysis._channel_metrics(wl, sig, 780.0, nominal_bw=0.2, pitch_nm=0.4)
        self.assertAlmostEqual(peak_wl, peak, places=2)     # centre relocated to peak
        self.assertGreater(window, 0.0)
        self.assertGreater(channel, window)                 # wider band integrates more
        self.assertGreater(total, channel)
        self.assertAlmostEqual(in_band, window / total)     # in-band = window / total


class MeasureThreadsColRatioTests(unittest.TestCase):
    def test_col_ratio_forwarded_to_encode(self) -> None:
        layout = _make_layout(width=10)
        ratio = np.ones(10)
        ratio[0] = 0.0
        ratio[-1] = 0.0

        seen: list = []
        real = analysis.encode_to_pattern

        def recorder(*args, **kwargs):
            seen.append(kwargs.get("col_ratio"))
            return real(*args, **kwargs)

        with mock.patch.object(analysis, "encode_to_pattern", recorder):
            measure_channel_spectra(
                _FakeOSA(), _FakeSLM(), layout, MeasurementSettings(),
                subtract_background=False, col_ratio=ratio,
            )
        # every encode call in the sweep received our exact profile
        self.assertTrue(seen)
        self.assertTrue(all(c is ratio for c in seen))


class _GaussianOSA:
    """Returns a narrow Gaussian centred at 780 nm (wavelengths in metres)."""

    def __init__(self) -> None:
        self.calls = 0

    def measure(self, settings, averages=1, stop_event=None):
        self.calls += 1
        wl = np.linspace(778.0e-9, 782.0e-9, 401)
        sig = 1e-6 * np.exp(-((wl * 1e9 - 780.0) ** 2) / (2 * 0.15 ** 2))
        return TraceData(wavelengths=wl, powers=sig, trace_id="TRA", y_unit="LINear")


class EncodingShapeTests(unittest.TestCase):
    def test_round_snaps_above_threshold(self) -> None:
        rounded = round_encoding_profile([0.5, 0.99, 0.991, 1.0])
        self.assertEqual(rounded.tolist(), [0.5, 0.99, 1.0, 1.0])

    def test_optimized_shape_rounded_flat_top(self) -> None:
        self.assertEqual(OPTIMIZED_ENCODING_SHAPE.shape, (8,))
        # every column that was > 0.99 is now exactly 1.0 (flat top)
        self.assertTrue(np.all(OPTIMIZED_ENCODING_SHAPE[3:] == 1.0))
        self.assertFalse(
            np.any((OPTIMIZED_ENCODING_SHAPE > 0.99) & (OPTIMIZED_ENCODING_SHAPE < 1.0))
        )
        self.assertAlmostEqual(OPTIMIZED_ENCODING_SHAPE[0], 0.38479, places=4)

    def test_flat_shape_all_ones_same_length(self) -> None:
        self.assertEqual(FLAT_ENCODING_SHAPE.shape, OPTIMIZED_ENCODING_SHAPE.shape)
        self.assertTrue(np.all(FLAT_ENCODING_SHAPE == 1.0))

    def test_mirror_to_15_is_symmetric_taper(self) -> None:
        full = mirror_intensity_profile(OPTIMIZED_ENCODING_SHAPE, 15)
        self.assertEqual(full.shape, (15,))
        self.assertTrue(np.allclose(full, full[::-1]))     # symmetric
        self.assertTrue(np.all(full[3:12] == 1.0))         # flat interior
        self.assertLess(full[0], full[3])                  # tapered edge


class TotalCrosstalkTests(unittest.TestCase):
    def test_sums_neighbour_dict(self) -> None:
        spec = _spectrum("x", 0, leak=0.05, in_band=0.9)
        spec.crosstalk = {-2: 0.01, -1: 0.03, 1: 0.02, 2: 0.005}
        self.assertAlmostEqual(spec.total_crosstalk, 0.065)

    def test_empty_is_zero(self) -> None:
        self.assertEqual(_spectrum("x", 0, 0.0, 0.0).total_crosstalk, 0.0)


class MeasureOneChannelTests(unittest.TestCase):
    def test_returns_metrics_for_selected_channel(self) -> None:
        layout = _make_layout(width=15)
        spec = measure_one_channel(
            _GaussianOSA(), _FakeSLM(), layout, MeasurementSettings(),
            side="x", index=0, subtract_background=False,
        )
        self.assertEqual((spec.side, spec.index), ("x", 0))
        self.assertAlmostEqual(spec.peak_wl_nm, 780.0, places=1)
        self.assertGreater(spec.in_band_fraction, 0.5)
        self.assertLess(spec.total_crosstalk, 0.05)         # isolated Gaussian
        self.assertAlmostEqual(spec.total_crosstalk, sum(spec.crosstalk.values()))

    def test_w_side_uses_list_position_not_channel_index(self) -> None:
        # w_channels[0] carries .index == 1 in this layout; the one-hot must use
        # the list position (0), so the spectrum reports index 0 and no IndexError.
        layout = _make_layout(width=15)
        spec = measure_one_channel(
            _GaussianOSA(), _FakeSLM(), layout, MeasurementSettings(),
            side="w", index=0, subtract_background=False,
        )
        self.assertEqual((spec.side, spec.index), ("w", 0))

    def test_col_ratio_forwarded_to_encode(self) -> None:
        layout = _make_layout(width=15)
        ratio = mirror_intensity_profile(OPTIMIZED_ENCODING_SHAPE, 15)
        seen: list = []
        real = analysis.encode_to_pattern

        def recorder(*args, **kwargs):
            seen.append(kwargs.get("col_ratio"))
            return real(*args, **kwargs)

        with mock.patch.object(analysis, "encode_to_pattern", recorder):
            measure_one_channel(
                _GaussianOSA(), _FakeSLM(), layout, MeasurementSettings(),
                side="x", index=0, subtract_background=True, col_ratio=ratio,
            )
        self.assertTrue(seen)                       # background + channel encodes
        self.assertTrue(all(c is ratio for c in seen))

    def test_out_of_range_index_raises(self) -> None:
        layout = _make_layout(width=15)
        with self.assertRaises(ValueError):
            measure_one_channel(
                _GaussianOSA(), _FakeSLM(), layout, MeasurementSettings(),
                side="x", index=5,
            )


class _FakeMonitor:
    """Minimal scope/DAQ monitor for the pair-grid sweep."""

    last_values = None

    def configure_monitor(self, *args, **kwargs) -> None:
        pass

    def monitor_cycle(self, timeout=30.0, **kwargs):
        class _Sample:
            value = 0.001
        return _Sample()


class CenterGapGeometryTests(unittest.TestCase):
    """center_gap_px widens only the central pad; None keeps legacy tiling."""

    # linear map: wl = 783 - 0.01 * x -> center_wl 778 sits exactly at x = 500
    COORDS = np.array([0.0, 1000.0])
    WLS = np.array([783.0, 773.0])

    def _geom(self, **kwargs):
        kwargs.setdefault("n_channels", 3)
        kwargs.setdefault("channel_width_px", 15)
        kwargs.setdefault("gap_px", 5)
        kwargs.setdefault("center_wl", 778.0)
        kwargs.setdefault("dark_wl_bands", ())
        return compute_channel_geometry(self.COORDS, self.WLS, **kwargs)

    def test_none_keeps_legacy_half_pitch_start(self) -> None:
        geom = self._geom()
        self.assertEqual(geom.c0, 500)
        self.assertIsNone(geom.center_gap_px)
        self.assertEqual([c.x_center for c in geom.x], [490, 470, 450])
        self.assertEqual([c.x_center for c in geom.w], [510, 530, 550])
        # central pad between the innermost pair is exactly gap_px wide
        self.assertEqual(geom.w[0].x_start - geom.x[0].x_end, 5)

    def test_gap_equal_to_gap_px_matches_legacy(self) -> None:
        # ceil((15 + 5)/2) == 10 == half pitch: same positions as None
        legacy = self._geom()
        explicit = self._geom(center_gap_px=5)
        self.assertEqual([c.x_center for c in explicit.x],
                         [c.x_center for c in legacy.x])
        self.assertEqual([c.x_center for c in explicit.w],
                         [c.x_center for c in legacy.w])
        self.assertEqual(explicit.center_gap_px, 5)

    def test_widened_center_pad(self) -> None:
        geom = self._geom(center_gap_px=10)
        # m0 = ceil((15 + 10)/2) = 13 -> innermost pair at c0 -/+ 13
        self.assertEqual([c.x_center for c in geom.x], [487, 467, 447])
        self.assertEqual([c.x_center for c in geom.w], [513, 533, 553])
        # pad = 2*13 - 15 = 11 >= requested 10
        self.assertEqual(geom.w[0].x_start - geom.x[0].x_end, 11)
        # mirror symmetry about c0 and unchanged same-side pitch
        for xc, wc in zip(geom.x, geom.w):
            self.assertEqual(xc.x_center + wc.x_center, 2 * geom.c0)
        self.assertEqual(geom.x[0].x_center - geom.x[1].x_center, 20)

    def test_odd_request_parity(self) -> None:
        # ceil((15 + 12)/2) = 14 -> pad 13 >= 12 (floor division would give 12)
        geom = self._geom(center_gap_px=12)
        self.assertEqual(geom.w[0].x_start - geom.x[0].x_end, 13)

    def test_guard_band_still_pushes_symmetrically(self) -> None:
        # guard band at px [505, 515] overlaps the w[0] window (506..520)
        # for center_gap_px=10 -> pair pushed outward together
        geom = self._geom(
            n_channels=1, center_gap_px=10,
            dark_wl_bands=((777.85, 777.95),),
        )
        self.assertEqual(geom.w[0].x_center, 523)   # cleared past guard hi+1
        self.assertEqual(geom.x[0].x_center, 477)   # mirror kept
        self.assertEqual(geom.x[0].x_center + geom.w[0].x_center, 2 * geom.c0)

    def test_negative_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "center_gap_px"):
            self._geom(center_gap_px=-1)

    def test_build_channel_layout_carries_center_gap(self) -> None:
        coords = np.arange(0.0, 1001.0, 20.0)
        calib = CalibrationResult(
            wavelength=783.0 - 0.01 * coords,
            coordinates=coords,
            max_level=900,
            min_level=100,
            level_range=np.asarray([100, 500, 900]),
            intensity_levels=np.tile(
                np.asarray([0.0, 0.5, 1.0]), (coords.size, 1)
            ),
        )
        layout = build_channel_layout(
            calib, n_channels=2, channel_width_px=15, gap_px=5,
            center_gap_px=10, center_wl=778.0, dark_wl_bands=(),
        )
        self.assertEqual(layout.center_gap_px, 10)
        self.assertEqual(layout.x_channels[0].x_center, 487)
        self.assertEqual(layout.w_channels[0].x_center, 513)
        # default (unset) stays None and keeps the legacy positions
        legacy = build_channel_layout(
            calib, n_channels=2, channel_width_px=15, gap_px=5,
            center_wl=778.0, dark_wl_bands=(),
        )
        self.assertIsNone(legacy.center_gap_px)
        self.assertEqual(legacy.x_channels[0].x_center, 490)


class MeasurePairGridsColRatioTests(unittest.TestCase):
    def test_col_ratio_forwarded_to_encode(self) -> None:
        from slm_module import encoding as encoding_module
        from slm_module import tpa_pair as tpa_pair_module

        layout = _make_layout(width=15)
        ratio = mirror_intensity_profile(OPTIMIZED_ENCODING_SHAPE, 15)
        seen: list = []
        real = encoding_module.encode_to_pattern

        def recorder(*args, **kwargs):
            seen.append(kwargs.get("col_ratio"))
            return real(*args, **kwargs)

        # measure_pair_grids does `from .encoding import encode_to_pattern` at
        # call time, so patch the module attribute it will re-import. fit_grid is
        # stubbed because the constant fake readings are a degenerate fit.
        with mock.patch.object(encoding_module, "encode_to_pattern", recorder), \
                mock.patch.object(tpa_pair_module, "fit_grid", lambda grid: None):
            tpa_pair_module.measure_pair_grids(
                _FakeMonitor(), _FakeSLM(), layout,
                pair_indices=[0], sweep=[0.0, 1.0], settle=0.0, col_ratio=ratio,
            )
        self.assertTrue(seen)                       # every grid point encoded
        self.assertTrue(all(c is ratio for c in seen))


if __name__ == "__main__":
    unittest.main()
