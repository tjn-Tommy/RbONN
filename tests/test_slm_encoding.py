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
    encode_to_pattern,
    optimize_from_osa,
)
import slm_module.analysis as analysis
from slm_module.analysis import (
    ChannelSpectrum,
    ModulationErrorResult,
    encoding_gain,
    measure_channel_spectra,
    write_gain_csv,
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


if __name__ == "__main__":
    unittest.main()
