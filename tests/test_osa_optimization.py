from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osa_module.controller import MeasurementSettings, TraceData
from slm_module.calibration.calibration_new import CalibrationResult
from slm_module.optimization import (
    AmplitudeLUT,
    FixedChannelBins,
    OSAEvaluator,
    OSAOptimizationConfig,
    RunStore,
    StageAmplitudeReference,
    amplitudes_to_intensity_commands,
    efficiency_penalty,
    independent_intensity_profile,
    load_optimization_result,
    mirror_intensity_profile,
    run_osa_optimization,
    stage1_loss,
    stage3_loss,
    validate_independent_profile,
)
from slm_module.encoding import (
    ChannelLayout,
    EncodingChannel,
    build_single_anchor_layout,
    interpolate_coordinate_for_wavelength,
)


class IntensityProfileTests(unittest.TestCase):
    def test_mirror_fifteen_pixel_intensity_profile(self) -> None:
        independent = np.arange(1, 9, dtype=float) / 10.0
        full = mirror_intensity_profile(independent)
        np.testing.assert_allclose(
            full,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
             0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        )
        np.testing.assert_allclose(independent_intensity_profile(full), independent)

    def test_rejects_amplitude_or_bad_intensity_input_shape(self) -> None:
        with self.assertRaises(ValueError):
            validate_independent_profile(np.ones(15))
        with self.assertRaises(ValueError):
            validate_independent_profile([0, 0, 0, 0, 0, 0, 0, 1.1])
        with self.assertRaises(ValueError):
            independent_intensity_profile([0.2, 0.5, 0.9, 0.4, 0.2])


class SingleAnchorLayoutTests(unittest.TestCase):
    @staticmethod
    def _step2() -> CalibrationResult:
        return CalibrationResult(
            wavelength=np.asarray([780.0, 778.0, 776.0]),
            coordinates=np.asarray([0.0, 100.0, 200.0]),
            max_level=1023,
            min_level=0,
            level_range=np.asarray([0, 512, 1023]),
        )

    def test_interpolates_target_wavelength_to_fractional_pixel(self) -> None:
        coordinate = interpolate_coordinate_for_wavelength(self._step2(), 778.5)
        self.assertAlmostEqual(coordinate, 75.0)

    def test_builds_layout_with_target_as_only_offset_zero_anchor(self) -> None:
        intensity = CalibrationResult(
            wavelength=np.asarray([778.0]),
            coordinates=np.asarray([100.0]),
            max_level=1023,
            min_level=0,
            level_range=np.asarray([0, 512, 1023]),
            intensity_levels=np.asarray([[0.0, 0.4, 1.0]]),
        )
        layout, coordinate = build_single_anchor_layout(
            self._step2(), intensity, target_wavelength_nm=778.0,
            channel_width_px=15, gap_px=5,
        )

        self.assertEqual(coordinate, 100.0)
        ordered = sorted(layout.all_channels, key=lambda channel: channel.wavelength_nm)
        anchor = ordered[len(ordered) // 2]
        self.assertEqual(anchor.x_center, 100)
        self.assertEqual(anchor.wavelength_nm, 778.0)
        self.assertEqual(layout.n_channels, 5)
        self.assertTrue(all(channel.levels.size == 3 for channel in ordered))

    def test_rejects_target_outside_step2_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the Step 2 range"):
            interpolate_coordinate_for_wavelength(self._step2(), 781.0)


class AmplitudeLUTTests(unittest.TestCase):
    def test_inverse_maps_amplitude_to_intensity_command(self) -> None:
        amplitudes = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        commands = amplitudes ** 2
        lut = AmplitudeLUT(commands, amplitudes)
        self.assertAlmostEqual(lut.command_for(0.5), 0.25)
        np.testing.assert_allclose(
            lut.command_for(np.array([0.0, 0.5, 1.0])), [0.0, 0.25, 1.0]
        )

    def test_monotone_envelope_handles_noisy_measurements(self) -> None:
        lut = AmplitudeLUT(
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([0.0, 0.55, 0.52, 0.9, 1.0]),
        )
        self.assertTrue(np.all(np.diff(lut.measured_amplitudes) >= 0.0))
        self.assertTrue(0.25 <= lut.command_for(0.7) <= 0.75)


class FixedBinTests(unittest.TestCase):
    @staticmethod
    def _trace(wl_nm: np.ndarray, power: np.ndarray) -> TraceData:
        return TraceData(
            wavelengths=wl_nm * 1e-9,
            powers=np.asarray(power, dtype=float),
            trace_id="TRA",
            y_unit="LINear",
        )

    def test_midpoint_edges_and_signed_dark_integration(self) -> None:
        centers = np.array([777.8, 777.9, 778.0, 778.1, 778.2])
        bins = FixedChannelBins.from_centers("x0", centers)
        np.testing.assert_allclose(
            bins.edges_nm, [777.75, 777.85, 777.95, 778.05, 778.15, 778.25]
        )
        wl = np.linspace(777.75, 778.25, 1001)
        dark = np.full(wl.size, 2.0)
        on = dark.copy()
        on[(wl >= 777.95) & (wl <= 778.05)] += 3.0
        # First bin lies below dark. Signed integration must be clamped only
        # after integration rather than clipping every sample first.
        on[(wl >= 777.75) & (wl <= 777.85)] -= 1.0
        energy = bins.integrate(self._trace(wl, on), self._trace(wl, dark))
        self.assertEqual(energy[0], 0.0)
        self.assertGreater(energy[2], 0.0)
        self.assertEqual(energy[4], 0.0)


class LossTests(unittest.TestCase):
    def test_stage1_formula(self) -> None:
        expected = np.mean([0.5 / 1.0, 0.4 / 0.8])
        expected += np.mean([efficiency_penalty(0.90), efficiency_penalty(0.85)])
        actual = stage1_loss(
            [0.5, 0.4], [0.90, 0.85], [1.0, 0.8], c_floor=1e-9
        )
        self.assertAlmostEqual(actual, expected)

    def test_stage3_formula(self) -> None:
        # First anchor stays inside the 5% crosstalk allowance; second reaches
        # 10%, so its guard is ((1.10-1.05)/0.05)^2 == 1.
        actual = stage3_loss(
            rmse=[0.05, 0.10],
            crosstalk=[0.10, 0.22],
            eta=[0.90, 0.87],
            rmse_stage1=[0.10, 0.20],
            c_stage1=[0.10, 0.20],
            sigma_a=0.01,
            c_floor=1e-9,
        )
        self.assertAlmostEqual(actual, 0.5 + 0.5)


class RunStoreTests(unittest.TestCase):
    def test_config_and_candidate_records_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = OSAOptimizationConfig(
                settings=MeasurementSettings(span="0.8nm"),
                output_root=tmp,
                run_name="unit",
            )
            store = RunStore(config)
            store.write_json("run_config.json", config.to_dict())
            self.assertTrue((Path(tmp) / "unit" / "run_config.json").is_file())
            self.assertTrue((Path(tmp) / "unit" / "candidates.csv").is_file())


def _hardware_layout() -> ChannelLayout:
    levels = np.arange(0, 1001, 10, dtype=int)
    curve = levels.astype(float) / 1000.0
    wavelengths = 777.65 + 0.1 * np.arange(8)
    channels: list[EncodingChannel] = []
    for position, wavelength in enumerate(wavelengths):
        side = "x" if position < 4 else "w"
        index = position if side == "x" else position - 4
        start = 10 + 20 * position
        channels.append(
            EncodingChannel(
                index=index,
                side=side,
                x_center=start + 7,
                x_start=start,
                x_end=start + 15,
                wavelength_nm=float(wavelength),
                levels=levels.copy(),
                intensity_curve=curve.copy(),
            )
        )
    return ChannelLayout(
        x_channels=channels[:4],
        w_channels=channels[4:],
        center_wl=778.0,
        center_x=87.0,
        channel_width_px=15,
        pitch_px=20,
        nm_per_px=0.005,
        calib_coords=np.array([0.0, 199.0]),
        calib_off_levels=np.array([0, 0]),
    )


class _HardwareSLM:
    def __init__(self) -> None:
        self.last = np.zeros((2, 200), dtype=np.uint16)

    def get_slm_info(self):
        return 200, 2

    def display_array(self, pattern):
        self.last = np.asarray(pattern, dtype=np.uint16)


class _HardwareOSA:
    def __init__(self, slm: _HardwareSLM, layout: ChannelLayout) -> None:
        self.slm = slm
        self.layout = layout

    def measure(self, settings, averages=1, stop_event=None):
        center = float(settings.center_wl.lower().replace("nm", ""))
        span = float(settings.span.lower().replace("nm", ""))
        wl = np.linspace(center - span / 2.0, center + span / 2.0, 1601)
        power = np.full(wl.size, 1e-9)
        for channel in self.layout.all_channels:
            requested_intensity = float(
                np.mean(self.slm.last[0, channel.x_start:channel.x_end]) / 1000.0
            )
            power += requested_intensity * np.exp(
                -0.5 * ((wl - channel.wavelength_nm) / 0.030) ** 2
            )
        return TraceData(
            wavelengths=wl * 1e-9,
            powers=power,
            trace_id="TRA",
            y_unit="LINear",
        )


class EvaluatorIntegrationTests(unittest.TestCase):
    def test_runtime_amplitudes_use_final_intensity_lut(self) -> None:
        layout = _hardware_layout()
        reference = StageAmplitudeReference(
            lut=AmplitudeLUT(
                commands=np.array([0.0, 0.25, 1.0]),
                measured_amplitudes=np.array([0.0, 0.5, 1.0]),
            ),
            e_blank=0.0,
            e_full=1.0,
        )
        x, w = amplitudes_to_intensity_commands(
            np.full(layout.n_channels, 0.5),
            np.full(layout.n_channels, 0.5),
            layout,
            {0: reference},
        )
        np.testing.assert_allclose(x, 0.25)
        np.testing.assert_allclose(w, 0.25)

    def test_fixed_bins_lut_and_modulation_with_fake_hardware(self) -> None:
        layout = _hardware_layout()
        slm = _HardwareSLM()
        osa = _HardwareOSA(slm, layout)
        with tempfile.TemporaryDirectory() as tmp:
            config = OSAOptimizationConfig(
                settings=MeasurementSettings(
                    center_wl="778nm", span="0.8nm", sampling_points="1001",
                    y_unit="LINear"
                ),
                anchor_offsets=(0,),
                averages=1,
                rerank_averages=1,
                baseline_repeats=2,
                stage2_repeats=1,
                output_root=tmp,
                run_name="hardware",
                lut_self_consistency=False,
                discrete_refine=False,
            )
            store = RunStore(config)
            evaluator = OSAEvaluator(osa, slm, layout, config, store)
            evaluator.calibrate_all()
            one_hot, _ = evaluator.measure_one_hot(
                0, np.ones(8), stage="test_one_hot"
            )
            self.assertGreater(one_hot.c_total, 0.0)
            self.assertAlmostEqual(one_hot.eta, 1.0, delta=0.03)

            reference = evaluator.build_amplitude_lut(
                0, np.ones(8), self_consistent=False, stage="test_lut"
            )
            self.assertAlmostEqual(reference.lut.command_for(0.5), 0.25, delta=0.05)
            metrics, hashes = evaluator.measure_modulation(
                0, np.ones(8), reference, stage="test_modulation"
            )
            self.assertTrue(np.isfinite(metrics.rmse))
            self.assertEqual(metrics.amplitude_errors.size, 14)
            self.assertTrue(hashes)

    def test_complete_tiny_budget_run_writes_final_result(self) -> None:
        layout = _hardware_layout()
        slm = _HardwareSLM()
        osa = _HardwareOSA(slm, layout)
        with tempfile.TemporaryDirectory() as tmp:
            config = OSAOptimizationConfig(
                settings=MeasurementSettings(
                    center_wl="778nm", span="0.8nm", sampling_points="1001",
                    y_unit="LINear"
                ),
                anchor_offsets=(0,),
                averages=1,
                rerank_averages=1,
                baseline_repeats=2,
                stage2_repeats=1,
                stage1_maxfev=1,
                stage3_maxfev=1,
                stage1_top_k=1,
                stage3_top_k=1,
                max_alternations=1,
                final_lut_points=5,
                lut_self_consistency=False,
                discrete_refine=False,
                reference_interval_candidates=0,
                output_root=tmp,
                run_name="complete",
            )
            result = run_osa_optimization(
                osa, slm, layout, np.ones(8), config=config
            )
            self.assertEqual(result.final_l.shape, (8,))
            self.assertEqual(result.final_profile.shape, (15,))
            self.assertTrue(Path(result.run_dir, "final_result.json").is_file())
            self.assertTrue(Path(result.run_dir, "optimizer_state.json").is_file())
            loaded = load_optimization_result(result.run_dir)
            np.testing.assert_allclose(loaded.final_l, result.final_l)
            self.assertEqual(set(loaded.final_luts), set(result.final_luts))
            self.assertEqual(loaded.accepted, result.accepted)


if __name__ == "__main__":
    unittest.main()
