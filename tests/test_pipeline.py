from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import slm_module.pipeline as pipeline
from slm_module.calibration.calibration_new import (
    CalibrationProgress,
    CalibrationResult,
    save_calibration_result,
)
from slm_module.pipeline import (
    InputSpec,
    IntensityConfig,
    LayoutConfig,
    PipelineAborted,
    PipelineInstruments,
    PipelineRequest,
    PipelineStageError,
    StagePlan,
    TPACenterConfig,
    WlMapConfig,
    plot_point,
    required_instruments,
    run_pipeline,
    validate_request,
)
from slm_module.tpa_center import TPACenterProgress
from slm_module.tpa_pair import TPAPairAborted, TPAPairProgress
from slm_module.tpa_phase_measure import TPAPhaseProgress


def _plan(stage_id, config, inputs, out_dir: Path, name: str) -> StagePlan:
    return StagePlan(
        stage_id=stage_id,
        config=config,
        inputs=inputs,
        output_path=out_dir / name,
    )


def _instruments() -> PipelineInstruments:
    return PipelineInstruments(slm=object(), osa=object(), monitor=object())


class ValidateRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_empty_request_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no pipeline stages"):
            validate_request(PipelineRequest(stages=[]))

    def test_unknown_stage_rejected(self) -> None:
        plan = _plan("bogus", WlMapConfig(), {}, self.out, "x.json")
        with self.assertRaisesRegex(ValueError, "unknown pipeline stage"):
            validate_request(PipelineRequest(stages=[plan]))

    def test_out_of_order_rejected(self) -> None:
        seed = self.out / "seed.json"
        seed.write_text("{}", encoding="utf-8")
        stages = [
            _plan("intensity", IntensityConfig(),
                  {"wl_map": InputSpec("file", seed)}, self.out, "b.json"),
            _plan("wl_map", WlMapConfig(), {}, self.out, "a.json"),
        ]
        with self.assertRaisesRegex(ValueError, "out of order"):
            validate_request(PipelineRequest(stages=stages))

    def test_memory_input_needs_earlier_producer(self) -> None:
        plan = _plan("intensity", IntensityConfig(),
                     {"wl_map": InputSpec("memory")}, self.out, "b.json")
        with self.assertRaisesRegex(ValueError, "from memory"):
            validate_request(PipelineRequest(stages=[plan]))

    def test_memory_input_accepted_when_producer_enabled(self) -> None:
        stages = [
            _plan("wl_map", WlMapConfig(), {}, self.out, "a.json"),
            _plan("intensity", IntensityConfig(),
                  {"wl_map": InputSpec("memory")}, self.out, "b.json"),
        ]
        validate_request(PipelineRequest(stages=stages))   # must not raise

    def test_missing_file_rejected(self) -> None:
        plan = _plan("intensity", IntensityConfig(),
                     {"wl_map": InputSpec("file", self.out / "absent.json")},
                     self.out, "b.json")
        with self.assertRaisesRegex(ValueError, "file not found"):
            validate_request(PipelineRequest(stages=[plan]))

    def test_missing_required_input_rejected(self) -> None:
        plan = _plan("intensity", IntensityConfig(), {}, self.out, "b.json")
        with self.assertRaisesRegex(ValueError, "missing required input"):
            validate_request(PipelineRequest(stages=[plan]))

    def test_unexpected_input_key_rejected(self) -> None:
        plan = _plan("wl_map", WlMapConfig(),
                     {"pair_etas": InputSpec("memory")}, self.out, "a.json")
        with self.assertRaisesRegex(ValueError, "does not accept input"):
            validate_request(PipelineRequest(stages=[plan]))

    def test_wrong_config_type_rejected(self) -> None:
        plan = _plan("wl_map", TPACenterConfig(), {}, self.out, "a.json")
        with self.assertRaisesRegex(ValueError, "WlMapConfig"):
            validate_request(PipelineRequest(stages=[plan]))

    def test_duplicate_output_paths_rejected(self) -> None:
        stages = [
            _plan("wl_map", WlMapConfig(), {}, self.out, "same.json"),
            _plan("intensity", IntensityConfig(),
                  {"wl_map": InputSpec("memory")}, self.out, "same.json"),
        ]
        with self.assertRaisesRegex(ValueError, "used twice"):
            validate_request(PipelineRequest(stages=stages))

    def test_required_instruments_union(self) -> None:
        seed = self.out / "seed.json"
        seed.write_text("{}", encoding="utf-8")
        osa_only = PipelineRequest(stages=[
            _plan("wl_map", WlMapConfig(), {}, self.out, "a.json"),
        ])
        self.assertEqual(required_instruments(osa_only), {"osa", "slm"})
        tpa_only = PipelineRequest(stages=[
            _plan("tpa_center", TPACenterConfig(),
                  {"intensity_calib": InputSpec("file", seed)},
                  self.out, "c.json"),
        ])
        self.assertEqual(required_instruments(tpa_only), {"monitor", "slm"})


class RunPipelineOrchestrationTests(unittest.TestCase):
    """Orchestration semantics with the stage runners monkeypatched out."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _stages(self) -> list[StagePlan]:
        return [
            _plan("wl_map", WlMapConfig(), {}, self.out, "wl.json"),
            _plan("intensity", IntensityConfig(),
                  {"wl_map": InputSpec("memory")}, self.out, "int.json"),
        ]

    def test_memory_artifact_flows_and_outputs_recorded(self) -> None:
        seen_inputs: list = []

        def fake_wl(ctx, plan):
            ctx.record(plan.output_path)
            return "WL_ARTIFACT"

        def fake_intensity(ctx, plan):
            seen_inputs.append(ctx.resolve(plan, "wl_map"))
            ctx.record(plan.output_path)
            return "INT_ARTIFACT"

        with mock.patch.dict(
            pipeline._STAGE_RUNNERS,
            {"wl_map": fake_wl, "intensity": fake_intensity},
        ):
            outcome = run_pipeline(
                PipelineRequest(stages=self._stages()), _instruments()
            )

        self.assertEqual(seen_inputs, ["WL_ARTIFACT"])
        self.assertEqual(outcome.artifacts["wl_map"], "WL_ARTIFACT")
        self.assertEqual(outcome.artifacts["intensity_calib"], "INT_ARTIFACT")
        self.assertEqual(
            [p.name for p in outcome.saved_files], ["wl.json", "int.json"]
        )

    def test_abort_mid_chain_keeps_earlier_saved_files(self) -> None:
        def fake_wl(ctx, plan):
            ctx.record(plan.output_path)
            return "WL_ARTIFACT"

        def aborting_intensity(ctx, plan):
            raise TPAPairAborted("stopped")

        with mock.patch.dict(
            pipeline._STAGE_RUNNERS,
            {"wl_map": fake_wl, "intensity": aborting_intensity},
        ):
            with self.assertRaises(PipelineAborted) as caught:
                run_pipeline(
                    PipelineRequest(stages=self._stages()), _instruments()
                )
        self.assertEqual(caught.exception.stage_id, "intensity")
        self.assertEqual(
            [p.name for p in caught.exception.saved_files], ["wl.json"]
        )

    def test_stage_failure_wrapped_with_stage_id(self) -> None:
        def fake_wl(ctx, plan):
            ctx.record(plan.output_path)
            return "WL_ARTIFACT"

        def broken_intensity(ctx, plan):
            raise RuntimeError("boom")

        with mock.patch.dict(
            pipeline._STAGE_RUNNERS,
            {"wl_map": fake_wl, "intensity": broken_intensity},
        ):
            with self.assertRaises(PipelineStageError) as caught:
                run_pipeline(
                    PipelineRequest(stages=self._stages()), _instruments()
                )
        self.assertEqual(caught.exception.stage_id, "intensity")
        self.assertIsInstance(caught.exception.error, RuntimeError)
        self.assertEqual(
            [p.name for p in caught.exception.saved_files], ["wl.json"]
        )

    def test_missing_instrument_rejected_before_any_stage(self) -> None:
        instruments = PipelineInstruments(slm=object(), osa=None, monitor=None)
        with self.assertRaisesRegex(ValueError, "'osa'"):
            run_pipeline(PipelineRequest(stages=self._stages()), instruments)

    def test_file_input_loaded_through_loader_table(self) -> None:
        calib = CalibrationResult(
            wavelength=np.asarray([780.0, 778.0]),
            coordinates=np.asarray([0.0, 100.0]),
            max_level=900,
            min_level=100,
            level_range=np.asarray([100, 900]),
        )
        wl_path = self.out / "wl_input.json"
        save_calibration_result(calib, wl_path)

        loaded: list = []

        def fake_intensity(ctx, plan):
            loaded.append(ctx.resolve(plan, "wl_map"))
            ctx.record(plan.output_path)
            return "INT_ARTIFACT"

        plan = _plan("intensity", IntensityConfig(),
                     {"wl_map": InputSpec("file", wl_path)},
                     self.out, "int.json")
        with mock.patch.dict(
            pipeline._STAGE_RUNNERS, {"intensity": fake_intensity}
        ):
            run_pipeline(PipelineRequest(stages=[plan]), _instruments())

        self.assertEqual(len(loaded), 1)
        self.assertIsInstance(loaded[0], CalibrationResult)
        np.testing.assert_allclose(loaded[0].coordinates, calib.coordinates)

    def test_progress_wrapped_with_stage_identity(self) -> None:
        reports: list = []

        def fake_wl(ctx, plan):
            cb = ctx.forward(pipeline.STAGE_BY_ID["wl_map"])
            cb(CalibrationProgress("min_max", 1, 10, "level 5", x=5.0, y=1e-6))
            ctx.record(plan.output_path)
            return "WL_ARTIFACT"

        def fake_intensity(ctx, plan):
            ctx.record(plan.output_path)
            return "INT_ARTIFACT"

        with mock.patch.dict(
            pipeline._STAGE_RUNNERS,
            {"wl_map": fake_wl, "intensity": fake_intensity},
        ):
            run_pipeline(
                PipelineRequest(stages=self._stages()), _instruments(),
                progress_callback=reports.append,
            )
        self.assertEqual(len(reports), 1)
        progress = reports[0]
        self.assertEqual(progress.stage_id, "wl_map")
        self.assertEqual(progress.stage_index, 0)
        self.assertEqual(progress.n_stages, 2)
        self.assertEqual(progress.inner.message, "level 5")


class LayoutConfigBuildLayoutTests(unittest.TestCase):
    """build_layout loads a Step-3 channel calibration verbatim (no re-tiling)."""

    def _channel_calib(self) -> CalibrationResult:
        # mirror-symmetric channel grid: pitch 20 px around centre 640
        coords = np.asarray([590.0, 610.0, 630.0, 650.0, 670.0, 690.0])
        wls = 778.0 - 0.01 * (coords - 640.0)   # a < 0: higher px -> lower wl
        levels = np.arange(100, 105)
        intens = np.tile(
            np.linspace(0.0, 1.0, levels.size), (coords.size, 1)
        ) + np.arange(coords.size)[:, None]     # distinct curve per row
        return CalibrationResult(
            wavelength=wls, coordinates=coords,
            max_level=int(levels.max()), min_level=int(levels.min()),
            level_range=levels, intensity_levels=intens,
        )

    def test_channels_are_calibration_rows(self) -> None:
        layout = LayoutConfig(gap_px=5).build_layout(self._channel_calib())
        self.assertEqual(layout.n_channels, 3)
        self.assertEqual(layout.pitch_px, 20)
        self.assertEqual(layout.channel_width_px, 15)  # pitch - gap_px
        # innermost-first x/w mirror pairs at exactly the measured centres
        self.assertEqual([ch.x_center for ch in layout.x_channels],
                         [630, 610, 590])
        self.assertEqual([ch.x_center for ch in layout.w_channels],
                         [650, 670, 690])

    def test_configured_gap_recovers_width(self) -> None:
        layout = LayoutConfig(gap_px=8).build_layout(self._channel_calib())
        self.assertEqual(layout.channel_width_px, 12)  # pitch 20 - gap 8

    def test_geometry_fields_do_not_retile(self) -> None:
        # n_channels / center_wl are grid-DESIGN inputs; a measured grid wins
        config = LayoutConfig(n_channels=1, center_wl=760.0)
        layout = config.build_layout(self._channel_calib())
        self.assertEqual(layout.n_channels, 3)
        self.assertAlmostEqual(layout.center_x, 640.0)


class PlotPointTests(unittest.TestCase):
    def test_calibration_progress_passthrough(self) -> None:
        inner = CalibrationProgress("wavelength", 3, 10, "x=5", x=5.0, y=778.1)
        self.assertEqual(
            plot_point(inner), ("wavelength", 3, 10, "x=5", 5.0, 778.1)
        )

    def test_tpa_center_progress(self) -> None:
        inner = TPACenterProgress(
            step=2, total=8, message="centre", center_wl_nm=778.01, signal_v=0.002
        )
        phase, step, total, message, x, y = plot_point(inner)
        self.assertEqual((phase, step, total), ("tpa_center", 2, 8))
        self.assertAlmostEqual(x, 778.01)
        self.assertAlmostEqual(y, 0.002)

    def test_tpa_pair_progress(self) -> None:
        inner = TPAPairProgress(step=4, total=30, message="pair 0", pair_index=0)
        phase, step, total, _message, x, _y = plot_point(inner)
        self.assertEqual((phase, step, total, x), ("pair_eta", 4, 30, 4.0))

    def test_tpa_phase_progress(self) -> None:
        inner = TPAPhaseProgress(step=7, total=150, message="theta 90")
        phase, step, total, _message, x, y = plot_point(inner)
        self.assertEqual((phase, step, total, x, y), ("comb_phase", 7, 150, 7.0, None))


class PhaseReportRenderTests(unittest.TestCase):
    def test_plot_fringe_renders_on_agg(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from slm_module.tpa_phase import PhaseFit
        from slm_module.tpa_phase_report import plot_fringe

        n = 15
        theta = np.linspace(0.0, np.pi, n)
        g = np.sin(theta / 2.0) ** 2
        dphi_slm = theta - np.pi
        a, b, dphi_comb = 0.03, 0.02, 0.4
        y = a**2 + b**2 * g**2 + 2 * a * b * g * np.cos(dphi_slm + dphi_comb)
        sem = np.full(n, 1e-5)
        fit = PhaseFit(
            dphi_comb=dphi_comb, dphi_comb_err=0.02,
            a=a, a_err=1e-3, b=b, b_err=1e-3,
            amp=2 * a * b, amp_err=1e-4,
            offset=0.0, offset_err=1e-5,
            chi2_red=1.1, dof=n - 3, birge=1.05, r2=0.99,
            eta_ref=a, eta_tgt=b, bound_frac=1.0,
            a_at_bound=False, b_at_bound=False,
            bg0=0.0, bg1=0.0, bg2=0.0,
            dphi_slm=dphi_slm, g=g, y=y, sem=sem,
            known=a**2 + b**2 * g**2, y_pred=y, residuals=np.zeros(n),
        )
        fig = Figure(figsize=(8, 4))
        plot_fringe(fig, fit, tgt=3)      # must render without raising
        self.assertEqual(len(fig.axes), 2)


class CombPhaseJsonTests(unittest.TestCase):
    """save_comb_phase_json / load_comb_phase_json round trip (no hardware)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    @staticmethod
    def _fit(dphi: float, frac: float):
        from slm_module.tpa_phase import PhaseFit

        n = 5
        arr = np.zeros(n)
        return PhaseFit(
            dphi_comb=dphi, dphi_comb_err=0.02,
            a=0.05, a_err=1e-3, b=0.048, b_err=1e-3,
            amp=2 * 0.05 * 0.048, amp_err=1e-4,
            offset=0.0, offset_err=1e-5,
            chi2_red=1.0, dof=n - 3, birge=1.0, r2=0.99,
            eta_ref=0.05, eta_tgt=0.048, bound_frac=frac,
            a_at_bound=False, b_at_bound=False,
            bg0=0.0, bg1=0.0, bg2=0.0,
            dphi_slm=arr, g=arr, y=arr, sem=np.ones(n),
            known=arr, y_pred=arr, residuals=arr,
        )

    def test_round_trip_and_method_selection(self) -> None:
        import json

        from slm_module.tpa_phase import load_comb_phase_json, save_comb_phase_json

        step6 = self.out / "step6.json"
        step6.write_text(
            json.dumps({"step3": {"probe": 1}, "step6": {"channels": []}}),
            encoding="utf-8",
        )
        fits = {
            (3, "bounded"): self._fit(0.30, 1.0),
            (3, "fix"): self._fit(0.29, 0.0),
            (5, "bounded"): self._fit(-0.23, 1.0),
            (5, "fix"): self._fit(-0.51, 0.0),
        }
        out = self.out / "step7.json"
        save_comb_phase_json(fits, step6, out, ref_index=1,
                             csv_path="meas.csv", single_beam_bg=True)

        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["step3"], {"probe": 1})   # carried over verbatim
        self.assertEqual(payload["step6"], {"channels": []})
        self.assertEqual(payload["step7"]["ref_index"], 1)

        ref, entries = load_comb_phase_json(out, method="bounded")
        self.assertEqual(ref, 1)
        self.assertEqual(sorted(entries), [3, 5])
        self.assertAlmostEqual(entries[3]["fit"]["dphi_comb_rad"], 0.30)
        self.assertAlmostEqual(entries[5]["fit"]["dphi_comb_rad"], -0.23)

        _, fixed = load_comb_phase_json(out, method="fix")
        self.assertAlmostEqual(fixed[5]["fit"]["dphi_comb_rad"], -0.51)
        self.assertEqual(fixed[3]["fit"]["bound_frac"], 0.0)

        with self.assertRaisesRegex(ValueError, "several stored fits"):
            load_comb_phase_json(out)                      # ambiguous without method
        with self.assertRaisesRegex(ValueError, "no 'other' fit"):
            load_comb_phase_json(out, method="other")


if __name__ == "__main__":
    unittest.main()
