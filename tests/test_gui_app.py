from __future__ import annotations

import json
import os

# Render Qt/matplotlib headless before either is imported by the app module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_module.gui.app import _format_duration


class FormatDurationTests(unittest.TestCase):
    def test_minutes_and_seconds(self) -> None:
        self.assertEqual(_format_duration(0), "0:00")
        self.assertEqual(_format_duration(7), "0:07")
        self.assertEqual(_format_duration(67), "1:07")
        self.assertEqual(_format_duration(750), "12:30")

    def test_rolls_over_to_hours(self) -> None:
        self.assertEqual(_format_duration(3725), "1:02:05")

    def test_non_finite_or_negative_is_dash(self) -> None:
        self.assertEqual(_format_duration(-1), "—")
        self.assertEqual(_format_duration(float("nan")), "—")
        self.assertEqual(_format_duration(float("inf")), "—")


class CalibrationDialogEtaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_eta_estimates_remaining_from_pace(self) -> None:
        from slm_module.calibration.calibration_new import CalibrationProgress
        from slm_module.gui.app import CalibrationProgressDialog

        dialog = CalibrationProgressDialog()
        try:
            # first update enters the phase and starts the clock
            dialog.update_progress(
                CalibrationProgress("intensity", step=0, total=10, message="start")
            )
            # pin the phase start 10 s in the past: 1/10 done -> ~90 s remaining
            dialog._phase_start = time.perf_counter() - 10.0
            dialog.update_progress(
                CalibrationProgress("intensity", step=0, total=10, message="tick")
            )
            text = dialog.eta_label.text()
            self.assertTrue(text.startswith("Elapsed 0:1"), text)
            self.assertIn("ETA 1:", text)
        finally:
            dialog.close()


class MainWindowStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_navigation_has_one_page_per_item(self) -> None:
        from slm_module.gui.app import MainWindow

        window = MainWindow()
        try:
            self.assertEqual(window.nav.count(), window.stack.count())
            labels = [window.nav.item(i).text() for i in range(window.nav.count())]
            self.assertEqual(sum("TPA Encoding" in label for label in labels), 1)
            self.assertFalse(any("Scope Monitor" in label for label in labels))
        finally:
            window.close()

    def test_pipeline_tab_supports_skipped_prerequisites(self) -> None:
        from slm_module.gui.app import MainWindow

        window = MainWindow()
        try:
            self.assertEqual(window.calibration_tabs.tabText(0), "Pipeline")
            self.assertFalse(window.pipeline_input_edits[2].isEnabled())
            self.assertFalse(window.pipeline_input_edits[3].isEnabled())

            window.pipeline_checks[1].setChecked(False)
            self.assertTrue(window.pipeline_input_edits[2].isEnabled())
            self.assertFalse(window.pipeline_input_edits[3].isEnabled())

            window.pipeline_checks[2].setChecked(False)
            self.assertTrue(window.pipeline_input_edits[3].isEnabled())
        finally:
            window.close()

    def test_pipeline_encoding_optimization_file_controls(self) -> None:
        from slm_module.gui.app import MainWindow

        window = MainWindow()
        try:
            self.assertFalse(window.pipeline_checks[4].isChecked())
            self.assertFalse(window.pipeline_profile_edit.isEnabled())
            self.assertFalse(window.pipeline_profile_values_edit.isEnabled())

            window.pipeline_checks[4].setChecked(True)
            self.assertTrue(window.pipeline_profile_values_edit.isEnabled())
            self.assertFalse(window.pipeline_profile_edit.isEnabled())
            self.assertFalse(window.pipeline_input_edits[4].isEnabled())
            self.assertIn("Step 3 output JSON", window.pipeline_source_labels[4].text())

            window.pipeline_profile_source_combo.setCurrentIndex(1)
            self.assertFalse(window.pipeline_profile_values_edit.isEnabled())
            self.assertTrue(window.pipeline_profile_edit.isEnabled())

            window.pipeline_checks[3].setChecked(False)
            self.assertTrue(window.pipeline_input_edits[4].isEnabled())
            self.assertIn("external Step 3 JSON", window.pipeline_source_labels[4].text())
        finally:
            window.close()

    def test_pipeline_parses_direct_initial_profile(self) -> None:
        from slm_module.gui.app import MainWindow

        window = MainWindow()
        try:
            expected = np.linspace(0.2, 0.9, 8)
            parsed = window._parse_pipeline_initial_profile(
                ", ".join(f"{value:g}" for value in expected)
            )
            np.testing.assert_allclose(parsed, expected)

            full = np.concatenate([expected, expected[-2::-1]])
            parsed_full = window._parse_pipeline_initial_profile(
                json.dumps(full.tolist())
            )
            np.testing.assert_allclose(parsed_full, expected)
        finally:
            window.close()

    def test_pipeline_step3_csv_reuses_step_panel_levels(self) -> None:
        from slm_module.calibration.calibration_new import CalibrationResult
        from slm_module.gui.app import MainWindow

        class ConnectedOSA:
            is_connected = True

        window = MainWindow()
        try:
            window.osa_controller = ConnectedOSA()
            window._controller = lambda: object()
            window.pipeline_checks[1].setChecked(False)
            window.pipeline_checks[2].setChecked(False)
            window.pipeline_checks[3].setChecked(True)
            window.step_widgets[3]["min"].setValue(123)
            window.step_widgets[3]["max"].setValue(900)
            window._launch_calibration = lambda _label, _work: None

            mapping = CalibrationResult(
                wavelength=np.asarray([778.0]),
                coordinates=np.asarray([100.0]),
                max_level=900,
                min_level=123,
                level_range=np.asarray([], dtype=int),
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                input_path = root / "wavelength_map.csv"
                input_path.write_text("placeholder", encoding="utf-8")
                window.pipeline_input_edits[3].setText(str(input_path))
                window.pipeline_output_edits[3].setText(str(root / "step3.json"))
                window.pipeline_csv_edit.setText(str(root / "calibration.csv"))

                with patch(
                    "slm_module.gui.app.load_wavelength_map_csv",
                    return_value=mapping,
                ) as load_map:
                    window._run_pipeline()

                self.assertEqual(load_map.call_count, 1)
                self.assertEqual(load_map.call_args.kwargs["min_level"], 123)
                self.assertEqual(load_map.call_args.kwargs["max_level"], 900)
                self.assertFalse(hasattr(window, "pipeline_csv_min_spin"))
                self.assertFalse(hasattr(window, "pipeline_csv_max_spin"))
        finally:
            window.close()

    def test_pipeline_reloads_each_selected_step_output(self) -> None:
        from slm_module.calibration.calibration_new import (
            CalibrationResult,
            load_calibration_result as real_load_calibration_result,
        )
        from slm_module.gui.app import MainWindow

        class ConnectedOSA:
            is_connected = True

        window = MainWindow()
        try:
            window.osa_controller = ConnectedOSA()
            window._controller = lambda: object()
            launched = {}
            window._launch_calibration = (
                lambda label, work: launched.update(label=label, work=work)
            )

            def wavelength_result(*_args, **_kwargs):
                return CalibrationResult(
                    wavelength=np.asarray([778.0]),
                    coordinates=np.asarray([100.0]),
                    max_level=1023,
                    min_level=0,
                    level_range=np.asarray([0, 1023]),
                )

            def intensity_result(_osa, _slm, levels, _settings, mapping, **_kwargs):
                self.assertEqual(mapping.coordinates.tolist(), [100.0])
                count = len(levels)
                return CalibrationResult(
                    wavelength=mapping.wavelength.copy(),
                    coordinates=mapping.coordinates.copy(),
                    max_level=mapping.max_level,
                    min_level=mapping.min_level,
                    level_range=np.asarray(levels),
                    intensity_levels=np.ones((1, count)),
                    raw_intensity_levels=np.ones((1, count)),
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                paths = {
                    1: Path(temp_dir, "step1.json"),
                    2: Path(temp_dir, "step2.json"),
                    3: Path(temp_dir, "step3.json"),
                }
                for step, path in paths.items():
                    window.pipeline_output_edits[step].setText(str(path))
                window.pipeline_csv_edit.setText(str(Path(temp_dir, "result.csv")))

                with (
                    patch(
                        "slm_module.gui.app.find_min_max_intensity_levels",
                        return_value=(0.0, 1.0, 0, 1023, {}),
                    ),
                    patch(
                        "slm_module.gui.app.wavelength_calibration",
                        side_effect=wavelength_result,
                    ),
                    patch(
                        "slm_module.gui.app.intensity_calibration",
                        side_effect=intensity_result,
                    ),
                    patch(
                        "slm_module.gui.app.load_calibration_result",
                        wraps=real_load_calibration_result,
                    ) as load_result,
                ):
                    window._run_pipeline()
                    payload = launched["work"](lambda _progress: None, threading.Event())

                self.assertEqual(launched["label"], "Run pipeline")
                self.assertEqual(payload["step"], "pipeline")
                self.assertEqual(load_result.call_count, 2)
                loaded_paths = [Path(call.args[0]) for call in load_result.call_args_list]
                self.assertEqual(loaded_paths, [paths[1].resolve(), paths[2].resolve()])
                self.assertTrue(paths[3].is_file())
                self.assertTrue(Path(temp_dir, "result.csv").is_file())
        finally:
            window.close()

    def test_pipeline_runs_encoding_optimization_from_direct_input(self) -> None:
        from slm_module.calibration.calibration_new import (
            CalibrationResult,
            load_calibration_result as real_load_calibration_result,
            save_calibration_result,
        )
        from slm_module.gui.app import MainWindow
        from slm_module.optimization import OptimizationResult

        class ConnectedOSA:
            is_connected = True

        class OpenSLM:
            is_open = True

        window = MainWindow()
        try:
            window.osa_controller = ConnectedOSA()
            window._controller = lambda: OpenSLM()
            for step in (1, 2, 3):
                window.pipeline_checks[step].setChecked(False)
            window.pipeline_checks[4].setChecked(True)

            launched = {}
            window._launch_calibration = (
                lambda label, work: launched.update(label=label, work=work)
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                calibration_path = root / "calibration.json"
                output_root = root / "optimization"
                levels = np.asarray([0, 1023])
                calibration = CalibrationResult(
                    wavelength=np.asarray([780.0, 778.0, 776.0]),
                    coordinates=np.asarray([0.0, 500.0, 1000.0]),
                    max_level=1023,
                    min_level=0,
                    level_range=levels,
                    intensity_levels=np.asarray(
                        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
                    ),
                    raw_intensity_levels=np.asarray(
                        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
                    ),
                )
                save_calibration_result(calibration, calibration_path)
                expected_initial = np.linspace(0.2, 0.9, 8)
                window.pipeline_input_edits[4].setText(str(calibration_path))
                window.pipeline_profile_values_edit.setText(
                    ", ".join(str(value) for value in expected_initial)
                )
                window.pipeline_optimization_root_edit.setText(str(output_root))
                window.pipeline_optimization_name_edit.setText("pipeline_test")

                def fake_optimize(layout, **kwargs):
                    np.testing.assert_allclose(kwargs["initial_l"], expected_initial)
                    self.assertEqual(layout.channel_width_px, 15)
                    self.assertEqual(Path(kwargs["config"].output_root), output_root)
                    run_dir = output_root / "pipeline_test"
                    run_dir.mkdir(parents=True)
                    (run_dir / "final_result.json").write_text("{}", encoding="utf-8")
                    final_profile = np.concatenate(
                        [expected_initial, expected_initial[-2::-1]]
                    )
                    return OptimizationResult(
                        initial_l=expected_initial.copy(),
                        stage1_l=expected_initial.copy(),
                        stage3_l=expected_initial.copy(),
                        final_l=expected_initial.copy(),
                        final_profile=final_profile,
                        final_luts={},
                        final_metrics={},
                        run_dir=str(run_dir),
                        accepted=True,
                    )

                with (
                    patch(
                        "slm_module.gui.app.optimize_from_osa",
                        side_effect=fake_optimize,
                    ),
                    patch(
                        "slm_module.gui.app.load_calibration_result",
                        wraps=real_load_calibration_result,
                    ) as load_result,
                ):
                    window._run_pipeline()
                    payload = launched["work"](
                        lambda _progress: None, threading.Event()
                    )

                self.assertEqual(launched["label"], "Run pipeline")
                self.assertEqual(load_result.call_count, 2)
                self.assertIsNotNone(payload["optimization_result"])
                self.assertEqual(payload["optimization_layout"].channel_width_px, 15)
                self.assertTrue(
                    Path(payload["optimization_result"].run_dir, "final_result.json").is_file()
                )

                window._on_step_finished(payload)
                self.assertIs(window._edge_optimization_result, payload["optimization_result"])
                self.assertTrue(window.enc_use_optimized_lut.isChecked())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
