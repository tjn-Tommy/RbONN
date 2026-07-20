from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_module.controller import validate_slm_csv
from slm_module.generator import (
    Segment,
    equal_segment_edges,
    export_center_scan_sequence,
    generate_center_scan,
    iter_center_scan_positions,
    make_equal_segments,
    make_vertical_window,
    make_segments,
    read_santec_csv,
    write_santec_csv,
)


class ReadSantecCsvTests(unittest.TestCase):
    def test_round_trips_written_pattern(self) -> None:
        data = make_vertical_window(
            width=10, height=4, x_start=3, level=700, window_px=2, background_level=120
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_santec_csv(data, Path(temp_dir) / "p.csv")
            loaded = read_santec_csv(path)
        self.assertEqual(loaded.dtype, np.uint16)
        self.assertEqual(loaded.shape, data.shape)
        self.assertTrue(np.array_equal(loaded, data))

    def test_rejects_out_of_range_grayscale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.csv"
            path.write_text("y/x,0,1\n0,5000,3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_santec_csv(path)


class PhaseGeneratorTests(unittest.TestCase):
    def test_make_vertical_window_places_level_and_clips_edge(self) -> None:
        data = make_vertical_window(width=8, height=3, x_start=6, level=700, window_px=5)

        self.assertEqual(data.shape, (3, 8))
        self.assertEqual(data.dtype, np.uint16)
        self.assertTrue(np.all(data[:, :6] == 0))
        self.assertTrue(np.all(data[:, 6:8] == 700))

    def test_make_vertical_window_uses_tunable_background(self) -> None:
        data = make_vertical_window(
            width=8, height=3, x_start=2, level=700, window_px=3, background_level=120
        )

        self.assertTrue(np.all(data[:, 2:5] == 700))
        self.assertTrue(np.all(data[:, :2] == 120))
        self.assertTrue(np.all(data[:, 5:] == 120))
        with self.assertRaises(ValueError):
            make_vertical_window(8, 3, 2, 700, 3, background_level=2000)

    def test_scan_positions_are_inclusive_start_positions(self) -> None:
        positions = list(iter_center_scan_positions(20, step_px=5, start_x=0, end_x=12))

        self.assertEqual(positions, [0, 5, 10])

    def test_generate_center_scan_returns_patterns(self) -> None:
        patterns = list(
            generate_center_scan(
                width=12,
                height=2,
                level=512,
                window_px=5,
                step_px=5,
                start_x=0,
                end_x=10,
            )
        )

        self.assertEqual([pattern.x_start for pattern in patterns], [0, 5, 10])
        self.assertEqual(patterns[-1].x_end, 12)

    def test_write_santec_csv_matches_validator(self) -> None:
        data = make_vertical_window(width=6, height=4, x_start=2, level=1023, window_px=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_santec_csv(data, Path(temp_dir) / "mask.csv")
            size = validate_slm_csv(path, expected_width=6, expected_height=4)

        self.assertEqual(size, (6, 4))

    def test_export_center_scan_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = export_center_scan_sequence(
                temp_dir,
                width=10,
                height=3,
                level=100,
                window_px=5,
                step_px=5,
            )

            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertTrue(path.exists())
                validate_slm_csv(path, expected_width=10, expected_height=3)


class SegmentTests(unittest.TestCase):
    def test_equal_edges_cover_full_size(self) -> None:
        self.assertEqual(equal_segment_edges(10, 3), [0, 3, 7, 10])
        self.assertEqual(equal_segment_edges(9, 3), [0, 3, 6, 9])

    def test_make_equal_segments_assigns_levels_per_part(self) -> None:
        data = make_equal_segments(width=10, height=2, levels=[100, 200, 300])

        self.assertEqual(data.shape, (2, 10))
        self.assertEqual(data.dtype, np.uint16)
        self.assertTrue(np.all(data[:, 0:3] == 100))
        self.assertTrue(np.all(data[:, 3:7] == 200))
        self.assertTrue(np.all(data[:, 7:10] == 300))

    def test_make_equal_segments_along_y_assigns_levels_per_part(self) -> None:
        data = make_equal_segments(width=2, height=10, levels=[100, 200, 300], axis="y")

        self.assertEqual(data.shape, (10, 2))
        self.assertTrue(np.all(data[0:3, :] == 100))
        self.assertTrue(np.all(data[3:7, :] == 200))
        self.assertTrue(np.all(data[7:10, :] == 300))

    def test_make_equal_segments_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            make_equal_segments(width=10, height=2, levels=[])
        with self.assertRaises(ValueError):
            make_equal_segments(width=3, height=2, levels=[1, 2, 3, 4])
        with self.assertRaises(ValueError):
            make_equal_segments(width=10, height=2, levels=[2000])
        with self.assertRaises(ValueError):
            make_equal_segments(width=10, height=3, levels=[1, 2, 3, 4], axis="y")

    def test_make_segments_places_bands_and_background(self) -> None:
        data = make_segments(
            width=10,
            height=3,
            segments=[Segment(0, 4, 100), (6, 10, 900)],
            background_level=50,
        )

        self.assertTrue(np.all(data[:, 0:4] == 100))
        self.assertTrue(np.all(data[:, 4:6] == 50))
        self.assertTrue(np.all(data[:, 6:10] == 900))

    def test_make_segments_along_y_places_bands_and_background(self) -> None:
        data = make_segments(
            width=3,
            height=10,
            segments=[Segment(0, 4, 100), (6, 10, 900)],
            axis="y",
            background_level=50,
        )

        self.assertTrue(np.all(data[0:4, :] == 100))
        self.assertTrue(np.all(data[4:6, :] == 50))
        self.assertTrue(np.all(data[6:10, :] == 900))

    def test_make_segments_rejects_overlap_and_bounds(self) -> None:
        with self.assertRaises(ValueError):
            make_segments(10, 2, [(0, 5, 100), (4, 8, 200)])
        with self.assertRaises(ValueError):
            make_segments(10, 2, [(5, 5, 100)])
        with self.assertRaises(ValueError):
            make_segments(10, 2, [(0, 11, 100)])
        with self.assertRaises(ValueError):
            make_segments(10, 2, [])
        with self.assertRaises(ValueError):
            make_segments(10, 2, [(0, 3, 100)], axis="y")
        with self.assertRaises(ValueError):
            make_segments(10, 2, [(0, 3, 100)], axis="diagonal")

    def test_segments_round_trip_through_csv(self) -> None:
        data = make_equal_segments(width=8, height=3, levels=[0, 512, 1023])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_santec_csv(data, Path(temp_dir) / "segments.csv")
            size = validate_slm_csv(path, expected_width=8, expected_height=3)

        self.assertEqual(size, (8, 3))


if __name__ == "__main__":
    unittest.main()
