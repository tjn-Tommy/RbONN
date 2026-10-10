"""The one calib.json: sections, stamps, and what a write drops."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration_module.calib_file import (  # noqa: E402
    CALIB_NAME,
    SCHEMA,
    STEP_ORDER,
    CalibFileError,
    CalibNotFound,
    SchemaMismatch,
    SectionMissing,
    calib_path,
    load_calib,
    step_key,
    step_number,
    write_section,
)


class _TempRun(unittest.TestCase):
    """A throwaway folder with one calib.json per test."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="rbonn_calib_"))
        self.run = self.tmp / "run_0931"
        self.run.mkdir()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def payload(self) -> dict:
        return json.loads((self.run / CALIB_NAME).read_text(encoding="utf-8"))

    def quiet(self, *args, **kwargs):
        kwargs.setdefault("log", None)
        return write_section(*args, **kwargs)


class StepKeyTests(unittest.TestCase):
    def test_accepts_int_str_and_prefixed(self) -> None:
        for given in (3, "3", "step3", "STEP3", " step3 "):
            self.assertEqual(step_key(given), "step3")
            self.assertEqual(step_number(given), 3)

    def test_rejects_a_step_that_is_not_one(self) -> None:
        # 4 and 5 never existed; 3b/3c collapsed into 3.
        for given in (4, 5, "3c", "nine", 0):
            with self.assertRaises(CalibFileError):
                step_key(given)

    def test_order_is_the_dependency_order(self) -> None:
        self.assertEqual(STEP_ORDER, (1, 2, 3, 6, 7, 8))


class PathTests(_TempRun):
    def test_directory_resolves_to_its_calib_json(self) -> None:
        self.assertEqual(calib_path(self.run), self.run / CALIB_NAME)

    def test_named_file_is_used_as_is(self) -> None:
        named = self.run / "calib.json"
        self.assertEqual(calib_path(named), named)

    def test_a_variant_name_in_the_same_folder_is_its_own_calibration(self) -> None:
        # The file is the unit, so two of them share a folder and its CSVs
        # without sharing a single section.
        variant = self.run / "calib_noYsat.json"
        self.assertEqual(calib_path(variant), variant)
        self.quiet(variant, 3, {"marker": "variant"})
        self.quiet(self.run, 3, {"marker": "main"})
        self.assertEqual(load_calib(variant).section(3)["marker"], "variant")
        self.assertEqual(load_calib(self.run).section(3)["marker"], "main")

    def test_beside_names_a_sibling_of_the_file(self) -> None:
        self.quiet(self.run, 3, {"marker": "x"})
        calib = load_calib(self.run)
        self.assertEqual(calib.folder, self.run)
        self.assertEqual(calib.beside("step3_0931_1802.csv"),
                         self.run / "step3_0931_1802.csv")


class LoadTests(_TempRun):
    def test_missing_file_raises_with_the_way_out(self) -> None:
        with self.assertRaises(CalibNotFound) as caught:
            load_calib(self.run)
        self.assertIn("create it with step 1", str(caught.exception))

    def test_missing_ok_gives_an_empty_one(self) -> None:
        calib = load_calib(self.run, missing_ok=True)
        self.assertEqual(calib.steps(), [])
        self.assertEqual(calib.folder, self.run)

    def test_pre_v3_result_is_refused_by_name(self) -> None:
        # What every existing run directory holds: a nested per-step result.
        (self.run / CALIB_NAME).write_text(
            json.dumps({"step3": {"schema": "calibration_result_v2"}, "step6": {}}),
            encoding="utf-8")
        with self.assertRaises(SchemaMismatch) as caught:
            load_calib(self.run)
        self.assertIn("migrate_calib_v3", str(caught.exception))

    def test_garbage_is_not_a_traceback(self) -> None:
        (self.run / CALIB_NAME).write_text("{not json", encoding="utf-8")
        with self.assertRaises(CalibFileError):
            load_calib(self.run)


class WriteTests(_TempRun):
    def test_section_is_stamped_and_schema_recorded(self) -> None:
        self.quiet(self.run, 3, {"channel_width_px": 20, "gap_px": 30})
        payload = self.payload()
        self.assertEqual(payload["schema"], SCHEMA)
        self.assertEqual(payload["step3"]["channel_width_px"], 20)
        self.assertRegex(payload["step3"]["ran"],
                         r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

    def test_ran_can_be_preserved(self) -> None:
        self.quiet(self.run, 2, {"min_level": 390}, ran="2026-09-28T17:44:00")
        self.assertEqual(self.payload()["step2"]["ran"], "2026-09-28T17:44:00")

    def test_sections_accumulate_in_one_file(self) -> None:
        self.quiet(self.run, 2, {"min_level": 390})
        self.quiet(self.run, 3, {"gap_px": 30})
        self.quiet(self.run, 6, {"channels": []})
        self.assertEqual(load_calib(self.run).steps(), [2, 3, 6])
        self.assertEqual(len(list(self.run.glob("*.json"))), 1)

    def test_keys_are_written_in_step_order(self) -> None:
        """A hand-assembled file comes back sorted, whatever order it was in."""
        (self.run / CALIB_NAME).write_text(json.dumps({
            "step6": {"ran": "2026-09-29T22:01:00"},
            "schema": SCHEMA,
            "step3": {"ran": "2026-09-29T18:02:00"},
        }), encoding="utf-8")
        self.quiet(self.run, 8, {"y_sat_v": 2.4})     # drops nothing
        self.assertEqual(list(self.payload()),
                         ["schema", "step3", "step6", "step8"])

    def test_caller_payload_is_not_mutated(self) -> None:
        given = {"gap_px": 30}
        self.quiet(self.run, 3, given)
        self.assertEqual(given, {"gap_px": 30})

    def test_rewriting_a_step_replaces_it_whole(self) -> None:
        """Replaced, not merged: a key the new fit dropped must not linger."""
        self.quiet(self.run, 3, {"gap_px": 30, "channel_width_px": 20})
        self.quiet(self.run, 3, {"gap_px": 5})
        section = self.payload()["step3"]
        self.assertEqual(section["gap_px"], 5)
        self.assertNotIn("channel_width_px", section)

    def test_no_stray_tmp_file_left_behind(self) -> None:
        self.quiet(self.run, 3, {"gap_px": 30})
        self.assertEqual([p.name for p in self.run.iterdir()], [CALIB_NAME])


class DownstreamDropTests(_TempRun):
    def _full_chain(self) -> None:
        for step in (2, 3, 6, 7, 8):
            self.quiet(self.run, step, {"marker": step}, ran=f"2026-09-2{step}T10:00:00")

    def test_new_step3_drops_6_7_8_and_keeps_2(self) -> None:
        self._full_chain()
        self.quiet(self.run, 3, {"marker": "fresh"})
        self.assertEqual(load_calib(self.run).steps(), [2, 3])

    def test_new_step7_keeps_3_and_6(self) -> None:
        self._full_chain()
        self.quiet(self.run, 7, {"marker": "fresh"})
        self.assertEqual(load_calib(self.run).steps(), [2, 3, 6, 7])

    def test_step8_drops_nothing(self) -> None:
        self._full_chain()
        self.quiet(self.run, 8, {"marker": "fresh"})
        self.assertEqual(load_calib(self.run).steps(), [2, 3, 6, 7, 8])

    def test_what_was_dropped_is_named(self) -> None:
        self._full_chain()
        said: list[str] = []
        write_section(self.run, 3, {"marker": "fresh"}, log=said.append)
        message = "\n".join(said)
        for name in ("step6", "step7", "step8"):
            self.assertIn(name, message)
        self.assertIn("previous step3", message)

    def test_a_plain_write_says_where_it_went(self) -> None:
        said: list[str] = []
        write_section(self.run, 3, {"gap_px": 30}, log=said.append)
        self.assertIn(CALIB_NAME, "\n".join(said))


class CsvPointerTests(_TempRun):
    def test_stored_relative_and_resolved_back(self) -> None:
        csv = self.run / "step3_0931_1802.csv"
        csv.write_text("x\n", encoding="utf-8")
        self.quiet(self.run, 3, {"gap_px": 30}, csv=csv)
        self.assertEqual(self.payload()["step3"]["csv"], "step3_0931_1802.csv")
        self.assertEqual(load_calib(self.run).csv_path(3), csv.resolve())

    def test_a_foreign_csv_is_stored_as_the_hop_it_is(self) -> None:
        other = self.tmp / "run_0930"
        other.mkdir()
        csv = other / "step6_meas_0930_2201.csv"
        csv.write_text("x\n", encoding="utf-8")
        self.quiet(self.run, 6, {"channels": []}, csv=csv)
        self.assertEqual(self.payload()["step6"]["csv"],
                         "../run_0930/step6_meas_0930_2201.csv")
        self.assertEqual(load_calib(self.run).csv_path(6), csv.resolve())

    def test_several_csvs_stay_a_list(self) -> None:
        # run_0916 merged two step-7 sweeps into one fit.
        names = ["step7_meas_a.csv", "step7_meas_b.csv"]
        for name in names:
            (self.run / name).write_text("x\n", encoding="utf-8")
        self.quiet(self.run, 7, {"ref_index": 2}, csv=[self.run / n for n in names])
        self.assertEqual(self.payload()["step7"]["csv"], names)
        self.assertEqual(len(load_calib(self.run).csv_paths(7)), 2)
        with self.assertRaises(CalibFileError):
            load_calib(self.run).csv_path(7)

    def test_no_csv_means_no_key(self) -> None:
        self.quiet(self.run, 3, {"gap_px": 30})
        self.assertNotIn("csv", self.payload()["step3"])
        self.assertEqual(load_calib(self.run).csv_paths(3), [])
        with self.assertRaises(SectionMissing):
            load_calib(self.run).csv_path(3)

    def test_the_directory_survives_being_copied(self) -> None:
        """Inheriting is a folder copy, so the pointers cannot be absolute."""
        (self.run / "step3_0931_1802.csv").write_text("x\n", encoding="utf-8")
        self.quiet(self.run, 3, {"gap_px": 30}, csv=self.run / "step3_0931_1802.csv")
        moved = self.tmp / "run_0932"
        shutil.copytree(self.run, moved)
        self.assertEqual(load_calib(moved).csv_path(3),
                         (moved / "step3_0931_1802.csv").resolve())

    def test_a_left_behind_csv_does_not_break_the_load(self) -> None:
        self.quiet(self.run, 3, {"gap_px": 30}, csv=self.run / "gone.csv")
        self.assertFalse(load_calib(self.run).csv_path(3).exists())


class SectionAccessTests(_TempRun):
    def test_missing_section_says_what_to_run(self) -> None:
        self.quiet(self.run, 2, {"min_level": 390})
        with self.assertRaises(SectionMissing) as caught:
            load_calib(self.run).section(3)
        message = str(caught.exception)
        self.assertIn("run step 3", message)
        self.assertIn("step2", message)          # what it does have

    def test_require_raises_on_the_first_gap(self) -> None:
        self.quiet(self.run, 3, {"gap_px": 30})
        calib = load_calib(self.run)
        calib.require(3)
        with self.assertRaises(SectionMissing):
            calib.require(3, 6)

    def test_contains_and_get(self) -> None:
        self.quiet(self.run, 3, {"gap_px": 30})
        calib = load_calib(self.run)
        self.assertIn(3, calib)
        self.assertNotIn(6, calib)
        self.assertIsNone(calib.get(6))


class StaleTests(_TempRun):
    def test_a_written_chain_is_never_stale(self) -> None:
        for step in (2, 3, 6):
            self.quiet(self.run, step, {"marker": step})
        self.assertEqual(load_calib(self.run).stale(), {})

    def test_a_hand_assembled_file_is_caught(self) -> None:
        """What a folder copy plus an edited-in step 3 would leave behind."""
        (self.run / CALIB_NAME).write_text(json.dumps({
            "schema": SCHEMA,
            "step2": {"ran": "2026-09-28T17:44:00"},
            "step3": {"ran": "2026-09-30T18:02:00"},
            "step6": {"ran": "2026-09-29T22:01:00"},
        }), encoding="utf-8")
        calib = load_calib(self.run)
        self.assertEqual(calib.stale(), {6: [3]})
        chain = "\n".join(calib.chain_lines())
        self.assertIn("older than step3", chain)
        self.assertIn("2026-09-29 22:01", chain)

    def test_an_unstamped_section_is_not_guessed_at(self) -> None:
        (self.run / CALIB_NAME).write_text(json.dumps({
            "schema": SCHEMA,
            "step3": {"ran": "2026-09-30T18:02:00"},
            "step6": {},
        }), encoding="utf-8")
        self.assertEqual(load_calib(self.run).stale(), {})

    def test_chain_of_an_empty_file(self) -> None:
        self.assertIn("(empty)",
                      "\n".join(load_calib(self.run, missing_ok=True).chain_lines()))


if __name__ == "__main__":
    unittest.main()
