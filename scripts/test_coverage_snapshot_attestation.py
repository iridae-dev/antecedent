"""Generated snapshots preserve timestamps and use only live attestations."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_facets
import generate_support_matrix_docs as generator


class GeneratedWriteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name) / "generated.txt"

    def pin_mtime(self, path):
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        return path.stat().st_mtime_ns

    def test_identical_utf8_preserves_mtime_and_character_count(self):
        content = "known truth: α\n"
        self.destination.write_bytes(content.encode("utf-8"))
        before = self.pin_mtime(self.destination)
        self.assertEqual(generator.write_text(self.destination, content), len(content))
        self.assertEqual(self.destination.stat().st_mtime_ns, before)
        self.assertEqual(self.destination.read_bytes(), content.encode("utf-8"))

    def test_changed_content_is_written(self):
        self.destination.write_text("old\n", encoding="utf-8")
        before = self.pin_mtime(self.destination)
        self.assertEqual(generator.write_text(self.destination, "new\n"), 4)
        self.assertEqual(self.destination.read_bytes(), b"new\n")
        self.assertNotEqual(self.destination.stat().st_mtime_ns, before)

    def test_missing_output_is_created(self):
        generator.write_text(str(self.destination), "α\n")
        self.assertEqual(self.destination.read_bytes(), "α\n".encode())

    def test_line_ending_differences_are_not_treated_as_identical(self):
        self.destination.write_bytes(b"value\r\n")
        generator.write_text(self.destination, "value\n")
        self.assertEqual(self.destination.read_bytes(), b"value\n")

    def test_snapshot_format_is_deterministic_and_unchanged_mtime_is_preserved(self):
        snapshot = Path(self.temp.name) / "nested" / "snapshot.json"
        payload = {"label": "α", "values": [1, None]}
        expected = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        generator.write_snapshot(snapshot, payload)
        self.assertEqual(snapshot.read_bytes(), expected)
        before = self.pin_mtime(snapshot)
        generator.write_snapshot(snapshot, payload)
        self.assertEqual(snapshot.stat().st_mtime_ns, before)
        generator.write_snapshot(snapshot, {**payload, "label": "changed"})
        self.assertEqual(json.loads(snapshot.read_bytes()), {**payload, "label": "changed"})
        self.assertNotEqual(snapshot.stat().st_mtime_ns, before)


class SnapshotAttestationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.snapshot = Path(self.temp.name) / "coverage_records.json"
        self.snapshot.write_text(
            json.dumps(
                {"records": [{"id": "cov.old"}], "attesting_record_ids": ["cov.old"]}
            ),
            encoding="utf-8",
        )

    def test_empty_live_assessment_invalidates_matching_cached_ids(self):
        with (
            patch.object(generator, "COVERAGE_SNAPSHOT", self.snapshot),
            patch.object(generator, "load", return_value={"record": [{"id": "cov.old"}]}),
            patch.object(generator, "_live_attesting_record_ids", return_value=[]),
        ):
            self.assertEqual(generator.attesting_record_ids(), [])

    def test_nonempty_live_assessment_is_the_only_authority(self):
        with (
            patch.object(generator, "COVERAGE_SNAPSHOT", self.snapshot),
            patch.object(generator, "load", return_value={"record": [{"id": "cov.old"}]}),
            patch.object(generator, "_live_attesting_record_ids", return_value=["cov.current"]),
        ):
            self.assertEqual(generator.attesting_record_ids(), ["cov.current"])

    def test_unresolvable_surface_does_not_restore_cached_ids(self):
        with (
            patch.object(generator, "COVERAGE_SNAPSHOT", self.snapshot),
            patch.object(generator, "load", return_value={"record": [{"id": "cov.old"}]}),
            patch.object(calibration_facets, "load_records", return_value={}),
            patch.object(calibration_facets, "load_surface", return_value=object()),
            patch.object(calibration_facets, "assess", side_effect=OSError("unresolvable Git surface")),
        ):
            self.assertEqual(generator.attesting_record_ids(), [])


if __name__ == "__main__":
    unittest.main()
