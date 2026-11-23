"""Generated snapshots preserve timestamps and use only live attestations."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_status
import generate_support_matrix_docs as generator

write_generated = generator.write_text  # the generator's own writer, not pathlib's


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
        self.assertEqual(write_generated(self.destination, content), len(content))
        self.assertEqual(self.destination.stat().st_mtime_ns, before)
        self.assertEqual(self.destination.read_bytes(), content.encode("utf-8"))

    def test_changed_content_is_written(self):
        self.destination.write_text("old\n", encoding="utf-8")
        before = self.pin_mtime(self.destination)
        self.assertEqual(write_generated(self.destination, "new\n"), 4)
        self.assertEqual(self.destination.read_bytes(), b"new\n")
        self.assertNotEqual(self.destination.stat().st_mtime_ns, before)

    def test_missing_output_is_created(self):
        write_generated(str(self.destination), "α\n")
        self.assertEqual(self.destination.read_bytes(), "α\n".encode())

    def test_line_ending_differences_are_not_treated_as_identical(self):
        self.destination.write_bytes(b"value\r\n")
        write_generated(self.destination, "value\n")
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
    """The snapshot's attesting ids are the live `current` set and nothing cached."""

    def test_live_current_set_is_the_only_authority(self):
        with patch.object(calibration_status, "current_ids", return_value=["cov.current"]):
            self.assertEqual(generator.attesting_record_ids(), ["cov.current"])

    def test_empty_live_set_attests_nothing(self):
        with patch.object(calibration_status, "current_ids", return_value=[]):
            self.assertEqual(generator.attesting_record_ids(), [])

    def test_unreadable_tree_attests_nothing(self):
        with patch.object(calibration_status, "current_ids", side_effect=OSError("no cargo")):
            self.assertEqual(generator.attesting_record_ids(), [])


if __name__ == "__main__":
    unittest.main()
