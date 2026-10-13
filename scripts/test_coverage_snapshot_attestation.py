"""Coverage snapshots must not revive a cached attestation after surface drift."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_facets
import generate_support_matrix_docs as generator


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
