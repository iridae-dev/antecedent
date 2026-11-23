import tempfile
import unittest
from pathlib import Path

import check_release_claims as release


class LiveCalibrationClaims(unittest.TestCase):
    def test_registered_coverage_backs_a_per_request_calibrated_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parity/release_snapshots").mkdir(parents=True)
            (root / "parity/capability_records.toml").write_text(
                '[[record]]\nid="r"\ninference_claim="interval"\ncoverage_records=["c"]\n'
                '[[record.routes]]\nstage="uncertainty"\nstatus="supported"\n', encoding="utf-8")
            (root / "parity/coverage_records.toml").write_text('[[record]]\nid="c"\n', encoding="utf-8")
            rows, present = release.live_projection(root)
            self.assertEqual(present, {"c"})
            self.assertEqual(release.effective_claim(rows[0], present), ("calibrated", False))
            entry = release.entry_text(rows[0], present)
            self.assertIn("Claim: `calibrated`", entry)
            self.assertIn("never withheld", entry)
            self.assertNotIn("`not_supported`", entry)

    def test_unmeasured_interval_is_returned_without_a_coverage_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parity/release_snapshots").mkdir(parents=True)
            (root / "parity/capability_records.toml").write_text(
                '[[record]]\nid="r"\ninference_claim="interval"\ncoverage_records=["missing"]\n'
                '[[record.routes]]\nstage="uncertainty"\nstatus="supported"\n', encoding="utf-8")
            (root / "parity/coverage_records.toml").write_text('[[record]]\nid="c"\n', encoding="utf-8")
            rows, present = release.live_projection(root)
            self.assertEqual(release.effective_claim(rows[0], present), ("point_only", True))
            entry = release.entry_text(rows[0], present)
            self.assertIn("`not_measured`", entry)
            self.assertNotIn("withheld", entry)
