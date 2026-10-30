from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import check_release_claims as release

class LiveCalibrationEligibility(unittest.TestCase):
    def test_present_id_does_not_prove_scope_but_snapshot_keeps_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parity/release_snapshots").mkdir(parents=True)
            (root / "parity/promotion.toml").write_text(
                '[[record]]\nid="r"\ninference_claim="calibrated"\ncoverage_records=["c"]\n'
                '[[record.routes]]\nstage="uncertainty"\nstatus="licensed"\n')
            (root / "parity/coverage_records.toml").write_text('[[record]]\nid="c"\n')
            rows, present = release.live_projection(root)
            self.assertEqual(release.effective_claim(rows[0], present), ("point_only", True))
            self.assertEqual(release.effective_claim(rows[0], {"c"}), ("calibrated", False))
