import tempfile
import unittest
from pathlib import Path
import check_release_claims as gate

class FrozenSnapshot(unittest.TestCase):
    def test_historical_snapshot_uses_frozen_coverage_not_live_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "parity/release_snapshots"
            folder.mkdir(parents=True)
            (folder / "v1.0.0.toml").write_text(
                'coverage_present=["c"]\nzero_cell_queries=[]\n'
                '[[record]]\nid="r"\ninference_claim="calibrated"\ncoverage_records=["c"]\n'
                '[[record.routes]]\nstage="uncertainty"\nstatus="licensed"\n')
            rows, present, queries = gate.source_for(root, "1.0.0", "2.0.0")
            self.assertEqual(queries, [])
            self.assertEqual(gate.effective_claim(rows[0], present), ("calibrated", False))
