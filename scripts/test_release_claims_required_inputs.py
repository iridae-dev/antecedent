"""Current release projection refuses missing source registries."""
import tempfile
import unittest
from pathlib import Path

import check_release_claims as gate


class RequiredReleaseInputs(unittest.TestCase):
    def test_current_inputs_are_required_but_historical_snapshot_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parity").mkdir()
            paths = [gate.RECORDS, gate.COVERAGE, "parity/support_axes.toml", "parity/support_licensed.toml"]
            for rel in paths:
                (root / rel).write_text("", encoding="utf-8")
            self.assertEqual(gate.live_projection(root), ([], set()))
            self.assertEqual(gate.zero_cell_queries(root), [])
            for rel in paths:
                path = root / rel
                path.unlink()
                with self.subTest(missing=rel), self.assertRaises(ValueError):
                    gate.source_for(root, "9.9.9", "9.9.9")
                path.write_text("", encoding="utf-8")
            self.assertIsNone(gate.source_for(root, "1.0.0", "9.9.9"))


if __name__ == "__main__":
    unittest.main()
