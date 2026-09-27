"""Focused guardrails for incremental coverage-record collection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_coverage_records as collector  # noqa: E402


class AppendAttestedRecordsTest(unittest.TestCase):
    def test_append_preserves_old_rows_and_original_sha(self) -> None:
        old = {"old": {"id": "old", "calibration_sha": "old-sha", "observed": 0.8}}
        new = {"new": {"id": "new", "calibration_sha": "new-sha", "observed": 0.9}}
        merged = collector.append_attested_records(old, new)
        self.assertEqual(set(merged), {"old", "new"})
        self.assertEqual(merged["old"], old["old"])
        self.assertEqual(merged["new"], new["new"])
        self.assertEqual(set(old), {"old"})

    def test_append_refuses_collision(self) -> None:
        with self.assertRaisesRegex(SystemExit, "refuses existing record ids"):
            collector.append_attested_records({"same": {}}, {"same": {}})

    def test_selected_cell_sync_leaves_other_cells_byte_identical(self) -> None:
        source = (
            '[[cell]]\nquery = "Elasticity"\ngraph_class = "Dag"\n'
            'structure = "explicit"\ninference = "Frequentist"\nvalidation = "none"\n'
            'calibration_reason = "estimator_grid_not_measured"\n\n'
            '[[cell]]\nquery = "Other"\ngraph_class = "Dag"\n'
            'structure = "explicit"\ninference = "Frequentist"\nvalidation = "none"\n'
            'calibration = ["old"]\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "licensed.toml"
            path.write_text(source)
            original = collector.LICENSED
            collector.LICENSED = path
            try:
                changed = collector.sync_licensed_cells(
                    {
                        "new": {
                            "query": "Elasticity",
                            "graph_class": "Dag",
                            "inference": "Frequentist",
                            "structure": "fixed",
                            "nominal": 0.95,
                            "boundary": False,
                        }
                    },
                    {("Elasticity", "Dag", "explicit", "Frequentist", "none"): ["new"]},
                )
            finally:
                collector.LICENSED = original
            self.assertEqual(changed, 1)
            self.assertIn('calibration = ["new"]', path.read_text())
            self.assertEqual(
                path.read_text().split('[[cell]]')[2],
                source.split('[[cell]]')[2],
            )


if __name__ == "__main__":
    unittest.main()
