"""Parity evidence cannot hide duplicate rows or empty exit artifacts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import parity_rows as gate


class ParityRowsIntegrity(unittest.TestCase):
    def test_duplicate_pending_and_done_rows_fail_both_lookup_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rows.toml").write_text('[[capabilities]]\nid="x"\nstatus="pending"\n[[capabilities]]\nid="x"\nstatus="done"\n', encoding="utf-8")
            (root / "evidence.txt").write_text("evidence", encoding="utf-8")
            with patch.object(gate, "ROOT", root):
                self.assertTrue(any("duplicate" in p for p in gate.evidence_map_problems({"x": "evidence.txt"}, ["rows.toml"])))
                self.assertTrue(any("duplicate" in p for p in gate.require_done("rows.toml", ["x"], "gate")))

    def test_exit_artifacts_are_nonempty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty").write_text("", encoding="utf-8")
            (root / "directory").mkdir()
            (root / "present").write_text("ok", encoding="utf-8")
            with patch.object(gate, "ROOT", root):
                self.assertEqual(len(gate.exit_artifact_problems(["empty", "directory", "present"])), 2)


if __name__ == "__main__":
    unittest.main()
