"""Revision checks handle TOML syntax without treating string contents as keys."""
import tempfile
import unittest
from pathlib import Path

import check_v3_test_registry as gate


class EvidenceRevisions(unittest.TestCase):
    def test_toml_assignments_and_embedded_strings(self):
        head = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "evidence.toml"
            cases = [
                (f'  revision = "{head}" # note\n', True),
                (f"revision = '{head}' # note\n", True),
                (f"revision = '{'0'*40}' # stale\n", False),
                ("revision = 42\n", False),
                ("# v3-evidence\nother = 1\n", False),
                ("description = '''\nrevision = 'not a key'\n'''\n", True),
            ]
            for body, valid in cases:
                with self.subTest(body=body):
                    path.write_text(body, encoding="utf-8")
                    self.assertEqual(gate.evidence_problems(root, "fixture", "evidence.toml", head) == [], valid)


if __name__ == "__main__":
    unittest.main()
