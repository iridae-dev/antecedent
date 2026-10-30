"""The provenance gate refuses vacuous source and review scans."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GATE = Path(__file__).with_name("gate_provenance_schema.py").resolve()


class ProvenanceNonempty(unittest.TestCase):
    def run_gate(self, factory):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "crates/antecedent/src/strategy_table/ids.rs": "// no pointers\n",
                "crates/antecedent-stats/src/ci/factory.rs": factory,
                "parity/method_children/children.toml": "child = []\n",
                "parity/method_specs/specs.toml": "spec = []\n",
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            return subprocess.run([sys.executable, str(GATE)], cwd=root, capture_output=True, text=True, check=False)

    def test_missing_definition_and_empty_collections_fail(self):
        result = self.run_gate("// renamed\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pointer scan found no pointers", result.stdout)
        self.assertIn("ci_from_name definition is missing", result.stdout)
        self.assertIn("no admitted method", result.stdout)
        result = self.run_gate("pub fn ci_from_name() {}\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ci_from_name contains no names", result.stdout)


if __name__ == "__main__":
    unittest.main()
