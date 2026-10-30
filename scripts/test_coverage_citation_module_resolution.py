"""Explicit missing test paths and a vacuous citation scan fail closed."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GATE = Path(__file__).with_name("gate_coverage_citations.py").resolve()


class CoverageModuleResolution(unittest.TestCase):
    def run_gate(self, limitations):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parity").mkdir()
            (root / "crates/antecedent/tests").mkdir(parents=True)
            (root / "crates/antecedent/tests/existing.rs").write_text("#[test]\nfn present() {}\n", encoding="utf-8")
            (root / "parity/support_licensed.toml").write_text(
                '[[cell]]\nquery="q"\ngraph_class="g"\nstructure="s"\ninference="i"\nvalidation="v"\nlimitations=' + repr(limitations) + '\n', encoding="utf-8")
            return subprocess.run([sys.executable, str(GATE)], cwd=root, capture_output=True, text=True, check=False)

    def test_deleted_explicit_module_and_vacuous_scan(self):
        missing = self.run_gate("coverage test crates/antecedent/tests/deleted.rs::case")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("deleted.rs does not exist", missing.stdout)
        none = self.run_gate("documentation mentions std::vec")
        self.assertNotEqual(none.returncode, 0)
        self.assertIn("no coverage test citations were checked", none.stdout)
        present = self.run_gate("coverage test existing::present")
        self.assertEqual(present.returncode, 0)


if __name__ == "__main__":
    unittest.main()
