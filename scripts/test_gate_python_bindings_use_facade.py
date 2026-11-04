import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GATE = Path(__file__).with_name("gate_python_bindings_use_facade.py")
CLEAN_MANIFEST = '[package]\nname = "antecedent-py"\n\n[features]\nml-neural = ["antecedent/ml-neural"]\n\n[dependencies]\nantecedent = { workspace = true }\nserde.workspace = true\n'
CLEAN_SRC = (
    "use antecedent::prelude::Value;\n"
    "// antecedent_core::Value is only named in this comment\n"
    'const S: &str = "antecedent_core::Value";\n'
    'const A: &str = "_antecedent_callback_failure";\n'
)


def run_gate(manifest: str, src: str):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "python" / "src").mkdir(parents=True)
        (root / "python" / "Cargo.toml").write_text(manifest, encoding="utf-8")
        (root / "python" / "src" / "lib.rs").write_text(src, encoding="utf-8")
        return subprocess.run([sys.executable, str(GATE), "--root", tmp], capture_output=True, text=True)


class PythonBindingsUseFacade(unittest.TestCase):
    def test_facade_only_passes(self):
        result = run_gate(CLEAN_MANIFEST, CLEAN_SRC)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_internal_source_path_fails(self):
        result = run_gate(CLEAN_MANIFEST, CLEAN_SRC + "fn f() { let _ = antecedent_core::VERSION; }\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("antecedent_core", result.stderr)

    def test_internal_dependency_fails(self):
        result = run_gate(CLEAN_MANIFEST + "antecedent-core = { workspace = true }\n", CLEAN_SRC)
        self.assertEqual(result.returncode, 1)
        self.assertIn("antecedent-core", result.stderr)

    def test_internal_feature_forward_fails(self):
        manifest = CLEAN_MANIFEST.replace('["antecedent/ml-neural"]', '["antecedent-estimate/ml-neural"]')
        result = run_gate(manifest, CLEAN_SRC)
        self.assertEqual(result.returncode, 1)
        self.assertIn("antecedent-estimate", result.stderr)


if __name__ == "__main__":
    unittest.main()
