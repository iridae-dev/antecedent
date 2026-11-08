"""Unit checks for the fixed, non-evidence Nix/PyO3 engineering fixture."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nix_python_engineering_fixture as fixture


class PythonFixtureTest(unittest.TestCase):
    def test_fixed_project_is_copied_and_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            fixture._prepare_fixture(root)
            self.assertIn('version = "=0.29.2"', (root / "Cargo.toml").read_text(encoding="utf-8"))
            self.assertIn("nix_python_boundary", (root / "src/lib.rs").read_text(encoding="utf-8"))
            self.assertTrue((root / ".git").is_dir())

    def test_inspector_checks_interpreter_graph_and_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "store-output"
            (output / "bin").mkdir(parents=True)
            (output / "lib").mkdir()
            python_bin = root / "python3.13"
            python_bin.write_text("fixture", encoding="utf-8")
            (output / "bin/python3.13").symlink_to(python_bin)
            source = "/nix/store/" + "1" * 32 + "-source"
            good_hash = "sha256-" + "A" * 43 + "="
            (output / "source-root").write_text(source + "\n", encoding="utf-8")
            (output / "rustc-version").write_text('release: 1.85.1\ncommit-hash: 4eb161250e340c8f48f66e2b929ef4a5bed7c181\n', encoding="utf-8")
            (output / "cargo-version").write_text("cargo 1.97.0 (fixture)\n", encoding="utf-8")
            (output / "python-config.json").write_text(json.dumps({"executable": str(python_bin), "implementation": "cpython", "version": "3.13.14"}), encoding="utf-8")
            (output / "lib/nix_python_boundary.so").write_bytes(b"fixture extension")
            (output / "cargo-metadata.json").write_text("{}", encoding="utf-8")
            (output / "cargo-unit-graph.json").write_text("{}", encoding="utf-8")
            graph = {"roots": [0], "units": [{"name": fixture.MODULE, "target_kind": ["cdylib"], "mode": "build"}]}
            with (
                patch.object(fixture, "STORE_ITEM", re.compile(r".*")),
                patch.object(fixture, "compiler_units", return_value=graph),
                patch.object(fixture, "_run", return_value=good_hash),
                patch.object(fixture.Path, "resolve", return_value=Path("/nix/store/fake/python")),
            ):
                # Test the Nix-path check using a synthetic store-shaped path.
                data = json.loads((output / "python-config.json").read_text(encoding="utf-8"))
                data["executable"] = "/nix/store/fake/python"
                (output / "python-config.json").write_text(json.dumps(data), encoding="utf-8")
                identity = fixture.inspect_output(output, source, good_hash)
                self.assertEqual(identity["python"]["version"], "3.13.14")
                self.assertEqual(identity["compiler_units"], 1)
                data["version"] = "3.12.0"
                (output / "python-config.json").write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "pinned Python"):
                    fixture.inspect_output(output, source, good_hash)

    def test_runner_never_attests_or_publishes_evidence(self):
        source = {"store_path": "/nix/store/source", "nar_hash": "nar", "commit": "abc"}
        identity = {"extension_sha256": "fixture", "compiler_units": 1}
        with patch.object(fixture, "_prepare_fixture"), \
             patch.object(fixture, "materialize_committed_source", return_value={"commit": "abc"}), \
             patch.object(fixture, "import_committed_snapshot", return_value=source), \
             patch.object(fixture, "verify_committed_snapshot") as verify, \
             patch.object(fixture, "_build", return_value="/nix/store/output"), \
             patch.object(fixture, "inspect_output", return_value=identity) as inspect, \
             patch.object(fixture, "_run") as execute:
            result = fixture.run_fixture()
        self.assertFalse(result["sandbox_verified"])
        self.assertFalse(result["evidence_published"])
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(inspect.call_count, 2)
        self.assertEqual(execute.call_args.args[0][2:4], ["-c", execute.call_args.args[0][3]])
        self.assertIn("nix_python_boundary", execute.call_args.args[0][3])


if __name__ == "__main__":
    unittest.main()
