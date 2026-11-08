"""Checks for the fixed, non-evidence Nix engineering fixture runner."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nix_evidence_engineering_fixture as fixture


class EngineeringFixtureTest(unittest.TestCase):
    def test_synthetic_project_is_fixed_and_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            fixture._prepare_fixture(root)
            self.assertEqual((root / "crates" / fixture.CRATE / "src/lib.rs").read_text(encoding="utf-8"), fixture.SOURCE)
            self.assertIn('default-members = ["crates/default_fixture"]', (root / "Cargo.toml").read_text(encoding="utf-8"))
            self.assertIn("guarded = []", (root / "crates" / fixture.CRATE / "Cargo.toml").read_text(encoding="utf-8"))
            self.assertTrue((root / ".git").is_dir())

    def test_runner_verifies_identity_and_never_attests_evidence(self):
        source = {"store_path": "/nix/store/source", "nar_hash": "nar", "commit": "abc"}
        artifact = {"schema": 1, "source": source["store_path"], "source_nar_hash": "nar",
                    "sandbox_verified": False, "binary_sha256": "binary", "output": "/nix/store/output",
                    "binary": f"targets/{fixture.CRATE}/lib/lib/bin/test",
                    "output_nar_hash": "out-nar", "derivation": "drv", "recipe_sha256": {"flake.lock": "hash"},
                    "crate": fixture.CRATE, "target_kind": "lib", "target_name": "", "features": fixture.FEATURES}
        with patch.object(fixture, "_prepare_fixture") as prepare, \
             patch.object(fixture, "materialize_committed_source", return_value={"commit": "abc"}), \
             patch.object(fixture, "import_committed_snapshot", return_value=source), \
             patch.object(fixture, "verify_committed_snapshot") as verify, \
             patch.object(fixture, "_build", return_value="/nix/store/output") as build, \
             patch.object(fixture, "inspect_artifact", return_value=artifact), \
             patch.object(fixture, "_tool_versions", return_value={"rustc": "1.85.1", "rustc_commit": "hash", "cargo": "cargo 1.97.0"}), \
             patch.object(fixture, "_compiler_graph", return_value={"units": [{}], "roots": [0]}), \
             patch.object(fixture, "_check_metadata_features"), \
             patch("nix_manifest_binding.verify_committed_snapshot") as binding_verify, \
             patch("nix_manifest_binding.recheck_artifact") as recheck, \
             patch.object(fixture, "_execute_and_measure") as execute:
            result = fixture.run_fixture()
        self.assertEqual(result["fixture"], "passed")
        self.assertFalse(result["sandbox_verified"])
        self.assertFalse(result["evidence_published"])
        prepare.assert_called_once()
        build.assert_called_once_with(source["store_path"])
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(binding_verify.call_count, 2)
        self.assertEqual(recheck.call_count, 2)
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0], Path(f'/nix/store/output/targets/{fixture.CRATE}/lib/lib/bin/test'))

    def test_mismatched_source_or_claimed_sandbox_refuses_execution(self):
        source = {"store_path": "/nix/store/source", "nar_hash": "nar", "commit": "abc"}
        for artifact in (
            {"source_nar_hash": "other", "sandbox_verified": False, "output": "/nix/store/output"},
            {"source_nar_hash": "nar", "sandbox_verified": True, "output": "/nix/store/output"},
        ):
            with self.subTest(artifact=artifact), \
                 patch.object(fixture, "_prepare_fixture"), \
                 patch.object(fixture, "materialize_committed_source", return_value={"commit": "abc"}), \
                 patch.object(fixture, "import_committed_snapshot", return_value=source), \
                 patch.object(fixture, "verify_committed_snapshot"), \
                 patch.object(fixture, "_build", return_value="/nix/store/output"), \
                 patch.object(fixture, "inspect_artifact", return_value=artifact), \
                 patch.object(fixture, "_execute_and_measure") as execute:
                with self.assertRaises(ValueError):
                    fixture.run_fixture()
                execute.assert_not_called()

    def test_tool_versions_require_the_compiler_pin_and_record_cargo(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "rustc-version").write_text(
                "release: 1.85.1\ncommit-hash: 4eb161250e340c8f48f66e2b929ef4a5bed7c181\n", encoding="utf-8")
            (output / "cargo-version").write_text("cargo 1.97.0 (fixture)\n", encoding="utf-8")
            versions = fixture._tool_versions(output)
            self.assertEqual(versions["rustc"], "1.85.1")
            self.assertEqual(versions["cargo"], "cargo 1.97.0 (fixture)")
            (output / "rustc-version").write_text("release: 1.85.1\ncommit-hash: changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "compiler"):
                fixture._tool_versions(output)

    def test_metadata_confirms_feature_on_nondefault_member(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = {"packages": [{"name": fixture.CRATE, "id": "selected"}],
                        "workspace_default_members": ["other"],
                        "resolve": {"nodes": [{"id": "selected", "features": ["guarded"]}]}}
            path = output / "targets" / fixture.CRATE / "lib/lib/cargo-metadata.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(metadata), encoding="utf-8")
            fixture._check_metadata_features(output)
            metadata["resolve"]["nodes"][0]["features"] = []
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not activate"):
                fixture._check_metadata_features(output)


if __name__ == "__main__":
    unittest.main()
