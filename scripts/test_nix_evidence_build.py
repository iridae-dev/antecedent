"""Build-only diagnostic tests; no scientific target is run or recorded."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import nix_evidence_build as build


class BuildOnlyTest(unittest.TestCase):
    def test_toolchain_requires_pinned_compiler_and_reviewed_host(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin = root / "tools/evidence-analyzer/pin.toml"
            pin.parent.mkdir(parents=True)
            pin.write_text('release = "1.85.1"\ncommit = "abc"\n', encoding="utf-8")
            out = root / "out"
            out.mkdir()
            (out / "cargo-version").write_text("cargo 1.97.0 (test)\n", encoding="utf-8")
            version = out / "rustc-version"
            version.write_text("release: 1.85.1\ncommit-hash: abc\nhost: aarch64-apple-darwin\n", encoding="utf-8")
            self.assertEqual(build._versions(root, out)["host"], "aarch64-apple-darwin")
            version.write_text("release: 1.85.1\ncommit-hash: abc\nhost: x86_64-unknown-linux-gnu\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "compiler"):
                build._versions(root, out)

    def test_target_features_use_the_stable_scientific_manifest_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "crates/toy"
            package.mkdir(parents=True)
            (package / "Cargo.toml").write_text(
                '[package]\nname = "toy"\nversion = "0.1.0"\n'
                '[features]\na = []\nb = []\n'
                '[package.metadata.scientific-test-features]\nsuite = ["b", "a"]\n', encoding="utf-8")
            self.assertEqual(build._target_features(root, "toy", "test", "suite"), ["a", "b"])
            with self.assertRaises(ValueError):
                build._target_features(root, "../bad", "test", "suite")

    def test_build_requests_offline_local_rebuild_and_refuses_untrusted_overrides(self):
        commands = []
        def fake_run(argv, **kwargs):
            commands.append(argv)
            return SimpleNamespace(stdout="/nix/store/output\n", stderr="")
        with patch.object(build.subprocess, "run", side_effect=fake_run):
            self.assertEqual(build._build(Path("/repo"), "/nix/store/source", "toy", "lib", "", []), "/nix/store/output")
        self.assertEqual(len(commands), 2)
        self.assertIn("--offline", commands[0])
        self.assertIn("--rebuild", commands[1])
        self.assertIn("--option", commands[1])
        self.assertNotIn("run", commands[1])
        with patch.object(build.subprocess, "run", return_value=SimpleNamespace(stdout="/nix/store/output", stderr="warning: ignoring untrusted setting")):
            with self.assertRaisesRegex(ValueError, "untrusted"):
                build._build(Path("/repo"), "/nix/store/source", "toy", "lib", "", [])

    def test_compiler_graph_must_match_the_selected_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = {"id": "path+fixture#toy@0.1.0", "name": "toy", "version": "0.1.0",
                       "source": None, "manifest_path": str(root / "Cargo.toml")}
            unit = {"pkg_id": package["id"], "target": {"name": "toy", "kind": ["lib"],
                    "src_path": str(root / "src/lib.rs")}, "dependencies": [], "features": ["f"],
                    "mode": "test", "platform": None, "profile": {}}
            (root / "cargo-metadata.json").write_text(json.dumps({"packages": [package]}), encoding="utf-8")
            (root / "cargo-unit-graph.json").write_text(json.dumps({"version": 1, "roots": [0], "units": [unit]}), encoding="utf-8")
            self.assertEqual(len(build._compiler_graph(root, "toy", "lib", "", ["f"])["units"]), 1)
            with self.assertRaisesRegex(ValueError, "does not match"):
                build._compiler_graph(root, "other", "lib", "", [])

    def test_build_result_never_authorizes_evidence(self):
        source = {"commit": "abc", "store_path": "/nix/store/source", "nar_hash": "nar"}
        artifact = {"source_nar_hash": "nar", "sandbox_verified": False}
        with patch.object(build, "_target_features", return_value=[]), \
             patch.object(build, "materialize_committed_source", return_value={"commit": "abc"}), \
             patch.object(build, "import_committed_snapshot", return_value=source), \
             patch.object(build, "verify_committed_snapshot") as verify, \
             patch.object(build, "_build", return_value="/nix/store/output"), \
             patch.object(build, "inspect_artifact", return_value=artifact), \
             patch.object(build, "_versions", return_value={"rustc": "1.85.1"}), \
             patch.object(build, "_compiler_graph", return_value={"roots": [0], "units": [{}]}), \
             patch.object(build, "recheck_artifact") as recheck:
            result = build.build_committed_target(Path("/repo"), "toy", "lib")
        self.assertFalse(result["evidence_authorized"])
        self.assertFalse(result["sandbox_verified"])
        self.assertEqual(verify.call_count, 2)
        recheck.assert_called_once()


if __name__ == "__main__":
    unittest.main()
