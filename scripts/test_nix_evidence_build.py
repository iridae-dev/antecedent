"""Build-only diagnostic tests; no scientific target is run or recorded."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nix_evidence_build as build

TOY = [{"crate": "toy", "kind": "lib", "name": "", "features": []}]


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
        recorded = []
        with patch.object(build.subprocess, "run", side_effect=fake_run), \
             patch.object(build, "_reproducibility_key", return_value="key"), \
             patch.object(build, "_reproduced", return_value=set()), \
             patch.object(build, "_record_reproduced", side_effect=lambda root, key: recorded.append(key)):
            self.assertEqual(build._build(Path("/repo"), "/nix/store/source", TOY, "/nix/store/deps"), "/nix/store/output")
        # A new recipe or lock rebuilds the evidence derivation once and records it.
        self.assertEqual(len(commands), 2)
        self.assertIn("--offline", commands[0])
        self.assertEqual(json.loads(commands[0][commands[0].index("targetsJson") + 1]), TOY)
        self.assertIn("/nix/store/deps", commands[0][commands[0].index("depsSrc") + 1])
        self.assertIn("--rebuild", commands[1])
        self.assertNotIn("run", commands[1])
        self.assertEqual(recorded, ["key"])
        commands.clear()
        with patch.object(build.subprocess, "run", side_effect=fake_run), \
             patch.object(build, "_reproducibility_key", return_value="key"), \
             patch.object(build, "_reproduced", return_value={"key"}):
            build._build(Path("/repo"), "/nix/store/source", TOY, "/nix/store/deps")
        # Already reproduced for this recipe and lock: one build, no rebuild.
        self.assertEqual(len(commands), 1)
        with (
            patch.object(build.subprocess, "run", return_value=SimpleNamespace(stdout="/nix/store/output", stderr="warning: ignoring untrusted setting")),
            self.assertRaisesRegex(ValueError, "untrusted"),
        ):
            build._build(Path("/repo"), "/nix/store/source", TOY)

    def test_compiler_graph_must_match_the_selected_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = {"id": "path+fixture#toy@0.1.0", "name": "toy", "version": "0.1.0",
                       "source": None, "manifest_path": str(root / "Cargo.toml")}
            unit = {"pkg_id": package["id"], "target": {"name": "toy", "kind": ["lib"],
                    "src_path": str(root / "src/lib.rs")}, "dependencies": [], "features": ["f"],
                    "mode": "test", "platform": None, "profile": {}}
            target = root / "targets/toy/lib/lib"
            target.mkdir(parents=True)
            (target / "cargo-metadata.json").write_text(json.dumps({"packages": [package]}), encoding="utf-8")
            (target / "cargo-unit-graph.json").write_text(json.dumps({"version": 1, "roots": [0], "units": [unit]}), encoding="utf-8")
            self.assertEqual(len(build._compiler_graph(root, "toy", "lib", "", ["f"])["units"]), 1)
            other = root / "targets/other/lib/lib"
            other.mkdir(parents=True)
            for name in ("cargo-metadata.json", "cargo-unit-graph.json"):
                (other / name).write_bytes((target / name).read_bytes())
            with self.assertRaisesRegex(ValueError, "does not match"):
                build._compiler_graph(root, "other", "lib", "", [])

    def test_build_result_never_authorizes_evidence(self):
        source = {"commit": "abc", "store_path": "/nix/store/source", "nar_hash": "nar"}
        artifact = {"source_nar_hash": "nar", "sandbox_verified": False}
        with patch.object(build, "_target_features", return_value=[]), \
             patch.object(build, "materialize_committed_source", return_value={"commit": "abc"}), \
             patch.object(build, "import_committed_snapshot", return_value=source), \
             patch.object(build, "verify_committed_snapshot") as verify, \
             patch.object(build, "write_deps_source"), \
             patch.object(build, "import_deps_source", return_value="/nix/store/deps"), \
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

    def test_several_targets_share_one_derivation_and_keep_their_own_records(self):
        source = {"commit": "abc", "store_path": "/nix/store/source", "nar_hash": "nar"}

        def artifact(root, output, source_path, crate, kind, name, features, build_targets,
                     deps_source, deps_targets):
            self.assertEqual(len(build_targets), 2)
            self.assertEqual(deps_source, "/nix/store/deps")
            self.assertEqual(len(deps_targets), 3)
            return {"source_nar_hash": "nar", "sandbox_verified": False, "crate": crate}
        with patch.object(build, "_target_features", return_value=[]), \
             patch.object(build, "materialize_committed_source", return_value={"commit": "abc"}), \
             patch.object(build, "import_committed_snapshot", return_value=source), \
             patch.object(build, "verify_committed_snapshot"), \
             patch.object(build, "write_deps_source"), \
             patch.object(build, "import_deps_source", return_value="/nix/store/deps"), \
             patch.object(build, "_build", return_value="/nix/store/output") as derivation, \
             patch.object(build, "inspect_artifact", side_effect=artifact), \
             patch.object(build, "_versions", return_value={"rustc": "1.85.1"}), \
             patch.object(build, "_compiler_graph", return_value={"roots": [0], "units": [{}]}), \
             patch.object(build, "recheck_artifact"):
            result = build.build_committed_targets(Path("/repo"), [("b", "test", "t"), ("a", "lib", ""), ("a", "lib", "")],
                                                   deps_targets=[("c", "lib", "")])
        derivation.assert_called_once()
        self.assertEqual(sorted(result), [("a", "lib", ""), ("b", "test", "t")])
        self.assertEqual(result[("b", "test", "t")]["artifact"]["crate"], "b")


if __name__ == "__main__":
    unittest.main()
