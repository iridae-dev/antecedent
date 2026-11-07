"""Checks for the lock-bound, scoped Nix compile-time source audit."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nix_evidence_policy as policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "nix").mkdir()
        source = Path(__file__).resolve().parent.parent
        for name in ("Cargo.lock", policy.REGISTRY, "nix/compile-time-audit.md", "nix/compile-time-suite01-review.md"):
            (self.root / name).write_bytes((source / name).read_bytes())
        self.registry = json.loads((self.root / policy.REGISTRY).read_text(encoding="utf-8"))
        row = next(x for x in self.registry["units"] if x["mode"] == "run-custom-build" and x["source_projection"] == "complete-source")
        self.unit = {k: row[k] for k in ("name", "version", "mode", "target_kind", "target_path", "features", "platform")}
        self.unit["dependencies"] = []
        self.built = {"compiler_graph": {"units": [self.unit], "roots": [0]}}
        self.sandbox = patch.object(policy, "_verify_sandbox")
        self.sandbox.start()
        self.addCleanup(self.sandbox.stop)

    def test_reviewed_unit_is_bound_to_lock_and_audit(self):
        result = policy.validate_build(self.root, self.built)
        self.assertEqual(result["id"], policy.POLICY)
        self.assertEqual(result["covered_units"], 1)
        self.assertEqual(len(result["audit_selection_sha256"]), 64)
        self.assertFalse(result["source_projection_allowed"])
        policy.recheck_selection(self.root, result)
        result["source_projection_allowed"] = True
        with self.assertRaisesRegex(ValueError, "projection review status"):
            policy.recheck_selection(self.root, result)

    def test_unrelated_audit_changes_do_not_invalidate_selection(self):
        result = policy.validate_build(self.root, self.built)
        selected_name = self.unit["name"]
        unrelated = next(x for x in self.registry["units"] if x["name"] != selected_name)
        unrelated["review_note"] = "unrelated review update"
        (self.root / policy.REGISTRY).write_text(json.dumps(self.registry), encoding="utf-8")
        policy.recheck_selection(self.root, result)
        selected = next(x for x in self.registry["units"] if x["name"] == selected_name
                        and x["mode"] == self.unit["mode"] and x["features"] == self.unit["features"])
        selected["review_note"] = "selected review update"
        (self.root / policy.REGISTRY).write_text(json.dumps(self.registry), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "audit changed"):
            policy.recheck_selection(self.root, result)

    def test_unknown_feature_and_new_helper_fail_closed(self):
        self.unit["features"] = self.unit["features"] + ["unreviewed_feature"]
        with self.assertRaisesRegex(ValueError, "unreviewed Nix compile-time unit"):
            policy.validate_build(self.root, self.built)
        self.unit["features"].pop()
        self.unit["dependencies"] = [{"index": 1}]
        self.built["compiler_graph"]["units"].append(self.unit | {"name": "unreviewed-helper", "dependencies": []})
        with self.assertRaisesRegex(ValueError, "unreviewed Nix compile-time unit"):
            policy.validate_build(self.root, self.built)

    def test_changed_lock_and_bad_dependency_index_refuse(self):
        (self.root / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            policy.validate_build(self.root, self.built)
        source = Path(__file__).resolve().parent.parent / "Cargo.lock"
        (self.root / "Cargo.lock").write_bytes(source.read_bytes())
        self.unit["dependencies"] = [{"index": 20}]
        with self.assertRaisesRegex(ValueError, "dependency index"):
            policy.validate_build(self.root, self.built)

    def test_missing_compile_time_graph_refuses(self):
        self.unit["mode"] = "build"
        self.unit["target_kind"] = ["lib"]
        with self.assertRaisesRegex(ValueError, "no reviewed compile-time units"):
            policy.validate_build(self.root, self.built)


class SandboxPolicyTests(unittest.TestCase):
    def config(self):
        return {"sandbox": {"value": True}, "sandbox-fallback": {"value": False},
                "sandbox-paths": {"value": ["/usr/lib", "/private/tmp"]}}

    def test_system_config_and_unreviewed_paths_are_checked(self):
        system = SimpleNamespace(is_symlink=lambda: False, is_file=lambda: True,
                                 stat=lambda: SimpleNamespace(st_uid=0, st_mode=0o644),
                                 read_text=lambda **kw: "sandbox = true\nsandbox-fallback = false\n")
        result = SimpleNamespace(stdout=json.dumps(self.config()))
        with patch.object(policy, "SYSTEM_CONFIG", system), patch.dict(policy.os.environ, {}, clear=True), \
             patch.object(policy.subprocess, "run", return_value=result):
            policy._verify_sandbox()
            cfg = self.config()
            cfg["sandbox-paths"]["value"].append("/Users")
            result.stdout = json.dumps(cfg)
            with self.assertRaisesRegex(ValueError, "unreviewed sandbox path"):
                policy._verify_sandbox()

    def test_caller_nix_config_override_refuses(self):
        with patch.dict(policy.os.environ, {"NIX_CONFIG": "sandbox = false"}, clear=True):
            with self.assertRaisesRegex(ValueError, "NIX_CONFIG"):
                policy._verify_sandbox()


if __name__ == "__main__":
    unittest.main()
