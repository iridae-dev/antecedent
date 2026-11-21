"""Focused checks for features inherited from included Rust test modules."""
import json
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch

from _repo import REPO_ROOT, ensure_scripts_on_path

ensure_scripts_on_path()
import test_evidence  # noqa: E402


class IncludedModuleFeatures(unittest.TestCase):
    def test_suite_03_cross_world_evidence_resolves_its_module_feature(self):
        source = REPO_ROOT / "crates/antecedent/tests/cross_world_edge_contrast.rs"
        self.assertEqual(
            test_evidence.rust_test_features(
                source, "exact_linear_scm_truth_for_every_edge_set"
            ),
            ["calibration-internal"],
        )

    def test_cargo_listing_enables_features_resolved_from_the_module(self):
        source = REPO_ROOT / "crates/antecedent/tests/cross_world_edge_contrast.rs"
        target = test_evidence.target_root(source)
        self.assertIsNotNone(target)
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        test_evidence._CARGO_LIST_CACHE.clear()
        with patch.object(test_evidence.subprocess, "run", side_effect=fake_run):
            test_evidence.cargo_listed(
                target[1], target[2], features=test_evidence.rust_test_features(
                    source, "exact_linear_scm_truth_for_every_edge_set"
                )
            )
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(
            "--features" in command
            and "calibration-internal" in command[command.index("--features") + 1]
            for command in commands
        ))

    def test_batch_keeps_ignored_names_and_feature_sets_separate(self):
        sources = [REPO_ROOT / "scripts" / name for name in ("test_evidence.py", "gate_support.py")]
        paths = [path.with_suffix(".rs") for path in sources]
        sources_by_name = {path: f"suite_{index}" for index, path in enumerate(paths)}
        def owner(path):
            return path, "pkg", ["--test", sources_by_name[path]]
        def listing(command, **kwargs):
            features = "--features" in command
            selected = [arg for i, arg in enumerate(command) if i and command[i - 1] == "--test"]
            suites = {
                "pkg::" + target: {"testcases": {"normal": {"ignored": False}, "manual": {"ignored": True}}}
                for target in selected
            }
            if features:
                self.assertEqual(command[command.index("--features") + 1], "extra")
            return CompletedProcess(command, 0, json.dumps({"rust-suites": suites}))
        with patch.object(test_evidence.shutil, "which", return_value="cargo-nextest"), \
             patch.object(test_evidence.Path, "is_file", return_value=True), \
             patch.object(test_evidence, "target_root", side_effect=owner), \
             patch.object(test_evidence, "rust_test_features", side_effect=lambda p, n: ["extra"] if n == "gated" else []), \
             patch.dict(test_evidence._CARGO_LIST_CACHE, clear=True), \
             patch.object(test_evidence.subprocess, "run", side_effect=listing) as run:
            test_evidence.prefetch_rust_listings([(path, "normal") for path in paths] + [(paths[0], "gated")])
            self.assertEqual(run.call_count, 2)
            for path in paths:
                self.assertEqual(test_evidence.cargo_listed("pkg", owner(path)[2]), ({"manual", "normal"}, {"manual"}, None))
            self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
