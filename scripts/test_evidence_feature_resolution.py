"""Focused checks for features inherited from included Rust test modules."""
import sys
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
