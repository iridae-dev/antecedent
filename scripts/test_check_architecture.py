"""Ownership gate must reject direct, transitive, aliased and platform edges."""
# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_architecture import dependency_graph, problems


class ArchitectureTests(unittest.TestCase):
    def test_transitive_estimator_dependency_reports_its_path(self):
        failures = problems({"antecedent-design": {"helper"}, "helper": {"antecedent-estimate"}})
        self.assertEqual(
            failures, ["ownership violation: antecedent-design -> helper -> antecedent-estimate"]
        )

    def test_permitted_io_core_and_prediction_dependencies(self):
        self.assertEqual(
            problems(
                {
                    "antecedent-design": {"antecedent-io-core"},
                    "antecedent-io-core": {"antecedent-learn"},
                    "antecedent-learn": {"antecedent-stats"},
                }
            ),
            [],
        )

    def test_core_cannot_depend_on_another_workspace_crate(self):
        self.assertTrue(
            problems({"antecedent-core": {"antecedent-kernels"}, "antecedent-kernels": set()})
        )

    def test_optional_target_build_alias_is_checked_but_dev_consumers_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text(
                """[workspace]
members = ["design"]
[workspace.dependencies]
engine = { package = "antecedent-estimate", version = "1" }
""",
                encoding="utf-8",
            )
            (root / "design").mkdir()
            manifest = root / "design/Cargo.toml"
            manifest.write_text(
                """[package]
name = "antecedent-design"
[dev-dependencies]
antecedent = "1"
[target.'cfg(windows)'.build-dependencies]
engine = { workspace = true, optional = true }
""",
                encoding="utf-8",
            )
            self.assertEqual(dependency_graph(root), {"antecedent-design": {"antecedent-estimate"}})
            self.assertTrue(problems(dependency_graph(root)))

    def test_cycles_do_not_hide_forbidden_paths_or_loop(self):
        self.assertTrue(
            problems(
                {"antecedent-expr": {"helper"}, "helper": {"antecedent-expr", "antecedent-model"}}
            )
        )


if __name__ == "__main__":
    unittest.main()
