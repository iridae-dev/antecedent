"""Calibration selection/prebuild ownership checks; never run a measurement."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_groups as groups  # noqa: E402


class WorkspaceSuiteOwnershipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "Cargo.toml").write_text('[workspace]\nmembers=["crates/*"]\n')
        self.member("antecedent", "antecedent", ["facade_current", "facade_new"])
        # Cargo package names, rather than directory spelling, own the command.
        self.member("io-folder", "antecedent-io", ["nested_bayesian"])
        self.member("estimate-folder", "antecedent-estimate", ["nested_fisher", "recovery"])
        self.member("antecedent-prob", "antecedent-prob", [])
        self.addCleanup(groups._workspace_packages.cache_clear)
        self.addCleanup(groups._suite_owner.cache_clear)
        self.root_patch = patch.object(groups, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def member(self, folder: str, name: str, suites: list[str], extra: str = "") -> None:
        path = self.root / "crates" / folder
        (path / "tests").mkdir(parents=True)
        (path / "Cargo.toml").write_text(f'[package]\nname="{name}"\nversion="1.0.0"\n' + extra)
        for suite in suites:
            (path / "tests" / f"{suite}.rs").write_text("// record-emitting fixture\n")

    def candidates(self) -> list[groups.Group]:
        return [groups.Group(i, label, False, True) for i, label in enumerate([
            "facade_current: existing", "facade_new: missing", "nested_bayesian: missing",
            "nested_fisher: missing", "recovery: missing", "antecedent-prob: sbc", "external_gate",
        ], 1)]

    def test_missing_io_estimate_and_facade_records_selected_when_all_existing_current(self) -> None:
        records = [{"id": "current", "test": "crates/antecedent/tests/facade_current.rs::existing"}]
        facets = SimpleNamespace(
            load_records=lambda: records,
            load_surface=lambda: {},
            assess=lambda surface, rows: [SimpleNamespace(stale=[], resolved=True, records=rows)],
        )
        with patch.dict(sys.modules, {"calibration_facets": facets}):
            selection = groups.select(self.candidates(), False)
        self.assertEqual(selection.owed_records, 0)
        self.assertEqual([g.index for g in selection.groups], [2, 3, 4, 5, 6, 7])
        for index in (2, 3, 4, 5):
            self.assertEqual(selection.reasons[index], "no record in the registry")
        self.assertEqual(selection.reasons[6], "pass/fail gate, no record")
        # If the missing cells disappear, current registry rows do not trigger gates.
        with patch.dict(sys.modules, {"calibration_facets": facets}):
            self.assertEqual(groups.select([self.candidates()[0], self.candidates()[5]], False).groups, [])

    def test_prebuild_groups_actual_package_targets_and_retains_lib_commands(self) -> None:
        candidates = self.candidates() + [groups.Group(8, "nested_bayesian: another", False, True)]
        with patch.object(groups.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(groups.prebuild(candidates), 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands, [
            ["cargo", "test", "--release", "-p", "antecedent-prob", "--lib", "--no-run"],
            ["cargo", "test", "--release", "-p", "antecedent", "--test", "facade_current", "--test", "facade_new", "--no-run"],
            ["cargo", "test", "--release", "-p", "antecedent-estimate", "--test", "nested_fisher", "--test", "recovery", "--no-run"],
            ["cargo", "test", "--release", "-p", "antecedent-io", "--test", "nested_bayesian", "--no-run"],
        ])
        self.assertTrue(all(call.kwargs == {"cwd": self.root} for call in run.call_args_list))

    def test_explicit_test_target_and_ambiguous_owner(self) -> None:
        self.member("custom", "custom-package", [], '\nautotests=false\n[[test]]\nname="custom_suite"\npath="oracle.rs"\n')
        (self.root / "crates/custom/oracle.rs").write_text("// explicit integration target\n")
        groups._workspace_packages.cache_clear()
        self.assertEqual(groups._suite_owner(self.root, "custom_suite"), "custom-package")
        self.member("duplicate", "duplicate-package", ["nested_bayesian"])
        groups._workspace_packages.cache_clear()
        with self.assertRaisesRegex(SystemExit, "ambiguous calibration test target"):
            groups._suite_owner(self.root, "nested_bayesian")

    def test_prebuild_failure_stops_before_later_packages(self) -> None:
        with patch.object(groups.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run:
            self.assertEqual(groups.prebuild(self.candidates()), 1)
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
