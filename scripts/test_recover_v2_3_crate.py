"""Ensure a partial-release retry cannot publish unverified or altered source."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import recover_v2_3_crate as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name)
        self.manifest = self.source / "crates/antecedent/Cargo.toml"
        self.manifest.parent.mkdir(parents=True)
        self.original = (
            '[package]\nname = "antecedent"\nversion = "2.3.0"\n'
            '[dependencies]\nantecedent-core = "2.3.0"\n'
            "[dev-dependencies]\n" + recovery.OLD_DEPENDENCY + "\n"
        ).encode()
        self.manifest.write_bytes(self.original)
        self.dirty = ""
        self.sha = recovery.SOURCE_SHA
        self.jobs = {
            "Prepare version": "success",
            "crates.io dry-run": "success",
            "Publish GitHub Release": "success",
            "Publish wheels to PyPI": "success",
        }
        self.commands = []
        self.fail_dry_run = False

    def external(self, source, *command, capture=False, env=None):
        self.commands.append(command)
        if command[:3] == ("git", "status", "--porcelain"):
            return self.dirty
        if command[:2] == ("git", "rev-parse"):
            return self.sha
        if command[:3] == ("gh", "release", "view"):
            return json.dumps({"tagName": recovery.TAG, "isDraft": False})
        if command[:3] == ("gh", "run", "view"):
            return json.dumps(
                {
                    "headSha": recovery.SOURCE_SHA,
                    "workflowName": "publish-release",
                    "jobs": [
                        {"name": name, "conclusion": status}
                        for name, status in self.jobs.items()
                    ],
                }
            )
        if command[:2] == ("git", "diff"):
            return "crates/antecedent/Cargo.toml"
        if command[0] == "bash":
            return ""
        if command[:2] == ("cargo", "publish"):
            self.assertEqual(
                self.manifest.read_bytes().decode(),
                self.original.decode().replace(
                    recovery.OLD_DEPENDENCY, recovery.NEW_DEPENDENCY
                ),
            )
            if "--dry-run" in command and self.fail_dry_run:
                raise subprocess.CalledProcessError(101, command)
            return ""
        raise AssertionError(f"Unexpected external operation: {command}")

    def attempt(self):
        with (
            patch.object(recovery, "run", side_effect=self.external),
            patch.dict(os.environ, {"CARGO_REGISTRY_TOKEN": "test-token"}),
        ):
            recovery.recover(self.source, execute=True)

    def assert_no_package(self):
        self.assertFalse(
            any(command[:2] == ("cargo", "publish") for command in self.commands)
        )
        self.assertEqual(self.manifest.read_bytes(), self.original)

    def test_wrong_source_cannot_package_or_modify_manifest(self):
        self.sha = "0" * 40
        with self.assertRaisesRegex(ValueError, "original immutable"):
            self.attempt()
        self.assert_no_package()

    def test_dirty_source_cannot_package_or_modify_manifest(self):
        self.dirty = " M crates/antecedent/src/lib.rs"
        with self.assertRaisesRegex(ValueError, "must be clean"):
            self.attempt()
        self.assert_no_package()

    def test_incomplete_python_release_cannot_package_or_modify_manifest(self):
        self.jobs["Publish wheels to PyPI"] = "failure"
        with self.assertRaisesRegex(ValueError, "must have succeeded"):
            self.attempt()
        self.assert_no_package()

    def test_failed_dry_run_cannot_upload_and_restores_manifest(self):
        self.fail_dry_run = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.attempt()
        publications = [c for c in self.commands if c[:2] == ("cargo", "publish")]
        self.assertEqual(len(publications), 1)
        self.assertIn("--dry-run", publications[0])
        self.assertEqual(self.manifest.read_bytes(), self.original)

    def test_verified_recovery_only_uploads_facade_after_dry_run(self):
        self.attempt()
        publications = [c for c in self.commands if c[:2] == ("cargo", "publish")]
        self.assertEqual(len(publications), 2)
        self.assertIn("--dry-run", publications[0])
        self.assertNotIn("--dry-run", publications[1])
        self.assertTrue(all(c[2:4] == ("-p", "antecedent") for c in publications))
        self.assertEqual(self.manifest.read_bytes(), self.original)

    def test_changed_self_dependency_is_not_a_packaging_only_repair(self):
        for before, after in (
            ("2.3.0", "2.4.0"),
            ('path = "."', 'path = "../foreign"'),
            ("calibration-internal", "ml-neural"),
        ):
            with self.subTest(change=after), self.assertRaises(ValueError):
                recovery.repair_manifest(self.original.decode().replace(before, after))


if __name__ == "__main__":
    unittest.main()
