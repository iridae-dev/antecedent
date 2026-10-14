#!/usr/bin/env python3
"""Same-run extension reuse fails closed before importing incompatible artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ci_python_native as native


class NativeArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "checkout"
        self.package = self.root / "python" / "antecedent"
        self.package.mkdir(parents=True)
        (self.root / "python" / "pyproject.toml").write_text(
            '[project]\nversion = "2.4.0"\n', encoding="utf-8"
        )
        self.extension = self.package / f"_native{native.runtime_identity()['extension_suffix']}"
        self.extension.write_bytes(b"native extension bytes")
        self.artifact = Path(self.temp.name) / "artifact"
        self.site_packages = Path(self.temp.name) / "site-packages"
        self.site_packages.mkdir()
        site_patch = patch.object(
            native.sysconfig, "get_path", return_value=str(self.site_packages)
        )
        site_patch.start()
        self.addCleanup(site_patch.stop)
        import_patch = patch.object(native, "verify_import")
        import_patch.start()
        self.addCleanup(import_patch.stop)
        fresh_patch = patch.object(native, "verify_fresh_import")
        self.import_check = fresh_patch.start()
        self.addCleanup(fresh_patch.stop)
        native.stage(self.root, self.artifact, "commit-a")
        self.import_check.reset_mock()
        self.extension.unlink()

    def rewrite_manifest(self, key: str, value: object) -> None:
        path = self.artifact / native.MANIFEST
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest[key] = value
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_installs_and_checks_import_from_matching_checkout(self) -> None:
        native.install(self.root, self.artifact, "commit-a")
        self.assertEqual(self.extension.read_bytes(), b"native extension bytes")
        self.assertEqual(
            (self.site_packages / "antecedent-ci-source.pth").read_text(encoding="utf-8"),
            str((self.root / "python").resolve()) + "\n",
        )
        self.import_check.assert_called_once_with(self.extension, "2.4.0")

    def test_refuses_other_commit_before_copy_or_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "commit"):
            native.install(self.root, self.artifact, "commit-b")
        self.assertFalse(self.extension.exists())
        self.import_check.assert_not_called()

    def test_refuses_incompatible_python_abi(self) -> None:
        runtime = native.runtime_identity()
        runtime["python"] = [3, 11] if runtime["python"] != [3, 11] else [3, 12]
        self.rewrite_manifest("runtime", runtime)
        with self.assertRaisesRegex(ValueError, "runtime"):
            native.install(self.root, self.artifact, "commit-a")
        self.assertFalse(self.extension.exists())

    def test_refuses_corrupted_extension(self) -> None:
        (self.artifact / self.extension.name).write_bytes(b"corrupt bytes")
        with self.assertRaisesRegex(ValueError, "checksum"):
            native.install(self.root, self.artifact, "commit-a")
        self.import_check.assert_not_called()

    def test_refuses_filename_outside_artifact(self) -> None:
        self.rewrite_manifest("filename", "../unexpected.so")
        with self.assertRaisesRegex(ValueError, "filename"):
            native.install(self.root, self.artifact, "commit-a")
        self.assertFalse(self.extension.exists())


if __name__ == "__main__":
    unittest.main()
