"""Exercise release dependency validation without registry access or uploads."""

import json
import os
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import set_version
from _repo import workspace_package_version

VERSION = workspace_package_version()
STALE = "0.1.0"


class PublishVersionTests(unittest.TestCase):
    def run_gate(
        self,
        requirement: str,
        kind: str | None = None,
        *,
        self_version: str | None = None,
        self_dependency: bool = False,
        private_dependency: bool = False,
        mode: str = "--dry-run",
        separate_source: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        packages = [
            {
                "id": "core",
                "name": "antecedent-core",
                "version": VERSION,
                "dependencies": [],
            },
            {
                "id": "consumer",
                "name": "antecedent-consumer",
                "version": VERSION,
                "dependencies": [
                    {
                        "name": "antecedent-consumer" if self_dependency else "antecedent-core",
                        "req": requirement,
                        "kind": kind,
                    }
                ],
            },
        ]
        if private_dependency:
            packages[0]["publish"] = []
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "consumer.toml"
            self_spec = 'path = "."'
            if self_version is not None:
                self_spec += f', version = "{self_version}"'
            manifest.write_text(
                f"[dev-dependencies]\n{'antecedent-core' if private_dependency else 'antecedent-consumer'} = {{ {self_spec} }}\n",
                encoding="utf-8",
            )
            packages[1]["manifest_path"] = str(manifest)
            cargo = Path(directory) / "cargo"
            cargo.write_text(
                "#!/usr/bin/env python3\nimport os, sys\n"
                'if "PUBLISH_TEST_SOURCE" in os.environ:\n'
                '    assert os.path.samefile(os.getcwd(), os.environ["PUBLISH_TEST_SOURCE"])\n'
                'if sys.argv[1] == "metadata":\n'
                '    print(os.environ["PUBLISH_TEST_METADATA"])\n'
                'elif sys.argv[1] == "publish":\n'
                '    print("TEST_PACKAGE_VERIFIED", " ".join(sys.argv[1:]))\n'
                "else:\n"
                '    sys.exit("unexpected cargo invocation")\n',
                encoding="utf-8",
                newline="\n",
            )
            cargo.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = directory + os.pathsep + env["PATH"]
            env["PUBLISH_TEST_METADATA"] = json.dumps(
                {"packages": packages, "workspace_members": ["core", "consumer"]}
            )
            arguments = [mode]
            if separate_source:
                source = Path(directory) / "release-source"
                source.mkdir()
                (source / "Cargo.toml").write_text(f'[workspace.package]\nversion = "{VERSION}"\n')
                env["PUBLISH_TEST_SOURCE"] = str(source)
                arguments += ["--source", str(source), "--allow-dirty"]
            if mode == "--execute":
                curl = Path(directory) / "curl"
                curl.write_text(
                    "#!/usr/bin/env python3\nimport sys\n"
                    'print("200" if "/antecedent-core/" in sys.argv[-1] else "404")\n'
                )
                curl.chmod(0o755)
                env["CARGO_REGISTRY_TOKEN"] = "test-token"
                env["PUBLISH_SLEEP_SECS"] = "0"
            return subprocess.run(
                [
                    "bash",
                    str(Path(__file__).with_name("publish_crates.sh")),
                    *arguments,
                ],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_current_internal_version_packages_in_dependency_order(self) -> None:
        result = self.run_gate(f"^{VERSION}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("packaged=2 check-only=0", result.stdout)
        self.assertLess(
            result.stdout.index("-p antecedent-core"),
            result.stdout.index("-p antecedent-consumer"),
        )

    def test_outdated_runtime_dependency_refuses_before_packaging(self) -> None:
        result = self.run_gate(f"^{STALE}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("internal dependency version mismatch", result.stderr)
        self.assertNotIn("TEST_PACKAGE_VERIFIED", result.stdout)

    def test_outdated_versioned_dev_dependency_also_refuses(self) -> None:
        result = self.run_gate(f"^{STALE}", "dev")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"expected ^{VERSION}", result.stderr)

    def test_unpublished_path_only_dev_dependency_is_allowed(self) -> None:
        result = self.run_gate("*", "dev")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_versioned_self_dev_dependency_refuses_before_any_packaging(self) -> None:
        result = self.run_gate(f"^{VERSION}", "dev", self_version=VERSION, self_dependency=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("versioned self dev dependency must be path-only", result.stderr)
        self.assertNotIn("TEST_PACKAGE_VERIFIED", result.stdout)

    def test_explicit_wildcard_self_dev_dependency_is_still_versioned(self) -> None:
        result = self.run_gate("*", "dev", self_version="*", self_dependency=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("versioned self dev dependency must be path-only", result.stderr)
        self.assertNotIn("TEST_PACKAGE_VERIFIED", result.stdout)

    def test_path_only_self_dev_dependency_is_omitted_and_allowed(self) -> None:
        result = self.run_gate("*", "dev", self_dependency=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("packaged=2 check-only=0", result.stdout)

    def test_versioned_private_dev_dependency_refuses_before_packaging(self) -> None:
        result = self.run_gate(f"^{VERSION}", "dev", private_dependency=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unpublished dependency antecedent-core", result.stderr)
        self.assertNotIn("TEST_PACKAGE_VERIFIED", result.stdout)

    def test_private_runtime_and_build_dependencies_refuse(self) -> None:
        for kind in (None, "build"):
            with self.subTest(kind=kind):
                result = self.run_gate("*", kind, private_dependency=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("path-only dev dependency", result.stderr)
                self.assertNotIn("TEST_PACKAGE_VERIFIED", result.stdout)

    def test_path_only_private_dev_dependency_is_omitted_and_allowed(self) -> None:
        result = self.run_gate("*", "dev", private_dependency=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("packaged=1 check-only=0", result.stdout)
        self.assertNotIn("-p antecedent-core", result.stdout)

    def test_wildcard_private_dev_dependency_is_still_versioned(self) -> None:
        result = self.run_gate("*", "dev", private_dependency=True, self_version="*")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("versioned unpublished dev dependency must be path-only", result.stderr)
        self.assertNotIn("TEST_PACKAGE_VERIFIED", result.stdout)

    def test_execute_skips_published_crates_and_verifies_missing_crate(self) -> None:
        result = self.run_gate(f"^{VERSION}", mode="--execute")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("published=1 skipped=1", result.stdout)
        self.assertIn(
            "TEST_PACKAGE_VERIFIED publish -p antecedent-consumer --locked",
            result.stdout,
        )
        self.assertNotIn("--allow-dirty", result.stdout)

    def test_recovery_packages_separate_source_with_reviewed_manifest_repair(
        self,
    ) -> None:
        result = self.run_gate(f"^{VERSION}", mode="--execute", separate_source=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("published=1 skipped=1", result.stdout)
        self.assertIn("--allow-dirty", result.stdout)


class ReleaseVersionBumpTests(unittest.TestCase):
    def test_next_release_updates_workspace_pins_and_keeps_private_helper_path_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text(
                '[workspace.package]\nversion = "9.8.7"\n'
                "[workspace.dependencies]\n"
                'antecedent-core = { version = "9.8.7", path = "crates/antecedent-core" }\n'
                'antecedent-test-support = { path = "crates/antecedent-test-support" }\n'
            )
            (root / "python/antecedent").mkdir(parents=True)
            (root / "python/pyproject.toml").write_text('[project]\nversion = "9.8.7"\n')
            (root / "python/antecedent/__init__.py").write_text(
                "from ._native import __version__ as __version__\n"
            )
            with (
                patch.object(set_version, "ROOT", root),
                patch.object(set_version, "apply_release_snippets"),
                patch.object(set_version.subprocess, "check_call"),
            ):
                set_version.main(["set_version.py", "9.8.8", "0"])
            manifest = tomllib.loads((root / "Cargo.toml").read_text())
            self.assertEqual(manifest["workspace"]["package"]["version"], "9.8.8")
            self.assertEqual(
                manifest["workspace"]["dependencies"]["antecedent-core"]["version"],
                "9.8.8",
            )
            self.assertEqual(
                manifest["workspace"]["dependencies"]["antecedent-test-support"],
                {"path": "crates/antecedent-test-support"},
            )


if __name__ == "__main__":
    unittest.main()
