"""Exercise release dependency validation without registry access or uploads."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

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
                        "name": "antecedent-consumer"
                        if self_dependency
                        else "antecedent-core",
                        "req": requirement,
                        "kind": kind,
                    }
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "consumer.toml"
            self_spec = 'path = "."'
            if self_version is not None:
                self_spec += f', version = "{self_version}"'
            manifest.write_text(
                f"[dev-dependencies]\nantecedent-consumer = {{ {self_spec} }}\n",
                encoding="utf-8",
            )
            packages[1]["manifest_path"] = str(manifest)
            cargo = Path(directory) / "cargo"
            cargo.write_text(
                "#!/usr/bin/env python3\nimport os, sys\n"
                'if sys.argv[1] == "metadata":\n'
                '    print(os.environ["PUBLISH_TEST_METADATA"])\n'
                'elif sys.argv[1] == "publish" and "--dry-run" in sys.argv:\n'
                '    print("TEST_PACKAGE_VERIFIED")\n'
                "else:\n"
                '    sys.exit("unexpected cargo invocation")\n'
            , encoding="utf-8", newline="\n")
            cargo.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = directory + os.pathsep + env["PATH"]
            env["PUBLISH_TEST_METADATA"] = json.dumps(
                {"packages": packages, "workspace_members": ["core", "consumer"]}
            )
            return subprocess.run(
                [
                    "bash",
                    str(Path(__file__).with_name("publish_crates.sh")),
                    "--dry-run",
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
        result = self.run_gate(
            f"^{VERSION}", "dev", self_version=VERSION, self_dependency=True
        )
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


if __name__ == "__main__":
    unittest.main()
