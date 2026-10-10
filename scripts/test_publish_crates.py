"""Exercise release dependency validation without registry access or uploads."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class PublishVersionTests(unittest.TestCase):
    def run_gate(
        self, requirement: str, kind: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        packages = [
            {
                "id": "core",
                "name": "antecedent-core",
                "version": "2.3.0",
                "dependencies": [],
            },
            {
                "id": "consumer",
                "name": "antecedent-consumer",
                "version": "2.3.0",
                "dependencies": [
                    {"name": "antecedent-core", "req": requirement, "kind": kind}
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            cargo = Path(directory) / "cargo"
            cargo.write_text(
                "#!/usr/bin/env python3\nimport os, sys\n"
                'if sys.argv[1] == "metadata":\n'
                '    print(os.environ["PUBLISH_TEST_METADATA"])\n'
                'elif sys.argv[1] == "publish" and "--dry-run" in sys.argv:\n'
                '    print("TEST_PACKAGE_VERIFIED")\n'
                "else:\n"
                '    sys.exit("unexpected cargo invocation")\n'
            )
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
        result = self.run_gate("^2.3.0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("packaged=2 check-only=0", result.stdout)
        self.assertLess(
            result.stdout.index("-p antecedent-core"),
            result.stdout.index("-p antecedent-consumer"),
        )

    def test_outdated_runtime_dependency_refuses_before_packaging(self) -> None:
        result = self.run_gate("^2.2.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("internal dependency version mismatch", result.stderr)
        self.assertNotIn("TEST_PACKAGE_VERIFIED", result.stdout)

    def test_outdated_versioned_dev_dependency_also_refuses(self) -> None:
        result = self.run_gate("^2.2.0", "dev")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected ^2.3.0", result.stderr)

    def test_unpublished_path_only_dev_dependency_is_allowed(self) -> None:
        result = self.run_gate("*", "dev")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
