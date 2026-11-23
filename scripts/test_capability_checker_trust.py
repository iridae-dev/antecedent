"""Direct capability checker controls; all cases stop before collecting evidence."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).with_name("check_capability_records.py")
OVERRIDES = (
    "CAPABILITY_TRANSPORT_STAGES", "CAPABILITY_SUPPORT_CELLS", "CAPABILITY_SUPPORT_REFUSAL_REASONS",
    "CAPABILITY_EXTRA_SOURCES", "CAPABILITY_DISABLE_RULES", "CAPABILITY_SYNTHETIC_EVIDENCE",
    "CAPABILITY_EVIDENCE_CACHE", "CAPABILITY_STATIC_ONLY",
)


class DirectCapabilityTrust(unittest.TestCase):
    def environment(self):
        return {k: v for k, v in os.environ.items() if k not in OVERRIDES}

    def invoke(self, *args, env=None):
        return subprocess.run([sys.executable, str(CHECKER), *args],
                              env=env or self.environment(), capture_output=True, text=True)

    def test_each_environment_override_refused_before_rules_are_listed(self):
        for name in OVERRIDES:
            with self.subTest(name=name):
                env = self.environment()
                env[name] = "1"
                result = self.invoke("--list-rules", env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)
                self.assertIn("--trusted-testing", result.stderr)
                self.assertEqual(result.stdout, "")
                trusted = self.invoke("--trusted-testing", "--list-rules", env=env)
                self.assertEqual(trusted.returncode, 0)
                self.assertIn("not capability evidence", trusted.stderr)

    def test_noncanonical_registry_requires_explicit_trust_before_opening(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.toml")
            result = self.invoke(missing)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("noncanonical registry requires --trusted-testing", result.stderr)
            self.assertNotIn("FileNotFoundError", result.stderr)

    def test_static_evidence_does_not_enable_environment_override(self):
        env = self.environment()
        env["CAPABILITY_EVIDENCE_CACHE"] = "/tmp/untrusted-cache"
        result = self.invoke("--static-evidence", "--list-rules", env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CAPABILITY_EVIDENCE_CACHE", result.stderr)


if __name__ == "__main__":
    unittest.main()
