"""Bash option forwarding with stubbed commands; no calibration is run."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("measure_calibration.sh").resolve()


class CalibrationShellArrays(unittest.TestCase):
    def test_empty_and_populated_options_under_nounset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, content in {"git": "#!/bin/sh\nexit 0\n", "python3": "#!/bin/sh\nprintf 'ARG<%s>\\n' \"$@\"\n"}.items():
                path = root / name
                path.write_text(content, encoding="utf-8")
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ.get("PATH", ""))
            for args, expected in [([], []), (["--group", "two words", "--replicates", "17"], ["ARG<--group>", "ARG<two words>", "ARG<--replicates>", "ARG<17>"])]:
                with self.subTest(args=args):
                    result = subprocess.run(["/bin/bash", str(SCRIPT), "--dry-run", "--jobs", "1", *args], env=env, capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    for token in expected:
                        self.assertIn(token, result.stdout)

    def test_owed_forwards_the_owed_selection_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, content in {"git": "#!/bin/sh\nexit 0\n", "python3": "#!/bin/sh\nprintf 'ARG<%s>\\n' \"$@\"\n"}.items():
                path = root / name
                path.write_text(content, encoding="utf-8")
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ.get("PATH", ""))
            owed = subprocess.run(["/bin/bash", str(SCRIPT), "--owed", "--dry-run", "--jobs", "1"], env=env, capture_output=True, text=True, check=False)
            plain = subprocess.run(["/bin/bash", str(SCRIPT), "--dry-run", "--jobs", "1"], env=env, capture_output=True, text=True, check=False)
            self.assertEqual(owed.returncode, 0, owed.stderr)
            self.assertIn("ARG<plan>", owed.stdout)
            self.assertIn("ARG<--owed>", owed.stdout)
            self.assertNotIn("ARG<--all>", owed.stdout)
            self.assertNotIn("ARG<--owed>", plain.stdout)

    def test_owed_refuses_to_combine_with_all_or_missing_only(self):
        for other in ("--all", "--missing-only"):
            with self.subTest(other=other):
                result = subprocess.run(["/bin/bash", str(SCRIPT), "--owed", other, "--dry-run"], capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
