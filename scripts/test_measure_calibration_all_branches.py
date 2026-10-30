import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("measure_calibration.sh").resolve()

class StubbedDriverBranches(unittest.TestCase):
    def test_pilot_and_run_forward_options_to_mocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            fake = {
                "git": "#!/bin/sh\ncase \"$1\" in rev-parse) echo 0123456789012345678901234567890123456789;; esac\n",
                "python3": "#!/bin/sh\nprintf 'PY' >> \"$CAPTURE\"; printf ' <%s>' \"$@\" >> \"$CAPTURE\"; printf '\n' >> \"$CAPTURE\"\n",
                "bash": "#!/bin/sh\nprintf 'BASH' >> \"$CAPTURE\"; printf ' <%s>' \"$@\" >> \"$CAPTURE\"; printf '\n' >> \"$CAPTURE\"\n",
            }
            for name, content in fake.items():
                path = root / name
                path.write_text(content)
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ.get("PATH", ""), CAPTURE=str(capture))
            for options, expected in [
                (["--pilot", "--group", "two words", "--replicates", "17"], ["<pilot>", "<two words>", "<17>"]),
                (["--group", "two words", "--replace-record", "r one"], ["<run>", "<two words>", "<--keep-existing>", "<r one>"]),
                ([], ["<run>", "<--keep-attested>"])
            ]:
                with self.subTest(options=options):
                    capture.write_text("")
                    result = subprocess.run(["/bin/bash", str(SCRIPT), "--jobs", "1", *options], env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    calls = capture.read_text()
                    for token in expected:
                        self.assertIn(token, calls)
