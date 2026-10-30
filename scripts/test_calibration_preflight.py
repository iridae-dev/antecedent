import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import calibration_decision as decision
import calibration_preflight as preflight

class DirectGatePreflight(unittest.TestCase):
    def test_missing_malformed_and_foreign_context_fail(self):
        config = decision.config_for_manifest({"groups": [{"label": "x"}]})
        for text in (None, "{", json.dumps({}), decision.canonical_json({**config, "family_digest": "f" * 64})):
            with self.subTest(text=text), self.assertRaises(ValueError):
                preflight.validate_inherited(text, config)
        preflight.validate_inherited(decision.canonical_json(config), config)
    def test_gate_preflight_is_skipped_for_dry_run_and_smoke(self):
        source = Path(__file__).with_name("gate_calibration.sh").read_text()
        start = source.index('# Ordinary direct execution must carry')
        end = source.index('\nFAILED=""', start)
        prefix = source[start:end]
        with tempfile.TemporaryDirectory() as tmp:
            mock = Path(tmp) / "python3"
            mock.write_text("#!/bin/sh\necho CALLED\nexit 7\n")
            mock.chmod(0o755)
            env = dict(os.environ, PATH=tmp + os.pathsep + os.environ.get("PATH", ""))
            for dry, smoke, expected in [("", "", 2), ("1", "", 0), ("", "1", 0)]:
                controlled = dict(env, ANTECEDENT_CALIBRATION_DRY_RUN=dry,
                                  ANTECEDENT_CALIBRATION_SMOKE=smoke)
                result = subprocess.run(["bash", "-c", prefix + "\necho AFTER"], env=controlled, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected)
                self.assertEqual("CALLED" in result.stdout, expected == 2)
                self.assertEqual("AFTER" in result.stdout, expected == 0)
