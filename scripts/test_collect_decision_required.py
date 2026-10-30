import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import collect_coverage_records as collector

class CollectionDecisionRequired(unittest.TestCase):
    def test_marker_free_records_refused_and_diagnostics_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "family.p0.log"
            def write(record, context=True):
                log.write_text(("calibration-decision-context {}\n" if context else "") +
                               "calibration-record " + json.dumps(record) + "\n")
            write({"id":"x"}, False)
            with self.assertRaises(SystemExit):
                collector.require_frozen_collection_decisions([log])
            write({"id":"x"})
            with self.assertRaises(SystemExit):
                collector.require_frozen_collection_decisions([log])
            write({"id":"x", "role":"named_boundary"})
            collector.require_frozen_collection_decisions([log])
            write({"id":"x", "decision_protocol":{}})
            with patch.object(collector, "validate_point_decision") as validate:
                collector.require_frozen_collection_decisions([log])
                validate.assert_called_once()
            confirmation = Path(tmp) / "family.p0.confirm.log"
            confirmation.write_text(log.read_text())
            with self.assertRaises(SystemExit):
                collector.require_frozen_collection_decisions([confirmation])

    def test_legacy_inspection_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "old.log"
            log.write_text("test result: ok. 1 passed; 0 failed; 0 ignored\n")
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            script = Path(collector.__file__).resolve()
            result = subprocess.run([sys.executable, str(script), "--inspect-legacy-logs", "--log-dir", str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("legacy inspection only", result.stdout)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            bad = subprocess.run([sys.executable, str(script), "--inspect-legacy-logs", "--keep-attested", "--log-dir", str(root)], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
