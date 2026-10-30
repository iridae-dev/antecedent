import json
from pathlib import Path
import tempfile
import unittest
import collect_coverage_records as collector

class DuplicateRecords(unittest.TestCase):
    def test_distinct_replicate_counts_refused_and_recheck_replaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.p0.log", root / "b.p0.log"
            def line(n):
                return "calibration-record " + json.dumps({"id":"x", "grid_point":0, "test":"t", "replicates":n}) + "\n"
            first.write_text(line(100))
            second.write_text(line(200))
            with self.assertRaisesRegex(SystemExit, "duplicate record"):
                collector.record_lines([first, second])
            recheck = root / "a.p0.recheck.log"
            recheck.write_text(line(200))
            self.assertEqual(collector.record_lines([first, recheck])["x"][0]["replicates"], 200)
            recheck.write_text(line(100) + line(200))
            with self.assertRaisesRegex(SystemExit, "duplicate record"):
                collector.record_lines([first, recheck])
