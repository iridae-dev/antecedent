"""Synthetic Criterion metadata tests; no benchmark or timing run."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gate_hot_path_baselines as gate


class CriterionMatching(unittest.TestCase):
    def test_parameterized_full_identity_and_unmatched_or_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "target/criterion/group/function/128/new"
            folder.mkdir(parents=True)
            (folder / "benchmark.json").write_text(json.dumps({"full_id": "group/function/128", "group_id": "group", "function_id": "function"}), encoding="utf-8")
            estimate = folder / "estimates.json"
            estimate.write_text(json.dumps({"mean": {"point_estimate": 500}}), encoding="utf-8")
            with patch.object(gate, "ROOT", root), patch.dict(gate.os.environ, {"GATE_CRITERION_MEANS": "1"}):
                self.assertEqual(gate.criterion_problems([("id", "group/function/128", 1.0, "us", "")]), [])
                self.assertTrue(any("matched no" in p for p in gate.criterion_problems([("id", "absent", 1.0, "us", "")])))
                estimate.write_text('{"mean":{"point_estimate":NaN}}', encoding="utf-8")
                self.assertTrue(any("nonfinite" in p for p in gate.criterion_problems([("id", "group/function/128", 1.0, "us", "")])))


if __name__ == "__main__":
    unittest.main()
