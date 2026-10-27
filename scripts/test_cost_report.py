"""Fault controls: a cost comparison is controlled only when conditions match."""

import json
import tempfile
import unittest
from pathlib import Path

import cost_report as report

RECEIPT = {
    "command": ["bash", "scripts/test.sh"], "exit_code": 0, "wall_seconds": 10.0,
    "user_seconds": 30.0, "system_seconds": 2.0, "settings": {"A": "1"}, "platform": "p", "cpus": 4,
    "checkout_before": {"commit": "c", "changed_inputs": {}},
    "checkout_after": {"commit": "c", "changed_inputs": {}},
}
JUNIT = {"cases": 5, "skipped": 0, "failed": 0, "sum_of_case_seconds": 4.0, "sha256": "x", "suite_seconds": 4.0}


class CostReport(unittest.TestCase):
    def test_matching_clean_passing_runs_are_controlled_and_report_signed_differences(self):
        result = report.compare(RECEIPT, {**RECEIPT, "wall_seconds": 8.0}, JUNIT, {**JUNIT, "sum_of_case_seconds": 3.0})
        self.assertTrue(result["controlled_comparison"])
        self.assertEqual(result["raw_differences_after_minus_before"]["wall_seconds"], -2.0)
        self.assertEqual(result["raw_differences_after_minus_before"]["sum_of_case_seconds"], -1.0)

    def test_any_mismatch_failure_dirty_checkout_or_missing_report_forfeits_the_claim(self):
        moved = {"commit": "d", "changed_inputs": {}}
        faults = {
            "settings": ({**RECEIPT, "settings": {"A": "2"}}, JUNIT),
            "failed run": ({**RECEIPT, "exit_code": 1}, JUNIT),
            "uncommitted": ({**RECEIPT, "checkout_before": {"commit": "c", "changed_inputs": {"f": "h"}}}, JUNIT),
            "moved during run": ({**RECEIPT, "checkout_after": moved}, JUNIT),
            "other commit": ({**RECEIPT, "checkout_before": moved, "checkout_after": moved}, JUNIT),
            "changed count": (RECEIPT, {**JUNIT, "cases": 4}),
            "skipped case": (RECEIPT, {**JUNIT, "skipped": 1}),
            "no report": (RECEIPT, None),
        }
        for name, (after, junit) in faults.items():
            with self.subTest(name):
                result = report.compare(RECEIPT, after, JUNIT, junit)
                self.assertFalse(result["controlled_comparison"])
                self.assertTrue(result["reasons_not_controlled"])
                self.assertIn("neither an improvement nor a regression", result["interpretation"])

    def test_unmeasured_values_stay_null_and_incomplete_receipts_are_rejected(self):
        bare = {k: v for k, v in RECEIPT.items() if k not in ("user_seconds", "system_seconds")}
        diff = report.compare(bare, bare, None, None)["raw_differences_after_minus_before"]
        self.assertIsNone(diff["user_seconds"])
        self.assertIsNone(diff["sum_of_case_seconds"])
        with self.assertRaisesRegex(ValueError, "lacks required"):
            report.compare({"command": []}, RECEIPT)

    def test_junit_reader_counts_cases_and_rejects_an_empty_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.xml"
            path.write_text(
                '<testsuites time="1"><testsuite><testcase name="a" time="0.5"/>'
                '<testcase name="b" time="0.25"><skipped/></testcase>'
                '<testcase name="c"><failure/></testcase></testsuite></testsuites>', encoding="utf-8")
            facts = report.junit_facts(path)
            self.assertEqual((facts["cases"], facts["skipped"], facts["failed"], facts["sum_of_case_seconds"]), (3, 1, 1, 0.75))
            path.write_text("<testsuites/>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no test cases"):
                report.junit_facts(path)


if __name__ == "__main__":
    unittest.main()
