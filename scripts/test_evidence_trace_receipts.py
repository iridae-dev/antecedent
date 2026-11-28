"""Regression checks for portable trace attribution, source currency and missing runs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evidence_links as links
from evidence_trace_receipts import TraceInputs, digest, observations
from test_scientific_records import Fixture


class SourceTests(Fixture):
    def test_key_binds_dependencies_but_reuses_unrelated_targets(self):
        src = self.root / "crates/top/tests/t.rs"
        def key():
            with patch("evidence_fingerprint.load_metadata", return_value=self.metadata):
                inputs = TraceInputs(self.root)
            return inputs.target_key("top", src, (), ["check_top"], [], {})
        original = key()
        self.write("crates/other/src/lib.rs", "fn other() { }")
        self.assertEqual(key(), original)
        self.write("crates/leaf/src/lib.rs", "fn new_leaf() { }")
        self.assertNotEqual(key(), original)
        other = key()
        self.write("crates/top/tests/t.rs", "#[test] fn check_new() {}")
        self.assertNotEqual(key(), other)


class ReceiptTests(unittest.TestCase):
    def test_missing_evidence_is_not_an_observed_refusal(self):
        self.assertEqual(links.record_link({}, "r", set()), ({}, False))
        facts = {"refused": ["matrix\tx"], "records": {"r": []}}
        entry, unadmitted = links.record_link(facts, "r", set())
        self.assertTrue(unadmitted)
        self.assertEqual(entry["refused"], {"matrix\tx"})

    def test_only_a_completed_target_can_supply_a_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.tsv"
            path.write_text("record\tsuite\ta\tr\n")
            with self.assertRaisesRegex(ValueError, "no successful test"):
                observations(path, "suite", {"b"})
            facts = observations(path, "suite", {"a"})
            receipt = {"status": "ok", "inputs": "first", **facts}
            receipt["digest"] = digest(receipt)
            self.assertTrue(links.current_receipt(receipt, "first"))
            self.assertFalse(links.current_receipt(receipt, "edited"))
            receipt["tests"]["a"]["records"]["r"] = ["capability\tz"]
            self.assertFalse(links.current_receipt(receipt, "first"))
            receipt["status"] = "timeout"
            receipt["digest"] = digest({k: v for k, v in receipt.items() if k != "digest"})
            self.assertFalse(links.current_receipt(receipt, "first"))

    def test_tallies_and_test_names_are_scoped_to_package_and_target(self):
        receipts = {"suite_01": {"tests": {"x::case": {"records": {"r": []}}}},
                    "other/suite_01": {"tests": {"x::case": {}}}}
        index = links.TestIndex(receipts)
        self.assertEqual(index.match("suite_01", "x::case"), [{"records": {"r": []}}])
        self.assertEqual(index.match("other/suite_01", "x::case"), [{}])
        self.assertEqual(index.match("suite_01", "different::case"), [])

    def test_a_measured_refusal_cannot_calibrate_the_declared_unit(self):
        facts = {"refused": ["capability\tz"], "records": {"r": ["capability\tz"]}}
        entry, unadmitted = links.record_link(facts, "r", {"capability\tz"})
        self.assertEqual(entry["refused"], {"capability\tz"})
        self.assertFalse(entry["declared"])
        self.assertFalse(entry["executed"])
        self.assertTrue(unadmitted)

    def test_an_owed_unit_can_be_witnessed_scientifically_on_successful_return(self):
        facts = {"refused": ["matrix\tx"], "returned": ["matrix\tx"], "records": {"r": []}}
        self.assertEqual(links.linked(facts)["returned"], {"matrix\tx"})
        measured, _ = links.record_link(facts, "r", set())
        self.assertFalse(measured["returned"])


if __name__ == "__main__":
    unittest.main()
