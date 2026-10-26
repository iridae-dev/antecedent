#!/usr/bin/env python3
"""Negative controls for capability scope, actual execution and freshness checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_python_capabilities import (
    MANIFEST,
    check,
    digest,
    evidence_problems,
    execution_problems,
    export_owners,
    rust_surface,
    source_digest,
)
from run_python_capabilities import JourneyRecorder


class CapabilityGateTests(unittest.TestCase):
    def test_aliases_require_exact_canonical_owner_not_a_shared_leaf(self):
        items = {
            "antecedent::family::Claim": {"kind": "type"},
            "antecedent::other::Claim": {"kind": "type"},
            "export:antecedent::api::Claim": {"kind": "reexport", "target": "super::family::Claim"},
            "export:antecedent::prelude::Claim": {
                "kind": "reexport",
                "target": "antecedent::api::Claim",
            },
            "export:antecedent::Unresolved": {"kind": "reexport", "target": "super::Claim"},
        }
        owners = export_owners(items)
        self.assertEqual(owners["export:antecedent::api::Claim"], "antecedent::family::Claim")
        self.assertEqual(owners["export:antecedent::prelude::Claim"], "antecedent::family::Claim")
        self.assertNotIn("export:antecedent::Unresolved", owners)

    def test_alias_cycle_does_not_establish_a_canonical_owner(self):
        items = {
            "export:antecedent::A": {"kind": "reexport", "target": "antecedent::B"},
            "export:antecedent::B": {"kind": "reexport", "target": "antecedent::A"},
        }
        self.assertEqual(export_owners(items), {})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "python/tests").mkdir(parents=True)
        (self.root / "parity/python_api_pass").mkdir(parents=True)
        (self.root / MANIFEST).write_text("schema=1\n")
        self.source = self.root / "python/tests/test_journey.py"
        self.source.write_text(
            "def helper():\n    assert 2 + 2 == 4\ndef test_live():\n    helper()\n"
        )
        self.evidence = {
            "test": "python/tests/test_journey.py::test_live",
            "assertion_scopes": ["helper"],
            "dimensions": ["success"],
            "assertion_contract": "Exact known arithmetic result.",
        }

    def execution(self):
        native = self.root / "native.so"
        native.write_bytes(b"actual extension fixture")
        return {
            "schema": 1,
            "source_digest": source_digest(self.root),
            "manifest_sha256": digest(self.root / MANIFEST),
            "native_path": str(native),
            "native_sha256": digest(native),
            "build_receipt_verified": True,
            "exit_code": 0,
            "selected": [self.evidence["test"]],
            "collected": [self.evidence["test"]],
            "tests": [
                {
                    "selector": self.evidence["test"],
                    "nodeid": self.evidence["test"],
                    "outcome": "passed",
                }
            ],
        }

    def manifest(self):
        return {"capability": [{"id": "live", "status": "covered", "evidence": [self.evidence]}]}

    def test_actual_helper_path_is_required(self):
        self.assertEqual(evidence_problems(self.root, self.evidence), [])
        self.evidence["assertion_scopes"] = ["unrelated"]
        self.assertTrue(evidence_problems(self.root, self.evidence))

    def test_assertion_cannot_be_replaced_with_export_count(self):
        self.source.write_text('def test_live():\n    print("exports exist")\n')
        self.evidence["assertion_scopes"] = ["test_live"]
        self.assertTrue(evidence_problems(self.root, self.evidence))

    def test_skipped_and_expected_failure_evidence_is_refused(self):
        for decorator in ["pytest.mark.skip", "pytest.mark.skipif(True)", "pytest.mark.xfail"]:
            with self.subTest(decorator=decorator):
                self.source.write_text(f"@{decorator}\ndef test_live():\n    assert True\n")
                self.evidence["assertion_scopes"] = ["test_live"]
                self.assertTrue(evidence_problems(self.root, self.evidence))

    def test_empty_selection_is_not_execution(self):
        run = self.execution()
        run["tests"] = []
        self.assertTrue(execution_problems(self.root, self.manifest(), run))

    def test_failed_skipped_or_ignored_execution_cannot_close(self):
        for outcome in ["failed", "skipped", "xfail", "incomplete"]:
            run = self.execution()
            run["tests"][0]["outcome"] = outcome
            self.assertTrue(execution_problems(self.root, self.manifest(), run))

    def test_a_collected_but_ignored_parameter_cannot_disappear(self):
        run = self.execution()
        run["collected"].append(self.evidence["test"] + "[ignored]")
        self.assertTrue(execution_problems(self.root, self.manifest(), run))

    def test_stale_extension_or_source_cannot_close(self):
        run = self.execution()
        self.assertEqual(execution_problems(self.root, self.manifest(), run), [])
        Path(run["native_path"]).write_bytes(b"replaced extension")
        self.assertTrue(execution_problems(self.root, self.manifest(), run))
        run = self.execution()
        self.source.write_text("def test_live():\n    assert False\n")
        self.assertTrue(execution_problems(self.root, self.manifest(), run))

    def test_runtime_cannot_promote_xpass_or_teardown_failure(self):
        recorder = JourneyRecorder(["test.py::test_live"])

        class Report:
            nodeid = "test.py::test_live"
            failed = False
            skipped = False
            when = "call"
            outcome = "passed"

        report = Report()
        recorder.pytest_runtest_logreport(report)
        report.when, report.failed, report.outcome = "teardown", True, "failed"
        recorder.pytest_runtest_logreport(report)
        self.assertEqual(recorder.tests[report.nodeid]["outcome"], "failed")
        report.failed, report.when, report.outcome, report.wasxfail = (
            False,
            "call",
            "passed",
            "known",
        )
        recorder.pytest_runtest_logreport(report)
        self.assertEqual(recorder.tests[report.nodeid]["outcome"], "xfail")

    def test_completion_cannot_ignore_a_named_gap_or_unmapped_method(self):
        (self.root / "parity/public_api").mkdir()
        snapshot = "pub fn <antecedent::Study>::fn inspect(&self) -> Report\n"
        (self.root / "parity/public_api/antecedent.txt").write_text(snapshot)
        (self.root / "parity/python_facade_surface.toml").write_text("entry=[]\n")
        manifest = {
            "schema": 1,
            "capability": [
                {
                    "id": "open",
                    "rust": ["<antecedent::Study>::inspect"],
                    "status": "gap",
                    "owner": "integration",
                    "behavior": "inspect without work",
                    "gap": "No asserting Python consumer yet",
                    "required_dimensions": ["inspection"],
                }
            ],
        }
        inventory = {"schema": 1, "items": rust_surface(snapshot)}
        problems, gaps = check(self.root, manifest, inventory, complete=True)
        self.assertTrue(gaps)
        self.assertTrue(any("No asserting Python consumer" in p for p in problems))
        manifest["capability"] = []
        problems, gaps = check(self.root, manifest, inventory, complete=True)
        self.assertTrue(any("unmapped Rust method" in p for p in problems))

    def test_inherent_methods_and_export_edges_are_inventory(self):
        snapshot = (
            "pub fn <antecedent::Study>::fn inspect(&self) -> Report\n"
            "pub use antecedent::Renamed = crate::Actual\n"
        )
        items = rust_surface(snapshot)
        self.assertIn("<antecedent::Study>::inspect", items)
        self.assertEqual(items["export:antecedent::Renamed"]["target"], "crate::Actual")


if __name__ == "__main__":
    unittest.main()
