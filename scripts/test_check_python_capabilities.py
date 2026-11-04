#!/usr/bin/env python3
"""Negative controls for capability scope, actual execution and freshness checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from check_python_capabilities import (
    MANIFEST,
    SURFACE,
    check,
    digest,
    evidence_problems,
    execution_problems,
    rust_surface,
    source_digest,
    write_surface,
)
from run_python_capabilities import JourneyRecorder


class CapabilityGateTests(unittest.TestCase):
    def test_regeneration_replaces_stale_entries_and_is_deterministic(self):
        target = self.root / SURFACE
        target.write_text('{"items": {"stale": {}}}', encoding="utf-8")
        items = {
            "antecedent::New": {
                "kind": "type",
                "declaration": "pub struct antecedent::New",
            }
        }
        with patch("check_python_capabilities.current_surface", return_value=items) as source:
            write_surface(self.root)
            first = target.read_bytes()
            write_surface(self.root)
        self.assertEqual(source.call_args.args, (self.root,))
        self.assertEqual(json.loads(first)["items"], items)
        self.assertEqual(target.read_bytes(), first)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "python/tests").mkdir(parents=True)
        (self.root / "parity/python_api_pass").mkdir(parents=True)
        (self.root / MANIFEST).write_text("schema=1\n", encoding="utf-8")
        self.source = self.root / "python/tests/test_journey.py"
        self.source.write_text(
            "def helper():\n    assert 2 + 2 == 4\ndef test_live():\n    helper()\n",
            encoding="utf-8",
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
        self.source.write_text('def test_live():\n    print("exports exist")\n', encoding="utf-8")
        self.evidence["assertion_scopes"] = ["test_live"]
        self.assertTrue(evidence_problems(self.root, self.evidence))

    def test_skipped_and_expected_failure_evidence_is_refused(self):
        for decorator in [
            "pytest.mark.skip",
            "pytest.mark.skipif(True)",
            "pytest.mark.xfail",
        ]:
            with self.subTest(decorator=decorator):
                self.source.write_text(
                    f"@{decorator}\ndef test_live():\n    assert True\n",
                    encoding="utf-8",
                )
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
        self.source.write_text("def test_live():\n    assert False\n", encoding="utf-8")
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

    def test_completion_cannot_ignore_a_named_gap(self):
        (self.root / "parity/public_api").mkdir()
        snapshot = "pub fn <antecedent::Study>::fn inspect(&self) -> Report\n"
        (self.root / "parity/public_api/antecedent.txt").write_text(snapshot, encoding="utf-8")
        (self.root / "parity/python_facade_surface.toml").write_text("entry=[]\n", encoding="utf-8")
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

    def test_a_rust_pointer_must_resolve_but_its_absence_is_never_a_finding(self):
        (self.root / "parity/public_api").mkdir()
        snapshot = (
            "pub struct antecedent::Study\n"
            "pub fn <antecedent::Study>::fn inspect(&self) -> Report\n"
            "pub struct antecedent::StudyBuilder\n"
        )
        (self.root / "parity/public_api/antecedent.txt").write_text(snapshot, encoding="utf-8")
        (self.root / "parity/python_facade_surface.toml").write_text("entry=[]\n", encoding="utf-8")
        row = {
            "id": "study",
            "status": "gap",
            "owner": "integration",
            "behavior": "study lifecycle",
            "gap": "open",
            "required_dimensions": ["inspection"],
        }
        inventory = {"schema": 1, "items": rust_surface(snapshot)}
        manifest = {"schema": 1, "capability": [row]}
        problems, gaps = check(self.root, manifest, inventory)
        self.assertEqual(problems, [])
        self.assertEqual(gaps, ["study: open"])
        row["rust_types"] = ["antecedent::Missing"]
        problems, _ = check(self.root, manifest, inventory)
        self.assertTrue(any("unknown Rust type antecedent::Missing" in p for p in problems))
        row["rust_types"] = []
        row["rust"] = ["<antecedent::Study>::missing"]
        problems, _ = check(self.root, manifest, inventory)
        self.assertTrue(any("unknown Rust symbol" in p for p in problems))

    def semantic_setup(self):
        """A facade with an unmapped binding-used item, one executing row and a universe."""
        patcher = patch("check_python_surface.PythonIndex.resolves", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        (self.root / "parity/public_api").mkdir()
        snapshot = "pub fn <antecedent::Study>::fn inspect(&self) -> Report\n"
        (self.root / "parity/public_api/antecedent.txt").write_text(snapshot, encoding="utf-8")
        (self.root / "parity/python_facade_surface.toml").write_text("entry=[]\n", encoding="utf-8")
        self.source.write_text(
            "def test_live():\n    assert Study().inspect() == 4\n", encoding="utf-8"
        )
        self.evidence["assertion_scopes"] = ["test_live"]
        row = {
            "id": "inspect",
            "status": "covered",
            "owner": "integration",
            "behavior": "inspect a study",
            "python": "antecedent",
            "required_dimensions": ["success"],
            "evidence": [self.evidence],
        }
        inventory = {"schema": 1, "items": rust_surface(snapshot)}
        return {"schema": 1, "capability": [row]}, inventory

    def test_a_binding_used_rust_item_without_a_row_is_not_a_gap(self):
        manifest, inventory = self.semantic_setup()
        universe = [{"id": "capability:inspect", "names": ["Study"]}]
        problems, gaps = check(self.root, manifest, inventory, complete=False, universe=universe)
        self.assertEqual((problems, gaps), ([], []))
        # The Rust method has no row anywhere and no pointer from this row: still no gap.
        self.assertNotIn("inspect", " ".join(manifest["capability"][0].get("rust", [])))

    def test_a_supported_capability_with_no_python_journey_is_a_gap(self):
        manifest, inventory = self.semantic_setup()
        universe = [
            {"id": "capability:inspect", "names": ["Study"]},
            {"id": "query:Absent", "names": ["Absent"]},
        ]
        problems, gaps = check(self.root, manifest, inventory, universe=universe)
        self.assertEqual(problems, [])
        self.assertEqual(len(gaps), 1)
        self.assertTrue(gaps[0].startswith("query:Absent"))
        problems, _ = check(self.root, manifest, inventory, complete=True, universe=universe)
        self.assertTrue(any(p.startswith("query:Absent") for p in problems))
        # An explicit statement still needs an executing row, not a bare assertion.
        manifest["capability"][0]["supports"] = ["query:Absent"]
        _, gaps = check(self.root, manifest, inventory, universe=universe)
        self.assertEqual(gaps, [])
        manifest["capability"][0]["supports"] = ["query:Unknown"]
        problems, _ = check(self.root, manifest, inventory, universe=universe)
        self.assertTrue(any("supports unknown capability unit" in p for p in problems))

    def test_a_rust_only_reason_row_is_not_a_gap(self):
        manifest, inventory = self.semantic_setup()
        universe = [{"id": "query:Absent", "names": ["Absent"]}]
        manifest["capability"].append(
            {
                "id": "absent-by-design",
                "status": "rust_only",
                "owner": "integration",
                "behavior": "an in-process hook with no meaning outside Rust",
                "supports": ["query:Absent"],
                "rust_only_reason": "Needs a Rust trait object; Python callers use the callback API.",
            }
        )
        problems, gaps = check(self.root, manifest, inventory, universe=universe)
        self.assertEqual((problems, gaps), ([], []))
        manifest["capability"][1]["rust_only_reason"] = " "
        problems, gaps = check(self.root, manifest, inventory, universe=universe)
        self.assertTrue(any("records why Python does not offer it" in p for p in problems))
        self.assertEqual(len(gaps), 1)

    def test_a_row_exercises_a_unit_by_the_python_name_its_test_calls(self):
        manifest, inventory = self.semantic_setup()
        universe = [
            {"id": "query:Study", "names": ["Study"]},
            {"id": "capability:routed", "routes": ["antecedent.module.Study.inspect"]},
            {"id": "capability:other", "routes": ["antecedent.module.Other.run"]},
        ]
        _, gaps = check(self.root, manifest, inventory, universe=universe)
        self.assertEqual([g.split(":")[0] + ":" + g.split(":")[1] for g in gaps], ["capability:other"])

    def test_a_deferred_row_still_needs_executing_evidence_for_what_it_does_not_wait_for(self):
        (self.root / "parity/public_api").mkdir()
        snapshot = "pub fn <antecedent::Study>::fn inspect(&self) -> Report\n"
        (self.root / "parity/public_api/antecedent.txt").write_text(snapshot, encoding="utf-8")
        (self.root / "parity/python_facade_surface.toml").write_text("entry=[]\n", encoding="utf-8")
        row = {
            "id": "measured",
            "rust": ["<antecedent::Study>::inspect"],
            "status": "deferred",
            "owner": "integration",
            "behavior": "a measured interval",
            "required_dimensions": ["success", "refusal"],
            "deferred_dimensions": ["success"],
            "deferred_to": "calibration measurement of the coverage record",
            "python": "antecedent",
        }
        inventory = {"schema": 1, "items": rust_surface(snapshot)}
        problems, gaps = check(self.root, {"schema": 1, "capability": [row]}, inventory)
        self.assertTrue(any("no executing evidence" in p for p in problems))
        self.assertFalse([gap for gap in gaps if gap.startswith("measured")])
        row["deferred_dimensions"] = ["success", "refusal"]
        problems, _ = check(self.root, {"schema": 1, "capability": [row]}, inventory)
        self.assertTrue(any("strict subset" in p for p in problems))
        row["deferred_dimensions"] = ["success"]
        row["deferred_to"] = ""
        problems, _ = check(self.root, {"schema": 1, "capability": [row]}, inventory)
        self.assertTrue(any("names what it waits for" in p for p in problems))

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
