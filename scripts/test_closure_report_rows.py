"""Closure report: every unresolved row is listed and the release condition is zero rows.

Module-level functions so a scenario cell can cite one by name; `load_tests` hands them to the
unittest discovery that scripts/run_script_tests.py uses.
"""

from __future__ import annotations

import unittest

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import closure_audit as audit  # noqa: E402

CHILD = {"id": "FAM.method.static.bayesian", "gap": "none"}


def _record(identifier: str, status: str, missing: list[str]) -> dict:
    return {
        "id": identifier, "list": "mandatory", "mandatory": True, "status": status,
        "missing": missing,
    }


def test_the_report_lists_every_unresolved_row_and_the_release_fails_on_each():
    records = [
        _record("OPEN.static.bayesian", "open", ["no consumer is named"]),
        _record("OTHER.static.bayesian", "open", ["no fixture is named"]),
        _record("SHUT.static.frequentist", "reconciled", []),
    ]
    report = audit.audit([CHILD], {}, [], records)
    assert [r["id"] for r in report["unresolved_mandatory"]] == [
        "OPEN.static.bayesian", "OTHER.static.bayesian",
    ]
    findings = audit.release_findings(report)
    assert any("OPEN.static.bayesian" in line for line in findings)
    assert any("OTHER.static.bayesian" in line for line in findings)


def test_a_report_with_every_row_resolved_has_no_release_finding():
    records = [_record("SHUT.static.frequentist", "reconciled", [])]
    report = audit.audit([CHILD], {}, [], records)
    assert report["unresolved_mandatory"] == []
    assert audit.release_findings(report) == []


_CASES = [
    test_the_report_lists_every_unresolved_row_and_the_release_fails_on_each,
    test_a_report_with_every_row_resolved_has_no_release_finding,
]


def load_tests(loader, tests, pattern):  # noqa: ARG001
    suite = unittest.TestSuite()
    suite.addTests(unittest.FunctionTestCase(case) for case in _CASES)
    return suite


if __name__ == "__main__":
    unittest.main()
