# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mkdocs==1.6.1", "PyYAML==6.0.3", "numpy==2.1.3", "pytest==9.1.1"]
# ///
"""Run discovered script unit tests in a retained, isolated tooling environment.

No calibration or performance campaign is launched. Two native performance
receipt consumers remain explicitly pending unless their real probe is supplied.
Every other skip, missing discovery or failed assertion fails this command.
"""

from __future__ import annotations

import argparse
import json
import os
import unittest
from pathlib import Path

from _repo import REPO_ROOT
from test_run_receipt import checkout

PENDING_PROBE_CONSUMERS = {
    "test_measurement_record.RustEmittedMeasurementTest.test_the_native_summary_is_accepted_and_shows_the_work_that_ran",
    "test_measurement_record.RustEmittedMeasurementTest.test_the_native_summary_rides_on_a_criterion_comparison",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument(
        "--exclude-module",
        action="append",
        default=[],
        help="Explicitly deselect a module executed by another job; recorded in the receipt",
    )
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "target/script-tests/latest.json"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    before = checkout(REPO_ROOT)
    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "scripts"), pattern=args.pattern
    )

    def cases(group):
        for test in group:
            if isinstance(test, unittest.TestSuite):
                yield from cases(test)
            else:
                yield test

    discovered_cases = list(cases(suite))
    excluded = set(args.exclude_module)
    unknown = excluded - {test.id().split(".")[0] for test in discovered_cases}
    if unknown:
        parser.error(f"excluded modules were not discovered: {sorted(unknown)}")
    deselected = [
        test.id() for test in discovered_cases if test.id().split(".")[0] in excluded
    ]
    suite = unittest.TestSuite(
        test for test in discovered_cases if test.id() not in deselected
    )
    discovered = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    allowed = (
        PENDING_PROBE_CONSUMERS
        if os.environ.get("ANTECEDENT_REQUIRE_NATIVE_MEASUREMENT") != "1"
        else set()
    )
    unexpected = [test.id() for test, _ in result.skipped if test.id() not in allowed]
    after = checkout(REPO_ROOT)
    report = {
        "schema": 1,
        "activity": "ordinary_script_unit_tests",
        "pattern": args.pattern,
        "discovered_before_selection": len(discovered_cases),
        "discovered": discovered,
        "explicit_deselections": deselected,
        "executed": result.testsRun,
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "skipped": [
            {
                "assertion": test.id(),
                "reason": reason,
                "activity": "pending_external_performance_probe"
                if test.id() in allowed
                else "unexpected_skip",
            }
            for test, reason in result.skipped
        ],
        "unexpected_skips": unexpected,
        "checkout_before": before,
        "checkout_after": after,
        "checkout_unchanged": before == after,
        "calibration_measurements_run": False,
        "performance_acceptance_run": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not discovered:
        print("FAIL: script test discovery selected no tests")
    if unexpected:
        print(f"FAIL: required script assertions skipped: {unexpected}")
    print(f"Script test execution receipt: {args.report}")
    return 0 if discovered and result.wasSuccessful() and not unexpected else 1


if __name__ == "__main__":
    raise SystemExit(main())
