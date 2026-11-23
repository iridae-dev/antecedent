#!/usr/bin/env python3
"""Compare two ordinary test-run receipts and state only what was measured.

Inputs are receipts written by ``scripts/test_run_receipt.py`` and, optionally,
the nextest JUnit report each run produced. The report lists each side's
measurements, the conditions that differ between the runs, and the raw
differences. A difference is called a controlled comparison only when the
command, settings, host, selected test count and checkout all match, both runs
passed, and neither checkout changed during its run. Otherwise the report says
no improvement or regression is established. A missing measurement is reported
as ``null``; nothing is estimated, and build, discovery, lock-wait and
execution time are not separated unless the inputs separate them.

    python3 scripts/cost_report.py BEFORE.json AFTER.json \
        [--before-junit B.xml] [--after-junit A.xml] [--output report.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CONDITION_FIELDS = ("command", "settings", "platform", "cpus")


def junit_facts(path: Path) -> dict:
    data = path.read_bytes()
    root = ET.fromstring(data)
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError(f"{path}: JUnit report contains no test cases")
    skipped = sum(1 for case in cases if case.find("skipped") is not None)
    failed = sum(1 for case in cases if case.find("failure") is not None or case.find("error") is not None)
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "cases": len(cases),
        "skipped": skipped,
        "failed": failed,
        "sum_of_case_seconds": round(sum(float(case.get("time", 0)) for case in cases), 3),
        "suite_seconds": float(root.get("time")) if root.get("time") is not None else None,
    }


def side(receipt: dict, junit: dict | None) -> dict:
    missing = [key for key in ("command", "exit_code", "wall_seconds", "checkout_before", "checkout_after") if key not in receipt]
    if missing:
        raise ValueError(f"receipt lacks required fields: {missing}")
    return {
        "command": receipt["command"],
        "exit_code": receipt["exit_code"],
        "wall_seconds": receipt["wall_seconds"],
        "user_seconds": receipt.get("user_seconds"),
        "system_seconds": receipt.get("system_seconds"),
        "peak_child_rss_bytes": receipt.get("peak_child_rss_bytes"),
        "settings": receipt.get("settings"),
        "platform": receipt.get("platform"),
        "cpus": receipt.get("cpus"),
        "commit": receipt["checkout_before"].get("commit"),
        "checkout_unchanged": receipt["checkout_before"] == receipt["checkout_after"],
        "dirty_inputs": len(receipt["checkout_before"].get("changed_inputs", {})),
        "junit": junit,
    }


def delta(before, after):
    if before is None or after is None:
        return None
    return round(after - before, 3)


def compare(before_receipt: dict, after_receipt: dict, before_junit: dict | None = None, after_junit: dict | None = None) -> dict:
    before, after = side(before_receipt, before_junit), side(after_receipt, after_junit)
    differences = [field for field in CONDITION_FIELDS if before[field] != after[field]]
    problems = [f"differs: {field}" for field in differences]
    if before["commit"] != after["commit"]:
        problems.append("differs: checkout commit")
    for name, run in (("before", before), ("after", after)):
        if run["exit_code"] != 0:
            problems.append(f"{name} run did not pass (exit {run['exit_code']})")
        if not run["checkout_unchanged"]:
            problems.append(f"{name} checkout changed during the run")
        if run["dirty_inputs"]:
            problems.append(f"{name} run had {run['dirty_inputs']} uncommitted inputs")
        if run["junit"] is None:
            problems.append(f"{name} has no JUnit report, so selected tests and execution time are not measured")
    if before["junit"] and after["junit"]:
        if before["junit"]["cases"] != after["junit"]["cases"]:
            problems.append("selected test counts differ")
        for name, run in (("before", before), ("after", after)):
            if run["junit"]["skipped"] or run["junit"]["failed"]:
                problems.append(f"{name} report contains skipped or failed cases")
    controlled = not problems
    exec_b = before["junit"]["sum_of_case_seconds"] if before["junit"] else None
    exec_a = after["junit"]["sum_of_case_seconds"] if after["junit"] else None
    return {
        "schema": 1,
        "activity": "ordinary_test_cost_comparison_not_acceptance",
        "before": before,
        "after": after,
        "controlled_comparison": controlled,
        "reasons_not_controlled": problems,
        "raw_differences_after_minus_before": {
            "wall_seconds": delta(before["wall_seconds"], after["wall_seconds"]),
            "user_seconds": delta(before["user_seconds"], after["user_seconds"]),
            "system_seconds": delta(before["system_seconds"], after["system_seconds"]),
            "sum_of_case_seconds": delta(exec_b, exec_a),
            "selected_cases": delta(
                before["junit"]["cases"] if before["junit"] else None,
                after["junit"]["cases"] if after["junit"] else None,
            ),
        },
        "interpretation": (
            "Single runs under matching conditions; the sign of each difference is observed, its size is not "
            "a variance-qualified estimate. Wall time includes discovery, build and lock waits that are not separated here."
            if controlled
            else "Not a controlled comparison; the raw differences are recorded but establish neither an improvement nor a regression."
        ),
    }


def render(report: dict) -> str:
    lines = [f"controlled comparison: {'yes' if report['controlled_comparison'] else 'no'}"]
    lines += [f"  - {reason}" for reason in report["reasons_not_controlled"]]
    for key, value in report["raw_differences_after_minus_before"].items():
        lines.append(f"{key} (after - before): {'not measured' if value is None else value}")
    lines.append(report["interpretation"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", nargs="?", type=Path)
    parser.add_argument("after", nargs="?", type=Path)
    parser.add_argument("--before-junit", type=Path)
    parser.add_argument("--after-junit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.before or not args.after:
        parser.error("two receipts are required")
    report = compare(
        json.loads(args.before.read_text(encoding="utf-8")),
        json.loads(args.after.read_text(encoding="utf-8")),
        junit_facts(args.before_junit) if args.before_junit else None,
        junit_facts(args.after_junit) if args.after_junit else None,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
