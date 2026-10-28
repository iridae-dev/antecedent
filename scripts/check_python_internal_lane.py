#!/usr/bin/env python3
"""Keep the `python-internal` lane honest.

Six Python test modules exercise private candidate hooks that exist only in a
native extension built with the `calibration-internal` feature, and skip
everywhere else. `bash scripts/test.sh python-internal` builds that extension
into an isolated environment and runs exactly these modules. A lane that ran
them against the ordinary extension would pass while every assertion skipped,
so this script reads the pytest JUnit report and fails when a module in the lane

  * is absent from the report or collected no tests,
  * passed no test, or
  * skipped a test because the internal hook was missing.

It also fails on a stale list: every module under python/tests whose skip reason
names the internal feature must be in `MODULES`. Neither command runs a test.

    python3 scripts/check_python_internal_lane.py sources
    python3 scripts/check_python_internal_lane.py junit REPORT.xml
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from _repo import REPO_ROOT

MODULES = (
    "test_joint_transport_candidate_lifecycle",
    "test_checked_temporal_interval",
    "test_nested_fisher_activation",
    "test_nested_bayesian_activation",
    "test_temporal_interval_activation",
    "test_sampled_recovery_activation",
)
# The skip reasons the six modules give when the internal hook is missing. A skip
# for any other reason (a normal-wheel refusal check) is by design in this lane.
MISSING_HOOK = re.compile(
    r"calibration-internal|internal candidate lifecycle is not a released positive"
    r"|private candidate hook"
)
SKIP_CALL = re.compile(r"skipif\(|pytest\.skip\(|importorskip\(")


def module_of(case: ET.Element) -> str | None:
    """`test_x` from a JUnit `classname` (`tests.test_x` or `tests.test_x.Class`)."""
    for part in (case.get("classname") or "").split("."):
        if part in MODULES:
            return part
    return None


def junit_problems(report: str) -> list[str]:
    try:
        root = ET.fromstring(report)
    except ET.ParseError as error:
        return [f"the JUnit report is not parseable: {error}"]
    counts = {m: {"collected": 0, "passed": 0, "bad": 0, "hook": []} for m in MODULES}
    for case in root.iter("testcase"):
        module = module_of(case)
        if module is None:
            continue
        entry = counts[module]
        entry["collected"] += 1
        skipped = case.find("skipped")
        if skipped is not None:
            text = f"{skipped.get('message', '')} {skipped.text or ''}"
            if MISSING_HOOK.search(text):
                entry["hook"].append(case.get("name") or "?")
        elif case.find("failure") is not None or case.find("error") is not None:
            entry["bad"] += 1
        else:
            entry["passed"] += 1
    problems = []
    for module, entry in counts.items():
        if entry["collected"] == 0:
            problems.append(f"{module}: no test was collected in the python-internal lane")
        elif entry["hook"]:
            problems.append(
                f"{module}: {len(entry['hook'])} test(s) skipped because the internal hook is "
                f"missing ({', '.join(sorted(entry['hook'])[:3])}); the extension was not built "
                "with calibration-internal"
            )
        elif entry["passed"] == 0 and entry["bad"] == 0:
            problems.append(f"{module}: every collected test skipped; none ran")
    return problems


def source_problems(root: Path) -> list[str]:
    problems = []
    tests = root / "python/tests"
    for module in MODULES:
        path = tests / f"{module}.py"
        if not path.is_file():
            problems.append(f"{module}: python/tests/{module}.py does not exist")
        elif not MISSING_HOOK.search(path.read_text(encoding="utf-8")):
            problems.append(f"{module}: no skip reason names the internal feature any more; drop it from MODULES")
    for path in sorted(tests.glob("test_*.py")):
        if path.stem in MODULES:
            continue
        text = path.read_text(encoding="utf-8")
        for match in SKIP_CALL.finditer(text):
            window = text[match.start(): match.start() + 300]
            if re.search(r"reason\s*=\s*[^)]*(requires calibration-internal|private candidate hook|internal candidate lifecycle)", window):
                problems.append(
                    f"{path.stem}: skips on the internal feature but is in no lane; add it to MODULES"
                )
                break
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sources")
    junit = sub.add_parser("junit")
    junit.add_argument("report", type=Path)
    sub.add_parser("modules")
    args = parser.parse_args(argv)
    if args.command == "modules":
        print("\n".join(f"python/tests/{m}.py" for m in MODULES))
        return 0
    if args.command == "sources":
        problems = source_problems(REPO_ROOT)
    else:
        problems = junit_problems(args.report.read_text(encoding="utf-8"))
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
