#!/usr/bin/env python3
"""Run the release-only scale cases of the acceptance manifest inside their frozen budgets.

A bound `#[ignore]` test that is not a calibration design closes its case only when it
runs here, in release mode, under the case's `runtime_seconds` wall-clock budget and
`memory_bytes` peak-resident budget. The test binary is built first so compilation is
not charged to the case.

    python3 scripts/run_scale_cases.py            # run every case (scripts/gate_scale_cases.sh)
    python3 scripts/run_scale_cases.py --self-test
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from typing import Any

from _repo import REPO_ROOT as ROOT

PASSED = re.compile(r"test result: ok\. (\d+) passed; 0 failed")


def command(case: dict[str, Any], *extra: str) -> list[str]:
    """The cargo invocation for `case`: its integration test target, or the crate's own tests."""
    selection = ["--test", case["target"]] if case["target"] else ["--lib"]
    return ["cargo", "test", "--release", "-p", case["package"], *selection, *extra]


def peak_bytes(maxrss: int) -> int:
    """`ru_maxrss` is kilobytes on Linux and bytes on macOS."""
    return maxrss if sys.platform == "darwin" else maxrss * 1024


def verdict(case: dict[str, Any], seconds: float, peak: int, status: int, output: str) -> list[str]:
    """Why the run does not close its case; empty when it does."""
    problems = []
    if status != 0:
        problems.append(f"{case['case']}: {case['test']} failed (exit {status})")
    elif not any(int(n) >= 1 for n in PASSED.findall(output)):
        problems.append(f"{case['case']}: {case['test']} ran no test")
    if seconds > case["runtime_seconds"]:
        problems.append(
            f"{case['case']}: ran {seconds:.0f}s, over the {case['runtime_seconds']}s budget"
        )
    if peak > case["memory_bytes"]:
        problems.append(
            f"{case['case']}: peaked at {peak} bytes, over the {case['memory_bytes']} byte budget"
        )
    return problems


def run(case: dict[str, Any]) -> list[str]:
    build = subprocess.run(command(case, "--no-run"), cwd=ROOT, capture_output=True, text=True)
    if build.returncode != 0:
        return [f"{case['case']}: build failed\n{build.stderr[-2000:]}"]
    start = time.monotonic()
    child = subprocess.Popen(
        command(case, case["test"], "--", "--ignored", "--nocapture"),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = child.stdout.read()
    _, status, usage = os.wait4(child.pid, 0)
    child.returncode = os.waitstatus_to_exitcode(status)
    return verdict(
        case, time.monotonic() - start, peak_bytes(usage.ru_maxrss), child.returncode, output
    )


def self_test() -> int:
    case = {
        "case": "c",
        "package": "p",
        "target": "t",
        "test": "x",
        "runtime_seconds": 10,
        "memory_bytes": 1000,
    }
    ok = "test result: ok. 1 passed; 0 failed; 0 ignored"
    failures = []
    if verdict(case, 5, 500, 0, ok):
        failures.append("a run inside its budgets was rejected")
    for label, args, needle in (
        ("slow", (11, 500, 0, ok), "budget"),
        ("large", (5, 5000, 0, ok), "byte budget"),
        ("failed", (5, 500, 101, ok), "failed"),
        ("empty", (5, 500, 0, "0 passed"), "ran no test"),
    ):
        if not any(needle in problem for problem in verdict(case, *args)):
            failures.append(f"{label} run was accepted")
    if command({**case, "target": None}, "x")[4:6] != ["p", "--lib"]:
        failures.append("a crate-internal test must select the library")
    print("run_scale_cases self-test:", "FAILED " + "; ".join(failures) if failures else "ok")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    cases = json.loads(
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_acceptance_manifest.py"), "--scale-cases"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    problems = [problem for case in cases for problem in run(case)]
    for problem in problems:
        print(f"FAIL: {problem}")
    print(f"scale cases: {len(cases)} run, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
