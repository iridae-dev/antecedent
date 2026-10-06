#!/usr/bin/env python3
"""2.2 A exit gate: join the executed story results into one per-story table and a verdict.

Inputs are the captured logs of the two story suites and the JSON of
scripts/check_a_intervals.py; this script decides nothing about the science, only whether every
story passed and whether the calibrated parts are measured:

    python3 scripts/a_exit_report.py --rust rust.log --python py.log --intervals iv.json \
        [--require-calibrated]
    python3 scripts/a_exit_report.py --self-test

Per-story status:
  PASS                   the Rust and the Python story both ran and passed
  FAIL                   a story failed, errored, or did not run
  PENDING_CALIBRATION    story 3's estimate/overlap/artifact parts pass, but its calibration
                         coverage records are absent (not pass, not fail)

Exit status: 1 on any FAIL; PENDING_CALIBRATION exits 0 unless `--require-calibrated`.
The X1 coverage records (not one of the six stories, but part of "zero unmeasured interval
coordinates") are reported as their own row and follow the same rule.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

STORIES = {
    1: "multi-source limited experiment (X1)",
    2: "finite scenario set: identified/unidentified/unevaluated (X2)",
    3: "learned continuous transport, overlap-supported (X4)",
    4: "two-step temporal transport, history-support refusal (X5)",
    5: "cross-world query, shared abduction, typed refusal (X8)",
    6: "mixed-source proof by bounded search + incomplete search (X9)",
}
RUST = re.compile(r"^test (?:\S+::)?story_(\d)_\w+ \.\.\. (ok|FAILED|ignored)\s*$", re.M)
PYTHON = re.compile(r"(?:^|/)test_a_exit_gate\.py::test_story_(\d)_\w+(?:\[[^\]]*\])?\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)", re.M)


def outcomes(pattern: re.Pattern[str], log: str, passing: set[str]) -> dict[int, str]:
    seen: dict[int, list[bool]] = {}
    for number, verdict in pattern.findall(log):
        seen.setdefault(int(number), []).append(verdict in passing)
    return {n: ("PASS" if all(v) else "FAIL") for n, v in seen.items()}


def evaluate(rust_log: str, python_log: str, intervals: dict, require_calibrated: bool) -> tuple[list[str], int]:
    rust = outcomes(RUST, rust_log, {"ok"})
    python = outcomes(PYTHON, python_log, {"PASSED"})
    calibration = intervals.get("calibration", {})
    x4 = calibration.get("X4", {"status": "FAIL", "detail": "no calibration state reported"})
    x1 = calibration.get("X1", {"status": "FAIL", "detail": "no calibration state reported"})
    rows: list[tuple[str, str, str, str, str]] = []
    for number, title in STORIES.items():
        r, p = rust.get(number, "MISSING"), python.get(number, "MISSING")
        status = "PASS" if (r, p) == ("PASS", "PASS") else "FAIL"
        detail = ""
        if status == "PASS" and number == 3:
            status = x4["status"]
            detail = f"calibrated: {x4['status']} ({x4['detail']})"
        rows.append((f"story {number}", title, r, p, status + (f"  [{detail}]" if detail else "")))
    rows.append((
        "X1 coverage", "multi-source interval records (interval-coordinate check)", "-", "-",
        f"{x1['status']}  [{x1['detail']}]",
    ))
    rows.append((
        "intervals", "zero newly introduced unmeasured interval coordinates", "-", "-",
        intervals.get("intervals", "FAIL"),
    ))
    width = max(len(r[0]) for r in rows), max(len(r[1]) for r in rows)
    lines = ["2.2 A exit gate", ""]
    lines.append(f"{'item':<{width[0]}}  {'story':<{width[1]}}  rust     python   status")
    for item, title, r, p, status in rows:
        lines.append(f"{item:<{width[0]}}  {title:<{width[1]}}  {r:<7}  {p:<7}  {status}")
    for error in intervals.get("errors", []):
        lines.append(f"  interval check: {error}")
    statuses = [row[4].split()[0] for row in rows]
    failed = "FAIL" in statuses
    pending = "PENDING_CALIBRATION" in statuses
    lines.append("")
    if failed:
        lines.append("verdict: FAIL")
        return lines, 1
    if pending and require_calibrated:
        lines.append("verdict: PENDING_CALIBRATION (--require-calibrated: not accepted)")
        return lines, 1
    if pending:
        lines.append("verdict: PENDING_CALIBRATION (every executable story passes; calibration is unmeasured)")
        return lines, 0
    lines.append("verdict: PASS")
    return lines, 0


def self_test() -> int:
    ok_rust = "".join(f"test story_{n}_x ... ok\n" for n in range(1, 7))
    ok_py = "".join(f"tests/test_a_exit_gate.py::test_story_{n}_x PASSED [ 10%]\n" for n in range(1, 7))
    ok_py += "tests/test_a_exit_gate.py::test_story_3_x[nested_cohort] PASSED\n"
    pending = {
        "intervals": "PASS",
        "calibration": {
            "X1": {"status": "PENDING_CALIBRATION", "detail": "2 of 2 absent"},
            "X4": {"status": "PENDING_CALIBRATION", "detail": "2 of 2 absent"},
        },
        "errors": [],
    }
    done = json.loads(json.dumps(pending))
    for state in done["calibration"].values():
        state.update(status="PASS", detail="records present and attested")
    failures: list[str] = []

    def expect(label: str, rust: str, py: str, iv: dict, strict: bool, code: int, verdict: str) -> None:
        lines, rc = evaluate(rust, py, iv, strict)
        text = "\n".join(lines)
        if rc != code or verdict not in text:
            failures.append(f"'{label}': rc={rc} (want {code}), want {verdict!r}\n{text}")
        else:
            print(f"self-test ok: {label}")

    expect("pending calibration exits 0 by default", ok_rust, ok_py, pending, False, 0, "verdict: PENDING_CALIBRATION")
    expect("pending calibration fails --require-calibrated", ok_rust, ok_py, pending, True, 1, "not accepted")
    expect("measured calibration passes", ok_rust, ok_py, done, True, 0, "verdict: PASS")
    expect("a failed Rust story fails", ok_rust.replace("story_4_x ... ok", "story_4_x ... FAILED"), ok_py, done, False, 1, "verdict: FAIL")
    expect("a failed Python story fails", ok_rust, ok_py.replace("story_2_x PASSED", "story_2_x FAILED"), done, False, 1, "verdict: FAIL")
    expect("a story that never ran fails", ok_rust.replace("test story_5_x ... ok\n", ""), ok_py, done, False, 1, "verdict: FAIL")
    expect("one failed parametrization fails story 3",
           ok_rust, ok_py + "tests/test_a_exit_gate.py::test_story_3_x[independent_samples] FAILED\n", done, False, 1, "verdict: FAIL")
    expect("a failed hard part outranks pending", ok_rust.replace("story_3_x ... ok", "story_3_x ... FAILED"), ok_py, pending, False, 1, "verdict: FAIL")
    broken = dict(pending, intervals="FAIL", errors=["licensed row carries estimator_grid_not_measured"])
    expect("an interval violation fails", ok_rust, ok_py, broken, False, 1, "verdict: FAIL")
    absent = dict(done, calibration={})
    expect("missing calibration state is a failure, not a pass", ok_rust, ok_py, absent, False, 1, "verdict: FAIL")
    if failures:
        for failure in failures:
            print(f"SELF-TEST FAIL: {failure}")
        return 1
    print("a_exit_report self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    flags = [a for a in argv if a != "--require-calibrated"]
    args = dict(zip(flags[::2], flags[1::2], strict=False))
    rust_log = Path(args["--rust"]).read_text() if "--rust" in args else ""
    python_log = Path(args["--python"]).read_text() if "--python" in args else ""
    intervals = json.loads(Path(args["--intervals"]).read_text()) if "--intervals" in args else {}
    lines, code = evaluate(rust_log, python_log, intervals, "--require-calibrated" in argv)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
