#!/usr/bin/env python3
"""Composition exit: join the executed story results into one per-story table and a verdict.

Inputs are the captured logs of the two story suites. This script decides nothing about the
science, only whether every story passed. Calibration is an annotation shown beside the table by
scripts/calibration_status.py and is never read here:

    python3 scripts/composition_exit_report.py --rust rust.log --python py.log
    python3 scripts/composition_exit_report.py --self-test

Per-story status:
  PASS   the Rust and the Python story both ran and passed
  FAIL   a story failed, errored, or did not run

Exit status: 1 on any FAIL.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

STORIES = {
    1: "multi-source limited experiment",
    2: "finite scenario set: identified/unidentified/unevaluated ",
    3: "learned continuous transport, overlap-supported ",
    4: "two-step temporal transport, history-support refusal ",
    5: "cross-world query, shared abduction, typed refusal ",
    6: "mixed-source proof by bounded search + incomplete search ",
    7: "release boundary: corrected measurement, pooled evidence, decision, revision ",
}
RUST = re.compile(r"^test (?:\S+::)?story_(\d)_\w+ \.\.\. (ok|FAILED|ignored)\s*$", re.M)
PYTHON = re.compile(
    r"(?:^|/)test_composition_exit_gate\.py::test_story_(\d)_\w+(?:\[[^\]]*\])?\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)",
    re.M,
)


def outcomes(pattern: re.Pattern[str], log: str, passing: set[str]) -> dict[int, str]:
    seen: dict[int, list[bool]] = {}
    for number, verdict in pattern.findall(log):
        seen.setdefault(int(number), []).append(verdict in passing)
    return {n: ("PASS" if all(v) else "FAIL") for n, v in seen.items()}


def evaluate(rust_log: str, python_log: str) -> tuple[list[str], int]:
    rust = outcomes(RUST, rust_log, {"ok"})
    python = outcomes(PYTHON, python_log, {"PASSED"})
    rows: list[tuple[str, str, str, str, str]] = []
    for number, title in STORIES.items():
        r, p = rust.get(number, "MISSING"), python.get(number, "MISSING")
        rows.append((f"story {number}", title, r, p, "PASS" if (r, p) == ("PASS", "PASS") else "FAIL"))
    width = max(len(r[0]) for r in rows), max(len(r[1]) for r in rows)
    lines = ["composition exit", ""]
    lines.append(f"{'item':<{width[0]}}  {'story':<{width[1]}}  rust     python   status")
    for item, title, r, p, status in rows:
        lines.append(f"{item:<{width[0]}}  {title:<{width[1]}}  {r:<7}  {p:<7}  {status}")
    lines.append("")
    if any(row[4] == "FAIL" for row in rows):
        lines.append("verdict: FAIL")
        return lines, 1
    lines.append("verdict: PASS")
    return lines, 0


def self_test() -> int:
    ok_rust = "".join(f"test story_{n}_x ... ok\n" for n in range(1, 8))
    ok_py = "".join(
        f"tests/test_composition_exit_gate.py::test_story_{n}_x PASSED [ 10%]\n"
        for n in range(1, 8)
    )
    ok_py += "tests/test_composition_exit_gate.py::test_story_3_x[nested_cohort] PASSED\n"
    failures: list[str] = []

    def expect(label: str, rust: str, py: str, code: int, verdict: str) -> None:
        lines, rc = evaluate(rust, py)
        text = "\n".join(lines)
        if rc != code or verdict not in text:
            failures.append(f"'{label}': rc={rc} (want {code}), want {verdict!r}\n{text}")
        else:
            print(f"self-test ok: {label}")

    expect("every story passing passes", ok_rust, ok_py, 0, "verdict: PASS")
    expect(
        "a failed Rust story fails",
        ok_rust.replace("story_4_x ... ok", "story_4_x ... FAILED"),
        ok_py,
        1,
        "verdict: FAIL",
    )
    expect(
        "a failed Python story fails",
        ok_rust,
        ok_py.replace("story_2_x PASSED", "story_2_x FAILED"),
        1,
        "verdict: FAIL",
    )
    expect(
        "a story that never ran fails",
        ok_rust.replace("test story_5_x ... ok\n", ""),
        ok_py,
        1,
        "verdict: FAIL",
    )
    expect(
        "the release-boundary story never ran",
        ok_rust.replace("test story_7_x ... ok\n", ""),
        ok_py,
        1,
        "verdict: FAIL",
    )
    expect(
        "a failed release-boundary Python story fails",
        ok_rust,
        ok_py.replace("story_7_x PASSED", "story_7_x FAILED"),
        1,
        "verdict: FAIL",
    )
    expect(
        "one failed parametrization fails story 3",
        ok_rust,
        ok_py + "tests/test_composition_exit_gate.py::test_story_3_x[independent_samples] FAILED\n",
        1,
        "verdict: FAIL",
    )
    if failures:
        for failure in failures:
            print(f"SELF-TEST FAIL: {failure}")
        return 1
    print("composition_exit_report self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    args = dict(zip(argv[::2], argv[1::2], strict=False))
    rust_log = Path(args["--rust"]).read_text(encoding="utf-8") if "--rust" in args else ""
    python_log = Path(args["--python"]).read_text(encoding="utf-8") if "--python" in args else ""
    lines, code = evaluate(rust_log, python_log)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
