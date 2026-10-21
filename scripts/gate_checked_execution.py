#!/usr/bin/env python3
"""Emit checked-execution citations, and fail when one does not run its retained program.

A licensed cell with no citation is not a silent pass and is not a license.
A citation whose test keeps or rebuilds a builder, or never runs the retained
program, fails this gate before any test binary starts. Gaussian compiler
integration is covered by cited public behavioral tests, which must be executed
by the acceptance runner; emitting a citation alone does not execute its test.
"""

import json
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from test_evidence import checked_execution_body_problems, closure

ROOT = Path(__file__).resolve().parents[1]
LICENSED = ROOT / "parity" / "support_licensed.toml"
AXIS = ("query", "graph_class", "structure", "inference", "validation")
_STUDY_BUILDER = re.compile(
    r"\bStudy::(?:tabular|series|series_multi|panel|events)\b|\bStudyBuilder\b"
)


def coordinate_of(cell: dict) -> str:
    return ":".join(str(cell.get(key)) for key in AXIS)


def execution_reentry_problems(body: str, assertion: str) -> list[str]:
    """The retained run must not be a public `Study` built after the builder is gone.

    Only the block that discards the builder is inspected, and only a `Study`
    constructor counts. A later refusal or a second coordinate in that same
    block is still a re-entry: the licensed execution was supposed to be over.
    """
    # Closures concatenate helpers as separate top-level blocks. Scan each.
    depth = 0
    start = None
    blocks: list[str] = []
    for index, character in enumerate(body):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(body[start : index + 1])
                start = None
    for block in blocks or [body]:
        discards = list(
            re.finditer(
                r"\b(?:drop\s*\(\s*(?:[A-Za-z_][A-Za-z0-9_]*)?builder[A-Za-z0-9_]*\s*\)"
                r"|del\s+(?:[A-Za-z_][A-Za-z0-9_]*)?builder[A-Za-z0-9_]*"
                r"|(?:[A-Za-z_][A-Za-z0-9_]*)?builder[A-Za-z0-9_]*\s*=\s*None)\b",
                block,
                re.I,
            )
        )
        if not discards:
            continue
        # A constructor before the retained run replaced that run. A constructor
        # after `estimate`/`execute` is a separate refusal or coordinate.
        executed = re.compile(
            r"\b(?:execute|estimate(?:_[A-Za-z0-9_]+)?|interval|refresh_series|run|evaluate_exact)\s*\("
        )
        for discard in discards:
            tail = block[discard.end() :]
            study = _STUDY_BUILDER.search(tail)
            if study is None:
                continue
            ran = executed.search(tail)
            if ran is not None and ran.start() < study.start():
                continue
            return [
                f"evidence assertion {assertion!r} re-enters the public builder "
                "instead of running its retained program"
            ]
    return []


def citation_problems(test_rel: str, assertion: str, cache: dict) -> list[str]:
    key = (test_rel, assertion)
    if key not in cache:
        path = ROOT / test_rel
        if not path.is_file():
            cache[key] = [f"test {test_rel!r} does not exist"]
        else:
            body = closure(path, assertion)
            cache[key] = checked_execution_body_problems(
                body, assertion
            ) + execution_reentry_problems(body, assertion)
    return cache[key]


def cell_problems(cell: dict, cache: dict) -> list[str]:
    label = coordinate_of(cell)
    estimators = cell.get("estimators")
    entries = cell.get("checked_execution")
    if not isinstance(estimators, list) or not estimators:
        return [f"{label}: licensed cell has no estimators, so it is not a checked-execution license"]
    if not isinstance(entries, list) or not entries:
        return [
            f"{label}: no retained-execution citation; an unlanded route is not a silent pass "
            "and is not a license"
        ]
    problems: list[str] = []
    named: list[str] = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            problems.append(f"{label}: checked_execution entry #{index} must be an inline table")
            continue
        estimator, test_rel, assertion = (
            entry.get("estimator"),
            entry.get("test"),
            entry.get("assertion"),
        )
        if not all(isinstance(value, str) and value.strip() for value in (estimator, test_rel, assertion)):
            problems.append(f"{label}: checked_execution entry #{index} needs estimator, test and assertion")
            continue
        named.append(estimator)
        for problem in citation_problems(test_rel, assertion, cache):
            problems.append(f"{label}::estimator={estimator}: {problem}")
    if len(named) != len(set(named)):
        problems.append(f"{label}: checked_execution names an estimator twice")
    if set(named) != set(estimators):
        problems.append(
            f"{label}: checked_execution estimators {sorted(set(named))} != "
            f"licensed estimators {sorted(set(estimators))}"
        )
    return problems


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} EVIDENCE_OUT SHARD")
    cells = tomllib.loads(LICENSED.read_text(encoding="utf-8")).get("cell", [])
    cache: dict = {}
    failures: list[str] = []
    rows = []
    for cell in cells:
        failures.extend(cell_problems(cell, cache))
        coordinate = coordinate_of(cell)
        for entry in cell.get("checked_execution") or []:
            if isinstance(entry, dict):
                rows.append(
                    (
                        f"{coordinate}::estimator={entry.get('estimator')}",
                        str(entry.get("test", "")),
                        str(entry.get("assertion", "")),
                    )
                )
    if failures:
        print(f"FAIL: checked execution: {len(failures)} licensed route(s) bypass the retained plan", flush=True)
        for failure in failures:
            print(f"FAIL: {failure}", flush=True)
        raise SystemExit(1)
    if not rows:
        raise SystemExit("FAIL: support_licensed.toml has no checked-execution citations")
    # Public behavioral assertions replace function-name checks of a parallel demonstration.
    gaussian_tests = "crates/antecedent/src/analysis/execute/prepared_gaussian_tests.rs"
    rows.extend([
        ("gaussian_compiled_execution", gaussian_tests,
         "prepared_gaussian_retains_the_compiled_program_after_builder_drop"),
        ("gaussian_compiled_resource_refusal", gaussian_tests,
         "prepared_gaussian_refuses_cancelled_and_insufficient_resource_execution"),
    ])
    shard = sys.argv[2]
    if shard != "all":
        # Keep all citations for one test target together, so one assertion is
        # executed by exactly one shard. Balance by distinct cited assertions.
        tests_by_path = defaultdict(set)
        for _, path, assertion in rows:
            if not path.startswith("crates/"):
                raise SystemExit(f"FAIL: checked shard needs Rust evidence; found {path}")
            tests_by_path[path].add(assertion)
        loads = [0] * 4
        assignment = {}
        for path, tests in sorted(tests_by_path.items(), key=lambda item: (-len(item[1]), item[0])):
            index = min(range(4), key=lambda i: (loads[i], i))
            assignment[path] = index
            loads[index] += len(tests) + 1
        index = int(shard.split("/")[0])
        total = len(rows)
        rows = [row for row in rows if assignment[row[1]] == index]
        if not rows:
            raise SystemExit(f"FAIL: checked shard {shard} has no citations")
        print(f"checked shard {shard}: {len(rows)}/{total} estimator citations", flush=True)
    with Path(sys.argv[1]).open("w", encoding="utf-8", newline="\n") as out:
        for label, path, assertion in rows:
            out.write("[[fixture_evidence]]\n")
            out.write(f"id = {json.dumps(label)}\n")
            out.write(f"evidence_test = {json.dumps(path)}\n")
            out.write(f"evidence_assertion = {json.dumps(assertion)}\n\n")
    print(f"checked execution: {len(cells)} licensed cells, {len(rows)} estimator citations", flush=True)


if __name__ == "__main__":
    main()
