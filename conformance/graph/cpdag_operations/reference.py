# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exhaustive enumeration reference for the CPDAG operation fixture.

mec_size: every orientation of the skeleton's edges is enumerated; an orientation is a Markov
equivalence class member when it is acyclic and has exactly the CPDAG's unshielded colliders
(no new ones, none removed). The class size is the number of members.

orient / insert cases: the fixture's operations are plain edge-set operations with no Meek
propagation. Orienting one undirected edge changes only that edge; inserting a directed edge that
closes a directed path is a cycle. Both are evaluated here by set manipulation and a depth-first
reachability test. No Antecedent code.

    uv run --frozen --script conformance/graph/cpdag_operations/reference.py --check
    uv run --frozen --script conformance/graph/cpdag_operations/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse(edge: str) -> tuple[int, int, bool]:
    """Return (a, b, directed) with a -> b when directed, a -- b otherwise."""
    if "->" in edge:
        a, b = edge.split("->")
        return int(a), int(b), True
    a, b = edge.split("--")
    return int(a), int(b), False


def reaches(arcs: set[tuple[int, int]], source: int, target: int) -> bool:
    seen, stack = set(), [source]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(b for a, b in arcs if a == node)
    return False


def colliders(arcs: set[tuple[int, int]], adjacent: set[frozenset[int]]) -> set[tuple[int, int, int]]:
    """Unshielded colliders a -> m <- c, with a < c and a, c non-adjacent."""
    found = set()
    for (a, m), (c, m2) in itertools.combinations(sorted(arcs), 2):
        if m == m2 and a != c and frozenset((a, c)) not in adjacent:
            found.add((min(a, c), m, max(a, c)))
    return found


def mec_size(edges: list[str]) -> int:
    parsed = [parse(e) for e in edges]
    adjacent = {frozenset((a, b)) for a, b, _ in parsed}
    fixed = {(a, b) for a, b, d in parsed if d}
    free = [(a, b) for a, b, d in parsed if not d]
    target = colliders(fixed, adjacent)
    count = 0
    for flips in itertools.product((False, True), repeat=len(free)):
        arcs = set(fixed) | {(b, a) if f else (a, b) for (a, b), f in zip(free, flips)}
        if any(reaches(arcs - {(a, b)}, b, a) for a, b in arcs):
            continue
        if colliders(arcs, adjacent) == target:
            count += 1
    return count


def apply(case: dict) -> dict:
    parsed = [parse(e) for e in case["input"]]
    operation = case["operation"]
    if operation.startswith("orient "):
        a, b, _ = parse(operation.removeprefix("orient "))
        out = []
        for x, y, d in parsed:
            if not d and {x, y} == {a, b}:
                out.append(f"{a}->{b}")
            else:
                out.append(f"{x}->{y}" if d else f"{x}--{y}")
        return {"output": out}
    if operation.startswith("insert "):
        a, b, _ = parse(operation.removeprefix("insert "))
        arcs = {(x, y) for x, y, d in parsed if d}
        return {"error": "cycle"} if reaches(arcs, b, a) else {"output": [*case["input"], f"{a}->{b}"]}
    raise ValueError(operation)


def check(document: dict) -> list[str]:
    problems = []
    for case in document["cases"]:
        if "mec_size" in case:
            got = mec_size(case["edges"])
            if got != case["mec_size"]:
                problems.append(f"{case['id']}: mec_size {case['mec_size']} vs {got}")
            continue
        got = apply(case)
        for key in ("output", "error"):
            if got.get(key) != case.get(key):
                problems.append(f"{case['id']}: {key} {case.get(key)} vs {got.get(key)}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["cases"][0]["mec_size"] = 4
    if not check(wrong):
        failures.append("wrong mec size accepted")
    wrong = copy.deepcopy(document)
    wrong["cases"][1]["output"] = ["1->0", "1--2"]
    if not check(wrong):
        failures.append("wrong orientation accepted")
    wrong = copy.deepcopy(document)
    del wrong["cases"][2]["error"]
    wrong["cases"][2]["output"] = ["0->1", "1->2", "2->0"]
    if not check(wrong):
        failures.append("cycle accepted")
    if mec_size(["0--1", "1--2", "0--2"]) != 6:
        failures.append("triangle class size is 6")
    if mec_size(["0->1", "2->1"]) != 1:
        failures.append("collider class is a single DAG")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
