# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent brute-force recanting-witness check for the path-specific identification fixture.

A directed T -> Y path is *selected* when it visits every included node (an empty list selects all
paths).  A node W (other than T and Y) is a recanting witness when some prefix q from T to W has
continuations s1, s2 from W to Y with q.s1 selected and q.s2 not selected (Avin-Shpitser-Pearl).
The effect is identified exactly when no witness exists.  Every directed T -> Y path is enumerated
explicitly; no reachability shortcut and no Antecedent code is used.

    uv run --frozen --script conformance/identify/path_specific/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def all_paths(edges, src, dst):
    out = []

    def walk(node, path):
        if node == dst:
            out.append(tuple(path))
            return
        for a, b in edges:
            if a == node and b not in path:
                walk(b, path + [b])

    walk(src, [src])
    return out


def witnesses(case):
    edges = [tuple(e.split("->")) for e in case["dag"]]
    edges = [(a.strip(), b.strip()) for a, b in edges]
    paths = all_paths(edges, "T", "Y")
    included = set(case["included_path_nodes"])
    selected = {p for p in paths if included <= set(p)}
    found = set()
    for w in {n for p in paths for n in p} - {"T", "Y"}:
        by_prefix = {}
        for p in paths:
            if w in p:
                i = p.index(w)
                by_prefix.setdefault(p[: i + 1], []).append(p)
        for group in by_prefix.values():
            if any(p in selected for p in group) and any(p not in selected for p in group):
                found.add(w)
    return found


def check(document):
    problems = []
    for case in document["cases"]:
        found = witnesses(case)
        derived = "identified" if not found else "not_identified"
        if derived != case["status"]:
            problems.append(f"{case['id']}: derived {derived} (witnesses {sorted(found)}) != {case['status']}")
        if case["status"] == "not_identified" and case["witness"] not in found:
            problems.append(f"{case['id']}: witness {case['witness']} not in derived {sorted(found)}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
