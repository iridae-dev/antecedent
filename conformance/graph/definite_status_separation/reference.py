# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent member-MAG enumeration for the definite-status separation fixture.

For each case the marked graph (marks: arrowhead, tail, circle) is treated as a partial ancestral
graph.  Every circle is resolved to a tail or an arrowhead in all ways; a completion is kept when
it is an ancestral graph (no directed cycle, no bidirected edge between an ancestor and its
descendant, no undirected edge), is maximal (every non-adjacent pair is m-separated by some set),
and has on every unshielded triple the collider status that the marked graph shows (a collider
exactly when both marks at the middle node are arrowheads).  Kept completions are grouped by the
set of all m-separation statements they entail (their Markov equivalence class); a case has a
determined answer only when exactly one class remains.  The status is "separated" when every
member separates x and y given the conditioning set, "connected" when no member does, and
"undetermined" otherwise.  m-separation is evaluated by its path definition: a simple path is open
given Z when every non-collider is outside Z and every collider is an ancestor of Z.  A conflict
mark ("x") records an unknown orientation and is classified undetermined by declaration, not
derived.

Standard library only; imports no Antecedent code.

    uv run --frozen --script conformance/graph/definite_status_separation/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_edge(text):
    left, right = text.split("-", 1)
    i = len(left)
    while i > 0 and left[i - 1] in "<ox":
        i -= 1
    j = 0
    while j < len(right) and right[j] in ">ox":
        j += 1
    mark = {"": "tail", "<": "arrow", ">": "arrow", "o": "circle", "x": "conflict"}
    return int(left[:i]), mark[left[i:]], mark[right[:j]], int(right[j:])


def ancestors(nodes, directed):
    anc = {v: {v} for v in nodes}
    changed = True
    while changed:
        changed = False
        for a, b in directed:
            before = len(anc[b])
            anc[b] |= anc[a]
            changed |= len(anc[b]) != before
    return anc


class Mag:
    def __init__(self, nodes, edges):
        # edges: dict frozenset({a, b}) -> {a: mark_at_a, b: mark_at_b}
        self.nodes = nodes
        self.edges = edges
        self.directed = [
            (a, b)
            for e, m in edges.items()
            for a, b in itertools.permutations(sorted(e))
            if m[a] == "tail" and m[b] == "arrow"
        ]
        self.anc = ancestors(nodes, self.directed)

    def adjacent(self, a, b):
        return frozenset((a, b)) in self.edges

    def mark_at(self, a, b):
        return self.edges[frozenset((a, b))][b]

    def ancestral(self):
        if any(m[a] == "tail" and m[b] == "tail" for e, m in self.edges.items() for a, b in [sorted(e)]):
            return False
        for a, b in self.directed:
            if a in self.anc[b] and b in self.anc[a] and a != b:
                return False
        for e, m in self.edges.items():
            a, b = sorted(e)
            if m[a] == "arrow" and m[b] == "arrow" and (a in self.anc[b] or b in self.anc[a]):
                return False
        return True

    def m_separated(self, x, y, z):
        z = set(z)
        anc_z = set().union(*(self.anc[v] for v in z)) if z else set()

        def open_path(path):
            for k in range(1, len(path) - 1):
                a, b, c = path[k - 1], path[k], path[k + 1]
                collider = self.mark_at(a, b) == "arrow" and self.mark_at(c, b) == "arrow"
                if collider:
                    if b not in anc_z:
                        return False
                elif b in z:
                    return False
            return True

        def walk(path):
            node = path[-1]
            if node == y:
                return open_path(path)
            for e in self.edges:
                if node in e:
                    (other,) = e - {node}
                    if other not in path and walk(path + [other]):
                        return True
            return False

        return not walk([x])

    def signature(self):
        out = set()
        for x, y in itertools.combinations(sorted(self.nodes), 2):
            rest = [v for v in sorted(self.nodes) if v not in (x, y)]
            for r in range(len(rest) + 1):
                for z in itertools.combinations(rest, r):
                    if self.m_separated(x, y, z):
                        out.add((x, y, z))
        return frozenset(out)

    def maximal(self):
        for x, y in itertools.combinations(sorted(self.nodes), 2):
            if self.adjacent(x, y):
                continue
            rest = [v for v in sorted(self.nodes) if v not in (x, y)]
            if not any(
                self.m_separated(x, y, z)
                for r in range(len(rest) + 1)
                for z in itertools.combinations(rest, r)
            ):
                return False
        return True


def members(case):
    parsed = [parse_edge(e) for e in case["edges"]]
    nodes = sorted({v for a, _, _, b in parsed for v in (a, b)})
    shown = {}
    for a, ma, mb, b in parsed:
        shown[frozenset((a, b))] = {a: ma, b: mb}
    circles = [(e, v) for e, m in shown.items() for v in sorted(e) if m[v] == "circle"]
    result = []
    for choice in itertools.product(("tail", "arrow"), repeat=len(circles)):
        edges = {e: dict(m) for e, m in shown.items()}
        for (e, v), mark in zip(circles, choice):
            edges[e][v] = mark
        mag = Mag(nodes, edges)
        if not mag.ancestral():
            continue
        ok = True
        for b in nodes:
            nbrs = [v for v in nodes if v != b and frozenset((v, b)) in shown]
            for a, c in itertools.combinations(nbrs, 2):
                if frozenset((a, c)) in shown:
                    continue
                want = shown[frozenset((a, b))][b] == "arrow" and shown[frozenset((c, b))][b] == "arrow"
                have = mag.mark_at(a, b) == "arrow" and mag.mark_at(c, b) == "arrow"
                ok &= want == have
        if ok and mag.maximal():
            result.append(mag)
    return result


def status(case):
    if any("x" in e.replace("-", "", 1) for e in case["edges"]):
        return "undetermined", 0, 0
    mags = members(case)
    classes = {}
    for mag in mags:
        classes.setdefault(mag.signature(), []).append(mag)
    if len(classes) != 1:
        return "undetermined", len(mags), len(classes)
    verdicts = {m.m_separated(case["x"], case["y"], case["conditioned"]) for m in mags}
    if verdicts == {True}:
        return "separated", len(mags), 1
    if verdicts == {False}:
        return "connected", len(mags), 1
    return "undetermined", len(mags), 1


def check(document):
    problems = []
    for case in document["cases"]:
        got, n_members, n_classes = status(case)
        if got != case["status"]:
            problems.append(f"{case['id']}: derived {got} ({n_members} members, {n_classes} classes) != {case['status']}")
        if "separated" in case and case["separated"] != (got == "separated"):
            problems.append(f"{case['id']}: separated flag disagrees")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    if args.report:
        for case in document["cases"]:
            print(case["id"], status(case))
    problems = check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
