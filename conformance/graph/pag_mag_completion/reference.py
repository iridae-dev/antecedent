# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent exhaustive enumeration of the MAG completions of a small partial ancestral graph.

Every circle endpoint of the declared input is assigned tail or arrowhead in all possible ways.
A completion is kept when it is an ancestral graph by the definition of Richardson and Spirtes
(no directed cycle; no vertex that is an ancestor of one of its spouses, that is no almost
directed cycle; no arrowhead into a vertex with an undirected edge) and has no tail-tail
(undirected, selection) edge, which the supported completion family excludes. Maximality is not
evaluated: the reference refuses any case whose underlying graph is not complete, where every
ancestral graph is maximal because no pair of vertices is non-adjacent.

The declared outputs, completion counts and the ``is_mag`` flag of every case are re-derived from
the ``input`` edge strings only (``AoB``, ``A->B``, ``A<->B``). Standard library only; imports no
Antecedent code.

    uv run --frozen --script conformance/graph/pag_mag_completion/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN = re.compile(r"^(\d+)(o-o|o->|<-o|->|<-|<->)(\d+)$")
MARKS = {
    "o-o": ("o", "o"), "o->": ("o", ">"), "<-o": (">", "o"),
    "->": ("-", ">"), "<-": (">", "-"), "<->": (">", ">"),
}


def parse(edges):
    parsed = []
    for token in edges:
        m = TOKEN.match(token)
        if not m:
            raise ValueError(f"unreadable edge {token!r}")
        a, kind, b = int(m.group(1)), m.group(2), int(m.group(3))
        parsed.append((a, b, *MARKS[kind]))
    return parsed


def is_ancestral(edges):
    """edges: (a, b, mark_at_a, mark_at_b) with marks in {'-', '>'} only."""
    children = {}
    nodes = set()
    for a, b, ma, mb in edges:
        nodes |= {a, b}
        if ma == "-" and mb == ">":
            children.setdefault(a, set()).add(b)
        if mb == "-" and ma == ">":
            children.setdefault(b, set()).add(a)

    def reach(start):
        seen, stack = set(), [start]
        while stack:
            v = stack.pop()
            for w in children.get(v, ()):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    descendants = {v: reach(v) for v in nodes}
    if any(v in descendants[v] for v in nodes):
        return False  # directed cycle
    for a, b, ma, mb in edges:
        if ma == ">" and mb == ">" and (a in descendants[b] or b in descendants[a]):
            return False  # almost directed cycle
        if ma == "-" and mb == "-":
            return False  # undirected edge, outside the supported family
    return True


def complete_graph(edges):
    nodes = {n for a, b, *_ in edges for n in (a, b)}
    pairs = {frozenset((a, b)) for a, b, *_ in edges}
    return all(frozenset(p) in pairs for p in itertools.combinations(sorted(nodes), 2))


def completions(edges):
    if not complete_graph(edges):
        raise ValueError("reference does not evaluate maximality; case graph is not complete")
    slots = [(i, side) for i, e in enumerate(edges) for side in (2, 3) if e[side] == "o"]
    out = []
    for choice in itertools.product("->", repeat=len(slots)):
        trial = [list(e) for e in edges]
        for (i, side), mark in zip(slots, choice):
            trial[i][side] = mark
        trial = [tuple(e) for e in trial]
        if is_ancestral(trial):
            out.append(trial)
    return out


def render(edge):
    a, b, ma, mb = edge
    if (ma, mb) == ("-", ">"):
        return f"{a}->{b}"
    if (ma, mb) == (">", "-"):
        return f"{b}->{a}"
    if (ma, mb) == (">", ">"):
        return f"{min(a, b)}<->{max(a, b)}"
    return f"{a}--{b}"


def check(document):
    problems = []
    for case in document["cases"]:
        edges = parse(case["input"])
        if "valid_completion_count" in case:
            found = completions(edges)
            if len(found) != case["valid_completion_count"]:
                problems.append(f"{case['id']}: count {case['valid_completion_count']} vs {len(found)}")
            if "outputs" in case:
                derived = sorted(" ".join(sorted(render(e) for e in c)) for c in found)
                if derived != sorted(case["outputs"]):
                    problems.append(f"{case['id']}: outputs {sorted(case['outputs'])} vs {derived}")
        if "is_mag" in case:
            if any(m == "o" for e in edges for m in e[2:]):
                problems.append(f"{case['id']}: is_mag asked of a graph with circle endpoints")
            elif is_ancestral(edges) != case["is_mag"]:
                problems.append(f"{case['id']}: is_mag {case['is_mag']} vs {is_ancestral(edges)}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = check(document)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    print("pag_mag_completion reference matches expected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
