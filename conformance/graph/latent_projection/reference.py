# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the latent-projection fixture by explicit path enumeration.

For a DAG over observed and latent nodes, the latent projection onto the observed nodes has

* a directed edge X -> Y when some directed path from X to Y has only latent nodes inside;
* a bidirected edge X <-> Y when some simple path between X and Y has only latent nodes inside,
  has an arrowhead at both ends (the first edge points into X, the last into Y) and no
  collider on it (so it passes through a latent common cause).

The path search below lists every simple path with latent interior and applies those two
definitions directly. Standard library only; it imports no Antecedent code and shares nothing
with the product's projection routine.

    uv run --frozen --script conformance/graph/latent_projection/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse(edges):
    return [tuple(edge.split("->")) for edge in edges]


def simple_paths(arcs, start, end, observed):
    """Every simple path start..end (as node lists) whose interior nodes are all latent."""
    neighbours = {}
    for a, b in arcs:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    found = []

    def walk(path):
        here = path[-1]
        for nxt in sorted(neighbours.get(here, ())):
            if nxt in path:
                continue
            if nxt == end:
                found.append(path + [nxt])
            elif nxt not in observed:
                walk(path + [nxt])

    walk([start])
    return found


def project(arcs, observed):
    arcset = set(arcs)
    names = sorted({n for arc in arcs for n in arc})
    directed, bidirected = set(), set()
    for x in observed:
        for y in observed:
            if x == y:
                continue
            for path in simple_paths(arcs, x, y, set(observed)):
                forward = [(path[i], path[i + 1]) in arcset for i in range(len(path) - 1)]
                if all(forward):
                    directed.add((x, y))
                points_into_start = (path[1], path[0]) in arcset
                points_into_end = (path[-2], path[-1]) in arcset
                collider = any(
                    (path[i - 1], path[i]) in arcset and (path[i + 1], path[i]) in arcset
                    for i in range(1, len(path) - 1)
                )
                if points_into_start and points_into_end and not collider:
                    bidirected.add(frozenset((x, y)))
    del names
    return directed, bidirected


def render(directed, bidirected, observed):
    rank = {name: i for i, name in enumerate(observed)}
    edges = {f"{a}->{b}" for a, b in directed}
    for pair in bidirected:
        a, b = sorted(pair, key=rank.get)
        edges.add(f"{a}<->{b}")
    return edges


def main() -> int:
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = []
    for case in document["cases"]:
        directed, bidirected = project(parse(case["dag"]), case["observed"])
        derived = render(directed, bidirected, case["observed"])
        stored = set(case["projection"])
        if derived != stored:
            problems.append(f"{case['id']}: derived {sorted(derived)} != fixture {sorted(stored)}")
    for problem in problems:
        print("FAIL", problem, file=sys.stderr)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
