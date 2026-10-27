# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact reference for the confounded temporal pulse fixture.

Everything is derived from the declared structural equations and the period-four
exogenous sequences, in exact rational arithmetic, with no Antecedent code:

* the unrolled temporal graph is built from the template edges and the backdoor
  adjustment sets for t@-1 -> y@0 (and t@-1 -> y@1) are decided by exhaustive d-separation
  over simple paths, so {z@-1} is shown to be sufficient and the empty set insufficient;
* the deterministic series (n = 241, pre-sample values 0) is generated from
  Z = V, A = Z + U, Y_s = 1 + 2 A_(s-1) + 5 Z_(s-1);
* the adjusted linear g-computation surface (OLS of Y_s on A_(s-1), Z_(s-1), standardized
  over the lagged Z window) and the unadjusted OLS surface are solved by exact normal
  equations, as are the horizon-2 unadjusted lag-2 association cells.

    uv run --frozen --script conformance/response/temporal_confounded_pulse/reference.py --check
    uv run --frozen --script conformance/response/temporal_confounded_pulse/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATOL = 1e-8


def solve(a, y):
    n = len(y)
    m = [row[:] + [v] for row, v in zip(a, y)]
    for i in range(n):
        pivot = next(r for r in range(i, n) if m[r][i] != 0)
        m[i], m[pivot] = m[pivot], m[i]
        m[i] = [v / m[i][i] for v in m[i]]
        for r in range(n):
            if r != i:
                m[r] = [v - m[r][i] * w for v, w in zip(m[r], m[i])]
    return [row[-1] for row in m]


def ols(columns, y):
    k = len(columns)
    xtx = [[sum(a * b for a, b in zip(columns[i], columns[j])) for j in range(k)] for i in range(k)]
    xty = [sum(a * b for a, b in zip(columns[i], y)) for i in range(k)]
    return solve(xtx, xty)


def series(n):
    z = [F([1, 1, -1, -1][i % 4]) for i in range(n)]
    u = [F([1, -1, 1, -1][i % 4]) for i in range(n)]
    a = [zi + ui for zi, ui in zip(z, u)]
    y = [(1 + 2 * a[i - 1] + 5 * z[i - 1]) if i >= 1 else F(0) for i in range(n)]
    return z, a, y


def unrolled_edges(template, low=-4, high=2):
    edges = set()
    for e in template:
        for t in range(low, high + 1):
            # template (from,lag)->(to,to_lag): the cause acts (lag - to_lag) steps before the effect.
            dst = (e["to"], t - e["to_lag"])
            src = (e["from"], dst[1] - (e["lag"] - e["to_lag"]))
            if low <= src[1] <= high and low <= dst[1] <= high:
                edges.add((src, dst))
    return edges


def blocked(path, edges, given, descendants):
    for i in range(1, len(path) - 1):
        left, mid, right = path[i - 1], path[i], path[i + 1]
        collider = (left, mid) in edges and (right, mid) in edges
        if collider:
            if mid not in given and not (descendants[mid] & given):
                return True
        elif mid in given:
            return True
    return False


def active_paths(edges, source, target, given):
    nodes = {n for e in edges for n in e}
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    children = {n: {b for a, b in edges if a == n} for n in nodes}

    def desc(n, seen=None):
        seen = seen if seen is not None else set()
        for c in children[n]:
            if c not in seen:
                seen.add(c)
                desc(c, seen)
        return seen

    descendants = {n: desc(n) for n in nodes}
    found = []

    def walk(path):
        if path[-1] == target:
            found.append(list(path))
            return
        for nxt in adj[path[-1]]:
            if nxt not in path:
                walk(path + [nxt])

    walk([source])
    # Backdoor paths: first edge points into the source.
    return [
        p
        for p in found
        if (p[1], p[0]) in edges and not blocked(p, edges, given, descendants)
    ]


def compute(document: dict) -> dict:
    contract = document["contract"]
    edges = unrolled_edges(contract["temporal_graph"])
    t0, y0, y1, z = ("t", -1), ("y", 0), ("y", 1), ("z", -1)
    ident = {
        "h1_empty_open_backdoor_paths": len(active_paths(edges, t0, y0, set())),
        "h1_z_open_backdoor_paths": len(active_paths(edges, t0, y0, {z})),
        "h2_empty_open_backdoor_paths": len(active_paths(edges, t0, y1, set())),
    }
    n = document["generation"]["n"]
    zs, a, y = series(n)
    rows = range(1, n)
    one = [F(1)] * len(rows)
    ylag = [y[s] for s in rows]
    alag = [a[s - 1] for s in rows]
    zlag = [zs[s - 1] for s in rows]
    b = ols([one, alag, zlag], ylag)
    mean_z = sum(zlag) / len(zlag)
    adjusted = [b[0] + b[1] * d + b[2] * mean_z for d in (0, 1)]
    c = ols([one, alag], ylag)
    unadjusted = [c[0] + c[1] * d for d in (0, 1)]
    rows2 = range(2, n)
    h2 = ols([[F(1)] * len(rows2), [a[s - 2] for s in rows2]], [y[s] for s in rows2])
    return {
        "ident": ident,
        "coefficients": b,
        "adjusted": adjusted,
        "unadjusted": unadjusted,
        "pulse_contrast": adjusted[1] - adjusted[0],
        "unadjusted_contrast": unadjusted[1] - unadjusted[0],
        "horizon2": [h2[0], h2[0] + h2[1]],
    }


def check(document: dict) -> list[str]:
    got = compute(document)
    problems = []
    ident = got["ident"]
    if ident["h1_empty_open_backdoor_paths"] == 0:
        problems.append("empty adjustment set blocks every horizon-1 backdoor path")
    if ident["h1_z_open_backdoor_paths"] != 0:
        problems.append("{z@-1} leaves an open horizon-1 backdoor path")
    if ident["h2_empty_open_backdoor_paths"] != 0:
        problems.append("horizon 2 has an open backdoor path under the empty set")
    declared = [(s["variable"], s["offset"]) for s in document["contract"]["identification"]["adjustment_set"]]
    if declared != [("z", -1)]:
        problems.append(f"declared adjustment set {declared}")
    if document["contract"]["multi_horizon"]["identification"][1]["adjustment_set"]:
        problems.append("declared horizon-2 adjustment set is nonempty")
    contract = document["contract"]
    pairs = [
        ("adjusted surface", [float(v) for v in got["adjusted"]], contract["surface"]["mean"]),
        ("unadjusted surface", [float(v) for v in got["unadjusted"]], contract["unadjusted"]["surface"]["mean"]),
        ("pulse contrast", [float(got["pulse_contrast"])], [contract["pulse_effect_projection"]["contrast"]]),
        ("sustained contrast", [float(got["pulse_contrast"])], [contract["sustained_effect_projection"]["contrast"]]),
        ("unadjusted contrast", [float(got["unadjusted_contrast"])], [contract["unadjusted"]["pulse_contrast"]]),
        (
            "horizon-2 cells",
            [float(got["horizon2"][0]), float(got["horizon2"][1])],
            [
                contract["multi_horizon"]["surface"]["mean"][1],
                contract["multi_horizon"]["surface"]["mean"][3],
            ],
        ),
        (
            "horizon-1 cells of the multi-horizon surface",
            [float(v) for v in got["adjusted"]],
            [contract["multi_horizon"]["surface"]["mean"][0], contract["multi_horizon"]["surface"]["mean"][2]],
        ),
    ]
    for name, reference, fixture in pairs:
        if len(reference) != len(fixture) or any(abs(r - f) > ATOL for r, f in zip(reference, fixture)):
            problems.append(f"{name}: fixture {fixture} reference {reference}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["contract"]["surface"]["mean"][1] = 5.5
    if not check(wrong):
        failures.append("confounded surface accepted as the adjusted truth")
    wrong = copy.deepcopy(document)
    wrong["contract"]["identification"]["adjustment_set"] = []
    if not check(wrong):
        failures.append("empty adjustment set accepted")
    wrong = copy.deepcopy(document)
    wrong["contract"]["unadjusted"]["pulse_contrast"] = 2.0
    if not check(wrong):
        failures.append("unadjusted contrast equal to the interventional one accepted")
    wrong = copy.deepcopy(document)
    wrong["contract"]["temporal_graph"] = [
        e for e in wrong["contract"]["temporal_graph"] if e["from"] != "z" or e["to"] != "t"
    ]
    if not check(wrong):
        failures.append("graph without the confounding edge accepted")
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
