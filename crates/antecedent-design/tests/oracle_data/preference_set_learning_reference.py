"""Exact compatible preference sets of small recorded-choice problems.

Standard library only. Every attribute mean, range end, relation coefficient and tolerance is
a dyadic rational, so the inputs are exact in floating point. The value model of every case is

    value = x0 + sum_k t_k * x_{k+1}

so the fixed part of an option is its x0 and its term for parameter k is its x_{k+1}. A record
with chosen option c and rejected option j gives the half-space

    sum_k t_k (x_{c,k+1} - x_{j,k+1}) >= -tolerance - (x_{c,0} - x_{j,0}).

The learned polytope is the declared box, the declared relations and every such half-space.
Its vertices are found by brute force: every choice of K hyperplanes among all constraints
(box faces included) is solved exactly with `fractions.Fraction`, and the feasible solutions
are kept. Each parameter's learned range is the minimum and maximum of that coordinate over
the vertices. A case with no vertex is contradictory; then

  * the smallest conflicting records are found by trying every subset of records in order of
    size (lexicographic within a size, records sorted by id) with the declared set;
  * the needed tolerance is `tolerance + s*`, where `s*` is the least `s >= 0` such that every
    half-space relaxed by `s` meets the declared set, found by vertex enumeration in (t, s).

Writes preference_set_learning_reference.json next to this script.
"""

import itertools
import json
from fractions import Fraction as F
from pathlib import Path


def solve_square(rows, rhs):
    """Exact Gaussian elimination; None when singular."""
    n = len(rows)
    a = [list(r) + [b] for r, b in zip(rows, rhs)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col and a[r][col] != 0:
                factor = a[r][col] / a[col][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
    return [a[i][n] / a[i][i] for i in range(n)]


def vertices(constraints, dim):
    """Vertices of {x : a.x <= b for (a, b) in constraints} (assumed bounded)."""
    found = set()
    for chosen in itertools.combinations(range(len(constraints)), dim):
        rows = [constraints[i][0] for i in chosen]
        rhs = [constraints[i][1] for i in chosen]
        x = solve_square(rows, rhs)
        if x is None:
            continue
        if all(sum(c * v for c, v in zip(a, x)) <= b for a, b in constraints):
            found.add(tuple(x))
    return sorted(found)


def box(bounds, extra=0):
    """Box faces as `a.x <= b` rows over len(bounds) + extra variables."""
    dim = len(bounds) + extra
    rows = []
    for k, (lo, hi) in enumerate(bounds):
        unit = [F(0)] * dim
        unit[k] = F(1)
        rows.append((unit, hi))
        rows.append(([-v for v in unit], -lo))
    return rows


def declared_rows(case, extra=0):
    rows = []
    k = len(case["bounds"])
    for rel in case["relations"]:
        a = [F(0)] * (k + extra)
        for j, c in rel["terms"]:
            a[j] = F(c)
        b = F(rel["bound"])
        if rel["sense"] in ("at_most", "equal"):
            rows.append((a, b))
        if rel["sense"] in ("at_least", "equal"):
            rows.append(([-v for v in a], -b))
    return rows


def choice_rows(case, records, tolerance, with_slack=False):
    """`a.x <= b` rows of every choice half-space (negated `>=`), optionally with slack s."""
    rows = []
    for rec in records:
        options = {o["id"]: [F(v) for v in o["means"]] for o in rec["options"]}
        c = options[rec["chosen"]]
        for o in rec["options"]:
            if o["id"] == rec["chosen"]:
                continue
            j = options[o["id"]]
            d = [c[k + 1] - j[k + 1] for k in range(len(case["bounds"]))]
            bound = -tolerance - (c[0] - j[0])
            # d.t (+ s) >= bound  ->  -d.t (- s) <= -bound
            a = [-v for v in d] + ([F(-1)] if with_slack else [])
            rows.append((a, -bound))
    return rows


def feasible(case, records, tolerance, bounds):
    rows = box(bounds) + declared_rows(case) + choice_rows(case, records, tolerance)
    return bool(vertices(rows, len(bounds)))


def solve_case(case):
    bounds = [(F(lo), F(hi)) for lo, hi in case["bounds"]]
    tolerance = F(case["tolerance"])
    dim = len(bounds)
    records = case["records"]
    rows = box(bounds) + declared_rows(case) + choice_rows(case, records, tolerance)
    points = vertices(rows, dim)
    if points:
        ranges = [[min(p[k] for p in points), max(p[k] for p in points)] for k in range(dim)]
        return {"learned": [[float(lo), float(hi)] for lo, hi in ranges]}
    ordered = sorted(records, key=lambda r: r["id"])
    smallest = None
    for size in range(1, len(ordered) + 1):
        for subset in itertools.combinations(ordered, size):
            if not feasible(case, list(subset), tolerance, bounds):
                smallest = [r["id"] for r in subset]
                break
        if smallest is not None:
            break
    # Slack program in (t, s): s >= 0, no upper bound needed for the minimum vertex.
    slack_rows = (
        box(bounds, extra=1)
        + declared_rows(case, extra=1)
        + choice_rows(case, records, tolerance, with_slack=True)
        + [([F(0)] * dim + [F(-1)], F(0))]
    )
    slack_points = vertices(slack_rows, dim + 1)
    slack = min(p[dim] for p in slack_points)
    return {
        "contradictory": {
            "records": smallest,
            "tolerance_needed": float(tolerance + slack),
            "tolerance_needed_exact": f"{(tolerance + slack).numerator}/{(tolerance + slack).denominator}",
        }
    }


def record(rid, options, chosen):
    return {"id": rid, "options": [{"id": i, "means": m} for i, m in options], "chosen": chosen}


ONE_PARAMETER_RECORDS = [
    record("r1", [("a", [0, 1]), ("b", [1, 0])], "a"),
    record("r2", [("a", [0, 1]), ("b", [3, 0]), ("c", [1, 0.75])], "b"),
]

TWO_PARAMETER_RECORDS = [
    record("r1", [("a", [0, 1, 1]), ("b", [1, 0, 0])], "a"),
    record("r2", [("a", [0, 1, -1]), ("b", [0.5, 0, 0])], "b"),
    record("r3", [("a", [0, 0, 1]), ("b", [-0.25, 1, 0]), ("c", [0.5, -0.5, 0.25])], "a"),
]

DECLARED = [{"terms": [[0, 1], [1, 2]], "sense": "at_most", "bound": 3}]

CASES = [
    {
        "name": "one_parameter_bracketed",
        "bounds": [[-4, 4]],
        "relations": [],
        "tolerance": 0,
        "records": ONE_PARAMETER_RECORDS,
    },
    {
        "name": "one_parameter_bracketed_with_tolerance",
        "bounds": [[-4, 4]],
        "relations": [],
        "tolerance": 0.25,
        "records": ONE_PARAMETER_RECORDS,
    },
    {
        "name": "two_parameters",
        "bounds": [[-2, 2], [-2, 2]],
        "relations": [],
        "tolerance": 0,
        "records": TWO_PARAMETER_RECORDS,
    },
    {
        "name": "two_parameters_declared_relation",
        "bounds": [[-2, 2], [-2, 2]],
        "relations": DECLARED,
        "tolerance": 0,
        "records": TWO_PARAMETER_RECORDS,
    },
    {
        "name": "two_parameters_declared_relation_with_tolerance",
        "bounds": [[-2, 2], [-2, 2]],
        "relations": DECLARED,
        "tolerance": 0.125,
        "records": TWO_PARAMETER_RECORDS,
    },
    {
        "name": "one_parameter_contradictory",
        "bounds": [[-4, 4]],
        "relations": [],
        "tolerance": 0,
        "records": ONE_PARAMETER_RECORDS + [record("r3", [("a", [0, 1]), ("b", [0.5, 0])], "b")],
    },
    {
        "name": "two_parameters_contradictory_through_declared_relation",
        "bounds": [[0, 2], [0, 2]],
        "relations": [{"terms": [[0, 1], [1, 1]], "sense": "at_most", "bound": 1}],
        "tolerance": 0.125,
        "records": [
            record("q1", [("a", [0, 1, 0]), ("b", [0.75, 0, 0])], "a"),
            record("q2", [("a", [0, 0, 1]), ("b", [0.75, 0, 0])], "a"),
            record("q3", [("a", [0, 0.5, 0.5]), ("b", [0.25, 0, 0])], "a"),
        ],
    },
]


def main():
    out = []
    for case in CASES:
        result = dict(case)
        result["expected"] = solve_case(case)
        out.append(result)
    path = Path(__file__).with_name("preference_set_learning_reference.json")
    path.write_text(json.dumps({"cases": out}, indent=1) + "\n")


if __name__ == "__main__":
    main()
