# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the interventional-distribution fixture for classes (CPDAG and PAG).

Binary structural causal models, every noise independent and uniform on four equally likely bins,
every threshold a multiple of 1/4. Enumerating the noise cells with exact rational arithmetic gives

* the interventional law P(Y = y | do(T = 1)) by evaluating the outcome equation under the
  assignment T = 1 on every noise cell (the structural definition, no identification formula), and
* the observed law, each noise cell contributing one row, so the contingency table is the law
  times the number of cells divided by the gcd of the counts.

class_point (columns w, t, x, y; graph w -> t, t -> x, x -> y, t -> y):

    W = 1[Uw < 2/4]      T = 1[Ut < (1 + 2w)/4]      X(t) = 1[Ux < (1 + 2t)/4]
    Y(t, x) = 1[Uy < q(t, x)/4], q = {(0,0): 1, (1,0): 2, (0,1): 2, (1,1): 3}

With every edge compelled (a CPDAG with one completion) or every mark decided and visible (a PAG)
the distribution is a point: P(y | do(t)) = sum_x P(x | t) P(y | t, x), which the table-only cross
check reproduces exactly.

class_disagree (columns t, x, y; the CPDAG has T - X undirected, X -> Y, T -> Y): the same model
without w and with T = 1[Ut < 2/4]. The class has two completions, T -> X and X -> T. Their
interventional probabilities are sum_x P(x | t) P(y | t, x) (the generating completion: the
structural truth) and sum_x P(x) P(y | t, x) (X is a non-descendant of T and so adjusted). The
two differ, so the distribution is partially identified: per atom the lowest and highest
probability over the completions, and the enumeration-weighted (equal weights) mean beside them.

    uv run --frozen --script conformance/estimate/interventional_distribution_classes/reference.py --check
    uv run --frozen --script conformance/estimate/interventional_distribution_classes/reference.py --self-test

Standard library only; imports no Antecedent code and no statistics package.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "expected.json"
BINS = 4
Y_QUARTERS = {(0, 0): 1, (1, 0): 2, (0, 1): 2, (1, 1): 3}


def below(bin_: int, quarters: int) -> int:
    return int(bin_ < quarters)


def point_cells():
    for uw, ut, ux, uy in itertools.product(range(BINS), repeat=4):
        w = below(uw, 2)
        t = below(ut, 1 + 2 * w)
        x_of = {tt: below(ux, 1 + 2 * tt) for tt in (0, 1)}
        y_of = {key: below(uy, q) for key, q in Y_QUARTERS.items()}
        yield {"w": w, "t": t}, x_of, y_of


def disagree_cells():
    for ut, ux, uy in itertools.product(range(BINS), repeat=3):
        t = below(ut, 2)
        x_of = {tt: below(ux, 1 + 2 * tt) for tt in (0, 1)}
        y_of = {key: below(uy, q) for key, q in Y_QUARTERS.items()}
        yield {"t": t}, x_of, y_of


def observed_table(cells, columns):
    counts: dict[tuple[int, ...], int] = {}
    for root, x_of, y_of in cells:
        t = root["t"]
        x = x_of[t]
        y = y_of[(t, x)]
        row = {**root, "x": x, "y": y}
        key = tuple(row[name] for name in columns)
        counts[key] = counts.get(key, 0) + 1
    common = math.gcd(*counts.values())
    return [
        {**{name: float(v) for name, v in zip(columns, key, strict=True)}, "count": n // common}
        for key, n in sorted(counts.items())
    ]


def structural_p_y1_do(cells, t_level: int) -> F:
    """P(Y = 1 | do(T = t)): the outcome equation under T = t on every noise cell."""
    cells = list(cells)
    total = sum(y_of[(t_level, x_of[t_level])] for _, x_of, y_of in cells)
    return F(total, len(cells))


def rows_of(table, columns):
    return [(tuple(int(c[name]) for name in columns), c["count"]) for c in table]


def prob(rows, index, target, level, given):
    num = den = 0
    for key, count in rows:
        if all(key[index[n]] == v for n, v in given.items()):
            den += count
            num += count * (key[index[target]] == level)
    return F(num, den)


def g_formula(table, columns, t_level: int, adjusted: bool) -> F:
    """P(Y = 1 | do(T = t)) from the table alone: sum_x P(x | t or marginal) P(y=1 | t, x)."""
    rows = rows_of(table, columns)
    index = {name: i for i, name in enumerate(columns)}
    total = F(0)
    for x in (0, 1):
        px = prob(rows, index, "x", x, {}) if adjusted else prob(rows, index, "x", x, {"t": t_level})
        total += px * prob(rows, index, "y", 1, {"t": t_level, "x": x})
    return total


def atoms(p1: F) -> dict[str, float]:
    return {"y0": float(1 - p1), "y1": float(p1)}


def build() -> dict:
    point_columns = ("w", "t", "x", "y")
    disagree_columns = ("t", "x", "y")
    point_table = observed_table(point_cells(), point_columns)
    disagree_table = observed_table(disagree_cells(), disagree_columns)
    point_truth = structural_p_y1_do(point_cells(), 1)
    generating = structural_p_y1_do(disagree_cells(), 1)
    adjusted = g_formula(disagree_table, disagree_columns, 1, True)
    low, high = min(generating, adjusted), max(generating, adjusted)
    mean = (generating + adjusted) / 2
    return {
        "schema_version": 1,
        "case": "interventional_distribution_classes",
        "query": {"outcome": "y", "do": {"t": 1.0}},
        "class_point": {
            "columns": list(point_columns),
            "graph": {
                "directed_edges": [["w", "t"], ["t", "x"], ["x", "y"], ["t", "y"]],
            },
            "contingency_table": point_table,
            "truth": atoms(point_truth),
        },
        "class_disagree": {
            "columns": list(disagree_columns),
            "cpdag": {
                "directed_edges": [["x", "y"], ["t", "y"]],
                "undirected_edges": [["t", "x"]],
            },
            "contingency_table": disagree_table,
            "completions": 2,
            "truth": {
                "y1": {"lower": float(low), "upper": float(high), "mean": float(mean)},
                "y0": {"lower": float(1 - high), "upper": float(1 - low), "mean": float(1 - mean)},
            },
        },
        "frequentist": {"estimator": "functional.distribution", "absolute_tolerance": 1e-9},
        "bayesian": {
            "estimator": "functional.distribution",
            "posterior_mean_tolerance": 0.03,
            "note": "Dirichlet row posterior mean within tolerance of the truth; declared acceptance band",
        },
    }


def close(got, want, label, problems, tol=1e-12):
    if not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got)!r} != fixture {want!r}")


def check(document: dict) -> list[str]:
    problems: list[str] = []
    point_columns = tuple(document["class_point"]["columns"])
    disagree_columns = tuple(document["class_disagree"]["columns"])
    if document["class_point"]["contingency_table"] != observed_table(point_cells(), point_columns):
        problems.append("class_point table is not the exact law of the SCM")
    if document["class_disagree"]["contingency_table"] != observed_table(
        disagree_cells(), disagree_columns
    ):
        problems.append("class_disagree table is not the exact law of the SCM")
    truth = structural_p_y1_do(point_cells(), 1)
    close(truth, document["class_point"]["truth"]["y1"], "class_point y1", problems)
    close(1 - truth, document["class_point"]["truth"]["y0"], "class_point y0", problems)
    close(
        g_formula(document["class_point"]["contingency_table"], point_columns, 1, False),
        document["class_point"]["truth"]["y1"],
        "class_point g-formula from the table",
        problems,
    )
    table = document["class_disagree"]["contingency_table"]
    generating = structural_p_y1_do(disagree_cells(), 1)
    close(
        g_formula(table, disagree_columns, 1, False),
        generating,
        "generating completion from the table",
        problems,
    )
    adjusted = g_formula(table, disagree_columns, 1, True)
    y1 = document["class_disagree"]["truth"]["y1"]
    y0 = document["class_disagree"]["truth"]["y0"]
    close(min(generating, adjusted), y1["lower"], "y1 lower", problems)
    close(max(generating, adjusted), y1["upper"], "y1 upper", problems)
    close((generating + adjusted) / 2, y1["mean"], "y1 mean", problems)
    close(1 - max(generating, adjusted), y0["lower"], "y0 lower", problems)
    close(1 - min(generating, adjusted), y0["upper"], "y0 upper", problems)
    if abs(generating - adjusted) < F(1, 100):
        problems.append("the completions must disagree materially")
    return problems


MUTATIONS = [
    ("a count changed", lambda d: d["class_point"]["contingency_table"][0].__setitem__("count", 99)),
    ("point truth changed", lambda d: d["class_point"]["truth"].__setitem__("y1", 0.5)),
    ("upper changed", lambda d: d["class_disagree"]["truth"]["y1"].__setitem__("upper", 0.9)),
    ("mean changed", lambda d: d["class_disagree"]["truth"]["y1"].__setitem__("mean", 0.1)),
]


def self_test(document: dict) -> list[str]:
    failures = []
    if check(document):
        failures.append("the unmodified fixture is rejected")
    for label, mutate in MUTATIONS:
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"accepted: {label}")
    return failures


def main() -> int:
    if "--write" in sys.argv:
        retained = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.is_file() else {}
        document = {"oracle": retained.get("oracle", {}), **build()}
        FIXTURE.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
        return 0
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if "--self-test" in sys.argv else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
