# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the cross-world class fixture from the enumerated count tables.

Every expected value is a rational function of the declared cell counts, evaluated with exact
arithmetic: the effect on the treated of the front-door ADMG, the unconfounded contrast, the
natural direct effect of a Markovian mediator model, the path-specific effect of a two-mediator
model and the exact least-squares slopes of a noiseless line. Standard library only; imports no
Antecedent code.

    uv run --frozen --script conformance/estimate/cross_world_class_truth/reference.py --check
    uv run --frozen --script conformance/estimate/cross_world_class_truth/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = Fraction
FIXTURE = HERE / "expected.json"


def counts(document, name):
    table = document["tables"][name]
    return table["columns"], {tuple(r["levels"]): F(r["count"]) for r in table["counts"]}


def marginal(n, columns, spec):
    """Sum of counts over the cells matching `spec` ({column: level})."""
    index = {c: i for i, c in enumerate(columns)}
    return sum(v for k, v in n.items() if all(k[index[c]] == l for c, l in spec.items()))


def p(n, columns, target, given):
    """P(target | given) from counts, both {column: level}."""
    return marginal(n, columns, {**target, **given}) / marginal(n, columns, given)


def ett_front_door(document, case):
    columns, n = counts(document, case["table"])
    t, m, y = case["treatment"], case["mediator"], case["outcome"]
    active, control = case["active"], case["control"]
    inside = sum(
        p(n, columns, {m: lv}, {t: active}) * p(n, columns, {y: 1}, {m: lv, t: control})
        for lv in (0, 1)
    )
    return inside - p(n, columns, {y: 1}, {t: control})


def ett_adjust_one(document, case):
    """Effect on the treated when the outcome's only other parent z is not a descendant of the treatment."""
    columns, n = counts(document, case["table"])
    t, z, y = case["treatment"], case["adjust"], case["outcome"]
    a, c = case["active"], case["control"]
    inside = sum(
        p(n, columns, {z: lv}, {t: c}) * p(n, columns, {y: 1}, {t: a, z: lv}) for lv in (0, 1)
    )
    return inside - p(n, columns, {y: 1}, {t: c})


def nde_markov(document, case):
    columns, n = counts(document, case["table"])
    t, m, y = case["treatment"], case["mediator"], case["outcome"]
    a, c = case["active"], case["control"]
    return sum(
        p(n, columns, {m: lv}, {t: c})
        * (p(n, columns, {y: 1}, {t: a, m: lv}) - p(n, columns, {y: 1}, {t: c, m: lv}))
        for lv in (0, 1)
    )


def nde_root_mediator(document, case):
    """Natural direct effect when the mediator is a root: the mediator keeps its marginal law."""
    columns, n = counts(document, case["table"])
    t, m, y = case["treatment"], case["mediator"], case["outcome"]
    a, c = case["active"], case["control"]
    return sum(
        marginal(n, columns, {m: lv})
        / marginal(n, columns, {})
        * (p(n, columns, {y: 1}, {t: a, m: lv}) - p(n, columns, {y: 1}, {t: c, m: lv}))
        for lv in (0, 1)
    )


def path_specific_two_mediators(document, case):
    """Active on t -> b, control on t -> a: sum_a P(a|c) sum_b [P(b|a_t) - P(b|c_t)] P(y|a,b)."""
    columns, n = counts(document, case["table"])
    t, a_, b_, y = case["treatment"], case["mediator"], case["other"], case["outcome"]
    act, con = case["active"], case["control"]
    total = F(0)
    for av in (0, 1):
        for bv in (0, 1):
            total += (
                p(n, columns, {a_: av}, {t: con})
                * (p(n, columns, {b_: bv}, {t: act}) - p(n, columns, {b_: bv}, {t: con}))
                * p(n, columns, {y: 1}, {a_: av, b_: bv})
            )
    return total


def line_slope(document, case):
    xs = [F(i, 20) - 2 for i in range(case["rows"])]
    ys = [F(case["intercept"]) + F(case["slope"]) * x for x in xs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


FORMULAS = {
    "ett_front_door": ett_front_door,
    "ett_adjust_one": ett_adjust_one,
    "nde_markov": nde_markov,
    "nde_root_mediator": nde_root_mediator,
    "path_specific_two_mediators": path_specific_two_mediators,
}


def check(document):
    problems = []
    for name, case in document["cases"].items():
        got = FORMULAS[case["formula"]](document, case)
        if not math.isclose(float(got), case["expected"], rel_tol=0.0, abs_tol=1e-12):
            problems.append(f"{name}: derived {float(got)!r} != fixture {case['expected']!r}")
    mixture = document["posterior"]
    atoms = mixture["atoms"]
    identified = sum(F(a["weight"]) for a in atoms if a["formula"] is not None)
    value = sum(
        F(a["weight"]) * FORMULAS[a["formula"]](document, a["case"]) for a in atoms if a["formula"] is not None
    ) / identified
    if not math.isclose(float(value), mixture["expected"], rel_tol=0.0, abs_tol=1e-12):
        problems.append(f"posterior: derived {float(value)!r} != fixture {mixture['expected']!r}")
    if float(1 - identified) != mixture["unidentified_mass"]:
        problems.append("posterior: unidentified mass is the weight of the atoms with no formula")
    line = document["line"]
    slope = line_slope(document, line)
    if slope != F(str(line["completion_slopes"][0])):
        problems.append(f"line: exact slope {slope} != fixture {line['completion_slopes'][0]!r}")
    if line["completion_slopes"][1] != 0.0 or line["mean"] != float(slope / 2):
        problems.append("line: reverse completion leaves y a root (slope 0) and the mean is the average")
    if line["identified_set"] != [0.0, float(slope)]:
        problems.append("line: identified set must be [0, slope]")
    return problems


MUTATIONS = [
    ("front-door effect changed", lambda d: d["cases"]["admg_counterfactual"].__setitem__("expected", 0.25)),
    ("count changed", lambda d: d["tables"]["front_door"]["counts"][7].__setitem__("count", 400)),
    ("nde changed", lambda d: d["cases"]["admg_natural_direct"].__setitem__("expected", 0.3)),
    ("two-mediator effect changed", lambda d: d["cases"]["cpdag_path_specific"].__setitem__("expected", 0.0)),
    ("posterior mixture changed", lambda d: d["posterior"].__setitem__("expected", 0.38)),
    ("slope mean changed", lambda d: d["line"].__setitem__("mean", 2.0)),
]


def self_test(document):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
