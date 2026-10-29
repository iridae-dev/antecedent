# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the counterfactual graph-posterior fixture from the enumerated count tables.

A counterfactual is the effect on the treated, E[Y_active - Y_control | T = active]. On the front-door ADMG
(treatment -> mediator -> outcome with a latent common cause of treatment and outcome) it is the front-door
functional of the cell counts; on the same variables with no latent cause it is the observed contrast
P(y | active) - P(y | control). The posterior holds those two ADMG atoms at weights 1/2 and 3/10 beside an
atom with no identification (weight 1/5, a cyclic sample): the posterior-weighted mean over the atoms that
identify it is (1/2 * front-door + 3/10 * contrast) / (4/5), and the unidentified mass is 1/5. The same
formulas give the PAG case (one visible atom, weight 4/5, beside an atom with an invisible treatment edge,
weight 1/5), the noiseless-line CPDAG case (identified-set mean) and the Gaussian DAG case (coefficient
times contrast). All values are rational functions of the counts, evaluated with exact arithmetic.

Standard library only; imports no Antecedent code.

    uv run --frozen --script conformance/estimate/cross_world_counterfactual_posterior/reference.py --check
    uv run --frozen --script conformance/estimate/cross_world_counterfactual_posterior/reference.py --self-test

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


def ett_unconfounded(document, case):
    """Effect on the treated with no latent cause: the observed contrast."""
    columns, n = counts(document, case["table"])
    t, y = case["treatment"], case["outcome"]
    return p(n, columns, {y: 1}, {t: case["active"]}) - p(n, columns, {y: 1}, {t: case["control"]})


FORMULAS = {
    "ett_front_door": ett_front_door,
    "ett_adjust_one": ett_adjust_one,
    "ett_unconfounded": ett_unconfounded,
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
        F(a["weight"]) * FORMULAS[a["formula"]](document, a["case"])
        for a in atoms
        if a["formula"] is not None
    ) / identified
    if not math.isclose(float(value), mixture["expected"], rel_tol=0.0, abs_tol=1e-12):
        problems.append(f"posterior: derived {float(value)!r} != fixture {mixture['expected']!r}")
    if float(1 - identified) != mixture["unidentified_mass"]:
        problems.append("posterior: unidentified mass is the weight of the atoms with no formula")
    line = document["line"]
    xs = [F(i, 20) - 2 for i in range(line["rows"])]
    ys = [F(line["intercept"]) + F(line["slope"]) * x for x in xs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    if float(slope / 2) != line["mean"]:
        problems.append("line: the mean of the completion slopes (slope and 0) is slope / 2")
    gaussian = document["gaussian"]
    if gaussian["expected"] != gaussian["coefficient"] * (gaussian["active"] - gaussian["control"]):
        problems.append("gaussian: effect is the outcome coefficient times the contrast")
    return problems


MUTATIONS = [
    ("front-door effect changed", lambda d: d["cases"]["admg_counterfactual"].__setitem__("expected", 0.25)),
    ("count changed", lambda d: d["tables"]["front_door"]["counts"][7].__setitem__("count", 400)),
    ("posterior mixture changed", lambda d: d["posterior"].__setitem__("expected", 0.38)),
    ("unidentified mass changed", lambda d: d["posterior"].__setitem__("unidentified_mass", 0.3)),
    ("line mean changed", lambda d: d["line"].__setitem__("mean", 2.0)),
    ("gaussian effect changed", lambda d: d["gaussian"].__setitem__("expected", 2.0)),
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
