# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Closed-form delta-method interval of a derived decision margin.

A derived margin is a function f of the component means in each arm, and its contrast is
f(treated means) - f(control means). With gradients h_t and h_c of f at the two arm points and
the covariance of the arm-mean estimators, the first-order variance is

    Var = h_t' S_t h_t + h_c' S_c h_c            (separate arm covariances), or
    Var = g' S g,  g = (h_t, -h_c)               (one covariance of both arms' means),

and the 95% interval is contrast +- c * sqrt(Var). The critical value c is the normal quantile
for cluster and complete designs and the Student quantile with n - 1 degrees of freedom for
matched-pair and switchback designs (n independent pairs or sequences).

Everything is exact rational arithmetic until the final square root: the gradient comes from
forward-mode differentiation of the expression over Fractions, the quadratic form is summed
exactly, the normal quantile is the standard library's inverse normal CDF, and the Student
quantile is the root of the closed-form finite-sum CDF for odd degrees of freedom (bisection
in 200 steps). No Antecedent code is imported and no decision routine is called.

    uv run --frozen --script conformance/design/decision_margin_interval/reference.py --check
    uv run --frozen --script conformance/design/decision_margin_interval/reference.py --write

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected.json"
RTOL = 1e-10

getcontext().prec = 60


def frac(x) -> Fraction:
    return Fraction(str(x))


def dual(expr, point):
    """(value, gradient over inputs) of an expression at `point`, exact."""
    n = len(point)
    if "in" in expr:
        k = expr["in"]
        return point[k], [Fraction(int(i == k)) for i in range(n)]
    if "const" in expr:
        return frac(expr["const"]), [Fraction(0)] * n
    op = expr["op"]
    a, ga = dual(expr["args"][0], point)
    b, gb = dual(expr["args"][1], point)
    if op == "add":
        return a + b, [x + y for x, y in zip(ga, gb)]
    if op == "sub":
        return a - b, [x - y for x, y in zip(ga, gb)]
    if op == "mul":
        return a * b, [x * b + a * y for x, y in zip(ga, gb)]
    raise ValueError(op)


def arm_point(case, means):
    point = []
    for binding in case["inputs"]:
        if "measured" in binding:
            point.append(frac(means[case["components"].index(binding["measured"])]))
        else:
            point.append(frac(binding["fixed"]))
    return point


def component_gradient(case, grad):
    h = [Fraction(0)] * len(case["components"])
    for binding, g in zip(case["inputs"], grad):
        if "measured" in binding:
            h[case["components"].index(binding["measured"])] += g
    return h


def student_cdf_odd(t: float, nu: int) -> float:
    """CDF of Student t for odd degrees of freedom: 1/2 + (theta + sin(theta) * sum) / pi with
    theta = atan(t / sqrt(nu)) and sum = c + (2/3) c^3 + (2*4)/(3*5) c^5 + ... (c = cos theta,
    (nu - 1) / 2 terms)."""
    theta = math.atan(t / math.sqrt(nu))
    s, c = math.sin(theta), math.cos(theta)
    total, term = 0.0, c
    for k in range((nu - 1) // 2):
        total += term
        term *= c * c * (2 * k + 2) / (2 * k + 3)
    return 0.5 + (theta + s * total) / math.pi


def student_quantile(p: float, nu: int) -> float:
    if nu % 2 == 0:
        raise ValueError("reference covers odd degrees of freedom")
    lo, hi = 0.0, 1.0
    while student_cdf_odd(hi, nu) < p:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_cdf_odd(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve(case: dict) -> dict:
    vt, gt = dual(case["expression"], arm_point(case, case["treated_means"]))
    vc, gc = dual(case["expression"], arm_point(case, case["control_means"]))
    ht, hc = component_gradient(case, gt), component_gradient(case, gc)
    k = len(case["components"])
    dep = case["dependence"]
    if dep["kind"] == "arm_covariances":
        var = Fraction(0)
        for h, key in ((ht, "treated"), (hc, "control")):
            m = [frac(v) for v in dep[key]]
            var += sum(h[i] * m[i * k + j] * h[j] for i in range(k) for j in range(k))
    else:
        m = [frac(v) for v in dep["covariance"]]
        g = ht + [-v for v in hc]
        d = 2 * k
        var = sum(g[i] * m[i * d + j] * g[j] for i in range(d) for j in range(d))
    design = case["design"]
    se = float(Decimal(var.numerator).sqrt() / Decimal(var.denominator).sqrt())
    if design["assignment_design"] in ("matched_pairs", "switchback"):
        critical = student_quantile(0.975, design["treated_units"] - 1)
    else:
        critical = NormalDist().inv_cdf(0.975)
    estimate = float(vt - vc)
    return {
        "estimate": estimate,
        "variance": float(var),
        "standard_error": se,
        "critical_value": critical,
        "lower": estimate - critical * se,
        "upper": estimate + critical * se,
    }


def close(a: float, b: float) -> bool:
    return abs(a - b) <= RTOL * max(1.0, abs(a), abs(b))


def check(document: dict) -> list[str]:
    problems = []
    for case in document["cases"]:
        got = solve(case)
        for key, value in got.items():
            if not close(value, case["expected"][key]):
                problems.append(f"{case['name']}: {key} {case['expected'][key]} vs {value}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    for index, case in enumerate(document["cases"]):
        for key in ("standard_error", "lower", "upper"):
            wrong = copy.deepcopy(document)
            wrong["cases"][index]["expected"][key] *= 1.0 + 1e-6
            if not check(wrong):
                failures.append(f"{case['name']}: a perturbed {key} was accepted")
    return failures


def gram(rows: list[list[str]], ridge: str) -> list[str]:
    """A positive definite matrix A'A + ridge I, as exact decimal strings."""
    n = len(rows[0])
    out = []
    for i in range(n):
        for j in range(n):
            v = sum(frac(r[i]) * frac(r[j]) for r in rows) + (frac(ridge) if i == j else 0)
            out.append(format(Decimal(v.numerator) / Decimal(v.denominator), "f"))
    return out


def build() -> dict:
    components = ["frequency", "value", "cost"]
    margin = {
        "op": "mul",
        "args": [{"in": 0}, {"op": "sub", "args": [{"in": 1}, {"in": 2}]}],
    }
    measured = [{"measured": n} for n in components]
    arm_t = [0.0004, -0.0001, 0.0, -0.0001, 0.04, 0.0, 0.0, 0.0, 0.01]
    arm_c = [0.0003, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.01]
    a_rows = [
        ["0.02", "0.10", "-0.03", "0.01", "0.05", "0.02"],
        ["0.00", "0.20", "0.04", "0.00", "-0.10", "0.03"],
        ["0.01", "-0.05", "0.12", "0.02", "0.06", "-0.04"],
        ["0.03", "0.08", "-0.02", "0.15", "0.01", "0.05"],
    ]
    cases = [
        {
            "name": "cluster_product_margin_arm_covariances",
            "components": components,
            "inputs": measured,
            "expression": margin,
            "treated_means": [2.2, 12.0, 6.0],
            "control_means": [2.0, 11.0, 6.0],
            "dependence": {"kind": "arm_covariances", "treated": arm_t, "control": arm_c},
            "design": {"assignment_design": "cluster", "treated_units": 30, "control_units": 30, "outcome_rows": 480},
        },
        {
            "name": "complete_fixed_cost_margin",
            "components": components,
            "inputs": [{"measured": "frequency"}, {"measured": "value"}, {"fixed": 6.5}],
            "expression": margin,
            "treated_means": [2.2, 12.0, 6.0],
            "control_means": [2.0, 11.0, 6.0],
            "dependence": {"kind": "arm_covariances", "treated": arm_t, "control": arm_c},
            "design": {"assignment_design": "complete", "treated_units": 40, "control_units": 35, "outcome_rows": 600},
        },
        {
            "name": "matched_pairs_joint_arm_covariance_student_29",
            "components": components,
            "inputs": measured,
            "expression": margin,
            "treated_means": [2.2, 12.0, 6.0],
            "control_means": [2.0, 11.0, 6.0],
            "dependence": {"kind": "joint_arms", "covariance": gram(a_rows, "0.01")},
            "design": {"assignment_design": "matched_pairs", "treated_units": 30, "control_units": 30, "outcome_rows": 60},
        },
        {
            "name": "switchback_profit_margin_joint_arm_covariance_student_29",
            "components": components,
            "inputs": measured,
            "expression": {
                "op": "sub",
                "args": [{"op": "mul", "args": [{"in": 0}, {"in": 1}]}, {"in": 2}],
            },
            "treated_means": [2.5, 9.0, 4.0],
            "control_means": [2.1, 9.5, 4.5],
            "dependence": {"kind": "joint_arms", "covariance": gram(a_rows[::-1], "0.02")},
            "design": {"assignment_design": "switchback", "treated_units": 30, "control_units": 30, "outcome_rows": 240},
        },
    ]
    for case in cases:
        case["expected"] = solve(case)
    return {
        "schema_version": 2,
        "scope": "Delta-method sampling interval of a derived decision margin (a function of arm component means) at four hand cases: cluster and complete designs with separate arm covariances, matched-pair and switchback designs with one covariance of both arms' means.",
        "tolerance": {"relative": RTOL},
        "oracle": {
            "kind": "closed_form",
            "project": "closed-form first-order delta method with exact rational arithmetic",
            "method": "forward-mode gradient of the margin expression over Fractions, exact quadratic form g'Sg, normal quantile from the standard library, Student quantile from the finite-sum CDF for odd degrees of freedom",
            "generator": "conformance/design/decision_margin_interval/reference.py",
            "command": "uv run --frozen --script conformance/design/decision_margin_interval/reference.py --check",
            "environment_lock": "conformance/design/decision_margin_interval/reference.py.lock",
            "generator_retained": True,
            "limitations": "Hand-constructed covariances; the interval is the first-order sampling interval and carries no coverage claim.",
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        fresh = build()
        if EXPECTED.exists():
            previous = json.loads(EXPECTED.read_text(encoding="utf-8"))
            review = previous.get("oracle", {}).get("reference_review")
            if review is not None:
                fresh["oracle"]["reference_review"] = review
        EXPECTED.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
        return 0
    document = json.loads(EXPECTED.read_text(encoding="utf-8"))
    problems = check(document) + self_test(document)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"{len(document['cases'])} cases agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
