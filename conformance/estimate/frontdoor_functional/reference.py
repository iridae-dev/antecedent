# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact enumeration reference for the front-door functional fixture.

Binary structural model U ~ Bern(1/2), P(T=1|U) = 1/10 + U/2, P(M=1|T) = 1/10 + 7T/10,
P(Y=1|M,U) = 1/20 + 9MU/10, with U unobserved.  Two quantities are derived in exact
rational arithmetic from those structural equations alone:

* the front-door effect of T on Y through M, by enumeration over the latent U:
  sum_u P(u) sum_m [P(m|do(T=1)) - P(m|do(T=0))] E[Y | m, u]  (P(m|do(t)) = P(m|t) here
  because U does not enter the mediator equation);
* the probability limit of the linear product-of-coefficients shortcut on the observed
  (U-marginalized) law: a * b with a the T->M slope and b the coefficient of M in the
  OLS regression of Y on (1, T, M), computed as the P(t) Var(M|t) weighted mean of the
  within-arm slopes.

It also verifies that every one of the 8 x 2 observed cell probabilities is an integer
multiple of 1/4000, so the 4000-row population table in the test is exact.  No Antecedent
code, no estimator and no sampling.

    uv run --frozen --script conformance/estimate/frontdoor_functional/reference.py --check
    uv run --frozen --script conformance/estimate/frontdoor_functional/reference.py --self-test

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
LIMIT_ATOL = 5e-5  # the fixture records the shortcut limit to four decimals


def p_t(t, u):
    p1 = F(1, 10) + F(1, 2) * u
    return p1 if t else 1 - p1


def p_m(m, t):
    p1 = F(1, 10) + F(7, 10) * t
    return p1 if m else 1 - p1


def p_y1(m, u):
    return F(1, 20) + F(9, 10) * m * u


def compute() -> dict:
    # Front-door contrast by enumeration over the latent confounder.
    truth = F(0)
    for u in (0, 1):
        for m in (0, 1):
            truth += F(1, 2) * (p_m(m, 1) - p_m(m, 0)) * p_y1(m, u)

    # Observed (U marginalized) cell probabilities over (t, m, y).
    cell = {}
    for u, t, m, y in itertools.product((0, 1), repeat=4):
        py = p_y1(m, u) if y else 1 - p_y1(m, u)
        cell[(t, m, y)] = cell.get((t, m, y), F(0)) + F(1, 2) * p_t(t, u) * p_m(m, t) * py
    exact_counts = all((c * 4000).denominator == 1 for c in cell.values())
    assert sum(cell.values()) == 1

    def arm(t):
        pt = sum(c for (tt, _, _), c in cell.items() if tt == t)
        e = lambda f: sum(c * f(m, y) for (tt, m, y), c in cell.items() if tt == t) / pt
        var_m = e(lambda m, y: m * m) - e(lambda m, y: m) ** 2
        cov = e(lambda m, y: m * y) - e(lambda m, y: m) * e(lambda m, y: y)
        return pt, var_m, cov / var_m if var_m else F(0)

    arms = [arm(0), arm(1)]
    pooled_b = sum(pt * v * b for pt, v, b in arms) / sum(pt * v for pt, v, _ in arms)
    a = p_m(1, 1) - p_m(1, 0)
    # The T->M slope from the observed law equals the structural one (U does not enter M).
    mean_m = lambda t: sum(c for (tt, m, _), c in cell.items() if tt == t and m) / arms[t][0]
    assert mean_m(1) - mean_m(0) == a
    return {
        "true_effect": float(truth),
        "linear_path_product_limit": float(a * pooled_b),
        "p_treated": float(arms[1][0]),
        "within_arm_slopes": [float(arms[0][2]), float(arms[1][2])],
        "exact_4000_cells": exact_counts,
    }


def check(document: dict) -> list[str]:
    got = compute()
    problems = []
    if not got["exact_4000_cells"]:
        problems.append("observed cell probabilities are not multiples of 1/4000")
    if abs(got["true_effect"] - document["true_effect"]) > document["tolerance"]:
        problems.append(f"true_effect {document['true_effect']} vs {got['true_effect']}")
    if abs(got["linear_path_product_limit"] - document["linear_path_product_limit"]) > LIMIT_ATOL:
        problems.append(
            f"linear limit {document['linear_path_product_limit']} vs {got['linear_path_product_limit']}"
        )
    if abs(got["p_treated"] - 0.35) > 1e-12:
        problems.append("P(T=1) is not 0.35")
    for slope, stated in zip(got["within_arm_slopes"], (0.2769, 0.7714), strict=True):
        if abs(slope - stated) > 5e-5:
            problems.append(f"within-arm slope {slope} vs documented {stated}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["true_effect"] = 0.3631
    if not check(wrong):
        failures.append("path-product value accepted as the front-door truth")
    wrong = copy.deepcopy(document)
    wrong["linear_path_product_limit"] = 0.315
    if not check(wrong):
        failures.append("front-door truth accepted as the shortcut limit")
    wrong = copy.deepcopy(document)
    wrong["true_effect"] += 1e-6
    if not check(wrong):
        failures.append("shifted truth accepted")
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
