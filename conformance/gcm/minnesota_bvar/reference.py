# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact-rational reference for the Minnesota-style standardized ridge fit.

The data are rebuilt from the fixture's stated formulas (the trigonometric values are taken once in
double precision and then treated as exact rationals). With Z = [1, x1/s1, x2/s2], s_j the
population standard deviation of parent j, the coefficients solve

    (Z'Z + diag(0, 1/phi, 1/phi)) b = Z'y        (intercept unpenalised, phi = 0.2)

by Gaussian elimination over Fractions; the reported slopes are b_j / s_j and the residual scale is
sqrt(RSS / (n - 3)) over the n data rows, with no prior pseudo-rows. The only floating-point steps
are the sines and cosines of the inputs, the two population standard deviations (a square root)
and the final residual-scale square root; the linear solve itself is exact. No Antecedent code.

    uv run --frozen --script conformance/gcm/minnesota_bvar/reference.py --check
    uv run --frozen --script conformance/gcm/minnesota_bvar/reference.py --self-test

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
PHI = Fraction(1, 5)


def solve(a: list[list[Fraction]], b: list[Fraction]) -> list[Fraction]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        pivot = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[pivot] = m[pivot], m[c]
        p = m[c][c]
        m[c] = [v / p for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [m[i][n] for i in range(n)]


def compute(n: int) -> dict:
    x1 = [Fraction(math.sin(0.13 * i)) for i in range(n)]
    x2 = [Fraction(math.cos(0.07 * i)) for i in range(n)]
    y = [
        Fraction(1.1)
        + Fraction(0.75) * x1[i]
        - Fraction(0.35) * x2[i]
        + Fraction(0.12) * Fraction(math.sin(0.41 * i))
        for i in range(n)
    ]
    scales = []
    for col in (x1, x2):
        mean = sum(col, Fraction(0)) / n
        var = sum(((v - mean) ** 2 for v in col), Fraction(0)) / n
        scales.append(Fraction(math.sqrt(var)))
    z = [[Fraction(1), x1[i] / scales[0], x2[i] / scales[1]] for i in range(n)]
    gram = [[sum((z[r][i] * z[r][j] for r in range(n)), Fraction(0)) for j in range(3)] for i in range(3)]
    for j in (1, 2):
        gram[j][j] += 1 / PHI
    rhs = [sum((z[r][i] * y[r] for r in range(n)), Fraction(0)) for i in range(3)]
    beta = solve(gram, rhs)
    rss = sum(
        ((y[r] - sum(beta[j] * z[r][j] for j in range(3))) ** 2 for r in range(n)), Fraction(0)
    )
    return {
        "intercept": float(beta[0]),
        "coefficients": [float(beta[1] / scales[0]), float(beta[2] / scales[1])],
        "data_residual_sigma": math.sqrt(float(rss / (n - 3))),
    }


def check(document: dict) -> list[str]:
    problems = []
    got = compute(document["data"]["n"])
    ref = document["reference"]
    atol = document["acceptance"]["atol"]
    if abs(got["intercept"] - ref["intercept"]) > atol:
        problems.append(f"intercept {ref['intercept']} vs {got['intercept']}")
    for index, (a, b) in enumerate(zip(got["coefficients"], ref["coefficients"])):
        if abs(a - b) > atol:
            problems.append(f"coefficient {index}: {b} vs {a}")
    if abs(got["data_residual_sigma"] - ref["data_residual_sigma"]) > atol:
        problems.append(f"sigma {ref['data_residual_sigma']} vs {got['data_residual_sigma']}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    for key in ("intercept", "data_residual_sigma"):
        wrong = copy.deepcopy(document)
        wrong["reference"][key] += 1e-6
        if not check(wrong):
            failures.append(f"shifted {key} accepted")
    wrong = copy.deepcopy(document)
    wrong["reference"]["coefficients"][1] += 1e-6
    if not check(wrong):
        failures.append("shifted coefficient accepted")
    wrong = copy.deepcopy(document)
    wrong["data"]["n"] = 91
    if not check(wrong):
        failures.append("changed n accepted")
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
