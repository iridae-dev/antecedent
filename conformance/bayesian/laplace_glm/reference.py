# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the coefficients behind the laplace_glm fixture: on the test's noiseless design (60 rows, x_r = r/10, y = 0.5 + 1.5 x) the exact least-squares coefficients over rationals are (0.5, 1.5); a weakly informative prior moves the MAP only by a prior-scale-sized amount.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/laplace_glm/reference.py --check
    uv run --frozen --script conformance/bayesian/laplace_glm/reference.py --self-test

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


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(float(got), want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


def solve(a, b):
    n = len(a)
    m = [row[:] + [bi] for row, bi in zip(a, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        m[c] = [v / m[c][c] for v in m[c]]
        for r in range(n):
            if r != c:
                m[r] = [x - m[r][c] * y for x, y in zip(m[r], m[c])]
    return [row[n] for row in m]

def least_squares(columns, y):
    a = [[sum(p * q for p, q in zip(ci, cj)) for cj in columns] for ci in columns]
    b = [sum(p * q for p, q in zip(ci, y)) for ci in columns]
    return solve(a, b)

N = 60

def derived():
    x = [Fraction(r, 10) for r in range(N)]
    y = [Fraction(1, 2) + Fraction(3, 2) * xi for xi in x]
    return least_squares([[Fraction(1)] * N, x], y)

def check(document):
    problems = []
    got = derived()
    if len(document["true_coefficients"]) != 2:
        return ["fixture does not state an intercept and a slope"]
    for i, want in enumerate(document["true_coefficients"]):
        close(got[i], want, 1e-12, f"true_coefficients[{i}]", problems)
    if not document["tolerance"] > 0:
        problems.append("tolerance is not positive")
    return problems

MUTATIONS = [
    ("slope shifted by 0.1", lambda d: d["true_coefficients"].__setitem__(1, 1.6)),
    ("intercept and slope swapped", lambda d: d["true_coefficients"].reverse()),
    ("tolerance made non-positive", lambda d: d.__setitem__("tolerance", 0.0)),
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
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
