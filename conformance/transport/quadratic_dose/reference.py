# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact closed-form reference for the quadratic conditional-mean dose fixture.

With m(d) = b0 + b1 d + b2 d^2 the level, derivative and contrast are polynomial
identities: m'(d) = b1 + 2 b2 d and m(to) - m(from) = b1 (to - from) + b2 (to^2 - from^2).
They are evaluated here in exact rational arithmetic from the fixture coefficients.

The zero-smoothing-bias statement is checked directly as well: a weighted local-quadratic
least-squares fit of noiseless m(d) on a window of grid points, with arbitrary positive
rational weights, reproduces (m(d0), m'(d0), m''(d0)/2) exactly, because the local design
spans the true mean.  The normal equations are solved by exact Gaussian elimination over
rationals.  No Antecedent code and no sampling.

    uv run --frozen --script conformance/transport/quadratic_dose/reference.py --check
    uv run --frozen --script conformance/transport/quadratic_dose/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mean(b, d):
    return b[0] + b[1] * d + b[2] * d * d


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


def local_fit(b, d0, half_width=F(2, 5), points=9):
    """Weighted local quadratic fit of noiseless m around d0 with triangular weights."""
    grid = [d0 - half_width + 2 * half_width * F(i, points - 1) for i in range(points)]
    weights = [1 - abs(d - d0) / (2 * half_width) for d in grid]
    basis = [[F(1), d - d0, (d - d0) ** 2] for d in grid]
    normal = [
        [sum(w * row[i] * row[j] for w, row in zip(weights, basis)) for j in range(3)]
        for i in range(3)
    ]
    rhs = [sum(w * row[i] * mean(b, d) for w, row, d in zip(weights, basis, grid)) for i in range(3)]
    return solve(normal, rhs)


def check(document: dict) -> list[str]:
    b = [F(str(c)) for c in document["conditional_mean_coefficients"]]
    problems = []
    lo, hi = F(str(document["from"])), F(str(document["to"]))
    contrast = mean(b, hi) - mean(b, lo)
    if abs(float(contrast) - document["contrast"]) > 1e-15:
        problems.append(f"contrast {document['contrast']} vs {float(contrast)}")
    if (lo, hi) != (1, 3):
        problems.append("contrast doses differ from the asserted 1 -> 3")
    for d0 in (F(1, 2), F(1), F(2), F(7, 2)):
        level, slope, curvature = local_fit(b, d0)
        if level != mean(b, d0) or slope != b[1] + 2 * b[2] * d0 or curvature != b[2]:
            problems.append(f"local quadratic fit is biased at {d0}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["contrast"] = 2.0
    if not check(wrong):
        failures.append("wrong contrast accepted")
    wrong = copy.deepcopy(document)
    wrong["conditional_mean_coefficients"][2] = 0.3
    if not check(wrong):
        failures.append("changed curvature accepted")
    wrong = copy.deepcopy(document)
    wrong["to"] = 2.0
    if not check(wrong):
        failures.append("changed contrast dose accepted")
    # A cubic mean is not spanned by the local quadratic design: the fitted slope must be biased.
    cubic = lambda d: d**3
    grid = [F(3, 5) + F(2, 5) * F(i, 4) for i in range(9)]
    weights = [1 - abs(d - F(1)) / F(4, 5) for d in grid]
    basis = [[F(1), d - 1, (d - 1) ** 2] for d in grid]
    normal = [
        [sum(w * r[i] * r[j] for w, r in zip(weights, basis)) for j in range(3)] for i in range(3)
    ]
    rhs = [sum(w * r[i] * cubic(d) for w, r, d in zip(weights, basis, grid)) for i in range(3)]
    if solve(normal, rhs)[1] == 3:
        failures.append("local quadratic slope was exact for a cubic mean")
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
