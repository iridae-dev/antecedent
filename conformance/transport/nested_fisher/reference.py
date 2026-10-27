# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact delta-method reference for the selected binary nested-Markov covariance.

The generating law is X1, X2 fair independent bits, P(M=0|X2) = 3/4, 1/4 and
P(Y=0|M) = 4/5, 2/5 for M = 0, 1.  The target functionals are the two Verma
do-means  mean_x = sum_m P(m|x) sum_x' P(x') E[Y|m,x']  and their contrast.  X1 does not
enter either functional and is independent of the rest, so the information about the
functionals lies in the 8-cell table over (X2, M, Y), whose 7-parameter factorization is
saturated; the large-sample covariance of the maximum-likelihood functionals is therefore
the multinomial delta method

    Cov = G (diag(p) - p p^T) G^T / N,

with G the exact gradient of each functional in the cell probabilities.  The gradient is
obtained by exact forward-mode (dual-number) differentiation over rational numbers.  This
is a different route from both the stratified block algebra written in the fixture README
and the product's inversion of the expected Fisher information in 11 parameters.

    uv run --frozen --script conformance/transport/nested_fisher/reference.py --check
    uv run --frozen --script conformance/transport/nested_fisher/reference.py --self-test

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
ATOL = 1e-12


class Dual:
    """Value with an exact gradient over the cell probabilities."""

    def __init__(self, value, grad):
        self.v, self.g = F(value), list(grad)

    def _lift(self, other):
        return other if isinstance(other, Dual) else Dual(other, [F(0)] * len(self.g))

    def __add__(self, other):
        o = self._lift(other)
        return Dual(self.v + o.v, [a + b for a, b in zip(self.g, o.g)])

    __radd__ = __add__

    def __sub__(self, other):
        o = self._lift(other)
        return Dual(self.v - o.v, [a - b for a, b in zip(self.g, o.g)])

    def __mul__(self, other):
        o = self._lift(other)
        return Dual(self.v * o.v, [self.v * b + o.v * a for a, b in zip(self.g, o.g)])

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._lift(other)
        return Dual(
            self.v / o.v, [(a * o.v - self.v * b) / (o.v * o.v) for a, b in zip(self.g, o.g)]
        )


def law() -> dict:
    """Probability of each (x2, m, y) cell from the generating equations."""
    cell = {}
    for x2, m, y in itertools.product((0, 1), repeat=3):
        p_m0 = F(3, 4) if x2 == 0 else F(1, 4)
        p_y0 = F(4, 5) if m == 0 else F(2, 5)
        cell[(x2, m, y)] = F(1, 2) * (p_m0 if m == 0 else 1 - p_m0) * (p_y0 if y == 0 else 1 - p_y0)
    assert sum(cell.values()) == 1
    return cell


def covariance(n: int) -> tuple[list[list[F]], bool]:
    cell = law()
    keys = sorted(cell)
    # X1 splits each (x2,m,y) cell in two equal halves; counts per 16-cell must be integers.
    integral = all((n * cell[k] / 2).denominator == 1 for k in keys)
    p = [cell[k] for k in keys]
    duals = {
        k: Dual(cell[k], [F(1) if i == j else F(0) for j in range(len(keys))])
        for i, k in enumerate(keys)
    }

    def mass(x2=None, m=None, y=None):
        total = Dual(0, [F(0)] * len(keys))
        for (a, b, c), d in duals.items():
            if (x2 is None or a == x2) and (m is None or b == m) and (y is None or c == y):
                total = total + d
        return total

    def mean(x):
        total = Dual(0, [F(0)] * len(keys))
        for m in (0, 1):
            p_m_given_x = mass(x2=x, m=m) / mass(x2=x)
            inner = Dual(0, [F(0)] * len(keys))
            for xp in (0, 1):
                inner = inner + mass(x2=xp) * (mass(x2=xp, m=m, y=1) / mass(x2=xp, m=m))
            total = total + p_m_given_x * inner
        return total

    m0, m1 = mean(0), mean(1)
    grads = [m0.g, m1.g, [b - a for a, b in zip(m0.g, m1.g)]]
    sigma = [[(p[i] if i == j else F(0)) - p[i] * p[j] for j in range(len(p))] for i in range(len(p))]
    cov = [
        [sum(g[i] * sigma[i][j] * h[j] for i in range(len(p)) for j in range(len(p))) for h in grads]
        for g in grads
    ]
    assert (m0.v, m1.v) == (F(3, 10), F(1, 2))
    return cov, integral


def check(document: dict) -> list[str]:
    n = document["sample_size"]
    cov, integral = covariance(n)
    problems = []
    if not integral:
        problems.append("expected cell counts are not integers at this sample size")
    if abs(document["contrast"] - 0.2) > ATOL:
        problems.append("contrast differs from mean1 - mean0 = 1/5")
    for i in range(3):
        for j in range(3):
            if abs(float(cov[i][j]) - document["covariance_times_n"][i][j]) > ATOL:
                problems.append(f"covariance_times_n[{i}][{j}] {document['covariance_times_n'][i][j]} vs {float(cov[i][j])}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["covariance_times_n"][0][1] += 1e-6
    if not check(wrong):
        failures.append("shifted covariance entry accepted")
    wrong = copy.deepcopy(document)
    wrong["covariance_times_n"][2][2] = wrong["covariance_times_n"][0][0] + wrong["covariance_times_n"][1][1]
    if not check(wrong):
        failures.append("covariance ignoring the shared block accepted")
    wrong = copy.deepcopy(document)
    wrong["sample_size"] = 16001
    if not check(wrong):
        failures.append("non-integral cell counts accepted")
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
