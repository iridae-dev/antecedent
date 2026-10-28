# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the structural ATE behind the shared_functional_ate fixture: on the test's 80-row design (z_i = i/10, t_i = 1 for even i) with Y = 2T + 0.5Z, the exact least-squares coefficient of T in the regression of Y on (1, T, Z) is 2, solved by Gaussian elimination over rationals.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/shared_functional_ate/reference.py --check
    uv run --frozen --script conformance/bayesian/shared_functional_ate/reference.py --self-test

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


N = 80
BETA_T, BETA_Z = Fraction(2), Fraction(1, 2)

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

def design():
    z = [Fraction(i, 10) for i in range(N)]
    t = [Fraction(1 if i % 2 == 0 else 0) for i in range(N)]
    y = [BETA_T * ti + BETA_Z * zi for ti, zi in zip(t, z)]
    return z, t, y

def derived():
    z, t, y = design()
    x = [[Fraction(1), ti, zi] for ti, zi in zip(t, z)]
    a = [[sum(r[i] * r[j] for r in x) for j in range(3)] for i in range(3)]
    b = [sum(r[i] * yi for r, yi in zip(x, y)) for i in range(3)]
    adjusted = solve(a, b)[1]
    n1 = sum(t)
    naive = sum(ti * yi for ti, yi in zip(t, y)) / n1 - sum((1 - ti) * yi for ti, yi in zip(t, y)) / (N - n1)
    return adjusted, naive

def check(document):
    problems = []
    adjusted, naive = derived()
    close(adjusted, document["true_ate"], 1e-12, "true_ate", problems)
    if not document["tolerance"] > 0:
        problems.append("tolerance is not positive")
    return problems

MUTATIONS = [
    ("true ATE shifted by 0.5", lambda d: d.__setitem__("true_ate", d["true_ate"] + 0.5)),
    ("true ATE replaced by the unadjusted contrast", lambda d: d.__setitem__("true_ate", 1.95)),
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
