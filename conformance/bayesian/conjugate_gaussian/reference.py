# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the conjugate-Gaussian fixture: with a diffuse isotropic Gaussian prior, known residual variance 1e-6 and 40 noiseless rows y = 1 + 2 x, the posterior mean (equal to the MAP) is the ridge solution (X'X + lam I)^-1 X'y with lam = sigma^2 / prior variance, which differs from (1, 2) by far less than the tolerance; solved in exact rational arithmetic.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/conjugate_gaussian/reference.py --check
    uv run --frozen --script conformance/bayesian/conjugate_gaussian/reference.py --self-test

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
    if got is None or want is None or not math.isclose(got, want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


def simpson(f, a=-10.0, b=10.0, n=4000):
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += f(a + i * h) * (4 if i % 2 else 2)
    return total * h / 3


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def solve_ridge(n_rows, lam):
    """Posterior mean of a Gaussian linear model with known variance: (X'X + lam I)^-1 X'y, lam = sigma^2 / prior variance."""
    xs = [Fraction(r) for r in range(n_rows)]
    ys = [1 + 2 * x for x in xs]               # noiseless y = 1 + 2 x
    a = Fraction(n_rows)
    b = sum(xs)
    d = sum(x * x for x in xs)
    r0 = sum(ys)
    r1 = sum(x * y for x, y in zip(xs, ys))
    a, d = a + lam, d + lam
    det = a * d - b * b
    return (d * r0 - b * r1) / det, (a * r1 - b * r0) / det


def check(document):
    problems = []
    sigma2 = Fraction(1, 10**6)                # known residual variance 1e-6 declared by the executing test
    truth = document["true_coefficients"]
    tol = document["tolerance"]
    # The prior scale (isotropic, 100 as a variance or as a standard deviation) is not declared in the fixture:
    # derive the posterior mean for both readings and require both to recover the coefficients.
    for label, prior_variance in (("variance 100", Fraction(100)), ("sd 100", Fraction(10000))):
        b0, b1 = solve_ridge(40, sigma2 / prior_variance)
        close(float(b0), truth[0], tol, f"posterior mean intercept ({label})", problems)
        close(float(b1), truth[1], tol, f"posterior mean slope ({label})", problems)
        shrink = max(abs(float(b0) - truth[0]), abs(float(b1) - truth[1]))
        if not shrink < tol / 100:
            problems.append(f"shrinkage {shrink} is not negligible against the tolerance ({label})")
    if document["backend"] != "conjugate_gaussian":
        problems.append("backend label differs")
    return problems


MUTATIONS = [
    ("slope shifted", lambda d: d["true_coefficients"].__setitem__(1, 2.1)),
    ("intercept shifted", lambda d: d["true_coefficients"].__setitem__(0, 1.5)),
    ("tolerance tightened below the ridge shrinkage floor", lambda d: d.__setitem__("tolerance", 1e-13)),
]


def self_test(document):
    failures = []
    if check(document):
        failures.append("the unmodified fixture is rejected")
    for label, mutate in MUTATIONS:
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"accepted: {{label}}")
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
