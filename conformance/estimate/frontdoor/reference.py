# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the linear front-door fixture: with T -> M -> Y and an unmeasured U -> T, U -> Y, the effect is the product of the stage coefficients 0.4 * 5 = 2, the confounded Y ~ T slope is 2 + Var(U)/Var(T), and the delta-method variance numerator quoted in the README is b^2 Var(a) + a^2 Var(b).

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/frontdoor/reference.py --check
    uv run --frozen --script conformance/estimate/frontdoor/reference.py --self-test

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

# Test SCM: U ~ N(0,1) unobserved; T = 3 + U + 0.1 e1; M = 0.4 T + 0.1 e2; Y = 5 M + U + 0.1 e3.
A_TM = Fraction(4, 10)
B_MY = Fraction(5)
SD = Fraction(1, 10)
README_SE_NUMERATOR = 0.566   # the delta-method numerator quoted in the fixture README (se^2 = 0.566 / n)


def derived():
    var_t = 1 + SD**2                       # Var(T), U has unit variance
    # Front-door: the effect of T on Y through M is the product a*b of the two stage coefficients.
    effect = A_TM * B_MY
    # Confounded Y ~ T slope: Y = a b T + (5 e2 sd) + U + e3 sd, Cov(T, Y) = a b Var(T) + Var(U).
    naive = effect + 1 / var_t
    # Delta-method variance of a_hat * b_hat times n (a_hat: M ~ T; b_hat: Y ~ M + T).
    var_a = SD**2 / var_t
    var_u_given_t = 1 - 1 / var_t
    resid_y = var_u_given_t + SD**2         # Var(Y | M, T)
    var_b = resid_y / SD**2                 # Var(M | T) = sd^2
    se_numerator = B_MY**2 * var_a + A_TM**2 * var_b
    return float(effect), float(naive), float(se_numerator)


def check(document):
    problems = []
    effect, naive, se_numerator = derived()
    close(effect, document["true_effect"], 1e-12, "true_effect (a*b)", problems)
    if not abs(naive - document["true_effect"]) > document["tolerance"]:
        problems.append(f"tolerance {document['tolerance']} does not exclude the confounded slope {naive:.4f}")
    for coefficient in (0.4, 5.0):
        if abs(coefficient - document["true_effect"]) <= document["tolerance"]:
            problems.append(f"tolerance admits the single coefficient {coefficient}")
    if abs(math.log(se_numerator / README_SE_NUMERATOR)) > 0.001:
        problems.append(f"delta-method numerator {se_numerator:.5f} != README {README_SE_NUMERATOR}")
    return problems


MUTATIONS = [
    ("true effect replaced by one coefficient", lambda d: d.__setitem__("true_effect", 5.0)),
    ("true effect replaced by the confounded slope", lambda d: d.__setitem__("true_effect", 2.99)),
    ("tolerance widened past the confounded slope", lambda d: d.__setitem__("tolerance", 1.5)),
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
