# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the instrumental-variable fixture: with T = a Z + U + noise and Y = b T + U + noise, Cov(Z,Y)/Cov(Z,T) = b exactly, whereas the confounded OLS slope is b + Var(U)/Var(T).

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/iv_2sls/reference.py --check
    uv run --frozen --script conformance/estimate/iv_2sls/reference.py --self-test

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

# Test SCM: Z alternates 0,1; U ~ N(0,1) unobserved; T = 0.6 Z + U + 0.1 e1; Y = 2 T + U + 0.1 e2.
A_ZT = Fraction(6, 10)
B_TY = Fraction(2)
VAR_U = Fraction(1)
VAR_Z = Fraction(1, 4)       # variance of a balanced 0/1 instrument
VAR_E1 = Fraction(1, 100)


def derived():
    cov_zt = A_ZT * VAR_Z
    cov_zy = B_TY * cov_zt   # Cov(Z, U) = 0 and Cov(Z, e2) = 0 (exclusion and as-if-random)
    wald = cov_zy / cov_zt
    var_t = A_ZT**2 * VAR_Z + VAR_U + VAR_E1
    cov_ty = B_TY * var_t + VAR_U
    ols = cov_ty / var_t
    return float(wald), float(ols)


def check(document):
    problems = []
    wald, ols = derived()
    close(wald, document["true_effect"], 1e-12, "true_effect (Cov(Z,Y)/Cov(Z,T))", problems)
    if not abs(ols - document["true_effect"]) > document["tolerance"]:
        problems.append(f"tolerance {document['tolerance']} does not exclude the confounded OLS slope {ols:.4f}")
    return problems


MUTATIONS = [
    ("true effect shifted by 0.5", lambda d: d.__setitem__("true_effect", 2.5)),
    ("true effect replaced by the confounded OLS slope", lambda d: d.__setitem__("true_effect", 2.909)),
    ("tolerance widened past the OLS bias", lambda d: d.__setitem__("tolerance", 1.0)),
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
