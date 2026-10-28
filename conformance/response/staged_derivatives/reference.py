# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the staged-derivative fixture: for the linear law Y = 5 + 2 A - 0.5 B and V = 1 + 0.25 A + 1.5 B, the point derivative, log-log elasticity, semi-elasticities, average derivative, Jacobian and directional derivative follow exactly from the coefficients (rational arithmetic).

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/staged_derivatives/reference.py --check
    uv run --frozen --script conformance/response/staged_derivatives/reference.py --self-test

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

A_AT, B_AT = Fraction(2), Fraction(0)   # evaluation point: A = 2 (the pinned `at`), B = 0 (the Jacobian point)
Y0, YA, YB = Fraction(5), Fraction(2), Fraction(-1, 2)    # Y = 5 + 2 A - 0.5 B
V0, VA, VB = Fraction(1), Fraction(1, 4), Fraction(3, 2)  # V = 1 + 0.25 A + 1.5 B


def check(document):
    problems = []
    tol = 1e-12
    y_at = Y0 + YA * A_AT + YB * B_AT
    derived = {
        "point": YA,
        "elasticity": YA * A_AT / y_at,         # d ln Y / d ln A
        "semi_treatment": YA * A_AT,            # dY / d ln A
        "semi_outcome": YA / y_at,              # d ln Y / dA
        "average": YA,                          # constant slope, so any weighting gives YA
    }
    for key, value in derived.items():
        close(float(value), document[key], tol, key, problems)
    jacobian = [YA, YB, VA, VB]
    for i, (got, want) in enumerate(zip(jacobian, document["jacobian"])):
        close(float(got), want, tol, f"jacobian[{i}]", problems)
    direction = (Fraction(1), Fraction(2))
    directional = [YA * direction[0] + YB * direction[1], VA * direction[0] + VB * direction[1]]
    for i, (got, want) in enumerate(zip(directional, document["directional"])):
        close(float(got), want, tol, f"directional[{i}]", problems)
    if document["at"] != float(A_AT) or document["dgp"].replace(" ", "") != "A=2+sin(i*.71)+.2*cos(i*.13);B=cos(i*1.13);Y=5+2*A-.5*B;V=1+.25*A+1.5*B":
        problems.append("declared point or data-generating law differs from the model this reference derives from")
    return problems


MUTATIONS = [
    ("elasticity shifted", lambda d: d.__setitem__("elasticity", 0.5)),
    ("jacobian entry shifted", lambda d: d["jacobian"].__setitem__(1, -0.4)),
    ("directional shifted", lambda d: d["directional"].__setitem__(1, 3.0)),
    ("average replaced by the elasticity", lambda d: d.__setitem__("average", 0.4444444444444444)),
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
