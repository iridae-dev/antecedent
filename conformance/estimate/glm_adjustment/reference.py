# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the logistic g-computation fixture: with Z ~ N(0,1), T ~ Bernoulli(sigmoid(-0.3 + 0.8 Z)) and Y ~ Bernoulli(sigmoid(-0.5 + 1.2 T + 0.8 Z)), the standardised ATE is E_Z[sigmoid(0.7 + 0.8 Z) - sigmoid(-0.5 + 0.8 Z)] (quadrature over the standard normal) and the unadjusted contrast is the difference of arm means.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/glm_adjustment/reference.py --check
    uv run --frozen --script conformance/estimate/glm_adjustment/reference.py --self-test

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

# Test SCM: Z ~ N(0,1); T ~ Bernoulli(sigmoid(-0.3 + 0.8 Z)); Y ~ Bernoulli(sigmoid(-0.5 + 1.2 T + 0.8 Z)).
def derived():
    pt = lambda z: sigmoid(-0.3 + 0.8 * z)
    py = lambda t, z: sigmoid(-0.5 + 1.2 * t + 0.8 * z)
    # do(T=t): standardise over the marginal Z (back-door adjustment), ATE = E_Z[py(1,Z) - py(0,Z)].
    ate = simpson(lambda z: phi(z) * (py(1, z) - py(0, z)))
    p1 = simpson(lambda z: phi(z) * pt(z))
    mean1 = simpson(lambda z: phi(z) * pt(z) * py(1, z)) / p1
    mean0 = simpson(lambda z: phi(z) * (1 - pt(z)) * py(0, z)) / (1 - p1)
    return ate, mean1 - mean0


def check(document):
    problems = []
    ate, naive = derived()
    tol = document["tolerance"]
    # The fixture states a rounded truth (0.25); the exact standardised ATE must lie within 0.01 of it.
    close(ate, document["true_effect"], 0.01, "true_effect (standardised ATE)", problems)
    if not abs(naive - document["true_effect"]) >= tol:
        problems.append(f"tolerance {tol} does not exclude the unadjusted contrast {naive:.4f}")
    if not abs(ate - document["true_effect"]) < tol:
        problems.append("the exact ATE falls outside the declared band")
    return problems


MUTATIONS = [
    ("true effect shifted by 0.1", lambda d: d.__setitem__("true_effect", 0.35)),
    ("true effect replaced by the unadjusted contrast", lambda d: d.__setitem__("true_effect", 0.3727)),
    ("tolerance widened past the unadjusted contrast", lambda d: d.__setitem__("tolerance", 0.2)),
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
