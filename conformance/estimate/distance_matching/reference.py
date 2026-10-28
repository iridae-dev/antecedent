# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the structural effect behind the distance_matching fixture: with Z ~ N(0,1), T ~ Bernoulli(sigmoid(-0.4 + 0.9 Z)) and Y = 2 T + Z + noise, do(T=t) leaves E[Y] = 2 t + E[Z], so the ATT (equal to the ATE here because the effect is constant) is the structural coefficient 2, and the unadjusted treated-minus-control contrast is 2 + E[Z|T=1] - E[Z|T=0] (numerical quadrature over the standard normal).

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/distance_matching/reference.py --check
    uv run --frozen --script conformance/estimate/distance_matching/reference.py --self-test

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

# Structural model declared by the executing test's SCM and by the fixture notes:
#   Z ~ N(0,1); T ~ Bernoulli(sigmoid(-0.4 + 0.9 Z)); Y = 2 T + Z + 0.4 eps, eps ~ N(0,1).
BETA_T = Fraction(2)   # coefficient of T in the Y equation
BETA_Z = Fraction(1)   # coefficient of Z in the Y equation
GAMMA0, GAMMA1 = -0.4, 0.9


def derived():
    mean_z = simpson(lambda z: z * phi(z))  # E[Z] = 0 under the standard normal
    # do(T=t) cuts Z -> T and leaves the Y equation: E[Y | do(T=t)] = BETA_T t + BETA_Z E[Z].
    do = lambda t: float(BETA_T) * t + float(BETA_Z) * mean_z
    ate = do(1.0) - do(0.0)
    # The effect is the same constant for every unit, so the effect on the treated equals the ATE.
    treat = lambda z: phi(z) * sigmoid(GAMMA0 + GAMMA1 * z)
    p1 = simpson(treat)
    mean_z_treated = simpson(lambda z: z * treat(z)) / p1
    mean_z_control = simpson(lambda z: z * phi(z) * (1 - sigmoid(GAMMA0 + GAMMA1 * z))) / (1 - p1)
    # Unadjusted treated-minus-control contrast = beta_T + beta_Z (E[Z|T=1] - E[Z|T=0]).
    naive = float(BETA_T) + float(BETA_Z) * (mean_z_treated - mean_z_control)
    return ate, naive


def check(document):
    problems = []
    ate, naive = derived()
    close(ate, document["true_effect"], 1e-9, "true_effect", problems)
    if not abs(naive - document["true_effect"]) > document["tolerance"]:
        problems.append(
            f"tolerance {document['tolerance']} does not exclude the confounded contrast {naive:.4f}"
        )
    if not 0 < document["tolerance"] < abs(naive - ate):
        problems.append("tolerance is not a positive band narrower than the confounding bias")
    return problems


MUTATIONS = [
    ("true effect shifted by 0.5", lambda d: d.__setitem__("true_effect", d["true_effect"] + 0.5)),
    ("true effect replaced by the confounded contrast", lambda d: d.__setitem__("true_effect", 2.77)),
    ("tolerance widened past the confounding bias", lambda d: d.__setitem__("tolerance", 0.9)),
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
