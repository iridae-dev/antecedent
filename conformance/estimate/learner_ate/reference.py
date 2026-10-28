# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the learner-ATE fixture: with Y = 2T + Z + 0.5 eps the interventional contrast is the coefficient 2, while the observed treated-minus-control contrast is 2 + E[Z|T=1] - E[Z|T=0] under T ~ Bernoulli(expit(-0.5 + Z)), evaluated by quadrature over the standard normal and required to lie outside the point tolerance.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/learner_ate/reference.py --check
    uv run --frozen --script conformance/estimate/learner_ate/reference.py --self-test

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
F = Fraction
FIXTURE = HERE / "fixture.json"


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got) if got is not None else got!r} != fixture {want!r} (tol {tol})")


def same(got, want, label, problems):
    if got != want:
        problems.append(f"{label}: derived {got!r} != fixture {want!r}")

def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def expit(x):
    return 1.0 / (1.0 + math.exp(-x))


def simpson(f, a=-12.0, b=12.0, n=20000):
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += f(a + i * h) * (4 if i % 2 else 2)
    return total * h / 3


def check(document):
    problems = []
    p1 = simpson(lambda z: phi(z) * expit(-0.5 + z))
    ez1 = simpson(lambda z: z * phi(z) * expit(-0.5 + z)) / p1
    ez0 = simpson(lambda z: z * phi(z) * (1 - expit(-0.5 + z))) / (1 - p1)
    observed = 2.0 + ez1 - ez0                       # confounded treated-minus-control contrast
    close(2.0, document["ate"], 1e-12, "ate (structural coefficient of T)", problems)
    if not abs(observed - document["ate"]) > document["point_conformance_absolute_tolerance"]:
        problems.append(f"the confounded contrast {observed} is within the point tolerance of the ATE")
    lo, hi = document["coverage_acceptance"]
    if not lo < document["nominal_coverage"] < hi:
        problems.append("nominal coverage is outside the acceptance interval")
    if not 0 < document["absolute_bias_limit"] < document["point_conformance_absolute_tolerance"]:
        problems.append("bias limit is not tighter than the point tolerance")
    return problems


MUTATIONS = [
    ("ate changed", lambda d: d.__setitem__("ate", 1.5)),
    ("point tolerance loosened past the confounded contrast", lambda d: d.__setitem__("point_conformance_absolute_tolerance", 5.0)),
    ("nominal coverage moved outside the acceptance interval", lambda d: d.__setitem__("nominal_coverage", 0.5)),
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
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
