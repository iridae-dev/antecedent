# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the Bayesian trial-transport fixture: with two equally weighted trial rows the normalized Dirichlet(1,1) row mass U is Uniform(0,1) (Beta(1,1)), and the transported effect draw 8U - 2 has mean 2 and variance 16/3.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/bayesian_trial_transport/reference.py --check
    uv run --frozen --script conformance/estimate/bayesian_trial_transport/reference.py --self-test

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
FIXTURE = HERE / "expected.json"


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got) if got is not None else got!r} != fixture {want!r} (tol {tol})")


def same(got, want, label, problems):
    if got != want:
        problems.append(f"{label}: derived {got!r} != fixture {want!r}")

def check(document):
    problems = []
    eu, eu2 = F(1, 2), F(1, 3)          # moments of Beta(1,1): E U = 1/2, E U^2 = 1/3
    mean = 8 * eu - 2
    var = 64 * (eu2 - eu * eu)
    close(mean, document["expected_mean"], 1e-12, "expected_mean", problems)
    close(var, document["expected_variance"], 1e-12, "expected_variance", problems)
    if document["mean_tolerance"] <= 0 or document["variance_tolerance"] <= 0:
        problems.append("tolerances must be positive")
    # a Monte Carlo band narrower than the sampling noise would be a flaky pin; report the implied relative width
    if document["variance_tolerance"] / float(var) > 0.25:
        problems.append("variance tolerance exceeds a quarter of the derived variance")
    return problems


MUTATIONS = [
    ("mean changed", lambda d: d.__setitem__("expected_mean", 2.5)),
    ("variance changed", lambda d: d.__setitem__("expected_variance", 4.0)),
    ("variance band loosened", lambda d: d.__setitem__("variance_tolerance", 2.0)),
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
