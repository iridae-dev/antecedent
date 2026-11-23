# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the marginal pulse effect behind the temporal_conditional fixture: with Y = 1.5 X + 0.2 W + 0.8 X W the unit slope is 1.5 + 0.8 W; its population value for W ~ Bernoulli(1/2) is 1.9, and averaged over the test's actual modifier column (W_t = 1 for odd t in 1..79, 40 ones in 79 rows) it is 1.5 + 0.8 * 40/79 = 1.9063, within the test's 0.05 band of 1.9.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/temporal_conditional/reference.py --check
    uv run --frozen --script conformance/estimate/temporal_conditional/reference.py --self-test

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
A, C = Fraction(3, 2), Fraction(4, 5)  # Y = A X + 0.2 W + C X W

def derived():
    population = A + C * Fraction(1, 2)
    w = [Fraction(t % 2) for t in range(1, N)]  # rows with a lagged treatment
    sample = sum(A + C * wi for wi in w) / len(w)
    return population, sample

def check(document):
    problems = []
    population, sample = derived()
    close(population, document["marginal_effect"], 1e-12, "marginal_effect", problems)
    if abs(float(sample) - document["marginal_effect"]) >= 0.05:
        problems.append(f"sample-weighted value {float(sample)!r} is outside the test's 0.05 band of the fixture")
    return problems

MUTATIONS = [
    ("effect shifted", lambda d: d.__setitem__("marginal_effect", 2.0)),
    ("effect replaced by the unmoderated slope", lambda d: d.__setitem__("marginal_effect", 1.5)),
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
