# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the ATE behind the bayesian_robust_ate fixture: the outcome equation is y = 1.4 x^2 + 2 t, so y(1) - y(0) = 2 for every unit, whatever the covariate or the (logistic) treatment assignment; evaluated over the test's covariate grid x_i = ((7919 i mod n) / n) * 2 - 1.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/bayesian_robust_ate/reference.py --check
    uv run --frozen --script conformance/estimate/bayesian_robust_ate/reference.py --self-test

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


def grid(n):
    return [Fraction(i * 7919 % n, n) * 2 - 1 for i in range(n)]

def outcome(x, t):
    return Fraction(14, 10) * x * x + 2 * t

def derived(n):
    xs = grid(n)
    effects = {outcome(x, 1) - outcome(x, 0) for x in xs}
    return effects

def check(document):
    problems = []
    n = document["n"]
    effects = derived(n)
    if len(effects) != 1:
        problems.append(f"unit effects are not constant: {sorted(effects)[:3]}")
    else:
        close(next(iter(effects)), document["truth"], 1e-12, "truth", problems)
    if len(set(grid(n))) != n:
        problems.append("covariate grid has repeated points, so the design is not a permutation of the grid")
    if not 0 < document["tolerance"] < document["truth"]:
        problems.append("tolerance is not a positive band narrower than the effect")
    return problems

MUTATIONS = [
    ("truth shifted by 0.5", lambda d: d.__setitem__("truth", d["truth"] + 0.5)),
    ("tolerance widened past the effect", lambda d: d.__setitem__("tolerance", 2.5)),
    ("row count changed to a divisor that repeats grid points", lambda d: d.__setitem__("n", 7919)),
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
