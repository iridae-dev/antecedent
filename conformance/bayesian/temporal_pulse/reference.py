# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the pulse effect behind the temporal_pulse fixture: on the test's noiseless series (pressure_t = sin(0.04 t), defect_t = 0.9 pressure_{t-1}, n = 400) the exact least-squares slope of defect on lagged pressure over rationals is 0.9, the effect of a unit pulse one step earlier.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/temporal_pulse/reference.py --check
    uv run --frozen --script conformance/bayesian/temporal_pulse/reference.py --self-test

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


def solve(a, b):
    n = len(a)
    m = [row[:] + [bi] for row, bi in zip(a, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        m[c] = [v / m[c][c] for v in m[c]]
        for r in range(n):
            if r != c:
                m[r] = [x - m[r][c] * y for x, y in zip(m[r], m[c])]
    return [row[n] for row in m]

def least_squares(columns, y):
    a = [[sum(p * q for p, q in zip(ci, cj)) for cj in columns] for ci in columns]
    b = [sum(p * q for p, q in zip(ci, y)) for ci in columns]
    return solve(a, b)

import math

def derived(n, effect):
    pressure = [Fraction(math.sin(0.04 * t)) for t in range(n)]
    lagged = pressure[: n - 1]
    defect = [effect * p for p in lagged]
    intercept, slope = least_squares([[Fraction(1)] * (n - 1), lagged], defect)
    return slope, intercept

def check(document):
    problems = []
    slope, intercept = derived(document["n"], Fraction(9, 10))
    close(slope, document["expected_ate"], 1e-9, "expected_ate", problems)
    close(slope, document["true_effect_per_unit"], 1e-9, "true_effect_per_unit", problems)
    close(intercept, 0.0, 1e-9, "intercept", problems)
    if not 0 < document["tolerance"] < document["expected_ate"]:
        problems.append("tolerance is not a positive band narrower than the effect")
    if document["treatment_lag"] != 1 or document["horizon_steps"] != 1:
        problems.append("lag or horizon differs from the single-step pulse this reference derived")
    return problems

MUTATIONS = [
    ("expected effect shifted", lambda d: d.__setitem__("expected_ate", 1.0)),
    ("per-unit effect shifted", lambda d: d.__setitem__("true_effect_per_unit", 0.8)),
    ("lag changed", lambda d: d.__setitem__("treatment_lag", 2)),
    ("tolerance widened past the effect", lambda d: d.__setitem__("tolerance", 1.0)),
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
