# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the prediction fixture: the noise-free recursion X_t = 0.5 X_{t-1} + 0.1, Y_t = 0.01 + 2 X_{t-1} is regressed exactly (rational least squares) on its lag-1 predictor, and the fitted line evaluated at X_{t-1} = 1 is 2.01.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/context/prediction_smoke/reference.py --check
    uv run --frozen --script conformance/context/prediction_smoke/reference.py --self-test

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

def ols_fit(points):
    n = Fraction(len(points))
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - slope * sx) / n, slope


def check(document):
    problems = []
    rows = 80
    half = Fraction(1, 2)
    tenth = Fraction(1, 10)
    hundredth = Fraction(1, 100)
    x = [Fraction(0)]
    y = [Fraction(0)]
    for t in range(1, rows):
        x.append(half * x[t - 1] + tenth)          # X_t = 0.5 X_{t-1} + 0.1, X_0 = 0
        y.append(2 * x[t - 1] + hundredth)         # Y_t = 0.01 + 2 X_{t-1}
    intercept, slope = ols_fit([(x[t - 1], y[t]) for t in range(1, rows)])   # lag-1 least squares
    prediction = float(intercept + slope * 1)
    close(prediction, document["mean_prediction_target"], 1e-12, "prediction at X_{t-1} = 1", problems)
    close(float(slope), 2.0, 1e-12, "fitted slope", problems)
    if not 0 < document["tol"] <= 1e-6:
        problems.append("tolerance is not a round-off scale")
    return problems


MUTATIONS = [
    ("target shifted", lambda d: d.__setitem__("mean_prediction_target", 2.02)),
    ("target replaced by the unconditional mean", lambda d: d.__setitem__("mean_prediction_target", 0.2)),
    ("tolerance loosened to a sampling scale", lambda d: d.__setitem__("tol", 0.1)),
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
