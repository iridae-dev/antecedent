# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the fitted-SCM intervention target: with Y = 1 + 2 X, do(X=3) gives E[Y] = 7.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/gcm/gcm_fit_intervene/reference.py --check
    uv run --frozen --script conformance/gcm/gcm_fit_intervene/reference.py --self-test

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

def check(document):
    problems = []
    intercept, slope = 1.0, 2.0   # noiseless Y = 1 + 2 X over x_i = 0.1 i, as the test's chain data
    level = document["intervention"]["x"]
    close(intercept + slope * level, document["true_interventional_mean_y"], 1e-12, "true_interventional_mean_y", problems)
    if not 0 < document["tolerance"] < 1.0:
        problems.append("tolerance must be a tight positive band")
    return problems


MUTATIONS = [
    ("true mean shifted", lambda d: d.__setitem__("true_interventional_mean_y", 7.5)),
    ("intervention level changed", lambda d: d["intervention"].__setitem__("x", 4.0)),
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
