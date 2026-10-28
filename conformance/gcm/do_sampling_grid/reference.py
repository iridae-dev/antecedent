# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the linear-Gaussian do-sampling target: with T ~ N(0,1) and Y = 1 + 2 T + 0.5 eps, do(T=1) gives Y ~ N(3, 0.25).

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/gcm/do_sampling_grid/reference.py --check
    uv run --frozen --script conformance/gcm/do_sampling_grid/reference.py --self-test

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
    model = document["model"]
    intercept, slope, noise_sd, level = 1.0, 2.0, 0.5, 1.0   # Y = 1 + 2 T + 0.5 eps, do(T = 1)
    close(intercept + slope * level, model["outcome_mean"], 1e-12, "outcome_mean", problems)
    close(noise_sd**2, model["outcome_variance"], 1e-12, "outcome_variance", problems)
    for kind, bound in document["acceptance"]["mean_atol"].items():
        if not 0 < bound < 0.2:
            problems.append(f"mean_atol {kind} is not a tight positive band")
    return problems


MUTATIONS = [
    ("mean shifted", lambda d: d["model"].__setitem__("outcome_mean", 3.5)),
    ("variance replaced by the unintervened marginal", lambda d: d["model"].__setitem__("outcome_variance", 4.25)),
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
