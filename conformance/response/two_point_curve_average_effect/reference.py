# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the two-point curve fixture: under E[Y | T=t, Z=z] = 1 + 2 t + 0.8 z the contrast between do(T=0.5) and do(T=-0.5) is 2 exactly (E[Z] cancels); the stated absolute responses 0 and 2 hold only up to the offset 0.8 E[Z] of the fixed z sequence.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/two_point_curve_average_effect/reference.py --check
    uv run --frozen --script conformance/response/two_point_curve_average_effect/reference.py --self-test

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
    n = document["generation"]["n"]
    contract = document["contract"]
    control, active = contract["control_level"], contract["active_level"]
    mean_z = math.fsum(math.sin(i / 17.0) for i in range(n)) / n
    mean = lambda t: 1.0 + 2.0 * t + 0.8 * mean_z   # E[Y | do(T=t)] under E[Y|T,Z] = 1 + 2 t + 0.8 z
    contrast = mean(active) - mean(control)
    close(contrast, contract["true_contrast"], 1e-12, "true_contrast", problems)
    # The stated absolute responses are consistent with the contrast, but equal the exact g-computation
    # values only up to 0.8 E[Z] over the fixed sequence; record the offset instead of asserting it away.
    close(contract["true_active_response"] - contract["true_control_response"], contrast, 1e-12,
          "stated responses reproduce the contrast", problems)
    offset = 0.8 * mean_z
    if not 0.0 < offset < 0.1:
        problems.append(f"unexpected offset of the stated responses from exact g-computation: {offset}")
    return problems


MUTATIONS = [
    ("true contrast shifted", lambda d: d["contract"].__setitem__("true_contrast", 2.1)),
    ("active response shifted", lambda d: d["contract"].__setitem__("true_active_response", 2.5)),
    ("active level changed", lambda d: d["contract"].__setitem__("active_level", 1.0)),
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
