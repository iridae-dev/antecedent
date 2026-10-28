# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the intervention-response fixture: for the zero-noise structural mean 1 + 2 t + 0.8 z, back-door adjustment on Z gives E[Y | do(T=t)] = 1 + 2 t + 0.8 mean(z) over the fixed z_i = sin(i/17) sequence.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/intervention_response/reference.py --check
    uv run --frozen --script conformance/response/intervention_response/reference.py --self-test

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
    dose = document["contract"]["intervention"]["value"]
    # Zero-noise structural mean E[Y | T=t, Z=z] = 1 + 2 t + 0.8 z, back-door adjusted on Z:
    # E[Y | do(T=t)] = 1 + 2 t + 0.8 E[Z], with E[Z] the mean of the fixed z_i = sin(i/17) sequence.
    mean_z = math.fsum(math.sin(i / 17.0) for i in range(n)) / n
    truth = 1.0 + 2.0 * dose + 0.8 * mean_z
    close(truth, document["contract"]["true_response"], 1e-12, "true_response", problems)
    if document["tolerance"]["truth_absolute"] <= 0:
        problems.append("tolerance must be positive")
    if document["contract"]["structural_mean"] != "E[Y|T=t,Z=z] = 1 + 2*t + 0.8*z":
        problems.append("structural mean text differs from the model this reference derives from")
    return problems


MUTATIONS = [
    ("true response shifted", lambda d: d["contract"].__setitem__("true_response", 1.6)),
    ("dose changed", lambda d: d["contract"]["intervention"].__setitem__("value", 0.5)),
    ("row count changed", lambda d: d["generation"].__setitem__("n", 200)),
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
