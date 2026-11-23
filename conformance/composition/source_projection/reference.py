# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact closed-form reference for the original-source projection fixture.

The native data are Y = 1 + 2A + 0.2X with each of 80 equally spaced X values in [-1, 1]
occurring once at every dose A in {0,1,2,3}, so A is independent of X by construction and
the backdoor standardization over X equals the structural intervention mean

    E[Y | do(A=a)] = 1 + 2a + 0.2 E[X] = 1 + 2a.

The caller-declared affine utilities are U_A = 2 m(1) - 1 and U_B = 0.5 m(2).  The script
evaluates these in exact rational arithmetic from the data-generating equation alone; it
imports no Antecedent code and fits no regression.

    uv run --frozen --script conformance/composition/source_projection/reference.py --check
    uv run --frozen --script conformance/composition/source_projection/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATOL = 1e-12


def compute() -> dict:
    xs = [Fraction(-1) + Fraction(2 * i, 79) for i in range(80)]
    mean_x = sum(xs) / len(xs)
    mean = {a: 1 + 2 * a + Fraction(1, 5) * mean_x for a in (1, 2)}
    return {
        "mean_x": mean_x,
        "native_means": [float(mean[1]), float(mean[2])],
        "utilities": {"A": float(2 * mean[1] - 1), "B": float(Fraction(1, 2) * mean[2])},
    }


def check(document: dict) -> list[str]:
    got = compute()
    problems = []
    if got["mean_x"] != 0:
        problems.append("covariate grid is not symmetric about zero")
    for name in ("native_means", "external_means"):
        for fixture, reference in zip(document[name], got["native_means"], strict=True):
            if abs(fixture - reference) > ATOL:
                problems.append(f"{name}: fixture {fixture} reference {reference}")
        if len(document[name]) != 2:
            problems.append(f"{name} has wrong length")
    for action in ("A", "B"):
        if abs(document["utilities"][action] - got["utilities"][action]) > ATOL:
            problems.append(f"utility {action} differs")
    rule = document["utility_rule"]
    if rule != {"A": "2 * E[Y|do(a=1)] - 1", "B": "0.5 * E[Y|do(a=2)]"}:
        problems.append("declared utility rule differs from the evaluated affine maps")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["native_means"][0] += 1e-6
    if not check(wrong):
        failures.append("shifted native mean accepted")
    wrong = copy.deepcopy(document)
    wrong["utilities"]["B"] = 5.0
    if not check(wrong):
        failures.append("changed utility accepted")
    wrong = copy.deepcopy(document)
    wrong["utility_rule"]["A"] = "2 * E[Y|do(a=2)] - 1"
    if not check(wrong):
        failures.append("changed utility rule accepted")
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
