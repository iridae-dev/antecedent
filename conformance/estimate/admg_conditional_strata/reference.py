# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the modifier-averaged effect behind the admg_conditional_strata fixture.

The table counts eight (z, x, y) cells. With the modifier z blocking every back-door path of
x -> y, the modifier-conditioned contrast is the stratified difference
sum_z P(z) (E[y | x = 1, z] - E[y | x = 0, z]), evaluated here by counting in exact rationals.
Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/admg_conditional_strata/reference.py --check
    uv run --frozen --script conformance/estimate/admg_conditional_strata/reference.py --self-test

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


def stratified_difference(cells):
    total = sum(cell["count"] for cell in cells)
    value = Fraction(0)
    strata = {}
    for z in (0, 1):
        in_stratum = [cell for cell in cells if cell["z"] == z]
        weight = Fraction(sum(cell["count"] for cell in in_stratum), total)

        def mean(x):
            arm = sum(cell["count"] for cell in in_stratum if cell["x"] == x)
            ones = sum(cell["count"] for cell in in_stratum if cell["x"] == x and cell["y"] == 1)
            return Fraction(ones, arm)

        effect = mean(1) - mean(0)
        strata[z] = (weight, effect)
        value += weight * effect
    return value, strata


def check(document):
    problems = []
    value, strata = stratified_difference(document["table"])
    if not math.isclose(float(value), document["expected_ate"], rel_tol=0.0, abs_tol=1e-15):
        problems.append(f"expected_ate: derived {float(value)!r} != fixture {document['expected_ate']!r}")
    if str(value) != document["expected_ate_exact"]:
        problems.append(f"expected_ate_exact: derived {value} != fixture {document['expected_ate_exact']}")
    effects = {str(z): float(effect) for z, (_, effect) in strata.items()}
    for z, effect in effects.items():
        if not math.isclose(effect, document["stratum_effects"][z], rel_tol=0.0, abs_tol=1e-15):
            problems.append(f"stratum_effects[{z}]: derived {effect!r}")
    if len({round(e, 9) for e in effects.values()}) < 2:
        problems.append("the strata must have different effects so a pooled difference is wrong")
    contract = document["refuter_contract"]
    for inference in ("Frequentist", "Bayesian"):
        if set(contract[inference]["cheap"]) - set(contract[inference]["full"]):
            problems.append(f"{inference}: full must run every refuter cheap runs")
    return problems


MUTATIONS = [
    ("effect shifted", lambda d: d.__setitem__("expected_ate", d["expected_ate"] + 0.01)),
    ("a cell count moved", lambda d: d["table"][0].__setitem__("count", d["table"][0]["count"] + 1)),
    ("pooled difference", lambda d: d.__setitem__("expected_ate", 0.2)),
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
