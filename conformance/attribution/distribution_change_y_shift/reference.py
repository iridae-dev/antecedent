# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the distribution-change fixture: with the X mechanism unchanged and the Y intercept moving from 1 to 6, the exact two-player Shapley value gives X nothing and Y the full +5 change.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/attribution/distribution_change_y_shift/reference.py --check
    uv run --frozen --script conformance/attribution/distribution_change_y_shift/reference.py --self-test

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

import itertools


def shapley(players, payoff):
    value = {p: 0.0 for p in players}
    orders = list(itertools.permutations(players))
    for order in orders:
        coalition = frozenset()
        for p in order:
            value[p] += payoff(coalition | {p}) - payoff(coalition)
            coalition = coalition | {p}
    return {p: v / len(orders) for p, v in value.items()}


def check(document):
    problems = []
    mean_x = 0.0   # any common mean of X cancels because the X mechanism is unchanged
    base, comparison, slope = 1.0, 6.0, 2.0   # Y = 1 + 2X in the baseline period, 6 + 2X in the comparison period

    def payoff(coalition):   # hybrid mean: mechanisms in the coalition come from the comparison period
        intercept = comparison if "y" in coalition else base
        return intercept + slope * mean_x

    phi_values = shapley(["x", "y"], payoff)
    total = payoff(frozenset({"x", "y"})) - payoff(frozenset())
    if not total >= document["total_change_min"]:
        problems.append(f"derived total change {total} is below total_change_min {document['total_change_min']}")
    dominant = max(phi_values, key=lambda p: abs(phi_values[p]))
    if dominant != document["dominant_component"]:
        problems.append(f"derived dominant component {dominant} != fixture {document['dominant_component']}")
    close(phi_values["x"], 0.0, 1e-12, "x contribution", problems)
    close(phi_values["y"], 5.0, 1e-12, "y contribution", problems)
    if total - document["total_change_min"] > 3.0:
        problems.append("total_change_min is more than 3 below the derived change; the bound is loose")
    return problems


MUTATIONS = [
    ("dominant component swapped", lambda d: d.__setitem__("dominant_component", "x")),
    ("lower bound above the derived change", lambda d: d.__setitem__("total_change_min", 6.0)),
    ("lower bound loosened to zero", lambda d: d.__setitem__("total_change_min", 0.0)),
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
