# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the structure-change efficiency proposition: with one structurally changed outcome node the exact Shapley value of that node equals the full hybrid-payoff difference, so the efficiency residual is zero.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/attribution/structure_change_grid/reference.py --check
    uv run --frozen --script conformance/attribution/structure_change_grid/reference.py --self-test

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
    tol = document["comparison"]["absolute_tolerance"]
    swap = next(case for case in document["cases"] if case["id"] == "single_parent_swap")
    # With a single changed player the exact Shapley value is v({y}) - v(empty set), which equals the full hybrid
    # payoff difference v(all) - v(empty set) because the coalition {y} is the grand coalition; try several payoffs.
    for baseline, comparison in ((0.0, 3.5), (1.25, -2.0), (-4.0, -4.0)):
        payoff = lambda coalition, b=baseline, c=comparison: c if "y" in coalition else b
        phi_y = shapley(swap["changed_players"], payoff)["y"]
        total = payoff(frozenset(swap["changed_players"])) - payoff(frozenset())
        close(phi_y - total, swap["efficiency_residual"], tol, f"efficiency residual (payoff {baseline}, {comparison})", problems)
    if len(swap["changed_players"]) != 1:
        problems.append("the efficiency derivation here covers a single changed player only")
    return problems


MUTATIONS = [
    ("efficiency residual non-zero", lambda d: d["cases"][0].__setitem__("efficiency_residual", 0.01)),
    ("two changed players", lambda d: d["cases"][0].__setitem__("changed_players", ["y", "x"])),
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
