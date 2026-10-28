# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the distribution-change fixture: with the X mechanism unchanged
and Y = a + b X, the hybrid payoff E[Y] moves only when the Y intercept is swapped, so the
exact two-player Shapley value (average over both player orders) gives X nothing and Y the
full intercept change.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/attribution/distribution_change_grid/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def close(got, want, tol, label, problems):
    if not math.isclose(got, want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


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
    tol = document["comparison"]["deterministic_absolute_tolerance"]
    mean_x = 0.7  # any common mean of X: it cancels because the X mechanism is unchanged
    for case in document["cases"]:
        def payoff(coalition, case=case):
            intercept = case["comparison_intercept"] if "y" in coalition else case["baseline_intercept"]
            return intercept + case["slope"] * mean_x
        phi = shapley(["x", "y"], payoff)
        total = payoff(frozenset({"x", "y"})) - payoff(frozenset())
        label = case["id"]
        close(total, case["total_change"], tol, f"{label}.total_change", problems)
        for name, want in case["component_contributions"].items():
            close(phi[name], want, tol, f"{label}.{name}", problems)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
