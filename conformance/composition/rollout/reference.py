# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact enumeration reference for the finite rollout/study-update fixture.

Equal-mass states theta in {1/4, 3/4}; waiting pays 0 and betting pays theta - 1/2.
Everything is evaluated in exact rational arithmetic by enumerating the states and the
Bernoulli outcome sequences, with no Antecedent code:

    prior expected utility  = max over actions of E[payoff]
    EVPI                    = E[max over actions of payoff(theta)] - prior expected utility
    EVSI(n)                 = sum over outcome counts k of P(k) max over actions of
                              E[payoff | k]  - prior expected utility

    uv run --frozen --script conformance/composition/rollout/reference.py --check
    uv run --frozen --script conformance/composition/rollout/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATOL = 1e-12


def compute(document: dict) -> dict:
    states = [Fraction(str(s)) for s in document["state_draws"]]
    mass = Fraction(1, len(states))
    payoffs = {
        "wait": [Fraction(0) for _ in states],
        "bet": [s - Fraction(1, 2) for s in states],
    }
    prior = max(sum(mass * p for p in row) for row in payoffs.values())
    evpi = sum(mass * max(row[i] for row in payoffs.values()) for i in range(len(states))) - prior
    n = document["sample_size"]
    evsi = Fraction(0)
    counts: dict[int, Fraction] = {}
    for outcome in itertools.product((0, 1), repeat=n):
        k = sum(outcome)
        likelihood = [s**k * (1 - s) ** (n - k) for s in states]
        counts[k] = counts.get(k, Fraction(0)) + sum(mass * l for l in likelihood)
        joint = [mass * l for l in likelihood]
        evsi += max(sum(j * row[i] for i, j in enumerate(joint)) for row in payoffs.values())
    evsi -= prior
    table = [[float(p) for p in payoffs[a]] for a in document["terminal_actions"]]
    return {
        "prior_expected_utility": float(prior),
        "evpi": float(evpi),
        "evsi": float(evsi),
        "utilities": table,
        "predictive_counts": {k: str(v) for k, v in sorted(counts.items())},
    }


def check(document: dict) -> list[str]:
    problems = []
    got = compute(document)
    for key in ("prior_expected_utility", "evpi", "evsi"):
        if abs(got[key] - document[key]) > ATOL:
            problems.append(f"{key}: fixture {document[key]} reference {got[key]}")
    if got["utilities"] != document["utilities"]:
        problems.append("utility table differs from the enumerated payoffs")
    if document["sample_size"] == 2 and got["predictive_counts"] != {
        0: "5/16",
        1: "3/8",
        2: "5/16",
    }:
        problems.append(f"predictive counts {got['predictive_counts']}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    for key, bump in (("evsi", 1e-6), ("evpi", 1e-6), ("prior_expected_utility", 1e-6)):
        wrong = copy.deepcopy(document)
        wrong[key] += bump
        if not check(wrong):
            failures.append(f"shifted {key} accepted")
    wrong = copy.deepcopy(document)
    wrong["sample_size"] = 3
    if not check(wrong):
        failures.append("changed sample size accepted")
    wrong = copy.deepcopy(document)
    wrong["utilities"][1][0] = 0.0
    if not check(wrong):
        failures.append("changed utility table accepted")
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
