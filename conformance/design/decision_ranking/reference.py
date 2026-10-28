# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact enumeration reference for the chance-constrained Bayes decision fixture.

For each case the action/outcome utility matrix u[a][o] = action[a] * outcome[o] is enumerated in
exact rational arithmetic. An action is admissible when its constraint satisfaction plus 1e-12 is at
least the chance threshold; the Bayes action maximises the outcome-mean utility over admissible
actions (lowest index on ties); the regret is the outcome mean of (best admissible utility minus the
Bayes action's utility), the expected value of perfect information over admissible actions. No
admissible action is a refusal.

    uv run --frozen --script conformance/design/decision_ranking/reference.py --check
    uv run --frozen --script conformance/design/decision_ranking/reference.py --self-test

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
SLACK = Fraction(1, 10**12)


def solve(case: dict) -> dict:
    actions = [Fraction(str(a)) for a in case["actions"]]
    outcomes = [Fraction(str(o)) for o in case["outcomes"]]
    sat = [Fraction(str(s)) for s in case["satisfaction"]]
    threshold = Fraction(str(case["threshold"]))
    admissible = [a for a in range(len(actions)) if sat[a] + SLACK >= threshold]
    if not admissible:
        return {"refusal": "no_admissible_action"}
    u = {a: [actions[a] * o for o in outcomes] for a in admissible}
    mean = {a: sum(u[a], Fraction(0)) / len(outcomes) for a in admissible}
    best = max(mean.values())
    chosen = min(a for a in admissible if mean[a] == best)
    regret = sum(
        (max(u[a][i] for a in admissible) - u[chosen][i] for i in range(len(outcomes))),
        Fraction(0),
    ) / len(outcomes)
    return {
        "chosen_action": chosen,
        "expected_utility": float(mean[chosen]),
        "regret": float(regret),
    }


def check(document: dict) -> list[str]:
    problems = []
    for case in document["cases"]:
        got = solve(case)
        if "refusal" in case or "refusal" in got:
            if case.get("refusal") != got.get("refusal"):
                problems.append(f"{case['name']}: refusal {case.get('refusal')} vs {got}")
            continue
        if got["chosen_action"] != case["chosen_action"]:
            problems.append(f"{case['name']}: chosen {case['chosen_action']} vs {got['chosen_action']}")
        for key in ("expected_utility", "regret"):
            if abs(got[key] - case[key]) > ATOL:
                problems.append(f"{case['name']}: {key} {case[key]} vs {got[key]}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    for index, case in enumerate(document["cases"]):
        for key, bump in (("expected_utility", 1e-6), ("regret", 1e-6)):
            if key in case:
                wrong = copy.deepcopy(document)
                wrong["cases"][index][key] += bump
                if not check(wrong):
                    failures.append(f"shifted {case['name']} {key} accepted")
        if "chosen_action" in case:
            wrong = copy.deepcopy(document)
            wrong["cases"][index]["chosen_action"] = 1 - case["chosen_action"]
            if not check(wrong):
                failures.append(f"changed {case['name']} action accepted")
    wrong = copy.deepcopy(document)
    wrong["cases"][0]["satisfaction"] = [0.0, 0.0]
    if not check(wrong):
        failures.append("infeasible variant accepted")
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
