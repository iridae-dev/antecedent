# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact enumeration reference for the whole-history recalculation fixture.

The structural model is binary and two-step: S0, A1, A2 uniform bits, U_L uniform on
0..3, U_Y uniform on 0..19, L2 = 1[U_L < S0 + A1 + 1] and
Y = 1[U_Y < 2 + 4 A1 + 3 L2 + 5 A2 + 2 S0 + 2 S0 A1].  The script

1. enumerates all 2*2*2*4*20 = 640 exogenous configurations and recovers the observational
   law the source histories come from (checking positivity and the stated response formula),
2. evaluates the interventional response E[Y | do(A1=a1, A2=a2)] under a target law
   P(S0=1) = p1 by exact integration of the structural probabilities
   (sum over S0, L2 of P(S0) P(L2 | S0, a1) P(Y=1 | S0, a1, L2, a2)), and
3. derives the additive paired-candidate pins (response 7.165, paired effect 5) from the
   stated additive model with balanced complete histories and the 24-unit disturbance
   0.01 * unit.

No Antecedent code, no estimator and no identification routine is used.  The
`control_first_unit_y_to_zero` pin is a perturbation control of the adapter's input
handling, not a structural quantity, and is not derived here.

    uv run --frozen --script conformance/recalculation/temporal_history/reference.py --check
    uv run --frozen --script conformance/recalculation/temporal_history/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATOL = 1e-12


def configs():
    for s0, a1, a2, ul, uy in itertools.product((0, 1), (0, 1), (0, 1), range(4), range(20)):
        l2 = int(ul < s0 + a1 + 1)
        y = int(uy < 2 + 4 * a1 + 3 * l2 + 5 * a2 + 2 * s0 + 2 * s0 * a1)
        yield s0, a1, l2, a2, y


def response(a1: int, a2: int, p1: F) -> F:
    """E[Y | do(a1, a2)] with S0 ~ Bernoulli(p1), by enumeration over the noise cells."""
    total = F(0)
    for s0, p_s0 in ((0, 1 - p1), (1, p1)):
        total += p_s0 * F(
            sum(
                int(uy < 2 + 4 * a1 + 3 * int(ul < s0 + a1 + 1) + 5 * a2 + 2 * s0 + 2 * s0 * a1)
                for ul in range(4)
                for uy in range(20)
            ),
            80,
        )
    return total


def observational_g_formula(a1: int, a2: int, p1: F) -> F:
    """The same quantity from the enumerated observational law, using only its counts."""
    rows = list(configs())
    count = lambda **k: sum(
        all(dict(zip(("s0", "a1", "l2", "a2", "y"), r))[n] == v for n, v in k.items())
        for r in rows
    )
    total = F(0)
    for s0, p_s0 in ((0, 1 - p1), (1, p1)):
        for l2 in (0, 1):
            p_l = F(count(s0=s0, a1=a1, l2=l2), count(s0=s0, a1=a1))
            e_y = F(count(s0=s0, a1=a1, l2=l2, a2=a2, y=1), count(s0=s0, a1=a1, l2=l2, a2=a2))
            total += p_s0 * p_l * e_y
    return total


def compute() -> dict:
    out = {"response_formula_ok": True}
    for p1 in (F(4, 5), F(1, 4)):
        for a1, a2 in itertools.product((0, 1), repeat=2):
            exact = response(a1, a2, p1)
            if exact != observational_g_formula(a1, a2, p1):
                out["response_formula_ok"] = False
            formula = (
                F(1375, 10000)
                + F(2375, 10000) * a1
                + F(25, 100) * a2
                + F(1375, 10000) * p1
                + F(1, 10) * p1 * a1
            )
            if exact != formula:
                out["response_formula_ok"] = False
            out[(float(p1), a1, a2)] = exact
    # Additive paired candidate with target P(S0=1)=4/5, balanced complete histories.
    unit_mean = F(sum(range(24)), 24) / 100
    p1 = F(4, 5)
    candidate = lambda a1, a2: 1 + p1 + 2 * a1 + 3 * a2 + F(1, 2) * F(1, 2) + unit_mean
    out["candidate_response_11"] = candidate(1, 1)
    out["candidate_effect"] = candidate(1, 1) - candidate(0, 0)
    return out


def check(document: dict) -> list[str]:
    got = compute()
    problems = []
    if not got["response_formula_ok"]:
        problems.append("response formula or observational g-formula disagrees with enumeration")
    if document["source_units"] != 640 or document["identical_histories_per_unit"] != 2:
        problems.append("source size differs from the 640 enumerated configurations")
    if len(list(configs())) != 640:
        problems.append("configuration count is not 640")
    for p1, key in ((0.8, "target_p1_0_8"), (0.25, "target_p1_0_25")):
        for name, value in document[key].items():
            if name.startswith("response_"):
                a1, a2 = int(name[-2]), int(name[-1])
                reference = float(got[(p1, a1, a2)])
            elif name == "effect_11_vs_00":
                reference = float(got[(p1, 1, 1)] - got[(p1, 0, 0)])
            elif name == "effect_10_vs_01":
                reference = float(got[(p1, 1, 0)] - got[(p1, 0, 1)])
            else:
                problems.append(f"{key}.{name} has no reference")
                continue
            if abs(reference - value) > ATOL:
                problems.append(f"{key}.{name}: fixture {value} reference {reference}")
    cand = document["paired_additive_candidate"]
    for name, reference in (
        ("response_11", got["candidate_response_11"]),
        ("effect_11_vs_00", got["candidate_effect"]),
    ):
        if abs(float(reference) - cand[name]) > ATOL:
            problems.append(f"paired_additive_candidate.{name}: {cand[name]} vs {float(reference)}")
    if document["structural_model"]["target_response"] != (
        "0.1375+0.2375*a1+0.25*a2+0.1375*p1+0.1*p1*a1"
    ):
        problems.append("stated response formula differs from the verified one")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["target_p1_0_8"]["response_11"] += 1e-6
    if not check(wrong):
        failures.append("shifted response accepted")
    wrong = copy.deepcopy(document)
    wrong["target_p1_0_25"]["effect_11_vs_00"] = wrong["target_p1_0_8"]["effect_11_vs_00"]
    if not check(wrong):
        failures.append("effect under the wrong target law accepted")
    wrong = copy.deepcopy(document)
    wrong["paired_additive_candidate"]["effect_11_vs_00"] = 5.1
    if not check(wrong):
        failures.append("shifted paired effect accepted")
    wrong = copy.deepcopy(document)
    wrong["structural_model"]["target_response"] = "0.1375+0.2375*a1"
    if not check(wrong):
        failures.append("changed stated formula accepted")
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
