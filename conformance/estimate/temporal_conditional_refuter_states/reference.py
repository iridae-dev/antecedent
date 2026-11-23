# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the marginal conditional effect behind the
temporal_conditional_refuter_states fixture.

The series obeys r_t, e_t, u_t independent standard normal, x_t = a r_t + e_t, w_t Bernoulli(p)
independent of the rest, y_t = b x_{t-1} + c w_t + d x_{t-1} w_t + s u_t. The slope of the pulse
in x_{t-1} at the modifier w_t is b + d w_t, so the effect marginal over the modifier law is
b + d p. Exact rationals; standard library only; imports no Antecedent code and no statistics
package.

    uv run --frozen --script conformance/estimate/temporal_conditional_refuter_states/reference.py --check
    uv run --frozen --script conformance/estimate/temporal_conditional_refuter_states/reference.py --self-test

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


def check(document):
    problems = []
    law = document["law"]
    b = Fraction(law["slope"]).limit_denominator(10**6)
    d = Fraction(law["interaction"]).limit_denominator(10**6)
    p = Fraction(law["modifier_probability"]).limit_denominator(10**6)
    effect = b + d * p
    if not math.isclose(float(effect), document["marginal_effect"], rel_tol=0.0, abs_tol=1e-12):
        problems.append(f"marginal_effect: derived {float(effect)!r} != fixture {document['marginal_effect']!r}")
    if abs(float(b) - document["marginal_effect"]) < 5 * document["tolerance"]["effect"] / 10:
        problems.append("the unmoderated slope must lie outside the tolerance band")
    contract = document["refuter_contract"]
    for inference in ("Frequentist", "Bayesian"):
        if set(contract[inference]["cheap"]) - set(contract[inference]["full"]):
            problems.append(f"{inference}: full must run every refuter cheap runs")
    return problems


MUTATIONS = [
    ("effect shifted", lambda d: d.__setitem__("marginal_effect", 1.9)),
    ("effect replaced by the unmoderated slope", lambda d: d.__setitem__("marginal_effect", 1.5)),
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
