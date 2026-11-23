# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the lag-0 average effect behind the lag_zero_average_effect fixture.

The series obeys z_t independent standard normal, x_t = a z_{t-1} + e_t with e_t standard normal,
y_t = beta x_t + gamma z_{t-1} + noise. The lag-0 contrast
E[y_t | do(x_t = 1)] - E[y_t | do(x_t = 0)] is beta whatever gamma is, because do(x_t) cuts
z_{t-1} -> x_t. Adjusting for the confounder z_{t-1} recovers beta; omitting it converges to
beta + gamma a / (a^2 + 1) (the regression slope of y_t on x_t). Exact rationals; standard
library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/lag_zero_average_effect/reference.py --check
    uv run --frozen --script conformance/estimate/lag_zero_average_effect/reference.py --self-test

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
    beta = Fraction(law["beta"]).limit_denominator(10**6)
    a = Fraction(law["treatment_loading"]).limit_denominator(10**6)
    gamma = Fraction(law["confounder_to_outcome"]).limit_denominator(10**6)
    omitted = beta + gamma * a / (a * a + 1)
    for label, got, want in (
        ("effect", beta, document["effect"]),
        ("unadjusted_plim", omitted, document["unadjusted_plim"]),
    ):
        if not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=1e-12):
            problems.append(f"{label}: derived {float(got)!r} != fixture {want!r}")
    if float(omitted) - document["effect"] < 5 * document["tolerance"]["effect"]:
        problems.append("the omitted-confounder slope must lie well outside the tolerance band")
    contract = document["refuter_contract"]
    for inference in ("Frequentist", "Bayesian", "Bayesian_class_graph_posterior"):
        if set(contract[inference]["cheap"]) - set(contract[inference]["full"]):
            problems.append(f"{inference}: full must run every refuter cheap runs")
    return problems


MUTATIONS = [
    ("effect shifted", lambda d: d.__setitem__("effect", 2.1)),
    ("effect replaced by the unadjusted slope", lambda d: d.__setitem__("effect", d["unadjusted_plim"])),
    ("a wider tolerance", lambda d: d["tolerance"].__setitem__("effect", 0.5)),
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
