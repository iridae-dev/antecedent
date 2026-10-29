# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the completion levels and witness effects behind the
intervention_response_class_refuter_states fixture.

Static law: z, e independent standard normal, t = 0.6 z + e, y = 1 + 2 t + z + 0.5 u. The
completion that adjusts for z identifies E[y | do(t = 1)] = 1 + 2 + E z = 3; the completion that
does not adjust identifies the association E[y | t = 1] = E y + slope (1 - E t) with
slope = 2 + cov(t, z) / var(t) = 2 + 0.6 / 1.36 (population values E z = E t = 0), 3.4412.
The PAG law y = 1 + 2 t + 0.8 z + e identifies 3 + 0.8 E z = 3.

Temporal law: r_i, e_i independent standard normal, z = 0.4 r + e_z, t = 0.3 + 0.5 r + 0.2 z + 0.7 e_t,
y_i = 1 + 2 t_{i-1} + 0.6 z_{i-1} + 0.3 u. The adjusting completion identifies 1 + 2 + 0.6 E z = 3 with
unit-contrast witness 2; the completion that treats z as a mediator identifies the association
mean_y + slope (1 - mean_t) with mean_t = 0.3, mean_y = 1.6 and slope = 2 + 0.6 cov(t, z) / var(t),
cov(t, z) = 0.5 * 0.4 + 0.2 * 1.16 = 0.432 and var(t) = 0.25 + 0.04 * 1.16 + 0.49 + 2 * 0.5 * 0.2 * 0.4
= 0.8664, so witness 2.2990 and level 3.2094. Exact rationals; standard library only; imports no
Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/intervention_response_class_refuter_states/reference.py --check
    uv run --frozen --script conformance/estimate/intervention_response_class_refuter_states/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent


def derived():
    static_slope = 2 + F(6, 10) / (F(36, 100) + 1)
    static_unadjusted = 1 + static_slope  # E y = 1, E t = 0
    cov_tz = F(5, 10) * F(4, 10) + F(2, 10) * (F(16, 100) + 1)
    var_t = F(25, 100) + F(4, 100) * (F(16, 100) + 1) + F(49, 100) + 2 * F(5, 10) * F(2, 10) * F(4, 10)
    temporal_slope = 2 + F(6, 10) * cov_tz / var_t
    temporal_unadjusted = (1 + 2 * F(3, 10)) + temporal_slope * (1 - F(3, 10))
    return {
        "static_unadjusted": float(static_unadjusted),
        "temporal_unadjusted": float(temporal_unadjusted),
        "temporal_witness_unadjusted": float(temporal_slope),
    }


def check(document):
    problems = []
    want = derived()
    for key, value in want.items():
        if not math.isclose(value, document["closed_form"][key], rel_tol=0.0, abs_tol=1e-9):
            problems.append(f"{key}: derived {value!r} != fixture {document['closed_form'][key]!r}")
    if document["closed_form"]["adjusted_level"] != 3.0 or document["closed_form"]["adjusted_witness"] != 2.0:
        problems.append("the adjusting completion identifies the level 3 and the unit contrast 2")
    contract = document["refuter_contract"]
    for inference in ("Frequentist", "Bayesian"):
        for family in ("static", "temporal"):
            entry = contract[inference][family]
            if set(entry["cheap"]) - set(entry["full"]):
                problems.append(f"{inference} {family}: full must run every refuter cheap runs")
    return problems


MUTATIONS = [
    ("unadjusted level replaced by the adjusted one", lambda d: d["closed_form"].__setitem__("static_unadjusted", 3.0)),
    ("witness shifted", lambda d: d["closed_form"].__setitem__("temporal_witness_unadjusted", 2.0)),
    ("full drops a cheap refuter", lambda d: d["refuter_contract"]["Frequentist"]["static"]["full"].clear()),
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
