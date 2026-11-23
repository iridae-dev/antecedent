# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the nested-counterfactual fixture: in M = 0.8 X + U_M, Y = 1.7 X + 4 M + U_Y the natural direct effect Y(x1, M(x0)) - Y(x0, M(x0)) with a shared exogenous draw is (x1 - x0) * 1.7, evaluated here by substitution with arbitrary exogenous values.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/nested_counterfactual/reference.py --check
    uv run --frozen --script conformance/estimate/nested_counterfactual/reference.py --self-test

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
F = Fraction
FIXTURE = HERE / "expected.json"


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got) if got is not None else got!r} != fixture {want!r} (tol {tol})")


def same(got, want, label, problems):
    if got != want:
        problems.append(f"{label}: derived {got!r} != fixture {want!r}")

def check(document):
    problems = []
    model = document["model"]
    x0, x1 = F(str(model["treatment_control"])), F(str(model["treatment_active"]))
    a, c, b = F("0.8"), F("1.7"), F("4.0")
    for u_m, u_y in ((F(0), F(0)), (F(3, 10), F(-1, 5)), (F(-7, 3), F(11, 4))):
        m0 = a * x0 + u_m
        y_x1_m0 = c * x1 + b * m0 + u_y
        y_x0_m0 = c * x0 + b * m0 + u_y
        nde = y_x1_m0 - y_x0_m0
        close(nde, document["reference"]["natural_direct_effect"], document["acceptance"]["atol"], f"NDE (u_m={u_m}, u_y={u_y})", problems)
    close((x1 - x0) * c, document["reference"]["natural_direct_effect"], 1e-12, "formula", problems)
    return problems


MUTATIONS = [
    ("effect changed", lambda d: d["reference"].__setitem__("natural_direct_effect", 5.2)),
    ("control level changed", lambda d: d["model"].__setitem__("treatment_control", 0.0)),
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
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
