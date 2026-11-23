# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the direct-trial IPW value behind the staged_transport fixture: trial rows (A,Y) = (1,3) and (0,1) with selection s = 0.5 and treatment probability e = 0.5, two target rows. Each trial row contributes the odds weight (1-s)/s times (A-e)/(e(1-e)) times Y; the sum is divided by the number of target rows.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/staged_transport/reference.py --check
    uv run --frozen --script conformance/estimate/staged_transport/reference.py --self-test

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
    if got is None or want is None or not math.isclose(float(got), want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


TRIAL = [(1, 3), (0, 1)]
N_TARGET = 2
S_PROB = E_PROB = Fraction(1, 2)

def derived():
    total = Fraction(0)
    for a, y in TRIAL:
        odds = (1 - S_PROB) / S_PROB
        total += odds * (a - E_PROB) / (E_PROB * (1 - E_PROB)) * y
    return total, total / N_TARGET

def check(document):
    problems = []
    total, ipw = derived()
    if total != 4:
        problems.append(f"IPW sum {total} differs from the 4 stated in the fixture notes")
    close(ipw, document["ipw"], 1e-12, "ipw", problems)
    if not 0 < document["tolerance"] <= 1e-9:
        problems.append("tolerance is not a positive round-off-scale band")
    return problems

MUTATIONS = [
    ("ipw shifted", lambda d: d.__setitem__("ipw", d["ipw"] + 0.5)),
    ("ipw replaced by the trial sum", lambda d: d.__setitem__("ipw", 4.0)),
    ("tolerance widened to a sampling-scale band", lambda d: d.__setitem__("tolerance", 0.1)),
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
