# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the modifier-averaged effect behind the context conditional_effect fixture: with Y = 1 + 2T + 0.5 T W and the test's modifier column W_i = i mod 5 over 200 rows, the average unit-level effect is 2 + 0.5 mean(W) = 3.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/context/conditional_effect/reference.py --check
    uv run --frozen --script conformance/context/conditional_effect/reference.py --self-test

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


N = 200
A, B = Fraction(2), Fraction(1, 2)

def derived():
    w = [Fraction(i % 5) for i in range(N)]
    return sum((1 + A + B * wi) - 1 for wi in w) / N

def check(document):
    problems = []
    effect = derived()
    close(effect, document["ate_target"], 1e-12, "ate_target", problems)
    if not 0 < document["ate_tol"] < abs(effect - A):
        problems.append("ate_tol is not a positive band narrower than the gap to the main-effect-only value 2")
    return problems

MUTATIONS = [
    ("target shifted by 0.5", lambda d: d.__setitem__("ate_target", d["ate_target"] + 0.5)),
    ("target replaced by the main effect alone", lambda d: d.__setitem__("ate_target", 2.0)),
    ("tolerance widened past the interaction contribution", lambda d: d.__setitem__("ate_tol", 1.5)),
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
