# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the nested temporal counterfactual fixture: with M = 0.7 T + 0.05 i and Y = 0.5 T + 1.5 M + 0.01 i over 40 rows, the nested Y(1, M(0)), the pure do(T=1) and do(T=0) means and their difference follow by substitution.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/gcm/nested_temporal_counterfactual/reference.py --check
    uv run --frozen --script conformance/gcm/nested_temporal_counterfactual/reference.py --self-test

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
    n = document["model"]["rows"]
    mean = lambda f: sum(F(f(i)) for i in range(n)) / n
    M = lambda t, i: F(7, 10) * t + F(5, 100) * i
    Y = lambda t, m, i: F(1, 2) * t + F(3, 2) * m + F(1, 100) * i
    ref = document["reference"]
    tol = document["acceptance"]["atol"]
    nested = mean(lambda i: Y(1, M(0, i), i))
    do1 = mean(lambda i: Y(1, M(1, i), i))
    do0 = mean(lambda i: Y(0, M(0, i), i))
    close(nested, ref["nested_y_t1_m_t0_mean"], tol, "nested_y_t1_m_t0_mean", problems)
    close(do1, ref["pure_do_t1_mean"], tol, "pure_do_t1_mean", problems)
    close(nested - do1, ref["difference"], tol, "difference", problems)
    close(do0, ref["trajectory_do_t0_mean"], tol, "trajectory_do_t0_mean", problems)
    close(do1, ref["trajectory_do_t1_mean"], tol, "trajectory_do_t1_mean", problems)
    return problems


MUTATIONS = [
    ("nested mean changed", lambda d: d["reference"].__setitem__("nested_y_t1_m_t0_mean", 2.2)),
    ("difference changed", lambda d: d["reference"].__setitem__("difference", -1.0)),
    ("rows changed", lambda d: d["model"].__setitem__("rows", 41)),
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
