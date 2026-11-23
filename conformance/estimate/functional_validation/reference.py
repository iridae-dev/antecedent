# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the two closed-form values behind the functional_validation fixture from the test's count tables: the path effect along T -> M -> Y equals P(Y=1|T=1) - P(Y=1|T=0) = 0.3 because the chain has no other path and no confounder of T, and the intervention mean P(Y=1|do(T=1)) = sum_z P(z) P(Y=1|T=1,z) = 0.7 on the confounded table.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/functional_validation/reference.py --check
    uv run --frozen --script conformance/estimate/functional_validation/reference.py --self-test

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


CHAIN = [(0, 0, 0, 40), (0, 0, 1, 10), (0, 1, 0, 10), (0, 1, 1, 40),
         (1, 0, 0, 10), (1, 0, 1, 10), (1, 1, 0, 10), (1, 1, 1, 70)]  # (t, m, y, count)
CONFOUNDED = [(0, 0, 0, 21), (0, 0, 1, 9), (0, 1, 0, 4), (0, 1, 1, 16),
              (1, 0, 0, 12), (1, 0, 1, 3), (1, 1, 0, 14), (1, 1, 1, 21)]  # (z, t, y, count)

def derived():
    def p_y1_given_t(t):
        rows = [r for r in CHAIN if r[0] == t]
        return Fraction(sum(r[3] for r in rows if r[2] == 1), sum(r[3] for r in rows))
    path = p_y1_given_t(1) - p_y1_given_t(0)
    n = sum(r[3] for r in CONFOUNDED)
    mean = Fraction(0)
    for z in (0, 1):
        pz = Fraction(sum(r[3] for r in CONFOUNDED if r[0] == z), n)
        rows = [r for r in CONFOUNDED if r[0] == z and r[1] == 1]
        mean += pz * Fraction(sum(r[3] for r in rows if r[2] == 1), sum(r[3] for r in rows))
    return path, mean

def check(document):
    problems = []
    path, mean = derived()
    close(path, document["path_effect"], 1e-12, "path_effect", problems)
    close(mean, document["distribution_mean"], 1e-12, "distribution_mean", problems)
    if not 0 < document["tolerance"] <= 1e-9:
        problems.append("tolerance is not a positive round-off-scale band")
    return problems

MUTATIONS = [
    ("path effect shifted", lambda d: d.__setitem__("path_effect", 0.35)),
    ("mean replaced by the confounded conditional", lambda d: d.__setitem__("distribution_mean", 0.7 + 0.05)),
    ("tolerance widened", lambda d: d.__setitem__("tolerance", 0.01)),
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
