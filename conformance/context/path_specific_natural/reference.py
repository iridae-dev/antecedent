# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the path-specific natural effect behind the context path_specific_natural fixture: on the chain T -> M -> Y with T unconfounded and no direct edge, the only path is the mediated one, so the effect equals P(Y=1|T=1) - P(Y=1|T=0) on the test's count table, which is 0.3.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/context/path_specific_natural/reference.py --check
    uv run --frozen --script conformance/context/path_specific_natural/reference.py --self-test

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


CELLS = [(0, 0, 0, 40), (0, 0, 1, 10), (0, 1, 0, 10), (0, 1, 1, 40),
         (1, 0, 0, 10), (1, 0, 1, 10), (1, 1, 0, 10), (1, 1, 1, 70)]  # (t, m, y, count)

def derived():
    def p_y1(t):
        rows = [c for c in CELLS if c[0] == t]
        return Fraction(sum(c[3] for c in rows if c[2] == 1), sum(c[3] for c in rows))
    # Mediation-formula route: sum_m P(y=1|m) [P(m|t=1) - P(m|t=0)] with Y independent of T given M
    # is the same number only when the table respects the chain; the observed contrast is used here
    # because the graph has no other path.
    return p_y1(1) - p_y1(0)

def check(document):
    problems = []
    close(derived(), document["ate"], 1e-12, "ate", problems)
    if not 0 < document["tolerance"] <= 1e-9:
        problems.append("tolerance is not a positive round-off-scale band")
    return problems

MUTATIONS = [
    ("effect shifted", lambda d: d.__setitem__("ate", 0.35)),
    ("effect replaced by the direct-edge contrast", lambda d: d.__setitem__("ate", 0.2)),
    ("tolerance widened", lambda d: d.__setitem__("tolerance", 0.05)),
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
