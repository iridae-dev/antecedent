# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the jump behind the bayesian_rd fixture: on the test's 400-row design (r_i = (i - 200)/100, y = 1 + 2.5 t + 0.8 r + 1.2 t r + 0.01 ((i mod 5) - 2)) a local-linear fit on each side of the cutoff within the fixed bandwidth, solved by exact rational least squares, gives a jump within 0.01 of the structural 2.5.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/bayesian_rd/reference.py --check
    uv run --frozen --script conformance/estimate/bayesian_rd/reference.py --self-test

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


N = 400

def rows():
    out = []
    for i in range(N):
        r = Fraction(i - 200, 100)
        t = 1 if r >= 0 else 0
        y = 1 + Fraction(5, 2) * t + Fraction(4, 5) * r + Fraction(6, 5) * t * r + Fraction((i % 5) - 2, 100)
        out.append((r, t, y))
    return out

def line(points):
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - slope * sx) / n  # intercept at the cutoff r = 0

def derived(bandwidth, inclusive):
    h = Fraction(str(bandwidth))
    inside = lambda r: abs(r) <= h if inclusive else abs(r) < h
    data = [(r, t, y) for r, t, y in rows() if inside(r)]
    right = line([(r, y) for r, t, y in data if t == 1])
    left = line([(r, y) for r, t, y in data if t == 0])
    return right - left

def check(document):
    problems = []
    for inclusive in (True, False):
        jump = derived(document["fixed_bandwidth"], inclusive)
        if abs(float(jump) - document["true_jump"]) > 0.01:
            problems.append(f"local-linear jump {float(jump)!r} (inclusive={inclusive}) is not within 0.01 of fixture {document['true_jump']!r}")
    if document["cutoff"] != 0.0:
        problems.append("cutoff differs from the zero used by this reference")
    return problems

MUTATIONS = [
    ("jump shifted by 0.5", lambda d: d.__setitem__("true_jump", d["true_jump"] + 0.5)),
    ("cutoff moved", lambda d: d.__setitem__("cutoff", 0.5)),
    ("bandwidth widened until curvature-free bias is irrelevant but jump changed", lambda d: d.__setitem__("true_jump", 1.2)),
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
