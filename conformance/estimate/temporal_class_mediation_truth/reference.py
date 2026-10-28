# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the temporal-mediation truth: in m_i = a t_{i-1} + 0.6 z_{i-1} + 0.15 e_i, y_i = 1 + c t_{i-1} + b m_i + 5 z_{i-1} the natural indirect effect is a*b, the natural direct effect c and the total c + a*b; they are recovered here by evaluating the equations on the stated sample rows under nested interventions.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/temporal_class_mediation_truth/reference.py --check
    uv run --frozen --script conformance/estimate/temporal_class_mediation_truth/reference.py --self-test

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
    s = document["structural"]
    a, b, c = F(str(s["a"])), F(str(s["b"])), F(str(s["direct_c"]))
    pattern = [1, -1, -1, 1]       # period-4 +-1 confounder pattern (any sign pattern leaves contrasts unchanged)
    totals, directs, indirects = set(), set(), set()
    for i in range(1, 41):
        z, e = F(pattern[(i - 1) % 4]), F((i % 5) - 2)
        m = lambda t: a * t + F("0.6") * z + F("0.15") * e
        y = lambda t, mm: 1 + c * t + b * mm + 5 * z
        totals.add(y(1, m(1)) - y(0, m(0)))
        directs.add(y(1, m(0)) - y(0, m(0)))
        indirects.add(y(0, m(1)) - y(0, m(0)))
    for name, values, key in (("total", totals, "total"), ("direct", directs, "direct"), ("mediated", indirects, "mediated")):
        if len(values) != 1:
            problems.append(f"{name} contrast varies with the exogenous rows: {sorted(values)}")
            continue
        close(next(iter(values)), document["horizon_1"][key], 1e-12, f"horizon_1.{key}", problems)
    return problems


MUTATIONS = [
    ("mediated changed", lambda d: d["horizon_1"].__setitem__("mediated", 0.5)),
    ("total changed", lambda d: d["horizon_1"].__setitem__("total", 0.7)),
    ("coefficient changed", lambda d: d["structural"].__setitem__("b", 0.5)),
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
