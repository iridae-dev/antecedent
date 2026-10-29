# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the joint intervention level behind the codetermined_joint_level fixture.

The design is a table of exact cell counts over (z, t1, t2); the outcome law is additive,
y = 0.2 + 0.4 t1 + 0.3 t2 + 0.2 z. The tier closure adjusts for z, so
E[y | do(t1 = 1, t2 = 1)] = 0.2 + 0.4 + 0.3 + 0.2 mean(z), with mean(z) the exact count
average. Exact rationals; standard library only; imports no Antecedent code and no statistics
package.

    uv run --frozen --script conformance/estimate/codetermined_joint_level/reference.py --check
    uv run --frozen --script conformance/estimate/codetermined_joint_level/reference.py --self-test

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
INTERCEPT, B1, B2, BZ = Fraction(1, 5), Fraction(2, 5), Fraction(3, 10), Fraction(1, 5)


def derived(document):
    cells = document["design"]["cells"]
    total = sum(cell["count"] for cell in cells)
    mean_z = Fraction(sum(cell["z"] * cell["count"] for cell in cells), total)
    return total, mean_z, INTERCEPT + B1 + B2 + BZ * mean_z


def check(document):
    problems = []
    total, mean_z, level = derived(document)
    for label, got, want in (
        ("rows", total, document["design"]["rows"]),
        ("mean_z", float(mean_z), document["design"]["mean_z"]),
        ("level", float(level), document["level"]["value"]),
    ):
        if not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=1e-12):
            problems.append(f"{label}: derived {got!r} != fixture {want!r}")
    return problems


MUTATIONS = [
    ("level shifted", lambda d: d["level"].__setitem__("value", d["level"]["value"] + 0.01)),
    ("level without the confounder term", lambda d: d["level"].__setitem__("value", 0.9)),
    ("a cell count moved", lambda d: d["design"]["cells"][7].__setitem__("count", 359)),
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
