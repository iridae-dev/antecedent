# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the seven derivative targets behind the temporal_derivative fixture: with Y = 3 + 2X + 0.5B, V = 1 + X - B and B alternating with mean 1/2, the response mean is 3.25 + 2x, so at x = 2 the slope is 2, the elasticity 4/7.25, the semi-elasticities 4 and 2/7.25, the Jacobian rows (2, 0.5) and (1, -1), and the directional derivative along (1, 2) is (3, -1).

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/temporal_derivative/reference.py --check
    uv run --frozen --script conformance/estimate/temporal_derivative/reference.py --self-test

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


Y0, YX, YB = Fraction(3), Fraction(2), Fraction(1, 2)
V0, VX, VB = Fraction(1), Fraction(1), Fraction(-1)
X_AT = Fraction(2)
DIRECTION = (Fraction(1), Fraction(2))

def derived():
    mean_b = Fraction(1, 2)
    mean_y = lambda x: Y0 + YX * x + YB * mean_b
    level = mean_y(X_AT)
    slope = YX  # d/dx of the structural mean
    jacobian = [YX, YB, VX, VB]
    directional = [YX * DIRECTION[0] + YB * DIRECTION[1], VX * DIRECTION[0] + VB * DIRECTION[1]]
    return {
        "point": slope,
        "elasticity": slope * X_AT / level,
        "semi_treatment": slope * X_AT,
        "semi_outcome": slope / level,
        "average": slope,
        "jacobian": jacobian,
        "directional": directional,
    }

def check(document):
    problems = []
    want = derived()
    for key, value in want.items():
        if isinstance(value, list):
            got = document[key]
            if len(got) != len(value):
                problems.append(f"{key}: length {len(got)} != {len(value)}")
                continue
            for i, v in enumerate(value):
                close(v, got[i], 1e-12, f"{key}[{i}]", problems)
        else:
            close(value, document[key], 1e-12, key, problems)
    if not 0 < document["tolerance"] < 0.1:
        problems.append("tolerance is not a small positive band")
    return problems

MUTATIONS = [
    ("point slope shifted", lambda d: d.__setitem__("point", 2.5)),
    ("elasticity replaced by the semi-elasticity", lambda d: d.__setitem__("elasticity", 0.27586206896551724)),
    ("jacobian entry swapped", lambda d: d["jacobian"].reverse()),
    ("directional shifted", lambda d: d["directional"].__setitem__(0, 2.0)),
    ("tolerance widened", lambda d: d.__setitem__("tolerance", 0.5)),
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
