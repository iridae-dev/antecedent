# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the four contrasts behind the temporal_cross_world fixture: with M = 0.5 X + U and Y = 1.5 X + 4 M, evaluating the structural equations at both treatment values and several shared disturbances gives nested direct 1.5, mediator-path 0.5 * 4 = 2, and total (counterfactual and distribution) 3.5, independent of U.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/temporal_cross_world/reference.py --check
    uv run --frozen --script conformance/estimate/temporal_cross_world/reference.py --self-test

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


F_M = lambda x, u: Fraction(1, 2) * x + u
F_Y = lambda x, m: Fraction(3, 2) * x + 4 * m
SHOCKS = (Fraction(3, 10), Fraction(-3, 10), Fraction(0), Fraction(7, 3))

def derived():
    found = set()
    for u in SHOCKS:
        m0, m1 = F_M(0, u), F_M(1, u)
        nested = F_Y(1, m0) - F_Y(0, m0)  # treatment held at the mediator's control-world value
        path = F_Y(0, m1) - F_Y(0, m0)  # only the mediator moves
        total = F_Y(1, m1) - F_Y(0, m0)
        found.add((nested, path, total))
    if len(found) != 1:
        raise SystemExit("contrasts depend on the shared disturbance")
    return next(iter(found))

def check(document):
    problems = []
    nested, path, total = derived()
    close(nested, document["nested"], 1e-12, "nested", problems)
    close(path, document["path"], 1e-12, "path", problems)
    close(total, document["counterfactual"], 1e-12, "counterfactual", problems)
    close(total, document["distribution"], 1e-12, "distribution", problems)
    if nested + path != total:
        problems.append("nested direct plus path effect is not the total")
    if not 0 < document["tolerance"] < min(float(nested), float(path)):
        problems.append("tolerance is not a positive band narrower than the smallest contrast")
    return problems

MUTATIONS = [
    ("nested shifted", lambda d: d.__setitem__("nested", 2.0)),
    ("path swapped with nested", lambda d: d.__setitem__("path", 1.5)),
    ("total shifted", lambda d: d.__setitem__("distribution", 3.0)),
    ("tolerance widened", lambda d: d.__setitem__("tolerance", 1.6)),
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
