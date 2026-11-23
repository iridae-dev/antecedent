# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the exposure contrast behind the bayesian_interference fixture: the stated outcomes (5, 6, 9, 10) are reproduced exactly by y = 2 + 3*own + 4*treated_in_neighbours on the stated assignment and edges (read as directed, source to target), so the to-minus-from contrast of exposure (own 1, 0) against (own 0, 1) is 3 - 4 = -1.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/bayesian_interference/reference.py --check
    uv run --frozen --script conformance/estimate/bayesian_interference/reference.py --self-test

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


OUTCOMES = [5, 6, 9, 10]
ASSIGNMENT = [True, False, True, False]
EDGES = [(0, 1), (0, 2), (0, 3), (2, 3)]
INTERCEPT, OWN, NEIGHBOUR = Fraction(2), Fraction(3), Fraction(4)

def treated_in_neighbours(unit):
    return sum(1 for a, b in EDGES if b == unit and ASSIGNMENT[a])

def derived():
    fitted = [INTERCEPT + OWN * int(ASSIGNMENT[u]) + NEIGHBOUR * treated_in_neighbours(u) for u in range(4)]
    contrast = (OWN * 1 + NEIGHBOUR * 0) - (OWN * 0 + NEIGHBOUR * 1)
    return fitted, contrast

def check(document):
    problems = []
    fitted, contrast = derived()
    if [float(v) for v in fitted] != [float(v) for v in document["outcomes"]]:
        problems.append(f"stated model reproduces {[float(v) for v in fitted]}, fixture outcomes {document['outcomes']}")
    if document["assignment"] != ASSIGNMENT or [tuple(e) for e in document["edges"]] != EDGES:
        problems.append("fixture assignment or edges differ from the ones this reference derived from")
    close(contrast, document["expected_contrast"], 1e-12, "expected_contrast", problems)
    if not 0 < document["tolerance"] < 1:
        problems.append("tolerance is not a positive band narrower than the contrast")
    return problems

MUTATIONS = [
    ("contrast sign flipped", lambda d: d.__setitem__("expected_contrast", 1.0)),
    ("an outcome perturbed", lambda d: d["outcomes"].__setitem__(0, 7.0)),
    ("an edge removed", lambda d: d["edges"].pop()),
    ("tolerance widened to the contrast", lambda d: d.__setitem__("tolerance", 1.5)),
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
