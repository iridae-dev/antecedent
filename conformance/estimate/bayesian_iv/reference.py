# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the structural effect behind the bayesian_iv fixture: on the test's 1200-row design (z_i = i mod 2, u_i = ((i mod 17) - 8)/5, t = 0.6 z + u, y = 2 t + u) the exact Wald/2SLS ratio cov(z, y) / cov(z, t) is computed in rationals and lies within 0.01 of the structural coefficient 2, while the ordinary least-squares slope is biased far from it.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/bayesian_iv/reference.py --check
    uv run --frozen --script conformance/estimate/bayesian_iv/reference.py --self-test

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


N = 1200

def design():
    z = [Fraction(i % 2) for i in range(N)]
    u = [Fraction((i % 17) - 8, 5) for i in range(N)]
    t = [Fraction(3, 5) * zi + ui for zi, ui in zip(z, u)]
    y = [2 * ti + ui for ti, ui in zip(t, u)]
    return z, t, y

def cov(a, b):
    ma, mb = sum(a) / N, sum(b) / N
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / N

def derived():
    z, t, y = design()
    return cov(z, y) / cov(z, t), cov(t, y) / cov(t, t)

def check(document):
    problems = []
    wald, ols = derived()
    if abs(float(wald) - document["true_structural_effect"]) > 0.01:
        problems.append(f"Wald ratio {float(wald)!r} is not within 0.01 of the fixture truth {document['true_structural_effect']!r}")
    if abs(float(ols) - document["true_structural_effect"]) < 0.2:
        problems.append("least-squares slope is not separated from the truth, so the design is not endogenous")
    if not document["first_stage_f_threshold"] > 0:
        problems.append("first-stage F threshold is not positive")
    return problems

MUTATIONS = [
    ("truth shifted by 0.5", lambda d: d.__setitem__("true_structural_effect", d["true_structural_effect"] + 0.5)),
    ("truth replaced by the confounded slope", lambda d: d.__setitem__("true_structural_effect", 2.6)),
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
