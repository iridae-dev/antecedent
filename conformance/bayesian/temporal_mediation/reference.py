# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the path-product effects behind the bayesian temporal_mediation fixture: with m_t = 0.8 t_{t-1} + noise and y_t = 0.25 t_{t-1} + 0.55 m_t + noise, the mediated effect is 0.8 * 0.55 = 0.44, the direct effect 0.25 and the total 0.69.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/temporal_mediation/reference.py --check
    uv run --frozen --script conformance/bayesian/temporal_mediation/reference.py --self-test

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


A, B, DIRECT = Fraction(4, 5), Fraction(11, 20), Fraction(1, 4)

def structural(treatment):
    # Noise-free parts of the stated equations, evaluated under do(treatment).
    m = A * treatment
    return m, DIRECT * treatment + B * m

def derived():
    m0, y0 = structural(Fraction(0))
    m1, y1 = structural(Fraction(1))
    total = y1 - y0
    mediated = B * (m1 - m0)
    direct = DIRECT
    return mediated, direct, total

def check(document):
    problems = []
    mediated, direct, total = derived()
    close(mediated, document["mediated"], 1e-12, "mediated", problems)
    close(direct, document["direct"], 1e-12, "direct", problems)
    close(total, document["total"], 1e-12, "total", problems)
    if mediated + direct != total:
        problems.append("mediated + direct is not the total")
    if not document["tolerance"] > 0:
        problems.append("tolerance is not positive")
    return problems

MUTATIONS = [
    ("mediated shifted by 0.1", lambda d: d.__setitem__("mediated", d["mediated"] + 0.1)),
    ("total no longer the sum", lambda d: d.__setitem__("total", d["total"] + 0.1)),
    ("direct replaced by the mediated effect", lambda d: d.__setitem__("direct", 0.44)),
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
