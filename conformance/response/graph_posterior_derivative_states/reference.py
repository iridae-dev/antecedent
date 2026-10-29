# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the graph-posterior derivative fixture.

The rows follow the linear law of the derivative refuter-state fixture (A = 2 + sin(.71 i) + .2 cos(.13 i),
B = .7 e, Y = 5 + 2 A - .2 B + .3 e, V = 1 + .25 A + .3 B + .3 e for i = 0..799, e = sum of twelve
uniforms minus six from one 64-bit linear congruential stream, state 20240607, multiplier 6364136223846793005,
increment 1442695040888963407, uniform = (state >> 11) / 2^53, drawn for B over every row, then Y, then V). The graph
posterior holds two DAG atoms that both identify the slope of Y in A (B is not a confounder in the first and a
confounder in the second) and a cyclic sample, which is not a member of the DAG class. Every identified atom has the
same structural derivative (the coefficient of A), so the posterior-weighted mean over the atoms that identify it is
that coefficient whatever the weights; the identified and unidentified posterior masses are the sums of the atoms'
weights. This reference derives the atoms' values, the mixture and the masses in exact rational arithmetic, and
regenerates the noise stream to confirm its mean and variance.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/derivative_refuter_states/reference.py --check
    uv run --frozen --script conformance/response/derivative_refuter_states/reference.py --self-test

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
ROWS = 800
MASK = (1 << 64) - 1


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(got, want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


def data():
    state = 20_240_607
    draws = []

    def gauss():
        nonlocal state
        total = 0.0
        for _ in range(12):
            state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & MASK
            total += (state >> 11) / 9_007_199_254_740_992.0
        draws.append(total - 6.0)
        return total - 6.0

    a = [2.0 + math.sin(i * 0.71) + 0.2 * math.cos(i * 0.13) for i in range(ROWS)]
    b = [0.7 * gauss() for i in range(ROWS)]
    y = [5.0 + 2.0 * ai - 0.2 * bi + 0.3 * gauss() for ai, bi in zip(a, b)]
    return a, b, y, draws


def exact_ols_slope(a, b, y):
    """Coefficient of A in the exact rational least squares of Y on 1, A, B."""
    cols = [[Fraction(1)] * len(y), [Fraction(x) for x in a], [Fraction(x) for x in b]]
    yy = [Fraction(v) for v in y]
    k = 3
    m = [[sum(cols[p][i] * cols[q][i] for i in range(len(y))) for q in range(k)] + [sum(cols[p][i] * yy[i] for i in range(len(y)))] for p in range(k)]
    for col in range(k):
        pivot = next(r for r in range(col, k) if m[r][col] != 0)
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(k):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * z for x, z in zip(m[r], m[col])]
    return m[1][k] / m[1][1]


EXPECTED_DGP = (
    "A=2+sin(.71*i)+.2*cos(.13*i); B=.7*e; Y=5+2*A-.2*B+.3*e; V=1+.25*A+.3*B+.3*e; "
    "i=0..799; e=sum of 12 uniforms minus 6 from one 64-bit LCG stream (B over all rows, then Y, then V)"
)


def check(document):
    problems = []
    tol = 1e-12
    slope = Fraction(2)
    level = Fraction(5) + slope * Fraction(2)  # E[Y | do(A=2)] with B = 0
    atoms = document["atoms"]
    weights = [Fraction(w).limit_denominator(1000) for w in atoms["weights"]]
    identified = [bool(x) for x in atoms["identified"]]
    mass = sum(w for w, ok in zip(weights, identified) if ok)
    close(float(mass), atoms["identified_mass"], tol, "identified_mass", problems)
    close(float(sum(weights) - mass), atoms["unidentified_mass"], tol, "unidentified_mass", problems)
    values = {
        "point": slope,
        "elasticity": slope * 2 / level,
        "semi_treatment": slope * 2,
        "semi_outcome": slope / level,
        "average": slope,
    }
    for key, value in values.items():
        # the posterior-weighted mean of identical atom values
        mixed = sum(w * value for w, ok in zip(weights, identified) if ok) / mass
        close(float(mixed), document[key], tol, key, problems)
    for key, entries in (("jacobian", [Fraction(2), Fraction(1, 4)]), ("directional", [Fraction(2), Fraction(1, 4)])):
        for i, entry in enumerate(entries):
            mixed = sum(w * entry for w, ok in zip(weights, identified) if ok) / mass
            close(float(mixed), document[key][i], tol, f"{key}[{i}]", problems)
    a, b, y, draws = data()
    mean = sum(draws) / len(draws)
    var = sum((d - mean) ** 2 for d in draws) / (len(draws) - 1)
    if abs(mean) > 0.1 or abs(var - 1.0) > 0.15:
        problems.append(f"noise stream mean {mean:.4f} / variance {var:.4f} is not near 0 / 1")
    ols = float(exact_ols_slope(a, b, y))
    if abs(ols - document["point"]) > 0.08:
        problems.append(f"exact least-squares slope {ols} is not near the structural slope {document['point']}")
    if not document["off_support_at"] > max(a):
        problems.append("the planted off-support point is not beyond the largest treatment value")
    if document["dgp"].replace(" ", "") != EXPECTED_DGP.replace(" ", ""):
        problems.append("declared data-generating law differs from the model this reference derives from")
    return problems


MUTATIONS = [
    ("identified mass counts the cyclic atom", lambda d: d["atoms"].__setitem__("identified_mass", 1.0)),
    ("elasticity shifted", lambda d: d.__setitem__("elasticity", 0.5)),
    ("jacobian entry shifted", lambda d: d["jacobian"].__setitem__(1, 0.5)),
    ("off-support point moved inside", lambda d: d.__setitem__("off_support_at", 3.0)),
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
