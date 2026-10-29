# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the temporal derivative refuter-state fixture.

The series follows the lag-one linear law X_t = 1 + .8 sin(.37 t) + .3 e, B_t = 1 if a draw e is positive, else 0,
Y_t = 3 + 2 X_{t-1} + .5 B_{t-1} + .3 e, V_t = 1 + X_{t-1} - B_{t-1} + .3 e for t = 0..399 (Y and V are zero at t = 0),
where each e is the sum of twelve uniforms minus six drawn from one 64-bit linear congruential stream (state
20240607, multiplier 6364136223846793005, increment 1442695040888963407, uniform = (state >> 11) / 2^53),
consumed in the order B_t, X_t, then (for t > 0) Y_t, then V_t. The structural slope of Y_t on X_{t-1} is the coefficient 2
whatever the noise; the level E[Y | do(X_{t-1} = x)] = 3.25 + 2 x averages B over its alternation, so at x = 1 it is
5.25. This reference derives the point derivative, elasticity, semi-elasticities, average derivative, Jacobian
and directional derivative in exact rational arithmetic, regenerates the noise stream to confirm its mean and
variance, solves the exact rational normal equations of Y_t on (1, X_{t-1}, B_{t-1}) on the regenerated rows to
confirm the finite-sample slope lies near the coefficient, and confirms the planted off-support point lies beyond
the largest X.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/temporal_derivative_refuter_states/reference.py --check
    uv run --frozen --script conformance/estimate/temporal_derivative_refuter_states/reference.py --self-test

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
ROWS = 400
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

    x = [0.0] * ROWS
    y = [0.0] * ROWS
    b = [0.0] * ROWS
    for t in range(ROWS):
        b[t] = 1.0 if gauss() > 0.0 else 0.0
        x[t] = 1.0 + 0.8 * math.sin(t * 0.37) + 0.3 * gauss()
        if t > 0:
            y[t] = 3.0 + 2.0 * x[t - 1] + 0.5 * b[t - 1] + 0.3 * gauss()
            gauss()  # V_t
    return x, b, y, draws


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


def check(document):
    problems = []
    tol = 1e-12
    slope = Fraction(2)
    at = Fraction(3) + slope * Fraction(1) + Fraction(1, 4)  # level E[Y | do(X=1)], B averaged (P(B=1) = 1/2)
    derived = {
        "point": slope,
        "elasticity": slope * 1 / at,
        "semi_treatment": slope * 1,
        "semi_outcome": slope / at,
        "average": slope,
    }
    for key, value in derived.items():
        close(float(value), document[key], tol, key, problems)
    for i, (got, want) in enumerate(zip([Fraction(2), Fraction(1, 2), Fraction(1), Fraction(-1)], document["jacobian"])):
        close(float(got), want, tol, f"jacobian[{i}]", problems)
    close(float(Fraction(2) * 1 + Fraction(1, 2) * 2), document["directional"][0], tol, "directional[0]", problems)
    close(float(Fraction(1) + Fraction(-1) * 2), document["directional"][1], tol, "directional[1]", problems)
    a, b, y, draws = data()
    a, b, y = a[:-1], b[:-1], y[1:]  # lag one: Y_t on (X_{t-1}, B_{t-1})
    mean = sum(draws) / len(draws)
    var = sum((d - mean) ** 2 for d in draws) / (len(draws) - 1)
    if abs(mean) > 0.08 or abs(var - 1.0) > 0.12:
        problems.append(f"noise stream mean {mean:.4f} / variance {var:.4f} is not near 0 / 1")
    ols = float(exact_ols_slope(a, b, y))
    if abs(ols - document["point"]) > 0.05:
        problems.append(f"exact least-squares slope {ols} is not near the structural slope {document['point']}")
    if not document["off_support_at"] > max(a):
        problems.append("the planted off-support point is not beyond the largest treatment value")
    if not min(a) < document["at"] < max(a):
        problems.append("the clean evaluation point is not inside the observed treatment range")
    if document["dgp"].replace(" ", "") != EXPECTED_DGP.replace(" ", ""):
        problems.append("declared data-generating law differs from the model this reference derives from")
    return problems


EXPECTED_DGP = (
    "X=1+.8*sin(.37*t)+.3*e; B=1 if e>0 else 0; Y=3+2*X[t-1]+.5*B[t-1]+.3*e; V=1+X[t-1]-B[t-1]+.3*e; t=0..399; "
    "e=sum of 12 uniforms minus 6 from one 64-bit LCG stream (B_t, X_t, then Y_t, then V_t for t>0)"
)

MUTATIONS = [
    ("elasticity shifted", lambda d: d.__setitem__("elasticity", 0.5)),
    ("semi-treatment shifted", lambda d: d.__setitem__("semi_treatment", 3.0)),
    ("jacobian entry shifted", lambda d: d["jacobian"].__setitem__(1, -0.4)),
    ("off-support point moved inside", lambda d: d.__setitem__("off_support_at", 1.5)),
    ("evaluation point moved outside", lambda d: d.__setitem__("at", 5.0)),
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
