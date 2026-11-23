# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the response-curve refuter-state fixture.

The data follow the linear structural law Z = sin(i / 17), T = .6 Z + .8 cos(1.13 i) + .4 e,
Y = 1 + 2 T + .8 Z + .3 e for i = 0..599, where each e is the sum of twelve uniforms minus six drawn
from one 64-bit linear congruential stream (state 20240611, multiplier 6364136223846793005, increment
1442695040888963407, uniform = (state >> 11) / 2^53), consumed for T over every row, then Y. The
interventional mean at level x is 1 + 2 x + .8 mean(Z) (the back-door average of the structural mean
over the observed confounder). This reference derives that curve at each stated grid point in exact
rational arithmetic, regenerates the noise stream to confirm it has the stated mean and variance,
solves the exact rational normal equations of Y on 1, T, Z on the regenerated rows to confirm the
finite-sample coefficients lie near the structural ones (so the curve the route estimates lies within
the declared band of the closed form), confirms the unadjusted slope is confounded away from 2, and
confirms the planted off-support grid point lies beyond the largest treatment value.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/curve_refuter_states/reference.py --check
    uv run --frozen --script conformance/response/curve_refuter_states/reference.py --self-test

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
ROWS = 600
MASK = (1 << 64) - 1

EXPECTED_DGP = (
    "Z=sin(i/17); T=.6*Z+.8*cos(1.13*i)+.4*e; Y=1+2*T+.8*Z+.3*e; i=0..599; "
    "e=sum of 12 uniforms minus 6 from one 64-bit LCG stream (state 20240611; T's noise over all rows, then Y's)"
)


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(got, want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


def data():
    state = 20_240_611
    draws = []

    def gauss():
        nonlocal state
        total = 0.0
        for _ in range(12):
            state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & MASK
            total += (state >> 11) / 9_007_199_254_740_992.0
        draws.append(total - 6.0)
        return total - 6.0

    z = [math.sin(i / 17.0) for i in range(ROWS)]
    t = [0.6 * z[i] + 0.8 * math.cos(i * 1.13) + 0.4 * gauss() for i in range(ROWS)]
    y = [1.0 + 2.0 * t[i] + 0.8 * z[i] + 0.3 * gauss() for i in range(ROWS)]
    return z, t, y, draws


def exact_ols(columns, y):
    """Coefficients of the exact rational least squares of y on the given columns."""
    cols = [[Fraction(x) for x in column] for column in columns]
    yy = [Fraction(v) for v in y]
    k = len(cols)
    m = [
        [sum(cols[p][i] * cols[q][i] for i in range(len(y))) for q in range(k)]
        + [sum(cols[p][i] * yy[i] for i in range(len(y)))]
        for p in range(k)
    ]
    for col in range(k):
        pivot = next(r for r in range(col, k) if m[r][col] != 0)
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(k):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * w for x, w in zip(m[r], m[col], strict=False)]
    return [m[p][k] / m[p][p] for p in range(k)]


def check(document):
    problems = []
    z, t, y, draws = data()
    mean_z = Fraction(sum(Fraction(v) for v in z), ROWS)
    one = [1.0] * ROWS
    tol = 1e-12
    if document["dgp"].replace(" ", "") != EXPECTED_DGP.replace(" ", ""):
        problems.append("declared data-generating law differs from the model this reference derives from")
    if document["rows"] != ROWS:
        problems.append("declared row count differs from the model this reference derives from")
    close(document["intercept"], 1.0, tol, "intercept", problems)
    close(document["treatment_coefficient"], 2.0, tol, "treatment_coefficient", problems)
    close(document["confounder_coefficient"], 0.8, tol, "confounder_coefficient", problems)
    truth = [1 + 2 * Fraction(x) + Fraction(4, 5) * mean_z for x in document["grid"]]
    mean = sum(draws) / len(draws)
    var = sum((d - mean) ** 2 for d in draws) / (len(draws) - 1)
    if abs(mean) > 0.08 or abs(var - 1.0) > 0.12:
        problems.append(f"noise stream mean {mean:.4f} / variance {var:.4f} is not near 0 / 1")
    b0, bt, bz = (float(v) for v in exact_ols([one, t, z], y))
    for x, want in zip(document["grid"], truth, strict=False):
        # The fitted adjusted curve b0 + bt x + bz mean(Z) lies near the closed form.
        fitted = b0 + bt * x + bz * float(mean_z)
        if abs(fitted - float(want)) > document["curve_absolute_tolerance"] / 3:
            problems.append(f"exact least-squares curve {fitted} is not near the closed form {float(want)} at {x}")
    unadjusted = float(exact_ols([one, t], y)[1])
    if abs(unadjusted - 2.0) < 0.15:
        problems.append(f"unadjusted slope {unadjusted} is not separated from the structural slope 2")
    if not document["off_support_at"] > max(t):
        problems.append("the planted off-support grid point is not beyond the largest treatment value")
    if not all(min(t) < x < max(t) for x in document["grid"]):
        problems.append("a clean grid point lies outside the observed treatment range")
    return problems


MUTATIONS = [
    ("intercept shifted", lambda d: d.__setitem__("intercept", 1.5)),
    ("treatment coefficient shifted", lambda d: d.__setitem__("treatment_coefficient", 2.5)),
    ("off-support point moved inside", lambda d: d.__setitem__("off_support_at", 0.5)),
    ("grid moved outside", lambda d: d.__setitem__("grid", [-0.8, 0.0, 9.0])),
    ("row count changed", lambda d: d.__setitem__("rows", 599)),
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
