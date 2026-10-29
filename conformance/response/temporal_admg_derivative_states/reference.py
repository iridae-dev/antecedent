# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the temporal-ADMG derivative fixture: the structural slope of a confounded lag-one law.

The series t = 0..599 follows latent = cos(.77 t + 2), w = sin(1.7 t + .3) + .8 latent, z = cos(.41 t) + .5 w,
x = 2 + .6 z + sin(.71 t) + .2 cos(.13 t) and, for t >= 1, y_t = 5 + 2 x_{t-1} + 3 z_{t-1} + .7 latent_{t-1} + .5 e_t
(y_0 = 0), where each e is the sum of twelve uniforms minus six drawn from one 64-bit linear congruential stream
(state 20240607, multiplier 6364136223846793005, increment 1442695040888963407, uniform = (state >> 11) / 2^53),
one draw per t >= 1. The observed confounder z(t-1) separates x(t-1) from the latent cause of w(t-1) and y(t), so the
structural slope of y(t) in x(t-1) is the coefficient of x(t-1) in the least-squares regression of y(t) on
(1, x(t-1), z(t-1), w(t-1)); the level at x = 2 averages z and w over the rows. This reference solves those normal
equations in exact rational arithmetic on the regenerated rows (the floats are converted exactly), derives the
scaled derivatives, and confirms the unadjusted lag-one slope is confounded.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/temporal_admg_derivative_states/reference.py --check
    uv run --frozen --script conformance/response/temporal_admg_derivative_states/reference.py --self-test

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
AT = 2.0
MASK = (1 << 64) - 1


def columns():
    state = 20_240_607

    def gauss():
        nonlocal state
        total = 0.0
        for _ in range(12):
            state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & MASK
            total += (state >> 11) / 9_007_199_254_740_992.0
        return total - 6.0

    latent, w, z, x, y = ([0.0] * ROWS for _ in range(5))
    for t in range(ROWS):
        c = float(t)
        latent[t] = math.cos(c * 0.77 + 2.0)
        w[t] = math.sin(c * 1.7 + 0.3) + 0.8 * latent[t]
        z[t] = math.cos(c * 0.41) + 0.5 * w[t]
        x[t] = 2.0 + 0.6 * z[t] + math.sin(c * 0.71) + 0.2 * math.cos(c * 0.13)
    for t in range(1, ROWS):
        y[t] = 5.0 + 2.0 * x[t - 1] + 3.0 * z[t - 1] + 0.7 * latent[t - 1] + 0.5 * gauss()
    return x, y, z, w


def ols(columns_, target):
    """Exact rational least squares of `target` on an intercept and `columns_`."""
    n = len(target)
    cols = [[Fraction(1)] * n] + [[Fraction(v) for v in c] for c in columns_]
    yy = [Fraction(v) for v in target]
    k = len(cols)
    m = [
        [sum(cols[p][i] * cols[q][i] for i in range(n)) for q in range(k)]
        + [sum(cols[p][i] * yy[i] for i in range(n))]
        for p in range(k)
    ]
    for col in range(k):
        pivot = next(r for r in range(col, k) if m[r][col] != 0)
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(k):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [u - f * v for u, v in zip(m[r], m[col])]
    return [m[p][k] / m[p][p] for p in range(k)]


def derive():
    x, y, z, w = columns()
    lag = lambda c: c[: ROWS - 1]
    target = y[1:]
    adjusted = ols([lag(x), lag(z), lag(w)], target)
    naive = ols([lag(x)], target)
    mean_z = sum(Fraction(v) for v in lag(z)) / (ROWS - 1)
    mean_w = sum(Fraction(v) for v in lag(w)) / (ROWS - 1)
    slope = adjusted[1]
    level = adjusted[0] + slope * Fraction(AT) + adjusted[2] * mean_z + adjusted[3] * mean_w
    return {
        "slope_y": slope,
        "level_y": level,
        "naive_y": naive[1],
        "elasticity": slope * Fraction(AT) / level,
        "semi_treatment": slope * Fraction(AT),
        "semi_outcome": slope / level,
    }


def close(got, want, tol, label, problems):
    if want is None or not math.isclose(float(got), want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got)!r} != fixture {want!r} (tol {tol})")


EXPECTED_DGP = (
    "latent=cos(.77*t+2); w=sin(1.7*t+.3)+.8*latent; z=cos(.41*t)+.5*w; x=2+.6*z+sin(.71*t)+.2*cos(.13*t); "
    "y[t]=5+2*x[t-1]+3*z[t-1]+.7*latent[t-1]+.5*e (y[0]=0); t=0..599; "
    "e=sum of 12 uniforms minus 6 from one 64-bit LCG stream (one draw per t>=1)"
)


def check(document):
    problems = []
    derived = derive()
    for key, value in derived.items():
        close(value, document[key], 1e-9, key, problems)
    if abs(float(derived["slope_y"]) - 2.0) > 0.05:
        problems.append("the structural slopes are not near the stated coefficients")
    if abs(float(derived["naive_y"] - derived["slope_y"])) < 0.5:
        problems.append("the unadjusted slope is not confounded")
    if document["at"] != AT:
        problems.append("evaluation point differs from the one this reference uses")
    if not document["off_support_at"] > max(columns()[0]):
        problems.append("the planted off-support point is not beyond the largest treatment value")
    if document["dgp"].replace(" ", "") != EXPECTED_DGP.replace(" ", ""):
        problems.append("declared data-generating law differs from the model this reference derives from")
    return problems


MUTATIONS = [
    ("slope replaced by the structural coefficient", lambda d: d.__setitem__("slope_y", 2.0)),
    ("confounded slope used", lambda d: d.__setitem__("slope_y", d["naive_y"])),
    ("elasticity shifted", lambda d: d.__setitem__("elasticity", d["elasticity"] + 0.01)),
    ("off-support point moved inside", lambda d: d.__setitem__("off_support_at", 2.0)),
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
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        d = {k: float(v) for k, v in derive().items()}
        print(json.dumps(d, indent=2))
        return 0
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
