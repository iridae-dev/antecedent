# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the class-derivative fixture: structural slopes of a confounded linear law.

The rows i = 0..799 follow, with t = i, latent = cos(.77 t + 2), w = sin(1.7 t + .3) + .8 latent,
r = sin(.29 t + 1), z = cos(.41 t) + .5 w, a = 2 + .6 z + .5 r + sin(.71 t) + .2 cos(.13 t),
y = 5 + 2 a + 3 z + .7 latent + .5 e, v = 1 + .25 a - 1.5 z + .5 e, where each e is the sum of twelve uniforms minus six from one 64-bit LCG stream (state 20240607, multiplier 6364136223846793005, increment 1442695040888963407, uniform = (state >> 11) / 2^53), drawn for y then v on each row.
The observed confounder z separates a from the latent cause of y, so the structural derivative of y
(and of v) in a is the coefficient of a in the least-squares regression on (1, a, z); the level at a = 2
averages z over the rows. This reference solves those normal equations in exact rational arithmetic on the
regenerated rows (the floats are converted exactly), derives the scaled derivatives, and confirms the
unadjusted slope of y on a is confounded.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/class_derivative_states/reference.py --check
    uv run --frozen --script conformance/response/class_derivative_states/reference.py --self-test

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
AT = 2.0


MASK = (1 << 64) - 1


def rows():
    out = []
    state = 20_240_607

    def gauss():
        nonlocal state
        total = 0.0
        for _ in range(12):
            state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & MASK
            total += (state >> 11) / 9_007_199_254_740_992.0
        return total - 6.0

    for i in range(ROWS):
        t = float(i)
        latent = math.cos(t * 0.77 + 2.0)
        w = math.sin(t * 1.7 + 0.3) + 0.8 * latent
        r = math.sin(t * 0.29 + 1.0)
        z = math.cos(t * 0.41) + 0.5 * w
        a = 2.0 + 0.6 * z + 0.5 * r + math.sin(t * 0.71) + 0.2 * math.cos(t * 0.13)
        y = 5.0 + 2.0 * a + 3.0 * z + 0.7 * latent + 0.5 * gauss()
        v = 1.0 + 0.25 * a - 1.5 * z + 0.5 * gauss()
        out.append((a, z, y, v))
    return out


def ols(columns, target):
    """Exact rational least squares of `target` on an intercept and `columns`."""
    n = len(target)
    cols = [[Fraction(1)] * n] + [[Fraction(x) for x in c] for c in columns]
    yy = [Fraction(x) for x in target]
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
                m[r] = [x - f * z for x, z in zip(m[r], m[col])]
    return [m[p][k] / m[p][p] for p in range(k)]


def derive():
    data = rows()
    a = [r[0] for r in data]
    z = [r[1] for r in data]
    y = [r[2] for r in data]
    v = [r[3] for r in data]
    adjusted_y = ols([a, z], y)
    adjusted_v = ols([a, z], v)
    naive = ols([a], y)
    mean_z = sum(Fraction(x) for x in z) / len(z)
    slope = adjusted_y[1]
    level = adjusted_y[0] + slope * Fraction(AT) + adjusted_y[2] * mean_z
    return {
        "slope_y": slope,
        "slope_v": adjusted_v[1],
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
    "latent=cos(.77*i+2); w=sin(1.7*i+.3)+.8*latent; r=sin(.29*i+1); z=cos(.41*i)+.5*w; "
    "a=2+.6*z+.5*r+sin(.71*i)+.2*cos(.13*i); y=5+2*a+3*z+.7*latent+.5*e; "
    "v=1+.25*a-1.5*z+.5*e; i=0..799; e=sum of 12 uniforms minus 6 from one 64-bit LCG stream (y then v per row)"
)


def check(document):
    problems = []
    derived = derive()
    for key, value in derived.items():
        close(value, document[key], 1e-9, key, problems)
    if abs(float(derived["slope_y"]) - 2.0) > 0.05 or abs(float(derived["slope_v"]) - 0.25) > 0.05:
        problems.append("the structural slopes are not near the stated coefficients")
    if abs(float(derived["naive_y"] - derived["slope_y"])) < 0.5:
        problems.append("the unadjusted slope is not confounded")
    if document["at"] != AT:
        problems.append("evaluation point differs from the one this reference uses")
    if not document["off_support_at"] > max(r[0] for r in rows()):
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
