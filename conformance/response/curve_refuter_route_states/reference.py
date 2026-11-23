# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the response-curve route refuter-state fixture.

Three linear laws, each driven by one 64-bit linear congruential stream (multiplier
6364136223846793005, increment 1442695040888963407, uniform = (state >> 11) / 2^53, gauss = sum of
twelve uniforms minus six):

* lag series (seed 20240701, 1200 rows): x_t = 4 u - 2 for every t, then y_t = 1.2 + 1.8 x_{t-1} + .3 e
  for t >= 1; the interventional mean at level x is 1.2 + 1.8 x.
* class series (seed 20240702, 600 rows): z_t = sin(.37 t), x_t = cos(.61 t) + .3 e for every t, then
  y_t = 1.4 x_{t-1} + .6 z_{t-1} + .3 e for t >= 1; the interventional mean is 1.4 x + .6 mean(z_{t-1}).
* static table (seed 20240704, 800 rows): z_i = sin(.37 i), t_i = .5 z_i + .8 sin(.61 i) + .3 e for every
  i, then y_i = 1 + 2 t_i + .3 e; the interventional mean is 1 + 2 x.

This reference regenerates each stream, solves the exact rational normal equations of the outcome on
its regressors to confirm the finite-sample coefficients lie within the declared band of the
structural ones (so the curve a route estimates lies within the band of the closed form), and
confirms the planted off-support grid point lies beyond the largest treatment value.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/curve_refuter_route_states/reference.py --check
    uv run --frozen --script conformance/response/curve_refuter_route_states/reference.py --self-test

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
MASK = (1 << 64) - 1


class Stream:
    def __init__(self, seed):
        self.state = seed

    def uniform(self):
        self.state = (self.state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & MASK
        return (self.state >> 11) / 9_007_199_254_740_992.0

    def gauss(self):
        return sum(self.uniform() for _ in range(12)) - 6.0


def exact_ols(columns, y):
    cols = [[Fraction(v) for v in column] for column in columns]
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
                m[r] = [a - f * b for a, b in zip(m[r], m[col], strict=False)]
    return [float(m[p][k] / m[p][p]) for p in range(k)]


def lag_series(pin):
    n = pin["rows"]
    s = Stream(pin["seed"])
    x = [4.0 * s.uniform() - 2.0 for _ in range(n)]
    y = [0.0] + [1.2 + 1.8 * x[i - 1] + 0.3 * s.gauss() for i in range(1, n)]
    return x, y


def panel_units(pin):
    s = Stream(pin["seed"])
    units = []
    for _ in range(pin["units"]):
        n = pin["rows"]
        x = [4.0 * s.uniform() - 2.0 for _ in range(n)]
        y = [0.0] + [1.2 + 1.8 * x[i - 1] + 0.3 * s.gauss() for i in range(1, n)]
        units.append((x, y))
    return units


def class_series(pin):
    n = pin["rows"]
    s = Stream(pin["seed"])
    z = [math.sin(i * 0.37) for i in range(n)]
    x = [math.cos(i * 0.61) + 0.3 * s.gauss() for i in range(n)]
    y = [0.0] + [1.4 * x[i - 1] + 0.6 * z[i - 1] + 0.3 * s.gauss() for i in range(1, n)]
    return z, x, y


def static_table(pin):
    n = pin["rows"]
    s = Stream(pin["seed"])
    z = [math.sin(i * 0.37) for i in range(n)]
    t = [0.5 * z[i] + 0.8 * math.sin(i * 0.61) + 0.3 * s.gauss() for i in range(n)]
    y = [1.0 + 2.0 * v + 0.3 * s.gauss() for v in t]
    return z, t, y


def check(document):
    problems = []
    band = document["curve_absolute_tolerance"]

    def near(got, want, label):
        if abs(got - want) > band / 3:
            problems.append(f"{label}: exact least squares {got} is not near the structural {want}")

    pin = document["lag_series"]
    x, y = lag_series(pin)
    one = [1.0] * (len(x) - 1)
    b0, b1 = exact_ols([one, x[:-1]], y[1:])
    for g in pin["grid"]:
        near(b0 + b1 * g, 1.2 + 1.8 * g, f"lag series at {g}")
    if not pin["off_support_at"] > max(x):
        problems.append("lag series: the planted off-support point is not beyond the largest treatment")
    if not all(min(x) < g < max(x) for g in pin["grid"]):
        problems.append("lag series: a clean grid point lies outside the treatment range")

    pin = document["panel"]
    xs, ys, ones = [], [], []
    for x, y in panel_units(pin):
        xs += x[:-1]
        ys += y[1:]
        ones += [1.0] * (len(x) - 1)
    b0, b1 = exact_ols([ones, xs], ys)
    for g in pin["grid"]:
        near(b0 + b1 * g, 1.2 + 1.8 * g, f"panel at {g}")
    if not all(min(xs) < g < max(xs) for g in pin["grid"]):
        problems.append("panel: a clean grid point lies outside the treatment range")

    pin = document["class_panel"]
    s = Stream(pin["seed"])
    n = pin["rows"]
    z = [math.sin(i * 0.37) for i in range(n)]
    cols = [[], [], []]
    ys = []
    allx = []
    for _ in range(pin["units"]):
        x = [math.cos(i * 0.61) + 0.3 * s.gauss() for i in range(n)]
        y = [0.0] + [1.4 * x[i - 1] + 0.6 * z[i - 1] + 0.3 * s.gauss() for i in range(1, n)]
        cols[0] += [1.0] * (n - 1)
        cols[1] += x[:-1]
        cols[2] += z[:-1]
        ys += y[1:]
        allx += x
    b0, b1, b2 = exact_ols(cols, ys)
    mean_z = sum(z[:-1]) / (n - 1)
    for g in pin["grid"]:
        near(b0 + b1 * g + b2 * mean_z, 1.4 * g + 0.6 * mean_z, f"class panel at {g}")
    if not all(min(allx) < g < max(allx) for g in pin["grid"]):
        problems.append("class panel: a clean grid point lies outside the treatment range")

    pin = document["class_series"]
    z, x, y = class_series(pin)
    one = [1.0] * (len(x) - 1)
    b0, b1, b2 = exact_ols([one, x[:-1], z[:-1]], y[1:])
    mean_z = sum(z[:-1]) / (len(z) - 1)
    for g in pin["grid"]:
        near(b0 + b1 * g + b2 * mean_z, 1.4 * g + 0.6 * mean_z, f"class series at {g}")
    if not all(min(x) < g < max(x) for g in pin["grid"]):
        problems.append("class series: a clean grid point lies outside the treatment range")

    pin = document["static_table"]
    z, t, y = static_table(pin)
    one = [1.0] * len(t)
    b0, b1, b2 = exact_ols([one, t, z], y)
    mean_z = sum(z) / len(z)
    for g in pin["grid"]:
        near(b0 + b1 * g + b2 * mean_z, 1.0 + 2.0 * g, f"static table at {g}")
    if not all(min(t) < g < max(t) for g in pin["grid"]):
        problems.append("static table: a clean grid point lies outside the treatment range")
    return problems


MUTATIONS = [
    ("lag series seed changed to a degenerate grid", lambda d: d["lag_series"].__setitem__("grid", [-1.0, 0.0, 9.0])),
    ("off-support point moved inside", lambda d: d["lag_series"].__setitem__("off_support_at", 0.5)),
    ("class grid moved outside", lambda d: d["class_series"].__setitem__("grid", [0.0, 9.0])),
    ("static grid moved outside", lambda d: d["static_table"].__setitem__("grid", [-0.5, 9.0])),
    ("panel grid moved outside", lambda d: d["panel"].__setitem__("grid", [-1.0, 0.0, 9.0])),
    ("class panel grid moved outside", lambda d: d["class_panel"].__setitem__("grid", [0.0, 9.0])),
    ("tolerance tightened to nothing", lambda d: d.__setitem__("curve_absolute_tolerance", 1e-9)),
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
