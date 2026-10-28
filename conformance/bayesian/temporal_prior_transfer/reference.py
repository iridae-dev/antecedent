# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the temporal prior-transfer effect: on the noise-free stationary law Y_s = 0.9 X_{s-1} with X_s = sin(0.04 s) the exact least-squares coefficient on the lagged treatment is 0.9, so the pulse, the single-step sustained shift and the two-point response-curve contrast between the declared grid doses are all 0.9.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/temporal_prior_transfer/reference.py --check
    uv run --frozen --script conformance/bayesian/temporal_prior_transfer/reference.py --self-test

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
F = Fraction
FIXTURE = HERE / "expected.json"


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got) if got is not None else got!r} != fixture {want!r} (tol {tol})")


def same(got, want, label, problems):
    if got != want:
        problems.append(f"{label}: derived {got!r} != fixture {want!r}")

def solve(a, b):
    """Exact Gauss-Jordan solve of a x = b over Fractions."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def ols(columns, y):
    """Least squares of y on the given columns (include a constant column yourself), exact on float inputs."""
    cols = [[F(v) for v in c] for c in columns]
    yy = [F(v) for v in y]
    xtx = [[sum(a * b for a, b in zip(ci, cj)) for cj in cols] for ci in cols]
    xty = [sum(a * b for a, b in zip(ci, yy)) for ci in cols]
    return solve(xtx, xty)


def check(document):
    problems = []
    n = document["n"]
    x = [math.sin(0.04 * s) for s in range(n)]
    y = [None] + [0.9 * x[s - 1] for s in range(1, n)]
    rows = range(1, n)
    beta = ols([[1] * (n - 1), [x[s - 1] for s in rows]], [y[s] for s in rows])
    close(beta[1], document["true_effect"], 1e-12, "true_effect (pulse and single-step sustained)", problems)
    grid = document["grid"]
    close(beta[1] * (F(str(grid[-1])) - F(str(grid[0]))), document["true_effect"], 1e-12, "grid contrast", problems)
    if document["horizons"] != [1] or document["treatment_lag"] != 1 or document["horizon_steps"] != 1:
        problems.append("the derivation covers horizon one with a lag-1 treatment only")
    if not document["atol"] > 0:
        problems.append("atol must be positive")
    return problems


MUTATIONS = [
    ("effect changed", lambda d: d.__setitem__("true_effect", 1.0)),
    ("grid widened", lambda d: d.__setitem__("grid", [0.0, 2.0])),
    ("horizon changed", lambda d: d.__setitem__("horizons", [2])),
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
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
