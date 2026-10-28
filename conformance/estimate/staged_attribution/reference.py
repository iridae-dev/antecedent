# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the staged attribution fixture: on y = 2x with the last of 20 rows replaced by 200, the planted row has by far the largest residual under both the true mechanism and the exact least-squares fit; and with the Y intercept moving from 1 to 6 at identical x, the mean change between the two 40-row periods is exactly 5.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/staged_attribution/reference.py --check
    uv run --frozen --script conformance/estimate/staged_attribution/reference.py --self-test

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
    n = 20
    x = list(range(n))
    y = [200 if i == n - 1 else 2 * i for i in range(n)]
    true_resid = [abs(F(y[i]) - 2 * x[i]) for i in range(n)]
    top_true = max(range(n), key=lambda i: true_resid[i])
    b0, b1 = ols([[1] * n, x], y)
    fit_resid = [abs(F(y[i]) - b0 - b1 * x[i]) for i in range(n)]
    top_fit = max(range(n), key=lambda i: fit_resid[i])
    same(top_true, document["anomaly"]["top_row"], "top_row under the true mechanism", problems)
    same(top_fit, document["anomaly"]["top_row"], "top_row under the least-squares fit", problems)
    ordered = sorted(true_resid, reverse=True)
    if not (ordered[0] > 0 and ordered[1] == 0):
        problems.append("the planted row is not the only row off the true mechanism")
    # two-period change: same x in both periods, intercept 1 -> 6, slope 2
    xs = [F(i % 40, 10) for i in range(80)]
    ys = [(1 if i < 40 else 6) + 2 * xs[i] for i in range(80)]
    first = sum(ys[:40]) / 40
    second = sum(ys[40:]) / 40
    close(second - first, document["change"]["total_change"], document["change"]["tolerance"], "total_change", problems)
    close(second - first, 5, 1e-12, "exact mean change", problems)
    if not second - first >= F(str(document["change"]["total_change_min"])):
        problems.append("total_change_min exceeds the derived change")
    return problems


MUTATIONS = [
    ("top row changed", lambda d: d["anomaly"].__setitem__("top_row", 0)),
    ("total change changed", lambda d: d["change"].__setitem__("total_change", 8.0)),
    ("lower bound above the change", lambda d: d["change"].__setitem__("total_change_min", 6.0)),
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
