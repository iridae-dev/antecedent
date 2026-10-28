# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the frequentist temporal class-envelope pulse: the equal-weight mean of the z-adjusting completion (exactly 2) and the completion without z (2 + 0.5 Cov(z,t)/Var(t)) on the 800-row noise-free series.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/temporal_class_envelope/reference.py --check
    uv run --frozen --script conformance/estimate/temporal_class_envelope/reference.py --self-test

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
    z = [i % 2 for i in range(n)]
    t = [0.3 + 0.4 * z[i] + 0.05 * math.sin(0.017 * i) for i in range(n)]
    y = [None] + [1 + 2 * t[i - 1] + 0.5 * z[i - 1] for i in range(1, n)]
    rows = range(1, n)
    one = [1] * (n - 1)
    adjusted = ols([one, [t[i - 1] for i in rows], [z[i - 1] for i in rows]], [y[i] for i in rows])[1]
    unadjusted = ols([one, [t[i - 1] for i in rows]], [y[i] for i in rows])[1]
    ate = (adjusted + unadjusted) / 2
    close(ate, document["cpdag"]["pulse"]["ate"], max(document["cpdag"]["pulse"]["absolute_tolerance"], 1e-8), "cpdag.pulse.ate", problems)
    close(ate, document["pag"]["withdrawn_numeric_outputs"]["pulse"]["ate"], 1e-8, "withdrawn pag pulse (same number)", problems)
    close(adjusted, 2, 1e-9, "adjusting completion", problems)
    if document["cpdag"]["identification"]["completion_count"] != 2:
        problems.append("expected two CPDAG completions")
    return problems


MUTATIONS = [
    ("pulse changed", lambda d: d["cpdag"]["pulse"].__setitem__("ate", 2.6)),
    ("withdrawn pulse changed", lambda d: d["pag"]["withdrawn_numeric_outputs"]["pulse"].__setitem__("ate", 2.0)),
    ("completion count changed", lambda d: d["cpdag"]["identification"].__setitem__("completion_count", 3)),
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
