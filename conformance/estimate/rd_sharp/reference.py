# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent reference for the sharp-RD fixture: the structural jump and the analytic standard errors.

The fixture's point pin (jump 3.0 within 0.5) is the structural jump coefficient of the seeded SCM
generated in Rust. The `se_reference` block is a small deterministic heteroskedastic design, frozen
row by row, with the local-linear jump and its analytic SEs computed here from the textbook formulas,
with exact rational linear algebra on the double-precision inputs:

    X = [1, T, R - c, T (R - c)] over rows with |R - c| <= h,  T = 1[R >= c]
    beta = (X'X)^-1 X'y,  e = y - X beta,  n rows, p = 4 columns
    homoskedastic: Var = e'e / (n - p) * (X'X)^-1
    HC1:           Var = n / (n - p) * (X'X)^-1 X' diag(e^2) X (X'X)^-1

The outcome noise scale grows with |R - c|, so the two SEs differ. `rd.sharp` reports HC1 by default
and the homoskedastic SE on explicit opt-in; `crates/antecedent/tests/estimate_conformance.rs` pins
both against this block.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/rd_sharp/reference.py --check
    uv run --frozen --script conformance/estimate/rd_sharp/reference.py --self-test

`--write` regenerates the numeric `se_reference` block only and keeps the recorded `oracle` block,
including its review, so a rewrite cannot erase it; `--check` compares the block and the structural
jump against the fixture and never writes.

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
FIXTURE = HERE / "expected.json"
N = 48
CUTOFF = 0.0
BANDWIDTH = 0.75
STRUCTURAL_JUMP = 3.0   # outcome = 2 + 0.5 r + 3 t - 0.8 t r + noise: the coefficient on t is the jump at the cutoff
RELATIVE_TOLERANCE = 1e-9


def rows() -> tuple[list[float], list[float]]:
    running = [-1.0 + 2.0 * (i + 0.5) / N for i in range(N)]
    outcome = []
    for i, r in enumerate(running):
        t = 1.0 if r >= CUTOFF else 0.0
        scale = 0.1 + 0.6 * abs(r - CUTOFF)
        noise = scale * math.sin(1.7 * i + 0.3)
        outcome.append(2.0 + 0.5 * r + STRUCTURAL_JUMP * t - 0.8 * t * r + noise)
    return running, outcome


def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def inverse(a):
    n = len(a)
    m = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [row[n:] for row in m]


def standard_errors(running: list[float], outcome: list[float]) -> dict[str, float]:
    keep = [i for i, r in enumerate(running) if abs(r - CUTOFF) <= BANDWIDTH]
    x = []
    for i in keep:
        centered = Fraction(running[i]) - Fraction(CUTOFF)
        t = Fraction(1 if centered >= 0 else 0)
        x.append([Fraction(1), t, centered, t * centered])
    y = [Fraction(outcome[i]) for i in keep]
    n, p = len(x), 4
    xt = [list(col) for col in zip(*x)]
    bread = inverse(matmul(xt, x))
    xty = [sum(a * b for a, b in zip(col, y)) for col in xt]
    beta = [sum(b * v for b, v in zip(row, xty)) for row in bread]
    e = [yi - sum(b * v for b, v in zip(beta, xi)) for xi, yi in zip(x, y)]
    sse = sum(v * v for v in e)
    var_hom = sse / (n - p) * bread[1][1]
    meat = [[sum(xi[a] * xi[b] * ei * ei for xi, ei in zip(x, e)) for b in range(p)] for a in range(p)]
    sandwich = matmul(matmul(bread, meat), bread)
    var_hc1 = Fraction(n, n - p) * sandwich[1][1]
    return {
        "window_rows": n,
        "jump": float(beta[1]),
        "se_hc1": math.sqrt(var_hc1),
        "se_homoskedastic": math.sqrt(var_hom),
    }


def build_block() -> dict:
    running, outcome = rows()
    return {
        "cutoff": CUTOFF,
        "bandwidth": BANDWIDTH,
        "running": running,
        "outcome": outcome,
        "expected": standard_errors(running, outcome),
        "relative_tolerance": RELATIVE_TOLERANCE,
    }


def compare_values(left: object, right: object, rtol: float, atol: float) -> bool:
    """Recursive comparison with an explicit relative tolerance (standard library only)."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(compare_values(left[k], right[k], rtol, atol) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(compare_values(a, b, rtol, atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return math.isclose(left, right, rel_tol=rtol, abs_tol=atol)
    return left == right


def values_close(left: object, right: object) -> bool:
    """Compare with the fixture's explicit tolerance."""
    return compare_values(left, right, rtol=1e-9, atol=0.0)


def check(document: dict) -> list[str]:
    problems = []
    if document.get("true_effect") != STRUCTURAL_JUMP:
        problems.append(f"true_effect {document.get('true_effect')!r} != structural jump {STRUCTURAL_JUMP}")
    block = build_block()
    if not values_close(document.get("se_reference"), block):
        problems.append("se_reference differs from the derived block (rerun with --write after reviewing)")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    if check(document):
        failures.append("the unmodified fixture is rejected")
    mutations = [
        ("true effect changed", lambda d: d.__setitem__("true_effect", 2.0)),
        ("hc1 se changed", lambda d: d["se_reference"]["expected"].__setitem__("se_hc1", 0.2)),
        ("homoskedastic se changed", lambda d: d["se_reference"]["expected"].__setitem__("se_homoskedastic", 0.12)),
        ("one outcome changed", lambda d: d["se_reference"]["outcome"].__setitem__(10, 0.0)),
    ]
    for label, mutate in mutations:
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"accepted: {label}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--write", action="store_true", help="regenerate se_reference, keeping the oracle block")
    args = parser.parse_args()
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if args.write:
        document["se_reference"] = build_block()
        FIXTURE.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(document["se_reference"]["expected"], indent=2))
        return 0
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
