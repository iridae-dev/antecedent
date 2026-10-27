# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Retained reference for the finite-sample F(1, n-2) critical values.

The F(1, d) 0.95 critical value is the square of the Student-t(d) 0.975 quantile.  Two
different stdlib-only computations are carried out and must agree before any comparison:

1. composite Simpson integration (4096 even panels) of the Student t density from 0 to a
   candidate quantile with 80 bisections on [0, 4], using math.lgamma for the density
   normalization (the method named in the fixture README);
2. the regularized incomplete beta function by Lentz's continued fraction,
   P(|T| <= x) = I_{x^2/(d+x^2)}(1/2, d/2), solved for the 0.95 two-sided probability by
   bisection on x^2.

No Antecedent code.  The structural effect and instrument strengths in the fixture are
declared data-generating inputs and are only checked for shape.

    uv run --frozen --script conformance/recalculation/design_ar/reference.py --check
    uv run --frozen --script conformance/recalculation/design_ar/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATOL = 1e-8


def t_density(x: float, d: int) -> float:
    log_c = math.lgamma((d + 1) / 2) - math.lgamma(d / 2) - 0.5 * math.log(d * math.pi)
    return math.exp(log_c - (d + 1) / 2 * math.log1p(x * x / d))


def simpson_cdf_half(x: float, d: int, panels: int = 4096) -> float:
    h = x / panels
    total = t_density(0.0, d) + t_density(x, d)
    for i in range(1, panels):
        total += (4 if i % 2 else 2) * t_density(i * h, d)
    return total * h / 3


def quantile_simpson(d: int, level: float = 0.975) -> float:
    lo, hi = 0.0, 4.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 + simpson_cdf_half(mid, d) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h


def reg_inc_beta(a: float, b: float, x: float) -> float:
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * betacf(a, b, x) / a
    return 1 - front * betacf(b, a, 1 - x) / b


def critical_beta(d: int, level: float = 0.95) -> float:
    """F(1, d) quantile: solve P(|T| <= x) = level through I_{x^2/(d+x^2)}(1/2, d/2)."""
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if reg_inc_beta(0.5, d / 2, mid / (d + mid)) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def check(document: dict) -> list[str]:
    problems = []
    if not document["finite_f_oracles"]:
        problems.append("empty oracle selection")
    for row in document["finite_f_oracles"]:
        n, d = row["n"], row["df"]
        if d != n - 2 or row["level"] != 0.95:
            problems.append(f"n={n}: df/level not F(1, n-2) at 0.95")
            continue
        simpson = quantile_simpson(d) ** 2
        beta = critical_beta(d)
        if abs(simpson - beta) > 1e-9:
            problems.append(f"n={n}: Simpson {simpson} and incomplete-beta {beta} disagree")
        if abs(simpson - row["f_critical"]) > ATOL:
            problems.append(f"n={n}: fixture {row['f_critical']} reference {simpson}")
    if not (document["effect"] == 2.0 and len(document["strengths"]) == 3):
        problems.append("declared data-generating inputs changed shape")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["finite_f_oracles"][0]["f_critical"] += 1e-6
    if not check(wrong):
        failures.append("shifted critical value accepted")
    wrong = copy.deepcopy(document)
    wrong["finite_f_oracles"][0]["f_critical"] = 3.841458820694124  # chi-square(1) limit
    if not check(wrong):
        failures.append("asymptotic critical value accepted for a finite sample")
    wrong = copy.deepcopy(document)
    wrong["finite_f_oracles"][1]["df"] = 299
    if not check(wrong):
        failures.append("wrong degrees of freedom accepted")
    wrong = copy.deepcopy(document)
    wrong["finite_f_oracles"] = []
    if not check(wrong):
        failures.append("empty selection accepted")
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
