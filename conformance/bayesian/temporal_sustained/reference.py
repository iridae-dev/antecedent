# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the sustained lag-2 effect behind the temporal_sustained fixture: the test's series (pressure_t = sin(0.04 t), defect_t = 0.7 pressure_{t-2} + 0.05 eps_t, eps from the test's SplitMix64 + Box-Muller stream, n = 400) is regenerated here and the exact least-squares lag-2 slope over rationals must lie within the fixture tolerance of 0.7 and within three of the stated standard errors; the noise-free map has slope exactly 0.7.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/temporal_sustained/reference.py --check
    uv run --frozen --script conformance/bayesian/temporal_sustained/reference.py --self-test

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


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(float(got), want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


def solve(a, b):
    n = len(a)
    m = [row[:] + [bi] for row, bi in zip(a, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        m[c] = [v / m[c][c] for v in m[c]]
        for r in range(n):
            if r != c:
                m[r] = [x - m[r][c] * y for x, y in zip(m[r], m[c])]
    return [row[n] for row in m]

def least_squares(columns, y):
    a = [[sum(p * q for p, q in zip(ci, cj)) for cj in columns] for ci in columns]
    b = [sum(p * q for p, q in zip(ci, y)) for ci in columns]
    return solve(a, b)

import math

MASK = (1 << 64) - 1
SEED = 0xA5EC_0DEF_5EED_0002
NOISE_SD = 0.05

class SplitMix64:
    def __init__(self, seed):
        self.s = seed & MASK
    def next_u64(self):
        self.s = (self.s + 0x9E37_79B9_7F4A_7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & MASK
        return z ^ (z >> 31)
    def next_f64(self):
        return 1.0 - (self.next_u64() >> 11) / float(1 << 53)
    def gaussian(self):
        u1, u2 = self.next_f64(), self.next_f64()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(math.tau * u2)

def series(n, effect):
    rng = SplitMix64(SEED)
    noise = [NOISE_SD * rng.gaussian() for _ in range(n)]
    pressure = [0.0, math.sin(0.04)] + [math.sin(t * 0.04) for t in range(2, n)]
    defect = [0.0, 0.0] + [effect * pressure[t - 2] + noise[t] for t in range(2, n)]
    return pressure, defect

def derived(n, effect):
    pressure, defect = series(n, effect)
    lagged = [Fraction(p) for p in pressure[: n - 2]]
    y = [Fraction(v) for v in defect[2:]]
    intercept, slope = least_squares([[Fraction(1)] * (n - 2), lagged], y)
    mean = sum(lagged) / len(lagged)
    sxx = sum((p - mean) ** 2 for p in lagged)
    se = NOISE_SD / math.sqrt(float(sxx))
    return float(slope), se

def check(document):
    problems = []
    n = document["n"]
    slope, se = derived(n, 0.7)
    close(0.7, document["expected_ate"], 1e-12, "expected_ate", problems)
    close(0.7, document["true_effect_per_unit"], 1e-12, "true_effect_per_unit", problems)
    if abs(slope - 0.7) > 3 * se:
        problems.append(f"regenerated lag-2 slope {slope!r} is more than 3 standard errors ({se!r}) from 0.7")
    if abs(slope - 0.7) >= document["tolerance"]:
        problems.append(f"regenerated lag-2 slope {slope!r} is outside the fixture tolerance")
    if not 0.002 < se < 0.005:
        problems.append(f"standard error {se!r} is not the ~0.0035 the derivation text states")
    if document["treatment_lag"] != 2:
        problems.append("lag differs from the lag-2 design this reference regenerated")
    return problems

MUTATIONS = [
    ("expected effect shifted", lambda d: d.__setitem__("expected_ate", 0.8)),
    ("lag changed to one", lambda d: d.__setitem__("treatment_lag", 1)),
    ("per-unit effect shifted", lambda d: d.__setitem__("true_effect_per_unit", 0.6)),
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
