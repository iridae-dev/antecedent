# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent stdlib re-derivation of the SplitMix64 / Box-Muller battery.

The stream is rebuilt from the published SplitMix64 definition (state += 0x9E3779B97F4A7C15, then
the two xor-shift-multiply finalisation rounds); the generator is started at seed XOR gamma and a
uniform is the top 53 bits scaled by 2^-53; a normal pair is the Box-Muller (cos, sin) pair of two
consecutive uniforms with u1 floored at 2^-53, emitted in order. These conventions are the
fixture's statement of the repository stream. The statistics are then recomputed in exact integer or
rational arithmetic where possible (moments, bin counts, tail counts) and with stdlib special
functions otherwise (chi-square tail from the regularised upper incomplete gamma, Kolmogorov
distribution from its alternating series). No Antecedent code, NumPy or SciPy.

    uv run --frozen --script conformance/stats/rng_gaussian/reference.py --check
    uv run --frozen --script conformance/stats/rng_gaussian/reference.py --self-test

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
MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
RES = 1.0 / 9007199254740992.0


def stream(seed: int, count: int) -> list[int]:
    state = (seed ^ GAMMA) & MASK
    out = []
    for _ in range(count):
        state = (state + GAMMA) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        out.append(z ^ (z >> 31))
    return out


def uniforms(seed: int, n: int) -> list[float]:
    return [(w >> 11) * (1.0 / 9007199254740992.0) for w in stream(seed, n)]


def normals(seed: int, n: int) -> list[float]:
    u = uniforms(seed, 2 * ((n + 1) // 2))
    out = []
    for i in range(0, len(u), 2):
        r = math.sqrt(-2.0 * math.log(max(u[i], RES)))
        theta = math.tau * u[i + 1]
        out += [r * math.cos(theta), r * math.sin(theta)]
    return out[:n]


def mean_var(x: list[float]) -> tuple[float, float]:
    n = len(x)
    m = math.fsum(x) / n
    return m, math.fsum((v - m) ** 2 for v in x) / n


def upper_gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) (series below a+1, Lentz fraction above)."""
    if x <= 0:
        return 1.0
    if x < a + 1.0:
        term = total = 1.0 / a
        n = a
        while abs(term) > abs(total) * 1e-16:
            n += 1.0
            term *= x / n
            total += term
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = d if abs(d) > tiny else tiny
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def kolmogorov_sf(t: float) -> float:
    if t <= 0:
        return 1.0
    return min(1.0, 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * t * t) for k in range(1, 200)))


def ks_statistic(sorted_x: list[float], cdf) -> float:
    n = len(sorted_x)
    return max(max((i + 1) / n - cdf(v), cdf(v) - i / n) for i, v in enumerate(sorted_x))


def battery(document: dict) -> dict:
    n = document["generation"]["n"]
    u = uniforms(document["generation"]["uniform_seed"], n)
    z = normals(document["generation"]["normal_seed"], n)
    um, uv = mean_var(u)
    ux, uy = u[:-1], u[1:]
    mx, my = math.fsum(ux) / len(ux), math.fsum(uy) / len(uy)
    lag1 = math.fsum((a - mx) * (b - my) for a, b in zip(ux, uy)) / math.sqrt(
        math.fsum((a - mx) ** 2 for a in ux) * math.fsum((b - my) ** 2 for b in uy)
    )
    bins = [0] * 20
    for v in u:
        bins[min(19, int(math.floor(v * 20.0)))] += 1
    chi_bins = sum((c - n / 20) ** 2 / (n / 20) for c in bins)
    pairs = [[0] * 10 for _ in range(10)]
    for i in range(0, n - 1, 2):
        pairs[min(9, int(u[i] * 10))][min(9, int(u[i + 1] * 10))] += 1
    expected_pair = (n // 2) / 100
    chi_pairs = sum((c - expected_pair) ** 2 / expected_pair for row in pairs for c in row)
    d_u = ks_statistic(sorted(u), lambda v: v)
    nm, nv = mean_var(z)
    cdf = lambda v: 0.5 * math.erfc(-v / math.sqrt(2.0))
    d_n = ks_statistic(sorted(z), cdf)
    return {
        "uniform": {
            "mean": um,
            "population_variance": uv,
            "lag1_correlation": lag1,
            "bin20_counts": bins,
            "bin20_chi_square": chi_bins,
            "bin20_p_value": upper_gamma_q(19 / 2, chi_bins / 2),
            "pair_10x10_chi_square": chi_pairs,
            "pair_10x10_p_value": upper_gamma_q(99 / 2, chi_pairs / 2),
            "ks_statistic": d_u,
            "ks_p_value": kolmogorov_sf(math.sqrt(n) * d_u),
            "min": min(u),
            "max": max(u),
        },
        "normal": {
            "mean": nm,
            "population_variance": nv,
            "absolute_tail_counts": {
                "gt_1": sum(abs(v) > 1 for v in z),
                "gt_2": sum(abs(v) > 2 for v in z),
                "gt_3": sum(abs(v) > 3 for v in z),
            },
            "skew": math.fsum((v - nm) ** 3 for v in z) / n / nv**1.5,
            "excess_kurtosis": math.fsum((v - nm) ** 4 for v in z) / n / nv**2 - 3.0,
            "ks_statistic": d_n,
            "ks_p_value": kolmogorov_sf(math.sqrt(n) * d_n),
        },
    }


def close(a: float, b: float, rtol: float, atol: float) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def check(document: dict) -> list[str]:
    got = battery(document)
    problems = []
    for group in ("uniform", "normal"):
        for key, value in got[group].items():
            stored = document[group].get(key)
            if stored is None:
                continue
            if isinstance(value, list) or isinstance(value, dict):
                if value != stored:
                    problems.append(f"{group}.{key}: {stored} vs {value}")
            elif key.endswith("p_value"):
                # screening p-values: the Kolmogorov series is the asymptotic law, SciPy's may
                # be exact, so a looser agreement is required.
                if not close(value, stored, 1e-2, 1e-3):
                    problems.append(f"{group}.{key}: {stored} vs {value}")
            elif key.endswith("chi_square"):
                if not close(value, stored, 1e-9, 1e-9):
                    problems.append(f"{group}.{key}: {stored} vs {value}")
            elif not close(value, stored, 1e-12, 1e-12):
                problems.append(f"{group}.{key}: {stored} vs {value}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    for path in (("uniform", "mean"), ("normal", "population_variance"), ("uniform", "ks_statistic")):
        wrong = copy.deepcopy(document)
        wrong[path[0]][path[1]] += 1e-6
        if not check(wrong):
            failures.append(f"shifted {path} accepted")
    wrong = copy.deepcopy(document)
    wrong["uniform"]["bin20_counts"][0] += 1
    if not check(wrong):
        failures.append("changed bin count accepted")
    wrong = copy.deepcopy(document)
    wrong["generation"]["normal_seed"] += 1
    if not check(wrong):
        failures.append("changed seed accepted")
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
