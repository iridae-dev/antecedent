# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact reference for the projections of a retained posterior's effect draws.

Regenerate with:  uv run --script recalc_posterior_summary_reference.py > recalc_posterior_summary_reference.json

For draws v_1..v_n (binary64 values taken as exact rationals) the reference forms, with no
floating-point rounding until the final rounding to binary64:
  mean = sum v / n;   sd = sqrt(sum (v - mean)^2 / (n - 1)) (mpmath, 50 digits);
  quantile(p) = v_(k) + (h - k) (v_(k+1) - v_(k)) with h = p (n - 1) computed exactly from the
                binary64 p, k = floor(h), v_(.) the ascending order statistics (linear interpolation
                of the order statistics, Hyndman and Fan type 7), v_(k+1) = v_(k) at the top;
  below = #{ v_i < threshold } / n  (strict inequality).
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import math
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50


def quantile(sorted_values, p):
    n = len(sorted_values)
    h = F(p) * (n - 1)
    k = math.floor(h)
    nxt = sorted_values[min(k + 1, n - 1)]
    return sorted_values[k] + (h - k) * (nxt - sorted_values[k])


def case(name, values, lower, upper, threshold):
    exact = [F(v) for v in values]
    n = len(exact)
    mean = sum(exact, F(0)) / n
    ss = sum(((v - mean) ** 2 for v in exact), F(0))
    sd = mp.sqrt(mp.mpf(ss.numerator) / mp.mpf(ss.denominator) / (n - 1))
    srt = sorted(exact)
    return {
        "name": name, "values": values, "lower": lower, "upper": upper, "threshold": threshold,
        "mean": float(mean), "sd": float(sd),
        "lower_quantile": float(quantile(srt, lower)), "upper_quantile": float(quantile(srt, upper)),
        "below": float(F(sum(1 for v in exact if v < F(threshold)), n)),
    }


def main():
    rng = random.Random(20261028)
    cases = [
        case("two_draws", [0.25, 1.75], 0.025, 0.975, 1.0),
        case("three_draws_median_pair", [3.0, -1.0, 2.0], 0.25, 0.5, 2.0),
        case("ties_and_threshold_on_a_draw", [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 5.0, 5.0], 0.1, 0.9, 2.0),
    ]
    for n, loc, spread, lo, hi, thr in [(7, 0.4, 1.0, 0.025, 0.975, 0.0), (100, -2.0, 3.0, 0.05, 0.95, -2.5),
                                        (1001, 10.0, 0.5, 0.025, 0.975, 10.0), (1000, 1.0e9, 0.25, 0.1, 0.9, 1.0e9),
                                        (257, 0.0, 1.0e-6, 0.3, 0.7, 1.0e-7)]:
        values = [round(loc + spread * rng.gauss(0, 1), 12) for _ in range(n)]
        cases.append(case(f"gauss_{n}", values, lo, hi, thr))
    print(json.dumps(cases, indent=1))


main()
