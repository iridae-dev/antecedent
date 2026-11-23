# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the PIT-window drift diagnostic (two-sample KS distance and PSI).

Regenerate with:  uv run --script distribution_drift_reference.py > distribution_drift_reference.json

Written from the statistics' definitions:
  KS  = max over the pooled values x of |F_a(x) - F_b(x)|, F the right-continuous empirical
        distribution function, in exact rational arithmetic (ties at a value are consumed together);
  PSI = sum_k (q_k - p_k) ln(q_k / p_k), with p_k = (count_a[k] + 1/2) / (n_a + bins/2) and q_k the same
        for the comparison window, equal-width bins on [0, 1] with the value 1 in the last bin,
        evaluated at 60 digits.
The values are decimal strings with at most three places so that ties and bin edges (0.25, 0.5,
0.3 with ten bins) occur; a value is binned by its decimal meaning, floor(value * bins), exactly.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random
from fractions import Fraction

import mpmath as mp

mp.mp.dps = 60


def ks(a, b):
    pts = sorted(set(a) | set(b))
    best = Fraction(0)
    for x in pts:
        fa = Fraction(sum(1 for v in a if v <= x), len(a))
        fb = Fraction(sum(1 for v in b if v <= x), len(b))
        best = max(best, abs(fa - fb))
    return best


def psi(a, b, bins):
    def counts(w):
        c = [0] * bins
        for v in w:
            k = min(int(v * bins // 1), bins - 1)
            c[k] += 1
        return c

    ca, cb = counts(a), counts(b)
    half = Fraction(1, 2)
    total = mp.mpf(0)
    for x, y in zip(ca, cb):
        p = (x + half) / (len(a) + half * bins)
        q = (y + half) / (len(b) + half * bins)
        total += (mp.mpf((q - p).numerator) / (q - p).denominator) * mp.log(
            mp.mpf(q.numerator) * p.denominator / (mp.mpf(q.denominator) * p.numerator)
        )
    return total


rng = random.Random(20261028)
cases = []


def add(name, a, b, bins):
    fa = [Fraction(s) for s in a]
    fb = [Fraction(s) for s in b]
    cases.append({
        "name": name, "reference": a, "comparison": b, "bins": bins,
        "ks": float(ks(fa, fb)), "psi": mp.nstr(psi(fa, fb, bins), 25),
    })


def draw(n, lo, hi):
    return [f"{rng.randint(int(lo * 1000), int(hi * 1000)) / 1000:.3f}" for _ in range(n)]


add("identical", ["0.100", "0.400", "0.600", "0.900"], ["0.100", "0.400", "0.600", "0.900"], 4)
add("disjoint", ["0.100", "0.200", "0.300"], ["0.700", "0.800", "0.900"], 4)
add("ties_at_edges", ["0.000", "0.000", "0.250", "0.500", "1.000"], ["0.250", "0.250", "0.500", "1.000", "1.000", "1.000"], 4)
add("tenth_edges", ["0.300", "0.700", "0.100", "0.900", "0.600"], ["0.300", "0.300", "0.700", "0.700", "0.000"], 10)
add("single_vs_many", ["0.500"], draw(40, 0, 1), 5)
for i in range(60):
    na, nb = rng.randint(1, 60), rng.randint(1, 60)
    bins = rng.choice([2, 3, 5, 10, 16, 20])
    lo, hi = sorted([rng.random() * 0.5, 0.5 + rng.random() * 0.5])
    add(f"random_{i}", draw(na, 0, 1), draw(nb, lo, hi), bins)
print(json.dumps(cases, indent=1))
