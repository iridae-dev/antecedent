# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact laws of Dirichlet-weighted score means and of the complier Wald ratio, in rationals.

Regenerate with:  uv run --script score_posterior_reference.py > score_posterior_reference.json

The Bayesian bootstrap draws weights omega ~ Dirichlet(1, ..., 1) on the n rows and reports
S(c) = sum_i omega_i c_i for a score column c.  For distinct scores c_1 .. c_n the law of S is the
B-spline law
    P(S <= x) = 1 - sum_i (c_i - x)_+^(n-1) / prod_{j != i} (c_i - c_j),
    density(x) = (n-1) sum_i (c_i - x)_+^(n-2) / prod_{j != i} (c_i - c_j)
(check n = 2: S is uniform between c_1 and c_2).  The complier posterior is the ratio
R = S(a) / S(b) of the intention-to-treat score a_i = s_i y_i and the receipt score b_i = s_i d_i
with s_i = 1/p_i for assigned rows and -1/(1 - p_i) otherwise.  When every b_i is nonnegative and
one is positive, S(b) > 0 and P(R <= r) = P(S(a - r b) <= 0), which is the formula at x = 0 for
the shifted scores.  All arithmetic is Fraction on the exact binary64 inputs (the cumulative
probabilities and quantiles are rounded to binary64 at the end; quantiles by bisection to 1e-13).
Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F


def cdf(c, x=F(0)):
    n = len(c)
    total = F(0)
    for i, ci in enumerate(c):
        if ci > x:
            den = F(1)
            for j, cj in enumerate(c):
                if j != i:
                    den *= ci - cj
            total += (ci - x) ** (n - 1) / den
    return 1 - total


def density(c, x):
    n = len(c)
    total = F(0)
    for i, ci in enumerate(c):
        if ci > x:
            den = F(1)
            for j, cj in enumerate(c):
                if j != i:
                    den *= ci - cj
            total += (ci - x) ** (n - 2) / den
    return (n - 1) * total


def quantile(fn, lo, hi, p):
    for _ in range(80):
        mid = (lo + hi) / 2
        if fn(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build(name, y, z, d, p):
    s = [1 / F(pi) if zi else -1 / (1 - F(pi)) for zi, pi in zip(z, p)]
    a = [si * F(yi) for si, yi in zip(s, y)]
    b = [si * F(1 if di else 0) for si, di in zip(s, d)]
    assert all(v >= 0 for v in b) and any(v > 0 for v in b)
    point = sum(a) / sum(b)
    ratio_cdf = lambda r: cdf([ai - r * bi for ai, bi in zip(a, b)])
    lo, hi = min(ai / bi for ai, bi in zip(a, b) if bi > 0) - 50, max(ai / bi for ai, bi in zip(a, b) if bi > 0) + 50
    # The ratio has mass outside the row ratios when rows with b = 0 carry a: widen generously.
    lo, hi = F(-1000), F(1000)
    q = {str(pp): quantile(ratio_cdf, lo, hi, F(pp)) for pp in ("0.025", "0.975")}
    grid = [point + F(k, 2) for k in range(-6, 7)]
    ratio_density = lambda r: None
    out = dict(
        name=name, y=y, assigned=z, received=d, probabilities=p,
        itt_mean=float(sum(a) / len(a)), first_stage_mean=float(sum(b) / len(b)), point=float(point),
        ratio_grid=[float(r) for r in grid], ratio_cdf=[float(ratio_cdf(r)) for r in grid],
        ratio_q025=float(q["0.025"]), ratio_q975=float(q["0.975"]),
    )
    # The sampler averages: the score mean is sum(omega * c) where omega is Dirichlet(1) scaled to mean one /n.
    n = len(a)
    itt_grid = [sum(a) / n + F(k, 2) for k in range(-4, 5)]
    out["itt_grid"] = [float(x) for x in itt_grid]
    out["itt_cdf"] = [float(cdf(a, x)) for x in itt_grid]
    fs_grid = [sum(b) / n + F(k, 4) for k in range(-3, 4)]
    out["first_stage_grid"] = [float(x) for x in fs_grid]
    out["first_stage_cdf"] = [float(cdf([bi for bi in b], x)) if len(set(b)) == len(b) else None for x in fs_grid]
    return out


# Scores must be distinct for the B-spline formula; the first-stage column repeats zeros, so
# only the intention-to-treat and the ratio laws are asserted.
y = [3.1, 0.7, 2.4, 4.9, 1.3, 0.2, 5.6, 2.2, 3.8, 1.1]
z = [True, False, True, True, False, False, True, False, True, False]
d = [True, False, False, True, False, False, True, False, True, False]
cases = [
    build("known_half", y, z, d, [0.5] * 10),
    build("varying_probabilities", y, z, d, [0.3, 0.6, 0.45, 0.7, 0.35, 0.55, 0.65, 0.4, 0.8, 0.25]),
]
for c in cases:
    del c["first_stage_cdf"], c["first_stage_grid"]
print(json.dumps(cases, separators=(",", ":")))
