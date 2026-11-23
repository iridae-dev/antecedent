# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact Wald interval and Dirichlet-weighted laws of cross-fit sequential contrast scores.

Regenerate with:  uv run --script sequential_contrast_reference.py > sequential_contrast_reference.json

For per-subject scores a_i (active regime), c_i (control regime) and d_i = a_i - c_i:
  * Wald: centre = mean(d), se = sqrt(sum (d_i - centre)^2 / (n (n - 1))), interval centre -/+
    z se with z the (1 + coverage) / 2 normal quantile (mpmath at 40 digits);
  * Bayesian bootstrap: with omega ~ Dirichlet(1, ..., 1) the posterior of each regime value and of the
    contrast is the weighted mean sum_i omega_i s_i, whose law for distinct scores is
        P(S <= x) = 1 - sum_i (s_i - x)_+^(n-1) / prod_{j != i} (s_i - s_j),
    and the three columns share omega.  Fractions throughout, from the exact binary64 scores.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
from fractions import Fraction as F
import mpmath as mp

mp.mp.dps = 40


def cdf(c, x):
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


active = [1.0, 6.0, -2.0, 4.0, 3.0, 9.0, 0.5, 2.0, 5.25, 7.5]
control = [0.0, 2.0, 1.0, -1.0, 2.5, 3.0, 0.75, 1.25, 4.125, 3.375]
fa, fc = [F(v) for v in active], [F(v) for v in control]
fd = [a - c for a, c in zip(fa, fc)]
n = len(fd)
assert all(len(set(col)) == n for col in (fa, fc, fd))
centre = sum(fd) / n
ss = sum((d - centre) ** 2 for d in fd)
se = mp.sqrt(mp.mpf(ss.numerator) / ss.denominator / (n * (n - 1)))
out = dict(active_scores=active, control_scores=control, centre=float(centre), standard_error=float(se), wald=[])
for cov in ("0.5", "0.9", "0.95", "0.99"):
    z = mp.sqrt(2) * mp.erfinv(mp.mpf(cov))
    c = mp.mpf(centre.numerator) / centre.denominator
    out["wald"].append(dict(coverage=float(cov), lower=float(c - z * se), upper=float(c + z * se)))
for name, col in (("active", fa), ("control", fc), ("contrast", fd)):
    mu = sum(col) / n
    grid = [mu + F(k, 2) for k in range(-6, 7)]
    out[name] = dict(mean=float(mu), grid=[float(x) for x in grid], cdf=[float(cdf(col, x)) for x in grid])
print(json.dumps(out, separators=(",", ":")))
