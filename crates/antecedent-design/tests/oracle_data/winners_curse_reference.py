# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the winner's-curse bound.

Regenerate with:  uv run --script winners_curse_reference.py > winners_curse_reference.json

For independent Z_j ~ N(0, s_j^2) the expected maximum is, by integrating the survival function of
the maximum on both sides of zero,
    E[max Z_j] = int_0^inf (1 - prod_j Phi(x / s_j)) dx - int_0^inf prod_j Phi(-x / s_j) dx
(a candidate with s_j = 0 contributes the factor 1 for x >= 0 and 0 for x < 0), evaluated here
with adaptive Gauss-Legendre quadrature at 20 digits with breakpoints at the scale of every
group.  The dependence-free bound is s_max sqrt(2 ln N) for N considered candidates, and
unseen candidates carry the largest supplied standard error.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 20
mpf = mp.mpf


def expected_max(groups):
    """groups: list of (s, count)."""
    smax = max(s for s, _ in groups)
    if smax <= 0:
        return mpf(0)

    def log_f(x):
        total = mpf(0)
        for s, c in groups:
            if s > 0:
                total += c * mp.log(mp.ncdf(mpf(x) / mpf(s)))
            elif x < 0:
                return -mp.inf
        return total

    def pos(x):  # 1 - F(x)
        return -mp.expm1(log_f(x))

    def neg(x):  # F(-x)
        return mp.exp(log_f(-x))

    scales = sorted({mpf(s) for s, _ in groups if s > 0})
    pts = [mpf(0)]
    for s in scales:
        pts += [s * k for k in (1, 3, 9)]
    pts = sorted(set(pts))
    # place breakpoints where the large-N maximum lives as well
    n_total = sum(c for _, c in groups)
    centre = smax * mp.sqrt(2 * mp.log(max(n_total, 2)))
    pts = sorted(set(pts + [centre * f for f in (0.5, 0.8, 1.0, 1.2, 1.6)] + [mpf(14) * smax]))
    up = mp.quad(pos, pts)
    down = mp.quad(neg, pts)
    return up - down


CASES = []


def add(name, ses, n_considered):
    s_max = max(ses)
    groups = {}
    for s in ses:
        groups[s] = groups.get(s, 0) + 1
    pad = n_considered - len(ses)
    if pad:
        groups[s_max] = groups.get(s_max, 0) + pad
    g = expected_max([(s, c) for s, c in groups.items()])
    union = mpf(s_max) * mp.sqrt(2 * mp.log(n_considered))
    CASES.append({"name": name, "ses": ses, "n_considered": n_considered,
                  "gaussian": float(g), "union": float(union)})


add("six mixed", [0.1, 0.5, 0.5, 1.0, 2.0, 0.01], 6)
add("six padded to 5000", [0.1, 0.5, 0.5, 1.0, 2.0, 0.01], 5000)
add("twelve distinct", [0.05 + 0.21 * i for i in range(12)], 12)
add("twelve distinct padded", [0.05 + 0.21 * i for i in range(12)], 100_000)
add("two zero one unit", [0.0, 0.0, 1.0], 3)
add("all zero", [0.0, 0.0, 0.0], 3)
add("single", [1.0], 1)
add("single padded to ten million", [1.0], 10_000_000)
add("wide scales", [1e-3, 1.0, 1e3], 3)
add("wide scales padded", [1e-3, 1.0, 1e3], 2_000)
print(json.dumps(CASES, indent=1))
