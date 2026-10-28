# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Augmented two-period panel ATT from supplied propensity and untreated-change predictions, exactly.

Regenerate with:  uv run --script augmented_did_reference.py > augmented_did_reference.json

With r_i = (post_i - pre_i) - m_i, odds_i = p_i/(1-p_i), treated set T and comparison set C:

    ATT = mean_{i in T} r_i  -  sum_{i in C} odds_i r_i / sum_{i in C} odds_i,

the improved doubly robust estimator with the comparison weights normalised to sum to one.
Kish effective size of the comparison weights = (sum odds)^2 / sum odds^2.  Everything is
Fraction arithmetic on the exact binary64 inputs and written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

rng = random.Random(11)


def att(pre, post, treated, p, m):
    r = [F(b) - F(a) - F(mi) for a, b, mi in zip(pre, post, m)]
    odds = [F(pi) / (1 - F(pi)) for pi in p]
    nt = sum(treated)
    tm = sum(ri for ri, t in zip(r, treated) if t) / nt
    cw = sum(o for o, t in zip(odds, treated) if not t)
    cm = sum(o * ri for o, ri, t in zip(odds, r, treated) if not t) / cw
    ess = cw * cw / sum(o * o for o, t in zip(odds, treated) if not t)
    return dict(effect=float(tm - cm), ess=float(ess))


def make(name, n, offset, pscale, shift=0.0, pmin=0.05):
    treated = [rng.random() < 0.4 for _ in range(n)]
    treated[0], treated[1] = True, False
    pre = [offset + rng.gauss(0, 1) for _ in range(n)]
    x = [rng.gauss(0, 1) for _ in range(n)]
    p = [min(1 - pmin, max(pmin, 0.5 + pscale * xi)) for xi in x]
    post = [a + 0.5 * xi + (2.0 if t else 0.0) + 0.1 * rng.gauss(0, 1) for a, xi, t in zip(pre, x, treated)]
    m = [0.5 * xi + shift for xi in x]
    return dict(name=name, pre=pre, post=post, treated=treated, p=p, m=m, **att(pre, post, treated, p, m))


cases = [
    make("moderate_overlap", 40, 0.0, 0.2),
    make("extreme_odds", 40, 0.0, 0.6, pmin=0.001),
    make("offset_1e8", 30, 1e8, 0.2),
    make("biased_prediction", 40, 0.0, 0.3, shift=5.0, pmin=0.01),
    make("tiny", 4, 0.0, 0.1),
]
print(json.dumps(dict(cases=cases)))
