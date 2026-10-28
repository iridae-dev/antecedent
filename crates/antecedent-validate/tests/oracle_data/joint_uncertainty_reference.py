# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Conservative endpoint percentile bootstrap of the joint mechanism range, by vertex enumeration.

Regenerate with:  uv run --script joint_uncertainty_reference.py draws.json > joint_uncertainty_reference.json

Input: for each case the count table of the source experiment (W x X x Y, W-major, binary X, numeric
Y levels) and the count table of every bootstrap draw (or null for a draw the sampler itself
dropped) that the sampler produced for the recorded seed.  For one draw with cell counts c the law
is P(w,x,y) = c / sum c, P_W(w) = sum_{x,y} P, P(y | w, x) = P(w,x,y) / sum_y P(w,x,y); a draw with an
empty (w, x) row has no conditional law and is a failed replicate.  The surrogate response is
    psi = sum_w P_W(w) [ m(w, 1) - m(w, 0) ],   m(w, x) = sum_y y P(y | w, x),
and the declared box contaminates each factor independently, Q = (1 - e) P + e R with R free in its
simplex: the kernel rows (e = e_K, one R per (w, x)) and the shared parent marginal (e = e_W).  psi is
linear in each replacement given the others, so its extremes are attained where every R is a point
mass; this script enumerates all such vertices (W parent point masses times |Y|^(2W) kernel point
masses) in exact rational arithmetic and takes the minimum and maximum.  The interval is
[q_{(1-level)/2}(minima), q_{(1+level)/2}(maxima)] with Hyndman-Fan type-7 quantiles over the
successful draws, and the baseline interval is the same percentile interval of the zero-deviation
responses.  Values are written as nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json
import sys
from fractions import Fraction as F


def counts_case(counts, W, Y):
    return [counts[w * 2 * Y + x * Y:(w * 2 * Y + x * Y) + Y] for w in range(W) for x in range(2)]


def draw_range(counts, W, y_values, e_k, e_w):
    Y = len(y_values)
    total = sum(counts)
    rows = counts_case(counts, W, Y)
    if any(sum(r) == 0 for r in rows):
        return None
    pw = [sum(sum(rows[w * 2 + x]) for x in range(2)) / F(total) for w in range(W)]
    m = [[sum(F(y) * F(c) for y, c in zip(y_values, rows[w * 2 + x])) / sum(rows[w * 2 + x])
          for x in range(2)] for w in range(W)]
    ek, ew = F(e_k), F(e_w)
    ymin, ymax = None, None
    for star in range(W):
        for choice in itertools.product(range(Y), repeat=2 * W):
            psi = F(0)
            for w in range(W):
                pw2 = (1 - ew) * pw[w] + (ew if w == star else 0)
                m1 = (1 - ek) * m[w][1] + ek * F(y_values[choice[w * 2 + 1]])
                m0 = (1 - ek) * m[w][0] + ek * F(y_values[choice[w * 2]])
                psi += pw2 * (m1 - m0)
            ymin = psi if ymin is None or psi < ymin else ymin
            ymax = psi if ymax is None or psi > ymax else ymax
    zero = sum(pw[w] * (m[w][1] - m[w][0]) for w in range(W))
    return ymin, ymax, zero


def type7(vals, p):
    s = sorted(vals)
    d = len(s)
    h = F(p) * (d - 1)
    lo = h.numerator // h.denominator
    hi = min(lo + 1, d - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


draws_in = json.load(open(sys.argv[1]))
out = []
for case in draws_in["cases"]:
    W, y_values = case["w_levels"], case["y_values"]
    lows, highs, zeros, failed = [], [], [], 0
    for counts in case["draws"]:
        r = None if counts is None else draw_range(counts, W, y_values, case["kernel_fraction"], case["parent_fraction"])
        if r is None:
            failed += 1
            continue
        lows.append(r[0]); highs.append(r[1]); zeros.append(r[2])
    level = F(case["level"])
    out.append({**case,
                "replicates_ok": len(lows), "replicates_failed": failed,
                "lower": float(type7(lows, (1 - level) / 2)), "upper": float(type7(highs, (1 + level) / 2)),
                "pointwise_at_zero": [float(type7(zeros, (1 - level) / 2)), float(type7(zeros, (1 + level) / 2))]})
print(json.dumps({"cases": out}))
