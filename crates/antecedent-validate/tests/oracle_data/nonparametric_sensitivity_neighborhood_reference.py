# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Brute-force 60-digit reference for the neighbourhood-truncated leave-one-out kernel smoother.

  uv run --script nonparametric_sensitivity_neighborhood_reference.py \
      > nonparametric_sensitivity_neighborhood_reference.json

Written from the declared definition, with no product code. Rows with identical covariates form
one cell carrying their count and per-target sums. For row i in cell c:
  * the contributing cells are the `neighbors` nearest OTHER cells (Euclidean distance between
    cell covariates), plus the row's own cell when it holds at least one other row;
  * a cell u contributes with multiplicity m_u = count_u - [u == c] (the row itself leaves its
    own cell) and its target sum with y_i removed from the own cell;
  * weight w_u = exp(-(d_u^2 - d_min^2) / (2 h^2)) with d_min the distance to the nearest
    contributing cell (own cell distance is 0);
  * prediction = sum_u w_u (sum_u y - [own] y_i) / sum_u w_u m_u.
The generator asserts there is no tie at the neighbour-count boundary, so the selected set is
unambiguous. With `neighbors` >= (cells - 1) the prediction is the exact leave-one-out
Nadaraya-Watson smoother over every other row.

Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60
mpf = mp.mpf


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield F(state >> 40, 1 << 24)


def d3(x):
    return F(round(x * 1000), 1000)


def make(name, n, dim, seed, h, neighbors, duplicate_every=0, scale=1):
    g = lcg(seed)
    cov = []
    for i in range(n):
        if duplicate_every and i % duplicate_every == 1:
            cov.append(list(cov[-1]))
        else:
            cov.append([d3((next(g) - F(1, 2)) * 4 * scale) for _ in range(dim)])
    y1 = [d3((next(g) - F(1, 2)) * 10) for _ in range(n)]
    y2 = [d3(cov[i][0] * 2 + (next(g) - F(1, 2))) for i in range(n)]
    return {"name": name, "dim": dim, "h": h, "neighbors": neighbors,
            "cov": [[float(v) for v in row] for row in cov],
            "y1": [float(v) for v in y1], "y2": [float(v) for v in y2]}


def predict(case):
    cov = [[F(v) for v in row] for row in case["cov"]]
    y = [[F(v) for v in case["y1"]], [F(v) for v in case["y2"]]]
    n = len(cov)
    cells = {}
    for i, row in enumerate(cov):
        cells.setdefault(tuple(row), []).append(i)
    keys = list(cells)
    h2 = mpf(case["h"]) ** 2
    out = [[None] * n, [None] * n]
    for ci, key in enumerate(keys):
        d2 = {}
        for cj, other in enumerate(keys):
            if cj != ci:
                d2[cj] = sum((a - b) ** 2 for a, b in zip(key, other))
        order = sorted(d2, key=lambda c: d2[c])
        k = case["neighbors"]
        if k < len(order):
            assert d2[order[k - 1]] != d2[order[k]], "tie at the neighbour boundary"
        chosen = order[:k]
        for i in cells[key]:
            members = [(ci, F(0))] + [(c, d2[c]) for c in chosen]
            rows = {}
            for c, dd in members:
                cnt = len(cells[keys[c]]) - (1 if c == ci else 0)
                if cnt > 0:
                    rows[c] = (cnt, dd)
            dmin = min(dd for _, dd in rows.values())
            for t in (0, 1):
                num = mpf(0)
                den = mpf(0)
                for c, (cnt, dd) in rows.items():
                    w = mp.exp(-(mpf(dd.numerator) / dd.denominator - mpf(dmin.numerator) / dmin.denominator) / (2 * h2))
                    s = sum(y[t][r] for r in cells[keys[c]]) - (y[t][i] if c == ci else 0)
                    num += w * (mpf(s.numerator) / s.denominator)
                    den += w * cnt
                out[t][i] = float(num / den)
    return out


CASES = [
    make("plain_k3", 30, 2, 61, 0.8, 3),
    make("plain_k8_wide_bandwidth", 30, 2, 62, 25.0, 8),
    make("tiny_bandwidth", 24, 2, 63, 1e-3, 4),
    make("covering_k", 18, 2, 64, 0.7, 40),
    make("duplicated_rows_k2", 30, 2, 65, 0.9, 2, duplicate_every=3),
    make("one_dimension_k5", 36, 1, 66, 0.4, 5),
    make("three_dimensions_k1", 28, 3, 67, 1.1, 1),
    make("two_rows", 2, 1, 68, 1.0, 3),
]
CASES[-1]["cov"] = [[0.25], [0.25]]
for c in CASES:
    c["expected"] = predict(c)
print(json.dumps(CASES))
