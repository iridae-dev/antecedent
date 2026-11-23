# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact reference for tie-shared ranking statements and their Monte Carlo errors.

Regenerate with:  uv run --script ranking_statement_reference.py > ranking_statement_reference.json

Tie sharing is defined here by averaging over every uniformly random linear order of the
candidates: at each draw the candidates are stably sorted by decreasing utility under each of the
m! permutations of the candidate list, and a candidate's share of the top k is the fraction of
permutations that put it in the first k positions.  (The product counts strictly greater and
tied candidates and splits slots arithmetically; this script does not.)  A baseline comparison
counts 1, 0 or 1/2.  Errors: iid sample variance / n; self-normalised weights
sqrt(sum (w (y - mean))^2) / sum w with Kish ESS (sum w)^2 / sum w^2; non-overlapping batch means
with b batches of floor(n/b) draws, standard error sqrt(var of batch means / b).  Values are
exact rationals converted once to float; mpmath only supplies the square roots at 50 digits.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import itertools
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50
rng = random.Random(77)


def sqrt(x):
    return float(mp.sqrt(mp.mpf(x.numerator) / x.denominator))


def var(v):
    n = len(v)
    m = sum(v) / n
    return sum((x - m) ** 2 for x in v) / (n - 1)


def stats(series, weights, dep, b):
    n = len(series)
    if weights is not None:
        sw = sum(weights)
        mean = sum(y * w for y, w in zip(series, weights)) / sw
        num = sum((w * (y - mean)) ** 2 for y, w in zip(series, weights))
        ess = sw * sw / sum(w * w for w in weights)
        return mean, sqrt(num) / float(sw) if num else 0.0, float(ess)
    if dep == "independent":
        return sum(series) / n, sqrt(var(series) / n), float(n)
    m = n // b
    used = b * m
    means = [sum(series[k * m:(k + 1) * m]) / m for k in range(b)]
    vb, vd = var(means), var(series[:used])
    ess = float(min(max(b * vd / vb, 1), n)) if vb > 0 else float(n)
    return sum(series) / n, sqrt(vb / b), ess


def case(name, ids, cols, weights, dep, batches, top_k, baseline):
    n = len(cols[0])
    m = len(ids)
    perms = list(itertools.permutations(range(m)))
    best = [[F(0)] * n for _ in range(m)]
    topk = {k: [[F(0)] * n for _ in range(m)] for k in top_k}
    for i in range(n):
        for p in perms:
            order = sorted(p, key=lambda j: -cols[j][i])  # stable sort over the permuted list
            for pos, j in enumerate(order):
                if pos == 0:
                    best[j][i] += F(1, len(perms))
                for k in top_k:
                    if pos < k:
                        topk[k][j][i] += F(1, len(perms))
    b = batches
    out = {"name": name, "ids": ids, "utilities": [[float(x) for x in c] for c in cols],
           "weights": None if weights is None else [float(w) for w in weights], "dependence": dep,
           "batches": batches, "top_k": top_k, "baseline": baseline, "statements": []}
    for j, cid in enumerate(ids):
        def pack(series):
            mean, se, ess = stats(series, weights, dep, b)
            return {"value": float(mean) if not isinstance(mean, float) else mean, "se": se, "ess": ess}
        st = {"id": cid, "expected_utility": pack(list(cols[j])), "best": pack(best[j]),
              "top_k": [pack(topk[k][j]) for k in top_k], "baseline": None}
        if baseline is not None and ids[j] != baseline:
            bj = ids.index(baseline)
            st["baseline"] = pack([F(1) if cols[j][i] > cols[bj][i] else F(0) if cols[j][i] < cols[bj][i] else F(1, 2)
                                   for i in range(n)])
        out["statements"].append(st)
    return out


def ints(n, m, lo, hi):
    return [[F(rng.randint(lo, hi)) for _ in range(n)] for _ in range(m)]


cases = [
    case("ties_independent", list("ABCDE"), ints(14, 5, 0, 3), None, "independent", None, [1, 2, 4], "C"),
    case("weighted_with_ties", list("ABCD"), ints(11, 4, 0, 4), [F(rng.randint(1, 9), 4) for _ in range(11)],
         "independent", None, [2, 3], "A"),
    case("all_tied_draws", list("ABC"), [[F(1)] * 8, [F(1)] * 7 + [F(2)], [F(1)] * 8], None, "independent", None, [1, 2], "B"),
    case("autocorrelated_sqrt_batches", list("ABC"), [[F(rng.randint(0, 5), 2) for _ in range(25)] for _ in range(3)],
         None, "autocorrelated", 5, [1, 2], "A"),
    case("autocorrelated_fixed_batches_with_remainder", list("ABCD"), ints(23, 4, -2, 2), None, "autocorrelated", 4,
         [3], "D"),
    case("signed_zero_ties", list("AB"), [[F(0), F(0), F(1), F(-1), F(0), F(2)], [F(0), F(1), F(1), F(-1), F(0), F(-2)]],
         None, "independent", None, [1], "A"),
]
print(json.dumps(cases, indent=1))
