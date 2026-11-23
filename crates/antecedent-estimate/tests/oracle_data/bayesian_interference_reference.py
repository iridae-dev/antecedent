# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact posterior of the Gaussian fixed-network exposure contrast, in 40-digit arithmetic.

Regenerate with:  uv run --script bayesian_interference_reference.py > bayesian_interference_reference.json

Unit i has exposure (own_i, k_i) where own_i is its realised assignment and k_i the number of
assigned units among those with an edge into i (edge weights do not enter the neighbour-count
mapping).  The outcome model is y_i = a + b own_i + g k_i + e_i with residual variance fixed at
one and independent N(0, tau^2) priors on (a, b, g), so the posterior is Normal with precision
I / tau^2 + X'X and mean  Vn X'y  for X = [1, own, k].  The contrast between exposure levels
(own', k') and (own, k) is  b (own' - own) + g (k' - k)  (the intercept cancels), with posterior
mean c'm and variance c'Vn c for c = (0, own' - own, k' - k).  The levels are chosen among the
exposure pairs that occur in the realised assignment.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40


def network(n, stride):
    edges = []
    for i in range(n):
        for step in (1, stride, 2 * stride + 1):
            j = (i + step) % n
            if j != i:
                edges.append((i, j, 0.25 + ((i + step) % 3)))
    return edges


def case(name, n, stride, tau, from_level, to_level):
    assign = [((i * 7 + 3) % 10) < 4 for i in range(n)]
    edges = network(n, stride)
    nbrs = [0] * n
    for a, b, _ in edges:
        nbrs[b] += 1 if assign[a] else 0
    own = [1.0 if a else 0.0 for a in assign]
    y = [2.0 + 3.0 * own[i] + 4.0 * nbrs[i] + 0.7 * ((((i * 13 + 5) % 19) - 9) / 9.0) for i in range(n)]
    levels = {(own[i], float(nbrs[i])) for i in range(n)}
    assert from_level in levels and to_level in levels, (name, sorted(levels))
    X = mp.matrix([[1, mp.mpf(own[i]), mp.mpf(nbrs[i])] for i in range(n)])
    Y = mp.matrix([mp.mpf(v) for v in y])
    vn = (X.T * X + mp.eye(3) / mp.mpf(tau) ** 2) ** -1
    m = vn * (X.T * Y)
    c = mp.matrix([0, mp.mpf(to_level[0] - from_level[0]), mp.mpf(to_level[1] - from_level[1])])
    return dict(name=name, n=n, prior_sd=tau, from_level=list(from_level), to_level=list(to_level),
                edges=[list(e) for e in edges], assignment=assign, outcome=y,
                mean=float((c.T * m)[0]), sd=float(mp.sqrt((c.T * vn * c)[0])))


print(json.dumps([
    case("wide_prior", 48, 5, 100.0, (0.0, 1.0), (1.0, 2.0)),
    case("tight_prior", 48, 5, 0.9, (0.0, 0.0), (1.0, 2.0)),
], separators=(",", ":")))
