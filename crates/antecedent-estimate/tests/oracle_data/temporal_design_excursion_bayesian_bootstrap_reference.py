# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact Dirichlet-weight distribution of the cluster ratio estimate.

Regenerate with:  uv run --script temporal_design_excursion_bayesian_bootstrap_reference.py > temporal_design_excursion_bayesian_bootstrap_reference.json

Per cluster g the design has a score sum S_g (one value per coordinate) and a pair count n_g.  A
Bayesian-bootstrap draw weighs the clusters by w ~ G * Dirichlet(1, ..., 1) and reports, per
coordinate, theta(w) = sum_g w_g S_g / sum_g w_g n_g.  The ratio does not change when the weights
are rescaled, so theta is a function of the Dirichlet vector alone.  For G = 2 the weight is
(u, 1 - u) with u uniform on (0, 1) and theta is monotone in u, so the exact mean, standard
deviation and every quantile follow from one-dimensional quadrature and from theta evaluated at
u = p (or 1 - p).  For G = 3 the Dirichlet density is 2 on the simplex and the mean and variance
are integrals over (u, v), u + v < 1.  Everything is evaluated at 30 digits.  Repository-authored
(MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 30


def theta_fn(sums, counts, j):
    G = len(counts)

    def theta(w):
        return sum(w[g] * mp.mpf(sums[g][j]) for g in range(G)) / sum(w[g] * mp.mpf(counts[g]) for g in range(G))

    return theta


def moments(sums, counts, j):
    G = len(counts)
    th = theta_fn(sums, counts, j)
    if G == 2:
        m1 = mp.quad(lambda u: th([u, 1 - u]), [0, 0.5, 1])
        m2 = mp.quad(lambda u: th([u, 1 - u]) ** 2, [0, 0.5, 1])
    else:
        m1 = 2 * mp.quad(lambda u: mp.quad(lambda v: th([u, v, 1 - u - v]), [0, (1 - u) / 2, 1 - u]), [0, 0.5, 1])
        m2 = 2 * mp.quad(lambda u: mp.quad(lambda v: th([u, v, 1 - u - v]) ** 2, [0, (1 - u) / 2, 1 - u]), [0, 0.5, 1])
    return m1, mp.sqrt(m2 - m1 * m1)


def case(name, sums, counts):
    G, k = len(counts), len(sums[0])
    mean, sd, quantiles = [], [], {}
    for j in range(k):
        m, s = moments(sums, counts, j)
        mean.append(float(m))
        sd.append(float(s))
    if G == 2:
        for p in (0.025, 0.5, 0.975):
            row = []
            for j in range(k):
                th = theta_fn(sums, counts, j)
                # theta is monotone in u; the p-quantile of theta is theta at the u-quantile on
                # the side where theta increases.
                increasing = th([mp.mpf(1), mp.mpf(0)]) > th([mp.mpf(0), mp.mpf(1)])
                u = mp.mpf(p) if increasing else 1 - mp.mpf(p)
                row.append(float(th([u, 1 - u])))
            quantiles[str(p)] = row
    return {"name": name, "coordinates": k, "sums": [v for row in sums for v in row], "counts": counts,
            "mean": mean, "sd": sd, "quantiles": quantiles,
            "estimate": [float(sum(mp.mpf(sums[g][j]) for g in range(G)) / sum(mp.mpf(c) for c in counts)) for j in range(k)]}


cases = [
    case("two_clusters_balanced", [[3.0], [1.0]], [5.0, 5.0]),
    case("two_clusters_unbalanced_counts", [[2.0], [30.0]], [1.0, 40.0]),
    case("two_clusters_offset_scores", [[1.0e6 + 0.5], [1.0e6 - 0.3]], [3.0, 7.0]),
    case("two_clusters_two_coordinates_sign_change", [[4.0, -2.0], [-1.0, 3.0]], [6.0, 2.0]),
    case("two_clusters_decreasing_in_weight", [[0.5, 9.0], [8.0, 1.0]], [4.0, 4.0]),
    case("three_clusters", [[2.0], [-1.0], [7.0]], [2.0, 5.0, 3.0]),
    case("three_clusters_heavy_cluster", [[0.1], [0.2], [50.0]], [10.0, 10.0, 1.0]),
]
print(json.dumps(cases))
