# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the proper scores and the paired score-difference standard errors.

Regenerate with:  uv run --script proper_scores_reference.py > proper_scores_reference.json

Written from definitions, in exact rational arithmetic wherever the quantity is rational and at
50 digits otherwise; none of it follows the product's evaluation order:
  Gaussian log score      -ln phi(z)/sigma, z = (y - mu)/sigma                 (mpmath log and erfc)
  Gaussian CRPS           the integral of (Phi((t - mu)/sigma) - 1{t >= y})^2 dt by tanh-sinh
                          quadrature (and the closed form of Gneiting and Raftery, to confirm it)
  finite / empirical CRPS the energy form E|X - y| - E|X - X'|/2 over the law's atoms (exact,
                          from the doubles' exact values); fair form divides the pair sum by m(m-1)
  Brier                   sum_k (p_k - 1{y = k})^2 (exact)
  energy score            E||X - y|| - E||X - X'||/2 over a finite joint law (50 digits)
  variance of the mean    exact rationals: iid s^2/n; Bartlett long-run variance with truncation L;
                          one-way cluster-robust G/(G - 1) sum_g (S_g)^2 / n^2

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random
from fractions import Fraction

import mpmath as mp

mp.mp.dps = 50
mpf = mp.mpf


def fr(x):
    return Fraction(x)


def to_text(q, digits=40):
    if isinstance(q, Fraction):
        return mp.nstr(mpf(q.numerator) / q.denominator, digits)
    return mp.nstr(q, digits)


def gaussian(mu, sd, y):
    mu, sd, y = mpf(mu), mpf(sd), mpf(y)
    z = (y - mu) / sd
    log_score = mp.log(sd) + z * z / 2 + mp.log(2 * mp.pi) / 2
    phi = mp.npdf(z)
    closed = sd * (z * (2 * mp.ncdf(z) - 1) + 2 * phi - 1 / mp.sqrt(mp.pi))
    return log_score, closed


def gaussian_crps_quadrature(mu, sd, y):
    mu, sd, y = mpf(mu), mpf(sd), mpf(y)
    pts = [mu - 40 * sd, mu - 8 * sd, mu, mu + 8 * sd, mu + 40 * sd]
    pts = sorted(set(pts + [y]))
    lo, hi = pts[0], pts[-1]
    if y < lo:
        lo = y
    return mp.quad(lambda t: (mp.ncdf((t - mu) / sd) - (1 if t >= y else 0)) ** 2, pts)


def atoms_crps(atoms, y, fair=False):
    """atoms: list of (Fraction value, Fraction weight); fair uses equal weights 1/m."""
    y = fr(y)
    first = sum(w * abs(x - y) for x, w in atoms)
    pairs = sum(wi * wj * abs(xi - xj) for xi, wi in atoms for xj, wj in atoms)
    if fair:
        m = len(atoms)
        pairs = pairs * Fraction(m, m - 1)
    return first - pairs / 2


def energy(atoms, weights, y):
    d = len(y)

    def dist(a, b):
        return mp.sqrt(sum((mpf(u) - mpf(v)) ** 2 for u, v in zip(a, b)))
    def mf(q):
        return mpf(q.numerator) / q.denominator
    first = sum(mf(w) * dist(a, y) for a, w in zip(atoms, weights))
    second = sum(mf(wi) * mf(wj) * dist(a, b) for a, wi in zip(atoms, weights) for b, wj in zip(atoms, weights))
    return first - second / 2


def variance_iid(d):
    n = len(d)
    mean = sum(d) / n
    return sum((x - mean) ** 2 for x in d) / (n - 1) / n


def variance_bartlett(d, lag):
    n = len(d)
    mean = sum(d) / n
    c = [x - mean for x in d]

    def gamma(k):
        return sum(c[i + k] * c[i] for i in range(n - k)) / n
    total = gamma(0) + sum(2 * (1 - Fraction(k, lag + 1)) * gamma(k) for k in range(1, lag + 1))
    return max(total, Fraction(0)) / n


def variance_cluster(d, labels):
    n = len(d)
    mean = sum(d) / n
    sums = {}
    for x, g in zip(d, labels):
        sums[g] = sums.get(g, Fraction(0)) + (x - mean)
    g = len(sums)
    return Fraction(g, g - 1) * sum(s * s for s in sums.values()) / (n * n)


def main():
    rnd = random.Random(20261028)
    out = {"gaussian": [], "gaussian_crps_quadrature": [], "brier": [], "finite": [], "sample": [], "energy": [], "variance": []}
    for mu, sd, y in [(0, 1, 0), (0, 1, 1), (0, 1, -37), (0, 1, 8.5), (3, 1e-8, 3 + 5e-8), (1e6, 1e-3, 1e6 + 0.01), (-2.5, 4e4, 1e5), (0, 1, 1e-9), (7, 1, 7 + 1e-5), (0, 1, 40), (5, 3, 5)]:
        ls, closed = gaussian(mu, sd, y)
        out["gaussian"].append({"mean": mu, "sd": sd, "y": y, "log_score": to_text(ls), "crps": to_text(closed)})
    for mu, sd, y in [(0, 1, 0), (0, 1, 1), (0, 1, -3), (2, 0.5, 2.7)]:
        out["gaussian_crps_quadrature"].append({"mean": mu, "sd": sd, "y": y, "crps": to_text(gaussian_crps_quadrature(mu, sd, y), 30),
                                               "closed": to_text(gaussian(mu, sd, y)[1], 30)})
    for probs, outcome in [([0.7, 0.3], 0), ([0.7, 0.3], 1), ([0.2, 0.5, 0.3], 2), ([1.0, 0.0], 0), ([1e-12, 1 - 1e-12], 0), ([0.25] * 4, 3)]:
        total = sum((fr(p) - (1 if k == outcome else 0)) ** 2 for k, p in enumerate(probs))
        log_p = mp.log(mpf(fr(probs[outcome]).numerator) / fr(probs[outcome]).denominator) if probs[outcome] > 0 else None
        out["brier"].append({"probabilities": probs, "outcome": outcome, "brier": to_text(total),
                             "log_score": None if log_p is None else to_text(-log_p)})
    # finite laws: strictly increasing support
    for support, probs, y in [([0.0, 1.0], [0.5, 0.5], 0.5), ([-1.0, 0.0, 2.5], [0.2, 0.5, 0.3], 0.0), ([1e9, 1e9 + 1, 1e9 + 3], [0.1, 0.6, 0.3], 1e9 + 2),
                              ([0.0, 1e-9, 2e-9], [0.3, 0.3, 0.4], 1e-9), ([0.0, 1.0], [1.0 - 1e-15, 1e-15], 1.0), ([-5.0, 5.0], [0.5, 0.5], 100.0)]:
        atoms = [(fr(x), fr(p)) for x, p in zip(support, probs)]
        total_w = sum(w for _, w in atoms)
        out["finite"].append({"support": support, "probabilities": probs, "y": y, "mass": to_text(total_w), "crps": to_text(atoms_crps(atoms, y) / total_w if total_w != 1 else atoms_crps(atoms, y))})
    # samples: draws with offsets that make cancellation visible
    for offset in (0.0, 1e3, 1e6, 1e9, 1e12):
        for m in (2, 5, 40):
            draws = [offset + round(rnd.gauss(0, 1), 3) for _ in range(m)]
            y = offset + round(rnd.gauss(0, 1), 3)
            atoms = [(fr(x), Fraction(1, m)) for x in draws]
            out["sample"].append({"draws": draws, "y": y, "offset": offset, "empirical": to_text(atoms_crps(atoms, y)),
                                  "fair": to_text(atoms_crps(atoms, y, fair=True))})
    for dim, m in ((2, 4), (3, 6), (5, 3)):
        atoms = [[round(rnd.gauss(0, 2), 2) for _ in range(dim)] for _ in range(m)]
        w = [Fraction(1, m)] * m
        y = [round(rnd.gauss(0, 2), 2) for _ in range(dim)]
        out["energy"].append({"dimension": dim, "atoms": [v for a in atoms for v in a], "y": y, "probabilities": [float(x) for x in w],
                              "energy": to_text(energy(atoms, w, y))})
    # equal weights not dyadic: 1/3 each, uneven weights
    atoms = [[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]]
    w = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)]
    out["energy"].append({"dimension": 2, "atoms": [v for a in atoms for v in a], "y": [1.0, 0.0], "probabilities": [0.5, 0.25, 0.25], "energy": to_text(energy(atoms, w, [1.0, 0.0]))})
    # variances of a mean of differences
    def dyadic(n, scale=1.0, off=0.0):
        return [off + scale * round(rnd.gauss(0, 1) * 64) / 64 for _ in range(n)]
    for n, lag in ((2, 1), (6, 2), (30, 4), (30, 29), (200, 10), (200, 0)):
        d = dyadic(n)
        out["variance"].append({"kind": "serial", "d": d, "lag": lag, "variance": to_text(variance_bartlett([fr(x) for x in d], lag), 30)})
    # autocorrelated and offset series, near-constant series
    d = [0.0]
    for _ in range(199):
        d.append(round((0.8 * d[-1] + rnd.gauss(0, 1)) * 256) / 256)
    out["variance"].append({"kind": "serial", "d": d, "lag": 12, "variance": to_text(variance_bartlett([fr(x) for x in d], 12), 30)})
    d = dyadic(50, 1e-3, 1e8)
    out["variance"].append({"kind": "serial", "d": d, "lag": 3, "variance": to_text(variance_bartlett([fr(x) for x in d], 3), 30)})
    out["variance"].append({"kind": "iid", "d": d, "variance": to_text(variance_iid([fr(x) for x in d]), 30)})
    for n in (2, 3, 25, 400):
        d = dyadic(n)
        out["variance"].append({"kind": "iid", "d": d, "variance": to_text(variance_iid([fr(x) for x in d]), 30)})
    for n, groups in ((6, 2), (40, 5), (40, 40), (100, 3)):
        d = dyadic(n)
        labels = [rnd.randrange(groups) for _ in range(n)]
        labels[0], labels[1] = 0, 1  # at least two clusters
        out["variance"].append({"kind": "cluster", "d": d, "labels": labels, "variance": to_text(variance_cluster([fr(x) for x in d], labels), 30)})
    print(json.dumps(out, separators=(",", ":")))


main()
