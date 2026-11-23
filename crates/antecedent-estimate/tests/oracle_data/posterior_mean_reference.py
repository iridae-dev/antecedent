# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact rational reference for the point-only posterior means in posterior_mean.rs.

Regenerate with:  uv run --script posterior_mean_reference.py > posterior_mean_reference.json

Every quantity is computed in exact rational arithmetic (fractions.Fraction) from the binary64
inputs, then rounded once to binary64 for the comparison.  Nothing here calls the product.

* Scalar normal update.  A sample (or contrast) of size n with mean m and Bessel variance s2 has
  likelihood precision n / s2 for its mean; with an independent N(0, tau^2) prior the posterior
  mean is  m * (n / s2) / (n / s2 + 1 / tau^2) = m * n tau^2 / (n tau^2 + s2).  With a known
  sampling variance v of a point p the same algebra gives  p * tau^2 / (tau^2 + v).
* Covariance entries: each entry e with sampling variance v updates as the known-variance case;
  v = 0 leaves the entry unchanged (the limit of the formula).
* Dirichlet(c, ..., c) prior on k atoms with weights w: posterior mean (w_i + c) / (sum w + k c).
  The supported variant updates only the strictly positive atoms (k = their number) and keeps
  zeros at zero.
* Ridge posterior mean of linear coefficients: with sigma2 = rss / (n - p) plugged in and an
  independent N(0, tau^2) prior on every non-intercept coefficient (intercept unpenalized),
  beta = (X'X + (sigma2 / tau^2) D)^{-1} X'y, D = identity with a zero in the intercept slot,
  solved by Gauss-Jordan elimination with exact pivots.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import random
from fractions import Fraction as F


def to_float(x: F) -> float:
    return float(x)


def bessel(values):
    n = len(values)
    mean = sum(values, F(0)) / n
    rss = sum(((v - mean) ** 2 for v in values), F(0))
    return n, mean, rss / (n - 1)


def normal_update(mean, n, s2, tau):
    t2 = F(tau) ** 2
    return mean * n * t2 / (n * t2 + s2)


def known_update(point, var, tau):
    t2 = F(tau) ** 2
    return point * t2 / (t2 + var)


def solve(a, b):
    p = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(p):
        pivot = next(r for r in range(col, p) if m[r][col] != 0)
        m[col], m[pivot] = m[pivot], m[col]
        inv = 1 / m[col][col]
        m[col] = [x * inv for x in m[col]]
        for r in range(p):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][p] for i in range(p)]


def ridge(gram, xty, rss, n, tau):
    p = len(xty)
    sigma2 = F(rss) / (n - p)
    pen = sigma2 / F(tau) ** 2
    a = [[F(gram[i * p + j]) for j in range(p)] for i in range(p)]
    for k in range(1, p):
        a[k][k] += pen
    return solve(a, [F(v) for v in xty])


def main():
    rng = random.Random(20261028)
    out = {"normal_mean": [], "score_contrast": [], "known_variance": [], "covariance": [],
           "dirichlet": [], "dirichlet_supported": [], "ridge": []}

    def sample(n, loc, spread):
        return [loc + spread * (rng.randint(-1000, 1000) / 997.0) for _ in range(n)]

    for n, loc, spread, tau in [(5, 0.7, 1.0, 0.5), (12, -3.0, 2.0, 1e6), (9, 1.0e9, 0.5, 2.0),
                                (30, 0.0, 4.0, 0.01), (2, 1.5, 1.0, 1.0), (50, 12.5, 3.0, 7.0)]:
        v = sample(n, loc, spread)
        m, mean, s2 = bessel([F(x) for x in v])
        out["normal_mean"].append({"values": v, "scale": tau, "want": to_float(normal_update(mean, m, s2, tau))})
        c = sample(n, loc / 2, spread)
        t = [x + y for x, y in zip(c, sample(n, loc, spread))]
        m, mean, s2 = bessel([F(a) - F(b) for a, b in zip(t, c)])
        out["score_contrast"].append({"control": c, "treated": t, "scale": tau,
                                      "want": to_float(normal_update(mean, m, s2, tau))})
    for point, var, tau in [(2.5, 0.04, 1.0), (-0.3, 1e-8, 0.2), (1e6, 25.0, 300.0), (0.0, 1.0, 1.0),
                            (7.0, 1e12, 3.0), (0.125, 0.5, 1e9)]:
        out["known_variance"].append({"point": point, "variance": var, "scale": tau,
                                      "want": to_float(known_update(F(point), F(var), tau))})
    for tau in (0.3, 1.0, 50.0):
        entries = [1.2, -0.4, 0.0, 3.75, 2.0]
        variances = [0.09, 0.0, 0.5, 0.02, 4.0]
        want = [to_float(F(e) if v == 0 else known_update(F(e), F(v), tau)) for e, v in zip(entries, variances)]
        out["covariance"].append({"entries": entries, "variances": variances, "scale": tau, "want": want})
    for weights, c in [([3.0, 1.0, 0.0, 6.0], 0.5), ([1.0, 1.0], 1e-9), ([0.0, 0.0, 5.0], 2.0),
                       ([2.5, 7.25, 0.125, 9.0, 4.0], 1000.0), ([0.2, 0.3, 0.5], 0.05)]:
        total = sum((F(w) for w in weights), F(0)) + F(c) * len(weights)
        out["dirichlet"].append({"weights": weights, "concentration": c,
                                 "want": [to_float((F(w) + F(c)) / total) for w in weights]})
    for probs, c in [([0.5, 0.0, 0.3, 0.2], 0.4), ([0.0, 1.0], 3.0), ([0.25, 0.25, 0.0, 0.5, 0.0], 1e-6),
                     ([0.1, 0.2, 0.3, 0.4], 10.0)]:
        free = [F(p) for p in probs if p > 0]
        total = sum(free, F(0)) + F(c) * len(free)
        want = [to_float((F(p) + F(c)) / total) if p > 0 else 0.0 for p in probs]
        out["dirichlet_supported"].append({"probabilities": probs, "concentration": c, "want": want})
    for p, n, tau, offset in [(2, 6, 1000.0, 0), (3, 40, 0.5, 0), (4, 60, 2.0, 0), (3, 25, 0.1, 100)]:
        rows = [[1.0] + [float(rng.randint(-8, 8) + offset) + rng.randint(0, 99) / 100.0 for _ in range(p - 1)]
                for _ in range(n)]
        beta_true = [rng.randint(-3, 3) / 2.0 for _ in range(p)]
        y = [sum(b * x for b, x in zip(beta_true, r)) + rng.randint(-100, 100) / 50.0 for r in rows]
        gram = [sum((F(r[i]) * F(r[j]) for r in rows), F(0)) for i in range(p) for j in range(p)]
        xty = [sum((F(r[i]) * F(yy) for r, yy in zip(rows, y)), F(0)) for i in range(p)]
        yty = sum((F(v) ** 2 for v in y), F(0))
        ols = solve([[gram[i * p + j] for j in range(p)] for i in range(p)], xty)
        rss = yty - sum((b * x for b, x in zip(ols, xty)), F(0))
        # inputs handed to the product are the binary64 roundings of the exact sufficient statistics;
        # the reference is evaluated on exactly those binary64 values so only the solve is compared
        g64, x64, r64 = [to_float(v) for v in gram], [to_float(v) for v in xty], to_float(rss)
        beta = ridge(g64, x64, r64, n, tau)
        out["ridge"].append({"n_rows": n, "gram": g64, "xty": x64, "rss": r64, "scale": tau,
                             "want": [to_float(b) for b in beta]})
    print(json.dumps(out, indent=1))


main()
