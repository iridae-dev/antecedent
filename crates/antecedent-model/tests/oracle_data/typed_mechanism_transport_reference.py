# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Gaussian-mechanism fits and the Wald invariance comparison of two populations, at 50 digits.

Regenerate with:  uv run --script typed_mechanism_transport_reference.py > typed_mechanism_transport_reference.json

A Gaussian mechanism with a degree-1 polynomial mean regresses the variable on its standardized
parents z_j = (x_j - mean_j)/sd_j (mean and sd of the SOURCE sample, sd with divisor n), by ordinary
least squares: beta = (X'X)^-1 X'y with X = [1, z], residual variance s2 = rss/(n - k), k the
mean width.  The parameter vector is (beta, log s = (1/2) ln s2) and its covariance is block
diagonal: s2 (X'X)^-1 for beta and 1/(2 (n - k)) for log s.  A target refit uses the source's
standardization.  The Wald comparison of a source fit and a target fit is
d' (S_source + S_target)^-1 d with d the parameter difference, compared with the upper tail of a
chi-square law with as many degrees of freedom as parameters.  Everything is evaluated at 50
digits with explicit matrix inverses; the tail uses the regularized upper incomplete gamma
function.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50


def M(x):
    x = F(x)
    return mp.mpf(x.numerator) / x.denominator


def fit(y, xs, scaling=None):
    n = len(y)
    p = len(xs)
    if scaling is None:
        scaling = []
        for col in xs:
            mean = sum(M(v) for v in col) / n
            var = sum((M(v) - mean) ** 2 for v in col) / n
            scaling.append((mean, mp.sqrt(var)))
    rows = [[mp.mpf(1)] + [(M(xs[j][i]) - scaling[j][0]) / scaling[j][1] for j in range(p)] for i in range(n)]
    X = mp.matrix(rows)
    Y = mp.matrix([M(v) for v in y])
    k = p + 1
    info = X.T * X
    inv = info ** -1
    beta = inv * (X.T * Y)
    resid = Y - X * beta
    rss = sum(r * r for r in resid)
    dof = n - k
    s2 = rss / dof
    theta = [beta[i] for i in range(k)] + [mp.log(s2) / 2]
    d = k + 1
    cov = mp.zeros(d, d)
    for a in range(k):
        for b in range(k):
            cov[a, b] = s2 * inv[a, b]
    cov[k, k] = 1 / (2 * dof)
    return theta, cov, scaling


def wald(src, tgt):
    (t1, c1, sc), (t2, c2, _) = src, tgt
    d = len(t1)
    diff = mp.matrix([t1[i] - t2[i] for i in range(d)])
    S = c1 + c2
    stat = (diff.T * (S ** -1) * diff)[0]
    if stat < 0:
        stat = mp.mpf(0)
    p = mp.gammainc(mp.mpf(d) / 2, stat / 2, mp.inf, regularized=True)
    return stat, p, d


def grid(n, offset, step, lift=F(0)):
    return [lift + F((offset + 7 * i) % 11 - 5) * step for i in range(n)]


def case(name, n_s, n_t, p, lift, shift, noise_t, identical=False):
    xs_s = [grid(n_s, 1 + 3 * j, F(1, 8), lift) for j in range(p)]
    xs_t = [grid(n_t, 2 + 5 * j, F(1, 8), lift) for j in range(p)]

    def outcome(xs, n, intercept, scale, salt):
        out = []
        for i in range(n):
            mean = intercept + sum(F(j + 1, 2) * xs[j][i] for j in range(p))
            out.append(mean + scale * F((salt + 5 * i) % 7 - 3, 16))
        return out

    y_s = outcome(xs_s, n_s, F(1), F(1), 0)
    y_t = outcome(xs_t, n_t, F(1) + shift, noise_t, 2)
    if identical:
        xs_t, y_t = xs_s, y_s
    src = fit(y_s, xs_s)
    tgt = fit(y_t, xs_t, src[2])
    stat, pv, d = wald(src, tgt)
    f = lambda x: float(x)
    return {
        "name": name, "parents": p,
        "source": {"columns": [[float(v) for v in c] for c in xs_s], "y": [float(v) for v in y_s]},
        "target": {"columns": [[float(v) for v in c] for c in xs_t], "y": [float(v) for v in y_t]},
        "source_theta": [f(v) for v in src[0]], "source_covariance": [[f(src[1][i, j]) for j in range(d)] for i in range(d)],
        "target_theta": [f(v) for v in tgt[0]], "target_covariance": [[f(tgt[1][i, j]) for j in range(d)] for i in range(d)],
        "statistic": f(stat), "p_value": f(pv), "degrees_of_freedom": d,
    }


cases = [
    case("mild_shift_one_parent", 12, 10, 1, F(0), F(1, 8), F(1)),
    case("smallest_admissible_samples", 5, 5, 1, F(0), F(1, 4), F(1)),
    case("intercept_shift_two_parents_far_offset", 14, 13, 2, F(10000), F(2), F(1)),
    case("scale_only_difference", 16, 16, 1, F(0), F(0), F(4)),
    case("identical_samples", 9, 9, 1, F(0), F(0), F(1), identical=True),
]
print(json.dumps({"cases": cases}, separators=(",", ":")))
