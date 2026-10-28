# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Natural direct and indirect effects with a Gaussian mediator and a polynomial outcome, at 80 digits.

Regenerate with:  uv run --script nonlinear_mediation_reference.py > nonlinear_mediation_reference.json

Fits the mediator M ~ [1, A, X] by least squares (variance RSS/(n-p)) and the outcome
Y ~ [1, A, X, z..z^d, A z..A z^d] with z = (M - mean M)/sd M (sample sd), by the normal equations
at 80 digits.  For a row the mediation formula integrates the outcome polynomial over
M | a', x ~ N(mu, sigma^2); the integral is exact: with z0 = (mu - mean)/sd and s = sigma/sd,
E[(z0 + s e)^k] = sum_j C(k, 2j) z0^(k-2j) s^(2j) (2j-1)!!  for e ~ N(0, 1).  Then
theta(a, a') = mean over rows, NDE = theta(1,0) - theta(0,0), NIE = theta(1,1) - theta(1,0).
Values are written as nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from math import comb

from mpmath import mp, mpf, matrix, lu_solve, sqrt

mp.dps = 80
rng = random.Random(4242)


def ols(cols, y):
    p = len(cols)
    A, b = matrix(p, p), matrix(p, 1)
    for a in range(p):
        b[a] = sum(cols[a][i] * y[i] for i in range(len(y)))
        for c in range(p):
            A[a, c] = sum(cols[a][i] * cols[c][i] for i in range(len(y)))
    beta = lu_solve(A, b)
    return [beta[k] for k in range(p)]


def dfact(k):
    out = mpf(1)
    while k > 1:
        out *= k
        k -= 2
    return out


def solve(a, m, y, xs, degree):
    n = len(a)
    a_, m_, y_ = [mpf(v) for v in a], [mpf(v) for v in m], [mpf(v) for v in y]
    xs_ = [[mpf(v) for v in col] for col in xs]
    med_cols = [[mpf(1)] * n, a_] + xs_
    med = ols(med_cols, m_)
    p = len(med_cols)
    res = [m_[i] - sum(med[k] * med_cols[k][i] for k in range(p)) for i in range(n)]
    sigma2 = sum(r * r for r in res) / (n - p)
    center = sum(m_) / n
    scale = sqrt(sum((v - center) ** 2 for v in m_) / (n - 1))
    z = [(v - center) / scale for v in m_]
    zc = [[zi ** d for zi in z] for d in range(1, degree + 1)]
    out_cols = med_cols + zc + [[zi * ai for zi, ai in zip(col, a_)] for col in zc]
    out = ols(out_cols, y_)
    base = 2 + len(xs)
    s2 = sigma2 / scale ** 2

    def theta(a_val, a_med):
        tot = mpf(0)
        for i in range(n):
            xb = sum(out[2 + k] * xs_[k][i] for k in range(len(xs)))
            mu = med[0] + med[1] * a_med + sum(med[2 + k] * xs_[k][i] for k in range(len(xs)))
            z0 = (mu - center) / scale
            acc = out[0] + out[1] * a_val + xb
            for d in range(1, degree + 1):
                moment = sum(comb(d, 2 * j) * z0 ** (d - 2 * j) * s2 ** j * dfact(2 * j - 1)
                             for j in range(d // 2 + 1))
                acc += (out[base + d - 1] + out[base + degree + d - 1] * a_val) * moment
            tot += acc
        return tot / n

    t11, t10, t00 = theta(1, 1), theta(1, 0), theta(0, 0)
    return dict(direct=float(t10 - t00), indirect=float(t11 - t10), total=float(t11 - t00),
                med_coefficients=[float(v) for v in med], residual_variance=float(sigma2))


def make(name, n, degree, n_cov, m_offset=0.0, x_offset=0.0, y_offset=0.0, m_noise=1.0,
         skew=False, effect=1.0):
    a = [1.0 if rng.random() < 0.5 else 0.0 for _ in range(n)]
    a[0], a[1] = 1.0, 0.0
    xs = [[x_offset + rng.gauss(0, 1) for _ in range(n)] for _ in range(n_cov)]
    m = []
    for i in range(n):
        e = rng.expovariate(1.0) - 1.0 if skew else rng.gauss(0, 1)
        m.append(m_offset + 0.8 * a[i] + sum(0.3 * (xs[k][i] - x_offset) for k in range(n_cov)) + m_noise * e)
    mc = sum(m) / n
    y = []
    for i in range(n):
        z = (m[i] - mc)
        y.append(y_offset + effect * a[i] + 0.7 * z + 0.25 * z * z - 0.1 * z ** 3 * (degree >= 3)
                 + 0.4 * a[i] * z + sum(0.2 * (xs[k][i] - x_offset) for k in range(n_cov)) + 0.3 * rng.gauss(0, 1))
    case = dict(name=name, degree=degree, a=a, m=m, y=y, x=xs)
    case.update(solve(a, m, y, xs, degree))
    return case


cases = [
    make("quadratic_two_covariates", 80, 2, 2),
    make("quartic", 120, 4, 2),
    make("mediator_offset_1e6", 80, 3, 1, m_offset=1e6),
    make("covariate_offset_1e3", 80, 2, 2, x_offset=1e3),
    make("outcome_offset_1e7", 80, 2, 1, y_offset=1e7),
    make("skewed_mediator", 100, 3, 1, skew=True),
    make("tiny_mediator_noise", 80, 2, 1, m_noise=1e-3),
]
print(json.dumps(dict(cases=cases)))
