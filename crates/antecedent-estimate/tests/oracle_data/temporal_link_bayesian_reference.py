# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0", "numpy"]
# ///
"""Bayesian temporal GLM contrast: two held lags, extreme scales, draw-level moments.

Regenerate with:  uv run --script temporal_link_bayesian_reference.py > temporal_link_bayesian_reference.json

Design columns: intercept, treatment A, lag z1, lag z2, A*z1 (the regime replaces A with the
active/control level and A*z1 with level*z1, the lags stay at their observed values).  For a
Bernoulli logit or probit link the average mean-scale contrast is
C(b) = (1/n) sum_i [mu(x_i(a)'b) - mu(x_i(c)'b)].
Maximum likelihood by Newton's method (40 digits) gives the estimate C(b_hat) and the Fisher
delta-method standard error sqrt(g' (X'WX)^-1 g), W the expected weights at the observed design
and g = (1/n) sum_i [mu'(eta_a) x_i(a) - mu'(eta_c) x_i(c)].
The Laplace posterior under N(0, sd^2 I) on every coefficient is N(m, S) with m the penalised mode
and S the inverse of the negative Hessian of the log posterior (observed curvature).  The mean
and variance of C over that normal are computed by a 13-point Gauss-Hermite rule in each of the
five dimensions (13^5 nodes) on b = m + chol(S) z.  Everything but the quadrature is 40 digits;
the quadrature is in binary64.  Repository-authored (MIT OR Apache-2.0); mpmath and numpy are not
product dependencies.
"""
import json

import mpmath as mp
import numpy as np

mp.mp.dps = 40
Phi = lambda x: mp.erfc(-x / mp.sqrt(2)) / 2
phi = lambda x: mp.e ** (-x * x / 2) / mp.sqrt(2 * mp.pi)
lam = lambda u: phi(u) / Phi(u)
LINKS = {
    "logit": dict(
        mu=lambda e: 1 / (1 + mp.e ** (-e)),
        dmu=lambda e: mp.e ** (-e) / (1 + mp.e ** (-e)) ** 2,
        fisher=lambda e: (1 / (1 + mp.e ** (-e))) * (1 - 1 / (1 + mp.e ** (-e))),
        score=lambda y, e: y - 1 / (1 + mp.e ** (-e)),
        curv=lambda y, e: (1 / (1 + mp.e ** (-e))) * (1 - 1 / (1 + mp.e ** (-e))),
    ),
    "probit": dict(
        mu=Phi, dmu=phi,
        fisher=lambda e: phi(e) ** 2 / (Phi(e) * (1 - Phi(e))),
        score=lambda y, e: lam(e) if y == 1 else -lam(-e),
        curv=lambda y, e: lam(e) * (lam(e) + e) if y == 1 else lam(-e) * (lam(-e) - e),
    ),
}


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)

    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0


def make_design(n, seed, lag_offset=0.0, lag_scale=1.0, treat_share=0.5):
    g = Lcg(seed)
    rows, act, ctl, y, treat = [], [], [], [], []
    shift = 0.0 if treat_share == 0.5 else -1.2
    for _ in range(n):
        z1 = lag_offset + lag_scale * g.normal()
        z2 = lag_scale * g.normal()
        a = 1.0 if g.u() < 1 / (1 + 2.718281828 ** (-(shift + 0.3 * (z1 - lag_offset) / lag_scale))) else 0.0
        eta = -0.3 + 0.8 * a + 0.5 * (z1 - lag_offset) / lag_scale - 0.4 * z2 / lag_scale + 0.3 * a * (z1 - lag_offset) / lag_scale
        y.append(1.0 if g.u() < 1 / (1 + 2.718281828 ** (-eta)) else 0.0)
        rows.append([1.0, a, z1, z2, a * z1])
        act.append([1.0, 1.0, z1, z2, z1])
        ctl.append([1.0, 0.0, z1, z2, 0.0])
        treat.append(a)
    return rows, act, ctl, y


def newton(L, rows, y, sd=None):
    p = len(rows[0])
    b = mp.zeros(p, 1)
    for _ in range(200):
        g = mp.zeros(p, 1)
        H = mp.zeros(p, p)
        for x, yi in zip(rows, y):
            xv = mp.matrix(x)
            e = (xv.T * b)[0]
            g += L["score"](yi, e) * xv
            H += L["curv"](yi, e) * (xv * xv.T)
        if sd is not None:
            g -= b / mp.mpf(sd) ** 2
            H += mp.eye(p) / mp.mpf(sd) ** 2
        step = H ** -1 * g
        b += step
        if mp.norm(step) < mp.mpf("1e-30"):
            return b, H
    raise RuntimeError("no convergence")


def contrast(L, act, ctl, b):
    return sum(L["mu"]((mp.matrix(xa).T * b)[0]) - L["mu"]((mp.matrix(xc).T * b)[0])
               for xa, xc in zip(act, ctl)) / len(act)


def fisher_se(L, rows, act, ctl, b):
    p = len(rows[0])
    n = len(rows)
    info = mp.zeros(p, p)
    for x in rows:
        xv = mp.matrix(x)
        info += L["fisher"]((xv.T * b)[0]) * (xv * xv.T)
    grad = mp.zeros(p, 1)
    for xa, xc in zip(act, ctl):
        grad += L["dmu"]((mp.matrix(xa).T * b)[0]) * mp.matrix(xa) - L["dmu"]((mp.matrix(xc).T * b)[0]) * mp.matrix(xc)
    grad = grad / n
    return mp.sqrt((grad.T * (info ** -1) * grad)[0])


def gauss_hermite_moments(L, act, ctl, mode, cov):
    p = len(mode)
    nodes, weights = np.polynomial.hermite_e.hermegauss(13)
    weights = weights / weights.sum()
    chol = np.linalg.cholesky(np.array([[float(cov[i, j]) for j in range(p)] for i in range(p)]))
    grids = np.meshgrid(*([nodes] * p), indexing="ij")
    z = np.stack([g.ravel() for g in grids], axis=1)
    w = np.prod(np.meshgrid(*([weights] * p), indexing="ij"), axis=0).ravel()
    beta = np.array([float(v) for v in mode]) + z @ chol.T
    xa, xc = np.array(act), np.array(ctl)
    if L is LINKS["logit"]:
        mu = lambda e: 1 / (1 + np.exp(-e))
    else:
        from math import erf, sqrt
        mu = lambda e: 0.5 * (1 + np.vectorize(erf)(e / sqrt(2)))
    c = (mu(beta @ xa.T) - mu(beta @ xc.T)).mean(axis=1)
    mean = float((w * c).sum())
    var = float((w * (c - mean) ** 2).sum())
    return mean, var


def case(name, link, n, seed, sd, **kw):
    L = LINKS[link]
    rows, act, ctl, y = make_design(n, seed, **kw)
    b_hat, _ = newton(L, rows, y)
    mode, hessian = newton(L, rows, y, sd=sd)
    cov = hessian ** -1
    mean, var = gauss_hermite_moments(L, act, ctl, mode, cov)
    return {
        "name": name, "link": link, "prior_sd": sd,
        "x": [r for r in rows], "act": act, "ctl": ctl, "y": y,
        "mle_estimate": float(contrast(L, act, ctl, b_hat)),
        "mle_se": float(fisher_se(L, rows, act, ctl, b_hat)),
        "mle_coefficients": [float(v) for v in b_hat],
        "mode_contrast": float(contrast(L, act, ctl, mode)),
        "posterior_mean": mean, "posterior_sd": var ** 0.5,
    }


cases = [
    case("logit_two_lags_moderate_prior", "logit", 60, 1, 1.0),
    case("logit_two_lags_tight_prior", "logit", 60, 2, 0.15),
    case("logit_two_lags_diffuse_prior", "logit", 80, 3, 20.0),
    case("probit_two_lags", "probit", 60, 4, 2.0),
    case("logit_lag_offset_100", "logit", 60, 5, 1.0, lag_offset=100.0, lag_scale=0.5),
    case("logit_unbalanced_treatment", "logit", 80, 6, 1.0, treat_share=0.25),
]
print(json.dumps(cases))
