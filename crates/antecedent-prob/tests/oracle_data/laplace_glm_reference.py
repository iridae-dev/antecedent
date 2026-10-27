#!/usr/bin/env python3
"""Repository-authored high-precision Laplace reference for the Bayesian GLM backend.

Computes the posterior mode and the inverse observed information at the mode for
Bernoulli-logit, Bernoulli-probit and Poisson-log models under an independent Gaussian
coefficient prior, with mpmath at 50 digits.  Likelihood derivatives are taken by
numerical differentiation of the log-likelihood itself (mpmath.diff), not from the
closed-form score/curvature the product uses.  The Newton iteration is separate code.

The data below are copied into crates/antecedent-prob/tests/bayes_posterior_oracle.rs;
printing this script's output regenerates the constants there.  Not a pinned
environment: the output is only a frozen reference, and mpmath is not a product
dependency.  Run: python3 laplace_glm_reference.py
"""
from mpmath import mp, mpf, matrix, exp, log, ncdf, diff, lu_solve, inverse

mp.dps = 50

X = [(1, -1.5), (1, -1.1), (1, -0.8), (1, -0.4), (1, -0.1), (1, 0.2), (1, 0.5),
     (1, 0.7), (1, 1.0), (1, 1.3), (1, 1.6), (1, -0.6), (1, 0.9), (1, 0.0)]
Y_BIN = [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1]
Y_CNT = [0, 1, 0, 1, 1, 2, 1, 3, 2, 4, 6, 0, 3, 1]
WEIGHTS = [1, 1, 2, 1, 0.5, 1, 1, 3, 1, 1, 1, 0, 1, 2]
OFFSETS = [0.0, 0.1, -0.1, 0.0, 0.2, 0.0, 0.3, -0.2, 0.0, 0.1, 0.0, 0.5, -0.3, 0.0]
PRIOR_MEAN = [mpf("0.25"), mpf("-0.5")]
PRIOR_VAR = [mpf("4"), mpf("2.25")]


def loglik(kind, y, eta):
    if kind == "logit":
        mu = 1 / (1 + exp(-eta))
        return y * log(mu) + (1 - y) * log(1 - mu)
    if kind == "probit":
        return log(ncdf(eta)) if y == 1 else log(ncdf(-eta))
    return y * eta - exp(eta)


def fit(kind, y, weights, offsets):
    p = 2
    rows = [[mpf(v) for v in r] for r in X]
    def eta(b, i):
        return offsets[i] + rows[i][0] * b[0] + rows[i][1] * b[1]
    def grad_hess(b):
        g = [-(b[j] - PRIOR_MEAN[j]) / PRIOR_VAR[j] for j in range(p)]
        h = matrix(p, p)
        for j in range(p):
            h[j, j] += 1 / PRIOR_VAR[j]
        for i in range(len(rows)):
            w = mpf(weights[i])
            if w == 0:
                continue
            f = lambda e, i=i: loglik(kind, y[i], e)
            e0 = eta(b, i)
            s = diff(f, e0, 1)
            c = -diff(f, e0, 2)
            for j in range(p):
                g[j] += w * s * rows[i][j]
                for k in range(p):
                    h[j, k] += w * c * rows[i][j] * rows[i][k]
        return g, h
    b = [mpf(0), mpf(0)]
    for _ in range(100):
        g, h = grad_hess(b)
        step = lu_solve(h, matrix(g))
        b = [b[j] + step[j] for j in range(p)]
        if max(abs(s) for s in step) < mpf(10) ** -40:
            break
    g, h = grad_hess(b)
    return b, inverse(h)


def show(name, kind, y, w, o):
    b, cov = fit(kind, y, w, o)
    print(name, "mode", [mp.nstr(v, 17) for v in b],
          "cov", [mp.nstr(cov[i, j], 17) for i in range(2) for j in range(2)])


if __name__ == "__main__":
    ones = [1] * len(X)
    zeros = [0.0] * len(X)
    show("logit_plain", "logit", Y_BIN, ones, zeros)
    show("logit_weighted_offset", "logit", Y_BIN, WEIGHTS, OFFSETS)
    show("probit_plain", "probit", Y_BIN, ones, zeros)
    show("probit_weighted_offset", "probit", Y_BIN, WEIGHTS, OFFSETS)
    show("poisson_plain", "poisson", Y_CNT, ones, zeros)
    show("poisson_weighted_offset", "poisson", Y_CNT, WEIGHTS, OFFSETS)
