# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""40-digit reference for the temporal GLM decision-time contrast (logit and probit).

Regenerate with:  uv run --script temporal_link_reference.py > temporal_link_reference.json

Design columns: intercept, treatment a, a held lag z and the interaction a * z.  The treatment and
the interaction are regime columns: under the active regime a = 1 and a z = z, under the control
regime a = 0 and a z = 0; the lag keeps its observed value.  With mean function mu (logistic or
the standard normal distribution function):
  estimate = (1/n) sum_i [mu(x_i(1)' b) - mu(x_i(0)' b)]
at the Bernoulli maximum-likelihood b (Newton on the log likelihood).  The Fisher delta-method
standard error is sqrt(g' (X'WX)^-1 g) with W = expected Fisher weight at the observed design
(logit mu(1 - mu); probit phi^2 / (Phi (1 - Phi))) and
g = (1/n) sum_i [mu'(x_i(1)'b) x_i(1) - mu'(x_i(0)'b) x_i(0)].
The Gaussian-prior posterior uses the Laplace approximation: mode b* of the log posterior
(log likelihood - |b|^2 / (2 sd^2)), covariance (-Hessian)^-1 with the OBSERVED Hessian of the
log posterior, the contrast at the mode, and the exact posterior-mean contrast under N(b*, C)
(row by row: the linear predictor is a scalar Gaussian, integrated by quadrature).
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 40


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)

    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0


def s(x):
    return mp.nstr(x, 30)


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
        mu=Phi,
        dmu=phi,
        fisher=lambda e: phi(e) ** 2 / (Phi(e) * (1 - Phi(e))),
        score=lambda y, e: lam(e) if y == 1 else -lam(-e),
        curv=lambda y, e: lam(e) * (lam(e) + e) if y == 1 else lam(-e) * (lam(-e) - e),
    ),
}


def design(n, seed):
    rng = Lcg(seed)
    a, z, y = [], [], []
    for _ in range(n):
        zi = rng.normal()
        ai = 1.0 if rng.u() < 1 / (1 + mp.e ** (-0.4 * zi)) else 0.0
        eta = -0.3 + 0.9 * ai + 0.7 * zi + 0.4 * ai * zi
        yi = 1.0 if rng.u() < 1 / (1 + mp.e ** (-eta)) else 0.0
        a.append(ai), z.append(zi), y.append(yi)
    return a, z, y


def rows(a, z):
    obs = [[1, ai, zi, ai * zi] for ai, zi in zip(a, z)]
    act = [[1, 1, zi, zi] for zi in z]
    ctl = [[1, 0, zi, 0] for zi in z]
    return obs, act, ctl


def newton(L, obs, y, sd=None):
    p = 4
    b = mp.zeros(p, 1)
    for _ in range(200):
        g = mp.zeros(p, 1)
        H = mp.zeros(p, p)
        for x, yi in zip(obs, y):
            xv = mp.matrix(x)
            e = (xv.T * b)[0]
            g += L["score"](yi, e) * xv
            H += L["curv"](yi, e) * (xv * xv.T)
        if sd is not None:
            g -= b / mp.mpf(sd) ** 2
            H += mp.eye(p) / mp.mpf(sd) ** 2
        step = H ** -1 * g
        b += step
        if mp.norm(step) < mp.mpf("1e-35"):
            return b, H
    raise RuntimeError("no convergence")


def contrast(L, act, ctl, b):
    n = len(act)
    return sum(L["mu"]((mp.matrix(xa).T * b)[0]) - L["mu"]((mp.matrix(xc).T * b)[0])
               for xa, xc in zip(act, ctl)) / n


def run(link, a, z, y, sd):
    L = LINKS[link]
    obs, act, ctl = rows(a, z)
    n = len(y)
    b, _ = newton(L, obs, y)
    est = contrast(L, act, ctl, b)
    W = mp.zeros(4, 4)
    for x in obs:
        xv = mp.matrix(x)
        W += L["fisher"]((xv.T * b)[0]) * (xv * xv.T)
    cov = W ** -1
    g = mp.zeros(4, 1)
    for xa, xc in zip(act, ctl):
        va, vc = mp.matrix(xa), mp.matrix(xc)
        g += L["dmu"]((va.T * b)[0]) * va - L["dmu"]((vc.T * b)[0]) * vc
    g /= n
    se = mp.sqrt((g.T * cov * g)[0])
    bp, Hp = newton(L, obs, y, sd)
    Cp = Hp ** -1
    at_mode = contrast(L, act, ctl, bp)
    total = mp.mpf(0)
    for xa, xc in zip(act, ctl):
        for x, sign in ((xa, 1), (xc, -1)):
            xv = mp.matrix(x)
            m = (xv.T * bp)[0]
            v = (xv.T * Cp * xv)[0]
            sdv = mp.sqrt(v)
            total += sign * mp.quad(lambda t: L["mu"](m + sdv * t) * phi(t), [-12, 0, 12])
    return {
        "link": link, "prior_sd": sd,
        "mle": [s(v) for v in b], "estimate": s(est), "se": s(se),
        "map": [s(v) for v in bp], "functional_at_mode": s(at_mode),
        "posterior_mean": s(total / n),
        "posterior_covariance": [[s(Cp[i, j]) for j in range(4)] for i in range(4)],
    }


def main():
    a, z, y = design(140, 4242)
    out = {"treatment": a, "lag": z, "outcome": y, "cases": [
        run("logit", a, z, y, 2.0), run("probit", a, z, y, 2.0), run("logit", a, z, y, 0.3),
    ]}
    print(json.dumps(out))


main()
