# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact posterior of the Bayesian conditional-effect average contrast, in 40-digit arithmetic.

Regenerate with:  uv run --script bayesian_conditional_reference.py > bayesian_conditional_reference.json

Outcome model y = b0 + b t + g w + d t (w - wbar) + e with the modifier w and the treatment
interaction centred at the observed modifier mean wbar, so b is the average contrast over the
observed rows.  Under the conjugate prior b | s2 ~ N(0, s2 c0^2 I), s2 ~ Inv-Gamma(1e-3, 1e-3) the
coefficient vector is multivariate Student-t with mean mn and covariance bn / (an - 1) Vn
(an = 1e-3 + n / 2, bn = 1e-3 + (y'y - mn' Vn^-1 mn) / 2, Vn = (I / c0^2 + X'X)^-1).
The reported effect also carries the uncertainty of the empirical modifier distribution: each
draw averages the row contrasts  b + d (w_i - wbar)  with Dirichlet(1, ..., 1) weights u_i, i.e.
the effect is  b + d eps  with eps = sum_i u_i (w_i - wbar) independent of the coefficients,
E[eps] = 0 and, for Dirichlet(1, ..., 1) weights, Var(eps) = (sum_i (w_i - wbar)^2 / n) / (n + 1)
(the weighted variance of the centred values divided by the Dirichlet total concentration plus one).
Therefore
    E[effect] = E[b],   Var(effect) = Var(b) + (Var(d) + E[d]^2) Var(eps).
Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40
A0 = B0 = mp.mpf("0.001")


def build(n, strength):
    w = [(((i * 7 + 2) % 15) - 7) / 7.0 for i in range(n)]
    t = [1.0 if ((i * 3 + 1) % 5) < 2 else 0.0 for i in range(n)]
    y = [0.5 + 1.2 * t[i] + 0.8 * w[i] + strength * t[i] * w[i] + 0.3 * ((((i * 11 + 5) % 17) - 8) / 8.0)
         for i in range(n)]
    return w, t, y


cases = []
for name, n, c0, strength in (("heterogeneous_wide_prior", 80, 10.0, 1.5), ("homogeneous_tight_prior", 60, 0.6, 0.0)):
    w, t, y = build(n, strength)
    wbar = sum(mp.mpf(v) for v in w) / n
    rows = [[mp.mpf(1), mp.mpf(t[i]), mp.mpf(w[i]), mp.mpf(t[i]) * (mp.mpf(w[i]) - wbar)] for i in range(n)]
    X, Y = mp.matrix(rows), mp.matrix([mp.mpf(v) for v in y])
    prec = X.T * X + mp.eye(4) / mp.mpf(c0) ** 2
    Vn = prec ** -1
    mn = Vn * (X.T * Y)
    an = A0 + mp.mpf(n) / 2
    bn = B0 + ((Y.T * Y)[0] - (mn.T * prec * mn)[0]) / 2
    cov = Vn * (bn / (an - 1))
    var_eps = sum((mp.mpf(v) - wbar) ** 2 for v in w) / n / (n + 1)
    var_effect = cov[1, 1] + (cov[3, 3] + mn[3] ** 2) * var_eps
    cases.append(dict(name=name, prior_scale=c0, modifier=w, treatment=t, outcome=y,
                      mean=float(mn[1]), sd=float(mp.sqrt(var_effect)), coefficient_sd=float(mp.sqrt(cov[1, 1])),
                      interaction_mean=float(mn[3]), modifier_mean_sd=float(mp.sqrt((cov[3, 3] + mn[3] ** 2) * var_eps))))
print(json.dumps(cases, separators=(",", ":")))
