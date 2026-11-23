# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact moments of the independent-product mediation posterior, in 40-digit arithmetic.

Regenerate with:  uv run --script bayesian_mediation_reference.py > bayesian_mediation_reference.json

Mediator mechanism M = a0 + a T + e and outcome mechanism Y = c0 + c T + b M + u, each with the
Normal-Inverse-Gamma conjugate posterior (b | s2 ~ N(0, s2 c0^2 I), s2 ~ Inv-Gamma(1e-3, 1e-3)).
The marginal of a coefficient vector is multivariate Student-t, so Cov = bn / (an - 1) * Vn, with
Vn = (I / c0^2 + X'X)^-1, mn = Vn X'y, an = 1e-3 + n / 2, bn = 1e-3 + (y'y - mn' Vn^-1 mn) / 2.
The two mechanisms are fitted independently, so a is independent of (c, b).  Writing delta for the
active-minus-control treatment gap, the composed quantities are
    mediated = delta a b,  direct = delta c,  total = delta (c + a b).
By independence E[a b] = E[a] E[b], Var(a b) = Va Vb + Va E[b]^2 + Vb E[a]^2,
Cov(c, a b) = E[a] Cov(c, b), hence Var(c + a b) = Vc + Var(a b) + 2 E[a] Cov(c, b).
Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40
A0 = B0 = mp.mpf("0.001")


def nig(X_rows, y, c0):
    n = len(y)
    X = mp.matrix(X_rows)
    Y = mp.matrix([mp.mpf(v) for v in y])
    prec = X.T * X + mp.eye(X.cols) / mp.mpf(c0) ** 2
    Vn = prec ** -1
    mn = Vn * (X.T * Y)
    an = A0 + mp.mpf(n) / 2
    bn = B0 + ((Y.T * Y)[0] - (mn.T * prec * mn)[0]) / 2
    return mn, Vn * (bn / (an - 1))


def build(n):
    t = [1.0 if ((i * 5 + 2) % 7) < 3 else 0.0 for i in range(n)]
    m = [0.3 + 0.9 * t[i] + 0.5 * ((((i * 7 + 1) % 13) - 6) / 6.0) for i in range(n)]
    y = [1.0 + 0.7 * t[i] + 1.4 * m[i] + 0.4 * ((((i * 11 + 3) % 17) - 8) / 8.0) for i in range(n)]
    return t, m, y


cases = []
for name, n, c0, delta in (("unit_gap_wide_prior", 70, 10.0, 1.0), ("gap_two_tight_prior", 70, 0.8, 2.0)):
    t, m, y = build(n)
    ma, ca = nig([[1, mp.mpf(v)] for v in t], m, c0)
    mo, co = nig([[1, mp.mpf(t[i]), mp.mpf(m[i])] for i in range(n)], y, c0)
    ea, va = ma[1], ca[1, 1]
    ec, vc = mo[1], co[1, 1]
    eb, vb = mo[2], co[2, 2]
    cov_cb = co[1, 2]
    d = mp.mpf(delta)
    var_ab = va * vb + va * eb ** 2 + vb * ea ** 2
    out = dict(
        mediated_mean=float(d * ea * eb), mediated_sd=float(d * mp.sqrt(var_ab)),
        direct_mean=float(d * ec), direct_sd=float(d * mp.sqrt(vc)),
        total_mean=float(d * (ec + ea * eb)),
        total_sd=float(d * mp.sqrt(vc + var_ab + 2 * ea * cov_cb)))
    cases.append(dict(name=name, prior_scale=c0, delta=delta, treatment=t, mediator=m, outcome=y, **out))
print(json.dumps(cases, separators=(",", ":")))
