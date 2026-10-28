# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact posterior of the quadratic-basis g-computation ATE and row effects, in 40-digit arithmetic.

Regenerate with:  uv run --script bayesian_basis_reference.py > bayesian_basis_reference.json

Basis for covariates z_1..z_k: (1, t, z_j, t z_j, z_j^2), p = 2 + 3k columns, in the order of the
estimator (1, t, z_1..z_k, t z_1..t z_k, z_1^2..z_k^2).  Under b | s2 ~ N(0, s2 c0^2 I),
s2 ~ Inv-Gamma(1e-3, 1e-3) the coefficient vector is multivariate Student-t with mean
mn = Vn X'y and covariance bn / (an - 1) Vn (Vn = (I / c0^2 + X'X)^-1, an = 1e-3 + n / 2,
bn = 1e-3 + (y'y - mn' Vn^-1 mn) / 2).  A posterior functional a'b therefore has mean a'mn and
variance a' Cov a.  The row effect at row r is b_t + sum_j b_tz_j z_rj, the ATE is its average over
the observed rows, and the control outcome at row r is b_0 + sum_j (b_z_j z_rj + b_zz_j z_rj^2).
Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40


def build(n):
    z = [[(((i * (7 + 2 * j) + 3 * j) % 19) - 9) / 9.0 for i in range(n)] for j in range(2)]
    t = [1.0 if ((i * 3 + 2) % 7) < 3 else 0.0 for i in range(n)]
    y = [0.5 + 1.8 * t[i] + 0.7 * z[0][i] - 0.4 * z[1][i] + 1.1 * t[i] * z[0][i] + 0.5 * z[0][i] ** 2
         + 0.25 * ((((i * 11 + 1) % 13) - 6) / 6.0) for i in range(n)]
    return z, t, y


def case(name, n, c0, rows):
    z, t, y = build(n)
    k = len(z)
    cols = [[1.0] * n, t, *z, *[[t[i] * z[j][i] for i in range(n)] for j in range(k)],
            *[[z[j][i] ** 2 for i in range(n)] for j in range(k)]]
    X = mp.matrix([[mp.mpf(c[i]) for c in cols] for i in range(n)])
    Y = mp.matrix([mp.mpf(v) for v in y])
    p = X.cols
    prec = X.T * X + mp.eye(p) / mp.mpf(c0) ** 2
    vn = prec ** -1
    mn = vn * (X.T * Y)
    an = mp.mpf("0.001") + mp.mpf(n) / 2
    bn = mp.mpf("0.001") + ((Y.T * Y)[0] - (mn.T * prec * mn)[0]) / 2
    cov = vn * (bn / (an - 1))

    def functional(a):
        a = mp.matrix(a)
        return float((a.T * mn)[0]), float(mp.sqrt((a.T * cov * a)[0]))

    def effect_vector(zs):
        a = [mp.mpf(0)] * p
        a[1] = mp.mpf(1)
        for j in range(k):
            a[2 + k + j] = mp.mpf(zs[j])
        return a

    zbar = [sum(mp.mpf(v) for v in z[j]) / n for j in range(k)]
    ate = functional(effect_vector(zbar))
    row_out = []
    for r in rows:
        zr = [z[j][r] for j in range(k)]
        control = [mp.mpf(0)] * p
        control[0] = mp.mpf(1)
        for j in range(k):
            control[2 + j] = mp.mpf(zr[j])
            control[2 + 2 * k + j] = mp.mpf(zr[j]) ** 2
        e = functional(effect_vector(zr))
        c = functional(control)
        row_out.append(dict(row=r, effect_mean=e[0], effect_sd=e[1], control_mean=c[0], control_sd=c[1]))
    return dict(name=name, prior_sd=c0, treatment=t, outcome=y, covariates=z, ate_mean=ate[0], ate_sd=ate[1],
                rows=row_out)


print(json.dumps([case("default_prior", 48, 2.0, [0, 7, 23, 41]), case("wide_prior", 48, 8.0, [3, 19, 30])],
                 separators=(",", ":")))
