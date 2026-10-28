"""Point derivative of the Kennedy curve: local fits, bias correction and scale transforms, exactly.

Regenerate with:  python3 estimate_response_point_derivative_reference.py > estimate_response_point_derivative_reference.json

The cross-fitted pseudo-outcome and the covariate term come from the exact pipeline of
estimate_response_kennedy_dr_reference.py.  At the point `at` with the caller bandwidth h and the
Gaussian kernel w_i = exp(-u_i^2 / 2), u_i = (a_i - at) / h, the references fit by weighted least
squares, at 50 digits, polynomials of degree 2 (the conventional point: level, slope gamma_1 / h,
curvature 2 gamma_2 / h^2), degree 3 (slope) and degree 4 (level, curvature).  The bias-corrected
coordinates are the quartic level, the cubic slope and the quartic curvature; their row influences are
w_i (e_k' G^-1 x_i) r_i with G = X'WX and r_i the residual of that fit (rescaled by 1/h and 2/h^2),
and the covariance is the sum of outer products, with the marginalization term (c_i - mean c) / n
added to the level coordinate.  A transform of the coordinates takes the delta method with a
gradient from numerical differentiation of the transform's definition, except that the identity and
log-treatment first derivatives and the identity second derivative use the coordinate's own error;
the log-log first derivative takes Fieller's interval.
Repository-authored (MIT OR Apache-2.0); needs mpmath (BSD-3-Clause) at run time, not redistributed.
"""
import json

from mpmath import mp, mpf, exp, matrix, lu_solve, sqrt, erfinv, diff

import estimate_response_kennedy_dr_reference as kd
from estimate_response_kennedy_dr_reference import F, mf

mp.dps = 50


def local_poly(a, y, at, h, N):
    """Coefficients in u = (x - at)/h, the inverse Gram and the per-row (w, powers, residual)."""
    G = matrix(N, N)
    r = matrix(N, 1)
    rows = []
    for ai, yi in zip(a, y):
        u = (mf(ai) - mf(at)) / mf(h)
        w = exp(-u * u / 2)
        p = [u ** k for k in range(N)]
        rows.append((w, p, yi))
        for j in range(N):
            r[j] += w * p[j] * yi
            for k in range(N):
                G[j, k] += w * p[j] * p[k]
    gamma = lu_solve(G, r)
    Ginv = G ** -1
    infl = []
    for w, p, yi in rows:
        res = yi - sum(gamma[k] * p[k] for k in range(N))
        infl.append([w * sum(Ginv[k, j] * p[j] for j in range(N)) * res for k in range(N)])
    return gamma, infl


def transform(order, scale, a, m, d1, d2):
    if order == 1:
        return {"identity": d1, "log_treatment": a * d1, "log_outcome": d1 / m, "log_log": a * d1 / m}[scale]
    return {
        "identity": d2,
        "log_treatment": a * d1 + a * a * d2,
        "log_outcome": d2 / m - (d1 / m) ** 2,
        "log_log": a * d1 / m + a * a * (d2 / m - (d1 / m) ** 2),
    }[scale]


def build(name, a, x, y, folds, K, lam, h, at, level=mpf("0.95")):
    phi, c, _ = kd.pseudo_outcomes(a, x, y, folds, K, lam)
    n = len(a)
    hh, aa = mf(h), mf(at)
    quad, _ = local_poly(a, phi, at, h, 3)
    q_level, q_slope, q_curv = quad[0], quad[1] / hh, 2 * quad[2] / hh ** 2
    cub, ci = local_poly(a, phi, at, h, 4)
    qua, qi = local_poly(a, phi, at, h, 5)
    centre = sum(c) / n
    rows = []
    for k in range(n):
        rows.append([qi[k][0] + (c[k] - centre) / n, ci[k][1] / hh, 2 * qi[k][2] / hh ** 2])
    rows_plain = [[qi[k][0], ci[k][1] / hh, 2 * qi[k][2] / hh ** 2] for k in range(n)]
    cov = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    cov_plain = [[sum(r[i] * r[j] for r in rows_plain) for j in range(3)] for i in range(3)]
    corr = (qua[0], cub[1] / hh, 2 * qua[2] / hh ** 2)
    z = sqrt(2) * erfinv(level)
    out = []
    for order, scale in [(1, "identity"), (1, "log_treatment"), (1, "log_outcome"), (1, "log_log"),
                         (2, "identity"), (2, "log_treatment"), (2, "log_outcome"), (2, "log_log")]:
        est = transform(order, scale, aa, q_level, q_slope, q_curv)
        center = transform(order, scale, aa, *corr)
        if (order, scale) == (1, "identity"):
            se = sqrt(cov_plain[1][1])
        elif (order, scale) == (1, "log_treatment"):
            se = abs(aa) * sqrt(cov_plain[1][1])
        elif (order, scale) == (2, "identity"):
            se = sqrt(cov_plain[2][2])
        else:
            f = lambda m_, d1_, d2_: transform(order, scale, aa, m_, d1_, d2_)
            grad = [diff(f, corr, (1, 0, 0)), diff(f, corr, (0, 1, 0)), diff(f, corr, (0, 0, 1))]
            se = sqrt(sum(grad[i] * cov[i][j] * grad[j] for i in range(3) for j in range(3)))
            # the same delta method without the marginalization term, kept to show what it changes
        entry = {"order": order, "scale": scale, "estimate": mp.nstr(est, 30), "center": mp.nstr(center, 30),
                 "se": mp.nstr(se, 30)}
        if (order, scale) == (1, "log_log"):
            m_, s_ = corr[0], corr[1]
            A = m_ ** 2 - z ** 2 * cov[0][0]
            B = -2 * aa * (m_ * s_ - z ** 2 * cov[0][1])
            C = aa ** 2 * (s_ ** 2 - z ** 2 * cov[1][1])
            disc = B * B - 4 * A * C
            if A > 0 and disc >= 0:
                entry["fieller"] = [mp.nstr((-B - sqrt(disc)) / (2 * A), 30), mp.nstr((-B + sqrt(disc)) / (2 * A), 30)]
            else:
                entry["fieller"] = None
        else:
            entry["interval"] = [mp.nstr(center - z * se, 30), mp.nstr(center + z * se, 30)]
        out.append(entry)
    return {
        "name": name, "folds": folds, "basis": K, "lambda": float(lam), "bandwidth": float(h), "at": float(at),
        "a": [str(float(v)) for v in a], "x": [str(float(v)) for v in x], "y": [str(float(v)) for v in y],
        "results": out,
        "level_variance_with_term": mp.nstr(cov[0][0], 30), "level_variance_without_term": mp.nstr(cov_plain[0][0], 30),
    }


if __name__ == "__main__":
    cases = []
    a, x, y = kd.make_data(80, F(0), F(5), F(1))
    cases.append(build("ordinary_positive_level", a, x, y, 4, 5, F(1), F(3, 4), F(1, 4)))
    a, x, y = kd.make_data(60, F(0), F(-1, 8), F(1))
    cases.append(build("level_near_zero", a, x, y, 3, 5, F(1), F(1, 2), F(1, 8)))
    a, x, y = kd.make_data(70, F(1000), F(100000), F(250))
    cases.append(build("offset_treatment_and_outcome", a, x, y, 5, 5, F(1), F(3, 4), F(1000)))
    print(json.dumps({"cases": cases}, indent=1))
