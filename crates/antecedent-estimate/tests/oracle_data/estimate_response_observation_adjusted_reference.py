"""Selected-outcome correction composed into the Kennedy curve, exactly.

Regenerate with:  python3 estimate_response_observation_adjusted_reference.py > estimate_response_observation_adjusted_reference.json

The outcome is observed only on rows with R = 1.  The selection probability p(R = 1 | a, x) is the
logistic maximum-likelihood fit on the columns (1, a, x) (Newton's method at 40 digits), clamped into
[floor, 1 - floor] with floor 0.01.  Inverse-probability weighting fits it on every row and uses the pseudo-value
R y / p.  The augmented estimator cross-fits: row i belongs to fold i mod K, the logistic model is fitted on
the other rows and the outcome regression m(a, x) by ordinary least squares of y on (1, a, x) over the other
rows with R = 1, and the pseudo-value is m + R (y - m) / p (m for an unobserved row), with p and m evaluated
out of fold.  The pseudo-values, taken to the nearest double as the estimator stores them, replace the outcome
in the exact Kennedy pipeline of estimate_response_kennedy_dr_reference.py, whose local-quadratic level at each
grid point is reported.  The estimator publishes no interval for this composition.
Repository-authored (MIT OR Apache-2.0); needs mpmath (BSD-3-Clause) at run time, not redistributed.
"""
import json
from fractions import Fraction as F

from mpmath import mp, mpf, matrix, lu_solve, exp

import estimate_response_kennedy_dr_reference as kd
from estimate_response_kennedy_dr_reference import mf

mp.dps = 40
FLOOR = mpf("0.01")


def design(a, x, rows):
    return [[mpf(1), mf(a[i]), mf(x[i])] for i in rows]


def logistic_fit(a, x, r, rows):
    X = design(a, x, rows)
    y = [mpf(int(r[i])) for i in rows]
    p = 3
    beta = [mpf(0)] * p
    for _ in range(100):
        grad = [mpf(0)] * p
        hess = matrix(p, p)
        for row, yi in zip(X, y):
            eta = sum(b * v for b, v in zip(beta, row))
            mu = 1 / (1 + exp(-eta))
            for j in range(p):
                grad[j] += (yi - mu) * row[j]
                for k in range(p):
                    hess[j, k] += mu * (1 - mu) * row[j] * row[k]
        step = lu_solve(hess, matrix(grad))
        beta = [beta[j] + step[j] for j in range(p)]
        if max(abs(step[j]) for j in range(p)) < mpf(10) ** -32:
            break
    return beta


def prob(beta, row):
    eta = sum(b * v for b, v in zip(beta, row))
    return min(max(1 / (1 + exp(-eta)), FLOOR), 1 - FLOOR)


def ols(a, x, y, rows):
    X = design(a, x, rows)
    p = 3
    G, b = matrix(p, p), matrix(p, 1)
    for row, i in zip(X, rows):
        for j in range(p):
            b[j] += row[j] * mf(y[i])
            for k in range(p):
                G[j, k] += row[j] * row[k]
    return lu_solve(G, b)


def pseudo_values(a, x, y, r, mode, folds):
    n = len(a)
    out = []
    if mode == "ipw":
        beta = logistic_fit(a, x, r, range(n))
        for i in range(n):
            p_i = prob(beta, [mpf(1), mf(a[i]), mf(x[i])])
            out.append(mf(y[i]) / p_i if r[i] == 1 else mpf(0))
        return out
    out = [None] * n
    for f in range(folds):
        train = [i for i in range(n) if i % folds != f]
        valid = [i for i in range(n) if i % folds == f]
        beta = logistic_fit(a, x, r, train)
        sel = [i for i in train if r[i] == 1]
        coef = ols(a, x, y, sel)
        for i in valid:
            row = [mpf(1), mf(a[i]), mf(x[i])]
            p_i = prob(beta, row)
            m = sum(c * v for c, v in zip(coef, row))
            out[i] = m + (mf(y[i]) - m) / p_i if r[i] == 1 else m
    return out


def build(name, n, mode, selection, a_off=F(0), y_off=F(0), x_scale=F(1), obs_folds=5, kennedy_folds=3, K=5, lam=F(1), h=F(3, 5), grid=(F(-1, 5), F(0), F(1, 5))):
    x = [kd.dy(kd.u01() * 4 - 2, 1024) * x_scale for _ in range(n)]
    a = [a_off + kd.dy(float(v / x_scale) * 0.5 + (kd.u01() - 0.5) * 1.5, 1024) for v in x]
    y = [y_off + kd.dy(1.5 * float(a[i] - a_off) + 0.8 * float(x[i] / x_scale) + (kd.u01() - 0.5), 1024) for i in range(n)]
    c0, c1, c2 = selection
    r = []
    for i in range(n):
        eta = c0 + c1 * float(a[i] - a_off) + c2 * float(x[i] / x_scale)
        pr = 1 / (1 + 2.718281828459045 ** (-eta))
        r.append(1 if kd.u01() < pr else 0)
    vals = pseudo_values(a, x, y, r, mode, obs_folds)
    # the estimator stores doubles
    yq = [F(float(v)) for v in vals]
    phi, c, _ = kd.pseudo_outcomes(a, x, yq, kennedy_folds, K, lam)
    return {
        "name": name, "mode": mode, "observation_folds": obs_folds, "folds": kennedy_folds, "basis": K,
        "lambda": float(lam), "bandwidth": float(h), "grid": [float(g) for g in grid],
        "a": [str(float(v)) for v in a], "x": [str(float(v)) for v in x],
        "y": [str(float(v)) if r[i] == 1 else "nan" for i, v in enumerate(y)], "r": [float(v) for v in r],
        "adjusted": [mp.nstr(v, 30) for v in vals],
        "level": [mp.nstr(kd.local_quadratic_level(a, phi, g, h), 30) for g in grid],
    }


if __name__ == "__main__":
    cases = [
        build("aipw_ordinary", 120, "aipw", (0.2, 0.6, 0.8)),
        build("ipw_ordinary", 120, "ipw", (0.2, 0.6, 0.8)),
        build("aipw_floor_binds", 160, "aipw", (-0.5, 0.8, 2.2)),
        build("aipw_offsets_and_scales", 120, "aipw", (0.3, 0.5, 0.8), a_off=F(1000), y_off=F(100000), x_scale=F(250), grid=(F(1000), F(1001, 1) - F(3, 4))),
        build("aipw_two_folds", 100, "aipw", (0.4, 0.5, 0.6), obs_folds=2),
    ]
    print(json.dumps({"cases": cases}, indent=1))
