# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Smoothed dose-response transport score, cross-fitted on a recorded fold assignment, at 50 digits.

Regenerate with:  uv run --script smoothed_dose_reference.py > smoothed_dose_reference.json

For a grid dose a, bandwidth h and the Epanechnikov kernel K_h(u) = 0.75 (1 - (u/h)^2) / h on
|u| <= h, the estimand is psi_h(a) = E_target[ integral K_h(a - t) mu(t, x) dt ].  The estimator
with fold ids f(i) is
  psi_hat(a) = (1/n0) [ sum_{S=0} nu_hat(x_i; a) + sum_{S=1} odds_i K_h(a - A_i)/pi_i (Y_i - mu_hat(A_i, x_i)) ],
  odds_i = (1 - p_i)/p_i,
where p_i is the logistic membership P(S = 1 | x_i), mu_hat(t, x) the least-squares outcome curve on
the declared dose basis, both fitted on the rows of the other folds only (membership on every row of
those folds, the outcome on their source rows) and predicted at row i with the model of its own
fold, and nu_hat(x; a) = integral over [a - h, a + h] of K_h(a - t) mu_hat(t, x) dt.  The integral is
evaluated by adaptive quadrature at 50 digits over the pieces cut at the basis knots (the curve is a
polynomial on each piece, so it is exact to working precision).  Also written: the plug-in mean, the
augmentation and the difference of the plug-in means at h and h / 2.  The data and the recorded
fold ids are inputs of each case.  Repository-authored (MIT OR Apache-2.0); mpmath is BSD-3-Clause.
"""
import json
from math import log, sqrt, cos, pi, tanh

from mpmath import mp, mpf, matrix, lu_solve, exp, quad

mp.dps = 50
_state = 0x082EFA98EC4E6C89


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def normal():
    return sqrt(-2.0 * log(1.0 - u01())) * cos(2.0 * pi * u01())


def density(a, z):
    return (1.0 + 0.6 * tanh(z) * (a - 2.0) / 2.0) / 4.0


def draw_dose(z):
    while True:
        a = 4.0 * u01()
        if u01() * 0.4 <= density(a, z):
            return a


def dataset(n_source, n_target, shift, z_offset, y_offset):
    z, a, y, dens, src = [], [], [], [], []
    for i in range(n_source + n_target):
        in_source = i < n_source
        zi = normal() if in_source else shift + normal()
        ai = draw_dose(zi) if in_source else 0.0
        yi = ai * ai + zi * (1.0 + ai / 2.0) + normal()
        z.append(zi + z_offset)
        a.append(ai)
        y.append(yi + y_offset if in_source else 0.0)
        dens.append(density(ai, zi) if in_source else 0.0)
        src.append(in_source)
    return z, a, y, dens, src


def basis_row(t, x, degree, knots, interactions):
    terms = [mpf(t) ** (k + 1) for k in range(degree)] + [max(mpf(t) - mpf(k), 0) for k in knots]
    row = [mpf(1)] + terms + [mpf(x)]
    if interactions:
        row += [term * mpf(x) for term in terms]
    return row


def ols(rows, y):
    p = len(rows[0])
    A, b = matrix(p, p), matrix(p, 1)
    for r, v in zip(rows, y):
        for i in range(p):
            b[i] += r[i] * v
            for j in range(p):
                A[i, j] += r[i] * r[j]
    return lu_solve(A, b)


def logit(rows, t):
    p = len(rows[0])
    beta = [mpf(0)] * p
    for _ in range(100):
        score = [mpf(0)] * p
        info = matrix(p, p)
        for r, ti in zip(rows, t):
            e = 1 / (1 + exp(-sum(beta[k] * r[k] for k in range(p))))
            for i in range(p):
                score[i] += (ti - e) * r[i]
                for j in range(p):
                    info[i, j] += e * (1 - e) * r[i] * r[j]
        step = lu_solve(info, matrix(score))
        beta = [beta[k] + step[k] for k in range(p)]
        if max(abs(s) for s in step) < mpf(10) ** -40:
            break
    return beta


def run(name, data, folds, n_folds, grid, h, degree, knots, interactions):
    z, a, y, dens, src = data
    n = len(z)
    p_oof, coef = [None] * n, []
    for f in range(n_folds):
        train_all = [i for i in range(n) if folds[i] != f]
        beta_m = logit([[mpf(1), mpf(z[i])] for i in train_all], [mpf(int(src[i])) for i in train_all])
        train_src = [i for i in train_all if src[i]]
        beta_o = ols([basis_row(a[i], z[i], degree, knots, interactions) for i in train_src],
                     [mpf(y[i]) for i in train_src])
        coef.append(beta_o)
        for i in range(n):
            if folds[i] == f:
                p_oof[i] = 1 / (1 + exp(-(beta_m[0] + beta_m[1] * mpf(z[i]))))
    mu = lambda f, t, x: sum(c * v for c, v in zip(coef[f], basis_row(t, x, degree, knots, interactions)))
    target = [i for i in range(n) if not src[i]]
    n0 = len(target)

    def kernel(u, bw):
        return mpf(3) / 4 * (1 - (u / bw) ** 2) / bw if abs(u) <= bw else mpf(0)

    def nu(i, center, bw):
        cuts = [mpf(center) - bw] + [mpf(k) for k in knots if center - bw < k < center + bw] + [mpf(center) + bw]
        return quad(lambda t: kernel(mpf(center) - t, bw) * mu(folds[i], t, z[i]), cuts)

    points = []
    for g in grid:
        plug = sum(nu(i, g, mpf(h)) for i in target) / n0
        plug_half = sum(nu(i, g, mpf(h) / 2) for i in target) / n0
        aug = mpf(0)
        for i in range(n):
            if src[i]:
                odds = (1 - p_oof[i]) / p_oof[i]
                aug += odds * kernel(mpf(a[i]) - mpf(g), mpf(h)) / mpf(dens[i]) * (mpf(y[i]) - mu(folds[i], a[i], z[i]))
        aug /= n0
        points.append({"dose": g, "plug_in": float(plug), "augmentation": float(aug),
                       "estimate": float(plug + aug), "half_bandwidth_difference": float(plug - plug_half)})
    return {
        "name": name, "folds": folds, "n_folds": n_folds, "grid": grid, "h": h, "degree": degree,
        "knots": knots, "interactions": interactions,
        "covariate": z, "dose": a, "outcome": y, "density": dens, "source": src,
        "membership_min": float(min(p_oof)), "min_membership": 0.05 if min(p_oof) >= 0.05 else float(min(p_oof)) / 2,
        "max_odds": float(max((1 - p_oof[i]) / p_oof[i] for i in range(n) if src[i])), "points": points,
    }


# Fold ids are recorded inputs (the product's stratified assignment, pinned by the test).
FOLDS = json.load(open("smoothed_dose_reference_folds.json"))
cases = [
    run("quadratic_interactions", dataset(60, 45, 0.3, 0.0, 0.0), FOLDS["quadratic_interactions"], 3,
        [1.0, 2.0, 3.0], 0.5, 2, [], True),
    run("hinge_knots_inside_window", dataset(60, 45, 0.3, 0.0, 0.0), FOLDS["hinge_knots_inside_window"], 3,
        [1.5, 2.0], 0.9, 1, [1.4, 2.6], False),
    run("extreme_odds_weights", dataset(40, 90, 0.9, 0.0, 0.0), FOLDS["extreme_odds_weights"], 3,
        [2.0], 0.6, 2, [], False),
    run("offsets", dataset(60, 45, 0.3, 50.0, 1.0e6), FOLDS["offsets"], 3, [1.0, 3.0], 0.5, 2, [], True),
]
print(json.dumps({"cases": cases}, indent=1))
