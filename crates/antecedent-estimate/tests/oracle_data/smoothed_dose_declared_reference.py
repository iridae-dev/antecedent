# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Smoothed dose-response transport score under declared sampling weights, clusters and serial
lags, cross-fitted on a stated fold assignment, at 50 digits.

Regenerate with:  uv run --script smoothed_dose_declared_reference.py > smoothed_dose_declared_reference.json

Written from the estimating equations, not from the product code.  Declared weights s_i (positive)
are normalized to mean one over all rows.  With fold ids f(i) the logistic membership
P(S = 1 | x) is fitted on the rows of the other folds by weighted maximum likelihood (score
sum_i s_i (t_i - e_i) r_i) and the outcome curve on their source rows by weighted least squares, and
each row is predicted by the model of its own fold.  With the Epanechnikov kernel K_h and
nu_hat(x; a) = integral K_h(a - t) mu_hat(t, x) dt over the window cut at the basis knots,
  n0   = sum_{S=0} s_i
  psi  = (1/n0) [ sum_{S=0} s_i nu_hat(x_i; a) + sum_{S=1} s_i odds_i K_h(a - A_i)/pi_i (Y_i - mu_hat_i) ]
and the row scores are psi_i = s_i (nu_hat_i - psi)/n0 (target) and s_i odds_i K_h/pi_i (Y_i - mu_i)/n0
(source), which sum to zero.  The influence-function standard errors are
  independent  sqrt(sum psi_i^2)                       cluster   sqrt(G/(G-1) sum_g (sum_{i in g} psi_i)^2)
  serial lag L sqrt(sum psi_t^2 + 2 sum_{l=1}^L (1 - l/(L+1)) sum_t psi_t psi_{t-l}) in row order.
Serial folds are the contiguous blocks floor(i * folds / n); cluster folds are recorded inputs (the
product's assignment, whose definition the test checks: every cluster in one fold, both roles in
every fold).  Also written: the Kish size of the weighted kernel weights s_i K_h/pi_i at each grid
dose.  Repository-authored (MIT OR Apache-2.0); mpmath is BSD-3-Clause.
"""
import json
import os
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


def dataset(n_source, n_target, shift, interleave=False):
    z, a, y, dens, src = [], [], [], [], []
    for i in range(n_source + n_target):
        in_source = (i % 3 != 0) if interleave else i < n_source
        zi = normal() if in_source else shift + normal()
        ai = draw_dose(zi) if in_source else 0.0
        yi = ai * ai + zi * (1.0 + ai / 2.0) + normal()
        z.append(zi)
        a.append(ai)
        y.append(yi if in_source else 0.0)
        dens.append(density(ai, zi) if in_source else 0.0)
        src.append(in_source)
    return z, a, y, dens, src


def basis_row(t, x, degree, knots, interactions):
    terms = [mpf(t) ** (k + 1) for k in range(degree)] + [max(mpf(t) - mpf(k), 0) for k in knots]
    row = [mpf(1)] + terms + [mpf(x)]
    if interactions:
        row += [term * mpf(x) for term in terms]
    return row


def wls(rows, y, w):
    p = len(rows[0])
    A, b = matrix(p, p), matrix(p, 1)
    for r, v, wi in zip(rows, y, w):
        for i in range(p):
            b[i] += wi * r[i] * v
            for j in range(p):
                A[i, j] += wi * r[i] * r[j]
    return lu_solve(A, b)


def wlogit(rows, t, w):
    p = len(rows[0])
    beta = [mpf(0)] * p
    for _ in range(100):
        score = [mpf(0)] * p
        info = matrix(p, p)
        for r, ti, wi in zip(rows, t, w):
            e = 1 / (1 + exp(-sum(beta[k] * r[k] for k in range(p))))
            for i in range(p):
                score[i] += wi * (ti - e) * r[i]
                for j in range(p):
                    info[i, j] += wi * e * (1 - e) * r[i] * r[j]
        step = lu_solve(info, matrix(score))
        beta = [beta[k] + step[k] for k in range(p)]
        if max(abs(s) for s in step) < mpf(10) ** -40:
            break
    return beta


def standard_error(psi, dep):
    if dep["kind"] == "independent":
        return sqrt_mp(sum(v * v for v in psi))
    if dep["kind"] == "cluster":
        sums = {}
        for v, g in zip(psi, dep["groups"]):
            sums[g] = sums.get(g, 0) + v
        G = len(sums)
        return sqrt_mp(mpf(G) / (G - 1) * sum(v * v for v in sums.values()))
    L = dep["lag"]
    n = len(psi)
    v = sum(p * p for p in psi)
    for l in range(1, L + 1):
        v += 2 * (1 - mpf(l) / (L + 1)) * sum(psi[t] * psi[t - l] for t in range(l, n))
    return sqrt_mp(v)


def sqrt_mp(v):
    return mp.sqrt(v)


def run(name, data, raw_weights, folds, n_folds, grid, h, degree, knots, interactions, dep):
    z, a, y, dens, src = data
    n = len(z)
    tot = sum(mpf(w) for w in raw_weights)
    s = [mpf(w) * n / tot for w in raw_weights]
    p_oof, coef = [None] * n, []
    for f in range(n_folds):
        train_all = [i for i in range(n) if folds[i] != f]
        beta_m = wlogit([[mpf(1), mpf(z[i])] for i in train_all], [mpf(int(src[i])) for i in train_all],
                        [s[i] for i in train_all])
        train_src = [i for i in train_all if src[i]]
        beta_o = wls([basis_row(a[i], z[i], degree, knots, interactions) for i in train_src],
                     [mpf(y[i]) for i in train_src], [s[i] for i in train_src])
        coef.append(beta_o)
        for i in range(n):
            if folds[i] == f:
                p_oof[i] = 1 / (1 + exp(-(beta_m[0] + beta_m[1] * mpf(z[i]))))
    mu = lambda f, t, x: sum(c * v for c, v in zip(coef[f], basis_row(t, x, degree, knots, interactions)))
    target = [i for i in range(n) if not src[i]]
    n0 = sum(s[i] for i in target)

    def kernel(u, bw):
        return mpf(3) / 4 * (1 - (u / bw) ** 2) / bw if abs(u) <= bw else mpf(0)

    def nu(i, center, bw):
        cuts = [mpf(center) - bw] + [mpf(k) for k in knots if center - bw < k < center + bw] + [mpf(center) + bw]
        return quad(lambda t: kernel(mpf(center) - t, bw) * mu(folds[i], t, z[i]), cuts)

    points = []
    for g in grid:
        nus = {i: nu(i, g, mpf(h)) for i in target}
        plug = sum(s[i] * nus[i] for i in target) / n0
        terms = {}
        for i in range(n):
            if src[i]:
                odds = (1 - p_oof[i]) / p_oof[i]
                terms[i] = s[i] * odds * kernel(mpf(a[i]) - mpf(g), mpf(h)) / mpf(dens[i]) * (mpf(y[i]) - mu(folds[i], a[i], z[i]))
        aug = sum(terms.values()) / n0
        est = plug + aug
        psi = [mpf(0)] * n
        for i in target:
            psi[i] = s[i] * (nus[i] - est) / n0
        for i, t in terms.items():
            psi[i] = t / n0
        kw = [s[i] * kernel(mpf(a[i]) - mpf(g), mpf(h)) / mpf(dens[i]) for i in range(n) if src[i]]
        points.append({"dose": g, "plug_in": float(plug), "augmentation": float(aug), "estimate": float(est),
                       "se": float(standard_error(psi, dep)), "psi_sum": float(sum(psi)),
                       "kernel_ess": float(sum(kw) ** 2 / sum(v * v for v in kw))})
    return {
        "name": name, "folds": folds, "n_folds": n_folds, "grid": grid, "h": h, "degree": degree,
        "knots": knots, "interactions": interactions, "weights": raw_weights,
        "groups": dep.get("groups"), "lag": dep.get("lag"),
        "covariate": z, "dose": a, "outcome": y, "density": dens, "source": src,
        "max_odds": float(max((1 - p_oof[i]) / p_oof[i] for i in range(n) if src[i])),
        "membership_min": float(min(p_oof)), "points": points,
    }


def weights_for(n, extreme_at=None):
    out = [round(0.4 + 2.2 * u01(), 3) for _ in range(n)]
    if extreme_at is not None:
        out[extreme_at] = 12.0
    return out


FOLDS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoothed_dose_declared_reference_folds.json"))) \
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoothed_dose_declared_reference_folds.json")) else {}


def blocks(n, folds):
    return [i * folds // n for i in range(n)]


def main():
    cases = []
    # Weights only, product folds recorded as in the unweighted reference.
    data = dataset(50, 40, 0.3)
    w = weights_for(90, extreme_at=7)
    cases.append(run("weights_only_extreme_weight", data, w, FOLDS.get("weights_only_extreme_weight", blocks(90, 3)), 3,
                     [1.0, 2.0, 3.0], 0.5, 2, [], True, {"kind": "independent"}))
    # Equal weights at a non-unit scale must equal the unweighted score.
    data = dataset(50, 40, 0.3)
    cases.append(run("equal_weights_scale_invariant", data, [5.0] * 90, FOLDS.get("equal_weights_scale_invariant", blocks(90, 3)), 3,
                     [1.0, 2.0], 0.5, 2, [], False, {"kind": "independent"}))
    # Clusters (whole clusters in a fold; folds recorded from the product, definition checked).
    data = dataset(50, 40, 0.3, interleave=True)
    groups = [(i * 7) % 12 for i in range(90)]
    w = weights_for(90)
    cases.append(run("clusters_weighted", data, w, FOLDS.get("clusters_weighted", [g % 3 for g in groups]), 3,
                     [1.0, 2.0, 3.0], 0.5, 2, [], True, {"kind": "cluster", "groups": groups}))
    # Serial dependence: contiguous-block folds and the Newey-West diagnostic, two lags.
    data = dataset(50, 40, 0.3, interleave=True)
    w = weights_for(90)
    cases.append(run("serial_lag_2_blocks", data, w, blocks(90, 3), 3, [1.5, 2.5], 0.6, 2, [], False,
                     {"kind": "serial", "lag": 2}))
    cases.append(run("serial_lag_5_blocks", data, [1.0] * 90, blocks(90, 3), 3, [1.5, 2.5], 0.6, 2, [], False,
                     {"kind": "serial", "lag": 5}))
    print(json.dumps({"cases": cases}, indent=1))


if __name__ == "__main__":
    main()
