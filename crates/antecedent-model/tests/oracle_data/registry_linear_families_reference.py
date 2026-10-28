# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Penalised and expanded linear-Gaussian mechanism fits, solved exactly in 60-digit arithmetic.

Regenerate with:  uv run --script registry_linear_families_reference.py > registry_linear_families_reference.json

Every fit is the closed-form minimiser of a penalised weighted least-squares objective, solved
from the normal equations in mp arithmetic (no QR, no pseudo-rows):

  bvar            min sum_i (y_i - b0 - sum_j b_j x_ij / s_j)^2 + (1/phi) sum_j b_j^2,  phi = 0.2,
                  s_j the population standard deviation of parent j (1 when it is not above 1e-12),
                  intercept unpenalised; coefficients returned as b_j / s_j.
  hierarchical    lambda = clamp(sigma0^2 / tau2, 1e-6, 1e6) with sigma0 the OLS residual scale,
                  tau2 = max(mean(b_ols^2) - sigma0^2 / n, 1e-8); then min rss + lambda sum b_j^2
                  on the raw parents.  With a UnitId column the outcome and every parent are
                  demeaned by unit, lambda is re-estimated on the demeaned OLS, the slopes are the
                  penalised within fit, the intercept is mean(y) - sum b_j mean(x_j), and sigma is
                  the residual scale on the raw rows over n - (1 + p) degrees of freedom.
  interactions    [1, z1, z2, z1 z2] with z the parent standardised by its population mean and
                  standard deviation; weighted least squares, weights rescaled to mean one.
  spline          [1 | z1 z1^2 z1^3 (z1-k)+^3 x3 | same for z2 | z1*{z2 columns} | z2*{z1^2 ...}]
                  with knots at the 0.25/0.5/0.75 order statistics of the standardised parent
                  (index round((n-1) q)), minimising weighted rss + 1e-6 n sum_{c>=1} (b_c s_c)^2 where
                  s_c is the root-mean-square of column c (a conditioning ridge on the scaled
                  columns, intercept unpenalised).

sigma is sqrt(weighted rss on the data rows / max(n - ncols, 1)) floored at 1e-8 rms(y).
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60


def mpf(f):
    f = F(f)
    return mp.mpf(f.numerator) / f.denominator


def solve(A, b):
    return mp.lu_solve(A, b)


def moments(col):
    n = len(col)
    m = sum(col) / n
    sd = mp.sqrt(sum((v - m) ** 2 for v in col) / n)
    return m, (sd if sd > mp.mpf("1e-12") else mp.mpf(1))


def wls_solve(D, y, w, penalty):
    """min sum_i w_i (y_i - D_i b)^2 + sum_c penalty_c b_c^2 ."""
    n, k = len(D), len(D[0])
    A = mp.zeros(k, k)
    r = mp.zeros(k, 1)
    for i in range(n):
        for a in range(k):
            r[a] += w[i] * D[i][a] * y[i]
            for b in range(k):
                A[a, b] += w[i] * D[i][a] * D[i][b]
    for c in range(k):
        A[c, c] += penalty[c]
    return [x for x in solve(A, r)]


def sigma_of(rss, n, k, y):
    rms = mp.sqrt(sum(v * v for v in y) / n)
    floor = max(mp.mpf("1e-8") * rms, mp.mpf(2) ** -1022)
    return max(mp.sqrt(rss / max(n - k, 1)), floor)


def data_rss(D, y, w, beta):
    return sum(w[i] * (y[i] - sum(beta[a] * D[i][a] for a in range(len(beta)))) ** 2 for i in range(len(y)))


def fl(v):
    return float(v)


def mean_one(weights, n):
    if weights is None:
        return [mp.mpf(1)] * n
    tot = sum(mpf(F(v)) for v in weights)
    return [mpf(F(v)) * n / tot for v in weights]


def raw_design(cols, n):
    return [[mp.mpf(1)] + [c[i] for c in cols] for i in range(n)]


def ols_lambda(cols, y, lo, hi, n):
    D = raw_design(cols, n)
    w = [mp.mpf(1)] * n
    beta = wls_solve(D, y, w, [0] * len(D[0]))
    sigma = sigma_of(data_rss(D, y, w, beta), n, len(D[0]), y)
    coeffs = beta[1:]
    mean_b2 = sum(b * b for b in coeffs) / max(len(coeffs), 1)
    tau2 = max(mean_b2 - sigma * sigma / n, mp.mpf("1e-8"))
    lam = min(max(sigma * sigma / tau2, mp.mpf(lo)), mp.mpf(hi))
    return lam


def bvar(cols, y):
    n = len(y)
    phi = mp.mpf("0.2")
    sc = [moments(c)[1] for c in cols]
    D = [[mp.mpf(1)] + [cols[j][i] / sc[j] for j in range(len(cols))] for i in range(n)]
    beta = wls_solve(D, y, [mp.mpf(1)] * n, [0] + [1 / phi] * len(cols))
    coeffs = [beta[1 + j] / sc[j] for j in range(len(cols))]
    D0 = raw_design(cols, n)
    full = [beta[0]] + coeffs
    sigma = sigma_of(data_rss(D0, y, [mp.mpf(1)] * n, full), n, len(D0[0]), y)
    return {"intercept": fl(beta[0]), "coeffs": [fl(c) for c in coeffs], "sigma": fl(sigma)}


def hierarchical(cols, y, groups):
    n = len(y)
    p = len(cols)
    lam = ols_lambda(cols, y, "1e-6", "1e6", n)
    w1 = [mp.mpf(1)] * n
    if groups is None:
        D = raw_design(cols, n)
        beta = wls_solve(D, y, w1, [0] + [lam] * p)
        sigma = sigma_of(data_rss(D, y, w1, beta), n, p + 1, y)
        return {"intercept": fl(beta[0]), "coeffs": [fl(b) for b in beta[1:]], "sigma": fl(sigma),
                "shrinkage": fl(lam)}

    def demean(v):
        out = []
        for i in range(n):
            members = [j for j in range(n) if groups[j] == groups[i]]
            out.append(v[i] - sum(v[j] for j in members) / len(members))
        return out

    yw = demean(y)
    cw = [demean(c) for c in cols]
    lam = ols_lambda(cw, yw, "1e-6", "1e6", n)
    Dw = raw_design(cw, n)
    beta = wls_solve(Dw, yw, w1, [0] + [lam] * p)
    coeffs = beta[1:]
    intercept = sum(y) / n - sum(coeffs[j] * sum(cols[j]) / n for j in range(p))
    D = raw_design(cols, n)
    sigma = sigma_of(data_rss(D, y, w1, [intercept] + coeffs), n, p + 1, y)
    return {"intercept": fl(intercept), "coeffs": [fl(b) for b in coeffs], "sigma": fl(sigma),
            "shrinkage": fl(lam)}


def basis_columns(cols, n, kind):
    """Basis columns (without the intercept) of the interaction or spline expansion."""
    ms = [moments(c) for c in cols]
    z = [[(c[i] - ms[j][0]) / ms[j][1] for i in range(n)] for j, c in enumerate(cols)]
    if kind == "interactions":
        return [z[0], z[1], [z[0][i] * z[1][i] for i in range(n)]]
    knots = []
    for j in range(2):
        vals = sorted(z[j])
        distinct = [vals[0]]
        for v in vals[1:]:
            if abs(v - distinct[-1]) >= mp.mpf("1e-9"):
                distinct.append(v)
        assert len(distinct) >= 5
        ks = []
        for q in (mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("0.75")):
            idx = int(mp.floor((len(vals) - 1) * q + mp.mpf("0.5")))
            k = vals[idx]
            if not ks or abs(k - ks[-1]) >= mp.mpf("1e-9"):
                ks.append(k)
        assert len(ks) == 3
        knots.append(ks)
    per = []
    for j in range(2):
        cs = [z[j], [v ** 2 for v in z[j]], [v ** 3 for v in z[j]]]
        for k in knots[j]:
            cs.append([(v - k) ** 3 if v - k > 0 else mp.mpf(0) for v in z[j]])
        per.append(cs)
    terms = per[0] + per[1]
    for right in per[1]:
        terms.append([per[0][0][i] * right[i] for i in range(n)])
    for r, right in enumerate(per[0]):
        if r == 0:
            continue
        terms.append([per[1][0][i] * right[i] for i in range(n)])
    return terms


def basis_fit(cols, y, weights, kind):
    n = len(y)
    terms = basis_columns(cols, n, kind)
    k = 1 + len(terms)
    w = mean_one(weights, n)
    ridge = mp.mpf("1e-6") * n if kind == "spline" else mp.mpf(0)
    scale = [mp.mpf(1)]
    for t in terms:
        rms = mp.sqrt(sum(v * v for v in t) / n)
        scale.append(rms if rms > mp.mpf("1e-12") else mp.mpf(1))
    D = [[mp.mpf(1)] + [terms[c][i] / scale[1 + c] for c in range(len(terms))] for i in range(n)]
    scaled = wls_solve(D, y, w, [0] + [ridge] * len(terms))
    beta = [scaled[c] / scale[c] for c in range(k)]
    Dfull = [[mp.mpf(1)] + [terms[c][i] for c in range(len(terms))] for i in range(n)]
    sigma = sigma_of(data_rss(Dfull, y, w, beta), n, k, y)
    return {
        "intercept": fl(beta[0]), "coeffs": [fl(b) for b in beta[1:]], "sigma": fl(sigma),
        "fitted": [fl(sum(beta[a] * Dfull[i][a] for a in range(k))) for i in range(n)],
        "n_terms": len(terms),
    }


def main():
    rng = random.Random(23)

    def g(lo, hi, den):
        return F(rng.randint(lo * den, hi * den), den)

    def cols_for(n, scale1=1, offset1=0):
        return [[g(-3, 3, 16) * scale1 + offset1 for _ in range(n)], [g(-2, 2, 16) for _ in range(n)]]

    cases = []
    # shrinkage families
    for name, n, off, noise in [("moderate", 30, 0, 1), ("offset_scale", 28, 2000, 1), ("strong_signal", 32, 0, 0)]:
        c = cols_for(n, 3 if off else 1, off)
        y = [F(3, 2) * a - F(1, 2) * b + F(rng.randint(-8, 8), 8) * noise + F(rng.randint(-4, 4), 64)
             for a, b in zip(c[0], c[1])]
        fc = [[float(v) for v in col] for col in c]
        fy = [float(v) for v in y]
        mc = [[mpf(F(v)) for v in col] for col in fc]
        my = [mpf(F(v)) for v in fy]
        cases.append({"name": name, "family": "shrinkage", "x": fc, "y": fy,
                      "bvar": bvar(mc, my), "hierarchical": hierarchical(mc, my, None)})
    # unit structure: six units of six rows
    n = 36
    groups = [i % 6 for i in range(n)]
    unit_shift = [F(5), F(-3), F(0), F(8), F(-6), F(2)]
    c = cols_for(n)
    c[0] = [v + unit_shift[groups[i]] / 2 for i, v in enumerate(c[0])]
    y = [F(7, 5) * a - F(1, 2) * b + unit_shift[groups[i]] + F(rng.randint(-8, 8), 8)
         for i, (a, b) in enumerate(zip(c[0], c[1]))]
    fc = [[float(v) for v in col] for col in c]
    fy = [float(v) for v in y]
    mc = [[mpf(F(v)) for v in col] for col in fc]
    my = [mpf(F(v)) for v in fy]
    cases.append({"name": "unit_structure", "family": "grouped", "x": fc, "y": fy, "groups": groups,
                  "hierarchical": hierarchical(mc, my, groups)})
    # interactions (n = 40) and spline (n = 250), unweighted and weighted
    for kind, n in [("interactions", 60), ("spline", 320)]:
        c = [[g(-3, 3, 32) for _ in range(n)], [g(-2, 2, 32) for _ in range(n)]]
        y = [F(1, 2) * a + F(1, 3) * b + F(3, 4) * a * b + F(1, 8) * a ** 3 + F(rng.randint(-8, 8), 8)
             for a, b in zip(*c)]
        fc = [[float(v) for v in col] for col in c]
        fy = [float(v) for v in y]
        mc = [[mpf(F(v)) for v in col] for col in fc]
        my = [mpf(F(v)) for v in fy]
        weights = [float(rng.choice([1, 2, 3, 5, 8])) for _ in range(n)]
        cases.append({"name": kind, "family": kind, "x": fc, "y": fy, "weights": weights,
                      "unweighted": basis_fit(mc, my, None, kind),
                      "weighted": basis_fit(mc, my, weights, kind)})
    print(json.dumps(cases, indent=1))


main()
