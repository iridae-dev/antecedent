# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the continuous measurement-error corrections.

Regenerate with:  uv run --script measurement_continuous_reference.py > measurement_continuous_reference.json

Written from the stated definitions at 60 digits:
  variance components   one-way random effects with groups of unequal size k_j: MSW = SSW/(N - m),
                        MSB = SSB/(m - 1), n0 = (N - sum k_j^2 / N)/(m - 1), between = (MSB - MSW)/n0,
                        xi_j = (SSW_j - (k_j - 1) MSW)/((N - m)/m), Var(MSW) = sum xi_j^2 / m^2
  validation error      s^2 of W - X with n - 1, and the variance of the mean of the squared centred
                        differences divided by n
  regression calibration  design v = (1, W, Z); M = mean(v v'), r = mean(v y); with the error variance
                        s2 and mean(1/k) = kbar the corrected coefficients solve
                        (M - s2 kbar e_W e_W') beta = r; psi_i = v_i (y_i - v_i' beta) + s2 (1/k_i) beta_W e_W;
                        covariance given s2 is A^-1 mean(psi psi') A^-1 / n with A = M - s2 kbar e e';
                        the estimated-s2 part uses c = d beta / d s2 (numerical derivative of the solve)
                        and u = A^-1 mean(psi xi), adding (c u' + u c' + c c' Var(xi)) / n, where for an
                        independent estimate Var(xi) = n Var(s2) and u = 0
  Rubin's rules         per draw OLS and its HC0 covariance; mean, within W, between B with S - 1,
                        total W + (1 + 1/S) B.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 60
mpf = mp.mpf
rng = random.Random(424242)


def components(values, sizes):
    groups, at = [], 0
    for k in sizes:
        groups.append([mpf(v) for v in values[at : at + k]])
        at += k
    N = sum(sizes)
    m = len(sizes)
    grand = sum(sum(g) for g in groups) / N
    means = [sum(g) / len(g) for g in groups]
    ssw_j = [sum((x - mu) ** 2 for x in g) for g, mu in zip(groups, means)]
    ssw = sum(ssw_j)
    ssb = sum(len(g) * (mu - grand) ** 2 for g, mu in zip(groups, means))
    msw = ssw / (N - m)
    msb = ssb / (m - 1)
    n0 = (N - sum(k * k for k in sizes) / mpf(N)) / (m - 1)
    between = (msb - msw) / n0
    nu = mpf(N - m) / m
    xi = [(s - (len(g) - 1) * msw) / nu for s, g in zip(ssw_j, groups)]
    var_msw = sum(x * x for x in xi) / (m * m)
    icc = between / (between + msw) if between + msw > 0 else None
    return {
        "within": msw, "between": between, "icc": icc, "n0": n0, "grand": grand,
        "var_within": var_msw, "xi": xi, "df": N - m,
    }


def validation(w, x):
    n = len(w)
    d = [mpf(a) - mpf(b) for a, b in zip(w, x)]
    mean = sum(d) / n
    sq = [(v - mean) ** 2 for v in d]
    var = sum(sq) / (n - 1)
    sqm = sum(sq) / n
    sqv = sum((s - sqm) ** 2 for s in sq) / (n - 1)
    return var, sqv / n


def design(w, z):
    return [[mpf(1), mpf(a), mpf(b)] for a, b in zip(w, z)]


def corrected(y, w, z, s2, kbar):
    n = len(y)
    V = design(w, z)
    d = 3
    M = mp.matrix(d, d)
    r = mp.matrix(d, 1)
    for row, yi in zip(V, y):
        for a in range(d):
            r[a] += row[a] * mpf(yi) / n
            for b in range(d):
                M[a, b] += row[a] * row[b] / n
    A = M.copy()
    A[1, 1] -= s2 * kbar
    beta = mp.lu_solve(A, r)
    naive = mp.lu_solve(M, r)
    return V, M, r, A, beta, naive


def fit(y, w, z, counts, s2, var_s2, xi):
    n = len(y)
    kbar = sum(mpf(1) / k for k in counts) / n
    s2 = mpf(s2)
    V, M, r, A, beta, naive = corrected(y, w, z, s2, kbar)
    d = 3
    meat = mp.matrix(d, d)
    nmeat = mp.matrix(d, d)
    cross = mp.matrix(d, 1)
    xi2 = mpf(0)
    for i, row in enumerate(V):
        fitted = sum(row[a] * beta[a] for a in range(d))
        nfit = sum(row[a] * naive[a] for a in range(d))
        e = mpf(y[i]) - fitted
        en = mpf(y[i]) - nfit
        psi = [row[a] * e for a in range(d)]
        psi[1] += s2 * (mpf(1) / counts[i]) * beta[1]
        for a in range(d):
            for b in range(d):
                meat[a, b] += psi[a] * psi[b] / n
                nmeat[a, b] += row[a] * row[b] * en * en / n
            if xi is not None:
                cross[a] += psi[a] * xi[i] / n
        if xi is not None:
            xi2 += xi[i] ** 2 / n
    Ainv = A ** -1
    Minv = M ** -1
    given = Ainv * meat * Ainv.T / n
    ncov = Minv * nmeat * Minv.T / n

    def beta_of(s):
        return corrected(y, w, z, s, kbar)[4]

    h = mpf(10) ** -20
    c = (beta_of(s2 + h) - beta_of(s2 - h)) / (2 * h)
    if xi is not None:
        u = Ainv * cross
        vxi = xi2
    else:
        u = mp.matrix(d, 1)
        vxi = n * mpf(var_s2)
    total = given.copy()
    for i in range(d):
        for j in range(d):
            total[i, j] += (c[i] * u[j] + u[i] * c[j] + c[i] * c[j] * vxi) / n
    # implied reliability: 1 - s2 kbar [M^-1]_WW
    rel = 1 - s2 * kbar * Minv[1, 1]
    return {
        "beta": [float(b) for b in beta], "naive": [float(b) for b in naive],
        "cov": [float(total[i, j]) for i in range(d) for j in range(d)],
        "given": [float(given[i, j]) for i in range(d) for j in range(d)],
        "naive_cov": [float(ncov[i, j]) for i in range(d) for j in range(d)],
        "reliability": float(rel),
    }


# --- replicate study: unbalanced ------------------------------------------------------------
sizes = [1, 2, 3, 2, 4, 1, 3, 2, 5, 2, 3, 1, 2, 3, 4, 2, 2, 3, 1, 2]
latent = [rng.gauss(1.0, 1.3) for _ in sizes]
values = []
for lat, k in zip(latent, sizes):
    values += [lat + rng.gauss(0, 0.6) for _ in range(k)]
comp = components(values, sizes)

# --- validation sample ------------------------------------------------------------------------
xv = [rng.gauss(0, 1) for _ in range(30)]
wv = [x + rng.gauss(0.05, 0.5) for x in xv]
val_var, val_est = validation(wv, xv)

# --- main sample with replicated W ---------------------------------------------------------
n = 150
counts = [rng.choice([1, 1, 2, 3]) for _ in range(n)]
true_x = [rng.gauss(0.5, 1.2) for _ in range(n)]
z = [rng.gauss(0, 1) + 0.3 * x for x in true_x]
y = [1.0 + 0.8 * x - 0.4 * zz + rng.gauss(0, 0.7) for x, zz in zip(true_x, z)]
rep_values, wbar = [], []
for x, k in zip(true_x, counts):
    reps = [x + rng.gauss(0, 0.8) for _ in range(k)]
    rep_values += reps
    wbar.append(sum(reps) / k)
main = components(rep_values, counts)
main_fit = fit(y, wbar, z, counts, main["within"], main["var_within"], main["xi"])
ext_fit = fit(y, wbar, z, counts, comp["within"], comp["var_within"], None)
known_fit = fit(y, wbar, z, counts, 0.64, 0.0, None)
shrunk = fit(y, wbar, z, counts, val_var, val_est, None)

# --- Rubin's rules ----------------------------------------------------------------------
S = 5
draws = [[x + rng.gauss(0, 0.5) for x in true_x] for _ in range(S)]


def ols(y, w, z):
    V = design(w, z)
    d = 3
    n = len(y)
    M = mp.matrix(d, d)
    r = mp.matrix(d, 1)
    for row, yi in zip(V, y):
        for a in range(d):
            r[a] += row[a] * mpf(yi) / n
            for b in range(d):
                M[a, b] += row[a] * row[b] / n
    beta = mp.lu_solve(M, r)
    meat = mp.matrix(d, d)
    for row, yi in zip(V, y):
        e = mpf(yi) - sum(row[a] * beta[a] for a in range(d))
        for a in range(d):
            for b in range(d):
                meat[a, b] += row[a] * row[b] * e * e / n
    Minv = M ** -1
    return beta, Minv * meat * Minv.T / n


fits = [ols(y, dr, z) for dr in draws]
mean = [sum(f[0][a] for f in fits) / S for a in range(3)]
within = [[sum(f[1][a, b] for f in fits) / S for b in range(3)] for a in range(3)]
between = [[sum((f[0][a] - mean[a]) * (f[0][b] - mean[b]) for f in fits) / (S - 1) for b in range(3)] for a in range(3)]
total = [[within[a][b] + (1 + mpf(1) / S) * between[a][b] for b in range(3)] for a in range(3)]
post_mean = [sum(dr[i] for dr in draws) / S for i in range(n)]

flat = lambda m: [float(m[a][b]) for a in range(3) for b in range(3)]
out = {
    "replicates": {"values": values, "sizes": sizes,
                   "within": float(comp["within"]), "between": float(comp["between"]),
                   "icc": float(comp["icc"]), "n0": float(comp["n0"]), "grand": float(comp["grand"]),
                   "var_within": float(comp["var_within"]), "df": comp["df"],
                   "xi": [float(v) for v in comp["xi"]]},
    "validation": {"w": wv, "x": xv, "variance": float(val_var), "estimate_variance": float(val_est)},
    "main": {"y": y, "wbar": wbar, "z": z, "counts": counts, "values": rep_values,
             "within": float(main["within"]), "var_within": float(main["var_within"]),
             "fit_main_replicates": main_fit, "fit_external": ext_fit, "fit_known": known_fit,
             "fit_validation": shrunk, "external_within": float(comp["within"]),
             "external_var_within": float(comp["var_within"]),
             "validation_variance": float(val_var), "validation_estimate": float(val_est)},
    "rubin": {"draws": draws, "mean": [float(m) for m in mean], "within": flat(within),
              "between": flat(between), "total": flat(total), "posterior_mean": [float(v) for v in post_mean]},
}
print(json.dumps(out))
