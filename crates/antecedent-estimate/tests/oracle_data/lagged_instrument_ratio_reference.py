# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact lagged-instrument Wald contrast and reduced-form ratio posterior, in 40-digit arithmetic.

Regenerate with:  uv run --script lagged_instrument_ratio_reference.py > lagged_instrument_ratio_reference.json

The instrument at time t - lag is paired with the treatment and outcome at t (n_a = n - lag rows).
Wald: with centred sums Szz, Szt, Szy, effect = Szy / Szt, first-stage slope Szt / Szz,
first-stage F = slope^2 Szz / (RSS_t / (n_a - 2)), homoskedastic SE of the effect
sqrt(RSS_s / (n_a - 2) * Szz / Szt^2) with RSS_s the residual sum of squares of y - a - effect t.
Posterior: the reduced forms t = a + pi z + u, y = c + gamma z + w are fitted by OLS; their
residual covariance S (divisor n_a - 2) is the plug-in, and with theta = (a, pi, c, gamma) and
independent N(0, tau^2) priors the posterior precision is  kron(S^-1, X'X) + I / tau^2  with
X = [1, z], and its mean solves precision * m = (S^-1 kron X') (t, y).  The effect is gamma / pi.
Because (pi, gamma) is bivariate Normal, P(gamma / pi <= x) = P(pi > 0, gamma - x pi <= 0)
+ P(pi < 0, gamma - x pi >= 0) is a one-dimensional integral of the Normal density of pi times the
conditional Normal CDF of gamma, evaluated here by adaptive quadrature; no draw is involved.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40


def series(n, seed_shift):
    z = [(((i * 13 + seed_shift) % 19) - 9) / 9.0 for i in range(n)]
    t, y = [], []
    for i in range(n):
        lagged = z[i - 2] if i >= 2 else 0.0
        e1 = (((i * 7 + 3) % 17) - 8) / 8.0
        e2 = (((i * 5 + 1) % 13) - 6) / 6.0
        ti = 0.4 + 0.9 * lagged + 0.5 * e1
        t.append(ti)
        y.append(1.0 + 1.7 * ti + 0.6 * e1 + 0.4 * e2)
    return z, t, y


def phi(x):
    return mp.npdf(x)


def reference(z, t, y, lag, tau):
    za, ta, ya = [mp.mpf(v) for v in z[:-lag]], [mp.mpf(v) for v in t[lag:]], [mp.mpf(v) for v in y[lag:]]
    n = len(za)
    mz, mt, my = sum(za) / n, sum(ta) / n, sum(ya) / n
    szz = sum((a - mz) ** 2 for a in za)
    szt = sum((a - mz) * (b - mt) for a, b in zip(za, ta))
    szy = sum((a - mz) * (b - my) for a, b in zip(za, ya))
    effect = szy / szt
    slope = szt / szz
    a1 = mt - slope * mz
    rss_t = sum((b - a1 - slope * a) ** 2 for a, b in zip(za, ta))
    f = slope ** 2 * szz / (rss_t / (n - 2))
    intercept = my - effect * mt
    rss_s = sum((c - intercept - effect * b) ** 2 for b, c in zip(ta, ya))
    se = mp.sqrt(rss_s / (n - 2) * szz / szt ** 2)
    gamma_hat = szy / szz
    c1 = my - gamma_hat * mz
    ut = [b - a1 - slope * a for a, b in zip(za, ta)]
    uy = [c - c1 - gamma_hat * a for a, c in zip(za, ya)]
    s = mp.matrix([[sum(u * u for u in ut), sum(u * v for u, v in zip(ut, uy))],
                   [sum(u * v for u, v in zip(ut, uy)), sum(v * v for v in uy)]]) / (n - 2)
    sinv = s ** -1
    X = mp.matrix([[1, a] for a in za])
    xtx, xt = X.T * X, X.T
    prec = mp.zeros(4, 4)
    rhs = mp.zeros(4, 1)
    w = [mp.matrix(ta), mp.matrix(ya)]
    for i in range(2):
        for j in range(2):
            for r in range(2):
                for c in range(2):
                    prec[2 * i + r, 2 * j + c] = sinv[i, j] * xtx[r, c]
            part = xt * w[j]
            for r in range(2):
                rhs[2 * i + r] += sinv[i, j] * part[r]
    for k in range(4):
        prec[k, k] += 1 / mp.mpf(tau) ** 2
    cov = prec ** -1
    m = cov * rhs
    mu_p, mu_g = m[1], m[3]
    vp, vg, cpg = cov[1, 1], cov[3, 3], cov[1, 3]
    cond_sd = mp.sqrt(vg - cpg ** 2 / vp)

    def cdf_ratio(x):
        x = mp.mpf(x)

        def below(p):
            cm = mu_g + cpg * (p - mu_p) / vp
            return phi((p - mu_p) / mp.sqrt(vp)) / mp.sqrt(vp) * mp.ncdf((x * p - cm) / cond_sd)

        def above(p):
            cm = mu_g + cpg * (p - mu_p) / vp
            return phi((p - mu_p) / mp.sqrt(vp)) / mp.sqrt(vp) * (1 - mp.ncdf((x * p - cm) / cond_sd))

        scale = mp.sqrt(vp)
        pts = [mu_p - 12 * scale, 0, mu_p + 12 * scale]
        lo = mp.quad(above, [pts[0], 0])
        hi = mp.quad(below, [0, pts[2]])
        return lo + hi

    ratio_centre = mu_g / mu_p
    grid = [ratio_centre + k * abs(ratio_centre) * 0.06 for k in (-3, -1, 0, 1, 3)]
    return dict(
        aligned_rows=n, effect=float(effect), intercept=float(intercept), first_stage_slope=float(slope),
        first_stage_f=float(f), homoskedastic_se=float(se),
        posterior_mean=[float(v) for v in m],
        cdf=[dict(x=float(x), p=float(cdf_ratio(x))) for x in grid])


cases = []
for name, n, lag, tau, shift in (("lag_two_wide_prior", 90, 2, 10.0, 0), ("lag_one_tight_prior", 70, 1, 1.5, 4)):
    z, t, y = series(n, shift)
    cases.append(dict(name=name, lag=lag, prior_sd=tau, instrument=z, treatment=t, outcome=y,
                      **reference(z, t, y, lag, tau)))
print(json.dumps(cases, separators=(",", ":")))
