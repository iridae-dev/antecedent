# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact bivariate-Normal ratio law of the Bayesian fuzzy-jump and kink local ratio.

Regenerate with:  uv run --script bayesian_local_ratio_reference.py > bayesian_local_ratio_reference.json

Window: rows with |R - c| <= h, design (1, A, R - c, A (R - c)) with A = 1{R >= c}.  For each response
(outcome y, treatment t) the posterior mean under the coefficient prior N(0, v prior_sd^2 I) scaled by the
plug-in residual covariance is m = L^-1 X'resp with L = X'X + I / prior_sd^2; the plug-in residual
covariance S = (1 / (n - 4)) sum_i e_i e_i' (diagonal floored at 1e-12, correlation clamped to
+-(1 - 1e-6)).  The coefficient pair (numerator, denominator) at index k (1 for the jump, 3 for the
kink) is bivariate Normal with mean (m_y[k], m_t[k]) and covariance (L^-1)_kk S.  Then
  P(N / D <= r) = int_{d>0} f_D(d) Phi((r d - mu_N(d)) / s) dd + int_{d<0} f_D(d) (1 - Phi((r d - mu_N(d)) / s)) dd,
with mu_N(d) = m_N + rho (sd_N / sd_D)(d - m_D) and s = sd_N sqrt(1 - rho^2).  The denominator
statistics are P(D > 0), P(|D| < floor), mean and sd.  Everything is mpmath at 30 digits from the
exact binary64 inputs.  Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import mpmath as mp

mp.mp.dps = 30


def ncdf(x):
    return mp.ncdf(x)


def fixture(kind, law):
    run = [(i - 200) / 100.0 for i in range(400)]
    tn = lambda i: 0.2 * (((i * 13) % 17) / 8.0 - 1.0)
    on = lambda i: 0.2 * (((i * 7) % 11) / 5.0 - 1.0)
    t = []
    for i, r in enumerate(run):
        a = 1.0 if r >= 0.0 else 0.0
        base = {"fuzzy": 0.2 + 0.6 * a + 0.1 * r, "kink": 1.0 + 0.5 * r + 0.7 * a * r,
                "weak": 0.5 + 0.02 * a + 0.1 * r}[law]
        t.append(base + tn(i))
    y = [1.0 + 0.8 * r + 1.5 * ti + on(i) for i, (r, ti) in enumerate(zip(run, t))]
    return run, t, y


def analyse(kind, law, bandwidth, prior_sd, floor, step, cutoff=0.0):
    run, t, y = fixture(kind, law)
    rows, ys, ts = [], [], []
    for r, ti, yi in zip(run, t, y):
        x = mp.mpf(r) - mp.mpf(cutoff)
        if abs(x) <= mp.mpf(bandwidth):
            a = mp.mpf(1) if r >= cutoff else mp.mpf(0)
            rows.append([mp.mpf(1), a, x, a * x])
            ys.append(mp.mpf(yi))
            ts.append(mp.mpf(ti))
    n = len(rows)
    X = mp.matrix(rows)
    L = X.T * X + mp.eye(4) / mp.mpf(prior_sd) ** 2
    Li = L ** -1
    my, mt = Li * (X.T * mp.matrix(ys)), Li * (X.T * mp.matrix(ts))
    ey, et = mp.matrix(ys) - X * my, mp.matrix(ts) - X * mt
    syy, stt, syt = (ey.T * ey)[0] / (n - 4), (et.T * et)[0] / (n - 4), (ey.T * et)[0] / (n - 4)
    syy, stt = max(syy, mp.mpf("1e-12")), max(stt, mp.mpf("1e-12"))
    lim = (1 - mp.mpf("1e-6")) * mp.sqrt(syy * stt)
    syt = max(-lim, min(lim, syt))
    k = 1 if kind == "jump" else 3
    sc = Li[k, k]
    mN, mD = my[k], mt[k]
    vN, vD, cND = sc * syy, sc * stt, sc * syt
    sN, sD = mp.sqrt(vN), mp.sqrt(vD)
    rho = cND / (sN * sD)
    s = sN * mp.sqrt(1 - rho ** 2)
    fD = lambda d: mp.npdf(d, mD, sD)
    muN = lambda d: mN + rho * sN / sD * (d - mD)

    def cdf(r):
        r = mp.mpf(r)
        pos = mp.quad(lambda d: fD(d) * ncdf((r * d - muN(d)) / s), [0, mD, mD + 6 * sD, mD + 40 * sD] if mD > 0 else [0, 6 * sD, 40 * sD])
        neg = mp.quad(lambda d: fD(d) * (1 - ncdf((r * d - muN(d)) / s)), [mD - 40 * sD, mD - 6 * sD, mD, 0] if mD < 0 else [-40 * sD, -6 * sD, 0])
        return pos + neg

    ratio_point = mN / mD
    grid = [ratio_point + mp.mpf(step) * j for j in range(-6, 7)]
    out = dict(kind=kind, law=law, bandwidth=bandwidth, prior_sd=prior_sd, floor=floor, cutoff=cutoff,
               running=run, treatment=t, outcome=y, n_window=n,
               numerator_mean=float(mN), denominator_mean=float(mD), denominator_sd=float(sD),
               denominator_positive=float(1 - ncdf(-mD / sD)),
               near_zero_mass=float(ncdf((floor - mD) / sD) - ncdf((-floor - mD) / sD)),
               ratio_grid=[float(g) for g in grid], ratio_cdf=[float(cdf(g)) for g in grid])
    out["weak"] = bool(out["near_zero_mass"] > 0.025)
    return out


cases = [
    analyse("jump", "fuzzy", 1.0, 20.0, 0.1, 0.03),
    analyse("kink", "kink", 1.0, 20.0, 0.1, 0.05),
    analyse("jump", "weak", 1.0, 20.0, 0.1, 15.0),
]
print(json.dumps(cases, separators=(",", ":")))
