# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference for the predictive checks, simulation-based calibration and prior sensitivity at 60 digits.

Stages (the Rust test regenerates every library draw stream and asserts it equals the frozen one):

  uv run --offline --script ppc_prior_sensitivity_reference.py data > data.json
  (Rust: PPC_REFERENCE_DATA=data.json PPC_REFERENCE_OUT=draws.json cargo test -p antecedent-validate \
         --lib ppc_prior_sensitivity_emit -- --ignored)
  uv run --offline --script ppc_prior_sensitivity_reference.py build data.json draws.json \
         > ppc_prior_sensitivity_reference.json

Written from the stated definitions, with no product code.

Posterior predictive check.  For used draw k (stride k*total//take over the stored draws) with
coefficients b, fitted means mu_i = g^-1(x_i b); a Gaussian replicate is mu_i + s z_i with
s^2 = SSR(b) / chi2 (chi2 a Gamma((n+p)/2, rate 1/2) draw: the conditional of sigma^2 given b under the
conjugate normal-inverse-gamma posterior with a diffuse prior, IG(a + (n+p)/2, b0 + (SSR + Q)/2)); a
Bernoulli replicate is 1{u < mu_i}; a Poisson replicate is Knuth's product of uniforms.  Location
statistic = mean(y_rep), dispersion = sample SD (n-1) of y_rep; Monte Carlo tails are (1 + count) /
(1 + draws) with count of replicate summaries <= (>=) the observed value; two-sided p = min(1,
2 min tail).  Residual axes use Pearson residuals r = (y - mu) / sqrt(V(mu)): skewness m3 / m2^1.5,
excess kurtosis m4 / m2^2 - 3 (population moments), correlation of r with (mu - mean mu)^2 and of r^2
with mu; the omnibus p is (1 + #{draws: dep(rep) >= dep(obs)}) / (1 + draws) with dep the largest
|stat - centre| / spread over the four statistics (centre, spread = mean, sample SD of the replicate
statistic).  Serial axis: lag-1 sum r_t r_{t-1} / sum r_t^2 of y - X b against iid standard normal
replicates.

Prior predictive check.  The Gaussian family draws sigma^2 first (known, or an inverse-gamma draw), then
b = m + sigma sqrt(V0) z, the conjugate scale V0 of the fitted prior; other families draw
b = m + sqrt(V) z.  A replicate adds the family's noise; an inverse-gamma residual prior without a
finite mean (shape <= 1) replicates fitted means only and has no upper dispersion tail.

Exact laws.  For the prior predictive of the mean, ybar_rep ~ N(xbar'm, sigma^2 (1/n + xbar' V0 xbar)) with
sigma^2 known, and a Student t with 2a degrees of freedom, location xbar'm and squared scale
(b/a)(1/n + xbar' V0 xbar) under an IG(a, b) residual prior.  The conjugate posterior has
Vn = (V0^-1 + X'X)^-1, mn = Vn (V0^-1 m + X'y), an = a + n/2, bn = b + (y'y + m'V0^-1 m - mn'Vn^-1 mn)/2.

Repository-authored (MIT OR Apache-2.0).
"""
import json
import math
import sys
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60
mpf = mp.mpf
MIN_POSITIVE = 2.2250738585072014e-308


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield F(state >> 40, 1 << 24)


def d4(x):
    return float(F(round(x * 10**4), 10**4))


def d64(x):
    return float(F(round(x * 64), 64))


# ---------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------


def gauss_data(n, seed, curvature=1.2, heavy=True):
    g = lcg(seed)
    t = [float(i % 2) for i in range(n)]
    z = [d4(2 * float(next(g)) - 1) for _ in range(n)]
    y = []
    for i in range(n):
        e = float(next(g)) - 0.5
        e = e * 3 if heavy and i % 7 == 0 else e
        y.append(d4(1.0 + 2.0 * t[i] + 0.3 * z[i] + curvature * z[i] ** 2 + e))
    return t, z, y


def binary_data(n, seed):
    g = lcg(seed)
    t = [float(i % 2) for i in range(n)]
    z = [d4(2 * float(next(g)) - 1) for _ in range(n)]
    y = []
    for i in range(n):
        p = 1 / (1 + math.exp(-(0.2 + 0.9 * t[i] + 0.7 * z[i])))
        y.append(1.0 if float(next(g)) < p else 0.0)
    return t, z, y


def count_data(n, seed):
    g = lcg(seed)
    t = [float(i % 2) for i in range(n)]
    z = [d4(2 * float(next(g)) - 1) for _ in range(n)]
    y = []
    for i in range(n):
        lam = math.exp(0.3 + 0.4 * t[i] + 0.3 * z[i])
        limit, count, prod = math.exp(-lam), 0, float(next(g))
        while prod > limit:
            count += 1
            prod *= float(next(g))
        y.append(float(count))
    return t, z, y


def data_stage():
    out = {"post": [], "prior": [], "shape": {}, "sbc": [], "sens": {}, "report_cases": []}
    t, z, y = gauss_data(40, 11)
    out["post"].append({"name": "gaussian_stride", "family": "gaussian", "t": t, "z": z, "y": y,
                        "fit_draws": 60, "fit_seed": 3, "n_sims": 45, "check_seed": 5,
                        "temporal_seed": 17, "u_count": 0})
    t, z, y = gauss_data(6, 12, heavy=False)
    out["post"].append({"name": "gaussian_small_capped", "family": "gaussian", "t": t, "z": z, "y": y,
                        "fit_draws": 30, "fit_seed": 4, "n_sims": 500, "check_seed": 6,
                        "temporal_seed": None, "u_count": 0})
    t, z, y = binary_data(30, 13)
    out["post"].append({"name": "logit", "family": "logit", "t": t, "z": z, "y": y,
                        "fit_draws": 40, "fit_seed": 5, "n_sims": 40, "check_seed": 7,
                        "temporal_seed": None, "u_count": 1300})
    t, z, y = binary_data(30, 14)
    out["post"].append({"name": "probit", "family": "probit", "t": t, "z": z, "y": y,
                        "fit_draws": 40, "fit_seed": 6, "n_sims": 25, "check_seed": 8,
                        "temporal_seed": None, "u_count": 1300})
    t, z, y = count_data(24, 15)
    out["post"].append({"name": "poisson", "family": "poisson", "t": t, "z": z, "y": y,
                        "fit_draws": 40, "fit_seed": 7, "n_sims": 30, "check_seed": 9,
                        "temporal_seed": None, "u_count": 9000})
    # Prior predictive: design [1, t, z], n = 20.
    base = {"coef_mean": [0.5, -0.3, 0.2], "coef_var": [4.0, 1.0, 2.25], "n_sims": 40}
    t, z, y = gauss_data(20, 21, heavy=False)
    out["prior"].append({**base, "name": "prior_known", "family": "gaussian", "kind": "known",
                         "sigma2": 2.5, "t": t, "z": z, "y": y, "seed": 31, "u_count": 0, "law": False})
    out["prior"].append({**base, "name": "prior_inverse_gamma", "family": "gaussian", "kind": "ig",
                         "shape": 3.5, "scale": 2.0, "t": t, "z": z, "y": y, "seed": 32, "u_count": 0,
                         "law": False})
    out["prior"].append({**base, "name": "prior_improper", "family": "gaussian", "kind": "improper",
                         "shape": 0.5, "scale": 1.0, "t": t, "z": z, "y": y, "seed": 33, "u_count": 0,
                         "law": False})
    t, z, y = binary_data(20, 22)
    out["prior"].append({**base, "name": "prior_logit", "family": "logit", "kind": "none", "t": t, "z": z,
                         "y": y, "seed": 34, "u_count": 1000, "coef_var": [0.5, 0.5, 0.5], "law": False})
    t, z, y = count_data(20, 23)
    out["prior"].append({**base, "name": "prior_poisson", "family": "poisson", "kind": "none", "t": t,
                         "z": z, "y": y, "seed": 35, "u_count": 8000, "coef_mean": [0.2, 0.0, 0.0],
                         "coef_var": [0.05, 0.05, 0.05], "n_sims": 30, "law": False})
    # Large prior-predictive runs checked against the exact mean law (no frozen draws).
    t, z, y = gauss_data(20, 24, heavy=False)
    xbar = [1.0, sum(t) / 20, sum(z) / 20]
    shift = 2.0 - (sum(y) / 20 - sum(a * b for a, b in zip(xbar, base["coef_mean"])))
    y = [d4(v + shift) for v in y]
    out["prior"].append({**base, "name": "law_known", "family": "gaussian", "kind": "known", "sigma2": 2.5,
                         "t": t, "z": z, "y": y, "seed": 41, "u_count": 0, "n_sims": 20000, "law": True})
    out["prior"].append({**base, "name": "law_inverse_gamma", "family": "gaussian", "kind": "ig",
                         "shape": 3.5, "scale": 2.0, "t": t, "z": z, "y": y, "seed": 42, "u_count": 0,
                         "n_sims": 20000, "law": True})
    # Residual-variance shape check.
    t, z, y = gauss_data(40, 51)
    out["shape"] = {"t": t, "z": z, "y": y, "fit_draws": 5000, "fit_seed": 61, "prior_scale": 10.0}
    # SBC and sensitivity share a design.
    g = lcg(71)
    n = 30
    t = [float(i % 2) for i in range(n)]
    z = [d64(2 * float(next(g)) - 1) for _ in range(n)]
    y = [d4(1 + 0.8 * t[i] + 0.5 * z[i] + 0.5 * (float(next(g)) - 0.5)) for i in range(n)]
    sbc_base = {"t": t, "z": z, "y": y, "coef_mean": [0.3, -0.4, 0.2], "coef_var": [4.0, 1.0, 2.25],
                "n_reps": 60, "n_draws": 99, "seed": 5, "fit_seed": 1}
    out["sbc"].append({**sbc_base, "name": "sbc_inverse_gamma", "kind": "ig", "shape": 3.0, "scale": 2.0})
    out["sbc"].append({**sbc_base, "name": "sbc_known", "kind": "known", "sigma2": 1.5})
    out["sens"] = {"t": t, "z": z, "y": y, "scales": [0.5, 2.0, 10.0],
                   "resolved": {"coef_mean": [0.0, 3.0, 0.0], "coef_var": [0.5, 0.05, 0.5], "shape": 3.0,
                                "scale": 2.0, "multipliers": [0.25, 1.0, 4.0]},
                   "alpha": {"source_mean": [0.0, 8.0, 0.0], "source_var": 0.05, "alphas": [0.0, 0.5, 1.0],
                             "alphas_applied": [0.8]},
                   "fit_draws": 1000, "fit_seed": 91}
    out["report_cases"] = [
        {"effect_means": [1.0, 1.2, 1.4], "original": 1.2, "max_relative_range": 0.5},
        {"effect_means": [-2.0, 3.0, 0.5], "original": 1.0, "max_relative_range": 0.5},
        {"effect_means": [0.0, 0.0], "original": 0.0, "max_relative_range": 0.5},
        {"effect_means": [1e300, -1e300], "original": 1e300, "max_relative_range": 3.0},
        {"effect_means": [2.0, 2.5], "original": 2.0, "max_relative_range": 0.25},
        {"effect_means": [2.0, 2.5], "original": 2.0, "max_relative_range": 0.2500001},
        {"effect_means": [], "original": 1.0, "max_relative_range": 0.5},
    ]
    return out


# ---------------------------------------------------------------------------------------------
# exact helpers
# ---------------------------------------------------------------------------------------------


def design(t, z):
    n = len(t)
    return [[mpf(1), mpf(t[i]), mpf(z[i])] for i in range(n)]


def eta_to_mean(family, eta):
    if family == "gaussian":
        return eta
    if family in ("logit",):
        return 1 / (1 + mp.exp(-eta))
    if family == "probit":
        return mp.ncdf(eta)
    return mp.exp(eta)


def pearson_scale(family, mu):
    if family == "gaussian":
        return mpf(1)
    if family in ("logit", "probit"):
        return mp.sqrt(max(mu * (1 - mu), mpf("1e-12")))
    return mp.sqrt(max(mu, mpf("1e-12")))


def mean_(v):
    return sum(v) / len(v)


def sample_sd(v):
    if len(v) < 2:
        return mpf(0)
    m = mean_(v)
    return mp.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def tails(summaries, observed):
    n = len(summaries)
    # Mathematically equal summaries tie (the SD of a binary outcome depends only on its count of
    # ones); 60-digit rounding noise must not break a tie, so equality is up to 1e-40 relative.
    tol = mpf("1e-40") * max([abs(observed)] + [abs(s) for s in summaries])
    below = sum(1 for s in summaries if s <= observed + tol)
    above = sum(1 for s in summaries if s >= observed - tol)
    return [float(F(1 + below, 1 + n)), float(F(1 + above, 1 + n))]


def two_sided(tl):
    return min(1.0, 2.0 * min(tl))


def correlation(a, b):
    ma, mb = mean_(a), mean_(b)
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    return sab / mp.sqrt(saa * sbb) if saa > 0 and sbb > 0 else mpf(0)


def residual_stats(r, curvature, mu):
    n = len(r)
    m = mean_(r)
    m2 = sum((v - m) ** 2 for v in r) / n
    m3 = sum((v - m) ** 3 for v in r) / n
    m4 = sum((v - m) ** 4 for v in r) / n
    if m2 > 0:
        skew, kurt = m3 / m2 ** mpf("1.5"), m4 / (m2 * m2) - 3
    else:
        skew, kurt = mpf(0), mpf(0)
    return [skew, kurt, correlation(r, curvature), correlation([v * v for v in r], mu)]


def knuth(lam, u, ptr):
    if lam <= 0:
        return 0.0, ptr
    remaining, count = lam, 0
    while remaining > 0:
        step = min(remaining, 500.0)
        remaining -= step
        limit = math.exp(-step)
        prod = u[ptr]
        ptr += 1
        while prod > limit:
            count += 1
            prod *= u[ptr]
            ptr += 1
    return float(count), ptr


def add_noise(family, mu_list, draws, ptr):
    """Replicate from fitted means; `draws` = {z: row normals | u: uniform stream}."""
    if family == "gaussian":
        return None, ptr
    out = []
    for mu in mu_list:
        if family in ("logit", "probit"):
            out.append(mpf(1) if draws["u"][ptr] < float(mu) else mpf(0))
            ptr += 1
        else:
            c, ptr = knuth(float(mu), draws["u"], ptr)
            out.append(mpf(c))
    return out, ptr


def summarize(kind, y, mean_s, disp_s, noise="likelihood"):
    ybar, ysd = mean_(y), sample_sd(y)
    loc = tails(mean_s, ybar)
    dis = tails(disp_s, ysd)
    if noise == "mean_only":
        dis = [dis[0], 1.0]
    return {
        "observed": float(ybar),
        "predictive_mean": float(mean_(mean_s)),
        "predictive_sd": float(sample_sd(mean_s)),
        "p_value": two_sided(loc),
        "location_tails": loc,
        "observed_dispersion": float(ysd),
        "predictive_dispersion_mean": float(mean_(disp_s)),
        "dispersion_p_value": two_sided(dis),
        "dispersion_tails": dis,
        "n_sims": len(mean_s),
        "noise": noise,
    }


def residual_family(family, y, mus, yreps):
    n = len(y)
    if n < 8:
        return [], 1.0
    obs_rows, rep_rows = [], []
    for mu, yrep in zip(mus, yreps):
        scale = [pearson_scale(family, m) for m in mu]
        mbar = mean_(mu)
        curv = [(m - mbar) ** 2 for m in mu]
        obs_rows.append(residual_stats([(a - m) / s for a, m, s in zip(y, mu, scale)], curv, mu))
        rep_rows.append(residual_stats([(a - m) / s for a, m, s in zip(yrep, mu, scale)], curv, mu))
    draws = len(mus)
    centre = [mean_([r[k] for r in rep_rows]) for k in range(4)]
    spread = [mp.sqrt(sum((r[k] - centre[k]) ** 2 for r in rep_rows) / max(draws - 1, 1)) for k in range(4)]

    def dep(row):
        return max([abs(row[k] - centre[k]) / spread[k] for k in range(4) if spread[k] > 0] + [mpf(0)])

    exceed = sum(1 for o, r in zip(obs_rows, rep_rows) if dep(r) >= dep(o))
    omnibus = float(F(1 + exceed, 1 + draws))
    names = ["residual skewness", "residual excess kurtosis", "curvature against the fitted mean",
             "variance trend against the fitted mean"]
    axes = []
    for k in range(4):
        below = sum(1 for o, r in zip(obs_rows, rep_rows) if r[k] <= o[k])
        above = sum(1 for o, r in zip(obs_rows, rep_rows) if r[k] >= o[k])
        tl = [float(F(1 + below, 1 + draws)), float(F(1 + above, 1 + draws))]
        axes.append({"name": names[k], "observed": float(mean_([o[k] for o in obs_rows])),
                     "predictive_mean": float(centre[k]), "p_value": two_sided(tl), "tails": tl})
    return axes, omnibus


def post_expected(case, d):
    family = case["family"]
    X = d["matrix"]  # row-major mpf rows built by caller
    y = [mpf(v) for v in d["outcome"]]
    n, p = len(y), len(X[0])
    betas = d["betas"]
    total = len(betas)
    take = min(total, case["n_sims"])
    mus, yreps, mean_s, disp_s = [], [], [], []
    ptr = 0
    for k in range(take):
        b = [mpf(v) for v in betas[k * total // take]]
        mu = [eta_to_mean(family, sum(X[i][c] * b[c] for c in range(p))) for i in range(n)]
        if family == "gaussian":
            ssr = sum((y[i] - mu[i]) ** 2 for i in range(n))
            sigma = mp.sqrt(ssr / max(mpf(d["chi2"][k]), mpf(MIN_POSITIVE)))
            yrep = [mu[i] + sigma * mpf(d["z"][k][i]) for i in range(n)] if sigma > 0 else list(mu)
        else:
            yrep, ptr = add_noise(family, mu, d, ptr)
        mus.append(mu)
        yreps.append(yrep)
        mean_s.append(mean_(yrep))
        disp_s.append(sample_sd(yrep))
    if family != "gaussian":
        assert ptr <= len(d["u"]), "uniform stream too short"
    rep = summarize("posterior", y, mean_s, disp_s)
    axes, omnibus = residual_family(family, y, mus, yreps)
    rep["residual"], rep["residual_p_value"] = axes, omnibus
    rep["replicates"] = [[float(v) for v in yr] for yr in yreps]
    if "serial_z" in d:
        obs_sum, rep_sum, below, above = mpf(0), mpf(0), 0, 0
        for k in range(take):
            b = [mpf(v) for v in betas[k * total // take]]
            r = [y[i] - sum(X[i][c] * b[c] for c in range(p)) for i in range(n)]
            zr = [mpf(v) for v in d["serial_z"][k]]

            def lag1(v):
                den = sum(x * x for x in v)
                return sum(v[i] * v[i - 1] for i in range(1, len(v))) / den if den > 0 else mpf(0)

            to, tr = lag1(r), lag1(zr)
            obs_sum += to
            rep_sum += tr
            below += 1 if tr <= to else 0
            above += 1 if tr >= to else 0
        tl = [float(F(1 + below, 1 + take)), float(F(1 + above, 1 + take))]
        rep["serial"] = {"observed": float(obs_sum / take), "predictive_mean": float(rep_sum / take),
                         "p_value": two_sided(tl), "tails": tl}
    return rep


def prior_expected(case, d):
    family = case["family"]
    X = d["matrix"]
    y = [mpf(v) for v in d["outcome"]]
    n, p = len(y), len(X[0])
    m = [mpf(v) for v in case["coef_mean"]]
    v0 = [mpf(v) for v in case["coef_var"]]
    mean_s, disp_s = [], []
    ptr = 0
    noise = "likelihood"
    for k in range(case["n_sims"]):
        z = [mpf(v) for v in d["z_beta"][k]]
        factor = mpf(1)
        if family == "gaussian" and case["kind"] in ("known", "ig"):
            factor = mp.sqrt(mpf(d["sigma2"][k]))
        b = [m[c] + factor * mp.sqrt(v0[c]) * z[c] for c in range(p)]
        mu = [eta_to_mean(family, sum(X[i][c] * b[c] for c in range(p))) for i in range(n)]
        if family == "gaussian":
            if case["kind"] == "improper":
                noise = "mean_only"
                yrep = mu
            else:
                sigma = mp.sqrt(mpf(d["sigma2"][k]))
                yrep = [mu[i] + sigma * mpf(d["z_noise"][k][i]) for i in range(n)] if sigma > 0 else mu
        else:
            yrep, ptr = add_noise(family, mu, d, ptr)
        mean_s.append(mean_(yrep))
        disp_s.append(sample_sd(yrep))
    if family != "gaussian":
        assert ptr <= len(d["u"]), "uniform stream too short"
    return summarize("prior", y, mean_s, disp_s, noise)


def t_cdf(x, nu):
    x, nu = mpf(x), mpf(nu)
    if x == 0:
        return mpf(1) / 2
    ib = mp.betainc(nu / 2, mpf(1) / 2, 0, nu / (nu + x * x), regularized=True)
    return 1 - ib / 2 if x > 0 else ib / 2


def conj_posterior(X, y, mean, var, a=None, b=None, sigma2=None):
    n, p = len(y), len(X[0])
    v0inv = mp.diag([1 / mpf(v) for v in var])
    xtx = mp.matrix(p, p)
    xty = mp.matrix(p, 1)
    for i in range(n):
        for c in range(p):
            xty[c] += X[i][c] * y[i]
            for e in range(p):
                xtx[c, e] += X[i][c] * X[i][e]
    prec = v0inv + xtx
    vn = prec ** -1
    m0 = mp.matrix([mpf(v) for v in mean])
    mn = vn * (v0inv * m0 + xty)
    out = {"mn": [mn[c] for c in range(p)], "vn": [[vn[c, e] for e in range(p)] for c in range(p)]}
    if a is not None:
        yy = sum(v * v for v in y)
        quad = (m0.T * v0inv * m0)[0] - (mn.T * prec * mn)[0]
        out["an"] = mpf(a) + mpf(n) / 2
        out["bn"] = mpf(b) + (yy + quad) / 2
    return out


def prior_law(case, d):
    X = d["matrix"]
    y = [mpf(v) for v in d["outcome"]]
    n, p = len(y), len(X[0])
    xbar = [sum(X[i][c] for i in range(n)) / n for c in range(p)]
    m = sum(xbar[c] * mpf(case["coef_mean"][c]) for c in range(p))
    q = mpf(1) / n + sum(xbar[c] ** 2 * mpf(case["coef_var"][c]) for c in range(p))
    ybar = mean_(y)
    if case["kind"] == "known":
        cdf = mp.ncdf((ybar - m) / mp.sqrt(mpf(case["sigma2"]) * q))
        wrong_cdf = mp.ncdf((ybar - m) / mp.sqrt(mpf(case["sigma2"]) / n + sum(
            xbar[c] ** 2 * mpf(case["coef_var"][c]) for c in range(p))))
    else:
        a, b = mpf(case["shape"]), mpf(case["scale"])
        cdf = t_cdf((ybar - m) / mp.sqrt(b / a * q), 2 * a)
        wrong_cdf = None
    return {"lower": float(cdf), "upper": float(1 - cdf),
            "convention_free_lower": None if wrong_cdf is None else float(wrong_cdf),
            "n_sims": case["n_sims"]}


def sbc_expected(case, d):
    X = d["matrix"]
    n, p = len(X), len(X[0])
    reps = case["n_reps"]
    mean = case["coef_mean"]
    var = case["coef_var"]
    kind = case["kind"]
    ranks = d["ranks"]
    L = case["n_draws"]
    fs, rep_out = [], []
    for r in range(reps):
        sigma2 = mpf(d["sigma2"][r])
        zb = [mpf(v) for v in d["z_beta"][r]]
        beta = [mpf(mean[c]) + mp.sqrt(sigma2) * mp.sqrt(mpf(var[c])) * zb[c] for c in range(p)]
        yrep = [sum(X[i][c] * beta[c] for c in range(p)) + mp.sqrt(sigma2) * mpf(d["z_noise"][r][i])
                for i in range(n)]
        theta = beta[1]
        if kind == "ig":
            post = conj_posterior(X, yrep, mean, var, a=case["shape"], b=case["scale"])
            scale = mp.sqrt(post["bn"] / post["an"] * post["vn"][1][1])
            f = t_cdf((theta - post["mn"][1]) / scale, 2 * post["an"])
        else:
            post = conj_posterior(X, yrep, mean, var)
            f = mp.ncdf((theta - post["mn"][1]) / mp.sqrt(mpf(case["sigma2"]) * post["vn"][1][1]))
        fs.append(f)
        # P(rank <= k) and P(rank >= k) for rank ~ Binomial(L, f)
        k = ranks[r]
        pmf = [mp.binomial(L, j) * f ** j * (1 - f) ** (L - j) for j in range(L + 1)]
        lo, hi = sum(pmf[: k + 1]), sum(pmf[k:])
        rep_out.append({"f": float(f), "p_le": float(lo), "p_ge": float(hi), "beta": [float(b) for b in beta],
                        "sigma2": float(sigma2), "theta": float(theta)})
    n_d = L
    fracs = [F(r_, n_d) for r_ in ranks]
    mean_frac = sum(fracs) / reps
    bins = [0] * 10
    for r_ in ranks:
        bins[min(r_ * 10 // (L + 1), 9)] += 1
    expected = F(reps, 10)
    chi2 = sum((F(c) - expected) ** 2 / expected for c in bins)
    # Hoeffding: S = sum of reps*L independent Bernoulli(f_r); delta = 1e-9
    total_mean = sum(mpf(L) * f for f in fs)
    t = mp.sqrt(mp.log(mpf(2) / mpf("1e-9")) / 2 * reps * L)
    return {"per_rep": rep_out, "mean_rank_frac": float(mean_frac), "uniformity_stat": float(chi2),
            "bins": bins, "rank_sum_expected": float(total_mean), "rank_sum_bound": float(t),
            "ranks": ranks}


def sens_expected(spec, d):
    X = d["matrix"]
    y = [mpf(v) for v in d["outcome"]]
    out = {}
    mn_list, post_sd_list, bound_list = [], [], []

    def one(mean, var, resid):
        if resid["kind"] == "ig":
            post = conj_posterior(X, y, mean, var, a=resid["shape"], b=resid["scale"])
            an, bn = post["an"], post["bn"]
            vtt = post["vn"][1][1]
            v_post = bn / (an - 1) * vtt
            # Chebyshev-type split: P(mean sigma^2 > c) <= 5e-4 (Cantelli), then a sub-Gaussian bound.
            s_mean = bn / (an - 1)
            s_sd = s_mean / mp.sqrt(an - 2)
            n_draws = min(spec["fit_draws"], 200)
            c = s_mean + s_sd / mp.sqrt(n_draws) * mp.sqrt((1 - mpf("5e-4")) / mpf("5e-4"))
            t = mp.sqrt(2 * vtt * c * mp.log(2 / mpf("5e-4")) / n_draws)
            return post["mn"][1], mp.sqrt(v_post), t
        post = conj_posterior(X, y, mean, var)
        vtt = post["vn"][1][1] * mpf(resid["sigma2"])
        n_draws = min(spec["fit_draws"], 200)
        t = mp.sqrt(2 * vtt * mp.log(2 / mpf("1e-9")) / n_draws)
        return post["mn"][1], mp.sqrt(vtt), t

    iso = []
    for s in spec["scales"]:
        iso.append(one([0.0, 0.0, 0.0], [s * s] * 3, {"kind": "ig", "shape": 1e-3, "scale": 1e-3}))
    out["isotropic"] = [{"mean": float(a), "sd": float(b), "bound": float(c)} for a, b, c in iso]
    res = spec["resolved"]
    rr = []
    for mult in res["multipliers"]:
        rr.append(one(res["coef_mean"], [v * mult for v in res["coef_var"]],
                      {"kind": "ig", "shape": res["shape"], "scale": res["scale"]}))
    out["resolved"] = [{"mean": float(a), "sd": float(b), "bound": float(c)} for a, b, c in rr]
    al = []
    for comp in d["composed"]:
        al.append(one(comp["mean"], comp["var"], comp["resid"]))
    out["alpha"] = [{"mean": float(a), "sd": float(b), "bound": float(c)} for a, b, c in al]
    return out


def report_expected(case):
    means = case["effect_means"]
    orig = case["original"]
    informative = len(means) > 0 and math.isfinite(orig) and all(math.isfinite(m) for m in means)
    if not informative:
        return {"relative": None, "informative": False, "passed": False}
    denom = max([abs(F(m)) for m in means] + [abs(F(orig))])
    rel = 0 if denom == 0 else (F(max(means)) - F(min(means))) / denom
    return {"relative": float(rel), "informative": True, "passed": bool(rel <= F(case["max_relative_range"]))}


def chi2_quantile(df, q):
    f = lambda x: mp.gammainc(mpf(df) / 2, 0, x / 2, regularized=True) - q
    return mp.findroot(f, mpf(df) * 2)


def build_stage(data, draws):
    out = {"post": [], "prior": [], "shape": {}, "sbc": [], "sens": {}, "report_cases": [], "constants": {}}
    for case in data["post"]:
        d = draws["post"][case["name"]]
        d = dict(d)
        n, p = d["nrows"], d["ncols"]
        d["matrix"] = [[mpf(d["matrix_colmajor"][c * n + i]) for c in range(p)] for i in range(n)]
        exp = post_expected(case, d)
        out["post"].append({**case, "draws": draws["post"][case["name"]], "expected": exp})
    for case in data["prior"]:
        if case["law"]:
            d = {"matrix": None}
            dd = draws["prior"][case["name"]]
            n, p = dd["nrows"], dd["ncols"]
            d["matrix"] = [[mpf(dd["matrix_colmajor"][c * n + i]) for c in range(p)] for i in range(n)]
            d["outcome"] = dd["outcome"]
            out["prior"].append({**case, "draws": {"nrows": n, "ncols": p, "matrix_colmajor": dd["matrix_colmajor"],
                                                   "outcome": dd["outcome"]}, "expected_law": prior_law(case, d)})
            continue
        d = dict(draws["prior"][case["name"]])
        n, p = d["nrows"], d["ncols"]
        d["matrix"] = [[mpf(d["matrix_colmajor"][c * n + i]) for c in range(p)] for i in range(n)]
        out["prior"].append({**case, "draws": draws["prior"][case["name"]], "expected": prior_expected(case, d)})
    # residual variance shape: exact marginal law of 1/sigma^2
    sh = dict(draws["shape"])
    n, p = sh["nrows"], sh["ncols"]
    X = [[mpf(sh["matrix_colmajor"][c * n + i]) for c in range(p)] for i in range(n)]
    post = conj_posterior(X, [mpf(v) for v in sh["outcome"]], [0.0] * p, [100.0] * p, a=1e-3, b=1e-3)
    an, bn = post["an"], post["bn"]
    big_n = sh["draws"]
    x = mp.log(mpf(2) / mpf("1e-9"))
    v, scale = an / bn ** 2, 1 / bn  # Gamma(an, rate bn): variance an / bn^2, scale 1/bn
    out["shape"] = {"an": float(an), "bn": float(bn), "mean_precision": float(an / bn),
                    "upper": float(mp.sqrt(2 * v * x / big_n) + scale * x / big_n),
                    "lower": float(mp.sqrt(2 * v * x / big_n)), "draws": big_n,
                    "data": {**sh, "t": data["shape"]["t"], "z": data["shape"]["z"]}}
    for case in data["sbc"]:
        d = dict(draws["sbc"][case["name"]])
        n, p = d["nrows"], d["ncols"]
        d["matrix"] = [[mpf(d["matrix_colmajor"][c * n + i]) for c in range(p)] for i in range(n)]
        out["sbc"].append({**case, "draws": draws["sbc"][case["name"]], "expected": sbc_expected(case, d)})
    sd = dict(draws["sens"])
    n, p = sd["nrows"], sd["ncols"]
    sd["matrix"] = [[mpf(sd["matrix_colmajor"][c * n + i]) for c in range(p)] for i in range(n)]
    out["sens"] = {**data["sens"], "draws": draws["sens"], "expected": sens_expected(data["sens"], sd)}
    out["report_cases"] = [{**c, "expected": report_expected(c)} for c in data["report_cases"]]
    # chi-square critical values and the exact rank-uniformity identity
    out["constants"]["chi2_9df_p99"] = float(chi2_quantile(9, mpf("0.99")))
    out["constants"]["chi2_4df_p99"] = float(chi2_quantile(4, mpf("0.99")))
    L = 99
    beta_ok = all(
        F(math.comb(L, k)) * F(math.factorial(k) * math.factorial(L - k), math.factorial(L + 1)) == F(1, L + 1)
        for k in range(L + 1))
    out["constants"]["rank_marginal_is_uniform"] = bool(beta_ok)
    bins = {}
    for n_draws in (9, 99, 299):
        counts = [0] * 10
        for r in range(n_draws + 1):
            counts[min(r * 10 // (n_draws + 1), 9)] += 1
        bins[str(n_draws)] = counts
    out["constants"]["bin_counts"] = bins
    return out


def main():
    stage = sys.argv[1]
    if stage == "data":
        json.dump(data_stage(), sys.stdout, allow_nan=True)
    else:
        data = json.load(open(sys.argv[2]))
        draws = json.load(open(sys.argv[3]))
        json.dump(build_stage(data, draws), sys.stdout, allow_nan=True)


if __name__ == "__main__":
    main()
