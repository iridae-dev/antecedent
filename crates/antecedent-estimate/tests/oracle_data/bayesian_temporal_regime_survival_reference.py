# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0", "numpy"]
# ///
"""Bayesian regime survival: tiny binary-confounder designs, exact pieces and an independent g-formula.

Regenerate with:
    uv run --script bayesian_temporal_regime_survival_reference.py > bayesian_temporal_regime_survival_reference.json

Model (the one the product states, written here from the definition):
  hazard      Y_k | L_k, A_k, k/K            pooled logistic, rows with C_k = 0, features [1, L_k, A_k, k/K]
  treatment   A_k | L_k, A_{k-1}, k/K        pooled logistic, all rows,          features [1, L_k, A_{k-1}, k/K]
  censoring   C_k | L_k, A_k, k/K            pooled logistic, all rows,          features [1, L_k, A_k, k/K]
  confounder  L_k | L_{k-1}, A_{k-1}, k/K    Gaussian linear, k >= 1, features [1, L_{k-1}, A_{k-1}, k/K]
Every logistic coefficient has an N(0, s^2) prior (intercept included); the posterior is the Laplace
normal N(mode, (X'WX + I/s^2)^-1).  The Gaussian law has beta | sigma^2 ~ N(0, sigma^2 s^2 I),
sigma^2 ~ InvGamma(1, 1): posterior mean (X'X + I/s^2)^-1 X'y, V = (X'X + I/s^2)^-1,
shape 1 + m/2, rate 1 + (y'y - mean'X'y)/2.  The confounder is binary in the data and modelled as
Gaussian by the product; the reference models it the same way.

Posterior draw of the g-formula: theta_h ~ N(mode_h, cov_h); sigma^2 ~ InvGamma(shape, rate);
beta ~ N(mean, sigma^2 V); w ~ Dirichlet(1,...,1) over subjects; for every subject and arm
(always treated a=1, never treated a=0) L_0 is the observed baseline, L_k = beta'[1, L_{k-1}, a, k/K]
+ sigma z_k with the same z_k in both arms, hazards sigmoid(theta_h'[1, L_k, a, k/K]);
S_a = sum_i w_i prod_k (1 - hazard) ; draw = (S_1 - S_0, S_1, S_0).

Exact pieces (50 digits, mpmath): the three Laplace modes and covariances and the Gaussian-law
posterior.  For horizon 2 the posterior mean of S_a is a four-dimensional integral: the Dirichlet
weights are independent of the coefficients with mean 1/n, and L_1 | data marginalises to a
Student t with 2*shape degrees of freedom, location x'mean and squared scale (rate/shape)(1 + x'Vx),
so E S_a = (1/n) sum_i E_h E_t[(1 - sigmoid(h0 + h1 L0_i + h2 a)) (1 - sigmoid(h0 + h2 a + h3/2 + h1 L1))],
computed with a Gauss-Hermite rule (14 nodes per dimension) over theta_h and a Gauss-Legendre rule
on the angle x = sqrt(nu) tan(phi) for the t (density proportional to cos^(nu-1) phi).
Independent Monte Carlo (numpy, 2e6 draws) of the whole draw gives the sd, higher moments,
quantiles and the positivity-shortfall share (share of draws whose smallest treatment probability
over the observed rows is below the floor).  Repository-authored (MIT OR Apache-2.0); mpmath and
numpy are not product dependencies.
"""
import json
import math

import mpmath as mp
import numpy as np

mp.mp.dps = 50
N_MC = 2_000_000
np.seterr(over="ignore")


def sig(e):
    return 1 / (1 + mp.e ** (-e))


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((self.s >> 11) + 0.5) / (1 << 53)

    def bern(self, p):
        return self.u() < p


def sg(x):
    return 1 / (1 + math.exp(-x))


def simulate(seed, n, horizon, cens):
    g = Lcg(seed)
    rows = []
    for unit in range(n):
        l = 1 if g.bern(0.5) else 0
        a_prev = 0
        for k in range(horizon):
            if k > 0:
                l = 1 if g.bern(sg(-0.3 + 1.3 * l - 0.6 * a_prev + 0.2 * k)) else 0
            a = 1 if g.bern(sg(-0.4 + 0.9 * (l - 0.5) * 2 + 0.7 * a_prev)) else 0
            c = g.bern(cens)
            y = False if c else g.bern(sg(-1.5 + 0.9 * l - 0.8 * a + 0.3 * k))
            rows.append(dict(unit=unit, period=k, treatment=bool(a), censored=bool(c), event=bool(y), L=float(l)))
            if c or y:
                break
            a_prev = a
    return rows


def features_h(l, a, k, K):
    return [1.0, l, a, k / K]


def laplace(X, y, lam):
    """50-digit penalised logistic mode, covariance; X list of lists of floats."""
    p = len(X[0])
    Xm = mp.matrix([[mp.mpf(v) for v in r] for r in X])
    ym = [mp.mpf(v) for v in y]
    b = mp.matrix([0] * p)
    for _ in range(200):
        eta = Xm * b
        mu = [sig(e) for e in eta]
        grad = mp.matrix([0] * p)
        H = mp.eye(p) * lam
        for i in range(len(X)):
            w = mu[i] * (1 - mu[i])
            for a in range(p):
                grad[a] += Xm[i, a] * (ym[i] - mu[i])
                for c in range(p):
                    H[a, c] += w * Xm[i, a] * Xm[i, c]
        grad -= lam * b
        step = mp.lu_solve(H, grad)
        b += step
        if max(abs(s) for s in step) < mp.mpf(10) ** (-45):
            break
    eta = Xm * b
    mu = [sig(e) for e in eta]
    H = mp.eye(p) * lam
    for i in range(len(X)):
        w = mu[i] * (1 - mu[i])
        for a in range(p):
            for c in range(p):
                H[a, c] += w * Xm[i, a] * Xm[i, c]
    cov = H ** -1
    return [float(v) for v in b], [[float(cov[i, j]) for j in range(p)] for i in range(p)], H


def gaussian_law(X, y, lam):
    p = len(X[0])
    Xm = mp.matrix([[mp.mpf(v) for v in r] for r in X])
    ym = mp.matrix([mp.mpf(v) for v in y])
    P = Xm.T * Xm + mp.eye(p) * lam
    V = P ** -1
    mean = V * (Xm.T * ym)
    quad = (ym.T * ym)[0] - (mean.T * (Xm.T * ym))[0]
    m = len(y)
    return dict(mean=[float(v) for v in mean], V=[[float(V[i, j]) for j in range(p)] for i in range(p)],
                shape=float(1 + mp.mpf(m) / 2), rate=float(1 + quad / 2), rows=m)


def build_case(name, seed, n, K, scale, cens, exact):
    rows = simulate(seed, n, K, cens)
    lam = 1 / (scale * scale)
    by = {}
    for r in rows:
        by.setdefault(r["unit"], []).append(r)
    hx, hy, tx, ty, cx, cy, gx, gy = [], [], [], [], [], [], [], []
    for u, rs in by.items():
        for k, r in enumerate(rs):
            a = 1.0 if r["treatment"] else 0.0
            ap = 1.0 if k > 0 and rs[k - 1]["treatment"] else 0.0
            tx.append(features_h(r["L"], ap, k, K)); ty.append(a)
            cx.append(features_h(r["L"], a, k, K)); cy.append(1.0 if r["censored"] else 0.0)
            if not r["censored"]:
                hx.append(features_h(r["L"], a, k, K)); hy.append(1.0 if r["event"] else 0.0)
            if k > 0:
                gx.append(features_h(rs[k - 1]["L"], ap, k, K)); gy.append(r["L"])
    hm, hc, _ = laplace(hx, hy, lam)
    tm, tc, _ = laplace(tx, ty, lam)
    cm, cc, _ = laplace(cx, cy, lam)
    gl = gaussian_law(gx, gy, lam)
    base = [by[u][0]["L"] for u in sorted(by)]
    case = dict(name=name, horizon=K, prior_scale=scale, subjects=len(by), person_periods=len(rows), rows=rows,
                hazard=dict(x=hx, y=hy, mode=hm, cov=hc), treatment=dict(x=tx, y=ty, mode=tm, cov=tc),
                censoring=dict(x=cx, y=cy, mode=cm, cov=cc), confounder=gl)
    # ---- independent Monte Carlo of the whole draw
    rng = np.random.default_rng(seed + 12345)
    Lh = np.linalg.cholesky(np.array(hc)); Lt = np.linalg.cholesky(np.array(tc))
    LV = np.linalg.cholesky(np.array(gl["V"]))
    hm_, tm_, gmu = np.array(hm), np.array(tm), np.array(gl["mean"])
    TX = np.array(tx)
    base_a = np.array(base)
    # floor: median of the smallest treatment probability over the observed rows
    def draws_min(C):
        th = tm_ + rng.standard_normal((C, 4)) @ Lt.T
        p = 1 / (1 + np.exp(-(th @ TX.T)))
        return np.minimum(p, 1 - p).min(axis=1)
    floor = round(float(np.median(draws_min(200000))), 4)
    case["floor"] = floor
    chunk = 100_000
    out = [[], [], []]
    short = 0
    for _ in range(N_MC // chunk):
        h = hm_ + rng.standard_normal((chunk, 4)) @ Lh.T
        s2 = 1.0 / rng.gamma(gl["shape"], 1.0 / gl["rate"], size=chunk)
        beta = gmu + np.sqrt(s2)[:, None] * (rng.standard_normal((chunk, 4)) @ LV.T)
        w = rng.exponential(size=(chunk, n)); w /= w.sum(1, keepdims=True)
        th = tm_ + rng.standard_normal((chunk, 4)) @ Lt.T
        p = 1 / (1 + np.exp(-(th @ TX.T)))
        short += int((np.minimum(p, 1 - p).min(axis=1) < floor).sum())
        S = []
        noise = rng.standard_normal((chunk, n, K))
        for a in (1.0, 0.0):
            L = np.broadcast_to(base_a, (chunk, n)).copy()
            surv = np.ones((chunk, n))
            for k in range(K):
                if k > 0:
                    mean = (beta[:, 0:1] + beta[:, 1:2] * L + beta[:, 2:3] * a + beta[:, 3:4] * (k / K))
                    L = mean + np.sqrt(s2)[:, None] * noise[:, :, k]
                eta = h[:, 0:1] + h[:, 1:2] * L + h[:, 2:3] * a + h[:, 3:4] * (k / K)
                surv *= 1 - 1 / (1 + np.exp(-eta))
            S.append((w * surv).sum(1))
        out[0].append(S[0] - S[1]); out[1].append(S[0]); out[2].append(S[1])
    levels = [0.025, 0.975]
    mc = dict(n=N_MC, mean=[], sd=[], kurtosis=[], cdf_grid=[], cdf_levels=[], lower=[], upper=[], shortfall=short / N_MC)
    for col in out:
        x = np.concatenate(col)
        mc["mean"].append(float(x.mean())); mc["sd"].append(float(x.std(ddof=1)))
        mc["kurtosis"].append(float(((x - x.mean()) ** 4).mean() / x.var() ** 2))
        qs = np.quantile(x, np.linspace(0.0005, 0.9995, 1999))
        mc["cdf_grid"].append([float(v) for v in qs])
        mc["lower"].append(float(np.quantile(x, 0.025))); mc["upper"].append(float(np.quantile(x, 0.975)))
    mc["cdf_levels"] = [float(v) for v in np.linspace(0.0005, 0.9995, 1999)]
    case["mc"] = mc
    # ---- exact mean (horizon 2) and the hazard-mode plug-in
    if exact:
        assert K == 2
        case["exact"] = exact_mean(base, hm, hc, gl, K)
    return case


def exact_mean(base, hmode, hcov, gl, K, nh=14, nphi=400):
    x, wgt = np.polynomial.hermite_e.hermegauss(nh)
    wgt = wgt / wgt.sum()
    Lh = np.linalg.cholesky(np.array(hcov))
    Z = np.array(np.meshgrid(x, x, x, x, indexing="ij")).reshape(4, -1).T
    W = np.prod(np.array(np.meshgrid(wgt, wgt, wgt, wgt, indexing="ij")).reshape(4, -1).T, axis=1)
    H = np.array(hmode) + Z @ Lh.T
    nu = 2 * gl["shape"]
    gmu, V = np.array(gl["mean"]), np.array(gl["V"])
    phi, wl = np.polynomial.legendre.leggauss(nphi)
    phi = phi * (math.pi / 2)
    wl = wl * np.cos(phi) ** (nu - 1)
    wl = wl / wl.sum()
    T = math.sqrt(nu) * np.tan(phi)
    sq = lambda e: 1 / (1 + np.exp(-e))
    res = {"full": [], "plug_in": []}
    out = []
    plug = []
    for a in (1.0, 0.0):
        tot = 0.0
        tot_p = 0.0
        for l0 in base:
            xv = np.array([1.0, l0, a, 0.5])
            loc = gmu @ xv
            sc = math.sqrt(gl["rate"] / gl["shape"] * (1 + xv @ V @ xv))
            L1 = loc + sc * T  # (nphi,)
            eta0 = H[:, 0] + H[:, 1] * l0 + H[:, 2] * a
            u = H[:, 0] + H[:, 2] * a + H[:, 3] * 0.5
            eta1 = u[:, None] + H[:, 1:2] * L1[None, :]
            val = (1 - sq(eta0))[:, None] * (1 - sq(eta1))
            tot += float((W[:, None] * wl[None, :] * val).sum())
            h = np.array(hmode)
            e0 = h[0] + h[1] * l0 + h[2] * a
            e1 = h[0] + h[2] * a + h[3] * 0.5 + h[1] * L1
            tot_p += float((wl * (1 - sq(e0)) * (1 - sq(e1))).sum())
        out.append(tot / len(base)); plug.append(tot_p / len(base))
    return dict(mean=[out[0] - out[1], out[0], out[1]], plug_in=[plug[0] - plug[1], plug[0], plug[1]], nodes=nh)


cases = [
    build_case("two_periods_scale_2", 20260301, 16, 2, 2.0, 0.12, True),
    build_case("two_periods_scale_half_censored", 20260302, 20, 2, 0.5, 0.2, True),
    build_case("three_periods_monte_carlo_only", 20260303, 18, 3, 1.0, 0.1, False),
]
print(json.dumps(dict(cases=cases)))
