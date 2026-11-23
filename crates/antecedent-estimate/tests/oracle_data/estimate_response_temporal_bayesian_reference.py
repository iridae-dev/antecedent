# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "mpmath==1.3.0"]
# ///
"""Long-run-tempered Gaussian posterior of a temporal response, written from its definition.

Regenerate with:
    uv run --script estimate_response_temporal_bayesian_reference.py > estimate_response_temporal_bayesian_reference.json

A case is a series (x treatment, y outcome, z confounder) and horizons h with the adjustment lags
of each horizon.  Horizon h regresses y_t on (1, x_{t-h}, z_{t-l} for each lag l) over the windows
t >= max(h, largest l).  Everything below is the definition, evaluated without any of the
product's routines:

1. Tempering factor kappa of a horizon.  The OLS residual e is modelled as a stationary Gaussian
   AR(q) process.  Its n x n correlation matrix R comes from the stationary covariance of the
   companion-form state (a discrete Lyapunov equation), never from a Yule-Walker or Levinson
   recursion on the data.  The restricted likelihood of the contrasts is evaluated densely,
       l(z) = -1/2 log|S| - 1/2 log|X'S^-1 X| - (n-p)/2 log(RSS_gls/(n-p)),
   with S the unit-innovation covariance, over the partial autocorrelations r = 0.97 tanh z, and
   maximised by multi-start BFGS plus a simplex polish; the order q is the BIC minimiser
   -2 l + q ln(n-p) over the orders the product's search rule visits.  For a targeted
   combination c, with w = X (X'X)^-1 c and H the hat matrix,
       kappa_unbounded = [w'Rw / w'w] * n / max(n - tr(HR), p + 2) * exp(tau^2 / 2),
   tau the delta-method standard deviation of log kappa from the observed information of l (the
   inverse of -Hessian l, both by central differences with Richardson extrapolation).  The factor
   of a horizon is the largest over the combinations of its dose grid, floored at 1 and capped at
   n / (p + 2); the factor of a combination is also bounded by three times the fixed-b-scaled long-run
   variance ratio of its estimating score w_t e_t (the larger of the AR(1)- and, when a
   Yule-Walker BIC fit picks order two or more, the AR(q)-prewhitened Bartlett readings; Kendall
   bias-corrected AR(1) coefficient, bandwidth floor(4 (n/100)^(2/9)), Kiefer-Vogelsang fixed-b
   polynomial at bandwidth + 1): kappa_combination = min(kappa_unbounded, that bound).
2. Tempered posterior.  Rows weigh 1 / kappa and the prior keeps full weight:
       L = I / c0^2 + X'X / kappa,  m = L^-1 X'y / kappa,  a_n = a0 + n / (2 kappa),
       b_n = b0 + (y'y / kappa - m'L m) / 2,   a0 = b0 = 1e-3,
   so the level w'beta is Student-t with 2 a_n degrees of freedom, location w'm and squared scale
   (b_n / a_n) w'L^-1 w.  Evaluated in 40-digit arithmetic.
3. Joint law of several horizons.  The OLS normal-equation scores x_tj e_t of every horizon are
   placed on the series calendar, stacked, and their Bartlett long-run covariance at the
   dependence block length is pushed through each horizon's posterior covariance; each diagonal
   block is whitened by its own Cholesky factor, leaving a cross-horizon correlation matrix.  The
   block length is max(rule, min(ceil(b_PW n^(1/6)), n // 3)) with rule = max(span, ceil(n^(1/3))),
   span = largest horizon + history lag, and b_PW the largest Politis-White flat-top length over
   the stacked score series.  Each horizon keeps its own posterior mean and covariance, so the
   level of horizon h is Gaussian with the tempered mean and variance b_n / (a_n - 1) w'L^-1 w,
   and two horizons' levels have covariance  w_a' F_a C_ab F_b' w_b  with F the covariance
   Cholesky factors and C_ab the correlation block.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import math

import mpmath as mp
import numpy as np
from scipy import linalg, optimize

mp.mp.dps = 40
A0 = B0 = mp.mpf("0.001")
MAX_PACF = 0.97
Z975 = 1.959963984540054

_state = 0x2F6C1D3B5A790E47


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def normal():
    return sum(u01() for _ in range(12)) - 6.0


def series(n, rho_x, rho_u):
    z, x, y, u = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
    for t in range(n):
        zp = z[t - 1] if t >= 1 else 0.0
        z[t] = round(0.5 * zp + 0.8 * normal(), 9)
        xp = x[t - 1] if t >= 1 else 0.0
        x[t] = round(rho_x * xp + 0.5 * z[t] + 0.3 * zp + 0.7 * normal(), 9)
        up = u[t - 1] if t >= 1 else 0.0
        u[t] = rho_u * up + 0.6 * normal()
        x1 = x[t - 1] if t >= 1 else 0.0
        x2 = x[t - 2] if t >= 2 else 0.0
        z2 = z[t - 2] if t >= 2 else 0.0
        y[t] = round(1.5 * x1 + 0.7 * x2 + 0.5 * zp + 0.8 * z2 + u[t], 9)
    return x, y, z


def design(x, y, z, h, lags):
    n = len(y)
    first = max([h] + lags)
    rows = [[1.0, x[t - h]] + [z[t - l] for l in lags] for t in range(first, n)]
    return np.array(rows), np.array([y[t] for t in range(first, n)]), first


# ---------------------------------------------------------------- AR(q) residual model


def pacf_to_phi(r):
    phi = []
    for m, k in enumerate(r):
        phi = [phi[j] - k * phi[m - 1 - j] for j in range(m)] + [k]
    return phi


def stationary_autocovariance(phi, n):
    """gamma(0..n-1) of a unit-innovation AR(q): e1' F^k P e1 with P = F P F' + e1 e1'."""
    q = len(phi)
    if q == 0:
        g = np.zeros(n)
        g[0] = 1.0
        return g
    F = np.zeros((q, q))
    F[0, :] = phi
    for i in range(1, q):
        F[i, i - 1] = 1.0
    Q = np.zeros((q, q))
    Q[0, 0] = 1.0
    P = linalg.solve_discrete_lyapunov(F, Q)
    u = P[:, 0].copy()
    g = np.empty(n)
    for k in range(n):
        g[k] = u[0]
        u = F @ u
    return g


def sigma_of(z, n):
    phi = pacf_to_phi([MAX_PACF * math.tanh(v) for v in z])
    return linalg.toeplitz(stationary_autocovariance(phi, n))


def reml_loglik(z, X, y):
    n, p = X.shape
    S = sigma_of(z, n)
    try:
        c = linalg.cho_factor(S, lower=True)
    except linalg.LinAlgError:
        return -math.inf
    logdet = 2.0 * np.sum(np.log(np.diag(c[0])))
    SiX = linalg.cho_solve(c, X)
    Siy = linalg.cho_solve(c, y)
    G = X.T @ SiX
    beta = np.linalg.solve(G, X.T @ Siy)
    rss = float(y @ Siy - (X.T @ Siy) @ beta)
    if not rss > 0:
        return -math.inf
    sign, logdet_g = np.linalg.slogdet(G)
    return -0.5 * logdet - 0.5 * logdet_g - 0.5 * (n - p) * math.log(rss / (n - p))


def fit_order(q, X, y, warm):
    def f(z):
        v = reml_loglik(z, X, y)
        return 1e300 if not math.isfinite(v) else -v

    starts = [np.append(warm, 0.0), np.zeros(q), np.append(warm, 0.3), np.append(warm, -0.3)]
    best = None
    for s in starts:
        r = optimize.minimize(f, s, method="BFGS", options={"gtol": 1e-9, "maxiter": 400})
        r = optimize.minimize(f, r.x, method="Nelder-Mead",
                              options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 4000 * q})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


def schwert(n):
    return max(4, int(math.floor(12.0 * (n / 100.0) ** 0.25)))


def select_ar(X, y):
    n, p = X.shape
    dof = n - p
    max_order = min(schwert(n), n - p - 2)
    best_q, best_z = 0, np.zeros(0)
    best_bic = -2.0 * reml_loglik(np.zeros(0), X, y)
    warm = np.zeros(0)
    bics = {0: best_bic}
    for q in range(1, max_order + 1):
        if q > 4 and q > best_q + 2:
            break
        z, ll = fit_order(q, X, y, warm)
        warm = z
        bic = -2.0 * ll + q * math.log(dof)
        bics[q] = bic
        if bic < best_bic:
            best_bic, best_q, best_z = bic, q, z
    return best_q, best_z, bics


def richardson_grad(f, z, i, h):
    def central(step):
        a = np.array(z, float)
        b = np.array(z, float)
        a[i] += step
        b[i] -= step
        return (f(a) - f(b)) / (2 * step)

    return (4 * central(h / 2) - central(h)) / 3


def hessian(f, z, h):
    q = len(z)
    H = np.zeros((q, q))

    def second(i, j, step):
        def at(si, sj):
            v = np.array(z, float)
            v[i] += si
            v[j] += sj
            return f(v)

        return (at(step, step) - at(step, -step) - at(-step, step) + at(-step, -step)) / (4 * step * step)

    for i in range(q):
        for j in range(q):
            H[i, j] = (4 * second(i, j, h / 2) - second(i, j, h)) / 3
    return 0.5 * (H + H.T)


def bartlett_lrv(u, bandwidth):
    m = len(u)
    lags = min(bandwidth, m - 1)
    lrv = float(u @ u)
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * float(u[k:] @ u[:-k])
    return max(lrv / m, 0.0)


def newey_west_ratio(s, bandwidth):
    """AR(1)-prewhitened Bartlett long-run variance of s over its variance (Kendall-corrected rho)."""
    n = len(s)
    g0 = float(s @ s) / n
    num = float(s[1:] @ s[:-1])
    den = float(s[:-1] @ s[:-1])
    rho_hat = num / den if den > 0 else 0.0
    rho = min(max(rho_hat + (1 + 3 * rho_hat) / n, -MAX_PACF), MAX_PACF)
    u = s[1:] - rho * s[:-1]
    return bartlett_lrv(u, bandwidth) / (1 - rho) ** 2 / g0


def yule_walker_bic(s, limit=4):
    """BIC-selected Yule-Walker AR(q<=4) of a mean-zero series (uncentred autocovariances)."""
    n = len(s)
    g = [float(s[k:] @ s[: n - k]) / n for k in range(limit + 1)]
    best_phi, best_bic = [], n * math.log(g[0])
    for q in range(1, limit + 1):
        T = linalg.toeplitz(g[:q])
        phi = np.linalg.solve(T, np.array(g[1: q + 1]))
        var = g[0] - float(phi @ np.array(g[1: q + 1]))
        bic = n * math.log(var) + q * math.log(n)
        if bic < best_bic:
            best_bic, best_phi = bic, list(phi)
    return best_phi


def score_hac_ratio(s, bandwidth):
    ratio = newey_west_ratio(s, bandwidth)
    phi = yule_walker_bic(s)
    q = len(phi)
    if q >= 2:
        n = len(s)
        g0 = float(s @ s) / n
        u = np.array([s[t] - sum(phi[j] * s[t - 1 - j] for j in range(q)) for t in range(q, n)])
        recolour = max(1.0 - sum(phi), 1.0 - MAX_PACF)
        ratio = max(ratio, bartlett_lrv(u, bandwidth) / recolour ** 2 / g0)
    return ratio, q


def fixed_b(block, n):
    b = min(max(block / n, 0.0), 1.0)
    return (1.96 + 2.9694 * b + 0.4160 * b * b - 0.5324 * b ** 3) / 1.96


def horizon_fit(X, y):
    """Design-level quantities: OLS fit, residual, REML AR model, observed information."""
    n, p = X.shape
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    q, z, bics = select_ar(X, y)
    fit = dict(n=n, p=p, e=e, y=y, q=q, z=z, bics=bics)
    ll = lambda v: reml_loglik(v, X, y)
    fit["info_pd"] = False
    if q > 0:
        info = -hessian(ll, z, 1e-3)
        try:
            np.linalg.cholesky(info)
            fit["info"] = info
            fit["info_pd"] = True
        except np.linalg.LinAlgError:
            pass
    return fit


def kappa_of(X, fit, directions):
    n, p = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    H = X @ xtx_inv @ X.T
    h_bw = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    fb = fixed_b(h_bw + 1, n)
    z = fit["z"]

    def operator(v):
        S = sigma_of(v, n)
        return S / S[0, 0]

    def ratio_scale(v, w):
        R = operator(v)
        ratio = float(w @ R @ w) / float(w @ w)
        scale = n / max(n - float(np.trace(H @ R)), p + 2)
        return ratio, scale

    out = []
    for c in directions:
        w = X @ (xtx_inv @ c)
        ar, scale = ratio_scale(z, w)
        tau = 0.0
        if fit["q"] > 0 and fit["info_pd"]:
            lk = lambda v: math.log(np.prod(ratio_scale(v, w)))
            g = np.array([richardson_grad(lk, z, i, 1e-3) for i in range(len(z))])
            var = float(g @ np.linalg.solve(fit["info"], g))
            tau = min(math.sqrt(var), 1.0) if math.isfinite(var) and var > 0 else 0.0
        unbounded = scale * ar * math.exp(0.5 * tau * tau)
        influence = w * fit["e"]
        hac, score_q = score_hac_ratio(influence, h_bw)
        bound = 3.0 * hac * fb * fb
        out.append(dict(ar_ratio=ar, scale=scale, tau=tau, unbounded=unbounded, hac=hac,
                        fixed_b=fb, score_ar_order=score_q, bound=bound,
                        combined=min(unbounded, bound), bounded=bool(bound < unbounded)))
    return out


def tempering(X, fit, directions):
    n, p = X.shape
    if n < max(8, p + 2):
        return dict(kappa=1.0, inestimable=True, capped=False, detail=[], q=0)
    if float(fit["e"] @ fit["e"]) <= 1e-20 * float(((fit["y"] - fit["y"].mean()) ** 2).sum()):
        return dict(kappa=1.0, inestimable=False, capped=False, detail=[], q=0, exact=True)
    detail = kappa_of(X, fit, directions)
    raw = max(d["combined"] for d in detail)
    cap = max(n / (p + 2), 1.0)
    return dict(kappa=min(max(raw, 1.0), cap), inestimable=False, capped=raw > cap,
                detail=detail, q=fit["q"], bounded=any(d["bounded"] for d in detail))


# ---------------------------------------------------------------- tempered posterior


def posterior(X, y, c0, kappa, direction):
    n, p = X.shape
    Xm = mp.matrix(X.tolist())
    Y = mp.matrix(y.tolist())
    k = mp.mpf(kappa)
    L = Xm.T * Xm / k + mp.eye(p) / mp.mpf(c0) ** 2
    Linv = L ** -1
    m = Linv * (Xm.T * Y) / k
    an = A0 + mp.mpf(n) / (2 * k)
    bn = B0 + ((Y.T * Y)[0] / k - (m.T * L * m)[0]) / 2
    return dict(m=m, Linv=Linv, an=an, bn=bn)


def level_summary(post, w):
    W = mp.matrix([mp.mpf(float(v)) for v in w])
    loc = (W.T * post["m"])[0]
    q = (W.T * post["Linv"] * W)[0]
    an, bn = post["an"], post["bn"]
    var = bn / (an - 1) * q
    scale = mp.sqrt(bn / an * q)
    nu = 2 * an

    def tq(prob):
        def cdf(v):
            u = (v - loc) / scale
            wgt = nu / (nu + u * u)
            tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, wgt, regularized=True) / 2
            return 1 - tail if u > 0 else tail

        return mp.findroot(lambda v: cdf(v) - prob, loc)

    return dict(mean=float(loc), sd=float(mp.sqrt(var)), var=float(var),
                t_lower=float(tq(mp.mpf("0.025"))), t_upper=float(tq(mp.mpf("0.975"))),
                nu=float(nu))


def cov_factor(post):
    cov = np.array((post["bn"] / (post["an"] - 1) * post["Linv"]).tolist(), dtype=float)
    return cov, np.linalg.cholesky(cov)


# ---------------------------------------------------------------- joint dependence


def politis_white(s):
    n = len(s)
    if n < 8:
        return None
    mean = s.sum() / n
    c = s - mean
    acov = lambda k: float(c[: n - k] @ c[k:]) / n
    g0 = acov(0)
    if not g0 > 1e-20:
        return None
    kn = max(5, math.ceil(math.sqrt(math.log10(n))))
    mmax = min(math.ceil(math.sqrt(n)) + kn, n - 1)
    thr = 2.0 * math.sqrt(math.log10(n) / n)
    rho = [acov(k) / g0 for k in range(mmax + 1)]
    mhat = max(mmax - kn, 0)
    for m in range(0, max(mmax - kn, 0) + 1):
        if all(abs(rho[m + j]) < thr for j in range(1, kn + 1) if m + j < len(rho)):
            mhat = m
            break
    M = max(min(2 * mhat, mmax), 1)
    lam = lambda t: 1.0 if abs(t) <= 0.5 else (2 * (1 - abs(t)) if abs(t) <= 1 else 0.0)
    gg, big_g = g0, 0.0
    for k in range(1, M + 1):
        wgt = lam(k / M) * rho[k] * g0
        gg += 2 * wgt
        big_g += 2 * k * wgt
    if gg <= 0:
        return None
    d = 4.0 / 3.0 * gg * gg
    b = (2.0 * big_g * big_g / d) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    cap = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    return math.ceil(min(max(b, 1.0), max(cap, 1)))


def icbrt(n):
    k = 0
    while k ** 3 < n:
        k += 1
    return k


def block_length(span, rows, series_list):
    rule = max(1, min(max(span, icbrt(rows)), rows))
    pws = [v for v in (politis_white(s) for s in series_list) if v is not None]
    testing = math.ceil(max(pws) * rows ** (1 / 6)) if pws else 0
    return min(max(rule, min(testing, rows // 3)), max(rows, 1))


def joint(parts, span_structural):
    """parts: list of dict(cov, scores (p x n_h array), times (list)). Returns corr blocks, block."""
    first = min(p["times"][0] for p in parts)
    last = max(p["times"][-1] for p in parts)
    length = last + 1 - first
    stacked = []
    for part in parts:
        for score in part["scores"]:
            axis = np.zeros(length)
            for t, v in zip(part["times"], score):
                axis[t - first] = v
            stacked.append(axis)
    S = np.array(stacked)
    block = block_length(span_structural, length, stacked)
    total = len(stacked)
    C = np.zeros((total, total))
    for lag in range(block):
        weight = 1.0 if lag == 0 else 1.0 - lag / block
        gamma = S[:, lag:] @ S[:, : length - lag].T
        C += gamma if lag == 0 else weight * (gamma + gamma.T)
    widths = [len(p["scores"]) for p in parts]
    offsets = np.cumsum([0] + widths)
    A = linalg.block_diag(*[p["cov"] for p in parts])
    sandwich = A @ C @ A.T
    chol = []
    for k, w in enumerate(widths):
        d = sandwich[offsets[k]: offsets[k] + w, offsets[k]: offsets[k] + w]
        chol.append(np.linalg.cholesky(d))
    corr = np.eye(total)
    for a in range(len(parts)):
        for b in range(len(parts)):
            if a == b:
                continue
            blk = sandwich[offsets[a]: offsets[a + 1], offsets[b]: offsets[b + 1]]
            corr[offsets[a]: offsets[a + 1], offsets[b]: offsets[b + 1]] = (
                np.linalg.solve(chol[a], blk) @ np.linalg.inv(chol[b]).T)
    for a in range(len(parts)):
        for b in range(a):
            u = corr[offsets[b]: offsets[b + 1], offsets[a]: offsets[a + 1]]
            l = corr[offsets[a]: offsets[a + 1], offsets[b]: offsets[b + 1]]
            m = 0.5 * (u + l.T)
            corr[offsets[b]: offsets[b + 1], offsets[a]: offsets[a + 1]] = m
            corr[offsets[a]: offsets[a + 1], offsets[b]: offsets[b + 1]] = m.T
    pd = True
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        pd = False
    return corr, offsets, block, pd


# ---------------------------------------------------------------- cases


def run(case, horizons, lags_by_h, grid, history_lag, tiny=False):
    x, y, z = case["x"], case["y"], case["z"]
    c0 = case["prior_scale"]
    fits, posts, kappas, levels, cbars, designs = {}, {}, {}, {}, {}, {}
    for h in horizons:
        X, Y, first = design(x, y, z, h, lags_by_h[h])
        fit = horizon_fit(X, Y) if len(Y) >= max(8, X.shape[1] + 2) else dict(e=Y - X @ np.linalg.solve(X.T @ X, X.T @ Y), y=Y)
        cbar = X.mean(axis=0)
        directions = []
        for a in grid:
            c = cbar.copy()
            c[1] = a
            directions.append(c)
        temper = tempering(X, fit, directions)
        post = posterior(X, Y, c0, temper["kappa"], None)
        designs[h] = (X, Y, first)
        fits[h], posts[h], kappas[h], cbars[h] = fit, post, temper, directions
        levels[h] = [level_summary(post, c) for c in directions]
    result = dict(
        horizons=horizons,
        grid=grid,
        kappa={str(h): kappas[h]["kappa"] for h in horizons},
        inestimable={str(h): kappas[h]["inestimable"] for h in horizons},
        capped={str(h): kappas[h]["capped"] for h in horizons},
        ar_order={str(h): kappas[h]["q"] for h in horizons},
        bounded={str(h): kappas[h].get("bounded", False) for h in horizons},
        kappa_detail={str(h): kappas[h]["detail"] for h in horizons},
        levels={str(h): levels[h] for h in horizons},
    )
    if len(horizons) > 1:
        parts = []
        covs = {}
        for h in horizons:
            X, Y, first = designs[h]
            cov, fac = cov_factor(posts[h])
            covs[h] = fac
            e = fits[h]["e"]
            parts.append(dict(cov=cov, scores=[X[:, j] * e for j in range(X.shape[1])],
                              times=list(range(first, first + len(Y)))))
        span = max(horizons) + history_lag
        corr, offsets, block, pd = joint(parts, span)
        result["block"] = block
        result["joint_pd"] = pd
        cross = {}
        for ai, a in enumerate(horizons):
            for bi, b in enumerate(horizons):
                if bi <= ai:
                    continue
                blk = corr[offsets[ai]: offsets[ai + 1], offsets[bi]: offsets[bi + 1]]
                cov_ab = [[float(cbars[a][i] @ covs[a] @ blk @ covs[b].T @ cbars[b][j])
                           for j in range(len(grid))] for i in range(len(grid))]
                cross[f"{a},{b}"] = cov_ab
        result["level_cross_covariance"] = cross
    return result


def case_data(name, n, rho_x, rho_u, prior_scale):
    x, y, z = series(n, rho_x, rho_u)
    return dict(name=name, x=x, y=y, z=z, prior_scale=prior_scale)


LAGS = {1: [1, 2], 2: [2]}
cases = []

main = case_data("persistent_ar_residual", 90, 0.3, 0.4, 3.0)
main["queries"] = {
    "curve_h12": run(main, [1, 2], LAGS, [-1.0, 0.0, 1.5], 6),
    "curve_h12_short_history": run(main, [1, 2], LAGS, [-1.0, 0.0, 1.5], 0),
    "level_h12": run(main, [1, 2], LAGS, [1.0], 6),
    "curve_h1": run(main, [1], LAGS, [-1.0, 0.0, 1.5], 6),
}
cases.append(main)

tiny = case_data("seven_rows_inestimable", 9, 0.5, 0.5, 10.0)
tiny["queries"] = {"curve_h1": run(tiny, [1], LAGS, [0.0, 1.0], 6)}
cases.append(tiny)

print(json.dumps(cases, allow_nan=False))
