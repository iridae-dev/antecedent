"""Independent Normal-inverse-gamma equations; no Antecedent imports.

SPDX-License-Identifier: MIT OR Apache-2.0
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(1200)
n = 96
t, w = rng.normal(size=(2, n))
y = 1 + 2 * t + 0.4 * w + 0.5 * t * w + rng.normal(scale=0.4, size=n)
r = 1 + 2 * t + rng.normal(scale=0.4, size=n)
m = 0.8 * np.roll(t, 1) + rng.normal(scale=0.4, size=n)
o = 0.25 * np.roll(t, 1) + 0.55 * m + rng.normal(scale=0.4, size=n)
s = 2 * np.roll(t, 1) + 3 * np.roll(t, 2) + rng.normal(scale=0.4, size=n)
u = np.roll(m, 1) + np.roll(m, 2) + rng.normal(scale=0.04, size=n)
z = rng.normal(size=n)
tc = z + rng.normal(size=n)
mc = np.roll(tc, 1) + 10 * np.roll(z, 1) + rng.normal(size=n)
oc = .25 * np.roll(tc, 1) + 2 * mc + 5 * np.roll(z, 1) + rng.normal(size=n)

MAX_PACF = 0.97
MAX_AR_ORDER = 4
HAC_BOUND = 3.0
MAX_KAPPA_LOG_SD = 1.0
DELTA_STEP = 1e-3
SIMPLEX_STEP = 0.2
SIMPLEX_F_TOL = 1e-12
SIMPLEX_X_TOL = 1e-8
SIMPLEX_ITERATIONS_PER_DIM = 500


def posterior(x, y, scale, kappa=1.0):
    """NIG posterior of the likelihood tempered by 1/kappa (kappa = 1: the plain posterior).

    Prior beta | s2 ~ N(0, s2 scale^2 I), s2 ~ InvGamma(.001, .001); likelihood
    prod_i N(y_i | x_i'beta, s2)^(1/kappa) = (2 pi s2)^(-n/(2 kappa))
    exp(-|y - X beta|^2 / (2 kappa s2)).
    """
    precision = x.T @ x / kappa + np.eye(x.shape[1]) / scale**2
    v = np.linalg.inv(precision)
    mean = v @ x.T @ y / kappa
    a = 0.001 + len(y) / (2 * kappa)
    b = 0.001 + (y @ y / kappa - mean @ np.linalg.solve(v, mean)) / 2
    return mean, v * b / (a - 1)


def kendall_rho(e):
    """Uncentred lag-1 autocorrelation with the Kendall bias correction, clamped to +-.97."""
    rho = (e[1:] @ e[:-1]) / (e[:-1] @ e[:-1])
    return float(np.clip(rho + (1 + 3 * rho) / len(e), -MAX_PACF, MAX_PACF))


def fixed_b(block, rows):
    """Kiefer-Vogelsang Bartlett fixed-b 95% critical value over 1.96."""
    b = min(max(block / rows, 0.0), 1.0)
    return (1.96 + 2.9694 * b + 0.4160 * b**2 - 0.5324 * b**3) / 1.96


def bic_autoregression(e, max_order=MAX_AR_ORDER):
    """Yule-Walker AR(q) of e (uncentred autocovariances), q <= 4 minimizing
    n ln sigma2_q + q ln n (the prewhitening filter of the bounding score HAC)."""
    n = len(e)
    max_order = min(max_order, n - 2)
    g = np.array([e[k:] @ e[:n - k] / n for k in range(max_order + 1)])
    best = (n * np.log(g[0]), np.zeros(0))
    for q in range(1, max_order + 1):
        r = np.array([[g[abs(i - j)] for j in range(q)] for i in range(q)])
        phi = np.linalg.solve(r, g[1:q + 1])
        sigma2 = g[0] - phi @ g[1:q + 1]
        bic = n * np.log(sigma2) + q * np.log(n)
        if bic < best[0]:
            best = (bic, phi)
    return best[1]


def bartlett(u, bandwidth):
    """Bartlett long-run variance of u (uncentred autocovariances over len(u))."""
    lags = min(bandwidth, len(u) - 1)
    lrv = u @ u + 2 * sum((1 - k / (lags + 1)) * (u[k:] @ u[:-k]) for k in range(1, lags + 1))
    return max(lrv / len(u), 0.0)


def score_hac_ratio(score, rows):
    """Prewhitened Newey-West long-run-variance ratio of the score: the larger of the
    AR(1) (Kendall rho) and the BIC AR(q >= 2) prewhitening, each recoloured."""
    bandwidth = int(np.floor(4 * (rows / 100) ** (2 / 9)))
    rho = kendall_rho(score)
    white = score[1:] - rho * score[:-1]
    hac = bartlett(white, bandwidth) / (1 - rho) ** 2 / (score @ score / rows)
    phi_s = bic_autoregression(score)
    if len(phi_s) >= 2:
        q = len(phi_s)
        white = np.array([score[t] - phi_s @ score[t - q:t][::-1] for t in range(q, rows)])
        recolour = max(1 - phi_s.sum(), 1 - MAX_PACF)
        hac = max(hac, bartlett(white, bandwidth) / recolour**2 / (score @ score / rows))
    return hac, bandwidth, len(phi_s)


def pacf_to_phi(r):
    """Levinson step-up: partial autocorrelations to AR coefficients."""
    phi = np.zeros(0)
    for k in r:
        phi = np.append(phi[:] - k * phi[::-1], k)
    return phi


def z_to_phi(z):
    return pacf_to_phi(MAX_PACF * np.tanh(np.asarray(z, float)))


def ar_correlation(phi, rows):
    """Toeplitz correlation matrix of a stationary AR(phi) over `rows` rows."""
    q = len(phi)
    acf = np.zeros(rows)
    acf[0] = 1.0
    if q:
        # Yule-Walker for rho_1..rho_q: rho_k = sum_j phi_j rho_|k-j|.
        a = -np.eye(q)
        b = np.zeros(q)
        for k in range(1, q + 1):
            for j in range(1, q + 1):
                if k == j:
                    b[k - 1] -= phi[j - 1]
                else:
                    a[k - 1, abs(k - j) - 1] += phi[j - 1]
        acf[1:q + 1] = np.linalg.solve(a, b)
        for k in range(q + 1, rows):
            acf[k] = phi @ acf[k - q:k][::-1]
    lag = np.abs(np.subtract.outer(np.arange(rows), np.arange(rows)))
    return acf[lag]


def reml_loglik(z, x, y):
    """Profile REML log-likelihood of AR(z) errors (unit innovations):
    -1/2 log|Gamma| - 1/2 log|X'Gamma^-1 X| - (n-p)/2 log(RSS_gls / (n-p))."""
    rows, cols = x.shape
    phi = z_to_phi(z)
    corr = ar_correlation(phi, rows)
    gamma0 = 1.0 / (1.0 - phi @ corr[0, 1:len(phi) + 1]) if len(phi) else 1.0
    try:
        chol = np.linalg.cholesky(gamma0 * corr)
    except np.linalg.LinAlgError:
        return -np.inf
    xw = np.linalg.solve(chol, x)
    yw = np.linalg.solve(chol, y)
    g = xw.T @ xw
    beta = np.linalg.solve(g, xw.T @ yw)
    rss = float((yw - xw @ beta) @ (yw - xw @ beta))
    if not rss > 0:
        return -np.inf
    log_det = 2 * np.log(np.diag(chol)).sum()
    return -0.5 * log_det - 0.5 * np.linalg.slogdet(g)[1] - 0.5 * (rows - cols) * np.log(rss / (rows - cols))


def nelder_mead(f, x0):
    """Standard Nelder-Mead (reflect 1, expand 2, contract 1/2, shrink 1/2) from the
    simplex x0 + SIMPLEX_STEP e_i; stops on objective range and simplex extent."""
    dim = len(x0)
    simplex = [(np.array(x0, float), f(x0))]
    for i in range(dim):
        v = np.array(x0, float)
        v[i] += SIMPLEX_STEP
        simplex.append((v, f(v)))
    simplex.sort(key=lambda e: e[1])
    for _ in range(SIMPLEX_ITERATIONS_PER_DIM * dim):
        f_range = simplex[dim][1] - simplex[0][1]
        x_range = max(np.max(np.abs(v - simplex[0][0])) for v, _ in simplex[1:])
        if abs(f_range) <= SIMPLEX_F_TOL and x_range <= SIMPLEX_X_TOL:
            break
        centroid = np.mean([v for v, _ in simplex[:dim]], axis=0)
        worst, f_worst = simplex[dim]
        point = lambda coef: centroid + coef * (centroid - worst)
        reflected = point(1.0)
        fr = f(reflected)
        if fr < simplex[0][1]:
            expanded = point(2.0)
            fe = f(expanded)
            simplex[dim] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < simplex[dim - 1][1]:
            simplex[dim] = (reflected, fr)
        else:
            contracted = point(0.5) if fr < f_worst else point(-0.5)
            fc = f(contracted)
            if fc < min(fr, f_worst):
                simplex[dim] = (contracted, fc)
            else:
                best = simplex[0][0]
                simplex = [simplex[0]] + [
                    (best + 0.5 * (v - best), f(best + 0.5 * (v - best))) for v, _ in simplex[1:]
                ]
        simplex.sort(key=lambda e: e[1])
    return simplex[0]


def reml_autoregression(x, y):
    """REML AR(q) at the BIC order -2 l_R + q ln(n - p), q <= 4, warm-started upward.
    Returns the optimizer coordinates z (empty for q = 0)."""
    rows, cols = x.shape
    max_order = min(MAX_AR_ORDER, rows - cols - 2)
    best = (-2 * reml_loglik(np.zeros(0), x, y), np.zeros(0))
    prev = np.zeros(0)
    for q in range(1, max_order + 1):
        start = np.append(prev, 0.0)
        zq, neg_ll = nelder_mead(lambda zz: -reml_loglik(zz, x, y), start)
        if not np.isfinite(neg_ll):
            break
        prev = zq
        bic = 2 * neg_ll + q * np.log(rows - cols)
        if bic < best[0]:
            best = (bic, zq)
    return best[1]


def kappa_terms(z, x, weights):
    """Per-combination variance ratios w'Rw / w'w and the scale loss tr(HR)."""
    rows, cols = x.shape
    corr = ar_correlation(z_to_phi(z), rows)
    ratios = [wt @ corr @ wt / (wt @ wt) for wt in weights]
    df_loss = float(np.trace(np.linalg.solve(x.T @ x, x.T @ corr @ x)))
    scale = rows / max(rows - df_loss, cols + 2)
    return np.array(ratios), df_loss, scale


def kappa_log_sd(z, x, y, weights):
    """Delta-method SD of log kappa per combination from the observed REML information."""
    q = len(z)
    if q == 0:
        return np.zeros(len(weights))
    h = DELTA_STEP

    def log_kappa(zz):
        ratios, _, scale = kappa_terms(zz, x, weights)
        return np.log(ratios * scale)

    grad = np.zeros((len(weights), q))
    for i in range(q):
        e = np.eye(q)[i] * h
        grad[:, i] = (log_kappa(z + e) - log_kappa(z - e)) / (2 * h)
    info = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            ei, ej = np.eye(q)[i] * h, np.eye(q)[j] * h
            second = (reml_loglik(z + ei + ej, x, y) - reml_loglik(z + ei - ej, x, y)
                      - reml_loglik(z - ei + ej, x, y) + reml_loglik(z - ei - ej, x, y)) / (4 * h * h)
            info[i, j] = -second
    info = 0.5 * (info + info.T)
    try:
        cov = np.linalg.inv(info)
        if np.any(np.linalg.eigvalsh(info) <= 0):
            return np.zeros(len(weights))
    except np.linalg.LinAlgError:
        return np.zeros(len(weights))
    var = np.einsum('ki,ij,kj->k', grad, cov, grad)
    return np.where(np.isfinite(var) & (var > 0), np.sqrt(np.maximum(var, 0)), 0.0).clip(max=MAX_KAPPA_LOG_SD)


def tempering(x, y, directions):
    """Tempering factor kappa of the largest-factor combination among `directions`
    (each a vector c in design-column order) on time-ordered rows.

    kappa = [w'Rw / w'w] n/(n - tr(HR)) exp(tau^2/2), R the REML AR(q) residual
    correlation, w = X(X'X)^-1 c, tau the delta-method SD of log kappa; bounded by
    3 x the fixed-b-scaled prewhitened Newey-West ratio of the score w * resid;
    clamped to [1, n/(p+2)].
    """
    rows, cols = x.shape
    resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    bandwidth = int(np.floor(4 * (rows / 100) ** (2 / 9)))
    base = {'rows': rows, 'kappa': 1.0, 'raw_ratio': 1.0, 'ar_ratio': 1.0, 'df_loss': float(cols),
            'kappa_log_sd': 0.0, 'hac_ratio': 1.0, 'fixed_b': 1.0, 'bandwidth': bandwidth,
            'score_ar_order': 0, 'residual_ar_order': 0, 'bounded': False}
    if resid @ resid <= 1e-20 * ((y - y.mean()) @ (y - y.mean())):
        # Exact fit: the residuals are rounding error, kappa stays 1.
        return base
    weights = [x @ np.linalg.solve(x.T @ x, np.asarray(c, float)) for c in directions]
    fb = fixed_b(bandwidth + 1, rows)
    z = reml_autoregression(x, y)
    ratios, df_loss, scale = kappa_terms(z, x, weights)
    taus = kappa_log_sd(z, x, y, weights)
    best = None
    for wt, ratio, tau in zip(weights, ratios, taus):
        hac, _, score_order = score_hac_ratio(wt * resid, rows)
        unbounded = scale * ratio * np.exp(0.5 * tau * tau)
        bound = HAC_BOUND * hac * fb**2
        raw = min(unbounded, bound)
        if best is None or raw > best['raw_ratio']:
            best = dict(base, raw_ratio=raw, ar_ratio=float(ratio), df_loss=df_loss,
                        kappa_log_sd=float(tau), hac_ratio=hac, fixed_b=fb,
                        score_ar_order=score_order, residual_ar_order=len(z),
                        bounded=bool(bound < unbounded))
    best['kappa'] = float(min(max(best['raw_ratio'], 1.0), max(rows / (cols + 2), 1.0)))
    return best


def mediation_tempering(x_m, y_m, x_o, y_o):
    """Per-mechanism kappa of a linear mediation (columns [1, t, (m,) z...]).

    The mediator mechanism is tempered along its path slope a (e_1); the outcome
    mechanism along the direct c' (e_1), mediated b (e_2) and total c' + a_hat b
    (e_1 + a_hat e_2) gradients, kappa the largest (a_hat the OLS mediator slope).
    """
    a_hat = np.linalg.lstsq(x_m, y_m, rcond=None)[0][1]
    unit = lambda p, entries: np.array([entries.get(i, 0.0) for i in range(p)])
    mediator = tempering(x_m, y_m, [unit(x_m.shape[1], {1: 1.0})])
    cols = x_o.shape[1]
    outcome = tempering(x_o, y_o, [unit(cols, d) for d in ({1: 1.0}, {2: 1.0}, {1: 1.0, 2: a_hat})])
    return [dict(mediator, mechanism='m'), dict(outcome, mechanism='y')]


def dirichlet_mean_variance(values):
    """Var(sum_i omega_i values_i), omega ~ Dirichlet(1, ..., 1): S_cc / (n (n + 1))."""
    centred = values - values.mean()
    return centred @ centred / (len(values) * (len(values) + 1))


# Lag-aligned designs (column order: intercept, then parents).
x_window, y_window = np.column_stack([np.ones(n-2), t[1:-1], t[:-2]]), s[2:]
x_a, y_a = np.column_stack([np.ones(n-1), t[:-1]]), m[1:]
x_b, y_b = np.column_stack([np.ones(n-2), m[1:-1], m[:-2]]), u[2:]
contrast = np.array([0, 1, 1])
# The window effect is b1+b2; the stationary effect a(b1+b2) has gradient
# proportional to e_a in the m mechanism and to (0, 1, 1) in the y mechanism
# (kappa is invariant to the scale and sign of the direction).
kappas = {
    'window': [tempering(x_window, y_window, [contrast])],
    'stationary_window': [
        dict(tempering(x_a, y_a, [np.array([0, 1])]), mechanism='m'),
        dict(tempering(x_b, y_b, [contrast]), mechanism='y'),
    ],
}
x_ma, y_ma = np.column_stack([np.ones(n-1), t[:-1]]), m[1:]
x_mb, y_mb = np.column_stack([np.ones(n-1), t[:-1], m[1:]]), o[1:]
x_ca, y_ca = np.column_stack([np.ones(n-1), tc[:-1], z[:-1]]), mc[1:]
x_cb, y_cb = np.column_stack([np.ones(n-1), tc[:-1], mc[1:], z[:-1]]), oc[1:]
# Bayesian temporal mediation tempers both mechanisms along their path gradients.
kappas['mediation'] = mediation_tempering(x_ma, y_ma, x_mb, y_mb)
kappas['confounded_mediation'] = mediation_tempering(x_ca, y_ca, x_cb, y_cb)
# Serially dependent Pulse design: AR(2)(0.3, 0.5) treatment and residual (unit
# innovations, 500-step burn-in) on the same 96 rows, y = 0.8 t_{t-1} + e. The REML
# order-2 path, its delta-method spread and the residual-scale factor all move here.


def ar2_series(rows, phi=(0.3, 0.5), burn=500):
    out = np.zeros(rows + burn)
    innovations = rng.normal(size=rows + burn)
    for i in range(rows + burn):
        out[i] = innovations[i] + sum(phi[j] * out[i - 1 - j] for j in range(2) if i > j)
    return out[burn:]


td = 0.5 * ar2_series(n)
ed = 0.35 * ar2_series(n)
yd = 0.8 * np.roll(td, 1) + ed
x_d, y_d = np.column_stack([np.ones(n - 1), td[:-1]]), yd[1:]
kappas['dependent_pulse'] = [tempering(x_d, y_d, [np.array([0, 1])])]
k_d = kappas['dependent_pulse'][0]['kappa']
k_window = kappas['window'][0]['kappa']
k_a, k_b = (f['kappa'] for f in kappas['stationary_window'])
k_ma, k_mb = (f['kappa'] for f in kappas['mediation'])
k_ca, k_cb = (f['kappa'] for f in kappas['confounded_mediation'])

expected = {}
for scale in [0.1, 10.0]:
    cm, cv = posterior(np.column_stack([np.ones(n), t, w, t * (w - w.mean())]), y, scale)
    rm, rv = posterior(np.column_stack([np.ones(n), t]), r, scale)
    am, av = posterior(x_ma, y_ma, scale, k_ma)
    bm, bv = posterior(x_mb, y_mb, scale, k_mb)
    sm, sv = posterior(x_window, y_window, scale, k_window)
    tam, tav = posterior(x_a, y_a, scale, k_a)
    um, uv = posterior(x_b, y_b, scale, k_b)
    cam, cav = posterior(x_ca, y_ca, scale, k_ca)
    cbm, cbv = posterior(x_cb, y_cb, scale, k_cb)
    dm, dv = posterior(x_d, y_d, scale, k_d)
    # One stationary a multiplies b1+b2, so covariance between its two
    # appearances is retained. Each regression uses unique observed rows.
    b_mean, b_var = contrast @ um, contrast @ uv @ contrast
    # Conditional: b_t + b_tx (wbar_D - wbar), wbar_D a Bayesian-bootstrap draw of
    # the modifier mean independent of beta; E[wbar_D - wbar] = 0.
    conditional_var = cv[1, 1] + (cv[3, 3] + cm[3]**2) * dirichlet_mean_variance(w)
    expected[str(scale)] = {
        'conditional': [cm[1], conditional_var],
        'response': [[float(np.array([1, level]) @ rm), float(np.array([1, level]) @ rv @ np.array([1, level]))] for level in [0, 1]],
        'mediation': [am[1] * bm[2], av[1, 1] * bv[2, 2] + av[1, 1] * bm[2]**2 + bv[2, 2] * am[1]**2],
        'window': [contrast @ sm, contrast @ sv @ contrast],
        'stationary_window': [tam[1] * b_mean, tav[1, 1] * b_var + tav[1, 1] * b_mean**2 + b_var * tam[1]**2],
        'confounded_mediation': [cam[1] * cbm[2], cav[1, 1] * cbv[2, 2] + cav[1, 1] * cbm[2]**2 + cbv[2, 2] * cam[1]**2],
        'dependent_pulse': [dm[1], dv[1, 1]],
    }
Path(__file__).with_name('expected.json').write_text(json.dumps({'data': {k: v.tolist() for k,v in dict(t=t,w=w,y=y,r=r,m=m,o=o,s=s,u=u,z=z,tc=tc,mc=mc,oc=oc,td=td,yd=yd).items()}, 'posterior': expected, 'tempering': kappas}, indent=2)+'\n')
