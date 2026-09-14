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
    return float(np.clip(rho + (1 + 3 * rho) / len(e), -0.97, 0.97))


def fixed_b(block, rows):
    """Kiefer-Vogelsang Bartlett fixed-b 95% critical value over 1.96."""
    b = min(max(block / rows, 0.0), 1.0)
    return (1.96 + 2.9694 * b + 0.4160 * b**2 - 0.5324 * b**3) / 1.96


def bic_autoregression(e, max_order=4):
    """Yule-Walker AR(q) of e (uncentred autocovariances), q <= 4 minimizing
    n ln sigma2_q + q ln n. Returns (phi, autocorrelations r_0..r_q)."""
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
    phi = best[1]
    return phi, g[:len(phi) + 1] / g[0]


def bartlett(u, bandwidth):
    """Bartlett long-run variance of u (uncentred autocovariances over len(u))."""
    lags = min(bandwidth, len(u) - 1)
    lrv = u @ u + 2 * sum((1 - k / (lags + 1)) * (u[k:] @ u[:-k]) for k in range(1, lags + 1))
    return max(lrv / len(u), 0.0)


def tempering(x, y, c):
    """Long-run-variance ratio kappa of the score of c'beta on time-ordered rows."""
    rows, cols = x.shape
    bandwidth = int(np.floor(4 * (rows / 100) ** (2 / 9)))
    resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    weight = x @ np.linalg.solve(x.T @ x, c)  # row weights of c'beta_hat
    score = weight * resid
    # AR(1)-prewhitened Bartlett long-run variance, recoloured by 1/(1 - rho)^2.
    rho = kendall_rho(score)
    white = score[1:] - rho * score[:-1]
    hac = bartlett(white, bandwidth) / (1 - rho) ** 2 / (score @ score / rows)
    # BIC-selected AR(q), q >= 2, prewhitening recoloured by 1/(1 - sum(phi))^2.
    phi_s, _ = bic_autoregression(score)
    if len(phi_s) >= 2:
        q = len(phi_s)
        white = np.array([score[t] - phi_s @ score[t - q:t][::-1] for t in range(q, rows)])
        recolour = max(1 - phi_s.sum(), 1 - 0.97)
        hac = max(hac, bartlett(white, bandwidth) / recolour**2 / (score @ score / rows))
    # AR(1)-residual quadratic-form ratio w' Gamma w / (gamma0 w'w) given the design.
    rho_e = kendall_rho(resid)
    lag = np.abs(np.subtract.outer(np.arange(rows), np.arange(rows)))
    ar = weight @ (rho_e**lag) @ weight / (weight @ weight)
    # Same with the BIC-selected AR(q), q >= 2, residual autocorrelation.
    # The AR(q) term is bounded by 3x the fixed-b-scaled HAC ratio.
    fb = fixed_b(bandwidth + 1, rows)
    phi_e, r_e = bic_autoregression(resid)
    if len(phi_e) >= 2:
        acf = np.zeros(rows)
        acf[:len(r_e)] = r_e
        for k in range(len(r_e), rows):
            acf[k] = phi_e @ acf[k - len(phi_e):k][::-1]
        quadratic = weight @ acf[lag] @ weight / (weight @ weight)
        ar = max(ar, min(quadratic, 3.0 * hac * fb**2))
    raw = max(hac * fb**2, ar)
    kappa = min(max(raw, 1.0), max(rows / (cols + 2), 1.0))
    return {
        'rows': rows, 'kappa': kappa, 'raw_ratio': raw, 'hac_ratio': hac,
        'fixed_b': fb, 'ar_ratio': ar, 'bandwidth': bandwidth,
        'score_ar_order': len(phi_s), 'residual_ar_order': len(phi_e),
    }


def mediation_tempering(x_m, y_m, x_o, y_o):
    """Per-mechanism kappa of a linear mediation (columns [1, t, (m,) z...]).

    The mediator mechanism is tempered along its path slope a (e_1); the outcome
    mechanism along the direct c' (e_1), mediated b (e_2) and total c' + a_hat b
    (e_1 + a_hat e_2) gradients, kappa the largest (a_hat the OLS mediator slope).
    """
    a_hat = np.linalg.lstsq(x_m, y_m, rcond=None)[0][1]
    unit = lambda p, entries: np.array([entries.get(i, 0.0) for i in range(p)])
    mediator = tempering(x_m, y_m, unit(x_m.shape[1], {1: 1.0}))
    cols = x_o.shape[1]
    outcome = max(
        (tempering(x_o, y_o, unit(cols, d)) for d in ({1: 1.0}, {2: 1.0}, {1: 1.0, 2: a_hat})),
        key=lambda f: f['raw_ratio'],
    )
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
    'window': [tempering(x_window, y_window, contrast)],
    'stationary_window': [
        dict(tempering(x_a, y_a, np.array([0, 1])), mechanism='m'),
        dict(tempering(x_b, y_b, contrast), mechanism='y'),
    ],
}
x_ma, y_ma = np.column_stack([np.ones(n-1), t[:-1]]), m[1:]
x_mb, y_mb = np.column_stack([np.ones(n-1), t[:-1], m[1:]]), o[1:]
x_ca, y_ca = np.column_stack([np.ones(n-1), tc[:-1], z[:-1]]), mc[1:]
x_cb, y_cb = np.column_stack([np.ones(n-1), tc[:-1], mc[1:], z[:-1]]), oc[1:]
# Bayesian temporal mediation tempers both mechanisms along their path gradients.
kappas['mediation'] = mediation_tempering(x_ma, y_ma, x_mb, y_mb)
kappas['confounded_mediation'] = mediation_tempering(x_ca, y_ca, x_cb, y_cb)
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
    }
Path(__file__).with_name('expected.json').write_text(json.dumps({'data': {k: v.tolist() for k,v in dict(t=t,w=w,y=y,r=r,m=m,o=o,s=s,u=u,z=z,tc=tc,mc=mc,oc=oc).items()}, 'posterior': expected, 'tempering': kappas}, indent=2)+'\n')
