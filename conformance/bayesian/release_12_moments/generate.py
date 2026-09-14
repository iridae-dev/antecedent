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
    lags = min(bandwidth, len(white) - 1)
    lrv = white @ white + 2 * sum(
        (1 - k / (lags + 1)) * (white[k:] @ white[:-k]) for k in range(1, lags + 1)
    )
    hac = max(lrv / len(white), 0.0) / (1 - rho) ** 2 / (score @ score / rows)
    # AR(1)-residual quadratic-form ratio w' Gamma w / (gamma0 w'w) given the design.
    rho_e = kendall_rho(resid)
    lag = np.abs(np.subtract.outer(np.arange(rows), np.arange(rows)))
    ar = weight @ (rho_e**lag) @ weight / (weight @ weight)
    fb = fixed_b(bandwidth + 1, rows)
    raw = max(hac * fb**2, ar)
    kappa = min(max(raw, 1.0), max(rows / (cols + 2), 1.0))
    return {
        'rows': rows, 'kappa': kappa, 'raw_ratio': raw, 'hac_ratio': hac,
        'fixed_b': fb, 'ar_ratio': ar, 'bandwidth': bandwidth,
    }


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
k_window = kappas['window'][0]['kappa']
k_a, k_b = (f['kappa'] for f in kappas['stationary_window'])

expected = {}
for scale in [0.1, 10.0]:
    cm, cv = posterior(np.column_stack([np.ones(n), t, w, t * (w - w.mean())]), y, scale)
    rm, rv = posterior(np.column_stack([np.ones(n), t]), r, scale)
    am, av = posterior(np.column_stack([np.ones(n-1), t[:-1]]), m[1:], scale)
    bm, bv = posterior(np.column_stack([np.ones(n-1), t[:-1], m[1:]]), o[1:], scale)
    sm, sv = posterior(x_window, y_window, scale, k_window)
    tam, tav = posterior(x_a, y_a, scale, k_a)
    um, uv = posterior(x_b, y_b, scale, k_b)
    cam, cav = posterior(np.column_stack([np.ones(n-1), tc[:-1], z[:-1]]), mc[1:], scale)
    cbm, cbv = posterior(np.column_stack([np.ones(n-1), tc[:-1], mc[1:], z[:-1]]), oc[1:], scale)
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
