# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the Gaussian-kernel local quadratic fit across scales, at 60 digits.

Regenerate with:  uv run --script local_quadratic_precision_reference.py > local_quadratic_precision_reference.json

Written from the definition.  With u_i = x_i - at, w_i = exp(-u_i^2 / (2 h^2)), X_i = (1, u_i, u_i^2):
  G = sum w X X', G2 = sum w^2 X X', beta = G^-1 sum w X y,
  level = beta_0, slope = beta_1, curvature = 2 beta_2,
  r_i = y_i - X_i' beta, influence_i = w_i (G^-1 X_i)_0 r_i, robust level SE = sqrt(sum influence^2),
  sigma2 = sum w r^2 / (sum w - tr(G^-1 G2)), plug-in level SE = sqrt(sigma2 (G^-1 G2 G^-1)_00),
  Kish effective size = (sum w)^2 / sum w^2.
Regimes: unit scale; doses spread over 1e4 and over 1e-4; bandwidths from a tenth of the spread to
fifty times it; a response offset of 1e6; and a dose offset of 1e6 with a unit spread.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 60
rng = random.Random(20261028)


def fit(x, y, at, h):
    xs = [mp.mpf(v) for v in x]
    ys = [mp.mpf(v) for v in y]
    at, h = mp.mpf(at), mp.mpf(h)
    u = [v - at for v in xs]
    w = [mp.exp(-(d / h) ** 2 / 2) for d in u]
    rows = [[mp.mpf(1), d, d * d] for d in u]
    g = mp.zeros(3, 3)
    g2 = mp.zeros(3, 3)
    rhs = mp.zeros(3, 1)
    for wi, r, yi in zip(w, rows, ys):
        for a in range(3):
            rhs[a] += wi * r[a] * yi
            for b in range(3):
                g[a, b] += wi * r[a] * r[b]
                g2[a, b] += wi * wi * r[a] * r[b]
    ginv = g ** -1
    beta = ginv * rhs
    resid = [yi - sum(r[a] * beta[a] for a in range(3)) for r, yi in zip(rows, ys)]
    infl = [wi * sum(ginv[0, a] * r[a] for a in range(3)) * ri for wi, r, ri in zip(w, rows, resid)]
    wsum = sum(w)
    trace = sum((ginv * g2)[a, a] for a in range(3))
    rss = sum(wi * ri * ri for wi, ri in zip(w, resid))
    sigma2 = rss / (wsum - trace)
    sand = ginv * g2 * ginv
    return {
        "value": mp.nstr(beta[0], 30), "slope": mp.nstr(beta[1], 30), "curvature": mp.nstr(2 * beta[2], 30),
        "robust_se": mp.nstr(mp.sqrt(sum(i * i for i in infl)), 30),
        "plugin_se": mp.nstr(mp.sqrt(sigma2 * sand[0, 0]), 30),
        "plugin_slope_se": mp.nstr(mp.sqrt(sigma2 * sand[1, 1]), 30),
        "ess": mp.nstr(wsum ** 2 / sum(wi * wi for wi in w), 30),
        "influence_sum": mp.nstr(sum(infl), 10),
        "influences": [mp.nstr(i, 30) for i in infl],
    }


cases = []


def add(name, x, y, at, h):
    c = {"name": name, "x": x, "y": y, "at": at, "bandwidth": h}
    c.update(fit(x, y, at, h))
    cases.append(c)


def make(n, spread, centre=0.0, y_offset=0.0, noise=0.3):
    x = [centre + spread * rng.uniform(-1, 1) for _ in range(n)]
    s = spread
    y = [y_offset + 1.0 + (v - centre) / s + 0.5 * ((v - centre) / s) ** 2 + 0.3 * mp.sin(3 * (v - centre) / s) + noise * rng.gauss(0, 1) for v in x]
    return x, [float(v) for v in y]


for spread in (1.0, 1e4, 1e-4):
    for factor in (0.1, 0.5, 2.0, 50.0):
        x, y = make(60, spread)
        add(f"spread_{spread:g}_bandwidth_{factor:g}x", x, y, 0.2 * spread, factor * spread)
x, y = make(60, 1.0, y_offset=1e6)
add("response_offset_1e6", x, y, 0.1, 0.4)
x, y = make(60, 1.0, centre=1e6)
add("dose_offset_1e6", x, y, 1e6 + 0.1, 0.4)
x = [round(v * 4) / 4 for v in (rng.uniform(-2, 2) for _ in range(50))]
add("heavy_ties", x, [float(v) for v in (1 + 0.5 * mp.mpf(a) + rng.gauss(0, 0.2) for a in x)], 0.0, 0.7)
print(json.dumps(cases, indent=1))
