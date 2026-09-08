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

def posterior(x, y, scale):
    v = np.linalg.inv(x.T @ x + np.eye(x.shape[1]) / scale**2)
    mean = v @ x.T @ y
    a = 0.001 + len(y) / 2
    b = 0.001 + (y @ y - mean @ np.linalg.solve(v, mean)) / 2
    return mean, v * b / (a - 1)

expected = {}
for scale in [0.1, 10.0]:
    cm, cv = posterior(np.column_stack([np.ones(n), t, w, t * (w - w.mean())]), y, scale)
    rm, rv = posterior(np.column_stack([np.ones(n), t]), r, scale)
    am, av = posterior(np.column_stack([np.ones(n-1), t[:-1]]), m[1:], scale)
    bm, bv = posterior(np.column_stack([np.ones(n-1), t[:-1], m[1:]]), o[1:], scale)
    sm, sv = posterior(np.column_stack([np.ones(n-2), t[1:-1], t[:-2]]), s[2:], scale)
    um, uv = posterior(np.column_stack([np.ones(n-2), m[1:-1], m[:-2]]), u[2:], scale)
    cam, cav = posterior(np.column_stack([np.ones(n-1), tc[:-1], z[:-1]]), mc[1:], scale)
    cbm, cbv = posterior(np.column_stack([np.ones(n-1), tc[:-1], mc[1:], z[:-1]]), oc[1:], scale)
    contrast = np.array([0, 1, 1])
    # One stationary a multiplies b1+b2, so covariance between its two
    # appearances is retained. Each regression uses unique observed rows.
    b_mean, b_var = contrast @ um, contrast @ uv @ contrast
    expected[str(scale)] = {
        'conditional': [cm[1], cv[1, 1]],
        'response': [[float(np.array([1, level]) @ rm), float(np.array([1, level]) @ rv @ np.array([1, level]))] for level in [0, 1]],
        'mediation': [am[1] * bm[2], av[1, 1] * bv[2, 2] + av[1, 1] * bm[2]**2 + bv[2, 2] * am[1]**2],
        'window': [contrast @ sm, contrast @ sv @ contrast],
        'stationary_window': [am[1] * b_mean, av[1, 1] * b_var + av[1, 1] * b_mean**2 + b_var * am[1]**2],
        'confounded_mediation': [cam[1] * cbm[2], cav[1, 1] * cbv[2, 2] + cav[1, 1] * cbm[2]**2 + cbv[2, 2] * cam[1]**2],
    }
Path(__file__).with_name('expected.json').write_text(json.dumps({'data': {k: v.tolist() for k,v in dict(t=t,w=w,y=y,r=r,m=m,o=o,s=s,u=u,z=z,tc=tc,mc=mc,oc=oc).items()}, 'posterior': expected}, indent=2)+'\n')
