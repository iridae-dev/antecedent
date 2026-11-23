"""Expected-information covariance of the Verma pilot's do-means, by high-precision differentiation.

At the maximum-likelihood parameters theta (from the SLSQP fit of nested_markov_binary_reference.py)
the expected information of N multinomial draws is I = N sum_i (dp_i/dtheta)(dp_i/dtheta)^T / p_i
over the eleven parameters (a, c0, c1, q2[0], q2[1], q4[0], q4[1], g00, g01, g10, g11). The cell
derivatives are taken here by mpmath central differences at 60 digits of the cell function (not the
analytic derivative the product uses), I is inverted with mpmath, and the delta covariance of
(mean0, mean1, mean1 - mean0) is J I^-1 J^T with J differentiated the same way. Integer count tables
only, since the product refuses fractional counts.

    python3 nested_markov_fisher_reference.py > nested_markov_fisher_reference.json
"""

import json

import mpmath as mp
import numpy as np

from nested_markov_binary_reference import build, cells, fit

mp.mp.dps = 60


def cells_mp(t):
    a, c0, c1, q20, q21, q40, q41, g00, g01, g10, g11 = t
    c = [c0, c1]
    q2 = [q20, q21]
    q4 = [q40, q41]
    g = [[g00, g01], [g10, g11]]
    out = []
    for x1 in range(2):
        for x2 in range(2):
            for x3 in range(2):
                for x4 in range(2):
                    gg = g[x1][x3]
                    q = [[gg, q2[x1] - gg], [q4[x3] - gg, 1 - q2[x1] - q4[x3] + gg]][x2][x4]
                    pa = a if x1 == 0 else 1 - a
                    pc = c[x2] if x3 == 0 else 1 - c[x2]
                    out.append(pa * pc * q)
    return out


def means_mp(t):
    c0, c1, q40, q41 = t[1], t[2], t[5], t[6]
    m0 = c0 * (1 - q40) + (1 - c0) * (1 - q41)
    m1 = c1 * (1 - q40) + (1 - c1) * (1 - q41)
    return [m0, m1, m1 - m0]


def partial(f, t, k, h):
    tp = list(t)
    tm = list(t)
    tp[k] += h
    tm[k] -= h
    fp, fm = f(tp), f(tm)
    return [(u - v) / (2 * h) for u, v in zip(fp, fm)]


out = []
_, tables = build()
for name, n in tables.items():
    n = np.round(n)
    if name in ("scm_direct_effect_1e12", "mixed_decades_boundary"):
        continue
    total = float(n.sum())
    ll, p = fit(n)
    if min(cells(p).min(), min(p[:7]), min(1 - np.array(p[:7]))) < 1e-8:
        continue  # the constrained maximum is on the boundary: the product refuses such a fit
    t = [mp.mpf(float(v)) for v in p]
    h = mp.mpf(10) ** -20
    dcell = [partial(cells_mp, t, k, h) for k in range(11)]  # dcell[k][i]
    pc = cells_mp(t)
    info = mp.matrix(11, 11)
    for i in range(11):
        for j in range(11):
            info[i, j] = mp.mpf(total) * mp.fsum(dcell[i][c] * dcell[j][c] / pc[c] for c in range(16))
    cov = info ** -1
    jac = mp.matrix(3, 11)
    for k in range(11):
        col = partial(means_mp, t, k, h)
        for r in range(3):
            jac[r, k] = col[r]
    eff = jac * cov * jac.T
    out.append({
        "name": name,
        "counts": [float(v) for v in n],
        "params": [float(v) for v in p],
        "effect_covariance": [[float(eff[i, j]) for j in range(3)] for i in range(3)],
        "sample_count": total,
        "mean_values": [float(v) for v in means_mp(t)],
        "condition_hint": float(mp.log10(mp.norm(info, 1) * mp.norm(cov, 1))),
    })
print(json.dumps({"cases": out}, indent=1))
