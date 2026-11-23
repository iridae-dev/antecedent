# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for mechanism-Shapley attribution of a distribution change.

Regenerate with:  uv run --script distribution_change_reference.py > distribution_change_reference.json

Two populations of a three-node system X -> Y -> Z with the chord X -> Z are drawn from linear
Gaussian mechanisms that differ in every node.  The reference fits each node on each population
by ordinary least squares in 80 digits (normal equations; sigma^2 = RSS / (n - p), p the number
of fitted columns including the intercept) and then, for every coalition S of the three
mechanisms, builds the hybrid system (the comparison fit for nodes in S, the baseline fit for the
rest) and computes the exact law of Z (mean and variance by propagation of moments).  The
measures are the mean, the variance and the Gaussian KL(N(mu_S, v_S) || N(mu_0, v_0)); the
Shapley value of a mechanism is the average of its marginal contribution over all 3! orders.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import itertools
import json
import random

import mpmath as mp

mp.mp.dps = 80
mpf = mp.mpf

N = 80
BASE = {"x": (0.5, 1.0), "y": (1.0, 0.8, 0.7), "z": (-0.5, 0.6, 0.9, 0.8)}
COMP = {"x": (1.5, 1.4), "y": (0.5, 1.3, 0.9), "z": (0.7, 0.2, 1.4, 1.2)}


def draw(params, n, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = params["x"][0] + params["x"][1] * rng.gauss(0, 1)
        y = params["y"][0] + params["y"][1] * x + params["y"][2] * rng.gauss(0, 1)
        z = params["z"][0] + params["z"][1] * x + params["z"][2] * y + params["z"][3] * rng.gauss(0, 1)
        rows.append([x, y, z])
    return rows


def ols(cols, y):
    n = len(y)
    X = [[mpf(1)] + [mpf(c[i]) for c in cols] for i in range(n)]
    p = len(X[0])
    A = mp.matrix(p, p)
    b = mp.matrix(p, 1)
    for i in range(n):
        for j in range(p):
            b[j] += X[i][j] * mpf(y[i])
            for k in range(p):
                A[j, k] += X[i][j] * X[i][k]
    beta = mp.lu_solve(A, b)
    rss = sum((mpf(y[i]) - sum(beta[j] * X[i][j] for j in range(p))) ** 2 for i in range(n))
    sigma = mp.sqrt(rss / (n - p))
    return [beta[j] for j in range(p)], sigma


def fit(rows):
    x = [r[0] for r in rows]
    y = [r[1] for r in rows]
    z = [r[2] for r in rows]
    bx, sx = ols([], x)
    by, sy = ols([x], y)
    bz, sz = ols([x, y], z)
    return {"x": (bx, sx), "y": (by, sy), "z": (bz, sz)}


base_rows = draw(BASE, N, 20240601)
comp_rows = draw(COMP, N, 20240602)
fb, fc = fit(base_rows), fit(comp_rows)


def law(mask):
    f = {k: (fc[k] if mask[i] else fb[k]) for i, k in enumerate("xyz")}
    (ax,), sx = f["x"][0], f["x"][1]
    (ay, by), sy = f["y"][0], f["y"][1]
    (az, bz1, bz2), sz = f["z"][0], f["z"][1]
    mx, vx = ax, sx**2
    my = ay + by * mx
    vy = by**2 * vx + sy**2
    cxy = by * vx
    mz = az + bz1 * mx + bz2 * my
    vz = bz1**2 * vx + bz2**2 * vy + 2 * bz1 * bz2 * cxy + sz**2
    return mz, vz


def kl(m1, v1, m0, v0):
    return (mp.log(v0 / v1) + (v1 + (m1 - m0) ** 2) / v0 - 1) / 2


def shapley(value):
    phi = [mpf(0)] * 3
    perms = list(itertools.permutations(range(3)))
    for perm in perms:
        mask = [0, 0, 0]
        for p in perm:
            before = value(tuple(mask))
            mask[p] = 1
            phi[p] += value(tuple(mask)) - before
    return [x / len(perms) for x in phi]


m0, v0 = law((0, 0, 0))
measures = {}
measures["mean"] = (lambda mk: law(mk)[0], lambda: law((1, 1, 1))[0] - m0)
measures["variance"] = (lambda mk: law(mk)[1], lambda: law((1, 1, 1))[1] - v0)


def kl_value(mk):
    if not any(mk):
        return mpf(0)
    m, v = law(mk)
    return kl(m, v, m0, v0)


measures["kl"] = (kl_value, lambda: kl_value((1, 1, 1)))
out = {
    "n": N,
    "rows": base_rows + comp_rows,
    "fit": {
        pop: {k: {"beta": [float(b) for b in v[0]], "sigma": float(v[1])} for k, v in f.items()}
        for pop, f in (("baseline", fb), ("comparison", fc))
    },
}
for name, (value, total) in measures.items():
    out[name] = {
        "total": float(total()),
        "phi": [float(p) for p in shapley(value)],
        "empty": float(value((0, 0, 0))),
        "full": float(value((1, 1, 1))),
    }
print(json.dumps(out, indent=1))
