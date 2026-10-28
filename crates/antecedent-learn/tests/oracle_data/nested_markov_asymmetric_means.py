"""Independent posterior means of the binary Verma model for an asymmetric count table.

Model (cell index i*8 + t*4 + j*2 + y over X1=i, X2=t, X3=j, X4=y):
  p(i, t, j, y) = a_i * c_{t,j} * cell_{ij}[t, y]
with a = P(X1=0), c_{t,0} = c_t = P(X3=0 | X2=t), and in context (i, j) the 2x2 table of (X2, X4)
having margins q2_i = P(X2=0 | X1=i) and q4_j = P(X4=0 | X3=j) and association g_ij = P(X2=0, X4=0):
  cells [g, q2_i - g, q4_j - g, 1 - q2_i - q4_j + g] in order (t, y) = (0,0), (0,1), (1,0), (1,1).
Prior: independent Beta(1, 1) kernels on the eleven coordinates, restricted to positive cells.

The three singleton coordinates a, c_0, c_1 are independent of the rest and have Beta posteriors
that are written down directly.  The eight remaining coordinates are integrated here by
Gauss-Legendre quadrature over (q20, q21, q40, q41) after integrating each association g_ij
over its feasible interval with a 16-node rule, which is exact for the polynomial likelihood.
No sampler, seed or production code is used.

    uv run --with numpy python nested_markov_asymmetric_means.py
"""

import json

import numpy as np

COUNTS = [3, 1, 2, 5, 1, 4, 2, 1, 6, 2, 1, 3, 2, 1, 4, 2]


def gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


def posterior_means(counts, n_margin=48, n_g=16):
    counts = np.asarray(counts, dtype=float).reshape(2, 2, 2, 2)  # i, t, j, y
    # singleton coordinates
    a0 = 1 + counts[0].sum()
    a1 = 1 + counts[1].sum()
    a = a0 / (a0 + a1)
    c = []
    for t in range(2):
        s0 = 1 + counts[:, t, 0, :].sum()
        s1 = 1 + counts[:, t, 1, :].sum()
        c.append(s0 / (s0 + s1))
    xs, ws = gl(n_margin)
    gx, gw = gl(n_g)
    Q = np.stack(np.meshgrid(xs, xs, xs, xs, indexing="ij"), axis=-1)  # q20 q21 q40 q41
    W = np.einsum("a,b,c,d->abcd", ws, ws, ws, ws)
    Z = np.ones_like(W)
    G = []
    for i in range(2):
        for j in range(2):
            q2 = Q[..., i]
            q4 = Q[..., 2 + j]
            lo = np.maximum(q2 + q4 - 1, 0)
            hi = np.minimum(q2, q4)
            zi = np.zeros_like(W)
            mi = np.zeros_like(W)
            for u, wu in zip(gx, gw):
                g = lo + (hi - lo) * u
                cells = [g, q2 - g, q4 - g, 1 - q2 - q4 + g]
                like = np.ones_like(W)
                for t in range(2):
                    for y in range(2):
                        like = like * cells[t * 2 + y] ** counts[i, t, j, y]
                zi += wu * (hi - lo) * like
                mi += wu * (hi - lo) * like * g
            Z *= zi
            G.append(mi / zi)
    total = (W * Z).sum()
    mean = {}
    names = ["q20", "q21", "q40", "q41"]
    for k, name in enumerate(names):
        mean[name] = float((W * Z * Q[..., k]).sum() / total)
    for idx, name in enumerate(["g00", "g01", "g10", "g11"]):
        mean[name] = float((W * Z * G[idx]).sum() / total)
    mean["a"] = float(a)
    mean["c0"] = float(c[0])
    mean["c1"] = float(c[1])
    # c and q4 are independent, so the do-mean is bilinear in their means
    for t in range(2):
        mean[f"mean_do_x2_{t}"] = float(c[t] * (1 - mean["q40"]) + (1 - c[t]) * (1 - mean["q41"]))
    mean["contrast_1_minus_0"] = mean["mean_do_x2_1"] - mean["mean_do_x2_0"]
    return mean


if __name__ == "__main__":
    out = {"counts": COUNTS}
    for n in (32, 48):
        out[f"mean_n{n}"] = posterior_means(COUNTS, n_margin=n)
    sym = posterior_means([1] * 16, n_margin=32)
    out["all_ones_n32"] = sym
    print(json.dumps(out, indent=1))
