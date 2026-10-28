"""Maximum-likelihood fit of the binary Verma nested-Markov model, by a different optimizer.

The 16-cell law of X1 -> X2 -> X3 -> X4, X2 <-> X4 is
    P = a(x1) * c(x3 | x2) * Q(x2, x4 | x1, x3),
    Q(0,0) = g, Q(0,1) = q2 - g, Q(1,0) = q4 - g, Q(1,1) = 1 - q2 - q4 + g,
with a, c0, c1, q2[x1], q4[x3], g[x1][x3] (eleven free parameters; g is four values). The product
profiles g inside block coordinate ascent over q2, q4. Here the whole eleven-parameter
log-likelihood is maximized with scipy SLSQP from twelve starts (the empirical projection and
eleven random interior points); the maximum log-likelihood, the do-means and the plug-in means are
written to the JSON. Cases include a table far from the model, tiny counts, a table of mixed scales
(counts spanning nine decades) and a table near the boundary of the parameter box.

    python3 nested_markov_binary_reference.py > nested_markov_binary_reference.json
"""

import json
import math

import numpy as np
from scipy.optimize import minimize


def idx(x1, x2, x3, x4):
    return x1 * 8 + x2 * 4 + x3 * 2 + x4


def cells(p):
    a, c0, c1, q20, q21, q40, q41, g00, g01, g10, g11 = p
    c = [c0, c1]
    q2 = [q20, q21]
    q4 = [q40, q41]
    g = [[g00, g01], [g10, g11]]
    out = np.zeros(16)
    for x1 in range(2):
        for x2 in range(2):
            for x3 in range(2):
                for x4 in range(2):
                    gg = g[x1][x3]
                    q = [[gg, q2[x1] - gg], [q4[x3] - gg, 1 - q2[x1] - q4[x3] + gg]][x2][x4]
                    pa = a if x1 == 0 else 1 - a
                    pc = c[x2] if x3 == 0 else 1 - c[x2]
                    out[idx(x1, x2, x3, x4)] = pa * pc * q
    return out


def feasible_margin(p):
    a, c0, c1, q20, q21, q40, q41, g00, g01, g10, g11 = p
    m = [a, c0, c1]
    for x1, q2 in enumerate((q20, q21)):
        for x3, q4 in enumerate((q40, q41)):
            g = (g00, g01, g10, g11)[2 * x1 + x3]
            m += [g, q2 - g, q4 - g, 1 - q2 - q4 + g]
    return m


def loglik(p, n):
    c = cells(p)
    if (c <= 0).any():
        return -1e300
    return float((n * np.log(c)).sum())


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def q_block(n, z):
    """Unconstrained z (8) -> q2[2], q4[2], g[4] strictly inside the feasible box."""
    q2 = sigmoid(z[0:2])
    q4 = sigmoid(z[2:4])
    g = []
    for x1 in range(2):
        for x3 in range(2):
            lo = max(0.0, q2[x1] + q4[x3] - 1.0)
            hi = min(q2[x1], q4[x3])
            g.append(lo + (hi - lo) * sigmoid(z[4 + 2 * x1 + x3]))
    return q2, q4, g


def q_loglik(n, q2, q4, g):
    t = n.reshape(2, 2, 2, 2)
    total = 0.0
    for x1 in range(2):
        for x3 in range(2):
            gg = g[2 * x1 + x3]
            cells4 = {(0, 0): gg, (0, 1): q2[x1] - gg, (1, 0): q4[x3] - gg, (1, 1): 1 - q2[x1] - q4[x3] + gg}
            for (x2, x4), pr in cells4.items():
                if pr <= 0:
                    return -1e300
                total += t[x1, x2, x3, x4] * math.log(pr)
    return total


def fit(n):
    """Maximum of the eleven-parameter likelihood.

    a and c(x3 | x2) multiply the c-factor and have the closed-form maximizers below (the
    log-likelihood is a sum of the three blocks). The Q block (q2, q4, g: eight parameters) is
    maximized in an unconstrained logistic chart with BFGS, Nelder-Mead polishing and SLSQP
    from many starts; the best log-likelihood found is kept.
    """
    tot = n.sum()
    t = n.reshape(2, 2, 2, 2)
    a = t[0].sum() / tot
    c = [t[:, x2, 0, :].sum() / t[:, x2].sum() for x2 in range(2)]
    best = None
    rng = np.random.default_rng(12345)

    def neg(z):
        q2, q4, g = q_block(n, z)
        return -q_loglik(n, q2, q4, g) / tot

    starts = [np.zeros(8)] + [rng.normal(0, 2.0, 8) for _ in range(40)]
    for s in starts:
        z = s
        for method in ("BFGS", "Nelder-Mead", "BFGS"):
            opts = {"gtol": 1e-13, "maxiter": 20000} if method == "BFGS" else {"xatol": 1e-12, "fatol": 1e-18, "maxiter": 40000, "maxfev": 80000}
            r = minimize(neg, z, method=method, options=opts)
            z = r.x
        v = -neg(z) * tot
        if best is None or v > best[0]:
            best = (v, z)
    q2, q4, g = q_block(n, best[1])
    p = [a, c[0], c[1], q2[0], q2[1], q4[0], q4[1], g[0], g[1], g[2], g[3]]
    ll = loglik(p, n)
    return ll, np.array(p)


def model_means(p):
    a, c0, c1, q20, q21, q40, q41 = p[:7]
    c = [c0, c1]
    q4 = [q40, q41]
    out = []
    for x2 in range(2):
        out.append(c[x2] * (1 - q4[0]) + (1 - c[x2]) * (1 - q4[1]))
    return out


def plugin_means(n):
    t = (n / n.sum()).reshape(2, 2, 2, 2)
    out = []
    for x2 in range(2):
        s = 0.0
        for x3 in range(2):
            p23 = t[:, x2, x3, :].sum() / t[:, x2, :, :].sum()
            inner = 0.0
            for x1 in range(2):
                for x2p in range(2):
                    inner += t[x1, x2p, :, :].sum() * t[x1, x2p, x3, 1] / t[x1, x2p, x3, :].sum()
            s += p23 * inner
        out.append(s)
    return out


def scm(px1, eps):
    pu = 0.5
    def p2(x1, u):
        return 0.2 + 0.5 * x1 * 0.6 + 0.3 * u * 0.9
    def p3(x2):
        return 0.25 + 0.5 * x2
    def p4(x3, u, x1):
        return 0.15 + 0.5 * x3 * 0.5 + 0.3 * u + eps * x1
    law = np.zeros(16)
    for x1 in range(2):
        for u in range(2):
            for x2 in range(2):
                for x3 in range(2):
                    for x4 in range(2):
                        pr = (px1 if x1 else 1 - px1) * (pu if u else 1 - pu)
                        pr *= p2(x1, u) if x2 else 1 - p2(x1, u)
                        pr *= p3(x2) if x3 else 1 - p3(x2)
                        pr *= p4(x3, u, x1) if x4 else 1 - p4(x3, u, x1)
                        law[idx(x1, x2, x3, x4)] += pr
    return law


def build():
    rng = np.random.default_rng(7)
    tables = {
        "scm_direct_effect_1e6": scm(0.4, 0.15) * 1e6,
        "scm_direct_effect_1e12": scm(0.4, 0.15) * 1e12,
        "tiny_counts": np.array([1, 2, 1, 3, 2, 1, 1, 2, 3, 1, 2, 1, 1, 4, 2, 1], dtype=float),
        "mixed_decades_boundary": np.array([10 ** (i % 10) for i in range(16)], dtype=float) * (1 + 0.1 * np.arange(16)),
    "mixed_decades": np.array([10 ** (i % 7) for i in range(16)], dtype=float) * (1 + 0.1 * np.arange(16)),
        "skewed_box": np.array([2000, 3, 5, 1, 9, 4000, 2, 8, 6, 1, 3000, 7, 4, 2, 5, 900], dtype=float),
        "random_a": rng.integers(5, 400, 16).astype(float),
        "random_b": rng.integers(1, 40, 16).astype(float),
        "random_c": rng.integers(50, 5000, 16).astype(float),
    }
    # heavy-tailed tables (about three decades of cell scale); keep drawing until the
    # constrained maximum is interior, so the fit is not decided by the boundary rule
    heavy = np.random.default_rng(99)
    kept = 0
    while kept < 3:
        n = np.round(np.exp(heavy.normal(5.0, 1.6, 16))) + 1.0
        params = fit(n)[1]
        if params[:7].min() > 1e-4 and cells(params).min() > 1e-6:
            tables[f"heavy_tail_{kept}"] = n
            kept += 1
    out = []
    for name, n in tables.items():
        ll, p = fit(n)
        out.append({
            "name": name,
            "counts": [float(v) for v in n],
            "max_log_likelihood": ll,
            "params": [float(v) for v in p],
            "model_means": [float(v) for v in model_means(p)],
            "plugin_means": [float(v) for v in plugin_means(n)],
            "min_fitted_cell": float(cells(p).min()),
        "min_fitted_margin": float(min(min(feasible_margin(p)), 1 - max(p[:7]))),
        "saturated_log_likelihood": float((n * np.log(n / n.sum())).sum()),
        })
    return out, tables


if __name__ == "__main__":
    out, _ = build()
    print(json.dumps({"cases": out}, indent=1))
