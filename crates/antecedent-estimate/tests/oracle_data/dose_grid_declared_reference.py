# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""High-precision reference for the dose-grid row under declared sampling weights, clusters and
serial dependence.

Regenerate with:  uv run --script dose_grid_declared_reference.py > dose_grid_declared_reference.json

Definitions (written from the estimating equation, not from the product code).  The declared
sampling weights s_i with s_i = 0 deleted are normalized to mean one.  At a dose d0 the fit is the
weighted least squares of y on (1, u, u^2), u = D - d0, with row weight w_i = s_i exp(-u_i^2 /
(2 h^2)); the level is the intercept and the derivative the slope.  The weighted score of row i for
coordinate k is psi_i = e_k' (X'WX)^-1 x_i w_i (y_i - x_i' beta).  Three variances of sum_i psi_i:
independent rows sum psi_i^2; clusters G/(G-1) sum_g (sum_{i in g} psi_i)^2; serial rows (in time
order at unit spacing) the Bartlett long-run variance sum_t psi_t^2 + 2 sum_{l=1}^L (1 - l/(L+1))
sum_t psi_t psi_{t-l}.  A contrast uses the difference of the two score series.  The local Kish size
is (sum w)^2 / sum w^2.  The worst-case smoothing bias is M/6 sum |l_i| |u_i|^3 with l_i = e_k'
(X'WX)^-1 x_i w_i, the folded-normal critical value solves Phi(c - t) - Phi(-c - t) = 0.95, and the
sup-t critical value of a two-dose grid is the 0.95 quantile of max(|Z1|, |Z2|) for a bivariate
normal whose correlation is the correlation of the two score series summed over the resampling
units (rows, clusters, or blocks of L + 1 rows).  Everything is evaluated at 50 digits from the
exact binary64 inputs.  Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import math

import mpmath as mp

mp.mp.dps = 50


class Lcg:
    def __init__(self, s): self.s = s
    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)


def fit(dose, y, w_s, d0, h, k):
    d0, h = mp.mpf(d0), mp.mpf(h)
    us = [mp.mpf(d) - d0 for d in dose]
    ws = [s * mp.e ** (-(u * u) / (2 * h * h)) for u, s in zip(us, w_s)]
    A = mp.zeros(3, 3)
    b = mp.zeros(3, 1)
    for u, w, yy in zip(us, ws, y):
        x = [1, u, u * u]
        for a in range(3):
            b[a] += w * x[a] * mp.mpf(yy)
            for c in range(3):
                A[a, c] += w * x[a] * x[c]
    Ainv = A ** -1
    beta = Ainv * b
    ls, es = [], []
    for u, w, yy in zip(us, ws, y):
        x = mp.matrix([1, u, u * u])
        ls.append((Ainv * x)[k] * w)
        es.append(mp.mpf(yy) - (beta[0] + beta[1] * u + beta[2] * u * u))
    return beta, ls, es, ws, us


def variance(psi, dep):
    kind = dep["kind"]
    if kind == "independent":
        return sum(p * p for p in psi)
    if kind == "cluster":
        sums = {}
        for p, g in zip(psi, dep["groups"]):
            sums[g] = sums.get(g, 0) + p
        G = len(sums)
        return mp.mpf(G) / (G - 1) * sum(v * v for v in sums.values())
    L = dep["lag"]
    n = len(psi)
    v = sum(p * p for p in psi)
    for l in range(1, L + 1):
        v += 2 * (1 - mp.mpf(l) / (L + 1)) * sum(psi[t] * psi[t - l] for t in range(l, n))
    return v


def units(n, dep):
    if dep["kind"] == "independent":
        return list(range(n))
    if dep["kind"] == "cluster":
        ids = sorted(set(dep["groups"]))
        return [ids.index(g) for g in dep["groups"]]
    return [i // (dep["lag"] + 1) for i in range(n)]


def folded(t):
    t = mp.mpf(t)
    cov = lambda c: (mp.erf((c - t) / mp.sqrt(2)) - mp.erf((-c - t) / mp.sqrt(2))) / 2
    return mp.findroot(lambda c: cov(c) - mp.mpf("0.95"), (mp.mpf(0), t + 10), solver="anderson", tol=1e-45, maxsteps=200)


def supt2(rho):
    rho = mp.mpf(rho)
    s = mp.sqrt(1 - rho * rho)
    Phi = lambda x: mp.erfc(-x / mp.sqrt(2)) / 2
    phi = lambda x: mp.e ** (-x * x / 2) / mp.sqrt(2 * mp.pi)

    def cover(c):
        return mp.quad(lambda z: phi(z) * (Phi((c - rho * z) / s) - Phi((-c - rho * z) / s)), [-c, 0, c])
    return mp.findroot(lambda c: cover(c) - mp.mpf("0.95"), (mp.mpf("1.9"), mp.mpf("2.6")), solver="anderson", tol=1e-30, maxsteps=100)


def prepare(dose, y, weights, groups):
    """Delete zero-weight rows and normalize the rest to mean one."""
    keep = [i for i in range(len(dose)) if weights is None or weights[i] > 0]
    d = [dose[i] for i in keep]
    yy = [y[i] for i in keep]
    g = None if groups is None else [groups[i] for i in keep]
    if weights is None:
        s = [mp.mpf(1)] * len(keep)
    else:
        raw = [mp.mpf(weights[i]) for i in keep]
        tot = sum(raw)
        s = [r * len(keep) / tot for r in raw]
    return d, yy, s, g, len(dose) - len(keep)


def point(dose, y, s, d0, h, M, dep):
    out = {"dose": d0}
    for name, k in (("level", 0), ("derivative", 1)):
        beta, ls, es, ws, us = fit(dose, y, s, d0, h, k)
        psi = [l * e for l, e in zip(ls, es)]
        se = mp.sqrt(variance(psi, dep))
        bias = M / mp.mpf(6) * sum(abs(l) * abs(u) ** 3 for l, u in zip(ls, us))
        out[name] = float(beta[k])
        out[name + "_se"] = float(se)
        out[name + "_bias"] = float(bias)
        out[name + "_half"] = float(folded(bias / se) * se) if se > 0 else float(bias)
        out[name + "_psi"] = psi
    _, _, _, ws, _ = fit(dose, y, s, d0, h, 0)
    out["ess"] = float(sum(ws) ** 2 / sum(w * w for w in ws))
    return out


def case(name, dose, y, weights, groups, lag, h, grid, M, contrast):
    d, yy, s, g, zero = prepare(dose, y, weights, groups)
    if groups is not None:
        dep = {"kind": "cluster", "groups": g}
    elif lag is not None:
        dep = {"kind": "serial", "lag": lag}
    else:
        dep = {"kind": "independent"}
    pts = [point(d, yy, s, gd, h, M, dep) for gd in grid]
    c = None
    if contrast is not None:
        f, t = contrast
        bt, lt, et, _, ut = fit(d, yy, s, t, h, 0)
        bf, lf, ef, _, uf = fit(d, yy, s, f, h, 0)
        psi = [a * b - c_ * dd for a, b, c_, dd in zip(lt, et, lf, ef)]
        se = mp.sqrt(variance(psi, dep))
        bias = M / mp.mpf(6) * (sum(abs(l) * abs(u) ** 3 for l, u in zip(lt, ut)) + sum(abs(l) * abs(u) ** 3 for l, u in zip(lf, uf)))
        c = {"from": f, "to": t, "estimate": float(bt[0] - bf[0]), "se": float(se),
             "bias_sum": float(bias), "half": float(folded(bias / se) * se)}
    out = {"name": name, "dose": dose, "outcome": y, "weights": weights, "groups": groups, "lag": lag,
           "bandwidth": h, "grid": grid, "curvature_bound": M, "zero_weight_rows": zero,
           "rows_used": len(d), "points": [{k: (v if not k.endswith("_psi") else None) for k, v in p.items()} for p in pts],
           "contrast": c}
    if weights is not None:
        out["kish_weights"] = float(sum(s) ** 2 / sum(v * v for v in s))
        out["weight_sum_used"] = float(sum(mp.mpf(weights[i]) for i in range(len(dose)) if weights[i] > 0))
    if groups is not None:
        out["clusters"] = len(set(g))
    if len(grid) == 2 and not (groups is not None and len(set(g)) < 3):
        u = units(len(d), dep)
        m = max(u) + 1
        r0 = [mp.mpf(0)] * m
        r1 = [mp.mpf(0)] * m
        for i, ui in enumerate(u):
            r0[ui] += pts[0]["level_psi"][i]
            r1[ui] += pts[1]["level_psi"][i]
        s0 = mp.sqrt(sum(v * v for v in r0)); s1 = mp.sqrt(sum(v * v for v in r1))
        rho = sum(a * b for a, b in zip(r0, r1)) / (s0 * s1)
        out["band_rho"] = float(rho)
        out["band_critical"] = float(supt2(rho))
    return out


def design(seed, n, lo, hi, noise=0.3):
    r = Lcg(seed)
    dose = [round(lo + (hi - lo) * r.u(), 3) for _ in range(n)]
    y = [round(math.sin(d) + 0.2 * d * d + noise * (r.u() - 0.5) * (1 + abs(d)), 5) for d in dose]
    return dose, y, r


def main():
    cases = []
    # Weights only: strongly unequal, with two zero-weight rows (deleted) and one extreme weight.
    d, y, r = design(11, 70, 0.0, 3.0)
    w = [round(0.2 + 3.0 * r.u(), 3) for _ in d]
    w[3] = 0.0; w[40] = 0.0; w[17] = 30.0
    cases.append(case("weights_unequal_zero_and_extreme", d, y, w, None, None, 0.6, [0.4, 1.5, 2.6], 2.0, (0.6, 2.4)))
    # Equal weights: must equal the unweighted fit (scale invariance), at a non-unit common scale.
    d, y, r = design(12, 60, 0.0, 3.0)
    cases.append(case("equal_weights_scale_invariant", d, y, [7.5] * len(d), None, None, 0.5, [0.7, 2.2], 1.0, (0.7, 2.2)))
    # Clusters: unbalanced (sizes 1..), singletons and one dominant cluster.
    d, y, r = design(13, 66, 0.0, 3.0)
    sizes = [1, 1, 2, 3, 5, 8, 13, 33]
    g = []
    for k, sz in enumerate(sizes):
        g += [k] * sz
    g = g[: len(d)]
    # Shuffle membership deterministically so clusters are not contiguous in dose.
    order = sorted(range(len(d)), key=lambda i: (i * 37) % 66)
    g = [g[order.index(i)] for i in range(len(d))]
    cases.append(case("clusters_unbalanced_singletons", d, y, None, g, None, 0.6, [0.5, 1.4], 1.5, (0.5, 2.5)))
    # Clusters with weights.
    d, y, r = design(14, 60, 0.0, 3.0)
    w = [round(0.3 + 2.0 * r.u(), 3) for _ in d]
    w[5] = 0.0; w[6] = 0.0
    g = [i % 7 for i in range(len(d))]
    cases.append(case("weighted_clusters_with_zero_weight_rows", d, y, w, g, None, 0.7, [0.6, 1.7, 2.5], 1.0, (0.6, 2.5)))
    # Serial dependence (rows in time order): AR(1)-like noise, lags 1 and 4.
    d, y, r = design(15, 80, 0.0, 3.0, noise=0.1)
    e = 0.0
    y = []
    for dd in d:
        e = 0.8 * e + (r.u() - 0.5)
        y.append(round(math.sin(dd) + 0.2 * dd * dd + e, 5))
    cases.append(case("serial_lag_1", d, y, None, None, 1, 0.5, [0.8, 2.0], 1.0, (0.8, 2.2)))
    cases.append(case("serial_lag_4_weighted", d, y, [round(0.5 + 1.5 * ((i * 0.618) % 1), 3) for i in range(len(d))], None, 4, 0.5, [0.8, 2.0, 2.7], 1.0, (0.8, 2.7)))
    # Tiny sample with two clusters (the smallest admissible).
    d, y, r = design(16, 14, 0.0, 3.0, noise=0.1)
    cases.append(case("two_clusters_tiny_sample", d, y, None, [i % 2 for i in range(len(d))], None, 1.5, [1.2, 1.8], 0.5, None))
    print(json.dumps(cases, indent=1))


if __name__ == "__main__":
    main()
