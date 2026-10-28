# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for summary-data synthesis: GLS pooling, random effects, hierarchical
borrowing and the equicorrelation envelope.

Regenerate with:  uv run --script evidence_synthesis_reference.py > evidence_synthesis_reference.json

Written from the models at 50 digits, not from the product's algebra:
  fixed effect / GLS      mp.lu_solve on the explicit covariance:  mu = 1'S^-1 y / 1'S^-1 1,
                          Var = 1/1'S^-1 1, Q = r'S^-1 r
  DerSimonian-Laird       tau2 = max(0, (Q - (k-1)) / (W - sum w^2/W)), w = 1/v
  REML                    the first sign change upward from 0 of the derivative of the restricted
                          log-likelihood  l(t) = -1/2 sum ln(v_i + t) - 1/2 ln sum w - 1/2 sum w (y - mu_w)^2
                          taken by mp.diff of l itself (not from a score formula), refined by bisection
  hierarchical borrowing  p(y | tau) and the posterior of mu by direct numerical integration
                          over mu of  prod N(y_i; mu, v_i + tau^2) * prior(mu)  (mp.quad), then
                          E[theta_i | mu, y_i] = b y_i + (1 - b) mu, Var = b v_i, b = tau2/(tau2 + v_i),
                          mixed over mu and tau
  equicorrelation         S = D R(rho) D formed explicitly; extremes over [lo, hi] by a 400-point scan
                          refined with mp.findroot on mp.diff, endpoints included

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 50
mpf = mp.mpf


def txt(x, d=30):
    return mp.nstr(x, d)


def gls(y, cov):
    k = len(y)
    S = mp.matrix(cov)
    ones = mp.matrix([1] * k)
    x = mp.lu_solve(S, ones)
    denom = sum(x)
    mu = sum(x[i] * y[i] for i in range(k)) / denom
    r = mp.matrix([mpf(v) - mu for v in y])
    z = mp.lu_solve(S, r)
    q = sum(r[i] * z[i] for i in range(k))
    weights = [x[i] / denom for i in range(k)]
    return mu, 1 / denom, weights, q


def gls_cases(rnd):
    out = []
    specs = []
    specs.append(([0.2, 0.5], [[0.04, 0.01], [0.01, 0.09]]))
    specs.append(([1.0, 2.0, 4.0], [[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 3]]))
    for k, rho in ((3, 0.95), (4, 0.999), (5, -0.2), (6, 0.5)):
        sds = [0.5 + 0.3 * i for i in range(k)]
        cov = [[(1.0 if i == j else rho) * sds[i] * sds[j] for j in range(k)] for i in range(k)]
        specs.append(([round(rnd.gauss(0, 1), 3) for _ in range(k)], cov))
    # variances across sixteen orders of magnitude, large offset
    specs.append(([1e6 + 0.5, 1e6 - 0.25, 1e6 + 1.5], [[1e-8, 0, 0], [0, 1.0, 0], [0, 0, 1e8]]))
    for y, cov in specs:
        mu, var, w, q = gls([mpf(v) for v in y], [[mpf(c) for c in row] for row in cov])
        out.append({"estimates": y, "covariance": [c for row in cov for c in row], "estimate": txt(mu), "variance": txt(var),
                    "weights": [txt(x) for x in w], "q": txt(q)})
    return out


def dl(y, v):
    k = len(y)
    w = [1 / mpf(x) for x in v]
    W = sum(w)
    mu = sum(wi * mpf(yi) for wi, yi in zip(w, y)) / W
    q = sum(wi * (mpf(yi) - mu) ** 2 for wi, yi in zip(w, y))
    c = W - sum(x * x for x in w) / W
    tau2 = max(mpf(0), (q - (k - 1)) / c)
    return tau2, q


def re_at(y, v, tau2):
    p = [1 / (mpf(vi) + tau2) for vi in v]
    P = sum(p)
    mu = sum(pi * mpf(yi) for pi, yi in zip(p, y)) / P
    return mu, 1 / P, [pi / P for pi in p]


def restricted_loglik(y, v):
    def l(t):
        w = [1 / (mpf(vi) + t) for vi in v]
        W = sum(w)
        mu = sum(wi * mpf(yi) for wi, yi in zip(w, y)) / W
        return -sum(mp.log(mpf(vi) + t) for vi in v) / 2 - mp.log(W) / 2 - sum(wi * (mpf(yi) - mu) ** 2 for wi, yi in zip(w, y)) / 2
    return l


def reml(y, v):
    l = restricted_loglik(y, v)
    d = lambda t: mp.diff(l, t)
    if d(mpf(0)) <= 0:
        return mpf(0)
    scale = max(mpf(max(v)), max(mpf(a) for a in y) - min(mpf(a) for a in y)) ** 2 + 1
    grid = [scale * mpf(10) ** (mpf(i) / 4 - 8) for i in range(0, 65)]
    lo = mpf(0)
    for g in grid:
        if d(g) < 0:
            hi = g
            break
        lo = g
    else:
        raise SystemExit("no bracket")
    for _ in range(170):
        mid = (lo + hi) / 2
        if d(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, d(lo)


def random_effects_cases(rnd):
    out = []
    specs = [
        ([0.1, 0.3, 0.5], [0.01, 0.02, 0.015]),
        ([0.0, 1.0], [0.1, 0.1]),
        ([0.2, 0.25, 0.22, 0.21], [0.01, 0.01, 0.01, 0.01]),  # homogeneous: tau2 = 0
        ([-1.2, 0.4, 2.9, 0.1, 1.7], [0.3, 0.5, 0.2, 0.9, 0.4]),
        ([5.0, 5.2, 4.9, 5.1], [1e-6, 4e-6, 2.5e-5, 1e-6]),  # tiny variances
        ([0.0, 10.0, 20.0], [1e4, 1e-4, 1.0]),  # variances across eight orders
        ([round(rnd.gauss(0, 1), 2) for _ in range(12)], [round(0.05 + 0.4 * rnd.random(), 3) for _ in range(12)]),
        ([1.0, 3.0], [0.5, 2.0]),
    ]
    for y, v in specs:
        tau_dl, q = dl(y, v)
        mu_dl, var_dl, w_dl = re_at(y, v, tau_dl)
        res = reml(y, v)
        if isinstance(res, tuple):
            tau_r, dres = res
            assert abs(dres) < mpf(10) ** -30, dres
        else:
            tau_r = res
        mu_r, var_r, w_r = re_at(y, v, tau_r)
        df = len(y) - 1
        i2 = max(mpf(0), (q - df) / q) if q > 0 else mpf(0)
        out.append({"estimates": y, "variances": v, "q": txt(q), "i_squared": txt(i2),
                    "dl": {"tau2": txt(tau_dl), "estimate": txt(mu_dl), "variance": txt(var_dl), "weights": [txt(x) for x in w_dl]},
                    "reml": {"tau2": txt(tau_r), "estimate": txt(mu_r), "variance": txt(var_r), "weights": [txt(x) for x in w_r]}})
    return out


def hierarchical(y, v, prior, support, probs):
    k = len(y)
    y = [mpf(a) for a in y]
    v = [mpf(a) for a in v]
    lo, hi = min(y) - 12 * mp.sqrt(max(v) + max(support) ** 2 + 1), max(y) + 12 * mp.sqrt(max(v) + max(support) ** 2 + 1)
    if prior is not None:
        m0, v0 = mpf(prior[0]), mpf(prior[1])
        lo, hi = min(lo, m0 - 12 * mp.sqrt(v0)), max(hi, m0 + 12 * mp.sqrt(v0))

    def like(mu, t2):
        value = mpf(1)
        for yi, vi in zip(y, v):
            value *= mp.npdf(yi, mu, mp.sqrt(vi + t2))
        if prior is not None:
            value *= mp.npdf(mu, m0, mp.sqrt(v0))
        return value

    cuts = sorted({lo, hi} | {a for a in y} | {sum(y) / k})
    cond = []
    for tau in support:
        t2 = mpf(tau) ** 2
        z = mp.quad(lambda mu: like(mu, t2), cuts)
        m1 = mp.quad(lambda mu: like(mu, t2) * mu, cuts) / z
        m2 = mp.quad(lambda mu: like(mu, t2) * mu * mu, cuts) / z
        cond.append((z, m1, m2 - m1 * m1, t2))
    zs = [mpf(p) * c[0] for p, c in zip(probs, cond)]
    total = sum(zs)
    post = [z / total for z in zs]
    mean = sum(p * c[1] for p, c in zip(post, cond))
    second = sum(p * (c[2] + c[1] ** 2) for p, c in zip(post, cond))
    sm, ss = [mpf(0)] * k, [mpf(0)] * k
    for p, (z, m1, var, t2) in zip(post, cond):
        for i in range(k):
            b = t2 / (t2 + v[i])
            m = b * y[i] + (1 - b) * m1
            vv = b * v[i] + (1 - b) ** 2 * var
            sm[i] += p * m
            ss[i] += p * (vv + m * m)
    return {"mean": txt(mean), "variance": txt(second - mean**2), "study_means": [txt(x) for x in sm],
            "study_variances": [txt(ss[i] - sm[i] ** 2) for i in range(k)], "tau_posterior": [txt(p) for p in post]}


def hierarchical_cases():
    out = []
    specs = [
        ([0.1, 0.3, 0.5], [0.01, 0.02, 0.015], None, [0.0, 0.1, 0.3, 1.0], [0.1, 0.4, 0.3, 0.2]),
        ([0.1, 0.3, 0.5], [0.01, 0.02, 0.015], (0.0, 1.0), [0.05, 0.2, 0.5], [0.25, 0.5, 0.25]),
        ([2.0, -1.0], [0.5, 0.5], None, [0.0, 0.5, 2.0, 8.0], [0.25, 0.25, 0.25, 0.25]),
        ([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], (3.0, 0.25), [0.0, 1.0], [0.5, 0.5]),
        ([5.0, 5.5, 4.0, 9.0, 5.2], [0.2, 0.3, 0.2, 0.5, 0.25], None, [0.01, 0.1, 0.5, 1.0, 3.0, 10.0], [0.05, 0.15, 0.3, 0.3, 0.15, 0.05]),
        ([1.0, 2.0], [100.0, 0.01], (0.0, 100.0), [0.0, 0.001, 50.0], [0.5, 0.25, 0.25]),
    ]
    for y, v, prior, support, probs in specs:
        out.append({"estimates": y, "variances": v, "mean_prior": prior, "support": support, "probabilities": probs,
                    **hierarchical(y, v, prior, support, probs)})
    return out


def equicorr(y, s, rho):
    k = len(y)
    cov = [[(mpf(1) if i == j else mpf(rho)) * mpf(s[i]) * mpf(s[j]) for j in range(k)] for i in range(k)]
    mu, var, _, _ = gls([mpf(a) for a in y], cov)
    return mu, var


def envelope(y, s, lo, hi):
    k = len(y)
    lo, hi = mpf(lo), mpf(hi)
    n = 400
    xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
    # geometric refinement toward both ends: a turning point can sit within 1e-7 of an endpoint
    for j in range(1, 15):
        xs.append(lo + (hi - lo) * mpf(10) ** -j)
        xs.append(hi - (hi - lo) * mpf(10) ** -j)
    xs = sorted(set(xs))
    n = len(xs) - 1
    results = {}
    for name, idx in (("estimate", 0), ("variance", 1)):
        f = lambda r, idx=idx: equicorr(y, s, r)[idx]
        values = [f(x) for x in xs]
        cands = [(values[0], xs[0]), (values[-1], xs[-1])]
        d = lambda r, f=f: mp.diff(f, r)
        for i in range(1, n):
            a, b = values[i - 1], values[i]
            c = values[i + 1]
            if (b - a) * (c - b) < 0:  # interior turning point near xs[i]
                try:
                    r = mp.findroot(d, (xs[i - 1], xs[i + 1]), solver="anderson", tol=mpf(10) ** -35)
                    if lo < r < hi:
                        cands.append((f(r), r))
                except Exception:
                    pass
        results[name] = {"min": txt(min(cands)[0]), "rho_min": txt(min(cands)[1]), "max": txt(max(cands)[0]), "rho_max": txt(max(cands)[1])}
    return results


def dependence_cases():
    out = {"points": [], "envelopes": []}
    specs = [
        ([0.2, 0.5, 0.35], [0.1, 0.2, 0.15]),
        ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        ([0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]),
        ([10.0, 12.0, 11.0], [1e-3, 1.0, 1e3]),
        ([0.5, 0.6], [0.1, 0.1000001]),
        ([0.1, 0.4, 0.25, 0.3, 0.2], [0.05, 0.2, 0.1, 0.08, 0.5]),
    ]
    for y, s in specs:
        k = len(y)
        rho_floor = -1 / (k - 1)
        for rho in (rho_floor * 0.95, -0.1 if -0.1 > rho_floor else rho_floor * 0.5, 0.0, 0.3, 0.9, 0.999999):
            mu, var = equicorr(y, s, rho)
            out["points"].append({"estimates": y, "se": s, "rho": float(rho), "estimate": txt(mu), "variance": txt(var)})
        for lo, hi in ((max(rho_floor * 0.9, -0.5), 0.9), (0.0, 0.5), (0.2, 0.999999), (rho_floor * 0.99, -rho_floor * 0.5 if k > 2 else 0.0)):
            if lo >= hi:
                continue
            e = envelope(y, s, lo, hi)
            out["envelopes"].append({"estimates": y, "se": s, "lo": float(lo), "hi": float(hi), **e})
    return out


def main():
    rnd = random.Random(20261028)
    print(json.dumps({"gls": gls_cases(rnd), "random_effects": random_effects_cases(rnd), "hierarchical": hierarchical_cases(),
                      "dependence": dependence_cases()}, separators=(",", ":")))


main()
