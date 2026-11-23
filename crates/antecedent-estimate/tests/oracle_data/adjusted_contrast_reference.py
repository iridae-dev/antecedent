# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""50-digit reference for the retained adjusted-model contrast (effect and delta-method SE).

Regenerate with:  uv run --script adjusted_contrast_reference.py > adjusted_contrast_reference.json

For a retained regression with design X = [1, adjustment, treatments], coefficients b and
covariance S, a contrast of an active against a control treatment setting is the target-law
mean of the response-scale difference
    effect = sum_r w_r (m(x_r' b | t = active) - m(x_r' b | t = control)),   sum_r w_r = 1,
with gradient g_c = sum_r w_r (m'(eta_a,r) x_a,rc - m'(eta_c,r) x_c,rc) and
SE = sqrt(g' S g).  The coefficients and covariance are refitted here from the data, not read from
the product: OLS with sigma^2 = rss / (n - p) and S = sigma^2 (X'X)^-1; binomial-logit and
Poisson-log by Newton's method on the likelihood to 40 digits with S = (X'WX)^-1, W the Fisher
weight (the logistic mean derivative mu (1 - mu), the Poisson mean exp(eta)).  For the identity
link the effect is delta'b_T and g = delta on the treatment columns (the intercept and adjustment
block cancel).  Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 40


def solve(a, b):
    return mp.lu_solve(mp.matrix(a), mp.matrix(b))


def design(zs, ts):
    return [[1.0] + [z[r] for z in zs] + [t[r] for t in ts] for r in range(len(ts[0]))]


def fit(x, y, family):
    n, p = len(x), len(x[0])
    X = mp.matrix(x)
    Y = mp.matrix(y)
    if family == "gaussian":
        xtx = X.T * X
        b = mp.lu_solve(xtx, X.T * Y)
        resid = Y - X * b
        s2 = sum(r * r for r in resid) / (n - p)
        return b, s2 * mp.inverse(xtx)
    b = mp.zeros(p, 1)
    for _ in range(200):
        eta = X * b
        if family == "logit":
            mu = [1 / (1 + mp.exp(-e)) for e in eta]
            w = [m * (1 - m) for m in mu]
        else:
            mu = [mp.exp(e) for e in eta]
            w = mu[:]
        grad = X.T * mp.matrix([Y[i] - mu[i] for i in range(n)])
        info = mp.zeros(p, p)
        for i in range(n):
            for j in range(p):
                for k in range(p):
                    info[j, k] += w[i] * X[i, j] * X[i, k]
        step = mp.lu_solve(info, grad)
        b += step
        if max(abs(s) for s in step) < mp.mpf(10) ** -35:
            break
    eta = X * b
    w = [(lambda m: m * (1 - m))(1 / (1 + mp.exp(-e))) if family == "logit" else mp.exp(e) for e in eta]
    info = mp.zeros(p, p)
    for i in range(n):
        for j in range(p):
            for k in range(p):
                info[j, k] += w[i] * X[i, j] * X[i, k]
    return b, mp.inverse(info)


def mean(family, eta):
    if family == "logit":
        return 1 / (1 + mp.exp(-eta))
    if family == "poisson":
        return mp.exp(eta)
    return eta


def dmean(family, eta):
    if family == "logit":
        m = 1 / (1 + mp.exp(-eta))
        return m * (1 - m)
    if family == "poisson":
        return mp.exp(eta)
    return mp.mpf(1)


def contrast(x, b, s, family, k_t, active, control, weights):
    n, p = len(x), len(x[0])
    first = p - k_t
    if family == "gaussian":
        g = [mp.mpf(0)] * p
        for i in range(k_t):
            g[first + i] = mp.mpf(active[i]) - mp.mpf(control[i])
        effect = sum(g[first + i] * b[first + i] for i in range(k_t))
    else:
        mass = sum(mp.mpf(w) for w in weights) if weights else mp.mpf(n)
        effect = mp.mpf(0)
        g = [mp.mpf(0)] * p
        for r in range(n):
            w = (mp.mpf(weights[r]) if weights else mp.mpf(1)) / mass
            base = sum(mp.mpf(x[r][c]) * b[c] for c in range(first))
            ea = base + sum(mp.mpf(active[i]) * b[first + i] for i in range(k_t))
            ec = base + sum(mp.mpf(control[i]) * b[first + i] for i in range(k_t))
            effect += w * (mean(family, ea) - mean(family, ec))
            da, dc = dmean(family, ea), dmean(family, ec)
            for c in range(p):
                if c < first:
                    xa = xc = mp.mpf(x[r][c])
                else:
                    xa, xc = mp.mpf(active[c - first]), mp.mpf(control[c - first])
                g[c] += w * (da * xa - dc * xc)
    var = sum(g[i] * s[i, j] * g[j] for i in range(p) for j in range(p))
    return effect, mp.sqrt(var)


def main():
    rng = random.Random(20261028)
    cases = []

    def uniform(n, lo, hi):
        return [round(lo + (hi - lo) * rng.random(), 3) for _ in range(n)]

    for name, family, n, k_t, n_adj, weighted in [
        ("ols_vector_two_treatments", "gaussian", 40, 2, 1, False),
        ("ols_scalar_two_adjusters", "gaussian", 25, 1, 2, True),
        ("logit_weighted", "logit", 120, 1, 1, True),
        ("logit_unweighted_vector", "logit", 150, 2, 1, False),
        ("poisson_weighted", "poisson", 90, 1, 1, True),
    ]:
        zs = [uniform(n, -1.5, 1.5) for _ in range(n_adj)]
        ts = [[float(rng.randint(0, 1)) if family != "poisson" else round(rng.random() * 2, 3) for _ in range(n)]
              for _ in range(k_t)]
        x = design(zs, ts)
        beta_true = [rng.choice([-0.5, 0.3, 0.6, -0.4, 0.5]) for _ in range(len(x[0]))]
        y = []
        for row in x:
            eta = sum(b * v for b, v in zip(beta_true, row))
            if family == "gaussian":
                y.append(round(eta + rng.gauss(0, 0.7), 4))
            elif family == "logit":
                y.append(1.0 if rng.random() < 1 / (1 + float(mp.exp(-eta))) else 0.0)
            else:
                lam = float(mp.exp(0.3 * eta))
                k, p_, limit = 0, 1.0, float(mp.exp(-lam))
                while True:
                    p_ *= rng.random()
                    if p_ <= limit:
                        break
                    k += 1
                y.append(float(k))
        b, s = fit(x, y, family)
        active = [1.0] * k_t if family != "poisson" else [1.5]
        control = [0.0] * k_t if family != "poisson" else [0.5]
        weights = [round(0.2 + rng.random(), 3) for _ in range(n)] if weighted else None
        effect, se = contrast(x, b, s, family, k_t, active, control, weights)
        cases.append({
            "name": name, "family": family, "outcome": y, "adjustment": zs, "treatments": ts,
            "active": active, "control": control, "weights": weights,
            "effect": float(effect), "se": float(se),
            "coefficients": [float(v) for v in b],
        })
    print(json.dumps(cases, indent=1))


main()
