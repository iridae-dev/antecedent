# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the overlap diagnostics and the Riesz-representer sensitivity.

Regenerate with:  uv run --script diagnostic_propensity_reference.py > diagnostic_propensity_reference.json

Written from the stated definitions at 60 digits.  A logistic propensity of the binary treatment
on an intercept and two covariates is fitted by Newton's method on the likelihood (to 1e-50);
then
  overlap.assessment   min/max propensity; weights 1/p (treated) and 1/(1-p) (control) with p
                       clamped to [1e-9, 1-1e-9]; ess = (sum w)^2 / sum w^2; comparison = 1 - ess/n;
                       pass iff min >= eps, max <= 1 - eps and ess/n >= min_ess_fraction
  overlap.rule         retained = fraction of rows with eps <= p <= 1 - eps; comparison = 1 -
                       retained; pass iff retained >= min_retained
  sensitivity.riesz    p clamped to [clip, 1-clip]; alpha = 1/p (treated), -1/(1-p) (control);
                       ||alpha||_2 = sqrt(mean alpha^2); sd(y) the sample SD; for a published
                       estimate c: the smallest grid delta with delta * sd * ||alpha||_2 >= |c|,
                       and the bound c - sign(c) delta sd ||alpha||_2
  continuous support   treatment on intercept and covariates by least squares, residuals sorted,
                       lower/upper the order statistics at the first index >= p (n - 1) for p = eps
                       and 1 - eps; supported = rows whose two doses minus the fitted mean both
                       lie in [lower, upper].

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 60
mpf = mp.mpf

rng = random.Random(7_654_321)
N = 160
z1 = [rng.gauss(0, 1) for _ in range(N)]
z2 = [rng.gauss(0.3, 1.2) for _ in range(N)]


def sigmoid(x):
    return 1 / (1 + mp.exp(-x))


t = []
for a, b in zip(z1, z2):
    p = float(sigmoid(mpf(-0.2) + mpf(0.9) * a - mpf(0.5) * b))
    t.append(1.0 if rng.random() < p else 0.0)
y = [1.0 + 2.0 * ti + 0.7 * a - 0.3 * b + rng.gauss(0, 1.1) for ti, a, b in zip(t, z1, z2)]
tc = [0.5 + 1.2 * a - 0.4 * b + (0.4 + 0.5 * abs(a)) * rng.gauss(0, 1) for a, b in zip(z1, z2)]


def fit_logistic(X, yv):
    p = len(X[0])
    beta = mp.matrix(p, 1)
    for _ in range(60):
        g = mp.matrix(p, 1)
        H = mp.matrix(p, p)
        for row, yi in zip(X, yv):
            eta = sum(beta[j] * row[j] for j in range(p))
            mu = sigmoid(eta)
            for j in range(p):
                g[j] += (mpf(yi) - mu) * row[j]
                for k in range(p):
                    H[j, k] += mu * (1 - mu) * row[j] * row[k]
        step = mp.lu_solve(H, g)
        beta += step
        if mp.norm(step) < mpf(10) ** -50:
            break
    return beta


X = [[mpf(1), mpf(a), mpf(b)] for a, b in zip(z1, z2)]
beta = fit_logistic(X, t)
prop = [sigmoid(sum(beta[j] * row[j] for j in range(3))) for row in X]


def overlap(eps, min_ess):
    w = []
    for ti, p in zip(t, prop):
        q = min(max(p, mpf(10) ** -9), 1 - mpf(10) ** -9)
        w.append(1 / q if ti > 0.5 else 1 / (1 - q))
    ess = sum(w) ** 2 / sum(x * x for x in w)
    frac = ess / N
    lo, hi = min(prop), max(prop)
    passed = lo >= mpf(eps) and hi <= 1 - mpf(eps) and frac >= mpf(min_ess)
    return {"eps": eps, "min_ess": min_ess, "comparison": float(1 - frac), "passed": bool(passed)}


def rule(eps, min_ret):
    retained = sum(1 for p in prop if mpf(eps) <= p <= 1 - mpf(eps)) / mpf(N)
    return {"eps": eps, "min_retained": min_ret, "comparison": float(1 - retained), "passed": bool(retained >= mpf(min_ret))}


def sample_sd(v):
    m = sum(mpf(x) for x in v) / len(v)
    return mp.sqrt(sum((mpf(x) - m) ** 2 for x in v) / (len(v) - 1))


def riesz(clip, published, grid, threshold):
    lo = mpf(clip)
    alpha = []
    ipw = mpf(0)
    for ti, yi, p in zip(t, y, prop):
        q = min(max(p, lo), 1 - lo)
        a = 1 / q if ti > 0.5 else -1 / (1 - q)
        alpha.append(a)
        ipw += a * mpf(yi)
    ipw /= N
    l2 = mp.sqrt(sum(a * a for a in alpha) / N)
    unit = sample_sd(y) * l2
    out = {"clip": clip, "published": published, "ipw": float(ipw), "unit": float(unit)}
    if ipw != 0 and published != 0 and (ipw > 0) != (published > 0):
        out["refused"] = True
        return out
    tip = next((d for d in grid if mpf(d) * unit >= abs(mpf(published))), None)
    out["refused"] = False
    out["threshold"] = threshold
    cleared = [g for g in grid if tip is None or g < tip]
    # pass iff the largest probed delta that left the effect standing reaches the bar
    out["passed"] = bool(cleared and max(cleared) >= threshold)
    if tip is None:
        out["comparison"] = None
        out["refuted"] = float(mpf(published) - mp.sign(mpf(published)) * mpf(grid[-1]) * unit)
    else:
        out["comparison"] = tip
        out["refuted"] = float(mpf(published) - mp.sign(mpf(published)) * mpf(tip) * unit)
    return out


GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def continuous(eps, min_frac, control, active):
    A = mp.matrix(3, 3)
    b = mp.matrix(3, 1)
    for a, c, d in zip(z1, z2, tc):
        row = [mpf(1), mpf(a), mpf(c)]
        for j in range(3):
            b[j] += row[j] * mpf(d)
            for k in range(3):
                A[j, k] += row[j] * row[k]
    coef = mp.lu_solve(A, b)
    means = [coef[0] + coef[1] * mpf(a) + coef[2] * mpf(c) for a, c in zip(z1, z2)]
    res = sorted(mpf(d) - m for d, m in zip(tc, means))

    def quant(p):
        for i in range(N):
            if i >= p * (N - 1):
                return res[i]
        return res[-1]

    p_lo = min(max(mpf(eps), 0), mpf(1) / 2)
    lower, upper = quant(p_lo), quant(1 - p_lo)
    sup = sum(1 for m in means if all(lower <= mpf(d) - m <= upper for d in (active, control)))
    frac = mpf(sup) / N
    return {"eps": eps, "min_frac": min_frac, "control": control, "active": active,
            "comparison": float(1 - frac), "passed": bool(frac >= mpf(min_frac)), "supported": sup}


out = {
    "n": N,
    "z1": z1, "z2": z2, "t": t, "y": y, "tc": tc,
    "propensity": [float(p) for p in prop],
    "overlap": [overlap(e, m) for e in (0.05, 0.12, 0.2) for m in (0.5, 0.8)],
    "rule": [rule(e, m) for e in (0.1, 0.25, 0.3) for m in (0.5, 0.8)],
    "riesz": [riesz(c, pub, GRID, thr) for c in (0.01, 0.15) for pub in (2.0, 0.5, 0.05, 4.0, 40.0, -0.4) for thr in (0.1, 0.25)],
    "continuous": [continuous(e, m, 0.0, 0.8) for e in (0.02, 0.1) for m in (0.5, 0.9)],
}
print(json.dumps(out, indent=1))
