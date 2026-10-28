# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Propensity-score stratified effect and its standard error, at 60 digits.

Regenerate with:  uv run --script stratification_reference.py > stratification_reference.json

Scores are the maximum-likelihood logistic fit (Newton's method), clipped into [c, 1-c].  Strata
are the empirical quantile classes of the clipped scores: the cut at level s/k is the order
statistic x_(ceil(s n / k)) (the type-1 empirical quantile), and a row belongs to the first class
whose cut is at or above its score, so equal scores share a class.  Within a class holding both
arms the contrast is the difference of the (mass-weighted) arm means; classes are pooled by the
mass of the target arms.  The standard error uses the arm-mean variances
sum w^2 (y - m)^2 n_a/(n_a-1)/(sum w)^2 pooled with squared pooling shares, plus the variation of
the pooling shares: both pieces of every row's influence are read off as derivatives of the pooled
contrast with respect to that row's mass, once through the arm means and once through the pooling
shares (central difference at step 1e-30), then combined as the product states.
Values are written as nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from math import ceil
from mpmath import mp, mpf, exp, matrix, lu_solve, sqrt

mp.dps = 60
_state = 0x1B873593CC9E2D51


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def fit_logit(t, X, mass):
    n, p = len(t), len(X[0])
    beta = [mpf(0)] * p
    for _ in range(200):
        score = [mpf(0)] * p
        info = matrix(p, p)
        for i in range(n):
            e = 1 / (1 + exp(-sum(beta[k] * X[i][k] for k in range(p))))
            for a in range(p):
                score[a] += mass[i] * (t[i] - e) * X[i][a]
                for b in range(p):
                    info[a, b] += mass[i] * e * (1 - e) * X[i][a] * X[i][b]
        step = lu_solve(info, matrix(score))
        beta = [beta[k] + step[k] for k in range(p)]
        if max(abs(s) for s in step) < mpf(10) ** -50:
            break
    return [1 / (1 + exp(-sum(beta[k] * X[i][k] for k in range(p)))) for i in range(len(t))]


def strata_of(scores, k):
    n = len(scores)
    order = sorted(scores)
    cuts = [order[min(max(ceil(s * n / k), 1) - 1, n - 1)] for s in range(1, k)]
    out = []
    for v in scores:
        c = next((j for j, cut in enumerate(cuts) if v <= cut), len(cuts))
        out.append(min(c, k - 1))
    return out


def pooled(t, y, stratum, k, arm_mass, share_mass, target):
    """Pooled contrast, its per-class arm variances (weighted by arm_mass) and shares."""
    classes = []
    for s in range(k):
        rows = [i for i in range(len(t)) if stratum[i] == s]
        treated = [i for i in rows if t[i] > 0.5]
        control = [i for i in rows if t[i] <= 0.5]
        if not treated or not control:
            continue
        mean = lambda idx: sum(arm_mass[i] * y[i] for i in idx) / sum(arm_mass[i] for i in idx)
        diff = mean(treated) - mean(control)
        if target == "att":
            share = sum(share_mass[i] for i in treated)
        elif target == "atc":
            share = sum(share_mass[i] for i in control)
        else:
            share = sum(share_mass[i] for i in rows)
        classes.append((s, treated, control, diff, share, mean(treated), mean(control)))
    total = sum(c[4] for c in classes)
    return sum(c[3] * c[4] for c in classes) / total, classes, total


def case(name, t, y, x, clip, trim, k, weights=None):
    n0 = len(t)
    X = [[mpf(1)] + [mpf(c) for c in row] for row in x]
    sw = [mpf(w) for w in weights] if weights else [mpf(1)] * n0
    raw = fit_logit(t, X, sw)
    keep = [i for i in range(n0) if trim is None or (trim <= raw[i] <= 1 - mpf(trim))]
    score = [min(max(raw[i], mpf(clip)), 1 - mpf(clip)) if clip else raw[i] for i in keep]
    tt = [t[i] for i in keep]
    yy = [mpf(y[i]) for i in keep]
    mass = [sw[i] for i in keep]
    n = len(keep)
    stratum = strata_of(score, k)
    out = {"name": name, "t": t, "y": y, "x": x, "clip": clip, "trim": trim, "strata": k,
           "weights": weights, "targets": {}}
    sizes = [stratum.count(s) for s in range(k)]
    out["stratum_sizes"] = sizes
    h = mpf(10) ** -30
    for target in ("ate", "att", "atc"):
        try:
            tau, classes, total = pooled(tt, yy, stratum, k, mass, mass, target)
        except ZeroDivisionError:
            out["targets"][target] = None
            continue
        if any(len(c[1]) < 2 or len(c[2]) < 2 for c in classes):
            out["targets"][target] = None
            continue
        weighted = weights is not None
        var_pool = mpf(0)
        for (s, tr, co, diff, share, m1, m0) in classes:
            v = mpf(0)
            for idx, m in ((tr, m1), (co, m0)):
                na = mpf(len(idx))
                if weighted:
                    ssq = sum((mass[i] * (yy[i] - m)) ** 2 for i in idx)
                    v += ssq * na / (na - 1) / sum(mass[i] for i in idx) ** 2
                else:
                    v += sum((yy[i] - m) ** 2 for i in idx) / (na * (na - 1))
            var_pool += v * (share / total) ** 2
        share_var = mpf(0)
        for i in range(n):
            ma = list(mass)
            ma[i] = mass[i] * (1 + h)
            hi = pooled(tt, yy, stratum, k, ma, mass, target)[0]
            ma[i] = mass[i] * (1 - h)
            lo = pooled(tt, yy, stratum, k, ma, mass, target)[0]
            res = n * (hi - lo) / (2 * mass[i] * h)
            ms = list(mass)
            ms[i] = mass[i] * (1 + h)
            hi = pooled(tt, yy, stratum, k, mass, ms, target)[0]
            ms[i] = mass[i] * (1 - h)
            lo = pooled(tt, yy, stratum, k, mass, ms, target)[0]
            sh = n * (hi - lo) / (2 * mass[i] * h) * mass[i]
            res *= mass[i]
            share_var += sh * sh + 2 * res * sh
        nf = mpf(n)
        se = sqrt(var_pool + share_var / (nf * (nf - 1)))
        out["targets"][target] = {"ate": float(tau), "se": float(se),
                                  "rows": sum(len(c[1]) + len(c[2]) for c in classes)}
    return out


def draw(n, slope, two=False):
    t, y, x = [], [], []
    for i in range(n):
        a = u01() * 4 - 2
        b = u01() * 2 - 1
        pr = 1 / (1 + 2.718281828459045 ** (-(slope * a + 0.2)))
        ti = 1.0 if u01() < pr else 0.0
        yi = 1.0 + 1.5 * ti + 0.9 * a + 0.4 * b + (u01() - 0.5)
        t.append(ti)
        y.append(float(yi))
        x.append([a, b] if two else [a])
    return t, y, x


cases = []
t, y, x = draw(100, 0.6)
cases.append(case("five_strata_divisible", t, y, x, None, None, 5))
t, y, x = draw(67, 0.9)
cases.append(case("five_strata_remainder", t, y, x, None, None, 5))
cases.append(case("two_strata", t, y, x, None, None, 2))
cases.append(case("three_strata_two_covariates", *draw(60, 0.6, two=True), None, None, 3))
t, y, x = draw(150, 1.8)
cases.append(case("clip_ties_in_boundary_strata", t, y, x, 0.1, None, 5))
cases.append(case("trim", t, y, x, None, 0.1, 5))
t, y, x = draw(70, 0.9)
cases.append(case("outcome_offset_1e8", t, [v + 1.0e8 for v in y], x, None, None, 5))
w = [0.5 + 3.5 * u01() for _ in range(70)]
cases.append(case("sampling_weights", t, y, x, None, None, 5, w))
print(json.dumps({"cases": cases}))
