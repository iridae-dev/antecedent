# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Hajek inverse-probability-weighted effect and its standard error, at 60 digits.

Regenerate with:  uv run --script ipw_weighting_reference.py > ipw_weighting_reference.json

The logistic propensity is the maximum-likelihood fit found by Newton's method on the
log-likelihood, the weights are the target's inverse-probability weights with the clip applied to
the weight (not to the fit) and trimming applied to the raw score, and the effect is the
difference of the two weighted arm means.  The standard error is not the product's influence
algebra: it is the derivative of the whole refitted estimator with respect to a case weight of
each row (central difference at step 1e-30), squared, summed, and scaled by n/(n-1) and
n/(n-p) as the product states for p design columns.  Because the refit changes the propensity,
the derivative contains the propensity-estimation term automatically.  Values are written as
nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from mpmath import mp, mpf, exp, matrix, lu_solve, sqrt

mp.dps = 60

_state = 0x2545F4914F6CDD1D


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def sigmoid(z):
    return 1 / (1 + exp(-z))


def fit_logit(t, X, mass):
    n, p = len(t), len(X[0])
    beta = [mpf(0)] * p
    for _ in range(200):
        score = [mpf(0)] * p
        info = matrix(p, p)
        for i in range(n):
            eta = sum(beta[k] * X[i][k] for k in range(p))
            e = sigmoid(eta)
            for a in range(p):
                score[a] += mass[i] * (t[i] - e) * X[i][a]
                for b in range(p):
                    info[a, b] += mass[i] * e * (1 - e) * X[i][a] * X[i][b]
        step = lu_solve(info, matrix(score))
        beta = [beta[k] + step[k] for k in range(p)]
        if max(abs(s) for s in step) < mpf(10) ** -50:
            break
    return [sigmoid(sum(beta[k] * X[i][k] for k in range(p))) for i in range(n)]


def weight(target, ti, e):
    if target == "ate":
        return 1 / e if ti > 0.5 else 1 / (1 - e)
    if target == "att":
        return mpf(1) if ti > 0.5 else e / (1 - e)
    return (1 - e) / e if ti > 0.5 else mpf(1)


def effect(t, y, X, mass, target, clip, trim):
    e = fit_logit(t, X, mass)
    num = [mpf(0), mpf(0)]
    den = [mpf(0), mpf(0)]
    for i in range(len(t)):
        if trim is not None and (e[i] < trim or e[i] > 1 - trim):
            continue
        s = e[i]
        if clip is not None:
            s = min(max(s, mpf(clip)), 1 - mpf(clip))
        w = mass[i] * weight(target, t[i], s)
        arm = 1 if t[i] > 0.5 else 0
        num[arm] += w * mpf(y[i])
        den[arm] += w
    return num[1] / den[1] - num[0] / den[0], e


def case(name, t, y, x, clip, trim):
    n = len(t)
    X = [[mpf(1)] + [mpf(c) for c in row] for row in x]
    p = len(X[0])
    out = {"name": name, "t": t, "y": y, "x": x, "clip": clip, "trim": trim, "targets": {}}
    h = mpf(10) ** -30
    for target in ("ate", "att", "atc"):
        mass = [mpf(1)] * n
        est, e = effect(t, y, X, mass, target, clip, trim)
        sq = mpf(0)
        for i in range(n):
            mass[i] = 1 + h
            hi = effect(t, y, X, mass, target, clip, trim)[0]
            mass[i] = 1 - h
            lo = effect(t, y, X, mass, target, clip, trim)[0]
            mass[i] = mpf(1)
            sq += ((hi - lo) / (2 * h)) ** 2
        nf = mpf(n)
        se = sqrt(sq * nf / (nf - 1) * nf / max(nf - p, 1))
        kept = sum(1 for v in e if not (trim is not None and (v < trim or v > 1 - trim)))
        out["targets"][target] = {"ate": float(est), "se": float(se)}
    out["max_score"] = float(max(e))
    out["min_score"] = float(min(e))
    return out


def draw(n, slope, offset=0.0, scale=1.0, two=False, collinear=0.0):
    t, y, x = [], [], []
    for i in range(n):
        a = u01() * 4 - 2
        b = u01() * 2 - 1
        pr = 1 / (1 + 2.718281828459045 ** (-(slope * a + 0.3)))
        ti = 1.0 if u01() < pr else 0.0
        if i % 7 == 0:
            ti = 1.0 - ti if i % 14 == 0 else ti
        yi = 1.0 + 2.0 * ti + 0.8 * a + 0.5 * b + (u01() - 0.5)
        t.append(ti)
        y.append(float(yi))
        row = [offset + scale * a]
        if two:
            row.append(offset + scale * (a + collinear * b))
        x.append(row)
    return t, y, x


cases = []
t, y, x = draw(36, 0.9)
cases.append(case("moderate_overlap", t, y, x, None, None))
t, y, x = draw(40, 2.2)
cases.append(case("strong_selection_clip", t, y, x, 0.1, None))
cases.append(case("strong_selection_trim", t, y, x, None, 0.1))
t, y, x = draw(30, 0.8, two=True, collinear=0.05)
cases.append(case("near_collinear_covariates", t, y, x, None, None))
t, y, x = draw(32, 0.9)
cases.append(case("outcome_offset_1e7", t, [v + 1.0e7 for v in y], x, None, None))
cases.append(case("covariate_offset_and_scale", t, y, [[1.0e3 + 250.0 * r[0]] for r in x], None, None))
t, y, x = draw(60, 3.2)
cases.append(case("extreme_weights_no_clip", t, y, x, None, None))
t, y, x = draw(9, 0.7)
cases.append(case("tiny_sample", t, y, x, None, None))
print(json.dumps({"cases": cases}))
