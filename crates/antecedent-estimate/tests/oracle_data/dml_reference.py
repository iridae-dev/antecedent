# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Cross-fitted AIPW and partially linear DML with fixed folds and exact least-squares nuisances.

Regenerate with:  uv run --script dml_reference.py > dml_reference.json

The fold of every row is part of the case, so the nuisances are fully determined.  For each fold
the outcome regressions mu_0 and mu_1 are ordinary least squares of the outcome on [1, x] over the
training rows of each arm (for the partially linear score, one regression m over all training
rows), and the propensity is the unpenalised logistic maximum likelihood over the training rows
(Newton's method); all are solved at 60 digits and predicted on the held-out fold.

AIPW: phi_i = mu_1 - mu_0 + T (y - mu_1)/e - (1 - T)(y - mu_0)/(1 - e), with e clipped into
[c, 1-c]; rows whose raw out-of-fold e lies outside [trim, 1-trim] are dropped; the estimate is the
mean of phi over the kept rows.  Partially linear: theta = sum dt dy / sum dt^2 over the kept rows
with dt = T - e (raw e, not clipped) and dy = y - m.
Standard error: both estimators are plug-in functionals of the case masses with the nuisances held
at their out-of-fold values, mass m_i of a kept row entering the numerator and denominator; the
derivative d_i of the estimate with respect to m_i is written out, and SE^2 = n/(n-1) sum d_i^2
over all n rows (dropped rows have d_i = 0).  Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from mpmath import mp, mpf, exp, matrix, lu_solve, sqrt

mp.dps = 60
_state = 0xBB67AE8584CAA73B


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def ols(rows, y):
    p = len(rows[0])
    A, b = matrix(p, p), matrix(p, 1)
    for r, v in zip(rows, y):
        for a in range(p):
            b[a] += r[a] * v
            for c in range(p):
                A[a, c] += r[a] * r[c]
    beta = lu_solve(A, b)
    return lambda r: sum(beta[k] * r[k] for k in range(p))


def logit(rows, t):
    p = len(rows[0])
    beta = [mpf(0)] * p
    for _ in range(200):
        score = [mpf(0)] * p
        info = matrix(p, p)
        for r, ti in zip(rows, t):
            e = 1 / (1 + exp(-sum(beta[k] * r[k] for k in range(p))))
            for a in range(p):
                score[a] += (ti - e) * r[a]
                for c in range(p):
                    info[a, c] += e * (1 - e) * r[a] * r[c]
        step = lu_solve(info, matrix(score))
        beta = [beta[k] + step[k] for k in range(p)]
        if max(abs(s) for s in step) < mpf(10) ** -50:
            break
    return lambda r: 1 / (1 + exp(-sum(beta[k] * r[k] for k in range(p))))


def crossfit(X, y, t, folds, k):
    n = len(y)
    mu0, mu1, m, e = [None] * n, [None] * n, [None] * n, [None] * n
    for f in range(k):
        train = [i for i in range(n) if folds[i] != f]
        valid = [i for i in range(n) if folds[i] == f]
        t0 = [i for i in train if t[i] < 0.5]
        t1 = [i for i in train if t[i] > 0.5]
        f0 = ols([X[i] for i in t0], [y[i] for i in t0])
        f1 = ols([X[i] for i in t1], [y[i] for i in t1])
        fm = ols([X[i] for i in train], [y[i] for i in train])
        fe = logit([X[i] for i in train], [mpf(t[i]) for i in train])
        for i in valid:
            mu0[i], mu1[i], m[i], e[i] = f0(X[i]), f1(X[i]), fm(X[i]), fe(X[i])
    return mu0, mu1, m, e


def case(name, t, y, x, k, clip, trim, folds=None):
    n = len(t)
    X = [[mpf(1)] + [mpf(c) for c in row] for row in x]
    yy = [mpf(v) for v in y]
    if folds is None:
        folds = [i % k for i in range(n)]
        for i in range(n - 1, 0, -1):
            j = int(u01() * (i + 1))
            folds[i], folds[j] = folds[j], folds[i]
    mu0, mu1, m, e = crossfit(X, yy, t, folds, k)
    keep = [i for i in range(n) if trim is None or (trim <= e[i] <= 1 - mpf(trim))]
    nf, kept = mpf(n), mpf(len(keep))
    # AIPW
    ec = [min(max(v, mpf(clip)), 1 - mpf(clip)) if clip else v for v in e]
    phi = [mu1[i] - mu0[i] + t[i] * (yy[i] - mu1[i]) / ec[i]
           - (1 - t[i]) * (yy[i] - mu0[i]) / (1 - ec[i]) for i in range(n)]
    ate = sum(phi[i] for i in keep) / kept
    d = [(phi[i] - ate) / kept if i in keep else mpf(0) for i in range(n)]
    se = sqrt(nf / (nf - 1) * sum(v * v for v in d))
    # partially linear
    dt = [t[i] - e[i] for i in range(n)]
    dy = [yy[i] - m[i] for i in range(n)]
    den = sum(dt[i] ** 2 for i in keep)
    theta = sum(dt[i] * dy[i] for i in keep) / den
    d = [(dt[i] * dy[i] - theta * dt[i] ** 2) / den if i in keep else mpf(0) for i in range(n)]
    se_plr = sqrt(nf / (nf - 1) * sum(v * v for v in d))
    return {"name": name, "t": t, "y": y, "x": x, "folds": folds, "k": k, "clip": clip,
            "trim": trim, "kept": len(keep), "aipw": {"ate": float(ate), "se": float(se)},
            "plr": {"ate": float(theta), "se": float(se_plr)},
            "min_e": float(min(e)), "max_e": float(max(e))}


def draw(n, p, slope=0.8):
    t, y, x = [], [], []
    for _ in range(n):
        row = [u01() * 4 - 2 for _ in range(p)]
        eta = slope * row[0] - 0.3 * (row[1] if p > 1 else 0.0)
        ti = 1.0 if u01() < 1 / (1 + 2.718281828459045 ** (-eta)) else 0.0
        yi = 1.0 + 1.5 * ti + 0.9 * row[0] + 0.4 * (row[1] if p > 1 else 0.0) + (u01() - 0.5)
        t.append(ti)
        y.append(float(yi))
        x.append(row)
    return t, y, x


cases = []
t, y, x = draw(60, 2)
cases.append(case("five_folds_two_covariates", t, y, x, 5, None, None))
cases.append(case("two_folds", t, y, x, 2, None, None))
cases.append(case("trim", t, y, x, 5, None, 0.2))
t2, y2, x2 = draw(80, 2, slope=2.2)
cases.append(case("clip_binds", t2, y2, x2, 5, 0.15, None))
cases.append(case("clip_and_trim", t2, y2, x2, 5, 0.15, 0.1))
cases.append(case("outcome_offset_1e7", t, [v + 1.0e7 for v in y], x, 5, None, None))
cases.append(case("covariate_offset_1e3", t, y, [[1.0e3 + 250.0 * r[0], r[1]] for r in x], 5, None, None))
t3, y3, x3 = draw(50, 2)
x3 = [[r[0], r[0] + 1.0e-4 * r[1]] for r in x3]
cases.append(case("near_collinear_covariates", t3, y3, x3, 5, None, None))
t4, y4, x4 = draw(24, 1, slope=0.5)
cases.append(case("small_sample_three_folds", t4, y4, x4, 3, None, None))
print(json.dumps({"cases": cases}))
