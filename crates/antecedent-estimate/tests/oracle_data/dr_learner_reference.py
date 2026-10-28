# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""DR-Learner: cross-fitted AIPW pseudo-outcome, linear final stage on the retained rows, HC0.

Regenerate with:  uv run --script dr_learner_reference.py > dr_learner_reference.json

The fold of every row is part of the case.  The nuisances are the same exact ones as the
cross-fitted AIPW reference (per-arm OLS outcome regressions and unpenalised logistic propensity
on each fold's training rows, 60 digits).  The pseudo-outcome is the AIPW score with the
propensity clipped into [c, 1-c]; rows whose raw out-of-fold propensity lies outside
[trim, 1-trim] are not retained.  The reported ATE is the mean of the pseudo-outcome over the
retained rows with the influence standard error n/(n-1) sum d_i^2.  The final stage is ordinary
least squares of the pseudo-outcome on [1, x] over the retained rows; the CATE is the fitted
value at every row, and its standard error is the HC0 sandwich sqrt(z' (X'X)^-1
(sum_i e_i^2 x_i x_i') (X'X)^-1 z) with z = [1, x] of the row and e the retained-row residuals.
Values are written as nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from mpmath import mp, mpf, exp, matrix, lu_solve, sqrt

mp.dps = 60
_state = 0x6A09E667F3BCC908



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


def case(name, t, y, x, k, clip, trim):
    n = len(t)
    X = [[mpf(1)] + [mpf(c) for c in row] for row in x]
    p = len(X[0])
    yy = [mpf(v) for v in y]
    folds = [i % k for i in range(n)]
    for i in range(n - 1, 0, -1):
        j = int(u01() * (i + 1))
        folds[i], folds[j] = folds[j], folds[i]
    mu0, mu1, m, e = crossfit(X, yy, t, folds, k)
    keep = [i for i in range(n) if trim is None or (trim <= e[i] <= 1 - mpf(trim))]
    nf, kept = mpf(n), mpf(len(keep))
    ec = [min(max(v, mpf(clip)), 1 - mpf(clip)) if clip else v for v in e]
    phi = [mu1[i] - mu0[i] + t[i] * (yy[i] - mu1[i]) / ec[i]
           - (1 - t[i]) * (yy[i] - mu0[i]) / (1 - ec[i]) for i in range(n)]
    ate = sum(phi[i] for i in keep) / kept
    d = [(phi[i] - ate) / kept if i in keep else mpf(0) for i in range(n)]
    se = sqrt(nf / (nf - 1) * sum(v * v for v in d))
    A = matrix(p, p)
    b = matrix(p, 1)
    for i in keep:
        for a in range(p):
            b[a] += X[i][a] * phi[i]
            for c in range(p):
                A[a, c] += X[i][a] * X[i][c]
    beta = lu_solve(A, b)
    cate = [sum(beta[a] * X[i][a] for a in range(p)) for i in range(n)]
    meat = matrix(p, p)
    for i in keep:
        r = phi[i] - cate[i]
        for a in range(p):
            for c in range(p):
                meat[a, c] += r * r * X[i][a] * X[i][c]
    Ainv = A ** -1
    cov = Ainv * meat * Ainv
    cate_se = []
    for i in range(n):
        v = sum(X[i][a] * cov[a, c] * X[i][c] for a in range(p) for c in range(p))
        cate_se.append(float(sqrt(v)))
    return {"name": name, "t": t, "y": y, "x": x, "folds": folds, "k": k, "clip": clip,
            "trim": trim, "kept": len(keep), "ate": float(ate), "se": float(se),
            "cate": [float(v) for v in cate], "cate_se": cate_se}


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
# A three-valued covariate gives every profile exact support in both arms.
t5, y5, x5 = [], [], []
for _ in range(90):
    z = float(int(u01() * 3))
    ti = 1.0 if u01() < [0.3, 0.5, 0.7][int(z)] else 0.0
    t5.append(ti)
    x5.append([z])
    y5.append(float(1.0 + 1.5 * ti + 0.6 * z + 0.5 * ti * z + (u01() - 0.5)))
cases.append(case("discrete_covariate_profiles", t5, y5, x5, 5, None, None))
print(json.dumps({"cases": cases}))
