# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Cross-fitted multi-level AIPW with a multinomial-logit generalized propensity, fixed folds.

Regenerate with:  uv run --script generalized_propensity_aipw_reference.py > generalized_propensity_aipw_reference.json

The fold of every row is part of the case (FOLDS below is the level-stratified plan the product
derives for the recorded seed; the test asserts the product's plan equals it, so the nuisances
are fully determined).  For each fold, on its training rows:
  * the generalized propensity is the unpenalised multinomial logit maximum likelihood with level
    0 as baseline, solved by Newton's method at 60 digits and predicted on the held-out rows;
  * the outcome regression of each level k is ordinary least squares of the outcome on [1, z]
    over the training rows at level k, predicted on the held-out rows.
The per-level score is phi_k = m_k + 1{T = k}(Y - m_k)/max(e_k, clip) (the clip is a floor and is
absent when null).  E[Y(k)] is the mean of phi_k; a contrast's standard error is
sqrt(sum_i (d_i - mean d)^2 / (n - 1) / n) for d = phi_active - phi_control.  Values are written as
nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from mpmath import mp, mpf, exp, matrix, lu_solve, sqrt

mp.dps = 60
_state = 0x510E527FADE682D1

FOLDS = {"three_levels_two_covariates": [3, 0, 4, 1, 4, 4, 3, 1, 2, 3, 0, 1, 0, 1, 1, 2, 4, 0, 1, 3, 1, 2, 2, 4, 3, 1, 2, 3, 0, 2, 3, 2, 0, 4, 3, 0, 3, 2, 2, 3, 3, 4, 2, 2, 0, 0, 2, 0, 1, 4, 1, 4, 2, 2, 3, 3, 4, 4, 1, 0, 3, 4, 1, 4, 0, 2, 1, 0, 1, 2, 1, 4, 1, 0, 4, 1, 3, 0, 2, 4, 1, 3, 0, 0, 4, 4, 0, 3, 3, 2], "three_levels_three_folds": [0, 1, 1, 0, 1, 0, 2, 0, 1, 2, 2, 2, 1, 1, 1, 1, 2, 0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 0, 1, 2, 2, 2, 1, 2, 2, 1, 0, 2, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 0, 1, 0, 1, 0, 1, 1, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 1, 0, 2, 2, 0, 1, 2, 1, 0, 2, 2, 2, 1, 0, 0, 1, 0, 0, 0, 1], "clip_floor_binds": [0, 2, 0, 4, 3, 0, 0, 1, 3, 4, 4, 0, 3, 4, 2, 3, 2, 2, 2, 2, 3, 4, 1, 4, 4, 1, 1, 4, 3, 2, 2, 0, 0, 4, 1, 4, 0, 1, 3, 0, 3, 0, 3, 4, 2, 1, 4, 3, 0, 1, 3, 1, 2, 3, 3, 3, 4, 1, 4, 3, 2, 4, 0, 2, 2, 1, 2, 3, 1, 0, 2, 0, 1, 4, 0, 2, 2, 3, 1, 0, 2, 1, 3, 0, 3, 2, 1, 3, 4, 0, 0, 4, 4, 3, 4, 4, 4, 0, 1, 0, 3, 1, 3, 1, 4, 1, 0, 2, 4, 2, 1, 1, 2, 1, 3, 0, 2, 1, 2, 0], "four_levels": [3, 1, 3, 3, 1, 0, 2, 1, 1, 3, 0, 0, 2, 2, 0, 2, 1, 3, 0, 0, 0, 0, 3, 1, 1, 0, 2, 3, 1, 2, 2, 0, 2, 3, 3, 0, 2, 1, 1, 2, 3, 1, 1, 1, 2, 2, 0, 0, 2, 3, 3, 3, 1, 3, 3, 2, 0, 1, 0, 3, 0, 2, 1, 2, 2, 0, 1, 3, 0, 3, 3, 3, 0, 1, 0, 3, 0, 1, 0, 3, 1, 1, 3, 1, 2, 1, 2, 2, 1, 2, 0, 2, 3, 2, 1, 2, 3, 0, 0, 2], "outcome_offset_1e7": [3, 0, 4, 1, 4, 4, 3, 1, 2, 3, 0, 1, 0, 1, 1, 2, 4, 0, 1, 3, 1, 2, 2, 4, 3, 1, 2, 3, 0, 2, 3, 2, 0, 4, 3, 0, 3, 2, 2, 3, 3, 4, 2, 2, 0, 0, 2, 0, 1, 4, 1, 4, 2, 2, 3, 3, 4, 4, 1, 0, 3, 4, 1, 4, 0, 2, 1, 0, 1, 2, 1, 4, 1, 0, 4, 1, 3, 0, 2, 4, 1, 3, 0, 0, 4, 4, 0, 3, 3, 2], "covariate_offset_1e3": [3, 0, 4, 1, 4, 4, 3, 1, 2, 3, 0, 1, 0, 1, 1, 2, 4, 0, 1, 3, 1, 2, 2, 4, 3, 1, 2, 3, 0, 2, 3, 2, 0, 4, 3, 0, 3, 2, 2, 3, 3, 4, 2, 2, 0, 0, 2, 0, 1, 4, 1, 4, 2, 2, 3, 3, 4, 4, 1, 0, 3, 4, 1, 4, 0, 2, 1, 0, 1, 2, 1, 4, 1, 0, 4, 1, 3, 0, 2, 4, 1, 3, 0, 0, 4, 4, 0, 3, 3, 2], "moderately_collinear_covariates": [0, 2, 3, 1, 0, 1, 0, 2, 3, 1, 3, 1, 2, 3, 2, 2, 0, 0, 2, 0, 2, 3, 1, 2, 0, 2, 3, 0, 1, 2, 0, 2, 0, 3, 1, 1, 3, 1, 0, 3, 0, 1, 2, 2, 1, 1, 2, 2, 0, 2, 3, 1, 0, 3, 1, 0, 1, 0, 3, 3, 3, 2, 0, 3, 3, 0, 0, 3, 2, 3, 1, 3, 1, 2, 3, 0, 1, 1, 1, 2]}  # name -> fold id of every row (the product's level-stratified plan for the recorded seed)


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


def softmax_fit(rows, cat, K):
    p = len(rows[0])
    q = p * (K - 1)
    beta = [mpf(0)] * q
    def probs(r, beta):
        eta = [mpf(0)] + [sum(beta[(k - 1) * p + j] * r[j] for j in range(p)) for k in range(1, K)]
        m = max(eta)
        ex = [exp(e - m) for e in eta]
        s = sum(ex)
        return [e / s for e in ex]
    for _ in range(200):
        grad = [mpf(0)] * q
        H = matrix(q, q)
        for r, c in zip(rows, cat):
            pr = probs(r, beta)
            for k in range(1, K):
                yk = 1 if c == k else 0
                for j in range(p):
                    grad[(k - 1) * p + j] += (yk - pr[k]) * r[j]
            for k in range(1, K):
                for l in range(1, K):
                    w = pr[k] * ((1 if k == l else 0) - pr[l])
                    for j in range(p):
                        for m_ in range(p):
                            H[(k - 1) * p + j, (l - 1) * p + m_] += w * r[j] * r[m_]
        step = lu_solve(H, matrix(grad))
        beta = [beta[i] + step[i] for i in range(q)]
        if max(abs(s) for s in step) < mpf(10) ** -45:
            break
    return lambda r: probs(r, beta)


def case(name, cat, y, z, K, k, clip, folds=None):
    n = len(cat)
    X = [[mpf(1)] + [mpf(c) for c in row] for row in z]
    yy = [mpf(v) for v in y]
    folds = FOLDS.get(name) or [i % k for i in range(n)]
    e = [[None] * K for _ in range(n)]
    m = [[None] * K for _ in range(n)]
    for f in range(k):
        train = [i for i in range(n) if folds[i] != f]
        valid = [i for i in range(n) if folds[i] == f]
        prob = softmax_fit([X[i] for i in train], [cat[i] for i in train], K)
        outc = []
        for c in range(K):
            rows = [i for i in train if cat[i] == c]
            outc.append(ols([X[i] for i in rows], [yy[i] for i in rows]))
        for i in valid:
            pr = prob(X[i])
            for c in range(K):
                e[i][c] = pr[c]
                m[i][c] = outc[c](X[i])
    phi = [[None] * K for _ in range(n)]
    for i in range(n):
        for c in range(K):
            ec = max(e[i][c], mpf(clip)) if clip else e[i][c]
            phi[i][c] = m[i][c] + (1 if cat[i] == c else 0) * (yy[i] - m[i][c]) / ec
    means = [sum(phi[i][c] for i in range(n)) / n for c in range(K)]
    out = {"name": name, "cat": cat, "y": y, "z": z, "K": K, "k": k, "clip": clip,
           "folds": folds, "means": [float(v) for v in means], "contrasts": []}
    for a in range(K):
        for b in range(a):
            d = [phi[i][a] - phi[i][b] for i in range(n)]
            md = sum(d) / n
            se = sqrt(sum((v - md) ** 2 for v in d) / (n - 1) / n)
            out["contrasts"].append({"active": a, "control": b, "effect": float(md), "se": float(se)})
    out["min_e"] = float(min(e[i][c] for i in range(n) for c in range(K)))
    return out


def draw(n, K, p, slope=0.35, spread=1.6):
    cat, y, z = [], [], []
    for _ in range(n):
        row = [u01() * spread * 2 - spread for _ in range(p)]
        eta = [0.0] + [slope * (k) * row[0] * (1 if k % 2 else -1) + 0.2 * k for k in range(1, K)]
        ex = [2.718281828459045 ** e for e in eta]
        u, acc, c = u01() * sum(ex), 0.0, K - 1
        for k in range(K):
            acc += ex[k]
            if u < acc:
                c = k
                break
        yi = 1.0 + 0.8 * c + 0.9 * row[0] + (0.3 * c * row[0]) + (u01() - 0.5)
        cat.append(c)
        y.append(float(yi))
        z.append(row)
    return cat, y, z


cases = []
cat, y, z = draw(90, 3, 2)
cases.append(case("three_levels_two_covariates", cat, y, z, 3, 5, None))
cases.append(case("three_levels_three_folds", cat, y, z, 3, 3, None))
cat2, y2, z2 = draw(120, 3, 1, slope=0.9)
cases.append(case("clip_floor_binds", cat2, y2, z2, 3, 5, 0.12))
cat3, y3, z3 = draw(100, 4, 2, slope=0.25)
cases.append(case("four_levels", cat3, y3, z3, 4, 4, None))
cases.append(case("outcome_offset_1e7", cat, [v + 1.0e7 for v in y], z, 3, 5, None))
cases.append(case("covariate_offset_1e3", cat, y, [[1.0e3 + 250.0 * r[0], r[1]] for r in z], 3, 5, None))
c5, y5, z5 = draw(80, 3, 2)
z5m = [[r[0], r[0] + 1.0e-2 * r[1]] for r in z5]
cases.append(case("moderately_collinear_covariates", c5, y5, z5m, 3, 4, None))
z5 = [[r[0], r[0] + 1.0e-4 * r[1]] for r in z5]
FOLDS["near_collinear_covariates"] = FOLDS["moderately_collinear_covariates"]
cases.append(case("near_collinear_covariates", c5, y5, z5, 3, 4, None))
print(json.dumps({"cases": cases}))
