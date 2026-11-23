# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Front-door path-sum product of coefficients with the stacked sandwich standard error, at 80 digits.

Regenerate with:  uv run --script frontdoor_two_stage_reference.py > frontdoor_two_stage_reference.json

Stage 1, one regression per mediator: M_j ~ [1, T] with slope a_j.  Stage 2: Y ~ [1, T, M_1..M_k]
with mediator coefficients b_j.  The effect is delta * sum_j a_j b_j, delta the treatment-level
difference.  The k + 1 regressions are stacked into one estimating-equation system with scores
e_ij x_i; the bread is block diagonal in the Gram matrices and the meat is the sum of outer
products of the row scores (HC0) or of cluster-summed scores (cluster).  The documented
small-sample convention is applied equation by equation: meat entry (r, s) is multiplied by
sqrt(f_r f_s) with f = n/(n-p) for HC1 and f = G/(G-1) (n-1)/(n-p) for the cluster form, p the
number of columns of the equation owning the parameter.  Var = grad' A^-1 B A^-1 grad with
grad(a_j) = delta b_j and grad(b_j) = delta a_j.  Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random

from mpmath import mp, mpf, matrix, lu_solve, inverse, sqrt

mp.dps = 80
rng = random.Random(77)


def ols(X, y):
    p = len(X[0])
    A = matrix(p, p)
    b = matrix(p, 1)
    for row, v in zip(X, y):
        for i in range(p):
            b[i] += row[i] * v
            for j in range(p):
                A[i, j] += row[i] * row[j]
    return lu_solve(A, b), A


def solve(t, ms, y, delta, clusters):
    n, k = len(t), len(ms)
    t_, y_ = [mpf(v) for v in t], [mpf(v) for v in y]
    ms_ = [[mpf(v) for v in m] for m in ms]
    X1 = [[mpf(1), t_[i]] for i in range(n)]
    X2 = [[mpf(1), t_[i]] + [ms_[j][i] for j in range(k)] for i in range(n)]
    coefs1, G1, res1 = [], None, []
    for j in range(k):
        beta, G1 = ols(X1, ms_[j])
        coefs1.append(beta)
        res1.append([ms_[j][i] - beta[0] - beta[1] * t_[i] for i in range(n)])
    beta2, G2 = ols(X2, y_)
    res2 = [y_[i] - sum(beta2[c] * X2[i][c] for c in range(2 + k)) for i in range(n)]
    path = sum(coefs1[j][1] * beta2[2 + j] for j in range(k))
    effect = path * delta
    npar = 2 * k + 2 + k
    owner_p = [2] * (2 * k) + [2 + k] * (2 + k)
    off2 = 2 * k
    scores = []
    for i in range(n):
        s = [mpf(0)] * npar
        for j in range(k):
            for c in range(2):
                s[2 * j + c] = res1[j][i] * X1[i][c]
        for c in range(2 + k):
            s[off2 + c] = res2[i] * X2[i][c]
        scores.append(s)
    inv1, inv2 = inverse(G1), inverse(G2)
    Ainv = matrix(npar, npar)
    for j in range(k):
        for r in range(2):
            for c in range(2):
                Ainv[2 * j + r, 2 * j + c] = inv1[r, c]
    for r in range(2 + k):
        for c in range(2 + k):
            Ainv[off2 + r, off2 + c] = inv2[r, c]
    grad = matrix(npar, 1)
    for j in range(k):
        grad[2 * j + 1] = delta * beta2[2 + j]
        grad[off2 + 2 + j] = delta * coefs1[j][1]

    def se(kind):
        if kind == "cluster":
            tot = {}
            for c, s in zip(clusters, scores):
                tot.setdefault(c, [mpf(0)] * npar)
                tot[c] = [a + b for a, b in zip(tot[c], s)]
            units = list(tot.values())
            Gc = len(units)
        else:
            units = scores
        B = matrix(npar, npar)
        for s in units:
            for r in range(npar):
                for c in range(npar):
                    B[r, c] += s[r] * s[c]
        for r in range(npar):
            for c in range(npar):
                def f(p):
                    if kind == "hc0":
                        return mpf(1)
                    if kind == "hc1":
                        return mpf(n) / (n - p)
                    return mpf(Gc) / (Gc - 1) * (n - 1) / (n - p)
                B[r, c] *= sqrt(f(owner_p[r]) * f(owner_p[c]))
        cov = Ainv * B * Ainv
        return float(sqrt((grad.T * cov * grad)[0]))

    return dict(effect=float(effect), se_hc0=se("hc0"), se_hc1=se("hc1"), se_cluster=se("cluster"))


def make(name, n, k, t_offset=0.0, m_offset=0.0, y_offset=0.0, delta=1.0, cluster_size=4):
    t = [t_offset + rng.gauss(0, 1) for _ in range(n)]
    ms = []
    for j in range(k):
        shared = [rng.gauss(0, 1) for _ in range(n)]
        ms.append([m_offset + (0.8 - 0.3 * j) * (t[i] - t_offset) + 0.5 * shared[i] + 0.2 * rng.gauss(0, 1)
                   for i in range(n)])
    y = [y_offset + sum((1.5 - 0.7 * j) * (ms[j][i] - m_offset) for j in range(k)) + 0.4 * rng.gauss(0, 1)
         for i in range(n)]
    clusters = [i // cluster_size for i in range(n)]
    case = dict(name=name, t=t, m=ms, y=y, delta=delta, t_low=t_offset, clusters=clusters)
    case.update(solve(t, ms, y, delta, clusters))
    return case


cases = [
    make("one_mediator", 40, 1),
    make("two_mediators_delta_two", 50, 2, delta=2.0),
    make("three_mediators", 60, 3, cluster_size=6),
    make("offsets", 50, 2, t_offset=1e3, m_offset=1e5, y_offset=1e6, delta=2.0),
]
print(json.dumps(dict(cases=cases)))
