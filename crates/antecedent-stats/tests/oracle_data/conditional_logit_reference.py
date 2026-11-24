"""Independent references for the conditional logit fit.

Regenerate with:  python3 conditional_logit_reference.py > conditional_logit_reference.json

Standard library only.  Each data set is drawn from `random.Random(seed)`: choice sets of 2 to 5
options, 1 to 4 features rounded to six decimals (exact binary64 values, carried into the
arithmetic below with `Decimal(float)`, which is exact), and one choice per set sampled from the
model at a fixed coefficient vector.

The maximum is found by Newton's method in 50-digit `decimal` arithmetic, written directly from
the defining formulas on the raw option features (no differencing against the chosen option):

* l(b)  = sum_s [ x_c . b - ln sum_j exp(x_j . b) ]
* g(b)  = sum_s [ x_c - sum_j p_j x_j ]
* H(b)  = sum_s [ sum_j p_j x_j x_j' - (sum_j p_j x_j)(sum_j p_j x_j)' ]    (observed information)

with p_j = exp(x_j . b) / sum_k exp(x_k . b).  The Newton system H d = g and the inverse of H are
solved by Gauss-Jordan elimination with partial pivoting in the same 50-digit arithmetic.  Steps
are halved while the log-likelihood decreases; iteration stops when every step entry is below
1e-40.  The covariance is H^-1 at the maximum; the cluster covariance is
C / (C - 1) H^-1 (sum_c s_c s_c') H^-1 with s_c the sum over the sets of cluster c of their score
vectors and C the number of clusters (the finite-cluster correction).

This script and its output are repository-authored (MIT OR Apache-2.0).
"""
import json
import math
import random
import sys
from decimal import Decimal as D, getcontext

getcontext().prec = 50

CASES = [
    # name, seed, sets, features, true coefficients, per-set offset on feature 0, clusters
    ("one_feature", 11, 60, 1, [0.8], 0.0, None),
    ("two_features_clustered", 23, 150, 2, [1.1, -0.6], 0.0, 30),
    ("three_features", 37, 220, 3, [0.4, -1.2, 0.7], 0.0, None),
    ("four_features_clustered", 41, 400, 4, [0.5, 0.9, -0.4, 0.3], 0.0, 57),
    ("offset_feature_clustered", 53, 120, 2, [0.9, -0.5], 1000.0, 15),
]


def draw(seed, n_sets, p, beta, offset, n_clusters):
    rng = random.Random(seed)
    features, starts, chosen, clusters = [], [0], [], []
    for _ in range(n_sets):
        k = rng.randint(2, 5)
        shift = offset * rng.uniform(-1.0, 1.0) if offset else 0.0
        rows = []
        for _ in range(k):
            row = [round(rng.gauss(0.0, 1.0), 6) for _ in range(p)]
            row[0] = round(row[0] + shift, 6)
            rows.append(row)
        eta = [sum(b * x for b, x in zip(beta, row)) for row in rows]
        top = max(eta)
        w = [math.exp(e - top) for e in eta]
        u = rng.random() * sum(w)
        pick, acc = k - 1, 0.0
        for j, wj in enumerate(w):
            acc += wj
            if u < acc:
                pick = j
                break
        for row in rows:
            features.extend(row)
        starts.append(starts[-1] + k)
        chosen.append(pick)
        if n_clusters:
            clusters.append(rng.randrange(n_clusters))
    return features, starts, chosen, (clusters if n_clusters else None)


def set_terms(rows, c, b):
    """Log-likelihood term, score and information of one set at b."""
    p = len(b)
    eta = [sum(x * bb for x, bb in zip(row, b)) for row in rows]
    top = max(eta)
    w = [(e - top).exp() for e in eta]
    total = sum(w, D(0))
    ll = eta[c] - top - total.ln()
    prob = [wi / total for wi in w]
    mean = [sum((prob[j] * rows[j][i] for j in range(len(rows))), D(0)) for i in range(p)]
    score = [rows[c][i] - mean[i] for i in range(p)]
    info = [[sum((prob[j] * rows[j][a] * rows[j][e] for j in range(len(rows))), D(0))
             - mean[a] * mean[e] for e in range(p)] for a in range(p)]
    return ll, score, info


def totals(sets, b):
    p = len(b)
    ll = D(0)
    g = [D(0)] * p
    h = [[D(0)] * p for _ in range(p)]
    scores = []
    for rows, c in sets:
        l, s, i = set_terms(rows, c, b)
        ll += l
        scores.append(s)
        g = [a + e for a, e in zip(g, s)]
        h = [[h[a][e] + i[a][e] for e in range(p)] for a in range(p)]
    return ll, g, h, scores


def gauss_jordan(a, rhs_cols):
    """Solve A X = B for square A (list of rows) and B given as a list of rows."""
    n = len(a)
    m = [list(a[i]) + list(rhs_cols[i]) for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        if pv == 0:
            raise ValueError("singular")
        m[col] = [v / pv for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [vr - f * vc for vr, vc in zip(m[r], m[col])]
    return [row[n:] for row in m]


def fit(features, starts, chosen, p):
    sets = []
    for s in range(len(chosen)):
        rows = [[D(features[r * p + i]) for i in range(p)] for r in range(starts[s], starts[s + 1])]
        sets.append((rows, chosen[s]))
    b = [D(0)] * p
    ll, g, h, _ = totals(sets, b)
    null_ll = ll
    for _ in range(200):
        step = [row[0] for row in gauss_jordan(h, [[gi] for gi in g])]
        t = D(1)
        while True:
            cand = [bi + t * di for bi, di in zip(b, step)]
            cll, cg, ch, _ = totals(sets, cand)
            if cll >= ll or t < D(10) ** -30:
                break
            t /= 2
        b, ll, g, h = cand, cll, cg, ch
        if max(abs(t * d) for d in step) < D(10) ** -40:
            break
    else:
        raise RuntimeError("no convergence")
    ll, g, h, scores = totals(sets, b)
    ident = [[D(1) if i == j else D(0) for j in range(p)] for i in range(p)]
    cov = gauss_jordan(h, ident)
    return sets, b, ll, null_ll, cov, scores


def sandwich(cov, scores, clusters, p):
    sums = {}
    for s, label in enumerate(clusters):
        acc = sums.setdefault(label, [D(0)] * p)
        sums[label] = [a + e for a, e in zip(acc, scores[s])]
    meat = [[sum((v[i] * v[j] for v in sums.values()), D(0)) for j in range(p)] for i in range(p)]
    left = [[sum((cov[i][k] * meat[k][j] for k in range(p)), D(0)) for j in range(p)]
            for i in range(p)]
    c = len(sums)
    correction = D(c) / D(c - 1) if c >= 2 else D(1)
    return [[correction * sum((left[i][k] * cov[k][j] for k in range(p)), D(0))
             for j in range(p)] for i in range(p)]


def flat(m):
    return [float(v) for row in m for v in row]


def main():
    out = []
    for name, seed, n_sets, p, beta, offset, n_clusters in CASES:
        features, starts, chosen, clusters = draw(seed, n_sets, p, beta, offset, n_clusters)
        _, b, ll, null_ll, cov, scores = fit(features, starts, chosen, p)
        case = {
            "name": name,
            "n_features": p,
            "features": features,
            "set_starts": starts,
            "chosen": chosen,
            "clusters": clusters,
            "coefficients": [float(v) for v in b],
            "log_likelihood": float(ll),
            "null_log_likelihood": float(null_ll),
            "covariance": flat(cov),
            "cluster_covariance": flat(sandwich(cov, scores, clusters, p)) if clusters else None,
        }
        out.append(case)
    json.dump(out, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
