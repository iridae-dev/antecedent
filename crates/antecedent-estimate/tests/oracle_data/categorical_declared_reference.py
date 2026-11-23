# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Dummy-coded categorical regression under declared sampling weights, clusters and serial lags,
in exact rational arithmetic.

Regenerate with:  python3 categorical_declared_reference.py > categorical_declared_reference.json

Written from the estimating equations, not from the product code.  Declared weights w_i with
w_i = 0 rows deleted are normalized to s_i = w_i n / sum w (n the rows kept).  The design is
x_i = [1, g_i, 1{L_i = l} for each non-reference level l]; the weighted normal equations give
b = A^-1 sum_i s_i x_i y_i with A = sum_i s_i x_i x_i', the residual is e_i = y_i - x_i'b and the
weighted score is z_i = s_i x_i e_i.  The covariance is A^-1 M A^-1 with the meat M:
  hc0   sum_i z_i z_i'                         hc1  n/(n-p) times hc0
  hc2   sum_i z_i z_i' / (1 - h_i)             hc3  sum_i z_i z_i' / (1 - h_i)^2, h_i = s_i x_i' A^-1 x_i
  cluster_robust  G/(G-1) (n-1)/(n-p) sum_g S_g S_g', S_g = sum_{i in g} z_i
  newey_west      Gamma_0 + sum_{l=1}^{L} (1 - l/(L+1)) (Gamma_l + Gamma_l'), Gamma_l = sum_t z_t z_{t-l}'
Contrast weights give the estimate c'b and the variance c'Vc over the dummy block; the omnibus is
b_d' V_dd^-1 b_d.  Everything is a Fraction of the exact binary64 inputs; floats are written only at the end.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F


def solve(A, B):
    n = len(A)
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def outer_add(M, u, v, c=1):
    for a in range(len(u)):
        for b in range(len(v)):
            M[a][b] += c * u[a] * v[b]


def fit(levels, reference, rows, adj, weights, groups, lag, kind, pairs):
    keep = [i for i in range(len(rows)) if weights is None or weights[i] != 0]
    rows = [rows[i] for i in keep]
    adj = None if adj is None else [adj[i] for i in keep]
    groups = None if groups is None else [groups[i] for i in keep]
    n = len(rows)
    if weights is None:
        s = [F(1)] * n
        used = None
    else:
        raw = [F(weights[i]) for i in keep]
        tot = sum(raw)
        s = [r * n / tot for r in raw]
        used = raw
    nonref = [l for l in levels if l != reference]
    X = [[F(1)] + ([F(adj[i])] if adj is not None else []) + [F(1 if rows[i][0] == l else 0) for l in nonref]
         for i in range(n)]
    y = [F(r[1]) for r in rows]
    p = len(X[0])
    A = [[sum(s[i] * X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Ainv = solve(A, [[F(int(a == b)) for b in range(p)] for a in range(p)])
    rhs = [[sum(s[i] * X[i][a] * y[i] for i in range(n))] for a in range(p)]
    beta = [r[0] for r in matmul(Ainv, rhs)]
    e = [y[i] - sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    z = [[s[i] * X[i][a] * e[i] for a in range(p)] for i in range(n)]
    M = [[F(0)] * p for _ in range(p)]
    if kind in ("hc0", "hc1", "hc2", "hc3"):
        for i in range(n):
            h = s[i] * sum(X[i][a] * sum(Ainv[a][b] * X[i][b] for b in range(p)) for a in range(p))
            c = {"hc0": F(1), "hc1": F(n, n - p), "hc2": 1 / (1 - h), "hc3": 1 / ((1 - h) ** 2)}[kind]
            outer_add(M, z[i], z[i], c)
    elif kind == "cluster_robust":
        ids = sorted(set(groups))
        G = len(ids)
        for g in ids:
            S = [sum(z[i][a] for i in range(n) if groups[i] == g) for a in range(p)]
            outer_add(M, S, S)
        f = F(G, G - 1) * F(n - 1, n - p)
        M = [[v * f for v in row] for row in M]
    else:
        for i in range(n):
            outer_add(M, z[i], z[i])
        for l in range(1, lag + 1):
            w = 1 - F(l, lag + 1)
            for t in range(l, n):
                outer_add(M, z[t], z[t - l], w)
                outer_add(M, z[t - l], z[t], w)
    V = matmul(matmul(Ainv, M), Ainv)
    t0 = p - len(nonref)
    idx = {l: t0 + j for j, l in enumerate(nonref)}

    def contrast(frm, to):
        w = {}
        if to != reference:
            w[idx[to]] = w.get(idx[to], 0) + 1
        if frm != reference:
            w[idx[frm]] = w.get(idx[frm], 0) - 1
        est = sum(c * beta[i] for i, c in w.items())
        var = sum(ci * cj * V[i][j] for i, ci in w.items() for j, cj in w.items())
        return est, var

    dummy = [[V[t0 + a][t0 + b] for b in range(len(nonref))] for a in range(len(nonref))]
    bd = [[beta[t0 + a]] for a in range(len(nonref))]
    try:
        sol = solve(dummy, bd)
        omnibus = sum(bd[a][0] * sol[a][0] for a in range(len(nonref)))
    except StopIteration:
        omnibus = None  # a singular dummy covariance: the joint test does not exist
    out = {
        "rows_used": n,
        "coefficients": [{"name": "level:" + l, "estimate": float(beta[idx[l]])} for l in nonref],
        "covariance": [[float(dummy[a][b]) for b in range(len(nonref))] for a in range(len(nonref))],
        "levels": [{"level": l, "estimate": float(contrast(reference, l)[0]),
                    "variance": float(contrast(reference, l)[1])} for l in nonref],
        "pairs": [{"from": a, "to": b, "estimate": float(contrast(a, b)[0]),
                   "variance": float(contrast(a, b)[1])} for a, b in pairs],
        "omnibus": None if omnibus is None else float(omnibus),
        "counts": {l: sum(1 for r in rows if r[0] == l) for l in levels},
    }
    if groups is not None:
        out["clusters"] = len(set(groups))
    if used is not None:
        out["zero_weight_rows"] = len(weights) - len(keep)
        out["weight_sum"] = float(sum(used))
        out["kish"] = float(sum(s) ** 2 / sum(v * v for v in s))
    return out


def case(name, seed, n, sizes, kind, weights=None, groups=None, lag=None, adjust=True, offset=0.0):
    r = random.Random(seed)
    levels = ["a", "b", "c"]
    labels = []
    for lv, k in zip(levels, sizes):
        labels += [lv] * k
    r.shuffle(labels)
    adj = [round(r.uniform(-1, 1), 3) for _ in labels] if adjust else None
    shift = {"a": 0.0, "b": 0.7, "c": -0.4}
    y = [round(offset + shift[l] + (0.5 * adj[i] if adj else 0.0) + r.uniform(-1, 1), 4) for i, l in enumerate(labels)]
    rows = list(zip(labels, y))
    pairs = [("b", "c")]
    res = fit(levels, "a", rows, adj, weights, groups, lag, kind, pairs)
    return {"name": name, "covariance": kind, "levels": levels, "reference": "a", "row_levels": labels,
            "outcomes": y, "adjustment": adj, "weights": weights, "groups": groups, "lag": lag,
            "pairs": [list(p) for p in pairs], "expected": res}


def main():
    cases = []
    r = random.Random(5)
    w = [round(0.2 + 3 * r.random(), 3) for _ in range(40)]
    w[3] = 0
    w[11] = 0
    w[20] = 60
    cases.append(case("weights_hc0_zero_and_extreme", 1, 40, [14, 13, 13], "hc0", weights=w))
    cases.append(case("weights_hc1", 2, 40, [14, 13, 13], "hc1", weights=w))
    cases.append(case("weights_hc3_leverage", 3, 40, [14, 13, 13], "hc3", weights=[round(0.5 + 2 * r.random(), 3) for _ in range(40)]))
    cases.append(case("weights_hc2_offset_1e8", 4, 36, [12, 12, 12], "hc2", weights=[round(0.5 + 2 * r.random(), 3) for _ in range(36)], offset=1.0e8))
    cases.append(case("equal_weights_at_scale", 6, 30, [10, 10, 10], "hc0", weights=[3] * 30))
    sizes = [1, 1, 2, 3, 5, 8, 13, 17]
    g = []
    for k, sz in enumerate(sizes):
        g += [k] * sz
    cases.append(case("clusters_unbalanced_singletons", 7, 50, [17, 17, 16], "cluster_robust",
                      groups=[g[(i * 29) % 50] for i in range(50)]))
    cases.append(case("clusters_with_weights_and_zero_rows", 8, 45, [15, 15, 15], "cluster_robust",
                      weights=[0 if i in (4, 9) else round(0.3 + 2 * r.random(), 3) for i in range(45)],
                      groups=[i % 6 for i in range(45)]))
    cases.append(case("three_clusters_tiny", 12, 15, [5, 5, 5], "cluster_robust", groups=[i % 3 for i in range(15)], adjust=False))
    cases.append(case("two_clusters_singular_joint_test", 9, 14, [5, 5, 4], "cluster_robust", groups=[i % 2 for i in range(14)], adjust=False))
    cases.append(case("serial_lag_1", 10, 40, [14, 13, 13], "newey_west", lag=1))
    cases.append(case("serial_lag_3_weighted", 11, 40, [14, 13, 13], "newey_west", lag=3,
                      weights=[round(0.4 + 1.6 * ((i * 0.618) % 1), 3) for i in range(40)]))
    print(json.dumps(cases, indent=1))


if __name__ == "__main__":
    main()
