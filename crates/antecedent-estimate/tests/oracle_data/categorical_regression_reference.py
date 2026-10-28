# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Dummy-coded categorical regression with adjustment columns, exactly.

Regenerate with:  uv run --with mpmath --script categorical_regression_reference.py > categorical_regression_reference.json

For rows i with outcome y_i, an adjustment column g_i and level L_i, the design is
X = [1, g, 1{L = l} for each non-reference level l].  Everything linear is Fraction arithmetic on
the exact binary64 inputs: b = (X'X)^-1 X'y, e = y - Xb, sigma^2 = e'e / (n - p), and the covariance
is sigma^2 (X'X)^-1 (model based) or (X'X)^-1 [sum_i w_i e_i^2 x_i x_i'] (X'X)^-1 with w_i = 1
(HC0), n/(n-p) (HC1), 1/(1-h_i) (HC2) or 1/(1-h_i)^2 (HC3), h_i = x_i' (X'X)^-1 x_i.  Contrast
weights give estimate w'b and variance w'Vw; two-sided p-values are normal (mpmath, 50 digits);
the family is the level-versus-reference contrasts followed by the requested pairs and carries the
Holm step-down adjustment; the omnibus is b_levels' V_levels^-1 b_levels against the chi-square
survival function; the monotonicity statistic is the oriented minimum adjacent-step z with the
Bonferroni bound min(1, m Phi(T)).  Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50


def solve(A, B):
    """Exact A^-1 B by Gauss-Jordan; B is a list of columns' rows (matrix)."""
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
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
            for i in range(len(A))]


def normal_two_sided(z):
    return 2 * mp.ncdf(-abs(mp.mpf(z.numerator) / z.denominator)) if isinstance(z, F) else 2 * mp.ncdf(-abs(z))


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: (ps[i], i))
    out = [None] * m
    run = mp.mpf(0)
    for rank, i in enumerate(order):
        run = max(run, min(mp.mpf(1), (m - rank) * ps[i]))
        out[i] = run
    return out


def mpf(x):
    return mp.mpf(x.numerator) / x.denominator


def fit(levels_decl, reference, rows, adj, cov, pairs, mono):
    """rows: list of (level, y); adj: list of g or None."""
    n = len(rows)
    order = list(levels_decl)
    nonref = [l for l in order if l != reference]
    X = []
    for i, (lv, _) in enumerate(rows):
        x = [F(1)] + ([F(adj[i])] if adj is not None else []) + [F(1 if lv == l else 0) for l in nonref]
        X.append(x)
    y = [F(r[1]) for r in rows]
    p = len(X[0])
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [[sum(X[i][a] * y[i] for i in range(n))] for a in range(p)]
    ginv = solve(XtX, [[F(int(a == b)) for b in range(p)] for a in range(p)])
    b = [row[0] for row in matmul(ginv, Xty)]
    e = [y[i] - sum(X[i][a] * b[a] for a in range(p)) for i in range(n)]
    sigma2 = sum(v * v for v in e) / (n - p)
    if cov == "model_based":
        V = [[sigma2 * v for v in row] for row in ginv]
    else:
        meat = [[F(0)] * p for _ in range(p)]
        for i in range(n):
            h = sum(X[i][a] * sum(ginv[a][c] * X[i][c] for c in range(p)) for a in range(p))
            w = {"hc0": F(1), "hc1": F(n, n - p), "hc2": 1 / (1 - h), "hc3": 1 / (1 - h) ** 2}[cov]
            for a in range(p):
                for c in range(p):
                    meat[a][c] += w * e[i] * e[i] * X[i][a] * X[i][c]
        V = matmul(matmul(ginv, meat), ginv)
    t0 = p - len(nonref)
    idx = {l: t0 + j for j, l in enumerate(nonref)}

    def contrast(frm, to):
        w = [F(0)] * p
        if to != reference:
            w[idx[to]] += 1
        if frm != reference:
            w[idx[frm]] -= 1
        est = sum(w[a] * b[a] for a in range(p))
        var = sum(w[a] * V[a][c] * w[c] for a in range(p) for c in range(p))
        z = mpf(est) / mp.sqrt(mpf(var))
        return est, var, z, 2 * mp.ncdf(-abs(z))

    fam = [(reference, l) for l in nonref] + list(pairs)
    cs = [contrast(f, t) for f, t in fam]
    hp = holm([c[3] for c in cs])
    out = dict(coefficients=[float(v) for v in b], covariance=[float(v) for row in V for v in row],
               contrasts=[dict(estimate=float(c[0]), variance=float(c[1]), z=float(c[2]),
                               p=float(c[3]), p_holm=float(h)) for c, h in zip(cs, hp)])
    # omnibus over the level dummies
    k = len(nonref)
    Vk = [[V[t0 + a][t0 + c] for c in range(k)] for a in range(k)]
    bk = [[b[t0 + a]] for a in range(k)]
    sol = solve(Vk, bk)
    stat = sum(bk[a][0] * sol[a][0] for a in range(k))
    out["omnibus"] = dict(statistic=float(stat), df=k,
                          p=float(mp.gammainc(mp.mpf(k) / 2, mpf(stat) / 2, mp.inf, regularized=True)))
    if mono:
        sign = 1 if mono == "non_decreasing" else -1
        zs = [contrast(order[j], order[j + 1])[2] for j in range(len(order) - 1)]
        T = min(sign * z for z in zs)
        out["monotonicity"] = dict(statistic=float(T), p=float(min(mp.mpf(1), len(zs) * mp.ncdf(T))),
                                   steps=[float(contrast(order[j], order[j + 1])[0])
                                          for j in range(len(order) - 1)])
    return out


def build(name, seed, counts, offset, adj_kind, reference, cov, pairs, mono, ordered=True):
    rng = random.Random(seed)
    levels = list(counts)
    rows, adj = [], []
    for lv in levels:
        for _ in range(counts[lv]):
            g = round(rng.gauss(0, 1), 3)
            level_effect = {"a": 0.0, "b": 0.7, "c": 0.9, "d": 2.0}[lv]
            y = offset + level_effect + 0.5 * g + round(rng.gauss(0, 1), 3)
            rows.append((lv, y))
            adj.append(g)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rows = [rows[i] for i in order]
    adj = [adj[i] for i in order]
    if adj_kind == "none":
        adj = None
    elif adj_kind == "near_collinear":
        # adjustment almost equal to the level-d indicator: exact but ill-conditioned
        adj = [(1.0 if lv == "d" else 0.0) + 1e-4 * g for (lv, _), g in zip(rows, adj)]
    d = dict(name=name, levels=levels, ordered=ordered, reference=reference, covariance=cov,
             pairs=pairs, monotonicity=mono, outcomes=[r[1] for r in rows],
             row_levels=[r[0] for r in rows], adjustment=adj)
    d["expected"] = fit(levels if ordered else sorted(levels), reference, rows, adj, cov,
                        [tuple(p) for p in pairs], mono)
    return d


def main():
    cases = []
    c = {"a": 5, "b": 9, "c": 4, "d": 7}
    for cov in ["model_based", "hc0", "hc1", "hc2", "hc3"]:
        cases.append(build(f"adjusted_{cov}", 5, c, 0.0, "adjusted", "a", cov,
                           [["b", "d"], ["c", "a"]], "non_decreasing"))
    cases.append(build("no_adjustment_reference_inside", 6, c, 0.0, "none", "c", "hc1",
                       [["a", "d"]], "non_increasing"))
    cases.append(build("outcome_offset", 7, c, 1e8, "adjusted", "a", "hc3", [["b", "c"]], "non_decreasing"))
    cases.append(build("near_collinear_adjustment", 8, c, 0.0, "near_collinear", "a", "model_based",
                       [["b", "c"]], None))
    cases.append(build("tiny_levels", 9, {"a": 2, "b": 2, "c": 3}, 0.0, "adjusted", "b", "hc0",
                       [["a", "c"]], "non_decreasing"))
    json.dump(dict(cases=cases), fp=__import__("sys").stdout, separators=(",", ":"))


main()
