# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Conditional temporal effect over declared modifiers, solved exactly.

Regenerate with:  uv run --script temporal_conditional_effect_reference.py > temporal_conditional_effect_reference.json

A case is a series (x treatment, y outcome, z and w confounders and modifiers; a missing flag per
time point).  The outcome at time t is regressed, over the windows whose every read value is
observed, on an intercept, the treatment at t - 1, the modifiers at t, their products with the
treatment, and the adjustment values z at t - 1 and t - 2 (the modifier z at t is its own column,
the same variable at other lags stays an adjustment node).  The conditional effect is
delta * (b_T + sum_j b_{T x W_j} mean(W_j)).  Its standard error comes from the influence function
of the pair (regression, modifier means), written in the uncentred parametrization:
    IF_i = g' (X'X)^-1 x_i e_i + (1/n) sum_j b_{T x W_j} (W_ij - mean(W_j)),
with g = (0, 1, 0.., mean(W).., 0..) the gradient in the regression coefficients.  The reported
variances are the HC0 sum of IF_i^2, the HC1 rescaling n / (n - p), and the homoskedastic form
sigma2 g' (X'X)^-1 g + sum_i ((1/n) sum_j b_{T x W_j} (W_ij - mean(W_j)))^2 with sigma2 = rss / (n - p).
Everything is in fractions.Fraction; values are written as the nearest binary64.  The series
values are binary64 numbers converted to Fractions exactly.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F
from math import isqrt

_state = 0x9E3779B97F4A7C15


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def solve(A, B):
    n = len(A)
    M = [row[:] + b[:] for row, b in zip(A, B)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def fsqrt(q, digits=60):
    scaled = q * 10 ** (2 * digits)
    return F(isqrt(scaled.numerator // scaled.denominator), 10**digits)


def fit(x, y, mods, z, missing, delta):
    q = len(mods)
    n = len(y)
    rows, ys, times, wrows = [], [], [], []
    for t in range(2, n):
        reads = [t, t - 1, t - 2]
        if any(missing[r] for r in reads):
            continue
        w = [F(m[t]) for m in mods]
        tr = F(x[t - 1])
        rows.append([F(1), tr] + w + [tr * v for v in w] + [F(z[t - 1]), F(z[t - 2])])
        wrows.append(w)
        ys.append(F(y[t]))
        times.append(t)
    m, p = len(rows), len(rows[0])
    A = [[sum(r[a] * r[b] for r in rows) for b in range(p)] for a in range(p)]
    ident = [[F(int(a == b)) for b in range(p)] for a in range(p)]
    inv = solve(A, ident)
    xty = [sum(r[a] * v for r, v in zip(rows, ys)) for a in range(p)]
    beta = [sum(inv[a][b] * xty[b] for b in range(p)) for a in range(p)]
    resid = [v - sum(bk * rk for bk, rk in zip(beta, r)) for r, v in zip(rows, ys)]
    rss = sum(e * e for e in resid)
    mu = [sum(w[j] for w in wrows) / m for j in range(q)]
    theta = beta[1] + sum(beta[2 + q + j] * mu[j] for j in range(q))
    g = [F(0)] * p
    g[1] = F(1)
    for j in range(q):
        g[2 + q + j] = mu[j]
    ginv = [sum(g[a] * inv[a][b] for a in range(p)) for b in range(p)]  # g'(X'X)^-1
    reg_if = [sum(ginv[b] * r[b] for b in range(p)) * e for r, e in zip(rows, resid)]
    mean_if = [sum(beta[2 + q + j] * (w[j] - mu[j]) for j in range(q)) / m for w in wrows]
    total = [a + b for a, b in zip(reg_if, mean_if)]
    hc0 = sum(u * u for u in total)
    hc1 = hc0 * m / (m - p)
    sigma2 = rss / (m - p)
    homo = sigma2 * sum(ginv[b] * g[b] for b in range(p)) + sum(b * b for b in mean_if)
    d = abs(F(delta))
    return {
        "rows": m,
        "times": times,
        "effect": float(F(delta) * theta),
        "se_homoskedastic": float(fsqrt(homo) * d),
        "se_hc0": float(fsqrt(hc0) * d),
        "se_hc1": float(fsqrt(hc1) * d),
        "modifier_means": [float(v) for v in mu],
        "coefficients": [float(b) for b in beta],
    }


def series(n, offs, drop):
    x_off, y_off, z_off, w_off = offs
    z, w, zv, wv = [], [], 0.0, 0.0
    for t in range(n):
        zv = 0.8 * zv + (u01() - 0.5)
        wv = 0.5 * wv + (u01() - 0.5)
        z.append(zv + z_off)
        w.append(wv + w_off)
    x, y = [], []
    for t in range(n):
        zt1 = z[t - 1] - z_off if t >= 1 else 0.0
        x.append(x_off + 0.6 * zt1 + (u01() - 0.5))
    for t in range(n):
        xt1 = x[t - 1] - x_off if t >= 1 else 0.0
        zt1 = z[t - 1] - z_off if t >= 1 else 0.0
        zt2 = z[t - 2] - z_off if t >= 2 else 0.0
        zt0 = z[t] - z_off
        wt0 = w[t] - w_off
        y.append(
            y_off + (1.5 + 0.7 * zt0 - 0.4 * wt0) * xt1 + 0.8 * zt1 - 0.5 * zt2 + 0.3 * wt0
            + 0.4 * (u01() - 0.5)
        )
    return x, y, z, w, [t in drop for t in range(n)]


def case(name, n, offs, drop, modifiers, delta):
    x, y, z, w, missing = series(n, offs, drop)
    mods = [z] if modifiers == ["z"] else [z, w]
    out = fit(x, y, mods, z, missing, delta)
    out.update(name=name, x=x, y=y, z=z, w=w, missing=missing, modifiers=modifiers, delta=delta)
    return out


cases = [
    case("one_modifier", 40, (0.0, 0.0, 0.0, 0.0), set(), ["z"], 1.0),
    case("two_modifiers", 40, (0.0, 0.0, 0.0, 0.0), set(), ["z", "w"], 1.0),
    case("offsets", 40, (1.0e3, 1.0e6, 5.0e2, -2.0e2), set(), ["z", "w"], 1.0),
    case("gaps", 44, (0.0, 0.0, 0.0, 0.0), {9, 10, 25, 33}, ["z", "w"], 1.0),
    case("tiny", 12, (0.0, 0.0, 0.0, 0.0), set(), ["z"], 1.0),
    case("delta_two", 40, (0.0, 0.0, 0.0, 0.0), set(), ["z", "w"], 2.5),
]
print(json.dumps({"cases": cases}, indent=1))
