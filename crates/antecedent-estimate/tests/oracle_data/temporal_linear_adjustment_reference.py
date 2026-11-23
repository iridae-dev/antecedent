# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Lag-aligned linear adjustment: the design the temporal route builds, solved exactly.

Regenerate with:  uv run --script temporal_linear_adjustment_reference.py > temporal_linear_adjustment_reference.json

A case is a series (x treatment, y outcome, z confounder; a missing flag per time point) and a
design written in the definition's terms: the outcome at time t is regressed on an intercept, the
treatment at time t - lag_t and the confounder at times t - lag_z for each declared lag.  The
windows are the times t >= the largest lag whose every read value is observed.  The coefficient
of the treatment, the effect (delta times the coefficient) and the homoskedastic standard error
sqrt(sigma2 [(X'X)^-1]_tt) with sigma2 = rss / (n - p) are computed in exact rational arithmetic
(fractions.Fraction); values are written as the nearest binary64.  The series values are binary64
numbers, converted to Fractions exactly.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F
from math import isqrt

_state = 0x243F6A8885A308D3


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def solve(A, B):
    n = len(A)
    M = [row[:] + b[:] for row, b in zip(A, B)]
    m = len(B[0])
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
    root = isqrt(scaled.numerator // scaled.denominator)
    return F(root, 10**digits)


def fit(x, y, z, missing, lag_t, lag_zs, delta):
    n = len(y)
    max_lag = max([lag_t] + lag_zs)
    rows, ys, times = [], [], []
    for t in range(max_lag, n):
        reads = [t, t - lag_t] + [t - l for l in lag_zs]
        if any(missing[r] for r in reads):
            continue
        rows.append([F(1), F(x[t - lag_t])] + [F(z[t - l]) for l in lag_zs])
        ys.append(F(y[t]))
        times.append(t)
    n_eff, p = len(rows), len(rows[0])
    A = [[sum(r[a] * r[b] for r in rows) for b in range(p)] for a in range(p)]
    rhs = [[sum(r[a] * v for r, v in zip(rows, ys))] for a in range(p)]
    ident = [[F(int(a == b)) for b in range(p)] for a in range(p)]
    inv = solve(A, ident)
    beta = [sum(inv[a][b] * rhs[b][0] for b in range(p)) for a in range(p)]
    rss = sum((v - sum(bk * rk for bk, rk in zip(beta, r))) ** 2 for r, v in zip(rows, ys))
    sigma2 = rss / (n_eff - p)
    se = fsqrt(sigma2 * inv[1][1]) * abs(F(delta))
    return {
        "rows": n_eff,
        "times": times,
        "coefficient": float(beta[1]),
        "effect": float(F(delta) * beta[1]),
        "se": float(se),
    }


def series(n, x_off, y_off, z_off, z_walk, drop):
    z, zv = [], 0.0
    for t in range(n):
        zv = 0.9 * zv + z_walk * (u01() - 0.5)
        z.append(zv + z_off)
    x, y = [], []
    for t in range(n):
        zt1 = z[t - 1] - z_off if t >= 1 else 0.0
        zt2 = z[t - 2] - z_off if t >= 2 else 0.0
        x.append(x_off + 0.6 * zt1 + 0.3 * zt2 + (u01() - 0.5))
    for t in range(n):
        xt1 = x[t - 1] - x_off if t >= 1 else 0.0
        zt1 = z[t - 1] - z_off if t >= 1 else 0.0
        zt2 = z[t - 2] - z_off if t >= 2 else 0.0
        y.append(y_off + 1.7 * xt1 + 0.8 * zt1 - 0.5 * zt2 + 0.3 * (u01() - 0.5))
    missing = [t in drop for t in range(n)]
    return x, y, z, missing


def case(name, n, offs, z_walk, drop, lag_t, lag_zs, delta):
    x, y, z, missing = series(n, *offs, z_walk, drop)
    out = fit(x, y, z, missing, lag_t, lag_zs, delta)
    out.update(name=name, x=x, y=y, z=z, missing=missing, lag_t=lag_t, lag_zs=lag_zs, delta=delta)
    return out


cases = [
    case("plain", 36, (0.0, 0.0, 0.0), 1.0, set(), 1, [1, 2], 1.0),
    case("offsets", 36, (1.0e3, 1.0e6, 5.0e2), 1.0, set(), 1, [1, 2], 1.0),
    # z moves by 1e-3 per step so that its two lags are nearly collinear
    case("near_collinear", 36, (0.0, 0.0, 10.0), 1.0e-3, set(), 1, [1, 2], 1.0),
    case("gaps", 40, (0.0, 0.0, 0.0), 1.0, {7, 8, 21, 30}, 1, [1, 2], 1.0),
    case("tiny", 8, (0.0, 0.0, 0.0), 1.0, set(), 1, [1, 2], 1.0),
    case("delta_two", 36, (0.0, 0.0, 0.0), 1.0, set(), 1, [1, 2], 2.5),
]
print(json.dumps({"cases": cases}, indent=1))
