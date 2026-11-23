# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact references for two-stage least squares, its first-stage test and the Anderson-Rubin set.

Regenerate with:  uv run --script twosls_reference.py > twosls_reference.json

Designs, instruments and outcomes are exact binary64 values.  Everything below is computed in
`fractions.Fraction` arithmetic (no rounding) and rounded once, from the defining formulas:

* first stage   T = W pi + v, W = [Z | X],  pi = (W'W)^-1 W'T,  That = W pi
* second stage  y on [That | X]: delta = (Xh'Xh)^-1 Xh'y with Xh = [That | X]
* structural RSS  sum (y - delta_0 T - X delta_1)^2   (actual T, not That)
* first-stage F  ((RSS_X - RSS_W)/k) / (RSS_W/df), RSS_X from T on X alone, df = n - k - p_x,
  and partial R^2 = (RSS_X - RSS_W)/RSS_X
* Anderson-Rubin statistic at beta0: k F for excluded instruments in (y - beta0 T) ~ [Z | X]
* Anderson-Rubin set {beta : AR(beta) <= k F_{k,df,level}}: after partialling X out of y, T, Z the
  statistic is df N(b)/(k D(b)) with N = |P y~|^2, D = |M y~|^2, y~ = ys - b ts, so the set is
  A b^2 + B b + C <= 0 with A = df N_tt - c D_tt, B = -2 df N_yt + 2 c D_yt, C = df N_yy - c D_yy
  and c = k F_crit.  The quadratic is classified and its roots found at 50 digits.
* F_crit is the `level` quantile of F(k, df): the root of the regularised incomplete beta
  I_x(k/2, df/2) = level at x = k f/(k f + df), found to 45 digits by bisection in mpmath.

This script and its frozen output are repository-authored (MIT OR Apache-2.0); mpmath is used
only here, never at product run time.
"""
import json
import sys
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.s >> 11) / float(1 << 53)

    def sym(self, scale=1.0):
        return round((self.u() * 2 - 1) * 1e6) / 1e6 * scale


def inv(a):
    n = len(a)
    m = [[F(x) for x in row] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        m[c] = [x / d for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def ols(cols, y):
    """Exact OLS: coefficients and residuals (cols are lists of numbers)."""
    p, n = len(cols), len(y)
    g = [[sum(F(cols[a][i]) * F(cols[b][i]) for i in range(n)) for b in range(p)] for a in range(p)]
    r = [sum(F(cols[a][i]) * F(y[i]) for i in range(n)) for a in range(p)]
    gi = inv(g)
    beta = [sum(gi[a][b] * r[b] for b in range(p)) for a in range(p)]
    res = [F(y[i]) - sum(beta[a] * F(cols[a][i]) for a in range(p)) for i in range(n)]
    return beta, res


def rss(res):
    return sum(v * v for v in res)


def f_crit(level, k, df):
    a, b = mp.mpf(k) / 2, mp.mpf(df) / 2
    lo, hi = mp.mpf(0), mp.mpf(1)
    for _ in range(170):
        mid = (lo + hi) / 2
        if mp.betainc(a, b, 0, mid, regularized=True) < level:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2
    return x * df / (k * (1 - x))


def mpf(fr):
    return mp.mpf(fr.numerator) / fr.denominator


def case(name, n, k, exog, strength, seed, level=0.95):
    rng = Lcg(seed)
    one = [1.0] * n
    z = [[rng.sym(2.0) for _ in range(n)] for _ in range(k)]
    x_extra = []
    if exog == "covariate":
        x_extra = [[rng.sym(1.0) for _ in range(n)]]
    elif exog == "offset":
        x_extra = [[1e5 + rng.sym(3.0) for _ in range(n)]]
    exogenous = [one] + x_extra
    u = [rng.sym(1.0) for _ in range(n)]
    t = [sum(strength * zc[i] for zc in z) / k + 0.5 * u[i] + rng.sym(0.5) for i in range(n)]
    y = [1.0 + 2.0 * t[i] + (0.3 * x_extra[0][i] if x_extra else 0.0) + u[i] for i in range(n)]
    px = len(exogenous)
    df = n - k - px

    # first stage on W = [Z | X]
    W = z + exogenous
    pi, v = ols(W, t)
    that = [F(t[i]) - v[i] for i in range(n)]
    rss_full = rss(v)
    _, vr = ols(exogenous, t)
    rss_restricted = rss(vr)
    fstat = ((rss_restricted - rss_full) / k) / (rss_full / df)
    pr2 = (rss_restricted - rss_full) / rss_restricted

    # second stage on [That | X]
    delta, _ = ols([that] + exogenous, y)
    struct = [F(y[i]) - delta[0] * F(t[i]) - sum(delta[1 + j] * F(exogenous[j][i]) for j in range(px))
              for i in range(n)]

    def ar_stat(beta0):
        ys = [F(y[i]) - F(beta0) * F(t[i]) for i in range(n)]
        _, full = ols(W, ys)
        _, restr = ols(exogenous, ys)
        k_f = ((rss(restr) - rss(full)) / k) / (rss(full) / df)
        return k * k_f

    # AR set: partial X out of y, t and each z, then the two quadratic forms
    ys, ts = ols(exogenous, y)[1], ols(exogenous, t)[1]
    zs = [ols(exogenous, zc)[1] for zc in z]
    _, ry = ols(zs, ys)
    _, rt = ols(zs, ts)
    dot = lambda a, b: sum(p * q for p, q in zip(a, b))
    yy, tt, yt = dot(ys, ys), dot(ts, ts), dot(ys, ts)
    d_yy, d_tt, d_yt = dot(ry, ry), dot(rt, rt), dot(ry, rt)
    n_yy, n_tt, n_yt = yy - d_yy, tt - d_tt, yt - d_yt
    fc = f_crit(level, k, df)
    c = k * fc
    A = df * mpf(n_tt) - c * mpf(d_tt)
    B = -2 * df * mpf(n_yt) + 2 * c * mpf(d_yt)
    C = df * mpf(n_yy) - c * mpf(d_yy)
    disc = B * B - 4 * A * C
    if A > 0 and disc >= 0:
        r1, r2 = sorted([(-B - mp.sqrt(disc)) / (2 * A), (-B + mp.sqrt(disc)) / (2 * A)])
        kind, lo, hi = "interval", float(r1), float(r2)
    elif A > 0:
        kind, lo, hi = "empty", None, None
    elif A < 0 and disc >= 0:
        kind, lo, hi = "union", None, None
    elif A < 0:
        kind, lo, hi = "whole_line", None, None
    else:
        kind, lo, hi = "degenerate", None, None
    fl = float
    # moments for the normal-equation entry point: [X | Z | T] Gram, cross with y, y'y
    cols = exogenous + z + [t]
    gram = [fl(sum(F(a[i]) * F(b[i]) for i in range(n))) for a in cols for b in cols]
    cross = [fl(sum(F(a[i]) * F(y[i]) for i in range(n))) for a in cols]
    return {"name": name, "n": n, "k": k, "x_ncols": px, "z": [v for zc in z for v in zc],
            "x": [v for xc in exogenous for v in xc], "t": t, "y": y,
            "coef": [fl(d) for d in delta], "structural_rss": fl(rss(struct)),
            "f_statistic": fl(fstat), "partial_r2": fl(pr2), "df2": df,
            "ar_beta0": 2.0, "ar_at_beta0": fl(ar_stat(2.0)), "ar_at_zero": fl(ar_stat(0.0)),
            "f_critical": float(fc), "level": level,
            "ar_kind": kind, "ar_lo": lo, "ar_hi": hi,
            "gram": gram, "cross": cross, "yy": fl(sum(F(v) * F(v) for v in y))}


if __name__ == "__main__":
    out = [
        case("single_instrument", 40, 1, "intercept", 1.0, 11),
        case("single_instrument_covariate", 40, 1, "covariate", 1.0, 12),
        case("two_instruments", 50, 2, "covariate", 1.0, 13),
        case("three_instruments_offset", 60, 3, "offset", 1.0, 14),
        case("weak_instrument", 40, 1, "covariate", 0.05, 15),
        case("weak_two_instruments", 40, 2, "covariate", 0.03, 16),
    ]
    json.dump(out, sys.stdout, allow_nan=False)
