# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Randomized ANCOVA: HC0 coefficient variance, HC2 variance and Bell-McCaffrey degrees of freedom.

Regenerate with:  uv run --script ancova_reference.py > ancova_reference.json

Every table holds binary64-exact values (integers and quarter multiples), so the design is a
rational matrix and the whole computation is exact rational arithmetic (`fractions.Fraction`):
beta = (X'X)^-1 X'y, residuals e = y - X beta, bread B = (X'X)^-1, w = X B e_T, leverages
h_i = x_i' B x_i.  HC0 = sum_i (w_i e_i)^2.  HC2 = sum_i (w_i e_i)^2 / (1 - h_i).  The
Bell-McCaffrey degrees of freedom are written from the definition rather than a trace identity:
the n-by-n matrix A = M D M with M = I - H and D = diag(w_i^2 / (1 - h_i)) is formed
explicitly and df = (tr A)^2 / tr(A^2).  The 97.5% Student quantile at that df comes from
bisection of the incomplete-beta survival function at 50 digits.  A design with a row of
leverage 1 is recorded as `undefined`.  Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F
from mpmath import mp, mpf, betainc

mp.dps = 50


def solve(a, b):
    """Gauss-Jordan on an exact square system; returns the solution or the inverse for matrix b."""
    n = len(a)
    m = [row[:] + brow[:] for row, brow in zip(a, b)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        m[c] = [v / d for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [row[n:] for row in m]


def t_quantile_975(df):
    df = mpf(df.numerator) / mpf(df.denominator)

    def sf(t):  # P(T > t), t >= 0
        return betainc(df / 2, mpf(1) / 2, 0, df / (df + t * t), regularized=True) / 2

    lo, hi = mpf(0), mpf(1)
    while sf(hi) > mpf("0.025"):
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if sf(mid) > mpf("0.025"):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit(y, t, covs):
    n = len(y)
    q = 2 + len(covs)
    x = [[F(1), F(1 if t[i] else 0)] + [F(c[i]) for c in covs] for i in range(n)]
    xtx = [[sum(x[i][a] * x[i][b] for i in range(n)) for b in range(q)] for a in range(q)]
    eye = [[F(int(a == b)) for b in range(q)] for a in range(q)]
    bread = solve(xtx, eye)
    xty = [sum(x[i][a] * F(y[i]) for i in range(n)) for a in range(q)]
    beta = [sum(bread[a][b] * xty[b] for b in range(q)) for a in range(q)]
    e = [F(y[i]) - sum(x[i][a] * beta[a] for a in range(q)) for i in range(n)]
    w = [sum(x[i][a] * bread[a][1] for a in range(q)) for i in range(n)]
    h = [sum(x[i][a] * bread[a][b] * x[i][b] for a in range(q) for b in range(q)) for i in range(n)]
    hc0 = sum((w[i] * e[i]) ** 2 for i in range(n))
    out = {"effect": beta[1], "hc0": hc0, "covariate_coefficients": beta[2:]}
    if any(v == 1 for v in h):
        out["undefined"] = True
        return out
    d = [w[i] ** 2 / (1 - h[i]) for i in range(n)]
    out["hc2"] = sum(d[i] * e[i] ** 2 for i in range(n))
    # M = I - H with H = X B X'
    m = [[F(int(i == j)) - sum(x[i][a] * bread[a][b] * x[j][b] for a in range(q) for b in range(q))
          for j in range(n)] for i in range(n)]
    dm = [[d[i] * m[i][j] for j in range(n)] for i in range(n)]
    amat = [[sum(m[i][k] * dm[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    tr = sum(amat[i][i] for i in range(n))
    tr2 = sum(amat[i][j] * amat[j][i] for i in range(n) for j in range(n))
    out["df"] = tr * tr / tr2
    out["t"] = t_quantile_975(out["df"])
    return out


def f64(v):
    if isinstance(v, F):
        return float(v)
    return float(mpf(v))


cases = []


def add(name, y, t, covs):
    r = fit(y, t, covs)
    rec = {"name": name, "y": [float(v) for v in y], "t": [bool(v) for v in t],
           "x": [[float(v) for v in c] for c in covs], "undefined": bool(r.get("undefined", False)),
           "effect": f64(r["effect"]), "hc0": f64(r["hc0"]),
           "covariate_coefficients": [f64(v) for v in r["covariate_coefficients"]]}
    if not rec["undefined"]:
        hw = r["t"] * mp.sqrt(mpf(r["hc2"].numerator) / mpf(r["hc2"].denominator))
        rec.update({"hc2": f64(r["hc2"]), "df": f64(r["df"]), "t975": f64(r["t"]),
                    "half_width": f64(hw)})
    cases.append(rec)


# two-group regression is Welch (no covariate): 3 treated, 5 control
add("difference_in_means_3_v_5", [1, 2, 6, 0, 1, 1, 2, 6], [1, 1, 1, 0, 0, 0, 0, 0], [])
# one covariate, balanced 6/6
x1 = [F(v, 4) for v in [3, 9, -2, 7, 5, 1, -6, 8, 2, 4, 10, -1]]
y1 = [F(v, 4) for v in [11, 25, 3, 19, 22, 9, 4, 30, 13, 17, 31, 6]]
t1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
add("one_covariate_6_v_6", y1, t1, [x1])
# two covariates, 2 treated vs 14 control (small arm)
n = 16
xa = [F((7 * i * i + 3 * i) % 23 - 11, 2) for i in range(n)]
xb = [F((5 * i + 2 * i * i) % 17 - 8, 2) for i in range(n)]
ya = [F((13 * i * i + 5 * i) % 29, 3) + 2 * (i < 2) for i in range(n)]
ya = [F(v) for v in ya]
add("two_covariates_2_v_14", ya, [1, 1] + [0] * 14, [xa, xb])
# a high leverage covariate value
xh = [F(v) for v in [0, 1, 2, -1, 1, 0, 2, -2, 1, 0, 40, 1]]
yh = [F(v, 2) for v in [5, 9, 3, 12, 7, 1, 10, 6, 8, 2, 70, 4]]
add("high_leverage_row", yh, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [xh])
# outcome offset 1e7 and covariate offset 1e3 (scale 250)
xo = [1000 + 250 * v for v in x1]
yo = [10_000_000 + v for v in y1]
add("outcome_offset_1e7_covariate_offset_1e3", yo, t1, [xo])
# near-collinear covariates
xc = [v + F(1, 1000) * F((3 * i * i) % 7 - 3) for i, v in enumerate(xa)]
add("near_collinear_covariates", ya, [1, 1] + [0] * 14, [xa, xc])
# a covariate that is nonzero in one row only: that row has leverage 1
xd = [F(0)] * 11 + [F(5)]
add("row_of_leverage_one", y1, t1, [xd])
print(json.dumps({"cases": cases}))
