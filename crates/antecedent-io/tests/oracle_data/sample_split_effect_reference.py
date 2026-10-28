# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Ordinary least squares coefficient, standard error and Student interval, in exact arithmetic.

Regenerate with:  uv run --script sample_split_effect_reference.py > sample_split_effect_reference.json

For a design [1, x, z_1 .. z_m] and outcome y (the binary64 values the product is given, taken as
exact rationals), the script solves the normal equations over the rationals, so the coefficient
of x, the residual sum of squares, the standard error sqrt(rss / (n - k) * [(X'X)^-1]_xx) and the
residual degrees of freedom n - k carry no rounding.  The two-sided Student interval uses the
quantile of the t distribution at 60 digits (root of the regularized incomplete beta function),
for several levels.  A design whose X'X is singular in exact arithmetic is recorded as such.
Cases include large offsets on every column, a covariate that equals the treatment up to noise
of order 1e-4 and 1e-6, the smallest admissible sample (n = k + 2), five covariates, and an exact
collinearity.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

from mpmath import mp, mpf, betainc, findroot, sqrt as msqrt

mp.dps = 60


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [v - f * w for v, w in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def inverse(A):
    n = len(A)
    cols = []
    for j in range(n):
        e = [F(int(i == j)) for i in range(n)]
        s = solve(A, e)
        if s is None:
            return None
        cols.append(s)
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def t_cdf(t, df):
    t = mpf(t)
    df = mpf(df)
    x = df / (df + t * t)
    tail = betainc(df / 2, mpf(1) / 2, 0, x, regularized=True) / 2
    return 1 - tail if t > 0 else tail


def t_ppf(p, df):
    guess = mpf(1.5)
    return findroot(lambda t: t_cdf(t, df) - mpf(p), guess, tol=mpf(10) ** -50, maxsteps=200)


LEVELS = [0.5, 0.9, 0.95, 0.99, 0.999]


def ols(y, x, zs):
    n = len(y)
    cols = [[F(1)] * n, [F(v) for v in x]] + [[F(v) for v in z] for z in zs]
    k = len(cols)
    G = [[sum(a * b for a, b in zip(cols[i], cols[j])) for j in range(k)] for i in range(k)]
    mom = [sum(a * F(b) for a, b in zip(cols[i], y)) for i in range(k)]
    inv = inverse(G)
    if inv is None:
        return {"singular": True}
    beta = [sum(inv[i][j] * mom[j] for j in range(k)) for i in range(k)]
    rss = sum((F(y[r]) - sum(beta[c] * cols[c][r] for c in range(k))) ** 2 for r in range(n))
    df = n - k
    var = rss / df * inv[1][1]
    se = msqrt(mpf(var.numerator) / mpf(var.denominator))
    out = {"singular": False, "point": float(beta[1]), "se": float(se), "df": df, "rows": n, "intervals": []}
    for lvl in LEVELS:
        q = t_ppf(1 - (1 - mpf(lvl)) / 2, df)
        pt = mpf(beta[1].numerator) / mpf(beta[1].denominator)
        out["intervals"].append({"level": lvl, "lower": float(pt - q * se), "upper": float(pt + q * se), "quantile": float(q)})
    return out


def make(seed, n, m, y0=0.0, x0=0.0, z0=0.0, collinear_noise=None, scale=1.0):
    r = random.Random(seed)
    x = [x0 + r.gauss(0, 1) * scale for _ in range(n)]
    zs = []
    for j in range(m):
        if collinear_noise is not None and j == 0:
            zs.append([v + z0 + collinear_noise * r.gauss(0, 1) for v in x])
        else:
            zs.append([z0 + r.gauss(0, 1) * scale + 0.5 * (x[i] - x0) for i in range(n)])
    y = [y0 + 0.7 * (x[i] - x0) + sum(0.3 * (zs[j][i] - z0) for j in range(m)) + r.gauss(0, 1) for i in range(n)]
    return y, x, zs


def case(name, **kw):
    y, x, zs = make(**kw)
    res = ols(y, x, zs)
    return {"name": name, "y": y, "x": x, "z": zs, **res}


cases = [
    case("plain", seed=1, n=60, m=2),
    case("offsets_1e6", seed=2, n=80, m=2, y0=1e6, x0=1e5, z0=-1e6),
    case("offsets_1e9_covariate", seed=3, n=80, m=1, y0=3e8, x0=2e4, z0=1e9),
    case("collinear_1e-4", seed=4, n=100, m=2, collinear_noise=1e-4),
    case("collinear_1e-6_with_offset", seed=5, n=100, m=1, collinear_noise=1e-6, x0=1e3, z0=0.0),
    case("smallest_sample", seed=6, n=6, m=2),
    case("five_covariates", seed=7, n=40, m=5, y0=50.0, x0=-30.0),
    case("tiny_scale_columns", seed=8, n=50, m=2, scale=1e-6, y0=0.0),
]
# exact collinearity: the covariate equals the treatment
y, x, _ = make(seed=9, n=30, m=1)
cases.append({"name": "exact_collinearity", "y": y, "x": x, "z": [list(x)], **ols(y, x, [list(x)])})
print(json.dumps(cases))
