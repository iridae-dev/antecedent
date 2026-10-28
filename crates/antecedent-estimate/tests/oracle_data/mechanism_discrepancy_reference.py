# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the two-sample mechanism discrepancy diagnostic.

Regenerate with:  uv run --script mechanism_discrepancy_reference.py > mechanism_discrepancy_reference.json

Written from the stated construction.  For each population the design is (1, parents...) and
  beta = (X'X)^-1 X'y,  RSS = y'y - beta'X'y,  sigma2 = RSS / (n - p),  V = sigma2 (X'X)^-1,
all in exact rational arithmetic.  With d = beta_t - beta_s over the compared coefficients and
V_d = V_s + V_t restricted to them:
  Wald W = d' V_d^-1 d, referred to chi-square with len(d) degrees of freedom (50 digits);
  per coefficient se = sqrt(V_d,jj), z = d_j / se, p = 2 (1 - Phi(|z|)), Holm-adjusted over the compared set
  (sort ascending, adjusted_(i) = min(1, max_{j<=i} (m - j + 1) p_(j))), and
  minimal detectable difference (z_{1-alpha/2} + z_{power}) se.
Observations are multiples of 1/8 below 64 in magnitude, so every sum is exact in binary doubles and the
Rust summaries equal the rational ones.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50
rng = random.Random(20261028)


def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        inv = 1 / m[c][c]
        m[c] = [v * inv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] for i in range(n)]


def inverse(a):
    n = len(a)
    cols = [solve(a, [F(int(i == j)) for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def fit(x, y):
    n, k = len(y), len(x[0])
    rows = [[F(1)] + [F(v) for v in r] for r in x]
    p = k + 1
    xtx = [[sum(r[i] * r[j] for r in rows) for j in range(p)] for i in range(p)]
    xty = [sum(r[i] * F(yy) for r, yy in zip(rows, y)) for i in range(p)]
    beta = solve(xtx, xty)
    rss = sum(F(yy) ** 2 for yy in y) - sum(b * c for b, c in zip(beta, xty))
    sigma2 = rss / (n - p)
    inv = inverse(xtx)
    cov = [[sigma2 * v for v in row] for row in inv]
    return beta, cov, sigma2, n - p


def mp_(q):
    return mp.mpf(q.numerator) / q.denominator


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [None] * m
    running = mp.mpf(0)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(mp.mpf(1), running)
    return adj


def quantile_up(tail):
    return mp.sqrt(2) * mp.erfinv(1 - 2 * mp.mpf(tail))


def grid(n, scale=64):
    return round(rng.uniform(-scale / 8, scale / 8) * 8) / 8


cases = []
for i in range(30):
    k = rng.randint(1, 3)
    ns, nt = rng.randint(k + 4, k + 30), rng.randint(k + 4, k + 30)
    shift = [grid(1, 4) * rng.choice([0, 0, 1]) for _ in range(k + 1)]
    truth = [grid(1, 8) for _ in range(k + 1)]

    def draw(n, coefs):
        xs = [[round(rng.gauss(0, 2) * 8) / 8 for _ in range(k)] for _ in range(n)]
        ys = [round((coefs[0] + sum(c * v for c, v in zip(coefs[1:], x)) + rng.gauss(0, 1.5)) * 8) / 8 for x in xs]
        return xs, ys

    xs_s, ys_s = draw(ns, truth)
    xs_t, ys_t = draw(nt, [t + s for t, s in zip(truth, shift)])
    alpha = rng.choice([0.05, 0.01, 0.2])
    power = rng.choice([0.8, 0.9, 0.5])
    compare_intercept = rng.choice([True, False])
    bs, vs, _, dfs = fit(xs_s, ys_s)
    bt, vt, _, dft = fit(xs_t, ys_t)
    idx = list(range(0 if compare_intercept else 1, k + 1))
    d = [bt[j] - bs[j] for j in idx]
    vd = [[vs[a][b] + vt[a][b] for b in idx] for a in idx]
    w = sum(a * b for a, b in zip(d, solve(vd, d)))
    wald_p = mp.gammainc(mp.mpf(len(idx)) / 2, mp_(w) / 2, mp.inf, regularized=True)
    zs, ps, ses = [], [], []
    for pos, j in enumerate(idx):
        se = mp.sqrt(mp_(vd[pos][pos]))
        z = mp_(d[pos]) / se
        zs.append(z)
        ses.append(se)
        ps.append(mp.erfc(abs(z) / mp.sqrt(2)))
    adj = holm(ps)
    factor = quantile_up(alpha / 2) + quantile_up(1 - power)
    cases.append({
        "x_source": xs_s, "y_source": ys_s, "x_target": xs_t, "y_target": ys_t,
        "alpha": alpha, "power": power, "compare_intercept": compare_intercept, "k": k,
        "beta_source": [mp.nstr(mp_(b), 25) for b in bs], "beta_target": [mp.nstr(mp_(b), 25) for b in bt],
        "wald": mp.nstr(mp_(w), 25), "df": len(idx), "wald_p": mp.nstr(wald_p, 25),
        "coefficients": [{"difference": mp.nstr(mp_(d[p]), 25), "se": mp.nstr(ses[p], 25), "z": mp.nstr(zs[p], 25),
                          "p": mp.nstr(ps[p], 25), "p_holm": mp.nstr(adj[p], 25), "mdd": mp.nstr(factor * ses[p], 25)}
                         for p in range(len(idx))],
        "factor": mp.nstr(factor, 25),
    })

# response offsets: the summaries (y'y, X'y) carry the offset, so RSS = y'y - beta'X'y cancels
offset_cases = []
for offset in (0.0, 1e3, 1e5, 1e6, 3e6):
    k = 1
    n = 12

    def draw_off(coefs):
        xs = [[round(rng.gauss(0, 2) * 8) / 8] for _ in range(n)]
        ys = [round((offset + coefs[0] + coefs[1] * x[0] + rng.gauss(0, 1.5)) * 8) / 8 for x in xs]
        return xs, ys

    xs_s, ys_s = draw_off([0.5, 1.0])
    xs_t, ys_t = draw_off([0.5, 1.5])
    bs, vs, _, _ = fit(xs_s, ys_s)
    bt, vt, _, _ = fit(xs_t, ys_t)
    idx = [0, 1]
    d = [bt[j] - bs[j] for j in idx]
    vd = [[vs[a][b] + vt[a][b] for b in idx] for a in idx]
    w = sum(a * b for a, b in zip(d, solve(vd, d)))
    offset_cases.append({"offset": offset, "x_source": xs_s, "y_source": ys_s, "x_target": xs_t, "y_target": ys_t,
                         "wald": mp.nstr(mp_(w), 25), "beta_source": [mp.nstr(mp_(b), 25) for b in bs],
                         "se_intercept": mp.nstr(mp.sqrt(mp_(vd[0][0])), 25), "se_slope": mp.nstr(mp.sqrt(mp_(vd[1][1])), 25)})
print(json.dumps({"cases": cases, "offset_cases": offset_cases}, indent=1))
