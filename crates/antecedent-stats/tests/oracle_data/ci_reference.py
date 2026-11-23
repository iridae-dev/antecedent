# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""High-precision reference values for the partial-correlation and G-squared tests.

Regenerate with:  uv run --script ci_reference.py > ci_reference.json

Partial correlation: the columns are generated here with a fixed integer recurrence and
stored in the JSON, so the Rust test receives exactly the binary64 values used here.  The
reference residualizes X and Y on [1, Z] by solving the normal equations in 60-digit
arithmetic (not by any floating-point factorization), takes the Pearson correlation of the
residuals, and evaluates the two-sided Student-t p-value and the Fisher-z interval from their
definitions.  G-squared: contingency tables per conditioning stratum are given as integer
counts; the statistic 2 sum O ln(O/E), degrees of freedom (r-1)(c-1) summed over strata, and
the chi-square survival function come from the definitions.  Repository-authored
(MIT OR Apache-2.0); mpmath is used only to produce the frozen file.
"""
import json
import sys

import mpmath as mp

mp.mp.dps = 60

_state = 0x9E3779B97F4A7C15


def u01():
    """Uniform in [0, 1) on a 2^-53 grid from a fixed 64-bit LCG (exactly representable)."""
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def noise(n):
    return [2.0 * (u01() - 0.5) for _ in range(n)]


def parcorr_ref(cols, z_idx, level=0.95):
    x, y = cols[0], cols[1]
    n = len(x)
    k = len(z_idx)
    X = mp.matrix(n, k + 1)
    for i in range(n):
        X[i, 0] = 1
        for j, zi in enumerate(z_idx):
            X[i, j + 1] = mp.mpf(cols[zi][i])
    XtX = X.T * X

    def resid(v):
        vv = mp.matrix([mp.mpf(t) for t in v])
        beta = mp.lu_solve(XtX, X.T * vv)
        return vv - X * beta

    rx, ry = resid(x), resid(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    sxy = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sxx = sum((rx[i] - mx) ** 2 for i in range(n))
    syy = sum((ry[i] - my) ** 2 for i in range(n))
    r = sxy / mp.sqrt(sxx * syy)
    df = n - 2 - k
    t = abs(r) * mp.sqrt(df / (1 - r * r))
    xx = mp.mpf(df) / (df + t * t)
    p = mp.betainc(mp.mpf(df) / 2, mp.mpf(1) / 2, 0, xx, regularized=True)
    zc = mp.sqrt(2) * mp.erfinv(level)
    z = mp.atanh(r)
    se = 1 / mp.sqrt(df - 1)
    lo, hi = mp.tanh(z - zc * se), mp.tanh(z + zc * se)
    return float(r), float(p), float(lo), float(hi), df


def parcorr_case(name, cols, z_idx):
    r, p, lo, hi, df = parcorr_ref(cols, z_idx)
    return {"name": name, "columns": cols, "z": z_idx, "df": df, "r": r, "p": p, "ci": [lo, hi]}


cases = []
# 1. two well-conditioned conditioning columns
n = 60
z1, z2, e1, e2, e3 = (noise(n) for _ in range(5))
x = [0.7 * a - 0.4 * b + 0.8 * c for a, b, c in zip(z1, z2, e1)]
y = [0.5 * a + 0.9 * b + 0.6 * c for a, b, c in zip(z1, z2, e2)]
cases.append(parcorr_case("two_conditioners", [x, y, z1, z2], [2, 3]))
# 2. empty conditioning set
cases.append(parcorr_case("marginal", [x, y], []))
# 3. nearly collinear conditioners (correlation 1 - 5e-11)
n = 100
a, b, c, d = (noise(n) for _ in range(4))
z1 = a
z2 = [p + 1e-5 * q for p, q in zip(a, b)]
x = [p + 3.0 * q + 0.5 * r for p, q, r in zip(z1, z2, c)]
y = [p - 2.0 * q + 0.5 * r for p, q, r in zip(z1, z2, d)]
cases.append(parcorr_case("near_collinear_conditioners", [x, y, z1, z2], [2, 3]))
# 4. large common offset and mixed scales
n = 80
a, b, c, d = (noise(n) for _ in range(4))
z = [1.0e8 + 3.0 * v for v in a]
x = [1.0e6 + 2.0e3 * v + 50.0 * w for v, w in zip(a, b)]
y = [-5.0e4 + 7.0 * v + 0.25 * w for v, w in zip(a, c)]
cases.append(parcorr_case("offset_and_scale", [x, y, z], [2]))
# 5. strong dependence after conditioning: |r| near 1, p in the far Student-t tail
n = 50
a, b, c = (noise(n) for _ in range(3))
x = [v + 0.01 * w for v, w in zip(a, b)]
y = [-2.0 * v + 0.01 * w for v, w in zip(a, c)]
z = noise(n)
cases.append(parcorr_case("near_perfect_dependence", [x, y, z], [2]))
# 6. larger n, weak partial correlation
n = 3000
a, b, c, d = (noise(n) for _ in range(4))
x = [v + w for v, w in zip(a, b)]
y = [v + 0.12 * w + u for v, w, u in zip(a, b, c)]
cases.append(parcorr_case("large_n_weak", [x, y, a], [2]))
# 7. few residual degrees of freedom
n = 7
a, b, c, d = (noise(n) for _ in range(4))
cases.append(parcorr_case("small_df", [[p + q for p, q in zip(a, b)], [p - q for p, q in zip(a, c)], d, b], [2, 3]))


def g2_ref(tables):
    g_tot = mp.mpf(0)
    df_tot = 0
    for t in tables:
        rows = [sum(r) for r in t]
        cols = [sum(t[i][j] for i in range(len(t))) for j in range(len(t[0]))]
        tot = sum(rows)
        nr = sum(1 for v in rows if v > 0)
        nc = sum(1 for v in cols if v > 0)
        for i in range(len(t)):
            for j in range(len(t[0])):
                o = t[i][j]
                if o > 0:
                    e = mp.mpf(rows[i]) * cols[j] / tot
                    g_tot += 2 * o * mp.log(o / e)
        df_tot += (nr - 1) * (nc - 1)
    p = mp.gammainc(mp.mpf(df_tot) / 2, g_tot / 2, mp.inf, regularized=True)
    return float(g_tot), df_tot, float(p)


g2_inputs = {
    "two_by_two": [[[30, 20], [15, 35]]],
    "three_by_four": [[[22, 18, 25, 15], [30, 12, 18, 20], [14, 26, 16, 24]]],
    "two_strata": [[[40, 10], [12, 38]], [[25, 25], [30, 20]]],
    "stratum_with_constant_y": [[[40, 10], [12, 38]], [[60, 0], [40, 0]]],
    "strong_dependence_tail": [[[500, 10], [10, 500]]],
    "large_counts_weak": [[[5100, 4900], [4850, 5150]]],
    "large_table_df56": [[[40 + (3 * i + 5 * j) % 11 for j in range(9)] for i in range(8)]],
    "five_strata_three_by_three": [
        [[20 + (i * j + s) % 7 + 4 * (i == j) for j in range(3)] for i in range(3)] for s in range(5)
    ],
}
g2 = []
for name, tables in g2_inputs.items():
    g, df, p = g2_ref(tables)
    g2.append({"name": name, "tables": tables, "g": g, "df": df, "p": p})

json.dump({"parcorr": cases, "gsquared": g2}, sys.stdout, allow_nan=False)
print()
