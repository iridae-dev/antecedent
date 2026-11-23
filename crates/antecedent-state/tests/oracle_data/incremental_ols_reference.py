# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reference values for the incremental least-squares summary, in exact rational arithmetic.

Regenerate with:  uv run --script incremental_ols_reference.py > incremental_ols_reference.json

For every design the rows are exact binary doubles (Fraction(float)).  The coefficients solve
X'X beta = X'y by Gauss-Jordan elimination over the rationals; the residual variance is
SSE / (n - p) with SSE = sum (y - x'beta)^2 for the OLS coefficients and for a second,
externally supplied coefficient vector.  Both are written as 40-digit decimals.

Designs: intercept and slope with a common offset of 0, 1e3, 1e6 and 1e8 on the regressor
(the Gram matrix of the raw rows has condition number growing as offset^2); a through-origin
design; three columns with a near-collinear pair; a single column; a large offset on the
response; and tiny n = p + 1.

Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F
from decimal import Decimal, getcontext

getcontext().prec = 60
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


def dec(q):
    return format(Decimal(q.numerator) / Decimal(q.denominator), ".40e")


def fit(rows, ys, extra=None):
    p = len(rows[0])
    r = [[F(v) for v in row] for row in rows]
    y = [F(v) for v in ys]
    xtx = [[sum(row[i] * row[j] for row in r) for j in range(p)] for i in range(p)]
    xty = [sum(row[i] * yy for row, yy in zip(r, y)) for i in range(p)]
    beta = solve(xtx, xty)

    def sigma2(b):
        sse = sum((yy - sum(bi * xi for bi, xi in zip(b, row))) ** 2 for row, yy in zip(r, y))
        return sse / (len(r) - p)

    out = {"beta": [dec(b) for b in beta], "sigma2": dec(sigma2(beta))}
    if extra is not None:
        out["extra_beta"] = extra
        out["extra_sigma2"] = dec(sigma2([F(e) for e in extra]))
    return out


cases = []


def add(name, rows, ys, extra=None):
    case = {"name": name, "rows": rows, "y": ys}
    case.update(fit(rows, ys, extra))
    cases.append(case)


def line(n, offset, spread=1.0, y_offset=0.0, noise=0.3):
    xs = [offset + spread * rng.uniform(-1, 1) for _ in range(n)]
    ys = [y_offset + 2.0 + 0.5 * (x - offset) + noise * rng.gauss(0, 1) for x in xs]
    return [[1.0, x] for x in xs], ys


for off in (0.0, 1e3, 1e6, 1e8):
    rows, ys = line(60, off)
    add(f"intercept_slope_offset_{off:g}", rows, ys, extra=[2.0, 0.5])
rows, ys = line(60, 1e3, y_offset=1e8)
add("intercept_slope_response_offset_1e8", rows, ys, extra=[1e8 + 2.0, 0.5])
xs = [rng.uniform(1, 5) for _ in range(40)]
add("through_origin", [[x] for x in xs], [3.0 * x + 0.2 * rng.gauss(0, 1) for x in xs], extra=[3.0])
rows = []
ys = []
for _ in range(50):
    a, b = rng.gauss(0, 1), rng.gauss(0, 1)
    rows.append([1.0, a, a + 1e-4 * b])
    ys.append(1.0 + a - b + 0.1 * rng.gauss(0, 1))
add("near_collinear_pair", rows, ys, extra=[1.0, 1.0, 0.0])
add("minimum_rows", [[1.0, 0.5], [1.0, 1.5], [1.0, 4.0]], [1.0, 3.0, 2.5], extra=[0.0, 1.0])
rows, ys = line(200, 0.0, spread=1e-3)
add("tiny_regressor_spread", rows, ys, extra=[2.0, 0.5])
print(json.dumps(cases, indent=1))
