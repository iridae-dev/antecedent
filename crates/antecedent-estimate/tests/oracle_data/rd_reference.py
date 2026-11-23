# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Sharp regression-discontinuity jump and its standard errors, in exact rational arithmetic.

Regenerate with:  uv run --script rd_reference.py > rd_reference.json

The uniform-kernel local-linear jump is computed in the form the literature states it: a
separate straight-line least-squares fit on each side of the cutoff over the rows with
|R - c| <= h, the jump being the difference of the two fitted values at c.  The product fits one
pooled regression on [1, T, R - c, T (R - c)]; the two coincide because that design is a
saturated interaction.  The variances are the pooled-regression sandwiches written out
directly from their definitions (HC0 = B X' diag(e^2) X B, HC1 = n/(n-4) HC0, HC2 and HC3 with
e^2 / (1 - h) and e^2 / (1 - h)^2, classical = rss/(n-4) B, with B = (X'X)^-1), all in exact
rationals; because the sides are separate blocks the same HC0 also equals the sum of the two
one-sided sandwiches, which the script asserts.  Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as Fr

_state = 0x9E3779B97F4A7C15


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def inv(a):
    n = len(a)
    m = [row[:] + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        d = m[c][c]
        m[c] = [x / d for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def mat(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def line(xs, ys):
    """Intercept and slope of the exact least-squares line."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    return (sxx * sy - sx * sxy) / den, (n * sxy - sx * sy) / den


def analyse(r, y, cutoff, h):
    c = Fr(cutoff)
    rows = [(Fr(ri) - c, Fr(yi)) for ri, yi in zip(r, y) if abs(Fr(ri) - c) <= Fr(h)]
    left = [(x, v) for x, v in rows if x < 0]
    right = [(x, v) for x, v in rows if x >= 0]
    a0, _ = line([x for x, _ in left], [v for _, v in left])
    a1, _ = line([x for x, _ in right], [v for _, v in right])
    jump = a1 - a0
    n, p = len(rows), 4
    X = [[Fr(1), Fr(int(x >= 0)), x, x * Fr(int(x >= 0))] for x, _ in rows]
    Y = [v for _, v in rows]
    B = inv(mat([list(col) for col in zip(*X)], X))
    beta = [sum(B[i][j] * sum(X[k][j] * Y[k] for k in range(n)) for j in range(p)) for i in range(p)]
    assert beta[1] == jump
    e = [Y[k] - sum(beta[j] * X[k][j] for j in range(p)) for k in range(n)]
    lev = [sum(X[k][i] * B[i][j] * X[k][j] for i in range(p) for j in range(p)) for k in range(n)]
    out = {"cutoff": float(cutoff), "bandwidth": float(h), "r": r, "y": y, "n_window": n,
           "jump": float(jump), "leverage_one": any(l == 1 for l in lev)}

    def var(weights):
        meat = [[sum(weights[k] * X[k][i] * X[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
        s = mat(mat(B, meat), B)
        return s[1][1]

    if n > p:
        hc0 = var([ek * ek for ek in e])
        out["var_hc0"] = float(hc0)
        out["var_hc1"] = float(hc0 * Fr(n, n - p))
        out["var_homoskedastic"] = float(sum(ek * ek for ek in e) / (n - p) * B[1][1])
        if not out["leverage_one"]:
            out["var_hc2"] = float(var([e[k] ** 2 / (1 - lev[k]) for k in range(n)]))
            out["var_hc3"] = float(var([e[k] ** 2 / (1 - lev[k]) ** 2 for k in range(n)]))
        # block structure: HC0 is the sum of the two one-sided sandwiches
        def side_var(pts):
            xs = [[Fr(1), x] for x, _ in pts]
            bi = inv(mat([list(c2) for c2 in zip(*xs)], xs))
            ys = [v for _, v in pts]
            co = [sum(bi[i][j] * sum(xs[k][j] * ys[k] for k in range(len(pts))) for j in range(2)) for i in range(2)]
            es = [ys[k] - co[0] - co[1] * pts[k][0] for k in range(len(pts))]
            mt = [[sum(es[k] ** 2 * xs[k][i] * xs[k][j] for k in range(len(pts))) for j in range(2)] for i in range(2)]
            return mat(mat(bi, mt), bi)[0][0]
        assert hc0 == side_var(left) + side_var(right)
    return out


def grid_case(name, cutoff, h, step, nleft, nright, noise, outside=(), hetero=False, seed_shift=0):
    """Rows on a dyadic grid around the cutoff, including the cutoff and +-h exactly."""
    for _ in range(seed_shift):
        u01()
    c = Fr(cutoff)
    r = []
    for k in range(-nleft, nright + 1):
        r.append(c + k * Fr(step))
    r = [float(x) for x in r]
    y = []
    for ri in r:
        x = ri - cutoff
        sd = noise * (1 + (3 * abs(x) / h if hetero else 0))
        y.append(round((1.0 + 0.5 * x + (2.0 + 1.5 * x if x >= 0 else 0.0) + (u01() - 0.5) * sd) * 4096) / 4096)
    for extra_r, extra_y in outside:
        r.append(float(cutoff + extra_r))
        y.append(extra_y)
    case = analyse(r, y, cutoff, h)
    case["name"] = name
    return case


def main():
    cases = [
        grid_case("symmetric_window_with_edge_rows", 0.0, 0.5, Fr(1, 16), 8, 8, 0.5, outside=[(-0.75, 40.0), (0.8125, -30.0), (1.5, 1e3)]),
        grid_case("heteroskedastic_unequal_sides", 0.25, 0.375, Fr(1, 64), 24, 9, 0.8, outside=[(-0.5, 7.0), (0.5, -9.0)], hetero=True),
        grid_case("large_cutoff_offset", 1048576.0, 0.25, Fr(1, 64), 16, 16, 0.6, outside=[(-1.0, 50.0), (1.0, 60.0)]),
        grid_case("small_window_three_residual_degrees", 0.0, 0.1875, Fr(1, 16), 3, 3, 0.4),
        grid_case("two_per_side_saturated", 0.0, 0.125, Fr(1, 16), 2, 1, 0.4),
        grid_case("negative_cutoff_wide", -3.0, 2.0, Fr(1, 8), 16, 16, 1.2, hetero=True),
    ]
    print(json.dumps({"cases": cases}))


main()
