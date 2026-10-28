# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Block, weighted and normal-score partial correlation tests in multiprecision arithmetic.

Regenerate with:  uv run --script parcorr_variants_reference.py > parcorr_variants_reference.json

All quantities are computed in 60-digit arithmetic from the stored binary64 columns:

* block test: both blocks are residualized on an intercept and Z by least squares; Wilks'
  Lambda is det(S) / (det(Sxx) det(Syy)) of the residual scatter; the leading canonical
  correlation is the square root of the largest eigenvalue of Sxx^-1 Sxy Syy^-1 Syx; the
  p-value is Bartlett's chi-square, -[(n - |Z| - 1) - (px + py + 1)/2] ln Lambda on px py
  degrees of freedom;
* weighted: weighted least-squares residuals, weighted Pearson correlation of the residuals and a
  Student-t p-value on the Kish effective sample size minus 2 minus |Z| degrees of freedom;
* robust: van der Waerden scores Phi^-1(midrank / (n + 1)), then the unweighted partial correlation
  with n - 2 - |Z| degrees of freedom.

Repository-authored (MIT OR Apache-2.0).
"""
import json
import math

import mpmath as mp

mp.mp.dps = 60
_state = 0x9E3779B97F4A7C15


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def normal():
    return math.sqrt(-2 * math.log(u01() + 2.0 ** -60)) * math.cos(2 * math.pi * u01())


def residuals(target, zs, w):
    n = len(target)
    cols = [[mp.mpf(1)] * n] + [[mp.mpf(v) for v in z] for z in zs]
    W = [mp.mpf(v) for v in w]
    p = len(cols)
    A = mp.matrix(p, p)
    b = mp.matrix(p, 1)
    for i in range(p):
        for j in range(p):
            A[i, j] = sum(W[r] * cols[i][r] * cols[j][r] for r in range(n))
        b[i] = sum(W[r] * cols[i][r] * mp.mpf(target[r]) for r in range(n))
    beta = mp.lu_solve(A, b)
    return [mp.mpf(target[r]) - sum(beta[i] * cols[i][r] for i in range(p)) for r in range(n)]


def block_case(name, X, Y, Z):
    n = len(X[0])
    ones = [1.0] * n
    rx = [residuals(c, Z, ones) for c in X]
    ry = [residuals(c, Z, ones) for c in Y]
    px, py = len(X), len(Y)

    def scatter(A, B):
        return mp.matrix([[sum(a[r] * b[r] for r in range(n)) for b in B] for a in A])

    sxx, syy, sxy = scatter(rx, rx), scatter(ry, ry), scatter(rx, ry)
    joint = mp.matrix(px + py, px + py)
    for i in range(px):
        for j in range(px):
            joint[i, j] = sxx[i, j]
        for j in range(py):
            joint[i, px + j] = sxy[i, j]
            joint[px + j, i] = sxy[i, j]
    for i in range(py):
        for j in range(py):
            joint[px + i, px + j] = syy[i, j]
    lam = mp.det(joint) / (mp.det(sxx) * mp.det(syy))
    M = mp.inverse(sxx) * sxy * mp.inverse(syy) * sxy.T
    ev = mp.eig(M, left=False, right=False)
    rho2 = max(mp.re(e) for e in ev)
    mult = (n - len(Z) - 1) - mp.mpf(px + py + 1) / 2
    chi2 = -mult * mp.log(lam)
    df = px * py
    p = mp.gammainc(mp.mpf(df) / 2, chi2 / 2, mp.inf, regularized=True)
    return {"name": name, "X": X, "Y": Y, "Z": Z, "rho": float(mp.sqrt(rho2)), "ln_lambda": float(mp.log(lam)),
            "p": float(p), "df": df}


def t_two_sided(r, df):
    t2 = r * r * df / (1 - r * r)
    return mp.betainc(mp.mpf(df) / 2, mp.mpf(1) / 2, 0, df / (df + t2), regularized=True)


def pearson(a, b, w):
    sw = sum(w)
    ma = sum(x * y for x, y in zip(a, w)) / sw
    mb = sum(x * y for x, y in zip(b, w)) / sw
    sab = sum(wi * (x - ma) * (y - mb) for x, y, wi in zip(a, b, w))
    saa = sum(wi * (x - ma) ** 2 for x, wi in zip(a, w))
    sbb = sum(wi * (y - mb) ** 2 for y, wi in zip(b, w))
    return sab / mp.sqrt(saa * sbb)


def weighted_case(name, x, y, Z, w):
    W = [mp.mpf(v) for v in w]
    ex, ey = residuals(x, Z, w), residuals(y, Z, w)
    r = pearson(ex, ey, W)
    neff = sum(W) ** 2 / sum(v * v for v in W)
    df = neff - 2 - len(Z)
    return {"name": name, "x": x, "y": y, "Z": Z, "w": w, "r": float(r), "df": float(df), "p": float(t_two_sided(r, df))}


def midranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def score(v):
    n = len(v)
    return [float(mp.sqrt(2) * mp.erfinv(2 * mp.mpf(r) / (n + 1) - 1)) for r in midranks(v)]


def robust_case(name, x, y, Z):
    n = len(x)
    sx, sy = score(x), score(y)
    sz = [score(z) for z in Z]
    ones = [1.0] * n
    ex, ey = residuals(sx, sz, ones), residuals(sy, sz, ones)
    r = pearson(ex, ey, [mp.mpf(1)] * n)
    df = n - 2 - len(Z)
    return {"name": name, "x": x, "y": y, "Z": Z, "r": float(r), "df": df, "p": float(t_two_sided(r, df))}


def main():
    blocks = []
    n = 60
    X = [[normal() for _ in range(n)] for _ in range(2)]
    Y = [[0.5 * X[0][i] + normal() for i in range(n)], [normal() for _ in range(n)]]
    blocks.append(block_case("two_by_two", X, Y, []))
    n = 80
    Z = [[normal() for _ in range(n)] for _ in range(2)]
    X = [[Z[0][i] + normal() for i in range(n)], [Z[1][i] + normal() for i in range(n)]]
    Y = [[0.4 * X[1][i] + Z[0][i] + normal() for i in range(n)], [normal() + Z[1][i] for i in range(n)],
         [normal() for _ in range(n)]]
    blocks.append(block_case("two_by_three_given_two", X, Y, Z))
    n = 50
    Z = [[normal() for _ in range(n)]]
    X = [[normal() + Z[0][i] for i in range(n)] for _ in range(3)]
    Y = [[X[0][i] + X[1][i] - X[2][i] + normal() + Z[0][i] for i in range(n)]] + [[normal() for _ in range(n)] for _ in range(2)]
    blocks.append(block_case("three_by_three_given_one", X, Y, Z))
    # Exchangeable X columns with a difference-direction dependence on Y: the whitened Gram of the
    # canonical problem is [[g, -g], [-g, g]] exactly, with the leading eigenvector (1, -1).
    g = [-1.0, 0.0, 1.0]
    rows = [(a, b, c, d) for a in g for b in g for c in g for d in g]
    X = [[r[0] for r in rows], [r[1] for r in rows]]
    Y = [[r[0] - r[1] + r[2] for r in rows], [r[3] for r in rows]]
    blocks.append(block_case("exchangeable_difference_direction", X, Y, []))

    weighted = []
    n = 40
    Z = [[normal() for _ in range(n)] for _ in range(2)]
    x = [0.5 * Z[0][i] + normal() for i in range(n)]
    y = [0.3 * x[i] - 0.4 * Z[1][i] + normal() for i in range(n)]
    w = [0.2 + 3 * u01() for _ in range(n)]
    weighted.append(weighted_case("random_weights", x, y, Z, w))
    w2 = list(w)
    w2[3] = 0.0
    w2[17] = 0.0
    weighted.append(weighted_case("zero_weights", x, y, Z, w2))
    w3 = [1.0] * 30 + [40.0] * 10
    weighted.append(weighted_case("concentrated_weights", x, y, Z, w3))

    robust = []
    n = 50
    z = [normal() for _ in range(n)]
    x = [round(math.exp(z[i] + 0.5 * normal()), 1) for i in range(n)]
    y = [round(math.exp(0.6 * z[i] + 0.8 * normal()) + 0.3 * x[i] ** 0.5, 1) for i in range(n)]
    robust.append(robust_case("ties_and_heavy_tails", x, y, [z]))
    robust.append(robust_case("no_conditioning", x, y, []))
    json.dump({"block": blocks, "weighted": weighted, "robust": robust}, __import__("sys").stdout, separators=(",", ":"))


main()
