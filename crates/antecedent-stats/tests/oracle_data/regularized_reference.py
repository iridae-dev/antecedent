# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Ridge, lasso and elastic-net fits in multiprecision.

Regenerate with:  uv run --script regularized_reference.py > regularized_reference.json

Ridge solves (X'X + lambda P) b = X'y exactly in rational arithmetic (P = I, or I with the
first diagonal entry zero for an unpenalized leading column).  The elastic net minimizes
0.5 ||y - a - X b||^2 + lambda alpha ||b||_1 + 0.5 lambda (1 - alpha) ||b||_2^2 (with an
unpenalized intercept a when requested; columns divided by sqrt(mean square) of the centered
column when standardized, coefficients reported on the original scale).  Instead of iterating, the
minimizer is found by enumerating every sign pattern of every active set (p <= 4), solving the
active-set equations in 80-digit arithmetic and keeping the pattern whose signs and
Karush-Kuhn-Tucker inequalities hold; strict convexity makes it unique.  Repository-authored
(MIT OR Apache-2.0); mpmath is used only to produce the frozen file.
"""
import itertools
import json
from fractions import Fraction

import mpmath as mp

mp.mp.dps = 80
_state = 0x853C49E6748FEA9B


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def normal():
    return float(mp.sqrt(-2 * mp.log(mp.mpf(u01()) + mp.mpf(2) ** -60)) * mp.cos(2 * mp.pi * u01()))


def solve_fraction(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [x / d for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def ridge(cols, y, lam, unpen_first):
    n, p = len(y), len(cols)
    X = [[Fraction(c[i]) for c in cols] for i in range(n)]
    Y = [Fraction(v) for v in y]
    lam = Fraction(lam)
    A = [[sum(X[r][i] * X[r][j] for r in range(n)) for j in range(p)] for i in range(p)]
    for i in range(p):
        if not (i == 0 and unpen_first):
            A[i][i] += lam
    b = [sum(X[r][i] * Y[r] for r in range(n)) for i in range(p)]
    return [float(v) for v in solve_fraction(A, b)]


def enet(cols, y, lam, alpha, intercept, standardize):
    n, p = len(y), len(cols)
    mpn = mp.mpf(n)
    Y = [mp.mpf(v) for v in y]
    ym = sum(Y) / mpn if intercept else mp.mpf(0)
    Yc = [v - ym for v in Y]
    means, scales, Xc = [], [], []
    for c in cols:
        C = [mp.mpf(v) for v in c]
        m = sum(C) / mpn if intercept else mp.mpf(0)
        Cc = [v - m for v in C]
        s = mp.sqrt(sum(v * v for v in Cc) / mpn) if standardize else mp.mpf(1)
        Xc.append([v / s for v in Cc])
        means.append(m)
        scales.append(s)
    l1 = mp.mpf(lam) * mp.mpf(alpha)
    l2 = mp.mpf(lam) * (1 - mp.mpf(alpha))
    G = [[sum(a * b for a, b in zip(Xc[i], Xc[j])) for j in range(p)] for i in range(p)]
    g = [sum(a * b for a, b in zip(Xc[i], Yc)) for i in range(p)]
    for k in range(p + 1):
        for A in itertools.combinations(range(p), k):
            for s in itertools.product([-1, 1], repeat=k):
                beta = [mp.mpf(0)] * p
                if k:
                    M = mp.matrix(k, k)
                    rhs = mp.matrix(k, 1)
                    for a, i in enumerate(A):
                        for b_, j in enumerate(A):
                            M[a, b_] = G[i][j] + (l2 if i == j else 0)
                        rhs[a] = g[i] - l1 * s[a]
                    sol = mp.lu_solve(M, rhs)
                    if any(sol[a] * s[a] <= 0 for a in range(k)):
                        continue
                    for a, i in enumerate(A):
                        beta[i] = sol[a]
                grad = [g[i] - sum(G[i][j] * beta[j] for j in range(p)) - l2 * beta[i] for i in range(p)]
                ok = all(abs(grad[i]) <= l1 * (1 + mp.mpf(10) ** -60) for i in range(p) if i not in A)
                if ok:
                    coef = [beta[i] / scales[i] for i in range(p)]
                    a0 = ym - sum(means[i] * coef[i] for i in range(p)) if intercept else mp.mpf(0)
                    return float(a0), [float(v) for v in coef]
    raise SystemExit("no KKT point found")


def lam_max(cols, y, alpha, intercept, standardize):
    n = len(y)
    mpn = mp.mpf(n)
    Y = [mp.mpf(v) for v in y]
    ym = sum(Y) / mpn if intercept else mp.mpf(0)
    best = mp.mpf(0)
    for c in cols:
        C = [mp.mpf(v) for v in c]
        m = sum(C) / mpn if intercept else mp.mpf(0)
        Cc = [v - m for v in C]
        sc = mp.sqrt(sum(v * v for v in Cc) / mpn) if standardize else mp.mpf(1)
        best = max(best, abs(sum(a / sc * (b - ym) for a, b in zip(Cc, Y))))
    return float(best / mp.mpf(alpha))


def design(n, p, rho):
    z = [normal() for _ in range(n)]
    cols = []
    for j in range(p):
        w = mp.sqrt(rho) if rho < 1 else 1
        cols.append([float(w * mp.mpf(z[i]) + mp.sqrt(1 - rho) * mp.mpf(normal()) + 0.3 * j) for i in range(n)])
    return cols


def main():
    out = {"ridge": [], "enet": [], "correlated": []}
    for n, p, rho, lam, unpen in [
        (12, 3, 0.2, 0.5, False),
        (15, 4, 0.9, 2.0, False),
        (10, 3, 0.999999, 1e-6, False),
        (20, 4, 0.5, 1e-3, True),
        (9, 3, 0.0, 1e3, True),
    ]:
        cols = design(n, p, rho)
        if unpen:
            cols[0] = [1.0] * n
        y = [float(c) for c in [sum(cols[j][i] * (j + 1) for j in range(p)) + normal() for i in range(n)]]
        out["ridge"].append({"n": n, "p": p, "cols": cols, "y": y, "lambda": lam, "unpenalized_leading": unpen,
                             "coef": ridge(cols, y, lam, unpen)})
    for n, p, rho, frac, alpha, icpt, std in [
        (16, 3, 0.3, 0.3, 1.0, True, False),
        (16, 3, 0.3, 0.3, 1.0, True, True),
        (18, 4, 0.7, 0.2, 1.0, True, True),
        (18, 4, 0.7, 0.2, 0.5, True, True),
        (14, 3, 0.5, 0.15, 1.0, False, False),
        (14, 3, 0.5, 0.15, 0.5, False, True),
        (20, 4, 0.9, 0.05, 1.0, True, False),
        (20, 4, 0.2, 1e-5, 1.0, True, False),
        (20, 3, 0.4, 0.95, 1.0, True, False),
    ]:
        cols = design(n, p, rho)
        cols = [[v * (10.0 ** (j - 1)) for v in c] for j, c in enumerate(cols)]  # unequal column scales
        y = [sum(cols[j][i] * (1.5 - j) / (10.0 ** (j - 1)) for j in range(p)) + 2.0 + normal() for i in range(n)]
        lam = float(mp.mpf(frac) * lam_max(cols, y, alpha, icpt, std))
        a0, coef = enet(cols, y, lam, alpha, icpt, std)
        out["enet"].append({"n": n, "p": p, "cols": cols, "y": y, "lambda": lam, "l1_ratio": alpha,
                            "fit_intercept": icpt, "standardize": std, "intercept": a0, "coef": coef})
    # strongly correlated designs: the default tolerance must hold the fitted values to a small
    # multiple of tol * ||y - ybar|| even though the coefficients are only weakly determined
    for rho in (0.9, 0.99):
        n, p = 200, 3
        cols = design(n, p, rho)
        y = [cols[0][i] + cols[1][i] + normal() for i in range(n)]
        for lam in (1.0, 20.0):
            a0, coef = enet(cols, y, lam, 1.0, True, False)
            out["correlated"].append({"n": n, "p": p, "cols": cols, "y": y, "lambda": lam,
                                      "intercept": a0, "coef": coef})
    print(json.dumps(out, separators=(",", ":")))


main()
