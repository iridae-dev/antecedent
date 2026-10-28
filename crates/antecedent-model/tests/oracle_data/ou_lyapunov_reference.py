# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact reference for the Ornstein-Uhlenbeck Lyapunov solve  M X + X M^T = -Q.

Regenerate with:  uv run --script ou_lyapunov_reference.py > ou_lyapunov_reference.json

The drift M and the symmetric direction Q are drawn from a seeded stdlib generator and rounded to
dyadic rationals, so the float64 inputs the product receives are exactly the rationals solved
here. The vectorized equation (I kron M + M kron I) vec(X) = -vec(Q) is solved in exact
Fraction arithmetic by Gaussian elimination with no rounding anywhere, and X is rounded to
float64 only at the end. Cases: a real-spectrum drift, a drift with complex-conjugate eigenvalue
pairs (rotation blocks), a non-normal drift, and an indefinite direction.

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import random
from fractions import Fraction as F


def dyadic(rng, lo, hi, bits=20):
    return float(F(rng.randrange(int(lo * 2**bits), int(hi * 2**bits)), 2**bits))


def exact_solve(m, q, n):
    k = n * n
    a = [[F(0)] * k for _ in range(k)]
    for i in range(n):
        for j in range(n):
            for c in range(n):
                a[i * n + j][c * n + j] += F(m[i][c])
                a[i * n + j][i * n + c] += F(m[j][c])
    b = [-F(q[i][j]) for i in range(n) for j in range(n)]
    for c in range(k):
        p = next(r for r in range(c, k) if a[r][c] != 0)
        a[c], a[p] = a[p], a[c]
        b[c], b[p] = b[p], b[c]
        for r in range(c + 1, k):
            f = a[r][c] / a[c][c]
            if f:
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
                b[r] -= f * b[c]
    x = [F(0)] * k
    for r in range(k - 1, -1, -1):
        x[r] = (b[r] - sum(a[r][c] * x[c] for c in range(r + 1, k))) / a[r][r]
    return [[float(x[i * n + j]) for j in range(n)] for i in range(n)]


def sym(rng, n, indefinite):
    g = [[dyadic(rng, -1, 1) for _ in range(n)] for _ in range(n)]
    q = [[sum(g[i][k] * g[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    if indefinite:
        for i in range(n):
            q[i][i] -= 3.0
    return [[float(F(q[i][j]) / 2 + F(q[j][i]) / 2) for j in range(n)] for i in range(n)]


def main():
    rng = random.Random(20261028)
    n = 8
    cases = []
    for name in ("real_spectrum", "complex_pairs", "non_normal", "indefinite_direction"):
        m = [[0.0] * n for _ in range(n)]
        if name == "real_spectrum":
            for i in range(n):
                m[i][i] = -dyadic(rng, 1.0, 2.5)
                for j in range(n):
                    if i != j:
                        m[i][j] = dyadic(rng, -0.15, 0.15)
        elif name == "complex_pairs":
            for b in range(0, n, 2):
                d, w = dyadic(rng, 0.5, 1.5), dyadic(rng, 1.0, 4.0)
                m[b][b] = m[b + 1][b + 1] = -d
                m[b][b + 1], m[b + 1][b] = -w, w
            for i in range(n):
                for j in range(n):
                    if i // 2 != j // 2:
                        m[i][j] = dyadic(rng, -0.1, 0.1)
        elif name == "non_normal":
            for i in range(n):
                m[i][i] = -dyadic(rng, 1.0, 1.5)
                for j in range(i + 1, n):
                    m[i][j] = dyadic(rng, 1.0, 3.0)
        else:
            for i in range(n):
                m[i][i] = -dyadic(rng, 1.0, 2.0)
                for j in range(n):
                    if i != j:
                        m[i][j] = dyadic(rng, -0.2, 0.2)
        q = sym(rng, n, name == "indefinite_direction")
        cases.append({"name": name, "n": n, "drift": m, "direction": q, "solution": exact_solve(m, q, n)})
    print(json.dumps({"cases": cases}, indent=1))


main()
