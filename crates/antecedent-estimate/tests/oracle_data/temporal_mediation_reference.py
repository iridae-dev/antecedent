# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Linear temporal mediation: the three mechanism regressions solved in exact rational arithmetic.

Regenerate with:  uv run --script temporal_mediation_reference.py > temporal_mediation_reference.json

For a window time t the treatment is read at t - h, the mediators and the outcome at t and each
adjustment column at its own lag; the windows are the times >= the deepest lag whose every read
value is observed.  With W = [1, adjustment columns], three least-squares problems are solved:
  M_j ~ [1, T, adj]            -> a_j   (the treatment coefficient)
  Y   ~ [1, T, M_1..M_q, adj]  -> c'    (the treatment coefficient), b_j (the mediator coefficients)
  Y   ~ [1, T, adj]            -> c
total = delta c, direct = delta c', mediated = delta sum_j a_j b_j, with delta the active minus the
control treatment level.  The iid standard errors the product publishes on request are
  total, direct : delta sqrt(sigma2 [(X'X)^-1]_TT) of the Y ~ T (+ M) regression, sigma2 = rss/(n-p),
  mediated      : delta sqrt(sum_j b_j^2 Var(a_j) + sum_{j,l} a_j a_l Cov(b_j, b_l)),
                  the Sobel form that takes the separate mediator regressions as independent of each
                  other and of the outcome regression (an approximation the product states; it is
                  the formula checked here, not the exact variance of a product of estimates).
Values are written as the nearest binary64; the series are binary64 numbers taken exactly.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F
from math import isqrt

_state = 0x13198A2E03707344


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def inverse(A):
    n = len(A)
    M = [row[:] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def fsqrt(q, digits=60):
    scaled = q * 10 ** (2 * digits)
    return F(isqrt(scaled.numerator // scaled.denominator), 10**digits)


def ols(rows, ys):
    n, p = len(rows), len(rows[0])
    A = [[sum(r[a] * r[b] for r in rows) for b in range(p)] for a in range(p)]
    inv = inverse(A)
    beta = [sum(inv[a][b] * sum(r[b] * v for r, v in zip(rows, ys)) for b in range(p)) for a in range(p)]
    rss = sum((v - sum(bk * rk for bk, rk in zip(beta, r))) ** 2 for r, v in zip(rows, ys))
    return beta, inv, rss / (n - p)


def run(name, n, h, q, adj_lags, drop, offsets, delta):
    T = [offsets[0] + 4 * (u01() - 0.5) for _ in range(n)]
    Z = [offsets[3] + (u01() - 0.5) for _ in range(n)]
    Ms = []
    for j in range(q):
        Ms.append([offsets[1] + (0.7 - 0.4 * j) * (T[t - h] - offsets[0] if t >= h else 0.0)
                   + 0.5 * (Z[t - 1] - offsets[3] if t >= 1 else 0.0) + 0.6 * (u01() - 0.5)
                   for t in range(n)])
    Y = [offsets[2] + 0.3 * (T[t - h] - offsets[0] if t >= h else 0.0)
         + sum((0.9 - 0.5 * j) * (Ms[j][t] - offsets[1]) for j in range(q))
         + 0.4 * (Z[t - 1] - offsets[3] if t >= 1 else 0.0) + 0.5 * (u01() - 0.5) for t in range(n)]
    missing = [t in drop for t in range(n)]
    max_lag = max([h] + adj_lags)
    keep = []
    for t in range(max_lag, n):
        reads = [t, t - h] + [t - l for l in adj_lags]
        if not any(missing[r] for r in reads):
            keep.append(t)
    adj = lambda t: [F(Z[t - l]) for l in adj_lags]
    rows_a = [[F(1), F(T[t - h])] + adj(t) for t in keep]
    a, var_a = [], []
    for j in range(q):
        beta, inv, s2 = ols(rows_a, [F(Ms[j][t]) for t in keep])
        a.append(beta[1])
        var_a.append(s2 * inv[1][1])
    rows_b = [[F(1), F(T[t - h])] + [F(Ms[j][t]) for j in range(q)] + adj(t) for t in keep]
    beta_b, inv_b, s2_b = ols(rows_b, [F(Y[t]) for t in keep])
    c_prime, b = beta_b[1], beta_b[2 : 2 + q]
    beta_c, inv_c, s2_c = ols(rows_a, [F(Y[t]) for t in keep])
    c = beta_c[1]
    d = F(delta)
    mediated = d * sum(x * y for x, y in zip(a, b))
    var_med = sum(b[j] ** 2 * var_a[j] for j in range(q)) + sum(
        a[j] * a[l] * s2_b * inv_b[2 + j][2 + l] for j in range(q) for l in range(q)
    )
    return {
        "name": name, "h": h, "q": q, "adj_lags": adj_lags, "delta": delta,
        "t": T, "mediators": Ms, "y": Y, "z": Z, "missing": missing,
        "rows": len(keep), "times": keep,
        "total": float(d * c), "direct": float(d * c_prime), "mediated": float(mediated),
        "se_total": float(abs(d) * fsqrt(s2_c * inv_c[1][1])),
        "se_direct": float(abs(d) * fsqrt(s2_b * inv_b[1][1])),
        "se_mediated": float(abs(d) * fsqrt(var_med)),
    }


cases = [
    run("one_mediator", 40, 1, 1, [], set(), (0.0, 0.0, 0.0, 0.0), 1.0),
    run("adjustment_and_delta", 40, 1, 1, [1, 2], set(), (0.0, 0.0, 0.0, 0.0), 2.5),
    run("two_mediators_lag2", 44, 2, 2, [1], set(), (0.0, 0.0, 0.0, 0.0), 1.0),
    run("offsets", 40, 1, 1, [1], set(), (1.0e3, 1.0e5, 1.0e6, 1.0e2), 1.0),
    run("gaps", 44, 1, 1, [1], {9, 10, 25, 37}, (0.0, 0.0, 0.0, 0.0), 1.0),
    run("tiny", 9, 1, 1, [], set(), (0.0, 0.0, 0.0, 0.0), 1.0),
]
print(json.dumps({"cases": cases}, indent=1))
