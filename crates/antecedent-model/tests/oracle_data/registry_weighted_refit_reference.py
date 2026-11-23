# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Weighted refit of linear-Gaussian, constant and parent-free discrete mechanisms, exactly.

Regenerate with:  uv run --script registry_weighted_refit_reference.py > registry_weighted_refit_reference.json

Row weights w are rescaled to mean one (so the weights sum to the row count n), as the refit
documents.  For a node with parent columns X (n x p) and outcome y, with the design D = [1 | X]:
  coefficients  b = (D' W D)^{-1} D' W y                 solved in Fraction arithmetic
  residual      sigma = max( sqrt( sum_i w_i (y_i - D_i b)^2 / max(n - (1 + p), 1) ),  1e-8 * rms(y) )
  constant      weighted mean  sum w_i y_i / sum w_i
  discrete      support = the sorted distinct values of y with positive weight mass;
                probability = sum of the weights of the rows at that value / sum w_i
(frequency counts are weights too; a zero count drops the row from the fit but not from n).
mpmath supplies the square root at 50 digits.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def wls(cols, y, w):
    n = len(y)
    D = [[F(1)] + [F(c[i]) for c in cols] for i in range(n)]
    k = len(D[0])
    A = [[sum(w[i] * D[i][a] * D[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    rhs = [sum(w[i] * D[i][a] * F(y[i]) for i in range(n)) for a in range(k)]
    beta = solve(A, rhs)
    rss = sum(w[i] * (F(y[i]) - sum(beta[a] * D[i][a] for a in range(k))) ** 2 for i in range(n))
    rms = mp.sqrt(sum(mp.mpf(F(v).numerator) / F(v).denominator * (mp.mpf(F(v).numerator) / F(v).denominator) for v in y) / n)
    floor = max(mp.mpf("1e-8") * rms, mp.mpf(2) ** -1022)
    s2 = rss / max(n - k, 1)
    sigma = max(mp.sqrt(mp.mpf(s2.numerator) / s2.denominator), floor)
    wls.fitted = [float(sum(beta[a] * D[i][a] for a in range(k))) for i in range(n)]
    return [float(b) for b in beta], float(sigma)


def case(name, x1, x2, y, weights):
    n = len(y)
    total = sum(F(v) for v in weights)
    w = [F(v) * n / total for v in weights]
    out = {"name": name, "x1": x1, "x2": x2, "y": y, "weights": weights}
    b1, s1 = wls([], x1, w)
    b2, s2 = wls([x1], x2, w)  # x2 given x1 is not a model edge; kept for the root fit below
    b0, s0 = wls([], x2, w)
    by, sy = wls([x1, x2], y, w)
    out["y_fitted"] = wls.fitted
    out.update(x1_intercept=b1[0], x1_sigma=s1, x2_intercept=b0[0], x2_sigma=s0, y_intercept=by[0],
               y_coeffs=by[1:], y_sigma=sy)
    wm = sum(w[i] * F(y[i]) for i in range(n)) / sum(w)
    out["y_weighted_mean"] = float(wm)
    return out


def main():
    rng = random.Random(3)

    def dy(k, scale=1):
        return [F(rng.randint(-40, 40), 8) * scale for _ in range(k)]

    cases = []
    n = 12
    x1, x2 = dy(n), dy(n)
    y = [F(2) * a - F(3, 2) * b + F(1, 2) + F(rng.randint(-8, 8), 16) for a, b in zip(x1, x2)]
    cases.append(case("unit_weights", [float(v) for v in x1], [float(v) for v in x2], [float(v) for v in y], [1.0] * n))
    cases.append(case("unbalanced_weights", [float(v) for v in x1], [float(v) for v in x2], [float(v) for v in y],
                      [float(2.0 ** rng.choice([-20, -6, 0, 3, 10, 20])) for _ in range(n)]))
    cases.append(case("frequency_counts_with_zeros", [float(v) for v in x1], [float(v) for v in x2], [float(v) for v in y],
                      [float(c) for c in [0, 2, 1, 0, 3, 1, 0, 1, 4, 0, 2, 1]]))
    big = F(10 ** 8)
    cases.append(case("large_offsets", [float(v + big) for v in x1], [float(v - big) for v in x2], [float(v + 3 * big) for v in y],
                      [float(rng.choice([1, 2, 3])) for _ in range(n)]))
    # parents equal up to 2^-20 jitter: near-collinear design
    jitter = [F(rng.randint(-4, 4), 2 ** 22) for _ in range(n)]
    xa = dy(n)
    xb = [a + j for a, j in zip(xa, jitter)]
    yc = [a + F(1, 3 * 8) * 0 + F(rng.randint(-8, 8), 16) for a in xa]
    cases.append(case("near_collinear_parents", [float(v) for v in xa], [float(v) for v in xb], [float(v) for v in yc], [1.0] * n))
    # four rows: one residual degree of freedom
    cases.append(case("one_residual_dof", [0.0, 1.0, 2.0, 5.0], [1.0, -1.0, 3.0, 0.5], [0.5, 2.0, 1.0, 4.0], [1.0, 2.0, 3.0, 1.0]))
    # three rows for three coefficients: zero residual degrees of freedom (divisor floored at one)
    cases.append(case("saturated_design", [0.0, 1.0, 2.0], [1.0, -1.0, 3.0], [0.5, 2.0, 1.5], [1.0, 1.0, 2.0]))
    print(json.dumps(cases, indent=1))


main()
