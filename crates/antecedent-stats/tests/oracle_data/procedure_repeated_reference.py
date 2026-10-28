# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference for adaptive_inference.procedure_repeated_measures.

Regenerate with:  uv run --script procedure_repeated_reference.py > procedure_repeated_reference.json
(or python3 with mpmath 1.3 installed).

Exact rational arithmetic throughout, except the normal distribution function (mpmath, 30 digits).

* `summaries`: for a unit of m measurements with marginal variance s2 and correlation matrix R
  (compound symmetry: rho off the diagonal; AR(1): rho^|i-j|; random effect: b / (b + w)), the
  variance s2 * w'Rw of the mean summary (w = 1/m) and of the generalized-least-squares summary
  w = R^-1 1 / (1' R^-1 1), obtained by exact Gauss-Jordan inversion of R (no closed form assumed),
  with variance s2 / (1' R^-1 1); and the design effect m * Var / s2.
* `mixtures`: for a fixed two-arm allocation with n revealed units, the number k of plus-arm units
  is Binomial(n, p); given 1 <= k <= n - 1 the contrast is N(delta, v (1/k + 1/(n-k))).  The exact
  weights are Binomial probabilities as Fractions (n up to 5000), the undefined mass is
  P(k = 0) + P(k = n), and the law's mean, variance and cdf given definedness follow.

Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F
from math import comb

import mpmath as mp

mp.mp.dps = 30


def frac_text(x):
    return f"{x.numerator}/{x.denominator}"


def cs_matrix(m, rho):
    return [[F(1) if i == j else rho for j in range(m)] for i in range(m)]


def ar_matrix(m, rho):
    return [[rho ** abs(i - j) for j in range(m)] for i in range(m)]


def solve(a, b):
    n = len(a)
    mat = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if mat[r][col] != 0)
        mat[col], mat[piv] = mat[piv], mat[col]
        p = mat[col][col]
        mat[col] = [x / p for x in mat[col]]
        for r in range(n):
            if r != col and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [x - f * y for x, y in zip(mat[r], mat[col])]
    return [mat[i][n] for i in range(n)]


def quad(w, r):
    m = len(w)
    return sum(w[i] * r[i][j] * w[j] for i in range(m) for j in range(m))


summaries = []


def add_summary(name, structure, m, matrix, s2):
    ones = [F(1)] * m
    mean_w = [F(1, m)] * m
    x = solve(matrix, ones)  # R^-1 1
    denom = sum(x)  # 1' R^-1 1
    gls_w = [xi / denom for xi in x]
    for kind, w in (("mean", mean_w), ("gls", gls_w)):
        var = s2 * quad(w, matrix)
        if kind == "gls":
            assert var == s2 / denom
        summaries.append(
            {
                "name": name,
                "structure": structure,
                "measurements": m,
                "summary": kind,
                "marginal_variance": float(s2),
                "unit_summary_variance": float(var),
                "design_effect": float(m * var / s2),
            }
        )


for m in (2, 3, 5, 12, 32):
    for rho in (F(0), F(1, 2), F(-9, 10) / (m - 1), F(99, 100), F(-1, 3) / (m - 1) * 2):
        if not (rho < 1 and rho * (m - 1) > -1):
            continue
        add_summary(
            "compound_symmetry",
            {"kind": "compound", "variance": frac_text(F(7, 4)), "rho": frac_text(rho)},
            m, cs_matrix(m, rho), F(7, 4),
        )
    for rho in (F(0), F(1, 2), F(-9, 10), F(99, 100)):
        add_summary(
            "autoregressive",
            {"kind": "autoregressive", "variance": frac_text(F(3, 2)), "rho": frac_text(rho)},
            m, ar_matrix(m, rho), F(3, 2),
        )
    for b, w in ((F(0), F(2)), (F(3, 2), F(1, 2)), (F(1, 100), F(5))):
        icc = b / (b + w)
        add_summary(
            "random_effect",
            {"kind": "hierarchical", "between": frac_text(b), "within": frac_text(w)},
            m, cs_matrix(m, icc), b + w,
        )


def binom(n, p, k):
    return F(comb(n, k)) * p**k * (1 - p) ** (n - k)


mixtures = []
for n, p, v, delta in (
    (5, F(1, 2), F(3, 4), F(2, 5)),
    (40, F(1, 10), F(3, 4), F(-1, 3)),
    (300, F(9, 10), F(5, 4), F(1, 2)),
    (1100, F(1, 2), F(3, 4), F(1, 5)),
    (2000, F(1, 2), F(3, 4), F(1, 5)),
    (2000, F(1, 3), F(3, 4), F(0)),
    (4000, F(1, 2), F(7, 4), F(1, 4)),
):
    q = 1 - p
    weights = {k: binom(n, p, k) for k in range(1, n)}
    defined = sum(weights.values())
    undefined = binom(n, p, 0) + binom(n, p, n)
    assert defined + undefined == 1
    variances = {k: v * (F(1, k) + F(1, n - k)) for k in weights}
    mean = delta
    var = sum(weights[k] * (variances[k] + (delta - mean) ** 2) for k in weights) / defined
    cdf_points = []
    for x in (delta - 3 * float(var) ** 0.5, delta, delta + float(var) ** 0.5):
        x_exact = F(x)
        total = mp.mpf(0)
        for k, w in weights.items():
            if w == 0:
                continue
            sd = mp.sqrt(mp.mpf(variances[k].numerator) / mp.mpf(variances[k].denominator))
            z = (mp.mpf(x_exact.numerator) / mp.mpf(x_exact.denominator) - mp.mpf(delta.numerator) / mp.mpf(delta.denominator)) / sd
            total += mp.mpf(w.numerator) / mp.mpf(w.denominator) * mp.ncdf(z)
        cdf_points.append({"x": float(x), "cdf": float(total / (mp.mpf(defined.numerator) / mp.mpf(defined.denominator)))})
    keep = sorted(weights)[:3] + sorted(weights)[n // 2 - 2 : n // 2 + 2] + sorted(weights)[-3:]
    mixtures.append(
        {
            "n": n,
            "plus_probability": frac_text(p),
            "summary_variance": float(v),
            "delta": float(delta),
            "undefined_mass": float(undefined),
            "defined_mass": float(defined),
            "variance": float(var),
            "components": {
                str(k): {"weight": float(weights[k]), "variance": float(variances[k])} for k in keep
            },
            "component_count": n - 1,
            "cdf": cdf_points,
        }
    )

print(json.dumps({"summaries": summaries, "mixtures": mixtures}, indent=1))
