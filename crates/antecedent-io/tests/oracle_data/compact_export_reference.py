# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact reference for the compact runtime export evaluation f(x) = b'phi(x), se = sqrt(phi'V phi).

Regenerate with:  uv run --script compact_export_reference.py > compact_export_reference.json

phi is the basis row of the declared terms (intercept 1, linear x, power x^d, one-hot indicator),
b the stored coefficients and V the stored covariance.  Inputs are dyadic rationals so the basis row,
the point and phi'V phi are exact in rational arithmetic (fractions.Fraction); the standard error is
the 50-digit square root of the exact quadratic form, rounded once to binary64.  The covariances are
A'A for a small dyadic A, hence exactly symmetric positive semidefinite.  Nothing here calls the
product.  Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50


def phi_row(terms, query):
    row = []
    for t in terms:
        if t["kind"] == "intercept":
            row.append(F(1))
        elif t["kind"] == "linear":
            row.append(F(query[t["quantity"]]))
        elif t["kind"] == "power":
            row.append(F(query[t["quantity"]]) ** t["degree"])
        else:
            row.append(F(1) if query[t["quantity"]] == t["level"] else F(0))
    return row


def evaluate(terms, b, v, query):
    phi = phi_row(terms, query)
    n = len(phi)
    point = sum((F(b[i]) * phi[i] for i in range(n)), F(0))
    var = sum((phi[i] * F(v[i * n + j]) * phi[j] for i in range(n) for j in range(n)), F(0))
    se = mp.sqrt(mp.mpf(var.numerator) / mp.mpf(var.denominator))
    return float(point), float(se)


def gram(rng, n):
    a = [[rng.randint(-6, 6) / 64.0 for _ in range(n)] for _ in range(n + 1)]
    return [float(sum((F(a[r][i]) * F(a[r][j]) for r in range(n + 1)), F(0))) for i in range(n) for j in range(n)]


def main():
    rng = random.Random(20261028)
    cases = []
    terms_a = [{"kind": "intercept"}, {"kind": "linear", "quantity": "x"}] + [
        {"kind": "power", "quantity": "x", "degree": d} for d in (2, 3, 4)]
    terms_b = [{"kind": "intercept"}, {"kind": "linear", "quantity": "x"}, {"kind": "power", "quantity": "x", "degree": 2},
               {"kind": "onehot", "quantity": "g", "level": "b"}, {"kind": "onehot", "quantity": "g", "level": "c"}]
    for name, terms in [("quartic_polynomial", terms_a), ("quadratic_with_levels", terms_b)]:
        n = len(terms)
        b = [rng.randint(-24, 24) / 8.0 for _ in range(n)]
        v = gram(rng, n)
        queries = []
        for x in (0.0, 0.5, 1.25, 3.0, 7.75, 10.0):
            for g in (("a", "b", "c") if name == "quadratic_with_levels" else (None,)):
                q = {"x": x}
                if g:
                    q["g"] = g
                point, se = evaluate(terms, b, v, q)
                queries.append({"x": x, "g": g, "point": point, "se": se})
        cases.append({"name": name, "terms": terms, "coefficients": b, "covariance": v, "queries": queries})
    print(json.dumps(cases, indent=1))


main()
