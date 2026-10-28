# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Additive marginal structural model: stabilised IPT-and-censoring weighted least squares with a
subject-clustered sandwich, in exact rational arithmetic.

Regenerate with:  uv run --script marginal_structural_model_reference.py > marginal_structural_model_reference.json

For subject i with observed terminal outcome, treatment a_it, treatment probability p_it,
probability of remaining observed c_it and stabilising numerator q_t, the weight is
    w_i = prod_t [ q_t^a (1-q_t)^(1-a) ] / [ p_it^a (1-p_it)^(1-a) c_it ].
The coefficients solve  sum_i w_i x_i x_i' b = sum_i w_i x_i y_i  with x_i = (1, a_i1..a_iT).
Covariance (one subject is one cluster): B^-1 [ sum_i w_i^2 x_i x_i' e_i^2 ] B^-1 * m / (m - (T+1)),
m the number of observed subjects, e_i = y_i - x_i'b.  Kish ESS = (sum w)^2 / sum w^2 over all
rows (unobserved rows have weight 0).  Linear systems are solved by exact Gauss-Jordan.
Variances (not standard errors) are written, as the nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F

_state = 0xC2B2AE3D27D4EB4F


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def solve_inverse(A):
    n = len(A)
    M = [[F(v) for v in row] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                fct = M[r][c]
                M[r] = [a - fct * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
            for i in range(len(A))]


def reference(y, a, p, obs, c, q, periods):
    n = len(y)
    cols = periods + 1
    w = []
    for i in range(n):
        if not obs[i]:
            w.append(F(0))
            continue
        wi = F(1)
        for t in range(periods):
            j = i * periods + t
            top = F(q[t]) if a[j] else 1 - F(q[t])
            bottom = F(p[j]) if a[j] else 1 - F(p[j])
            wi *= top / (bottom * F(c[j]))
        w.append(wi)
    X = [[F(1)] + [F(int(a[i * periods + t])) for t in range(periods)] for i in range(n)]
    A = [[sum(w[i] * X[i][j] * X[i][k] for i in range(n) if obs[i]) for k in range(cols)]
         for j in range(cols)]
    rhs = [sum(w[i] * X[i][j] * F(y[i]) for i in range(n) if obs[i]) for j in range(cols)]
    Ainv = solve_inverse(A)
    beta = [sum(Ainv[j][k] * rhs[k] for k in range(cols)) for j in range(cols)]
    meat = [[F(0)] * cols for _ in range(cols)]
    for i in range(n):
        if not obs[i]:
            continue
        e = F(y[i]) - sum(X[i][j] * beta[j] for j in range(cols))
        for j in range(cols):
            for k in range(cols):
                meat[j][k] += w[i] ** 2 * X[i][j] * X[i][k] * e * e
    m = sum(1 for o in obs if o)
    cov = mm(mm(Ainv, meat), Ainv)
    corr = F(m, m - cols)
    sw = sum(w)
    ess = sw * sw / sum(x * x for x in w)
    return {
        "beta": [float(b) for b in beta],
        "var": [float(cov[j][j] * corr) for j in range(cols)],
        "ess": float(ess),
        "max_weight": float(max(w)),
        "observed": m,
    }


cases = []


def add(name, n, periods, pfun, cfun, q, yfun, obsfun, treat=None):
    a, p, c, y, obs = [], [], [], [], []
    for i in range(n):
        for t in range(periods):
            pi = pfun(i, t)
            p.append(pi)
            c.append(cfun(i, t))
            a.append(u01() < (treat if treat is not None else pi))
        obs.append(obsfun(i))
    for i in range(n):
        y.append(yfun(i, a[i * periods:(i + 1) * periods]))
    cases.append({"name": name, "periods": periods, "y": y, "treatment": a, "treatment_probability": p,
                  "observed": obs, "censoring_probability": c, "numerator": q,
                  "minimum_probability": 0.01, "expected": reference(y, a, p, obs, c, q, periods)})


dyn = lambda i, t: 0.25 + 0.5 * ((i * 7 + t * 3) % 11) / 10
add("two periods, varying propensity", 30, 2, dyn, lambda i, t: 1.0, [0.5, 0.5],
    lambda i, a: 1 + 2 * a[0] + 3 * a[1] + (u01() - 0.5), lambda i: True)
add("three periods with censoring and unobserved subjects", 45, 3, dyn,
    lambda i, t: 0.6 + 0.4 * ((i + t) % 5) / 4, [0.4, 0.5, 0.6],
    lambda i, a: 0.5 - a[0] + 2 * a[1] + 0.3 * a[2] + (u01() - 0.5), lambda i: i % 6 != 0)
add("extreme weights from propensities near the floor", 40, 2,
    lambda i, t: 0.02 if (i + t) % 4 == 0 else 0.97, lambda i, t: 0.05 if i % 5 == 0 else 1.0,
    [0.5, 0.5], lambda i, a: 2 * a[0] - a[1] + (u01() - 0.5), lambda i: True, treat=0.5)
add("outcome offset 1e7", 36, 2, dyn, lambda i, t: 0.9, [0.5, 0.5],
    lambda i, a: 1e7 + 2 * a[0] + 3 * a[1] + (u01() - 0.5), lambda i: i % 9 != 4)
add("rare second-period treatment", 50, 2, lambda i, t: 0.5 if t == 0 else 0.1, lambda i, t: 1.0,
    [0.5, 0.2], lambda i, a: 1 + a[0] + 4 * a[1] + (u01() - 0.5), lambda i: True)
add("smallest admissible sample", 4, 1, lambda i, t: 0.5, lambda i, t: 1.0, [0.5],
    lambda i, a: 1 + 2 * a[0] + (u01() - 0.5), lambda i: True)

print(json.dumps({"cases": cases}, indent=1))
