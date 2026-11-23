"""Exact-rational one-period weighted g-formula, including the declared ridge."""

import json
from fractions import Fraction as F
from pathlib import Path

x = [F(-1), F(1), F(-1), F(1), F(-1), F(1), F(-1), F(1)]
a = [0, 0, 1, 1, 0, 0, 1, 1]
w = [F(1), F(2), F(3), F(1), F(2), F(1), F(1), F(2)]
y = [F(2) + x[i] / 2 + a[i] * F(3, 4) + F(i % 3, 5) for i in range(8)]
rows = [[F(1), x[i], F(a[i]), x[i] * a[i]] for i in range(8)]
mat = [
    [
        sum(w[t] * rows[t][i] * rows[t][j] for t in range(8))
        + (F(1, 10**6) if i == j else 0)
        for j in range(4)
    ]
    for i in range(4)
]
rhs = [sum(w[t] * rows[t][i] * y[t] for t in range(8)) for i in range(4)]
# Rational Gauss-Jordan, independent of the production Cholesky implementation.
for i in range(4):
    pivot = mat[i][i]
    mat[i] = [v / pivot for v in mat[i]]
    rhs[i] /= pivot
    for j in range(4):
        if i != j:
            factor = mat[j][i]
            mat[j] = [v - factor * p for v, p in zip(mat[j], mat[i], strict=True)]
            rhs[j] -= factor * rhs[i]
mean_x = sum(wi * xi for wi, xi in zip(w, x, strict=True)) / sum(w)
control = rhs[0] + rhs[1] * mean_x
active = control + rhs[2] + rhs[3] * mean_x
Path(__file__).with_suffix(".json").write_text(
    json.dumps({"active": float(active), "reference": float(control)}, indent=2) + "\n"
)
