# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Frozen-weight influence-function mixture, weighted mean and joint covariance in exact rationals.

Regenerate with:  uv run --script joint_if_reference.py > joint_if_reference.json

Every score and weight is an exactly representable binary64 value, converted to a Fraction, so
the references below carry no rounding at all.  Definitions (the delta method for a Hajek mean):
  theta_j = sum_i w_i phi_ij / sum_i w_i                       (w_i = 1 when unweighted)
  psi_ij  = n (w_i / W) (phi_ij - theta_j)                     (psi_ij = phi_ij - theta_j unweighted)
  Cov_jk  = sum_i psi_ij psi_ik / (n (n - 1))
Mixture of atoms with frozen weights g_a >= 0 (zero-weight atoms contribute nothing):
  mix_i = sum_a (g_a / sum g) phi_ia.
The nearest binary64 of each exact value is written.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F

_state = 0x9E3779B97F4A7C15


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def wmean(col, w):
    n = len(col)
    if w is None:
        return sum(F(x) for x in col) / n
    W = sum(F(x) for x in w)
    return sum(F(a) * F(b) for a, b in zip(w, col)) / W


def joint(cols, w):
    n = len(cols[0])
    th = [wmean(c, w) for c in cols]
    if w is None:
        psi = [[F(x) - t for x in c] for c, t in zip(cols, th)]
    else:
        W = sum(F(x) for x in w)
        psi = [[n * F(wi) / W * (F(x) - t) for wi, x in zip(w, c)] for c, t in zip(cols, th)]
    k = len(cols)
    cov = [[sum(psi[i][r] * psi[j][r] for r in range(n)) / (n * (n - 1)) for j in range(k)]
           for i in range(k)]
    return th, cov


def mix(atoms, g):
    G = sum(F(x) for x in g)
    n = len(atoms[0])
    return [sum(F(gi) / G * F(a[r]) for a, gi in zip(atoms, g)) for r in range(n)]


def f(x):
    return float(x)


cases = []


def add_joint(name, cols, w):
    th, cov = joint(cols, w)
    cases.append({"kind": "joint", "name": name, "cols": cols, "weights": w,
                  "means": [f(t) for t in th],
                  "cov": [[f(v) for v in row] for row in cov]})


def add_mix(name, atoms, g):
    m = mix(atoms, g)
    th, cov = joint([[float(x) for x in m]], None)
    # the product mixes in binary64; the reference mixes exactly, so also freeze the exact column
    cases.append({"kind": "mixture", "name": name, "atoms": atoms, "weights": g,
                  "mixed": [f(x) for x in m],
                  "mixed_se2": f(sum((x - sum(m) / len(m)) ** 2 for x in m) / (len(m) * (len(m) - 1)))})


# tiny sample, two rows
add_joint("two rows unweighted", [[1.0, 4.0], [0.5, -2.0]], None)
add_joint("two rows weighted", [[1.0, 4.0], [0.5, -2.0]], [3.0, 1.0])
# large offset, small spread
c = [1e8 + (u01() - 0.5) for _ in range(40)]
d = [-3e7 + 2 * (u01() - 0.5) for _ in range(40)]
add_joint("offset 1e8 unweighted", [c, d], None)
w = [0.25 + u01() for _ in range(40)]
add_joint("offset 1e8 weighted", [c, d], w)
# extreme weights: nine orders of magnitude, ties in scores
phi = [2.0, 2.0, 2.0, 5.0, 5.0, -1.0, 2.0, 2.0]
wt = [1e-9, 1e9, 1.0, 1e-9, 1e9, 1e3, 1e-6, 1.0]
add_joint("extreme weights with ties", [phi, [x * x for x in phi]], wt)
# zero weights on some rows (rows stay in n)
add_joint("zero weights keep n", [[1.0, 2.0, 7.0, 4.0, 9.0]], [0.0, 1.0, 0.0, 2.0, 1.0])
# huge uniform weight magnitude
add_joint("weights 1e150", [[1.0, 2.0, 7.0, 4.0]], [1e150, 2e150, 3e150, 1e150])
# unit-scale weights, four correlated columns
base = [u01() for _ in range(25)]
cols = [[b + 0.1 * u01() for b in base], [-b + 0.1 * u01() for b in base],
        [2 * b for b in base], [u01() for _ in base]]
add_joint("four columns unweighted", cols, None)
add_joint("four columns weighted", cols, [0.1 + 3 * u01() for _ in base])
# identical (collinear) columns give a singular covariance
add_joint("collinear columns", [base, base, [3 * b for b in base]], None)

add_mix("two atoms equal", [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], [1.0, 1.0])
add_mix("zero-weight atom ignored", [[1.0, 2.0, 3.0, 9.0], [3.0, 2.0, 1.0, 0.0]], [2.0, 0.0])
add_mix("unnormalised weights", [[1.0, -2.0, 3.0], [0.5, 2.0, -1.0], [4.0, 4.0, 4.0]], [1e-3, 3e-3, 6e-3])
add_mix("extreme weights", [[1.0, -2.0, 3.0], [0.5, 2.0, -1.0]], [1e-12, 1e12])
add_mix("offset atoms", [[1e8 + 1, 1e8 - 1, 1e8], [1e8 - 1, 1e8 + 1, 1e8 + 2]], [0.3, 0.7])

print(json.dumps({"cases": cases}, indent=1))
