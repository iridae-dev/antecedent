# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Local Gaussian BIC scores in 60-digit arithmetic, for ill-conditioned designs.

Regenerate with:  uv run --script gaussian_bic_reference.py > gaussian_bic_reference.json

The score is -n/2 (1 + ln 2 pi + ln(RSS/n)) - (k/2) ln n with k = |Pa| + 2 and RSS the residual
sum of squares of the node regressed on an intercept and its parents.  RSS is obtained by
solving the normal equations exactly (multiprecision LU), so it is the true value for the
stored binary64 columns.  Columns are stored in the JSON.  Repository-authored
(MIT OR Apache-2.0); mpmath is used only to produce the frozen file.
"""
import json
import sys

import mpmath as mp

mp.mp.dps = 80
_state = 0xD1B54A32D192ED03


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def noise(n):
    return [2.0 * (u01() - 0.5) for _ in range(n)]


def bic(cols, node, parents):
    n = len(cols[0])
    k = len(parents) + 1
    X = mp.matrix(n, k)
    for i in range(n):
        X[i, 0] = 1
        for j, p in enumerate(parents):
            X[i, j + 1] = mp.mpf(cols[p][i])
    y = mp.matrix([mp.mpf(v) for v in cols[node]])
    beta = mp.lu_solve(X.T * X, X.T * y)
    r = y - X * beta
    rss = sum(v * v for v in r)
    s2 = rss / n
    return float(-mp.mpf(n) / 2 * (1 + mp.log(2 * mp.pi) + mp.log(s2)) - mp.mpf(k + 1) / 2 * mp.log(n)), float(rss)


cases = []


def add(name, cols, queries):
    out = []
    for node, parents in queries:
        score, rss = bic(cols, node, parents)
        out.append({"node": node, "parents": parents, "score": score, "rss": rss})
    cases.append({"name": name, "columns": cols, "queries": out})


n = 100
x1, x2, e, f = noise(n), noise(n), noise(n), noise(n)
y = [1 + 2 * a - b + 0.4 * c for a, b, c in zip(x1, x2, e)]
add("well_conditioned", [x1, x2, y], [(2, [0, 1]), (2, [0]), (1, [0]), (2, [])])

x1, b, e = noise(n), noise(n), noise(n)
x2 = [a + 1e-6 * c for a, c in zip(x1, b)]
y = [a + c + 0.3 * d for a, c, d in zip(x1, x2, e)]
add("nearly_collinear_parents", [x1, x2, y], [(2, [0, 1]), (2, [0]), (1, [0]), (0, [1, 2])])

x1, e = noise(n), noise(n)
y = [3 * a + 1e-5 * c for a, c in zip(x1, e)]
add("nearly_deterministic_child", [x1, y], [(1, [0]), (0, [1])])

x1, x2, e = noise(n), noise(n), noise(n)
x1 = [1.0e7 + a for a in x1]
x2 = [-3.0e6 + 2 * a for a in x2]
y = [-2.0e7 + 3 * (a - 1.0e7) + 0.5 * (b + 3.0e6) + 0.2 * c for a, b, c in zip(x1, x2, e)]
add("large_offsets", [x1, x2, y], [(2, [0, 1]), (2, [0]), (0, [2]), (2, [])])

json.dump(cases, sys.stdout, allow_nan=False)
print()
