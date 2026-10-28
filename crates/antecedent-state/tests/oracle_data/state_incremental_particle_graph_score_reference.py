# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Exact reference for the Gaussian BIC local graph score and its incremental total.

Regenerate with:  python3 state_incremental_particle_graph_score_reference.py > state_incremental_particle_graph_score_reference.json

The local score of node y given parents P on n rows is
    -n/2 (1 + ln(2 pi) + ln(RSS/n)) - (|P| + 1)/2 ln n
where RSS is the residual sum of squares of y on an intercept and P.  Every column is a dyadic
rational, so RSS is computed with Fractions: the columns are centered by their exact means, the
normal equations of the centered parents are solved exactly, and RSS = S_yy - S_yP S_PP^-1 S_Py.
The logarithms are evaluated by mpmath at 60 digits.  A parent set that repeats a column or
fits y with zero residual has no score (rank-deficient / zero variance); the case lists carry
those as refusals.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F
from mpmath import mp, mpf, log, pi

mp.dps = 60
_state = 0x6A09E667F3BCC908


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def dec(x):
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    x = F(x)
    return str(Decimal(x.numerator) / Decimal(x.denominator))


def dy(x, den=1 << 12):
    return F(int(round(x * den)), den)


def center(col):
    m = sum(col) / len(col)
    return [v - m for v in col]


def solve(a, b):
    n = len(b)
    a = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = next(r for r in range(c, n) if a[r][c] != 0)
        a[c], a[piv] = a[piv], a[c]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [a[i][n] / a[i][i] for i in range(n)]


def rss(cols, node, parents):
    y = center(cols[node])
    syy = sum(v * v for v in y)
    if not parents:
        return syy
    xs = [center(cols[p]) for p in parents]
    spp = [[sum(a * b for a, b in zip(u, v)) for v in xs] for u in xs]
    spy = [sum(a * b for a, b in zip(u, y)) for u in xs]
    beta = solve(spp, spy)
    return syy - sum(b * s for b, s in zip(beta, spy))


def score(cols, node, parents):
    n = len(cols[0])
    r = rss(cols, node, parents)
    if r == 0:
        return None
    k = len(parents) + 1
    val = -mpf(n) / 2 * (1 + log(2 * pi) + log(mpf(r.numerator) / mpf(r.denominator) / n)) - mpf(k) / 2 * log(n)
    return mp.nstr(val, 40)


def case(name, cols, ops):
    """ops: list of (node, parents) edits applied in order; the expected local scores are the
    exact scores before and after, and the exact total of the whole graph after each edit."""
    m = len(cols)
    parents = {j: [] for j in range(m)}
    steps = []
    for node, ps in ops:
        parents[node] = sorted(ps)
        local = {j: score(cols, j, parents[j]) for j in range(m)}
        steps.append({
            "node": node, "parents": sorted(ps),
            "local": local,
            "total": mp.nstr(sum(mpf(v) for v in local.values()), 40),
        })
    return {"name": name, "columns": [[dec(v) for v in c] for c in cols], "steps": steps}


cases = []
# chain with a large offset: the Schur complement of the raw Gram would cancel completely
n = 30
x0 = [dy(u01() * 8 - 4) + 10**9 for _ in range(n)]
x1 = [2 * (v - 10**9) + dy(u01() - 0.5) + 10**9 for v in x0]
x2 = [(v - 10**9) / 2 + dy(u01() * 2 - 1) + 10**9 for v in x1]
cases.append(case("offset_1e9_chain", [x0, x1, x2],
                  [(1, [0]), (2, [1]), (2, [0, 1]), (1, []), (0, [1, 2])]))
# two parents differing by 2^-18 jitter: nearly collinear
n = 40
a = [dy(u01() * 6 - 3, 1 << 20) for _ in range(n)]
b = [v + F(int((u01() - 0.5) * 8), 1 << 18) for v in a]
c = [3 * v - 2 * w + dy(u01() - 0.5, 1 << 10) for v, w in zip(a, b)]
cases.append(case("near_collinear_parents", [a, b, c], [(2, [0, 1]), (2, [0]), (2, [1]), (1, [0])]))
# a child fitted to 2^-17 noise (RSS is about 1e-11 of its centered sum of squares)
n = 100
p = [dy(u01() * 10 - 5, 1 << 14) for _ in range(n)]
q = [5 * v + F(int((u01() - 0.5) * 16), 1 << 17) for v in p]
cases.append(case("child_fitted_to_tiny_noise", [p, q], [(1, [0]), (0, [1])]))
# columns on scales 2^-40 and 2^40
n = 25
s = [dy(u01() * 4 - 2, 1 << 10) for _ in range(n)]
t = [F(2) ** 40 * (2 * v + dy(u01() - 0.5, 1 << 10)) for v in s]
u = [F(1, 2 ** 40) * (v + dy(u01() - 0.5, 1 << 10)) for v in s]
cases.append(case("unit_scales_2e40", [s, t, u], [(1, [0]), (2, [0]), (2, [0, 1]), (0, [1, 2])]))
# the smallest samples: n = 3 with one parent leaves one degree of freedom; n = 4 with two
cases.append(case("three_rows", [[F(0), F(1), F(3)], [F(1), F(5, 2), F(4)]], [(1, [0])]))
cases.append(case("four_rows_two_parents",
                  [[F(0), F(1), F(3), F(2)], [F(1), F(0), F(2), F(5)], [F(1), F(5, 2), F(4), F(1)]],
                  [(2, [0, 1])]))
# a ten variable graph, built edge by edge
n = 60
cols = []
for j in range(6):
    base = [dy(u01() * 4 - 2, 1 << 10) for _ in range(n)]
    if j:
        base = [v + (cols[j - 1][i] - sum(cols[j - 1]) / n) * F(j, 3) for i, v in enumerate(base)]
    cols.append(base)
cases.append(case("six_variable_edits", cols,
                  [(1, [0]), (2, [1]), (3, [1, 2]), (4, [3]), (5, [2, 4]), (3, [2]), (1, []), (5, [0, 1, 2, 3, 4])]))
print(json.dumps({"cases": cases}, indent=1))
