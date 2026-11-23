"""Generic identifiability of the direct effects into each target, by exact rank.

For a fully observed linear system x = B x (zero diagonal) and an experiment set (each
experiment clamps J and leaves U passive), the experimental effect of clamping j on the
passive units is T = (I - B_UU)^-1 B_UJ and the direct effects b_t. of a passive target t
satisfy, for every probe j in J,

    b_tj + sum_{u in U, u != t} T[u][j] b_tu = T[t][j]            (derived by differentiating
                                                                   x_t = sum_s B_ts x_s in x_j)

The stacked rows over all experiments in which t is passive form a matrix with n-1 unknown columns.
Target t is determined exactly when that matrix has rank n-1 (exact Fraction elimination). Two fixed
dense rational matrices B stand in for a generic system; a target counts as determined only if both
agree. The product's pair-coverage verdict is compared with this rank for every experiment set of
at most three experiments over three and four units.

    python3 experimental_recovery_reference.py > experimental_recovery_reference.json
"""

import itertools
import json
from fractions import Fraction as F


def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def rank(rows):
    m = [r[:] for r in rows]
    rk = 0
    cols = len(m[0]) if m else 0
    for c in range(cols):
        piv = next((r for r in range(rk, len(m)) if m[r][c] != 0), None)
        if piv is None:
            continue
        m[rk], m[piv] = m[piv], m[rk]
        for r in range(len(m)):
            if r != rk and m[r][c] != 0:
                f = m[r][c] / m[rk][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[rk])]
        rk += 1
    return rk


def system(n, seed):
    s = seed
    b = [[F(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                s = (s * 6364136223846793005 + 1442695040888963407) % 2**64
                b[i][j] = F((s >> 33) % 17 - 8 or 3, 11)
    return b


def rows_for(b, n, clamped, target):
    u = [i for i in range(n) if i not in clamped]
    a = [[(F(1) if i == j else F(0)) - b[u[i]][u[j]] for j in range(len(u))] for i in range(len(u))]
    out = []
    srcs = [s for s in range(n) if s != target]
    for j in clamped:
        t_col = solve(a, [b[x][j] for x in u])
        tv = {x: t_col[k] for k, x in enumerate(u)}
        row = []
        for s in srcs:
            if s == j:
                row.append(F(1))
            elif s in clamped:
                row.append(F(0))
            else:
                row.append(tv[s])
        # self-check: the true direct effects satisfy the row
        lhs = sum(r * b[target][s] for r, s in zip(row, srcs))
        assert lhs == tv[target], "row not satisfied by B"
        out.append(row)
    return out


def determined(b, n, experiments, target):
    rows = []
    for clamped in experiments:
        if target in clamped:
            continue
        rows += rows_for(b, n, clamped, target)
    return bool(rows) and rank(rows) == n - 1


cases = []
for n in (3, 4):
    systems = [system(n, 7), system(n, 91)]
    designs = [tuple(c) for k in range(1, n) for c in itertools.combinations(range(n), k)]
    for size in range(1, 4):
        for exps in itertools.combinations(designs, size):
            det = []
            for t in range(n):
                v = [determined(b, n, exps, t) for b in systems]
                assert v[0] == v[1], (n, exps, t)
                if v[0]:
                    det.append(t)
            cases.append({"units": n, "clamped": [list(e) for e in exps], "determined": det})
print(json.dumps({"cases": cases}, separators=(",", ":")))
