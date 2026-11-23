# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Acyclicity function and global optimum of the NOTEARS objective, by exact definitions.

Regenerate with:  uv run --script notears_reference.py > notears_reference.json

acyclicity: h(W) = tr(exp(W o W)) - d and its gradient exp(W o W)^T o 2W, from mpmath's matrix
exponential at 40 digits, for matrices from a nilpotent (acyclic) one, through tiny and
unit-sized cycles, to entries of size 5 whose Hadamard square has 1-norm in the hundreds.

solver: for centered data X (n x d) the objective is  F(W) = (1/2n) ||X - XW||_F^2 + lam |W|_1  with
zero diagonal.  Because tr(exp(W o W)) = d exactly when the support of W is acyclic, the
constrained global minimum is  min over node orderings of  sum_j  min_{S subset of predecessors(j)}
f_j(S),  f_j(S) = min_w  (1/2) (G_jj - 2 w'G_Sj + w'G_SS w) + lam |w|_1  with G = X'X/n.  Each f_j(S)
is a small convex lasso, solved to 1e-30 by cyclic coordinate descent in mpmath; the orderings
are enumerated (d <= 5).  The minimising weights, the objective and the gap to the runner-up
support are recorded.  Data have exactly zero column means (every row appears with its negative),
so the product's uncentred objective equals the centred one.  Repository-authored
(MIT OR Apache-2.0).
"""
import itertools
import json

import mpmath as mp

mp.mp.dps = 40

_state = 0x1234567890ABCDEF


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def gauss():
    return sum(u01() for _ in range(12)) - 6.0


def q(x, bits=12):
    return round(x * (1 << bits)) / (1 << bits)


def h_and_grad(w, d):
    a = mp.matrix(d, d)
    for i in range(d):
        for j in range(d):
            a[i, j] = mp.mpf(w[i * d + j]) ** 2
    e = mp.expm(a)
    h = sum(e[i, i] for i in range(d)) - d
    g = [e[j, i] * 2 * mp.mpf(w[i * d + j]) for i in range(d) for j in range(d)]
    return h, g


def acyclicity_cases():
    cases = []

    def add(name, d, w):
        h, g = h_and_grad(w, d)
        cases.append({"name": name, "d": d, "w": w, "h": mp.nstr(h, 25), "grad": [mp.nstr(x, 25) for x in g]})

    d = 5
    nil = [0.0] * 25
    for i in range(d):
        for j in range(i + 1, d):
            nil[i * d + j] = q(0.9 * gauss())
    add("upper_triangular_acyclic", d, nil)
    add("two_cycle_tiny", 3, [0, 1e-3, 0, 2e-3, 0, 0, 0, 0, 0])
    add("two_cycle_unit", 3, [0, 1.0, 0, 0.5, 0, 0, 0, 0, 0])
    add("three_cycle_mixed", 3, [0, 0.7, 0, 0, 0, -1.1, 0.9, 0, 0])
    dense = [0.0 if i == j else q(gauss()) for i in range(6) for j in range(6)]
    add("dense_six", 6, dense)
    big = [0.0 if i == j else q(5 * (u01() - 0.5) * 2) for i in range(4) for j in range(4)]
    add("large_entries", 4, big)
    wide = [0.0 if i == j else q(0.4 * gauss()) for i in range(12) for j in range(12)]
    add("twelve_variables", 12, wide)
    return cases


def lasso(g, j, s, lam):
    """min_w  1/2 (G_jj - 2 w'G_sj + w'G_ss w) + lam |w|_1 over the support s (list of indices)."""
    k = len(s)
    if k == 0:
        return g[j][j] / 2, []
    w = [mp.mpf(0)] * k
    for _ in range(20000):
        change = mp.mpf(0)
        for a in range(k):
            rho = g[s[a]][j] - sum(g[s[a]][s[b]] * w[b] for b in range(k) if b != a)
            if rho > lam:
                new = (rho - lam) / g[s[a]][s[a]]
            elif rho < -lam:
                new = (rho + lam) / g[s[a]][s[a]]
            else:
                new = mp.mpf(0)
            change = max(change, abs(new - w[a]))
            w[a] = new
        if change < mp.mpf(10) ** -32:
            break
    f = g[j][j] / 2 - sum(w[a] * g[s[a]][j] for a in range(k)) + sum(w[a] * sum(g[s[a]][s[b]] * w[b] for b in range(k)) for a in range(k)) / 2 + lam * sum(abs(x) for x in w)
    return f, w


def solver_case(name, d, edges, n_half, lam, noise=1.0):
    """Linear SEM over the given (parent, child, weight) edges in topological order 0..d-1."""
    rows = []
    for _ in range(n_half):
        x = [0.0] * d
        for j in range(d):
            x[j] = noise * gauss() + sum(wt * x[p] for p, c, wt in edges if c == j)
        rows.append([q(v) for v in x])
    rows = rows + [[-v for v in r] for r in rows]
    n = len(rows)
    cols = [[rows[r][j] for r in range(n)] for j in range(d)]
    g = [[sum(mp.mpf(cols[i][r]) * mp.mpf(cols[j][r]) for r in range(n)) / n for j in range(d)] for i in range(d)]
    f = {}
    for j in range(d):
        others = [i for i in range(d) if i != j]
        for k in range(len(others) + 1):
            for s in itertools.combinations(others, k):
                f[(j, s)] = lasso(g, j, list(s), mp.mpf(lam))
    best = None
    scores = []
    for order in itertools.permutations(range(d)):
        total = mp.mpf(0)
        chosen = {}
        for pos, j in enumerate(order):
            preds = order[:pos]
            cand = None
            for k in range(len(preds) + 1):
                for s in itertools.combinations(sorted(preds), k):
                    val = f[(j, s)][0]
                    if cand is None or val < cand[0]:
                        cand = (val, s)
            total += cand[0]
            chosen[j] = cand[1]
        scores.append((total, chosen))
    scores.sort(key=lambda t: t[0])
    best_total, chosen = scores[0]
    # runner-up with a different weight pattern
    def pattern(c):
        return tuple(sorted((j, tuple(c[j])) for j in c))
    gap = None
    for total, c in scores[1:]:
        if pattern(c) != pattern(chosen):
            gap = total - best_total
            break
    w = [0.0] * (d * d)
    for j, s in chosen.items():
        _, ws = f[(j, s)]
        for a, p in enumerate(s):
            w[p * d + j] = float(ws[a])
    return {
        "name": name, "d": d, "n": n, "lam": lam, "cols": cols,
        "gram": [[mp.nstr(g[i][j], 30) for j in range(d)] for i in range(d)],
        "objective": mp.nstr(best_total, 30), "weights": w,
        "runner_up_gap": None if gap is None else float(gap),
    }


def main():
    solver = [
        solver_case("chain_three_no_penalty", 3, [(0, 1, 0.9), (1, 2, -0.8)], 30, 0.0),
        solver_case("chain_three_penalised", 3, [(0, 1, 0.9), (1, 2, -0.8)], 30, 0.1),
        solver_case("collider_four_penalised", 4, [(0, 2, 0.9), (1, 2, 0.8), (2, 3, -0.7)], 40, 0.05),
        solver_case("diamond_four_no_penalty", 4, [(0, 1, 0.8), (0, 2, -0.7), (1, 3, 0.9), (2, 3, 0.6)], 40, 0.0),
        solver_case("five_variable_mixed", 5, [(0, 1, 0.9), (1, 2, 0.8), (0, 3, -0.7), (3, 4, 0.9), (2, 4, 0.5)], 50, 0.1),
        solver_case("weak_signal_heavy_penalty", 4, [(0, 1, 0.3), (1, 2, 0.25), (2, 3, 0.2)], 40, 0.2),
    ]
    print(json.dumps({"acyclicity": acyclicity_cases(), "solver": solver}))


main()
