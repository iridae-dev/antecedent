# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Permutation falsification of a DAG: violation counts, p-value and equivalence share, exactly.

Regenerate with:  uv run --script graph_falsification_reference.py > graph_falsification_reference.json

Written from the definitions, with mpmath at 60 digits (the doubles in the data are taken as
exact).  For a DAG on nodes 0..n-1 and data columns c_0..c_{n-1}:
  implications  for every node i and every node j that is not i, not a parent of i and not a
                descendant of i:  c_i independent of c_j given the columns of parents(i)
  test          the partial correlation r of the residuals of c_i and c_j after least-squares
                regression on an intercept and the conditioning columns; t = r sqrt(df/(1-r^2)),
                df = rows - 2 - |Z|; two-sided Student-t p-value  I_{df/(df+t^2)}(df/2, 1/2)
  violation     p < alpha
  relabelling   a permutation pi of the nodes: implication (i, j | Z) is tested on the columns
                (pi(i), pi(j) | pi(Z)); the identity is excluded, every other permutation is used
                (the node counts here keep n! below the request)
  p-value       (1 + #{relabellings with at most as many violations as the graph}) / (1 + #relabellings)
  equivalent    relabellings whose edge set has the same skeleton and the same unshielded
                colliders as the graph (Verma and Pearl); share = (1 + #equivalent)/(1 + #relabellings)
  outcome       Uninformative when there is no implication or share > significance; otherwise
                Rejected when the p-value exceeds the significance, else NotRejected
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import itertools
import json
import random

import mpmath as mp

mp.mp.dps = 60
mpf = mp.mpf
ALPHA = 0.05
SIGNIFICANCE = 0.05


def residuals(y, zs):
    n = len(y)
    cols = [[mpf(1)] * n] + [[mpf(v) for v in z] for z in zs]
    A = mp.matrix(n, len(cols))
    for j, c in enumerate(cols):
        for i in range(n):
            A[i, j] = c[i]
    b = mp.matrix([mpf(v) for v in y])
    coef = mp.lu_solve(A.T * A, A.T * b)
    res = b - A * coef
    return [res[i] for i in range(n)]


def parcorr(cols, i, j, z):
    n = len(cols[0])
    rx = residuals(cols[i], [cols[k] for k in z])
    ry = residuals(cols[j], [cols[k] for k in z])
    sxy = sum(a * b for a, b in zip(rx, ry))
    r = sxy / mp.sqrt(sum(a * a for a in rx) * sum(b * b for b in ry))
    df = n - 2 - len(z)
    t2 = r * r * df / (1 - r * r)
    p = mp.betainc(mpf(df) / 2, mpf(1) / 2, 0, mpf(df) / (df + t2), regularized=True)
    return r, p


def descendants(edges, n, i):
    out, stack = set(), [i]
    while stack:
        u = stack.pop()
        for a, b in edges:
            if a == u and b not in out:
                out.add(b)
                stack.append(b)
    return out


def implications(edges, n):
    out = []
    for i in range(n):
        parents = sorted(a for a, b in edges if b == i)
        desc = descendants(edges, n, i)
        for j in range(n):
            if j == i or j in parents or j in desc:
                continue
            out.append((i, j, parents))
    return out


def signature(edges, n):
    skel = {(min(a, b), max(a, b)) for a, b in edges}
    colliders = set()
    for c in range(n):
        ps = sorted(a for a, b in edges if b == c)
        for x, y in itertools.combinations(ps, 2):
            if (x, y) not in skel:
                colliders.add((x, c, y))
    return skel, colliders


def design(name, n, edges, cols, max_share=None):
    cache = {}

    def test(a, b, z):
        key = (min(a, b), max(a, b), tuple(sorted(z)))
        if key not in cache:
            cache[key] = parcorr(cols, key[0], key[1], list(key[2]))
        return cache[key]

    imps = implications(edges, n)
    ident = list(range(n))

    def violations(perm):
        c = 0
        for i, j, z in imps:
            if test(perm[i], perm[j], [perm[k] for k in z])[1] < ALPHA:
                c += 1
        return c

    threshold = SIGNIFICANCE if max_share is None else max_share
    out = {"name": name, "n": n, "edges": edges, "columns": cols, "n_implications": len(imps),
           "max_equivalent_share": max_share}
    if not imps:
        out.update(outcome="uninformative", reason="no_testable_implication")
        return out
    sig = signature(edges, n)
    perms = [list(p) for p in itertools.permutations(range(n)) if list(p) != ident]
    equivalent = sum(1 for p in perms if signature([(p[a], p[b]) for a, b in edges], n) == sig)
    share = (1 + equivalent) / (1 + len(perms))
    out.update(equivalent=equivalent, share=share, permutations=len(perms))
    if share > threshold:
        out.update(outcome="uninformative", reason="equivalence_class_share")
        return out
    observed = violations(ident)
    counts = sorted(violations(p) for p in perms)
    at_most = sum(1 for c in counts if c <= observed)
    pv = (1 + at_most) / (1 + len(counts))
    out.update(
        observed_violations=observed,
        permutation_violations=counts,
        p_value=pv,
        outcome="rejected" if pv > SIGNIFICANCE else "not_rejected",
        evidence=[{"node": i, "other": j, "conditioning": z,
                   "statistic": float(test(i, j, z)[0]), "p_value": float(test(i, j, z)[1])} for i, j, z in imps],
    )
    return out


def gen(rng, n_rows, spec):
    """spec: list of (intercept, [(parent, coef)], sd) per node in index order (parents precede)."""
    cols = [[] for _ in spec]
    for _ in range(n_rows):
        row = []
        for k, (b, par, sd) in enumerate(spec):
            row.append(b + sum(c * row[p] for p, c in par) + rng.gauss(0, sd))
        for k, v in enumerate(row):
            cols[k].append(v)
    return cols


def main():
    rng = random.Random(7)
    out = []
    fork = [(0, 1), (1, 2), (1, 3)]
    # data from the graph itself
    cols = gen(rng, 80, [(0, [], 1), (0, [(0, 0.9)], 1), (0, [(1, 0.8)], 1), (0, [(1, -0.7)], 1)])
    out.append(design("fork_true_data", 4, fork, cols, 0.5))
    # same data, shifted by 1e8 and scaled by 1e-6 per column: partial correlation is invariant
    shifted = [[1e8 + 1e-6 * v for v in c] for c in cols]
    out.append(design("fork_true_data_offset_scaled", 4, fork, shifted, 0.5))
    # data in which every pair is nearly independent: the graph implies independencies that hold,
    # but so does every relabelling
    indep = gen(rng, 80, [(0, [], 1)] * 4)
    out.append(design("independent_columns", 4, fork, indep, 0.5))
    # data from a different graph (3 -> 2 -> 1 -> 0 plus 3 -> 0): the fork graph's implications fail
    other = gen(rng, 80, [(0, [], 1), (0, [], 1), (0, [], 1), (0, [], 1)])
    chain = gen(rng, 80, [(0, [], 1), (0, [(0, 1.2)], 0.6), (0, [(1, 1.2)], 0.6), (0, [(2, 1.2)], 0.6)])
    out.append(design("fork_graph_chain_data", 4, fork, chain[::-1], 0.5))
    # collider graph with its own data
    coll = gen(rng, 60, [(0, [], 1), (0, [], 1), (0, [(0, 1.0), (1, 1.0)], 0.7), (0, [(2, 0.9)], 1)])
    out.append(design("collider_true_data", 4, [(0, 2), (1, 2), (2, 3)], coll, 0.5))
    # a node duplicated to within 1e-4: near-perfect collinearity in the conditioning set
    base = gen(rng, 50, [(0, [], 1), (0, [(0, 1.0)], 0.5), (0, [(0, 0.5)], 1)])
    near = base + [[v + rng.gauss(0, 1e-4) for v in base[1]]]
    out.append(design("near_collinear_duplicate", 4, [(0, 1), (0, 2), (1, 3)], near, 0.5))
    # the default threshold (the significance level) calls the same small graph uninformative
    out.append(design("fork_default_threshold", 4, fork, cols))
    # five nodes: 119 relabellings, none Markov equivalent, so the default threshold applies
    five = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    good = gen(rng, 120, [(0, [], 1), (0, [(0, 0.9)], 1), (0, [(0, -0.8)], 1), (0, [(1, 0.7), (2, 0.7)], 1), (0, [(3, 0.9)], 1)])
    out.append(design("five_node_true_data", 5, five, good))
    drop = [[v for v in good[0]], good[1], good[2], good[3], [rng.gauss(0, 1) for _ in good[0]]]
    out.append(design("five_node_wrong_data", 5, five, [good[4], good[3], good[0], good[1], good[2]]))
    # tiny sample: 7 rows, 5 nodes, up to two conditioning columns, 3 residual degrees of freedom
    tiny = gen(rng, 7, [(0, [], 1), (0, [(0, 1.0)], 1), (0, [(0, 0.8)], 1), (0, [(1, 1), (2, 1)], 1), (0, [], 1)])
    out.append(design("tiny_sample", 5, [(0, 1), (0, 2), (1, 3), (2, 3)], tiny))
    # a complete graph implies nothing testable
    out.append(design("complete_graph", 3, [(0, 1), (0, 2), (1, 2)], gen(rng, 30, [(0, [], 1)] * 3)))
    # a chain's relabellings are largely Markov equivalent to it
    out.append(design("chain_equivalence", 3, [(0, 1), (1, 2)], gen(rng, 30, [(0, [], 1), (0, [(0, 1)], 1), (0, [(1, 1)], 1)])))
    print(json.dumps(out))


main()
