# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Directed-path products of linear structural models, against the exact total effect.

Regenerate with:  uv run --script path_reference.py > path_reference.json

For a linear model Y = B^T Y + noise the total effect of node a on node b is the (b, a) entry of
(I - B^T)^-1, and equals the sum over directed paths a -> ... -> b of the product of edge
coefficients (Wright's rule for unstandardized coefficients).  Both are computed here in exact
rational arithmetic: the path enumeration lists every product, and a Gauss-Jordan inversion
gives the total effect; they must agree before a case is written.  Edges are listed in a
shuffled order so a consumer cannot rely on index order.  Repository-authored
(MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction

_state = 0xA24BAED4963EE407


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def total_effects(n, edges):
    # M = I - B where B[child][parent] = beta; total effect matrix = M^-1 - I... use M^-1 itself
    m = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for (p, c), b in edges.items():
        m[c][p] -= b
    inv = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        inv[col], inv[piv] = inv[piv], inv[col]
        d = m[col][col]
        m[col] = [x / d for x in m[col]]
        inv[col] = [x / d for x in inv[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
                inv[r] = [a - f * b for a, b in zip(inv[r], inv[col])]
    return inv


def paths(n, edges, src, dst):
    out = []

    def go(node, trail, prod):
        if node == dst:
            out.append((list(trail), prod))
            return
        for (p, c), b in edges.items():
            if p == node:
                go(c, trail + [c], prod * b)

    go(src, [src], Fraction(1))
    return out


def case(name, n, edge_list, src, dst):
    edges = {(p, c): Fraction(b) for p, c, b in edge_list}
    total = total_effects(n, edges)[dst][src]
    ps = paths(n, edges, src, dst)
    assert sum(pr for _, pr in ps) == total, name
    return {
        "name": name,
        "n": n,
        "edges": [[p, c, b] for p, c, b in edge_list],
        "source": src,
        "outcome": dst,
        "paths": [{"path": t, "product": float(pr)} for t, pr in ps],
        "total": float(total),
    }


def main():
    cases = []
    # chain, as in the pinned conformance case
    cases.append(case("chain", 3, [(0, 1, 2.0), (1, 2, 3.0)], 0, 2))
    # diamond with exactly cancelling branches: 0->1->3 (+2*3) and 0->2->3 (-3*2)
    cases.append(
        case("cancelling_diamond", 4, [(2, 3, 2.0), (0, 1, 2.0), (1, 3, 3.0), (0, 2, -3.0)], 0, 3)
    )
    # edges inserted so a child's parent order differs from node order (collider at 4)
    cases.append(
        case(
            "unsorted_parents",
            5,
            [(3, 4, 0.5), (1, 4, -1.25), (0, 4, 7.0), (2, 4, 0.125), (1, 3, 2.0), (0, 1, 3.0), (0, 2, 4.0)],
            0,
            4,
        )
    )
    # random dense DAG on 8 nodes (topological order = index order), quarter-integer coefficients
    edges = []
    for c in range(1, 8):
        for p in range(c):
            if u01() < 0.7:
                edges.append((p, c, round((u01() * 8 - 4) * 4) / 4 or 0.25))
    for k in range(len(edges) - 1, 0, -1):
        j = int(u01() * (k + 1))
        edges[k], edges[j] = edges[j], edges[k]
    cases.append(case("dense_8", 8, edges, 0, 7))
    # a source with no directed path to the outcome contributes nothing
    cases.append(case("no_path", 3, [(0, 1, 1.5), (2, 1, 2.5)], 0, 2))
    print(json.dumps({"cases": cases}, separators=(",", ":")))


main()
