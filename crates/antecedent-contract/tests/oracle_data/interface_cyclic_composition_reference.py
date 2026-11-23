"""Definition-level oracle for feed-forward ordering and cycle refusal of a component graph.

Regenerate with:  python3 interface_cyclic_composition_reference.py > interface_cyclic_composition_reference.txt

For every digraph on a fixed vertex set the oracle decides, from the definition alone:
  * the graph is cyclic iff some vertex reaches itself by a path of one or more edges
    (transitive closure by repeated boolean matrix squaring, no ordering involved);
  * if acyclic, the answer is the lexicographically smallest permutation of the vertices in
    which every edge goes from an earlier vertex to a later one (found by scanning all
    permutations in lexicographic order).
Output lines are "<mask> <order>" or "<mask> -" (cyclic). Edge k of the listing is bit k of the mask.
Vertex sets: 3 vertices with self-loops allowed (9 edges, 512 graphs) and 4 vertices without
self-loops (12 edges, 4096 graphs).

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import itertools

NAMES = "abcd"


def reach(n, edges):
    r = [[False] * n for _ in range(n)]
    for i, j in edges:
        r[i][j] = True
    for _ in range(n + 1):
        r = [[r[i][j] or any(r[i][k] and r[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return r


def solve(n, edges):
    r = reach(n, edges)
    if any(r[i][i] for i in range(n)):
        return "-"
    for perm in itertools.permutations(range(n)):
        pos = {v: p for p, v in enumerate(perm)}
        if all(pos[i] < pos[j] for i, j in edges):
            return "".join(NAMES[v] for v in perm)
    raise AssertionError("acyclic graph without a topological order")


def block(n, self_loops):
    listing = [(i, j) for i in range(n) for j in range(n) if self_loops or i != j]
    print(f"# n={n} self_loops={int(self_loops)} edges={len(listing)}")
    for mask in range(1 << len(listing)):
        edges = [e for k, e in enumerate(listing) if mask >> k & 1]
        print(mask, solve(n, edges))


block(3, True)
block(4, False)
