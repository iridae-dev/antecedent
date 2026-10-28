# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Parent-set acceptance over every marked graph on four nodes, by definition.

Regenerate with:  uv run --script sample_split_parent_set_effect_reference.py > sample_split_parent_set_effect_reference.json

Each unordered pair of the four nodes is absent ('.'), undirected ('-'), directed from the smaller
to the larger node ('>') or from the larger to the smaller ('<'): 4^6 = 4096 patterns, those with a
directed cycle dropped.  For a treatment t and an outcome o, the parent set of t is accepted when
every edge incident to t is directed (an undirected edge there leaves graphs with different parent
sets); the parents are the sources of the directed edges into t, the children the sinks of the
directed edges out of t, and the undirected edges not incident to t are listed unresolved, never
completed; an outcome that is a parent of t is refused after the ambiguity check.  The fixed-graph
effect route is stricter: any undirected edge anywhere refuses, then an outcome parent refuses.
Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json

N = 4
PAIRS = list(itertools.combinations(range(N), 2))


def edges_of(pattern):
    out = []
    for (a, b), c in zip(PAIRS, pattern):
        if c == '>':
            out.append(('d', a, b))
        elif c == '<':
            out.append(('d', b, a))
        elif c == '-':
            out.append(('u', a, b))
    return out


def acyclic(directed):
    reach = {v: set() for v in range(N)}
    for _, a, b in directed:
        reach[a].add(b)
    changed = True
    while changed:
        changed = False
        for v in range(N):
            new = set().union(*[reach[w] for w in reach[v]]) if reach[v] else set()
            if not new <= reach[v]:
                reach[v] |= new
                changed = True
    return all(v not in reach[v] for v in range(N))


def expect(pattern, t, o):
    es = edges_of(pattern)
    parents = sorted(a for k, a, b in es if k == 'd' and b == t)
    children = sorted(b for k, a, b in es if k == 'd' and a == t)
    ambiguous = [(a, b) for k, a, b in es if k == 'u' and t in (a, b)]
    unresolved = [(a, b) for k, a, b in es if k == 'u' and t not in (a, b)]
    directed = sorted((a, b) for k, a, b in es if k == 'd')
    if ambiguous:
        parent_set = "ambiguous"
    elif o in parents:
        parent_set = "outcome_precedes_treatment"
    else:
        parent_set = {"parents": parents, "children": children, "unresolved": unresolved,
                      "directed": directed, "edges": len(es)}
    if any(k == 'u' for k, _, _ in es):
        fixed = "unresolved"
    elif o in parents:
        fixed = "outcome_precedes_treatment"
    else:
        fixed = {"parents": parents}
    return parent_set, fixed


rows = []
for pattern in itertools.product('.-<>', repeat=len(PAIRS)):
    es = edges_of(pattern)
    if not acyclic([e for e in es if e[0] == 'd']):
        continue
    for t, o in [(1, 3), (3, 0)]:
        ps, fx = expect(pattern, t, o)
        rows.append({"pattern": ''.join(pattern), "treatment": t, "outcome": o, "parent_set": ps, "fixed": fx})
print(json.dumps(rows, separators=(',', ':')))
