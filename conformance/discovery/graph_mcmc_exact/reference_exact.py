# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Independent exhaustive Gaussian-BIC DAG posterior over the retained fixture data.

Pure standard library.  For every labelled DAG on the case's variables it scores each node
by the textbook Gaussian log-likelihood at the maximum-likelihood fit, minus (k/2) ln n with
k = |parents| + 1 (intercept; the shared noise-variance parameter is the same for every graph
and cancels in the posterior), adds the graph prior, and normalises.  Least squares is solved
by Gaussian elimination with partial pivoting, a different algorithm from the product's
equilibrated Cholesky on a cached Gram matrix.

    uv run --script reference_exact.py            # prints max |difference| to the fixture
    uv run --script reference_exact.py --emit     # prints edge marginals as JSON

SPDX-License-Identifier: MIT OR Apache-2.0
"""

import itertools
import json
import math
import sys
from pathlib import Path

FIXTURE = Path(__file__).with_name("expected.json")


def solve(a, b):
    k = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, k):
            f = m[r][c] / m[c][c]
            for j in range(c, k + 1):
                m[r][j] -= f * m[c][j]
    x = [0.0] * k
    for r in range(k - 1, -1, -1):
        x[r] = (m[r][k] - sum(m[r][j] * x[j] for j in range(r + 1, k))) / m[r][r]
    return x


def local_score(cols, node, parents):
    n = len(cols[0])
    design = [[1.0] * n] + [cols[p] for p in parents]
    gram = [[sum(u * v for u, v in zip(a, b)) for b in design] for a in design]
    rhs = [sum(u * v for u, v in zip(a, cols[node])) for a in design]
    beta = solve(gram, rhs)
    rss = 0.0
    for i in range(n):
        fit = sum(beta[j] * design[j][i] for j in range(len(design)))
        rss += (cols[node][i] - fit) ** 2
    sigma2 = rss / n
    k = len(parents) + 1
    return -0.5 * n * (1.0 + math.log(2.0 * math.pi) + math.log(sigma2)) - 0.5 * k * math.log(n)


def is_dag(n, edges):
    indeg = [0] * n
    for _, j in edges:
        indeg[j] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    while ready:
        i = ready.pop()
        seen += 1
        for a, b in edges:
            if a == i:
                indeg[b] -= 1
                if indeg[b] == 0:
                    ready.append(b)
    return seen == n


def posterior(case, edge_p):
    names = case["var_names"]
    n = len(names)
    cols = [[row[c] for row in case["data"]] for c in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    local = {}
    logs = []
    graphs = []
    for bits in itertools.product([0, 1], repeat=len(pairs)):
        edges = [p for p, b in zip(pairs, bits) if b]
        if not is_dag(n, edges):
            continue
        total = 0.0
        for node in range(n):
            ps = tuple(sorted(i for i, j in edges if j == node))
            if (node, ps) not in local:
                local[(node, ps)] = local_score(cols, node, ps)
            total += local[(node, ps)]
        if edge_p is not None:
            total += len(edges) * math.log(edge_p) + (len(pairs) - len(edges)) * math.log(1 - edge_p)
        logs.append(total)
        graphs.append(edges)
    top = max(logs)
    weights = [math.exp(v - top) for v in logs]
    z = sum(weights)
    marginals = {f"{names[i]}->{names[j]}": 0.0 for i, j in pairs}
    for edges, w in zip(graphs, weights):
        for i, j in edges:
            marginals[f"{names[i]}->{names[j]}"] += w / z
    return len(graphs), marginals


def main():
    fixture = json.loads(FIXTURE.read_text())
    emit = {}
    worst = 0.0
    for case in fixture["cases"]:
        for key, p in (("uniform_all_dags", None), ("bernoulli_0_2_all_dags", 0.2)):
            count, marg = posterior(case, p)
            target = case["targets"][key]
            assert count == target["n_dags"], (case["name"], count)
            emit[f"{case['name']}/{key}"] = marg
            for edge, value in marg.items():
                worst = max(worst, abs(value - target["edge_marginals"][edge]))
    if "--emit" in sys.argv:
        print(json.dumps(emit, indent=1, sort_keys=True))
    else:
        print(f"max |reference - fixture| over all edge marginals: {worst:.3e}")
        assert worst < 1e-9, worst


main()
