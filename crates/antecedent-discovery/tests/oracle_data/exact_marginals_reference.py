# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Exact edge marginals and normalizing constant of the Gaussian-BIC DAG posterior by brute force.

Regenerate with:  uv run --script exact_marginals_reference.py > exact_marginals_reference.json

Every labelled DAG on 4 or 5 nodes is enumerated.  The local score of a node with parent set Pa is
    -n/2 (1 + ln 2 pi + ln(RSS / n)) - (|Pa| + 1)/2 ln n,
RSS the residual sum of squares of the least-squares fit on an intercept and the parents, evaluated at
60 digits from the data (3-decimal values, so exactly representable as read).  The graph prior is
uniform over the valid DAGs, or independent Bernoulli(p) on every ordered pair that is not forbidden
(log p for a present edge, log(1-p) for an absent one); a DAG is valid when it has no forbidden edge,
all required edges, at most max_parents parents per node.  The posterior weight of a DAG is
exp(sum of local scores + log prior); the evidence is the sum of weights, the marginal of i -> j the
weight of the DAGs containing it over the evidence.
Repository-authored (MIT OR Apache-2.0).
"""
import itertools, json, random
from mpmath import mp, mpf, matrix, lu_solve, log, exp, pi

mp.dps = 60


def make(n_rows, seed, spec):
    r = random.Random(seed)
    cols = [[] for _ in spec]
    for _ in range(n_rows):
        row = []
        for j, (coefs, noise) in enumerate(spec):
            v = sum(c * row[i] for i, c in coefs) + noise * r.gauss(0, 1)
            row.append(round(v, 3))
        for j, v in enumerate(row):
            cols[j].append(v)
    return cols


def local_score(cols, node, parents):
    n = len(cols[0])
    X = [[mpf(str(v)) for v in cols[j]] for j in range(len(cols))]
    mean = [sum(c) / n for c in X]
    C = [[v - mean[j] for v in c] for j, c in enumerate(X)]
    syy = sum(v * v for v in C[node])
    if parents:
        k = len(parents)
        A = matrix(k, k); b = matrix(k, 1)
        for a, pa in enumerate(parents):
            b[a] = sum(u * v for u, v in zip(C[pa], C[node]))
            for c, pc in enumerate(parents):
                A[a, c] = sum(u * v for u, v in zip(C[pa], C[pc]))
        beta = lu_solve(A, b)
        rss = syy - sum(beta[a] * b[a] for a in range(k))
    else:
        rss = syy
    return -mpf(n) / 2 * (1 + log(2 * pi) + log(rss / n)) - mpf(len(parents) + 1) / 2 * log(n)


def acyclic(parent_sets):
    n = len(parent_sets)
    state = [0] * n
    def visit(v):
        state[v] = 1
        for p in parent_sets[v]:
            if state[p] == 1 or (state[p] == 0 and not visit(p)):
                return False
        state[v] = 2
        return True
    return all(state[v] == 2 or visit(v) for v in range(n))


def posterior(cols, p=None, max_parents=None, forbidden=(), required=()):
    n = len(cols)
    cap = n - 1 if max_parents is None else max_parents
    allowed = []
    for node in range(n):
        forb = {i for (i, j) in forbidden if j == node}
        req = {i for (i, j) in required if j == node}
        sets = []
        for r in range(0, n):
            for pa in itertools.combinations([i for i in range(n) if i != node], r):
                s = set(pa)
                if len(s) > cap or s & forb or not req <= s:
                    continue
                lp = mpf(0)
                if p is not None:
                    cand = n - 1 - len(forb)
                    lp = len(s) * log(mpf(str(p))) + (cand - len(s)) * log(1 - mpf(str(p)))
                sets.append((pa, local_score(cols, node, list(pa)) + lp))
        allowed.append(sets)
    evidence = mpf(0)
    marg = [[mpf(0)] * n for _ in range(n)]
    for combo in itertools.product(*allowed):
        if not acyclic([c[0] for c in combo]):
            continue
        w = exp(sum(c[1] for c in combo))
        evidence += w
        for node, (pa, _) in enumerate(combo):
            for i in pa:
                marg[i][node] += w
    return float(log(evidence)), [float(marg[i][j] / evidence) for i in range(n) for j in range(n)]


def case(name, cols, **prior):
    ev, m = posterior(cols, **prior)
    return {"name": name, "n_vars": len(cols), "columns": cols,
            "p": prior.get("p"), "max_parents": prior.get("max_parents"),
            "forbidden": [list(e) for e in prior.get("forbidden", ())],
            "required": [list(e) for e in prior.get("required", ())],
            "log_evidence": ev, "marginals": m}


chain = make(40, 5, [([], 1.0), ([(0, 0.8)], 0.7), ([(1, 0.7)], 0.7), ([(2, 0.6)], 0.8)])
strong = make(400, 9, [([], 1.0), ([], 1.0), ([(0, 0.9), (1, 0.9)], 0.15), ([(2, 0.8)], 0.15), ([(3, 0.7), (0, 0.5)], 0.15)])
collinear = make(60, 13, [([], 1.0), ([(0, 1.0)], 0.001), ([(0, 0.5)], 0.9), ([(1, 0.4)], 0.9)])
cases = [
    case("chain_uniform", chain),
    case("chain_bernoulli_cap_two", chain, p=0.2, max_parents=2),
    case("chain_forbidden_and_required", chain, forbidden=[(0, 1)], required=[(2, 3)]),
    case("five_nodes_concentrated_posterior", strong),
    case("five_nodes_bernoulli_forbidden", strong, p=0.5, forbidden=[(4, 0), (3, 1)]),
    case("near_collinear_parents", collinear),
]
print(json.dumps({"cases": cases}))
