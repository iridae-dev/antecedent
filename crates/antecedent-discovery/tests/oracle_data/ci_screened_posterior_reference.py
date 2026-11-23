# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Exact edge marginals of the CI-screened Gaussian-BIC DAG posterior, by brute force.

Regenerate with:  uv run --script ci_screened_posterior_reference.py > ci_screened_posterior_reference.json

For each case this script (1) screens the pairs of four variables the way a conditional-independence
screen is defined: a pair is dropped when some conditioning set of at most three of the other
variables has a Student-t partial-correlation p-value above alpha (no multiplicity correction), a
pair is kept when no such set exists, a pair whose both directions are forbidden is never a
candidate and a pair with a required direction is always a candidate; the script asserts the decisions are not marginal, so the screen does not
depend on the order in which a search visits pairs (a dropped pair has a set with p of at least 3 alpha, a kept pair has every set below alpha / 20); (2) enumerates every labelled DAG on the four
nodes whose directed edges lie on candidate pairs, avoid the forbidden directed edges, contain the
required ones and give every node at most `max_parents` parents; and (3) weights each by
exp(sum of 60-digit Gaussian-BIC local scores + log prior), the prior uniform or independent
Bernoulli(p) on every ordered pair, and sums edge marginals.  The constraints of the engine and of
the caller are given separately and merged here as set unions with the smaller parent cap.  The
floor for a screened-out pair is min(p, alpha) / 2 on both cells.
Repository-authored (MIT OR Apache-2.0).
"""
import itertools, json, random
from mpmath import mp, mpf, matrix, lu_solve, log, exp, pi, betainc, sqrt

mp.dps = 60
N = 4
ALPHA = 0.05


def make(n_rows, seed, spec):
    r = random.Random(seed)
    cols = [[] for _ in spec]
    for _ in range(n_rows):
        row = []
        for coefs, noise in spec:
            row.append(round(sum(c * row[i] for i, c in coefs) + noise * r.gauss(0, 1), 3))
        for j, v in enumerate(row):
            cols[j].append(v)
    return cols


def centred(cols):
    X = [[mpf(str(v)) for v in c] for c in cols]
    return [[v - sum(c) / len(c) for v in c] for c in X]


def partial_p(cols, x, y, z):
    C = centred(cols)
    n = len(C[0])

    def resid(t):
        if not z:
            return C[t]
        k = len(z)
        A = matrix(k, k); b = matrix(k, 1)
        for a, pa in enumerate(z):
            b[a] = sum(u * v for u, v in zip(C[pa], C[t]))
            for c, pc in enumerate(z):
                A[a, c] = sum(u * v for u, v in zip(C[pa], C[pc]))
        beta = lu_solve(A, b)
        return [C[t][i] - sum(beta[a] * C[z[a]][i] for a in range(k)) for i in range(n)]

    rx, ry = resid(x), resid(y)
    r = sum(u * v for u, v in zip(rx, ry)) / sqrt(sum(u * u for u in rx) * sum(v * v for v in ry))
    df = n - len(z) - 2
    t2 = r * r * df / (1 - r * r)
    return float(betainc(mpf(df) / 2, mpf(1) / 2, 0, df / (df + t2), regularized=True))


def screen(cols, forbidden, required):
    pairs = []
    for a, b in itertools.combinations(range(N), 2):
        if (a, b) in forbidden and (b, a) in forbidden:
            continue
        if (a, b) in required or (b, a) in required:
            pairs.append((a, b))
            continue
        others = [v for v in range(N) if v not in (a, b)]
        ps = [partial_p(cols, a, b, list(z)) for r in range(0, 4) for z in itertools.combinations(others, r)]
        if max(ps) > ALPHA:
            # dropped: some conditioning set clearly does not reject independence
            assert max(ps) >= 3 * ALPHA, (a, b, max(ps))
        else:
            # kept: every conditioning set rejects independence with a wide margin
            assert max(ps) <= ALPHA / 20, (a, b, max(ps))
            pairs.append((a, b))
    return pairs


def local_score(cols, node, parents):
    C = centred(cols)
    n = len(C[0])
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
    return -mpf(n) / 2 * (1 + log(2 * pi) + log(rss / n)) - mpf(len(parents) + 2) / 2 * log(n)


def acyclic(parent_sets):
    state = [0] * N
    def visit(v):
        state[v] = 1
        for p in parent_sets[v]:
            if state[p] == 1 or (state[p] == 0 and not visit(p)):
                return False
        state[v] = 2
        return True
    return all(state[v] == 2 or visit(v) for v in range(N))


def posterior(cols, pairs, forbidden, required, cap, p):
    cand = {(a, b) for a, b in pairs} | {(b, a) for a, b in pairs}
    allowed = []
    for node in range(N):
        req = {i for (i, j) in required if j == node}
        sets = []
        for r in range(0, N):
            for pa in itertools.combinations([i for i in range(N) if i != node], r):
                s = set(pa)
                if len(s) > cap or any((i, node) not in cand or (i, node) in forbidden for i in s):
                    continue
                if not req <= s:
                    continue
                lp = mpf(0)
                if p is not None:
                    lp = len(s) * log(mpf(str(p))) + (N - 1 - len(s)) * log(1 - mpf(str(p)))
                sets.append((pa, local_score(cols, node, list(pa)) + lp))
        allowed.append(sets)
    evidence = mpf(0)
    marg = [[mpf(0)] * N for _ in range(N)]
    for combo in itertools.product(*allowed):
        if not acyclic([c[0] for c in combo]):
            continue
        w = exp(sum(c[1] for c in combo))
        evidence += w
        for node, (pa, _) in enumerate(combo):
            for i in pa:
                marg[i][node] += w
    return [float(marg[i][j] / evidence) for i in range(N) for j in range(N)]


def case(name, cols, engine=None, caller=None, p=None, alpha=ALPHA):
    engine = engine or {}
    caller = caller or {}
    forbidden = {tuple(e) for e in engine.get("forbidden", [])} | {tuple(e) for e in caller.get("forbidden", [])}
    required = {tuple(e) for e in engine.get("required", [])} | {tuple(e) for e in caller.get("required", [])}
    caps = [c for c in (engine.get("max_parents"), caller.get("max_parents")) if c is not None]
    cap = min(caps) if caps else N - 1
    pairs = screen(cols, forbidden, required)
    marg = posterior(cols, pairs, forbidden, required, cap, p)
    out = {"name": name, "columns": cols, "engine": engine, "caller": caller, "p": p, "alpha": alpha,
           "pairs": [list(x) for x in pairs], "merged_cap": cap}
    floor = min(p, alpha) if p is not None else None
    cells = list(marg)
    if floor is not None:
        for a, b in itertools.combinations(range(N), 2):
            if (a, b) in pairs or ((a, b) in forbidden and (b, a) in forbidden):
                continue
            for cell in (a * N + b, b * N + a):
                cells[cell] = max(cells[cell], floor / 2)
    out["marginals"] = cells
    return out


A = make(250, 31, [([], 1.0), ([(0, 1.0)], 0.5), ([(1, 0.9)], 0.5), ([(1, -0.8)], 0.5)])
B = make(250, 37, [([], 1.0), ([], 1.0), ([(0, 0.9), (1, 0.8)], 0.5), ([(2, 0.9)], 0.5)])
cases = [
    case("fork_screen_only", A),
    case("collider_screen_only", B),
    case("fork_forbid_one_direction_each_side_is_a_union", A, engine={"forbidden": [[1, 2]]}, caller={"forbidden": [[3, 1]]}),
    case("fork_forbid_both_directions_split_across_sides_removes_the_pair", A, engine={"forbidden": [[1, 2]]}, caller={"forbidden": [[2, 1]]}),
    case("fork_required_edge_off_the_skeleton", A, caller={"required": [[0, 3]]}),
    case("collider_cap_engine_one_caller_two", B, engine={"max_parents": 1}, caller={"max_parents": 2}),
    case("collider_cap_caller_one_engine_none", B, caller={"max_parents": 1}),
    case("collider_cap_engine_two_caller_one", B, engine={"max_parents": 2}, caller={"max_parents": 1}),
    case("fork_bernoulli_floor_below_alpha", A, p=0.01),
    case("fork_bernoulli_floor_at_alpha", A, p=0.3),
    case("collider_bernoulli_forbidden_pair_keeps_zero", B, p=0.2, caller={"forbidden": [[0, 1], [1, 0]]}),
]
print(json.dumps({"cases": cases}))
