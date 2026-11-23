# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the efficient-adjustment fixture by exhaustive enumeration.

For each DAG the reference

1. enumerates every subset Z of the other nodes and keeps the valid back-door sets by the
   definition: no member of Z is a descendant of the treatment, and Z d-separates the treatment
   from the outcome once the arrows out of the treatment are removed (d-separation is decided
   on the moral graph of the ancestral subgraph, not by a path search);
2. gives the DAG the linear-Gaussian parameterisation with every edge coefficient 1 and every
   noise variance 1, builds the exact covariance with fractions, and computes for each valid
   set the asymptotic variance factor sigma2_{Y|T,Z} / sigma2_{T|Z} of the regression
   coefficient of the treatment;
3. selects the valid set with the smallest factor and requires it to be unique.

Standard library only (fractions). It imports no Antecedent code and shares nothing with the
product's identifier. The treatment is T and the outcome is Y unless a case names others.

    uv run --frozen --script conformance/identify/efficient_adjustment/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parents_of(edges):
    nodes = sorted({n for e in edges for n in e.split("->")})
    parents = {n: set() for n in nodes}
    for e in edges:
        a, b = e.split("->")
        parents[b].add(a)
    return nodes, parents


def ancestors(parents, seeds):
    seen, stack = set(seeds), list(seeds)
    while stack:
        for p in parents[stack.pop()]:
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def descendants(parents, node):
    return {n for n in parents if node in ancestors(parents, {n}) and n != node}


def d_separated(parents, x, y, given):
    """Moral-graph test: x and y are separated by `given` iff they are disconnected in the
    moralised ancestral graph of {x, y} U given once `given` is deleted."""
    keep = ancestors(parents, {x, y} | set(given))
    adjacency = {n: set() for n in keep}
    for child in keep:
        ps = [p for p in parents[child] if p in keep]
        for p in ps:
            adjacency[p].add(child)
            adjacency[child].add(p)
        for a, b in itertools.combinations(ps, 2):
            adjacency[a].add(b)
            adjacency[b].add(a)
    frontier, seen = [x], {x}
    while frontier:
        here = frontier.pop()
        for nxt in adjacency[here]:
            if nxt in given or nxt in seen:
                continue
            seen.add(nxt)
            frontier.append(nxt)
    return y not in seen


def valid_sets(nodes, parents, t, y):
    # Remove the arrows out of the treatment: no node keeps the treatment as a parent.
    cut = {n: {p for p in ps if p != t} for n, ps in parents.items()}
    forbidden = descendants(parents, t) | {t, y}
    others = [n for n in nodes if n not in (t, y)]
    out = []
    for size in range(len(others) + 1):
        for z in itertools.combinations(others, size):
            if set(z) & forbidden:
                continue
            if d_separated(cut, t, y, set(z)):
                out.append(frozenset(z))
    return out


def covariance(nodes, parents):
    """Sigma = (I - B)^-1 (I - B)^-T for unit coefficients and unit noise variances."""
    order = []
    placed = set()
    while len(order) < len(nodes):
        for n in nodes:
            if n not in placed and parents[n] <= placed:
                order.append(n)
                placed.add(n)
    # Express every node as a vector of noise coefficients, in topological order.
    loading = {}
    for n in order:
        row = {n: Fraction(1)}
        for p in parents[n]:
            for k, v in loading[p].items():
                row[k] = row.get(k, Fraction(0)) + v
        loading[n] = row
    return {
        (a, b): sum(loading[a].get(k, 0) * loading[b].get(k, 0) for k in nodes)
        for a in nodes
        for b in nodes
    }


def solve(matrix, rhs):
    n = len(matrix)
    aug = [list(map(Fraction, matrix[i])) + [Fraction(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if aug[r][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col], strict=True)]
    return [aug[i][n] for i in range(n)]


def conditional_variance(sigma, target, given):
    given = sorted(given)
    if not given:
        return sigma[(target, target)]
    matrix = [[sigma[(a, b)] for b in given] for a in given]
    cross = [sigma[(target, g)] for g in given]
    weights = solve(matrix, cross)
    return sigma[(target, target)] - sum(w * c for w, c in zip(weights, cross, strict=True))


def optimal_set(edges, t, y):
    nodes, parents = parents_of(edges)
    sigma = covariance(nodes, parents)
    scored = []
    for z in valid_sets(nodes, parents, t, y):
        factor = conditional_variance(sigma, y, set(z) | {t}) / conditional_variance(sigma, t, z)
        scored.append((factor, sorted(z)))
    scored.sort(key=lambda row: (row[0], len(row[1]), row[1]))
    best = scored[0][0]
    winners = [row[1] for row in scored if row[0] == best]
    if len(winners) != 1:
        raise SystemExit(f"the minimum-variance valid set is not unique: {winners}")
    return winners[0], scored


def main() -> int:
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = []
    for case in document["cases"]:
        t = case.get("treatment", "T")
        y = case.get("outcome", "Y")
        chosen, scored = optimal_set(case["edges"], t, y)
        if sorted(case["optimal_set"]) != chosen:
            problems.append(f"{case['id']}: derived {chosen} != fixture {case['optimal_set']}")
    for problem in problems:
        print("FAIL", problem, file=sys.stderr)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
