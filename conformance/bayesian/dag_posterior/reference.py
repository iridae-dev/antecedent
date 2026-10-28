# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact Gaussian-BIC DAG posterior enumeration for the graph-posterior fixture.

The chain (a -> b -> c, n = 200, stream seed 7) and collider (a -> c <- b, n = 400, stream seed 11)
tables are regenerated from the SplitMix64 stream definition (state seed XOR 0x9E3779B97F4A7C15,
state += gamma before each output, two xor-shift-multiply rounds, top 53 bits scaled by 2^-53; a
draw is 2u - 1), with the structural equations stated in the consuming test's docstrings. Every one
of the 25 DAGs on three nodes is enumerated; node v with parent set P scores
-(n/2) ln(RSS_v / n) - (|P| / 2) ln n, where RSS_v is the residual sum of squares of v regressed on
an intercept and P, solved over Fractions from the stored doubles. Terms that do not depend on the
graph cancel; the prior over DAGs is uniform. The directed edge marginals (index from*3+to) are the
weighted fraction of DAGs containing each edge. No Antecedent code.

    uv run --frozen --script conformance/bayesian/dag_posterior/reference.py --check
    uv run --frozen --script conformance/bayesian/dag_posterior/reference.py --self-test
    uv run --frozen --script conformance/bayesian/dag_posterior/reference.py --write

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


class Stream:
    def __init__(self, seed: int) -> None:
        self.state = (seed ^ GAMMA) & MASK

    def uniform(self) -> float:
        self.state = (self.state + GAMMA) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
        return (z >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0


def chain(n: int) -> list[list[float]]:
    rng = Stream(7)
    a, b, c = [0.0] * n, [0.0] * n, [0.0] * n
    for i in range(n):
        a[i] = rng.uniform()
        b[i] = 1.5 * a[i] + 0.15 * rng.uniform()
        c[i] = 1.2 * b[i] + 0.15 * rng.uniform()
    return [a, b, c]


def collider(n: int) -> list[list[float]]:
    rng = Stream(11)
    a, b, c = [0.0] * n, [0.0] * n, [0.0] * n
    for i in range(n):
        a[i] = rng.uniform()
        b[i] = rng.uniform()
        c[i] = a[i] + b[i] + 0.15 * rng.uniform()
    return [a, b, c]


def rss(columns: list[list[float]], target: int, parents: tuple[int, ...]) -> Fraction:
    n = len(columns[0])
    rows = [[Fraction(1)] + [Fraction(columns[p][r]) for p in parents] for r in range(n)]
    y = [Fraction(v) for v in columns[target]]
    k = len(rows[0])
    m = [
        [sum((rows[r][i] * rows[r][j] for r in range(n)), Fraction(0)) for j in range(k)]
        + [sum((rows[r][i] * y[r] for r in range(n)), Fraction(0))]
        for i in range(k)
    ]
    for c in range(k):
        p = m[c][c]
        m[c] = [v / p for v in m[c]]
        for r in range(k):
            if r != c:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    beta = [m[i][k] for i in range(k)]
    return sum(
        ((y[r] - sum(beta[j] * rows[r][j] for j in range(k))) ** 2 for r in range(n)), Fraction(0)
    )


def acyclic(edges: set[tuple[int, int]], v: int) -> bool:
    remaining = set(range(v))
    while remaining:
        free = [x for x in remaining if not any(t == x and f in remaining for f, t in edges)]
        if not free:
            return False
        remaining.discard(free[0])
    return True


def marginals(columns: list[list[float]]) -> list[float]:
    v, n = len(columns), len(columns[0])
    pairs = [(i, j) for i in range(v) for j in range(v) if i != j]
    cache: dict[tuple[int, tuple[int, ...]], float] = {}

    def node_score(node: int, parents: tuple[int, ...]) -> float:
        key = (node, parents)
        if key not in cache:
            cache[key] = -(n / 2.0) * math.log(float(rss(columns, node, parents) / n)) - (
                len(parents) / 2.0
            ) * math.log(n)
        return cache[key]

    graphs, weights = [], []
    for mask in itertools.product((0, 1), repeat=len(pairs)):
        edges = {e for e, bit in zip(pairs, mask) if bit}
        if not acyclic(edges, v):
            continue
        weights.append(
            sum(node_score(x, tuple(sorted(f for f, t in edges if t == x))) for x in range(v))
        )
        graphs.append(edges)
    top = max(weights)
    total = sum(math.exp(w - top) for w in weights)
    out = [0.0] * (v * v)
    for edges, w in zip(graphs, weights):
        for f, t in edges:
            out[f * v + t] += math.exp(w - top) / total
    assert len(graphs) == 25
    return out


def compute() -> dict:
    return {"chain": marginals(chain(200)), "collider": marginals(collider(400))}


def check(document: dict) -> list[str]:
    problems = []
    got = compute()
    stored = document.get("exact_marginals")
    if stored is None:
        return ["exact_marginals block missing (run --write)"]
    for name in ("chain", "collider"):
        if len(stored.get(name, [])) != 9:
            problems.append(f"{name}: stored marginals malformed")
            continue
        for cell, (a, b) in enumerate(zip(got[name], stored[name])):
            if abs(a - b) > 1e-12:
                problems.append(f"{name} cell {cell}: {b} vs {a}")
    if not problems:
        chain_mass = min(got["chain"][1] + got["chain"][3], got["chain"][5] + got["chain"][7])
        if chain_mass < document["chain_fixture"]["min_undirected_mass"]:
            problems.append(f"chain skeleton mass {chain_mass} below the declared minimum")
        arrows = min(got["collider"][2], got["collider"][5])
        if arrows < document["collider_fixture"]["min_directed_mass"]:
            problems.append(f"collider arrow mass {arrows} below the declared minimum")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["exact_marginals"]["collider"][2] += 1e-6
    if not check(wrong):
        failures.append("shifted marginal accepted")
    wrong = copy.deepcopy(document)
    wrong["collider_fixture"]["min_directed_mass"] = 0.95
    if not check(wrong):
        failures.append("threshold above the exact mass accepted")
    wrong = copy.deepcopy(document)
    del wrong["exact_marginals"]
    if not check(wrong):
        failures.append("missing block accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = HERE / "expected.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if args.write:
        document["exact_marginals"] = compute()
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("written")
        return 0
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
