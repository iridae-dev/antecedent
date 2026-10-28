# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact evaluation of a linear structural model from given noise, under interventions.

Regenerate with:  uv run --script scm_replay_reference.py > scm_replay_reference.json

A scenario is a directed acyclic graph with node ids in a deliberately non-topological order,
a linear mechanism per node (intercept and one coefficient per parent id), and a table of
exogenous noise values.  The value of a node is evaluated from the structural equation
    x_i = b_i + sum_{p in parents(i)} c_{i,p} x_p + u_i                    (observed model)
in a topological order found here by repeated sweeps, with every number a Fraction of a dyadic
(or exactly representable) input.  Interventions are the definitions of the mutilated model:
    hard set   x_i := v                    (the equation and the noise of i are discarded)
    shift      x_i := b_i + sum c x_p + u_i + s   (hard-set nodes are not shifted)
    soft       the equation of i is replaced by b'_i + sum c'_{i,p} x_p + u_i
Coalition replays read the noise of masked nodes from a first table and of the others from a
second table.  Each expected value also carries a magnitude  m_i  (the same recursion with
absolute values) so a test can scale its tolerance to the rounding the evaluation may incur.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F


def evaluate(sc, noise, hard, shift, soft, rows=None):
    n = sc["n"]
    parents = {i: [p for (p, c) in sc["edges"] if c == i] for i in range(n)}
    order, placed = [], set()
    while len(order) < n:
        for i in range(n):
            if i not in placed and all(p in placed for p in parents[i]):
                order.append(i)
                placed.add(i)
    out, mag = {}, {}
    for i in order:
        if i in hard:
            out[i] = F(hard[i])
            mag[i] = abs(F(hard[i]))
            continue
        b, coef = (soft[i] if i in soft else (sc["intercept"][i], sc["coef"][i]))
        v = F(b) + noise[i]
        m = abs(F(b)) + abs(noise[i])
        for p in parents[i]:
            c = F(coef[str(p)])
            v += c * out[p]
            m += abs(c) * mag[p]
        v += F(shift.get(i, 0))
        m += abs(F(shift.get(i, 0)))
        out[i], mag[i] = v, m
    return out, mag


def scenario(name, n, edges, rng, big=False, rows=4):
    sc = {"name": name, "n": n, "edges": edges, "intercept": [], "coef": []}
    for i in range(n):
        sc["intercept"].append(float(rng.randint(-8, 8) * (2 ** 17 if big else 1) / 4))
        d = {}
        for (p, c) in edges:
            if c == i:
                d[str(p)] = float(rng.choice([-3, -1, 1, 2, 3, 7]) * (2.0 ** rng.choice([-20, -3, 0, 6]) if big else 1) / 2)
        sc["coef"].append(d)
    tables = [[[F(rng.randint(-16, 16), 8) for _ in range(n)] for _ in range(rows)] for _ in range(2)]
    sc["noise"] = [[[float(x) for x in r] for r in t] for t in tables]
    cases = []
    overlays = [
        {"hard": {}, "shift": {}, "soft": {}},
        {"hard": {1: 2.5}, "shift": {}, "soft": {}},
        {"hard": {}, "shift": {0: 0.75, n - 1: -4.0}, "soft": {}},
        {"hard": {2: -1.5}, "shift": {1: 0.5, 0: -0.25}, "soft": {}},
        {"hard": {n - 2: 7.0, 0: 0.0}, "shift": {}, "soft": {}},
        {"hard": {}, "shift": {}, "soft": {1: (3.5, {str(p): 0.25 * (k + 1) for k, (p, c) in enumerate([e for e in edges if e[1] == 1])})}},
        {"hard": {3 % n: 1.0}, "shift": {1: 1.0}, "soft": {n - 1: (-2.0, {str(p): -1.5 for (p, c) in edges if c == n - 1})}},
    ]
    for ov in overlays:
        hard = {int(k): v for k, v in ov["hard"].items()}
        shift = {int(k): v for k, v in ov["shift"].items()}
        soft = {int(k): v for k, v in ov["soft"].items()}
        vals, mags = [], []
        for r in range(rows):
            o, m = evaluate(sc, tables[0][r], hard, shift, soft)
            vals.append([float(o[i]) for i in range(n)])
            mags.append([float(m[i]) for i in range(n)])
        # coalition: nodes {0,2,...} read table 0, the rest table 1
        mask = [i % 2 == 0 for i in range(n)]
        cvals, cmags = [], []
        for r in range(rows):
            mixed = [tables[0][r][i] if mask[i] else tables[1][r][i] for i in range(n)]
            o, m = evaluate(sc, mixed, hard, shift, soft)
            cvals.append([float(o[i]) for i in range(n)])
            cmags.append([float(m[i]) for i in range(n)])
        zero = [F(0)] * n
        o, m = evaluate(sc, zero, hard, shift, soft)
        cases.append({"deterministic": [float(o[i]) for i in range(n)],
                      "deterministic_magnitude": [float(m[i]) for i in range(n)],
                      "hard": {str(k): v for k, v in hard.items()},
                      "shift": {str(k): v for k, v in shift.items()},
                      "soft": {str(k): {"intercept": v[0], "coef": v[1]} for k, v in soft.items()},
                      "values": vals, "magnitude": mags, "mask": mask,
                      "coalition_values": cvals, "coalition_magnitude": cmags})
    sc["cases"] = cases
    return sc


def main():
    rng = random.Random(2026)
    out = [
        # diamond plus collider, ids against the topological order
        scenario("diamond_collider_reversed_ids", 6, [(5, 3), (5, 1), (3, 0), (1, 0), (0, 2), (4, 2)], rng),
        # the same shape with intercepts near 1e5 and coefficients spanning 2^-20 .. 2^6
        scenario("large_offsets_wide_scales", 6, [(5, 3), (5, 1), (3, 0), (1, 0), (0, 2), (4, 2)], rng, big=True),
        # a deep chain in reverse id order (amplifying)
        scenario("deep_chain_reversed", 8, [(7, 6), (6, 5), (5, 4), (4, 3), (3, 2), (2, 1), (1, 0)], rng),
        # five parents into one node and an isolated node
        scenario("wide_collider_isolated", 8, [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 4), (6, 3)], rng, big=True),
    ]
    print(json.dumps(out, indent=1))


main()
