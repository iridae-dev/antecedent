# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact-rational enumeration oracle for the ordered-response recovery of a binary m-graph.

Regenerate with:  python3 recovery_chain_observation_reference.py > recovery_chain_observation_reference.json

A structural model over X1, X2 (binary), the response indicators R1, R2 (1 observed) and the
deterministic proxies X*_i = X_i if R_i = 1, else missing, is enumerated over its 16 joint cells
with exact Fractions.  Head h is the response with no response parent, tail t has the response
edge R_h -> R_t and, in the dependent shape, also X_h -> R_t.  The observed pattern law is the
marginal of the joint on (R1, R2, X*1, X*2): nine cells, both (four), only_first (two),
only_second (two), neither.  The oracle answer is the exact P(X1, X2).  Probabilities are chosen
by a fixed generator, including extremes (1/1000 and 999/1000) that make a complete-case cell as
small as 1e-8.  The self-censoring witness is the two-model example of the module documentation
re-enumerated over exact Fractions: all mechanisms fair coins except X_i ~ Bern(1/2),
P(R_i=1 | X_i) = (2/5, 3/5) against X_i ~ Bern(2/5), (1/3, 3/4); both models give the same nine
cells and different P(X_i = 1).  Everything is written as nearest binary64 (and as the exact
fraction string).  Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)

    def prob(self, extreme):
        v = self.u()
        if extreme and v < 0.3:
            return F(1, 1000)
        if extreme and v > 0.7:
            return F(999, 1000)
        return F(int(v * 998) + 1, 1000)


def enumerate_model(p_x1, p_x2_given, p_head, tail, head, dependent):
    tail_axis = 1 - head
    both = [[F(0)] * 2 for _ in range(2)]
    only = [[F(0)] * 2 for _ in range(2)]
    neither = F(0)
    truth = [[F(0)] * 2 for _ in range(2)]
    for x1 in range(2):
        for x2 in range(2):
            for r1 in range(2):
                for r2 in range(2):
                    x, r = (x1, x2), (r1, r2)
                    mass = (p_x1 if x1 else 1 - p_x1)
                    mass *= p_x2_given[x1] if x2 else 1 - p_x2_given[x1]
                    mass *= p_head if r[head] else 1 - p_head
                    q = tail[r[head]][x[head]] if dependent else tail[r[head]][0]
                    mass *= q if r[tail_axis] else 1 - q
                    truth[x1][x2] += mass
                    if r1 and r2:
                        both[x1][x2] += mass
                    elif r1:
                        only[0][x1] += mass
                    elif r2:
                        only[1][x2] += mass
                    else:
                        neither += mass
    return both, only, neither, truth


def fl(v):
    return float(v)


cases = []
g = Lcg(20240607)
for k in range(24):
    extreme = k >= 8
    head = k % 2
    dependent = (k // 2) % 2 == 0
    p_x1 = g.prob(extreme)
    p_x2 = [g.prob(extreme), g.prob(extreme)]
    p_head = g.prob(extreme)
    tail = [[g.prob(extreme), g.prob(extreme)], [g.prob(extreme), g.prob(extreme)]]
    if not dependent:
        tail = [[t[0], t[0]] for t in tail]
    both, only, neither, truth = enumerate_model(p_x1, p_x2, p_head, tail, head, dependent)
    cases.append({
        "name": f"case{k}", "head": head, "dependent": dependent,
        "both": [[fl(v) for v in row] for row in both],
        "only_first": [fl(v) for v in only[0]], "only_second": [fl(v) for v in only[1]],
        "neither": fl(neither),
        "truth": [[fl(v) for v in row] for row in truth],
        "truth_exact": [[str(v) for v in row] for row in truth],
        "smallest_complete_cell": fl(min(v for row in both for v in row)),
    })


def witness(model, axis):
    """Nine cells and P(X_axis = 1) of one witness model, self-censoring on `axis` (0 or 1)."""
    px, pr = model
    both = [[F(0)] * 2 for _ in range(2)]
    only = [[F(0)] * 2 for _ in range(2)]
    neither = F(0)
    p_axis_one = F(0)
    for x1 in range(2):
        for x2 in range(2):
            for r1 in range(2):
                for r2 in range(2):
                    x, r = (x1, x2), (r1, r2)
                    mass = F(1)
                    for ax in range(2):
                        if ax == axis:
                            pa = px if x[ax] else 1 - px
                            q = pr[x[ax]]
                            mass *= pa * (q if r[ax] else 1 - q)
                        else:
                            mass *= F(1, 2) * F(1, 2)
                    if x[axis] == 1:
                        p_axis_one += mass
                    if r1 and r2:
                        both[x1][x2] += mass
                    elif r1:
                        only[0][x1] += mass
                    elif r2:
                        only[1][x2] += mass
                    else:
                        neither += mass
    return both, only, neither, p_axis_one


witnesses = []
for axis in range(2):
    a = witness((F(1, 2), (F(2, 5), F(3, 5))), axis)
    b = witness((F(2, 5), (F(1, 3), F(3, 4))), axis)
    cells = lambda w: [w[0][0][0], w[0][0][1], w[0][1][0], w[0][1][1], w[1][0][0], w[1][0][1],
                       w[1][1][0], w[1][1][1], w[2]]
    witnesses.append({
        "axis": axis,
        "first_cells": [str(v) for v in cells(a)], "second_cells": [str(v) for v in cells(b)],
        "cells_equal": cells(a) == cells(b),
        "first_p_one": str(a[3]), "second_p_one": str(b[3]),
    })
print(json.dumps({"cases": cases, "witnesses": witnesses}))
