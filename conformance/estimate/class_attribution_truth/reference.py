# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of anomaly and change attribution over a structure class.

A structure that stands for several (a CPDAG, a PAG, their temporal forms) is attributed member by
member and the members are merged; a mixed graph is attributed to the noise blocks (districts) of
the ancestral sub-graph. This script derives every published number from the definitions, with
exact rationals for the regressions and `math.erfc` for the Gaussian tail, and imports no
Antecedent code and no statistics package.

Definitions used (each case states its members, their parents and their players by hand):

* Every mechanism is the least-squares regression of the node on its parents in the member
  (intercept included; a root is its mean); the noise of a row is its residual (abduction).
* Anomaly score of a value v: -ln P(|Z| >= |v - c| / s) = -ln erfc(|v - c| / (s sqrt 2)), with
  (c, s) fixed by the query or, for the empirical reference, c = median(target column) and
  s = 1.482602218505602 * MAD (mean absolute deviation if MAD = 0, and 1 if that is 0 as well).
* The Shapley game of a row has the players of the member (the ancestors of the target, each
  alone, or a whole block of a mixed graph together); v(S) is the score of the target rebuilt from
  the noise of the players in S with the noise of the others set to 0; a player's contribution is
  its marginal contribution averaged over all orders of the players.
* A class publishes the weighted mean of the members that identify (absent players contribute 0),
  the per-component minimum and maximum over those members of the row mean (the identified set),
  and the share of the weight that identified, did not identify (a hedge) or failed.
* Change attribution compares two row ranges (populations). Each mechanism is fitted on each
  population; the players are the mechanisms of the ancestors of the outcome (a block swaps all
  its mechanisms together); v(S) is the mean of the outcome in the model whose mechanisms in S are
  the comparison fits and the others the baseline fits (noise has mean 0), so the value is linear
  in the means and exact; the contribution is the Shapley value of the mechanism swap. The product
  evaluates the value by Monte Carlo (2000 noise draws, common to every coalition), so the
  tolerance of each published number is 5 standard errors of that draw, derived here.
* A series is analysed on its lag-aligned design: the column of lag L at row t reads the series at
  row max(t - L, 0), so the first rows repeat the first period. That convention is the route's
  stated alignment and the one object this reference shares with it.

    uv run --frozen --script conformance/estimate/class_attribution_truth/reference.py --check
    uv run --frozen --script conformance/estimate/class_attribution_truth/reference.py --self-test
    uv run --frozen --script conformance/estimate/class_attribution_truth/reference.py --write

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
FIXTURE = HERE / "expected.json"
F = Fraction
MAD_SCALE = F(1.482602218505602)
DRAWS = 2000
SIGMAS = 5


# ----------------------------------------------------------------------------------------------
# Exact regression


def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def fit(design, rows, node, parents):
    """Least squares of `node` on `parents` over `rows`: ([intercept, *slopes], residual variance)."""
    cols = [[F(1)] * len(rows)] + [[F(design[p][i]) for i in rows] for p in parents]
    y = [F(design[node][i]) for i in rows]
    xtx = [[sum(a * b for a, b in zip(ci, cj)) for cj in cols] for ci in cols]
    xty = [sum(a * b for a, b in zip(ci, y)) for ci in cols]
    beta = solve(xtx, xty)
    rss = sum((y[k] - sum(beta[j] * cols[j][k] for j in range(len(cols)))) ** 2 for k in range(len(rows)))
    dof = len(rows) - len(cols)
    return beta, (rss / dof if dof > 0 else F(0))


def median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def reference_of(spec, target_values):
    if spec["kind"] == "fixed":
        return F(spec["center"]), F(spec["scale"])
    vals = [F(v) for v in target_values]
    c = median(vals)
    dev = [abs(v - c) for v in vals]
    mad = median(dev)
    if mad > 0:
        return c, MAD_SCALE * mad
    mean_abs = sum(dev) / len(dev)
    return c, (mean_abs if mean_abs > 0 else F(1))


def score(value, center, scale):
    z = abs(float((F(value) - center) / scale))
    if z > 26:
        raise SystemExit("standardized deviation too large for the plain tail")
    return -math.log(math.erfc(z / math.sqrt(2)))


# ----------------------------------------------------------------------------------------------
# Designs


def design_of(case):
    """The columns the member graphs are fitted on: the series columns, plus `name@L` lag columns."""
    cols = case["columns"]
    out = {name: list(values) for name, values in cols.items()}
    for name, lag in case.get("lags", []):
        values = cols[name]
        out[f"{name}@{lag}"] = [values[max(t - lag, 0)] for t in range(len(values))]
    return out


def ancestors_of(parents, target):
    seen, stack = set(), [target]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        stack.extend(parents.get(v, []))
    return seen


def topo(parents, nodes):
    order, done = [], set()

    def visit(v):
        if v in done:
            return
        for p in parents.get(v, []):
            visit(p)
        done.add(v)
        order.append(v)

    for v in sorted(nodes):
        visit(v)
    return order


def shapley(players, value):
    """Exact Shapley values of a game over `players` (indices) with `value(frozenset)`."""
    k = len(players)
    phi = [0.0] * k
    orders = list(itertools.permutations(range(k)))
    for order in orders:
        held = set()
        before = value(frozenset(held))
        for p in order:
            held.add(p)
            after = value(frozenset(held))
            phi[p] += after - before
            before = after
    return [v / len(orders) for v in phi]


# ----------------------------------------------------------------------------------------------
# Anomaly


def anomaly_member(case, member):
    design = design_of(case)
    n = len(next(iter(design.values())))
    rows = list(range(n))
    target = case["target"]
    parents = member["parents"]
    nodes = ancestors_of(parents, target)
    order = topo(parents, nodes)
    fits = {v: fit(design, rows, v, parents.get(v, []))[0] for v in order}
    noise = {}
    for v in order:
        beta = fits[v]
        noise[v] = [
            F(design[v][i]) - beta[0] - sum(b * F(design[p][i]) for b, p in zip(beta[1:], parents.get(v, [])))
            for i in range(n)
        ]
    players = member["players"]
    for node in nodes:
        if sum(node in p["nodes"] for p in players) != 1:
            raise SystemExit(f"node {node} is not in exactly one player")
    center, scale = reference_of(case["reference"], design[target])

    def rebuild(i, restored):
        held = {v for idx in restored for v in players[idx]["nodes"]}
        value = {}
        for v in order:
            beta = fits[v]
            value[v] = beta[0] + sum(b * value[p] for b, p in zip(beta[1:], parents.get(v, [])))
            if v in held:
                value[v] += noise[v][i]
        return value[target]

    scores, baseline, phi = [], [], []
    for i in range(n):
        scores.append(score(design[target][i], center, scale))
        cache = {}

        def game(s, i=i, cache=cache):
            if s not in cache:
                cache[s] = score(rebuild(i, s), center, scale)
            return cache[s]

        baseline.append(game(frozenset()))
        phi.append(shapley(range(len(players)), game))
    return {
        "components": [p["component"] for p in players],
        "scores": scores,
        "baseline": baseline,
        "phi": phi,
    }


def merge_anomaly(case, members_out):
    weights = [m["weight"] for m in case["members"]]
    outside = case.get("outside_weight", 0.0)
    ident = [i for i, m in enumerate(case["members"]) if m["status"] == "identified"]
    total = sum(weights)
    w = [weights[i] / sum(weights[j] for j in ident) for i in ident]
    ids = case["component_ids"]
    comps = sorted({c for i in ident for c in members_out[i]["components"]}, key=lambda c: ids[c])
    n = len(members_out[ident[0]]["scores"])
    merged = [[0.0] * len(comps) for _ in range(n)]
    base = [0.0] * n
    row_mean = []
    for wi, i in zip(w, ident):
        out = members_out[i]
        means = [0.0] * len(comps)
        for r in range(n):
            base[r] += wi * out["baseline"][r]
            for col, c in enumerate(out["components"]):
                pos = comps.index(c)
                merged[r][pos] += wi * out["phi"][r][col]
                means[pos] += out["phi"][r][col] / n
        row_mean.append(means)
    return {
        "components": comps,
        "baseline": base,
        "contributions": merged,
        "member_row_mean": row_mean,
        "lower": [min(m[c] for m in row_mean) for c in range(len(comps))],
        "upper": [max(m[c] for m in row_mean) for c in range(len(comps))],
        "identified_mass": sum(weights[i] for i in ident) / (total + outside),
        "unidentified_mass": (sum(m["weight"] for m in case["members"] if m["status"] == "hedged") + outside)
        / (total + outside),
        "unevaluable_mass": sum(m["weight"] for m in case["members"] if m["status"] == "failed") / (total + outside),
    }


# ----------------------------------------------------------------------------------------------
# Change


def change_member(case, member):
    design = design_of(case)
    (b0, b1), (c0, c1) = case["populations"]
    base_rows, comp_rows = list(range(b0, b1)), list(range(c0, c1))
    target = case["target"]
    parents = member["parents"]
    nodes = ancestors_of(parents, target)
    order = topo(parents, nodes)
    fits = {
        v: (fit(design, base_rows, v, parents.get(v, [])), fit(design, comp_rows, v, parents.get(v, [])))
        for v in order
    }
    players = member["players"]
    for node in nodes:
        if sum(node in p["nodes"] for p in players) != 1:
            raise SystemExit(f"node {node} is not in exactly one player")

    def hybrid(swapped):
        held = {v for idx in swapped for v in players[idx]["nodes"]}
        return {v: fits[v][1 if v in held else 0] for v in order}

    def outcome_mean(swapped):
        mech = hybrid(swapped)
        value = {}
        for v in order:
            beta = mech[v][0]
            value[v] = beta[0] + sum(b * value[p] for b, p in zip(beta[1:], parents.get(v, [])))
        return value[target]

    def effect(swapped, source):
        """d outcome / d (standardized noise draw) of `source` under the hybrid of `swapped`: the
        noise enters the node scaled by the standard deviation of the mechanism in use."""
        mech = hybrid(swapped)
        eff = {}
        for v in order:
            beta = mech[v][0]
            own = F(math.sqrt(float(mech[v][1]))) if v == source else F(0)
            eff[v] = own + sum(b * eff[p] for b, p in zip(beta[1:], parents.get(v, [])))
        return eff[target]

    k = len(players)
    phi = shapley(range(k), lambda s: float(outcome_mean(s)))
    everyone = frozenset(range(k))
    total = float(outcome_mean(everyone) - outcome_mean(frozenset()))

    # The product evaluates every coalition by its own Monte Carlo draw of the outcome mean, and the
    # draws are not assumed common to coalitions. The mean of the outcome under the hybrid S carries
    # the error sum_k e_k(S) z_k / sqrt(DRAWS) with e_k(S) the effect of node k's own noise (scaled
    # by its mechanism's standard deviation), the z_k independent standard normals; a contribution
    # is an average of differences of two such values. Bound: SIGMAS * sqrt(2 sum_k max_S e_k(S)^2)
    # / sqrt(DRAWS), a block resampling its residual rows having the same scale.
    subsets = [frozenset(c) for r in range(k + 1) for c in itertools.combinations(range(k), r)]
    value_sd = math.sqrt(
        sum(max(float(effect(s, v)) ** 2 for s in subsets) for v in order) / DRAWS
    )
    bound = SIGMAS * math.sqrt(2.0) * value_sd + 1e-9
    tol = [bound for _ in range(k)]
    total_tol = bound
    observed = float(
        sum(F(design[target][i]) for i in comp_rows) / len(comp_rows)
        - sum(F(design[target][i]) for i in base_rows) / len(base_rows)
    )
    return {
        "components": [p["component"] for p in players],
        "total": total,
        "observed": observed,
        "phi": phi,
        "tolerance": tol,
        "total_tolerance": total_tol,
    }


def merge_change(case, members_out):
    weights = [m["weight"] for m in case["members"]]
    outside = case.get("outside_weight", 0.0)
    ident = [i for i, m in enumerate(case["members"]) if m["status"] == "identified"]
    w = [weights[i] / sum(weights[j] for j in ident) for i in ident]
    ids = case["component_ids"]
    comps = sorted({c for i in ident for c in members_out[i]["components"]}, key=lambda c: ids[c])
    per_member = []
    for i in ident:
        out = members_out[i]
        per_member.append([out["phi"][out["components"].index(c)] if c in out["components"] else 0.0 for c in comps])
    mean = [sum(wi * m[c] for wi, m in zip(w, per_member)) for c in range(len(comps))]
    tol = []
    for c, comp in enumerate(comps):
        # the mean of the members' contributions carries the weighted sum of their tolerances
        bound = 0.0
        for wi, i in zip(w, ident):
            out = members_out[i]
            if comp in out["components"]:
                bound += wi * out["tolerance"][out["components"].index(comp)]
        tol.append(bound)
    return {
        "components": comps,
        "member_contributions": per_member,
        "contributions": mean,
        "tolerance": tol,
        "lower": [min(m[c] for m in per_member) for c in range(len(comps))],
        "upper": [max(m[c] for m in per_member) for c in range(len(comps))],
        "total": sum(wi * members_out[i]["total"] for wi, i in zip(w, ident)),
        "total_tolerance": sum(wi * members_out[i]["total_tolerance"] for wi, i in zip(w, ident)),
        "observed": members_out[ident[0]]["observed"],
        "identified_mass": sum(weights[i] for i in ident) / (sum(weights) + outside),
        "unidentified_mass": (sum(m["weight"] for m in case["members"] if m["status"] == "hedged") + outside)
        / (sum(weights) + outside),
        "unevaluable_mass": sum(m["weight"] for m in case["members"] if m["status"] == "failed")
        / (sum(weights) + outside),
    }


def cube_root_ceil(n):
    k = 1
    while k**3 < n:
        k += 1
    return k


def dirichlet_mean_var(values, series):
    """Mean and variance of sum_i w_i v_i for w ~ Dirichlet(1, ..., 1) over the rows, or, for a
    series, of sum_b W_b m_b with W ~ Dirichlet(1, ..., 1) over its contiguous blocks of rows and
    m_b the block mean (the block length is the larger of the cube root of the rows, rounded up,
    and the lag span plus one, which is at most 2 here; the rows divide into whole blocks)."""
    v = [F(x) for x in values]
    n = len(v)
    if series:
        block = max(cube_root_ceil(n), 2)
        if n % block:
            raise SystemExit("the series does not divide into whole blocks")
        v = [sum(v[i : i + block]) / block for i in range(0, n, block)]
        n = len(v)
    mean = sum(v) / n
    spread = sum((x - mean) ** 2 for x in v) / n
    return float(mean), float(spread / (n + 1))


def bayes_moments(case, members):
    """Posterior of the Bayesian-bootstrap cells: the mean anomaly score of the first target, or the
    total change, under Dirichlet(1, ..., 1) weights on the rows (blocks for a series)."""
    series = bool(case.get("series"))
    if case["kind"] == "anomaly":
        first = next(m for m in members if m is not None)
        mean, var = dirichlet_mean_var(first["scores"], series)
        return {"mean": mean, "sd": math.sqrt(var)}
    design = design_of(case)
    (b0, b1), (c0, c1) = case["populations"]
    target = design[case["target"]]
    mb, vb = dirichlet_mean_var(target[b0:b1], series)
    mc, vc = dirichlet_mean_var(target[c0:c1], series)
    return {"mean": mc - mb, "sd": math.sqrt(vb + vc)}


def derive(case):
    run = anomaly_member if case["kind"] == "anomaly" else change_member
    members = []
    for member in case["members"]:
        members.append(run(case, member) if member["status"] == "identified" else None)
    merged = (merge_anomaly if case["kind"] == "anomaly" else merge_change)(case, members)
    merged["bayes"] = bayes_moments(case, members)
    return {"members": [m for m in members], "merged": merged}


# ----------------------------------------------------------------------------------------------
# Cases


def sines(n):
    seq = lambda f, p: [math.sin(f * i + p) for i in range(n)]
    return seq(0.37, 0.2), seq(1.13, 0.7), seq(2.9, 1.3)


def player(*nodes, component=None):
    return {"component": component or nodes[0], "nodes": list(nodes)}


def member(label, weight, status, parents, players):
    return {"label": label, "weight": weight, "status": status, "parents": parents, "players": players}


def chain_pair_table(n, outlier, shift_pop=False):
    s1, s2, s3 = sines(n)
    x = [2.0 * v for v in s1]
    y = [2.0 * x[i] + 0.4 * s3[i] for i in range(n)]
    y[-1] += outlier
    return {"x": x, "y": y}


def build_cases():
    cases = {}
    n = 40
    s1, s2, s3 = sines(n)

    # Anomaly ------------------------------------------------------------------------------
    cases["cpdag_pair_anomaly"] = {
        "kind": "anomaly",
        "graph": {"class": "cpdag", "nodes": 2, "directed": [], "undirected": [[0, 1]]},
        "columns": chain_pair_table(n, 12.0),
        "reference": {"kind": "empirical"},
        "target": "y",
        "component_ids": {"x": 0, "y": 1},
        "members": [
            member("x -> y", 1, "identified", {"x": [], "y": ["x"]}, [player("x"), player("y")]),
            member("y -> x", 1, "identified", {"y": [], "x": ["y"]}, [player("y")]),
        ],
    }
    x = [2.0 * v for v in s1]
    y = [1.0 * x[i] + 0.5 * s2[i] for i in range(n)]
    z = [0.8 * y[i] + 0.4 * s3[i] for i in range(n)]
    z[-1] += 9.0
    cases["cpdag_chain_anomaly"] = {
        "kind": "anomaly",
        "graph": {"class": "cpdag", "nodes": 3, "directed": [], "undirected": [[0, 1], [1, 2]]},
        "columns": {"x": x, "y": y, "z": z},
        "reference": {"kind": "fixed", "center": 0.0, "scale": 2.5},
        "target": "z",
        "component_ids": {"x": 0, "y": 1, "z": 2},
        "members": [
            member("x -> y -> z", 1, "identified", {"x": [], "y": ["x"], "z": ["y"]}, [player("x"), player("y"), player("z")]),
            member("x <- y -> z", 1, "identified", {"y": [], "x": ["y"], "z": ["y"]}, [player("y"), player("z")]),
            member("x <- y <- z", 1, "identified", {"z": [], "y": ["z"], "x": ["y"]}, [player("z")]),
        ],
    }
    iso = [3.0 * v + 0.5 * w for v, w in zip(s3, s2)]
    iso[-1] += 20.0
    cases["cpdag_agreeing_anomaly"] = {
        "kind": "anomaly",
        "graph": {"class": "cpdag", "nodes": 3, "directed": [], "undirected": [[0, 1]]},
        "columns": {"x": x, "y": [2.0 * a + 0.4 * b for a, b in zip(x, s3)], "z": iso},
        "reference": {"kind": "empirical"},
        "target": "z",
        "component_ids": {"x": 0, "y": 1, "z": 2},
        "members": [
            member("x -> y, z", 1, "identified", {"x": [], "y": ["x"], "z": []}, [player("z")]),
            member("x <- y, z", 1, "identified", {"y": [], "x": ["y"], "z": []}, [player("z")]),
        ],
    }
    yy = [2.0 * a + 0.4 * b for a, b in zip(x, s3)]
    yy[-1] += 14.0
    pair = {"x": x, "y": yy}
    cases["pag_circle_arrow_anomaly"] = {
        "kind": "anomaly",
        "graph": {"class": "pag", "nodes": 2, "marked": [[0, 1, "circle", "arrow"]]},
        "columns": pair,
        "reference": {"kind": "empirical"},
        "target": "y",
        "component_ids": {"x": 0, "y": 1},
        "members": [
            member("x -> y (an invisible edge may carry a latent common cause)", 1, "hedged", {"x": [], "y": ["x"]}, []),
            member("x <-> y", 1, "identified", {"x": [], "y": []}, [player("y")]),
        ],
    }
    cases["pag_circle_circle_anomaly"] = {
        "kind": "anomaly",
        "graph": {"class": "pag", "nodes": 2, "marked": [[0, 1, "circle", "circle"]]},
        "columns": pair,
        "reference": {"kind": "fixed", "center": 0.0, "scale": 3.0},
        "target": "y",
        "component_ids": {"x": 0, "y": 1},
        "members": [
            member("x -> y (hedged)", 1, "hedged", {"x": [], "y": ["x"]}, []),
            member("x <- y", 1, "identified", {"y": [], "x": ["y"]}, [player("y")]),
            member("x <-> y", 1, "identified", {"x": [], "y": []}, [player("y")]),
        ],
    }
    a = list(s1)
    b = [-0.6 * p + 0.8 * q for p, q in zip(s1, s2)]
    b[-1] += 4.0
    t = [a[i] + 2.0 * b[i] + 0.3 * s3[i] for i in range(n)]
    cases["admg_collider_anomaly"] = {
        "kind": "anomaly",
        "graph": {"class": "admg", "nodes": 3, "directed": [[0, 2], [1, 2]], "bidirected": [[0, 1]]},
        "columns": {"a": a, "b": b, "t": t},
        "reference": {"kind": "fixed", "center": 0.5, "scale": 2.5},
        "target": "t",
        "component_ids": {"a": 0, "b": 1, "t": 2},
        "members": [
            member("a -> t <- b, a <-> b", 1, "identified", {"a": [], "b": [], "t": ["a", "b"]}, [player("a", "b"), player("t")]),
        ],
    }
    # one-lag template, series of 48 periods
    m = 48
    q1, q2, q3 = sines(m)
    sx = [q1[0]]
    for i in range(1, m):
        sx.append(0.5 * sx[i - 1] + q1[i])
    sy = [0.8 * sx[max(i - 1, 0)] + 0.4 * q2[i] for i in range(m)]
    sy[-1] += 10.0
    cases["temporal_cpdag_anomaly"] = {
        "kind": "anomaly",
        "series": True,
        "graph": {
            "class": "temporal_cpdag",
            "nodes": [[0, 1], [0, 0], [1, 0]],
            "directed": [[0, 1], [0, 2]],
            "undirected": [[1, 2]],
        },
        "columns": {"x": sx, "y": sy},
        "lags": [["x", 1], ["y", 1]],
        "reference": {"kind": "empirical"},
        "target": "y",
        "component_ids": {"x": 0, "y": 1, "x@1": 2, "y@1": 3},
        "members": [
            member(
                "x -> y in every period",
                1,
                "identified",
                {"x@1": [], "y@1": ["x@1"], "x": ["x@1"], "y": ["x@1", "x"]},
                [player("x@1"), player("x"), player("y")],
            ),
            member(
                "y -> x in every period",
                1,
                "identified",
                {"y@1": [], "x@1": ["y@1"], "y": ["x@1"], "x": ["x@1", "y"]},
                [player("y@1"), player("x@1"), player("y")],
            ),
        ],
    }
    ca = list(q1)
    cb = [-0.6 * p + 0.8 * r for p, r in zip(q1, q2)]
    ct = [ca[max(i - 1, 0)] + 2.0 * cb[max(i - 1, 0)] + 0.3 * q3[i] for i in range(m)]
    ct[-1] += 7.0
    cases["temporal_admg_anomaly"] = {
        "kind": "anomaly",
        "series": True,
        "graph": {
            "class": "temporal_admg",
            "nodes": [[0, 1], [1, 1], [2, 0]],
            "directed": [[0, 2], [1, 2]],
            "bidirected": [[0, 1]],
        },
        "columns": {"a": ca, "b": cb, "t": ct},
        "lags": [["a", 1], ["b", 1]],
        "reference": {"kind": "fixed", "center": 0.0, "scale": 2.5},
        "target": "t",
        "component_ids": {"a@1": 3, "b@1": 4, "t": 2},
        "members": [
            member(
                "a@1 -> t <- b@1, a@1 <-> b@1",
                1,
                "identified",
                {"a@1": [], "b@1": [], "t": ["a@1", "b@1"]},
                [player("a@1", "b@1"), player("t")],
            ),
        ],
    }
    cases["temporal_pag_anomaly"] = {
        "kind": "anomaly",
        "series": True,
        "graph": {"class": "temporal_pag", "nodes": [[0, 1], [1, 0]], "marked": [[0, 1, "circle", "arrow"]]},
        "columns": {"x": sx, "y": sy},
        "lags": [["x", 1]],
        "reference": {"kind": "fixed", "center": 0.0, "scale": 2.0},
        "target": "y",
        "component_ids": {"x": 0, "y": 1, "x@1": 2},
        "members": [
            member("x@1 -> y (hedged)", 1, "hedged", {"x@1": [], "y": ["x@1"]}, []),
            member("x@1 <-> y", 1, "identified", {"x@1": [], "y": []}, [player("y")]),
        ],
    }

    # Change -------------------------------------------------------------------------------
    h = 60
    t1, t2, t3 = sines(h)

    def two_pop(slope0, slope1, shift_x, shift_y):
        xs, ys = [], []
        for pop in (0, 1):
            for i in range(h):
                xv = 0.5 * t1[i] + (shift_x if pop else 0.0)
                slope = slope1 if pop else slope0
                ys.append(slope * xv + 0.4 * t3[i] + (shift_y if pop else 0.0))
                xs.append(xv)
        return {"x": xs, "y": ys}

    pops = [[0, h], [h, 2 * h]]
    cases["cpdag_pair_change"] = {
        "kind": "change",
        "graph": {"class": "cpdag", "nodes": 2, "directed": [], "undirected": [[0, 1]]},
        "columns": two_pop(2.0, 2.5, 1.0, 3.0),
        "populations": pops,
        "target": "y",
        "component_ids": {"x": 0, "y": 1},
        "members": [
            member("x -> y", 1, "identified", {"x": [], "y": ["x"]}, [player("x"), player("y")]),
            member("y -> x", 1, "identified", {"y": [], "x": ["y"]}, [player("y")]),
        ],
    }
    cases["cpdag_agreeing_change"] = {
        "kind": "change",
        "graph": {"class": "cpdag", "nodes": 2, "directed": [], "undirected": [[0, 1]]},
        "columns": two_pop(2.0, 2.0, 0.0, 5.0),
        "populations": pops,
        "target": "y",
        "component_ids": {"x": 0, "y": 1},
        "members": [
            member("x -> y", 1, "identified", {"x": [], "y": ["x"]}, [player("x"), player("y")]),
            member("y -> x", 1, "identified", {"y": [], "x": ["y"]}, [player("y")]),
        ],
    }
    cases["pag_circle_arrow_change"] = {
        "kind": "change",
        "graph": {"class": "pag", "nodes": 2, "marked": [[0, 1, "circle", "arrow"]]},
        "columns": two_pop(2.0, 2.5, 1.0, 3.0),
        "populations": pops,
        "target": "y",
        "component_ids": {"x": 0, "y": 1},
        "members": [
            member("x -> y (hedged)", 1, "hedged", {"x": [], "y": ["x"]}, []),
            member("x <-> y", 1, "identified", {"x": [], "y": []}, [player("y")]),
        ],
    }
    xa, xb, xt = [], [], []
    for pop in (0, 1):
        off, const = (1.0, 0.5) if pop else (0.0, 0.0)
        for i in range(h):
            av = 0.5 * t1[i] + off
            bv = 0.5 * (-0.6 * t1[i] + 0.8 * t2[i]) + off
            xa.append(av)
            xb.append(bv)
            xt.append(av + 2.0 * bv + const + 0.2 * t3[i])
    cases["admg_collider_change"] = {
        "kind": "change",
        "graph": {"class": "admg", "nodes": 3, "directed": [[0, 2], [1, 2]], "bidirected": [[0, 1]]},
        "columns": {"a": xa, "b": xb, "t": xt},
        "populations": pops,
        "target": "t",
        "component_ids": {"a": 0, "b": 1, "t": 2},
        "members": [
            member("a -> t <- b, a <-> b", 1, "identified", {"a": [], "b": [], "t": ["a", "b"]}, [player("a", "b"), player("t")]),
        ],
    }
    return cases, h


def build_temporal_change_cases(h):
    """Two-period series (rows [0, h) and [h, 2h)) of the one-lag templates."""
    out = {}
    n = 2 * h
    q1, q2, q3 = sines(h)

    def series_xy(slope, shift_y, drift):
        xs, ys = [], []
        for pop in (0, 1):
            for i in range(h):
                prev = xs[-1] if i > 0 else q1[0]
                xs.append(0.5 * prev + 0.5 * q1[i] + (drift if pop else 0.0))
        for i in range(n):
            pop = 1 if i >= h else 0
            lag = xs[max(i - 1, 0)]
            ys.append((slope[pop]) * lag + 0.4 * q2[i % h] + (shift_y if pop else 0.0))
        return xs, ys

    sx, sy = series_xy((0.8, 1.1), 2.0, 0.5)
    pops = [[0, h], [h, n]]
    out["temporal_cpdag_change"] = {
        "kind": "change",
        "series": True,
        "graph": {
            "class": "temporal_cpdag",
            "nodes": [[0, 1], [0, 0], [1, 0]],
            "directed": [[0, 1], [0, 2]],
            "undirected": [[1, 2]],
        },
        "columns": {"x": sx, "y": sy},
        "lags": [["x", 1], ["y", 1]],
        "populations": pops,
        "target": "y",
        "component_ids": {"x": 0, "y": 1, "x@1": 2, "y@1": 3},
        "members": [
            member(
                "x -> y in every period",
                1,
                "identified",
                {"x@1": [], "y@1": ["x@1"], "x": ["x@1"], "y": ["x@1", "x"]},
                [player("x@1"), player("x"), player("y")],
            ),
            member(
                "y -> x in every period",
                1,
                "identified",
                {"y@1": [], "x@1": ["y@1"], "y": ["x@1"], "x": ["x@1", "y"]},
                [player("y@1"), player("x@1"), player("y")],
            ),
        ],
    }
    out["temporal_pag_change"] = {
        "kind": "change",
        "series": True,
        "graph": {"class": "temporal_pag", "nodes": [[0, 1], [1, 0]], "marked": [[0, 1, "circle", "arrow"]]},
        "columns": {"x": sx, "y": sy},
        "lags": [["x", 1]],
        "populations": pops,
        "target": "y",
        "component_ids": {"x": 0, "y": 1, "x@1": 2},
        "members": [
            member("x@1 -> y (hedged)", 1, "hedged", {"x@1": [], "y": ["x@1"]}, []),
            member("x@1 <-> y", 1, "identified", {"x@1": [], "y": []}, [player("y")]),
        ],
    }
    ca, cb, ct = [], [], []
    for pop in (0, 1):
        off = 1.0 if pop else 0.0
        for i in range(h):
            ca.append(0.5 * q1[i] + off)
            cb.append(0.5 * (-0.6 * q1[i] + 0.8 * q2[i]) + off)
    for i in range(n):
        pop = 1 if i >= h else 0
        lag = max(i - 1, 0)
        ct.append(ca[lag] + 2.0 * cb[lag] + (0.5 if pop else 0.0) + 0.2 * q3[i % h])
    out["temporal_admg_change"] = {
        "kind": "change",
        "series": True,
        "graph": {
            "class": "temporal_admg",
            "nodes": [[0, 1], [1, 1], [2, 0]],
            "directed": [[0, 2], [1, 2]],
            "bidirected": [[0, 1]],
        },
        "columns": {"a": ca, "b": cb, "t": ct},
        "lags": [["a", 1], ["b", 1]],
        "populations": pops,
        "target": "t",
        "component_ids": {"a@1": 3, "b@1": 4, "t": 2},
        "members": [
            member("a@1 -> t <- b@1, a@1 <-> b@1", 1, "identified", {"a@1": [], "b@1": [], "t": ["a@1", "b@1"]}, [player("a@1", "b@1"), player("t")]),
        ],
    }
    return out




def build_posterior_cases(anomaly_cases, change_cases):
    """Graph posteriors over the atoms of the collider table and of the x-y table: each atom is
    attributed as the structure it stands for, a class atom shares its probability equally among
    its completions, and a mask that is no structure of the posterior's kind is unidentified mass."""
    out = {}
    base = anomaly_cases["admg_collider_anomaly"]
    chg = change_cases["admg_collider_change"]
    dag_atoms = [
        member("a -> t <- b", 0.5, "identified", {"a": [], "b": [], "t": ["a", "b"]}, [player("a"), player("b"), player("t")]),
        member("a -> t, b apart", 0.3, "identified", {"a": [], "t": ["a"]}, [player("a"), player("t")]),
    ]
    admg_atoms = [
        member("a -> t <- b, a <-> b", 0.6, "identified", {"a": [], "b": [], "t": ["a", "b"]}, [player("a", "b"), player("t")]),
        member("a -> t <- b", 0.25, "identified", {"a": [], "b": [], "t": ["a", "b"]}, [player("a"), player("b"), player("t")]),
    ]
    for kind, atoms, outside, spec in [
        ("dag", dag_atoms, 0.2, {"kind": "dag", "weights": [0.5, 0.3, 0.2], "edges": [[[0, 2], [1, 2]], [[0, 2]], [[0, 2], [2, 0]]]}),
        ("admg", admg_atoms, 0.15, {"kind": "admg", "weights": [0.6, 0.25, 0.15], "edges": [[[0, 2], [1, 2], [0, 1], [1, 0]], [[0, 2], [1, 2]], [[0, 1], [1, 2], [2, 0]]]}),
    ]:
        for query, src in (("anomaly", base), ("change", chg)):
            case = {k: v for k, v in src.items() if k not in ("graph", "members", "derived")}
            case.update({"graph": {"class": "posterior"}, "posterior": spec, "members": copy.deepcopy(atoms), "outside_weight": outside})
            out[f"posterior_{kind}_{query}"] = case
    # An atom whose mechanisms cannot be fitted (a constant parent leaves the regression rank
    # deficient) is unevaluable mass, neither unidentified nor redistributed.
    for query, src in (("anomaly", base), ("change", chg)):
        cols = {"a": src["columns"]["a"], "c": [1.0] * len(src["columns"]["a"]), "t": src["columns"]["t"]}
        case = {k: v for k, v in src.items() if k not in ("graph", "members", "derived", "columns")}
        case.update(
            {
                "columns": cols,
                "component_ids": {"a": 0, "c": 1, "t": 2},
                "graph": {"class": "posterior"},
                "posterior": {"kind": "dag", "weights": [0.5, 0.5], "edges": [[[0, 2]], [[1, 2]]]},
                "members": [
                    member("a -> t", 0.5, "identified", {"a": [], "t": ["a"]}, [player("a"), player("t")]),
                    member("c -> t, c constant", 0.5, "failed", {"c": [], "t": ["c"]}, []),
                ],
                "outside_weight": 0.0,
            }
        )
        out[f"posterior_dag_unevaluable_{query}"] = case
    xy_a = anomaly_cases["cpdag_pair_anomaly"]
    xy_c = change_cases["cpdag_pair_change"]
    cpdag_members = [
        member("x -> y (completion of the class atom)", 0.3, "identified", {"x": [], "y": ["x"]}, [player("x"), player("y")]),
        member("y -> x (completion of the class atom)", 0.3, "identified", {"y": [], "x": ["y"]}, [player("y")]),
        member("x -> y (compelled atom)", 0.4, "identified", {"x": [], "y": ["x"]}, [player("x"), player("y")]),
    ]
    cpdag_spec = {"kind": "cpdag", "weights": [0.6, 0.4], "atoms": [{"undirected": [[0, 1]], "directed": []}, {"undirected": [], "directed": [[0, 1]]}]}
    for query, src in (("anomaly", xy_a), ("change", xy_c)):
        case = {k: v for k, v in src.items() if k not in ("graph", "members", "derived")}
        case.update({"graph": {"class": "posterior"}, "posterior": cpdag_spec, "members": copy.deepcopy(cpdag_members), "outside_weight": 0.0})
        out[f"posterior_cpdag_{query}"] = case
    # a PAG atom x o-> y shares its probability among its MAG completions
    pag_members = [
        member("x -> y (completion of the circle-arrow atom, hedged)", 0.3, "hedged", {"x": [], "y": ["x"]}, []),
        member("x <-> y (completion of the circle-arrow atom)", 0.3, "identified", {"x": [], "y": []}, [player("y")]),
        member("x <-> y (bidirected atom)", 0.4, "identified", {"x": [], "y": []}, [player("y")]),
    ]
    pag_spec = {"kind": "pag", "weights": [0.6, 0.4], "atoms": [{"edges": [[0, 1]], "circles": [[0, 1]]}, {"edges": [[0, 1], [1, 0]], "circles": []}]}
    for query, src in (("anomaly", xy_a), ("change", xy_c)):
        case = {k: v for k, v in src.items() if k not in ("graph", "members", "derived")}
        case.update({"graph": {"class": "posterior"}, "posterior": pag_spec, "members": copy.deepcopy(pag_members), "outside_weight": 0.0})
        out[f"posterior_pag_{query}"] = case
    # lag templates of the two-variable series
    ts_a = anomaly_cases["temporal_cpdag_anomaly"]
    ts_c = change_cases["temporal_cpdag_change"]
    dag_members = [
        member("x@1 -> y", 0.7, "identified", {"x@1": [], "y": ["x@1"]}, [player("x@1"), player("y")]),
        member("no lagged edge", 0.3, "identified", {"y": []}, [player("y")]),
    ]
    dag_spec = {"kind": "temporal_dag", "weights": [0.7, 0.3], "atoms": [{"lag_edges": [[0, 1]]}, {"lag_edges": []}]}
    cpdag_t_members = [
        member("x -> y in every period (class atom)", 0.3, "identified", ts_a["members"][0]["parents"], ts_a["members"][0]["players"]),
        member("y -> x in every period (class atom)", 0.3, "identified", ts_a["members"][1]["parents"], ts_a["members"][1]["players"]),
        member("x -> y in every period (compelled atom)", 0.4, "identified", ts_a["members"][0]["parents"], ts_a["members"][0]["players"]),
    ]
    cpdag_t_spec = {
        "kind": "temporal_cpdag",
        "weights": [0.6, 0.4],
        "atoms": [
            {"undirected": [[0, 1]], "directed": [], "lag_edges": [[0, 0], [0, 1]]},
            {"undirected": [], "directed": [[0, 1]], "lag_edges": [[0, 0], [0, 1]]},
        ],
    }
    pag_t_members = [
        member("x -> y in every period (completion of the circle-arrow atom, hedged)", 0.3, "hedged", {"x@1": [], "y": ["x@1"]}, []),
        member("x <-> y in every period (completion of the circle-arrow atom)", 0.3, "identified", {"x@1": [], "y": ["x@1"]}, [player("x@1"), player("y")]),
        member("x <-> y in every period (bidirected atom)", 0.4, "identified", {"x@1": [], "y": ["x@1"]}, [player("x@1"), player("y")]),
    ]
    pag_t_spec = {
        "kind": "temporal_pag",
        "weights": [0.6, 0.4],
        "atoms": [
            {"edges": [[0, 1]], "circles": [[0, 1]], "lag_edges": [[0, 1]]},
            {"edges": [[0, 1], [1, 0]], "circles": [], "lag_edges": [[0, 1]]},
        ],
    }
    for kind, spec, members_ in (
        ("temporal_dag", dag_spec, dag_members),
        ("temporal_cpdag", cpdag_t_spec, cpdag_t_members),
        ("temporal_pag", pag_t_spec, pag_t_members),
    ):
        for query, src in (("anomaly", ts_a), ("change", ts_c)):
            case = {k: v for k, v in src.items() if k not in ("graph", "members", "derived")}
            case["lags"] = [["x", 1], ["y", 1]]
            case["component_ids"] = {"x": 0, "y": 1, "x@1": 2, "y@1": 3}
            case.update({"graph": {"class": "posterior"}, "posterior": spec, "members": copy.deepcopy(members_), "outside_weight": 0.0})
            out[f"posterior_{kind}_{query}"] = case
    # lag templates of a three-variable series: a(t-1) -> t and b(t) -> t, with and without the
    # latent common cause a(t-1) <-> b(t) a lagged latent edge can name
    m = 48
    q1, q2, q3 = sines(m)
    pa = [0.5 * v for v in q1]
    pb = [0.5 * (-0.6 * p + 0.8 * q) for p, q in zip(q1, q2)]
    pt = [pa[max(i - 1, 0)] + 2.0 * pb[i] + 0.3 * q3[i] for i in range(m)]
    pt[-1] += 7.0
    admg_t_members = [
        member(
            "a@1 -> t <- b, a@1 <-> b",
            0.7,
            "identified",
            {"a@1": [], "b": [], "t": ["a@1", "b"]},
            [player("a@1", "b", component="b"), player("t")],
        ),
        member(
            "a@1 -> t <- b",
            0.3,
            "identified",
            {"a@1": [], "b": [], "t": ["a@1", "b"]},
            [player("a@1"), player("b"), player("t")],
        ),
    ]
    admg_t_spec = {
        "kind": "temporal_admg",
        "weights": [0.7, 0.3],
        "atoms": [
            {"directed": [[1, 2]], "lag_edges": [[0, 2]], "lag_latent": [[0, 1]]},
            {"directed": [[1, 2]], "lag_edges": [[0, 2]], "lag_latent": []},
        ],
    }
    h = 60
    t1, t2, t3 = sines(h)
    ca, cb, ct = [], [], []
    for pop in (0, 1):
        off = 1.0 if pop else 0.0
        for i in range(h):
            ca.append(0.5 * t1[i] + off)
            cb.append(0.5 * (-0.6 * t1[i] + 0.8 * t2[i]) + off)
    for i in range(2 * h):
        pop = 1 if i >= h else 0
        ct.append(ca[max(i - 1, 0)] + 2.0 * cb[i] + (0.5 if pop else 0.0) + 0.2 * t3[i % h])
    tail = {
        "series": True,
        "lags": [["a", 1]],
        "target": "t",
        "component_ids": {"a@1": 3, "b": 1, "t": 2},
        "graph": {"class": "posterior"},
        "posterior": admg_t_spec,
        "members": admg_t_members,
        "outside_weight": 0.0,
    }
    out["posterior_temporal_admg_anomaly"] = {
        "kind": "anomaly",
        "columns": {"a": pa, "b": pb, "t": pt},
        "reference": {"kind": "fixed", "center": 0.0, "scale": 2.5},
        **copy.deepcopy(tail),
    }
    out["posterior_temporal_admg_change"] = {
        "kind": "change",
        "columns": {"a": ca, "b": cb, "t": ct},
        "populations": [[0, h], [h, 2 * h]],
        **copy.deepcopy(tail),
    }
    return out


def all_inputs():
    cases, h = build_cases()
    cases.update(build_temporal_change_cases(h))
    anomaly = {k: v for k, v in cases.items() if v["kind"] == "anomaly"}
    change = {k: v for k, v in cases.items() if v["kind"] == "change"}
    cases.update(build_posterior_cases(anomaly, change))
    return cases


# ----------------------------------------------------------------------------------------------
# Fixture assembly and checks


def inputs_of(case):
    return {k: v for k, v in case.items() if k != "derived"}


def build_document(old_oracle=None):
    cases = all_inputs()
    for case in cases.values():
        case["column_order"] = list(case["columns"])
        case["derived"] = derive(case)
    document = {
        "fixture_id": "class_attribution_truth",
        "description": "Anomaly and change attribution over a CPDAG, a PAG, a mixed graph and their one-lag temporal forms, derived from the definitions in exact rationals (see reference.py).",
        "draws": DRAWS,
        "draw_sigmas": SIGMAS,
        "cases": cases,
    }
    if old_oracle is not None:
        document["oracle"] = old_oracle
    return document


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


def same_tree(got, want, label, problems, tol=1e-9):
    if isinstance(want, dict):
        if not isinstance(got, dict) or got.keys() != want.keys():
            problems.append(f"{label}: keys differ")
            return
        for key in want:
            same_tree(got[key], want[key], f"{label}.{key}", problems, tol)
    elif isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            problems.append(f"{label}: length differs")
            return
        for i, (g, w) in enumerate(zip(got, want)):
            same_tree(g, w, f"{label}[{i}]", problems, tol)
    elif isinstance(want, (int, float)) and not isinstance(want, bool):
        close(got, want, tol, label, problems)
    else:
        if got != want:
            problems.append(f"{label}: derived {got!r} != fixture {want!r}")


def check(document):
    problems = []
    for name, case in document["cases"].items():
        derived = derive(inputs_of(case))
        same_tree(json.loads(json.dumps(derived)), case["derived"], name, problems)
    return problems


MUTATIONS = [
    ("an anomaly contribution changed", lambda d: d["cases"]["cpdag_pair_anomaly"]["derived"]["merged"]["contributions"][3].__setitem__(0, 9.0)),
    ("an identified-set bound changed", lambda d: d["cases"]["cpdag_pair_anomaly"]["derived"]["merged"]["upper"].__setitem__(0, 9.0)),
    ("a change contribution changed", lambda d: d["cases"]["cpdag_pair_change"]["derived"]["merged"]["contributions"].__setitem__(0, 9.0)),
    ("the identified mass changed", lambda d: d["cases"]["pag_circle_arrow_anomaly"]["derived"]["merged"].__setitem__("identified_mass", 1.0)),
    ("a member graph changed", lambda d: d["cases"]["cpdag_pair_anomaly"]["members"][0]["parents"].__setitem__("y", [])),
    ("a table value changed", lambda d: d["cases"]["admg_collider_anomaly"]["columns"]["t"].__setitem__(5, 1.5)),
]


def self_test(document):
    failures = []
    if check(document):
        failures.append("the unmodified fixture is rejected")
    for label, mutate in MUTATIONS:
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"accepted: {label}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        old = None
        if FIXTURE.exists():
            old = json.loads(FIXTURE.read_text(encoding="utf-8")).get("oracle")
        FIXTURE.write_text(json.dumps(build_document(old), indent=1) + "\n", encoding="utf-8")
        print("written")
        return 0
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
