# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the count-fitted finite-table posterior.

Regenerate with:  uv run --script count_fitted_posterior_reference.py > count_fitted_posterior_reference.json

Two independent references, both written from the model:

1. Point values.  The posterior-mean tables are (count + alpha) / (row total + k alpha); the
   conditional P(target = level | evidence) is the sum over the whole joint assignment space in exact
   rational arithmetic (Fraction), for random topologically ordered models with up to five nodes,
   counted and declared nodes, and evidence on a child as well as a root.

2. Posterior quantiles.  For a child Y with parent X and evidence X = x, the row of Y's table for
   X = x has a Dirichlet(counts + alpha) posterior, so the probability of one level is
   Beta(c_l + alpha, sum_{m != l} (c_m + alpha)) exactly (aggregation property of the Dirichlet).
   The quantile function is found at 60 digits by bisection on log x of the regularised incomplete
   beta function.  For a Monte Carlo interval from m independent draws the endpoint at probability p
   must lie between the true quantiles at p -/+ 5 sqrt(p (1 - p) / m); both brackets are recorded.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import math
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60
rng = random.Random(20261028)


def beta_quantile(a, b, p):
    a, b, p = mp.mpf(a), mp.mpf(b), mp.mpf(p)
    if p <= 0:
        return mp.mpf(0)
    if p >= 1:
        return mp.mpf(1)
    lo, hi = mp.mpf(-4000), mp.mpf(0)  # log x
    for _ in range(260):
        mid = (lo + hi) / 2
        if mp.betainc(a, b, 0, mp.exp(mid), regularized=True) < p:
            lo = mid
        else:
            hi = mid
    return mp.exp((lo + hi) / 2)


def fmt(x):
    # the smallest double is about 5e-324; anything below is written as zero
    return "0" if x < mp.mpf("1e-320") else mp.nstr(x, 25)


# ---- 1. exact point values -------------------------------------------------
def exact_point(nodes, alpha, target, level, evidence):
    tables = []
    for node in nodes:
        if node["kind"] == "declared":
            tables.append([F(x) for x in node["table"]])
        else:
            k = node["card"]
            t = []
            for r in range(0, len(node["table"]), k):
                row = node["table"][r:r + k]
                tot = sum(F(c) + F(alpha) for c in row)
                t += [(F(c) + F(alpha)) / tot for c in row]
            tables.append(t)
    n = len(nodes)
    cards = [nd["card"] for nd in nodes]
    joint = ev = F(0)

    def rec(i, assign):
        nonlocal joint, ev
        if i == n:
            if all(assign[nd] == lv for nd, lv in evidence):
                p = F(1)
                for j, nd in enumerate(nodes):
                    cfg = 0
                    for par in nd["parents"]:
                        cfg = cfg * cards[par] + assign[par]
                    p *= tables[j][cfg * nd["card"] + assign[j]]
                ev += p
                if assign[target] == level:
                    joint += p
            return
        for v in range(cards[i]):
            rec(i + 1, assign + [v])

    rec(0, [])
    return joint / ev if ev else None


point_cases = []
for i in range(40):
    n = rng.randint(2, 5)
    nodes = []
    for j in range(n):
        card = rng.randint(2, 3)
        parents = sorted(rng.sample(range(j), rng.randint(0, min(j, 2)))) if j else []
        cfgs = 1
        for p in parents:
            cfgs *= nodes[p]["card"]
        if rng.random() < 0.25:
            # probabilities with denominator 8 are exact in binary, so every row sums to one
            rows = []
            for _ in range(cfgs):
                cuts = sorted(rng.sample(range(1, 8), card - 1))
                parts = [y - x for x, y in zip([0] + cuts, cuts + [8])]
                rows += [x / 8 for x in parts]
            nodes.append({"kind": "declared", "card": card, "parents": parents, "table": rows})
        else:
            nodes.append({"kind": "counts", "card": card, "parents": parents,
                          "table": [rng.randint(0, 12) for _ in range(cfgs * card)]})
    alpha = rng.choice([0.5, 1.0, 2.0, 0.1])
    target = rng.randrange(n)
    level = rng.randrange(nodes[target]["card"])
    others = [j for j in range(n) if j != target]
    evidence = [(j, rng.randrange(nodes[j]["card"])) for j in rng.sample(others, rng.randint(0, min(2, len(others))))]
    want = exact_point(nodes, alpha, target, level, evidence)
    if want is None:
        continue
    point_cases.append({"nodes": nodes, "alpha": alpha, "target": target, "level": level,
                        "evidence": evidence, "point": float(want)})

# ---- 2. Beta quantile brackets ----------------------------------------------
quantile_cases = []
for counts, alpha, level, draws in [
    ([6, 14], 1.0, 0.95, 20000),
    ([47, 3], 1.0, 0.90, 20000),
    ([0, 0], 1.0, 0.95, 20000),
    ([2, 1, 9], 0.5, 0.95, 20000),
    ([0, 0], 0.001, 0.90, 20000),
    ([0, 3], 0.01, 0.95, 20000),
    ([30, 30], 20.0, 0.95, 2000),
    ([5, 5], 1.0, 0.99, 8000),
]:
    k = len(counts)
    lv = k - 1
    a = counts[lv] + alpha
    b = sum(c + alpha for j, c in enumerate(counts) if j != lv)
    lp, up = (1 - level) / 2, 1 - (1 - level) / 2
    rec = {"counts": counts, "alpha": alpha, "level": level, "draws": draws, "target_level": lv}
    for name, p in (("lower", lp), ("upper", up)):
        sd = math.sqrt(p * (1 - p) / draws)
        rec[name] = {"p": p,
                     "min": fmt(beta_quantile(a, b, max(p - 5 * sd, 0))),
                     "max": fmt(beta_quantile(a, b, min(p + 5 * sd, 1)))}
    rec["point"] = float((F(counts[lv]) + F(alpha)) / (sum(F(c) for c in counts) + k * F(alpha)))
    quantile_cases.append(rec)
print(json.dumps({"point": point_cases, "quantile": quantile_cases}, indent=1))
