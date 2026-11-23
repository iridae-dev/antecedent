# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Finite-law conditioning, intervention, measured-indicator and threshold queries, exactly.

Regenerate with:  uv run --script finite_law_conditioning_reference.py > finite_law_conditioning_reference.json

Written from the definitions with Fraction arithmetic on dyadic tables (so the doubles handed to
the product are the exact rationals).  A model is a list of variables in topological order, each
with a conditional table.  The oracle never uses the factorization identities the product relies
on in the form it uses them:
  * observational joint  P(v) = prod_i T_i(v_i | parents)   enumerated over every assignment;
  * P(Y | e)             = sum over assignments matching (Y, e) / sum matching e;
  * do(A = a)            builds a NEW model with the tables of the intervened variables replaced by
                         point masses and enumerates it (the mutilated model of the definition);
  * observation after intervention conditions that mutilated joint;
  * a measured indicator is ADDED as an extra variable M with P(M = m | C) = L[c], and the answer
    is conditioning on M = m in the enlarged model (so P(y | m, o) and the construct posterior
    come from the definition of conditional probability, not the sum formula in the docstring);
  * threshold crossing lists P(signal = s | background) and P(outcome | s, background) per level
    and compares each with the threshold exactly.
Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json
import random
from fractions import Fraction as F


def dy(rng, k, grain=16):
    """k dyadic probabilities summing to 1 (some may be zero)."""
    cuts = sorted(rng.randint(0, grain) for _ in range(k - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [grain])]
    return [F(p, grain) for p in parts]


def table_rows(rng, parents, levels, zero_level=None):
    rows = []
    for config in itertools.product(*[p[1] for p in parents]):
        if zero_level is None:
            probs = dy(rng, len(levels))
        else:
            probs = dy(rng, len(levels) - 1)
            probs.insert(zero_level, F(0))
        rows.append((list(config), probs))
    return rows


def model_joint(variables, override=None):
    """Enumerate the joint of the model; override maps a variable name to a fixed level."""
    override = override or {}
    names = [v["name"] for v in variables]
    out = {}
    for assign in itertools.product(*[v["levels"] for v in variables]):
        a = dict(zip(names, assign))
        p = F(1)
        for v in variables:
            if v["name"] in override:
                p *= 1 if a[v["name"]] == override[v["name"]] else 0
                continue
            cfg = [a[q] for q in v["parents"]]
            probs = next(pr for c, pr in v["rows"] if c == cfg)
            p *= probs[v["levels"].index(a[v["name"]])]
        if p:
            out[assign] = p
    return names, out


def cond(names, joint, outcome, levels, evidence):
    oi = names.index(outcome)
    tot = F(0)
    num = {l: F(0) for l in levels}
    for assign, p in joint.items():
        if all(assign[names.index(k)] == v for k, v in evidence.items()):
            tot += p
            num[assign[oi]] += p
    if tot == 0:
        return None, None
    return tot, {l: num[l] / tot for l in levels}


def fl(x):
    return float(x)


def build(seed):
    rng = random.Random(seed)
    Z = {"name": "Z", "levels": ["a", "b", "c"], "parents": [], "rows": [([], dy(rng, 3))]}
    S = {"name": "S", "levels": ["lo", "hi"], "parents": ["Z"]}
    S["rows"] = table_rows(rng, [(None, Z["levels"])], S["levels"])
    W = {"name": "W", "levels": ["n", "p", "x"], "parents": ["S"]}
    W["rows"] = table_rows(rng, [(None, S["levels"])], W["levels"], zero_level=2)
    Y = {"name": "Y", "levels": ["0", "1", "2"], "parents": ["S", "Z"]}
    Y["rows"] = table_rows(rng, [(None, S["levels"]), (None, Z["levels"])], Y["levels"])
    return [Z, S, W, Y]


def jv(v):
    return {"name": v["name"], "levels": v["levels"], "parents": v["parents"],
            "rows": [{"config": c, "probs": [fl(x) for x in p]} for c, p in v["rows"]]}


def dist(d, levels):
    return [[l, fl(d[l])] for l in levels]


def main():
    models = []
    for seed in (11, 12, 13):
        variables = build(seed)
        by = {v["name"]: v for v in variables}
        names, joint = model_joint(variables)
        queries = []
        # conditioning
        for evid in ({"S": "hi"}, {"S": "lo", "W": "p"}, {"W": "n", "Z": "c"}, {"Z": "a", "S": "hi", "W": "p"}):
            for out in ("Y", "Z"):
                if out in evid:
                    continue
                tot, d = cond(names, joint, out, by[out]["levels"], evid)
                q = {"kind": "condition", "outcome": out, "evidence": evid}
                if tot is None:
                    q["error"] = "zero_evidence"
                else:
                    q["evidence_probability"] = fl(tot)
                    q["levels"] = dist(d, by[out]["levels"])
                queries.append(q)
        # intervention (mutilated model) and conditioning vs intervention gap
        for do in ({"S": "hi"}, {"S": "lo"}, {"S": "hi", "W": "n"}, {"Z": "b"}):
            n2, j2 = model_joint(variables, override=do)
            tot, d = cond(n2, j2, "Y", by["Y"]["levels"], {})
            queries.append({"kind": "intervene", "outcome": "Y", "do": do, "levels": dist(d, by["Y"]["levels"])})
        # observation after intervention
        for do, obs in (({"S": "hi"}, {"W": "p"}), ({"S": "lo"}, {"Z": "b"}), ({"S": "hi"}, {"W": "n", "Z": "a"})):
            n2, j2 = model_joint(variables, override=do)
            tot, d = cond(n2, j2, "Y", by["Y"]["levels"], obs)
            q = {"kind": "after", "outcome": "Y", "do": do, "observed": obs}
            if tot is None:
                q["error"] = "zero_evidence"
            else:
                q["evidence_probability"] = fl(tot)
                q["levels"] = dist(d, by["Y"]["levels"])
            queries.append(q)
        # measured indicator: add M with P(M=m|Z) = L[Z]; condition on M = m in the enlarged model
        for lik, others in (([F(3, 4), F(1, 8), F(1, 16)], {}), ([F(1, 2), F(1, 2), F(1, 2)], {"W": "p"}),
                            ([F(0), F(1), F(1, 4)], {"S": "hi"}), ([F(1, 16), F(0), F(0)], {})):
            Mrows = [([z], [l, 1 - l]) for z, l in zip(by["Z"]["levels"], lik)]
            M = {"name": "M", "levels": ["m", "notm"], "parents": ["Z"], "rows": Mrows}
            nm, jm = model_joint(variables + [M])
            ev = dict(others)
            ev["M"] = "m"
            for out in ("Y", "Z", "S"):
                if out in others:
                    continue
                tot, d = cond(nm, jm, out, by[out]["levels"], ev)
                tot2, dc = cond(nm, jm, "Z", by["Z"]["levels"], ev)
                q = {"kind": "measured", "outcome": out, "construct": "Z", "observed": others,
                     "likelihood": [fl(x) for x in lik]}
                if tot is None:
                    q["error"] = "zero_evidence"
                else:
                    q["evidence_probability"] = fl(tot)
                    q["outcome_levels"] = dist(d, by[out]["levels"])
                    q["construct_levels"] = dist(dc, by["Z"]["levels"])
                queries.append(q)
        # threshold crossing: thresholds at an attained conditional (equality counts), just above, just below
        for signal, background in (("W", {"Z": "b"}), ("S", {}), ("W", {})):
            for target in ("0", "2"):
                for direction in ("raise", "lower"):
                    # per-level exact values
                    oi = names.index("Y")
                    si = names.index(signal)
                    mass, hit = {}, {}
                    total = F(0)
                    for assign, p in joint.items():
                        if all(assign[names.index(k)] == v for k, v in background.items()):
                            total += p
                            mass[assign[si]] = mass.get(assign[si], F(0)) + p
                            if assign[oi] == target:
                                hit[assign[si]] = hit.get(assign[si], F(0)) + p
                    if total == 0:
                        continue
                    conds = {s: hit.get(s, F(0)) / m for s, m in mass.items()}
                    vals = sorted(set(conds.values()))
                    for thr in (vals[len(vals) // 2], vals[0], vals[-1], vals[-1] + F(1, 1000) if vals[-1] < 1 else F(1)):
                        thr = min(max(thr, F(0)), F(1))
                        crosses = {s: (c >= thr if direction == "raise" else c <= thr) for s, c in conds.items()}
                        q = {"kind": "threshold", "outcome": "Y", "outcome_level": target, "signal": signal,
                             "background": background, "threshold": fl(thr), "direction": direction}
                        if not any(crosses.values()):
                            q["error"] = "unreachable"
                            ext = max(conds.values()) if direction == "raise" else min(conds.values())
                            q["extremum"] = fl(ext)
                        else:
                            q["levels"] = [{"level": s, "probability": fl(mass[s] / total), "conditional": fl(conds[s]),
                                            "crosses": crosses[s]} for s in by[signal]["levels"] if s in mass]
                            q["crossing_probability"] = fl(sum(mass[s] / total for s in mass if crosses[s]))
                        queries.append(q)
        models.append({"seed": seed, "variables": [jv(v) for v in variables], "queries": queries})
    print(json.dumps(models, indent=1))


main()
