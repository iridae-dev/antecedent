# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Brute-force reference for exact cross-world queries on finite structural models.

Regenerate with:  uv run --script finite_scm_reference.py > finite_scm_reference.json

Written from the semantics of abduction, action and prediction (Pearl 2009, ch. 7; Balke and
Pearl 1994), by enumerating every joint exogenous state with integer weights and exact fractions:

  1. weight of a joint state u is the product of the exogenous masses, all dyadic rationals;
  2. a unit's evidence keeps the states whose factual (non-intervened) evaluation reproduces every
     observed (variable, level), the posterior being those weights divided by their sum;
  3. each world re-evaluates the variables in index order with its interventions overriding the
     mechanism, a cross-world route making one parent of one child be read from the source world
     (all worlds advance variable by variable, so a source value is always available), every
     mechanism reading the single shared state u;
  4. the quantity is the posterior mass of the event (Probability, JointProbability) or the
     posterior difference of two mean levels (MeanContrast); the answer is the mean over units.

Models include shared exogenous terms feeding several variables, three-level variables, routes
that read through other worlds, and one chain of seventeen binary exogenous terms (131072 joint
states), which the product executes by variable elimination rather than by enumeration.

Repository-authored (MIT OR Apache-2.0).  Uses only the standard library.
"""
import itertools
import json
import random
from fractions import Fraction


def dyadic_pmf(rnd, states, bits=4):
    total = 1 << bits
    cuts = sorted(rnd.sample(range(1, total), states - 1)) if states > 1 else []
    parts = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    return [p / total for p in parts]  # exact doubles


def random_model(rnd, n, shared_prob):
    levels = [rnd.choice([2, 2, 3]) for _ in range(n)]
    parents = []
    for i in range(n):
        pool = list(range(i))
        rnd.shuffle(pool)
        parents.append(sorted(pool[: rnd.randint(0, min(2, i))]))
    exogenous, assign = [], []
    for i in range(n):
        if exogenous and rnd.random() < shared_prob:
            assign.append(rnd.randrange(len(exogenous)))
        else:
            exogenous.append(dyadic_pmf(rnd, rnd.choice([2, 3, 4])))
            assign.append(len(exogenous) - 1)
    variables = []
    for i in range(n):
        n_cfg = 1
        for p in parents[i]:
            n_cfg *= levels[p]
        states = len(exogenous[assign[i]])
        variables.append({"name": f"v{i}", "levels": levels[i], "parents": parents[i], "exogenous": assign[i],
                          "table": [rnd.randrange(levels[i]) for _ in range(states * n_cfg)]})
    return {"exogenous": exogenous, "variables": variables}


def evaluate(model, u, worlds):
    """Level of every variable in every world under the joint exogenous state u (a tuple)."""
    n = len(model["variables"])
    values = [[None] * n for _ in worlds]
    for i, var in enumerate(model["variables"]):
        n_cfg = 1
        for p in var["parents"]:
            n_cfg *= model["variables"][p]["levels"]
        for w, world in enumerate(worlds):
            forced = dict((v, l) for v, l in world["interventions"])
            if i in forced:
                values[w][i] = forced[i]
                continue
            routes = {(p, c): s for p, c, s in world["routes"]}
            cfg = 0
            for p in var["parents"]:
                source = routes.get((p, i), w)
                cfg = cfg * model["variables"][p]["levels"] + values[source][p]
            values[w][i] = var["table"][u[var["exogenous"]] * n_cfg + cfg]
    return values


def joint_states(model):
    ranges = [range(len(pmf)) for pmf in model["exogenous"]]
    for u in itertools.product(*ranges):
        weight = Fraction(1)
        for pmf, state in zip(model["exogenous"], u):
            weight *= Fraction(pmf[state])
        yield u, weight


def unit_value(model, worlds, observed, quantity):
    factual = [{"interventions": [], "routes": []}]
    total, mass = Fraction(0), Fraction(0)
    for u, weight in joint_states(model):
        f = evaluate(model, u, factual)[0]
        if any(f[v] != l for v, l in observed):
            continue
        mass += weight
        if weight == 0:
            continue
        values = evaluate(model, u, worlds)
        kind = quantity["kind"]
        if kind == "probability":
            a = quantity["atom"]
            total += weight * (values[a[0]][a[1]] == a[2])
        elif kind == "joint":
            total += weight * all(values[a[0]][a[1]] == a[2] for a in quantity["atoms"])
        else:
            (pw, pv), (mw, mv) = quantity["plus"], quantity["minus"]
            total += weight * (values[pw][pv] - values[mw][mv])
    return total / mass, mass


def random_case(rnd, model=None, n=None, units=3):
    n = n or rnd.randint(3, 6)
    model = model or random_model(rnd, n, 0.4)
    nv = len(model["variables"])
    worlds = []
    for w in range(rnd.randint(2, 3)):
        interventions = []
        for v in rnd.sample(range(nv), rnd.randint(0, 2)):
            interventions.append([v, rnd.randrange(model["variables"][v]["levels"])])
        worlds.append({"name": f"w{w}", "interventions": interventions, "routes": []})
    for w, world in enumerate(worlds):
        forced = {v for v, _ in world["interventions"]}
        for c, var in enumerate(model["variables"]):
            if c in forced:
                continue
            for p in var["parents"]:
                if rnd.random() < 0.3:
                    world["routes"].append([p, c, rnd.choice([s for s in range(len(worlds)) if s != w])])
        # one route per (parent, child) pair
        seen, kept = set(), []
        for r in world["routes"]:
            if (r[0], r[1]) not in seen:
                seen.add((r[0], r[1]))
                kept.append(r)
        world["routes"] = kept
    case_units = []
    for k in range(units):
        # observe a random subset of one sampled factual outcome so the evidence has positive mass
        choices = list(joint_states(model))
        u = rnd.choices([c[0] for c in choices], weights=[float(c[1]) for c in choices])[0]
        f = evaluate(model, u, [{"interventions": [], "routes": []}])[0]
        observed = [[v, f[v]] for v in sorted(rnd.sample(range(nv), rnd.randint(0, nv)))]
        case_units.append({"unit": f"unit-{k}", "observed": observed})
    nw = len(worlds)
    atom = lambda: [rnd.randrange(nw), rnd.randrange(nv), rnd.randrange(2)]
    kind = rnd.choice(["probability", "joint", "contrast"])
    if kind == "probability":
        quantity = {"kind": kind, "atom": atom()}
    elif kind == "joint":
        quantity = {"kind": kind, "atoms": [atom() for _ in range(rnd.randint(1, 3))]}
    else:
        quantity = {"kind": kind, "plus": [rnd.randrange(nw), rnd.randrange(nv)], "minus": [rnd.randrange(nw), rnd.randrange(nv)]}
    values = []
    for unit in case_units:
        value, mass = unit_value(model, worlds, unit["observed"], quantity)
        values.append({"value": float(value), "value_exact": f"{value.numerator}/{value.denominator}", "evidence": float(mass)})
    mean = sum(Fraction(v["value_exact"].split("/")[0]) / Fraction(v["value_exact"].split("/")[1]) for v in values) / len(values)
    return {"model": model, "worlds": worlds, "units": case_units, "quantity": quantity, "unit_values": values, "mean": float(mean)}


def chain_case(rnd):
    n = 17
    exogenous = [dyadic_pmf(rnd, 2, bits=3) for _ in range(n)]
    variables = []
    for i in range(n):
        parents = [i - 1] if i else []
        n_cfg = 2 if i else 1
        variables.append({"name": f"c{i}", "levels": 2, "parents": parents, "exogenous": i,
                          "table": [rnd.randrange(2) for _ in range(2 * n_cfg)]})
    model = {"exogenous": exogenous, "variables": variables}
    worlds = [{"name": "w0", "interventions": [[3, 0]], "routes": []},
              {"name": "w1", "interventions": [[3, 1]], "routes": [[15, 16, 0]]}]
    # evidence consistent with one sampled state
    u = tuple(rnd.randrange(2) for _ in range(n))
    f = evaluate(model, u, [{"interventions": [], "routes": []}])[0]
    units = [{"unit": "chain-a", "observed": [[16, f[16]], [9, f[9]]]}, {"unit": "chain-b", "observed": []}]
    quantity = {"kind": "joint", "atoms": [[1, 16, 1], [0, 16, 0]]}
    values = []
    for unit in units:
        value, mass = unit_value(model, worlds, unit["observed"], quantity)
        values.append({"value": float(value), "value_exact": f"{value.numerator}/{value.denominator}", "evidence": float(mass)})
    mean = sum(Fraction(v["value_exact"].split("/")[0]) / Fraction(v["value_exact"].split("/")[1]) for v in values) / 2
    return {"model": model, "worlds": worlds, "units": units, "quantity": quantity, "unit_values": values, "mean": float(mean)}


def main():
    rnd = random.Random(20261028)
    cases = [random_case(rnd) for _ in range(120)]
    cases.append(chain_case(rnd))
    print(json.dumps({"cases": cases}, separators=(",", ":")))


main()
