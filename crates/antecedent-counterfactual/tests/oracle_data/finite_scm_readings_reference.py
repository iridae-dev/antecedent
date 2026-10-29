# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact references for the spread and envelope readings of the finite cross-world executor.

Regenerate with:  uv run --script finite_scm_readings_reference.py > finite_scm_readings_reference.json

Built on the definition-level enumeration of finite_scm_reference.py (every joint exogenous
state with exact fractions, abduction by keeping the states that reproduce the evidence):

  spread    per unit, the exact posterior distribution of the quantity's value over the joint
            exogenous states (value -> Fraction mass) and its variance E[v^2] - E[v]^2 as a
            Fraction; the variance of the mean over units is the sum of the unit variances over
            the squared unit count (units are independent individuals);
  envelope  a declared set of parameterisations (the model and alternatives with other
            exogenous laws and mechanism tables over the same structure); per member the exact
            unit values, and the minimum and maximum of the member means and of each unit.

Repository-authored (MIT OR Apache-2.0).  Uses only the standard library.
"""
import json
import os
import random
from fractions import Fraction

here = os.path.dirname(os.path.abspath(__file__))
# finite_scm_reference.py runs its own main() at import; load its definitions without that call.
source = open(os.path.join(here, "finite_scm_reference.py")).read().rstrip()
assert source.endswith("main()")
namespace = {"__name__": "finite_scm_reference"}
exec(compile(source[: -len("main()")], "finite_scm_reference.py", "exec"), namespace)


class base:  # noqa: N801 - a namespace over the loaded definitions
    joint_states = staticmethod(namespace["joint_states"])
    evaluate = staticmethod(namespace["evaluate"])
    random_case = staticmethod(namespace["random_case"])
    chain_case = staticmethod(namespace["chain_case"])
    dyadic_pmf = staticmethod(namespace["dyadic_pmf"])


def quantity_value(values, q):
    kind = q["kind"]
    if kind == "probability":
        a = q["atom"]
        return int(values[a[0]][a[1]] == a[2])
    if kind == "joint":
        return int(all(values[a[0]][a[1]] == a[2] for a in q["atoms"]))
    (pw, pv), (mw, mv) = q["plus"], q["minus"]
    return values[pw][pv] - values[mw][mv]


def unit_distribution(model, worlds, observed, q):
    factual = [{"interventions": [], "routes": []}]
    dist, mass = {}, Fraction(0)
    for u, weight in base.joint_states(model):
        f = base.evaluate(model, u, factual)[0]
        if any(f[v] != l for v, l in observed):
            continue
        mass += weight
        if weight == 0:
            continue
        v = quantity_value(base.evaluate(model, u, worlds), q)
        dist[v] = dist.get(v, Fraction(0)) + weight
    return {v: w / mass for v, w in dist.items()}, mass


def frac(x):
    return f"{x.numerator}/{x.denominator}"


def spread_of(model, case):
    units, var_sum = [], Fraction(0)
    for unit in case["units"]:
        dist, _ = unit_distribution(model, case["worlds"], unit["observed"], case["quantity"])
        mean = sum(v * w for v, w in dist.items())
        var = sum(v * v * w for v, w in dist.items()) - mean * mean
        var_sum += var
        units.append({
            "unit": unit["unit"],
            "distribution": [[float(v), float(w), frac(w)] for v, w in sorted(dist.items())],
            "variance": float(var), "variance_exact": frac(var), "mean": float(mean),
        })
    n = len(units)
    mv = var_sum / (n * n)
    return {"units": units, "mean_variance": float(mv), "mean_variance_exact": frac(mv)}


def perturbed(rnd, model):
    exogenous = [base.dyadic_pmf(rnd, len(p)) for p in model["exogenous"]]
    variables = []
    for var in model["variables"]:
        v = dict(var)
        v["table"] = [rnd.randrange(var["levels"]) for _ in var["table"]]
        variables.append(v)
    return {"exogenous": exogenous, "variables": variables}


def envelope_of(rnd, case):
    members = [case["model"]]
    for _ in range(rnd.randint(1, 3)):
        members.append(perturbed(rnd, case["model"]))
    rows, usable = [], []
    for m in members:
        vals = []
        ok = True
        for unit in case["units"]:
            dist, mass = unit_distribution(m, case["worlds"], unit["observed"], case["quantity"])
            if mass == 0:
                ok = False
                break
            vals.append(sum(v * w for v, w in dist.items()))
        if ok:
            usable.append(m)
            rows.append(vals)
    means = [sum(v) / len(v) for v in rows]
    n_units = len(case["units"])
    return {
        "members": usable,
        "member_means": [float(m) for m in means],
        "lower": float(min(means)), "upper": float(max(means)),
        "lower_member": means.index(min(means)), "upper_member": means.index(max(means)),
        "unit_lower": [float(min(r[k] for r in rows)) for k in range(n_units)],
        "unit_upper": [float(max(r[k] for r in rows)) for k in range(n_units)],
    }


def main():
    rnd = random.Random(20261029)
    cases = []
    for _ in range(60):
        case = base.random_case(rnd)
        case["spread"] = spread_of(case["model"], case)
        env = envelope_of(rnd, case)
        if len(env["members"]) >= 2:
            case["envelope"] = env
        cases.append(case)
    chain = base.chain_case(rnd)
    chain["spread"] = spread_of(chain["model"], chain)
    print(json.dumps({"cases": cases, "chain": chain}, separators=(",", ":")))


if __name__ == "__main__":
    main()
