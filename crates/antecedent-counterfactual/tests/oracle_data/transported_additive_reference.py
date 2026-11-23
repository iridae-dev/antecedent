# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact-rational reference for the transported additive path-specific contrast.

Regenerate with:  uv run --offline --script transported_additive_reference.py > transported_additive_reference.json

Written from the definition, not from the product's closed form.  A world is evaluated by a
recursive structural evaluator in which the edge A -> c carries the treated value when c is in
the world's treated set and the control value otherwise (every other parent is read from the
same world).  The reference does NOT use the fact that the contrast is free of the noise: each
mechanism gets a three-point exogenous noise law (dyadic probabilities), the covariate is drawn
from the TARGET law, and the contrast  E[Y_plus - Y_minus]  is the exact sum over the product
of covariate points and all noise points of  P(z) P(u) (Y_plus(z, u) - Y_minus(z, u)),  the same
noise being fed to both worlds.  The source contrast uses the source law.  Every number is a
Fraction built from doubles (exact), and the expected values are the correctly rounded doubles.
Cases cover a diamond with covariate-dependent intercepts, slopes and coefficients, a two
mediator chain with a direct edge, covariate offsets near 1e6, target weights 2^-20 apart, a
treatment contrast offset by 1e6, a target that is a strict subset of the source, and a
selection node on a mediator with the witness the product must produce (the documented search
order: parents by name, the constant before the slopes by name, first coefficient whose
perturbation by +1 moves the target contrast).

Repository-authored (MIT OR Apache-2.0).  Uses only the standard library.
"""
import itertools
import json
from fractions import Fraction as F


def aff(c, slopes=None):
    return {"constant": c, "slopes": slopes or {}}


def aff_at(a, z):
    return F(a["constant"]) + sum(F(s) * z[k] for k, s in a["slopes"].items())


def world_outcome(model, q, assignment, z, noise):
    order = model["order"]
    mech = {m["name"]: m for m in model["mechanisms"]}
    vals = {}

    def value_of(name):
        if name in vals:
            return vals[name]
        m = mech[name]
        v = aff_at(m["intercept"], z) + F(noise[name])
        for pname, coef in m["parents"]:
            if pname == model["treatment"]:
                pv = F(q["a1"]) if name in assignment else F(q["a0"])
            elif pname in mech:
                pv = value_of(pname)
            else:
                pv = z[pname]
            v += aff_at(coef, z) * pv
        vals[name] = v
        return v

    return value_of(q["outcome"])


def noise_support(model):
    names = [m["name"] for m in model["mechanisms"]]
    pts = [(F(-1, 2), F(1, 4)), (F(0), F(1, 2)), (F(3, 4), F(1, 4))]
    for combo in itertools.product(range(3), repeat=len(names)):
        w = F(1)
        nz = {}
        for n, i in zip(names, combo):
            nz[n] = pts[i][0]
            w *= pts[i][1]
        yield nz, w


def contrast(model, q, law):
    total = F(0)
    for point, w in law:
        z = {k: F(v) for k, v in point.items()}
        for nz, pw in noise_support(model):
            yp = world_outcome(model, q, set(q["plus"]), z, nz)
            ym = world_outcome(model, q, set(q["minus"]), z, nz)
            total += F(w) * pw * (yp - ym)
    return total


def perturb(model, mech, parent, slope, delta=1.0):
    import copy
    m2 = copy.deepcopy(model)
    for m in m2["mechanisms"]:
        if m["name"] != mech:
            continue
        for pname, coef in m["parents"]:
            if pname == parent:
                if slope is None:
                    coef["constant"] += delta
                else:
                    coef["slopes"][slope] += delta
    return m2


def witness(model, q, law, node):
    m = next(x for x in model["mechanisms"] if x["name"] == node)
    base = contrast(model, q, law)
    for pname, coef in sorted(m["parents"], key=lambda p: p[0]):
        for slope in [None] + sorted(coef["slopes"]):
            other = perturb(model, node, pname, slope)
            moved = contrast(other, q, law)
            if abs(moved - base) > F(1, 10**9):
                return {"node": node, "parent": pname, "slope": slope,
                        "base": float(base), "moved": float(moved)}
    return None


def case(name, model, source, target, q, selection=None):
    out = {"name": name, "model": model, "source": source, "target": target, "query": q,
           "target_contrast": float(contrast(model, q, target)),
           "source_contrast": float(contrast(model, q, source))}
    if selection:
        out["selection_node"] = selection
        out["witness"] = witness(model, q, target, selection)
    return out


def diamond():
    return {"treatment": "a", "covariates": ["z"], "order": ["m", "y"], "mechanisms": [
        {"name": "m", "intercept": aff(0.25, {"z": 0.5}),
         "parents": [["a", aff(2.0, {"z": -0.5})], ["z", aff(0.5)]]},
        {"name": "y", "intercept": aff(0.0),
         "parents": [["a", aff(3.0, {"z": 0.75})], ["m", aff(4.0, {"z": 0.125})], ["z", aff(1.5)]]},
    ]}


def diamond_non_dyadic():
    return {"treatment": "a", "covariates": ["z"], "order": ["m", "y"], "mechanisms": [
        {"name": "m", "intercept": aff(0.1, {"z": 0.7}),
         "parents": [["a", aff(1.3, {"z": -0.9})], ["z", aff(0.3)]]},
        {"name": "y", "intercept": aff(0.0),
         "parents": [["a", aff(2.1, {"z": 0.45})], ["m", aff(3.7, {"z": 0.11})], ["z", aff(-1.7)]]},
    ]}


def chain():
    return {"treatment": "a", "covariates": ["z1", "z2"], "order": ["m1", "m2", "y"], "mechanisms": [
        {"name": "m1", "intercept": aff(1.0, {"z1": 0.5}),
         "parents": [["a", aff(1.5, {"z2": 0.25})]]},
        {"name": "m2", "intercept": aff(-0.5),
         "parents": [["a", aff(-1.0, {"z1": 0.5})], ["m1", aff(0.75, {"z2": -0.5})]]},
        {"name": "y", "intercept": aff(0.5, {"z1": 0.25, "z2": -0.25}),
         "parents": [["a", aff(0.5)], ["m1", aff(2.0, {"z1": 0.125})], ["m2", aff(1.25, {"z2": 0.5})]]},
    ]}


def law1(pts):
    return [({"z": z}, w) for z, w in pts]


def law2(pts):
    return [({"z1": a, "z2": b}, w) for a, b, w in pts]


def main():
    cases = []
    q_nde = {"outcome": "y", "a1": 2.0, "a0": 0.5, "plus": ["y"], "minus": []}
    q_nie = {"outcome": "y", "a1": 2.0, "a0": 0.5, "plus": ["m", "y"], "minus": ["y"]}
    q_te = {"outcome": "y", "a1": 2.0, "a0": 0.5, "plus": ["m", "y"], "minus": []}
    src = law1([(0.0, 0.25), (1.0, 0.5), (2.0, 0.25)])
    tgt = law1([(0.0, 0.25), (1.0, 0.75)])
    for nm, q in (("nde", q_nde), ("nie", q_nie), ("total", q_te)):
        cases.append(case(f"diamond_{nm}", diamond(), src, tgt, q))
    # non-dyadic coefficients and laws: the reference works on the exact rationals of the doubles
    nd_src = law1([(0.0, 0.3), (1.0, 0.45), (2.0, 0.25)])
    nd_tgt = law1([(0.0, 0.2), (1.0, 0.8)])
    cases.append(case("diamond_non_dyadic_nie", diamond_non_dyadic(), nd_src, nd_tgt, q_nie))
    cases.append(case("diamond_non_dyadic_total", diamond_non_dyadic(), nd_src, nd_tgt, q_te))
    # a strict-subset target with a different support point weighting, and zero-contrast query
    cases.append(case("diamond_empty_assignments_zero", diamond(), src, tgt,
                      {"outcome": "y", "a1": 2.0, "a0": 0.5, "plus": [], "minus": []}))
    # extreme covariate offsets and unbalanced weights, dyadic so the weights sum to one exactly
    big = law1([(1.0e6, 1 - 2.0 ** -20), (1.0e6 + 4.0, 2.0 ** -20)])
    bigsrc = law1([(1.0e6, 0.5), (1.0e6 + 4.0, 0.25), (1.0e6 - 8.0, 0.25)])
    cases.append(case("diamond_offset_1e6_unbalanced", diamond(), bigsrc, big, q_nie))
    # treatment contrast offset by 1e6
    cases.append(case("diamond_treatment_offset_1e6", diamond(), src, tgt,
                      {"outcome": "y", "a1": 1.0e6 + 1.0, "a0": 1.0e6, "plus": ["m", "y"], "minus": ["y"]}))
    # two mediators, two covariates, direct edge
    s2 = law2([(0.0, 0.0, 0.25), (1.0, 0.0, 0.25), (0.0, 1.0, 0.25), (1.0, 1.0, 0.25)])
    t2 = law2([(1.0, 0.0, 0.125), (0.0, 1.0, 0.875)])
    for nm, plus, minus in (("m1_only", ["m1", "m2", "y"], ["m2", "y"]),
                            ("m2_only", ["m2", "y"], ["y"]),
                            ("direct", ["y"], []),
                            ("total", ["m1", "m2", "y"], [])):
        cases.append(case(f"chain_{nm}", chain(), s2, t2,
                          {"outcome": "y", "a1": 1.0, "a0": -1.0, "plus": plus, "minus": minus}))
    # a selection node on the first mediator: the contrast is sensitive, witness expected
    cases.append(case("diamond_selection_on_mediator", diamond(), src, tgt, q_nie, selection="m"))
    # selection on a mediator the zero contrast does not depend on: no witness
    cases.append(case("diamond_selection_insensitive", diamond(), src, tgt,
                      {"outcome": "y", "a1": 2.0, "a0": 0.5, "plus": [], "minus": []}, selection="m"))
    json.dump(cases, open("/dev/stdout", "w"), separators=(",", ":"))


main()
