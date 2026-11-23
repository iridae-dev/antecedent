# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Frozen tree maps evaluated exactly: the part of the tree-ensemble families this repository owns.

Regenerate with:  uv run --script registry_tree_ensemble_reference.py > registry_tree_ensemble_reference.json

Tree growth (split search, bootstrap and feature subsets, boosting) is done by the pinned
providers forust-ml and smartcore and exported to a provider-independent tree map; nothing here
re-derives it.  The tree maps below are fixed by hand (thresholds and leaves are dyadic
rationals), so every number is an exact rational except the sigmoid, taken at 60 digits:

  one row   walks each tree from node 0: NaN goes to the `missing` child, otherwise left when
            x < threshold (or x == threshold when `inclusive`), else right; a leaf with no value
            casts no vote.
  sum       base + sum of the voting leaves                      (boosted maps)
  mean      base + (sum of the voting leaves) / (votes)          (forest maps)
  link      boosted classifier: sigmoid of the sum, then clipped to [1e-9, 1 - 1e-9]; forest
            classifier: the mean clipped to [1e-9, 1 - 1e-9]; regression, and a map that sets
            neither flag: the value as it is.
  classes   a binary support has one classifier giving P(second level); k > 2 levels have one
            classifier per level and the k scores are divided by their total.
  density   regression: Gaussian about the map with the stored sigma; categorical: ln of the
            class probability floored at 2^-52, -inf for a value outside the support.

Rows include values equal to a threshold, below every threshold, above every threshold, NaN and
a raw boosted score large enough to saturate the sigmoid to exactly one.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60

PROV = {"spec": "gradient_boosted_trees", "implementation": "frozen", "version": "1"}


def leaf(v):
    return {"kind": "leaf", "value": v}


def split(feature, threshold, left, right, missing, inclusive=False):
    return {"kind": "split", "feature": feature, "threshold": threshold, "inclusive": inclusive,
            "left": left, "right": right, "missing": missing}


def predictor(columns, trees, base, average, logistic, probability, spec):
    return {"version": 1, "columns": columns, "provenance": dict(PROV, spec=spec),
            "model": {"kind": "trees", "trees": trees, "base": base, "average": average,
                      "logistic": logistic, "probability": probability}}


def walk(tree, row):
    i = 0
    while True:
        node = tree[i]
        if node["kind"] == "leaf":
            return node["value"]
        v = row[node["feature"]]
        if v is None:  # NaN
            i = node["missing"]
        elif F(v) < F(node["threshold"]) or (node["inclusive"] and F(v) == F(node["threshold"])):
            i = node["left"]
        else:
            i = node["right"]


def evaluate(p, row):
    m = p["model"]
    votes = [walk(t, row) for t in m["trees"]]
    votes = [F(v) for v in votes if v is not None]
    assert votes
    total = sum(votes)
    raw = F(m["base"]) + (total / len(votes) if m["average"] else total)
    r = mp.mpf(raw.numerator) / raw.denominator
    if m["logistic"]:
        return min(max(1 / (1 + mp.e ** (-r)), mp.mpf("1e-9")), 1 - mp.mpf("1e-9"))
    if m["probability"]:
        return min(max(r, mp.mpf("1e-9")), 1 - mp.mpf("1e-9"))
    return r


ROWS = [
    [0.0, 0.0], [0.5, 1.0], [0.25, 0.75], [-1000.0, -1000.0], [1000.0, 1000.0], [0.5, None],
    [None, 0.25], [None, None], [0.75, 0.5], [0.125, 2.0], [1.0, -0.5], [0.5, 0.5],
]


def boosted_regression():
    t1 = [split(0, 0.5, 1, 2, 2), leaf(-1.5), split(1, 0.75, 3, 4, 3), leaf(0.5), leaf(2.0)]
    t2 = [split(1, 0.25, 1, 2, 1), leaf(0.125), leaf(-0.375)]
    t3 = [leaf(0.0625)]
    return predictor(2, [t1, t2, t3], 0.25, False, False, False, "gradient_boosted_trees")


def forest_regression():
    t1 = [split(0, 0.5, 1, 2, 2, inclusive=True), leaf(1.0), split(1, 1.0, 3, 4, 4, inclusive=True),
          leaf(2.0), leaf(None)]
    t2 = [split(1, 0.5, 1, 2, 1, inclusive=True), leaf(0.5), leaf(3.0)]
    t3 = [split(0, 0.125, 1, 2, 1, inclusive=True), leaf(-1.0), leaf(1.5)]
    return predictor(2, [t1, t2, t3], 0.0, True, False, False, "random_forest")


def boosted_class(offset, scale):
    t1 = [split(0, 0.5, 1, 2, 2), leaf(-1.0 * scale), leaf(1.5 * scale + offset)]
    t2 = [split(1, 0.5, 1, 2, 1), leaf(-0.5 * scale), leaf(0.75 * scale)]
    return predictor(2, [t1, t2], 0.125, False, True, False, "gradient_boosted_trees")


def forest_class(a, b, c):
    t1 = [split(0, 0.5, 1, 2, 2, inclusive=True), leaf(a), leaf(b)]
    t2 = [split(1, 0.5, 1, 2, 1, inclusive=True), leaf(c), leaf(b)]
    return predictor(2, [t1, t2], 0.0, True, False, True, "random_forest")


def preds(p):
    return [float(evaluate(p, r)) for r in ROWS]


def main():
    out = {"rows": ROWS}
    reg = {"boosted_regression": boosted_regression(), "forest_regression": forest_regression()}
    out["regression"] = [{"name": k, "predictor": v, "predictions": preds(v)} for k, v in reg.items()]
    sigma = 0.3125
    obs = [0.25, -0.5, 1.0, 0.0, 2.0, 0.125, 1.75, 0.5, -1.25, 0.875, 0.0, 3.0]
    out["additive_density"] = []
    for case in out["regression"]:
        p = case["predictor"]
        lp = []
        for r, y in zip(ROWS, obs):
            mu = evaluate(p, r)
            s = mp.mpf(F(sigma).numerator) / F(sigma).denominator
            lp.append(float(-mp.log(2 * mp.pi * s * s) / 2 - (mp.mpf(F(y).numerator) / F(y).denominator - mu) ** 2 / (2 * s * s)))
        case["sigma"] = sigma
        case["observations"] = obs
        case["log_density"] = lp
    eps = mp.mpf(2) ** -52

    def floor_ln(p):
        return float(mp.log(max(p, eps)))

    # binary categorical: one classifier for P(second level); the boosted one saturates on some rows
    unclipped = forest_class(0.0, 1.0, 0.0)
    unclipped["model"]["probability"] = False  # a foreign map of 0/1 leaf means: exactly 0 and 1 occur
    binary = {"boosted": boosted_class(45.0, 2.0), "forest": forest_class(0.125, 0.875, 0.25),
              "unclipped_indicator": unclipped}
    out["binary"] = []
    for name, p in binary.items():
        probs = [evaluate(p, r) for r in ROWS]
        rows_lp = []
        for q in probs:
            rows_lp.append([floor_ln(1 - q), floor_ln(q)])
        out["binary"].append({"name": name, "predictor": p, "support": [10.0, 20.0],
                              "p_second": [float(q) for q in probs], "log_probability": rows_lp,
                              "clipped_rows": [i for i, q in enumerate(probs) if q in (mp.mpf("1e-9"), 1 - mp.mpf("1e-9"))]})
    # three categories: one classifier per level, renormalised
    ps = [forest_class(0.125, 0.875, 0.25), forest_class(0.5, 0.25, 0.75), forest_class(0.0625, 0.375, 0.5)]
    # the boosted three-class variant: scores are sigmoids, not probabilities
    pb = [boosted_class(0.0, 1.0), boosted_class(0.0, -1.0), boosted_class(0.0, 1.5)]
    out["three_level"] = []
    for name, group in [("forest", ps), ("boosted", pb)]:
        rows_p, rows_lp = [], []
        for r in ROWS:
            q = [evaluate(p, r) for p in group]
            tot = sum(q)
            pr = [v / tot for v in q]
            rows_p.append([float(v) for v in pr])
            rows_lp.append([floor_ln(v) for v in pr])
        out["three_level"].append({"name": name, "predictors": group, "support": [1.0, 2.0, 4.0],
                                   "probabilities": rows_p, "log_probability": rows_lp})
    print(json.dumps(out, indent=1))


main()
