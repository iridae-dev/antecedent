# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Mutual information of the graph-label observation channel, at 60 digits.

Regenerate with:  uv run --script graph_entropy_channel_reference.py > graph_entropy_channel_reference.json

The channel observes the categorical feature c(G) of the graph with probability
  p(y | G) = r * 1[y = c(G)] + (1 - r) / k,
r the declared reliability of the design and k the number of distinct feature values.  The
reference builds the joint law p(g, y) = p(g) p(y | g) explicitly and evaluates
I(G; Y) = H(G) - H(G | Y) = sum_{g,y} p(g,y) ln( p(g,y) / (p(g) p(y)) ) in nats, so it does not
use the posterior-entropy decomposition of the product.  The reliability of each design amount
is the declared heuristic map (the product's documented one) evaluated here independently:
  measure k variables:      min(0.99, 1 - exp(-0.75 k))
  intervene on k targets:   min(0.99, 1 - exp(-k))
  observe n rows / n more samples: min(0.95, 1 - 1/(1 + n/50)).

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 60

DESIGNS = [
    ("measure", 1), ("measure", 2), ("measure", 10), ("intervene", 1), ("intervene", 3),
    ("observe", 0), ("observe", 1), ("observe", 50), ("observe", 1000000),
    ("sample", 1), ("sample", 100), ("sample", 3), ("sample", 1000000),
]

PRIORS = {
    "six_graphs_four_labels": {
        "weights": [0.8, 0.2, 0.5, 0.1, 0.3, 0.1],
        "features": [0, 1, 1, 2, 3, 3],
    },
    "near_degenerate_prior": {
        "weights": [1.0, 1.0e-9, 1.0e-12],
        "features": [0, 1, 2],
    },
    "uniform_eight_labels": {
        "weights": [1.0] * 8,
        "features": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    "single_label": {
        "weights": [0.3, 0.7],
        "features": [4, 4],
    },
    "two_labels_skewed": {
        "weights": [0.95, 0.04, 0.01],
        "features": [0, 1, 1],
    },
}


def reliability(kind, amount):
    a = mp.mpf(amount)
    if kind == "measure":
        return min(mp.mpf("0.99"), 1 - mp.exp(-mp.mpf("0.75") * a))
    if kind == "intervene":
        return min(mp.mpf("0.99"), 1 - mp.exp(-a))
    return min(mp.mpf("0.95"), 1 - 1 / (1 + a / 50))


def mutual_information(weights, features, r):
    labels = sorted(set(features))
    k = len(labels)
    if k < 2:
        return mp.mpf(0)
    tot = sum(mp.mpf(w) for w in weights)
    p = [mp.mpf(w) / tot for w in weights]
    mi = mp.mpf(0)
    for y in labels:
        py = sum(pg * (r * (1 if f == y else 0) + (1 - r) / k) for pg, f in zip(p, features))
        for pg, f in zip(p, features):
            lik = r * (1 if f == y else 0) + (1 - r) / k
            joint = pg * lik
            if joint > 0:
                mi += joint * mp.log(joint / (pg * py))
    return mi


out = []
for name, prior in PRIORS.items():
    cases = []
    for kind, amount in DESIGNS:
        r = reliability(kind, amount)
        cases.append(
            {
                "kind": kind,
                "amount": amount,
                "reliability": float(r),
                "information_gain": float(mutual_information(prior["weights"], prior["features"], r)),
            }
        )
    out.append({"name": name, **prior, "cases": cases})
print(json.dumps(out, indent=1))
