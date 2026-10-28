# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact reference for the completion-weighted mean and identified-set bounds of the
identified multi-completion temporal PAG (conformance/estimate/temporal_class_envelope/identified_pag.json).

Regenerate with:  uv run --offline --script temporal_class_envelope_weighting_reference.py > temporal_class_envelope_weighting_reference.json

Definition-level derivation, standard library only, no Antecedent code. The series is the closed-form
law of identified_pag.json (sin/cos of the row index; each float is taken as an exact rational). Each
identified completion's effect is the coefficient of t_{i-1} in the least-squares fit of y_i on
[1, t_{i-1}] (adjust nothing) or [1, t_{i-1}, z_{i-1}] (adjust z), rows i = 1..n-1, solved from the
normal equations with Fractions. The class point is the weight-averaged completion effect over the
identified completions (equal enumeration weight 1 each, unidentified mass reported and never mixed):
    point = sum_g w_g e_g / sum_g w_g.
The identified set of the effect is [min_g e_g, max_g e_g]. Also written are the values a wrong
weighting would produce, so a test can assert the product matches none of them:
  - distinct_effect_mean: each distinct effect counted once (uniform over models, not over completions);
  - unidentified_as_zero: the unidentified completion mixed in at effect 0 (dropped mass);
  - adjust_z_only / adjust_none_only: a single model's effect.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import math
from fractions import Fraction as F
from statistics import NormalDist


def series(n):
    z, t, m, y = [], [], [], [0.0] * n
    for i in range(n):
        x = float(i)
        zi = math.sin(0.37 * x) + 0.5 * math.cos(1.3 * x)
        z.append(zi)
        t.append(0.6 * zi + 0.8 * math.sin(0.23 * x + 0.4))
        m.append(0.7 * zi + 0.6 * math.cos(0.29 * x + 0.2))
    for i in range(1, n):
        y[i] = 1.0 + 2.0 * t[i - 1] + 1.5 * m[i - 1] + 0.3 * math.sin(0.53 * float(i))
    return {"t": t, "y": y, "z": z}


def solve(a, b):
    k = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(k):
        p = next(r for r in range(c, k) if aug[r][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        for r in range(k):
            if r != c and aug[r][c] != 0:
                f = aug[r][c] / aug[c][c]
                aug[r] = [u - f * v for u, v in zip(aug[r], aug[c])]
    return [aug[i][k] / aug[i][i] for i in range(k)]


def effect(d, adjust):
    n = len(d["y"])
    rows = [[F(1), F(d["t"][i - 1])] + [F(d[a][i - 1]) for a in adjust] for i in range(1, n)]
    y = [F(d["y"][i]) for i in range(1, n)]
    k = len(rows[0])
    xtx = [[sum(r[a] * r[b] for r in rows) for b in range(k)] for a in range(k)]
    xty = [sum(r[a] * yy for r, yy in zip(rows, y)) for a in range(k)]
    return solve(xtx, xty)[1]


def main():
    n = 600
    d = series(n)
    adjustments = [["z"], ["z"], [], [], ["z"], ["z"]]  # identified_pag.json identified_completions
    cache = {}
    effects = []
    for adj in adjustments:
        key = tuple(adj)
        if key not in cache:
            cache[key] = effect(d, adj)
        effects.append(cache[key])
    weights = [F(1)] * len(effects)
    point = sum(w * e for w, e in zip(weights, effects)) / sum(weights)
    e_z, e_none = cache[("z",)], cache[()]
    nd = NormalDist()
    out = {
        "n": n,
        "adjustments": adjustments,
        "completion_effects": [float(e) for e in effects],
        "weights": [float(w) for w in weights],
        "identified_mass": 6.0 / 7.0,
        "unidentified_mass": 1.0 / 7.0,
        "point": float(point),
        "bound_lower": float(min(effects)),
        "bound_upper": float(max(effects)),
        "wrong_weightings": {
            "distinct_effect_mean": float((e_z + e_none) / 2),
            "unidentified_as_zero": float(point * 6 / 7),
            "adjust_z_only": float(e_z),
            "adjust_none_only": float(e_none),
        },
        "interval_level": 0.9,
        "critical_value_one_sided": nd.inv_cdf(0.9),
        "critical_value_two_sided": nd.inv_cdf(0.95),
        "absolute_tolerance": 1e-9,
    }
    print(json.dumps(out, indent=2))


main()
