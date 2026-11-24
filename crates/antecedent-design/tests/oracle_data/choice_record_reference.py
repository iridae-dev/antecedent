"""Exact expected linear-form terms of recorded choice options.

Standard library only. Every belief value and weight is a dyadic rational, so the exact
`fractions.Fraction` results are the values a correct floating-point evaluation reproduces to
rounding. Each model's parts are written out directly as functions of one attribute row (no
expression trees), and an expectation is the weighted mean of the part over the draw rows, or
the part at the means for a means belief (used only where every part is affine).

Models (attributes x0, x1; parameters theta):
  two_rates      value = t0*x0 + t1*x1
                 fixed 0; terms [x0, x1]
  rate_and_kink  value = t0*x0 + t1*max(x1 - t2, 0) + x0/2, with t2 fixed at 3/2
                 fixed x0/2; terms [x0, max(x1 - 3/2, 0), 0]
  three_segment  value = sum_i s_i * (clamp(x0, b_i, b_{i+1}) - b_i) + t3*x1,
                 breakpoints 0, 1, 2, 4, with t3 fixed at -2
                 fixed -2*x1; terms [seg0, seg1, seg2, 0]

A fixed parameter's column is zero and its contribution is part of the fixed value.

Writes choice_record_reference.json next to this script.
"""

import json
import random
from fractions import Fraction as F
from pathlib import Path

BREAKPOINTS = [F(0), F(1), F(2), F(4)]


def segment_width(x, i):
    lo, hi = BREAKPOINTS[i], BREAKPOINTS[i + 1]
    if x <= lo:
        return F(0)
    if x >= hi:
        return hi - lo
    return x - lo


MODELS = {
    "two_rates": {
        "n_parameters": 2,
        "fixed": lambda r: F(0),
        "terms": [lambda r: r[0], lambda r: r[1]],
        "means_allowed": True,
    },
    "rate_and_kink": {
        "n_parameters": 3,
        "fixed": lambda r: r[0] / 2,
        "terms": [
            lambda r: r[0],
            lambda r: max(r[1] - F(3, 2), F(0)),
            lambda r: F(0),
        ],
        "means_allowed": False,
    },
    "three_segment": {
        "n_parameters": 4,
        "fixed": lambda r: -2 * r[1],
        "terms": [
            lambda r: segment_width(r[0], 0),
            lambda r: segment_width(r[0], 1),
            lambda r: segment_width(r[0], 2),
            lambda r: F(0),
        ],
        "means_allowed": False,
    },
}

WIDTH = 2


def dyadic(rng, low, high, denominator):
    return F(rng.randint(low * denominator, high * denominator), denominator)


def expectation(part, belief):
    if belief["kind"] == "means":
        return part(belief["means"])
    rows = belief["rows"]
    weights = belief["weights"] or [F(1)] * len(rows)
    total = sum(weights)
    return sum(w * part(row) for w, row in zip(weights, rows)) / total


def make_belief(rng, means_allowed):
    if means_allowed and rng.random() < 0.5:
        return {"kind": "means", "means": [dyadic(rng, -4, 6, 8) for _ in range(WIDTH)]}
    n_rows = rng.randint(1, 4)
    rows = [[dyadic(rng, -1, 5, 8) for _ in range(WIDTH)] for _ in range(n_rows)]
    weights = None
    if rng.random() < 0.6:
        weights = [F(rng.randint(0, 12), 4) for _ in range(n_rows)]
        if sum(weights) == 0:
            weights[0] = F(1, 4)
    return {"kind": "draws", "rows": rows, "weights": weights}


def to_json_belief(belief):
    if belief["kind"] == "means":
        return {"means": [float(v) for v in belief["means"]]}
    return {
        "rows": [float(v) for row in belief["rows"] for v in row],
        "weights": None if belief["weights"] is None else [float(w) for w in belief["weights"]],
    }


def main():
    rng = random.Random(20261123)
    cases = []
    for name, model in MODELS.items():
        for case_index in range(3):
            records = []
            for record_index in range(rng.randint(2, 4)):
                options = []
                fixed = []
                terms = []
                for option_index in range(rng.randint(2, 3)):
                    belief = make_belief(rng, model["means_allowed"])
                    options.append({"id": f"option-{option_index}", "belief": to_json_belief(belief)})
                    fixed.append(float(expectation(model["fixed"], belief)))
                    terms.extend(float(expectation(t, belief)) for t in model["terms"])
                records.append(
                    {
                        "id": f"{name}-{case_index}-record-{record_index}",
                        "options": options,
                        "fixed": fixed,
                        "terms": terms,
                    }
                )
            cases.append({"model": name, "n_parameters": model["n_parameters"], "records": records})
    out = Path(__file__).with_name("choice_record_reference.json")
    out.write_text(json.dumps({"cases": cases}, indent=1) + "\n")


if __name__ == "__main__":
    main()
