# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact enumeration of the two-unit Bernoulli design behind the randomized-interference fixture.

Every assignment of the declared Bernoulli design is enumerated with its exact probability. From
that enumeration the reference computes, in rational arithmetic and only from the declared
design, edges, mapping and exposure levels:

* each unit's exposure probability for the ``from`` and ``to`` exposure levels,
* the minimum of those probabilities,
* the Horvitz-Thompson contrast: the mean over units of 1(e_i = level) y_i / pi_i(level) at
  the `to` level minus the same at the `from` level, pi_i the enumerated exposure probability, and
* the Hajek (self-normalised) contrast.

The exposure of a unit is (own treatment, sum of in-neighbour treatments weighted by edge weight)
(the ``neighbor_count`` mapping). The conservative variance of the fixture is Antecedent's
documented repository variance bound, not a quantity derived here: this reference does not check it.

Standard library only; imports no Antecedent code.

    uv run --frozen --script conformance/response/randomized_interference/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent


def exposure(assignment, edges, unit):
    own = F(int(assignment[unit]))
    neighbors = sum(F(str(w)) * int(assignment[src]) for src, dst, w in edges if dst == unit)
    return (own, neighbors)


def derive(document):
    if document["design"]["kind"] != "bernoulli" or document["mapping"] != "neighbor_count":
        raise ValueError("reference covers the Bernoulli design with the neighbour-count mapping")
    p = F(str(document["design"]["probability"]))
    n = len(document["assignment"])
    edges = [(int(a), int(b), w) for a, b, w in document["directed_edges"]]
    levels = {
        name: (F(str(document[name]["own"])), F(str(document[name]["neighbors"])))
        for name in ("from", "to")
    }
    prob = {name: [F(0)] * n for name in levels}
    for assignment in itertools.product((False, True), repeat=n):
        weight = F(1)
        for z in assignment:
            weight *= p if z else 1 - p
        for unit in range(n):
            for name, level in levels.items():
                if exposure(assignment, edges, unit) == level:
                    prob[name][unit] += weight
    observed = document["assignment"]
    y = [F(str(v)) for v in document["outcomes"]]
    out = {}
    for name, level in levels.items():
        hit = [exposure(observed, edges, u) == level for u in range(n)]
        terms = [y[u] / prob[name][u] if hit[u] else F(0) for u in range(n)]
        out[f"ht_{name}"] = sum(terms) / n
        mass = sum(F(1) / prob[name][u] for u in range(n) if hit[u])
        out[f"hajek_{name}"] = sum(terms) / mass
    return {
        "exposure_probabilities_from": [float(v) for v in prob["from"]],
        "exposure_probabilities_to": [float(v) for v in prob["to"]],
        "minimum_exposure_probability": float(min(prob["from"] + prob["to"])),
        "horvitz_thompson_contrast": float(out["ht_to"] - out["ht_from"]),
        "hajek_contrast": float(out["hajek_to"] - out["hajek_from"]),
    }


def check(document):
    problems = []
    atol = document["tolerance"]["atol"]
    for key, value in derive(document).items():
        frozen = document["expected"][key]
        pairs = zip(value, frozen) if isinstance(value, list) else [(value, frozen)]
        for got, want in pairs:
            if abs(got - want) > atol:
                problems.append(f"{key}: fixture {want!r} vs derived {got!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = check(document)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    print("randomized_interference reference matches expected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
