# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the known-truth mixture fixture: the identified-mass-weighted mixture of per-atom effects, the per-atom response curves of the stated linear laws, the temporal and mediation atom effects, and the lagged-edge marginals implied by the atom masks and posterior weights.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/known_truth_mixtures/reference.py --check
    uv run --frozen --script conformance/bayesian/known_truth_mixtures/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = Fraction
FIXTURE = HERE / "expected.json"


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got) if got is not None else got!r} != fixture {want!r} (tol {tol})")


def same(got, want, label, problems):
    if got != want:
        problems.append(f"{label}: derived {got!r} != fixture {want!r}")

def mix(weights, effects):
    mass = sum(F(str(w)) for w in weights[: len(effects)])
    total = sum(F(str(w)) * F(str(e)) for w, e in zip(weights, effects))
    return total, mass, total / mass


def marginals(atoms, width):
    out = []
    for bit in range(width):
        out.append(sum(F(str(w)) for w, mask in atoms if (mask >> bit) & 1))
    return out


def check(document):
    problems = []
    s = document["static_average_effect"]
    # adjusted atom: do(T) cuts Z -> T, so the effect is the structural coefficient 2; the unadjusted atom is 2 + 2 * gap with
    # gap = E[Z|T=1] - E[Z|T=0] = 1/2 in the test design, which is an input here.
    adjusted, unadjusted = F(2), F(2) + 2 * F(1, 2)
    same([float(unadjusted), float(adjusted)], s["identified_atom_effects"], "identified_atom_effects", problems)
    total, mass, conditional = mix(s["posterior_weights"], [unadjusted, adjusted])
    close(total, s["identified_weighted_sum"], 1e-12, "identified_weighted_sum", problems)
    close(mass, s["identified_mass"], 1e-12, "identified_mass", problems)
    close(conditional, s["expected_effect_given_identified"], 1e-12, "expected_effect_given_identified", problems)
    close(1 - mass, s["expected_unidentified_mass"], 1e-12, "static unidentified mass", problems)

    r = document["static_response"]
    w = [F(str(x)) for x in r["posterior_weights"]]
    grid = [F(str(g)) for g in r["treatment_grid"]]
    # Z = 0.5 T +- 0.5 over four sign pairs per level: E[Z | T=t] = 0.5 t, and Z averages to zero over the symmetric grid
    direct = [1 + 2 * t + F(3, 2) * (F(1, 2) * t) for t in grid]
    adjusted_curve = [1 + 2 * t + F(3, 2) * 0 for t in grid]
    for got, want, label in ((direct, r["identified_atom_curve_direct"], "direct curve"), (adjusted_curve, r["identified_atom_curve_adjusted"], "adjusted curve")):
        for g, v in zip(got, want):
            close(g, v, 1e-12, label, problems)
    for d, a, want in zip(direct, adjusted_curve, r["expected_conditional_on_identified_curve"]):
        close((w[0] * d + w[1] * a) / (w[0] + w[1]), want, 1e-12, "conditional curve", problems)
    close(1 - w[0] - w[1], r["expected_unidentified_mass"], 1e-12, "static response unidentified mass", problems)

    t = document["temporal_effect"]
    close(F(9, 10), t["identified_atom"]["effect"], 1e-12, "temporal effect (coefficient)", problems)
    atoms = [(t["identified_atom"]["weight"], t["identified_atom"]["lag_mask"]), (t["unidentified_atom"]["weight"], t["unidentified_atom"]["lag_mask"])]
    close(sum(F(str(a[0])) for a in atoms[:1]), t["identified_mass"], 1e-12, "temporal identified mass", problems)
    close(1 - F(str(atoms[0][0])), t["expected_unidentified_mass"], 1e-12, "temporal unidentified mass", problems)
    close(F(9, 10), t["expected_effect_given_identified"], 1e-12, "temporal conditional effect", problems)
    for got, want in zip(marginals(atoms, 4), t["lagged_edge_marginals"]):
        close(got, want, 1e-12, "temporal lagged edge marginal", problems)
    sm = document["temporal_sustained_multistep"]
    close(F(9, 10), sm["expected_effect_given_identified"], 1e-12, "sustained conditional effect", problems)
    close(1 - F(str(sm["posterior_weights"][0])), sm["expected_unidentified_mass"], 1e-12, "sustained unidentified mass", problems)

    m = document["temporal_mediation"]
    close(F("0.8") * F("0.55"), m["identified_atom"]["effect"], 1e-12, "mediation atom effect (a*b)", problems)
    close(F("0.8") * F("0.55"), m["expected_effect_given_identified"], 1e-12, "mediation conditional effect", problems)
    close(1 - F(str(m["identified_atom"]["weight"])), m["expected_unidentified_mass"], 1e-12, "mediation unidentified mass", problems)
    atoms = [(m["identified_atom"]["weight"], m["identified_atom"]["lag_mask"]), (m["unidentified_atom"]["weight"], m["unidentified_atom"]["lag_mask"])]
    for got, want in zip(marginals(atoms, 9), m["lagged_edge_marginals"]):
        close(got, want, 1e-12, "mediation lagged edge marginal", problems)
    return problems


MUTATIONS = [
    ("mixture mean changed", lambda d: d["static_average_effect"].__setitem__("expected_effect_given_identified", 2.5)),
    ("curve mixture changed", lambda d: d["static_response"]["expected_conditional_on_identified_curve"].__setitem__(2, 2.2)),
    ("temporal effect changed", lambda d: d["temporal_effect"].__setitem__("expected_effect_given_identified", 1.0)),
    ("mediation effect changed", lambda d: d["temporal_mediation"].__setitem__("expected_effect_given_identified", 0.5)),
    ("edge marginal changed", lambda d: d["temporal_effect"]["lagged_edge_marginals"].__setitem__(0, 0.7)),
]


def self_test(document):
    failures = []
    if check(document):
        failures.append("the unmodified fixture is rejected")
    for label, mutate in MUTATIONS:
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"accepted: {label}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
