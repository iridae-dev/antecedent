# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the identified curves and the placebo level behind the
response_curve_posterior_refuter_states fixture.

* Front-door ADMG table (conformance/estimate/admg_frontdoor_functional): P(T=1)=0.3,
  P(M=1|T)=0.2, 0.8 and P(Y=1|M,T)=0.1, 0.5 (M=0) and 0.6, 0.9 (M=1). The front-door curve is
  E[Y | do(t)] = sum_m P(m | t) g(m) with g(m) = sum_t' P(t') P(Y=1 | m, t'), giving
  g(0) = 0.22, g(1) = 0.69, curve(0) = 0.8 g(0) + 0.2 g(1) = 0.314 and curve(1) = 0.2 g(0) +
  0.8 g(1) = 0.596; the placebo (permuted-treatment) curve is flat at E[Y] = 0.386, the mean outcome of the table.
* Cpdag law y = t + z + e: the curve is a (population, E z = 0). PAG law y = 1 + 2 t + 0.8 z + e:
  the curve is 1 + 2 a.

Exact rationals; standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/response_curve_posterior_refuter_states/reference.py --check
    uv run --frozen --script conformance/estimate/response_curve_posterior_refuter_states/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent


def frontdoor():
    p_t = {0: F(7, 10), 1: F(3, 10)}
    p_m = {0: F(2, 10), 1: F(8, 10)}  # P(M=1 | T)
    p_y = {(0, 0): F(1, 10), (0, 1): F(5, 10), (1, 0): F(6, 10), (1, 1): F(9, 10)}  # (m, t)
    g = {m: sum(p_t[t] * p_y[(m, t)] for t in (0, 1)) for m in (0, 1)}

    def curve(t):
        return (1 - p_m[t]) * g[0] + p_m[t] * g[1]

    e_y = sum(
        p_t[t] * (((1 - p_m[t]) * p_y[(0, t)]) + p_m[t] * p_y[(1, t)]) for t in (0, 1)
    )
    return curve(0), curve(1), e_y


def check(document):
    problems = []
    c0, c1, e_y = frontdoor()
    for label, got, want in (
        ("frontdoor_curve_0", c0, document["closed_form"]["frontdoor_curve"][0]),
        ("frontdoor_curve_1", c1, document["closed_form"]["frontdoor_curve"][1]),
        ("frontdoor_mean_outcome", e_y, document["closed_form"]["frontdoor_mean_outcome"]),
    ):
        if not math.isclose(float(got), want, rel_tol=0.0, abs_tol=1e-12):
            problems.append(f"{label}: derived {float(got)!r} != fixture {want!r}")
    if document["closed_form"]["cpdag_curve"] != [0.0, 1.0] or document["closed_form"]["pag_curve"] != [1.0, 3.0]:
        problems.append("population curves: cpdag a, pag 1 + 2 a on the grid 0, 1")
    contract = document["refuter_contract"]
    for key in ("Frequentist", "Bayesian_admg", "Bayesian_class"):
        if set(contract[key]["cheap"]) - set(contract[key]["full"]):
            problems.append(f"{key}: full must run every refuter cheap runs")
    return problems


MUTATIONS = [
    ("front-door curve shifted", lambda d: d["closed_form"]["frontdoor_curve"].__setitem__(0, 0.32)),
    ("mean outcome shifted", lambda d: d["closed_form"].__setitem__("frontdoor_mean_outcome", 0.4)),
    ("pag curve without the intercept", lambda d: d["closed_form"].__setitem__("pag_curve", [0.0, 2.0])),
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
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
