# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the response-surface truths: the static two-point mean curve of E[Y|T=t,Z=z] = 1 + 2t + 0.8z (Z averaging to a small finite-sample mean), and the temporal dose-by-horizon means and sustained-window effect of y_i = 1 + 2 t_{i-1} + 3 t_{i-2} with treatment pattern 0, 1, 0, -1.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/response_surfaces/reference.py --check
    uv run --frozen --script conformance/bayesian/response_surfaces/reference.py --self-test

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

def check(document):
    problems = []
    two_point = json.loads((HERE.parent.parent / "response/two_point_curve_average_effect/expected.json").read_text(encoding="utf-8"))
    n = two_point["generation"]["n"]
    levels = [two_point["contract"]["control_level"], two_point["contract"]["active_level"]]
    mean_z = sum(F(math.sin(i / 17.0)) for i in range(n)) / n
    structural = [1 + 2 * F(str(t)) for t in levels]          # E[Y | do(t)] with E[Z] = 0
    for got, want in zip(structural, document["static_mean"]):
        close(got, want, 1e-12, "static_mean (structural, E[Z] = 0)", problems)
    deviation = F(8, 10) * abs(mean_z)                           # what the sample mean of Z adds
    if not deviation < F(str(document["static_tolerance"])):
        problems.append(f"finite-sample confounder mean shifts the level by {float(deviation)}, beyond the tolerance")

    dose = json.loads((HERE.parent.parent / "response/temporal_dose_horizon/expected.json").read_text(encoding="utf-8"))
    rows = dose["generation"]["n"]
    pattern = [0, 1, 0, -1]
    t = [F(pattern[i % 4]) for i in range(rows)]
    mean_t = sum(t) / rows
    expected = []
    for d in (0, 1):                                              # dose-major, horizons 1 then 2
        for horizon in (1, 2):
            coefficient = {1: 2, 2: 3}[horizon]
            other = {1: 3, 2: 2}[horizon]
            expected.append(1 + coefficient * d + other * mean_t)
    structural = []
    for d in (0, 1):
        for horizon in (1, 2):
            structural.append(1 + {1: 2, 2: 3}[horizon] * d)          # mean(t) = 0 in the period-4 population
    for got, exact, want in zip(expected, structural, document["temporal_mean"]):
        close(exact, want, 1e-12, "temporal_mean (period-4 population)", problems)
        close(got, want, document["temporal_tolerance"], "temporal_mean with the finite-sample mean of t", problems)
    close(2 + 3, document["window_effect"], 1e-12, "window_effect = sum of lag coefficients", problems)
    return problems


MUTATIONS = [
    ("static truth changed", lambda d: d["static_mean"].__setitem__(1, 2.5)),
    ("temporal mean changed", lambda d: d["temporal_mean"].__setitem__(3, 5.0)),
    ("window effect changed", lambda d: d.__setitem__("window_effect", 4.0)),
    ("static tolerance tightened below the sample shift", lambda d: d.__setitem__("static_tolerance", 0.01)),
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
