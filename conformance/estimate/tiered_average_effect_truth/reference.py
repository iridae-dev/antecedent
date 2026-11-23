# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the tiered AverageEffect truths by exact path algebra.

CoDetermined: y = 2 t + z - 0.5 u + e adjusted for the closure {z, u} is linear in (t, z, u),
so the closure functional E[E[y | t=1, z, u] - E[y | t=0, z, u]] is the coefficient of t, 2.

Unknown (true order era -> t -> m -> y): y = -t + 2 m + 0.2 era + e and m = t + 0.2 era + e.
Scenario 0 (treatment precedes its peer, adjust {era}): total effect = -1 + 2 * 1 = 1.
Scenario 1 (treatment follows its peer, adjust {era, m}): direct effect = -1.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/tiered_average_effect_truth/reference.py --check
    uv run --frozen --script conformance/estimate/tiered_average_effect_truth/reference.py --self-test

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


def close(got, want, label, problems):
    if want is None or not math.isclose(float(got), want, rel_tol=0.0, abs_tol=1e-12):
        problems.append(f"{label}: derived {got!r} != fixture {want!r}")


def derived():
    beta_t_codetermined = Fraction(2)
    beta_t, beta_m, gamma_mt = Fraction(-1), Fraction(2), Fraction(1)
    return beta_t_codetermined, [beta_t + beta_m * gamma_mt, beta_t]


def check(document):
    problems = []
    closure, scenarios = derived()
    close(closure, document["codetermined"]["closure_effect"], "codetermined.closure_effect", problems)
    got = document["unknown"]["scenario_effects"]
    if len(got) != len(scenarios):
        problems.append("unknown.scenario_effects has the wrong length")
    else:
        for index, (want, value) in enumerate(zip(scenarios, got, strict=True)):
            close(want, value, f"unknown.scenario_effects[{index}]", problems)
    return problems


MUTATIONS = [
    ("closure effect shifted", lambda d: d["codetermined"].__setitem__("closure_effect", 2.5)),
    ("scenario order swapped", lambda d: d["unknown"].__setitem__("scenario_effects", [-1.0, 1.0])),
    ("total effect replaced by direct", lambda d: d["unknown"].__setitem__("scenario_effects", [-1.0, -1.0])),
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
