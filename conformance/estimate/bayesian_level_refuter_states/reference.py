# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the intervention level and the refuter contract behind the
bayesian_level_refuter_states fixture.

The covariate is the wave z_i = sin(0.37 i), i = 0..n-1. The outcome law is
y = 1 + 2 t + 1.5 z + noise with t confounded by z, so the adjusted level of the intervention
do(t = 1) is E[y | do(t = 1)] = 1 + 2 + 1.5 mean(z), with mean(z) the exact sample average of
the wave. The contract lists the refuters a Bayesian level study runs at each validation depth.
Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/bayesian_level_refuter_states/reference.py --check
    uv run --frozen --script conformance/estimate/bayesian_level_refuter_states/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def derived(document):
    law = document["law"]
    n = document["design"]["rows"]
    mean_z = math.fsum(math.sin(law["wave_frequency"] * i) for i in range(n)) / n
    level = law["intercept"] + law["effect"] * law["level_at"] + law["covariate_coefficient"] * mean_z
    return mean_z, level


def check(document):
    problems = []
    mean_z, level = derived(document)
    for label, got, want in (
        ("mean_z", mean_z, document["design"]["mean_z"]),
        ("level", level, document["level"]["value"]),
    ):
        if not math.isclose(got, want, rel_tol=0.0, abs_tol=1e-12):
            problems.append(f"{label}: derived {got!r} != fixture {want!r}")
    contract = document["refuter_contract"]
    if set(contract["cheap"]) - set(contract["full"]):
        problems.append("full must run every refuter cheap runs")
    if contract["predictive_checks"] != 2:
        problems.append("a Bayesian level study reports the prior and the posterior check")
    return problems


MUTATIONS = [
    ("level shifted", lambda d: d["level"].__setitem__("value", d["level"]["value"] + 0.01)),
    ("level without the covariate term", lambda d: d["level"].__setitem__("value", 3.0)),
    ("full drops a cheap refuter", lambda d: d["refuter_contract"]["full"].pop(0)),
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
