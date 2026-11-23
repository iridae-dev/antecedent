# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["tigramite==5.2.1.30", "numpy==1.23.5", "scipy==1.10.1", "scikit-learn==1.3.2", "dcor==0.6"]
# ///
"""Retained black-box regeneration of the advanced conditional-independence references.

Reads only the fixture's declared columns and conditioning indices and runs the public
``tigramite`` tests CMIsymb (discrete conditional mutual information, natural log) and GPDC
(Gaussian-process distance correlation) with the shuffle-test p-value.  Statistics are
compared at the tight fixture tolerance; shuffle p-values are Monte Carlo and compared at the
fixture's absolute bands.  ``null_q95`` is regenerated only as context.  No Antecedent code is
imported and no tigramite source is stored.

    uv run --frozen --script conformance/discovery/advanced_ci/reference.py --check
    uv run --frozen --script conformance/discovery/advanced_ci/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.gpdc import GPDC

HERE = Path(__file__).resolve().parent
EXPECTED_TOLERANCES = {
    "symbolic_stat_atol": 1e-12,
    "symbolic_p_atol": 0.08,
    "gpdc_stat_atol": 0.35,
    "gpdc_stat_rtol": 1.0,
    "gpdc_p_atol": 0.2,
}
REGEN_STAT_ATOL = 1e-9


def run(kind: str, case: dict) -> tuple[float, float]:
    cols = [np.asarray(c, dtype=float).reshape(-1, 1) for c in case["columns"]]
    x, y = cols[0], cols[1]
    z = (
        np.hstack([cols[i] for i in case["z_indices"]])
        if case["z_indices"]
        else None
    )
    if kind == "symbolic":
        test = CMIsymb(significance="shuffle_test", sig_samples=1000, seed=1)
    else:
        test = GPDC(significance="shuffle_test", sig_samples=200, seed=1)
    stat, p = test.run_test_raw(x, y, z)
    return float(stat), float(p)


def check(document: dict) -> list[str]:
    problems = []
    if document["tolerances"] != EXPECTED_TOLERANCES:
        problems.append("product comparison tolerances changed from the reviewed values")
    for kind, key in (("symbolic", "symbolic_cases"), ("gpdc", "gpdc_cases")):
        if not document[key]:
            problems.append(f"empty {key}")
        for case in document[key]:
            stat, p = run(kind, case)
            ref = case["reference"]
            if abs(ref["statistic"] - stat) > REGEN_STAT_ATOL:
                problems.append(f"{case['name']} statistic differs: {ref['statistic']} vs {stat}")
            band = EXPECTED_TOLERANCES["symbolic_p_atol" if kind == "symbolic" else "gpdc_p_atol"]
            if abs(ref["p_value"] - p) > band:
                problems.append(f"{case['name']} p-value differs beyond the Monte Carlo band")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["symbolic_cases"][0]["reference"]["statistic"] += 1e-4
    if not check(wrong):
        failures.append("shifted symbolic statistic accepted")
    wrong = copy.deepcopy(document)
    wrong["gpdc_cases"][0]["reference"]["statistic"] += 0.01
    if not check(wrong):
        failures.append("shifted gpdc statistic accepted")
    wrong = copy.deepcopy(document)
    wrong["symbolic_cases"][1]["reference"]["p_value"] = 0.9
    if not check(wrong):
        failures.append("wrong null p-value accepted")
    wrong = copy.deepcopy(document)
    wrong["tolerances"]["gpdc_stat_atol"] = 5.0
    if not check(wrong):
        failures.append("widened tolerance accepted")
    wrong = copy.deepcopy(document)
    wrong["symbolic_cases"][0]["columns"][0][0] += 1.0
    if not check(wrong):
        failures.append("corrupted input accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
