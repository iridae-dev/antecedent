# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact scalar Kalman filter reference for the particle-filter fixture.

Linear-Gaussian state space x_t = a x_{t-1} + w_t, y_t = x_t + v_t with w ~ N(0, process_std^2)
and v ~ N(0, obs_std^2), started from x_0 ~ N(0, process_std^2) before the first prediction. Each
step predicts (mean a m, variance a^2 P + q^2) and updates with gain K = P_pred / (P_pred + r^2).
The recursion runs in exact rational arithmetic on the observations exactly as stored (each double
taken as the rational it denotes); only the final conversion to double rounds. No Antecedent code.

    uv run --frozen --script conformance/design_state/particle_filter_kalman/reference.py --check
    uv run --frozen --script conformance/design_state/particle_filter_kalman/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATOL = 1e-12


def kalman(params: dict, observations: list[float]) -> tuple[list[float], list[float]]:
    a = Fraction(params["a"])
    q2 = Fraction(params["process_std"]) ** 2
    r2 = Fraction(params["obs_std"]) ** 2
    mean, var = Fraction(0), q2
    means, variances = [], []
    for y in observations:
        pred_mean, pred_var = a * mean, a * a * var + q2
        gain = pred_var / (pred_var + r2)
        mean = pred_mean + gain * (Fraction(y) - pred_mean)
        var = (1 - gain) * pred_var
        means.append(float(mean))
        variances.append(float(var))
    return means, variances


def check(document: dict) -> list[str]:
    problems = []
    for case in document["cases"]:
        means, variances = kalman(case["params"], case["observations"])
        ref = case["reference"]
        if len(means) != len(ref["filtered_means"]) or len(variances) != len(ref["filtered_variances"]):
            problems.append(f"{case['name']}: length differs")
            continue
        for t, (m, v) in enumerate(zip(means, variances)):
            if abs(m - ref["filtered_means"][t]) > ATOL:
                problems.append(f"{case['name']} t={t}: mean {ref['filtered_means'][t]} vs {m}")
            if abs(v - ref["filtered_variances"][t]) > ATOL:
                problems.append(f"{case['name']} t={t}: variance {ref['filtered_variances'][t]} vs {v}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["cases"][0]["reference"]["filtered_means"][9] += 1e-6
    if not check(wrong):
        failures.append("shifted mean accepted")
    wrong = copy.deepcopy(document)
    wrong["cases"][1]["reference"]["filtered_variances"][24] += 1e-6
    if not check(wrong):
        failures.append("shifted variance accepted")
    wrong = copy.deepcopy(document)
    wrong["cases"][2]["params"]["a"] = 0.97
    if not check(wrong):
        failures.append("changed persistence accepted")
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
