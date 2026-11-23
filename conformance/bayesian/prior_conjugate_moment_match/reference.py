# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the Beta and Gamma moment-matching equations. Beta(a, b) with
mean m and variance v: a + b = m(1 - m)/v - 1, a = m (a + b). Gamma(shape k, rate r):
k = m^2/v, r = m/v. The fixture states its ESS convention (Beta: a + b - 2, Gamma: k - 1),
which this script applies as stated; the validity conditions are the support bounds.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/prior_conjugate_moment_match/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def close(got, want, tol, label, problems):
    if not math.isclose(got, want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


def beta_valid(mean, variance):
    return 0.0 < mean < 1.0 and 0.0 < variance < mean * (1.0 - mean)


def check(document):
    problems = []
    tol = document["tol"]
    b = document["beta_moment_match"]
    total = b["mean"] * (1 - b["mean"]) / b["variance"] - 1
    close(b["mean"] * total, b["expected_alpha"], tol, "beta alpha", problems)
    close((1 - b["mean"]) * total, b["expected_beta"], tol, "beta beta", problems)
    close(total - 2.0, b["expected_ess"], tol, "beta ess", problems)
    if not beta_valid(document["beta_moment_match_negative_ess"]["mean"], document["beta_moment_match_negative_ess"]["variance"]):
        problems.append("beta_moment_match_negative_ess should be a valid Beta with total below 2")
    for ess_key in ("beta_mean_and_ess_zero", "beta_mean_and_ess_matches_any_request"):
        c = document[ess_key]
        total = c["ess"] + 2.0
        close(c["mean"] * total, c["expected_alpha"], tol, f"{ess_key} alpha", problems)
        close((1 - c["mean"]) * total, c["expected_beta"], tol, f"{ess_key} beta", problems)
    for case in document["beta_moment_match_rejected_inputs"]:
        if beta_valid(case["mean"], case["variance"]):
            problems.append(f"beta ({case['mean']}, {case['variance']}) is valid but the fixture rejects it")
    for case in document["beta_mean_and_ess_rejected_inputs"]:
        if 0.0 < case["mean"] < 1.0 and case["ess"] >= 0.0:
            problems.append(f"beta mean/ess ({case['mean']}, {case['ess']}) is valid but the fixture rejects it")
    g = document["gamma_moment_match"]
    close(g["mean"] ** 2 / g["variance"], g["expected_shape"], tol, "gamma shape", problems)
    close(g["mean"] / g["variance"], g["expected_rate"], tol, "gamma rate", problems)
    close(g["expected_shape"] - 1.0, g["expected_ess"], tol, "gamma ess", problems)
    for key in ("gamma_mean_and_ess_zero", "gamma_mean_and_ess_matches_any_request"):
        c = document[key]
        shape = c["ess"] + 1.0
        close(shape, c["expected_shape"], tol, f"{key} shape", problems)
        close(shape / c["mean"], c["expected_rate"], tol, f"{key} rate", problems)
    for case in document["gamma_moment_match_rejected_inputs"]:
        if case["mean"] > 0 and case["variance"] > 0:
            problems.append(f"gamma ({case['mean']}, {case['variance']}) is valid but the fixture rejects it")
    for case in document["gamma_mean_and_ess_rejected_inputs"]:
        if case["mean"] > 0 and case["ess"] >= 0:
            problems.append(f"gamma mean/ess ({case['mean']}, {case['ess']}) is valid but the fixture rejects it")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
