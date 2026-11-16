# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the kernel weights and the weighted-target transport moments.

With u = distance / bandwidth the kernels are Gaussian exp(-u^2/2), Epanechnikov 1 - u^2 and
tricube (1 - u^3)^3 (the last two zero for u >= 1). For rows with effects e_i and weights w_i
the transported mean is sum(w e) / sum(w), its sampling variance is
sum(w^2 (e - mean)^2) / sum(w)^2, and the Kish effective sample size is sum(w)^2 / sum(w^2).
Rows 0..n-1 with effect i and weight i + 1 are summed term by term below.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/prior_bank_weighted_target/reference.py --check

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


def check(document):
    problems = []
    tol = document["tol"]
    k = document["kernel_weights"]
    h = k["bandwidth"]
    for x, g, e, t in zip(k["distances"], k["gaussian"], k["epanechnikov"], k["tricube"], strict=True):
        u = x / h
        close(math.exp(-0.5 * u * u), g, tol, f"gaussian at {x}", problems)
        close(1 - u * u if u < 1 else 0.0, e, tol, f"epanechnikov at {x}", problems)
        close((1 - u**3) ** 3 if u < 1 else 0.0, t, tol, f"tricube at {x}", problems)
    r = document["graded_rows"]
    rows = [(float(i), float(i + 1)) for i in range(r["n_rows"])]
    mass = sum(w for _, w in rows)
    mean = sum(w * e for e, w in rows) / mass
    close(mean, r["expected_mean"], tol, "graded mean", problems)
    close(mass * mass / sum(w * w for _, w in rows), r["expected_kish_ess"], tol, "graded ess", problems)
    variance = sum(w * w * (e - mean) ** 2 for e, w in rows) / (mass * mass)
    close(variance, r["expected_sampling_variance"], tol, "graded sampling variance", problems)
    n = document["neighbourhood_rows"]
    inside = [n["inside_effect"]] * n["neighbourhood_rows"]
    close(sum(inside) / len(inside), n["expected_mean"], tol, "neighbourhood mean", problems)
    close(float(len(inside)), n["expected_kish_ess"], tol, "neighbourhood ess", problems)
    f = document["refusals"]
    if not f["equal_weight_rows_below_floor"] < f["absolute_ess_floor"]:
        problems.append("the equal-weight refusal case must sit below the floor")
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
