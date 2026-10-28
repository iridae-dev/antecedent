# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the 1-D Gaussian power-prior composition: precisions add,
Lambda = Lambda0 + alpha * Lambda_source, and the mean is the precision-weighted average.
Exact rational arithmetic.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/prior_bank_power_mixture/reference.py --check

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


from fractions import Fraction as F


def check(document):
    problems = []
    tol = document["tol"]
    p0 = 1 / F(str(document["baseline_variance"]))
    ps = F(str(document["alpha"])) / F(str(document["source_variance"]))
    precision = p0 + ps
    mean = (p0 * F(str(document["baseline_mean"])) + ps * F(str(document["source_mean"]))) / precision
    close(float(precision), document["expected_precision"], tol, "precision", problems)
    close(float(mean), document["expected_mean"], tol, "mean", problems)
    close(float(1 / precision), document["expected_variance"], tol, "variance", problems)
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
