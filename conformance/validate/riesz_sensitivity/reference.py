# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the balanced binary-ATE Riesz representer: alpha = T/pi - (1-T)/(1-pi)
so alpha is 1/pi for the treated and -1/(1-pi) for the control, and its L2 norm is
sqrt(pi/pi^2 + (1-pi)/(1-pi)^2) = sqrt(1/(pi(1-pi))).

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/validate/riesz_sensitivity/reference.py --check

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
    case = document["balanced_case"]
    pi = case["propensity"]
    treated, control = 1.0 / pi, -1.0 / (1.0 - pi)
    l2 = math.sqrt(pi * treated**2 + (1.0 - pi) * control**2)
    close(treated, case["alpha_treated"], 1e-12, "alpha_treated", problems)
    close(control, case["alpha_control"], 1e-12, "alpha_control", problems)
    close(l2, case["alpha_l2"], 1e-12, "alpha_l2", problems)
    close(math.sqrt(1.0 / (pi * (1.0 - pi))), case["alpha_l2"], 1e-12, "alpha_l2 closed form", problems)
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
