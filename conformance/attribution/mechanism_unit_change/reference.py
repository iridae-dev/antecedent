# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the unit-change fixture: for Y = b X and reference mean xbar the
unit contribution is b (x - xbar); b is read from the declared baseline equation.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/attribution/mechanism_unit_change/reference.py --check

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


import re


def parse(equation):
    match = re.fullmatch(r"Y=(?:(-?[0-9.]+)\+)?(-?[0-9.]+)X", equation)
    return float(match.group(1) or 0.0), float(match.group(2))


def check(document):
    problems = []
    tol = document["tolerance"]["absolute"]
    unit = document["unit_case"]
    _, slope = parse(document["mechanism_detection"]["baseline"].replace(" ", ""))
    derived = [slope * (x - unit["reference_mean"]) for x in unit["x"]]
    for index, (got, want) in enumerate(zip(derived, unit["contributions"])):
        close(got, want, tol, f"contributions[{index}]", problems)
    close(sum(derived) / len(derived), unit["mean_contribution"], tol, "mean_contribution", problems)
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
