# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the path-allocation fixture: every directed path contributes the
product of its edge coefficients, and the total change is the sum over paths.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/attribution/path_allocation/reference.py --check

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
    tol = document["tolerance"]["absolute"]
    for case in document["cases"]:
        paths = case.get("edge_coefficients_by_path") or [case["edge_coefficients"]]
        contributions = [math.prod(path) for path in paths]
        label = case["id"]
        if len(contributions) != len(case["path_contributions"]):
            problems.append(f"{label}: path count {len(contributions)} != {len(case['path_contributions'])}")
            continue
        for index, (got, want) in enumerate(zip(contributions, case["path_contributions"])):
            close(got, want, tol, f"{label}.path_contributions[{index}]", problems)
        close(sum(contributions), case["total_change"], tol, f"{label}.total_change", problems)
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
