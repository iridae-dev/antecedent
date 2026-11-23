# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Definition-level check of the identification status of the declared conditional-intervention cases.

For each case the target is P(outcomes | do(interventions), conditioning) on the declared
graph. Every declared case is a single-confounder back-door graph (one common parent ``Z`` of
treatment and outcome); the reference verifies the identification claim by exact enumeration of
binary structural models, not by running any identification algorithm:

* a model is a parameter set (P(Z=1), P(T=1|Z), P(Y=1|T,Z)) of exact rationals drawn from a
  fixed grid of 4 x 4 x 4 x 4 x 4 x 4 x 4 = 16384 points;
* the interventional law is computed from the definition (truncated factorisation: the
  treatment's mechanism is replaced by a point mass), and the observational law from the joint;
* a case is ``identified`` when the interventional quantity equals the stated functional of the
  observational joint at every grid point, the functional being sum_z P(y|t,z) P(z) for the
  unconditional case and P(y|t,z) for the case that conditions on ``Z``.

The functional is named in each case below from the do-calculus (adjustment for the common
parent); the enumeration is what shows it reproduces the intervention for every parameter set.
Standard library only; imports no Antecedent code.

    uv run --frozen --script conformance/identify/idc/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = [F(1, 5), F(2, 5), F(3, 5), F(4, 5)]


def joint(pz, pt, py):
    return {
        (z, t, y): (pz if z else 1 - pz)
        * (pt[z] if t else 1 - pt[z])
        * (py[t, z] if y else 1 - py[t, z])
        for z in (0, 1)
        for t in (0, 1)
        for y in (0, 1)
    }


def interventional(pz, py, t_set):
    """P(y, z | do(T = t_set)) by truncated factorisation."""
    return {
        (z, y): (pz if z else 1 - pz) * (py[t_set, z] if y else 1 - py[t_set, z])
        for z in (0, 1)
        for y in (0, 1)
    }


def case_holds(case):
    edges = sorted(case["edges"])
    if edges != sorted(["Z->T", "Z->Y", "T->Y"]):
        raise ValueError(f"{case['id']}: reference only covers the single-confounder back-door graph")
    if case["outcomes"] != ["Y"] or case["interventions"] != ["do(T=1)"]:
        raise ValueError(f"{case['id']}: unsupported query")
    conditioning = case["conditioning"]
    if conditioning not in ([], ["Z"]):
        raise ValueError(f"{case['id']}: unsupported conditioning")
    t_set = 1
    for pz, pt0, pt1, py00, py01, py10, py11 in itertools.product(GRID, repeat=7):
        pt = {0: pt0, 1: pt1}
        py = {(0, 0): py00, (0, 1): py01, (1, 0): py10, (1, 1): py11}
        obs = joint(pz, pt, py)
        do = interventional(pz, py, t_set)
        for y in (0, 1):
            if conditioning == []:
                truth = sum(do[z, y] for z in (0, 1))
                functional = sum(
                    obs[z, t_set, y] / (obs[z, t_set, 0] + obs[z, t_set, 1])
                    * (obs[z, 0, 0] + obs[z, 0, 1] + obs[z, 1, 0] + obs[z, 1, 1])
                    for z in (0, 1)
                )
                if truth != functional:
                    return False
            else:
                for z in (0, 1):
                    truth = do[z, y] / (do[z, 0] + do[z, 1])
                    functional = obs[z, t_set, y] / (obs[z, t_set, 0] + obs[z, t_set, 1])
                    if truth != functional:
                        return False
    return True


def check(document):
    problems = []
    for case in document["cases"]:
        derived = "identified" if case_holds(case) else "not_identified_by_the_adjustment_functional"
        if derived != case["status"]:
            problems.append(f"{case['id']}: fixture {case['status']} vs derived {derived}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = check(document)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    print("idc reference matches expected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
