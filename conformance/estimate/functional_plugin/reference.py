# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the plug-in interventional mean of the binary confounding table.

The table is the eight (z, t, y, count) cells that the executing assertion builds, transcribed
below as input.  The back-door functional E[Y | do(T = 1)] = sum_z P(Z = z) P(Y = 1 | T = 1, Z = z)
is evaluated in exact rational arithmetic from those counts; the interventional distribution
over y = 0, 1 must sum to one.

Standard library only; imports no Antecedent code.

    uv run --frozen --script conformance/estimate/functional_plugin/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (z, t, y, count), transcribed from the table built by the executing assertion.
CELLS = [
    (0, 0, 0, 21), (0, 0, 1, 9), (0, 1, 0, 4), (0, 1, 1, 16),
    (1, 0, 0, 12), (1, 0, 1, 3), (1, 1, 0, 14), (1, 1, 1, 21),
]


def derive():
    total = sum(c for *_, c in CELLS)
    p_z = {z: Fraction(sum(c for zz, _, _, c in CELLS if zz == z), total) for z in (0, 1)}
    cond = {}
    for z in (0, 1):
        n_t1 = sum(c for zz, t, _, c in CELLS if zz == z and t == 1)
        n_y1 = sum(c for zz, t, y, c in CELLS if zz == z and t == 1 and y == 1)
        cond[z] = Fraction(n_y1, n_t1)
    mean = sum(p_z[z] * cond[z] for z in (0, 1))
    atoms = {1: mean, 0: sum(p_z[z] * (1 - cond[z]) for z in (0, 1))}
    return p_z, cond, mean, atoms


def check(document):
    problems = []
    p_z, cond, mean, atoms = derive()
    case = document["case"]
    tol = document["acceptance"]["atol"]
    for z in (0, 1):
        if abs(float(p_z[z]) - case["z_mass"][z]) > tol:
            problems.append(f"z_mass[{z}] {float(p_z[z])} != {case['z_mass'][z]}")
        if abs(float(cond[z]) - case["conditional_outcome_probability"][z]) > tol:
            problems.append(f"P(Y=1|T=1,Z={z}) {float(cond[z])} != fixture")
    if abs(float(mean) - case["interventional_mean"]) > tol:
        problems.append(f"interventional_mean {float(mean)} != {case['interventional_mean']}")
    if abs(float(sum(atoms.values())) - case["atom_probability_sum"]) > tol:
        problems.append("atom probability sum differs")
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
