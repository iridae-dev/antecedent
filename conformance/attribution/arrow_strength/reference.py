# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the arrow-strength fixture: strength = beta^2 * Var(parent) for a
root parent with standard deviation sigma (Var = sigma^2), and a do contrast of width delta
is beta * delta.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/attribution/arrow_strength/reference.py --check

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
    variance = {}
    for parent in document["parents"]:
        derived = parent["sigma"] ** 2
        variance[parent["raw"]] = derived
        close(derived, parent["variance"], tol, f"parent {parent['raw']} variance", problems)
    for edge in document["edges"]:
        derived = edge["coefficient"] ** 2 * variance[edge["parent_raw"]]
        close(derived, edge["strength"], tol, f"edge {edge['parent_raw']}->{edge['child_raw']} strength", problems)
    by_strength = sorted(document["edges"], key=lambda e: -(e["coefficient"] ** 2 * variance[e["parent_raw"]]))
    by_beta = sorted(document["edges"], key=lambda e: -abs(e["coefficient"]))
    inverted = [e["parent_raw"] for e in by_strength] != [e["parent_raw"] for e in by_beta]
    if inverted != document["abs_beta_ranking_is_inverted"]:
        problems.append("abs_beta_ranking_is_inverted disagrees with the derived rankings")
    contrast = document["do_contrast"]
    close(contrast["coefficient"] * contrast["delta"], contrast["expected"], tol, "do_contrast", problems)
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
