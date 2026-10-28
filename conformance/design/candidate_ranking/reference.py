# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Derive the ranking targets of the candidate-ranking fixture from the exact information gains.

The exact expected information gains of the six candidate designs are the independent
``dit``-based values frozen in the sibling fixture ``conformance/design/expected_information_gain``
(their own retained generator recomputes them). This script does not recompute them: it reads
them and checks that the candidate-ranking fixture's declared quantities follow from them:

* ``expected_native_order`` is the candidate list sorted by exact gain, largest first, with no
  ties and a separation between neighbours larger than twice ``score_atol`` (so a score within
  ``score_atol`` of its exact gain cannot swap two neighbours);
* ``adaptive_top_candidate`` is the exact-gain argmax;
* every candidate named in the order has an exact gain recorded.

Standard library only; imports no Antecedent code.

    uv run --frozen --script conformance/design/candidate_ranking/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EIG = HERE.parent / "expected_information_gain" / "expected.json"


def check(document, eig):
    gains = {c["name"]: c["expected_information_gain"] for c in eig["cases"]}
    problems = []
    order = document["expected_native_order"]
    missing = [name for name in order if name not in gains]
    if missing:
        return [f"no exact gain recorded for {missing}"]
    derived = sorted(order, key=lambda name: -gains[name])
    if derived != order:
        problems.append(f"native order {order} vs exact-gain order {derived}")
    if set(order) != set(gains):
        problems.append("candidate sets differ between the two fixtures")
    if document["acceptance"]["adaptive_top_candidate"] != derived[0]:
        problems.append(f"top candidate {document['acceptance']['adaptive_top_candidate']} vs {derived[0]}")
    atol = document["acceptance"]["score_atol"]
    for left, right in zip(derived, derived[1:]):
        if gains[left] - gains[right] <= 2 * atol:
            problems.append(f"{left} and {right} are within 2 * score_atol of each other")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    eig = json.loads(EIG.read_text(encoding="utf-8"))
    problems = check(document, eig)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    print("candidate_ranking reference matches expected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
