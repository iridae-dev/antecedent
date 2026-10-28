# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact interventional mean for the declared eight-cell count table.

The graph is Z -> T, Z -> Y, T -> Y over binary variables with cell counts for (z, t, y) as in
the executing assertion. The interventional law P(Y | do(T = 1)) of that graph is, by the
truncated factorisation, sum_z P(z) P(Y | T = 1, z), evaluated here in exact rational arithmetic
from the counts only (the empirical distribution is the plug-in distribution). The expected mean
is E[Y | do(T = 1)] = P(Y = 1 | do(T = 1)).

Standard library only; imports no Antecedent code.

    uv run --frozen --script conformance/estimate/interventional_distribution/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
# (z, t, y) -> count; the same table as the executing assertion, restated here.
COUNTS = {
    (0, 0, 0): 21, (0, 0, 1): 9, (0, 1, 0): 4, (0, 1, 1): 16,
    (1, 0, 0): 12, (1, 0, 1): 3, (1, 1, 0): 14, (1, 1, 1): 21,
}


def interventional_law(counts, t_set):
    n = sum(counts.values())
    law = {0: F(0), 1: F(0)}
    for z in (0, 1):
        pz = F(sum(c for (zz, _, _), c in counts.items() if zz == z), n)
        nt = sum(c for (zz, t, _), c in counts.items() if zz == z and t == t_set)
        for y in (0, 1):
            law[y] += pz * F(counts[z, t_set, y], nt)
    return law


def check(document):
    law = interventional_law(COUNTS, 1)
    problems = []
    if sum(law.values()) != 1:
        problems.append("interventional law does not sum to one")
    mean = float(law[1])
    if abs(mean - document["mean"]) > 1e-12:
        problems.append(f"mean {document['mean']!r} vs derived {mean!r}")
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
    print("interventional_distribution reference matches expected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
