# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the identified-atom mixture mean of a graph-posterior effect
envelope: the mean is sum(w_i e_i) / sum(w_i) over the identified atoms, and the unidentified
mass is one minus the identified weights. Exact rational arithmetic.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/graph_effect_envelope/reference.py --check

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
    weights = [F(str(w)) for w in document["identified_weights"]]
    effects = [F(str(e)) for e in document["effect_means"]]
    mixture = sum(w * e for w, e in zip(weights, effects)) / sum(weights)
    close(float(mixture), document["expected_mixture_mean"], 1e-12, "mixture mean", problems)
    close(float(1 - sum(weights)), document["unidentified_mass"], 1e-12, "unidentified mass", problems)
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
