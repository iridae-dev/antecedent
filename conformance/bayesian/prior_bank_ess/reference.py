# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the prior-strength ESS accounting: effective ESS of a source is
alpha * ess, the composed ESS is the sum of effective ESS when every contributing source
declares one (otherwise absent), and the Kish concentration is (sum w)^2 / sum w^2 over the
applied alphas.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/prior_bank_ess/reference.py --check

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


def kish(weights):
    return sum(weights) ** 2 / sum(w * w for w in weights)


def source_list(case):
    if "source_a" in case:
        return [case["source_a"], case["source_b"]]
    return [{"alpha": case["alpha"], "ess": case["ess"]}]


def check(document):
    problems = []
    tol = document["tol"]
    for name in ("power", "power_partial_coverage", "power_dropped_source"):
        case = document[name]
        sources = source_list(case)
        effective = [None if s["ess"] is None else s["alpha"] * s["ess"] for s in sources]
        for index, (got, want) in enumerate(zip(effective, case["expected_effective_ess"])):
            if (got is None) != (want is None):
                problems.append(f"{name}.effective_ess[{index}] presence differs")
            elif got is not None:
                close(got, want, tol, f"{name}.effective_ess[{index}]", problems)
        composed = None if any(e is None for e in effective) else sum(effective)
        want = case["expected_composed_ess"]
        if (composed is None) != (want is None):
            problems.append(f"{name}.composed_ess presence differs")
        elif composed is not None:
            close(composed, want, tol, f"{name}.composed_ess", problems)
        close(kish([s["alpha"] for s in sources]), case["expected_kish_ess"], tol, f"{name}.kish_ess", problems)
    mixture = document["mixture"]
    close(mixture["alpha"] * mixture["ess"], mixture["expected_effective_ess"][0], tol, "mixture.effective_ess", problems)
    if mixture["expected_composed_ess"] is not None:
        problems.append("mixture composed ESS is declared absent by the fixture notes")
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
