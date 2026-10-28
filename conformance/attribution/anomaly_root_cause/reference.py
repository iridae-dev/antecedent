# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the root-cause ranking fixture from the posterior draws: score is
the absolute posterior mean of a component, graph_std its population standard deviation
sqrt(E[d^2] - E[d]^2), and components rank by descending score.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/attribution/anomaly_root_cause/reference.py --check

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
    n = document["n_draws"]
    flat = document["posterior_draws_colmajor"]
    columns = [flat[i * n:(i + 1) * n] for i in range(len(flat) // n)]
    rows = []
    for raw, column in enumerate(columns):
        mean = sum(column) / n
        mean_square = sum(d * d for d in column) / n
        rows.append((abs(mean), math.sqrt(max(mean_square - mean * mean, 0.0)), raw))
    rows.sort(key=lambda r: -r[0])
    if len(rows) != len(document["expected_ranking"]):
        return [f"component count {len(rows)} != {len(document['expected_ranking'])}"]
    for rank, ((score, std, raw), want) in enumerate(zip(rows, document["expected_ranking"])):
        if raw != want["component_raw"]:
            problems.append(f"rank {rank}: component {raw} != fixture {want['component_raw']}")
        close(score, want["score"], tol, f"rank {rank} score", problems)
        close(std, want["graph_std"], tol, f"rank {rank} graph_std", problems)
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
