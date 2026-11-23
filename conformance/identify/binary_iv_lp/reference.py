# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.1.3", "scipy==1.14.1"]
# ///
"""Retained independent linear-programming reference for sharp binary-IV bounds.

The sharp bounds on E[Y(1)-Y(0)] under instrument exogeneity and exclusion are the
optimum of a linear program over the distribution q of the 16 latent response types
(d0, d1, y0, y1).  This script solves that program with SciPy's HiGHS simplex/interior
solvers, a different algorithm from the product's enumeration of basic solutions, and writes
the case table.  The latent type is encoded here as a tuple of four named response
functions, independently of the product's bit packing.  It imports no Antecedent code and no
causal-inference package.

    uv run --frozen --script conformance/identify/binary_iv_lp/reference.py --write
    uv run --frozen --script conformance/identify/binary_iv_lp/reference.py --check
    uv run --frozen --script conformance/identify/binary_iv_lp/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_compare import values_close  # noqa: E402

ATOL = 1e-8
# Law cells within a Z arm are ordered (Y,D) = (0,0), (1,0), (0,1), (1,1).
CELLS = [(0, 0), (1, 0), (0, 1), (1, 1)]
# A latent type is (D(0), D(1), Y(0), Y(1)).
TYPES = list(itertools.product((0, 1), repeat=4))


def cell_of(t: tuple[int, int, int, int], z: int) -> int:
    d = t[z]
    y = t[2 + d]
    return CELLS.index((y, d))


def program(law: list[list[float]]):
    a = np.zeros((8, 16))
    b = np.zeros(8)
    for z in (0, 1):
        for cell in range(4):
            row = 4 * z + cell
            b[row] = law[z][cell]
            for k, t in enumerate(TYPES):
                a[row, k] = 1.0 if cell_of(t, z) == cell else 0.0
    return a, b


def bounds(law: list[list[float]]) -> dict:
    a, b = program(law)
    effect = np.array([t[3] - t[2] for t in TYPES], dtype=float)
    lo = linprog(effect, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
    hi = linprog(-effect, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
    if lo.status == 2 or hi.status == 2:
        return {"feasible": False, "lower": None, "upper": None}
    assert lo.status == 0 and hi.status == 0, (lo.message, hi.message)
    return {"feasible": True, "lower": float(lo.fun) + 0.0, "upper": float(-hi.fun) + 0.0}


def law_from_types(q: np.ndarray, digits: int) -> list[list[float]]:
    law = []
    for z in (0, 1):
        cells = np.zeros(4)
        for k, t in enumerate(TYPES):
            cells[cell_of(t, z)] += q[k]
        rounded = [round(float(c), digits) for c in cells[:3]]
        rounded.append(round(1.0 - sum(rounded), digits))
        law.append(rounded)
    return law


def named_cases() -> list[dict]:
    return [
        {"name": "balke_pearl_table1", "cells": [[0.0064, 0.9936, 0.0, 0.0], [0.0028, 0.1972, 0.001, 0.799]]},
        {"name": "perfect_compliance_effect_one", "cells": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]},
        {"name": "never_takers_only", "cells": [[0.3, 0.7, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0]]},
        {"name": "always_takers_only", "cells": [[0.0, 0.0, 0.4, 0.6], [0.0, 0.0, 0.4, 0.6]]},
        {"name": "null_instrument_mixed", "cells": [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]},
        {"name": "violates_instrumental_inequality", "cells": [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]},
        {"name": "violates_instrumental_inequality_treated_arm_sum", "cells": [[0.1, 0.1, 0.0, 0.8], [0.1, 0.0, 0.8, 0.1]]},
        {"name": "point_identified_zero_effect", "cells": [[0.0, 0.5, 0.5, 0.0], [0.5, 0.0, 0.0, 0.5]]},
    ]


def random_cases(seed: int = 20261027, count: int = 24) -> list[dict]:
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(count):
        alpha = rng.choice([0.2, 0.5, 1.0])
        q = rng.dirichlet(np.full(16, alpha))
        cases.append({"name": f"random_feasible_{i:02d}", "cells": law_from_types(q, 6)})
    return cases


def build_cases() -> list[dict]:
    out = []
    for case in named_cases() + random_cases():
        out.append({**case, **bounds(case["cells"])})
    return out


def check(document: dict) -> list[str]:
    problems = []
    if document.get("atol") != ATOL:
        problems.append("fixture tolerance changed from the reviewed bound")
    if not document["cases"]:
        problems.append("empty case selection")
    for case in document["cases"]:
        actual = bounds(case["cells"])
        for field in ("feasible", "lower", "upper"):
            if not values_close(case[field], actual[field], rtol=0.0, atol=ATOL):
                problems.append(f"{case['name']}.{field} differs from the LP optimum")
    if sum(not c["feasible"] for c in document["cases"]) < 2:
        problems.append("no infeasible counterexample retained")
    if not any(c["feasible"] and c["upper"] - c["lower"] > 0.05 for c in document["cases"]):
        problems.append("no case with a non-degenerate identified interval")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["cases"][0]["lower"] += 0.01
    if not check(wrong):
        failures.append("shifted lower bound accepted")
    flipped = copy.deepcopy(document)
    flipped["cases"][0]["feasible"] = False
    if not check(flipped):
        failures.append("flipped feasibility accepted")
    corrupted = copy.deepcopy(document)
    corrupted["cases"][1]["cells"][0][0] = 0.9
    if not check(corrupted):
        failures.append("corrupted input law accepted")
    widened = copy.deepcopy(document)
    widened["atol"] = 1e-2
    if not check(widened):
        failures.append("widened tolerance accepted")
    empty = copy.deepcopy(document)
    empty["cases"] = []
    if not check(empty):
        failures.append("empty selection accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    path = HERE / "expected.json"
    if args.write:
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        document = {
            **previous,
            "schema_version": 1,
            "estimand": "E[Y(1)-Y(0)] without monotonicity under instrument exogeneity and exclusion",
            "cell_order": "within each Z arm: (Y,D)=(0,0),(1,0),(0,1),(1,1)",
            "atol": ATOL,
            "cases": build_cases(),
        }
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {len(document['cases'])} cases")
        return 0
    document = json.loads(path.read_text(encoding="utf-8"))
    problems = check(document) if args.check or not args.self_test else []
    if args.self_test:
        problems += self_test(document)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
