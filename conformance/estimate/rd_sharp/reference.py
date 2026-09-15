"""Independent reference for the sharp-RD analytic standard errors.

The `rd_sharp` fixture's point pin (jump 3.0 within 0.5) is recovered from a
seeded SCM generated in Rust. This script adds the `se_reference` block: a
small deterministic heteroskedastic design, frozen row by row, with the jump
coefficient and its analytic SEs computed here from the textbook formulas:

    X = [1, T, R - c, T (R - c)] over rows with |R - c| <= h,  T = 1[R >= c]
    beta = (X'X)^-1 X'y,  e = y - X beta,  n rows, p = 4 columns
    homoskedastic: Var = e'e / (n - p) * (X'X)^-1
    HC1:           Var = n / (n - p) * (X'X)^-1 X' diag(e^2) X (X'X)^-1

The outcome noise scale grows with |R - c|, so the two SEs differ. `rd.sharp`
reports HC1 by default and the homoskedastic SE on explicit opt-in;
`crates/antecedent/tests/estimate_conformance.rs` pins both against this block.

Run: ``python3 conformance/estimate/rd_sharp/reference.py`` (add ``--write``
to update expected.json, ``--check`` to compare).

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N = 48
CUTOFF = 0.0
BANDWIDTH = 0.75


def rows() -> tuple[list[float], list[float]]:
    running = [-1.0 + 2.0 * (i + 0.5) / N for i in range(N)]
    outcome = []
    for i, r in enumerate(running):
        t = 1.0 if r >= CUTOFF else 0.0
        scale = 0.1 + 0.6 * abs(r - CUTOFF)
        noise = scale * math.sin(1.7 * i + 0.3)
        outcome.append(2.0 + 0.5 * r + 3.0 * t - 0.8 * t * r + noise)
    return running, outcome


def standard_errors(running: list[float], outcome: list[float]) -> dict[str, float]:
    r = np.asarray(running)
    y = np.asarray(outcome)
    keep = np.abs(r - CUTOFF) <= BANDWIDTH
    centered = r[keep] - CUTOFF
    t = (centered >= 0.0).astype(float)
    x = np.column_stack([np.ones_like(t), t, centered, t * centered])
    y = y[keep]
    n, p = x.shape
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    e = y - x @ beta
    homoskedastic = (e @ e) / (n - p) * bread
    meat = x.T @ (x * (e**2)[:, None])
    hc1 = n / (n - p) * bread @ meat @ bread
    return {
        "window_rows": int(n),
        "jump": float(beta[1]),
        "se_hc1": float(math.sqrt(hc1[1, 1])),
        "se_homoskedastic": float(math.sqrt(homoskedastic[1, 1])),
    }


def build_block() -> dict:
    running, outcome = rows()
    return {
        "cutoff": CUTOFF,
        "bandwidth": BANDWIDTH,
        "running": running,
        "outcome": outcome,
        "expected": standard_errors(running, outcome),
        "relative_tolerance": 1e-9,
    }


def main() -> None:
    path = HERE / "expected.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    block = build_block()
    if "--write" in sys.argv:
        fixture["se_reference"] = block
        path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    if "--check" in sys.argv and fixture.get("se_reference") != block:
        raise SystemExit("expected.json se_reference is stale; rerun with --write")
    print(json.dumps(block["expected"], indent=2))


if __name__ == "__main__":
    main()
