#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent reference for sparse feedback equilibria with hundreds of variables.

Each case is an affine equilibrium ``x = c + W x + u`` on a ring whose variable ``i`` reads
its predecessor, successor and itself. The truth is the dense solution of ``(I - W) x = c + u``
from ``numpy.linalg`` (LAPACK), the exact infinity-norm condition number of ``I - W`` from
its dense inverse, and the exact spectral radius of ``|W|`` from ``numpy.linalg.eigvals``.
Nothing here shares code with the production sparse solver.

    python3 conformance/gcm/sparse_feedback_equilibrium/reference.py          # write
    python3 conformance/gcm/sparse_feedback_equilibrium/reference.py --check  # compare
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected.json"

CASES = (
    {
        "case": "contractive_ring",
        "order": 320,
        "predecessor": 0.35,
        "successor": 0.25,
        "own": 0.1,
        "clamps": [],
    },
    {
        "case": "noncontractive_directed_ring",
        "order": 300,
        "predecessor": 1.5,
        "successor": 0.0,
        "own": 0.0,
        "clamps": [],
    },
    {
        "case": "clamped_directed_ring",
        "order": 300,
        "predecessor": 0.5,
        "successor": 0.0,
        "own": 0.0,
        "clamps": [[10, 2.0], [150, -1.0], [299, 0.5]],
    },
)


def solve_case(spec: dict) -> dict:
    n = spec["order"]
    weights = np.zeros((n, n))
    for i in range(n):
        weights[i, (i - 1) % n] += spec["predecessor"]
        weights[i, (i + 1) % n] += spec["successor"]
        weights[i, i] += spec["own"]
    intercept = np.array([0.05 * ((i % 7) - 3) for i in range(n)])
    noise = np.sin(0.37 * np.arange(n))
    a = np.eye(n) - weights
    rhs = intercept + noise
    for variable, value in spec["clamps"]:
        a[variable, :] = 0.0
        a[variable, variable] = 1.0
        rhs[variable] = value
    x = np.linalg.solve(a, rhs)
    condition = float(
        np.linalg.norm(a, np.inf) * np.linalg.norm(np.linalg.inv(a), np.inf)
    )
    radius = float(np.max(np.abs(np.linalg.eigvals(np.abs(weights)))))
    return {
        **spec,
        "intercept_rule": "0.05 * ((i % 7) - 3)",
        "noise_rule": "sin(0.37 * i)",
        "values_head": [float(v) for v in x[:6]],
        "values_sum": float(x.sum()),
        "values_max_abs": float(np.abs(x).max()),
        "infinity_condition": condition,
        "abs_weight_spectral_radius": radius,
        "abs_weight_row_sum": float(np.abs(weights).sum(axis=1).max()),
    }


def build() -> dict:
    return {
        "oracle": {
            "kind": "dense_linear_algebra",
            "method": "numpy.linalg dense solve, inverse and eigenvalues of the affine equilibrium",
            "independence_basis": (
                "Dense LAPACK solves of (I - W) x = c + u on rings of 300 to 320 variables, the "
                "exact infinity-norm condition number from the dense inverse and the exact "
                "spectral radius of |W|. The generator is retained and checked by the "
                "conformance reference check; the author of the reference is the repository "
                "author and independence from the production implementation has not been reviewed."
            ),
            "generator": "conformance/gcm/sparse_feedback_equilibrium/reference.py",
            "historical_exposure": "Historical author exposure is unrecorded and remains unknown.",
        },
        "schema_version": 1,
        "cases": [solve_case(spec) for spec in CASES],
    }


def close(left, right) -> bool:
    return bool(
        np.allclose(
            np.asarray(left, float), np.asarray(right, float), rtol=1e-9, atol=0
        )
    )


def check(document: dict, committed: dict) -> list[str]:
    problems = []
    if document["oracle"] != committed["oracle"]:
        problems.append("oracle record differs")
    for fresh, stored in zip(document["cases"], committed["cases"], strict=True):
        for key, value in fresh.items():
            same = (
                close(value, stored[key])
                if isinstance(value, (int, float, list))
                and key not in ("clamps", "order")
                else value == stored[key]
            )
            if not same:
                problems.append(f"{fresh['case']}.{key}")
    return problems


def main() -> int:
    document = build()
    if "--check" in sys.argv[1:]:
        committed = json.loads(EXPECTED.read_text(encoding="utf-8"))
        problems = check(document, committed)
        if len(document["cases"]) != len(committed["cases"]):
            problems.append("case count")
        for problem in problems:
            print(f"sparse feedback reference mismatch: {problem}", file=sys.stderr)
        return 1 if problems else 0
    EXPECTED.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
