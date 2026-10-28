# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent batch recomputation of the rolling mechanism diagnostics checkpoints.

For each declared case and checkpoint row the reference rebuilds the trailing window of at
most ``window`` rows ending at that row from the declared data generator, fits the
two-coefficient (intercept, slope) least-squares line from the exact normal equations in
rational arithmetic (the double-precision inputs are taken as the exact rationals they are),
and recomputes directly: the coefficients, the residual sum of squares, the residual variance
SSE / (n - 2), the mean absolute residual, and the maximum absolute running sum of the
residuals taken in time order. There is no incremental update, so the product's downdating
across its internal rebuild boundary is compared with a from-scratch fit.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/state/rolling_mechanism_diagnostics/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOLERANCE = 1e-12


def row(i):
    """x = ((i mod 23) - 11)/7 + 0.15 sin(0.19 i); y per the fixture's declared response."""
    x = ((i % 23) - 11) / 7 + 0.15 * math.sin(0.19 * i)
    shift = 0.4 if i >= 90 else 0.0
    y = 1.5 - 0.7 * x + shift + 0.2 * math.sin(0.37 * i) + 0.05 * math.cos(0.11 * i)
    return x, y


def window_fit(end, window):
    start = max(0, end + 1 - window)
    rows = [tuple(F(v) for v in row(i)) for i in range(start, end + 1)]
    n = len(rows)
    sx = sum(x for x, _ in rows)
    sy = sum(y for _, y in rows)
    sxx = sum(x * x for x, _ in rows)
    sxy = sum(x * y for x, y in rows)
    det = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / det
    intercept = (sy - slope * sx) / n
    residuals = [y - intercept - slope * x for x, y in rows]
    sse = sum(e * e for e in residuals)
    running, peak = F(0), F(0)
    for e in residuals:
        running += e
        peak = max(peak, abs(running))
    return {
        "n": n,
        "beta": [float(intercept), float(slope)],
        "sse": float(sse),
        "residual_variance": float(sse / (n - 2)),
        "mean_absolute_residual": float(sum(abs(e) for e in residuals) / n),
        "max_absolute_cusum": float(peak),
    }


def check(document):
    problems = []
    for case in document["cases"]:
        for frozen in case["checkpoints"]:
            derived = window_fit(frozen["row"], case["window"])
            label = f"{case['name']}@{frozen['row']}"
            if derived["n"] != frozen["n"]:
                problems.append(f"{label}: n {frozen['n']} vs {derived['n']}")
            for key in ("sse", "residual_variance", "mean_absolute_residual", "max_absolute_cusum"):
                if abs(derived[key] - frozen[key]) > TOLERANCE:
                    problems.append(f"{label} {key}: {frozen[key]!r} vs {derived[key]!r}")
            for k in range(2):
                if abs(derived["beta"][k] - frozen["beta"][k]) > TOLERANCE:
                    problems.append(f"{label} beta[{k}]: {frozen['beta'][k]!r} vs {derived['beta'][k]!r}")
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
    print("rolling_mechanism_diagnostics reference matches expected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
