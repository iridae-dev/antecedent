#!/usr/bin/env python3
"""Temporal dose × horizon response on a TemporalDag.

Shows ``ResponseCurve`` with a temporal attachment (horizons, pulse policy,
treatment lag). Requires a built antecedent extension (``maturin develop``).
"""

from __future__ import annotations

import math

import numpy as np
from antecedent import InterventionResponse, ResponseCurve, analyze
from antecedent.intervention import Set

# Pressure at lag 1 and 2 drives defect; pulse policy at t-1.
N = 400
pressure = np.array([math.sin(0.04 * t) for t in range(N)], dtype=np.float64)
defect = np.zeros(N, dtype=np.float64)
for t in range(1, N):
    defect[t] = 0.9 * pressure[t - 1] + 0.1 * pressure[t - 2]

data = {"pressure": pressure, "defect": defect}
graph = [("pressure", 1, "defect", 0), ("pressure", 2, "defect", 0)]

curve = ResponseCurve(
    "pressure",
    "defect",
    grid=[0.0, 0.5, 1.0],
    horizons=[1, 2],
    policy="pulse",
    treatment_lag=1,
)

result = analyze(data, graph=graph, query=curve, refute=False, bootstrap=0, seed=42)
assert result.response is not None
curve_result = result
print("dose × horizon surface (mean, lower, upper):")
assert curve_result.uncertainty.lower is not None
assert curve_result.uncertainty.upper is not None
for point, mean_row, lo_row, hi_row in zip(
    curve_result.response.points,
    curve_result.response.values,
    curve_result.uncertainty.lower,
    curve_result.uncertainty.upper,
):
    dose, horizon = point[0], point[1]
    print(
        f"  dose={dose:.1f} horizon={horizon:.0f}  "
        f"mean={mean_row[0]:.4f}  [{lo_row[0]:.4f}, {hi_row[0]:.4f}]"
    )

# Intervention path at a fixed level (same licensed temporal cell family).
path = analyze(
    data,
    graph=graph,
    query=InterventionResponse(
        "defect",
        intervention=Set("pressure", 1.0),
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    ),
    refute=False,
    bootstrap=0,
    seed=42,
)
assert path.response is not None
print("intervention path:", [row[0] for row in path.response.values])
