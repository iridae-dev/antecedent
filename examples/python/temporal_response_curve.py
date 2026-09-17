#!/usr/bin/env python3
"""Compare pressure levels and their delayed effects on defects.

A response curve asks what would happen at each pressure level and time
horizon. We also estimate the path after setting pressure to one fixed level.
The graph states which earlier pressure measurements affect today's outcome.

Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import math

import numpy as np
from antecedent import InterventionResponse, ResponseCurve, analyze, load
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

# The band comes from joint circular-block bootstrap replicates of the whole
# surface; bootstrap=0 keeps the point surface and withholds
# the band, because an analytic band would treat lag-aligned rows as independent.
result = analyze(data, graph=graph, query=curve, refute="none", bootstrap=100, seed=42)
assert result.response is not None
curve_result = result
study = result.study
print("Calibration:", result.calibration.status)
loaded = load(result.export())
assert loaded.acceptance.verified
assert study.estimate().response.values == result.response.values
print("dose × horizon surface (mean, pointwise 95% lower, upper):")
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

# The simultaneous band covers every (dose, horizon) cell at once.
band = curve_result.simultaneous_band
assert band is not None
print(
    f"simultaneous band: critical={band.critical:.3f} from {band.replicates} replicates"
)
for point, lo_row, hi_row in zip(curve_result.response.points, band.lower, band.upper):
    print(
        f"  dose={point[0]:.1f} horizon={point[1]:.0f}  [{lo_row[0]:.4f}, {hi_row[0]:.4f}]"
    )

# Next, hold the intervention level fixed and follow its effect over time.
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
    refute="none",
    bootstrap=0,
    seed=42,
)
assert path.response is not None
print("intervention path:", [row[0] for row in path.response.values])
