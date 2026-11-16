#!/usr/bin/env python3
"""What would defects be at each pressure level and horizon, and after one fixed setting?

A response curve estimates the outcome at each pressure level (dose) and time horizon; an
intervention response follows the path after setting pressure to one level. The declared temporal
graph says which earlier pressure values affect today's defects.

The data are 400 deterministic steps: defects equal 0.9 times the pressure one step earlier plus
0.1 times the pressure two steps earlier, with no noise. The script asserts that the surface
carries pointwise and simultaneous bands, that the exported result loads, and that estimating the
study again reproduces it; it prints the values rather than asserting them, and reports each
band's calibration status as computed. The slowly varying pressure series is strongly
autocorrelated, which the declared graph omits, so the printed surface does not separate the two
lag coefficients; the Rust example uses a series whose lags are uncorrelated and checks the
surface against 0.9 and 0.1.

Run: `python examples/python/temporal_response_curve.py`."""

from __future__ import annotations

import math

import antecedent as _ant_api
import numpy as np
from antecedent import InterventionResponse, ResponseCurve, analyze, load
from antecedent.intervention import Set
from antecedent.outcomes import require_success

# Defects respond to pressure at lags 1 and 2. A pulse sets pressure once, one step back.
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

# The bands come from circular-block bootstrap replicates of the whole surface, which keep
# the serial dependence of the series. With bootstrap=0 the surface has no band: an analytic
# band would treat lag-aligned rows as independent.
result = require_success(
    analyze(
        data,
        graph=graph,
        query=curve,
        refute="none",
        inference=_ant_api.Frequentist(bootstrap=100),
        execution=_ant_api.Execution(seed=42),
    )
)
assert result.response is not None
curve_result = result
study = result.study
print("Calibration:", result.calibration.status)
artifact = result.export()
loaded = load(artifact, expected=artifact.expectation)
assert loaded.acceptance.recognized
assert require_success(_ant_api.estimate(study)).response.values == result.response.values
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
print(f"simultaneous band: critical={band.critical:.3f} from {band.replicates} replicates")
for point, lo_row, hi_row in zip(curve_result.response.points, band.lower, band.upper):
    print(f"  dose={point[0]:.1f} horizon={point[1]:.0f}  [{lo_row[0]:.4f}, {hi_row[0]:.4f}]")

# Next, hold the intervention level fixed and follow its effect over time.
path = require_success(
    analyze(
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
        inference=_ant_api.Frequentist(bootstrap=0),
        execution=_ant_api.Execution(seed=42),
    )
)
assert path.response is not None
print("intervention path:", [row[0] for row in path.response.values])
