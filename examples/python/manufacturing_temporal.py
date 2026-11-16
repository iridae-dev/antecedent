#!/usr/bin/env python3
"""How much does a one-step pressure pulse move defects one step later?

Data: a simulated series of 400 steps with no noise, in which defects equal 0.9 times the
pressure one step earlier. The graph declares that single lag-1 edge; the lag structure is
declared, not discovered.

The script asserts the pulse effect is a point within 0.05 of 0.9. Bootstrap is off, so there is
no interval, and the estimate is conditional on the declared lagged graph.

Run: `python examples/python/manufacturing_temporal.py`."""

from __future__ import annotations

import math

import antecedent as _ant_api
import numpy as np
from antecedent import PulseEffect, analyze
from antecedent.outcomes import require_success


def main() -> None:
    n = 400
    pressure = np.array([math.sin(0.04 * t) for t in range(n)], dtype=np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    # A lagged edge (pressure at t-1 -> defect at t) states when the cause acts. The pulse query
    # raises pressure to 1.0 for one step and reads defects one step later.
    outcome = analyze(
        {"pressure": pressure, "defect": defect},
        graph=[("pressure", 1, "defect", 0)],
        query=PulseEffect(
            treatment="pressure",
            outcome="defect",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        inference=_ant_api.Frequentist(bootstrap=0),
        execution=_ant_api.Execution(seed=42),
    )
    result = require_success(outcome)
    assert result.answer.kind == "point"
    print("Calibration:", result.calibration.status)
    print(
        f"ATE={result.answer.value:.4f} plan={result.performance.plan_id} "
        f"peak_mem={result.performance.peak_memory_bytes} method={result.estimate.method}"
    )
    assert abs(result.answer.value - 0.9) < 0.05, result.answer.value


if __name__ == "__main__":
    main()
