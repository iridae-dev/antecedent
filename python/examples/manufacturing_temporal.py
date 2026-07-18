#!/usr/bin/env python3
"""Manufacturing-style temporal analyze() example.

Requires a built causal extension (`maturin develop` in python/).
"""

from __future__ import annotations

import math

import numpy as np

from causal import PulseEffect, analyze


def main() -> None:
    n = 400
    pressure = np.array([math.sin(0.04 * t) for t in range(n)], dtype=np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    result = analyze(
        {"pressure": pressure, "defect": defect},
        graph=[("pressure", 1, "defect", 0)],
        query=PulseEffect(
            treatment="pressure",
            outcome="defect",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        bootstrap=0,
        seed=42,
    )
    print(
        f"ATE={result.ate:.4f} plan={result.performance.plan_id} "
        f"peak_mem={result.performance.peak_memory_bytes} method={result.estimate.method}"
    )
    assert abs(result.ate - 0.9) < 0.05, result.ate


if __name__ == "__main__":
    main()
