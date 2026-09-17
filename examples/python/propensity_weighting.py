#!/usr/bin/env python3
"""Adjust for treatment selection with inverse-probability weights.

The baseline variable z affects both treatment assignment and the outcome.
A raw group comparison therefore mixes the treatment effect with differences
in z. Weighting adjusts for that selection; the known treatment effect is 2.

This example selects propensity.weighting explicitly. It also prints overlap
diagnostics, which help you check whether the weights rely on too few rows.
Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import math
import random

import numpy as np
from antecedent import AverageEffect, analyze


def main() -> None:
    rng = random.Random(7)
    n = 1200
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi

    result = analyze(
        {"t": t, "y": y, "z": z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect(treatment="t", outcome="y"),
        estimator="propensity.weighting",
        bootstrap=30,
        seed=11,
    )
    assert result.answer.kind == "point"
    print("Calibration:", result.calibration.status)
    print(
        f"ATE={result.answer.value:.4f} method={result.estimate.method} "
        f"estimator={result.estimate.estimator_id} "
        f"overlap_ess={result.estimate.overlap_ess} "
        f"overlap_propensity_min={result.estimate.overlap_propensity_min}"
    )
    assert abs(result.answer.value - 2.0) < 0.35, result.answer.value
    assert result.estimate.estimator_id == "propensity.weighting"
    assert result.estimate.overlap_ess is not None


if __name__ == "__main__":
    main()
