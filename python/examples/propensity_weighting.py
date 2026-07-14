#!/usr/bin/env python3
"""Propensity-weighting (IPW) `analyze_ate` example .

Requires a built causal extension (`maturin develop` in python/).

Confounded SCM: `Z ~ N(0,1)`, `T ~ Bernoulli(sigmoid(-0.4 + 0.9 Z))`,
`Y = 2T + Z + noise`. True ATE = 2; a naive unadjusted contrast is biased by
`Z`, so this exercises the `propensity.weighting` estimator explicitly rather
than the -3 default (`linear.adjustment.ate`).
"""

from __future__ import annotations

import math
import random

import numpy as np

from causal import analyze_ate


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

    result = analyze_ate(
        ["t", "y", "z"],
        [t, y, z],
        edges=[("z", "t"), ("z", "y"), ("t", "y")],
        treatment="t",
        outcome="y",
        estimator="propensity.weighting",
        bootstrap=30,
        seed=11,
    )
    print(
        f"ATE={result.ate:.4f} method={result.method} estimator={result.estimator_id} "
        f"overlap_ess={result.overlap_ess} overlap_propensity_min={result.overlap_propensity_min}"
    )
    assert abs(result.ate - 2.0) < 0.35, result.ate
    assert result.estimator_id == "propensity.weighting"
    assert result.overlap_ess is not None


if __name__ == "__main__":
    main()
