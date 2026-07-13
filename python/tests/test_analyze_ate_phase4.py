"""Phase 4: analyze_ate identifier/estimator kwargs and enriched result schema."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("causal")
import causal

SCHEMA_FIELDS = {
    "ate",
    "se_analytic",
    "se_bootstrap",
    "adjustment_set",
    "identification_status",
    "refutation_passed",
    "refutation_count",
    "assumption_count",
    "derivation_step_count",
    "method",
    "estimator_id",
    "overlap_ess",
    "overlap_propensity_min",
}


def _confounded_scm(n: int = 800, seed: int = 5):
    rng = random.Random(seed)
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
    names = ["t", "y", "z"]
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    return names, [t, y, z], edges


def test_analyze_ate_default_pair_schema_and_fields():
    names, cols, edges = _confounded_scm()
    result = causal.analyze_ate(
        names, cols, edges, treatment="t", outcome="y", bootstrap=10, seed=1
    )
    for field in SCHEMA_FIELDS:
        assert hasattr(result, field), field
    assert result.method == "backdoor.adjustment"
    assert result.estimator_id in ("", "linear.adjustment.ate")
    assert result.overlap_ess is None
    assert result.overlap_propensity_min is None


def test_analyze_ate_propensity_weighting_recovers_ate_and_overlap():
    names, cols, edges = _confounded_scm()
    result = causal.analyze_ate(
        names,
        cols,
        edges,
        treatment="t",
        outcome="y",
        identifier="backdoor.adjustment",
        estimator="propensity.weighting",
        bootstrap=10,
        seed=1,
    )
    assert abs(result.ate - 2.0) < 0.4, result.ate
    assert result.estimator_id == "propensity.weighting"
    assert result.overlap_ess is not None
    assert result.overlap_propensity_min is not None
    # Placebo/RCC refuters are hardwired to the default estimator; skipped otherwise.
    assert result.refutation_count == 0
