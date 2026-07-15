"""analyze_ate identifier/estimator kwargs and enriched result schema."""

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
    "bootstrap_replicates_failed",
    "adjustment_set",
    "identification_status",
    "refutation_passed",
    "refutation_ran",
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
    # Default PlaceboAndRcc suite is NotApplicable for non-linear estimators.
    assert result.refutation_count == 0


@pytest.mark.parametrize(
    "estimator",
    [
        "propensity.stratification",
        "aipw",
        "propensity.matching",
        "distance.matching",
    ],
)
def test_analyze_ate_estimate_estimators_smoke(estimator):
    names, cols, edges = _confounded_scm(n=1000, seed=9)
    result = causal.analyze_ate(
        names,
        cols,
        edges,
        treatment="t",
        outcome="y",
        identifier="backdoor.adjustment",
        estimator=estimator,
        bootstrap=5,
        seed=2,
    )
    assert result.estimator_id == estimator
    assert abs(result.ate - 2.0) < 0.6, (estimator, result.ate)


def test_analyze_ate_iv_2sls_smoke():
    rng = random.Random(3)
    n = 1500
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = float(i % 2)
        u = rng.gauss(0.0, 1.0)
        ti = 0.6 * zi + u + 0.1 * rng.gauss(0.0, 1.0)
        yi = 2.0 * ti + u + 0.1 * rng.gauss(0.0, 1.0)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    result = causal.analyze_ate(
        ["t", "y", "z"],
        [t, y, z],
        [("z", "t"), ("t", "y")],
        treatment="t",
        outcome="y",
        identifier="iv",
        estimator="iv.2sls",
        bootstrap=5,
        seed=4,
    )
    assert result.estimator_id == "iv.2sls"
    assert abs(result.ate - 2.0) < 0.6, result.ate


def test_analyze_ate_frontdoor_smoke():
    rng = random.Random(4)
    n = 1500
    t = np.empty(n, dtype=np.float64)
    m = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        u = rng.gauss(0.0, 1.0)
        ti = u + 0.1 * rng.gauss(0.0, 1.0)
        mi = ti + 0.1 * rng.gauss(0.0, 1.0)
        yi = 2.0 * mi + u + 0.1 * rng.gauss(0.0, 1.0)
        t[i] = ti
        m[i] = mi
        y[i] = yi
    result = causal.analyze_ate(
        ["t", "y", "m"],
        [t, y, m],
        [("t", "m"), ("m", "y")],
        treatment="t",
        outcome="y",
        identifier="frontdoor",
        estimator="frontdoor.two_stage",
        bootstrap=5,
        seed=5,
    )
    assert result.estimator_id == "frontdoor.two_stage"
    assert abs(result.ate - 2.0) < 0.6, result.ate
