"""analyze() identifier/estimator kwargs and nested result schema.

IPW ATE dual: shares structural confounded SCM (true ATE=2) and acceptance band
with Rust `end_to_end_propensity_weighting_recovers_confounded_effect`
(`crates/causal/src/lib.rs`). Cross-language floor: |ate − 2| < 0.4
(Rust unit test uses a tighter 0.3 on its RNG stream).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded_scm(n: int = 800, seed: int = 5):
    """Confounded Z→T, Z→Y, T→Y with structural ATE=2 (Python dual of Rust IPW fixture)."""
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
    data = {"t": t, "y": y, "z": z}
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    return data, edges


def test_analyze_default_pair_schema_and_fields():
    data, edges = _confounded_scm()
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        bootstrap=10,
        seed=1,
    )
    assert result.identification.method == "backdoor.adjustment"
    assert result.estimate.estimator_id in ("", "linear.adjustment.ate")
    assert result.estimate.overlap_ess is None
    assert result.validation.count >= 0


def test_analyze_propensity_weighting_recovers_ate_and_overlap():
    # Shared dual band with Rust IPW ATE≈2 (see module docstring).
    data, edges = _confounded_scm(n=800, seed=5)
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        identifier="backdoor.adjustment",
        estimator="propensity.weighting",
        bootstrap=10,
        seed=1,
    )
    assert abs(result.ate - 2.0) < 0.4, result.ate
    assert result.estimate.estimator_id == "propensity.weighting"
    assert result.estimate.overlap_ess is not None
    assert result.estimate.overlap_propensity_min is not None
    assert result.validation.count == 0


@pytest.mark.parametrize(
    "estimator",
    [
        "propensity.stratification",
        "aipw",
        "propensity.matching",
        "distance.matching",
    ],
)
def test_analyze_estimate_estimators_smoke(estimator):
    data, edges = _confounded_scm(n=1000, seed=9)
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        estimator=estimator,
        bootstrap=5,
        seed=1,
        refute=False,
    )
    assert np.isfinite(result.ate)


def test_analyze_iv_2sls_smoke():
    rng = random.Random(2)
    n = 800
    z = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float64)
    u = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float64)
    t = (0.8 * z + 0.5 * u + np.array([rng.gauss(0, 0.3) for _ in range(n)]) > 0).astype(
        np.float64
    )
    y = 1.5 * t + u + np.array([rng.gauss(0, 0.3) for _ in range(n)], dtype=np.float64)
    result = causal.analyze(
        {"t": t, "y": y, "z": z},
        graph=[("z", "t"), ("t", "y")],
        query=causal.AverageEffect(treatment="t", outcome="y"),
        identifier="iv",
        estimator="iv.2sls",
        bootstrap=5,
        seed=1,
        refute=False,
    )
    assert np.isfinite(result.ate)


def test_analyze_frontdoor_smoke():
    rng = random.Random(4)
    n = 900
    t = np.array([1.0 if rng.random() < 0.5 else 0.0 for _ in range(n)], dtype=np.float64)
    m = t + np.array([rng.gauss(0, 0.4) for _ in range(n)], dtype=np.float64)
    y = m + np.array([rng.gauss(0, 0.4) for _ in range(n)], dtype=np.float64)
    result = causal.analyze(
        {"t": t, "m": m, "y": y},
        graph=[("t", "m"), ("m", "y")],
        query=causal.AverageEffect(treatment="t", outcome="y"),
        identifier="frontdoor",
        estimator="frontdoor.two_stage",
        bootstrap=5,
        seed=1,
        refute=False,
    )
    assert np.isfinite(result.ate)
