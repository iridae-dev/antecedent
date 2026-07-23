"""Prepared analysis re-estimate (Python dual of Rust backlog B)."""

from __future__ import annotations

import math
import random
import time

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def _confounded_scm(n: int = 500, seed: int = 19):
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
    return {"t": t, "y": y, "z": z}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_prepared_reestimate_matches_fresh_analyze():
    data, edges = _confounded_scm()
    fresh = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    prepared = causal.PreparedAnalysis.prepare(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    first = prepared.estimate(data, seed=1)
    second = prepared.refresh(data, seed=1)
    assert math.isfinite(first.ate)
    assert abs(first.ate - 2.0) < 0.5
    assert abs(first.ate - fresh.ate) < 1e-12
    assert abs(second.ate - fresh.ate) < 1e-12
    assert first.identification.status == fresh.identification.status
    assert first.identification.adjustment_set == fresh.identification.adjustment_set
    assert first.performance.plan_id == fresh.performance.plan_id
    # Result.refresh uses the retained prepared handle.
    via_result = first.refresh(data, seed=1)
    assert abs(via_result.ate - fresh.ate) < 1e-12


def test_oneshot_analyze_result_cannot_refresh():
    data, edges = _confounded_scm(n=200, seed=5)
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    with pytest.raises(TypeError, match="PreparedAnalysis"):
        result.refresh(data)


def test_prepared_second_shot_not_slower_than_prepare_plus_first():
    data, edges = _confounded_scm(n=800, seed=31)
    t0 = time.perf_counter()
    prepared = causal.PreparedAnalysis.prepare(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    _ = prepared.estimate(data, seed=1)
    prepare_plus_first = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = prepared.estimate(data, seed=1)
    second = time.perf_counter() - t1
    assert second <= prepare_plus_first * 2.0 + 0.05
