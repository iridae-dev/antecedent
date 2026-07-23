"""AcceptedGraph session: estimate clicks never rediscover (backlog D)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


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
    return {"t": t, "y": y, "z": z}


def test_from_discovery_estimates_without_rediscover(monkeypatch):
    data = _confounded_scm()
    calls = {"n": 0}
    real = causal.discover_pc

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(causal, "discover_pc", spy)
    monkeypatch.setattr("causal.accepted_graph.discover_pc", spy)
    monkeypatch.setattr("causal.discovery.discover_pc", spy)

    result = causal.discover_pc(
        data, alpha=0.5, fdr=False, max_cond_size=0, seed=1
    )
    assert calls["n"] == 1
    accepted = causal.AcceptedGraph.from_discovery(result, algorithm_id="pc")
    assert accepted.version == 1
    assert accepted.algorithm_id == "pc"
    # PC may leave undirected marks — hold as Cpdag, or Dag when fully oriented.
    assert isinstance(accepted.graph, (causal.Dag, causal.Cpdag))

    # Estimate clicks use a reviewed/accepted DAG (spreadsheet: accept then click).
    # Spy still proves rediscovery does not run when knobs change.
    estimate_handle = causal.AcceptedGraph.from_graph(
        [("z", "t"), ("z", "y"), ("t", "y")],
        algorithm_id=accepted.algorithm_id,
        version=accepted.version,
    )
    q = causal.AverageEffect(treatment="t", outcome="y")
    first = estimate_handle.analyze(data, query=q, seed=1)
    second = estimate_handle.analyze(data, query=q, seed=1, bootstrap=0)
    assert calls["n"] == 1, "estimate clicks must not re-enter discover_pc"
    assert estimate_handle.version == 1
    assert math.isfinite(first.ate)
    assert math.isfinite(second.ate)
    assert abs(first.ate - 2.0) < 0.75
    assert first.identification.status == second.identification.status


def test_bootstrap_tweak_does_not_bump_version_or_rediscover(monkeypatch):
    data = _confounded_scm(seed=23)
    dag = causal.Dag.from_edges(
        ["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")]
    )
    accepted = causal.AcceptedGraph.from_graph(dag, algorithm_id=None)
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("discovery must not run on estimate knobs")

    monkeypatch.setattr("causal.accepted_graph.discover_pc", boom)
    monkeypatch.setattr(causal, "discover_pc", boom)

    q = causal.AverageEffect(treatment="t", outcome="y")
    a = accepted.analyze(data, query=q, seed=1, bootstrap=0)
    b = accepted.analyze(data, query=q, seed=1, bootstrap=10, refute=False)
    assert calls["n"] == 0
    assert accepted.version == 1
    assert math.isfinite(a.ate) and math.isfinite(b.ate)


def test_rediscover_bumps_version_and_calls_discovery(monkeypatch):
    data = _confounded_scm(seed=29)
    dag = causal.Dag.from_edges(
        ["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")]
    )
    accepted = causal.AcceptedGraph.from_graph(dag, algorithm_id="hand")
    calls = {"n": 0}
    real = causal.discover_pc

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("causal.accepted_graph.discover_pc", spy)

    refreshed = accepted.rediscover(
        data, causal.PC(alpha=0.5, fdr=False, max_cond_size=0), seed=1
    )
    assert calls["n"] == 1
    assert refreshed.version == accepted.version + 1
    assert refreshed.algorithm_id == "pc"
    assert accepted.version == 1  # original handle unchanged
    assert isinstance(refreshed.graph, (causal.Dag, causal.Cpdag))

def test_analyze_rejects_discovery_kwarg():
    data = _confounded_scm(n=200, seed=3)
    accepted = causal.AcceptedGraph.from_graph(
        [("z", "t"), ("z", "y"), ("t", "y")], algorithm_id=None
    )
    with pytest.raises(causal.CausalUnsupportedError, match="rejects discovery="):
        accepted.analyze(
            data,
            query=causal.AverageEffect(treatment="t", outcome="y"),
            discovery=causal.PC(),
        )


def test_json_roundtrip_preserves_version_and_edges():
    dag = causal.Dag.from_edges(
        ["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")]
    )
    accepted = causal.AcceptedGraph.from_graph(dag, algorithm_id="pc", version=3)
    restored = causal.AcceptedGraph.from_json(accepted.to_json())
    assert restored.version == 3
    assert restored.algorithm_id == "pc"
    assert set(restored.graph.edges()) == set(dag.edges())  # type: ignore[union-attr]


def test_prepare_on_accepted_graph():
    data = _confounded_scm(n=300, seed=41)
    accepted = causal.AcceptedGraph.from_graph(
        [("z", "t"), ("z", "y"), ("t", "y")], algorithm_id="supplied"
    )
    prepared = accepted.prepare(
        data, query=causal.AverageEffect(treatment="t", outcome="y"), seed=1
    )
    assert prepared is not None
    first = prepared.estimate(data, seed=1)
    second = prepared.refresh(data, seed=1)
    assert accepted.version == 1
    assert abs(first.ate - second.ate) < 1e-12
    assert abs(first.ate - 2.0) < 0.6


def _lag1_series(n: int = 300, seed: int = 9):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    x = np.sin(t * 0.01) + 0.05 * rng.normal(size=n)
    y = np.zeros(n, dtype=np.float64)
    y[1:] = 0.8 * x[:-1] + 0.05 * rng.normal(size=n - 1)
    return {"x": x, "y": y}


def test_temporal_accepted_graph_estimates_without_rediscover(monkeypatch):
    data = _lag1_series()
    calls = {"n": 0}
    real = causal.discover_pcmci

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(causal, "discover_pcmci", spy)
    monkeypatch.setattr("causal.accepted_graph.discover_pcmci", spy)
    monkeypatch.setattr("causal.discovery.discover_pcmci", spy)

    result = causal.discover_pcmci(data=data, max_lag=2, alpha=0.05, fdr=False, seed=9)
    assert calls["n"] == 1
    accepted = causal.AcceptedGraph.from_discovery(result, algorithm_id="pcmci")
    assert accepted.version == 1
    assert accepted.algorithm_id == "pcmci"
    assert isinstance(accepted.graph, causal.TemporalDag)

    # Estimate clicks on a known TemporalDag; spy proves rediscovery does not run.
    estimate_handle = causal.AcceptedGraph.from_graph(
        causal.TemporalDag.from_lagged_edges(
            ["x", "y"], [("x", 1, "y", 0)]
        ),
        algorithm_id=accepted.algorithm_id,
        version=accepted.version,
    )
    q = causal.PulseEffect(
        treatment="x", outcome="y", treatment_lag=1, horizon_steps=1
    )
    first = estimate_handle.analyze(data, query=q, seed=1, bootstrap=0, refute=False)
    second = estimate_handle.analyze(data, query=q, seed=1, bootstrap=10, refute=False)
    assert calls["n"] == 1, "temporal estimate clicks must not re-enter discover_pcmci"
    assert estimate_handle.version == 1
    assert math.isfinite(first.ate)
    assert math.isfinite(second.ate)


def test_temporal_rediscover_bumps_version(monkeypatch):
    data = _lag1_series(seed=11)
    tdag = causal.TemporalDag.from_lagged_edges(["x", "y"], [("x", 1, "y", 0)])
    accepted = causal.AcceptedGraph.from_graph(tdag, algorithm_id="hand")
    calls = {"n": 0}
    real = causal.discover_pcmci

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("causal.accepted_graph.discover_pcmci", spy)

    refreshed = accepted.rediscover(data, causal.PCMCI(max_lag=2, alpha=0.05, fdr=False), seed=1)
    assert calls["n"] == 1
    assert refreshed.version == accepted.version + 1
    assert refreshed.algorithm_id == "pcmci"
    assert isinstance(refreshed.graph, causal.TemporalDag)


def test_temporal_json_roundtrip():
    tdag = causal.TemporalDag.from_lagged_edges(
        ["pressure", "defect"], [("pressure", 1, "defect", 0)]
    )
    accepted = causal.AcceptedGraph.from_graph(tdag, algorithm_id="pcmci", version=2)
    restored = causal.AcceptedGraph.from_json(accepted.to_json())
    assert restored.version == 2
    assert restored.algorithm_id == "pcmci"
    assert isinstance(restored.graph, causal.TemporalDag)
    assert set(restored.graph.edges()) == set(tdag.edges())
