"""AcceptedGraph session: estimate clicks never rediscover (backlog D)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _patch_discovery(monkeypatch, name, fn):
    """Patch a dispatcher at every name that resolves it.

    ``accepted_graph`` does ``from .discovery import run_static_discovery`` at
    import time, so patching only ``antecedent.discovery.<name>`` leaves
    ``rediscover``'s call untouched — and a "discovery must not run" assertion
    against that patch passes vacuously. Patch both bindings.
    """
    monkeypatch.setattr(f"antecedent.discovery.{name}", fn)
    monkeypatch.setattr(f"antecedent.accepted_graph.{name}", fn)


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
    real = antecedent.discovery.run_static_discovery

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    # `discover_pc` (a free function) is gone; `run_static_discovery` is the
    # stable dispatch point `AcceptedGraph.rediscover` itself calls, and the
    # one this test actually needs to prove is never re-entered.
    _patch_discovery(monkeypatch, "run_static_discovery", spy)

    result, algo = antecedent.discovery.run_static_discovery(
        data, antecedent.discovery.PC(alpha=0.5, fdr=False, max_cond_size=0), seed=1
    )
    assert calls["n"] == 1
    assert algo == "pc"
    accepted = antecedent.AcceptedGraph.from_discovery(result, algorithm_id=algo)
    assert accepted.version == 1
    assert accepted.algorithm_id == "pc"
    # PC may leave undirected marks — hold as Cpdag, or Dag when fully oriented.
    assert isinstance(accepted.graph, (antecedent.Dag, antecedent.Cpdag))

    # Estimate clicks use a reviewed/accepted DAG (spreadsheet: accept then click).
    # Spy still proves rediscovery does not run when knobs change.
    estimate_handle = antecedent.AcceptedGraph.from_graph(
        [("z", "t"), ("z", "y"), ("t", "y")],
        algorithm_id=accepted.algorithm_id,
        version=accepted.version,
    )
    q = antecedent.AverageEffect(treatment="t", outcome="y")
    first = estimate_handle.analyze(data, query=q, seed=1)
    second = estimate_handle.analyze(data, query=q, seed=1, bootstrap=0)
    assert calls["n"] == 1, "estimate clicks must not re-enter discovery"
    assert estimate_handle.version == 1
    assert math.isfinite(first.ate)
    assert math.isfinite(second.ate)
    assert abs(first.ate - 2.0) < 0.75
    assert first.identification.status == second.identification.status


def test_bootstrap_tweak_does_not_bump_version_or_rediscover(monkeypatch):
    data = _confounded_scm(seed=23)
    dag = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    accepted = antecedent.AcceptedGraph.from_graph(dag, algorithm_id=None)
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("discovery must not run on estimate knobs")

    _patch_discovery(monkeypatch, "run_static_discovery", boom)

    q = antecedent.AverageEffect(treatment="t", outcome="y")
    a = accepted.analyze(data, query=q, seed=1, bootstrap=0)
    b = accepted.analyze(data, query=q, seed=1, bootstrap=10, refute=False)
    assert calls["n"] == 0
    assert accepted.version == 1
    assert math.isfinite(a.ate) and math.isfinite(b.ate)


def test_rediscover_bumps_version_and_calls_discovery(monkeypatch):
    data = _confounded_scm(seed=29)
    dag = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    accepted = antecedent.AcceptedGraph.from_graph(dag, algorithm_id="hand")
    calls = {"n": 0}
    real = antecedent.discovery.run_static_discovery

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _patch_discovery(monkeypatch, "run_static_discovery", spy)

    refreshed = accepted.rediscover(
        data, antecedent.discovery.PC(alpha=0.5, fdr=False, max_cond_size=0), seed=1
    )
    assert calls["n"] == 1
    assert refreshed.version == accepted.version + 1
    assert refreshed.algorithm_id == "pc"
    assert accepted.version == 1  # original handle unchanged
    assert isinstance(refreshed.graph, (antecedent.Dag, antecedent.Cpdag))


def test_analyze_rejects_discovery_kwarg():
    data = _confounded_scm(n=200, seed=3)
    accepted = antecedent.AcceptedGraph.from_graph(
        [("z", "t"), ("z", "y"), ("t", "y")], algorithm_id=None
    )
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="rejects discovery="):
        accepted.analyze(
            data,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            discovery=antecedent.discovery.PC(),
        )


def test_json_roundtrip_preserves_version_and_edges():
    dag = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    accepted = antecedent.AcceptedGraph.from_graph(dag, algorithm_id="pc", version=3)
    restored = antecedent.AcceptedGraph.from_json(accepted.to_json())
    assert restored.version == 3
    assert restored.algorithm_id == "pc"
    assert set(restored.graph.edges()) == set(dag.edges())  # type: ignore[union-attr]


def test_json_roundtrip_preserves_node_names_cpdag():
    """Round-trip through Cpdag.to_json()/from_json() must keep real variable names.

    This currently depends on a native fix (owned elsewhere in this refactor):
    `Cpdag.from_json` / `Pag.from_json` / `Admg.from_json` discard variable
    names, so `AcceptedGraph.from_json(ag.to_json())` today comes back with
    placeholder nodes ("0", "1", "2", ...) for these three graph kinds. This
    test asserts the *intended* behavior and will fail until that native fix
    lands — see the task brief for this refactor phase.
    """
    cpdag = antecedent.Cpdag.from_directed_undirected(
        ["z", "t", "y"], [("z", "t"), ("z", "y")], [("t", "y")]
    )
    accepted = antecedent.AcceptedGraph.from_graph(cpdag, algorithm_id="pc")
    restored = antecedent.AcceptedGraph.from_json(accepted.to_json())
    assert isinstance(restored.graph, antecedent.Cpdag)
    assert set(restored.graph.nodes()) == {"z", "t", "y"}


def test_prepare_on_accepted_graph():
    data = _confounded_scm(n=300, seed=41)
    accepted = antecedent.AcceptedGraph.from_graph(
        [("z", "t"), ("z", "y"), ("t", "y")], algorithm_id="supplied"
    )
    prepared = accepted.prepare(
        data, query=antecedent.AverageEffect(treatment="t", outcome="y"), seed=1
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
    real = antecedent.discovery.run_temporal_discovery

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _patch_discovery(monkeypatch, "run_temporal_discovery", spy)

    result, algo = antecedent.discovery.run_temporal_discovery(
        data, antecedent.discovery.PCMCI(max_lag=2, alpha=0.05, fdr=False), seed=9
    )
    assert calls["n"] == 1
    assert algo == "pcmci"
    accepted = antecedent.AcceptedGraph.from_discovery(result, algorithm_id=algo)
    assert accepted.version == 1
    assert accepted.algorithm_id == "pcmci"
    assert isinstance(accepted.graph, antecedent.TemporalDag)

    # Estimate clicks on a known TemporalDag; spy proves rediscovery does not run.
    estimate_handle = antecedent.AcceptedGraph.from_graph(
        antecedent.TemporalDag.from_lagged_edges(["x", "y"], [("x", 1, "y", 0)]),
        algorithm_id=accepted.algorithm_id,
        version=accepted.version,
    )
    q = antecedent.PulseEffect(treatment="x", outcome="y", treatment_lag=1, horizon_steps=1)
    first = estimate_handle.analyze(data, query=q, seed=1, bootstrap=0, refute=False)
    second = estimate_handle.analyze(data, query=q, seed=1, bootstrap=10, refute=False)
    assert calls["n"] == 1, "temporal estimate clicks must not re-enter discovery"
    assert estimate_handle.version == 1
    assert math.isfinite(first.ate)
    assert math.isfinite(second.ate)


def test_temporal_rediscover_bumps_version(monkeypatch):
    data = _lag1_series(seed=11)
    tdag = antecedent.TemporalDag.from_lagged_edges(["x", "y"], [("x", 1, "y", 0)])
    accepted = antecedent.AcceptedGraph.from_graph(tdag, algorithm_id="hand")
    calls = {"n": 0}
    real = antecedent.discovery.run_temporal_discovery

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _patch_discovery(monkeypatch, "run_temporal_discovery", spy)

    refreshed = accepted.rediscover(
        data, antecedent.discovery.PCMCI(max_lag=2, alpha=0.05, fdr=False), seed=1
    )
    assert calls["n"] == 1
    assert refreshed.version == accepted.version + 1
    assert refreshed.algorithm_id == "pcmci"
    assert isinstance(refreshed.graph, antecedent.TemporalDag)


def test_temporal_json_roundtrip():
    tdag = antecedent.TemporalDag.from_lagged_edges(
        ["pressure", "defect"], [("pressure", 1, "defect", 0)]
    )
    accepted = antecedent.AcceptedGraph.from_graph(tdag, algorithm_id="pcmci", version=2)
    restored = antecedent.AcceptedGraph.from_json(accepted.to_json())
    assert restored.version == 2
    assert restored.algorithm_id == "pcmci"
    assert isinstance(restored.graph, antecedent.TemporalDag)
    assert set(restored.graph.edges()) == set(tdag.edges())


# --- asserted / accepted: documented spellings of from_graph / from_discovery ---


def test_asserted_is_from_graph_alias():
    dag = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    handle = antecedent.AcceptedGraph.asserted(dag, algorithm_id="hand")
    assert handle.algorithm_id == "hand"
    assert handle.version == 1
    assert isinstance(handle.graph, antecedent.Dag)


def test_accepted_is_from_discovery_alias():
    result, algo = antecedent.discovery.run_static_discovery(
        _confounded_scm(seed=5), antecedent.discovery.PC(alpha=0.5, fdr=False), seed=1
    )
    via_accepted = antecedent.AcceptedGraph.accepted(result, algorithm_id=algo)
    via_from_discovery = antecedent.AcceptedGraph.from_discovery(result, algorithm_id=algo)
    assert via_accepted.algorithm_id == via_from_discovery.algorithm_id == algo


# --- .pending / .review() ---


def test_cpdag_pending_and_review_orients_edge():
    cpdag = antecedent.Cpdag.from_directed_undirected(
        ["z", "t", "y"], [("z", "t"), ("z", "y")], [("t", "y")]
    )
    accepted = antecedent.AcceptedGraph.from_graph(cpdag, algorithm_id="pc")
    pending = accepted.pending
    assert len(pending) == 1
    edge = pending[0]
    assert {edge.source, edge.target} == {"t", "y"}
    assert edge.at_source == "tail"
    assert edge.at_target == "tail"

    reviewed = accepted.review({(edge.source, edge.target): ("tail", "arrow")})
    assert reviewed.version == accepted.version + 1
    assert accepted.version == 1  # original handle unchanged
    assert reviewed.pending == ()
    # Fully resolved Cpdag collapses to a Dag.
    assert isinstance(reviewed.graph, antecedent.Dag)
    assert (edge.source, edge.target) in set(reviewed.graph.edges())


def test_review_rejects_non_pending_edge():
    dag = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    accepted = antecedent.AcceptedGraph.from_graph(dag, algorithm_id="hand")
    assert accepted.pending == ()
    with pytest.raises(ValueError, match="not a pending edge"):
        accepted.review({("t", "y"): ("tail", "arrow")})


def test_review_rejects_cycle():
    # z - t, z - y, t - y all undirected: orienting z->t, t->y, y->z is a cycle.
    cpdag = antecedent.Cpdag.from_directed_undirected(
        ["z", "t", "y"], [], [("z", "t"), ("t", "y"), ("y", "z")]
    )
    accepted = antecedent.AcceptedGraph.from_graph(cpdag, algorithm_id="hand")
    assert len(accepted.pending) == 3
    with pytest.raises(ValueError, match="cycle"):
        accepted.review(
            {
                ("z", "t"): ("tail", "arrow"),
                ("t", "y"): ("tail", "arrow"),
                ("y", "z"): ("tail", "arrow"),
            }
        )


def test_pag_pending_marks_circle_edges():
    pag = antecedent.Pag.from_marked_edges(
        ["x", "y", "z"],
        [
            ("x", "y", "circle", "arrow"),
            ("y", "z", "tail", "arrow"),
        ],
    )
    accepted = antecedent.AcceptedGraph.from_graph(pag, algorithm_id="fci")
    pending = accepted.pending
    assert len(pending) == 1
    assert (pending[0].source, pending[0].target) == ("x", "y")
    assert pending[0].at_source == "circle"
    assert pending[0].at_target == "arrow"

    reviewed = accepted.review({("x", "y"): ("tail", "arrow")})
    assert reviewed.pending == ()
    assert isinstance(reviewed.graph, antecedent.Pag)


# --- dunders ---


def test_dunders_on_dag():
    dag = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    accepted = antecedent.AcceptedGraph.from_graph(dag, algorithm_id="hand", version=2)
    assert len(accepted) == 3
    assert set(iter(accepted)) == {("z", "t"), ("z", "y"), ("t", "y")}
    assert ("z", "t") in accepted
    assert "z" in accepted
    assert "q" not in accepted
    assert ("t", "z") not in accepted  # direction matters for containment via edges
    text = repr(accepted)
    assert "Dag" in text
    assert "nodes=3" in text
    assert "version=2" in text
    assert "pending=0" in text
