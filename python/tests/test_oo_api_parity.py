"""OO API parity smoke tests (graphs, queries, state, discovery configs)."""

from __future__ import annotations

import numpy as np
import pytest

import causal


def _ate_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 0.5 * t + 0.8 * z + rng.normal(size=n)
    return {"z": z, "t": t, "y": y}


def test_dag_from_edges_parents_children():
    g = causal.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    assert set(g.nodes()) == {"z", "t", "y"}
    assert ("z", "t") in g.edges()
    assert g.parents("y") == ["z", "t"] or set(g.parents("y")) == {"z", "t"}
    assert "t" in g.children("z")


def test_analyze_with_dag_object_and_levels():
    data = _ate_data()
    g = causal.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    r0 = causal.analyze(
        data,
        graph=g,
        query=causal.AverageEffect("t", "y", control_level=0.0, active_level=1.0),
        refute=False,
        bootstrap=0,
    )
    r1 = causal.analyze(
        data,
        graph=g,
        query=causal.AverageEffect("t", "y", control_level=0.0, active_level=2.0),
        refute=False,
        bootstrap=0,
    )
    assert r0.ate != pytest.approx(r1.ate, abs=1e-9) or abs(r1.ate) > abs(r0.ate) * 1.2


def test_sustained_vs_pulse_policy_accepted():
    rng = np.random.default_rng(1)
    n = 120
    data = {
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
    }
    # a@1 → b@0
    edges = [("a", 1, "b", 0)]
    pulse = causal.analyze(
        data,
        graph=edges,
        query=causal.PulseEffect("a", "b", treatment_lag=1, horizon_steps=1),
        bootstrap=0,
    )
    sustained = causal.analyze(
        data,
        graph=edges,
        query=causal.SustainedEffect("a", "b", treatment_lag=1, horizon_steps=1),
        bootstrap=0,
    )
    assert np.isfinite(pulse.ate)
    assert np.isfinite(sustained.ate)


def test_temporal_rejects_bayesian_loudly():
    rng = np.random.default_rng(2)
    n = 80
    data = {"a": rng.normal(size=n), "b": rng.normal(size=n)}
    with pytest.raises(TypeError, match="Bayesian"):
        causal.analyze(
            data,
            graph=[("a", 1, "b", 0)],
            query=causal.PulseEffect("a", "b"),
            inference=causal.Bayesian(n_draws=10),
            bootstrap=0,
        )


def test_fitted_gcm_sample_do():
    data = _ate_data(n=100)
    names = list(data.keys())
    cols = [data[n] for n in names]
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    gcm = causal.fit_gcm(names, cols, edges)
    out = gcm.sample_do({"t": 1.0}, n=50, seed=3)
    assert out.n_draws == 50
    assert out.draws.shape[1] == 50


def test_causal_state_append_data():
    state = causal.CausalState(cache_bytes=1 << 20)
    v0 = state.version
    data = _ate_data(n=20)
    names = list(data.keys())
    cols = [data[n] for n in names]
    v1 = state.append_data(names, cols)
    assert v1 > v0
    assert state.version == v1


def test_exact_dag_posterior_tiny():
    rng = np.random.default_rng(4)
    n = 80
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(size=n) * 0.3
    data = {"x": x, "y": y}
    post = causal.discover_exact_dag_posterior(data)
    assert post.n_vars == 2
    assert post.n_graphs >= 1
    assert len(post.weights) == post.n_graphs


def test_discovery_result_alias_and_ges_config():
    assert causal.discovery.DiscoveryResult is causal.PcmciDiscoveryResult
    cfg = causal.GES(alpha=0.1)
    assert cfg.kind == "ges"


def test_path_specific_and_distribution_queries():
    # Discrete chain t → m → y (matches Rust end_to_end_path_specific fixture).
    t_vals: list[float] = []
    m_vals: list[float] = []
    y_vals: list[float] = []
    for t in (0.0, 1.0):
        for _ in range(50):
            t_vals.append(t)
            m_vals.append(t)
            y_vals.append(t)
    data = {
        "t": np.asarray(t_vals, dtype=np.float64),
        "m": np.asarray(m_vals, dtype=np.float64),
        "y": np.asarray(y_vals, dtype=np.float64),
    }
    edges = [("t", "m"), ("m", "y")]
    path = causal.analyze(
        data,
        graph=edges,
        query=causal.PathSpecificEffect("t", "y", path_nodes=["m"]),
        refute=False,
        bootstrap=0,
    )
    assert abs(path.ate - 1.0) < 0.1
    dist = causal.analyze(
        data,
        graph=edges,
        query=causal.InterventionalDistribution("y", interventions={"t": 1.0}),
        refute=False,
        bootstrap=0,
    )
    assert np.isfinite(dist.ate)


def test_extensibility_exported():
    assert hasattr(causal.extensibility, "CiBatchTest")
