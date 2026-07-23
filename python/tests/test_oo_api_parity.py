"""OO API parity smoke tests (graphs, queries, state, discovery configs)."""

from __future__ import annotations

import numpy as np
import pytest

import antecedent


def _ate_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 0.5 * t + 0.8 * z + rng.normal(size=n)
    return {"z": z, "t": t, "y": y}


def test_dag_from_edges_parents_children():
    g = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    assert set(g.nodes()) == {"z", "t", "y"}
    assert ("z", "t") in g.edges()
    assert g.parents("y") == ["z", "t"] or set(g.parents("y")) == {"z", "t"}
    assert "t" in g.children("z")


def test_analyze_with_dag_object_and_levels():
    data = _ate_data()
    g = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    r0 = antecedent.analyze(
        data,
        graph=g,
        query=antecedent.AverageEffect("t", "y", control_level=0.0, active_level=1.0),
        refute=False,
        bootstrap=0,
    )
    r1 = antecedent.analyze(
        data,
        graph=g,
        query=antecedent.AverageEffect("t", "y", control_level=0.0, active_level=2.0),
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
    pulse = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.PulseEffect("a", "b", treatment_lag=1, horizon_steps=1),
        bootstrap=0,
        refute=False,
    )
    sustained = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.SustainedEffect("a", "b", treatment_lag=1, horizon_steps=1),
        bootstrap=0,
        refute=False,
    )
    assert np.isfinite(pulse.ate)
    assert np.isfinite(sustained.ate)
    assert pulse.validation.passed is False
    assert pulse.validation.ran is False
    assert pulse.validation.count == 0


def test_temporal_refute_runs_by_default():
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = 0.8 * x[t - 1]
    result = antecedent.analyze(
        {"a": x, "b": y},
        graph=[("a", 1, "b", 0)],
        query=antecedent.PulseEffect("a", "b", treatment_lag=1, horizon_steps=1),
        refute=True,
        bootstrap=0,
        seed=1,
    )
    assert result.validation.ran
    assert result.validation.count >= 1


def test_temporal_accepts_bayesian_pulse():
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = 0.8 * x[t - 1]
    result = antecedent.analyze(
        {"a": x, "b": y},
        graph=[("a", 1, "b", 0)],
        query=antecedent.PulseEffect("a", "b", treatment_lag=1, horizon_steps=1),
        inference=antecedent.Bayesian(n_draws=64),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert result.posterior is not None
    assert np.isfinite(result.posterior.p_below_zero)
    assert abs(result.posterior.effect_mean - 0.8) < 0.15
    assert result.estimate.estimator_id == "bayesian.temporal.gcomp"


def test_fitted_gcm_sample_do():
    data = _ate_data(n=100)
    names = list(data.keys())
    cols = [data[n] for n in names]
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    gcm = antecedent.fit_gcm(names, cols, edges)
    out = gcm.sample_do({"t": 1.0}, n=50, seed=3)
    assert out.n_draws == 50
    assert out.draws.shape[1] == 50


def test_causal_state_append_data():
    state = antecedent.CausalState(cache_bytes=1 << 20)
    v0 = state.version
    data = _ate_data(n=20)
    names = list(data.keys())
    cols = [data[n] for n in names]
    v1 = state.append_data(names, cols)
    assert v1 > v0
    assert state.version == v1
    ids = state.batch_ids()
    assert len(ids) == 1
    got_names, got_cols = state.get_batch(ids[0])
    assert got_names == names
    assert len(got_cols) == len(cols)
    assert state.batch_nrows(ids[0]) == 20

    state.ols_ensure("m1", 2)
    state.ols_append_row("m1", [1.0, 0.5], 1.2)
    ols = state.ols_get("m1")
    assert ols["n"] == 1
    assert ols["ncols"] == 2

    state.cov_ensure("c1", 2)
    state.cov_update("c1", [0.1, 0.2])
    cov = state.cov_get("c1")
    assert cov["n"] == 1

    state.particle_filter_init("pf", 32, seed=2)
    state.particle_filter_step("pf", 0.0)
    pf = state.particle_filter_get("pf")
    assert pf["n_obs"] == 1
    assert pf["n_particles"] == 32

    ver, qid = state.register_average_effect(0, 1)
    assert ver >= state.version - 1
    state.refresh_results([(qid, 1, 8)])
    assert state.stale_query_count() == 0

    v_rep = state.replace_data(names, cols)
    assert v_rep > v1
    assert len(state.batch_ids()) == 1
    # Replace invalidates registered query results until explicit refresh (ADR 0016).
    assert state.stale_query_count() >= 1
    state.refresh_results([(qid, 1, 8)])
    assert state.stale_query_count() == 0


def test_causal_state_ols_append_matches_full_recompute():
    """Python dual of Rust incremental_ols_match: append batches ≡ full XtX/XtY."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(30, 2))
    x[:, 0] = 1.0
    beta_true = np.array([0.5, -1.25])
    y = x @ beta_true + rng.normal(size=30) * 0.05

    state = antecedent.CausalState(cache_bytes=1 << 20)
    state.ols_ensure("ols", 2)
    # Two append batches (online path).
    for i in range(0, 12):
        state.ols_append_row("ols", [float(x[i, 0]), float(x[i, 1])], float(y[i]))
    for i in range(12, 30):
        state.ols_append_row("ols", [float(x[i, 0]), float(x[i, 1])], float(y[i]))
    inc = state.ols_get("ols")
    assert inc["n"] == 30
    xtx = np.asarray(inc["xtx"], dtype=np.float64).reshape(2, 2)
    xty = np.asarray(inc["xty"], dtype=np.float64)
    beta_inc = np.linalg.solve(xtx, xty)

    xtx_full = x.T @ x
    xty_full = x.T @ y
    beta_full = np.linalg.solve(xtx_full, xty_full)
    assert np.allclose(xtx, xtx_full, rtol=0, atol=1e-10)
    assert np.allclose(xty, xty_full, rtol=0, atol=1e-10)
    assert np.allclose(beta_inc, beta_full, rtol=0, atol=1e-9)

    _, qid = state.register_average_effect(0, 1)
    state.refresh_results([(qid, 2, 16)])
    assert state.stale_query_count() == 0
    # Append data must not auto-refresh results.
    state.append_data(["t", "y"], [y[:5], y[5:10]])
    assert state.stale_query_count() >= 1


def test_rank_designs_full_surface():
    ranking = antecedent.rank_designs(
        [0.5, 0.3, 0.2],
        [1, 0, 0],
        [10, 20, 30],
        [
            {"kind": "measure", "variables": [3], "tag": 1},
            {"kind": "observe_environment", "environment": 7, "additional_rows": 50},
            {"kind": "increase_sampling_rate", "additional_samples": 10},
            {"kind": "intervene", "targets": [0]},
        ],
        objective="increase_identification_probability",
        query_id=0,
        query_id_unlock=[(0, [3])],
        env_id_unlock=[(0, [7])],
        min_batches=2,
        max_batches=4,
        batch_size=4,
        rank_uncertainty_threshold=1.0,
        seed=3,
    )
    assert ranking.mc_samples > 0
    assert len(ranking.ranked) == 4
    assert ranking.best_index in {r.candidate_index for r in ranking.ranked}
    kinds = {r.kind for r in ranking.ranked}
    assert "measure" in kinds
    assert "observe_environment" in kinds


def test_exact_dag_posterior_tiny():
    rng = np.random.default_rng(4)
    n = 80
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(size=n) * 0.3
    data = {"x": x, "y": y}
    post = antecedent.discover_exact_dag_posterior(data)
    assert post.n_vars == 2
    assert post.n_graphs >= 1
    assert len(post.weights) == post.n_graphs


def test_discovery_result_alias_and_ges_config():
    assert antecedent.discovery.DiscoveryResult is antecedent.PcmciDiscoveryResult
    cfg = antecedent.GES(alpha=0.1)
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
    path = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.PathSpecificEffect("t", "y", path_nodes=["m"]),
        refute=False,
        bootstrap=0,
    )
    assert abs(path.ate - 1.0) < 0.1
    dist = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.InterventionalDistribution("y", interventions={"t": 1.0}),
        refute=False,
        bootstrap=0,
    )
    assert np.isfinite(dist.ate)


def test_extensibility_exported():
    assert hasattr(antecedent.extensibility, "CiBatchTest")
