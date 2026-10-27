"""OO API parity smoke tests (graphs, queries, state, discovery configs)."""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent import _native, design
from antecedent.errors import CausalStateError, CausalValueError
from antecedent.execution import CachePolicy, Execution
from antecedent.outcomes import require_success


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
    r0 = require_success(
        antecedent.analyze(
            data,
            graph=g,
            query=antecedent.AverageEffect("t", "y", control_level=0.0, active_level=1.0),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
        )
    )
    r1 = require_success(
        antecedent.analyze(
            data,
            graph=g,
            query=antecedent.AverageEffect("t", "y", control_level=0.0, active_level=2.0),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
        )
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
    pulse = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.PulseEffect("a", "b", treatment_lag=1, horizon_steps=1),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
        )
    )
    # Licensed SustainedEffect is the single-step window at -treatment_lag,
    # so it matches Pulse at the same lag under temporal linear adjustment.
    sustained = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.SustainedEffect("a", "b", treatment_lag=1, horizon_steps=1),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
        )
    )
    assert np.isfinite(pulse.ate)
    assert np.isfinite(sustained.ate)
    assert sustained.ate == pytest.approx(pulse.ate, abs=1e-10)
    assert pulse.validation.passed is False
    assert pulse.validation.ran is False
    assert pulse.validation.count == 0
    assert sustained.validation.passed is False
    assert sustained.validation.ran is False
    assert sustained.validation.count == 0


def test_temporal_refute_runs_by_default():
    # `refute=True` is now a TypeError (it carried no information beyond
    # "unset" and was easy to confuse with an explicit choice) — leaving
    # `refute` unspecified is the current spelling of "run the default suite",
    # which is exactly the behavior this test pins.
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = 0.8 * x[t - 1]
    result = require_success(
        antecedent.analyze(
            {"a": x, "b": y},
            graph=[("a", 1, "b", 0)],
            query=antecedent.PulseEffect("a", "b", treatment_lag=1, horizon_steps=1),
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.validation.ran
    assert result.validation.count >= 1


def test_refute_true_is_rejected():
    rng = np.random.default_rng(3)
    n = 60
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = 0.8 * x[t - 1]
    with pytest.raises(TypeError, match="refute=True is ambiguous"):
        require_success(
            antecedent.analyze(
                {"a": x, "b": y},
                graph=[("a", 1, "b", 0)],
                query=antecedent.PulseEffect("a", "b", treatment_lag=1, horizon_steps=1),
                refute=True,
                inference=antecedent.Frequentist(bootstrap=0),
                execution=antecedent.Execution(seed=1),
            )
        )


def test_temporal_accepts_bayesian_pulse():
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = 0.8 * x[t - 1]
    result = require_success(
        antecedent.analyze(
            {"a": x, "b": y},
            graph=[("a", 1, "b", 0)],
            query=antecedent.PulseEffect("a", "b", treatment_lag=1, horizon_steps=1),
            inference=antecedent.Bayesian(n_draws=64),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
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
    gcm = antecedent.model.fit_gcm(names, cols, edges)
    out = gcm.sample_do({"t": 1.0}, n=50, execution=antecedent.Execution(seed=3))
    assert out.n_draws == 50
    assert out.draws.shape[1] == 50


def test_antecedent_state_append_data():
    state = antecedent.state.CausalState(execution=Execution(cache=CachePolicy.bounded(1 << 20)))
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

    state.particle_filter_init("pf", 32, execution=Execution(seed=2))
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


def test_antecedent_state_ols_append_matches_full_recompute():
    """Python dual of Rust incremental_ols_match: append batches ≡ full XtX/XtY."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(30, 2))
    x[:, 0] = 1.0
    beta_true = np.array([0.5, -1.25])
    y = x @ beta_true + rng.normal(size=30) * 0.05

    state = antecedent.state.CausalState(execution=Execution(cache=CachePolicy.bounded(1 << 20)))
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


def test_antecedent_state_rejected_replace_preserves_existing_data():
    state = antecedent.state.CausalState(execution=Execution(cache=CachePolicy.bounded(1 << 20)))
    state.append_data(
        ["t", "y"],
        [np.array([0.0, 1.0]), np.array([1.0, 3.0])],
    )
    before_version = state.version
    before_ids = state.batch_ids()
    before_names, before_columns = state.get_batch(before_ids[0])

    with pytest.raises(ValueError, match="names/columns length mismatch"):
        state.replace_data(
            ["t", "y"],
            [np.array([2.0, 3.0])],
        )

    assert state.version == before_version
    assert state.batch_ids() == before_ids
    after_names, after_columns = state.get_batch(before_ids[0])
    assert after_names == before_names
    assert all(
        np.array_equal(after, before)
        for after, before in zip(after_columns, before_columns, strict=True)
    )

    with pytest.raises(ValueError, match="both names and columns"):
        state.replace_data(names=["t", "y"])
    assert state.version == before_version
    assert state.batch_ids() == before_ids


def test_rank_designs_full_surface():
    ranking = design.rank_designs(
        [
            design.Measurement([3], tag=1),
            design.Environment(7, additional_rows=50),
            design.Sampling(10),
            design.Experiment([0]),
        ],
        prior=design.StructurePrior(
            weights=(0.5, 0.3, 0.2), identified=(True, False, False), keys=(10, 20, 30)
        ),
        query_id=0,
        variable_unlocks={0: [3]},
        environment_unlocks={0: [7]},
        monte_carlo=design.MonteCarlo(2, 4, 4, 1.0),
        rng_seed=3,
    )
    assert ranking.basis == "identification"
    assert ranking.mc_samples > 0
    assert len(ranking.candidates) == 4
    assert ranking.best is not None and ranking.best.rank == 0  # ranks are zero-based
    assert [c.rank for c in ranking.candidates] == [0, 1, 2, 3]
    kinds = {c.kind for c in ranking.candidates}
    assert "measure" in kinds
    assert "observe_environment" in kinds
    assert "basis='identification'" in repr(ranking)
    assert ranking.explain().startswith("Basis: identification.")
    assert ranking.to_dict()["basis"] == "identification"
    with pytest.raises(antecedent.errors.CausalValueError, match="decision ranking"):
        bytes(ranking.export())


def test_native_rank_designs_decision_regret_is_conjugate_normal_evsi():
    # The native legacy entry point (dict plans, decision-regret objective) is still exercised
    # directly; the typed surface is `antecedent.design.rank_designs`.
    ranking = _native.rank_designs(
        [1.0],
        [1],
        [1],
        [
            {"kind": "increase_sampling_rate", "additional_samples": 4},
            {"kind": "increase_sampling_rate", "additional_samples": 64},
            {"kind": "measure", "variables": [0]},
        ],
        objective="reduce_decision_regret",
        decision_id=0,
        decision={
            "utility": {"intercepts": [1.0, -0.5], "slopes": [0.2, 1.1]},
            "prior": {"kind": "normal", "mean": 1.0, "variance": 2.0},
            "signal": {"kind": "gaussian_mean", "noise_variance": 3.0},
        },
        min_batches=1,
        max_batches=1,
        batch_size=1,
        seed=1,
    )
    assert [r.candidate_index for r in ranking.ranked] == [1, 0]
    assert [(v.candidate_index, v.constraint) for v in ranking.violations] == [
        (2, "unlicensed_candidate")
    ]

    # EVSI(n) = |Δβ| s_n G(|μ_b − μ₀| / s_n), s_n² = τ⁴n / (nτ² + σ²),
    # G(u) = φ(u) − u (1 − Φ(u)); here Δβ = 0.9, μ_b = 1.5 / 0.9, μ₀ = 1.
    def evsi(n: float) -> float:
        s = math.sqrt(4.0 * n / (2.0 * n + 3.0))
        u = abs(1.5 / 0.9 - 1.0) / s
        g = math.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi) - u * 0.5 * math.erfc(
            u / math.sqrt(2.0)
        )
        return 0.9 * s * g

    for ranked in ranking.ranked:
        n = [4.0, 64.0][ranked.candidate_index]
        assert ranked.score == pytest.approx(evsi(n), abs=1e-13)
        assert ranked.evaluation == "exact"
        assert ranked.stderr == 0.0
        assert ranked.implemented_functional == "preposterior_expected_value_of_sample_information"


def test_native_rank_designs_decision_regret_callable_utility_on_draws():
    def bet(actions, outcomes):
        a = np.asarray(actions, dtype=np.float64)
        o = np.asarray(outcomes, dtype=np.float64)
        return np.outer(a, o - 0.45).ravel()

    draws = [0.2, 0.5, 0.8, 0.35]
    common = {
        "objective": "reduce_decision_regret",
        "decision_id": 0,
        "min_batches": 2,
        "max_batches": 2,
        "batch_size": 4,
        "seed": 5,
    }
    exact = _native.rank_designs(
        [1.0],
        [1],
        [1],
        [{"kind": "increase_sampling_rate", "additional_samples": 1}],
        decision={
            "actions": [0.0, 1.0],
            "utility": bet,
            "prior": {"kind": "draws", "draws": draws},
            "signal": {"kind": "binomial"},
        },
        **common,
    )
    # One coin flip, enumerated: E_y[max(0, E[θ − 0.45 | y])] − max(0, E[θ − 0.45]).
    heads = sum(draws) / 4.0
    after_heads = sum(t * (t - 0.45) for t in draws) / 4.0 / heads
    after_tails = sum((1.0 - t) * (t - 0.45) for t in draws) / 4.0 / (1.0 - heads)
    expected = (
        heads * max(after_heads, 0.0)
        + (1.0 - heads) * max(after_tails, 0.0)
        - max(heads - 0.45, 0.0)
    )
    assert exact.ranked[0].evaluation == "exact"
    assert exact.ranked[0].score == pytest.approx(expected, abs=1e-15)

    mc = _native.rank_designs(
        [1.0],
        [1],
        [1],
        [{"kind": "increase_sampling_rate", "additional_samples": 3}],
        decision={
            "actions": [0.0, 1.0],
            "utility": bet,
            "prior": {"kind": "draws", "draws": draws},
            "signal": {"kind": "gaussian_mean", "noise_variance": 0.2},
        },
        **common,
    )
    assert mc.ranked[0].evaluation == "monte_carlo"
    assert mc.ranked[0].stderr > 0.0


def test_native_rank_designs_decision_regret_requires_a_decision_model():
    with pytest.raises(antecedent.errors.CausalDesignError, match="decisions context"):
        _native.rank_designs(
            [1.0],
            [1],
            [1],
            [{"kind": "increase_sampling_rate", "additional_samples": 1}],
            objective="reduce_decision_regret",
            decision_id=0,
        )


def test_exact_dag_posterior_tiny():
    rng = np.random.default_rng(4)
    n = 80
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(size=n) * 0.3
    data = {"x": x, "y": y}
    post = antecedent.discovery.ExactDagPosterior().run(data)
    assert post.n_vars == 2
    assert post.n_graphs >= 1
    assert len(post.weights) == post.n_graphs


def test_discovery_result_alias_and_ges_config():
    assert antecedent.discovery.DiscoveryResult is antecedent.discovery.PcmciDiscoveryResult
    cfg = antecedent.discovery.GES(alpha=0.1)
    assert cfg.kind == "ges"


def test_path_specific_and_distribution_queries():
    # Discrete chain t → m → y (matches Rust end_to_end_path_specific fixture).
    t_vals: list[float] = []
    m_vals: list[float] = []
    y_vals: list[float] = []
    for t, m, y, count in (
        (0.0, 0.0, 0.0, 40),
        (0.0, 0.0, 1.0, 10),
        (0.0, 1.0, 0.0, 10),
        (0.0, 1.0, 1.0, 40),
        (1.0, 0.0, 0.0, 10),
        (1.0, 0.0, 1.0, 10),
        (1.0, 1.0, 0.0, 10),
        (1.0, 1.0, 1.0, 70),
    ):
        t_vals.extend([t] * count)
        m_vals.extend([m] * count)
        y_vals.extend([y] * count)
    data = {
        "t": np.asarray(t_vals, dtype=np.float64),
        "m": np.asarray(m_vals, dtype=np.float64),
        "y": np.asarray(y_vals, dtype=np.float64),
    }
    edges = [("t", "m"), ("m", "y")]
    path = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.PathSpecificEffect("t", "y", path_nodes=["m"]),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
        )
    )
    assert abs(path.ate - 0.3) < 1e-9
    dist = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.InterventionalDistribution("y", interventions={"t": 1.0}),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
        )
    )
    assert np.isfinite(dist.ate)


def test_extensibility_exported():
    assert hasattr(antecedent.extensibility, "CiBatchTest")


@pytest.mark.parametrize(
    "name,parameters,result",
    [
        (
            "joint_bayesian_transport",
            "sources target features graph graph_class dependence varying sharing draws identification priors treatment outcome max_unsupported_mass conflict_z_threshold level execution",
            "MeasuredInference",
        ),
        (
            "learned_joint_transport",
            "sources target features graph graph_class dependence varying sharing basis_degree draws identification priors treatment outcome max_unsupported_mass conflict_z_threshold level execution",
            "MeasuredInference",
        ),
        (
            "temporal_dependent_interval",
            "panel sequence estimand fixed_state target_law replicates level method min_units max_failed_fraction execution",
            "MeasuredInference",
        ),
        (
            "binary_nested_markov",
            "graph regimes max_iterations tolerance prior execution",
            "MeasuredInference",
        ),
        (
            "binary_nested_markov_fisher_interval",
            "graph regimes nominal_level max_iterations tolerance execution",
            "MeasuredInference",
        ),
        (
            "sampled_observation_recovery",
            "stage query rows snapshot replicates interval_method execution",
            "MeasuredInference",
        ),
        (
            "consume_joint_transport_posterior",
            "data kind expected_identity",
            "JointTransportPosterior",
        ),
    ],
)
def test_candidate_public_signature_contract(name, parameters, result):
    """Measured normal producers and the historical source consumer have explicit contracts."""
    import inspect

    from antecedent.transport import advanced

    signature = inspect.signature(getattr(advanced, name))
    assert tuple(signature.parameters) == tuple(parameters.split())
    assert signature.return_annotation == result
    assert all(
        p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()
    )


def test_checked_prior_bridge_signature_and_identity_defaults():
    import inspect

    signature = inspect.signature(design.adapt_prior_to_signal)
    assert tuple(signature.parameters) == (
        "catalog",
        "query",
        "sources",
        "signal",
        "state",
        "target_population",
        "prior_id",
        "variables",
        "transport_policy_id",
        "candidate_observation_ids",
        "resolution",
    )
    assert signature.return_annotation == "CheckedPriorSignal"
    assert signature.parameters["transport_policy_id"].default is None
    assert signature.parameters["candidate_observation_ids"].default == ()
    for required in ("sources", "state", "target_population", "prior_id"):
        assert signature.parameters[required].default is inspect.Parameter.empty


def test_graph_evidence_is_versioned_and_listed_in_the_order_it_was_added():
    state = antecedent.state.CausalState(execution=Execution(cache=CachePolicy.bounded(1 << 20)))
    assert state.graph_evidence() == []
    v0 = state.version
    v1 = state.add_graph_evidence("pc-run", 7, 128)
    assert v1 > v0 and state.version == v1
    assert state.graph_evidence() == [("pc-run", 7, 128)]
    v2 = state.add_graph_evidence("ges-run", 9, 64)
    assert v2 > v1
    assert state.graph_evidence() == [("pc-run", 7, 128), ("ges-run", 9, 64)]
    assert state.constraints() == [], "evidence is not a constraint"


def _chain_columns(n: int = 60) -> list[np.ndarray]:
    i = np.arange(n, dtype=np.float64)
    x0 = np.sin(0.37 * i) + 0.1 * i / n
    x1 = 2.0 * x0 + 0.3 * np.cos(1.9 * i)
    x2 = x1 - 0.5 * x0 + 0.25 * np.sin(3.1 * i + 0.4)
    return [x0, x1, x2]


def _gaussian_bic(columns: list[np.ndarray], node: int, parents: tuple[int, ...]) -> float:
    """-n/2 (1 + ln 2pi + ln sigma^2) - (k/2) ln n, sigma^2 = SSE/n of OLS on intercept+parents."""
    y = columns[node]
    n = len(y)
    design = np.column_stack([np.ones(n), *[columns[p] for p in parents]])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    sigma2 = float(np.sum((y - design @ beta) ** 2)) / n
    return -0.5 * n * (1.0 + math.log(2.0 * math.pi) + math.log(sigma2)) - (
        0.5 * (len(parents) + 1) * math.log(n)
    )


def test_a_parent_set_change_rescores_one_node_and_matches_a_full_recomputation():
    columns = _chain_columns()
    cache = antecedent.state.GraphScoreCache(columns)
    assert (cache.n_rows, cache.n_vars) == (60, 3)
    empty = sum(_gaussian_bic(columns, j, ()) for j in range(3))
    assert cache.score() == pytest.approx(empty, rel=1e-9)

    delta, total = cache.set_parents(1, [0])
    expected = _gaussian_bic(columns, 1, (0,)) - _gaussian_bic(columns, 1, ())
    assert delta == pytest.approx(expected, rel=1e-7)
    assert delta > 0, "x0 explains x1"
    assert total == pytest.approx(empty + delta, rel=1e-9)
    assert cache.parents_of(1) == [0] and cache.parents_of(0) == []

    # Parents are stored sorted and deduplicated; the running total follows each change.
    delta2, total2 = cache.set_parents(2, [1, 0, 1])
    assert cache.parents_of(2) == [0, 1]
    assert total2 == pytest.approx(total + delta2, rel=1e-9)
    graph_score = (
        _gaussian_bic(columns, 0, ())
        + _gaussian_bic(columns, 1, (0,))
        + _gaussian_bic(columns, 2, (0, 1))
    )
    assert total2 == pytest.approx(graph_score, rel=1e-7)
    assert total2 == pytest.approx(cache.full_score({1: [0], 2: [0, 1]}), rel=1e-9)

    # Removing the parents again returns to the score before the change.
    delta3, total3 = cache.set_parents(2, [])
    assert delta3 == pytest.approx(-delta2, rel=1e-7)
    assert total3 == pytest.approx(total, rel=1e-9)
    assert cache.retained_bytes > 0
    cache.clear()
    assert cache.parents_of(1) == [] and cache.retained_bytes == 0


def test_a_graph_score_cache_refuses_self_parents_unknown_nodes_and_malformed_data():
    columns = _chain_columns()
    cache = antecedent.state.GraphScoreCache(columns)
    with pytest.raises(CausalStateError, match="own parent"):
        cache.set_parents(1, [1])
    with pytest.raises(CausalStateError, match="out of range"):
        cache.set_parents(7, [0])
    assert cache.parents_of(1) == [], "a refused change leaves the parent sets alone"
    with pytest.raises(CausalValueError, match="same number of rows"):
        antecedent.state.GraphScoreCache([columns[0], columns[1][:-1]])
    with pytest.raises(CausalValueError, match="finite"):
        antecedent.state.GraphScoreCache([np.array([0.0, np.nan, 1.0]), np.array([0.0, 1.0, 2.0])])
    with pytest.raises(CausalValueError, match="family"):
        antecedent.state.GraphScoreCache(columns, family="no_such_family")
    laplace = antecedent.state.GraphScoreCache(columns, family="laplace_bic")
    delta, total = laplace.set_parents(1, [0])
    assert total == pytest.approx(laplace.full_score({1: [0]}), rel=1e-6)
    assert delta > 0


def test_rolling_mechanism_diagnostics_fit_the_last_window_of_rows():
    n, window = 30, 10
    i = np.arange(n, dtype=np.float64)
    design = np.column_stack([np.ones(n), i])
    y = 1.5 + 0.25 * i + 0.1 * np.sin(1.7 * i)
    diag = antecedent.state.RollingMechanismDiagnostics(2, window)
    assert (diag.ncols, diag.window, diag.n, diag.beta) == (2, window, 0, [])
    for row, response in zip(design[:20], y[:20], strict=True):
        diag.append_row([float(v) for v in row], float(response))
    rest = design[20:]
    diag.append_batch([float(v) for v in rest.ravel()], [float(v) for v in y[20:]])
    assert diag.n == window, "older rows left the window"
    diag.refresh()

    rows, ys = design[-window:], y[-window:]
    beta, *_ = np.linalg.lstsq(rows, ys, rcond=None)
    residuals = ys - rows @ beta
    assert diag.beta == pytest.approx(list(beta), abs=1e-7)
    assert diag.residual_sse == pytest.approx(float(np.sum(residuals**2)), abs=1e-9)
    assert diag.residual_var == pytest.approx(float(np.sum(residuals**2)) / (window - 2), abs=1e-9)
    assert diag.mean_abs_residual == pytest.approx(float(np.mean(np.abs(residuals))), abs=1e-9)
    assert diag.max_abs_cusum is not None and diag.max_abs_cusum >= 0.0
    assert 0 < diag.bytes <= diag.worst_case_bytes


def test_rolling_mechanism_diagnostics_report_what_a_short_window_cannot_support():
    diag = antecedent.state.RollingMechanismDiagnostics(1, 4)
    with pytest.raises(CausalStateError):
        diag.refresh()
    diag.append_row([2.0], 4.0)
    diag.refresh()
    assert diag.beta == pytest.approx([2.0])
    assert diag.residual_var is None, "one row cannot estimate a variance for one column"
    assert diag.max_abs_cusum is None, "fewer than four residuals carry no CUSUM"
    with pytest.raises(CausalStateError):
        diag.append_row([1.0, 2.0], 1.0)
    with pytest.raises(CausalStateError):
        diag.append_row([1.0], float("nan"))
    with pytest.raises(CausalStateError):
        antecedent.state.RollingMechanismDiagnostics(0, 4)
    with pytest.raises(CausalStateError):
        antecedent.state.RollingMechanismDiagnostics(2, 0)
