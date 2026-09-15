"""One omitted-default table."""

from __future__ import annotations

import numpy as np

import antecedent as ant
from antecedent._defaults import OMITTED, is_temporal_query, resolve_omitted
from antecedent.estimation import PreparedAnalysis
from antecedent.inference import Bayesian, Frequentist


def test_snapshot():
    assert OMITTED == {"bootstrap": 199, "refute": "placebo", "latency": None, "n_draws": 1000}


def test_omitted_table_is_the_native_snapshot():
    assert OMITTED["bootstrap"] == 199
    assert OMITTED["refute"] == "placebo"
    assert OMITTED["n_draws"] == 1000
    assert OMITTED["latency"] is None


def test_temporal_kind_helper():
    assert is_temporal_query(kind="pulse")
    assert is_temporal_query(kind="sustained")
    assert is_temporal_query(kind="temporal_mediation")
    assert not is_temporal_query(kind="average")
    assert is_temporal_query(ant.PulseEffect("t", "y"))


def test_resolve_omitted_static_response_family_zeros_bootstrap():
    for kind in (
        "response_curve",
        "intervention_response",
        "counterfactual",
        "point_derivative",
        "average_derivative",
        "elasticity",
        "semi_elasticity",
        "directional_derivative",
        "response_jacobian",
    ):
        refute, bootstrap, latency = resolve_omitted(
            kind=kind,
            inference=Frequentist(),
            is_temporal=False,
            refute=None,
            bootstrap=None,
            latency=None,
        )
        assert refute == "none"
        assert bootstrap == 0
        assert latency is None


def test_resolve_omitted_frequentist_temporal_response_keeps_replicates():
    _, bootstrap, _ = resolve_omitted(
        kind="response_curve",
        inference=Frequentist(),
        is_temporal=True,
        refute=None,
        bootstrap=None,
        latency=None,
    )
    assert bootstrap == 199
    _, bootstrap, _ = resolve_omitted(
        kind="response_curve",
        inference=Bayesian(),
        is_temporal=True,
        refute=None,
        bootstrap=None,
        latency=None,
    )
    assert bootstrap == 0


def test_resolve_omitted_does_not_inject_latency():
    refute, bootstrap, latency = resolve_omitted(
        kind="average",
        inference=Frequentist(),
        is_temporal=False,
        refute=None,
        bootstrap=None,
        latency=None,
    )
    assert refute == "placebo"
    assert bootstrap == 199
    assert latency is None


def test_bayesian_omitted_n_draws_is_not_explicit():
    assert Bayesian().n_draws is None
    assert Bayesian().n_draws_explicit is False
    assert Bayesian(n_draws=32).n_draws_explicit is True


def test_entry_points_agree():
    rng = np.random.default_rng(1)
    z = rng.normal(size=80)
    t = (rng.uniform(size=80) < 0.5).astype(float)
    y = t + z
    data = {"t": t, "y": y, "z": z}
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    query = ant.AverageEffect("t", "y")
    analyzed = ant.analyze(data, graph=graph, query=query, refute="none")
    prepared = PreparedAnalysis.prepare(data, graph=graph, query=query, refute="none").estimate()
    workflow = ant.prepare(data, graph=graph, query=query, refute="none").estimate()
    assert analyzed.performance.bootstrap_requested == prepared.performance.bootstrap_requested
    assert prepared.performance.bootstrap_requested == workflow.performance.bootstrap_requested
    assert analyzed.performance.latency_mode == prepared.performance.latency_mode
    assert analyzed.inspect().to_dict()["inference_binding_id"] == prepared.inspect().to_dict()[
        "inference_binding_id"
    ]


def test_explicit_budget_reaches_plan():
    rng = np.random.default_rng(2)
    z = rng.normal(size=80)
    t = (rng.uniform(size=80) < 0.5).astype(float)
    y = t + z
    result = ant.analyze(
        {"t": t, "y": y, "z": z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Bayesian(n_draws=1000),
        latency="interactive",
        refute="none",
        bootstrap=0,
    )
    assert result.performance.n_draws == 1000
    joined = " ".join(str(d) for d in result.diagnostics)
    assert "latency.explicit_budget_kept" in joined


def test_omitted_bayesian_draws_are_1000():
    rng = np.random.default_rng(3)
    z = rng.normal(size=64)
    t = (rng.uniform(size=64) < 0.5).astype(float)
    y = t + z
    result = ant.analyze(
        {"t": t, "y": y, "z": z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Bayesian(),
        refute="none",
        bootstrap=0,
    )
    assert result.performance.n_draws == 1000
