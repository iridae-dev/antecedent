"""Consuming regressions discovered by the independent matrix edge suite."""

import antecedent as ant
import numpy as np
import pytest


def data():
    rng = np.random.default_rng(19019)
    t = rng.uniform(0.3, 1.7, 160)
    return {"t": t, "y": 4 + 2.5 * t + rng.normal(0, 0.06, len(t))}


DERIVATIVES = [
    ant.PointDerivative("t", "y", at=1.0),
    ant.AverageDerivative("t", "y"),
    ant.Elasticity("t", "y", at=1.0),
    ant.SemiElasticity("t", "y", at=1.0),
    ant.DirectionalDerivative(["t"], ["y"], at=[1.0], direction=[-2.0]),
    ant.ResponseJacobian(["t"], ["y"], at=[1.0]),
]


@pytest.mark.parametrize("query", DERIVATIVES)
@pytest.mark.parametrize("accepted", [False, True])
def test_cpdag_derivatives_normalize_only_fully_oriented_graphs(query, accepted):
    for complete in (True, False):
        graph = ant.Cpdag.from_directed_undirected(
            ["t", "y"],
            [("t", "y")] if complete else [],
            [] if complete else [("t", "y")],
        )
        if accepted:
            graph = ant.AcceptedGraph.from_graph(graph)

        def prepared(graph=graph):
            return ant.prepare(
                data(),
                graph=graph,
                query=query,
                refute="none",
                bootstrap=0,
                estimator_config={"bandwidth": 0.3},
            )

        if complete:
            assert prepared().inspect().support.payload["matrix_coordinate"] == (
                f"{type(query).__name__}:Dag:{'accepted' if accepted else 'explicit'}:Frequentist:none"
            )
        else:
            with pytest.raises(ValueError, match="undirected|orient"):
                prepared()


@pytest.mark.parametrize(
    "query",
    [
        ant.ResponseCurve("t", "y", grid=[0.5, 1.0, 1.5]),
        ant.DirectionalDerivative(["t"], ["y"], at=[1.0], direction=[-2.0]),
        ant.ResponseJacobian(["t"], ["y"], at=[1.0]),
    ],
)
def test_bayesian_response_band_has_portable_uncertainty_source(query):
    result = ant.analyze(
        data(),
        graph=[("t", "y")],
        query=query,
        inference=ant.Bayesian(n_draws=128),
        refute="none",
        bootstrap=0,
        seed=19019,
        estimator_config={"bandwidth": 0.3},
    )
    loaded = ant.load(result.export())
    for item in (result, loaded):
        uncertainty = item.inspect().uncertainty
        assert uncertainty.available
        assert {
            "source": "parameter",
            "target": "posterior_pointwise_band",
            "omitted": False,
        } in uncertainty.payload["components"]
    assert loaded.acceptance.verified


@pytest.mark.parametrize(
    "query",
    [
        ant.MediationEffect("t", "y", mediators=["m"]),
        ant.Counterfactual("t", "y"),
        ant.PathSpecificEffect("t", "y", path_nodes=["m"]),
        ant.InterventionalDistribution("y", interventions={"t": 1.0}),
    ],
)
@pytest.mark.parametrize("accepted", [False, True])
def test_static_dag_families_preserve_cpdag_compatibility(query, accepted):
    values = data()
    values["m"] = values["t"] + np.sin(np.arange(len(values["t"])))
    if isinstance(query, ant.InterventionalDistribution):
        # The functional-distribution estimator has a finite discrete support
        # contract. Keep this graph-class compatibility fixture inside it.
        values = {name: (column > np.median(column)).astype(float) for name, column in values.items()}
    for complete in (True, False):
        graph = ant.Cpdag.from_directed_undirected(
            ["t", "y", "m"],
            [("m", "y"), ("t", "y")] + ([("t", "m")] if complete else []),
            [] if complete else [("t", "m")],
        )
        if accepted:
            graph = ant.AcceptedGraph.from_graph(graph)

        def prepared(graph=graph):
            return ant.prepare(values, graph=graph, query=query, refute="none", bootstrap=0)

        if complete:
            assert prepared().inspect().support.payload["matrix_coordinate"] == (
                f"{type(query).__name__}:Dag:{'accepted' if accepted else 'explicit'}:Frequentist:none"
            )
        else:
            with pytest.raises(ValueError, match="undirected|orient"):
                prepared()
