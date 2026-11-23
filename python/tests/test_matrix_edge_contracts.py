"""Consuming regressions discovered by the independent matrix edge suite."""

import antecedent as ant
import numpy as np
import pytest
from antecedent.outcomes import require_success

from _loaded_results import assert_answer_kept


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


def confounded_data():
    """`z -> t`, `z -> y`, `t -> y`: the treatment's confounder is observed."""
    rng = np.random.default_rng(19019)
    z = rng.uniform(0.3, 1.7, 160)
    t = 0.6 * z + rng.uniform(0.0, 0.8, 160)
    return {"t": t, "y": 4 + 2.5 * t + 1.5 * z + rng.normal(0, 0.06, 160), "z": z}


@pytest.mark.parametrize("query", DERIVATIVES)
@pytest.mark.parametrize("accepted", [False, True])
def test_cpdag_derivatives_run_on_the_class_whether_or_not_it_is_oriented(query, accepted):
    # A CPDAG is executed as the class it is, never coerced to a Dag: a fully oriented one has
    # one member, and `z -- t` leaves two (`z -> t` adjusts `{z}`, `t -> z` adjusts nothing).
    complete = ant.Cpdag.from_directed_undirected(["t", "y"], [("t", "y")], [])
    incomplete = ant.Cpdag.from_directed_undirected(
        ["t", "y", "z"], [("z", "y"), ("t", "y")], [("z", "t")]
    )
    for graph, values in ((complete, data()), (incomplete, confounded_data())):
        prepared = ant.prepare(
            values,
            graph=ant.AcceptedGraph.from_graph(graph) if accepted else graph,
            query=query,
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
            estimator_config={"bandwidth": 0.3},
        )
        assert prepared.inspect().support.payload["matrix_coordinate"] == (
            f"{type(query).__name__}:Cpdag:{'accepted' if accepted else 'explicit'}:Frequentist:none"
        )
        require_success(prepared.estimate())


@pytest.mark.parametrize(
    "query",
    [
        ant.ResponseCurve("t", "y", grid=[0.5, 1.0, 1.5]),
        ant.DirectionalDerivative(["t"], ["y"], at=[1.0], direction=[-2.0]),
        ant.ResponseJacobian(["t"], ["y"], at=[1.0]),
    ],
)
def test_bayesian_response_band_has_portable_uncertainty_source(query):
    result = require_success(
        ant.analyze(
            data(),
            graph=[("t", "y")],
            query=query,
            inference=ant.Bayesian(n_draws=128),
            refute="none",
            execution=ant.Execution(seed=19019),
            estimator_config={"bandwidth": 0.3},
        )
    )
    loaded = require_success(ant.load(bytes(result.export())))
    for item in (result, loaded):
        uncertainty = item.inspect().uncertainty
        assert uncertainty.available
        assert {
            "source": "parameter",
            "target": "posterior_pointwise_band",
            "omitted": False,
        } in uncertainty.payload["components"]
    assert_answer_kept(loaded)


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
def test_cross_world_families_run_on_the_cpdag_class_whether_or_not_it_is_oriented(query, accepted):
    values = data()
    values["m"] = values["t"] + np.sin(np.arange(len(values["t"])))
    if not isinstance(query, ant.Counterfactual):
        # The class routes of mediation, path-specific effects and distributions run the
        # finite-discrete functional plug-in on every member; a CPDAG counterfactual fits the
        # mechanisms of each member instead and reads the continuous law.
        values = {
            name: (column > np.median(column)).astype(float) for name, column in values.items()
        }
    # Oriented, the class has one member; with `t -- m` it has two (`t -> m` and `m -> t`).
    for complete in (True, False):
        graph = ant.Cpdag.from_directed_undirected(
            ["t", "y", "m"],
            [("m", "y"), ("t", "y")] + ([("t", "m")] if complete else []),
            [] if complete else [("t", "m")],
        )
        prepared = ant.prepare(
            values,
            graph=ant.AcceptedGraph.from_graph(graph) if accepted else graph,
            query=query,
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
        )
        assert prepared.inspect().support.payload["matrix_coordinate"] == (
            f"{type(query).__name__}:Cpdag:{'accepted' if accepted else 'explicit'}:Frequentist:none"
        )
        require_success(prepared.estimate())
