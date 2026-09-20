"""T2 done-when fixtures: population-aware functionals and kernel round-trips."""

from __future__ import annotations

import antecedent
from antecedent import transport


def _mean_curve() -> antecedent.ResponseCurve:
    return antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0])


def test_direct_formula_exposes_population_and_lowered_expr() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", []),
        catalog=transport.EvidenceCatalog.empty(),
    )
    result = transport.identify(graph=graph, query=query)
    assert isinstance(result.formula, transport.RecursiveFactorizationFormula)
    assert result.formula.factors[0].population == "target"
    assert result.formula.factors[0].regime is None
    assert not result.formula.factors[0].interventions
    assert result.pretty is not None
    assert "P_target" in result.pretty
    assert result.leaf_bindings
    assert all(population == "target" for population, _ in result.leaf_bindings)


def test_nested_kernel_round_trips_through_artifact() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "m", "y"], [("a", "m"), ("m", "y")])
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", ["m"]),
        source_experiments=["a"],
    )
    result = transport.identify(graph=graph, query=query)
    assert isinstance(result.formula, transport.RecursiveFactorizationFormula)
    assert result.pretty is not None
    assert "K_" in result.pretty or "K_{" in result.pretty
    pretty, latex, bindings, free = transport.reload_lowered_expression(result)
    assert pretty == result.pretty
    assert latex == result.latex
    assert bindings == result.leaf_bindings
    assert free


def test_source_and_target_leaves_stay_distinct() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "z", "y"], [("z", "y"), ("a", "y")])
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", ["z"]),
        source_experiments=["a"],
    )
    result = transport.identify(graph=graph, query=query)
    assert isinstance(result.formula, transport.StandardizationFormula)
    populations = {
        result.formula.source_response.population,
        result.formula.target_law.population,
    }
    assert "trial" in populations
    assert "target" in populations
    leaf_pops = {population for population, _ in result.leaf_bindings}
    assert "trial" in leaf_pops
    assert "target" in leaf_pops


def test_symbolic_source_experiment_reloads_with_derivation() -> None:
    import json

    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        _mean_curve(), transport.SelectionDiagram("trial", "target", []), source_experiments=["a"]
    )
    result = transport.identify(graph=graph, query=query)
    pretty, _, bindings, free = transport.reload_lowered_expression(result)
    assert pretty == result.pretty
    assert bindings == result.leaf_bindings
    assert len(free) == 2
    wire = json.loads(result.expr_wire_json)
    assert wire["derivations"]
    assert any(a.get("symbolic") for assignments in wire["interventions"] for a in assignments)
