"""T2 done-when fixtures: population-aware functionals and kernel round-trips."""

from __future__ import annotations

import antecedent
import pytest
from antecedent.errors import CausalResourceError, CausalSerializationError, CausalValueError
from antecedent.transport import advanced as transport


def _mean_curve() -> antecedent.ResponseCurve:
    return antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0])


def test_direct_formula_exposes_population_and_lowered_expr() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    query = transport.TransportQuery(
        _mean_curve(),
        transport.SelectionDiagram("trial", "target", []),
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


def test_reloaded_functional_program_executes_and_replays_checked_wire() -> None:
    graph = antecedent.graph.Admg.from_edges(["a", "y"], [("a", "y")])
    catalog = transport.EvidenceCatalog(
        regimes=[transport.EvidenceRegime("observational", "target", measured=["a", "y"])],
    )
    query = transport.TransportQuery(
        antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
        catalog=catalog,
    )
    identification_builder = transport.identify(graph=graph, query=query)
    identification = identification_builder
    del identification_builder
    pretty, latex, bindings, free = transport.reload_lowered_expression(identification)
    assert pretty == identification.pretty
    assert latex == identification.latex
    assert bindings == identification.leaf_bindings
    assert free
    program = transport.reload_lowered_program(identification)
    assert program.source_root == program.executable_root
    with pytest.raises(CausalValueError, match="provider catalog"):
        program.evaluate_exact()
    with pytest.raises(CausalValueError, match="provider laws"):
        program.evaluate_exact(catalog)

    law = transport.ExactDiscreteLaw(
        "target",
        "observational",
        (("a", (0.0, 1.0)), ("y", (0.0, 1.0))),
        (0.4, 0.1, 0.15, 0.35),
        "exact-response-law",
    )
    # Independent reference: P(Y=1 | A=1) = .35 / (.15 + .35) = .7.
    estimate = program.evaluate_exact(catalog, (law,), {"a": 1.0, "y": 1.0})
    assert estimate == pytest.approx(0.7)
    changed_catalog = transport.EvidenceCatalog(
        regimes=[transport.EvidenceRegime("observational", "target", measured=["a"])],
    )
    with pytest.raises(CausalValueError, match="catalog differs"):
        program.evaluate_exact(changed_catalog, (law,), {"a": 1.0, "y": 1.0})
    with pytest.raises(CausalValueError, match="budget must be positive"):
        program.evaluate_exact(catalog, (law,), {"a": 1.0, "y": 1.0}, max_operations=0)

    wire = program.to_wire_json()
    replayed = transport.restore_lowered_program(identification, wire)
    assert replayed.source_root == program.source_root
    assert replayed.executable_root == program.executable_root
    assert replayed.evaluate_exact(catalog, (law,), {"a": 1.0, "y": 1.0}) == pytest.approx(estimate)
    import json

    payload = json.loads(wire)
    payload["executable"] = 999
    with pytest.raises(CausalSerializationError, match="checked functional program"):
        transport.restore_lowered_program(identification, json.dumps(payload))
    # The wire denies unknown fields before any root is checked.
    with pytest.raises(CausalSerializationError, match="unknown field `ignored`"):
        transport.restore_lowered_program(identification, json.dumps({**payload, "ignored": 1}))
    with pytest.raises(CausalSerializationError, match="key must be a string"):
        transport.restore_lowered_program(identification, "{not json")
    del identification
    assert program.evaluate_exact(catalog, (law,), {"a": 1.0, "y": 1.0}) == pytest.approx(estimate)
    assert replayed.evaluate_exact(catalog, (law,), {"a": 1.0, "y": 1.0}) == pytest.approx(estimate)


def test_exhausted_evaluation_budget_is_a_resource_error() -> None:
    graph = antecedent.graph.Admg.from_edges(["c", "a", "y"], [("c", "a"), ("c", "y"), ("a", "y")])
    catalog = transport.EvidenceCatalog(
        regimes=[transport.EvidenceRegime("observational", "target", measured=["c", "a", "y"])],
    )
    query = transport.TransportQuery(
        antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
        catalog=catalog,
    )
    program = transport.reload_lowered_program(transport.identify(graph=graph, query=query))
    law = transport.ExactDiscreteLaw(
        "target",
        "observational",
        (("c", (0.0, 1.0)), ("a", (0.0, 1.0)), ("y", (0.0, 1.0))),
        (0.1, 0.15, 0.1, 0.15, 0.1, 0.15, 0.1, 0.15),
        "exact-confounded-law",
    )
    at = {"a": 1.0, "y": 1.0}
    assert program.evaluate_exact(catalog, (law,), at) == pytest.approx(0.6)
    with pytest.raises(CausalResourceError, match="operation budget exceeded"):
        program.evaluate_exact(catalog, (law,), at, max_operations=1)
    with pytest.raises(CausalResourceError, match="support budget exceeded"):
        program.evaluate_exact(catalog, (law,), at, max_support_rows=1)


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
