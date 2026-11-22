"""The derivative and cross-world query families on class graphs, prepared and executed.

Each test prepares a derivative, mediation decomposition, counterfactual, natural direct
effect, path-specific effect or interventional distribution on a supplied or accepted
``Cpdag``, ``Pag``, latent-confounded ``Admg`` or ``TemporalAdmg`` (read as its
contemporaneous projection), or over a graph posterior of such atoms, then executes it. The
oracle of every test is the conformance fixture the Rust route evidence reads, with the same
rows, graph and query and the fixture's declared tolerance:

* derivatives: ``conformance/response/class_derivative_states`` (the slope of ``y`` and ``v`` in
  ``a`` adjusted for the observed confounder ``z``, solved in exact rational arithmetic);
* counterfactual and natural direct effect: ``conformance/estimate/cross_world_class_truth``;
* mediation: ``conformance/estimate/mediation_mixed_graphs``;
* path-specific effect: ``conformance/estimate/path_specific_mixed_graphs``;
* interventional distribution: ``conformance/estimate/interventional_distribution_classes`` and
  ``conformance/estimate/admg_frontdoor_distribution``;
* graph posteriors: the same fixtures, the posterior value being the weighted mean of the atoms'
  fixture values (``conformance/estimate/cross_world_counterfactual_posterior`` for the
  counterfactual).

Every prepared study must bind the coordinate of the class it was handed, never a ``Dag`` the
class was coerced to.
"""

from __future__ import annotations

import math
from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError
from antecedent.graph import TemporalAdmg
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, load_json


def _pin(path: str) -> dict[str, Any]:
    return load_json(REPO_ROOT / "conformance" / path / "expected.json")


_DERIVATIVE_PIN = _pin("response/class_derivative_states")
_CROSS_WORLD_PIN = _pin("estimate/cross_world_class_truth")
_POSTERIOR_PIN = _pin("estimate/cross_world_counterfactual_posterior")
_MEDIATION_PIN = _pin("estimate/mediation_mixed_graphs")
_PATH_PIN = _pin("estimate/path_specific_mixed_graphs")
_DISTRIBUTION_PIN = _pin("estimate/interventional_distribution_classes")
_FRONT_DOOR_PIN = _pin("estimate/admg_frontdoor_distribution")

_FREQUENTIST = ant.Frequentist(bootstrap=0)


def _coordinate(item: Any) -> str:
    return item.inspect().support.payload["matrix_coordinate"]


def _source(accepted: bool) -> str:
    return "accepted" if accepted else "explicit"


def _bind(graph: Any, accepted: bool) -> Any:
    return ant.AcceptedGraph.from_graph(graph, algorithm_id="fixture.class") if accepted else graph


def _lifted(names: list[str], directed: list[tuple[str, str]], bidirected=()) -> TemporalAdmg:
    """A temporal ADMG with every node contemporaneous.

    The edges are listed so that the nodes first appear in the order of ``names``, which keeps
    the template's node order the variable order, as the Rust contemporaneous lift does.
    """
    return TemporalAdmg.from_lagged_edges(
        names,
        [(a, 0, b, 0) for a, b in directed],
        [(a, 0, b, 0) for a, b in bidirected] or None,
    )


def _interleaved(columns: list[str], table: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """A contingency table's rows in the fixed interleaved order the Rust evidence reads."""
    values = {name: [] for name in columns}
    for cell in table:
        for name in columns:
            values[name].extend([float(cell[name])] * int(cell["count"]))
    rows = len(values[columns[0]])
    order = [(row * 97) % rows for row in range(rows)]
    assert len(set(order)) == rows
    return {name: np.asarray(column, dtype=float)[order] for name, column in values.items()}


def _expanded(columns: list[str], table: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values = {name: [] for name in columns}
    for cell in table:
        for name in columns:
            values[name].extend([float(cell[name])] * int(cell["count"]))
    return {name: np.asarray(column, dtype=float) for name, column in values.items()}


def _counts(name: str) -> dict[str, np.ndarray]:
    spec = _CROSS_WORLD_PIN["tables"][name]
    columns = spec["columns"]
    values: dict[str, list[float]] = {column: [] for column in columns}
    for cell in spec["counts"]:
        for column, level in zip(columns, cell["levels"], strict=True):
            values[column].extend([float(level)] * int(cell["count"]))
    return {column: np.asarray(values[column], dtype=float) for column in columns}


def _line() -> dict[str, np.ndarray]:
    line = _CROSS_WORLD_PIN["line"]
    x = np.arange(int(line["rows"]), dtype=float) / 20.0 - 2.0
    return {"x": x, "y": float(line["intercept"]) + float(line["slope"]) * x}


# -- derivatives ------------------------------------------------------------------------------

#: `r -> a <- z`, `a -> y`, `z -> y`, `a -> v`, `z -> v`: a v-structure, so every edge is
#: compelled and the class has one completion.
_CLASS_EDGES = [("r", "a"), ("z", "a"), ("a", "y"), ("z", "y"), ("a", "v"), ("z", "v")]
#: The same edges plus `w -> z` and the latent common cause `w <-> y`.
_ADMG_DIRECTED = [*_CLASS_EDGES, ("w", "z")]
_ADMG_BIDIRECTED = [("w", "y")]


def _derivative_rows() -> dict[str, np.ndarray]:
    """The 800 rows of the fixture's `dgp`, with its 64-bit LCG noise (`y` then `v` per row)."""
    mask = (1 << 64) - 1
    state = 20_240_607

    def gauss() -> float:
        nonlocal state
        total = 0.0
        for _ in range(12):
            state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & mask
            total += (state >> 11) / 9_007_199_254_740_992.0
        return total - 6.0

    rows: dict[str, list[float]] = {name: [] for name in ("a", "z", "y", "r", "v", "w")}
    for i in range(800):
        t = float(i)
        latent = math.cos(t * 0.77 + 2.0)
        w = math.sin(t * 1.7 + 0.3) + 0.8 * latent
        r = math.sin(t * 0.29 + 1.0)
        z = math.cos(t * 0.41) + 0.5 * w
        a = 2.0 + 0.6 * z + 0.5 * r + math.sin(t * 0.71) + 0.2 * math.cos(t * 0.13)
        y = 5.0 + 2.0 * a + 3.0 * z + 0.7 * latent + 0.5 * gauss()
        v = 1.0 + 0.25 * a - 1.5 * z + 0.5 * gauss()
        for name, value in (("a", a), ("z", z), ("y", y), ("r", r), ("v", v), ("w", w)):
            rows[name].append(value)
    return {name: np.asarray(column) for name, column in rows.items()}


_DERIVATIVE_ROWS = _derivative_rows()


def _derivative_structure(structure: str) -> tuple[dict[str, np.ndarray], Any]:
    names = ["a", "z", "y", "r", "v", "w"]
    if structure == "Cpdag":
        graph: Any = ant.Cpdag.from_directed_undirected(names, _CLASS_EDGES, [])
    elif structure == "Pag":
        graph = ant.Pag.from_marked_edges(names, [(a, b, "tail", "arrow") for a, b in _CLASS_EDGES])
    elif structure == "Admg":
        graph = ant.Admg.from_edges(names, _ADMG_DIRECTED, bidirected=_ADMG_BIDIRECTED)
    else:
        # The node order of first appearance in the edges below.
        names = ["r", "a", "z", "y", "v", "w"]
        graph = _lifted(names, _ADMG_DIRECTED, _ADMG_BIDIRECTED)
    return {name: _DERIVATIVE_ROWS[name] for name in names}, graph


def _derivative_case(case: str) -> tuple[Any, list[float]]:
    pin = _DERIVATIVE_PIN
    at = float(pin["at"])
    slope, slope_v = float(pin["slope_y"]), float(pin["slope_v"])
    return {
        "point": (ant.PointDerivative("a", "y", at=at), [slope]),
        "elasticity": (ant.Elasticity("a", "y", at=at), [float(pin["elasticity"])]),
        "semi_treatment": (
            ant.SemiElasticity("a", "y", at=at, log_scale="treatment"),
            [float(pin["semi_treatment"])],
        ),
        "semi_outcome": (
            ant.SemiElasticity("a", "y", at=at, log_scale="outcome"),
            [float(pin["semi_outcome"])],
        ),
        "average": (ant.AverageDerivative("a", "y"), [slope]),
        "jacobian": (ant.ResponseJacobian(["a"], ["y", "v"], at=[at]), [slope, slope_v]),
        "directional": (
            ant.DirectionalDerivative(["a"], ["y", "v"], at=[at], direction=[1.0]),
            [slope, slope_v],
        ),
    }[case]


_DERIVATIVE_CASES = [
    "point",
    "elasticity",
    "semi_treatment",
    "semi_outcome",
    "average",
    "jacobian",
    "directional",
]
_DERIVATIVE_STRUCTURES = ["Cpdag", "Pag", "Admg", "TemporalAdmg"]


def _assert_derivative(result: Any, case: str, truths: list[float]) -> None:
    band = float(_DERIVATIVE_PIN["value_relative_tolerance"])
    values = np.asarray(result.estimate, dtype=float).ravel()
    assert len(values) == len(truths), (case, values)
    for value, truth in zip(values, truths, strict=True):
        assert abs(value - truth) <= max(band * abs(truth), 0.1), (case, value, truth)
    if case in ("point", "average"):
        # The confounded slope is not what a route publishes.
        assert abs(values[0] - float(_DERIVATIVE_PIN["naive_y"])) > 0.3, (case, values)


@pytest.mark.parametrize("case", _DERIVATIVE_CASES)
@pytest.mark.parametrize("structure", _DERIVATIVE_STRUCTURES)
@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
def test_class_derivative_is_the_adjusted_structural_slope(case, structure, accepted):
    data, graph = _derivative_structure(structure)
    query, truths = _derivative_case(case)
    prepared = ant.prepare(
        data,
        graph=_bind(graph, accepted),
        query=query,
        refute="none",
        inference=_FREQUENTIST,
        estimator_config={"bandwidth": 0.35},
        execution=ant.Execution(seed=814),
    )
    assert _coordinate(prepared) == (
        f"{type(query).__name__}:{structure}:{_source(accepted)}:Frequentist:none"
    )
    result = require_success(prepared.estimate())
    _assert_derivative(result, case, truths)
    # The retained study re-executes the same program.
    again = require_success(prepared.estimate())
    assert np.asarray(again.estimate, dtype=float).ravel().tolist() == (
        np.asarray(result.estimate, dtype=float).ravel().tolist()
    )


@pytest.mark.parametrize("case", _DERIVATIVE_CASES)
@pytest.mark.parametrize("structure", _DERIVATIVE_STRUCTURES)
def test_bayesian_class_derivative_is_the_adjusted_structural_slope(case, structure):
    data, graph = _derivative_structure(structure)
    query, truths = _derivative_case(case)
    prepared = ant.prepare(
        data,
        graph=graph,
        query=query,
        refute="none",
        inference=ant.Bayesian(n_draws=24),
        estimator_config={"bandwidth": 0.35},
        execution=ant.Execution(seed=814),
    )
    assert _coordinate(prepared) == f"{type(query).__name__}:{structure}:explicit:Bayesian:none"
    _assert_derivative(require_success(prepared.estimate()), case, truths)


def _two_member_rows() -> dict[str, np.ndarray]:
    """`z ~ N(0, 1)`, `t = .8 z + e`, `y = t + z + e` from the Rust fixture's LCG normals."""
    mask = (1 << 64) - 1
    state = ((11 * 0x9E37_79B9_7F4A_7C15) & mask) | 1

    def gauss() -> float:
        nonlocal state
        total = 0.0
        for _ in range(12):
            state = (state * 6_364_136_223_846_793_005 + 1) & mask
            total += (state >> 11) / float(1 << 53)
        return total - 6.0

    t, y, z = [], [], []
    for _ in range(4000):
        zi = gauss()
        ti = 0.8 * zi + gauss()
        yi = ti + zi + gauss()
        t.append(ti)
        y.append(yi)
        z.append(zi)
    return {"t": np.asarray(t), "y": np.asarray(y), "z": np.asarray(z)}


def test_disagreeing_cpdag_members_publish_the_identified_set_of_their_slopes():
    """`z -> y`, `t -> y` with `z -- t`: the member `z -> t` adjusts `{z}`, `t -> z` nothing."""
    data = _two_member_rows()
    ones = np.ones_like(data["t"])
    adjusted = np.linalg.lstsq(
        np.column_stack([ones, data["t"], data["z"]]), data["y"], rcond=None
    )[0][1]
    unadjusted = np.linalg.lstsq(np.column_stack([ones, data["t"]]), data["y"], rcond=None)[0][1]
    assert unadjusted - adjusted > 0.3
    graph = ant.Cpdag.from_directed_undirected(
        ["t", "y", "z"], [("z", "y"), ("t", "y")], [("z", "t")]
    )
    prepared = ant.prepare(
        data,
        graph=graph,
        query=ant.PointDerivative("t", "y", at=0.0),
        refute="none",
        inference=_FREQUENTIST,
        estimator_config={"bandwidth": 0.8},
        execution=ant.Execution(seed=5),
    )
    assert _coordinate(prepared) == "PointDerivative:Cpdag:explicit:Frequentist:none"
    result = require_success(prepared.estimate())
    assert result.envelope is not None
    lower = float(np.min(np.asarray(result.envelope.lower, dtype=float)))
    upper = float(np.max(np.asarray(result.envelope.upper, dtype=float)))
    for got, want in ((lower, adjusted), (upper, unadjusted)):
        assert abs(got - want) < 0.1 * max(abs(want), 0.5) + 0.02, (got, want)


# -- counterfactual and natural direct effect ---------------------------------------------------


def _cross_world_case(name: str) -> tuple[str, dict[str, np.ndarray], Any, Any, float]:
    truth = _CROSS_WORLD_PIN["cases"]
    front_door = _counts("front_door")
    if name == "admg_counterfactual":
        graph = ant.Admg.from_edges(
            ["t", "m", "y"], [("t", "m"), ("m", "y")], bidirected=[("t", "y")]
        )
        return "Admg", front_door, graph, ant.Counterfactual("t", "y"), truth[name]["expected"]
    if name == "temporal_admg_counterfactual":
        graph = _lifted(["t", "m", "y"], [("t", "m"), ("m", "y")], [("t", "y")])
        query = ant.Counterfactual("t", "y")
        return "TemporalAdmg", front_door, graph, query, truth["admg_counterfactual"]["expected"]
    if name == "pag_counterfactual":
        names = ["w", "x", "z", "y"]
        edges = [("w", "x"), ("w", "z"), ("z", "y"), ("x", "y")]
        graph = ant.Pag.from_marked_edges(names, [(a, b, "tail", "arrow") for a, b in edges])
        query = ant.Counterfactual("x", "y")
        return "Pag", _counts("side_path_chain"), graph, query, truth[name]["expected"]
    if name == "admg_natural_direct":
        graph = ant.Admg.from_edges(["t", "m", "y"], [("t", "m"), ("t", "y"), ("m", "y")])
        query = ant.NestedCounterfactual("t", "m", "y")
        return "Admg", front_door, graph, query, truth[name]["expected"]
    if name == "temporal_admg_natural_direct":
        graph = _lifted(["t", "m", "y"], [("t", "m"), ("t", "y"), ("m", "y")])
        query = ant.NestedCounterfactual("t", "m", "y")
        return "TemporalAdmg", front_door, graph, query, truth["admg_natural_direct"]["expected"]
    if name == "pag_natural_direct":
        names = ["c", "w", "t", "m", "y"]
        edges = [("c", "t"), ("w", "m"), ("t", "m"), ("t", "y"), ("m", "y")]
        graph = ant.Pag.from_marked_edges(names, [(a, b, "tail", "arrow") for a, b in edges])
        query = ant.NestedCounterfactual("t", "m", "y")
        return "Pag", _counts("mediated_two_parents"), graph, query, truth[name]["expected"]
    assert name == "cpdag_natural_direct"
    edges = [("t", "a"), ("t", "b"), ("a", "y"), ("b", "y")]
    graph = ant.Cpdag.from_directed_undirected(["t", "a", "b", "y"], edges, [])
    query = ant.NestedCounterfactual("t", "a", "y")
    return "Cpdag", _counts("two_mediators"), graph, query, truth["cpdag_path_specific"]["expected"]


_CROSS_WORLD_CASES = [
    "admg_counterfactual",
    "temporal_admg_counterfactual",
    "pag_counterfactual",
    "admg_natural_direct",
    "temporal_admg_natural_direct",
    "pag_natural_direct",
    "cpdag_natural_direct",
]


def _query_name(query: Any) -> str:
    return (
        "NestedCounterfactualEffect"
        if isinstance(query, ant.NestedCounterfactual)
        else type(query).__name__
    )


@pytest.mark.parametrize("case", _CROSS_WORLD_CASES)
@pytest.mark.parametrize("bayesian", [False, True], ids=["frequentist", "bayesian"])
@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
def test_class_cross_world_effect_is_the_exact_functional_of_the_count_table(
    case, bayesian, accepted
):
    structure, data, graph, query, truth = _cross_world_case(case)
    inference = ant.Bayesian(n_draws=200, backend="conjugate") if bayesian else _FREQUENTIST
    prepared = ant.prepare(
        data,
        graph=_bind(graph, accepted),
        query=query,
        refute="none",
        inference=inference,
        execution=ant.Execution(seed=5),
    )
    assert _coordinate(prepared) == (
        f"{_query_name(query)}:{structure}:{_source(accepted)}:"
        f"{'Bayesian' if bayesian else 'Frequentist'}:none"
    )
    result = require_success(prepared.estimate())
    tolerance = (0.05 if structure == "Cpdag" else 0.03) if bayesian else 1e-9
    assert abs(result.estimate.ate - float(truth)) <= tolerance, (result.estimate.ate, truth)


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
def test_incomplete_cpdag_counterfactual_publishes_the_set_of_its_members(accepted):
    """`x -- y` over the noiseless line: the members `x -> y` and `y -> x` disagree."""
    prepared = ant.prepare(
        _line(),
        graph=_bind(ant.Cpdag.from_directed_undirected(["x", "y"], [], [("x", "y")]), accepted),
        query=ant.Counterfactual("x", "y"),
        refute="none",
        inference=_FREQUENTIST,
        execution=ant.Execution(seed=5),
    )
    assert _coordinate(prepared) == f"Counterfactual:Cpdag:{_source(accepted)}:Frequentist:none"
    result = require_success(prepared.estimate())
    line = _CROSS_WORLD_PIN["line"]
    assert result.estimate.ate == pytest.approx(float(line["mean"]), abs=1e-6)
    assert result.structural_identified_set == pytest.approx(
        tuple(float(bound) for bound in line["identified_set"]), abs=1e-6
    )


@pytest.mark.parametrize("refute", ["cheap", "full"])
def test_a_pag_counterfactual_under_validation_keeps_its_refusal(refute):
    """The matrix refuses this cell with a documented reason; the route keeps that refusal."""
    _, data, graph, query, _ = _cross_world_case("pag_counterfactual")
    with pytest.raises(CausalError, match="not_supported") as caught:
        ant.prepare(
            data,
            graph=graph,
            query=query,
            refute=refute,
            inference=_FREQUENTIST,
            execution=ant.Execution(seed=5),
        )
    assert "placebo treatment permutes the treatment" in str(caught.value)


# -- mediation ------------------------------------------------------------------------------


def _mixed_graph_case(pin: dict[str, Any], structure: str, key: str) -> tuple[Any, Any]:
    case = pin[key]
    columns = case["columns"]
    directed = [tuple(edge) for edge in case["graph"]["directed_edges"]]
    bidirected = [tuple(edge) for edge in case["graph"].get("bidirected_edges", [])]
    if structure == "Admg":
        graph: Any = ant.Admg.from_edges(columns, directed, bidirected=bidirected)
    elif structure == "TemporalAdmg":
        graph = _lifted(columns, directed, bidirected)
    elif structure == "Cpdag":
        graph = ant.Cpdag.from_directed_undirected(columns, directed, [])
    else:
        graph = ant.Pag.from_marked_edges(columns, [(a, b, "tail", "arrow") for a, b in directed])
    return case, graph


@pytest.mark.parametrize("structure", ["Admg", "TemporalAdmg", "Cpdag", "Pag"])
@pytest.mark.parametrize("bayesian", [False, True], ids=["frequentist", "bayesian"])
@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
def test_class_mediation_decomposition_is_the_enumerated_truth(structure, bayesian, accepted):
    key = "covariate_latent" if structure in ("Admg", "TemporalAdmg") else "markov_parents"
    case, graph = _mixed_graph_case(_MEDIATION_PIN, structure, key)
    spec = _MEDIATION_PIN["query"]
    inference = (
        ant.Bayesian(n_draws=200, prior_scale=10.0, backend="conjugate")
        if bayesian
        else _FREQUENTIST
    )
    prepared = ant.prepare(
        _interleaved(case["columns"], case["contingency_table"]),
        graph=_bind(graph, accepted),
        query=ant.MediationEffect(
            spec["treatment"],
            spec["outcome"],
            mediators=list(spec["mediators"]),
            contrast="natural_direct",
        ),
        refute="none",
        inference=inference,
        execution=ant.Execution(seed=2129),
    )
    assert _coordinate(prepared) == (
        f"MediationEffect:{structure}:{_source(accepted)}:"
        f"{'Bayesian' if bayesian else 'Frequentist'}:none"
    )
    result = require_success(prepared.estimate())
    tolerance = (
        float(_MEDIATION_PIN["bayesian"]["posterior_mean_tolerance"])
        if bayesian
        else float(_MEDIATION_PIN["frequentist"]["absolute_tolerance"])
    )
    truth = case["truth"]
    assert abs(result.estimate.ate - truth["natural_direct_effect"]) < tolerance
    assert result.mediation is not None
    for name, key in (
        ("total", "total_effect"),
        ("direct", "natural_direct_effect"),
        ("mediated", "natural_indirect_effect"),
    ):
        assert abs(getattr(result.mediation, name) - truth[key]) < tolerance, name


# -- path-specific effect ---------------------------------------------------------------------


def _path_query(case: dict[str, Any]) -> Any:
    spec = case["query"]
    return ant.PathSpecificEffect(
        spec["treatment"],
        spec["outcome"],
        path_nodes=list(spec["path_nodes"]),
        control_level=float(spec["control_level"]),
        active_level=float(spec["active_level"]),
        max_len=4,
    )


@pytest.mark.parametrize("structure", ["Admg", "TemporalAdmg", "Cpdag", "Pag"])
@pytest.mark.parametrize("bayesian", [False, True], ids=["frequentist", "bayesian"])
@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
def test_class_path_specific_effect_is_the_enumerated_truth(structure, bayesian, accepted):
    key = "district_admg" if structure in ("Admg", "TemporalAdmg") else "routed_markov"
    case, graph = _mixed_graph_case(_PATH_PIN, structure, key)
    inference = (
        ant.Bayesian(n_draws=100, prior_scale=10.0, backend="conjugate")
        if bayesian
        else _FREQUENTIST
    )
    prepared = ant.prepare(
        _interleaved(case["columns"], case["contingency_table"]),
        graph=_bind(graph, accepted),
        query=_path_query(case),
        refute="none",
        inference=inference,
        execution=ant.Execution(seed=2124),
    )
    assert _coordinate(prepared) == (
        f"PathSpecificEffect:{structure}:{_source(accepted)}:"
        f"{'Bayesian' if bayesian else 'Frequentist'}:none"
    )
    result = require_success(prepared.estimate())
    tolerance = (
        float(_PATH_PIN["bayesian"]["posterior_mean_tolerance"])
        if bayesian
        else float(_PATH_PIN["frequentist"]["absolute_tolerance"])
    )
    truth = case["truth"]
    assert abs(result.estimate.ate - truth["path_specific_effect"]) < tolerance
    # Neither the total effect nor the unselected-path effect is the answer.
    assert abs(result.estimate.ate - truth["total_effect"]) > 0.05


def test_collider_cpdag_path_specific_effect_publishes_the_enumerated_identified_set():
    case = _PATH_PIN["collider_class"]
    graph = ant.Cpdag.from_directed_undirected(
        case["columns"],
        [tuple(edge) for edge in case["cpdag"]["directed_edges"]],
        [tuple(edge) for edge in case["cpdag"]["undirected_edges"]],
    )
    prepared = ant.prepare(
        _interleaved(case["columns"], case["contingency_table"]),
        graph=graph,
        query=_path_query(case),
        refute="none",
        inference=_FREQUENTIST,
        execution=ant.Execution(seed=2125),
    )
    assert _coordinate(prepared) == "PathSpecificEffect:Cpdag:explicit:Frequentist:none"
    result = require_success(prepared.estimate())
    truth = case["truth"]
    assert result.structural_identified_set == pytest.approx(
        (truth["identified_set_lower"], truth["identified_set_upper"]), abs=1e-9
    )
    assert (result.structural_unidentified_mass or 0.0) == pytest.approx(
        truth["unidentified_mass"], abs=1e-12
    )


# -- interventional distribution --------------------------------------------------------------


def _probability(result: Any, level: float) -> float:
    return next(
        atom.probability for atom in result.estimate.distribution if atom.outcomes[0][1] == level
    )


@pytest.mark.parametrize("structure", ["Cpdag", "Pag"])
@pytest.mark.parametrize("bayesian", [False, True], ids=["frequentist", "bayesian"])
@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
def test_class_interventional_distribution_is_the_enumerated_truth(structure, bayesian, accepted):
    case = _DISTRIBUTION_PIN["class_point"]
    directed = [tuple(edge) for edge in case["graph"]["directed_edges"]]
    graph = (
        ant.Cpdag.from_directed_undirected(case["columns"], directed, [])
        if structure == "Cpdag"
        else ant.Pag.from_marked_edges(
            case["columns"], [(a, b, "tail", "arrow") for a, b in directed]
        )
    )
    spec = _DISTRIBUTION_PIN["query"]
    inference = ant.Bayesian(n_draws=256, backend="conjugate") if bayesian else _FREQUENTIST
    prepared = ant.prepare(
        _expanded(case["columns"], case["contingency_table"]),
        graph=_bind(graph, accepted),
        query=ant.InterventionalDistribution(spec["outcome"], interventions=dict(spec["do"])),
        refute="none",
        inference=inference,
        execution=ant.Execution(seed=2127),
    )
    assert _coordinate(prepared) == (
        f"InterventionalDistribution:{structure}:{_source(accepted)}:"
        f"{'Bayesian' if bayesian else 'Frequentist'}:none"
    )
    result = require_success(prepared.estimate())
    tolerance = (
        float(_DISTRIBUTION_PIN["bayesian"]["posterior_mean_tolerance"])
        if bayesian
        else float(_DISTRIBUTION_PIN["frequentist"]["absolute_tolerance"])
    )
    for key, level in (("y0", 0.0), ("y1", 1.0)):
        assert abs(_probability(result, level) - case["truth"][key]) < tolerance, key
    # One member (or one functional over the PAG's candidates) is a point, not an envelope.
    assert not any("estimate.distribution.class_envelope" in str(d) for d in result.diagnostics)


def test_disagreeing_cpdag_distribution_publishes_the_enumeration_weighted_atoms():
    case = _DISTRIBUTION_PIN["class_disagree"]
    graph = ant.Cpdag.from_directed_undirected(
        case["columns"],
        [tuple(edge) for edge in case["cpdag"]["directed_edges"]],
        [tuple(edge) for edge in case["cpdag"]["undirected_edges"]],
    )
    spec = _DISTRIBUTION_PIN["query"]
    prepared = ant.prepare(
        _expanded(case["columns"], case["contingency_table"]),
        graph=graph,
        query=ant.InterventionalDistribution(spec["outcome"], interventions=dict(spec["do"])),
        refute="none",
        inference=_FREQUENTIST,
        execution=ant.Execution(seed=2128),
    )
    assert _coordinate(prepared) == "InterventionalDistribution:Cpdag:explicit:Frequentist:none"
    result = require_success(prepared.estimate())
    assert any("estimate.distribution.class_envelope" in str(d) for d in result.diagnostics)
    assert _probability(result, 1.0) == pytest.approx(case["truth"]["y1"]["mean"], abs=1e-9)


@pytest.mark.parametrize("bayesian", [False, True], ids=["frequentist", "bayesian"])
@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
def test_temporal_admg_interventional_distribution_is_the_front_door_truth(bayesian, accepted):
    pin = _FRONT_DOOR_PIN
    data = _expanded(pin["columns"], pin["contingency_table"])
    graph = _lifted(
        pin["columns"],
        [tuple(edge) for edge in pin["graph"]["directed_edges"]],
        [tuple(edge) for edge in pin["graph"]["bidirected_edges"]],
    )
    inference = ant.Bayesian(n_draws=256, backend="conjugate") if bayesian else _FREQUENTIST
    for row in pin["truth"]:
        prepared = ant.prepare(
            data,
            graph=_bind(graph, accepted),
            query=ant.InterventionalDistribution("y", interventions={"t": float(row["t"])}),
            refute="none",
            inference=inference,
            execution=ant.Execution(seed=11017),
        )
        assert _coordinate(prepared) == (
            f"InterventionalDistribution:TemporalAdmg:{_source(accepted)}:"
            f"{'Bayesian' if bayesian else 'Frequentist'}:none"
        )
        result = require_success(prepared.estimate())
        tolerance = float(pin["bayesian"]["absolute_tolerance"]) if bayesian else 1e-9
        assert abs(result.estimate.ate - float(row["p_y1"])) < tolerance, row


# -- graph posteriors of class atoms ----------------------------------------------------------


@pytest.mark.parametrize("structure", ["Cpdag", "Pag", "Admg"])
@pytest.mark.parametrize("case", ["point", "average"])
def test_class_posterior_derivative_is_the_adjusted_structural_slope(structure, case):
    data, graph = _derivative_structure(structure)
    posterior = ant.discovery.GraphPosterior.from_graphs(list(data), [1.0], [graph])
    query, truths = _derivative_case(case)
    prepared = ant.prepare(
        data,
        discovery=posterior,
        query=query,
        refute="none",
        inference=_FREQUENTIST,
        estimator_config={"bandwidth": 0.35},
        execution=ant.Execution(seed=814),
    )
    assert _coordinate(prepared) == (
        f"{type(query).__name__}:{structure}:graph_posterior:Frequentist:none"
    )
    _assert_derivative(require_success(prepared.estimate()), case, truths)


def test_admg_posterior_counterfactual_is_the_weighted_mean_of_its_atoms():
    atoms = {atom["name"]: atom for atom in _POSTERIOR_PIN["posterior"]["atoms"]}
    names = ["t", "m", "y"]
    front_door = ant.Admg.from_edges(names, [("t", "m"), ("m", "y")], bidirected=[("t", "y")])
    unconfounded = ant.Admg.from_edges(names, [("t", "m"), ("t", "y"), ("m", "y")])
    weights = [0.625, 0.375]
    posterior = ant.discovery.GraphPosterior.from_graphs(names, weights, [front_door, unconfounded])
    prepared = ant.prepare(
        _counts("front_door"),
        discovery=posterior,
        query=ant.Counterfactual("t", "y"),
        refute="none",
        inference=_FREQUENTIST,
        execution=ant.Execution(seed=3),
    )
    assert _coordinate(prepared) == "Counterfactual:Admg:graph_posterior:Frequentist:none"
    expected = weights[0] * float(atoms["front_door"]["case"]["expected"]) + weights[1] * float(
        atoms["unconfounded"]["case"]["expected"]
    )
    assert expected == pytest.approx(float(_POSTERIOR_PIN["posterior"]["expected"]), abs=1e-12)
    result = require_success(prepared.estimate())
    assert result.estimate.ate == pytest.approx(expected, abs=1e-9)


def test_cpdag_posterior_counterfactual_is_the_mean_of_the_members():
    line = _POSTERIOR_PIN["line"]
    x = np.arange(int(line["rows"]), dtype=float) / 20.0 - 2.0
    data = {"x": x, "y": float(line["intercept"]) + float(line["slope"]) * x}
    graph = ant.Cpdag.from_directed_undirected(["x", "y"], [], [("x", "y")])
    posterior = ant.discovery.GraphPosterior.from_graphs(["x", "y"], [1.0], [graph])
    prepared = ant.prepare(
        data,
        discovery=posterior,
        query=ant.Counterfactual("x", "y"),
        refute="none",
        inference=_FREQUENTIST,
        execution=ant.Execution(seed=3),
    )
    assert _coordinate(prepared) == "Counterfactual:Cpdag:graph_posterior:Frequentist:none"
    result = require_success(prepared.estimate())
    assert result.estimate.ate == pytest.approx(float(line["mean"]), abs=1e-6)
