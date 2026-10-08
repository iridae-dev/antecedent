"""2.3A closed calibrated-interval pilots: joint Bayesian transport (A2), the binary
nested-Markov pilot (A3) and sampled observation recovery (A6).

Each public route is closed until calibration is measured at the release cut: it validates
its request against the pilot's scope through the Rust core, raises the pilot's own scope
refusal for a request outside the pilot, and otherwise raises ``cell_not_licensed`` with the
route-frozen detail. The Rust engines and their io artifacts are tested in Rust.
"""

from types import SimpleNamespace

import pytest
from antecedent import Admg
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.transport import advanced as transport

ObservationRecoveryQuery = transport.ObservationRecoveryQuery
PartiallyObservedVariable = transport.PartiallyObservedVariable

NODES = ["X1", "X2", "X3", "X4"]
VERMA_DIRECTED = [("X1", "X2"), ("X2", "X3"), ("X3", "X4")]
VERMA_BIDIRECTED = [("X2", "X4")]
CELLS = [100.0 + index for index in range(16)]


def verma() -> Admg:
    return Admg.from_edges(NODES, VERMA_DIRECTED, VERMA_BIDIRECTED)


def observational(cells=None, levels=None):
    regime = {"counts": CELLS if cells is None else cells}
    if levels is not None:
        regime["levels"] = levels
    return regime


def refusal_of(call, error=CausalUnsupportedError):
    with pytest.raises(error) as caught:
        call()
    return caught.value


def assert_refused(error, code, detail):
    assert error.reason_code == code
    assert detail in str(error)


# ------------------------------------------------------------------ A2 joint Bayesian


def joint_kwargs(**overrides):
    kwargs = {
        "sources": [{"id": "s1"}, {"id": "s2"}],
        "target": {"x": [0.1, 0.2, 0.3]},
        "features": ["x"],
        "draws": 1000,
        "seed": 7,
    }
    kwargs.update(overrides)
    return kwargs


def test_x4_joint_bayesian_route_is_closed_with_cell_not_licensed():
    error = refusal_of(lambda: transport.joint_bayesian_transport(**joint_kwargs()))
    assert_refused(error, "cell_not_licensed", "bayesian_transport.route_frozen")


def test_x4_joint_bayesian_adjacent_graph_refuses_before_the_frozen_route():
    adjacent = Admg.from_edges(["x", "a", "y"], [("x", "a"), ("a", "y")], [("x", "y")])
    error = refusal_of(lambda: transport.joint_bayesian_transport(**joint_kwargs(graph=adjacent)))
    assert_refused(error, "route_not_supported", "bayesian_transport.unsupported_graph")
    error = refusal_of(
        lambda: transport.joint_bayesian_transport(**joint_kwargs(graph_class="graph_posterior"))
    )
    assert_refused(error, "route_not_supported", "bayesian_transport.unsupported_graph")
    # A graph without a bidirected edge stays a fixed DAG: the frozen refusal, not the scope one.
    dag = Admg.from_edges(["x", "a", "y"], [("x", "a"), ("a", "y")])
    error = refusal_of(lambda: transport.joint_bayesian_transport(**joint_kwargs(graph=dag)))
    assert_refused(error, "cell_not_licensed", "bayesian_transport.route_frozen")


def test_x4_joint_bayesian_scope_refusals_carry_their_own_details():
    error = refusal_of(
        lambda: transport.joint_bayesian_transport(**joint_kwargs(draws=0)), CausalValueError
    )
    assert_refused(error, "invalid_argument", "bayesian_transport.too_many_draws")
    error = refusal_of(
        lambda: transport.joint_bayesian_transport(**joint_kwargs(draws=100_001)), CausalValueError
    )
    assert_refused(error, "invalid_argument", "bayesian_transport.too_many_draws")
    error = refusal_of(
        lambda: transport.joint_bayesian_transport(**joint_kwargs(dependence="overlapping_units"))
    )
    assert_refused(error, "sampling_dependence_unknown", "bayesian_transport.source_dependence")
    error = refusal_of(lambda: transport.joint_bayesian_transport(**joint_kwargs(sources=[])))
    assert_refused(error, "transport_missing_evidence", "bayesian_transport.missing_source")
    error = refusal_of(lambda: transport.joint_bayesian_transport(**joint_kwargs(target=None)))
    assert_refused(error, "joint_law_required", "bayesian_transport.missing_law")
    many = [f"x{index}" for index in range(300)]
    error = refusal_of(
        lambda: transport.joint_bayesian_transport(**joint_kwargs(features=many)),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", "bayesian_transport.too_many_parameters")


def test_x4_joint_bayesian_validates_its_arguments_as_typed_errors():
    with pytest.raises(CausalTypeError):
        transport.joint_bayesian_transport(**joint_kwargs(sources="s1"))
    with pytest.raises(CausalTypeError):
        transport.joint_bayesian_transport(**joint_kwargs(graph="not a graph"))
    with pytest.raises(CausalValueError):
        transport.joint_bayesian_transport(**joint_kwargs(graph_class="cyclic"))
    with pytest.raises(CausalValueError):
        transport.joint_bayesian_transport(**joint_kwargs(features=["x", "x"]))
    with pytest.raises(CausalValueError):
        transport.joint_bayesian_transport(**joint_kwargs(draws=-1))


# ------------------------------------------------------------------ A3 nested Markov


def test_x4_nested_markov_route_is_closed_with_cell_not_licensed():
    error = refusal_of(
        lambda: transport.binary_nested_markov(graph=verma(), regimes=[observational()])
    )
    assert_refused(error, "cell_not_licensed", "nested_markov.route_frozen")


def test_nested_fisher_public_interval_remains_closed_and_validates_level():
    error = refusal_of(
        lambda: transport.binary_nested_markov_fisher_interval(
            graph=verma(), regimes=[observational()], nominal_level=0.95
        )
    )
    assert_refused(error, "cell_not_licensed", "nested_markov.route_frozen")
    for level in [0.0, 1.0, float("nan")]:
        error = refusal_of(
            lambda level=level: transport.binary_nested_markov_fisher_interval(
                graph=verma(), regimes=[observational()], nominal_level=level
            ),
            CausalValueError,
        )
        assert_refused(error, "invalid_argument", "nested_markov.fisher_invalid_level")


def test_x4_nested_markov_outside_class_requests_refuse_without_a_nonidentification_claim():
    adjacent = Admg.from_edges(NODES, VERMA_DIRECTED, [("X1", "X4")])
    cases = [
        {"graph": adjacent, "regimes": [observational()]},
        {"graph": Admg.from_edges(NODES, VERMA_DIRECTED), "regimes": [observational()]},
        {"graph": verma(), "regimes": [{"counts": CELLS, "intervened": ["X2"]}]},
        {"graph": verma(), "regimes": [observational(levels=[2, 3, 2, 2])]},
        {"graph": verma(), "regimes": [observational(cells=[1.0] * 81, levels=[3, 3, 3, 3])]},
        {"graph": verma(), "regimes": [observational(), observational()]},
        {"graph": verma(), "regimes": [observational(cells=[1.0] * 8)]},
    ]
    for case in cases:
        error = refusal_of(lambda case=case: transport.binary_nested_markov(**case))
        assert_refused(error, "route_not_supported", "nested_markov.outside_binary_pilot")
        assert "nonidentif" not in str(error)


def test_x4_nested_markov_invalid_counts_and_arguments_are_typed_errors():
    error = refusal_of(
        lambda: transport.binary_nested_markov(
            graph=verma(), regimes=[observational(cells=[-1.0] + CELLS[1:])]
        ),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", "nested_markov.invalid_counts")
    with pytest.raises(CausalTypeError):
        transport.binary_nested_markov(graph="not a graph", regimes=[observational()])
    with pytest.raises(CausalTypeError):
        transport.binary_nested_markov(graph=verma(), regimes=[{"counts": "many"}])
    with pytest.raises(CausalValueError):
        transport.binary_nested_markov(graph=verma(), regimes=[observational()], tolerance=0.0)
    with pytest.raises(CausalValueError):
        transport.binary_nested_markov(
            graph=verma(), regimes=[{"counts": CELLS, "intervened": ["nope"]}]
        )


# ------------------------------------------------------------------ A6 sampled recovery


def recovery_query() -> ObservationRecoveryQuery:
    return ObservationRecoveryQuery(
        population="clinic",
        observed_regime="observed",
        partially_observed=[PartiallyObservedVariable("X", "R", "X_star")],
    )


def recovered_stage():
    return SimpleNamespace(outcome="recovered")


def complete_rows(proxies=(0, 1)):
    """Rows of one partially observed variable: ``R = 1`` with each listed proxy value."""
    return [(index, 1, proxies[index % len(proxies)], 0) for index in range(40)]


def sampled_kwargs(**overrides):
    kwargs = {
        "stage": recovered_stage(),
        "query": recovery_query(),
        "rows": complete_rows(),
        "snapshot": "snap-observed",
        "replicates": 100,
        "seed": 3,
    }
    kwargs.update(overrides)
    return kwargs


def test_x10_sampled_recovery_route_is_closed_with_cell_not_licensed():
    error = refusal_of(lambda: transport.sampled_observation_recovery(**sampled_kwargs()))
    assert_refused(error, "cell_not_licensed", "sampled_recovery.route_frozen")


def test_x10_sampled_recovery_unrecoverable_inputs_refuse_before_the_frozen_route():
    # A zero complete-case cell (no row with X* = 1) is a zero denominator of the formula.
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(rows=complete_rows((0,))))
    )
    assert_refused(error, "route_not_supported", "sampled_recovery.unrecoverable_pattern")
    # An adjacent m-graph with a verified nonrecoverability witness has no recovered law.
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(
            **sampled_kwargs(stage=SimpleNamespace(outcome="nonrecoverable"))
        )
    )
    assert_refused(error, "route_not_supported", "sampled_recovery.unrecoverable_pattern")


def test_x10_sampled_recovery_bounds_and_malformed_rows_carry_their_own_details():
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(replicates=5000))
    )
    assert_refused(error, "route_not_supported", "sampled_recovery.bounds_exceeded")
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(replicates=5)),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", "sampled_recovery.invalid_input")
    # A proxy value with no response is a pattern the proxy model excludes.
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(rows=[(0, 0, 1, 0)])),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", "sampled_recovery.invalid_input")
    duplicate = complete_rows() + [(0, 1, 0, 0)]
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(rows=duplicate)),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", "sampled_recovery.invalid_input")
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(rows=[])), CausalValueError
    )
    assert_refused(error, "invalid_argument", "sampled_recovery.invalid_input")


def test_x10_sampled_recovery_validates_its_arguments_as_typed_errors():
    with pytest.raises(CausalTypeError):
        transport.sampled_observation_recovery(**sampled_kwargs(stage=object()))
    with pytest.raises(CausalTypeError):
        transport.sampled_observation_recovery(**sampled_kwargs(query="not a query"))
    with pytest.raises(CausalTypeError):
        transport.sampled_observation_recovery(**sampled_kwargs(rows=[(1, 2, 3)]))
    with pytest.raises(CausalValueError):
        transport.sampled_observation_recovery(**sampled_kwargs(rows=[(0, 1, 300, 0)]))
    with pytest.raises(CausalValueError):
        transport.sampled_observation_recovery(**sampled_kwargs(snapshot=" "))
    with pytest.raises(CausalValueError):
        transport.sampled_observation_recovery(**sampled_kwargs(seed=-1))


def test_x10_closed_pilot_routes_are_exported_from_the_advanced_namespace():
    for name in (
        "joint_bayesian_transport",
        "binary_nested_markov",
        "sampled_observation_recovery",
    ):
        assert name in transport.__all__
        assert callable(getattr(transport, name))
