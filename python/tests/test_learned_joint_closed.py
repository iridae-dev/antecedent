"""2.3 learned joint source-target transport: the closed route (Python surface).

The Rust core fits the outcome mechanism through ``antecedent-learn`` and replays through an
independent io artifact, but its posterior claims a calibrated interval whose coverage is
measured only at the release cut. The public producer therefore validates its request against
the row's scope through the Rust core, raises the row's own scope refusal for a request outside
it, and otherwise raises ``cell_not_licensed`` / ``learned_joint_transport.route_frozen``.
"""

import pytest
from antecedent import Admg
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.transport import advanced as transport


def refusal_of(call, error=CausalUnsupportedError):
    with pytest.raises(error) as caught:
        call()
    return caught.value


def assert_refused(error, code, detail):
    assert error.reason_code == code
    assert detail in str(error)


def learned_kwargs(**overrides):
    kwargs = {
        "sources": [{"id": "s1"}, {"id": "s2"}],
        "target": {"x": [0.1, 0.2, 0.3]},
        "features": ["x"],
        "basis_degree": 2,
        "draws": 1000,
        "seed": 7,
    }
    kwargs.update(overrides)
    return kwargs


def test_a2_learned_route_is_closed_with_cell_not_licensed():
    error = refusal_of(lambda: transport.learned_joint_transport(**learned_kwargs()))
    assert_refused(error, "cell_not_licensed", "learned_joint_transport.route_frozen")
    assert "calibration evidence" in str(error)


def test_a2_learned_adjacent_graph_refuses_before_the_frozen_route():
    adjacent = Admg.from_edges(["x", "a", "y"], [("x", "a"), ("a", "y")], [("x", "y")])
    error = refusal_of(lambda: transport.learned_joint_transport(**learned_kwargs(graph=adjacent)))
    assert_refused(error, "route_not_supported", "learned_joint_transport.unsupported_graph")
    error = refusal_of(
        lambda: transport.learned_joint_transport(**learned_kwargs(graph_class="graph_posterior"))
    )
    assert_refused(error, "route_not_supported", "learned_joint_transport.unsupported_graph")
    # A graph without a bidirected edge stays a fixed DAG: the frozen refusal, not the scope one.
    dag = Admg.from_edges(["x", "a", "y"], [("x", "a"), ("a", "y")])
    error = refusal_of(lambda: transport.learned_joint_transport(**learned_kwargs(graph=dag)))
    assert_refused(error, "cell_not_licensed", "learned_joint_transport.route_frozen")


def test_a2_learned_bad_basis_refuses_with_its_own_detail_first():
    for degree in (0, 7, 100):
        error = refusal_of(
            lambda degree=degree: transport.learned_joint_transport(
                **learned_kwargs(basis_degree=degree)
            ),
            CausalValueError,
        )
        assert_refused(error, "invalid_argument", "learned_joint_transport.invalid_basis")
    # The basis refusal precedes the draw-count refusal; the graph refusal precedes the basis.
    error = refusal_of(
        lambda: transport.learned_joint_transport(**learned_kwargs(basis_degree=0, draws=0)),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", "learned_joint_transport.invalid_basis")
    error = refusal_of(
        lambda: transport.learned_joint_transport(
            **learned_kwargs(basis_degree=0, graph_class="admg")
        )
    )
    assert_refused(error, "route_not_supported", "learned_joint_transport.unsupported_graph")
    # Every supported degree reaches the frozen route.
    for degree in range(1, 7):
        error = refusal_of(
            lambda degree=degree: transport.learned_joint_transport(
                **learned_kwargs(basis_degree=degree)
            )
        )
        assert_refused(error, "cell_not_licensed", "learned_joint_transport.route_frozen")


def test_a2_learned_scope_refusals_carry_their_own_details():
    for draws in (0, 100_001):
        error = refusal_of(
            lambda draws=draws: transport.learned_joint_transport(**learned_kwargs(draws=draws)),
            CausalValueError,
        )
        assert_refused(error, "invalid_argument", "learned_joint_transport.too_many_draws")
    for dependence in ("overlapping_units", "unknown"):
        error = refusal_of(
            lambda dependence=dependence: transport.learned_joint_transport(
                **learned_kwargs(dependence=dependence)
            )
        )
        assert_refused(
            error, "sampling_dependence_unknown", "learned_joint_transport.source_dependence"
        )
    error = refusal_of(lambda: transport.learned_joint_transport(**learned_kwargs(sources=[])))
    assert_refused(error, "transport_missing_evidence", "learned_joint_transport.missing_source")
    error = refusal_of(lambda: transport.learned_joint_transport(**learned_kwargs(target=None)))
    assert_refused(error, "joint_law_required", "learned_joint_transport.missing_law")
    error = refusal_of(lambda: transport.learned_joint_transport(**learned_kwargs(target={})))
    assert_refused(error, "joint_law_required", "learned_joint_transport.missing_law")
    many = [f"x{index}" for index in range(300)]
    error = refusal_of(
        lambda: transport.learned_joint_transport(**learned_kwargs(features=many)),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", "learned_joint_transport.too_many_parameters")


def test_a2_learned_validates_its_arguments_as_typed_errors():
    with pytest.raises(CausalTypeError):
        transport.learned_joint_transport(**learned_kwargs(sources="s1"))
    with pytest.raises(CausalTypeError):
        transport.learned_joint_transport(**learned_kwargs(graph="not a graph"))
    with pytest.raises(CausalTypeError):
        transport.learned_joint_transport(**learned_kwargs(basis_degree=1.5))
    with pytest.raises(CausalValueError):
        transport.learned_joint_transport(**learned_kwargs(graph_class="cyclic"))
    with pytest.raises(CausalValueError):
        transport.learned_joint_transport(**learned_kwargs(features=["x", "x"]))
    with pytest.raises(CausalValueError):
        transport.learned_joint_transport(**learned_kwargs(draws=-1))
    with pytest.raises(CausalValueError):
        transport.learned_joint_transport(**learned_kwargs(basis_degree=-1))
