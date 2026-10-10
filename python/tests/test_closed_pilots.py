"""Measured pilot scope boundaries and typed refusal precedence.

Success lifecycles and independent numerical oracles live in the measured-family
suites. These adjacent requests cannot borrow evidence from those exact protocols.
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


def test_nested_bayesian_below_measured_row_range_cannot_borrow_fisher_evidence():
    error = refusal_of(
        lambda: transport.binary_nested_markov(graph=verma(), regimes=[observational()])
    )
    assert_refused(error, "cell_not_licensed", "sample_size_outside_measured_range")


def test_nested_fisher_measured_interval_and_invalid_level():
    from antecedent.inference import MeasuredInference

    result = transport.binary_nested_markov_fisher_interval(
        graph=verma(), regimes=[observational()], nominal_level=0.95
    )
    assert isinstance(result, MeasuredInference)
    assert [scalar.name for scalar in result.scalars] == ["mean0", "mean1", "contrast"]
    for scalar in result.scalars:
        assert scalar.calibration == "calibrated"
        assert scalar.basis["scope"]["row_count"] == sum(CELLS)
        assert scalar.interval[0] <= scalar.point <= scalar.interval[1]
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
    from test_observation_recovery import query

    return query()


def recovered_stage(self_censoring=False):
    from test_observation_recovery import catalog, graph

    return transport.identify_observation_recovery(
        graph=graph(self_censoring=self_censoring),
        query=recovery_query(),
        catalog=catalog(),
        effect_outcomes=["y"],
        effect_treatments=["t"],
    )


def complete_rows(proxies=(0, 1, 2, 3)):
    """Positive complete-case cells for both missing variables and confounder.

    Two hundred rows stay deliberately below the measured 1000-row minimum;
    every full cell has 25 rows, so this is a protocol boundary, not zero support.
    """
    return [(index, 3, proxies[(index // 2) % len(proxies)], index % 2) for index in range(200)]


def sampled_kwargs(**overrides):
    kwargs = {
        "stage": recovered_stage(),
        "query": recovery_query(),
        "rows": complete_rows(),
        "snapshot": "snap-1",
        "replicates": 100,
        "interval_method": "bootstrap_percentile",
        "seed": 3,
    }
    kwargs.update(overrides)
    return kwargs


def test_sampled_percentile_cannot_borrow_measured_bca_evidence():
    error = refusal_of(lambda: transport.sampled_observation_recovery(**sampled_kwargs()))
    assert_refused(error, "cell_not_licensed", "sampled_recovery.protocol_not_measured")


def test_sampled_recovery_zero_support_and_native_nonrecoverability_refuse():
    # Missing complete-case proxy cells give a zero denominator of the formula.
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(rows=complete_rows((0,))))
    )
    assert_refused(error, "route_not_supported", "sampled_recovery.unrecoverable_pattern")
    # An adjacent m-graph with a verified nonrecoverability witness has no recovered law.
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(
            **sampled_kwargs(stage=recovered_stage(self_censoring=True))
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
    duplicate = complete_rows() + [complete_rows()[0]]
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


def test_scoped_pilot_routes_are_exported_from_the_advanced_namespace():
    for name in (
        "joint_bayesian_transport",
        "binary_nested_markov",
        "sampled_observation_recovery",
    ):
        assert name in transport.__all__
        assert callable(getattr(transport, name))


def test_sampled_bca_default_rejects_below_measured_rows_and_distinct_method():
    values = sampled_kwargs()
    values.pop("replicates")
    values.pop("interval_method")
    error = refusal_of(lambda: transport.sampled_observation_recovery(**values))
    assert_refused(error, "cell_not_licensed", "sampled_recovery.protocol_not_measured")
    with pytest.raises(CausalValueError, match="exactly 2000"):
        transport.sampled_observation_recovery(**values, replicates=500)
    with pytest.raises(CausalValueError, match="interval_method"):
        transport.sampled_observation_recovery(**values, interval_method="percentile_alias")


def test_sampled_recovery_rejects_fake_stage_before_calling_authority_callbacks():
    calls = []
    fake = SimpleNamespace(outcome="recovered", sampled_measured=lambda *args: calls.append(args))
    with pytest.raises(CausalTypeError, match="original native recovery stage"):
        transport.sampled_observation_recovery(**sampled_kwargs(stage=fake))
    assert calls == []
