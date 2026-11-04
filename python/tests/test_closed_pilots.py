"""Measured pilot scope boundaries and typed refusal precedence.

Success lifecycles and independent numerical oracles live in the measured-family
suites. These adjacent requests cannot borrow evidence from those exact protocols.
"""

import re
from types import SimpleNamespace

import pytest
from antecedent import Admg, Execution
from antecedent import _reason_details as reason_details
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
        "execution": Execution(seed=7),
    }
    kwargs.update(overrides)
    return kwargs


def test_joint_bayesian_in_scope_request_without_identification_asks_for_it():
    error = refusal_of(lambda: transport.joint_bayesian_transport(**joint_kwargs()))
    assert_refused(
        error,
        "transport_missing_evidence",
        reason_details.bayesian_transport.IDENTIFICATION_REQUIRED,
    )


def test_joint_bayesian_adjacent_graph_refuses_before_the_producer():
    adjacent = Admg.from_edges(["x", "a", "y"], [("x", "a"), ("a", "y")], [("x", "y")])
    error = refusal_of(lambda: transport.joint_bayesian_transport(**joint_kwargs(graph=adjacent)))
    assert_refused(
        error, "route_not_supported", reason_details.bayesian_transport.UNSUPPORTED_GRAPH
    )
    error = refusal_of(
        lambda: transport.joint_bayesian_transport(**joint_kwargs(graph_class="graph_posterior"))
    )
    assert_refused(
        error, "route_not_supported", reason_details.bayesian_transport.UNSUPPORTED_GRAPH
    )
    # A graph without a bidirected edge stays a fixed DAG: in scope, so it asks for identification.
    dag = Admg.from_edges(["x", "a", "y"], [("x", "a"), ("a", "y")])
    error = refusal_of(lambda: transport.joint_bayesian_transport(**joint_kwargs(graph=dag)))
    assert_refused(
        error,
        "transport_missing_evidence",
        reason_details.bayesian_transport.IDENTIFICATION_REQUIRED,
    )


def test_joint_bayesian_scope_refusals_carry_their_own_details():
    error = refusal_of(
        lambda: transport.joint_bayesian_transport(**joint_kwargs(draws=0)), CausalValueError
    )
    assert_refused(error, "invalid_argument", reason_details.bayesian_transport.TOO_MANY_DRAWS)
    error = refusal_of(
        lambda: transport.joint_bayesian_transport(**joint_kwargs(draws=100_001)), CausalValueError
    )
    assert_refused(error, "invalid_argument", reason_details.bayesian_transport.TOO_MANY_DRAWS)
    error = refusal_of(
        lambda: transport.joint_bayesian_transport(**joint_kwargs(dependence="overlapping_units"))
    )
    assert_refused(
        error, "sampling_dependence_unknown", reason_details.bayesian_transport.SOURCE_DEPENDENCE
    )
    error = refusal_of(lambda: transport.joint_bayesian_transport(**joint_kwargs(sources=[])))
    assert_refused(
        error, "transport_missing_evidence", reason_details.bayesian_transport.MISSING_SOURCE
    )
    error = refusal_of(lambda: transport.joint_bayesian_transport(**joint_kwargs(target=None)))
    assert_refused(error, "joint_law_required", reason_details.bayesian_transport.MISSING_LAW)
    many = [f"x{index}" for index in range(300)]
    error = refusal_of(
        lambda: transport.joint_bayesian_transport(**joint_kwargs(features=many)),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", reason_details.bayesian_transport.TOO_MANY_PARAMETERS)


def test_joint_bayesian_validates_its_arguments_as_typed_errors():
    with pytest.raises(CausalTypeError):
        transport.joint_bayesian_transport(**joint_kwargs(sources="s1"))
    with pytest.raises(CausalTypeError, match=re.escape("graph must be an Admg or None")):
        transport.joint_bayesian_transport(**joint_kwargs(graph="not a graph"))
    with pytest.raises(
        CausalValueError,
        match=re.escape("graph_class must be one of ('fixed_dag', 'admg', 'graph_posterior')"),
    ):
        transport.joint_bayesian_transport(**joint_kwargs(graph_class="cyclic"))
    with pytest.raises(CausalValueError, match=re.escape("features must not repeat a name")):
        transport.joint_bayesian_transport(**joint_kwargs(features=["x", "x"]))
    with pytest.raises(
        CausalValueError, match=re.escape("draws must be a non-negative integer, got -1")
    ):
        transport.joint_bayesian_transport(**joint_kwargs(draws=-1))


# ------------------------------------------------------------------ A3 nested Markov


def test_nested_bayesian_below_measured_row_range_is_annotated_not_borrowed():
    from antecedent.inference import MeasuredInference

    result = transport.binary_nested_markov(graph=verma(), regimes=[observational()])
    assert isinstance(result, MeasuredInference)
    assert [scalar.name for scalar in result.scalars] == ["mean0", "mean1", "contrast"]
    for scalar in result.scalars:
        assert scalar.calibration != "calibrated"
        assert scalar.interval[0] <= scalar.point <= scalar.interval[1]


@pytest.mark.requires_calibration
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
        assert_refused(error, "invalid_argument", reason_details.nested_markov.FISHER_INVALID_LEVEL)


def test_nested_markov_outside_class_requests_refuse_without_a_nonidentification_claim():
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
        assert_refused(
            error, "route_not_supported", reason_details.nested_markov.OUTSIDE_BINARY_PILOT
        )
        assert "nonidentif" not in str(error)


def test_nested_markov_invalid_counts_and_arguments_are_typed_errors():
    error = refusal_of(
        lambda: transport.binary_nested_markov(
            graph=verma(), regimes=[observational(cells=[-1.0] + CELLS[1:])]
        ),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", reason_details.nested_markov.INVALID_COUNTS)
    with pytest.raises(
        CausalTypeError, match=re.escape("binary_nested_markov requires graph=Admg(...)")
    ):
        transport.binary_nested_markov(graph="not a graph", regimes=[observational()])
    with pytest.raises(
        CausalTypeError, match=re.escape("a regime needs counts, a sequence of cell counts")
    ):
        transport.binary_nested_markov(graph=verma(), regimes=[{"counts": "many"}])
    with pytest.raises(CausalValueError, match=re.escape("tolerance must be finite and positive")):
        transport.binary_nested_markov(graph=verma(), regimes=[observational()], tolerance=0.0)
    with pytest.raises(
        CausalValueError, match=re.escape("intervened variable 'nope' is not in the graph")
    ):
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
        "replicates": 2000,
        "interval_method": "bootstrap_bca",
        "execution": Execution(seed=3),
    }
    kwargs.update(overrides)
    return kwargs


def test_retired_sampled_percentile_is_not_an_available_method():
    with pytest.raises(CausalValueError, match="interval_method must be bootstrap_bca"):
        transport.sampled_observation_recovery(
            **sampled_kwargs(interval_method="bootstrap_percentile")
        )


def test_sampled_recovery_zero_support_and_native_nonrecoverability_refuse():
    # Missing complete-case proxy cells give a zero denominator of the formula.
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(rows=complete_rows((0,))))
    )
    assert_refused(
        error, "route_not_supported", reason_details.sampled_recovery.UNRECOVERABLE_PATTERN
    )
    # An adjacent m-graph with a verified nonrecoverability witness has no recovered law.
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(
            **sampled_kwargs(stage=recovered_stage(self_censoring=True))
        )
    )
    assert_refused(
        error, "route_not_supported", reason_details.sampled_recovery.UNRECOVERABLE_PATTERN
    )


def test_sampled_recovery_bounds_and_malformed_rows_carry_their_own_details():
    # Any replicate count of at least two runs; a count the grid did not measure is annotated.
    fewer = transport.sampled_observation_recovery(**sampled_kwargs(replicates=500))
    assert fewer.calibration != "calibrated"
    with pytest.raises(CausalValueError, match="at least 2"):
        transport.sampled_observation_recovery(**sampled_kwargs(replicates=1))
    # The sealed artifact carries at most 2000 replicates: a larger count is a resource bound of
    # the export, not a measured-scope refusal.
    with pytest.raises(CausalValueError, match="export bounds"):
        transport.sampled_observation_recovery(**sampled_kwargs(replicates=2500))
    # A proxy value with no response is a pattern the proxy model excludes.
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(rows=[(0, 0, 1, 0)])),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", reason_details.sampled_recovery.INVALID_INPUT)
    duplicate = complete_rows() + [complete_rows()[0]]
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(rows=duplicate)),
        CausalValueError,
    )
    assert_refused(error, "invalid_argument", reason_details.sampled_recovery.INVALID_INPUT)
    error = refusal_of(
        lambda: transport.sampled_observation_recovery(**sampled_kwargs(rows=[])), CausalValueError
    )
    assert_refused(error, "invalid_argument", reason_details.sampled_recovery.INVALID_INPUT)


def test_sampled_recovery_validates_its_arguments_as_typed_errors():
    with pytest.raises(CausalTypeError):
        transport.sampled_observation_recovery(**sampled_kwargs(stage=object()))
    with pytest.raises(
        CausalTypeError, match=re.escape("query must be an ObservationRecoveryQuery")
    ):
        transport.sampled_observation_recovery(**sampled_kwargs(query="not a query"))
    with pytest.raises(
        CausalTypeError, match=re.escape("a row is (id, responses, proxies, fully)")
    ):
        transport.sampled_observation_recovery(**sampled_kwargs(rows=[(1, 2, 3)]))
    with pytest.raises(
        CausalValueError,
        match=re.escape("a row id must be in 0..=2**64-1 and each pattern field in 0..=255"),
    ):
        transport.sampled_observation_recovery(**sampled_kwargs(rows=[(0, 1, 300, 0)]))
    with pytest.raises(CausalValueError, match=re.escape("snapshot must name the rows' snapshot")):
        transport.sampled_observation_recovery(**sampled_kwargs(snapshot=" "))
    with pytest.raises(CausalValueError, match=re.escape("seed is outside its supported range")):
        transport.sampled_observation_recovery(**sampled_kwargs(execution=Execution(seed=-1)))


def test_scoped_pilot_routes_are_exported_from_the_advanced_namespace():
    for name in (
        "joint_bayesian_transport",
        "binary_nested_markov",
        "sampled_observation_recovery",
    ):
        assert name in transport.__all__
        assert callable(getattr(transport, name))


def test_sampled_bca_default_annotates_below_measured_rows_and_rejects_distinct_method():
    values = sampled_kwargs()
    values.pop("replicates")
    values.pop("interval_method")
    # Fewer rows than the measured range still report the interval; calibration annotates it.
    reported = transport.sampled_observation_recovery(**values)
    assert reported.calibration != "calibrated"
    # Another replicate count is computed; calibration annotates it as not measured.
    other = transport.sampled_observation_recovery(**values, replicates=500)
    assert other.calibration != "calibrated"
    with pytest.raises(CausalValueError, match="interval_method"):
        transport.sampled_observation_recovery(**values, interval_method="percentile_alias")


def test_sampled_recovery_rejects_fake_stage_before_calling_authority_callbacks():
    calls = []
    fake = SimpleNamespace(outcome="recovered", sampled_measured=lambda *args: calls.append(args))
    with pytest.raises(CausalTypeError, match="original native recovery stage"):
        transport.sampled_observation_recovery(**sampled_kwargs(stage=fake))
    assert calls == []
