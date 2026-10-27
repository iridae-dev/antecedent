"""One physical policy, with actual finite evaluator bounds on typed questions."""

import dataclasses
import inspect

import antecedent as ant
import pytest
from antecedent import transport
from antecedent.errors import CausalCancelledError, CausalResourceError, CausalTypeError
from antecedent.outcomes import require_success
from antecedent.transport import advanced

from test_canonical_trial_transport import fixture as trial_fixture
from test_transport_exact import fixture as exact_fixture
from test_transport_statistical import fixture as statistical_fixture


def test_canonical_transport_has_no_parallel_physical_controls_location():
    assert "controls" not in inspect.signature(ant.prepare).parameters
    assert not hasattr(transport, "TransportControls")
    with pytest.raises(TypeError, match="controls"):
        ant.prepare({}, query=object(), controls=object())
    query, data = trial_fixture()
    study = ant.prepare(
        data,
        query=query,
        inference=transport.TransportInference(bootstrap=0),
        execution=ant.Execution(seed=8),
    )
    assert require_success(study.estimate()).estimate == pytest.approx(2.0, abs=1e-10)
    assert study.execution.seed == 8


def test_exact_query_preserves_distribution_and_declared_evaluator_bounds():
    identified, catalog, data = exact_fixture()
    evaluation = transport.TransportEvaluation(
        max_operations=10_000_000, max_depth=256, max_support_rows=1_000_000
    )
    query = advanced.ExactTransportQuery(identified, catalog, {"x": 1.0}, evaluation=evaluation)
    study = ant.prepare(data, query=query)
    result = require_success(study.estimate())
    # Supplied joint: P(Y=1 | X=1) = .35 / (.15 + .35).
    assert result.mean("y") == pytest.approx(0.7)
    assert result.probabilities == pytest.approx((0.3, 0.7))
    reference = require_success(advanced.evaluate_exact(identified, catalog, data, at={"x": 1.0}))
    assert bytes(result.export()) == bytes(reference.export())
    assert study.query.evaluation == evaluation


def test_statistical_support_bound_reaches_the_real_native_row_predicate():
    identified, catalog, data = statistical_fixture()
    query = advanced.StatisticalTransportQuery(
        identified,
        catalog,
        {"x": 1.0},
        bootstrap=0,
        evaluation=transport.TransportEvaluation(max_support_rows=1),
    )
    with pytest.raises(CausalResourceError, match="max_support_rows"):
        ant.prepare(data, query=query)
    allowed = dataclasses.replace(query, evaluation=transport.TransportEvaluation())
    result = require_success(ant.estimate(ant.prepare(data, query=allowed)))
    # The supplied source experiment contains 80 successes in 100 rows.
    assert result.mean("y") == pytest.approx(0.8)
    assert not result.uncertainty["available"]


def _grid_catalog(catalog):
    """A grid evaluates declared finite worlds: the catalog declares each variable's domain."""
    return dataclasses.replace(
        catalog,
        environments=[
            advanced.Environment(
                "target",
                [advanced.VariableCoordinate(name, "binary") for name in ("x", "y")],
            )
        ],
    )


def test_grid_keeps_ordered_worlds_and_its_joint_native_result():
    identified, catalog, data = exact_fixture()
    query = advanced.TransportResponseGridQuery(
        identified,
        _grid_catalog(catalog),
        ({"x": 0.0}, {"x": 1.0}),
        evaluation=transport.TransportEvaluation(),
    )
    result = require_success(ant.estimate(ant.prepare(data, query=query)))
    # Independently condition the supplied two rows for each treatment level.
    assert [result.mean(index, "y") for index in range(2)] == pytest.approx((0.2, 0.7))
    assert result.contrast(1, 0, "y").estimate == pytest.approx(0.5)


def test_typed_statistical_provider_and_inference_still_reach_native_plan():
    identified, catalog, data = statistical_fixture()
    query = advanced.StatisticalTransportQuery(identified, catalog, {"x": 1.0}, bootstrap=5)
    study = ant.prepare(
        data,
        query=query,
        provider=transport.EmpiricalTable(),
        inference=transport.TransportInference(bootstrap=0),
        execution=ant.Execution(seed=19),
    )
    result = require_success(study.estimate())
    assert result.mean("y") == pytest.approx(0.8)
    assert not result.uncertainty["available"]
    assert study.execution.seed == 19


@pytest.mark.parametrize("family", ["exact", "statistical", "grid", "trial"])
def test_execution_owns_cancellation_for_every_canonical_transport_modality(family):
    from antecedent.state import CancellationToken

    if family == "trial":
        query, data = trial_fixture()
    elif family == "statistical":
        identified, catalog, data = statistical_fixture()
        query = advanced.StatisticalTransportQuery(identified, catalog, {"x": 1.0}, bootstrap=0)
    else:
        identified, catalog, data = exact_fixture()
        query = (
            advanced.ExactTransportQuery(identified, catalog, {"x": 1.0})
            if family == "exact"
            else advanced.TransportResponseGridQuery(
                identified, _grid_catalog(catalog), ({"x": 0.0}, {"x": 1.0})
            )
        )
    token = CancellationToken()
    study = ant.prepare(data, query=query, execution=ant.Execution(cancel=token))
    token.cancel()
    with pytest.raises(CausalCancelledError):
        ant.estimate(study)
    # Replacing the physical policy explicitly removes the earlier cancellation.
    independent = ant.reconfigure(study, execution=ant.Execution())
    result = require_success(independent.estimate())
    assert (
        result.estimate
        if family == "trial"
        else result.mean(0, "y")
        if family == "grid"
        else result.mean("y")
    ) == pytest.approx(
        2.0
        if family == "trial"
        else 0.2
        if family == "grid"
        else 0.8
        if family == "statistical"
        else 0.7
    )


def test_evaluation_declaration_rejects_invalid_arguments_without_execution():
    with pytest.raises(CausalTypeError, match="integer"):
        transport.TransportEvaluation(max_depth=True)
    with pytest.raises(CausalTypeError, match="integer"):
        transport.TransportEvaluation(max_depth=None)
