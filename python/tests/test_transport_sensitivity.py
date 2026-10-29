"""The assumption range of a transported response through the Python surface: built from the
response's own point value, quantity and receipt, sealed into the one sensitivity artifact, and
never inherited by another target.

Oracle (the Python spelling of ``crates/antecedent/tests/transport_sensitivity_facade.rs``). The
transported mean of ``effect`` in ``clinic_b`` is ``2.0``. An additive shift bound of ``0.5`` on
three grid points is ``0, 0.25, 0.5``, so the range is ``[2, 2]``, ``[1.75, 2.25]``,
``[1.5, 2.5]``. A proportional bound of ``0.25`` of ``2.0`` is a shift of ``0.5`` at the bound:
``[1.5, 2.5]``.
"""

from __future__ import annotations

import dataclasses

import pytest
from antecedent import cohort_decision as cd
from antecedent import sensitivity_decision as sd
from antecedent import transported_response as tr
from antecedent.compact_export import CompactExport, Intercept, Linear, Numeric, Quantity
from antecedent.errors import CausalTypeError
from antecedent.outcomes import require_success
from antecedent.transport import sensitivity as ts

from _refusal import assert_registered_refusal
from _transport_fixtures import effect, point, transported_law

TOL = 1e-12


def sensitivity(response, *, family="additive_shift", bound=0.5, reference_value=2.2, **options):
    options.setdefault("route", "cohort_decision")
    return ts.transport_sensitivity(
        response,
        family=family,
        bound=bound,
        grid_points=3,
        reference=effect(response.coordinates[0].population_id, "keep"),
        reference_value=reference_value,
        **options,
    )


def refusal_of(call) -> tr.TransportedResponseRefusal:
    with pytest.raises(tr.TransportedResponseRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def test_the_range_is_built_from_the_responses_value_quantity_and_receipt() -> None:
    result = sensitivity(point(2.0))
    assert result.label == "assumption_range"
    assert result.route == "cohort_decision"
    assert result.premise == "selection_diagram_invariance"
    assert result.point_value == 2.0
    assert result.grid == pytest.approx((0.0, 0.25, 0.5), abs=TOL)
    assert result.lower == pytest.approx((2.0, 1.75, 1.5), abs=TOL)
    assert result.upper == pytest.approx((2.0, 2.25, 2.5), abs=TOL)
    assert result.range_at_bound == pytest.approx((1.5, 2.5), abs=TOL)
    assert result.statement

    proportional = sensitivity(point(2.0), family="proportional_shift", bound=0.25)
    assert proportional.range_at_bound == pytest.approx((1.5, 2.5), abs=TOL)

    # The smallest shift at which the range reaches the reference is a closed form on the
    # declared family: |2.0 - 2.2| = 0.2 additive; 0.2 / 2 = 0.1 proportional.
    assert result.reference_reached_at == pytest.approx(0.2, abs=TOL)
    assert proportional.reference_reached_at == pytest.approx(0.1, abs=TOL)
    # A reference further than the bound is not reached inside it.
    assert sensitivity(point(2.0), reference_value=9.0).reference_reached_at is None


def test_the_sealed_artifact_exports_and_a_consumer_recomputes_it() -> None:
    result = sensitivity(point(2.0), reference_value=1.0)
    artifact = result.decision_source
    assert isinstance(artifact, sd.SensitivityArtifact)
    assert artifact.grid == pytest.approx((0.0, 0.25, 0.5), abs=TOL)
    # The reference 1.0 lies below the whole range [1.5, 2.5]: acting is the invariant best.
    assert artifact.outcome is not None
    assert artifact.outcome.kind == "invariant_action"

    data = bytes(artifact.export(artifact_id="transport-sensitivity"))
    consumed = require_success(
        sd.SensitivityArtifact.consume(data, expected_identity=artifact.identity)
    )
    assert consumed.identity == artifact.identity

    # The same family on a response about another target is another range with another identity:
    # an assumption range is never inherited by another target.
    other = sensitivity(point(2.0, "clinic_c"), reference_value=1.0)
    assert other.identity != result.identity
    assert other.decision_source.identity != artifact.identity
    # And a different point value at the same declared family moves the range.
    moved = sensitivity(point(3.0))
    assert moved.lower == pytest.approx((3.0, 2.75, 2.5), abs=TOL)
    # A reference above the range flips the invariant action.
    above = sensitivity(point(2.0), reference_value=3.0)
    assert above.decision_source.outcome is not None
    assert above.decision_source.outcome.kind == "invariant_action"
    assert above.decision_source.outcome.action != artifact.outcome.action


def test_a_response_that_cannot_be_shifted_is_refused_with_its_own_reason() -> None:
    response = point(2.0)
    # A declared transport names no route of its own.
    undeclared = refusal_of(lambda: sensitivity(response, route=None))
    assert undeclared.reason_code == "route_not_supported"
    assert undeclared.detail == "transport_sensitivity.route_undeclared"

    # A coordinate outside the response.
    outside = refusal_of(lambda: sensitivity(response, coordinate=3))
    assert outside.reason_code == "invalid_argument"
    assert outside.detail == "transport_sensitivity.coordinate_out_of_range"

    # A structural premise has no scalar family.
    structural = refusal_of(lambda: sensitivity(response, route="discovered_structure"))
    assert structural.detail == "transport_sensitivity.no_scalar_family"

    # A proportional shift of zero is no range.
    zero = refusal_of(lambda: sensitivity(point(0.0), family="proportional_shift", bound=0.5))
    assert zero.reason_code == "invalid_argument"
    assert zero.detail == "transport_sensitivity.proportional_at_zero"


def test_a_transported_law_has_no_single_point_to_shift() -> None:
    law = refusal_of(lambda: sensitivity(transported_law()))
    assert law.reason_code == "route_not_supported"
    assert law.detail == "transport_sensitivity.law_response_not_scalar"
    assert law.remedy


def test_only_a_transported_response_has_an_assumption_range() -> None:
    with pytest.raises(CausalTypeError):
        ts.transport_sensitivity(
            object(),  # type: ignore[arg-type]
            family="additive_shift",
            bound=0.5,
            grid_points=3,
            reference=effect(),
            reference_value=1.0,
        )


def test_a_shifted_cohort_response_names_its_own_sensitivity_route() -> None:
    # tau(z) = 2 + 0.5 z over the cohort z = (-1, 0.5, 1) with weights (0.2, 0.3, 0.5): the
    # weighted mean is 0.2 * 1.5 + 0.3 * 2.25 + 0.5 * 2.5 = 2.225 in clinic_b.
    export = CompactExport.build(
        response=Quantity(
            "y",
            units="mmHg",
            role="outcome",
            regime="contrast:do(t=1)-do(t=0)",
            functional="conditional_mean_difference",
        ),
        inputs=[Numeric("z", -1.0, 1.0, units="1")],
        terms=[Intercept(), Linear("z")],
        coefficients=[2.0, 0.5],
        covariance=[[0.04, -0.01], [-0.01, 0.09]],
    )
    cohort = cd.decision_source_from_compact_export(
        export,
        cohort=cd.CohortBinding(
            snapshot_id="clinic-2026-q1",
            population="clinic_b",
            columns={"z": [-1.0, 0.5, 1.0]},
            weights=[0.2, 0.3, 0.5],
        ),
        outcome_units="mmHg",
        shift=cd.CovariateShift("target", "clinic_b", ("z",)),
    )
    response = tr.TransportedResponse.from_shifted_cohort(cohort)
    assert response.receipt.route == "shifted_cohort"
    reference = dataclasses.replace(
        response.coordinates[0], variable_id="keep", variable_name="keep"
    )
    # No route is named: the response's own route implies the cohort-decision premise.
    result = ts.transport_sensitivity(
        response,
        family="additive_shift",
        bound=0.5,
        grid_points=3,
        reference=reference,
        reference_value=2.4,
    )
    assert result.route == "cohort_decision"
    assert result.premise == "selection_diagram_invariance"
    assert result.point_value == pytest.approx(2.225, abs=TOL)
    assert result.range_at_bound == pytest.approx((1.725, 2.725), abs=TOL)
    assert result.reference_reached_at == pytest.approx(0.175, abs=TOL)
