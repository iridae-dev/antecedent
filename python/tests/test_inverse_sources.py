"""An inverse query reaches the composed, cohort and transported forward sources and the
associational observational route through the Python surface, each with its own refusals and its
identity carried into the result and the sealed artifact.

Oracles (hand-derived, the Python spelling of
``crates/antecedent/tests/inverse_forward_sources_facade.rs``):

- Linear-Gaussian chain ``X = 1 + 2 e``, ``M = 0.5 + 2 X + e``, ``Y = -1 + 0.5 X + 3 M + 0.5 e``:
  ``E[Y | do(X = 0, 1, 2)] = 0.5, 7, 13.5``; with ``M`` refit as ``1 + 3 X``,
  ``E[Y | do(X = 1)] = 11.5``.
- Binary chain: ``P(Y = 1 | do(X = 0)) = 0.18`` and ``P(Y = 1 | do(X = 1)) = 0.72``.
- Eight observed units ``(x, z, y)`` with ``y = 1`` in rows 2, 5, 6 and 8 and ``x = 1`` in rows
  5-8: ``P(E) = 1/2``, ``P(x = 1) = 1/2``, ``P(E | x = 1) = 3/4``, ``P(x = 1 | E) = 3/4``.
- A transported point ``(1, 3)`` against a target mean of 2 keeps only the second action.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import cohort_decision as cd
from antecedent import decision
from antecedent import inverse_query as iq
from antecedent import inverse_sources as src
from antecedent import mechanism_composition as mc
from antecedent import transported_response as tr
from antecedent.compact_export import CompactExport, Intercept, Linear, Numeric, Quantity
from antecedent.errors import CausalTypeError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)

from _refusal import assert_registered_refusal
from _transport_fixtures import (
    compose,
    finite_chain,
    gaussian_chain,
    identity_contract,
    linear,
    outcome_under,
    point,
    transport_declaration,
)

TOL = 1e-9


def query(arms, constraints, *, functional: str = "mean") -> iq.InverseQuery:
    return iq.InverseQuery(
        identity_contract(arms, functional),
        tuple(f"x{arm}" for arm in arms),
        tuple(constraints),
    )


def gaussian_actions() -> list[src.ComposedAction]:
    return [
        src.ComposedAction(
            f"x{arm}", outcome_under(arm), mc.MeanContrast.mean_at("Y", {"X": float(arm)})
        )
        for arm in ("0", "1", "2")
    ]


def finite_actions(levels: dict[str, float]) -> list[src.ComposedAction]:
    return [
        src.ComposedAction(
            f"x{arm}",
            outcome_under(arm, "outcome"),
            {"X": arm},
            outcome="Y",
            level_values=levels,
        )
        for arm in ("0", "1")
    ]


def refusal_of(call) -> iq.InverseQueryRefusal:
    with pytest.raises(iq.InverseQueryRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


# --------------------------------------------------------------------------- composed


def test_a_linear_gaussian_composition_supplies_a_replayed_point_per_action() -> None:
    source = src.composed_forward_source(compose(gaussian_chain()), gaussian_actions())
    origin = source.origin
    assert origin.kind == "composed_mechanism"
    assert origin.representation == "point_only"
    assert not origin.transported
    assert origin.interval_region == "not_licensed"
    # Every mechanism of the chain was written down, not estimated.
    assert set(origin.assumed_mechanisms) == {"X", "M", "Y"}

    result = source.evaluate(query(("0", "1", "2"), [iq.target_mean(5.0)]))
    by_id = {a.id: a for a in result.actions}
    assert [by_id[f"x{arm}"].point for arm in ("0", "1", "2")] == [
        "infeasible",
        "feasible",
        "feasible",
    ]
    values = [by_id[f"x{arm}"].point_values[0].value for arm in ("0", "1", "2")]
    assert values == pytest.approx([0.5, 7.0, 13.5], abs=TOL)
    assert result.selected == "x1"
    assert result.selection == "selected"
    assert result.origin == origin
    assert not result.per_action


def test_a_changed_refit_changes_the_source_identity_and_the_values() -> None:
    base = src.composed_forward_source(compose(gaussian_chain()), gaussian_actions())
    # M = 1 + 3 X, so E[Y | do(X = 1)] = -1 + 0.5 + 3 * (1 + 3) = 11.5.
    refit = src.composed_forward_source(compose(gaussian_chain(1.0, 3.0)), gaussian_actions())
    assert base.origin.source_identity != refit.origin.source_identity
    assert base.origin.claim_digest != refit.origin.claim_digest
    result = refit.evaluate(query(("0", "1", "2"), [iq.target_mean(5.0)]))
    assert result.action("x1").point_values[0].value == pytest.approx(11.5, abs=TOL)


def test_a_point_composition_refuses_laws_intervals_and_the_wrong_family() -> None:
    composition = compose(gaussian_chain())
    source = src.composed_forward_source(composition, gaussian_actions())
    arms = ("0", "1", "2")

    quantile = refusal_of(
        lambda: source.evaluate(query(arms, [iq.target_quantile(0.5, 1.0)]))
    )
    assert quantile.reason_code == "decision_contract_unsatisfied"
    assert quantile.detail == "inverse_forward_source.composed_law_not_materialized"
    assert quantile.remedy

    interval = refusal_of(
        lambda: source.evaluate(query(arms, [iq.target_mean(5.0)]), interval_region=True)
    )
    assert interval.reason_code == "cell_not_licensed"
    assert interval.detail == "inverse_forward_source.interval_not_licensed"

    # A finite-distribution query of a linear-Gaussian composition is refused by the composition.
    wrong = finite_actions({"0": 0.0, "1": 1.0})
    refusal_of(lambda: src.composed_forward_source(composition, wrong))


# --------------------------------------------------------------------------- finite


def test_a_finite_composition_supplies_the_exact_law_of_each_action() -> None:
    composition = compose(finite_chain())
    source = src.composed_forward_source(composition, finite_actions({"0": 0.0, "1": 1.0}))
    assert source.origin.representation == "law"
    assert source.per_action
    assert len(source.origin.assumed_mechanisms) == 3

    contract_arms = ("0", "1")
    mean = query(contract_arms, [iq.target_mean(0.5)], functional="outcome")
    result = source.evaluate(mean)
    # E[Y | do(X = 0)] = 0.18 and E[Y | do(X = 1)] = 0.72 against 0.5.
    assert result.action("x0").point_values[0].value == pytest.approx(0.18, abs=TOL)
    assert result.action("x1").point_values[0].value == pytest.approx(0.72, abs=TOL)
    assert result.action("x0").point == "infeasible"
    assert result.action("x1").point == "feasible"
    assert result.per_action

    # The left-inverse median is 0 under x0 (F(0) = 0.82) and 1 under x1 (F(0) = 0.28).
    median = source.evaluate(
        query(contract_arms, [iq.target_quantile(0.5, 1.0)], functional="outcome")
    )
    assert median.action("x0").point == "infeasible"
    assert median.action("x1").point == "feasible"

    # Each action is its own intervention: no single claim exists to seal, so each action is
    # sealed separately and bound to its own claim.
    unsealable = refusal_of(lambda: source.seal(mean))
    assert unsealable.detail == "inverse_forward_source.per_action_not_sealable"
    sealed = source.seal_per_action(mean)
    assert set(sealed) == {"x0", "x1"}
    assert sealed["x0"].identity != sealed["x1"].identity
    assert sealed["x0"].origin is not None


def test_an_uncoded_finite_outcome_is_refused_with_its_level_coding() -> None:
    composition = compose(finite_chain())
    error = refusal_of(
        lambda: src.composed_forward_source(composition, finite_actions({"1": 1.0}))
    )
    assert error.reason_code == "mechanism_composition_incoherent"
    assert error.detail == "inverse_forward_source.composed_level_coding"


# --------------------------------------------------------------------------- admitted


def admitted_composition() -> mc.PreparedMechanismComposition:
    mechanisms = gaussian_chain()
    mechanisms[1] = linear("M", 0.5, 1.0, [("X", 2.0)], population="source")
    return compose(
        mechanisms,
        admissions=[
            mc.Admission(
                variable_id="M",
                coordinate="population",
                source_id="source",
                target_id="pop",
                selection_targets=("X",),
            )
        ],
    )


def receipt() -> tr.TransportReceipt:
    return tr.TransportReceipt(
        route="mechanism",
        source_identity="mechanism-transport",
        source_populations=("source",),
        target_population="pop",
        transport_identities=(f"{7:064x}",),
        assumptions=("M is invariant across populations",),
        support="extrapolative",
        representation="point_only",
    )


def test_a_composition_with_admitted_mechanisms_is_a_transported_claim() -> None:
    composition = admitted_composition()
    # No receipt: the claim would say every mechanism is native to the target.
    missing = refusal_of(
        lambda: src.composed_forward_source(
            composition, gaussian_actions(), accepts_transported=True
        )
    )
    assert missing.detail == "inverse_forward_source.composed_response_invalid"

    # A consumer that accepts only native target data refuses it.
    native = refusal_of(
        lambda: src.composed_forward_source(
            composition, gaussian_actions(), transport=receipt(), accepts_transported=False
        )
    )
    assert native.reason_code == "cell_not_licensed"
    assert native.detail == "inverse_forward_source.transported_not_accepted"

    source = src.composed_forward_source(
        composition, gaussian_actions(), transport=receipt(), accepts_transported=True
    )
    assert source.origin.support == "extrapolative"
    assert source.origin.transported
    assert source.origin.source_populations == ("source",)
    assert source.origin.target_population == "pop"


def test_a_sealed_composed_query_exports_and_a_consumer_re_derives_it() -> None:
    source = src.composed_forward_source(compose(gaussian_chain()), gaussian_actions())
    sealed = source.seal(query(("0", "1", "2"), [iq.target_mean(5.0)]))
    assert sealed.origin == source.origin
    data = sealed.export(artifact_id="composed-inverse")
    consumed = iq.consume(data, expected_identity=sealed.identity)
    assert consumed.origin == source.origin
    assert consumed.identity == sealed.identity
    assert consumed.selected == "x1"


# --------------------------------------------------------------------------- transported


def action_response(declaration=None) -> tr.TransportedResponse:
    return tr.TransportedResponse.declared_point(
        [outcome_under("a"), outcome_under("b")],
        [1.0, 3.0],
        declaration or transport_declaration("pop"),
    )


def test_a_transported_response_is_read_only_when_accepted_and_carries_its_receipt() -> None:
    response = action_response()
    # A consumer that accepts only native target data refuses it.
    native = refusal_of(lambda: src.transported_forward_source(response))
    assert native.detail == "inverse_forward_source.transported_not_accepted"

    source = src.transported_forward_source(response, accepts_transported=True)
    assert source.origin.kind == "transported_response"
    assert source.origin.support == "extrapolative"
    assert source.origin.transported
    assert source.origin.target_population == "pop"

    contract_query = query(("a", "b"), [iq.target_mean(2.0)])
    result = source.evaluate(contract_query)
    assert result.action("xa").point == "infeasible"
    assert result.action("xb").point == "feasible"
    assert result.action("xb").point_values[0].value == pytest.approx(3.0, abs=TOL)
    # The source model's standard error is a diagnostic: no feasibility inherits it.
    assert result.action("xb").point_values[0].standard_error is None
    assert result.origin.source_populations == ("trial",)

    # A transported point supplies no law and no interval region.
    quantile = refusal_of(
        lambda: source.evaluate(query(("a", "b"), [iq.target_quantile(0.5, 1.0)]))
    )
    assert quantile.detail == "inverse_forward_source.point_only_law_requested"
    interval = refusal_of(lambda: source.evaluate(contract_query, interval_region=True))
    assert interval.detail == "inverse_forward_source.interval_not_licensed"

    # The sealed artifact binds the whole receipt: another transport at the same values is
    # another artifact.
    sealed = source.seal(contract_query)
    moved = src.transported_forward_source(
        action_response(transport_declaration("pop", diagram="diagram-2")),
        accepts_transported=True,
    ).seal(contract_query)
    assert sealed.identity != moved.identity
    assert sealed.origin.transport is not None
    assert moved.origin.transport is not None
    assert sealed.origin.transport.transport_identities == ("diagram-1",)
    assert moved.origin.transport.transport_identities == ("diagram-2",)


def test_only_a_transported_response_is_a_transported_source() -> None:
    with pytest.raises(CausalTypeError):
        src.transported_forward_source(object())  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        src.composed_forward_source(object(), gaussian_actions())  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        src.cohort_forward_source(point(2.0))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- observational

X = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
Z = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
Y = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def observed(name: str, role: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=name,
        variable_name=name,
        role=role,  # type: ignore[arg-type]
        units="units",
        population_id="target",
        regime_id="observational",
        horizon=0,
        functional_id="realization",
    )


def observed_law(meaning: str = "empirical_outcome", outcome=Y) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic=meaning,  # type: ignore[arg-type]
        quantities=(observed("x", "covariate"), observed("z", "covariate"), observed("y", "outcome")),
        alignment="joint",
        source_id="observed-units",
        provider_id="enumerated",
        rng_id="deterministic_exact",
        snapshot_id="snapshot-1",
        causal_contract_id="contract-1",
    )
    draws = np.array(list(zip(X, Z, outcome, strict=True)), dtype=np.float64)
    return JointDistributionArtifact(identity, draws, calibration="exact")


def x_is_one() -> src.ObservationalScenario:
    return src.ObservationalScenario(
        "x_is_1", (src.ScenarioCondition(observed("x", "covariate"), 1.0, 1.0),)
    )


def test_observational_scenarios_are_associational_and_re_derive() -> None:
    law = observed_law()
    result = src.observational_scenarios(
        law, observed("y", "outcome"), [x_is_one()], event="at_least", threshold=1.0
    )
    assert result.standing == "associational"
    assert result.ranks_actions is False
    assert result.is_intervention_effect is False
    assert result.event_mass == pytest.approx(0.5, abs=TOL)
    scenario = result.scenario("x_is_1")
    assert scenario.mass == pytest.approx(0.5, abs=TOL)
    assert scenario.event_given_scenario == pytest.approx(0.75, abs=TOL)
    assert scenario.scenario_given_event == pytest.approx(0.75, abs=TOL)
    assert result.exact
    result.verify(law)

    # A law whose meaning is interventional answers an intervention, not an observed population.
    interventional = observed_law("interventional_predictive")
    meaning = refusal_of(
        lambda: src.observational_scenarios(
            interventional, observed("y", "outcome"), [x_is_one()], threshold=1.0
        )
    )
    assert meaning.reason_code == "distribution_meaning_mismatch"

    # A stored result of another law does not replay.
    other = observed_law(outcome=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    mismatch = refusal_of(lambda: result.verify(other))
    assert mismatch.detail == "inverse_observational_scenarios.result_mismatch"


def test_an_observational_result_is_not_an_inverse_source() -> None:
    result = src.observational_scenarios(
        observed_law(), observed("y", "outcome"), [x_is_one()], threshold=1.0
    )
    with pytest.raises(CausalTypeError):
        src.transported_forward_source(result)  # type: ignore[arg-type]
    inspected = result.inspect()
    assert inspected.family == "ObservationalScenarioResult"


# --------------------------------------------------------------------------- cohort


def cohort_source(*, shifted: bool = False) -> cd.CohortDecisionSource:
    """``tau(z) = 2 + 0.5 z`` as a compact contrast model; cohort ``z = (-1, 0.5, 1)`` with
    weights ``(0.2, 0.3, 0.5)``, so the weighted cohort mean is ``0.2 * 1.5 + 0.3 * 2.25 +
    0.5 * 2.5 = 2.225`` and the utility ``4 m - 1.5`` is ``7.4``."""
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
    cohort = cd.CohortBinding(
        snapshot_id="clinic-2026-q1",
        population="clinic_b" if shifted else "target",
        columns={"z": [-1.0, 0.5, 1.0]},
        weights=[0.2, 0.3, 0.5],
    )
    return cd.decision_source_from_compact_export(
        export,
        cohort=cohort,
        outcome_units="mmHg",
        shift=cd.CovariateShift("target", "clinic_b", ("z",)) if shifted else None,
    )


def cohort_query(source: cd.CohortDecisionSource, *constraints) -> iq.InverseQuery:
    contract = decision.Contract(
        actions=(
            decision.Action("treat", inputs=(source.quantity,), utility=decision.x(0) * 4.0 - 1.5),
            decision.Action("wait", inputs=(source.quantity,), utility=decision.x(0) * 0.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population=source.quantity.population_id,
    )
    return iq.InverseQuery(contract, ("wait", "treat"), tuple(constraints))


def test_a_cohort_contrast_is_a_point_source_with_its_own_law_refusal() -> None:
    cohort = cohort_source()
    assert cohort.mean_contrast == pytest.approx(2.225, abs=TOL)
    source = src.cohort_forward_source(cohort)
    assert source.origin.kind == "cohort_contrast"
    assert source.origin.representation == "point_only"
    assert not source.origin.transported
    answered = source.evaluate(cohort_query(cohort, iq.target_mean(5.0)))
    assert answered.feasible_actions == ("treat",)
    assert answered.action("treat").point_values[0].value == pytest.approx(7.4, abs=TOL)
    refused = refusal_of(lambda: source.evaluate(cohort_query(cohort, iq.target_quantile(0.5, 1.0))))
    assert refused.detail == "inverse_forward_source.point_only_law_requested"


def test_a_shifted_cohort_is_a_transported_source_only_when_accepted() -> None:
    cohort = cohort_source(shifted=True)
    unaccepted = refusal_of(lambda: src.cohort_forward_source(cohort))
    assert unaccepted.reason_code == "cell_not_licensed"
    assert unaccepted.detail == "inverse_forward_source.transported_not_accepted"
    source = src.cohort_forward_source(cohort, accepts_transported=True)
    assert source.origin.transported
    assert source.origin.support == "extrapolative"
    assert source.origin.target_population == "clinic_b"
    answered = source.evaluate(cohort_query(cohort, iq.target_mean(5.0)))
    assert answered.feasible_actions == ("treat",)
    # The same cohort is a transported response of the decision boundary.
    response = tr.TransportedResponse.from_shifted_cohort(cohort)
    assert response.receipt.route == "shifted_cohort"
    assert response.receipt.target_population == "clinic_b"
    assert response.means == pytest.approx((2.225,), abs=TOL)
    # A cohort in the model's own population is not transported.
    with pytest.raises(tr.TransportedResponseRefusal) as native:
        tr.TransportedResponse.from_shifted_cohort(cohort_source())
    assert_registered_refusal(native.value)
    assert native.value.detail == "transported_decision.not_transported"
