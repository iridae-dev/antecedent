"""The mixed-evidence decision: a cluster trial, a temporal period effect, a cost model and a bridge.

Oracles. The cost model is fitted to ``cost = 0.5 + 1.5 * action +- 0.25`` on a balanced design, so
its contrast is 1.5 up to the flat prior's shrinkage and a tight prior pulls it toward zero. The
combination is the weighted sum ``benefit * margin + temporal - cost`` of the three published
moments, so its mean is ``benefit * margin_mean + temporal_effect - cost_mean`` and its variance
is the quadratic form of the declared covariance with weights ``(benefit, 1, -1)``: the diagonal
alone when the routes are independent, the diagonal plus twice each signed cross term otherwise.
The shared-parameter case is checked against the exact joint Gaussian posterior of ``(theta, phi)``
computed here in numpy. Authority is the ledger's: withdrawing the trial retires the decision, the
export and every replay that asks for current standing, and leaves the numbers of a historical
replay as they were.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from antecedent import composition_verification as cv
from antecedent import mixed_decision as md
from antecedent.errors import CausalSerializationError, CausalTypeError, CausalValueError

from _refusal import assert_registered_refusal

SEED = 0xF1A6


def run(dependence: md.Dependence | None = None, **options) -> md.MixedDecisionWorkflow:
    return md.run_mixed_decision(dependence, seed=SEED, **options)


@pytest.fixture(scope="module")
def base() -> md.MixedDecisionWorkflow:
    return run(md.Dependence.independent())


def refused(call) -> md.MixedDecisionRefusal:
    with pytest.raises(md.MixedDecisionRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


# ------------------------------------------------------------------------------ the decision


def test_the_independent_decision_combines_the_published_moments(base) -> None:
    decision, routes = base.decision(), base.routes
    assert routes.cluster_trials == 1 and routes.temporal_means == 2
    # the balanced fixture's least-squares contrast is exactly 1.5; the weak default prior shrinks
    # it toward zero by a few thousandths, never past it and never away from it
    assert routes.cost_mean == pytest.approx(1.5, abs=1e-2)
    assert 1.49 < routes.cost_mean < 1.5
    assert 0.0 < decision.cost_variance < 0.05
    assert decision.bridge_value == pytest.approx(routes.cluster_effect, abs=1e-12)
    assert decision.dependence == "independent" and decision.dependence_id is None
    assert decision.mean == pytest.approx(
        decision.margin_mean + routes.temporal_effect - routes.cost_mean, rel=1e-9, abs=1e-12
    )
    assert decision.variance == pytest.approx(
        decision.margin_variance + decision.temporal_variance + decision.cost_variance, rel=1e-12
    )
    assert decision.temporal_variance > 0.0 and decision.margin_variance > 0.0

    evaluation = decision.evaluation
    by_id = {a.id: a for a in evaluation.actions}
    assert by_id["decline"].expected_utility == pytest.approx(0.0, abs=1e-12)
    assert by_id["act"].expected_utility == pytest.approx(decision.mean, rel=1e-12, abs=1e-12)
    assert evaluation.verdict == "uniquely_optimal"
    assert evaluation.chosen == (("act",) if decision.mean > 0.0 else ("decline",))
    assert len(decision.identity) == 64


def test_a_declared_covariance_moves_the_variance_by_twice_each_signed_cross_term() -> None:
    reference = run(md.Dependence.independent()).decision()
    vm, vt, vc = reference.margin_variance, reference.temporal_variance, reference.cost_variance
    margin_temporal, margin_cost = 0.3 * math.sqrt(vm * vt), -0.2 * math.sqrt(vm * vc)
    temporal_cost = 0.1 * math.sqrt(vt * vc)
    declared = md.Dependence.covariance(
        "cross-terms",
        margin_temporal=margin_temporal,
        margin_cost=margin_cost,
        temporal_cost=temporal_cost,
    )
    decision = run(declared).decision()
    # weights (1, 1, -1): the margin-cost and temporal-cost terms enter with a negative sign
    expected = (
        decision.margin_variance
        + decision.temporal_variance
        + decision.cost_variance
        + 2.0 * margin_temporal
        - 2.0 * margin_cost
        - 2.0 * temporal_cost
    )
    assert decision.variance == pytest.approx(expected, rel=1e-9)
    assert decision.mean == pytest.approx(reference.mean, rel=1e-9, abs=1e-12)
    assert decision.dependence == "covariance" and decision.dependence_id == "cross-terms"
    assert decision.identity != reference.identity

    # a correlation of the published standard deviations is the same covariance
    correlated = run(
        md.Dependence.correlated(
            "cross-terms",
            margin_temporal=margin_temporal,
            margin_cost_correlation=-0.2,
            temporal_cost=temporal_cost,
        )
    ).decision()
    assert correlated.variance == pytest.approx(decision.variance, rel=1e-9)


def test_an_unlawful_dependence_is_refused_and_a_malformed_declaration_raises() -> None:
    reference = run(md.Dependence.independent()).decision()
    scale = math.sqrt(reference.margin_variance * reference.temporal_variance)
    # a correlation of two between the margin and the temporal effect is not positive semidefinite
    not_psd = md.Dependence.covariance(
        "not-psd", margin_temporal=2.0 * scale, margin_cost=0.0, temporal_cost=0.0
    )
    error = refused(lambda: run(not_psd))
    assert error.reason_code == "invalid_argument"
    assert error.detail == "mixed_decision.invalid_request"
    assert error.remedy
    impossible = md.Dependence.correlated(
        "too-correlated", margin_temporal=0.0, margin_cost_correlation=1.5, temporal_cost=0.0
    )
    assert refused(lambda: run(impossible)).detail == "mixed_decision.invalid_request"
    # a feedback policy declares no covariance to carry the feedback over independence
    assert (
        refused(
            lambda: run(md.Dependence.independent(), policy=md.Policy.feedback("returned"))
        ).detail
        == "mixed_decision.invalid_request"
    )
    with pytest.raises(CausalTypeError):
        md.Dependence.covariance("", margin_temporal=0.0, margin_cost=0.0, temporal_cost=0.0)
    with pytest.raises(CausalValueError):
        md.Dependence.covariance("c", margin_temporal=math.nan, margin_cost=0.0, temporal_cost=0.0)
    with pytest.raises(CausalTypeError):
        md.run_mixed_decision("independent")  # type: ignore[arg-type]
    with pytest.raises(CausalValueError):
        md.run_mixed_decision(seed=-1)


def _shared_model(feedback_weight: float = 0.5) -> md.SharedParameterModel:
    return md.SharedParameterModel(
        margin_prior=md.Gaussian(0.5, 2.0),
        cost_prior=md.Gaussian(-0.4, 1.0),
        pooling_weight=0.4,
        margin_observation=md.Gaussian(1.3, 0.25),
        cost_observation=md.Gaussian(2.1, 0.5),
        specific_prior=md.Gaussian(0.3, 1.5),
        feedback_weight=feedback_weight,
    )


def _joint_posterior_correlation(model: md.SharedParameterModel) -> float:
    """Correlation of ``theta`` and ``theta + phi`` in the exact joint Gaussian posterior."""
    w = model.pooling_weight
    pooled = w / model.margin_prior.variance + (1.0 - w) / model.cost_prior.variance
    precision = np.diag([pooled, 1.0 / model.specific_prior.variance])
    precision += np.outer([1.0, 0.0], [1.0, 0.0]) / model.margin_observation.variance
    precision += np.outer([1.0, 1.0], [1.0, 1.0]) / model.cost_observation.variance
    sigma = np.linalg.inv(precision)
    theta, contrast = sigma[0, 0], sigma[0, 0] + sigma[1, 1] + 2.0 * sigma[0, 1]
    return float((sigma[0, 0] + sigma[0, 1]) / math.sqrt(theta * contrast))


def test_a_shared_parameter_takes_its_correlation_from_the_joint_posterior_under_the_policy() -> (
    None
):
    model = _shared_model()
    reference = run(md.Dependence.independent()).decision()
    dependence = md.Dependence.shared_parameter("shared-theta", model)

    joint = run(dependence).decision()
    vm, vc = joint.margin_variance, joint.cost_variance
    diagonal = vm + joint.temporal_variance + vc
    correlation = _joint_posterior_correlation(model)
    # the coherent joint model is the full posterior; weights (1, -1) make the term negative
    expected = diagonal - 2.0 * correlation * math.sqrt(vm * vc)
    assert joint.variance == pytest.approx(expected, rel=1e-8)
    assert joint.mean == pytest.approx(reference.mean, rel=1e-9, abs=1e-12)

    cut = run(dependence, policy=md.Policy.cut("pooled theta, margin leads")).decision()
    feedback = run(dependence, policy=md.Policy.feedback("pooled theta, partial return")).decision()
    variances = [joint.variance, cut.variance, feedback.variance]
    identities = [joint.identity, cut.identity, feedback.identity]
    # three policies are three decisions; none of them moves the combined mean
    assert len(set(identities)) == 3
    for left, right in ((0, 1), (0, 2), (1, 2)):
        assert abs(variances[left] - variances[right]) > 1e-9 * variances[left]
    assert cut.mean == pytest.approx(joint.mean, rel=1e-9, abs=1e-12)
    # a cut is not independence: the dependence that flows forward remains
    assert abs(cut.variance - diagonal) > 1e-6 * diagonal
    # a malformed declaration is refused, not run
    bad = md.SharedParameterModel(
        margin_prior=model.margin_prior,
        cost_prior=model.cost_prior,
        pooling_weight=1.5,
        margin_observation=model.margin_observation,
        cost_observation=model.cost_observation,
        specific_prior=model.specific_prior,
        feedback_weight=1.0,
    )
    assert refused(lambda: run(md.Dependence.shared_parameter("bad", bad))).detail == (
        "mixed_decision.invalid_request"
    )


# ------------------------------------------------------------------------------ export and load


def test_an_exported_decision_loads_against_current_authority_and_replays_the_same_numbers() -> (
    None
):
    workflow = run(md.Dependence.independent())
    decision = workflow.decision()
    exported = workflow.export_decision()
    assert len(exported.identity) == 64

    loaded = workflow.load(exported.artifact, exported.identity)
    assert loaded.mean == decision.mean and loaded.variance == decision.variance
    assert loaded.evaluation.verdict == decision.evaluation.verdict
    assert loaded.evaluation.chosen == decision.evaluation.chosen
    # the independent replay of the same bytes grants no authority and reads the same numbers
    assert md.replay_decision(exported.artifact, exported.identity, seed=SEED) == (
        decision.mean,
        decision.variance,
    )
    historical = workflow.export_historical_replay()
    assert historical.identity == decision.identity
    assert md.replay_decision(historical.artifact, historical.identity, seed=SEED) == (
        decision.mean,
        decision.variance,
    )

    # an identity the consumer retained independently must match the bytes
    mismatch = refused(lambda: workflow.load(exported.artifact, "0" * 64))
    assert mismatch.detail == "mixed_decision.invalid_request"
    assert refused(lambda: md.replay_decision(exported.artifact, "0" * 64)).detail == (
        "mixed_decision.invalid_request"
    )
    # corrupt or foreign bytes are not a refusal of the decision
    with pytest.raises(CausalSerializationError):
        workflow.load(exported.artifact[:-7])
    with pytest.raises(CausalSerializationError):
        workflow.load(historical.artifact)
    with pytest.raises(CausalTypeError):
        workflow.load("not bytes")  # type: ignore[arg-type]


def test_a_replacement_supersedes_an_earlier_export_and_names_the_work_it_did() -> None:
    workflow = run(md.Dependence.independent())
    first = workflow.export_decision()
    before = workflow.decision()
    routes = workflow.routes

    replay = workflow.utility_only(2.0)
    # a new utility refits no trial, no mechanism and no posterior
    assert replay == md.UtilityReplay(0, 0, 0, 0, 0)
    after, current = workflow.decision(), workflow.routes
    assert after.mean == pytest.approx(
        2.0 * after.margin_mean + current.temporal_effect - current.cost_mean, rel=1e-9, abs=1e-12
    )
    assert after.margin_mean == before.margin_mean
    assert after.variance == pytest.approx(
        4.0 * after.margin_variance + after.temporal_variance + after.cost_variance, rel=1e-9
    )
    assert current.cluster_effect == routes.cluster_effect and current.cluster_trials == 1
    assert current.temporal_effect == routes.temporal_effect
    # the earlier export was bound to the earlier standing of its sources
    stale = refused(lambda: workflow.load(first.artifact, first.identity))
    assert stale.detail.startswith("evidence_ledger.")
    assert workflow.load(workflow.export_decision().artifact).mean == after.mean
    with pytest.raises(CausalValueError):
        workflow.utility_only(math.inf)


def test_a_prior_change_refits_only_the_cost_posterior() -> None:
    workflow = run(md.Dependence.independent())
    before = workflow.routes
    replay = workflow.prior_change(0.5)
    assert replay.score_recomputed and replay.identification_reused
    assert replay.bayesian_model_fits >= 1
    after = workflow.routes
    # a tight prior shrinks the contrast of the balanced fixture toward zero
    assert 0.0 < after.cost_mean < before.cost_mean
    assert after.temporal_effect == before.temporal_effect
    assert after.cluster_effect == before.cluster_effect and after.cluster_trials == 1
    assert refused(lambda: workflow.prior_change(-1.0)).detail == "mixed_decision.invalid_request"
    with pytest.raises(CausalValueError):
        workflow.prior_change(math.nan)


def test_a_period_change_refits_only_the_temporal_mechanisms() -> None:
    workflow = run(md.Dependence.independent())
    before = workflow.routes
    replay = workflow.period_change(0, 49)
    assert replay.score_recomputed and replay.identification_reused
    assert replay.temporal_factor_builds >= 1
    after = workflow.routes
    assert after.cost_mean == before.cost_mean
    assert after.cluster_effect == before.cluster_effect and after.cluster_trials == 1
    # the retained two-step histories must stay inside the period
    assert refused(lambda: workflow.period_change(0, 10)).detail == "mixed_decision.invalid_request"
    with pytest.raises(CausalValueError):
        workflow.period_change(-1, 49)


# ------------------------------------------------------------------------------ withdrawal


def test_withdrawing_the_trial_retires_the_decision_and_leaves_the_other_claims() -> None:
    workflow = run(md.Dependence.independent())
    exported = workflow.export_decision()
    historical = md.replay_decision(exported.artifact, exported.identity, seed=SEED)
    cost, temporal = workflow.routes.cost_mean, workflow.routes.temporal_effect

    verification = workflow.withdraw_cluster()
    assert isinstance(verification, cv.CompositionVerification)
    for key in ("cluster_trial", "margin_bridge", "decision"):
        assert verification.classify(key).klass == "withdrawn", key
    for key in ("temporal_drift", "bayesian_cost"):
        assert verification.classify(key).klass == "still_valid", key
    assert verification.classify("decision").authority_retained is False
    assert verification.require_valid("temporal_drift").klass == "still_valid"
    with pytest.raises(cv.CompositionVerificationRefusal) as withdrawn:
        verification.require_valid("decision")
    assert_registered_refusal(withdrawn.value)
    assert withdrawn.value.reason_code == "claim_license_withdrawn"
    assert withdrawn.value.detail == "composition_verification.input_retired"

    # nothing asks the withdrawn source for permission any more
    for call in (
        workflow.decision,
        workflow.export_decision,
        lambda: workflow.utility_only(2.0),
        lambda: workflow.prior_change(1.0),
        lambda: workflow.period_change(0, 49),
        lambda: workflow.load(exported.artifact, exported.identity),
    ):
        error = refused(call)
        assert error.detail.startswith("evidence_ledger."), error.detail
    # the historical numbers are unchanged and the other routes keep their estimates
    assert md.replay_decision(exported.artifact, exported.identity, seed=SEED) == historical
    assert workflow.routes.cost_mean == cost and workflow.routes.temporal_effect == temporal
    # the sealed classification replays in a fresh consumer
    again = cv.consume_composition_verification_artifact(
        bytes(verification.export()), verification.identity
    )
    assert again.classify("decision").klass == "withdrawn"
    assert again.classify("bayesian_cost").klass == "still_valid"
