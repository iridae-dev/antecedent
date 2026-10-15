"""A fitted effect model decides for a cohort: actual DR-learner producer, cohort adapter,
affine contract with hand-calculated cohort means and utilities, fresh-process replay and the
negative gates.

Oracle. The data are ``y = t (effect + 0.4 z) + z`` with a randomized binary treatment, so the
true conditional contrast is ``tau(z) = effect + 0.4 z``. The cohort rows and weights are fixed
in the test; the expected cohort mean is the weighted sum of the model's own CATE predictions
(computed here in numpy) and, within the estimator's tolerance, of the truth.
"""

from __future__ import annotations

import json
import subprocess
import sys

import antecedent as ac
import numpy as np
import pytest
from antecedent import cohort_decision as cd
from antecedent import decision
from antecedent import inverse_query as iq
from antecedent.compact_export import CompactExport, Intercept, Linear, Numeric, Quantity
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.estimators import DRLearner
from antecedent.prediction import FittedEffectModel

Z = [-0.8, 0.0, 0.5]
W = [0.25, 0.25, 0.5]
ROW_TOLERANCE = 1e-9


def fitted_result(effect: float = 2.0):
    rng = np.random.default_rng(19)
    z = np.tile(np.linspace(-1.0, 1.0, 21), 80)
    treatment = rng.binomial(1, 0.5, len(z)).astype(float)
    return ac.analyze(
        {"z": z, "t": treatment, "y": treatment * (effect + 0.4 * z) + z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ac.AverageEffect(treatment="t", outcome="y"),
        estimator=DRLearner(),
        bootstrap=0,
        refute=False,
        seed=3,
    )


@pytest.fixture(scope="module")
def model() -> FittedEffectModel:
    return fitted_result().fitted_model


@pytest.fixture(scope="module")
def other_model() -> FittedEffectModel:
    return fitted_result(3.0).fitted_model


def cohort(z=Z, w=W, population: str = "target", snapshot: str = "clinic-2026-q1"):
    return cd.CohortBinding(
        snapshot_id=snapshot, population=population, columns={"z": list(z)}, weights=list(w)
    )


def affine_contract(source: cd.CohortDecisionSource, slope: float, offset: float):
    return decision.Contract(
        actions=(
            decision.Action(
                "treat", inputs=(source.quantity,), utility=decision.x(0) * slope + offset
            ),
            decision.Action("wait", inputs=(source.quantity,), utility=decision.x(0) * 0.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population=source.quantity.population_id,
    )


def compact_model() -> CompactExport:
    return CompactExport.build(
        response=Quantity(
            "y",
            units="mmHg",
            role="outcome",
            regime="contrast:do(t=1)-do(t=0)",
            functional="conditional_mean_difference",
        ),
        inputs=[Numeric("z", -1.0, 1.0, units="1")],
        terms=[Intercept(), Linear("z")],
        coefficients=[2.0, 0.4],
        covariance=[[0.04, -0.01], [-0.01, 0.09]],
    )


# --------------------------------------------------------------------------- the producer path


def test_producer_to_cohort_to_affine_contract_matches_the_hand_calculation(model) -> None:
    source = model.decision_source(cohort=cohort(), outcome_units="mmHg")
    assert source.is_native
    assert source.parent_claim == model.parent_claim
    # Hand calculation from the model's own CATE points.
    points = np.asarray(model.predict({"z": Z}).values)
    expected_mean = float(np.dot(W, points))
    assert source.mean_contrast == pytest.approx(expected_mean, abs=ROW_TOLERANCE)
    assert source.row_contrasts == pytest.approx(tuple(points), abs=ROW_TOLERANCE)
    # ... and, within the estimator's tolerance, the structural truth 2 + 0.4 * 0.0 + ...
    truth = 2.0 + 0.4 * float(np.dot(W, Z))
    assert source.mean_contrast == pytest.approx(truth, abs=0.12)
    assert source.support["rows_outside"] == 0
    assert source.support_status == "supported"
    assert source.quantity.functional_id == "mean_difference"
    assert source.quantity.regime_id == "contrast:do(t=1)-do(t=0)"
    assert source.quantity.conditioning[0].value_id == cohort().digest

    # Affine utilities: 5 x - 2 and the do-nothing action; computed by hand from the mean.
    contract = affine_contract(source, 5.0, -2.0)
    for evaluated in (contract.evaluate(source), contract.evaluate(source.mean_source)):
        by_id = {a.id: a for a in evaluated.actions}
        assert by_id["treat"].expected_utility == pytest.approx(5.0 * expected_mean - 2.0, abs=1e-8)
        assert by_id["wait"].expected_utility == pytest.approx(0.0, abs=1e-12)
        assert by_id["treat"].standard_error is None and evaluated.evpi is None
        assert evaluated.selected == ("treat",)
    # A loss-making utility waits.
    hold = affine_contract(source, 1.0, -10.0).evaluate(source)
    assert hold.selected == ("wait",)
    # A pure Rust-issued mean source: a hand-edited one is refused.
    tampered = decision.MeanSource(
        coordinates=source.mean_source.coordinates,
        means=(source.mean_contrast + 1.0,),
        provider_id=source.mean_source.provider_id,
        snapshot_id=source.mean_source.snapshot_id,
        causal_contract_id=source.mean_source.causal_contract_id,
        _original_cohort=source,
    )
    with pytest.raises(CausalValueError):
        contract.evaluate(tampered)


def test_probability_quantile_nonlinear_and_joint_law_requests_are_refused(model) -> None:
    source = model.decision_source(cohort=cohort(), outcome_units="mmHg")
    contract = affine_contract(source, 5.0, -2.0)
    squared = decision.Contract(
        actions=(
            decision.Action(
                "treat", inputs=(source.quantity,), utility=decision.x(0) * decision.x(0)
            ),
            contract.actions[1],
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    with pytest.raises(decision.DecisionRefusal) as nonlinear:
        squared.evaluate(source)
    assert nonlinear.value.reason_code == "decision_contract_unsatisfied"
    for criterion in (
        decision.Criterion.threshold_probability(1.0),
        decision.Criterion.quantile(0.5),
    ):
        asked = decision.Contract(
            actions=contract.actions,
            utility_units="utility",
            criterion=criterion,
            target_population="target",
        )
        with pytest.raises(decision.DecisionRefusal) as law:
            asked.evaluate(source)
        assert law.value.reason_code == "decision_contract_unsatisfied"
    for kind, code, detail in (
        ("probability", "decision_contract_unsatisfied", "cohort_decision.law_not_supplied"),
        ("quantile", "decision_contract_unsatisfied", "cohort_decision.law_not_supplied"),
        ("nonlinear_utility", "decision_contract_unsatisfied", "cohort_decision.law_not_supplied"),
        ("joint_law", "joint_law_required", "cohort_decision.joint_law_required"),
    ):
        with pytest.raises(cd.CohortDecisionRefusal) as refused:
            source.refuse(kind)
        assert (refused.value.reason_code, refused.value.detail) == (code, detail)
        assert refused.value.remedy
    with pytest.raises(cd.CohortDecisionRefusal) as interval:
        source.interval()
    assert interval.value.reason_code == "cell_not_licensed"
    assert interval.value.detail == "cohort_decision.interval_not_licensed"


def test_the_source_replays_in_a_fresh_process_under_the_retained_identity(model, tmp_path) -> None:
    source = model.decision_source(cohort=cohort(), outcome_units="mmHg")
    artifact = tmp_path / "cohort.ant"
    artifact.write_bytes(source.export())
    code = """
import json, sys
from pathlib import Path
import antecedent as ac
from antecedent import cohort_decision as cd

def forbidden(*args, **kwargs):
    raise AssertionError('replay cannot analyze or refit')
ac.analyze = forbidden
replayed = cd.consume_cohort_decision_source(
    Path(sys.argv[1]).read_bytes(),
    expected_identity=sys.argv[2],
    parent_claim=sys.argv[3],
    population='target',
    outcome_units='mmHg',
    cohort_digest=sys.argv[4],
)
print(json.dumps({
    'mean': replayed.mean_contrast,
    'identity': replayed.identity,
    'native': replayed.is_native,
}))
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(artifact),
            source.identity,
            model.parent_claim,
            cohort().digest,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    replayed = json.loads(child.stdout)
    assert replayed["mean"] == source.mean_contrast
    assert replayed["identity"] == source.identity
    assert replayed["native"] is True


# ----------------------------------------------------------------------- the negative gates


def test_missing_or_outside_support_and_schema_unit_population_contrast_are_refused(
    model,
) -> None:
    # Outside the fitted support: the row is named.
    with pytest.raises(cd.CohortDecisionRefusal) as outside:
        model.decision_source(
            cohort=cd.CohortBinding(
                snapshot_id="s",
                population="target",
                columns={"z": [0.0, 1.5]},
                row_ids=["patient-a", "patient-b"],
            ),
            outcome_units="mmHg",
        )
    assert outside.value.reason_code == "cell_not_licensed"
    assert outside.value.detail == "cohort_decision.outside_support"
    assert "patient-b" in str(outside.value.offending)
    assert outside.value.remedy
    # Missing and extra features.
    with pytest.raises(cd.CohortDecisionRefusal) as missing:
        model.decision_source(
            cohort=cd.CohortBinding(
                snapshot_id="s", population="target", columns={"age": [40.0]}, weights=[1.0]
            ),
            outcome_units="mmHg",
        )
    assert missing.value.reason_code == "schema_mismatch"
    assert missing.value.detail == "cohort_decision.feature_schema_mismatch"
    with pytest.raises(cd.CohortDecisionRefusal) as extra:
        model.decision_source(
            cohort=cd.CohortBinding(
                snapshot_id="s",
                population="target",
                columns={"z": [0.0], "age": [40.0]},
                weights=[1.0],
            ),
            outcome_units="mmHg",
        )
    assert extra.value.detail == "cohort_decision.feature_schema_mismatch"
    # Units are required and never inferred.
    with pytest.raises(CausalValueError):
        model.decision_source(cohort=cohort(), outcome_units="")
    # Another population needs a declared shift; without it the premise is named.
    foreign = cohort(population="clinic_b")
    with pytest.raises(cd.CohortDecisionRefusal) as population:
        model.decision_source(cohort=foreign, outcome_units="mmHg")
    assert population.value.reason_code == "transport_missing_evidence"
    assert population.value.detail == "cohort_decision.shift_declaration_required"
    # A different contrast than the one the decision names.
    with pytest.raises(cd.CohortDecisionRefusal) as contrast:
        model.decision_source(
            cohort=cohort(), outcome_units="mmHg", expected_contrast="contrast:do(t=2)-do(t=0)"
        )
    assert contrast.value.detail == "cohort_decision.contrast_mismatch"
    # Invalid cohorts are refused at binding.
    with pytest.raises(cd.CohortDecisionRefusal) as weights:
        cd.CohortBinding(
            snapshot_id="s", population="target", columns={"z": [0.0, 1.0]}, weights=[0.5, 0.6]
        )
    assert weights.value.detail == "cohort_decision.weights_not_normalized"
    with pytest.raises(cd.CohortDecisionRefusal) as nonfinite:
        cd.CohortBinding(snapshot_id="s", population="target", columns={"z": [float("nan")]})
    assert nonfinite.value.detail == "cohort_decision.cohort_invalid"
    # NumPy normally coerces this mixed sequence to strings; binding must retain the caller's
    # types so numeric values cannot silently become categorical levels.
    with pytest.raises(CausalTypeError, match="mixes levels and numbers"):
        cd.CohortBinding(snapshot_id="s", population="target", columns={"z": [1, "high"]})
    with pytest.raises(CausalTypeError, match="row_id"):
        cd.CohortBinding(snapshot_id="s", population="target", columns={"z": [1.0]}, row_ids=[17])


def test_a_declared_shift_is_extrapolative_and_must_be_accepted(model) -> None:
    foreign = cohort(population="clinic_b")
    shift = cd.CovariateShift("target", "clinic_b", ("z",))
    source = model.decision_source(cohort=foreign, outcome_units="mmHg", shift=shift)
    assert source.support_status == "extrapolative"
    assert source.quantity.population_id == "clinic_b"
    assert any("declared covariate shift" in line for line in source.assumptions)
    contract = affine_contract(source, 5.0, -2.0)
    with pytest.raises(cd.CohortDecisionRefusal) as unaccepted:
        contract.evaluate(source)
    assert unaccepted.value.detail == "cohort_decision.extrapolative_not_accepted"
    accepted = contract.evaluate(source.accept_extrapolative())
    assert accepted.selected == ("treat",)
    # The shifted cohort must still lie inside the fitted support.
    with pytest.raises(cd.CohortDecisionRefusal) as beyond:
        model.decision_source(
            cohort=cohort(z=[0.0, 3.0], w=[0.5, 0.5], population="clinic_b"),
            outcome_units="mmHg",
            shift=shift,
        )
    assert beyond.value.reason_code == "transport_support_failure"
    # A declaration naming other modifiers is not certified.
    with pytest.raises(cd.CohortDecisionRefusal) as wrong:
        model.decision_source(
            cohort=foreign,
            outcome_units="mmHg",
            shift=cd.CovariateShift("target", "clinic_b", ("z", "age")),
        )
    assert wrong.value.reason_code == "transport_not_certified"
    # An inverse query does not take a transported mean.
    with pytest.raises(cd.CohortDecisionRefusal) as inverse:
        source.forward_claim()
    assert inverse.value.reason_code == "cell_not_licensed"
    assert inverse.value.detail == "inverse_forward_source.transported_not_accepted"


def test_changed_rows_weights_cohort_wrong_parent_and_resealed_artifacts_are_refused(
    model, other_model
) -> None:
    base = model.decision_source(cohort=cohort(), outcome_units="mmHg")
    artifact = base.export()
    retained = base.identity
    # A changed weight and a changed row are different cohorts and different identities.
    for changed in (cohort(w=[0.5, 0.25, 0.25]), cohort(z=[-0.8, 0.0, 0.4])):
        other = model.decision_source(cohort=changed, outcome_units="mmHg")
        assert other.identity != retained
        assert other.cohort_digest != base.cohort_digest
        with pytest.raises(cd.CohortDecisionRefusal) as swapped:
            cd.consume_cohort_decision_source(
                other.export(), expected_identity=retained, cohort_digest=base.cohort_digest
            )
        assert swapped.value.detail == "cohort_decision.cohort_changed"
        # Without the cohort expectation the consistently resealed artifact is another identity.
        with pytest.raises(cd.CohortDecisionRefusal) as resealed:
            cd.consume_cohort_decision_source(other.export(), expected_identity=retained)
        assert resealed.value.detail == "cohort_decision.identity_unexpected"
    # A different parent claim.
    foreign = other_model.decision_source(cohort=cohort(), outcome_units="mmHg")
    assert foreign.parent_claim != base.parent_claim
    with pytest.raises(cd.CohortDecisionRefusal) as parent:
        cd.consume_cohort_decision_source(
            foreign.export(), expected_identity=foreign.identity, parent_claim=base.parent_claim
        )
    assert parent.value.detail == "cohort_decision.wrong_parent_identity"
    with pytest.raises(cd.CohortDecisionRefusal) as load:
        FittedEffectModel.load(other_model.export(), expected_parent_claim=model.parent_claim)
    assert load.value.detail == "cohort_decision.wrong_parent_identity"
    # Other retained expectations are typed gates.
    for field, detail in (
        ("population", "cohort_decision.population_mismatch"),
        ("outcome_units", "cohort_decision.units_mismatch"),
        ("contrast", "cohort_decision.contrast_mismatch"),
    ):
        with pytest.raises(cd.CohortDecisionRefusal) as gate:
            cd.consume_cohort_decision_source(
                artifact, expected_identity=retained, **{field: "something-else"}
            )
        assert gate.value.detail == detail
    # A corrupted container is refused before any number is read.
    corrupted = bytearray(artifact)
    corrupted[len(corrupted) // 2] ^= 0xFF
    with pytest.raises(cd.CohortDecisionRefusal):
        cd.consume_cohort_decision_source(bytes(corrupted), expected_identity=retained)
    with pytest.raises(cd.CohortDecisionRefusal) as foreign_bytes:
        cd.consume_cohort_decision_source(b"not an artifact", expected_identity=retained)
    assert foreign_bytes.value.reason_code == "invalid_argument"


# ------------------------------------------------------------- compact model and the variance


def test_a_declared_compact_model_carries_the_uncalibrated_variance_and_is_never_native() -> None:
    export = compact_model()
    source = cd.decision_source_from_compact_export(export, cohort=cohort(), outcome_units="mmHg")
    # Oracle: tau = 2 + 0.4 z; z_bar = 0.25 * -0.8 + 0.5 * 0.5 = 0.05; mean = 2.02.
    zbar = float(np.dot(W, Z))
    assert source.mean_contrast == pytest.approx(2.0 + 0.4 * zbar, abs=1e-12)
    assert not source.is_native
    # Variance of the weighted mean: phibar' V phibar with phibar = (1, z_bar).
    variance = 0.04 + 2 * (-0.01) * zbar + 0.09 * zbar**2
    diagnostic = source.uncalibrated_standard_error
    assert diagnostic is not None
    assert diagnostic.value == pytest.approx(variance**0.5, abs=1e-12)
    assert diagnostic.calibration == "calibration unmeasured"
    # A double-sum oracle agrees.
    phi = np.array([[1.0, z] for z in Z])
    v = np.array([[0.04, -0.01], [-0.01, 0.09]])
    total = sum(W[i] * W[j] * phi[i] @ v @ phi[j] for i in range(len(Z)) for j in range(len(Z)))
    assert diagnostic.value == pytest.approx(float(total) ** 0.5, abs=1e-12)
    # It never reaches the decision, and the interval route is closed.
    decided = affine_contract(source, 5.0, -2.0).evaluate(source)
    assert all(a.standard_error is None for a in decided.actions)
    with pytest.raises(cd.CohortDecisionRefusal) as closed:
        source.interval()
    assert closed.value.reason_code == "cell_not_licensed"
    # An export of an absolute mean is not a contrast; a retained identity is required.
    absolute = CompactExport.build(
        response=Quantity("y", units="mmHg", role="outcome"),
        inputs=[Numeric("z", -1.0, 1.0, units="1")],
        terms=[Intercept(), Linear("z")],
        coefficients=[2.0, 0.4],
        covariance=[[0.04, 0.0], [0.0, 0.09]],
    )
    with pytest.raises(cd.CohortDecisionRefusal) as not_contrast:
        cd.decision_source_from_compact_export(absolute, cohort=cohort(), outcome_units="mmHg")
    assert not_contrast.value.detail == "cohort_decision.contrast_not_declared"
    with pytest.raises(cd.CohortDecisionRefusal) as units:
        cd.decision_source_from_compact_export(export, cohort=cohort(), outcome_units="g/dL")
    assert units.value.detail == "cohort_decision.units_mismatch"
    # A portable (analysis result) model has no variance diagnostic.


def test_a_portable_model_has_no_variance_diagnostic_and_feeds_an_inverse_query(model) -> None:
    source = model.decision_source(cohort=cohort(), outcome_units="mmHg")
    assert source.uncalibrated_standard_error is None
    contract = affine_contract(source, 5.0, -2.0)
    target = 5.0 * source.mean_contrast - 2.0 - 0.5
    answered = iq.InverseQuery(contract, ("wait", "treat"), (iq.target_mean(target),)).evaluate(
        source.forward_claim()
    )
    assert answered.feasible_actions == ("treat",)
    with pytest.raises(iq.InverseQueryRefusal) as refused:
        iq.InverseQuery(contract, ("wait", "treat"), (iq.target_quantile(0.5, 1.0),)).evaluate(
            source.forward_claim()
        )
    assert refused.value.detail == "decision_evaluation.mean_source_insufficient"
    with pytest.raises(CausalTypeError):
        iq.InverseQuery(contract, ("wait", "treat"), (iq.target_mean(target),)).evaluate(source)


def _native_fit(n: int = 400):
    rng = np.random.default_rng(23)
    x = rng.uniform(-2.0, 2.0, n)
    t = (np.arange(n) % 2).astype(float)
    y = 2.0 + x + t * (1.0 + 0.5 * x) + rng.normal(0.0, 0.5, n)
    return x, cd.fit_native_cohort_effect(
        outcome=y,
        treatment=t,
        features={"x": x},
        outcome_name="y",
        treatment_name="t",
        outcome_units="mmHg",
        population="target",
    )


def test_a_native_fit_issues_a_measured_interval_or_a_typed_refusal() -> None:
    x, fit = _native_fit()
    assert (fit.rows, fit.width) == (400, 4)
    binding = cd.CohortBinding(
        snapshot_id="clinic",
        population="target",
        columns={"x": [float(v) for v in x[:80]]},
        weights=[1.0 / 80.0] * 80,
    )
    source = cd.decision_source_from_native_fit(fit, cohort=binding, outcome_units="mmHg")
    try:
        measured = source.interval(level=0.95)
    except cd.CohortDecisionRefusal as refusal:
        # No attesting coverage record yet: the typed refusal, never an unmeasured interval.
        assert refusal.reason_code == "cell_not_licensed"
        assert refusal.detail == "measured_inference.record_not_attesting"
    else:
        scalar = measured.scalars[0]
        assert measured.route == "cohort_interval"
        assert scalar.interval[0] <= scalar.point <= scalar.interval[1]
    # The point route is unchanged and never carries an interval by itself.
    assert abs(source.mean_contrast - (1.0 + 0.5 * float(np.mean(x[:80])))) < 0.2
    with pytest.raises(CausalValueError):
        source.interval(level=1.0)


def test_a_declared_compact_export_is_point_only_with_its_own_detail() -> None:
    x, fit = _native_fit()
    binding = cd.CohortBinding(
        snapshot_id="clinic", population="target", columns={"x": [0.0, 0.5]}, weights=[0.5, 0.5]
    )
    source = cd.decision_source_from_native_fit(fit, cohort=binding, outcome_units="mmHg")
    plain = cd.consume_cohort_decision_source(
        source.export(), expected_identity=source.identity, outcome_units="mmHg"
    )
    with pytest.raises(cd.CohortDecisionRefusal) as refused:
        plain.interval()
    assert refused.value.reason_code == "cell_not_licensed"
    assert refused.value.detail == "cohort_decision.interval_requires_native_fit"
