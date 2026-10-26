"""A retained fitted effect model carried to a cohort of another population under a declared
selection-diagram premise: value against the model's own CATE points, typed receipt and identity,
the existing decision surface, the named refusal of every unmet premise and fresh re-derivation.

Oracle. The data are ``y = t (effect + 0.4 z) + z`` with a randomized binary treatment, so the
conditional contrast is ``tau(z) = effect + 0.4 z`` and the fitted rows cover ``z in [-1, 1]``. The
cohort mean is the weighted sum of the model's own CATE predictions (computed here in numpy) and,
within the estimator's tolerance, of the truth ``effect + 0.4 * sum(w z)``.
"""

from __future__ import annotations

import dataclasses

import antecedent as ac
import numpy as np
import pytest
from antecedent import cohort_decision as cd
from antecedent import decision
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.estimators import DRLearner
from antecedent.outcomes import require_success
from antecedent.prediction import FittedEffectModel

Z = [-0.8, 0.0, 0.5]
W = [0.25, 0.25, 0.5]
TOLERANCE = 1e-9


def fitted_result(effect: float = 2.0):
    rng = np.random.default_rng(19)
    z = np.tile(np.linspace(-1.0, 1.0, 21), 80)
    treatment = rng.binomial(1, 0.5, len(z)).astype(float)
    return require_success(
        ac.analyze(
            {"z": z, "t": treatment, "y": treatment * (effect + 0.4 * z) + z},
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=ac.AverageEffect(treatment="t", outcome="y"),
            estimator=DRLearner(),
            bootstrap=0,
            refute=False,
            seed=3,
        )
    )


@pytest.fixture(scope="module")
def model() -> FittedEffectModel:
    return fitted_result().fitted_model


@pytest.fixture(scope="module")
def other_model() -> FittedEffectModel:
    return fitted_result(3.0).fitted_model


def cohort(z=Z, w=W, population: str = "clinic_b", snapshot: str = "clinic-b-2026-q1"):
    return cd.CohortBinding(
        snapshot_id=snapshot, population=population, columns={"z": list(z)}, weights=list(w)
    )


def shift(
    *,
    source: str = "clinic_a",
    target: str = "clinic_b",
    node: str = "S",
    children=("z",),
    modifiers=("z",),
) -> cd.SelectionShift:
    return cd.SelectionShift(source, target, node, tuple(children), tuple(modifiers))


def transport(model, *, cohort_=None, declared="default", units="mmHg", source="clinic_a"):
    return cd.transport_cohort_effect(
        model,
        cohort=cohort_ if cohort_ is not None else cohort(),
        source_population=source,
        shift=shift() if declared == "default" else declared,
        outcome_units=units,
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


# ------------------------------------------------------------------------------ the value


def test_transported_mean_matches_the_models_own_cate_points_with_a_typed_receipt(model) -> None:
    effect = transport(model)
    points = np.asarray(require_success(model.predict({"z": Z})).values)
    expected = float(np.dot(W, points))
    assert effect.mean_contrast == pytest.approx(expected, abs=TOLERANCE)
    assert effect.row_contrasts == pytest.approx(tuple(points), abs=TOLERANCE)
    # Within the estimator's tolerance, the structural truth.
    assert effect.mean_contrast == pytest.approx(2.0 + 0.4 * float(np.dot(W, Z)), abs=0.12)

    receipt = effect.receipt
    assert receipt.route == "cohort_effect"
    assert receipt.identity == effect.identity and len(effect.identity) == 64
    assert receipt.parent_claim == model.parent_claim
    assert receipt.source_population == "clinic_a" and receipt.target_population == "clinic_b"
    assert receipt.selection_node == "S"
    assert receipt.selection_children == ("z",) and receipt.effect_modifiers == ("z",)
    assert len(receipt.cohort_digest) == 64
    # Point-only, extrapolative, never target-population data.
    assert receipt.support == "extrapolative" and receipt.inference == "point_only"
    assert effect.native_target_data is False
    assert receipt.overlap["rows_outside"] == 0 and receipt.overlap["rows_checked"] == 3
    (feature,) = receipt.overlap["features"]
    assert feature["feature"] == "z" and feature["disjoint"] is False
    assert feature["target_lo"] == pytest.approx(min(Z)) and feature["target_hi"] == pytest.approx(
        max(Z)
    )
    assert any("selection" in line for line in receipt.assumptions)
    assert any("not a density or positivity guarantee" in line for line in receipt.assumptions)
    shared = receipt.as_transport_receipt()
    assert shared.route == "cohort_effect" and shared.support == "extrapolative"
    assert shared.transport_identities == (effect.identity,)


def test_the_decision_source_is_the_existing_surface_with_the_same_mean(model) -> None:
    effect = transport(model)
    source = effect.decision_source
    assert source.mean_contrast == pytest.approx(effect.mean_contrast, abs=TOLERANCE)
    assert source.support_status == "extrapolative"
    assert source.quantity.population_id == "clinic_b"
    contract = affine_contract(source, 5.0, -2.0)
    with pytest.raises(cd.CohortDecisionRefusal) as unaccepted:
        contract.evaluate(source)
    assert unaccepted.value.detail == "cohort_decision.extrapolative_not_accepted"
    decided = contract.evaluate(source.accept_extrapolative())
    assert effect.mean_source.means == pytest.approx((effect.mean_contrast,), abs=TOLERANCE)
    by_id = {a.id: a for a in decided.actions}
    assert by_id["treat"].expected_utility == pytest.approx(
        5.0 * effect.mean_contrast - 2.0, abs=1e-8
    )
    # The source replays under its own identity in a fresh consumer.
    again = require_success(
        cd.CohortDecisionSource.consume(bytes(source.export()), expected_identity=source.identity)
    )
    assert again.mean_contrast == source.mean_contrast


def test_changed_inputs_at_an_unchanged_shape_move_the_value_and_identity(
    model, other_model
) -> None:
    base = transport(model)
    reweighted = transport(model, cohort_=cohort(w=[0.5, 0.25, 0.25]))
    expected = float(
        np.dot([0.5, 0.25, 0.25], np.asarray(require_success(model.predict({"z": Z})).values))
    )
    assert reweighted.mean_contrast == pytest.approx(expected, abs=TOLERANCE)
    assert reweighted.identity != base.identity
    moved = transport(model, cohort_=cohort(z=[-0.8, 0.0, 0.6]))
    assert moved.identity != base.identity
    other = transport(other_model)
    assert other.identity != base.identity
    assert other.mean_contrast - base.mean_contrast == pytest.approx(1.0, abs=0.25)
    # Another declared selection node is a different declaration, so a different identity; the
    # number is unchanged because the premise licenses the same function.
    renamed = cd.transport_cohort_effect(
        model,
        cohort=cohort(),
        source_population="clinic_a",
        shift=shift(node="S2"),
        outcome_units="mmHg",
    )
    assert renamed.mean_contrast == pytest.approx(base.mean_contrast, abs=TOLERANCE)
    assert renamed.identity != base.identity


def test_a_fresh_consumer_rederives_the_identity_and_refuses_a_wrong_one(model) -> None:
    effect = transport(model)
    assert effect.verify().identity == effect.identity
    forged = dataclasses.replace(effect, identity="0" * 64)
    with pytest.raises(cd.CohortDecisionRefusal) as mismatch:
        forged.verify()
    assert mismatch.value.reason_code == "transport_not_certified"
    assert mismatch.value.detail == "transported_effect.identity_mismatch"


# ------------------------------------------------------------------------------ refusals


def test_every_unmet_premise_is_named(model) -> None:
    with pytest.raises(cd.CohortDecisionRefusal) as missing:
        transport(model, declared=None)
    assert missing.value.reason_code == "transport_missing_evidence"
    assert missing.value.detail == "transported_effect.shift_declaration_required"
    assert missing.value.remedy

    # A selection arrow into the outcome mechanism breaks the invariance premise.
    with pytest.raises(cd.CohortDecisionRefusal) as arrow:
        transport(model, declared=shift(children=("z", "y")))
    assert arrow.value.reason_code == "transport_not_certified"
    assert arrow.value.detail == "transported_effect.selection_enters_mechanism"
    assert arrow.value.offending == "y"

    # Modifiers that are not exactly the model's features.
    with pytest.raises(cd.CohortDecisionRefusal) as modifiers:
        transport(model, declared=shift(modifiers=("z", "age")))
    assert modifiers.value.reason_code == "transport_not_certified"
    assert modifiers.value.detail == "transported_effect.modifier_set_mismatch"

    # A declaration for other populations.
    with pytest.raises(cd.CohortDecisionRefusal) as populations:
        transport(model, declared=shift(source="clinic_x"))
    assert populations.value.detail == "transported_effect.shift_declaration_mismatch"

    # The cohort is declared to be the model's own population.
    with pytest.raises(cd.CohortDecisionRefusal) as same:
        transport(model, cohort_=cohort(population="clinic_a"))
    assert same.value.reason_code == "quantity_semantics_mismatch"
    assert same.value.detail == "transported_effect.population_not_different"


def test_overlap_failures_name_the_feature_and_row(model) -> None:
    with pytest.raises(cd.CohortDecisionRefusal) as disjoint:
        transport(model, cohort_=cohort(z=[3.0, 4.0], w=[0.5, 0.5]))
    assert disjoint.value.reason_code == "transport_support_failure"
    assert disjoint.value.detail == "transported_effect.no_overlap"
    assert "feature=z" in str(disjoint.value.offending)

    with pytest.raises(cd.CohortDecisionRefusal) as beyond:
        transport(model, cohort_=cohort(z=[0.0, 3.0], w=[0.5, 0.5]))
    assert beyond.value.reason_code == "transport_support_failure"
    assert beyond.value.detail == "transported_effect.out_of_support"
    assert "row=row-1" in str(beyond.value.offending)


def test_schema_units_and_type_gates(model) -> None:
    wrong_feature = cd.CohortBinding(
        snapshot_id="s", population="clinic_b", columns={"age": [1.0, 2.0]}
    )
    with pytest.raises(cd.CohortDecisionRefusal) as schema:
        transport(model, cohort_=wrong_feature)
    assert schema.value.reason_code == "schema_mismatch"
    assert schema.value.detail == "transported_effect.feature_schema_mismatch"

    levels = cd.CohortBinding(
        snapshot_id="s", population="clinic_b", columns={"z": ["low", "high"]}
    )
    with pytest.raises(cd.CohortDecisionRefusal) as categorical:
        transport(model, cohort_=levels)
    assert categorical.value.detail == "transported_effect.cohort_not_numeric"

    with pytest.raises(CausalValueError):
        transport(model, units="")
    with pytest.raises(CausalTypeError):
        cd.transport_cohort_effect(
            model,
            cohort=cohort(),
            source_population="clinic_a",
            shift={"selection_node": "S"},  # type: ignore[arg-type]
            outcome_units="mmHg",
        )


def test_interval_and_law_are_refused_with_the_licensing_evidence(model) -> None:
    effect = transport(model)
    with pytest.raises(cd.CohortDecisionRefusal) as interval:
        effect.interval()
    assert interval.value.reason_code == "cell_not_licensed"
    assert interval.value.detail == "transported_effect.interval_not_licensed"
    with pytest.raises(cd.CohortDecisionRefusal) as law:
        effect.refuse("law")
    assert law.value.reason_code == "decision_contract_unsatisfied"
    assert law.value.detail == "transported_effect.law_not_supplied"
    with pytest.raises(cd.CohortDecisionRefusal) as quantile:
        effect.refuse("quantile")
    assert quantile.value.reason_code == "decision_contract_unsatisfied"
