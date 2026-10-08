"""2.3.0 A5: the transported static path-specific counterfactual, narrow class.

Structural model (covariate ``z``, treatment ``a``, mediator ``m``, outcome ``y``)::

    M = 0.5 Z + (2 - 0.5 Z) A + U_M
    Y = (3 + 0.75 Z) A + 4 M + 1.5 Z + U_Y

with ``a1 = 2`` and ``a0 = 0.5`` (``a1 - a0 = 1.5``). By hand, per unit with covariate ``z``::

    NDE(z) = (3 + 0.75 z) * 1.5        = 4.5 + 1.125 z
    NIE(z) = 4 * (2 - 0.5 z) * 1.5     = 12 - 3 z
    total  = NDE + NIE                 = 16.5 - 1.875 z

Target law ``{0: 0.25, 1: 0.75}`` (``E z = 0.75``); source law ``{0: 0.4, 1: 0.4, 2: 0.2}``
(``E z = 0.8``). The noise cancels in every unit's contrast, so it is not modelled.
"""

import pytest
from antecedent.errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.temporal_counterfactual import transported_path_specific
from antecedent.transported_counterfactual import (
    AdditiveNoiseScm,
    Affine,
    CovariateLaw,
    EdgeAssignment,
    Mechanism,
    MechanismSelection,
    Premises,
    TransportedCounterfactualRefusal,
    TransportedPathSpecificEffect,
    consume_transported_counterfactual_artifact,
    transported_path_specific_effect,
)

from _refusal import assert_registered_refusal

A1, A0 = 2.0, 0.5


def scm(m_on_a=None):
    return AdditiveNoiseScm(
        treatment="a",
        covariates=("z",),
        mechanisms=(
            Mechanism("m", {"z": 0.5, "a": m_on_a or Affine(2.0, {"z": -0.5})}),
            Mechanism("y", {"a": Affine(3.0, {"z": 0.75}), "m": 4.0, "z": 1.5}),
        ),
    )


def source():
    return CovariateLaw.univariate("z", {0.0: 0.4, 1.0: 0.4, 2.0: 0.2})


def target(weights=None):
    return CovariateLaw.univariate("z", weights or {0.0: 0.25, 1.0: 0.75})


def nde():
    return EdgeAssignment.natural_direct("y", treated_value=A1, control_value=A0)


def nie():
    return EdgeAssignment.natural_indirect("y", ["m"], treated_value=A1, control_value=A0)


def total():
    return EdgeAssignment.total("y", ["m"], treated_value=A1, control_value=A0)


DECLARED = Premises(additive_noise=True, noise_laws_shared=True, cross_world_independence=True)


def effect(
    assignment=None,
    *,
    model=None,
    target_law=None,
    selection=None,
    premises=DECLARED,
    **kwargs,
):
    return transported_path_specific_effect(
        model or scm(),
        source(),
        target_law or target(),
        selection if selection is not None else MechanismSelection.on("z"),
        assignment or nde(),
        premises=premises,
        **kwargs,
    )


def refusal_of(**kwargs):
    with pytest.raises(TransportedCounterfactualRefusal) as raised:
        effect(**kwargs)
    error = raised.value
    assert isinstance(error, CausalUnsupportedError)
    assert_registered_refusal(error)
    return error


def test_x8t_hand_values_nde_nie_total():
    nde_result = effect(nde())
    assert nde_result.target_contrast == pytest.approx(4.5 + 1.125 * 0.75, abs=1e-12)
    assert nde_result.target_contrast == pytest.approx(5.34375, abs=1e-12)
    nie_result = effect(nie())
    assert nie_result.target_contrast == pytest.approx(9.75, abs=1e-12)
    total_result = effect(total())
    assert total_result.target_contrast == pytest.approx(15.09375, abs=1e-12)
    assert total_result.target_contrast == pytest.approx(
        nde_result.target_contrast + nie_result.target_contrast, abs=1e-12
    )
    # Per-unit contrasts at the target support points.
    by_z = {u.point["z"]: u.contrast for u in nde_result.unit_contrasts}
    assert by_z == pytest.approx({0.0: 4.5, 1.0: 5.625}, abs=1e-12)
    assert sum(u.target_weight for u in nde_result.unit_contrasts) == pytest.approx(1.0)
    assert nde_result.inference_claim == nde_result.derivation.claim == "point_only"


def test_x8t_derivation_separates_checked_from_declared_assumptions():
    derivation = effect().derivation
    assert derivation.theorem == "covariate_selected_additive_path_specific_transport"
    assert "no_selection_on_mediator_or_outcome" in derivation.checked
    assert "target_support_within_source_support" in derivation.checked
    assert "additive_noise" in derivation.declared
    assert "unit_level_cross_world_independence" in derivation.declared
    assert not set(derivation.checked) & set(derivation.declared)


def test_x8t_the_target_law_moves_the_answer_by_exactly_the_hand_amount():
    base = effect().target_contrast
    moved = effect(target_law=target({0.0: 0.5, 1.0: 0.5})).target_contrast
    # Mass moves from z=1 to z=0: E z goes 0.75 -> 0.5 and the NDE slope is 1.125.
    assert moved - base == pytest.approx(-0.28125, abs=1e-12)
    assert effect().source_contrast == pytest.approx(5.4, abs=1e-12)
    assert abs(effect().source_contrast - base) > 0.05


def test_x8t_selection_on_a_covariate_or_the_treatment_is_allowed():
    result = effect(selection=MechanismSelection.on("z", "a"))
    assert result.target_contrast == pytest.approx(5.34375, abs=1e-12)


def test_x8t_premises_default_to_undeclared_and_each_refuses_with_its_own_detail():
    args = (scm(), source(), target(), MechanismSelection.on("z"), nde())
    with pytest.raises(TransportedCounterfactualRefusal) as raised:
        transported_path_specific_effect(*args)
    assert raised.value.detail == "transported_counterfactual.nonadditive_mechanism"
    assert raised.value.reason_code == "cell_not_licensed"
    assert raised.value.offending == "additive_noise"
    cases = {
        "noise_law_not_shared": Premises(additive_noise=True),
        "cross_world_independence_missing": Premises(additive_noise=True, noise_laws_shared=True),
    }
    for detail, premises in cases.items():
        error = refusal_of(premises=premises)
        assert error.detail == f"transported_counterfactual.{detail}"
        assert error.reason_code == "cell_not_licensed"
        assert error.witness is None


def test_x8t_selection_on_the_outcome_refuses_with_a_two_model_witness():
    error = refusal_of(selection=MechanismSelection.on("y"))
    assert error.reason_code == "transport_proven_non_transportable"
    assert error.detail == "transported_counterfactual.selection_on_mechanism"
    witness = error.witness
    assert witness is not None
    assert (witness.selected_node, witness.perturbed_parent) == ("y", "a")
    assert witness.perturbed_slope_covariate is None
    # Both target models share the source model and agree on the source answer.
    assert witness.source_model == scm() == witness.target_model_a
    assert witness.target_model_b != witness.target_model_a
    assert witness.source_contrast == pytest.approx(5.4, abs=1e-12)
    # Raising the A coefficient of Y by 1 raises the NDE by 1 * (a1 - a0) = 1.5.
    assert witness.target_contrast_a == pytest.approx(5.34375, abs=1e-12)
    assert witness.target_contrast_b - witness.target_contrast_a == pytest.approx(1.5, abs=1e-12)


def test_x8t_selection_on_a_mediator_is_witnessed_only_when_the_contrast_depends_on_it():
    # The NDE does not depend on the mediator's equation: refused, no impossibility claimed.
    error = refusal_of(selection=MechanismSelection.on("m"))
    assert error.reason_code == "cell_not_licensed"
    assert error.detail == "transported_counterfactual.selection_on_mechanism"
    assert error.witness is None and error.offending == "m"
    # The NIE depends on it: witness (perturbing the A coefficient of M by 1 moves it by 6).
    error = refusal_of(assignment=nie(), selection=MechanismSelection.on("m"))
    assert error.reason_code == "transport_proven_non_transportable"
    witness = error.witness
    assert witness is not None and witness.selected_node == "m"
    assert witness.target_contrast_a == pytest.approx(9.75, abs=1e-12)
    assert witness.target_contrast_b == pytest.approx(15.75, abs=1e-12)
    assert witness.source_contrast == pytest.approx(9.6, abs=1e-12)


def test_x8t_target_outside_the_source_support_is_an_overlap_failure():
    error = refusal_of(target_law=target({0.0: 0.5, 3.0: 0.5}))
    assert error.reason_code == "transport_support_failure"
    assert error.detail == "transported_counterfactual.overlap_failure"
    assert error.offending == "z=3"


def test_x8t_absent_regime_evidence_is_factor_missing():
    error = refusal_of(evidence=[("source", "observational", "fit")])
    assert error.reason_code == "transport_missing_evidence"
    assert error.detail == "transported_counterfactual.factor_missing"
    assert error.missing_factors == ("target:observational",)
    withheld = refusal_of(evidence=())
    assert withheld.missing_factors == ("source:observational", "target:observational")


def test_x8t_malformed_inputs_are_refused_or_rejected():
    cyclic = AdditiveNoiseScm(
        "a",
        ("z",),
        (Mechanism("m", {"a": 1.0, "y": 1.0}), Mechanism("y", {"m": 1.0, "a": 1.0})),
    )
    error = refusal_of(model=cyclic)
    assert (error.reason_code, error.detail) == (
        "invalid_argument",
        "transported_counterfactual.invalid_model",
    )
    not_a_child = EdgeAssignment("y", A1, A0, ("z",))
    assert refusal_of(assignment=not_a_child).detail == "transported_counterfactual.invalid_query"
    bad_law = CovariateLaw.univariate("z", {0.0: 0.5, 1.0: 0.25})
    assert refusal_of(target_law=bad_law).detail == "transported_counterfactual.invalid_law"
    nowhere = MechanismSelection.on("nowhere")
    assert refusal_of(selection=nowhere).detail == "transported_counterfactual.invalid_diagram"
    with pytest.raises(CausalValueError):
        Affine(float("nan"))
    with pytest.raises(CausalTypeError):
        effect(premises=True)


def test_x8t_artifact_round_trips_and_replays():
    original = effect(nie())
    data = original.export()
    assert isinstance(data, bytes)
    replayed = consume_transported_counterfactual_artifact(
        data, expected_identity=original.identity
    )
    assert replayed.target_contrast == original.target_contrast
    assert replayed.source_contrast == original.source_contrast
    assert replayed.unit_contrasts == original.unit_contrasts
    assert replayed.derivation == original.derivation
    assert replayed.identity == original.identity
    again = TransportedPathSpecificEffect.consume(data)
    assert again.target_contrast == pytest.approx(9.75, abs=1e-12)
    # The identity is a function of the request: the same request gives the same digests.
    assert effect(nie()).identity == original.identity


def test_x8t_resealed_changes_are_refused_against_the_retained_identity():
    original = effect()
    retained = original.identity
    changed = {
        "model": effect(model=scm(m_on_a=Affine(3.0, {"z": -0.5}))),
        "target_law": effect(target_law=target({0.0: 0.5, 1.0: 0.5})),
        "selection": effect(selection=MechanismSelection.on("z", "a")),
        "assignment": effect(nie()),
        "evidence": effect(
            evidence=[
                ("source", "observational", "other-fit"),
                ("target", "observational", "target_covariate_law"),
            ]
        ),
    }
    for field, forged in changed.items():
        # A forger who reseals consistently is accepted without a retained identity ...
        assert consume_transported_counterfactual_artifact(forged.export()) is not None
        # ... and refused with one.
        with pytest.raises(TransportedCounterfactualRefusal) as raised:
            consume_transported_counterfactual_artifact(forged.export(), expected_identity=retained)
        assert raised.value.detail == "transported_counterfactual.artifact_changed"
        assert raised.value.reason_code == "route_not_supported"
        assert raised.value.offending == field
        assert_registered_refusal(raised.value)


def test_x8t_corrupt_artifacts_are_serialization_errors():
    data = effect().export()
    with pytest.raises(CausalSerializationError):
        consume_transported_counterfactual_artifact(data[: len(data) // 2])
    flipped = bytearray(data)
    flipped[len(flipped) // 2] ^= 0xFF
    with pytest.raises((CausalSerializationError, TransportedCounterfactualRefusal)):
        consume_transported_counterfactual_artifact(bytes(flipped))
    with pytest.raises(CausalTypeError):
        consume_transported_counterfactual_artifact("not bytes")  # type: ignore[arg-type]


def test_x8t_the_general_class_stays_closed():
    # The narrow-class cell does not open the closed general route.
    with pytest.raises(CausalUnsupportedError) as raised:
        transported_path_specific()
    assert raised.value.detail == "transported_counterfactual.route_frozen"
    assert raised.value.reason_code == "cell_not_licensed"
