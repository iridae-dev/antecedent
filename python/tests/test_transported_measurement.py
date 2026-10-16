"""A measurement model carried to another population under a declared invariance premise.

Oracles are computed here, never by native code: the inverted misclassification algebra
``pi = (p* + Sp - 1) / J`` with ``J = Se + Sp - 1`` and its delta-method variance term by term
(``p*(1 - p*)/n + (1 - pi)^2 Var(Sp) + pi^2 Var(Se)``, divided by ``J^2``). The carried parameters
are the source population's, so the transported value equals the algebra evaluated at the source's
rates and the target's observed positives; a model established for the target is used as is and
borrows nothing.
"""

from __future__ import annotations

import dataclasses

import pytest
from antecedent import measurement_models as mm
from antecedent._freeze import thaw
from antecedent.errors import CausalTypeError
from antecedent.measurement_models import MeasurementModelRefusal

SCALAR = mm.RepresentationVersion("scalar", 1)
REGIME = "observational"
CONSTRUCT = "construct:latent"
INDICATOR = "schema:indicator"
BOTH = ("sensitivity", "specificity")
SE, SP, VSE, VSP = 0.9, 0.8, 1e-4, 4e-4
EVENTS, UNITS = 520, 1000


def model(population: str, se: float = SE, sp: float = SP, *, regime: str = REGIME):
    return mm.MeasurementModel(
        indicator=mm.MeasurementVariable(INDICATOR, "noisy_misclassified", "unit"),
        construct=mm.MeasurementVariable(CONSTRUCT, "latent_scientific", "unit"),
        parameters=mm.BinaryParameters(
            (mm.BinaryRates(se, sp, VSE, VSP, arm_id=None),), min_youden=0.1
        ),
        population_id=population,
        regime_id=regime,
        evidence=mm.ParameterEvidence.validation_sample(f"val-{population}", units=500),
        structure="non_differential",
        representation=SCALAR,
    )


def analysis(population: str = "clinic_b", regime: str = REGIME) -> mm.AnalysisBinding:
    return mm.AnalysisBinding(construct_id=CONSTRUCT, population_id=population, regime_id=regime)


def invariance(
    parameters=BOTH, source: str = "clinic_a", target: str = "clinic_b"
) -> mm.InvarianceDeclaration:
    return mm.InvarianceDeclaration(source, target, "analyst", tuple(parameters))


DATA = mm.BinaryPrevalenceData(EVENTS, UNITS)


def oracle(se: float, sp: float):
    j = se + sp - 1.0
    p_star = EVENTS / UNITS
    pi = (p_star + sp - 1.0) / j
    variance = (p_star * (1 - p_star) / UNITS + (1 - pi) ** 2 * VSP + pi**2 * VSE) / j**2
    return pi, variance


def refused(call, code: str, detail: str) -> MeasurementModelRefusal:
    with pytest.raises(MeasurementModelRefusal) as caught:
        call()
    assert caught.value.reason_code == code, (caught.value.reason_code, str(caught.value))
    assert caught.value.detail == detail, (caught.value.detail, str(caught.value))
    return caught.value


# ------------------------------------------------------------------------------ carried


def test_carried_rates_correct_the_target_prevalence_with_a_typed_receipt() -> None:
    result = mm.transport_measurement_model(
        [model("clinic_a")], analysis(), DATA, structure=invariance()
    )
    pi, variance = oracle(SE, SP)
    assert pi == pytest.approx(0.32 / 0.7, abs=1e-15)
    estimate = result.estimate
    assert estimate.point == pytest.approx(pi, abs=1e-12)
    assert estimate.naive_point == pytest.approx(EVENTS / UNITS, abs=1e-15)
    assert estimate.standard_error**2 == pytest.approx(variance, abs=1e-12)
    assert estimate.route == "binary_misclassification"
    assert estimate.inference == "uncalibrated_standard_error"

    receipt = result.receipt
    assert result.identity == receipt.identity and len(receipt.identity) == 64
    assert receipt.carried is True
    assert receipt.support == "extrapolative"
    assert receipt.source_population == "clinic_a" and receipt.target_population == "clinic_b"
    assert receipt.basis["kind"] == "declared_invariance"
    assert tuple(sorted(receipt.basis["invariant"])) == BOTH
    assert receipt.basis["declarer"] == "analyst"
    assert any("invariant" in line for line in receipt.assumptions)
    assert any("not propagated" in line for line in receipt.assumptions)
    assert result.native_target_data is False
    # The source model's own identity is bound, and differs from the re-bound model's.
    assert receipt.source_model != estimate.model_identity

    # There is no interval, whoever asks.
    refused(result.interval, "cell_not_licensed", "measurement_model.interval_closed")


def test_the_transported_artifact_is_the_existing_one_and_replays() -> None:
    result = mm.transport_measurement_model(
        [model("clinic_a")], analysis(), DATA, structure=invariance()
    )
    consumed = mm.consume_measurement_model_artifact(
        result.export("transported"), result.estimate.model_identity
    )
    assert consumed.point == pytest.approx(result.estimate.point, abs=1e-12)
    assert consumed.model.population_id == "clinic_b"
    # The artifact never carries the receipt: a stale or foreign identity is refused on re-derivation.
    again = mm.verify_measurement_transport(
        [model("clinic_a")],
        analysis(),
        structure=invariance(),
        expected_identity=result.identity,
    )
    assert thaw(again) == thaw(result.receipt)
    refused(
        lambda: mm.verify_measurement_transport(
            [model("clinic_a")],
            analysis(),
            structure=invariance(),
            expected_identity="0" * 64,
        ),
        "measurement_binding_mismatch",
        "measurement_model.transport_receipt_mismatch",
    )


def test_changed_numbers_and_declarations_move_the_identity() -> None:
    base = mm.transport_measurement_model(
        [model("clinic_a")], analysis(), DATA, structure=invariance()
    )
    shifted = mm.transport_measurement_model(
        [model("clinic_a", se=0.85, sp=0.8)], analysis(), DATA, structure=invariance()
    )
    pi, _ = oracle(0.85, 0.8)
    assert shifted.estimate.point == pytest.approx(pi, abs=1e-12)
    assert shifted.identity != base.identity
    # Declaring a parameter the route does not use changes the declaration, not the value.
    wider = mm.transport_measurement_model(
        [model("clinic_a")],
        analysis(),
        DATA,
        structure=invariance(BOTH + ("error_variance",)),
    )
    assert wider.estimate.point == pytest.approx(base.estimate.point, abs=1e-15)
    assert wider.identity != base.identity
    assert wider.receipt.source_model == base.receipt.source_model


# ------------------------------------------------------------------------------ not carried


def test_a_model_established_for_the_target_borrows_nothing() -> None:
    models = [model("clinic_a"), model("clinic_b", se=0.7, sp=0.95)]
    result = mm.transport_measurement_model(
        models,
        analysis(),
        DATA,
        structure=mm.DifferentialErrorDeclaration("analyst"),
    )
    pi, _ = oracle(0.7, 0.95)
    assert pi == pytest.approx(0.47 / 0.65, abs=1e-15)
    assert result.estimate.point == pytest.approx(pi, abs=1e-12)
    assert result.receipt.carried is False
    assert result.receipt.support == "supported"
    assert result.receipt.basis["kind"] == "differential_established"
    # Without a declaration the same target model is simply its own.
    own = mm.transport_measurement_model(models, analysis(), DATA)
    assert own.receipt.basis["kind"] == "established"
    assert own.estimate.point == pytest.approx(pi, abs=1e-12)


# ------------------------------------------------------------------------------ refusals


def test_every_unmet_premise_is_named() -> None:
    source = [model("clinic_a")]
    missing = refused(
        lambda: mm.transport_measurement_model(source, analysis(), DATA),
        "transport_missing_evidence",
        "measurement_model.invariance_premise_required",
    )
    assert missing.remedy
    # A premise about specificity alone does not license a sensitivity-and-specificity correction.
    refused(
        lambda: mm.transport_measurement_model(
            source, analysis(), DATA, structure=invariance(("specificity",))
        ),
        "transport_not_certified",
        "measurement_model.invariance_not_declared",
    )
    # A declared population-differential error never falls back to another population's rates.
    refused(
        lambda: mm.transport_measurement_model(
            source, analysis(), DATA, structure=mm.DifferentialErrorDeclaration("analyst")
        ),
        "differential_error_not_pooled",
        "measurement_model.differential_population_unestablished",
    )
    # A premise for another target or an unknown source.
    refused(
        lambda: mm.transport_measurement_model(
            source, analysis(), DATA, structure=invariance(target="clinic_c")
        ),
        "transport_not_certified",
        "measurement_model.invariance_target_mismatch",
    )
    refused(
        lambda: mm.transport_measurement_model(
            source, analysis(), DATA, structure=invariance(source="clinic_z")
        ),
        "transport_not_certified",
        "measurement_model.invariance_source_unknown",
    )
    # The premise covers the population only: a model of another regime is not carried.
    refused(
        lambda: mm.transport_measurement_model(
            source, analysis(regime="do(a=1)"), DATA, structure=invariance()
        ),
        "measurement_binding_mismatch",
        "measurement_model.regime_mismatch",
    )


def test_a_route_with_nothing_to_carry_is_refused() -> None:
    repeated = dataclasses.replace(model("clinic_a"), parameters=mm.RepeatedParameters(20, 0.1))
    data = mm.RepeatedLinearData([1.0, 2.0, 3.0], [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [2, 2, 2])
    refused(
        lambda: mm.transport_measurement_model(
            [repeated], analysis(), data, structure=invariance(("error_variance",))
        ),
        "route_not_supported",
        "measurement_model.nothing_to_transport",
    )


def test_declaration_types_are_checked() -> None:
    with pytest.raises(CausalTypeError):
        mm.InvarianceDeclaration("clinic_a", "clinic_b", "analyst", "sensitivity")
    with pytest.raises(CausalTypeError):
        mm.transport_measurement_model(
            [model("clinic_a")],
            analysis(),
            DATA,
            structure={"kind": "invariant"},  # type: ignore[arg-type]
        )
    with pytest.raises(CausalTypeError):
        mm.transport_measurement_model([], analysis(), DATA, structure=invariance())
