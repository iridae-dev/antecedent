//! Typed bridges between scientific quantities: licensed bridges against hand
//! computed truth, the named substitution refusals, population and regime
//! binding, transportability, value-kind separation, identity and the derived
//! claim's relation to its source.
//!
//! Independent oracles used here:
//!
//! * trial-level surrogate regression: four validation trials with effects
//!   `(0, 0.1), (0.5, 0.5), (1, 0.9), (1.5, 1.5)` give by hand
//!   `Sxx = 1.25`, `Sxy = 1.15`, `Syy = 1.07`, slope `0.92`, intercept
//!   `0.06`, `R^2 = 1.15^2 / (1.25 * 1.07) = 529/535`;
//! * binary misclassification with sensitivity 0.9 and specificity 0.8:
//!   a recorded prevalence 0.31 is a latent prevalence
//!   `(0.31 + 0.8 - 1) / (0.9 + 0.8 - 1) = 11/70`, and the forward map of
//!   `11/70` is `0.9 * 11/70 + 0.2 * 59/70 = 0.31`;
//! * an enumerated confounded structural model (`Z ~ Bernoulli(0.5)`,
//!   `P(X=1|Z) = 0.2 + 0.6 Z`, `P(Y=1|X,Z) = 0.1 + 0.3 X + 0.4 Z`) whose
//!   association is `0.54` and whose interventional effect is `0.30`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeRefusal, BridgeScope, BridgeSource, MappingKind, QuantityClass, TransportabilityPremise,
    TrialEffectPair, TrialLevelValidation, ValueKind, derive_bridged_claim, fit_trial_level_linear,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, ClaimOperation, DerivedClaimOutcome, DomainStatus,
    QuantityRole, ReasoningView, ScientificQuantity, SharedEvidenceRef, SlotAvailability,
    compose_claims,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, DifferentialAxis, ErrorStructure, MeasurementBinding,
    MeasurementKind, MeasurementModel, MeasurementParameters, MeasurementRoute,
    MeasurementVariable, ParameterEvidence, RepresentationVersion,
};
use antecedent_core::{ContractIdentities, SemanticDigest};

// -- fixtures -------------------------------------------------------------------------------

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn close(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance
}

fn quantity(
    variable: &str,
    role: QuantityRole,
    units: &str,
    functional: &str,
    population: &str,
    regime: &str,
) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role,
        units: units.into(),
        population_id: population.into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn claim(byte: u8, value: f64, units: &str) -> ClaimEnvelope {
    ClaimEnvelope::new(
        digest(byte),
        ContractIdentities::new(
            digest(10),
            digest(11),
            Some(digest(12)),
            Some(digest(13)),
            digest(14),
            digest(15),
            digest(16),
        ),
        ClaimKind::Point,
        Some(value),
        Some(Arc::from(units)),
        ReasoningView::new(
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
        ),
        ClaimDomains::new(
            DomainStatus::Identified,
            DomainStatus::Supported,
            DomainStatus::Evaluated,
        ),
        None,
        [Arc::<str>::from("prior:p0"), Arc::<str>::from("joint_draws:draw-set-1")],
    )
}

fn assumption(id: &str, kind: AssumptionKind, evidence: Option<&str>) -> BridgeAssumption {
    BridgeAssumption {
        id: id.into(),
        kind,
        statement: format!("{id} holds"),
        evidence: evidence.map(str::to_owned),
    }
}

fn scope(population: &str, regime: &str) -> BridgeScope {
    BridgeScope { population_id: population.into(), regime_id: regime.into() }
}

fn contract(
    kind: BridgeKind,
    source: (QuantityClass, ScientificQuantity),
    target: (QuantityClass, ScientificQuantity),
    mapping: BridgeMapping,
    assumptions: Vec<BridgeAssumption>,
    evidence_scope: BridgeScope,
) -> BridgeContract {
    BridgeContract {
        kind,
        source: BridgeEndpoint { quantity: source.1, class: source.0 },
        target: BridgeEndpoint { quantity: target.1, class: target.0 },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope,
        mapping,
        assumptions,
        transport: None,

        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn assert_refused<T: std::fmt::Debug>(
    result: Result<T, BridgeRefusal>,
    code: &str,
    detail: &str,
) -> BridgeRefusal {
    let refusal = result.expect_err("the bridge must be refused");
    assert_eq!(refusal.code, code, "{refusal:?}");
    assert_eq!(refusal.detail, detail, "{refusal:?}");
    assert_eq!(refusal.validate(), Ok(()), "{refusal:?}");
    assert!(refusal.remedy.is_some(), "a refusal states what would license it: {refusal:?}");
    refusal
}

// -- surrogate to terminal outcome ---------------------------------------------------------------

fn validation_trials() -> Vec<TrialEffectPair> {
    [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)]
        .into_iter()
        .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
            surrogate_effect,
            terminal_effect,
        })
        .collect()
}

fn surrogate_quantity(population: &str) -> ScientificQuantity {
    quantity(
        "trial:biomarker",
        QuantityRole::Outcome,
        "mmol_per_l",
        "average_effect",
        population,
        "do(treatment=1)",
    )
}

fn terminal_quantity(population: &str) -> ScientificQuantity {
    quantity(
        "trial:survival",
        QuantityRole::Outcome,
        "log_hazard",
        "average_effect",
        population,
        "do(treatment=1)",
    )
}

fn surrogate_contract(min_trials: u32, min_r_squared: f64) -> BridgeContract {
    contract(
        BridgeKind::SurrogateTerminalOutcome,
        (QuantityClass::SurrogateEndpoint, surrogate_quantity("adults")),
        (QuantityClass::TerminalOutcome, terminal_quantity("adults")),
        BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials: validation_trials(),
            min_trials,
            min_r_squared,
        }),
        vec![assumption(
            "validation",
            AssumptionKind::SurrogateValidation,
            Some("meta-analysis:ckd-2024"),
        )],
        scope("adults", "do(treatment=1)"),
    )
}

#[test]
fn surrogate_bridge_matches_the_hand_computed_trial_level_regression() {
    let source_claim = claim(1, 0.8, "mmol_per_l");
    let source_quantity = surrogate_quantity("adults");
    let source = BridgeSource { claim: &source_claim, quantity: &source_quantity };
    let before = source_claim.clone();

    let bridged = derive_bridged_claim(&surrogate_contract(4, 0.9), &source).unwrap();

    // Independent truth: 0.06 + 0.92 * 0.8 = 0.796.
    assert!(close(bridged.value, 0.796, 1e-12), "{}", bridged.value);
    assert_eq!(bridged.claim.value, Some(bridged.value));
    let fit = bridged.fit.unwrap();
    assert!(close(fit.slope, 0.92, 1e-12));
    assert!(close(fit.intercept, 0.06, 1e-12));
    assert!(close(fit.r_squared, 529.0 / 535.0, 1e-12));
    assert_eq!(fit.trials, 4);
    assert!(close(fit.surrogate_min, 0.0, 0.0) && close(fit.surrogate_max, 1.5, 0.0));
    assert_eq!(bridged.mapping, MappingKind::TrialLevelLinear);
    assert!(!bridged.transported);

    // The bridge is its own claim; the source stays what it was.
    assert_eq!(source_claim, before, "the source claim is not modified");
    assert_ne!(bridged.claim.claim_id, source_claim.claim_id);
    assert_eq!(bridged.source_claim_id, source_claim.claim_id);
    assert_eq!(bridged.claim.kind, ClaimKind::Point);
    assert_eq!(bridged.claim.outcome_units.as_deref(), Some("log_hazard"));
    assert_eq!(bridged.value_kind, ValueKind::Mean);

    // Provenance: the source, then this derivation under it.
    assert_eq!(bridged.provenance.nodes.len(), 2);
    let derived_node = &bridged.provenance.nodes[1];
    assert_eq!(&*derived_node.artifact_id, bridged.claim.claim_id.to_hex());
    assert_eq!(&*derived_node.parents[0], source_claim.claim_id.to_hex());
    assert_eq!(&*derived_node.operation, "scientific_bridge:surrogate_terminal_outcome");
    assert_eq!(bridged.provenance.validate().unwrap().len(), 0);
    assert!(!bridged.receipt.equivalent_claim());
    assert_eq!(bridged.receipt.input, source_claim.claim_id);
    assert_eq!(bridged.receipt.output, Some(bridged.claim.claim_id));
    assert!(bridged.receipt.unresolved.iter().any(|item| &**item == "assumption:validation"));
    assert!(bridged.receipt.unavailable_operations.iter().any(|op| &**op == "interval"));

    // Nothing is established for the target quantity beyond the declared
    // assumptions: identification, support and uncertainty are unavailable.
    assert!(!bridged.claim.reasoning.identification.is_available());
    assert!(!bridged.claim.reasoning.support.is_available());
    assert!(!bridged.claim.reasoning.uncertainty.is_available());
    let obligations = &bridged.claim.reasoning.assumptions.as_ref().unwrap().obligations;
    assert_eq!(obligations.len(), 1);
    assert!(obligations[0].is_unresolved());
    assert_eq!(bridged.claim.domains.identification, DomainStatus::Unknown);
    assert_eq!(bridged.claim.domains.support, DomainStatus::Unknown);

    // The derived claim's evidence keeps the source's, drops its draw
    // alignment (a mapped point is not an aligned draw), and names the bridge.
    let evidence: Vec<&str> = bridged.claim.evidence.iter().map(|e| &**e).collect();
    assert!(evidence.contains(&"prior:p0"));
    assert!(!evidence.iter().any(|e| e.starts_with("joint_draws")));
    assert!(evidence.contains(&format!("bridge:{}", bridged.contract_identity.to_hex()).as_str()));
    assert!(
        evidence.contains(&format!("bridge_source:{}", source_claim.claim_id.to_hex()).as_str())
    );
}

#[test]
fn the_fit_is_the_ordinary_least_squares_line_at_every_trial_order() {
    let mut forward =
        TrialLevelValidation { trials: validation_trials(), min_trials: 3, min_r_squared: 0.0 };
    let reference = fit_trial_level_linear(&forward).unwrap();
    forward.trials.reverse();
    let reversed = fit_trial_level_linear(&forward).unwrap();
    forward.trials.swap(0, 2);
    let shuffled = fit_trial_level_linear(&forward).unwrap();
    for fit in [reversed, shuffled] {
        assert_eq!(fit.slope.to_bits(), reference.slope.to_bits());
        assert_eq!(fit.intercept.to_bits(), reference.intercept.to_bits());
        assert_eq!(fit.r_squared.to_bits(), reference.r_squared.to_bits());
    }

    // A second hand-computed set: collinear trials give R^2 = 1 and the
    // exact line y = 2x - 1.
    let line = TrialLevelValidation {
        trials: [(1.0, 1.0), (2.0, 3.0), (3.0, 5.0)]
            .into_iter()
            .map(|(s, t)| TrialEffectPair { surrogate_effect: s, terminal_effect: t })
            .collect(),
        min_trials: 3,
        min_r_squared: 1.0,
    };
    let exact = fit_trial_level_linear(&line).unwrap();
    assert!(close(exact.slope, 2.0, 1e-12) && close(exact.intercept, -1.0, 1e-12));
    assert!(close(exact.r_squared, 1.0, 1e-12));
}

#[test]
fn validation_minimums_and_the_validated_range_are_enforced() {
    let source_quantity = surrogate_quantity("adults");
    let at = |value: f64| claim(1, value, "mmol_per_l");

    let too_few = surrogate_contract(5, 0.9);
    let source_claim = at(0.8);
    let source = BridgeSource { claim: &source_claim, quantity: &source_quantity };
    assert_refused(
        derive_bridged_claim(&too_few, &source),
        "bridge_mapping_not_licensed",
        "scientific_bridge.validation_trials_too_few",
    );
    let weak = surrogate_contract(4, 0.995);
    let refusal = assert_refused(
        derive_bridged_claim(&weak, &source),
        "bridge_mapping_not_licensed",
        "scientific_bridge.validation_r_squared_too_low",
    );
    assert_eq!(refusal.expected.as_deref(), Some("0.995"));

    // The range of validated surrogate effects is [0, 1.5]; the ends are in,
    // anything beyond is an extrapolation and refused.
    let licensed = surrogate_contract(4, 0.9);
    for (value, expected) in [(0.0, 0.06), (1.5, 1.44)] {
        let claim_at = at(value);
        let source = BridgeSource { claim: &claim_at, quantity: &source_quantity };
        let bridged = derive_bridged_claim(&licensed, &source).unwrap();
        assert!(close(bridged.value, expected, 1e-12), "{value}: {}", bridged.value);
    }
    for value in [-0.01, 1.51, 3.0] {
        let claim_at = at(value);
        let source = BridgeSource { claim: &claim_at, quantity: &source_quantity };
        assert_refused(
            derive_bridged_claim(&licensed, &source),
            "bridge_mapping_not_licensed",
            "scientific_bridge.outside_validated_range",
        );
    }

    // Constant surrogate effects cannot identify a slope; constant terminal
    // effects have no association to assess; two trials always fit a line.
    let mut flat = surrogate_contract(4, 0.0);
    flat.mapping = BridgeMapping::TrialLevelLinear(TrialLevelValidation {
        trials: [(1.0, 0.1), (1.0, 0.5), (1.0, 0.9), (1.0, 1.5)]
            .into_iter()
            .map(|(s, t)| TrialEffectPair { surrogate_effect: s, terminal_effect: t })
            .collect(),
        min_trials: 4,
        min_r_squared: 0.0,
    });
    let claim_at = at(1.0);
    let source = BridgeSource { claim: &claim_at, quantity: &source_quantity };
    assert_refused(
        derive_bridged_claim(&flat, &source),
        "bridge_mapping_not_licensed",
        "scientific_bridge.degenerate_validation",
    );
    let mut two = surrogate_contract(2, 0.0);
    assert_refused(
        derive_bridged_claim(&two, &source),
        "scientific_bridge_invalid",
        "scientific_bridge.malformed_mapping",
    );
    two = surrogate_contract(4, 1.5);
    assert_refused(
        derive_bridged_claim(&two, &source),
        "scientific_bridge_invalid",
        "scientific_bridge.malformed_mapping",
    );
    let mut nan = surrogate_contract(4, 0.9);
    if let BridgeMapping::TrialLevelLinear(validation) = &mut nan.mapping {
        validation.trials[1].terminal_effect = f64::NAN;
    }
    assert_refused(
        derive_bridged_claim(&nan, &source),
        "scientific_bridge_invalid",
        "scientific_bridge.non_finite_value",
    );
}

// -- latent construct and measured outcome ---------------------------------------------------

fn clinic_model(sensitivity: f64, specificity: f64, min_youden: f64) -> MeasurementModel {
    MeasurementModel {
        model_id: "screening-flag".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "schema:flag".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "binary".into(),
        },
        construct: MeasurementVariable {
            id: "construct:disease".into(),
            kind: MeasurementKind::LatentScientific,
            units: "binary".into(),
        },
        binding: MeasurementBinding {
            population_id: "clinic-a".into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity,
                specificity,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            }],
            min_youden,
        },
        representation: RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 },
        evidence: ParameterEvidence::Declared { declarer: "lab".into() },
    }
}

fn prevalence(variable: &str, role: QuantityRole) -> ScientificQuantity {
    quantity(variable, role, "probability", "prevalence", "clinic-a", "observational")
}

fn correction_contract(
    model: MeasurementModel,
    source: (QuantityClass, ScientificQuantity),
    target: (QuantityClass, ScientificQuantity),
) -> BridgeContract {
    let measured_assumption = BridgeAssumption::for_measurement_model("flag-model", &model);
    contract(
        BridgeKind::LatentMeasuredOutcome,
        source,
        target,
        BridgeMapping::MeasurementCorrection(Box::new(model)),
        vec![measured_assumption],
        scope("clinic-a", "observational"),
    )
}

fn measured_to_latent(model: MeasurementModel) -> BridgeContract {
    correction_contract(
        model,
        (QuantityClass::MeasuredOutcome, prevalence("schema:flag", QuantityRole::Outcome)),
        (QuantityClass::LatentConstruct, prevalence("construct:disease", QuantityRole::Outcome)),
    )
}

#[test]
fn a_measured_prevalence_is_the_latent_prevalence_only_through_the_model() {
    let measured = prevalence("schema:flag", QuantityRole::Outcome);
    let source_claim = claim(2, 0.31, "probability");
    let source = BridgeSource { claim: &source_claim, quantity: &measured };

    let bridged =
        derive_bridged_claim(&measured_to_latent(clinic_model(0.9, 0.8, 0.1)), &source).unwrap();
    // Independent truth: (0.31 + 0.8 - 1) / (0.9 + 0.8 - 1) = 0.11 / 0.7 = 11/70.
    assert!(close(bridged.value, 11.0 / 70.0, 1e-12), "{}", bridged.value);
    assert!(!close(bridged.value, 0.31, 0.1), "the measured value is not the construct");
    assert_eq!(bridged.mapping, MappingKind::MeasurementCorrection);
    assert_eq!(bridged.contract.assumptions[0].kind, AssumptionKind::MeasurementModel);

    // The forward direction recovers the recorded prevalence: 0.9 * 11/70 +
    // 0.2 * 59/70 = 0.31, chained from the derived claim.
    let latent = prevalence("construct:disease", QuantityRole::Outcome);
    let forward = correction_contract(
        clinic_model(0.9, 0.8, 0.1),
        (QuantityClass::LatentConstruct, latent.clone()),
        (QuantityClass::MeasuredOutcome, measured),
    );
    let chained_source = BridgeSource { claim: &bridged.claim, quantity: &latent };
    let round_trip = derive_bridged_claim(&forward, &chained_source).unwrap();
    assert!(close(round_trip.value, 0.31, 1e-12), "{}", round_trip.value);
    // The chain keeps both derivations separately identified.
    assert_eq!(round_trip.source_claim_id, bridged.claim.claim_id);
    assert_ne!(round_trip.claim.claim_id, bridged.claim.claim_id);
    assert_ne!(round_trip.claim.claim_id, source_claim.claim_id);

    // A direct forward value too: latent 0.5 -> 0.9 * 0.5 + 0.2 * 0.5 = 0.55.
    let half = claim(3, 0.5, "probability");
    let direct = BridgeSource { claim: &half, quantity: &latent };
    let value = derive_bridged_claim(&forward, &direct).unwrap().value;
    assert!(close(value, 0.55, 1e-12), "{value}");
}

#[test]
fn the_correction_refuses_what_the_declared_rates_cannot_produce() {
    let measured = prevalence("schema:flag", QuantityRole::Outcome);
    let model = clinic_model(0.9, 0.8, 0.1);
    let at = |value: f64| claim(2, value, "probability");

    // Recorded prevalences outside [1 - Sp, Se] = [0.2, 0.9] cannot arise.
    for value in [0.05, 0.95] {
        let source_claim = at(value);
        let source = BridgeSource { claim: &source_claim, quantity: &measured };
        assert_refused(
            derive_bridged_claim(&measured_to_latent(model.clone()), &source),
            "bridge_mapping_not_licensed",
            "scientific_bridge.outside_attainable_range",
        );
    }
    // The ends of the attainable range map to 0 and 1.
    for (value, expected) in [(0.2, 0.0), (0.9, 1.0)] {
        let source_claim = at(value);
        let source = BridgeSource { claim: &source_claim, quantity: &measured };
        let bridged = derive_bridged_claim(&measured_to_latent(model.clone()), &source).unwrap();
        assert!(close(bridged.value, expected, 1e-12), "{value}: {}", bridged.value);
    }

    // Youden's index 0.05 is not above the declared minimum 0.1.
    let weak = clinic_model(0.55, 0.5, 0.1);
    let source_claim = at(0.5);
    let source = BridgeSource { claim: &source_claim, quantity: &measured };
    assert_refused(
        derive_bridged_claim(&measured_to_latent(weak), &source),
        "measurement_not_identified",
        "scientific_bridge.youden_too_small",
    );

    // A differential model has no single pair of rates for a claim without an arm.
    let mut differential = clinic_model(0.9, 0.8, 0.1);
    differential.structure = ErrorStructure::Differential(DifferentialAxis::Arm);
    differential.parameters = MeasurementParameters::Binary {
        rates: ["a", "b"]
            .into_iter()
            .map(|arm| BinaryArmRates {
                arm_id: Some(arm.into()),
                sensitivity: 0.9,
                specificity: 0.8,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            })
            .collect(),
        min_youden: 0.1,
    };
    assert_refused(
        derive_bridged_claim(&measured_to_latent(differential), &source),
        "differential_error_not_pooled",
        "scientific_bridge.differential_error",
    );

    // A prevalence on another scale, or another functional, is not inverted.
    let mut log_scale = measured.clone();
    log_scale.transform_id = "log_odds".into();
    let source = BridgeSource { claim: &source_claim, quantity: &log_scale };
    let mut on_log = measured_to_latent(model.clone());
    on_log.source.quantity = log_scale.clone();
    assert_refused(
        derive_bridged_claim(&on_log, &source),
        "bridge_mapping_not_licensed",
        "scientific_bridge.functional_not_supported",
    );
}

#[test]
fn the_measurement_model_binding_is_checked_against_the_bridge() {
    let measured = prevalence("schema:flag", QuantityRole::Outcome);
    let source_claim = claim(2, 0.31, "probability");
    let source = BridgeSource { claim: &source_claim, quantity: &measured };

    // The assumption must name exactly the model the mapping uses.
    let mut other_identity = measured_to_latent(clinic_model(0.9, 0.8, 0.1));
    other_identity.assumptions[0].evidence =
        Some(clinic_model(0.95, 0.8, 0.1).identities().model.to_hex());
    assert_refused(
        derive_bridged_claim(&other_identity, &source),
        "bridge_binding_mismatch",
        "scientific_bridge.model_identity_mismatch",
    );

    // The model's indicator and construct are the bridge's two quantities.
    let mut wrong_construct = clinic_model(0.9, 0.8, 0.1);
    wrong_construct.construct.id = "construct:other".into();
    assert_refused(
        derive_bridged_claim(&measured_to_latent(wrong_construct), &source),
        "bridge_binding_mismatch",
        "scientific_bridge.measurement_model_mismatch",
    );

    // The model was established for a population and regime; the bridge's
    // evidence scope must be that scope.
    let mut other_population = clinic_model(0.9, 0.8, 0.1);
    other_population.binding.population_id = "clinic-b".into();
    assert_refused(
        derive_bridged_claim(&measured_to_latent(other_population), &source),
        "bridge_binding_mismatch",
        "scientific_bridge.mapping_scope_mismatch",
    );
}

#[test]
fn a_proxy_is_never_the_construct_without_a_measurement_model() {
    let measured = prevalence("schema:flag", QuantityRole::Outcome);
    let source_claim = claim(2, 0.31, "probability");
    let source = BridgeSource { claim: &source_claim, quantity: &measured };

    // No measurement model declared.
    let mut undeclared = measured_to_latent(clinic_model(0.9, 0.8, 0.1));
    undeclared.assumptions.clear();
    let refusal = assert_refused(
        derive_bridged_claim(&undeclared, &source),
        "proxy_offered_as_construct",
        "scientific_bridge.proxy_as_construct",
    );
    assert_eq!(refusal.supplied.as_deref(), Some("measured_outcome"));
    assert_eq!(refusal.expected.as_deref(), Some("latent_construct"));

    // A declared model whose mapping leaves the value unchanged is the proxy
    // renamed: refused under the same name.
    let mut renamed = measured_to_latent(clinic_model(0.9, 0.8, 0.1));
    renamed.mapping = BridgeMapping::Identity;
    assert_refused(
        derive_bridged_claim(&renamed, &source),
        "proxy_offered_as_construct",
        "scientific_bridge.proxy_as_construct",
    );

    // A measurement-model assumption that names no model is not evidence.
    let mut anonymous = measured_to_latent(clinic_model(0.9, 0.8, 0.1));
    anonymous.assumptions[0].evidence = None;
    assert_refused(
        derive_bridged_claim(&anonymous, &source),
        "bridge_assumption_missing",
        "scientific_bridge.assumption_evidence_missing",
    );
}

// -- predictive quantity and causal quantity ---------------------------------------------------

/// The enumerated confounded model: returns `(association, causal_effect)`.
fn confounded_model_truth() -> (f64, f64) {
    let pz = [0.5, 0.5];
    let px1 = |z: usize| 0.2 + 0.6 * z as f64;
    let py1 = |x: usize, z: usize| 0.1 + 0.3 * x as f64 + 0.4 * z as f64;
    let mut joint_x = [0.0_f64; 2];
    let mut joint_xy = [0.0_f64; 2];
    let mut interventional = [0.0_f64; 2];
    for (z, &p_z) in pz.iter().enumerate() {
        for x in 0..2 {
            let p_x = if x == 1 { px1(z) } else { 1.0 - px1(z) };
            joint_x[x] += p_z * p_x;
            joint_xy[x] += p_z * p_x * py1(x, z);
            interventional[x] += p_z * py1(x, z);
        }
    }
    let association = joint_xy[1] / joint_x[1] - joint_xy[0] / joint_x[0];
    (association, interventional[1] - interventional[0])
}

fn predictive_contract(assumptions: Vec<BridgeAssumption>) -> BridgeContract {
    contract(
        BridgeKind::PredictiveCausalQuantity,
        (
            QuantityClass::PredictiveQuantity,
            quantity(
                "cohort:y",
                QuantityRole::Outcome,
                "probability",
                "contrast",
                "registry",
                "observational",
            ),
        ),
        (
            QuantityClass::CausalQuantity,
            quantity(
                "cohort:y",
                QuantityRole::Outcome,
                "probability",
                "contrast",
                "registry",
                "do(x=1)",
            ),
        ),
        BridgeMapping::Identity,
        assumptions,
        scope("registry", "observational"),
    )
}

#[test]
fn an_association_is_a_causal_quantity_only_under_a_named_identification() {
    // Enumerated truth: the confounded association is 0.54, the effect 0.30.
    let (association, effect) = confounded_model_truth();
    assert!(close(association, 0.54, 1e-12), "{association}");
    assert!(close(effect, 0.30, 1e-12), "{effect}");

    let predictive = quantity(
        "cohort:y",
        QuantityRole::Outcome,
        "probability",
        "contrast",
        "registry",
        "observational",
    );

    // The unadjusted association offered as the intervention: refused, and the
    // number it would have published is wrong by 0.24.
    let unadjusted = claim(4, association, "probability");
    let source = BridgeSource { claim: &unadjusted, quantity: &predictive };
    let refusal = assert_refused(
        derive_bridged_claim(&predictive_contract(Vec::new()), &source),
        "bridge_substitution_refused",
        "scientific_bridge.association_as_intervention",
    );
    assert_eq!(refusal.supplied.as_deref(), Some("predictive_quantity"));
    assert_eq!(refusal.expected.as_deref(), Some("causal_quantity"));
    assert!(close(association - effect, 0.24, 1e-12));

    // The back-door standardized contrast, with its identification named, is
    // licensed and carries its value unchanged to the interventional regime.
    let identified = vec![assumption(
        "backdoor",
        AssumptionKind::Identification,
        Some("identification:backdoor-z"),
    )];
    let adjusted = claim(5, effect, "probability");
    let source = BridgeSource { claim: &adjusted, quantity: &predictive };
    let bridged = derive_bridged_claim(&predictive_contract(identified.clone()), &source).unwrap();
    assert!(close(bridged.value, 0.30, 1e-12));
    assert_eq!(bridged.mapping, MappingKind::Identity);
    assert_eq!(bridged.contract.target.quantity.regime_id, "do(x=1)");

    // An identification assumption that names nothing is not an identification.
    let mut anonymous = identified.clone();
    anonymous[0].evidence = None;
    assert_refused(
        derive_bridged_claim(&predictive_contract(anonymous), &source),
        "bridge_assumption_missing",
        "scientific_bridge.assumption_evidence_missing",
    );
    // An affine rescaling is not the identification of the same quantity.
    let mut rescaled = predictive_contract(identified);
    rescaled.mapping = BridgeMapping::Affine { intercept: 0.0, slope: 1.0 };
    assert_refused(
        derive_bridged_claim(&rescaled, &source),
        "scientific_bridge_invalid",
        "scientific_bridge.mapping_not_allowed",
    );
}

fn associational_contract(assumptions: Vec<BridgeAssumption>) -> BridgeContract {
    let mut associational = predictive_contract(assumptions);
    associational.source.class = QuantityClass::AssociationalQuantity;
    associational
}

#[test]
fn an_associational_quantity_is_refused_as_a_causal_quantity_until_identified() {
    assert_eq!(QuantityClass::AssociationalQuantity.as_str(), "associational_quantity");
    assert_eq!(
        QuantityClass::from_name("associational_quantity"),
        Some(QuantityClass::AssociationalQuantity)
    );
    let (association, effect) = confounded_model_truth();
    let observational = quantity(
        "cohort:y",
        QuantityRole::Outcome,
        "probability",
        "contrast",
        "registry",
        "observational",
    );

    // The association offered where a causal quantity is required is the named substitution.
    let unadjusted = claim(4, association, "probability");
    let source = BridgeSource { claim: &unadjusted, quantity: &observational };
    let refusal = assert_refused(
        derive_bridged_claim(&associational_contract(Vec::new()), &source),
        "bridge_substitution_refused",
        "scientific_bridge.association_as_intervention",
    );
    assert_eq!(refusal.supplied.as_deref(), Some("associational_quantity"));
    assert_eq!(refusal.expected.as_deref(), Some("causal_quantity"));

    // An assumption of another kind does not stand in for the identification.
    let linkage = vec![assumption("linkage", AssumptionKind::BehavioralLinkage, Some("study"))];
    assert_refused(
        derive_bridged_claim(&associational_contract(linkage), &source),
        "bridge_substitution_refused",
        "scientific_bridge.association_as_intervention",
    );

    // Only a named identification licenses the same pair, and it carries the identified value.
    let identified = vec![assumption(
        "backdoor",
        AssumptionKind::Identification,
        Some("identification:backdoor-z"),
    )];
    let adjusted = claim(5, effect, "probability");
    let source = BridgeSource { claim: &adjusted, quantity: &observational };
    let bridged = derive_bridged_claim(&associational_contract(identified), &source).unwrap();
    assert!(close(bridged.value, effect, 1e-12));
}

#[test]
fn an_associational_claim_is_never_a_bridge_source() {
    let observational = quantity(
        "cohort:y",
        QuantityRole::Outcome,
        "probability",
        "contrast",
        "registry",
        "observational",
    );
    let identified = vec![assumption(
        "backdoor",
        AssumptionKind::Identification,
        Some("identification:backdoor-z"),
    )];
    let mut association = claim(6, 0.54, "probability");
    association.kind = ClaimKind::Associational;
    association.domains = ClaimDomains::new(
        DomainStatus::NotClaimed,
        DomainStatus::Supported,
        DomainStatus::Evaluated,
    );
    let source = BridgeSource { claim: &association, quantity: &observational };
    let refusal = assert_refused(
        derive_bridged_claim(&associational_contract(identified), &source),
        "cell_not_licensed",
        "scientific_bridge.source_not_point",
    );
    assert_eq!(refusal.supplied.as_deref(), Some("associational"));
}

#[test]
fn predictive_importance_and_model_scores_are_closed_whatever_is_declared() {
    let everything = vec![
        assumption("backdoor", AssumptionKind::Identification, Some("identification:any")),
        assumption("linkage", AssumptionKind::BehavioralLinkage, Some("study")),
        assumption("calibration", AssumptionKind::Additional, None),
    ];
    let feature = quantity(
        "m:feature",
        QuantityRole::Outcome,
        "probability",
        "importance",
        "registry",
        "observational",
    );
    let effect = quantity(
        "m:feature",
        QuantityRole::Outcome,
        "probability",
        "effect",
        "registry",
        "do(x=1)",
    );
    let importance = contract(
        BridgeKind::PredictiveCausalQuantity,
        (QuantityClass::PredictiveImportance, feature.clone()),
        (QuantityClass::CausalQuantity, effect),
        BridgeMapping::Identity,
        everything.clone(),
        scope("registry", "observational"),
    );
    let source_claim = claim(6, 0.12, "probability");
    let source = BridgeSource { claim: &source_claim, quantity: &feature };
    let refusal = assert_refused(
        derive_bridged_claim(&importance, &source),
        "bridge_substitution_refused",
        "scientific_bridge.predictive_importance_as_causal_effect",
    );
    assert_eq!(refusal.supplied.as_deref(), Some("predictive_importance"));

    let score =
        quantity("m:score", QuantityRole::Outcome, "score", "mean", "registry", "observational");
    let probability = quantity(
        "m:score",
        QuantityRole::Outcome,
        "probability",
        "mean",
        "registry",
        "observational",
    );
    let model_score = contract(
        BridgeKind::PredictiveCausalQuantity,
        (QuantityClass::ModelScore, score.clone()),
        (QuantityClass::PredictiveQuantity, probability),
        BridgeMapping::Identity,
        everything,
        scope("registry", "observational"),
    );
    let source_claim = claim(7, 0.7, "score");
    let source = BridgeSource { claim: &source_claim, quantity: &score };
    let refusal = assert_refused(
        derive_bridged_claim(&model_score, &source),
        "bridge_substitution_refused",
        "scientific_bridge.model_score_as_probability",
    );
    assert_eq!(refusal.supplied.as_deref(), Some("model_score"));
    assert_eq!(refusal.expected.as_deref(), Some("predictive_quantity"));
}

// -- preference and behavior, parameter and response, endpoint and utility ---------------------

#[test]
fn a_stated_preference_is_realized_behavior_only_under_a_declared_linkage() {
    let stated = quantity(
        "survey:take_up",
        QuantityRole::Outcome,
        "probability",
        "mean",
        "panel",
        "observational",
    );
    let realized = quantity(
        "admin:take_up",
        QuantityRole::Outcome,
        "probability",
        "mean",
        "panel",
        "observational",
    );
    let make = |mapping: BridgeMapping, assumptions: Vec<BridgeAssumption>| {
        contract(
            BridgeKind::PreferenceRealizedBehavior,
            (QuantityClass::StatedPreference, stated.clone()),
            (QuantityClass::RealizedBehavior, realized.clone()),
            mapping,
            assumptions,
            scope("panel", "observational"),
        )
    };
    let source_claim = claim(8, 0.5, "probability");
    let source = BridgeSource { claim: &source_claim, quantity: &stated };

    let refusal = assert_refused(
        derive_bridged_claim(&make(BridgeMapping::Identity, Vec::new()), &source),
        "bridge_substitution_refused",
        "scientific_bridge.stated_preference_as_realized_behavior",
    );
    assert_eq!(refusal.supplied.as_deref(), Some("stated_preference"));
    assert_eq!(refusal.expected.as_deref(), Some("realized_behavior"));

    // A declared hypothetical-bias calibration: realized = 0.6 * stated, so
    // a stated take-up of 0.5 is a realized take-up of 0.3.
    let linkage = vec![assumption(
        "hypothetical-bias",
        AssumptionKind::BehavioralLinkage,
        Some("validation-panel-2023"),
    )];
    let calibrated = derive_bridged_claim(
        &make(BridgeMapping::Affine { intercept: 0.0, slope: 0.6 }, linkage.clone()),
        &source,
    )
    .unwrap();
    assert!(close(calibrated.value, 0.3, 1e-12), "{}", calibrated.value);
    // The declared linkage with an unchanged value is the assertion that stated
    // equals realized; it is licensed only because it is declared.
    let unchanged = derive_bridged_claim(&make(BridgeMapping::Identity, linkage), &source).unwrap();
    assert!(close(unchanged.value, 0.5, 0.0));
    assert_ne!(unchanged.claim.claim_id, calibrated.claim.claim_id);
}

#[test]
fn a_model_parameter_is_an_intervention_response_only_through_its_declared_contrast() {
    let parameter = quantity(
        "fit:dose_coefficient",
        QuantityRole::Outcome,
        "per_unit_dose",
        "coefficient",
        "trial",
        "observational",
    );
    let response = quantity(
        "fit:dose_coefficient",
        QuantityRole::Outcome,
        "outcome_units",
        "average_effect",
        "trial",
        "do(dose=+0.4)",
    );
    let make = |assumptions: Vec<BridgeAssumption>| {
        contract(
            BridgeKind::ParameterInterventionResponse,
            (QuantityClass::ModelParameter, parameter.clone()),
            (QuantityClass::InterventionResponse, response.clone()),
            BridgeMapping::Affine { intercept: 0.0, slope: 0.4 },
            assumptions,
            scope("trial", "observational"),
        )
    };
    let source_claim = claim(9, 2.5, "per_unit_dose");
    let source = BridgeSource { claim: &source_claim, quantity: &parameter };
    assert_refused(
        derive_bridged_claim(&make(Vec::new()), &source),
        "bridge_assumption_missing",
        "scientific_bridge.assumption_missing",
    );
    let interpreted = vec![assumption(
        "linear-structural-equation",
        AssumptionKind::ParameterInterpretation,
        None,
    )];
    let bridged = derive_bridged_claim(&make(interpreted), &source).unwrap();
    // Hand value: a +0.4 dose change moves the outcome by 2.5 * 0.4 = 1.0.
    assert!(close(bridged.value, 1.0, 1e-12), "{}", bridged.value);
    assert_eq!(bridged.claim.outcome_units.as_deref(), Some("outcome_units"));
}

#[test]
fn a_scientific_endpoint_is_a_utility_only_through_a_declared_utility_function() {
    let endpoint = quantity(
        "trial:qaly_gain",
        QuantityRole::Outcome,
        "qaly",
        "average_effect",
        "adults",
        "do(treatment=1)",
    );
    let utility = quantity(
        "payer:value",
        QuantityRole::Utility,
        "currency",
        "expected_utility",
        "adults",
        "do(treatment=1)",
    );
    let make = |mapping: BridgeMapping, assumptions: Vec<BridgeAssumption>| {
        contract(
            BridgeKind::ScientificDecisionUtility,
            (QuantityClass::ScientificEndpoint, endpoint.clone()),
            (QuantityClass::DecisionUtility, utility.clone()),
            mapping,
            assumptions,
            scope("adults", "do(treatment=1)"),
        )
    };
    let source_claim = claim(10, 0.3, "qaly");
    let source = BridgeSource { claim: &source_claim, quantity: &endpoint };
    let declared = vec![assumption(
        "willingness-to-pay",
        AssumptionKind::UtilityMapping,
        Some("utility-function:wtp-30000"),
    )];

    // Neither no mapping assumption nor an unchanged value is a utility.
    let refusal = assert_refused(
        derive_bridged_claim(
            &make(BridgeMapping::Affine { intercept: 0.0, slope: 30_000.0 }, Vec::new()),
            &source,
        ),
        "bridge_substitution_refused",
        "scientific_bridge.scientific_endpoint_as_utility",
    );
    assert_eq!(refusal.supplied.as_deref(), Some("scientific_endpoint"));
    assert_refused(
        derive_bridged_claim(&make(BridgeMapping::Identity, declared.clone()), &source),
        "bridge_substitution_refused",
        "scientific_bridge.scientific_endpoint_as_utility",
    );
    let mut anonymous = declared.clone();
    anonymous[0].evidence = None;
    assert_refused(
        derive_bridged_claim(
            &make(BridgeMapping::Affine { intercept: 0.0, slope: 30_000.0 }, anonymous),
            &source,
        ),
        "bridge_assumption_missing",
        "scientific_bridge.assumption_evidence_missing",
    );

    // A declared linear utility of 30,000 per QALY: 0.3 QALY is worth 9,000.
    // Exact for a mean, because the expectation of an affine function is the
    // affine function of the expectation.
    let bridged = derive_bridged_claim(
        &make(BridgeMapping::Affine { intercept: 0.0, slope: 30_000.0 }, declared),
        &source,
    )
    .unwrap();
    assert!(close(bridged.value, 9_000.0, 1e-9), "{}", bridged.value);
}

// -- value kinds, binding, transport, identity, derived claim -----------------------------------

#[test]
fn a_mean_a_law_a_posterior_a_set_and_a_joint_law_are_not_interchangeable() {
    let source_claim = claim(1, 0.8, "mmol_per_l");
    let source_quantity = surrogate_quantity("adults");
    let source = BridgeSource { claim: &source_claim, quantity: &source_quantity };

    for held in ValueKind::ALL {
        for requested in ValueKind::ALL {
            let mut request = surrogate_contract(4, 0.9);
            request.source_value = held;
            request.target_value = requested;
            let result = derive_bridged_claim(&request, &source);
            if held == requested && held == ValueKind::Mean {
                assert!(result.is_ok(), "a mean bridges to a mean");
            } else if held == requested {
                let refusal = assert_refused(
                    result,
                    "cell_not_licensed",
                    "scientific_bridge.value_kind_not_carried",
                );
                assert_eq!(refusal.supplied.as_deref(), Some(held.as_str()));
            } else {
                let refusal = assert_refused(
                    result,
                    "bridge_substitution_refused",
                    "scientific_bridge.value_kind_substitution",
                );
                assert_eq!(refusal.supplied.as_deref(), Some(held.as_str()));
                assert_eq!(refusal.expected.as_deref(), Some(requested.as_str()));
            }
        }
    }

    // A consumer that needs a law from a bridged mean is refused.
    let bridged = derive_bridged_claim(&surrogate_contract(4, 0.9), &source).unwrap();
    assert!(bridged.require_value_kind(ValueKind::Mean).is_ok());
    for needed in [
        ValueKind::PredictiveLaw,
        ValueKind::Posterior,
        ValueKind::IdentifiedSet,
        ValueKind::JointOutcomeLaw,
    ] {
        let refusal = bridged.require_value_kind(needed).unwrap_err();
        assert_eq!(refusal.detail, "scientific_bridge.value_kind_substitution");
        assert_eq!(refusal.code, "bridge_substitution_refused");
    }
    // And no interval or posterior is reachable from it.
    let refusal = bridged.request_interval().unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "scientific_bridge.interval_not_licensed");
}

#[test]
fn population_regime_conditioning_and_the_claimed_quantity_must_agree() {
    let source_claim = claim(1, 0.8, "mmol_per_l");
    let source_quantity = surrogate_quantity("adults");
    let source = BridgeSource { claim: &source_claim, quantity: &source_quantity };

    let mut other_population = surrogate_contract(4, 0.9);
    other_population.target.quantity = terminal_quantity("children");
    assert_refused(
        derive_bridged_claim(&other_population, &source),
        "bridge_binding_mismatch",
        "scientific_bridge.population_mismatch",
    );
    let mut other_regime = surrogate_contract(4, 0.9);
    other_regime.target.quantity.regime_id = "do(treatment=2)".into();
    assert_refused(
        derive_bridged_claim(&other_regime, &source),
        "bridge_binding_mismatch",
        "scientific_bridge.regime_mismatch",
    );
    let mut other_horizon = surrogate_contract(4, 0.9);
    other_horizon.target.quantity.horizon = 3;
    assert_refused(
        derive_bridged_claim(&other_horizon, &source),
        "bridge_binding_mismatch",
        "scientific_bridge.horizon_mismatch",
    );
    let mut other_condition = surrogate_contract(4, 0.9);
    other_condition.target.quantity.conditioning = vec![antecedent_contract::QuantityCondition {
        variable_id: "schema:age".into(),
        value_id: "over_65".into(),
    }];
    assert_refused(
        derive_bridged_claim(&other_condition, &source),
        "bridge_binding_mismatch",
        "scientific_bridge.conditioning_mismatch",
    );

    // The claim must be about the contract's source quantity, in its units.
    let licensed = surrogate_contract(4, 0.9);
    let wrong = quantity(
        "trial:other",
        QuantityRole::Outcome,
        "mmol_per_l",
        "average_effect",
        "adults",
        "do(treatment=1)",
    );
    let wrong_source = BridgeSource { claim: &source_claim, quantity: &wrong };
    let refusal = assert_refused(
        derive_bridged_claim(&licensed, &wrong_source),
        "bridge_binding_mismatch",
        "scientific_bridge.quantity_mismatch",
    );
    assert_eq!(refusal.offending.as_deref(), Some("variable"));
    let in_other_units = claim(1, 0.8, "mg_per_dl");
    let source = BridgeSource { claim: &in_other_units, quantity: &source_quantity };
    assert_refused(
        derive_bridged_claim(&licensed, &source),
        "bridge_binding_mismatch",
        "scientific_bridge.units_mismatch",
    );
}

#[test]
fn a_bridge_established_elsewhere_names_its_transportability_premise() {
    // The surrogate relationship was validated in adult trials; the claim is
    // about children.
    let children = surrogate_quantity("children");
    let mut transported = surrogate_contract(4, 0.9);
    transported.source.quantity = children.clone();
    transported.target.quantity = terminal_quantity("children");
    let source_claim = claim(1, 0.8, "mmol_per_l");
    let source = BridgeSource { claim: &source_claim, quantity: &children };

    let refusal = assert_refused(
        derive_bridged_claim(&transported, &source),
        "bridge_assumption_missing",
        "scientific_bridge.transport_premise_missing",
    );
    let named = refusal.expected.unwrap();
    assert!(named.contains("adults") && named.contains("children"), "{named}");
    assert!(named.contains("surrogate_terminal_outcome"), "{named}");

    // A premise for another transport is not this one.
    let premise = |from: BridgeScope, to: BridgeScope| TransportabilityPremise {
        id: "invariant-effect-modification".into(),
        statement: "the trial-level surrogate relationship is the same in children".into(),
        established_in: from,
        applies_to: to,
        evidence: Some("selection-diagram:pediatric".into()),
    };
    let mut wrong = transported.clone();
    wrong.transport =
        Some(premise(scope("adults", "do(treatment=1)"), scope("infants", "do(treatment=1)")));
    assert_refused(
        derive_bridged_claim(&wrong, &source),
        "bridge_binding_mismatch",
        "scientific_bridge.transport_premise_mismatch",
    );

    // The declared premise licenses the bridge, labelled transported, with
    // the premise carried as an unresolved obligation.
    let mut licensed = transported.clone();
    licensed.transport =
        Some(premise(scope("adults", "do(treatment=1)"), scope("children", "do(treatment=1)")));
    let bridged = derive_bridged_claim(&licensed, &source).unwrap();
    assert!(bridged.transported);
    assert!(close(bridged.value, 0.796, 1e-12));
    assert!(
        bridged
            .receipt
            .unresolved
            .iter()
            .any(|item| &**item == "transportability:invariant-effect-modification")
    );
    let obligations = &bridged.claim.reasoning.assumptions.as_ref().unwrap().obligations;
    assert!(obligations.iter().any(|o| &*o.id == "bridge_transport:invariant-effect-modification"));
    // The premise is part of the bridge's identity.
    let mut other_statement = licensed.clone();
    other_statement.transport.as_mut().unwrap().statement = "a weaker premise".into();
    assert_ne!(licensed.identity(), other_statement.identity());

    // Where nothing is transported, a premise is refused rather than carried
    // as decoration.
    let mut unused = surrogate_contract(4, 0.9);
    unused.transport =
        Some(premise(scope("adults", "do(treatment=1)"), scope("adults", "do(treatment=1)")));
    let adults = surrogate_quantity("adults");
    let adult_source = BridgeSource { claim: &source_claim, quantity: &adults };
    assert_refused(
        derive_bridged_claim(&unused, &adult_source),
        "bridge_binding_mismatch",
        "scientific_bridge.transport_premise_unused",
    );
}

#[test]
fn malformed_declarations_and_unusable_sources_are_refused() {
    let source_claim = claim(1, 0.8, "mmol_per_l");
    let source_quantity = surrogate_quantity("adults");
    let source = BridgeSource { claim: &source_claim, quantity: &source_quantity };

    let mut duplicated = surrogate_contract(4, 0.9);
    duplicated.assumptions.push(assumption("validation", AssumptionKind::Additional, None));
    assert_refused(
        derive_bridged_claim(&duplicated, &source),
        "scientific_bridge_invalid",
        "scientific_bridge.duplicate_assumption",
    );
    let mut blank = surrogate_contract(4, 0.9);
    blank.assumptions[0].statement = "  ".into();
    assert_refused(
        derive_bridged_claim(&blank, &source),
        "scientific_bridge_invalid",
        "scientific_bridge.blank_field",
    );
    let mut wrong_pair = surrogate_contract(4, 0.9);
    wrong_pair.target.class = QuantityClass::DecisionUtility;
    assert_refused(
        derive_bridged_claim(&wrong_pair, &source),
        "scientific_bridge_invalid",
        "scientific_bridge.endpoint_classes",
    );
    let mut wrong_mapping = surrogate_contract(4, 0.9);
    wrong_mapping.mapping = BridgeMapping::Identity;
    assert_refused(
        derive_bridged_claim(&wrong_mapping, &source),
        "scientific_bridge_invalid",
        "scientific_bridge.mapping_not_allowed",
    );
    let mut missing = surrogate_contract(4, 0.9);
    missing.assumptions.clear();
    assert_refused(
        derive_bridged_claim(&missing, &source),
        "bridge_assumption_missing",
        "scientific_bridge.assumption_missing",
    );

    // A source claim that is not a finite point value.
    let incomplete = ClaimEnvelope::new(
        digest(1),
        source_claim.identities,
        ClaimKind::Incomplete,
        None,
        Some(Arc::from("mmol_per_l")),
        source_claim.reasoning.clone(),
        ClaimDomains::new(DomainStatus::Unknown, DomainStatus::Unknown, DomainStatus::Unknown),
        None,
        [],
    );
    let source_bad = BridgeSource { claim: &incomplete, quantity: &source_quantity };
    assert_refused(
        derive_bridged_claim(&surrogate_contract(4, 0.9), &source_bad),
        "cell_not_licensed",
        "scientific_bridge.source_not_point",
    );
    let not_finite = claim(1, f64::NAN, "mmol_per_l");
    let source_nan = BridgeSource { claim: &not_finite, quantity: &source_quantity };
    assert_refused(
        derive_bridged_claim(&surrogate_contract(4, 0.9), &source_nan),
        "cell_not_licensed",
        "scientific_bridge.source_not_point",
    );
}

#[test]
fn identity_binds_the_contract_the_assumptions_and_the_source_claim() {
    let base = surrogate_contract(4, 0.9);
    let reference = base.identity();

    // Listing order of assumptions and trials is not identity.
    let mut reordered = base.clone();
    reordered.assumptions.push(assumption("second", AssumptionKind::Additional, Some("note")));
    let with_second = reordered.identity();
    reordered.assumptions.reverse();
    assert_eq!(reordered.identity(), with_second);
    assert_ne!(with_second, reference);
    let mut shuffled = base.clone();
    if let BridgeMapping::TrialLevelLinear(validation) = &mut shuffled.mapping {
        validation.trials.reverse();
    }
    assert_eq!(shuffled.identity(), reference);

    // Every semantic edit moves it.
    let edits: Vec<BridgeContract> = {
        let mut list = Vec::new();
        let mut edit = base.clone();
        edit.assumptions[0].statement.push('!');
        list.push(edit);
        let mut edit = base.clone();
        edit.assumptions[0].evidence = Some("meta-analysis:other".into());
        list.push(edit);
        let mut edit = base.clone();
        edit.evidence_scope.population_id = "elderly".into();
        list.push(edit);
        let mut edit = base.clone();
        edit.source.quantity.units = "mg_per_dl".into();
        list.push(edit);
        let mut edit = base.clone();
        edit.target.class = QuantityClass::ScientificEndpoint;
        list.push(edit);
        let mut edit = base.clone();
        if let BridgeMapping::TrialLevelLinear(validation) = &mut edit.mapping {
            validation.min_r_squared = f64::from_bits(0.9_f64.to_bits() + 1);
        }
        list.push(edit);
        list
    };
    for edit in &edits {
        assert_ne!(edit.identity(), reference);
    }

    // The derived claim identity binds the source claim as well.
    let quantity_a = surrogate_quantity("adults");
    let claim_a = claim(1, 0.8, "mmol_per_l");
    let claim_b = claim(2, 0.8, "mmol_per_l");
    let a = derive_bridged_claim(&base, &BridgeSource { claim: &claim_a, quantity: &quantity_a })
        .unwrap();
    let b = derive_bridged_claim(&base, &BridgeSource { claim: &claim_b, quantity: &quantity_a })
        .unwrap();
    assert_eq!(a.contract_identity, b.contract_identity);
    assert_ne!(a.claim.claim_id, b.claim.claim_id);
    assert_eq!(a.value.to_bits(), b.value.to_bits());
    // And the derivation is deterministic.
    let again =
        derive_bridged_claim(&base, &BridgeSource { claim: &claim_a, quantity: &quantity_a })
            .unwrap();
    assert_eq!(a.claim.claim_id, again.claim.claim_id);
    assert_eq!(a.claim, again.claim);
}

#[test]
fn a_bridged_claim_shares_its_sources_evidence_and_is_not_poolable_with_it() {
    let source_claim = claim(1, 0.8, "mmol_per_l");
    let source_quantity = surrogate_quantity("adults");
    let source = BridgeSource { claim: &source_claim, quantity: &source_quantity };
    let bridged = derive_bridged_claim(&surrogate_contract(4, 0.9), &source).unwrap();

    // Same data snapshot: the derived claim is not independent evidence.
    let derived_ref = SharedEvidenceRef::from_claim(&bridged.claim);
    let source_ref = SharedEvidenceRef::from_claim(&source_claim);
    assert!(derived_ref.same_source(&source_ref));
    // Pooling the source with its own bridge is refused as unlicensed
    // synthesis, with both parents intact.
    match compose_claims(&[&source_claim, &bridged.claim], ClaimOperation::Pool, None, None) {
        DerivedClaimOutcome::Refused { restriction, parents, .. } => {
            assert_eq!(&*restriction, "unlicensed_synthesis");
            assert_eq!(parents.as_ref(), [source_claim.claim_id, bridged.claim.claim_id]);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_withdrawn_source_makes_the_bridged_claim_inapplicable() {
    let source_claim = claim(1, 0.8, "mmol_per_l");
    let source_quantity = surrogate_quantity("adults");
    let source = BridgeSource { claim: &source_claim, quantity: &source_quantity };
    let bridged = derive_bridged_claim(&surrogate_contract(4, 0.9), &source).unwrap();
    let snapshot = source_claim.identities.data_snapshot;

    let current = source_claim.freshness(Some(snapshot), Some(snapshot), None, false, true);
    let derived = bridged.freshness(&current);
    assert_eq!(derived.currently_applicable, Some(true));
    assert!(derived.historically_valid);

    // History is the source's: a source that was never valid yields a bridged claim that
    // was never valid.
    let mut invalid_source = current.clone();
    invalid_source.historically_valid = false;
    assert!(!bridged.freshness(&invalid_source).historically_valid);

    let withdrawn = source_claim.freshness(Some(snapshot), Some(snapshot), None, true, true);
    let derived = bridged.freshness(&withdrawn);
    assert_eq!(derived.currently_applicable, Some(false));
    assert!(derived.historically_valid, "history is preserved, only applicability changes");

    let superseded =
        source_claim.freshness(Some(snapshot), Some(snapshot), Some(digest(99)), false, true);
    assert_eq!(bridged.freshness(&superseded).currently_applicable, Some(false));
}

#[test]
fn every_kind_and_class_has_a_stable_name_and_a_principal_assumption() {
    for kind in BridgeKind::ALL {
        assert_eq!(BridgeKind::from_name(kind.as_str()), Some(kind));
        assert!(!kind.endpoint_pairs().is_empty());
        assert!(!kind.allowed_mappings().is_empty());
        assert!(AssumptionKind::ALL.contains(&kind.principal_assumption()));
        for (source, target) in kind.endpoint_pairs() {
            assert_eq!(QuantityClass::from_name(source.as_str()), Some(*source));
            assert_eq!(QuantityClass::from_name(target.as_str()), Some(*target));
        }
    }
    assert_eq!(BridgeKind::ALL.len(), 6, "the six bridges of the contract");
    for kind in AssumptionKind::ALL {
        assert_eq!(AssumptionKind::from_name(kind.as_str()), Some(kind));
    }
    for kind in ValueKind::ALL {
        assert_eq!(ValueKind::from_name(kind.as_str()), Some(kind));
    }
    for mapping in [
        MappingKind::Identity,
        MappingKind::Affine,
        MappingKind::TrialLevelLinear,
        MappingKind::MeasurementCorrection,
    ] {
        assert_eq!(MappingKind::from_name(mapping.as_str()), Some(mapping));
    }
}
