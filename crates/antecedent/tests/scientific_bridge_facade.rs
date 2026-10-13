//! Facade integration of typed scientific bridges: checked preparation,
//! derivation, inspection, export and independent consumption, against
//! hand-computed and enumerated truth.
//!
//! Oracles that are not the code under test:
//!
//! * four validation trials `(0, 0.1), (0.5, 0.5), (1, 0.9), (1.5, 1.5)` have,
//!   by hand, `Sxx = 1.25`, `Sxy = 1.15`, slope `0.92`, intercept `0.06`, so a
//!   surrogate effect `0.8` is a terminal effect `0.796` and `1.0` is `0.98`;
//! * the Rogan-Gladen correction of a recorded prevalence `0.31` under
//!   sensitivity 0.9 and specificity 0.8 is `11/70`, which the
//!   `antecedent-stats` corrected-prevalence routine reproduces from counts
//!   (310 of 1000) without sharing the bridge's code;
//! * an enumerated confounded structural model whose association `0.54`
//!   differs from its interventional effect `0.30`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    PreparedScientificBridge, ScientificBridgeRefusal, consume_scientific_bridge_artifact,
};
use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, MappingKind, QuantityClass, TransportabilityPremise,
    TrialEffectPair, TrialLevelValidation, ValueKind,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, QuantityRole, ReasoningView,
    ScientificQuantity, SlotAvailability,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_core::{ContractIdentities, SemanticDigest};
use antecedent_stats::measurement_error::{BinaryErrorRates, corrected_prevalence};

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn close(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance
}

fn quantity(
    variable: &str,
    units: &str,
    functional: &str,
    population: &str,
    regime: &str,
) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
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
        [Arc::<str>::from("prior:p0")],
    )
}

fn surrogate_contract(min_r_squared: f64) -> BridgeContract {
    let trials = [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)]
        .into_iter()
        .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
            surrogate_effect,
            terminal_effect,
        })
        .collect();
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: quantity(
                "trial:biomarker",
                "mmol_per_l",
                "average_effect",
                "adults",
                "do(t=1)",
            ),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: quantity(
                "trial:survival",
                "log_hazard",
                "average_effect",
                "adults",
                "do(t=1)",
            ),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping: BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials,
            min_trials: 4,
            min_r_squared,
        }),
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:ckd-2024".into()),
        }],
        transport: None,
    }
}

fn clinic_model() -> MeasurementModel {
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
                sensitivity: 0.9,
                specificity: 0.8,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 },
        evidence: ParameterEvidence::Declared { declarer: "lab".into() },
    }
}

fn correction_contract() -> BridgeContract {
    let model = clinic_model();
    BridgeContract {
        kind: BridgeKind::LatentMeasuredOutcome,
        source: BridgeEndpoint {
            quantity: quantity(
                "schema:flag",
                "probability",
                "prevalence",
                "clinic-a",
                "observational",
            ),
            class: QuantityClass::MeasuredOutcome,
        },
        target: BridgeEndpoint {
            quantity: quantity(
                "construct:disease",
                "probability",
                "prevalence",
                "clinic-a",
                "observational",
            ),
            class: QuantityClass::LatentConstruct,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: "clinic-a".into(),
            regime_id: "observational".into(),
        },
        assumptions: vec![BridgeAssumption::for_measurement_model("flag-model", &model)],
        mapping: BridgeMapping::MeasurementCorrection(Box::new(model)),
        transport: None,
    }
}

fn source<'a>(claim: &'a ClaimEnvelope, contract: &'a BridgeContract) -> BridgeSource<'a> {
    BridgeSource { claim, quantity: &contract.source.quantity }
}

// -- prepare, derive, export, consume ---------------------------------------------------------

#[test]
fn a_surrogate_bridge_is_prepared_derived_exported_and_consumed_independently() {
    let contract = surrogate_contract(0.9);
    let prepared = PreparedScientificBridge::prepare(contract.clone()).expect("licensed");
    assert!(!prepared.transported());
    assert_eq!(prepared.contract_identity(), contract.identity());

    let source_claim = claim(1, 0.8, "mmol_per_l");
    let before = source_claim.clone();
    let result = prepared.derive(&source(&source_claim, &contract)).expect("derived");

    // Hand value: 0.06 + 0.92 * 0.8 = 0.796.
    assert!(close(result.value(), 0.796, 1e-12), "{}", result.value());
    assert_eq!(result.source_claim(), &before, "the source claim is not modified");
    assert_ne!(result.claim().claim_id, source_claim.claim_id);
    let inspection = result.inspect();
    assert_eq!(inspection.kind, BridgeKind::SurrogateTerminalOutcome);
    assert_eq!(inspection.mapping, MappingKind::TrialLevelLinear);
    assert_eq!(inspection.value_kind, ValueKind::Mean);
    assert_eq!(inspection.source_claim_id, source_claim.claim_id.to_hex());
    assert_eq!(inspection.claim_id, result.claim().claim_id.to_hex());
    assert!(close(inspection.fit.expect("fit").slope, 0.92, 1e-12));
    assert_eq!(
        inspection.assumptions,
        vec![(
            "validation".to_owned(),
            "surrogate_validation".to_owned(),
            Some("meta-analysis:ckd-2024".to_owned())
        )]
    );
    assert!(inspection.unavailable.iter().any(|op| op == "interval"));
    let text = inspection.to_string();
    assert!(text.contains("surrogate_terminal_outcome") && text.contains("validation"), "{text}");

    // Export, then an independent consumer that retained the identity.
    let export = result.export("surrogate-bridge").expect("export");
    let consumed = consume_scientific_bridge_artifact(&export.bytes, Some(&export.identity))
        .expect("consumed");
    assert_eq!(consumed.bridged.value.to_bits(), result.value().to_bits());
    assert_eq!(consumed.bridged.claim.claim_id, result.claim().claim_id);
    // The artifact stores the source as the bridge saw it: identity, value,
    // units, identity layers and evidence. The program identity, reasoning
    // slots and domain statuses are not carried, and the replayed source
    // never claims identification or support it cannot show.
    let replayed = &consumed.source;
    assert_eq!(replayed.claim_id, before.claim_id);
    assert_eq!(replayed.value, before.value);
    assert_eq!(replayed.outcome_units, before.outcome_units);
    assert_eq!(replayed.evidence, before.evidence);
    assert_eq!(replayed.execution, before.execution);
    assert_eq!(replayed.identities.target, before.identities.target);
    assert_eq!(replayed.identities.identification, before.identities.identification);
    assert_eq!(replayed.identities.data_snapshot, before.identities.data_snapshot);
    assert_eq!(replayed.identities.program, None);
    assert!(!replayed.reasoning.identification.is_available());
    assert_eq!(replayed.domains.identification, DomainStatus::Unknown);
    assert_eq!(replayed.domains.support, DomainStatus::Unknown);
    assert_eq!(replayed.domains.evaluated, before.domains.evaluated);
    assert_eq!(consumed.identity, export.identity);

    // A byte flip is a refusal; another identity is a refusal.
    let mut flipped = export.bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xff;
    let refusal = consume_scientific_bridge_artifact(&flipped, None).expect_err("corrupt");
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, "scientific_bridge.artifact_invalid");
    let refusal =
        consume_scientific_bridge_artifact(&export.bytes, Some(&digest(5).to_hex())).unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "scientific_bridge.identity_mismatch");
}

#[test]
fn the_correction_agrees_with_the_independent_corrected_prevalence_routine() {
    let contract = correction_contract();
    let prepared = PreparedScientificBridge::prepare(contract.clone()).expect("licensed");
    let source_claim = claim(2, 0.31, "probability");
    let result = prepared.derive(&source(&source_claim, &contract)).expect("derived");
    assert!(close(result.value(), 11.0 / 70.0, 1e-12), "{}", result.value());

    // 310 of 1000 recorded positives, through the stats-level routine that
    // shares no code with the bridge's mapping.
    let rates = BinaryErrorRates::new(0.9, 0.8, 0.0, 0.0).expect("rates");
    let corrected = corrected_prevalence(310, 1000, &rates, 0.1).expect("corrected");
    assert!(close(corrected.observed, 0.31, 1e-15));
    assert!(close(result.value(), corrected.corrected, 1e-12));

    // The artifact carries the measurement model and replays it.
    let export = result.export("flag-bridge").expect("export");
    let consumed = consume_scientific_bridge_artifact(&export.bytes, Some(&export.identity))
        .expect("consumed");
    assert!(close(consumed.bridged.value, corrected.corrected, 1e-12));
    assert_eq!(consumed.bridged.mapping, MappingKind::MeasurementCorrection);
    assert!(consumed.bridged.fit.is_none());
}

#[test]
fn the_unlicensed_adjacent_substitutions_are_refused_before_any_number() {
    // A measured prevalence offered as the construct.
    let mut proxy = correction_contract();
    proxy.assumptions.clear();
    let refusal = PreparedScientificBridge::prepare(proxy).unwrap_err();
    assert_eq!(refusal.code, "proxy_offered_as_construct");
    assert_eq!(refusal.detail, "scientific_bridge.proxy_as_construct");
    assert!(refusal.remedy.is_some());
    assert!(refusal.to_string().contains("scientific_bridge.proxy_as_construct"));

    // An association offered as the intervention; the enumerated truth shows
    // the number it would have published is wrong.
    let association = {
        let (mut joint_x, mut joint_xy) = ([0.0_f64; 2], [0.0_f64; 2]);
        for z in 0..2 {
            for x in 0..2 {
                let px1 = 0.2 + 0.6 * z as f64;
                let p_x = if x == 1 { px1 } else { 1.0 - px1 };
                let py1 = 0.1 + 0.3 * x as f64 + 0.4 * z as f64;
                joint_x[x] += 0.5 * p_x;
                joint_xy[x] += 0.5 * p_x * py1;
            }
        }
        joint_xy[1] / joint_x[1] - joint_xy[0] / joint_x[0]
    };
    assert!(close(association, 0.54, 1e-12));
    let predictive = BridgeContract {
        kind: BridgeKind::PredictiveCausalQuantity,
        source: BridgeEndpoint {
            quantity: quantity("cohort:y", "probability", "contrast", "registry", "observational"),
            class: QuantityClass::PredictiveQuantity,
        },
        target: BridgeEndpoint {
            quantity: quantity("cohort:y", "probability", "contrast", "registry", "do(x=1)"),
            class: QuantityClass::CausalQuantity,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: "registry".into(),
            regime_id: "observational".into(),
        },
        mapping: BridgeMapping::Identity,
        assumptions: Vec::new(),
        transport: None,
    };
    let refusal = PreparedScientificBridge::prepare(predictive.clone()).unwrap_err();
    assert_eq!(refusal.code, "bridge_substitution_refused");
    assert_eq!(refusal.detail, "scientific_bridge.association_as_intervention");

    // Declared identification licenses the standardized contrast (0.30), which
    // is the interventional truth of the same enumerated model.
    let mut identified = predictive;
    identified.assumptions.push(BridgeAssumption {
        id: "backdoor".into(),
        kind: AssumptionKind::Identification,
        statement: "Z blocks every back-door path".into(),
        evidence: Some("identification:backdoor-z".into()),
    });
    let prepared = PreparedScientificBridge::prepare(identified.clone()).expect("licensed");
    let adjusted = claim(3, 0.30, "probability");
    let result = prepared.derive(&source(&adjusted, &identified)).expect("derived");
    assert!(close(result.value(), 0.30, 1e-12));
    assert_eq!(result.claim().outcome_units.as_deref(), Some("probability"));
}

fn derived_surrogate() -> (BridgeContract, antecedent::ScientificBridgeResult) {
    let contract = surrogate_contract(0.9);
    let prepared = PreparedScientificBridge::prepare(contract.clone()).expect("licensed");
    let source_claim = claim(1, 0.8, "mmol_per_l");
    let result = prepared.derive(&source(&source_claim, &contract)).expect("derived");
    (contract, result)
}

#[test]
fn a_bridged_mean_satisfies_a_consumer_that_requires_a_mean() {
    let (_, result) = derived_surrogate();
    assert!(result.require_value_kind(ValueKind::Mean).is_ok());
    assert_eq!(result.inspect().value_kind, ValueKind::Mean);
}

#[test]
fn a_bridged_mean_is_not_a_law_a_posterior_an_identified_set_or_a_joint_law() {
    let (contract, result) = derived_surrogate();
    for needed in [
        ValueKind::PredictiveLaw,
        ValueKind::Posterior,
        ValueKind::IdentifiedSet,
        ValueKind::JointOutcomeLaw,
    ] {
        let refusal = result.require_value_kind(needed).unwrap_err();
        assert_eq!(refusal.code, "bridge_substitution_refused");
        assert_eq!(refusal.detail, "scientific_bridge.value_kind_substitution");
        assert!(refusal.message.contains(needed.as_str()), "{}", refusal.message);
    }

    // A contract that requires a law from a mean is refused at preparation,
    // and a law held as a law is a route this bridge does not carry.
    let mut wants_law = contract.clone();
    wants_law.target_value = ValueKind::PredictiveLaw;
    let refusal = PreparedScientificBridge::prepare(wants_law).unwrap_err();
    assert_eq!(refusal.detail, "scientific_bridge.value_kind_substitution");
    let mut holds_law = contract;
    holds_law.source_value = ValueKind::PredictiveLaw;
    holds_law.target_value = ValueKind::PredictiveLaw;
    let refusal = PreparedScientificBridge::prepare(holds_law).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "scientific_bridge.value_kind_not_carried");
}

#[test]
fn a_bridged_claim_reports_a_point_value_and_its_labels() {
    let (_, result) = derived_surrogate();
    let claim = result.claim();
    assert_eq!(claim.kind, ClaimKind::Point);
    assert!(claim.value.is_some_and(|value| close(value, 0.796, 1e-12)));
    // Identification, support and uncertainty are not established for the
    // target quantity by the bridge; the declared assumptions are listed.
    assert!(!claim.reasoning.identification.is_available());
    assert!(!claim.reasoning.uncertainty.is_available());
    assert!(claim.reasoning.assumptions.is_available());
    assert_eq!(&*claim.calibration.status, "unavailable");
}

#[test]
fn an_interval_posterior_or_joint_law_request_on_a_bridged_claim_is_refused() {
    let (_, result) = derived_surrogate();
    let refusal = result.interval().unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "scientific_bridge.interval_not_licensed");
    assert!(refusal.remedy.is_some());
    // The refusal does not change the claim.
    assert!(close(result.value(), 0.796, 1e-12));
}

#[test]
fn a_transported_bridge_names_its_premise_or_is_refused() {
    let mut contract = surrogate_contract(0.9);
    contract.source.quantity.population_id = "children".into();
    contract.target.quantity.population_id = "children".into();

    let refusal = PreparedScientificBridge::prepare(contract.clone()).unwrap_err();
    assert_eq!(refusal.code, "bridge_assumption_missing");
    assert_eq!(refusal.detail, "scientific_bridge.transport_premise_missing");
    assert!(refusal.message.contains("adults") && refusal.message.contains("children"));

    contract.transport = Some(TransportabilityPremise {
        id: "same-surrogate-relationship".into(),
        statement: "the trial-level relationship is the same in children".into(),
        established_in: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        applies_to: BridgeScope { population_id: "children".into(), regime_id: "do(t=1)".into() },
        evidence: Some("selection-diagram:pediatric".into()),
    });
    let prepared = PreparedScientificBridge::prepare(contract.clone()).expect("licensed");
    assert!(prepared.transported());
    let source_claim = claim(1, 0.8, "mmol_per_l");
    let result = prepared.derive(&source(&source_claim, &contract)).expect("derived");
    assert!(close(result.value(), 0.796, 1e-12));
    let inspection = result.inspect();
    assert!(inspection.transported);
    assert_eq!(inspection.transportability_premise.as_deref(), Some("same-surrogate-relationship"));
    // The premise travels with the artifact.
    let export = result.export("transported-bridge").expect("export");
    let consumed = consume_scientific_bridge_artifact(&export.bytes, Some(&export.identity))
        .expect("consumed");
    assert!(consumed.bridged.transported);
    assert!(consumed.bridged.contract.transport.is_some());
}

// -- lifecycle ----------------------------------------------------------------------------------

#[test]
fn refresh_keeps_the_contract_and_recomputes_and_a_refusal_changes_nothing() {
    let contract = surrogate_contract(0.9);
    let prepared = PreparedScientificBridge::prepare(contract.clone()).expect("licensed");
    let first_claim = claim(1, 0.8, "mmol_per_l");
    let first = prepared.derive(&source(&first_claim, &contract)).expect("first");

    // A refreshed source claim about the same quantity: same contract and
    // verdict, a recomputed value and claim identity.
    let second_claim = claim(2, 1.0, "mmol_per_l");
    let second = prepared.derive(&source(&second_claim, &contract)).expect("second");
    assert_eq!(first.bridged().contract_identity, second.bridged().contract_identity);
    assert!(close(second.value(), 0.98, 1e-12), "{}", second.value());
    assert_ne!(first.claim().claim_id, second.claim().claim_id);

    // A claim outside the validated range, and one in other units, are
    // refused; the prepared bridge and the first result are untouched.
    let beyond = claim(3, 2.0, "mmol_per_l");
    let refusal = prepared.derive(&source(&beyond, &contract)).unwrap_err();
    assert_eq!(refusal.code, "bridge_mapping_not_licensed");
    assert_eq!(refusal.detail, "scientific_bridge.outside_validated_range");
    let other_units = claim(4, 0.8, "mg_per_dl");
    assert_eq!(
        prepared.derive(&source(&other_units, &contract)).unwrap_err().detail,
        "scientific_bridge.units_mismatch"
    );
    assert_eq!(prepared.contract_identity(), contract.identity());
    let again = prepared.derive(&source(&first_claim, &contract)).expect("again");
    assert_eq!(again.claim().claim_id, first.claim().claim_id);

    // Replacing the evidence requirement re-runs every preparation check: a
    // minimum the validation trials do not meet is refused at preparation.
    let stricter = surrogate_contract(0.995);
    let refusal: ScientificBridgeRefusal = prepared.with_contract(stricter).unwrap_err();
    assert_eq!(refusal.detail, "scientific_bridge.validation_r_squared_too_low");
    // The prepared bridge is unchanged by the refused replacement.
    assert!(prepared.derive(&source(&first_claim, &contract)).is_ok());
}

#[test]
fn refusals_convert_to_the_library_error_types() {
    let mut contract = surrogate_contract(0.9);
    contract.assumptions.clear();
    let refusal = PreparedScientificBridge::prepare(contract).unwrap_err();
    let io: antecedent_io::IoError = refusal.clone().into();
    match io {
        antecedent_io::IoError::Refused { code, message } => {
            assert_eq!(code, "bridge_assumption_missing");
            assert!(message.starts_with("scientific_bridge.assumption_missing"), "{message}");
        }
        other => panic!("unexpected {other:?}"),
    }
    let error: antecedent::CausalError = refusal.into();
    assert!(error.to_string().contains("scientific_bridge.assumption_missing"), "{error}");
}
