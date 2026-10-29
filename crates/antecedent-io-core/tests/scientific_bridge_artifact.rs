//! Scientific bridge artifact: a fresh consumer replays the derivation from
//! the stored contract and source claim, recomputes the identity, value and fit
//! and refuses semantic mutations.
//!
//! The values are the hand computed ones of the contract tests: the validation
//! trials `(0, 0.1), (0.5, 0.5), (1, 0.9), (1.5, 1.5)` give slope `0.92` and
//! intercept `0.06`, so a surrogate effect `0.8` bridges to `0.796`; sensitivity
//! 0.9 and specificity 0.8 turn a recorded prevalence `0.31` into `11/70`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, BridgedClaim, QuantityClass, TrialEffectPair, TrialLevelValidation,
    ValueKind, derive_bridged_claim,
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
use antecedent_io_core::bridge_artifact::{
    MappingWire, SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE, ScientificBridgeArtifact,
    ScientificBridgeArtifactError, ScientificBridgeMeta,
};
use antecedent_io_core::measurement_model_wire::model_to_wire;
use antecedent_io_core::{ArtifactContainer, CONTENT_CBOR, OutSection, container_kind};

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
        Some(digest(77)),
        [Arc::<str>::from("prior:p0")],
    )
}

fn surrogate_contract() -> BridgeContract {
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
            min_r_squared: 0.9,
        }),
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:ckd-2024".into()),
        }],
        transport: None,

        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn clinic_model(sensitivity: f64) -> MeasurementModel {
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

fn correction_contract(model: MeasurementModel) -> BridgeContract {
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

        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

struct Sealed {
    bridged: BridgedClaim,
    source: ClaimEnvelope,
    contract: BridgeContract,
    meta: ScientificBridgeMeta,
    identity: String,
}

fn seal(contract: BridgeContract, value: f64, units: &str) -> Sealed {
    let source = claim(1, value, units);
    let quantity = contract.source.quantity.clone();
    let bridged =
        derive_bridged_claim(&contract, &BridgeSource { claim: &source, quantity: &quantity })
            .expect("licensed bridge");
    let (artifact, meta) = ScientificBridgeArtifact::seal(
        &bridged,
        &BridgeSource { claim: &source, quantity: &quantity },
    )
    .expect("seal");
    Sealed { bridged, source, contract, meta, identity: artifact.identity().to_owned() }
}

fn reseal(mut meta: ScientificBridgeMeta) -> ScientificBridgeMeta {
    ScientificBridgeArtifact::reseal_identity(&mut meta);
    meta
}

fn surrogate() -> Sealed {
    seal(surrogate_contract(), 0.8, "mmol_per_l")
}

// -- replay --------------------------------------------------------------------------------

#[test]
fn a_fresh_consumer_replays_the_bridge_and_recomputes_its_values() {
    let s = surrogate();
    assert!(close(s.bridged.value, 0.796, 1e-12));

    let bytes = {
        let quantity = s.contract.source.quantity.clone();
        ScientificBridgeArtifact::to_bytes(
            &s.bridged,
            &BridgeSource { claim: &s.source, quantity: &quantity },
            "surrogate-bridge-1",
        )
        .expect("bytes")
    };
    assert_eq!(container_kind(&bytes).as_deref(), Some("scientific_bridge_v1"));

    // A consumer that retained the identity independently of the bytes.
    let consumed =
        ScientificBridgeArtifact::from_bytes(&bytes, Some(&s.identity)).expect("consumed");
    let replayed = consumed.bridged();
    assert_eq!(replayed.value.to_bits(), s.bridged.value.to_bits());
    assert_eq!(replayed.claim.claim_id, s.bridged.claim.claim_id);
    assert_eq!(replayed.contract_identity, s.bridged.contract_identity);
    assert_eq!(replayed.source_claim_id, s.source.claim_id);
    let fit = replayed.fit.expect("fit");
    assert!(close(fit.slope, 0.92, 1e-12) && close(fit.intercept, 0.06, 1e-12));
    assert!(close(fit.r_squared, 529.0 / 535.0, 1e-12));
    assert!(!replayed.transported);
    assert_eq!(consumed.identity(), s.identity);

    // The source claim stays separately inspectable, with the identity layers
    // and value the bridge saw; the derived claim is a different claim.
    let source = consumed.source_claim();
    assert_eq!(source.claim_id, s.source.claim_id);
    assert_eq!(source.value, s.source.value);
    assert_eq!(source.identities.data_snapshot, s.source.identities.data_snapshot);
    assert_eq!(source.execution, s.source.execution);
    assert_ne!(source.claim_id, replayed.claim.claim_id);
}

#[test]
fn a_measurement_correction_is_carried_with_its_model_and_replayed() {
    let s = seal(correction_contract(clinic_model(0.9)), 0.31, "probability");
    assert!(close(s.bridged.value, 11.0 / 70.0, 1e-12));
    let artifact = ScientificBridgeArtifact::from_meta(&s.meta, Some(&s.identity)).expect("replay");
    assert!(close(artifact.bridged().value, 11.0 / 70.0, 1e-12));
    assert!(artifact.bridged().fit.is_none());

    // The model inside the artifact is the model whose identity the
    // assumption names; another model with the same label is another model.
    let mut swapped = s.meta.clone();
    swapped.contract.mapping =
        MappingWire::MeasurementCorrection { model: model_to_wire(&clinic_model(0.95)) };
    let error = ScientificBridgeArtifact::from_meta(&reseal(swapped), None).expect_err("refused");
    let (code, detail, _) = error.refusal().expect("registered refusal");
    assert_eq!(code, "bridge_binding_mismatch");
    assert_eq!(detail, "scientific_bridge.model_identity_mismatch");
}

// -- mutation ------------------------------------------------------------------------------

#[test]
fn a_resealed_edit_of_a_stored_number_does_not_replay() {
    let s = surrogate();

    let mut value = s.meta.clone();
    value.value_bits = f64::from_bits(value.value_bits + 1).to_bits();
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&reseal(value), None).expect_err("value"),
        ScientificBridgeArtifactError::ReplayMismatch("value")
    );

    let mut source_value = s.meta.clone();
    source_value.source.value_bits = 0.9_f64.to_bits();
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&reseal(source_value), None).expect_err("source"),
        ScientificBridgeArtifactError::ReplayMismatch("value")
    );

    let mut source_id = s.meta.clone();
    source_id.source.claim_id = digest(2).to_hex();
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&reseal(source_id), None).expect_err("source id"),
        ScientificBridgeArtifactError::ReplayMismatch("claim identity")
    );

    let mut flag = s.meta.clone();
    flag.transported = true;
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&reseal(flag), None).expect_err("flag"),
        ScientificBridgeArtifactError::ReplayMismatch("transport flag")
    );

    let mut fit = s.meta.clone();
    fit.fit.as_mut().expect("fit").slope_bits = 0.93_f64.to_bits();
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&reseal(fit), None).expect_err("fit"),
        ScientificBridgeArtifactError::ReplayMismatch("trial-level fit")
    );

    let mut contract_identity = s.meta.clone();
    contract_identity.contract_identity = digest(3).to_hex();
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&reseal(contract_identity), None).expect_err("id"),
        ScientificBridgeArtifactError::ReplayMismatch("contract identity")
    );
}

#[test]
fn a_resealed_edit_of_the_declaration_changes_the_derivation_or_is_refused() {
    let s = surrogate();

    // A different validation trial changes the fit, hence the contract identity.
    let mut trial = s.meta.clone();
    if let MappingWire::TrialLevelLinear { trials, .. } = &mut trial.contract.mapping {
        trials[3].terminal_bits = 1.6_f64.to_bits();
    }
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&reseal(trial), None).expect_err("trial"),
        ScientificBridgeArtifactError::ReplayMismatch("contract identity")
    );

    // An assumption statement is part of the contract identity.
    let mut statement = s.meta.clone();
    statement.contract.assumptions[0].statement = "a weaker statement".into();
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&reseal(statement), None).expect_err("statement"),
        ScientificBridgeArtifactError::ReplayMismatch("contract identity")
    );

    // Removing the principal assumption is refused by the library on replay,
    // with the registered code and a scientific_bridge detail.
    let mut dropped = s.meta.clone();
    dropped.contract.assumptions.clear();
    let error = ScientificBridgeArtifact::from_meta(&reseal(dropped), None).expect_err("dropped");
    assert!(matches!(error, ScientificBridgeArtifactError::Refused { .. }), "{error}");
    let (code, detail, _) = error.refusal().expect("registered refusal");
    assert_eq!(code, "bridge_assumption_missing");
    assert_eq!(detail, "scientific_bridge.assumption_missing");

    // Moving the evidence scope away from the claim's population needs a
    // transportability premise that the stored contract does not carry.
    let mut scope = s.meta.clone();
    scope.contract.evidence_scope.population_id = "children".into();
    let error = ScientificBridgeArtifact::from_meta(&reseal(scope), None).expect_err("scope");
    let (code, detail, _) = error.refusal().expect("registered refusal");
    assert_eq!(code, "bridge_assumption_missing");
    assert_eq!(detail, "scientific_bridge.transport_premise_missing");

    // A value kind the bridge does not carry is refused, not replayed.
    let mut law = s.meta.clone();
    law.contract.source_value = "predictive_law".into();
    law.contract.target_value = "predictive_law".into();
    let error = ScientificBridgeArtifact::from_meta(&reseal(law), None).expect_err("law");
    let (code, detail, _) = error.refusal().expect("registered refusal");
    assert_eq!(code, "cell_not_licensed");
    assert_eq!(detail, "scientific_bridge.value_kind_not_carried");
}

#[test]
fn an_unsealed_edit_is_an_identity_mismatch_and_a_retained_identity_binds_the_history() {
    let s = surrogate();

    // Evidence references are stored data a replay does not consult; an edit
    // that is not resealed is still caught by the identity.
    let mut evidence = s.meta.clone();
    evidence.source.evidence.push("prior:another".into());
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&evidence, None).expect_err("unsealed"),
        ScientificBridgeArtifactError::IdentityMismatch { field: "identity" }
    );
    // Resealed, it is a different artifact; a consumer that retained the
    // original identity refuses it.
    let resealed = reseal(evidence);
    let other = ScientificBridgeArtifact::from_meta(&resealed, None).expect("self-consistent");
    assert_ne!(other.identity(), s.identity);
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&resealed, Some(&s.identity)).expect_err("retained"),
        ScientificBridgeArtifactError::IdentityMismatch { field: "retained_identity" }
    );
    // The refusal is registered.
    let (code, detail, _) = ScientificBridgeArtifactError::IdentityMismatch { field: "identity" }
        .refusal()
        .expect("registered");
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "scientific_bridge.identity_mismatch");
}

#[test]
fn versions_markers_layouts_and_corruption_are_refused() {
    let s = surrogate();

    let mut newer = s.meta.clone();
    newer.version = 2;
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&newer, None),
        Err(ScientificBridgeArtifactError::UnsupportedVersion { version: 2 })
    ));
    let mut foreign = s.meta.clone();
    foreign.feature = "something_else".to_owned();
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&foreign, None),
        Err(ScientificBridgeArtifactError::UnsupportedSemantics("feature marker"))
    ));
    assert_eq!(SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE, "scientific_bridge_v1");

    let quantity = s.contract.source.quantity.clone();
    let source = BridgeSource { claim: &s.source, quantity: &quantity };
    let mut bytes = ScientificBridgeArtifact::to_bytes(&s.bridged, &source, "surrogate-bridge-2")
        .expect("bytes");
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    assert!(matches!(
        ScientificBridgeArtifact::from_bytes(&bytes, None),
        Err(ScientificBridgeArtifactError::Container(antecedent_io_core::IoError::ChecksumMismatch { section })) if section == "scientific_bridge_meta"
    ));
    assert!(matches!(
        ScientificBridgeArtifact::from_bytes(&bytes[..bytes.len() / 3], None),
        Err(ScientificBridgeArtifactError::Decode(m)) if m.contains("failed to fill whole buffer")
    ));
    assert!(matches!(
        ScientificBridgeArtifact::to_bytes(&s.bridged, &source, "  "),
        Err(ScientificBridgeArtifactError::Encode(m)) if m == "missing artifact id"
    ));

    // A container of another artifact kind is not a bridge.
    let other_kind = antecedent_io_core::encode_container(
        &ArtifactContainer {
            kind: "measurement_model_v1",
            note: "not_a_bridge",
            max_bytes: 1 << 20,
            artifact_id: "other",
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[OutSection { id: "scientific_bridge_meta", content_type: CONTENT_CBOR, bytes: &[0xa0] }],
    )
    .expect("container");
    assert!(matches!(
        ScientificBridgeArtifact::from_bytes(&other_kind, None),
        Err(ScientificBridgeArtifactError::Malformed(m)) if m == "unsupported container layout"
    ));
}

#[test]
fn a_bridge_that_was_not_derived_from_its_source_does_not_seal() {
    let s = surrogate();
    // The bridged claim names source claim 1; a different source claim, with
    // the same value, is not the claim it was derived from.
    let other = claim(2, 0.8, "mmol_per_l");
    let quantity = s.contract.source.quantity.clone();
    let error = ScientificBridgeArtifact::seal(
        &s.bridged,
        &BridgeSource { claim: &other, quantity: &quantity },
    )
    .expect_err("seal");
    assert_eq!(error, ScientificBridgeArtifactError::ReplayMismatch("claim identity"));
}

#[test]
fn a_stored_value_kind_that_is_not_a_mean_is_refused_on_replay() {
    let s = surrogate();
    for kind in ["predictive_law", "posterior", "identified_set", "joint_outcome_law"] {
        // A law held and required as a law is a route the bridge does not carry.
        let mut same = s.meta.clone();
        same.contract.source_value = kind.into();
        same.contract.target_value = kind.into();
        let error = ScientificBridgeArtifact::from_meta(&reseal(same), None).expect_err("law");
        let (code, detail, _) = error.refusal().expect("registered refusal");
        assert_eq!(code, "cell_not_licensed");
        assert_eq!(detail, "scientific_bridge.value_kind_not_carried");

        // A mean held and a different object required is a substitution.
        let mut required = s.meta.clone();
        required.contract.target_value = kind.into();
        let error = ScientificBridgeArtifact::from_meta(&reseal(required), None).expect_err("req");
        let (code, detail, _) = error.refusal().expect("registered refusal");
        assert_eq!(code, "bridge_substitution_refused");
        assert_eq!(detail, "scientific_bridge.value_kind_substitution");
    }
    // An unknown kind name is unsupported semantics, never a mean.
    let mut unknown = s.meta.clone();
    unknown.contract.target_value = "interval".into();
    assert_eq!(
        ScientificBridgeArtifact::from_meta(&reseal(unknown), None).expect_err("unknown"),
        ScientificBridgeArtifactError::UnsupportedSemantics("value kind")
    );
}

#[test]
fn a_consumed_bridge_has_no_interval_license() {
    let s = surrogate();
    let artifact = ScientificBridgeArtifact::from_meta(&s.meta, Some(&s.identity)).expect("replay");
    let refusal = artifact.bridged().request_interval().unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "scientific_bridge.interval_not_licensed");
    // The replayed claim reports no uncertainty and names what it cannot offer.
    assert!(!artifact.bridged().claim.reasoning.uncertainty.is_available());
    assert!(artifact.bridged().receipt.unavailable_operations.iter().any(|op| &**op == "interval"));
    assert!(
        artifact.bridged().receipt.unavailable_operations.iter().any(|op| &**op == "posterior")
    );
}
