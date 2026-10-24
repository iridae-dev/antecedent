//! The extended bridge artifact: the mapping uncertainty, the source dependence
//! and the estimand bridge a contract declares, and the draws a bridge of a
//! distribution maps, are carried in format version 2 and replayed by an
//! independent consumer; a version-1 artifact keeps its bytes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::bridge::{
    BridgeSource, MappingUncertainty, NonlinearMap, SourceMappingDependence, ValueKind,
    derive_bridged_claim, derive_bridged_claim_propagating,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, QuantityRole, ReasoningView,
    ScientificQuantity, SlotAvailability,
};
use antecedent_core::{ContractIdentities, SemanticDigest};
use antecedent_stats::normal_ppf;

use super::*;
use crate::distribution_artifact::{
    DistributionCalibration, DistributionMetadata, DistributionProvenance, DistributionTrust,
    DrawAlignment,
};
use crate::quantity_wire::DistributionMeaningWire;

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn quantity(variable: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "adults".into(),
        regime_id: "do(t=1)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn setting(dose: f64, start: &str, end: &str) -> ExposureSetting {
    ExposureSetting {
        population_id: "adults".into(),
        time_start: start.into(),
        time_end: end.into(),
        dose,
        dose_units: "usd_per_capita".into(),
        duration: 12.0,
        duration_units: "week".into(),
        exposure_regime: "do(t=1)".into(),
    }
}

fn contract(value: ValueKind, extended: bool) -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: quantity("trial:biomarker", "mmol_per_l"),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: quantity("trial:survival", "log_hazard"),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: value,
        target_value: value,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping: BridgeMapping::Affine { intercept: 0.06, slope: 0.92 },
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:ckd-2024".into()),
        }],
        transport: None,
        mapping_uncertainty: extended.then(|| MappingUncertainty {
            id: "calibration-fit".into(),
            intercept_variance: 0.0004,
            slope_variance: 0.0025,
            intercept_slope_covariance: -0.0002,
        }),
        source_dependence: extended.then(|| SourceMappingDependence::SlopeCovariance {
            covariance: 0.001,
            evidence: "shared-sample".into(),
        }),
        estimand_bridge: extended.then(|| EstimandBridge {
            id: "geo-to-panel".into(),
            experiment_id: "geo-2024".into(),
            experiment: setting(10.0, "2024-01", "2024-04"),
            target: setting(25.0, "2024-06", "2024-09"),
            recency_delta: 8.0,
            recency_units: "week".into(),
            spend_delta: 15.0,
            spend_units: "usd_per_capita".into(),
        }),
    }
}

fn source_claim() -> ClaimEnvelope {
    ClaimEnvelope::new(
        digest(1),
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
        Some(0.8),
        Some(Arc::from("mmol_per_l")),
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

fn sealed(
    contract: &BridgeContract,
) -> (ScientificBridgeArtifact, ScientificBridgeMeta, ClaimEnvelope) {
    let claim = source_claim();
    let quantity = contract.source.quantity.clone();
    let source = BridgeSource { claim: &claim, quantity: &quantity };
    let bridged = derive_bridged_claim(contract, &source).expect("licensed bridge");
    let (artifact, meta) = ScientificBridgeArtifact::seal(&bridged, &source).expect("seal");
    (artifact, meta, claim)
}

fn reseal(mut meta: ScientificBridgeMeta) -> ScientificBridgeMeta {
    ScientificBridgeArtifact::reseal_identity(&mut meta);
    meta
}

#[test]
fn the_declared_mapping_uncertainty_dependence_and_estimand_bridge_are_carried_and_replayed() {
    let declared = contract(ValueKind::Mean, true);
    let (artifact, meta, claim) = sealed(&declared);
    assert_eq!(meta.version, SCIENTIFIC_BRIDGE_EXTENDED_ARTIFACT_VERSION);
    assert_eq!(meta.feature, SCIENTIFIC_BRIDGE_EXTENDED_FEATURE);
    assert!(meta.contract.is_extended());

    // A fresh consumer rebuilds every declaration and the contract identity binds them.
    let quantity = declared.source.quantity.clone();
    let source = BridgeSource { claim: &claim, quantity: &quantity };
    let bridged = derive_bridged_claim(&declared, &source).expect("bridge");
    let bytes = ScientificBridgeArtifact::to_bytes(&bridged, &source, "uncertain-bridge").unwrap();
    let consumed = ScientificBridgeArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    let replayed = &consumed.bridged().contract;
    assert_eq!(replayed.mapping_uncertainty, declared.mapping_uncertainty);
    assert_eq!(replayed.source_dependence, declared.source_dependence);
    assert_eq!(replayed.estimand_bridge, declared.estimand_bridge);
    assert_eq!(consumed.bridged().contract_identity, declared.identity());
    // The value of a point bridge is the plug-in map: the declarations are carried, not applied.
    assert!((consumed.bridged().value - (0.06 + 0.92 * 0.8)).abs() < 1e-12);

    // A resealed edit of any declared number is a different contract and does not replay.
    let mut slope = meta.clone();
    slope.contract.mapping_uncertainty.as_mut().unwrap().slope ^= 1;
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&reseal(slope), None),
        Err(ScientificBridgeArtifactError::ReplayMismatch("contract identity"))
    ));
    let mut dependence = meta.clone();
    dependence.contract.source_dependence =
        Some(DependenceWire::Independent { evidence: "shared-sample".into() });
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&reseal(dependence), None),
        Err(ScientificBridgeArtifactError::ReplayMismatch("contract identity"))
    ));
    let mut dropped = meta.clone();
    dropped.contract.estimand_bridge = None;
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&reseal(dropped), None),
        Err(ScientificBridgeArtifactError::ReplayMismatch("contract identity"))
    ));
    // An estimand bridge that no longer names the target is refused by the library.
    let mut moved = meta.clone();
    moved.contract.estimand_bridge.as_mut().unwrap().target.population = "children".into();
    let error = ScientificBridgeArtifact::from_meta(&reseal(moved), None).expect_err("target");
    assert_eq!(
        error.refusal().map(|(_, detail, _)| detail).as_deref(),
        Some("scientific_bridge.estimand_bridge_mismatch")
    );
}

#[test]
fn a_version_one_artifact_declares_nothing_extended_and_keeps_its_bytes() {
    let plain = contract(ValueKind::Mean, false);
    let (_, meta, _) = sealed(&plain);
    assert_eq!(meta.version, SCIENTIFIC_BRIDGE_ARTIFACT_VERSION);
    assert_eq!(meta.feature, SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE);
    assert!(!meta.contract.is_extended());
    assert!(meta.draws.is_none());
    // Nothing of the extension is written, so the contract's bytes are those of version 1.
    let bytes = to_cbor(&meta.contract).unwrap();
    for key in ["mapping_uncertainty", "source_dependence", "estimand_bridge"] {
        assert!(!bytes.windows(key.len()).any(|window| window == key.as_bytes()), "{key}");
    }
    let bytes = to_cbor(&meta).unwrap();
    assert!(!bytes.windows(5).any(|window| window == b"draws"));

    // An extended declaration under the version-1 marker is not version 1.
    let (_, extended, _) = sealed(&contract(ValueKind::Mean, true));
    let mut downgraded = extended.clone();
    downgraded.version = SCIENTIFIC_BRIDGE_ARTIFACT_VERSION;
    downgraded.feature = SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE.to_owned();
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&reseal(downgraded), None),
        Err(ScientificBridgeArtifactError::UnsupportedSemantics(
            "extended declarations in a version-1 artifact"
        ))
    ));
    // An extended marker with nothing extended is not the canonical encoding of a plain bridge.
    let mut upgraded = meta.clone();
    upgraded.version = SCIENTIFIC_BRIDGE_EXTENDED_ARTIFACT_VERSION;
    upgraded.feature = SCIENTIFIC_BRIDGE_EXTENDED_FEATURE.to_owned();
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&reseal(upgraded), None),
        Err(ScientificBridgeArtifactError::UnsupportedSemantics(
            "extended format without extended declarations"
        ))
    ));
    // The joint-panel request wrapper writes the contract without the extension fields.
    let full = FullBridgeContractWire::from_domain(&contract(ValueKind::Mean, true)).unwrap();
    let bytes = to_cbor(&full).unwrap();
    assert!(!bytes.windows(19).any(|window| window == b"mapping_uncertainty"));
    assert_eq!(full.to_domain().unwrap(), contract(ValueKind::Mean, true));
}

// -- a point source with a propagated variance --------------------------------------------------

fn sealed_propagated(
    contract: &BridgeContract,
    standard_error: f64,
    nonlinear: Option<NonlinearMap>,
) -> (ScientificBridgeArtifact, ScientificBridgeMeta, BridgedClaim, Vec<u8>) {
    let claim = source_claim();
    let quantity = contract.source.quantity.clone();
    let source = BridgeSource { claim: &claim, quantity: &quantity };
    let bridged =
        derive_bridged_claim_propagating(contract, &source, Some(standard_error), nonlinear)
            .expect("propagated bridge");
    let (artifact, meta) = ScientificBridgeArtifact::seal(&bridged, &source).expect("seal");
    let bytes = ScientificBridgeArtifact::to_bytes(&bridged, &source, "propagated").unwrap();
    (artifact, meta, bridged, bytes)
}

#[test]
fn a_propagated_variance_is_stored_and_replayed_bit_for_bit_by_a_fresh_consumer() {
    let declared = contract(ValueKind::Mean, true);
    let (artifact, meta, bridged, bytes) =
        sealed_propagated(&declared, 0.1, Some(NonlinearMap::Square));
    assert_eq!(meta.version, SCIENTIFIC_BRIDGE_EXTENDED_ARTIFACT_VERSION);
    let wire = meta.propagation.as_ref().expect("the propagation is stored");
    assert_eq!(wire.source_standard_error_bits, 0.1_f64.to_bits());
    assert_eq!(wire.nonlinear.as_deref(), Some("square"));

    // The delta method by hand: a = 0.06, b = 0.92, x = 0.8, se = 0.1, c = 0.001.
    let (x, b, v, c) = (0.8_f64, 0.92_f64, 0.01_f64, 0.001_f64);
    let by_hand = 0.0004
        + x * x * 0.0025
        + b * b * v
        + 2.0 * x * b * c
        + 2.0 * x * -0.0002
        + 0.0025 * v
        + c * c;
    let propagation = bridged.propagation.as_ref().unwrap();
    assert!((propagation.variance - by_hand).abs() <= 1e-12 * by_hand, "{}", propagation.variance);
    assert!((propagation.mean - (0.06 + b * x + c)).abs() < 1e-12);

    // A fresh consumer derives the claim again from the stored inputs and gets the same bits.
    let consumed = ScientificBridgeArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    let replayed = consumed.bridged().propagation.as_ref().expect("replayed");
    assert_eq!(replayed.variance.to_bits(), propagation.variance.to_bits());
    assert_eq!(replayed.mean.to_bits(), propagation.mean.to_bits());
    assert_eq!(replayed.nonlinear, propagation.nonlinear);
    assert_eq!(consumed.bridged().claim.claim_id, bridged.claim.claim_id);
    assert_eq!(consumed.bridged().value.to_bits(), bridged.value.to_bits());
    let square = replayed.nonlinear.expect("square");
    let closed_form = replayed.mean * replayed.mean + replayed.variance;
    assert_eq!(square.expectation.to_bits(), closed_form.to_bits());
}

#[test]
fn a_resealed_edit_of_the_propagation_does_not_replay() {
    let declared = contract(ValueKind::Mean, true);
    let (_, meta, bridged, _) = sealed_propagated(&declared, 0.1, None);

    // The variance bits are an output: a resealed edit is a stored value that does not replay.
    let mut variance = meta.clone();
    variance.propagation.as_mut().unwrap().variance_bits ^= 1;
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&reseal(variance), None),
        Err(ScientificBridgeArtifactError::ReplayMismatch("propagated uncertainty"))
    ));
    // The standard error is an input and binds the claim identity.
    let mut input = meta.clone();
    input.propagation.as_mut().unwrap().source_standard_error_bits = 0.2_f64.to_bits();
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&reseal(input), None),
        Err(ScientificBridgeArtifactError::ReplayMismatch("claim identity"))
    ));
    // Dropping the propagation is another claim, not a plain bridge.
    let mut dropped = meta.clone();
    dropped.propagation = None;
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&reseal(dropped), None),
        Err(ScientificBridgeArtifactError::ReplayMismatch("claim identity"))
    ));
    // A nonlinear map no consumer propagates is malformed, not a silent plug-in.
    let mut unknown = meta.clone();
    unknown.propagation.as_mut().unwrap().nonlinear = Some("cube".into());
    assert!(ScientificBridgeArtifact::from_meta(&reseal(unknown), None).is_err());
    // A propagation under the version-1 marker is another version.
    let mut downgraded = meta.clone();
    downgraded.version = SCIENTIFIC_BRIDGE_ARTIFACT_VERSION;
    downgraded.feature = SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE.to_owned();
    assert!(ScientificBridgeArtifact::from_meta(&reseal(downgraded), None).is_err());
    // The plug-in derivation of the same source carries no propagation and another identity.
    let (_, plain, _) = sealed(&declared);
    assert!(plain.propagation.is_none());
    assert_ne!(plain.claim_id, meta.claim_id);
    assert_eq!(plain.value_bits, bridged.value.to_bits());
}

#[test]
fn a_bridge_without_a_propagation_keeps_its_bytes() {
    let (_, meta, _) = sealed(&contract(ValueKind::Mean, false));
    assert!(meta.propagation.is_none());
    let bytes = to_cbor(&meta).unwrap();
    assert!(!bytes.windows(11).any(|window| window == b"propagation"));
}

// -- bridges of draws ------------------------------------------------------------------------------

fn draw_table() -> DistributionArtifact {
    let rows = 800;
    let source: Vec<f64> =
        (0..rows).map(|i| 0.8 + 0.1 * normal_ppf((i as f64 + 0.5) / rows as f64)).collect();
    let sibling: Vec<f64> = (0..rows)
        .map(|i| {
            0.2 + 0.3 * normal_ppf(((i * 487) % rows) as f64 / rows as f64 + 0.5 / rows as f64)
        })
        .collect();
    let quantities = [quantity("trial:biomarker", "mmol_per_l"), quantity("trial:cost", "usd")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::CausalFunctionalPosterior,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "study-1".into(),
            provider_id: "posterior.provider".into(),
            rng_id: "seed:11".into(),
            snapshot_id: "snapshot-1".into(),
            causal_contract_id: "contract-1".into(),
        },
    )
    .expect("identity");
    let mut draws = Vec::with_capacity(rows * 2);
    for (a, b) in source.iter().zip(&sibling) {
        draws.push(*a);
        draws.push(*b);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows, 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .expect("table")
}

fn bridged_draws(seed: u64) -> BridgedDistribution {
    bridge_draws(&contract(ValueKind::Posterior, true), &draw_table(), None, seed).expect("bridge")
}

fn reseal_draws(mut meta: ScientificBridgeMeta) -> ScientificBridgeMeta {
    ScientificBridgeDrawsArtifact::reseal_identity(&mut meta);
    meta
}

#[test]
fn a_bridge_of_draws_is_replayed_bit_for_bit_from_its_stored_table() {
    let bridged = bridged_draws(5);
    let (artifact, meta, source_bytes) =
        ScientificBridgeDrawsArtifact::seal(&bridged).expect("seal");
    assert_eq!(meta.version, SCIENTIFIC_BRIDGE_EXTENDED_ARTIFACT_VERSION);
    let bytes = ScientificBridgeDrawsArtifact::to_bytes(&bridged, "draws-bridge").expect("bytes");
    assert_eq!(ScientificBridgeAny::shape(&bytes).unwrap(), "draws");
    let consumed = ScientificBridgeDrawsArtifact::from_bytes(&bytes, Some(artifact.identity()))
        .expect("consume");
    let replayed = consumed.bridged();
    assert_eq!(replayed.claim().draws_digest, bridged.claim().draws_digest);
    assert_eq!(replayed.claim().claim.claim_id, bridged.claim().claim.claim_id);
    assert_eq!(replayed.claim().stream, 5);
    for (a, b) in replayed.mapped_column().iter().zip(bridged.mapped_column()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(matches!(
        ScientificBridgeAny::from_bytes(&bytes, None).unwrap(),
        ScientificBridgeAny::Draws(_)
    ));
    // The reader of a bridged mean does not read a bridge of draws.
    assert!(ScientificBridgeArtifact::from_bytes(&bytes, None).is_err());
    // A consumer that retained another identity refuses it.
    assert!(matches!(
        ScientificBridgeDrawsArtifact::from_bytes(&bytes, Some(&"0".repeat(64))),
        Err(ScientificBridgeArtifactError::IdentityMismatch { field: "retained_identity" })
    ));

    // A resealed edit of the seed is another realization of the mapping.
    let mut seed = meta.clone();
    seed.draws.as_mut().unwrap().seed = 6;
    assert!(matches!(
        ScientificBridgeDrawsArtifact::from_meta(&reseal_draws(seed), &source_bytes, None),
        Err(ScientificBridgeArtifactError::ReplayMismatch("claim identity"))
    ));
    // Resealed edits of the stored summary do not replay.
    let mut mean = meta.clone();
    mean.draws.as_mut().unwrap().mean_bits ^= 1;
    assert!(matches!(
        ScientificBridgeDrawsArtifact::from_meta(&reseal_draws(mean), &source_bytes, None),
        Err(ScientificBridgeArtifactError::ReplayMismatch("value"))
    ));
    let mut mapped = meta.clone();
    mapped.draws.as_mut().unwrap().draws_digest = "0".repeat(64);
    assert!(matches!(
        ScientificBridgeDrawsArtifact::from_meta(&reseal_draws(mapped), &source_bytes, None),
        Err(ScientificBridgeArtifactError::ReplayMismatch("mapped draws"))
    ));
    // A source table that is not the stored one fails its digest before it is mapped.
    let mut swapped = source_bytes.clone();
    let last = swapped.len() - 1;
    swapped[last] ^= 0xff;
    assert!(matches!(
        ScientificBridgeDrawsArtifact::from_meta(&meta, &swapped, None),
        Err(ScientificBridgeArtifactError::ReplayMismatch("source draws"))
    ));
    // A foreign table with its own digest and identity maps to another claim.
    let foreign = {
        let table = draw_table();
        let mut metadata = table.metadata().clone();
        metadata.identity.snapshot_id = "snapshot-2".into();
        DistributionArtifact::new(metadata, table.draws().to_vec()).unwrap()
    };
    let foreign_bytes = foreign.to_bytes(SOURCE_DRAWS_ARTIFACT_ID).unwrap();
    let mut other = meta.clone();
    {
        let draws = other.draws.as_mut().unwrap();
        draws.source_identity = foreign.metadata().identity.clone();
        draws.source_digest = CanonicalWriter::digest_bytes_hex(TAG_BODY, &foreign_bytes);
    }
    assert!(
        ScientificBridgeDrawsArtifact::from_meta(&reseal_draws(other), &foreign_bytes, None)
            .is_err()
    );
}
