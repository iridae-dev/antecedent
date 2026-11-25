//! Bind a composition-bundle reference to a producer artifact's existing identity.
//!
//! The digest on the reference is that identity. Acceptance compares it with the
//! identity the producer already seals. This module does not replay the artifact.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io_core::catalog_role::{
    self, COMPOSED_MODEL, DISCOVERY, LEDGER, MEASUREMENT, PROPAGATION, RELIABILITY,
};
use antecedent_io_core::convert::from_cbor;
use antecedent_io_core::evidence_ledger_artifact::EvidenceLedgerArtifact;
use antecedent_io_core::joint_law_propagation_artifact::{
    JointLawPropagationArtifactWire, joint_law_propagation_identity,
};
use antecedent_io_core::measurement_model_wire::MeasurementModelArtifact;
use antecedent_io_core::mechanism_composition_artifact::{
    MechanismCompositionArtifactWire, mechanism_composition_identity,
};
use antecedent_io_core::selective_discovery_artifact::SelectiveDiscoveryArtifact;
use antecedent_io_core::{
    EncodedArtifact, PROPAGATION_CHAIN_SECTION, PropagationChainRecordWire,
    propagation_chain_identity,
};

use super::{BundleError, BundleNode, BundleStage, NodeKind, ProviderOrData};

/// Fact that names the catalog role of a producer reference.
pub const CATALOG_ROLE_FACT: &str = "catalog_role";

/// What comparing a reference digest with a producer artifact found.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProducerCheck {
    /// The producer identity is the reference digest.
    Match,
    /// No bytes were supplied for the role.
    Missing,
    /// The bytes are not that producer, or their identity is different.
    Refused {
        /// Failed stage.
        stage: BundleStage,
        /// What was refused.
        reason: String,
    },
}

/// Node kind already used for this producer. The kind table is unchanged.
#[must_use]
pub fn node_kind_for_role(role: &str) -> Option<NodeKind> {
    Some(match role {
        PROPAGATION => NodeKind::PropagationChain,
        LEDGER => NodeKind::EvidenceRelationship,
        COMPOSED_MODEL => NodeKind::ExecutionOrFit,
        MEASUREMENT => NodeKind::SupportTrustCalibration,
        DISCOVERY => NodeKind::CausalContract,
        RELIABILITY => NodeKind::SequentialReliabilityDiagnostic,
        _ => return None,
    })
}

/// Identity the producer already seals for `role`.
///
/// # Errors
///
/// `role` is not a producer role, or the bytes fail the producer's identity checks.
pub fn producer_identity(role: &str, bytes: &[u8]) -> Result<String, String> {
    if !catalog_role::is_producer(role) {
        return Err(format!("unknown catalog role `{role}`"));
    }
    match role {
        PROPAGATION => propagation_identity(bytes),
        LEDGER => ledger_identity(bytes),
        COMPOSED_MODEL => composed_identity(bytes),
        MEASUREMENT => measurement_identity(bytes),
        DISCOVERY => discovery_identity(bytes),
        RELIABILITY => reliability_identity(bytes),
        _ => Err(format!("unknown catalog role `{role}`")),
    }
}

/// Compare `expected` with the identity of `bytes`.
#[must_use]
pub fn check_producer(role: &str, expected: &str, bytes: Option<&[u8]>) -> ProducerCheck {
    let Some(bytes) = bytes else {
        return ProducerCheck::Missing;
    };
    match producer_identity(role, bytes) {
        Ok(identity) if identity == expected => ProducerCheck::Match,
        Ok(_) => ProducerCheck::Refused {
            stage: BundleStage::TamperedQuantity,
            reason: format!("producer identity for `{role}` does not match the reference"),
        },
        Err(reason) => ProducerCheck::Refused { stage: BundleStage::SwappedEvidence, reason },
    }
}

/// A reference node whose digest is the producer identity of `bytes`.
///
/// # Errors
///
/// `role` is not a producer role, or `bytes` do not carry that identity.
pub fn producer_reference(id: &str, role: &str, bytes: &[u8]) -> Result<BundleNode, BundleError> {
    let identity = producer_identity(role, bytes).map_err(|reason| BundleError::Refused {
        stage: BundleStage::SwappedEvidence,
        reason,
        offending: None,
    })?;
    let kind = node_kind_for_role(role).ok_or_else(|| BundleError::Refused {
        stage: BundleStage::SwappedEvidence,
        reason: format!("unknown catalog role `{role}`"),
        offending: None,
    })?;
    Ok(BundleNode::reference(
        id,
        kind,
        &identity,
        ProviderOrData::Data { snapshot_id: role.to_owned(), digest: identity.clone() },
    )
    .with_fact(CATALOG_ROLE_FACT, role))
}

fn propagation_identity(bytes: &[u8]) -> Result<String, String> {
    if let Ok(wire) = JointLawPropagationArtifactWire::decode(bytes) {
        let identity = joint_law_propagation_identity(
            &wire.rule,
            &wire.inputs,
            &wire.request,
            &wire.route,
            &wire.label,
            wire.alignment_id.as_deref(),
            wire.covariance_id.as_deref(),
            &wire.shared_evidence,
        )
        .map_err(|error| error.to_string())?;
        if identity != wire.identity {
            return Err("joint-law identity does not match the stored digest".into());
        }
        return Ok(identity);
    }
    let record = chain_record(bytes)?;
    propagation_chain_identity(&record).map_err(|error| error.to_string())
}

fn chain_record(bytes: &[u8]) -> Result<PropagationChainRecordWire, String> {
    if let Ok(record) = from_cbor::<PropagationChainRecordWire>(bytes) {
        return Ok(record);
    }
    let artifact = EncodedArtifact::read_from(std::io::Cursor::new(bytes))
        .map_err(|_| "not a propagation artifact".to_owned())?;
    let section = artifact
        .sections
        .iter()
        .find(|item| item.id == PROPAGATION_CHAIN_SECTION)
        .ok_or_else(|| "not a propagation artifact".to_owned())?;
    from_cbor(section.data.as_ref()).map_err(|error| error.to_string())
}

fn ledger_identity(bytes: &[u8]) -> Result<String, String> {
    EvidenceLedgerArtifact::from_bytes(bytes, None)
        .map(|artifact| artifact.identity().to_owned())
        .map_err(|error| error.to_string())
}

fn composed_identity(bytes: &[u8]) -> Result<String, String> {
    let wire =
        MechanismCompositionArtifactWire::decode(bytes).map_err(|error| error.to_string())?;
    let identity = mechanism_composition_identity(&wire.premises, &wire.query)
        .map_err(|error| error.to_string())?;
    if identity != wire.premises_digest {
        return Err("composed-model identity does not match the stored digest".into());
    }
    Ok(identity)
}

fn measurement_identity(bytes: &[u8]) -> Result<String, String> {
    MeasurementModelArtifact::validate_model_identity(bytes).map_err(|error| error.to_string())
}

fn discovery_identity(bytes: &[u8]) -> Result<String, String> {
    SelectiveDiscoveryArtifact::from_bytes(bytes, None)
        .map(|artifact| artifact.identity().to_owned())
        .map_err(|error| error.to_string())
}

fn reliability_identity(bytes: &[u8]) -> Result<String, String> {
    let artifact = crate::decision_reliability::DecisionReliabilityArtifact::from_bytes(bytes)
        .map_err(|error| error.to_string())?;
    let digest = artifact.recomputed_digest().map_err(|error| error.to_string())?;
    if digest != artifact.digest {
        return Err("reliability digest does not match the body".into());
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::{
        CATALOG_ROLE_FACT, ProducerCheck, check_producer, producer_identity, producer_reference,
    };
    use crate::composition_bundle::{
        BundleConsumer, BundleLimits, BundleNode, BundleStage, CompositionBundle, NodeKind,
        NodeStatus, ProviderOrData, SuppliedSources,
    };
    use antecedent_contract::evidence_ledger::EvidenceLedger;
    use antecedent_contract::{
        InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
        PropagationRequest, RequestedReading,
    };
    use antecedent_core::ExecutionContext;
    use antecedent_core::measurement_model::{
        ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
        MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
        ParameterEvidence, RepresentationVersion,
    };
    use antecedent_io_core::artifact_codec::{ArtifactContainer, encode_sectioned_io};
    use antecedent_io_core::catalog_role::{
        COMPOSED_MODEL, DISCOVERY, LEDGER, MEASUREMENT, PROPAGATION, RELIABILITY,
    };
    use antecedent_io_core::evidence_ledger_artifact::EvidenceLedgerArtifact;
    use antecedent_io_core::joint_law_propagation_artifact::{
        JointLawPropagationArtifactWire, PropagationRequestWire, float_bits,
    };
    use antecedent_io_core::measurement_model_wire::MeasurementModelArtifact;
    use antecedent_io_core::selective_discovery_artifact::SelectiveDiscoveryArtifact;
    use serde::Serialize;

    fn contrast_bytes(first_mean: f64) -> (Vec<u8>, String) {
        let wire = JointLawPropagationArtifactWire::seal(
            &PropagationRequest {
                operation: PropagationOperation::Contrast,
                inputs: vec![
                    PropagationInput {
                        id: "effect_a".into(),
                        evidence_id: "snapshot_1".into(),
                        role: InputRole::Estimate,
                        alignment: InputAlignment::MomentsOnly,
                    },
                    PropagationInput {
                        id: "effect_b".into(),
                        evidence_id: "snapshot_1".into(),
                        role: InputRole::Estimate,
                        alignment: InputAlignment::MomentsOnly,
                    },
                ],
                dependence: PropagationDependence::Covariance { id: "cov_ab".into() },
                reading: RequestedReading::Point,
            },
            PropagationRequestWire::LinearCombination {
                coefficients: float_bits(&[1.0, -1.0]),
                means: float_bits(&[first_mean, 2.0]),
                covariance: float_bits(&[4.0, 3.0, 3.0, 9.0]),
            },
            &ExecutionContext::for_tests(11),
        )
        .expect("seal");
        let bytes = wire.export().expect("export");
        (bytes, wire.identity)
    }

    fn container<T: Serialize>(kind: &str, section: &str, meta: &T) -> Vec<u8> {
        encode_sectioned_io(
            &ArtifactContainer {
                kind,
                note: kind,
                max_bytes: 1024 * 1024,
                artifact_id: kind,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            section,
            meta,
            &[],
            "producer reference",
        )
        .expect("container")
    }

    #[test]
    fn a_propagation_reference_accepts_its_identity_and_refuses_a_tampered_digest() {
        let (bytes, identity) = contrast_bytes(5.0);
        let node = producer_reference("prop", PROPAGATION, &bytes).unwrap();
        assert_eq!(node.identity, identity);
        assert_eq!(node.kind, NodeKind::PropagationChain);
        assert_eq!(node.facts.get(CATALOG_ROLE_FACT).map(String::as_str), Some(PROPAGATION));
        let limits = BundleLimits::default();
        let bundle = CompositionBundle::new(vec![node], &[], &limits).unwrap();
        let exported = bundle.to_bytes("producer-ref", &limits).unwrap();
        let supplied = SuppliedSources::default().with_producer(PROPAGATION, bytes.clone());
        let consumed = BundleConsumer::new()
            .consume(&exported, &limits, bundle.identity(), &supplied)
            .unwrap();
        assert_eq!(consumed.nodes()[0].status, NodeStatus::Verified);

        let tampered_id = "00".repeat(32);
        let tampered = BundleNode::reference(
            "prop",
            NodeKind::PropagationChain,
            &tampered_id,
            ProviderOrData::Data { snapshot_id: PROPAGATION.into(), digest: tampered_id.clone() },
        )
        .with_fact(CATALOG_ROLE_FACT, PROPAGATION);
        let bundle = CompositionBundle::new(vec![tampered], &[], &limits).unwrap();
        let exported = bundle.to_bytes("producer-ref", &limits).unwrap();
        let consumed = BundleConsumer::new()
            .consume(&exported, &limits, bundle.identity(), &supplied)
            .unwrap();
        assert!(matches!(
            consumed.nodes()[0].status,
            NodeStatus::Failed { stage: BundleStage::TamperedQuantity, .. }
        ));

        let node = producer_reference("prop", PROPAGATION, &bytes).unwrap();
        let bundle = CompositionBundle::new(vec![node], &[], &limits).unwrap();
        let exported = bundle.to_bytes("producer-ref", &limits).unwrap();
        let consumed = BundleConsumer::new()
            .consume(&exported, &limits, bundle.identity(), &SuppliedSources::default())
            .unwrap();
        assert!(matches!(consumed.nodes()[0].status, NodeStatus::ReferenceUnresolved { .. }));
    }

    #[test]
    fn two_references_resolve_their_own_producer_in_either_order() {
        let (first, first_identity) = contrast_bytes(5.0);
        let (second, second_identity) = contrast_bytes(7.0);
        assert_ne!(first_identity, second_identity);
        let nodes = vec![
            producer_reference("first", PROPAGATION, &first).unwrap(),
            producer_reference("second", PROPAGATION, &second).unwrap(),
        ];
        let limits = BundleLimits::default();
        let bundle = CompositionBundle::new(nodes, &[], &limits).unwrap();
        let bytes = bundle.to_bytes("producer-references", &limits).unwrap();
        for ordered in [[&first, &second], [&second, &first]] {
            let supplied = SuppliedSources::default()
                .with_producer(PROPAGATION, ordered[0].clone())
                .with_producer(PROPAGATION, ordered[1].clone());
            let consumed = BundleConsumer::new()
                .consume(&bytes, &limits, bundle.identity(), &supplied)
                .unwrap();
            assert_eq!(consumed.nodes().len(), 2);
            assert!(consumed.nodes().iter().all(|node| node.status == NodeStatus::Verified));
        }
    }

    #[test]
    fn ledger_measurement_and_discovery_references_validate_the_producer_body() {
        let ledger = EvidenceLedger::new();
        let ledger_bytes = EvidenceLedgerArtifact::to_bytes(&ledger, &[], "ledger").unwrap();
        let (_, mut ledger_meta) = EvidenceLedgerArtifact::seal(&ledger, &[]).unwrap();
        ledger_meta.state_digest = "00".repeat(32);
        let edited_ledger = container("evidence_ledger_v1", "evidence_ledger_meta", &ledger_meta);

        let mut model = MeasurementModel {
            model_id: "screen".into(),
            route: MeasurementRoute::BinaryMisclassification,
            indicator: MeasurementVariable {
                id: "indicator".into(),
                kind: MeasurementKind::NoisyMisclassified,
                units: "binary".into(),
            },
            construct: MeasurementVariable {
                id: "construct".into(),
                kind: MeasurementKind::LatentScientific,
                units: "binary".into(),
            },
            binding: MeasurementBinding {
                population_id: "population".into(),
                regime_id: "observational".into(),
                arm: ArmBinding::AllArms,
            },
            structure: ErrorStructure::NonDifferential,
            parameters: MeasurementParameters::Binary {
                rates: vec![BinaryArmRates {
                    arm_id: None,
                    sensitivity: 0.9,
                    specificity: 0.8,
                    sensitivity_variance: 0.0004,
                    specificity_variance: 0.0009,
                }],
                min_youden: 0.1,
            },
            representation: RepresentationVersion {
                coordinate_system_id: "scalar".into(),
                version: 1,
            },
            evidence: ParameterEvidence::ValidationSample { sample_id: "val".into(), units: 400 },
        };
        let measurement = MeasurementModelArtifact::new(&model, None).unwrap();
        let measurement_bytes = measurement.to_bytes("measurement").unwrap();
        let mut measurement_meta = measurement.metadata().clone();
        model.binding.population_id = "different_population".into();
        measurement_meta.model =
            MeasurementModelArtifact::new(&model, None).unwrap().metadata().model.clone();
        let edited_measurement =
            container("measurement_model_v1", "measurement_model_meta", &measurement_meta);

        let discovery_bytes =
            include_bytes!("../../../antecedent-io-core/tests/fixtures/selective_discovery_v1.bin");
        let mut discovery_meta = SelectiveDiscoveryArtifact::inspect(discovery_bytes).unwrap();
        discovery_meta.work.algorithm_executions += 1;
        let edited_discovery =
            SelectiveDiscoveryArtifact::meta_to_bytes(&discovery_meta, "edited").unwrap();

        for (role, bytes, edited) in [
            (LEDGER, ledger_bytes.as_slice(), &edited_ledger),
            (MEASUREMENT, measurement_bytes.as_slice(), &edited_measurement),
            (DISCOVERY, discovery_bytes.as_slice(), &edited_discovery),
        ] {
            let node = producer_reference("producer", role, bytes).unwrap();
            let limits = BundleLimits::default();
            let bundle = CompositionBundle::new(vec![node], &[], &limits).unwrap();
            let exported = bundle.to_bytes("producer-ref", &limits).unwrap();
            let consume = |bytes: &[u8]| {
                let supplied = SuppliedSources::default().with_producer(role, bytes.to_vec());
                BundleConsumer::new()
                    .consume(&exported, &limits, bundle.identity(), &supplied)
                    .unwrap()
            };
            assert_eq!(consume(bytes).nodes()[0].status, NodeStatus::Verified, "{role}");
            assert!(
                matches!(
                    consume(edited).nodes()[0].status,
                    NodeStatus::Failed { stage: BundleStage::SwappedEvidence, .. }
                ),
                "{role}"
            );
        }

        let (propagation, _) = contrast_bytes(5.0);
        assert!(matches!(
            check_producer(LEDGER, &ledger_meta.identity, Some(&propagation)),
            ProducerCheck::Refused { stage: BundleStage::SwappedEvidence, .. }
        ));
        assert!(producer_identity(COMPOSED_MODEL, b"not a model").is_err());
        assert!(producer_identity(RELIABILITY, &ledger_bytes).is_err());
    }

    #[test]
    fn a_reference_without_the_catalog_fact_still_resolves_from_supplied_data() {
        let node = BundleNode::reference(
            "data",
            NodeKind::CausalContract,
            "digest-1",
            ProviderOrData::Data { snapshot_id: "snap".into(), digest: "digest-1".into() },
        );
        let limits = BundleLimits::default();
        let bundle = CompositionBundle::new(vec![node], &[], &limits).unwrap();
        let exported = bundle.to_bytes("plain", &limits).unwrap();
        let supplied = SuppliedSources::default().with_data("snap", "digest-1");
        let consumed = BundleConsumer::new()
            .consume(&exported, &limits, bundle.identity(), &supplied)
            .unwrap();
        assert_eq!(consumed.nodes()[0].status, NodeStatus::Verified);
        assert!(consumed.nodes()[0].facts.get(CATALOG_ROLE_FACT).is_none());
        assert_eq!(consumed.nodes()[0].kind, NodeKind::CausalContract);
    }
}
