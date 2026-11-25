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
use antecedent_io_core::evidence_ledger_artifact::EvidenceLedgerMeta;
use antecedent_io_core::joint_law_propagation_artifact::{
    JointLawPropagationArtifactWire, joint_law_propagation_identity,
};
use antecedent_io_core::measurement_model_wire::MeasurementIdentityWire;
use antecedent_io_core::mechanism_composition_artifact::{
    MechanismCompositionArtifactWire, mechanism_composition_identity,
};
use antecedent_io_core::selective_discovery_artifact::SelectiveDiscoveryArtifact;
use antecedent_io_core::{
    ArtifactKind, EncodedArtifact, PropagationChainRecordWire, propagation_chain_identity,
    PROPAGATION_CHAIN_SECTION,
};
use serde::Deserialize;

use super::{BundleError, BundleNode, BundleStage, NodeKind, ProviderOrData};

/// Fact that names the catalog role of a producer reference.
pub const CATALOG_ROLE_FACT: &str = "catalog_role";

const LEDGER_KIND: &str = "evidence_ledger_v1";
const LEDGER_SECTION: &str = "evidence_ledger_meta";
const MEASUREMENT_KIND: &str = "measurement_model_v1";
const MEASUREMENT_SECTION: &str = "measurement_model_meta";

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
/// `role` is not a producer role, or `bytes` are not that artifact. A stored
/// digest that differs from the producer's identity function is an error for
/// joint-law propagation, mechanism composition, and decision reliability.
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
    let identity = producer_identity(role, bytes).map_err(|reason| {
        BundleError::Refused { stage: BundleStage::SwappedEvidence, reason, offending: None }
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
    let payload = section(bytes, LEDGER_KIND, LEDGER_SECTION)?;
    let meta: EvidenceLedgerMeta = from_cbor(&payload).map_err(|error| error.to_string())?;
    if meta.identity.is_empty() {
        return Err("ledger identity is empty".into());
    }
    Ok(meta.identity)
}

fn composed_identity(bytes: &[u8]) -> Result<String, String> {
    let wire = MechanismCompositionArtifactWire::decode(bytes).map_err(|error| error.to_string())?;
    let identity = mechanism_composition_identity(&wire.premises, &wire.query)
        .map_err(|error| error.to_string())?;
    if identity != wire.premises_digest {
        return Err("composed-model identity does not match the stored digest".into());
    }
    Ok(identity)
}

fn measurement_identity(bytes: &[u8]) -> Result<String, String> {
    #[derive(Deserialize)]
    struct Read {
        identities: MeasurementIdentityWire,
    }
    let payload = section(bytes, MEASUREMENT_KIND, MEASUREMENT_SECTION)?;
    let read: Read = from_cbor(&payload).map_err(|error| error.to_string())?;
    if read.identities.model.is_empty() {
        return Err("measurement identity is empty".into());
    }
    Ok(read.identities.model)
}

fn discovery_identity(bytes: &[u8]) -> Result<String, String> {
    let meta = SelectiveDiscoveryArtifact::inspect(bytes).map_err(|error| error.to_string())?;
    if meta.identity.is_empty() {
        return Err("discovery identity is empty".into());
    }
    Ok(meta.identity)
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

fn section(bytes: &[u8], kind: &str, id: &str) -> Result<Vec<u8>, String> {
    let artifact = EncodedArtifact::read_from(std::io::Cursor::new(bytes))
        .map_err(|error| error.to_string())?;
    match &artifact.manifest.artifact_kind {
        ArtifactKind::Other(found) if found == kind => {}
        _ => return Err(format!("producer kind is not {kind}")),
    }
    artifact
        .sections
        .iter()
        .find(|item| item.id == id)
        .map(|item| item.data.as_ref().to_vec())
        .ok_or_else(|| format!("missing section {id}"))
}

#[cfg(test)]
mod tests {
    use super::{
        CATALOG_ROLE_FACT, LEDGER_KIND, LEDGER_SECTION, MEASUREMENT_KIND, MEASUREMENT_SECTION,
        ProducerCheck, check_producer, producer_identity, producer_reference,
    };
    use crate::composition_bundle::{
        BundleConsumer, BundleLimits, BundleNode, BundleStage, CompositionBundle, NodeKind,
        NodeStatus, ProviderOrData, SuppliedSources,
    };
    use antecedent_contract::{
        InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
        PropagationRequest, RequestedReading,
    };
    use antecedent_core::ExecutionContext;
    use antecedent_io_core::artifact_codec::{ArtifactContainer, encode_sectioned_io};
    use antecedent_io_core::catalog_role::{
        COMPOSED_MODEL, DISCOVERY, LEDGER, MEASUREMENT, PROPAGATION, RELIABILITY,
    };
    use antecedent_io_core::evidence_ledger_artifact::EvidenceLedgerMeta;
    use antecedent_io_core::joint_law_propagation_artifact::{
        JointLawPropagationArtifactWire, PropagationRequestWire, float_bits,
    };
    use antecedent_io_core::measurement_model_wire::MeasurementIdentityWire;
    use antecedent_io_core::selective_discovery_artifact::{
        SELECTIVE_DISCOVERY_ARTIFACT_FEATURE, SELECTIVE_DISCOVERY_ARTIFACT_VERSION,
        SelectiveDiscoveryMeta, StructureWire, WorkWire,
    };
    use serde::Serialize;

    fn contrast_bytes() -> (Vec<u8>, String) {
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
                means: float_bits(&[5.0, 2.0]),
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
        let (bytes, identity) = contrast_bytes();
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
        assert!(matches!(
            consumed.nodes()[0].status,
            NodeStatus::ReferenceUnresolved { .. }
        ));
    }

    #[test]
    fn ledger_measurement_and_discovery_identities_bind_and_a_foreign_role_does_not() {
        let ledger = container(
            LEDGER_KIND,
            LEDGER_SECTION,
            &EvidenceLedgerMeta {
                version: 1,
                feature: LEDGER_KIND.into(),
                operations: Vec::new(),
                syntheses: Vec::new(),
                head: "head".into(),
                state_digest: "state".into(),
                identity: "ledger-identity".into(),
            },
        );
        assert_eq!(producer_identity(LEDGER, &ledger).unwrap(), "ledger-identity");
        assert!(matches!(
            check_producer(LEDGER, "other", Some(&ledger)),
            ProducerCheck::Refused { stage: BundleStage::TamperedQuantity, .. }
        ));

        let measurement = container(
            MEASUREMENT_KIND,
            MEASUREMENT_SECTION,
            &MeasurementWrap {
                identities: MeasurementIdentityWire {
                    measurement: "m".into(),
                    population: "p".into(),
                    regime: "r".into(),
                    arm: "a".into(),
                    representation: "rep".into(),
                    evidence: "e".into(),
                    model: "measurement-identity".into(),
                },
            },
        );
        assert_eq!(
            producer_identity(MEASUREMENT, &measurement).unwrap(),
            "measurement-identity"
        );

        let discovery = container(
            "selective_discovery_v1",
            "selective_discovery_meta",
            &SelectiveDiscoveryMeta {
                version: SELECTIVE_DISCOVERY_ARTIFACT_VERSION,
                feature: SELECTIVE_DISCOVERY_ARTIFACT_FEATURE.into(),
                required_features: Vec::new(),
                inference: "algorithmic_match_only".into(),
                route: "route".into(),
                components: Vec::new(),
                contract_identity: "c".into(),
                data_identity: "d".into(),
                rows: 0,
                variables: Vec::new(),
                receipts: Vec::new(),
                receipts_identity: "r".into(),
                structure: StructureWire { class: "cpdag".into(), nodes: Vec::new(), edges: Vec::new() },
                graph_identity: "g".into(),
                result_identity: "result".into(),
                search_incomplete: false,
                orientation_conflict: false,
                bounded_search_receipts: Vec::new(),
                work: WorkWire {
                    ci_tests_attempted: 0,
                    ci_tests_completed: 0,
                    ci_tests_reused: 0,
                    ci_tests_consumed: 0,
                    search_passes: 0,
                    orientation_runs: 0,
                    orientation_rounds: 0,
                    orientations_applied: 0,
                    score_evaluations: 0,
                    algorithm_executions: 0,
                },
                identity: "discovery-identity".into(),
            },
        );
        assert_eq!(producer_identity(DISCOVERY, &discovery).unwrap(), "discovery-identity");

        let (propagation, _) = contrast_bytes();
        assert!(matches!(
            check_producer(LEDGER, "ledger-identity", Some(&propagation)),
            ProducerCheck::Refused { stage: BundleStage::SwappedEvidence, .. }
        ));
        assert!(producer_identity(COMPOSED_MODEL, b"not a model").is_err());
        assert!(producer_identity(RELIABILITY, b"not a reliability artifact").is_err());
        assert!(producer_identity(RELIABILITY, &ledger).is_err());
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

    #[derive(Serialize)]
    struct MeasurementWrap {
        identities: MeasurementIdentityWire,
    }
}
