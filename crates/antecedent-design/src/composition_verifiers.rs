//! Node verifiers for the composition bundle.
//!
//! [`crate::composition_bundle`] decodes embedded nodes only through the
//! [`NodeVerifier`] registered for their [`NodeKind`]. This module supplies one
//! verifier per portable artifact kind, [`standard_consumer`] registering them
//! all, and the shared [`describe_artifact`] a bundle builder uses to read an
//! artifact's identity and binding facts from its bytes.
//!
//! # What a verifier does
//!
//! Each verifier decodes the artifact through that artifact's own consumer (the
//! container reader, the digest and lineage recomputation, the replay of a stored
//! table), recomputes its identity from the decoded content and publishes the
//! binding facts the bundle cross-checks across edges. The facts are read from the
//! decoded artifact's fields, never from text a producer wrote next to it:
//!
//! * a **source** (a distribution or an external claim) publishes
//!   `quantity_digest` over its whole coordinate set, `graph_or_snapshot`, its
//!   `law` and its `trust`; an external claim also publishes `request_fingerprint`
//!   and `evidence_digest`;
//! * a **consumer** (a decision, a sensitivity artifact, a ranking) publishes the
//!   coordinates it reads in `quantity_coordinates` and is refused when its
//!   sources do not carry them, and publishes `requires_law` when its functional
//!   needs an aligned joint law;
//! * a decision result rests on a decision contract and on draws or means by
//!   digest: the result is verified against the contract and the source nodes
//!   upstream of it, so replacing either upstream node, even with a resealed and
//!   self-consistent artifact, fails the dependent node at a named stage.
//!
//! Only a joint distribution publishes the aligned joint law. An external claim is
//! a finite response grid: it publishes `law = mean_only` and trust
//! `externally_attested`, upgraded to `verified_extension` only when the artifact
//! retains its exact-request verification receipt.
//!
//! # Retained identity
//!
//! An artifact's own consumer wants an identity the consumer retained
//! independently of the bytes. Inside a bundle that identity is the node's declared
//! identity, which the bundle's Merkle chain covers and the consumer's retained
//! bundle identity anchors: a verifier recomputes the identity from the decoded
//! content and the bundle refuses a node whose recomputation differs.

use std::collections::BTreeMap;
use std::io::Cursor;

use antecedent_core::{ExecutionContext, ScientificQuantity};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionMetadata, DistributionTrust, DrawAlignment,
    MAX_DISTRIBUTION_ARTIFACT_BYTES,
};
use antecedent_io::error::IoError;
use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimMetadata, ExternalClaimTrust,
    MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::reader::ArtifactReader;
use antecedent_io::sensitivity_artifact::{MAX_SENSITIVITY_ARTIFACT_BYTES, SensitivityArtifact};
use antecedent_io::wire::ArtifactKind;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::composition_bundle::{
    BundleConsumer, BundleStage, EvidenceRelationship, FACT_EVIDENCE_DIGEST, FACT_LAW,
    FACT_QUANTITY_DIGEST, FACT_REQUEST, FACT_REQUIRES_LAW, FACT_SNAPSHOT, FACT_TRUST, LAW_JOINT,
    LAW_MEAN_ONLY, NodeFailure, NodeKind, NodeVerifier, UpstreamNode, VerifiedNode,
};
use crate::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, MAX_DECISION_ARTIFACT_BYTES,
    mean_result_to_bytes, mean_source_digest, source_digest,
};
use crate::decision_contract::{DecisionContract, SourceRepresentation};
use crate::decision_eval::{DecisionEvalError, MeanSource, evaluate_contract_on_means};
use crate::decision_robust_artifact::{
    AdmissibleContractArtifact, ExternalTrustLimit, RobustResultArtifact,
};
use crate::decision_structural_artifact::StructuralResultArtifact;
use crate::design_ranking_artifact::{
    ConsumeExpectation, DESIGN_RANKING_ARTIFACT_KIND, DesignRankingArtifactError,
    MAX_DESIGN_RANKING_ARTIFACT_BYTES, consume as consume_ranking,
};
use crate::inverse_query_artifact::{
    INVERSE_QUERY_ARTIFACT_FEATURE, InverseQueryArtifact, MAX_INVERSE_QUERY_ARTIFACT_BYTES,
};
use crate::repair_artifact::{
    REPAIR_ARTIFACT_KIND, RepairArtifactError, RepairConsumeLimits, RepairReportArtifact,
};
use crate::signal::SignalTrustLabel;

/// Fact key: the coordinates a node reads or carries, as a sorted comma-joined
/// list of per-coordinate digests.
pub const FACT_COORDINATES: &str = "quantity_coordinates";
/// Fact key: the digest of the draws a distribution holds (the digest a decision
/// result retains of its source).
pub const FACT_SOURCE_DIGEST: &str = "source_digest";
/// Fact key: the digest of the mean source an external claim yields (the digest a
/// point-only decision result retains of its source).
pub const FACT_MEAN_SOURCE: &str = "mean_source_digest";
/// Fact key: `provider|snapshot|request` an external claim answered.
pub const FACT_PROVIDER_BINDING: &str = "provider_binding";
/// Fact key: `true` when the node's functional cannot be answered from a mean.
pub const FACT_REQUIRES_DISTRIBUTION: &str = "requires_distribution";
/// Fact value of [`FACT_LAW`]: independently sampled marginals, no pairing.
pub const LAW_MARGINAL: &str = "marginal_draw";
/// Fact value of [`FACT_TRUST`]: a supplier assertion only.
pub const TRUST_ATTESTED: &str = "externally_attested";
/// Fact value of [`FACT_TRUST`]: the exact request passed object-level
/// verification and the artifact retains the receipt.
pub const TRUST_VERIFIED: &str = "verified_extension";
/// Fact value of [`FACT_TRUST`]: a natively licensed route.
pub const TRUST_NATIVE: &str = "native_licensed";
/// Fact value of [`FACT_TRUST`]: no provider verification claim.
pub const TRUST_UNVERIFIED: &str = "unverified";

/// Node kinds that have an embedded-artifact verifier, in registration order.
pub const VERIFIABLE_KINDS: [NodeKind; 11] = [
    NodeKind::Distribution,
    NodeKind::ExternalClaim,
    NodeKind::DecisionContract,
    NodeKind::DecisionResult,
    NodeKind::Sensitivity,
    NodeKind::StudyRanking,
    NodeKind::InverseQuery,
    NodeKind::RepairReport,
    NodeKind::EvidenceRelationship,
    NodeKind::RecalculationReceipt,
    NodeKind::FrozenScores,
];

const KIND_DISTRIBUTION: &str = "joint_distribution_v1";
const KIND_EXTERNAL_CLAIM: &str = "external_response_claim_v1";
const KIND_CONTRACT: &str = "decision_contract_v1";
const KIND_ADMISSIBLE: &str = "admissible_decision_contract_v1";
const KIND_RESULT: &str = "decision_result_v1";
const KIND_ROBUST: &str = "robust_decision_result_v1";
const KIND_STRUCTURAL: &str = "decision_structural_result_v1";
const KIND_SENSITIVITY: &str = "sensitivity_decision_v1";
const MAX_SNIFF_BYTES: usize = 64 * 1024 * 1024;
const MAX_VALUES: usize = 256;
const RELATIONSHIP_MAGIC: &str = "evidence_relationship_v1";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn failure(stage: BundleStage, reason: impl Into<String>) -> NodeFailure {
    NodeFailure { stage, reason: reason.into() }
}

/// Stage an artifact that does not decode or verify fails at: a value-bearing
/// artifact was altered, any other was swapped.
const fn default_stage(kind: NodeKind) -> BundleStage {
    match kind {
        NodeKind::QuantityCoordinates
        | NodeKind::DecisionResult
        | NodeKind::Sensitivity
        | NodeKind::StudyRanking
        | NodeKind::InverseQuery => BundleStage::TamperedQuantity,
        _ => BundleStage::SwappedEvidence,
    }
}

fn io_failure(kind: NodeKind, error: &IoError) -> NodeFailure {
    let stage = match error {
        IoError::TooLarge => BundleStage::Oversized,
        IoError::UnsupportedVersion { .. } | IoError::UnsupportedFormat { .. } => {
            BundleStage::IncompatibleVersion
        }
        _ => default_stage(kind),
    };
    failure(stage, error.to_string())
}

fn put(hasher: &mut blake3::Hasher, text: &str) {
    hasher.update(&(text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
}

fn digest_parts(parts: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    put(&mut hasher, "antecedent.composition_verifiers.v1");
    for part in parts {
        put(&mut hasher, part);
    }
    hasher.finalize().to_hex().to_string()
}

fn open(bytes: &[u8], limit: usize) -> Result<ArtifactReader<Cursor<&[u8]>>, IoError> {
    if bytes.len() > limit {
        return Err(IoError::TooLarge);
    }
    ArtifactReader::open_seek(Cursor::new(bytes))
}

/// Decode the first section of a container as CBOR, refusing an oversized
/// section before any allocation.
fn first_section<T: DeserializeOwned>(bytes: &[u8], limit: usize) -> Result<T, IoError> {
    let mut reader = open(bytes, limit)?;
    let Some(section) = reader.manifest().sections.first() else {
        return Err(IoError::Convert("the artifact has no sections".into()));
    };
    if section.uncompressed_size > limit as u64 {
        return Err(IoError::TooLarge);
    }
    let id = section.id.clone();
    let loaded = reader.load_section(&id)?;
    from_cbor(loaded.as_bytes())
}

/// The artifact kind label of a container.
fn container_label(bytes: &[u8], limit: usize) -> Result<String, IoError> {
    let reader = open(bytes, limit)?;
    match &reader.manifest().artifact_kind {
        ArtifactKind::Other(name) => Ok(name.clone()),
        _ => Err(IoError::Convert("not a composition-bundle artifact kind".into())),
    }
}

/// Digest over every section of a container (not its artifact id or library
/// version), for an artifact that carries no identity of its own. Call it only
/// after the artifact's own consumer accepted the bytes.
fn content_identity(tag: &str, bytes: &[u8], limit: usize) -> Result<String, IoError> {
    let mut reader = open(bytes, limit)?;
    let ids: Vec<String> = reader.manifest().sections.iter().map(|s| s.id.clone()).collect();
    let mut hasher = blake3::Hasher::new();
    put(&mut hasher, "antecedent.composition_verifiers.content.v1");
    put(&mut hasher, tag);
    for id in ids {
        let section = reader.load_section(&id)?;
        put(&mut hasher, &id);
        hasher.update(blake3::hash(section.as_bytes()).as_bytes());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn coordinate_id(quantity: &ScientificQuantityWire) -> String {
    let digest = blake3::hash(format!("{quantity:?}").as_bytes()).to_hex().to_string();
    digest[..16].to_owned()
}

fn coordinate_ids<'a>(
    quantities: impl IntoIterator<Item = &'a ScientificQuantityWire>,
) -> Vec<String> {
    let mut ids: Vec<String> = quantities.into_iter().map(coordinate_id).collect();
    ids.sort();
    ids.dedup();
    ids
}

fn publish(facts: &mut BTreeMap<String, String>, key: &str, value: impl Into<String>) {
    facts.insert(key.to_owned(), value.into());
}

fn publish_coordinates(facts: &mut BTreeMap<String, String>, ids: &[String], whole_set: bool) {
    publish(facts, FACT_COORDINATES, ids.join(","));
    if whole_set {
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        publish(facts, FACT_QUANTITY_DIGEST, digest_parts(&refs));
    }
}

fn contract_coordinates(contract: &DecisionContract) -> Vec<String> {
    let wires: Vec<ScientificQuantityWire> = contract
        .actions
        .iter()
        .flat_map(|action| action.inputs.iter())
        .map(ScientificQuantityWire::from)
        .collect();
    coordinate_ids(&wires)
}

/// Publish what the contract's functional needs of a source: a mean is enough or
/// it is not, and whether an aligned joint law is the only representation.
fn publish_requirement(
    contract: &DecisionContract,
    facts: &mut BTreeMap<String, String>,
) -> Result<(), NodeFailure> {
    let requirement = contract
        .source_requirement()
        .map_err(|e| failure(BundleStage::TamperedQuantity, format!("{e:?}")))?;
    let Some(requirement) = requirement else {
        return Ok(());
    };
    if !requirement.any_of.contains(&SourceRepresentation::Mean) {
        publish(facts, FACT_REQUIRES_DISTRIBUTION, "true");
    }
    let joint_only = !requirement.any_of.is_empty()
        && requirement.any_of.iter().all(|r| *r == SourceRepresentation::JointDraws);
    if joint_only {
        publish(facts, FACT_REQUIRES_LAW, LAW_JOINT);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Inspection: decode one artifact on its own
// ---------------------------------------------------------------------------

/// A receipt of an external callback a result retains.
struct ReceiptInfo {
    provider_id: String,
    snapshot_id: String,
    request: String,
    verified: bool,
}

/// What an artifact's own consumer established, before any upstream is consulted.
#[derive(Default)]
struct Inspected {
    identity: String,
    facts: BTreeMap<String, String>,
    values: Vec<(String, f64)>,
    /// Identity of the decision contract the artifact is bound to.
    contract: Option<String>,
    /// The contract must be an upstream node (a result), not merely agree with one.
    contract_required: bool,
    /// Digests of the draws or means the artifact rests on.
    sources: Vec<String>,
    /// Every digest in `sources` must be an upstream node's (a single-source
    /// result); otherwise each upstream distribution must be one of `sources`.
    sources_must_be_upstream: bool,
    /// The artifact inherits the contract's coordinate and law requirements.
    bound_to_contract: bool,
    receipts: Vec<ReceiptInfo>,
    /// For an evidence relationship: relationship, left and right node ids.
    relationship: Option<(EvidenceRelationship, String, String)>,
}

fn push_value(values: &mut Vec<(String, f64)>, key: String, value: f64) {
    if values.len() < MAX_VALUES && value.is_finite() {
        values.push((key, value));
    }
}

fn trust_label(trust: DistributionTrust) -> &'static str {
    match trust {
        DistributionTrust::NativeLicensed => TRUST_NATIVE,
        DistributionTrust::ExternalAttested => TRUST_ATTESTED,
        DistributionTrust::VerifiedExtension => TRUST_VERIFIED,
        DistributionTrust::Unverified => TRUST_UNVERIFIED,
    }
}

fn inspect_distribution(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::Distribution;
    let meta: DistributionMetadata =
        first_section(bytes, MAX_DISTRIBUTION_ARTIFACT_BYTES).map_err(|e| io_failure(kind, &e))?;
    let artifact = DistributionArtifact::from_bytes(bytes, &meta.identity)
        .map_err(|e| io_failure(kind, &e))?;
    let metadata = artifact.metadata();
    let identity = &metadata.identity;
    let draws = source_digest(&artifact);
    let metadata_bytes = to_cbor(metadata).map_err(|e| io_failure(kind, &e))?;
    let metadata_digest = blake3::hash(&metadata_bytes).to_hex().to_string();
    let mut facts = BTreeMap::new();
    let law = match identity.alignment {
        DrawAlignment::Joint => LAW_JOINT,
        DrawAlignment::IndependentMarginals => LAW_MARGINAL,
    };
    publish(&mut facts, FACT_LAW, law);
    publish(&mut facts, FACT_TRUST, trust_label(metadata.trust));
    publish(
        &mut facts,
        FACT_SNAPSHOT,
        format!("{}|{}", identity.snapshot_id, identity.causal_contract_id),
    );
    publish(&mut facts, FACT_SOURCE_DIGEST, draws.clone());
    publish_coordinates(&mut facts, &coordinate_ids(artifact.quantities()), true);
    let mut values = Vec::new();
    push_value(&mut values, "n_draws".to_owned(), artifact.n_draws() as f64);
    for coordinate in 0..artifact.shape()[1].min(MAX_VALUES) {
        if let Ok(mean) = artifact.mean(coordinate) {
            push_value(&mut values, format!("mean.{coordinate}"), mean);
        }
    }
    Ok(Inspected {
        identity: digest_parts(&["distribution", &draws, &metadata_digest]),
        facts,
        values,
        ..Inspected::default()
    })
}

/// The mean source an external claim yields: its coordinates and values under the
/// provider's snapshot and the claim's causal contract. A point-only decision
/// computed from this source retains [`mean_source_digest`] of it.
///
/// # Errors
/// A coordinate that does not convert refuses.
pub fn mean_source_of(claim: &ExternalClaimArtifact) -> Result<MeanSource, IoError> {
    let identity = &claim.metadata().identity;
    let coordinates = identity
        .quantities
        .iter()
        .map(|wire| {
            ScientificQuantity::try_from(wire.clone())
                .map_err(|reason| IoError::Convert(reason.into()))
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    Ok(MeanSource {
        coordinates,
        means: claim.values().to_vec(),
        provider_id: identity.provider_id.clone(),
        snapshot_id: identity.snapshot_id.clone(),
        causal_contract_id: identity.causal_contract_id.clone(),
        rng_id: "none:mean_grid".to_owned(),
    })
}

/// Evaluate a decision contract on the means of an external claim's container
/// bytes and return the point-only result as container bytes, ready to embed as a
/// decision-result node beneath the claim.
///
/// The claim is decoded through its own consumer first. A contract whose functional
/// a mean cannot answer (a quantile, a probability or a nonlinear utility) is
/// refused as `unsupported_law`: a mean never yields an outcome distribution.
///
/// # Errors
/// A claim its own consumer refuses, a contract that cannot be evaluated on the
/// means, or an encoding failure.
pub fn mean_decision_bytes(
    contract: &DecisionContract,
    claim_bytes: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, NodeFailure> {
    let kind = NodeKind::ExternalClaim;
    let meta: ExternalClaimMetadata = first_section(claim_bytes, MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES)
        .map_err(|e| io_failure(kind, &e))?;
    let claim = ExternalClaimArtifact::from_bytes(claim_bytes, &meta.identity)
        .map_err(|e| io_failure(kind, &e))?;
    let source = mean_source_of(&claim).map_err(|e| io_failure(kind, &e))?;
    let result = evaluate_contract_on_means(contract, &source).map_err(|e| {
        let stage = match e {
            DecisionEvalError::MeanSourceInsufficient { .. } => BundleStage::UnsupportedLaw,
            _ => BundleStage::TamperedQuantity,
        };
        failure(stage, format!("{e:?}"))
    })?;
    mean_result_to_bytes(&result, &source, artifact_id)
        .map_err(|e| io_failure(NodeKind::DecisionResult, &e))
}

fn inspect_external_claim(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::ExternalClaim;
    let meta: ExternalClaimMetadata = first_section(bytes, MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES)
        .map_err(|e| io_failure(kind, &e))?;
    let artifact = ExternalClaimArtifact::from_bytes(bytes, &meta.identity)
        .map_err(|e| io_failure(kind, &e))?;
    let identity = &artifact.metadata().identity;
    let trust = match (identity.trust, identity.verification.is_some()) {
        (ExternalClaimTrust::ExactRequestVerified, true) => TRUST_VERIFIED,
        _ => TRUST_ATTESTED,
    };
    let mut facts = BTreeMap::new();
    publish(&mut facts, FACT_LAW, LAW_MEAN_ONLY);
    publish(&mut facts, FACT_TRUST, trust);
    publish(
        &mut facts,
        FACT_SNAPSHOT,
        format!("{}|{}", identity.snapshot_id, identity.causal_contract_id),
    );
    publish(&mut facts, FACT_REQUEST, identity.request_id.clone());
    publish(
        &mut facts,
        FACT_PROVIDER_BINDING,
        format!("{}|{}|{}", identity.provider_id, identity.snapshot_id, identity.request_id),
    );
    let sorted = |ids: &[String]| {
        let mut ids = ids.to_vec();
        ids.sort();
        ids.join(",")
    };
    publish(
        &mut facts,
        FACT_EVIDENCE_DIGEST,
        digest_parts(&[
            "evidence",
            &sorted(&identity.evidence_ids),
            &sorted(&identity.assumption_ids),
            &sorted(&identity.equivalence_ids),
        ]),
    );
    let mean_source = mean_source_of(&artifact).map_err(|e| io_failure(kind, &e))?;
    publish(&mut facts, FACT_MEAN_SOURCE, mean_source_digest(&mean_source));
    publish_coordinates(&mut facts, &coordinate_ids(&identity.quantities), true);
    let identity_bytes = to_cbor(identity).map_err(|e| io_failure(kind, &e))?;
    let identity_digest = blake3::hash(&identity_bytes).to_hex().to_string();
    let mut values = Vec::new();
    for (index, value) in artifact.values().iter().enumerate() {
        push_value(&mut values, format!("value.{index}"), *value);
    }
    Ok(Inspected {
        identity: digest_parts(&["external_claim", &identity_digest]),
        facts,
        values,
        ..Inspected::default()
    })
}

fn body_str(body: &Value, path: &[&str]) -> String {
    let mut node = body;
    for key in path {
        node = &node[*key];
    }
    node.as_str().unwrap_or_default().to_owned()
}

fn inspect_contract(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::DecisionContract;
    let io = |e: &IoError| io_failure(kind, e);
    let label = container_label(bytes, MAX_DECISION_ARTIFACT_BYTES).map_err(|e| io(&e))?;
    let body: Value = first_section(bytes, MAX_DECISION_ARTIFACT_BYTES).map_err(|e| io(&e))?;
    let stored = body_str(&body, &["identity"]);
    let (identity, contract) = match label.as_str() {
        KIND_CONTRACT => {
            let artifact =
                DecisionContractArtifact::from_bytes(bytes, &stored).map_err(|e| io(&e))?;
            (artifact.identity().to_owned(), artifact.contract().clone())
        }
        KIND_ADMISSIBLE => {
            let artifact =
                AdmissibleContractArtifact::from_bytes(bytes, &stored).map_err(|e| io(&e))?;
            (artifact.identity().to_owned(), artifact.contract().contract.clone())
        }
        other => {
            return Err(failure(
                BundleStage::SwappedEvidence,
                format!("`{other}` is not a decision contract artifact"),
            ));
        }
    };
    let mut facts = BTreeMap::new();
    publish_coordinates(&mut facts, &contract_coordinates(&contract), false);
    publish_requirement(&contract, &mut facts)?;
    Ok(Inspected { identity, facts, ..Inspected::default() })
}

fn atom_digests(body: &Value) -> Result<Vec<(String, Option<String>)>, NodeFailure> {
    let invalid = || failure(BundleStage::TamperedQuantity, "the structure records are malformed");
    let atoms = body["atoms"].as_array().ok_or_else(invalid)?;
    atoms
        .iter()
        .map(|atom| {
            let id = atom["id"].as_str().ok_or_else(invalid)?.to_owned();
            let digest = match &atom["source_digest"] {
                Value::Null => None,
                Value::String(text) => Some(text.clone()),
                _ => return Err(invalid()),
            };
            Ok((id, digest))
        })
        .collect()
}

// One arm per stored result kind: splitting them would only scatter the shared tail.
#[allow(clippy::too_many_lines)]
fn inspect_result(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::DecisionResult;
    let io = |e: &IoError| io_failure(kind, e);
    let label = container_label(bytes, MAX_DECISION_ARTIFACT_BYTES).map_err(|e| io(&e))?;
    let body: Value = first_section(bytes, MAX_DECISION_ARTIFACT_BYTES).map_err(|e| io(&e))?;
    let mut inspected =
        Inspected { contract_required: true, bound_to_contract: true, ..Inspected::default() };
    match label.as_str() {
        KIND_RESULT => {
            let contract = body_str(&body, &["contract_identity"]);
            let source = body_str(&body, &["source_digest"]);
            let artifact = DecisionResultArtifact::from_bytes(bytes, &contract, &source)
                .map_err(|e| io(&e))?;
            let result = artifact.result();
            publish(
                &mut inspected.facts,
                FACT_SNAPSHOT,
                format!("{}|{}", result.source.snapshot_id, result.source.causal_contract_id),
            );
            for action in &result.actions {
                push_value(
                    &mut inspected.values,
                    format!("{}.expected_utility", action.id),
                    action.expected_utility,
                );
                push_value(&mut inspected.values, format!("{}.value", action.id), action.value);
            }
            if let Some(evpi) = result.evpi {
                push_value(&mut inspected.values, "evpi".to_owned(), evpi);
            }
            inspected.contract = Some(contract);
            inspected.sources = vec![source];
            inspected.sources_must_be_upstream = true;
        }
        KIND_ROBUST => {
            let contract = body_str(&body, &["result", "contract_identity"]);
            let atoms = atom_digests(&body)?;
            let artifact =
                RobustResultArtifact::from_bytes(bytes, &contract, &atoms).map_err(|e| io(&e))?;
            for action in &artifact.result().actions {
                if let Some((low, high)) = action.range {
                    push_value(&mut inspected.values, format!("{}.range_low", action.id), low);
                    push_value(&mut inspected.values, format!("{}.range_high", action.id), high);
                }
            }
            push_value(
                &mut inspected.values,
                "native_verified".to_owned(),
                f64::from(u8::from(artifact.native_verified())),
            );
            let receipts = artifact.receipts();
            if !receipts.is_empty() {
                let all_verified = receipts
                    .iter()
                    .all(|r| matches!(r.trust, ExternalTrustLimit::VerifiedExtension));
                publish(
                    &mut inspected.facts,
                    FACT_TRUST,
                    if all_verified { TRUST_VERIFIED } else { TRUST_ATTESTED },
                );
            }
            inspected.receipts = receipts
                .iter()
                .map(|r| ReceiptInfo {
                    provider_id: r.provider_id.clone(),
                    snapshot_id: r.snapshot_id.clone(),
                    request: r.request_fingerprint.clone(),
                    verified: matches!(r.trust, ExternalTrustLimit::VerifiedExtension),
                })
                .collect();
            inspected.contract = Some(contract);
            inspected.sources = atoms.into_iter().filter_map(|(_, digest)| digest).collect();
        }
        KIND_STRUCTURAL => {
            let contract = body_str(&body, &["contract_identity"]);
            let atoms = atom_digests(&body)?;
            let artifact = StructuralResultArtifact::from_bytes(bytes, &contract, &atoms)
                .map_err(|e| io(&e))?;
            for action in &artifact.result().actions {
                if let Some(weighted) = action.weighted_value {
                    push_value(
                        &mut inspected.values,
                        format!("{}.weighted_value", action.id),
                        weighted,
                    );
                }
                if let Some((low, high)) = action.range {
                    push_value(&mut inspected.values, format!("{}.range_low", action.id), low);
                    push_value(&mut inspected.values, format!("{}.range_high", action.id), high);
                }
            }
            inspected.contract = Some(contract);
            inspected.sources = atoms.into_iter().filter_map(|(_, digest)| digest).collect();
        }
        other => {
            return Err(failure(
                BundleStage::TamperedQuantity,
                format!("`{other}` is not a decision result artifact"),
            ));
        }
    }
    inspected.identity =
        content_identity(&label, bytes, MAX_DECISION_ARTIFACT_BYTES).map_err(|e| io(&e))?;
    Ok(inspected)
}

fn inspect_sensitivity(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::Sensitivity;
    if bytes.len() > MAX_SENSITIVITY_ARTIFACT_BYTES {
        return Err(failure(BundleStage::Oversized, "the sensitivity artifact exceeds its bound"));
    }
    let artifact =
        SensitivityArtifact::from_bytes(bytes, None).map_err(|e| io_failure(kind, &e))?;
    let provenance = artifact.provenance();
    let mut facts = BTreeMap::new();
    publish(
        &mut facts,
        FACT_SNAPSHOT,
        format!("{}|{}", provenance.provider_snapshot, provenance.causal_contract_id),
    );
    let wires: Vec<ScientificQuantityWire> =
        artifact.quantities().iter().map(|q| q.quantity.clone()).collect();
    publish_coordinates(&mut facts, &coordinate_ids(&wires), false);
    let mut values = Vec::new();
    push_value(&mut values, "grid_points".to_owned(), artifact.grid().len() as f64);
    Ok(Inspected {
        identity: artifact.identity().digest.clone(),
        facts,
        values,
        ..Inspected::default()
    })
}

fn inspect_ranking(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    if bytes.len() > MAX_DESIGN_RANKING_ARTIFACT_BYTES {
        return Err(failure(BundleStage::Oversized, "the ranking artifact exceeds its bound"));
    }
    let consumed = consume_ranking(bytes, &ConsumeExpectation::default()).map_err(|e| {
        let stage = match &e {
            DesignRankingArtifactError::UnsupportedVersion(_) => BundleStage::IncompatibleVersion,
            DesignRankingArtifactError::Bounds(_) => BundleStage::Oversized,
            _ => BundleStage::TamperedQuantity,
        };
        failure(stage, e.to_string())
    })?;
    let mut facts = BTreeMap::new();
    let trust =
        if consumed.candidates.iter().any(|c| c.trust == SignalTrustLabel::ExternallyAttested) {
            TRUST_ATTESTED
        } else if consumed.candidates.iter().all(|c| c.trust == SignalTrustLabel::NativeLicensed) {
            TRUST_NATIVE
        } else {
            TRUST_VERIFIED
        };
    publish(&mut facts, FACT_TRUST, trust);
    let mut values = Vec::new();
    for entry in &consumed.ranking.entries {
        push_value(&mut values, format!("{}.evsi", entry.semantic_id), entry.evsi);
        if let Some(net) = entry.net_value {
            push_value(&mut values, format!("{}.net_value", entry.semantic_id), net);
        }
        push_value(&mut values, format!("{}.rank", entry.semantic_id), entry.rank as f64);
    }
    Ok(Inspected {
        identity: consumed.identity.clone(),
        facts,
        values,
        contract: consumed.ranking.decision_contract_identity.clone(),
        sources: consumed.ranking.source_digests.clone(),
        ..Inspected::default()
    })
}

fn inspect_inverse(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::InverseQuery;
    if bytes.len() > MAX_INVERSE_QUERY_ARTIFACT_BYTES {
        return Err(failure(BundleStage::Oversized, "the inverse query exceeds its bound"));
    }
    let artifact =
        InverseQueryArtifact::from_bytes(bytes, None).map_err(|e| io_failure(kind, &e))?;
    let mut facts = BTreeMap::new();
    publish_coordinates(&mut facts, &contract_coordinates(&artifact.query().contract), false);
    let mut values = Vec::new();
    push_value(
        &mut values,
        "feasible_actions".to_owned(),
        artifact.result().feasible_actions.len() as f64,
    );
    Ok(Inspected {
        identity: artifact.identity().digest.clone(),
        facts,
        values,
        contract: Some(artifact.result().contract_identity.clone()),
        ..Inspected::default()
    })
}

fn repair_failure(error: &RepairArtifactError) -> NodeFailure {
    let stage = match error.detail {
        "repair_artifact.unsupported_version" => BundleStage::IncompatibleVersion,
        "repair_artifact.bounds_exceeded" => BundleStage::Oversized,
        "repair_artifact.budget" => BundleStage::CallbackUnavailable,
        _ => BundleStage::SwappedEvidence,
    };
    failure(stage, error.to_string())
}

fn inspect_repair(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let artifact = RepairReportArtifact::from_bytes(bytes).map_err(|e| repair_failure(&e))?;
    let ctx = ExecutionContext::production(1, 1);
    artifact.consume(RepairConsumeLimits::default(), &ctx).map_err(|e| repair_failure(&e))?;
    let mut values = Vec::new();
    push_value(&mut values, "outcomes".to_owned(), artifact.report.outcomes.len() as f64);
    push_value(
        &mut values,
        "sufficient".to_owned(),
        artifact.report.ranked_sufficient.len() as f64,
    );
    Ok(Inspected { identity: artifact.report_digest.clone(), values, ..Inspected::default() })
}

/// The bytes of an evidence-relationship node: how the evidence behind the
/// bundle nodes `left` and `right` depends on each other. Connect both nodes to the
/// relationship node so the verifier can check the declaration against them.
#[must_use]
pub fn evidence_relationship_bytes(
    relationship: EvidenceRelationship,
    left: &str,
    right: &str,
) -> Vec<u8> {
    format!(
        "{RELATIONSHIP_MAGIC}\nrelationship={}\nleft={left}\nright={right}\n",
        relationship.as_str()
    )
    .into_bytes()
}

fn inspect_relationship(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let invalid = |why: &str| failure(BundleStage::SwappedEvidence, why.to_owned());
    if bytes.len() > 4096 {
        return Err(failure(BundleStage::Oversized, "an evidence relationship is a short record"));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("the relationship is not text"))?;
    let mut lines = text.lines();
    if lines.next() != Some(RELATIONSHIP_MAGIC) {
        return Err(invalid("not an evidence relationship record"));
    }
    let field = |lines: &mut std::str::Lines<'_>, key: &str| -> Result<String, NodeFailure> {
        lines
            .next()
            .and_then(|line| line.strip_prefix(key))
            .and_then(|rest| rest.strip_prefix('='))
            .map(str::to_owned)
            .ok_or_else(|| invalid("a relationship field is missing"))
    };
    let name = field(&mut lines, "relationship")?;
    let left = field(&mut lines, "left")?;
    let right = field(&mut lines, "right")?;
    let relationship = EvidenceRelationship::from_name(&name)
        .ok_or_else(|| invalid("the relationship is not a known dependence"))?;
    if left.is_empty() || right.is_empty() || left == right || lines.next().is_some() {
        return Err(invalid("a relationship names two distinct nodes"));
    }
    Ok(Inspected {
        identity: digest_parts(&["evidence_relationship", &name, &left, &right]),
        relationship: Some((relationship, left, right)),
        ..Inspected::default()
    })
}

// Historical work and frozen scores are separate independently validated objects.
// Neither reports a joint outcome law or authenticates a native provider.
fn inspect_recalculation_receipt(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    use antecedent_io::recalc_receipt_artifact::RecalcReceiptArtifact;
    let artifact = RecalcReceiptArtifact::from_bytes(bytes, None)
        .map_err(|e| failure(BundleStage::SwappedEvidence, e.to_string()))?;
    let meta = artifact.meta();
    let mut facts = BTreeMap::from([
        ("execution_status".into(), "historical_receipt".into()),
        (FACT_TRUST.into(), TRUST_UNVERIFIED.into()),
    ]);
    if let Some(stage) = meta.requested.iter().find(|s| s.stage == "data_snapshot") {
        facts.insert(FACT_SNAPSHOT.into(), stage.own.clone());
    }
    if let Some(stage) =
        meta.entries.iter().find(|s| s.stage == "score_artifact" && s.tag != "refused")
    {
        facts.insert("recalc_score_identity".into(), stage.identity.clone());
    }
    Ok(Inspected {
        identity: artifact.receipt_identity().to_owned(),
        facts,
        ..Inspected::default()
    })
}
fn inspect_frozen_scores(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    use antecedent_io::frozen_scores_artifact::FrozenScoreTable;
    let artifact = FrozenScoreTable::from_bytes(bytes, None)
        .map_err(|e| failure(BundleStage::SwappedEvidence, e.to_string()))?;
    Ok(Inspected {
        identity: artifact.identity().to_owned(),
        facts: BTreeMap::from([
            (FACT_SNAPSHOT.into(), artifact.meta().snapshot_digest.clone()),
            ("recalc_score_identity".into(), artifact.meta().fit_identity.clone()),
            ("reusable_state".into(), "frozen_same_row_scores".into()),
            (FACT_TRUST.into(), TRUST_UNVERIFIED.into()),
        ]),
        ..Inspected::default()
    })
}

fn inspect(kind: NodeKind, bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    match kind {
        NodeKind::Distribution => inspect_distribution(bytes),
        NodeKind::ExternalClaim => inspect_external_claim(bytes),
        NodeKind::DecisionContract => inspect_contract(bytes),
        NodeKind::DecisionResult => inspect_result(bytes),
        NodeKind::Sensitivity => inspect_sensitivity(bytes),
        NodeKind::StudyRanking => inspect_ranking(bytes),
        NodeKind::InverseQuery => inspect_inverse(bytes),
        NodeKind::RepairReport => inspect_repair(bytes),
        NodeKind::EvidenceRelationship => inspect_relationship(bytes),
        NodeKind::RecalculationReceipt => inspect_recalculation_receipt(bytes),
        NodeKind::FrozenScores => inspect_frozen_scores(bytes),
        other => Err(failure(
            BundleStage::UnknownNodeKind,
            format!("no verifier is registered for kind `{}`", other.as_str()),
        )),
    }
}

// ---------------------------------------------------------------------------
// Cross-checks against the upstream nodes
// ---------------------------------------------------------------------------

fn source_digest_of<'a>(node: &UpstreamNode<'a>) -> Option<&'a str> {
    node.facts
        .get(FACT_SOURCE_DIGEST)
        .or_else(|| node.facts.get(FACT_MEAN_SOURCE))
        .map(String::as_str)
}

fn covered(contract_coordinates: &str, sources: &[&UpstreamNode<'_>]) -> bool {
    let mut carried: Vec<&str> = Vec::new();
    for source in sources {
        if let Some(list) = source.facts.get(FACT_COORDINATES) {
            carried.extend(list.split(',').filter(|c| !c.is_empty()));
        }
    }
    contract_coordinates.split(',').filter(|c| !c.is_empty()).all(|c| carried.contains(&c))
}

fn check_relationship(
    inspected: &Inspected,
    upstream: &[UpstreamNode<'_>],
) -> Result<(), NodeFailure> {
    let Some((relationship, left, right)) = &inspected.relationship else {
        return Ok(());
    };
    let find = |id: &str| upstream.iter().find(|u| u.id == id);
    let (Some(a), Some(b)) = (find(left), find(right)) else {
        return Err(failure(
            BundleStage::GraphOrSnapshotMismatch,
            "a node the relationship names is not verified upstream of it",
        ));
    };
    if *relationship == EvidenceRelationship::Independent {
        if let (Some(x), Some(y)) =
            (a.facts.get(FACT_EVIDENCE_DIGEST), b.facts.get(FACT_EVIDENCE_DIGEST))
        {
            if x == y {
                return Err(failure(
                    BundleStage::SwappedEvidence,
                    "the nodes are declared independent but rest on the same evidence",
                ));
            }
        }
    }
    Ok(())
}

fn check_receipts(inspected: &Inspected, upstream: &[UpstreamNode<'_>]) -> Result<(), NodeFailure> {
    for receipt in &inspected.receipts {
        let prefix = format!("{}|{}|", receipt.provider_id, receipt.snapshot_id);
        for claim in upstream.iter().filter(|u| u.kind == NodeKind::ExternalClaim) {
            let Some(request) = claim
                .facts
                .get(FACT_PROVIDER_BINDING)
                .and_then(|binding| binding.strip_prefix(prefix.as_str()))
            else {
                continue;
            };
            if request != receipt.request {
                return Err(failure(
                    BundleStage::ProviderRequestChanged,
                    format!(
                        "the receipt answered request `{}` but upstream node `{}` answered `{request}`",
                        receipt.request, claim.id
                    ),
                ));
            }
            if receipt.verified
                && claim.facts.get(FACT_TRUST).map(String::as_str) != Some(TRUST_VERIFIED)
            {
                return Err(failure(
                    BundleStage::SwappedEvidence,
                    format!(
                        "the receipt claims verification that node `{}` does not retain",
                        claim.id
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn cross_check(
    inspected: &mut Inspected,
    upstream: &[UpstreamNode<'_>],
) -> Result<(), NodeFailure> {
    if let Some(expected) = inspected.facts.get("recalc_score_identity") {
        for source in upstream {
            if let Some(actual) = source.facts.get("recalc_score_identity") {
                if expected != actual {
                    return Err(failure(
                        BundleStage::SwappedEvidence,
                        "frozen scores and execution receipt name different executed fits",
                    ));
                }
            }
        }
    }
    check_relationship(inspected, upstream)?;
    check_receipts(inspected, upstream)?;
    let contracts: Vec<&UpstreamNode<'_>> =
        upstream.iter().filter(|u| u.kind == NodeKind::DecisionContract).collect();
    let matched_contract = match &inspected.contract {
        Some(identity) => {
            let found = contracts.iter().copied().find(|c| c.identity == identity.as_str());
            if found.is_none() && (inspected.contract_required || !contracts.is_empty()) {
                return Err(failure(
                    BundleStage::GraphOrSnapshotMismatch,
                    "the artifact is bound to a decision contract that is not upstream of it",
                ));
            }
            found
        }
        None => None,
    };
    let sources: Vec<&UpstreamNode<'_>> = upstream
        .iter()
        .filter(|u| matches!(u.kind, NodeKind::Distribution | NodeKind::ExternalClaim))
        .collect();
    let mut matched: Vec<&UpstreamNode<'_>> = Vec::new();
    if inspected.sources_must_be_upstream {
        for digest in &inspected.sources {
            let Some(found) =
                sources.iter().copied().find(|s| source_digest_of(s) == Some(digest.as_str()))
            else {
                return Err(failure(
                    BundleStage::TamperedQuantity,
                    "the artifact rests on draws or means that are not upstream of it",
                ));
            };
            matched.push(found);
        }
    } else {
        for source in sources.iter().copied().filter(|s| s.kind == NodeKind::Distribution) {
            if let Some(digest) = source_digest_of(source) {
                if !inspected.sources.iter().any(|own| own == digest) {
                    return Err(failure(
                        BundleStage::TamperedQuantity,
                        format!(
                            "the artifact was not computed from the draws of upstream node `{}`",
                            source.id
                        ),
                    ));
                }
                matched.push(source);
            }
        }
    }
    if let (true, Some(contract)) = (inspected.bound_to_contract, matched_contract) {
        for key in [FACT_REQUIRES_LAW, FACT_REQUIRES_DISTRIBUTION] {
            if let Some(value) = contract.facts.get(key) {
                publish(&mut inspected.facts, key, value.clone());
            }
        }
        if let Some(own) = contract.facts.get(FACT_COORDINATES) {
            if !matched.is_empty() && !covered(own, &matched) {
                return Err(failure(
                    BundleStage::TamperedQuantity,
                    "an action reads a coordinate its sources do not carry",
                ));
            }
        }
        if contract.facts.contains_key(FACT_REQUIRES_DISTRIBUTION) {
            if let Some(mean_only) = matched
                .iter()
                .find(|s| s.facts.get(FACT_LAW).map(String::as_str) == Some(LAW_MEAN_ONLY))
            {
                return Err(failure(
                    BundleStage::UnsupportedLaw,
                    format!(
                        "the decision needs a distribution but upstream node `{}` supplies a mean only",
                        mean_only.id
                    ),
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// What a bundle builder reads from an artifact's bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct ArtifactDescription {
    /// The node kind the artifact fills.
    pub kind: NodeKind,
    /// The identity the artifact's own consumer recomputed.
    pub identity: String,
    /// The facts the artifact publishes.
    pub facts: BTreeMap<String, String>,
    /// Named numbers read from the decoded artifact.
    pub values: Vec<(String, f64)>,
}

/// Decode an artifact on its own through its own consumer and report its
/// identity and published facts, without any upstream node. A decision result is
/// checked against the contract and source digests it states; the bundle
/// verifier then requires those to be the nodes upstream of it.
///
/// # Errors
/// An artifact its own consumer refuses, or a kind without a verifier.
pub fn describe_artifact(kind: NodeKind, bytes: &[u8]) -> Result<ArtifactDescription, NodeFailure> {
    let inspected = inspect(kind, bytes)?;
    Ok(ArtifactDescription {
        kind,
        identity: inspected.identity,
        facts: inspected.facts,
        values: inspected.values,
    })
}

/// The node kind a container's artifact fills, read from its manifest kind label.
#[must_use]
pub fn detect_node_kind(bytes: &[u8]) -> Option<NodeKind> {
    // A relationship is a short text record, not a sectioned container.
    if bytes.starts_with(RELATIONSHIP_MAGIC.as_bytes()) {
        return Some(NodeKind::EvidenceRelationship);
    }
    let label = container_label(bytes, MAX_SNIFF_BYTES).ok()?;
    match label.as_str() {
        KIND_DISTRIBUTION => Some(NodeKind::Distribution),
        KIND_EXTERNAL_CLAIM => Some(NodeKind::ExternalClaim),
        KIND_CONTRACT | KIND_ADMISSIBLE => Some(NodeKind::DecisionContract),
        KIND_RESULT | KIND_ROBUST | KIND_STRUCTURAL => Some(NodeKind::DecisionResult),
        KIND_SENSITIVITY => Some(NodeKind::Sensitivity),
        DESIGN_RANKING_ARTIFACT_KIND => Some(NodeKind::StudyRanking),
        INVERSE_QUERY_ARTIFACT_FEATURE => Some(NodeKind::InverseQuery),
        REPAIR_ARTIFACT_KIND => Some(NodeKind::RepairReport),
        antecedent_io::recalc_receipt_artifact::RECALC_RECEIPT_ARTIFACT_FEATURE => {
            Some(NodeKind::RecalculationReceipt)
        }
        antecedent_io::frozen_scores_artifact::FROZEN_SCORES_ARTIFACT_FEATURE => {
            Some(NodeKind::FrozenScores)
        }
        _ => None,
    }
}

fn verify_kind(
    kind: NodeKind,
    bytes: &[u8],
    upstream: &[UpstreamNode<'_>],
) -> Result<VerifiedNode, NodeFailure> {
    let mut inspected = inspect(kind, bytes)?;
    cross_check(&mut inspected, upstream)?;
    Ok(VerifiedNode {
        identity: inspected.identity,
        facts: inspected.facts,
        values: inspected.values,
    })
}

macro_rules! kind_verifier {
    ($(#[$doc:meta])* $name:ident, $kind:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl NodeVerifier for $name {
            fn kind(&self) -> NodeKind {
                $kind
            }

            fn verify(
                &self,
                _node_id: &str,
                bytes: &[u8],
                upstream: &[UpstreamNode<'_>],
            ) -> Result<VerifiedNode, NodeFailure> {
                verify_kind($kind, bytes, upstream)
            }
        }
    };
}

kind_verifier!(
    /// Verifies a `joint_distribution_v1` artifact and publishes its law
    /// (`joint_draw` only for aligned joint draws), coordinates, snapshot and trust.
    DistributionVerifier,
    NodeKind::Distribution
);
kind_verifier!(
    /// Verifies an `external_response_claim_v1` artifact: `mean_only`, trust
    /// `externally_attested` unless the verification receipt is retained, its exact
    /// request, evidence digest and snapshot.
    ExternalClaimVerifier,
    NodeKind::ExternalClaim
);
kind_verifier!(
    /// Verifies a decision contract or an admissible decision contract and
    /// publishes the coordinates its actions read and what its functional needs of a
    /// source.
    DecisionContractVerifier,
    NodeKind::DecisionContract
);
kind_verifier!(
    /// Verifies a decision, robust decision or structural decision result against
    /// the contract and the sources upstream of it.
    DecisionResultVerifier,
    NodeKind::DecisionResult
);
kind_verifier!(
    /// Verifies a `sensitivity_decision_v1` artifact by recomputation.
    SensitivityVerifier,
    NodeKind::Sensitivity
);
kind_verifier!(
    /// Verifies a `design_ranking_v1` artifact by replaying every candidate.
    StudyRankingVerifier,
    NodeKind::StudyRanking
);
kind_verifier!(
    /// Verifies an inverse functional query by re-evaluating it.
    InverseQueryVerifier,
    NodeKind::InverseQuery
);
kind_verifier!(
    /// Verifies a repair search receipt by replaying the whole search.
    RepairReportVerifier,
    NodeKind::RepairReport
);
kind_verifier!(
    /// Verifies an evidence-relationship record against the nodes it names.
    EvidenceRelationshipVerifier,
    NodeKind::EvidenceRelationship
);

kind_verifier!(
    /// Verifies a historical selective receipt without supplying executable state.
    RecalculationReceiptVerifier,
    NodeKind::RecalculationReceipt
);
kind_verifier!(
    /// Verifies frozen same-row scores and their producing snapshot/fit bindings.
    FrozenScoresVerifier,
    NodeKind::FrozenScores
);

/// The verifier of one node kind, when it has one.
#[must_use]
pub fn verifier_for(kind: NodeKind) -> Option<Box<dyn NodeVerifier>> {
    let verifier: Box<dyn NodeVerifier> = match kind {
        NodeKind::Distribution => Box::new(DistributionVerifier),
        NodeKind::ExternalClaim => Box::new(ExternalClaimVerifier),
        NodeKind::DecisionContract => Box::new(DecisionContractVerifier),
        NodeKind::DecisionResult => Box::new(DecisionResultVerifier),
        NodeKind::Sensitivity => Box::new(SensitivityVerifier),
        NodeKind::StudyRanking => Box::new(StudyRankingVerifier),
        NodeKind::InverseQuery => Box::new(InverseQueryVerifier),
        NodeKind::RepairReport => Box::new(RepairReportVerifier),
        NodeKind::EvidenceRelationship => Box::new(EvidenceRelationshipVerifier),
        NodeKind::RecalculationReceipt => Box::new(RecalculationReceiptVerifier),
        NodeKind::FrozenScores => Box::new(FrozenScoresVerifier),
        _ => return None,
    };
    Some(verifier)
}

/// A consumer with a verifier registered for every kind in [`VERIFIABLE_KINDS`].
/// A kind without one (a causal contract, an attestation, quantity coordinates, a
/// transformation) fails as `unknown_node_kind` when embedded; it is carried as a
/// reference node instead.
#[must_use]
pub fn standard_consumer() -> BundleConsumer {
    VERIFIABLE_KINDS
        .into_iter()
        .filter_map(verifier_for)
        .fold(BundleConsumer::new(), BundleConsumer::register)
}
