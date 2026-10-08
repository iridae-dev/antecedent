//! Portable composed result: the `composition_bundle_v1` artifact.
//!
//! A [`CompositionBundle`] is a typed directed graph. Its nodes are the parts of a
//! composed decision (causal contract, execution or fit, distribution artifact,
//! external claim and attestation, evidence-relationship declaration, quantity
//! coordinates, support/trust/calibration, transformation, decision contract and
//! result, sensitivity artifact, study ranking). Its edges are dependencies, and
//! each edge carries the Merkle digest of the upstream node, so a changed upstream
//! identity changes the digest of every dependent node and of the bundle.
//!
//! Every node is either [`NodeSource::Embedded`] (an artifact the consumer decodes
//! and verifies itself) or [`NodeSource::Reference`] (it needs a supplied provider
//! or data source, and is never claimed to have been replayed).
//!
//! # Crate placement
//!
//! The bundle lives in `antecedent-design`: `antecedent-io` sits below it and
//! cannot decode the decision, robust-result or study-ranking artifacts, while the
//! facade crate is the wrong layer for a wire format. The bundle itself does not
//! hard-depend on any artifact: embedded nodes are decoded through the
//! [`NodeVerifier`] implementations registered per [`NodeKind`] in a
//! [`BundleConsumer`], so an artifact crate or a host registers exactly the
//! consumers it trusts.
//!
//! # Verification order
//!
//! Version, size and container checks come first, then node kinds, then the
//! recomputation of every node and edge digest (a mismatch is
//! [`BundleStage::EdgeDigestMismatch`]), then the consumer's retained identity
//! ([`BundleStage::ExpectedIdentityMismatch`]). Only then are nodes verified; a
//! node-level failure is reported on that node and the rest of the bundle keeps its
//! inspected result.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use antecedent_core::{
    CompositionLink, CompositionStage, ExternalRefusal, ProvenanceChain, ProvenanceChainError,
};
use antecedent_io::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::error::IoError;
use antecedent_io::reader::ArtifactReader;
use antecedent_io::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};
use serde::{Deserialize, Serialize};

/// Artifact kind label of a composition bundle.
pub const BUNDLE_KIND: &str = "composition_bundle_v1";
/// Body format version this module reads and writes.
pub const BUNDLE_VERSION: u16 = 1;
const BODY_SECTION: &str = "bundle.body";

/// Fact key: the law a node supplies or a result rests on.
pub const FACT_LAW: &str = "law";
/// Fact value of [`FACT_LAW`]: aligned joint draws.
pub const LAW_JOINT: &str = "joint_draw";
/// Fact value of [`FACT_LAW`]: an external mean only.
pub const LAW_MEAN_ONLY: &str = "mean_only";
/// Fact key: the law a dependent node requires of its upstream nodes.
pub const FACT_REQUIRES_LAW: &str = "requires_law";
/// Fact key: digest of the quantity coordinates a node is stated over.
pub const FACT_QUANTITY_DIGEST: &str = "quantity_digest";
/// Fact key: digest of the evidence a node was assessed under.
pub const FACT_EVIDENCE_DIGEST: &str = "evidence_digest";
/// Fact key: the graph or data snapshot a node is bound to.
pub const FACT_SNAPSHOT: &str = "graph_or_snapshot";
/// Fact key: the exact provider request fingerprint a node answered.
pub const FACT_REQUEST: &str = "request_fingerprint";
/// Fact key: how far a value can be trusted, as declared.
pub const FACT_TRUST: &str = "trust";

/// Facts that must agree across an edge when both ends state them, with the
/// stage a disagreement fails at.
const BINDING_FACTS: [(&str, BundleStage); 4] = [
    (FACT_QUANTITY_DIGEST, BundleStage::TamperedQuantity),
    (FACT_EVIDENCE_DIGEST, BundleStage::SwappedEvidence),
    (FACT_SNAPSHOT, BundleStage::GraphOrSnapshotMismatch),
    (FACT_REQUEST, BundleStage::ProviderRequestChanged),
];

// ---------------------------------------------------------------------------
// Stages and errors
// ---------------------------------------------------------------------------

/// The structured stage a bundle refusal or node failure carries, under the
/// `composition_bundle` namespace.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BundleStage {
    /// A quantity coordinate differs between connected nodes.
    TamperedQuantity,
    /// Evidence differs from the evidence the bundle declares.
    SwappedEvidence,
    /// A graph or data snapshot differs between connected nodes.
    GraphOrSnapshotMismatch,
    /// A node requires a law its upstream nodes do not supply.
    UnsupportedLaw,
    /// A supplied provider answered a different request.
    ProviderRequestChanged,
    /// A needed provider callback or data source was not supplied.
    CallbackUnavailable,
    /// The body format version is not supported.
    IncompatibleVersion,
    /// The bundle exceeds a size or count bound.
    Oversized,
    /// A node kind is unknown or has no registered verifier.
    UnknownNodeKind,
    /// A node or edge digest differs from its recomputation, or the graph is
    /// malformed.
    EdgeDigestMismatch,
    /// The bundle identity differs from the consumer's retained identity.
    ExpectedIdentityMismatch,
}

impl BundleStage {
    /// Every stage.
    pub const ALL: [Self; 11] = [
        Self::TamperedQuantity,
        Self::SwappedEvidence,
        Self::GraphOrSnapshotMismatch,
        Self::UnsupportedLaw,
        Self::ProviderRequestChanged,
        Self::CallbackUnavailable,
        Self::IncompatibleVersion,
        Self::Oversized,
        Self::UnknownNodeKind,
        Self::EdgeDigestMismatch,
        Self::ExpectedIdentityMismatch,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TamperedQuantity => "tampered_quantity",
            Self::SwappedEvidence => "swapped_evidence",
            Self::GraphOrSnapshotMismatch => "graph_or_snapshot_mismatch",
            Self::UnsupportedLaw => "unsupported_law",
            Self::ProviderRequestChanged => "provider_request_changed",
            Self::CallbackUnavailable => "callback_unavailable",
            Self::IncompatibleVersion => "incompatible_version",
            Self::Oversized => "oversized",
            Self::UnknownNodeKind => "unknown_node_kind",
            Self::EdgeDigestMismatch => "edge_digest_mismatch",
            Self::ExpectedIdentityMismatch => "expected_identity_mismatch",
        }
    }

    /// Namespaced refusal detail, `composition_bundle.<stage>`.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::TamperedQuantity => "composition_bundle.tampered_quantity",
            Self::SwappedEvidence => "composition_bundle.swapped_evidence",
            Self::GraphOrSnapshotMismatch => "composition_bundle.graph_or_snapshot_mismatch",
            Self::UnsupportedLaw => "composition_bundle.unsupported_law",
            Self::ProviderRequestChanged => "composition_bundle.provider_request_changed",
            Self::CallbackUnavailable => "composition_bundle.callback_unavailable",
            Self::IncompatibleVersion => "composition_bundle.incompatible_version",
            Self::Oversized => "composition_bundle.oversized",
            Self::UnknownNodeKind => "composition_bundle.unknown_node_kind",
            Self::EdgeDigestMismatch => "composition_bundle.edge_digest_mismatch",
            Self::ExpectedIdentityMismatch => "composition_bundle.expected_identity_mismatch",
        }
    }

    /// Registered reason code the stage refuses under.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::TamperedQuantity => antecedent_core::reason_code!("quantity_semantics_mismatch"),
            Self::UnsupportedLaw => antecedent_core::reason_code!("joint_law_required"),
            Self::CallbackUnavailable => {
                antecedent_core::reason_code!("external_capability_missing")
            }
            Self::IncompatibleVersion => antecedent_core::reason_code!("schema_mismatch"),
            Self::Oversized | Self::UnknownNodeKind => {
                antecedent_core::reason_code!("invalid_argument")
            }
            Self::SwappedEvidence
            | Self::GraphOrSnapshotMismatch
            | Self::ProviderRequestChanged
            | Self::EdgeDigestMismatch
            | Self::ExpectedIdentityMismatch => {
                antecedent_core::reason_code!("external_binding_mismatch")
            }
        }
    }
}

/// Why a bundle could not be built or consumed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BundleError {
    /// A structured refusal at a named stage.
    Refused {
        /// Failed stage.
        stage: BundleStage,
        /// What was refused.
        reason: String,
        /// Offending node, when there is one.
        offending: Option<String>,
    },
    /// The container itself is corrupt or unreadable.
    Container(Box<IoError>),
}

impl BundleError {
    /// The failed stage, for a structured refusal.
    #[must_use]
    pub fn stage(&self) -> Option<BundleStage> {
        match self {
            Self::Refused { stage, .. } => Some(*stage),
            Self::Container(_) => None,
        }
    }

    /// Structured refusal with a `composition_bundle.<stage>` detail.
    #[must_use]
    pub fn to_refusal(&self) -> Option<ExternalRefusal> {
        match self {
            Self::Refused { stage, reason, offending } => Some(ExternalRefusal {
                code: stage.code(),
                stage: "bind",
                detail: stage.detail().to_owned(),
                offending: offending.clone(),
                expected: None,
                supplied: Some(reason.clone()),
                capability: None,
                remedy: None,
            }),
            Self::Container(_) => None,
        }
    }
}

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused { stage, reason, offending } => match offending {
                Some(node) => write!(f, "{}: {reason} (node `{node}`)", stage.detail()),
                None => write!(f, "{}: {reason}", stage.detail()),
            },
            Self::Container(error) => write!(f, "composition bundle container: {error}"),
        }
    }
}

impl std::error::Error for BundleError {}

impl From<IoError> for BundleError {
    fn from(error: IoError) -> Self {
        match error {
            IoError::TooLarge => {
                refused(BundleStage::Oversized, "the container exceeds a size bound", None)
            }
            other => Self::Container(Box::new(other)),
        }
    }
}

fn refused(stage: BundleStage, reason: &str, offending: Option<String>) -> BundleError {
    BundleError::Refused { stage, reason: reason.to_owned(), offending }
}

fn graph_error(reason: &str, offending: Option<String>) -> BundleError {
    refused(BundleStage::EdgeDigestMismatch, reason, offending)
}

fn chain_refusal(error: &ProvenanceChainError) -> BundleError {
    let offending = match error {
        ProvenanceChainError::DigestMismatch { link, .. } => Some(link.clone()),
        ProvenanceChainError::DuplicateLink(id) | ProvenanceChainError::UnknownLink(id) => {
            Some(id.clone())
        }
        ProvenanceChainError::UnresolvedParent { child, .. } => Some(child.clone()),
        ProvenanceChainError::InvalidLink | ProvenanceChainError::MissingStage(_) => None,
    };
    BundleError::Refused {
        stage: BundleStage::EdgeDigestMismatch,
        reason: format!("{error:?}"),
        offending,
    }
}

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/// What a node stands for.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum NodeKind {
    /// A causal contract or identified program.
    CausalContract,
    /// A native execution or a portable fit.
    ExecutionOrFit,
    /// An external claim.
    ExternalClaim,
    /// An attestation of an external claim.
    Attestation,
    /// An evidence-relationship declaration ([`EvidenceRelationship`]).
    EvidenceRelationship,
    /// Quantity coordinates.
    QuantityCoordinates,
    /// Support, trust and calibration statements.
    SupportTrustCalibration,
    /// A portable distribution artifact.
    Distribution,
    /// A named transformation.
    Transformation,
    /// A decision contract.
    DecisionContract,
    /// A decision result.
    DecisionResult,
    /// A sensitivity artifact.
    Sensitivity,
    /// A study ranking.
    StudyRanking,
    /// An inverse functional query with its forward evidence and result table.
    InverseQuery,
    /// An identification-repair search report.
    RepairReport,
    /// Independently checked historical selective execution receipt; no live state.
    RecalculationReceipt,
    /// Independently checked frozen same-row scores for their licensed retarget operation.
    FrozenScores,
}

impl NodeKind {
    /// Every kind.
    pub const ALL: [Self; 17] = [
        Self::CausalContract,
        Self::ExecutionOrFit,
        Self::ExternalClaim,
        Self::Attestation,
        Self::EvidenceRelationship,
        Self::QuantityCoordinates,
        Self::SupportTrustCalibration,
        Self::Distribution,
        Self::Transformation,
        Self::DecisionContract,
        Self::DecisionResult,
        Self::Sensitivity,
        Self::StudyRanking,
        Self::InverseQuery,
        Self::RepairReport,
        Self::RecalculationReceipt,
        Self::FrozenScores,
    ];

    /// Stable `snake_case` wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CausalContract => "causal_contract",
            Self::ExecutionOrFit => "execution_or_fit",
            Self::ExternalClaim => "external_claim",
            Self::Attestation => "attestation",
            Self::EvidenceRelationship => "evidence_relationship",
            Self::QuantityCoordinates => "quantity_coordinates",
            Self::SupportTrustCalibration => "support_trust_calibration",
            Self::Distribution => "distribution",
            Self::Transformation => "transformation",
            Self::DecisionContract => "decision_contract",
            Self::DecisionResult => "decision_result",
            Self::Sensitivity => "sensitivity",
            Self::StudyRanking => "study_ranking",
            Self::InverseQuery => "inverse_query",
            Self::RepairReport => "repair_report",
            Self::RecalculationReceipt => "recalculation_receipt",
            Self::FrozenScores => "frozen_scores",
        }
    }

    /// Parse the wire name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    /// The lineage stage the kind occupies in the bundle's Merkle chain.
    #[must_use]
    pub const fn stage(self) -> CompositionStage {
        match self {
            Self::CausalContract => CompositionStage::CausalContract,
            Self::ExecutionOrFit | Self::Distribution | Self::FrozenScores => {
                CompositionStage::DistributionArtifact
            }
            Self::ExternalClaim | Self::Attestation => CompositionStage::ExternalProvider,
            Self::EvidenceRelationship
            | Self::QuantityCoordinates
            | Self::SupportTrustCalibration
            | Self::RepairReport
            | Self::RecalculationReceipt => CompositionStage::Evidence,
            Self::Transformation => CompositionStage::Transformation,
            Self::DecisionContract => CompositionStage::DecisionContract,
            Self::DecisionResult | Self::InverseQuery => CompositionStage::Claim,
            Self::Sensitivity => CompositionStage::SensitivityInput,
            Self::StudyRanking => CompositionStage::StudyRankingProvider,
        }
    }

    /// Stage a recomputed identity that differs from the declared one fails at:
    /// a value-bearing artifact was altered, any other was swapped.
    const fn identity_mismatch_stage(self) -> BundleStage {
        match self {
            Self::QuantityCoordinates
            | Self::DecisionResult
            | Self::Sensitivity
            | Self::StudyRanking
            | Self::InverseQuery => BundleStage::TamperedQuantity,
            _ => BundleStage::SwappedEvidence,
        }
    }
}

/// How two pieces of evidence depend on each other (the C1 vocabulary, declared
/// locally until that module is shared).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EvidenceRelationship {
    /// Both rest on the same data.
    SharedData,
    /// Both rest on the same prior.
    SharedPrior,
    /// Both rest on the same fitted model.
    SharedFittedModel,
    /// The dependence is not known.
    UnknownDependence,
    /// The evidence is independent.
    Independent,
}

impl EvidenceRelationship {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SharedData => "shared_data",
            Self::SharedPrior => "shared_prior",
            Self::SharedFittedModel => "shared_fitted_model",
            Self::UnknownDependence => "unknown_dependence",
            Self::Independent => "independent",
        }
    }

    /// Parse the wire name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [
            Self::SharedData,
            Self::SharedPrior,
            Self::SharedFittedModel,
            Self::UnknownDependence,
            Self::Independent,
        ]
        .into_iter()
        .find(|relationship| relationship.as_str() == name)
    }
}

/// What a reference node needs the consumer to supply.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderOrData {
    /// A provider or callback at an exact request.
    Provider {
        /// Provider service identity.
        provider_id: String,
        /// Provider input or model snapshot.
        snapshot_id: String,
        /// Exact request fingerprint the provider must answer.
        request_fingerprint: String,
    },
    /// A data snapshot with the digest it must have.
    Data {
        /// Snapshot identity.
        snapshot_id: String,
        /// Expected content digest.
        digest: String,
    },
}

impl ProviderOrData {
    fn canonical(&self) -> String {
        match self {
            Self::Provider { provider_id, snapshot_id, request_fingerprint } => {
                format!("provider|{provider_id}|{snapshot_id}|{request_fingerprint}")
            }
            Self::Data { snapshot_id, digest } => format!("data|{snapshot_id}|{digest}"),
        }
    }
}

/// Where a node's content lives.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NodeSource {
    /// An artifact the consumer can decode and verify independently.
    Embedded {
        /// The artifact's container bytes.
        bytes: Vec<u8>,
    },
    /// An artifact held elsewhere: it needs a supplied provider or data source
    /// and is never claimed replayed.
    Reference {
        /// Identity of the referenced artifact.
        identity: String,
        /// What must be supplied to resolve it.
        requires: ProviderOrData,
    },
}

/// One node of the composed result.
#[derive(Clone, Debug, PartialEq)]
pub struct BundleNode {
    /// Identity within the bundle.
    pub id: String,
    /// What the node stands for.
    pub kind: NodeKind,
    /// Declared identity of the node's artifact (for an embedded node, the
    /// identity its verifier must recompute; for a reference, the referenced one).
    pub identity: String,
    /// Embedded bytes or an external reference.
    pub source: NodeSource,
    /// Declared facts of a reference node (embedded nodes publish theirs through
    /// their verifier). Unverified.
    pub facts: BTreeMap<String, String>,
    /// Declared, unverified values of a reference node, kept so an inspected
    /// result survives when execution is unavailable.
    pub inspected: Vec<(String, f64)>,
}

impl BundleNode {
    /// An embedded node.
    #[must_use]
    pub fn embedded(id: &str, kind: NodeKind, identity: &str, bytes: Vec<u8>) -> Self {
        Self {
            id: id.to_owned(),
            kind,
            identity: identity.to_owned(),
            source: NodeSource::Embedded { bytes },
            facts: BTreeMap::new(),
            inspected: Vec::new(),
        }
    }

    /// A reference node.
    #[must_use]
    pub fn reference(id: &str, kind: NodeKind, identity: &str, requires: ProviderOrData) -> Self {
        Self {
            id: id.to_owned(),
            kind,
            identity: identity.to_owned(),
            source: NodeSource::Reference { identity: identity.to_owned(), requires },
            facts: BTreeMap::new(),
            inspected: Vec::new(),
        }
    }

    /// Declare a fact on a reference node (ignored on an embedded node).
    #[must_use]
    pub fn with_fact(mut self, key: &str, value: &str) -> Self {
        if matches!(self.source, NodeSource::Reference { .. }) {
            self.facts.insert(key.to_owned(), value.to_owned());
        }
        self
    }

    /// Declare an inspected value on a reference node (ignored on an embedded node).
    #[must_use]
    pub fn with_inspected(mut self, key: &str, value: f64) -> Self {
        if matches!(self.source, NodeSource::Reference { .. }) {
            self.inspected.push((key.to_owned(), value));
        }
        self
    }
}

/// A dependency carrying the upstream node's Merkle digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleEdge {
    /// Upstream node.
    pub from: String,
    /// Dependent node.
    pub to: String,
    /// Merkle digest of the upstream node (lowercase hex).
    pub upstream_digest: String,
}

/// Bounds on a bundle, checked before any decode allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleLimits {
    /// Most container bytes.
    pub max_bundle_bytes: usize,
    /// Most nodes.
    pub max_nodes: usize,
    /// Most edges.
    pub max_edges: usize,
    /// Most bytes in one embedded artifact.
    pub max_embedded_bytes: usize,
}

impl Default for BundleLimits {
    fn default() -> Self {
        Self {
            max_bundle_bytes: 64 * 1024 * 1024,
            max_nodes: 1_024,
            max_edges: 8_192,
            max_embedded_bytes: 16 * 1024 * 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// Digests and graph derivation
// ---------------------------------------------------------------------------

fn put(hasher: &mut blake3::Hasher, text: &str) {
    hasher.update(&(text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
}

/// Digest of one node's own content (not of its upstream nodes).
fn node_digest(node: &BundleNode) -> String {
    let mut hasher = blake3::Hasher::new();
    put(&mut hasher, "antecedent.composition_bundle.node.v1");
    put(&mut hasher, &node.id);
    put(&mut hasher, node.kind.as_str());
    put(&mut hasher, &node.identity);
    match &node.source {
        NodeSource::Embedded { bytes } => {
            put(&mut hasher, "embedded");
            put(&mut hasher, blake3::hash(bytes).to_hex().as_str());
        }
        NodeSource::Reference { identity, requires } => {
            put(&mut hasher, "reference");
            put(&mut hasher, identity);
            put(&mut hasher, &requires.canonical());
            hasher.update(&(node.facts.len() as u64).to_le_bytes());
            for (key, value) in &node.facts {
                put(&mut hasher, key);
                put(&mut hasher, value);
            }
            hasher.update(&(node.inspected.len() as u64).to_le_bytes());
            for (key, value) in &node.inspected {
                put(&mut hasher, key);
                hasher.update(&value.to_bits().to_le_bytes());
            }
        }
    }
    hasher.finalize().to_hex().to_string()
}

struct Derived {
    order: Vec<usize>,
    parents: Vec<Vec<usize>>,
    digests: Vec<String>,
    edges: Vec<BundleEdge>,
}

fn validate_nodes(nodes: &[BundleNode]) -> Result<HashMap<&str, usize>, BundleError> {
    let mut index = HashMap::with_capacity(nodes.len());
    for (position, node) in nodes.iter().enumerate() {
        if node.id.trim().is_empty() || node.identity.trim().is_empty() {
            return Err(graph_error("a node has a blank id or identity", Some(node.id.clone())));
        }
        if index.insert(node.id.as_str(), position).is_some() {
            return Err(graph_error("a node id is repeated", Some(node.id.clone())));
        }
    }
    Ok(index)
}

type Adjacency = (Vec<Vec<usize>>, Vec<Vec<usize>>);

fn adjacency(
    index: &HashMap<&str, usize>,
    deps: &BTreeSet<(String, String)>,
    count: usize,
) -> Result<Adjacency, BundleError> {
    let mut parents = vec![Vec::new(); count];
    let mut children = vec![Vec::new(); count];
    for (from, to) in deps {
        let (Some(&f), Some(&t)) = (index.get(from.as_str()), index.get(to.as_str())) else {
            return Err(graph_error("an edge names an unknown node", Some(from.clone())));
        };
        if f == t {
            return Err(graph_error("a node depends on itself", Some(from.clone())));
        }
        parents[t].push(f);
        children[f].push(t);
    }
    Ok((parents, children))
}

fn topological(parents: &[Vec<usize>], children: &[Vec<usize>]) -> Result<Vec<usize>, BundleError> {
    let mut indegree: Vec<usize> = parents.iter().map(Vec::len).collect();
    let mut ready: BTreeSet<usize> = indegree
        .iter()
        .enumerate()
        .filter(|(_, degree)| **degree == 0)
        .map(|(position, _)| position)
        .collect();
    let mut order = Vec::with_capacity(parents.len());
    while let Some(next) = ready.pop_first() {
        order.push(next);
        for &child in &children[next] {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                ready.insert(child);
            }
        }
    }
    if order.len() != parents.len() {
        return Err(graph_error("the dependency graph has a cycle", None));
    }
    Ok(order)
}

/// Build the Merkle chain over the nodes in dependency order. With `declared`
/// edges, every dependent link states the digests it believes its parents have,
/// so a changed upstream refuses.
fn derive(
    nodes: &[BundleNode],
    deps: &BTreeSet<(String, String)>,
    declared: Option<&[BundleEdge]>,
) -> Result<Derived, BundleError> {
    let index = validate_nodes(nodes)?;
    let (parents, children) = adjacency(&index, deps, nodes.len())?;
    let order = topological(&parents, &children)?;
    let declared_digests: Option<HashMap<(&str, &str), &str>> = declared.map(|edges| {
        edges
            .iter()
            .map(|e| ((e.from.as_str(), e.to.as_str()), e.upstream_digest.as_str()))
            .collect()
    });
    if let Some(edges) = declared {
        if edges.len() != deps.len() {
            return Err(graph_error("an edge is repeated", None));
        }
    }
    let mut link_ids = vec![String::new(); nodes.len()];
    let mut links = Vec::with_capacity(nodes.len());
    for &position in &order {
        let node = &nodes[position];
        let link_id = format!("{}@{}", node.id, node_digest(node));
        let parent_links: Vec<String> =
            parents[position].iter().map(|p| link_ids[*p].clone()).collect();
        let declared_parent_digests = match &declared_digests {
            Some(map) => {
                let mut digests = Vec::with_capacity(parents[position].len());
                for parent in &parents[position] {
                    let key = (nodes[*parent].id.as_str(), node.id.as_str());
                    let Some(digest) = map.get(&key) else {
                        return Err(graph_error(
                            "an edge digest is missing",
                            Some(node.id.clone()),
                        ));
                    };
                    digests.push((*digest).to_owned());
                }
                Some(digests)
            }
            None => None,
        };
        link_ids[position].clone_from(&link_id);
        links.push(CompositionLink {
            id: link_id,
            stage: node.kind.stage(),
            parents: parent_links,
            declared_parent_digests,
        });
    }
    let chain = ProvenanceChain::new(links).map_err(|e| chain_refusal(&e))?;
    let mut digests = Vec::with_capacity(nodes.len());
    for link_id in &link_ids {
        digests.push(chain.digest_of(link_id).map_err(|e| chain_refusal(&e))?.to_owned());
    }
    let edges = deps
        .iter()
        .map(|(from, to)| {
            let position = index.get(from.as_str()).copied().unwrap_or(0);
            BundleEdge {
                from: from.clone(),
                to: to.clone(),
                upstream_digest: digests.get(position).cloned().unwrap_or_default(),
            }
        })
        .collect();
    Ok(Derived { order, parents, digests, edges })
}

fn bundle_identity(nodes: &[BundleNode], digests: &[String], edges: &[BundleEdge]) -> String {
    let mut hasher = blake3::Hasher::new();
    put(&mut hasher, "antecedent.composition_bundle.v1");
    hasher.update(&(nodes.len() as u64).to_le_bytes());
    for (node, digest) in nodes.iter().zip(digests) {
        put(&mut hasher, &node.id);
        put(&mut hasher, &node_digest(node));
        put(&mut hasher, digest);
    }
    hasher.update(&(edges.len() as u64).to_le_bytes());
    for edge in edges {
        put(&mut hasher, &edge.from);
        put(&mut hasher, &edge.to);
        put(&mut hasher, &edge.upstream_digest);
    }
    hasher.finalize().to_hex().to_string()
}

fn embedded_section(id: &str) -> String {
    format!("embedded.{id}")
}

// ---------------------------------------------------------------------------
// The bundle
// ---------------------------------------------------------------------------

/// A bounded, versioned, BLAKE3-identified composed result.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositionBundle {
    nodes: Vec<BundleNode>,
    edges: Vec<BundleEdge>,
    digests: Vec<String>,
    identity: String,
}

impl CompositionBundle {
    /// Build a bundle from nodes and dependencies `(upstream, dependent)`. Node
    /// and edge declaration order does not affect the identity.
    ///
    /// # Errors
    /// A bound exceeded ([`BundleStage::Oversized`]); a blank or repeated node, a
    /// repeated or dangling edge, a self-dependency or a cycle
    /// ([`BundleStage::EdgeDigestMismatch`]).
    pub fn new(
        mut nodes: Vec<BundleNode>,
        dependencies: &[(String, String)],
        limits: &BundleLimits,
    ) -> Result<Self, BundleError> {
        if nodes.len() > limits.max_nodes || dependencies.len() > limits.max_edges {
            return Err(refused(BundleStage::Oversized, "too many nodes or edges", None));
        }
        for node in &nodes {
            if let NodeSource::Embedded { bytes } = &node.source {
                if bytes.len() > limits.max_embedded_bytes {
                    return Err(refused(
                        BundleStage::Oversized,
                        "an embedded artifact exceeds its bound",
                        Some(node.id.clone()),
                    ));
                }
            }
        }
        for node in &mut nodes {
            if matches!(node.source, NodeSource::Embedded { .. }) {
                node.facts.clear();
                node.inspected.clear();
            }
        }
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        let deps: BTreeSet<(String, String)> = dependencies.iter().cloned().collect();
        if deps.len() != dependencies.len() {
            return Err(graph_error("an edge is repeated", None));
        }
        let derived = derive(&nodes, &deps, None)?;
        let identity = bundle_identity(&nodes, &derived.digests, &derived.edges);
        Ok(Self { nodes, edges: derived.edges, digests: derived.digests, identity })
    }

    /// The bundle identity (lowercase hex), covering every node, edge and digest.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Nodes ordered by id.
    #[must_use]
    pub fn nodes(&self) -> &[BundleNode] {
        &self.nodes
    }

    /// Edges ordered by `(from, to)`, each with the upstream Merkle digest.
    #[must_use]
    pub fn edges(&self) -> &[BundleEdge] {
        &self.edges
    }

    /// Merkle digest of a node: it covers the node's content and the digest of
    /// every upstream node, so a changed upstream identity changes it.
    #[must_use]
    pub fn chain_digest(&self, id: &str) -> Option<&str> {
        let position = self.nodes.iter().position(|n| n.id == id)?;
        self.digests.get(position).map(String::as_str)
    }

    fn body(&self) -> BodyWire {
        BodyWire {
            version: BUNDLE_VERSION,
            nodes: self.nodes.iter().map(node_to_wire).collect(),
            edges: self
                .edges
                .iter()
                .map(|e| EdgeWire {
                    from: e.from.clone(),
                    to: e.to.clone(),
                    upstream_digest: e.upstream_digest.clone(),
                })
                .collect(),
            identity: self.identity.clone(),
        }
    }

    /// Serialize through the checksummed container: one body section plus one
    /// section per embedded node.
    ///
    /// # Errors
    /// A blank artifact id, a CBOR failure or a result over the bound refuses.
    pub fn to_bytes(
        &self,
        artifact_id: &str,
        limits: &BundleLimits,
    ) -> Result<Vec<u8>, BundleError> {
        if artifact_id.trim().is_empty() {
            return Err(IoError::Convert("missing artifact id".into()).into());
        }
        let body = to_cbor(&self.body())?;
        let mut descriptors = vec![section_descriptor(BODY_SECTION, "application/cbor", &body)];
        let mut sections = vec![SectionBytes::new(BODY_SECTION, body)];
        for node in &self.nodes {
            if let NodeSource::Embedded { bytes } = &node.source {
                let id = embedded_section(&node.id);
                descriptors.push(section_descriptor(id.clone(), "application/octet-stream", bytes));
                sections.push(SectionBytes::new(id, bytes.clone()));
            }
        }
        let encoded = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: antecedent_io::migrate::STABLE_FORMAT,
                minimum_reader_version: antecedent_io::migrate::STABLE_FORMAT,
                artifact_kind: ArtifactKind::Other(BUNDLE_KIND.into()),
                library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))?,
                artifact_id: artifact_id.into(),
                sections: descriptors,
                provenance: ProvenanceWire { note: "composition_bundle".into() },
            },
            sections,
        };
        let mut out = Vec::new();
        encoded.write_to(&mut out)?;
        if out.len() > limits.max_bundle_bytes {
            return Err(refused(BundleStage::Oversized, "the bundle exceeds its bound", None));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Wire
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeWire {
    id: String,
    kind: String,
    identity: String,
    embedded: bool,
    #[serde(default)]
    requires: Option<ProviderOrData>,
    #[serde(default)]
    facts: BTreeMap<String, String>,
    #[serde(default)]
    inspected: Vec<(String, f64)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EdgeWire {
    from: String,
    to: String,
    upstream_digest: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BodyWire {
    version: u16,
    nodes: Vec<NodeWire>,
    edges: Vec<EdgeWire>,
    identity: String,
}

#[derive(Deserialize)]
struct VersionProbe {
    version: u16,
}

fn node_to_wire(node: &BundleNode) -> NodeWire {
    let (embedded, requires) = match &node.source {
        NodeSource::Embedded { .. } => (true, None),
        NodeSource::Reference { requires, .. } => (false, Some(requires.clone())),
    };
    NodeWire {
        id: node.id.clone(),
        kind: node.kind.as_str().to_owned(),
        identity: node.identity.clone(),
        embedded,
        requires,
        facts: node.facts.clone(),
        inspected: node.inspected.clone(),
    }
}

struct Decoded {
    nodes: Vec<BundleNode>,
    edges: Vec<BundleEdge>,
    stored_identity: String,
}

fn node_from_wire(
    wire: NodeWire,
    reader: &mut ArtifactReader<std::io::Cursor<&[u8]>>,
    limits: &BundleLimits,
) -> Result<BundleNode, BundleError> {
    let Some(kind) = NodeKind::from_name(&wire.kind) else {
        return Err(refused(
            BundleStage::UnknownNodeKind,
            "the node kind is not known",
            Some(wire.id),
        ));
    };
    if wire.embedded {
        let section = embedded_section(&wire.id);
        let declared = reader.manifest().sections.iter().find(|s| s.id == section);
        match declared {
            Some(descriptor) if descriptor.uncompressed_size > limits.max_embedded_bytes as u64 => {
                return Err(refused(
                    BundleStage::Oversized,
                    "an embedded artifact exceeds its bound",
                    Some(wire.id),
                ));
            }
            Some(_) => {}
            None => {
                return Err(graph_error("an embedded node has no content section", Some(wire.id)));
            }
        }
        let bytes = reader.load_section(&section)?.as_bytes().to_vec();
        return Ok(BundleNode {
            id: wire.id,
            kind,
            identity: wire.identity,
            source: NodeSource::Embedded { bytes },
            facts: BTreeMap::new(),
            inspected: Vec::new(),
        });
    }
    let Some(requires) = wire.requires else {
        return Err(graph_error("a reference node states no requirement", Some(wire.id)));
    };
    Ok(BundleNode {
        id: wire.id,
        kind,
        source: NodeSource::Reference { identity: wire.identity.clone(), requires },
        identity: wire.identity,
        facts: wire.facts,
        inspected: wire.inspected,
    })
}

fn decode(bytes: &[u8], limits: &BundleLimits) -> Result<Decoded, BundleError> {
    if bytes.len() > limits.max_bundle_bytes {
        return Err(refused(BundleStage::Oversized, "the bundle exceeds its bound", None));
    }
    let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
    if reader.manifest().artifact_kind != ArtifactKind::Other(BUNDLE_KIND.into()) {
        return Err(IoError::Convert("not a composition bundle".into()).into());
    }
    let body_size = reader
        .manifest()
        .sections
        .iter()
        .find(|s| s.id == BODY_SECTION)
        .map(|s| s.uncompressed_size)
        .ok_or_else(|| IoError::Convert("composition bundle has no body".into()))?;
    if body_size > limits.max_bundle_bytes as u64 {
        return Err(refused(BundleStage::Oversized, "the bundle body exceeds its bound", None));
    }
    let body_bytes = reader.load_section(BODY_SECTION)?.as_bytes().to_vec();
    let probe: VersionProbe = from_cbor(&body_bytes)?;
    if probe.version != BUNDLE_VERSION {
        return Err(refused(
            BundleStage::IncompatibleVersion,
            "the bundle body version is not supported",
            None,
        ));
    }
    let body: BodyWire = from_cbor(&body_bytes)?;
    if body.nodes.len() > limits.max_nodes || body.edges.len() > limits.max_edges {
        return Err(refused(BundleStage::Oversized, "too many nodes or edges", None));
    }
    let embedded = body.nodes.iter().filter(|n| n.embedded).count();
    if reader.manifest().sections.len() != embedded + 1 {
        return Err(IoError::Convert("composition bundle has unexpected sections".into()).into());
    }
    let mut nodes = Vec::with_capacity(body.nodes.len());
    for wire in body.nodes {
        nodes.push(node_from_wire(wire, &mut reader, limits)?);
    }
    nodes.sort_by(|a, b| a.id.cmp(&b.id));
    let edges = body
        .edges
        .into_iter()
        .map(|e| BundleEdge { from: e.from, to: e.to, upstream_digest: e.upstream_digest })
        .collect();
    Ok(Decoded { nodes, edges, stored_identity: body.identity })
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// A verified upstream node handed to a [`NodeVerifier`].
#[derive(Clone, Copy, Debug)]
pub struct UpstreamNode<'a> {
    /// Node id.
    pub id: &'a str,
    /// Node kind.
    pub kind: NodeKind,
    /// Declared identity of the node's artifact.
    pub identity: &'a str,
    /// Effective facts: verified ones for a verified node, declared ones for a
    /// reference.
    pub facts: &'a BTreeMap<String, String>,
}

/// What a verifier established about an embedded artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedNode {
    /// The artifact's identity, recomputed from its decoded content.
    pub identity: String,
    /// Facts the bundle cross-checks across edges (see the `FACT_*` keys).
    pub facts: BTreeMap<String, String>,
    /// Named numbers read from the decoded artifact.
    pub values: Vec<(String, f64)>,
}

/// Why a verifier refused an embedded artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeFailure {
    /// Failed stage.
    pub stage: BundleStage,
    /// What failed.
    pub reason: String,
}

/// Decodes and verifies embedded artifacts of one [`NodeKind`] through that
/// artifact's own consumer.
pub trait NodeVerifier {
    /// The node kind this verifier handles.
    fn kind(&self) -> NodeKind;

    /// Decode `bytes` independently and recompute the artifact's identity.
    /// `upstream` lists the node's verified or declared upstream nodes.
    fn verify(
        &self,
        node_id: &str,
        bytes: &[u8],
        upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure>;
}

/// A provider the consumer supplies at an exact request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuppliedProvider {
    /// Provider service identity.
    pub provider_id: String,
    /// Provider input or model snapshot.
    pub snapshot_id: String,
    /// Fingerprint of the request the provider will answer.
    pub request_fingerprint: String,
}

/// A data snapshot the consumer supplies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuppliedData {
    /// Snapshot identity.
    pub snapshot_id: String,
    /// Digest of the supplied content.
    pub digest: String,
}

/// Providers and data the consumer can supply to resolve reference nodes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SuppliedSources {
    /// Supplied providers.
    pub providers: Vec<SuppliedProvider>,
    /// Supplied data snapshots.
    pub data: Vec<SuppliedData>,
}

impl SuppliedSources {
    /// Add a provider.
    #[must_use]
    pub fn with_provider(
        mut self,
        provider_id: &str,
        snapshot_id: &str,
        request_fingerprint: &str,
    ) -> Self {
        self.providers.push(SuppliedProvider {
            provider_id: provider_id.to_owned(),
            snapshot_id: snapshot_id.to_owned(),
            request_fingerprint: request_fingerprint.to_owned(),
        });
        self
    }

    /// Add a data snapshot.
    #[must_use]
    pub fn with_data(mut self, snapshot_id: &str, digest: &str) -> Self {
        self.data
            .push(SuppliedData { snapshot_id: snapshot_id.to_owned(), digest: digest.to_owned() });
        self
    }
}

/// Outcome of consuming one node.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeStatus {
    /// An embedded artifact decoded and recomputed to its declared identity, or a
    /// reference matched a supplied source. A resolved reference is never claimed
    /// replayed.
    Verified,
    /// A reference whose provider or data source was not supplied.
    ReferenceUnresolved {
        /// What must be supplied.
        requires: ProviderOrData,
    },
    /// Verification failed.
    Failed {
        /// Failed stage.
        stage: BundleStage,
        /// What failed.
        reason: String,
    },
}

/// The claim a decision result carries, derived from its ancestors (never read
/// from the bytes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimLabel {
    /// The decision rests on an external mean only; exported as a point-only
    /// attested result.
    PointOnlyAttested,
    /// The decision rests on aligned joint draws.
    JointDraw,
}

impl ClaimLabel {
    /// Stable wire label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PointOnlyAttested => "point_only_attested",
            Self::JointDraw => "joint_draw",
        }
    }
}

/// One consumed node.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedNode {
    /// Node id.
    pub id: String,
    /// Node kind.
    pub kind: NodeKind,
    /// Declared identity.
    pub identity: String,
    /// Recomputed Merkle digest.
    pub chain_digest: String,
    /// Verification outcome.
    pub status: NodeStatus,
    /// Effective facts (verified for a verified embedded node, else declared).
    pub facts: BTreeMap<String, String>,
    /// Numbers read from a verified embedded artifact.
    pub values: Vec<(String, f64)>,
    /// Declared, unverified values of a reference node.
    pub inspected: Vec<(String, f64)>,
    /// Claim label of a decision result that did not fail.
    pub claim_label: Option<ClaimLabel>,
}

/// A consumed bundle: every node's status, with inspected results preserved.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedBundle {
    identity: String,
    nodes: Vec<ConsumedNode>,
    edges: Vec<BundleEdge>,
}

impl ConsumedBundle {
    /// The recomputed bundle identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Nodes ordered by id.
    #[must_use]
    pub fn nodes(&self) -> &[ConsumedNode] {
        &self.nodes
    }

    /// Edges with their recomputed upstream digests.
    #[must_use]
    pub fn edges(&self) -> &[BundleEdge] {
        &self.edges
    }

    /// One node by id.
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&ConsumedNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// A number read from a node that verified; `None` for any other node.
    #[must_use]
    pub fn value(&self, id: &str, key: &str) -> Option<f64> {
        let node = self.node(id)?;
        if node.status != NodeStatus::Verified {
            return None;
        }
        node.values.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    /// Whether every node verified.
    #[must_use]
    pub fn all_verified(&self) -> bool {
        self.nodes.iter().all(|n| n.status == NodeStatus::Verified)
    }

    /// Require every node to have verified.
    ///
    /// # Errors
    /// The first failed node refuses at its stage; the first unresolved
    /// reference refuses as [`BundleStage::CallbackUnavailable`].
    pub fn require_verified(&self) -> Result<(), BundleError> {
        for node in &self.nodes {
            match &node.status {
                NodeStatus::Verified => {}
                NodeStatus::Failed { stage, reason } => {
                    return Err(BundleError::Refused {
                        stage: *stage,
                        reason: reason.clone(),
                        offending: Some(node.id.clone()),
                    });
                }
                NodeStatus::ReferenceUnresolved { .. } => {
                    return Err(refused(
                        BundleStage::CallbackUnavailable,
                        "a node needs a provider or data source that was not supplied",
                        Some(node.id.clone()),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct NodeState {
    status: NodeStatus,
    facts: BTreeMap<String, String>,
    values: Vec<(String, f64)>,
    label: Option<ClaimLabel>,
}

impl NodeState {
    fn failed(stage: BundleStage, reason: String) -> Self {
        Self {
            status: NodeStatus::Failed { stage, reason },
            facts: BTreeMap::new(),
            values: Vec::new(),
            label: None,
        }
    }

    fn is_failed(&self) -> bool {
        matches!(self.status, NodeStatus::Failed { .. })
    }
}

fn resolve_reference(
    node: &BundleNode,
    requires: &ProviderOrData,
    supplied: &SuppliedSources,
) -> NodeState {
    let status = match requires {
        ProviderOrData::Provider { provider_id, snapshot_id, request_fingerprint } => {
            let found = supplied
                .providers
                .iter()
                .find(|p| &p.provider_id == provider_id && &p.snapshot_id == snapshot_id);
            match found {
                None => NodeStatus::ReferenceUnresolved { requires: requires.clone() },
                Some(p) if &p.request_fingerprint == request_fingerprint => NodeStatus::Verified,
                Some(_) => NodeStatus::Failed {
                    stage: BundleStage::ProviderRequestChanged,
                    reason: "the supplied provider answers a different request".to_owned(),
                },
            }
        }
        ProviderOrData::Data { snapshot_id, digest } => {
            match supplied.data.iter().find(|d| &d.snapshot_id == snapshot_id) {
                None => NodeStatus::ReferenceUnresolved { requires: requires.clone() },
                Some(d) if &d.digest == digest => NodeStatus::Verified,
                Some(_) => NodeStatus::Failed {
                    stage: BundleStage::GraphOrSnapshotMismatch,
                    reason: "the supplied data differs from the referenced snapshot".to_owned(),
                },
            }
        }
    };
    if let NodeStatus::Failed { stage, reason } = status {
        return NodeState::failed(stage, reason);
    }
    NodeState { status, facts: node.facts.clone(), values: Vec::new(), label: None }
}

fn verify_embedded(
    node: &BundleNode,
    bytes: &[u8],
    upstream: &[UpstreamNode<'_>],
    verifiers: &[Box<dyn NodeVerifier>],
) -> NodeState {
    let Some(verifier) = verifiers.iter().find(|v| v.kind() == node.kind) else {
        return NodeState::failed(
            BundleStage::UnknownNodeKind,
            format!("no verifier is registered for kind `{}`", node.kind.as_str()),
        );
    };
    match verifier.verify(&node.id, bytes, upstream) {
        Err(failure) => NodeState::failed(failure.stage, failure.reason),
        Ok(verified) if verified.identity != node.identity => NodeState::failed(
            node.kind.identity_mismatch_stage(),
            "the artifact's recomputed identity differs from the declared one".to_owned(),
        ),
        Ok(verified) => NodeState {
            status: NodeStatus::Verified,
            facts: verified.facts,
            values: verified.values,
            label: None,
        },
    }
}

fn find_conflict(
    facts: &BTreeMap<String, String>,
    upstream: &[UpstreamNode<'_>],
) -> Option<(BundleStage, String)> {
    for up in upstream {
        for (key, stage) in BINDING_FACTS {
            if let (Some(a), Some(b)) = (up.facts.get(key), facts.get(key)) {
                if a != b {
                    return Some((
                        stage,
                        format!("`{key}` differs from upstream node `{}`", up.id),
                    ));
                }
            }
        }
        if let (Some(required), Some(law)) = (facts.get(FACT_REQUIRES_LAW), up.facts.get(FACT_LAW))
        {
            if required != law {
                return Some((
                    BundleStage::UnsupportedLaw,
                    format!("upstream node `{}` supplies `{law}`, not `{required}`", up.id),
                ));
            }
        }
    }
    None
}

fn claim_label(
    position: usize,
    parents: &[Vec<usize>],
    states: &[NodeState],
) -> Option<ClaimLabel> {
    let mut seen = vec![false; states.len()];
    let mut stack: Vec<usize> = parents[position].clone();
    let (mut mean_only, mut joint) = (false, false);
    while let Some(next) = stack.pop() {
        if seen[next] {
            continue;
        }
        seen[next] = true;
        stack.extend(parents[next].iter().copied());
        if states[next].is_failed() {
            continue;
        }
        match states[next].facts.get(FACT_LAW).map(String::as_str) {
            Some(LAW_MEAN_ONLY) => mean_only = true,
            Some(LAW_JOINT) => joint = true,
            _ => {}
        }
    }
    if mean_only {
        Some(ClaimLabel::PointOnlyAttested)
    } else if joint {
        Some(ClaimLabel::JointDraw)
    } else {
        None
    }
}

/// Consumes bundles, decoding embedded artifacts through the verifiers
/// registered per node kind.
#[derive(Default)]
pub struct BundleConsumer {
    verifiers: Vec<Box<dyn NodeVerifier>>,
}

impl BundleConsumer {
    /// A consumer with no verifiers: every embedded node fails as
    /// [`BundleStage::UnknownNodeKind`] until its kind is registered.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the verifier of one node kind (a later registration of the same
    /// kind is ignored: the first one wins).
    #[must_use]
    pub fn register(mut self, verifier: Box<dyn NodeVerifier>) -> Self {
        if !self.verifiers.iter().any(|v| v.kind() == verifier.kind()) {
            self.verifiers.push(verifier);
        }
        self
    }

    /// Consume a bundle: verify the container, recompute every node and edge
    /// digest, require the retained identity, then verify each node.
    ///
    /// # Errors
    /// An oversized or corrupt container, an unsupported version, an unknown node
    /// kind, an edge digest that differs from its recomputation, or an identity
    /// different from `expected_identity` refuses. A node-level failure does not
    /// refuse the bundle; it is reported on that node.
    pub fn consume(
        &self,
        bytes: &[u8],
        limits: &BundleLimits,
        expected_identity: &str,
        supplied: &SuppliedSources,
    ) -> Result<ConsumedBundle, BundleError> {
        let decoded = decode(bytes, limits)?;
        let deps: BTreeSet<(String, String)> =
            decoded.edges.iter().map(|e| (e.from.clone(), e.to.clone())).collect();
        let derived = derive(&decoded.nodes, &deps, Some(&decoded.edges))?;
        let identity = bundle_identity(&decoded.nodes, &derived.digests, &derived.edges);
        if identity != decoded.stored_identity || identity != expected_identity {
            return Err(refused(
                BundleStage::ExpectedIdentityMismatch,
                "the bundle identity differs from the consumer's retained identity",
                None,
            ));
        }
        let states = self.evaluate(&decoded.nodes, &derived, supplied);
        let nodes = decoded
            .nodes
            .iter()
            .zip(states)
            .zip(&derived.digests)
            .map(|((node, state), digest)| ConsumedNode {
                id: node.id.clone(),
                kind: node.kind,
                identity: node.identity.clone(),
                chain_digest: digest.clone(),
                status: state.status,
                facts: state.facts,
                values: state.values,
                inspected: node.inspected.clone(),
                claim_label: state.label,
            })
            .collect();
        Ok(ConsumedBundle { identity, nodes, edges: derived.edges })
    }

    fn evaluate(
        &self,
        nodes: &[BundleNode],
        derived: &Derived,
        supplied: &SuppliedSources,
    ) -> Vec<NodeState> {
        let placeholder =
            NodeState::failed(BundleStage::EdgeDigestMismatch, "node was not evaluated".to_owned());
        let mut states = vec![placeholder; nodes.len()];
        for &position in &derived.order {
            let state = {
                let upstream: Vec<UpstreamNode<'_>> = derived.parents[position]
                    .iter()
                    .filter(|p| !states[**p].is_failed())
                    .map(|&p| UpstreamNode {
                        id: &nodes[p].id,
                        kind: nodes[p].kind,
                        identity: &nodes[p].identity,
                        facts: &states[p].facts,
                    })
                    .collect();
                let node = &nodes[position];
                let mut state = match &node.source {
                    NodeSource::Embedded { bytes } => {
                        verify_embedded(node, bytes, &upstream, &self.verifiers)
                    }
                    NodeSource::Reference { requires, .. } => {
                        resolve_reference(node, requires, supplied)
                    }
                };
                if !state.is_failed() {
                    if let Some((stage, reason)) = find_conflict(&state.facts, &upstream) {
                        state = NodeState::failed(stage, reason);
                    }
                }
                state
            };
            states[position] = state;
        }
        for position in 0..nodes.len() {
            if nodes[position].kind == NodeKind::DecisionResult && !states[position].is_failed() {
                let label = claim_label(position, &derived.parents, &states);
                states[position].label = label;
            }
        }
        states
    }
}
