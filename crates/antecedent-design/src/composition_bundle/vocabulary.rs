//! Vocabulary and node declarations for composition bundles.

use super::BundleStage;
use antecedent_contract::CompositionStage;
use antecedent_core::named_variants;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    /// A source-law binding to terminal decision state and study ranking.
    Rollout,
    /// An inverse functional query with its forward evidence and result table.
    InverseQuery,
    /// An identification-repair search report.
    RepairReport,
    /// Independently checked historical selective execution receipt; no live state.
    RecalculationReceipt,
    /// Independently checked frozen same-row scores for their supported retarget operation.
    FrozenScores,
    /// An observation query: a conditional or interventional law, a posterior, a
    /// required-observation count, or aligned draws of parameter redraw.
    ObservationQuery,
    /// A scientific bridge: a mean, or aligned draws, carried to another
    /// quantity under declared assumptions.
    ScientificBridge,
    /// A propagation chain: uncertain quantities carried through several
    /// transformations, with the summary of every stage.
    PropagationChain,
}

named_variants! {
    NodeKind {
        CausalContract => "causal_contract",
        ExecutionOrFit => "execution_or_fit",
        ExternalClaim => "external_claim",
        Attestation => "attestation",
        EvidenceRelationship => "evidence_relationship",
        QuantityCoordinates => "quantity_coordinates",
        SupportTrustCalibration => "support_trust_calibration",
        Distribution => "distribution",
        Transformation => "transformation",
        DecisionContract => "decision_contract",
        DecisionResult => "decision_result",
        Sensitivity => "sensitivity",
        StudyRanking => "study_ranking",
        Rollout => "rollout",
        InverseQuery => "inverse_query",
        RepairReport => "repair_report",
        RecalculationReceipt => "recalculation_receipt",
        FrozenScores => "frozen_scores",
        ObservationQuery => "observation_query",
        ScientificBridge => "scientific_bridge",
        PropagationChain => "propagation_chain",
    }
}

impl NodeKind {
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
            Self::Transformation | Self::Rollout => CompositionStage::Transformation,
            Self::DecisionContract => CompositionStage::DecisionContract,
            Self::DecisionResult
            | Self::InverseQuery
            | Self::ObservationQuery
            | Self::ScientificBridge
            | Self::PropagationChain => CompositionStage::Claim,
            Self::Sensitivity => CompositionStage::SensitivityInput,
            Self::StudyRanking => CompositionStage::StudyRankingProvider,
        }
    }

    /// Stage a recomputed identity that differs from the declared one fails at:
    /// a value-bearing artifact was altered, any other was swapped.
    pub(super) const fn identity_mismatch_stage(self) -> BundleStage {
        match self {
            Self::QuantityCoordinates
            | Self::DecisionResult
            | Self::Sensitivity
            | Self::StudyRanking
            | Self::Rollout
            | Self::InverseQuery
            | Self::ObservationQuery
            | Self::ScientificBridge
            | Self::PropagationChain => BundleStage::TamperedQuantity,
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

named_variants! {
    EvidenceRelationship {
        SharedData => "shared_data",
        SharedPrior => "shared_prior",
        SharedFittedModel => "shared_fitted_model",
        UnknownDependence => "unknown_dependence",
        Independent => "independent",
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
    pub(super) fn canonical(&self) -> String {
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
