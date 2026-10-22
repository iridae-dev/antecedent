//! Portable composition verification (`composition_verification_v1`).
//!
//! A bounded, checksummed container holding one revalidation of a claim-level
//! dependency graph so that a consumer in a fresh process can see, for every
//! claim, whether it is still valid, must be recomputed or is no longer
//! licensed, and the lineage behind it:
//!
//! * the earlier and the current graph: every node with its kind, content
//!   identity, structural identity, representation version, coordinate,
//!   support, provider trust, calibration declaration, assumptions,
//!   transformation and ledger sources, and every typed dependency;
//! * the observed effects (a recomputed stage, a stale or withdrawn ledger
//!   result) the classification was made under;
//! * the shared-data relations among nodes, declared from the evidence ledger
//!   the producer held, with that ledger's state digest;
//! * the classification of every claim, with its numerical reasons, its
//!   withdrawals, whether its earlier fit and uncertainty authority stand,
//!   its calibration state, and the identity of its lineage report.
//!
//! A consumer trusts none of the stored classification. It rebuilds both
//! graphs (every structural check of the library applies), classifies again,
//! derives each lineage report again and requires every stored class, reason,
//! withdrawal, authority, calibration state and lineage identity to equal the
//! derivation. A resealed edit of a node, an edge, an effect or a stored
//! class either fails the graph checks or changes the replayed
//! classification; an edit that is not resealed is an identity mismatch, and a
//! consumer that retained the identity independently refuses a consistently
//! resealed artifact of another verification.
//!
//! What replay protects against: corruption, unsealed edits and resealed edits
//! that change a classification. What it does not protect against: a producer
//! that declares a node's identity, coordinate, support or calibration
//! falsely (they are declarations; the artifact certifies the dependency
//! logic over them, not their truth), or a bug shared by producer and
//! consumer (the same classification code replays; the independent oracles
//! are in the test suites). The shared-data relations are declarations of
//! the ledger the producer held; [`CompositionVerificationArtifact::verify_ledger`]
//! checks them against a ledger the consumer holds. The artifact carries no
//! interval, posterior or calibration record: a calibration declaration names
//! a record identity and coordinate, and the artifact never measures one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::composition_verification::{
    CalibrationDeclaration, CompositionGraph, CompositionNode, CompositionView, DependencyKind,
    GraphError, GraphRevalidation, MAX_NODES, MAX_TEXT_BYTES, NodeKind, ObservedEffect,
    SharedDataRelation, SupportStanding, TrustStanding, ViewError, revalidate,
    shared_data_relations,
};
use antecedent_contract::evidence_ledger::{EvidenceLedger, SourceId, parse_digest_hex};
use antecedent_core::{CancellationToken, CanonicalWriter, SemanticDigest};
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;

/// The artifact major version this reader writes and accepts.
pub const COMPOSITION_VERIFICATION_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const COMPOSITION_VERIFICATION_ARTIFACT_FEATURE: &str = "composition_verification_v1";
/// Most bytes a composition verification artifact may occupy. The content limits
/// (nodes, edges, items per node, text bytes, relations) bound the real payload
/// well under this; the cap is checked before any section is parsed.
pub const MAX_COMPOSITION_VERIFICATION_ARTIFACT_BYTES: usize = 32 * 1024 * 1024;
/// Most shared-data relations one artifact may carry.
pub const MAX_SHARED_DATA_RELATIONS: usize = 65_536;

const ARTIFACT_KIND: &str = crate::artifact_kinds::COMPOSITION_VERIFICATION;
const META_SECTION: &str = "composition_verification_meta";
const TAG_ARTIFACT: &str = "antecedent.composition_verification.artifact";

crate::sectioned_artifact_enum! {
    /// Why a composition verification artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    CompositionVerificationArtifactError {
        noun: "composition verification",
        identity: "composition_verification.identity_mismatch",
        variants {
            /// A stored classification, report identity or relation does not replay.
            ReplayMismatch(&'static str),
            /// The consumer's ledger does not hold what the artifact declared.
            LedgerMismatch(&'static str),
        }
        display(f) {
            Self::ReplayMismatch(what) => {
                write!(f, "composition_verification.replay_mismatch: stored {what} does not replay")
            }
            Self::LedgerMismatch(what) => {
                write!(f, "composition_verification.ledger_mismatch: {what} differs from the ledger")
            }
        }
    }
}

impl CompositionVerificationArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a stored value that does not replay, a ledger that does
    /// not hold the declared relations and a changed identity; absent for
    /// corruption, unsupported versions, malformed values, limits and
    /// encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::IdentityMismatch { field } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "composition_verification.identity_mismatch".to_owned(),
                format!("{field} changed"),
            )),
            Self::ReplayMismatch(what) => Some((
                antecedent_core::reason_code!("invalid_argument"),
                "composition_verification.replay_mismatch".to_owned(),
                format!("the stored {what} does not replay"),
            )),
            Self::LedgerMismatch(what) => Some((
                antecedent_core::reason_code!("invalid_argument"),
                "composition_verification.ledger_mismatch".to_owned(),
                format!("the declared {what} differs from the consumer's evidence ledger"),
            )),
            _ => None,
        }
    }
}

fn malformed(text: impl Into<String>) -> CompositionVerificationArtifactError {
    CompositionVerificationArtifactError::Malformed(text.into())
}

fn graph_error(error: &GraphError) -> CompositionVerificationArtifactError {
    match error {
        GraphError::LimitExceeded(what) => {
            CompositionVerificationArtifactError::LimitsExceeded(what)
        }
        other => malformed(other.to_string()),
    }
}

// -- wire types -----------------------------------------------------------------------------------

/// One calibration declaration on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationDeclarationWire {
    /// `point_only`, `required`, `held` or `inherited`.
    pub kind: String,
    /// The node the record is claimed from (`inherited`).
    pub from: Option<String>,
    /// Record identity (hex), for `held` and `inherited`.
    pub record: Option<String>,
    /// Coordinate (hex), for `held` and `inherited`.
    pub coordinate: Option<String>,
}

/// One node on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeWire {
    /// Node key.
    pub key: String,
    /// `NodeKind` name.
    pub kind: String,
    /// Content identity (hex).
    pub identity: String,
    /// Structural identity (hex).
    pub structure: Option<String>,
    /// Model or representation version.
    pub representation: Option<String>,
    /// Evaluation coordinate (hex).
    pub coordinate: String,
    /// Unsupported coordinates; `None` when supported.
    pub unsupported: Option<Vec<String>>,
    /// `TrustStanding` name.
    pub trust: String,
    /// Calibration declaration.
    pub calibration: CalibrationDeclarationWire,
    /// Assumptions introduced at the node.
    pub assumptions: Vec<String>,
    /// Transformation applied.
    pub transformation: Option<String>,
    /// Ledger sources (hex).
    pub evidence: Vec<String>,
}

/// One dependency on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeWire {
    /// The dependent node.
    pub downstream: String,
    /// The node depended on.
    pub upstream: String,
    /// `DependencyKind` name.
    pub kind: String,
}

/// A graph on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphWire {
    /// Nodes in key order.
    pub nodes: Vec<NodeWire>,
    /// Edges in canonical order.
    pub edges: Vec<EdgeWire>,
}

/// One observed effect on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectWire {
    /// The node it concerns.
    pub key: String,
    /// `recomputed`, `withdrawn` or `refused`.
    pub kind: String,
    /// The cause or refusal detail.
    pub text: String,
}

/// One shared-data relation on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedDataWire {
    /// First node key.
    pub first: String,
    /// Second node key.
    pub second: String,
    /// Relation name.
    pub relation: String,
    /// Observations shared.
    pub shared_observations: u64,
}

/// One numerical reason on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasonWire {
    /// The node whose own input changed.
    pub origin: String,
    /// The dependency it arrived through.
    pub via_node: Option<String>,
    /// The dependency kind.
    pub via_kind: Option<String>,
    /// `NumericalCause` name.
    pub cause: String,
}

/// One withdrawal on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WithdrawalWire {
    /// `WithdrawalKind` name.
    pub kind: String,
    /// Where the loss originates.
    pub origin: String,
    /// The dependency it arrived through.
    pub via_node: Option<String>,
    /// The dependency kind.
    pub via_kind: Option<String>,
    /// Slot or coordinates.
    pub detail: String,
}

/// One claim's classification on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimWire {
    /// Node key.
    pub key: String,
    /// `still_valid`, `recompute` or `withdrawn`.
    pub class: String,
    /// Numerical reasons.
    pub numerical: Vec<ReasonWire>,
    /// Withdrawals.
    pub withdrawals: Vec<WithdrawalWire>,
    /// Whether the earlier fit and uncertainty authority stand.
    pub authority_retained: bool,
    /// `CalibrationState` name.
    pub calibration_state: String,
    /// Where an inherited record came from.
    pub calibration_from: Option<String>,
    /// Whether a new calibration is needed at this coordinate.
    pub calibration_newly_required: bool,
    /// Why a record does not hold.
    pub calibration_detail: Option<String>,
    /// Identity (hex) of the claim's lineage report.
    pub lineage: String,
}

/// Stored metadata of a composition verification artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositionVerificationMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Features a reader must understand. This version defines none; a reader
    /// refuses any entry.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_features: Vec<String>,
    /// The earlier snapshot; `None` for a first verification.
    pub previous: Option<GraphWire>,
    /// The current snapshot.
    pub current: GraphWire,
    /// Observed effects, in key order.
    pub effects: Vec<EffectWire>,
    /// Shared-data relations among nodes, declared from a ledger.
    pub shared_data: Vec<SharedDataWire>,
    /// State digest (hex) of the ledger the relations were read from.
    pub ledger_state: Option<String>,
    /// Identity (hex) of the classification as a whole.
    pub revalidation: String,
    /// Every claim's classification, in key order.
    pub claims: Vec<ClaimWire>,
    /// BLAKE3 identity (hex) of everything above.
    pub identity: String,
}

// -- conversion -----------------------------------------------------------------------------------

fn digest_of(
    text: &str,
    what: &'static str,
) -> Result<SemanticDigest, CompositionVerificationArtifactError> {
    parse_digest_hex(text).ok_or_else(|| malformed(format!("{what} is not 64 hex digits")))
}

fn calibration_to_wire(c: &CalibrationDeclaration) -> CalibrationDeclarationWire {
    let (kind, from, record, coordinate) = match c {
        CalibrationDeclaration::PointOnly => ("point_only", None, None, None),
        CalibrationDeclaration::Required => ("required", None, None, None),
        CalibrationDeclaration::Held { record, coordinate } => {
            ("held", None, Some(record.to_hex()), Some(coordinate.to_hex()))
        }
        CalibrationDeclaration::Inherited { from, record, coordinate } => {
            ("inherited", Some(from.clone()), Some(record.to_hex()), Some(coordinate.to_hex()))
        }
    };
    CalibrationDeclarationWire { kind: kind.to_owned(), from, record, coordinate }
}

fn calibration_from_wire(
    wire: &CalibrationDeclarationWire,
) -> Result<CalibrationDeclaration, CompositionVerificationArtifactError> {
    let pair = || -> Result<_, CompositionVerificationArtifactError> {
        let record = wire.record.as_deref().ok_or_else(|| malformed("calibration record"))?;
        let coordinate =
            wire.coordinate.as_deref().ok_or_else(|| malformed("calibration coordinate"))?;
        Ok((digest_of(record, "calibration record")?, digest_of(coordinate, "coordinate")?))
    };
    let bare = wire.from.is_none() && wire.record.is_none() && wire.coordinate.is_none();
    Ok(match wire.kind.as_str() {
        "point_only" if bare => CalibrationDeclaration::PointOnly,
        "required" if bare => CalibrationDeclaration::Required,
        "held" if wire.from.is_none() => {
            let (record, coordinate) = pair()?;
            CalibrationDeclaration::Held { record, coordinate }
        }
        "inherited" => {
            let from = wire.from.clone().ok_or_else(|| malformed("calibration source"))?;
            let (record, coordinate) = pair()?;
            CalibrationDeclaration::Inherited { from, record, coordinate }
        }
        _ => {
            return Err(CompositionVerificationArtifactError::UnsupportedSemantics(
                "calibration declaration",
            ));
        }
    })
}

fn node_to_wire(node: &CompositionNode) -> NodeWire {
    NodeWire {
        key: node.key.clone(),
        kind: node.kind.as_str().to_owned(),
        identity: node.identity.to_hex(),
        structure: node.structure.map(|digest| digest.to_hex()),
        representation: node.representation.clone(),
        coordinate: node.coordinate.to_hex(),
        unsupported: match &node.support {
            SupportStanding::Supported => None,
            SupportStanding::Unsupported { coordinates } => Some(coordinates.clone()),
        },
        trust: node.trust.as_str().to_owned(),
        calibration: calibration_to_wire(&node.calibration),
        assumptions: node.assumptions.clone(),
        transformation: node.transformation.clone(),
        evidence: node
            .evidence
            .iter()
            .map(antecedent_contract::evidence_ledger::SourceId::to_hex)
            .collect(),
    }
}

fn node_from_wire(
    wire: &NodeWire,
) -> Result<CompositionNode, CompositionVerificationArtifactError> {
    let unsupported = CompositionVerificationArtifactError::UnsupportedSemantics;
    let kind = NodeKind::from_name(&wire.kind).ok_or(unsupported("node kind"))?;
    let mut node =
        CompositionNode::new(wire.key.clone(), kind, digest_of(&wire.identity, "node identity")?);
    node.structure =
        wire.structure.as_deref().map(|t| digest_of(t, "structure identity")).transpose()?;
    node.representation.clone_from(&wire.representation);
    node.coordinate = digest_of(&wire.coordinate, "node coordinate")?;
    if let Some(coordinates) = &wire.unsupported {
        node.support = SupportStanding::Unsupported { coordinates: coordinates.clone() };
    }
    node.trust = TrustStanding::from_name(&wire.trust).ok_or(unsupported("trust standing"))?;
    node.calibration = calibration_from_wire(&wire.calibration)?;
    node.assumptions.clone_from(&wire.assumptions);
    node.transformation.clone_from(&wire.transformation);
    node.evidence = wire
        .evidence
        .iter()
        .map(|t| SourceId::from_hex(t).ok_or_else(|| malformed("evidence source is not hex")))
        .collect::<Result<_, _>>()?;
    Ok(node)
}

fn graph_to_wire(graph: &CompositionGraph) -> GraphWire {
    GraphWire {
        nodes: graph.nodes().map(node_to_wire).collect(),
        edges: graph
            .edges()
            .map(|e| EdgeWire {
                downstream: e.downstream.clone(),
                upstream: e.upstream.clone(),
                kind: e.kind.as_str().to_owned(),
            })
            .collect(),
    }
}

fn graph_from_wire(
    wire: &GraphWire,
) -> Result<CompositionGraph, CompositionVerificationArtifactError> {
    if wire.nodes.len() > MAX_NODES {
        return Err(CompositionVerificationArtifactError::LimitsExceeded("nodes"));
    }
    let mut graph = CompositionGraph::new();
    for node in &wire.nodes {
        graph.add_node(node_from_wire(node)?).map_err(|e| graph_error(&e))?;
    }
    for edge in &wire.edges {
        let kind = DependencyKind::from_name(&edge.kind)
            .ok_or(CompositionVerificationArtifactError::UnsupportedSemantics("dependency kind"))?;
        graph.add_edge(&edge.downstream, &edge.upstream, kind).map_err(|e| graph_error(&e))?;
    }
    // A stored graph is in canonical order, so a rebuilt one lists the same
    // wire. Anything else was edited or was not produced by this writer.
    if graph_to_wire(&graph) != *wire {
        return Err(malformed("graph is not in canonical form"));
    }
    graph.validate().map_err(|e| graph_error(&e))?;
    Ok(graph)
}

fn effects_to_wire(
    effects: &BTreeMap<String, ObservedEffect>,
) -> Result<Vec<EffectWire>, CompositionVerificationArtifactError> {
    let mut out = Vec::with_capacity(effects.len());
    for (key, effect) in effects {
        let (kind, text) = match effect {
            ObservedEffect::Recomputed { cause } => ("recomputed", cause),
            ObservedEffect::Withdrawn { cause } => ("withdrawn", cause),
            ObservedEffect::Refused { detail } => ("refused", detail),
            _ => {
                return Err(CompositionVerificationArtifactError::UnsupportedSemantics(
                    "effect kind",
                ));
            }
        };
        out.push(EffectWire { key: key.clone(), kind: kind.to_owned(), text: text.clone() });
    }
    Ok(out)
}

fn effects_from_wire(
    wire: &[EffectWire],
) -> Result<BTreeMap<String, ObservedEffect>, CompositionVerificationArtifactError> {
    if wire.len() > MAX_NODES {
        return Err(CompositionVerificationArtifactError::LimitsExceeded("effects"));
    }
    let mut out = BTreeMap::new();
    for effect in wire {
        if effect.text.trim().is_empty() || effect.text.len() > MAX_TEXT_BYTES {
            return Err(malformed("effect text is blank or too long"));
        }
        let value = match effect.kind.as_str() {
            "recomputed" => ObservedEffect::Recomputed { cause: effect.text.clone() },
            "withdrawn" => ObservedEffect::Withdrawn { cause: effect.text.clone() },
            "refused" => ObservedEffect::Refused { detail: effect.text.clone() },
            _ => {
                return Err(CompositionVerificationArtifactError::UnsupportedSemantics(
                    "effect kind",
                ));
            }
        };
        if out.insert(effect.key.clone(), value).is_some() {
            return Err(malformed("effect declared twice for one node"));
        }
    }
    Ok(out)
}

fn shared_to_wire(relations: &[SharedDataRelation]) -> Vec<SharedDataWire> {
    let mut wire: Vec<SharedDataWire> = relations
        .iter()
        .map(|r| SharedDataWire {
            first: r.first.clone(),
            second: r.second.clone(),
            relation: r.relation.to_owned(),
            shared_observations: r.shared_observations,
        })
        .collect();
    wire.sort_by(|a, b| {
        (&a.first, &a.second, &a.relation).cmp(&(&b.first, &b.second, &b.relation))
    });
    wire
}

fn shared_from_wire(
    wire: &[SharedDataWire],
    graph: &CompositionGraph,
) -> Result<Vec<SharedDataRelation>, CompositionVerificationArtifactError> {
    static NAMES: [&str; 4] =
        ["shared_data", "shared_prior", "shared_fitted_model", "unknown_dependence"];
    if wire.len() > MAX_SHARED_DATA_RELATIONS {
        return Err(CompositionVerificationArtifactError::LimitsExceeded("shared-data relations"));
    }
    let mut out = Vec::with_capacity(wire.len());
    for r in wire {
        let relation = NAMES
            .iter()
            .find(|name| **name == r.relation)
            .ok_or(CompositionVerificationArtifactError::UnsupportedSemantics("relation kind"))?;
        if graph.node(&r.first).is_none() || graph.node(&r.second).is_none() || r.first >= r.second
        {
            return Err(malformed("shared-data relation names unknown or unordered nodes"));
        }
        out.push(SharedDataRelation {
            first: r.first.clone(),
            second: r.second.clone(),
            relation,
            shared_observations: r.shared_observations,
        });
    }
    if shared_to_wire(&out) != wire {
        return Err(malformed("shared-data relations are not in canonical form"));
    }
    Ok(out)
}

fn via_parts(via: Option<&(String, DependencyKind)>) -> (Option<String>, Option<String>) {
    via.map_or((None, None), |(node, kind)| (Some(node.clone()), Some(kind.as_str().to_owned())))
}

fn claims_to_wire(
    graph: &CompositionGraph,
    revalidation: &GraphRevalidation,
    shared: &[SharedDataRelation],
) -> Result<Vec<ClaimWire>, CompositionVerificationArtifactError> {
    let mut out = Vec::new();
    for claim in revalidation.claims() {
        let view = CompositionView::of(graph, revalidation, &claim.key, shared)
            .map_err(|e| malformed(e.to_string()))?;
        out.push(ClaimWire {
            key: claim.key.clone(),
            class: claim.class.as_str().to_owned(),
            numerical: claim
                .numerical
                .iter()
                .map(|r| {
                    let (via_node, via_kind) = via_parts(r.via.as_ref());
                    ReasonWire {
                        origin: r.origin.clone(),
                        via_node,
                        via_kind,
                        cause: r.cause.as_str().to_owned(),
                    }
                })
                .collect(),
            withdrawals: claim
                .withdrawals
                .iter()
                .map(|w| {
                    let (via_node, via_kind) = via_parts(w.via.as_ref());
                    WithdrawalWire {
                        kind: w.kind.as_str().to_owned(),
                        origin: w.origin.clone(),
                        via_node,
                        via_kind,
                        detail: w.detail.clone(),
                    }
                })
                .collect(),
            authority_retained: claim.authority
                == antecedent_contract::composition_verification::PriorAuthority::Retained,
            calibration_state: claim.calibration.state.as_str().to_owned(),
            calibration_from: claim.calibration.from.clone(),
            calibration_newly_required: claim.calibration.newly_required,
            calibration_detail: claim.calibration.detail.map(str::to_owned),
            lineage: view.identity().to_hex(),
        });
    }
    Ok(out)
}

// -- identity -------------------------------------------------------------------------------------

fn compute_identity(
    meta: &CompositionVerificationMeta,
) -> Result<String, CompositionVerificationArtifactError> {
    let mut blanked = meta.clone();
    blanked.identity = String::new();
    let bytes = to_cbor(&blanked)
        .map_err(|error| CompositionVerificationArtifactError::Encode(error.to_string()))?;
    let mut writer = CanonicalWriter::new(TAG_ARTIFACT);
    writer.word(u64::from(meta.version)).text(&meta.feature).bytes(&bytes);
    Ok(writer.finish_hex())
}

// -- consumption ----------------------------------------------------------------------------------

fn check(
    meta: &CompositionVerificationMeta,
    expected_identity: Option<&str>,
) -> Result<CompositionVerificationArtifact, CompositionVerificationArtifactError> {
    if meta.version != COMPOSITION_VERIFICATION_ARTIFACT_VERSION {
        return Err(CompositionVerificationArtifactError::UnsupportedVersion {
            version: meta.version,
        });
    }
    if meta.feature != COMPOSITION_VERIFICATION_ARTIFACT_FEATURE {
        return Err(CompositionVerificationArtifactError::UnsupportedSemantics("feature marker"));
    }
    if !meta.required_features.is_empty() {
        return Err(CompositionVerificationArtifactError::UnsupportedSemantics("required feature"));
    }
    let current = graph_from_wire(&meta.current)?;
    let previous = meta.previous.as_ref().map(graph_from_wire).transpose()?;
    let effects = effects_from_wire(&meta.effects)?;
    let shared = shared_from_wire(&meta.shared_data, &current)?;
    let ledger_state =
        meta.ledger_state.as_deref().map(|t| digest_of(t, "ledger state digest")).transpose()?;
    let revalidation = revalidate(previous.as_ref().unwrap_or(&current), &current, &effects)
        .map_err(|e| graph_error(&e))?;
    let mismatch = CompositionVerificationArtifactError::ReplayMismatch;
    if claims_to_wire(&current, &revalidation, &shared)? != meta.claims {
        return Err(mismatch("classification"));
    }
    if revalidation.identity().to_hex() != meta.revalidation {
        return Err(mismatch("classification identity"));
    }
    let identity = compute_identity(meta)?;
    if meta.identity != identity {
        return Err(CompositionVerificationArtifactError::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(CompositionVerificationArtifactError::IdentityMismatch {
            field: "retained_identity",
        });
    }
    Ok(CompositionVerificationArtifact {
        previous,
        current,
        effects,
        shared,
        revalidation,
        identity,
        ledger_state,
    })
}

fn encode_meta(
    meta: &CompositionVerificationMeta,
    artifact_id: &str,
) -> Result<Vec<u8>, CompositionVerificationArtifactError> {
    if artifact_id.trim().is_empty() {
        return Err(CompositionVerificationArtifactError::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta)
        .map_err(|error| CompositionVerificationArtifactError::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "composition_verification",
            max_bytes: MAX_COMPOSITION_VERIFICATION_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: META_SECTION,
            content_type: crate::CONTENT_CBOR,
            bytes: &meta_bytes,
        }],
    )
    .map_err(|error| crate::map_artifact_check!(error, CompositionVerificationArtifactError))
}

fn decode_meta(
    bytes: &[u8],
) -> Result<CompositionVerificationMeta, CompositionVerificationArtifactError> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_COMPOSITION_VERIFICATION_ARTIFACT_BYTES,
        &[META_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, CompositionVerificationArtifactError))?;
    let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(CompositionVerificationArtifactError::Malformed(
            "unsupported container layout".into(),
        ));
    };
    crate::decode_with_version(&meta_bytes, |version: u16| {
        if version == COMPOSITION_VERIFICATION_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(CompositionVerificationArtifactError::UnsupportedVersion { version })
        }
    })
}

fn build_meta(
    previous: Option<&CompositionGraph>,
    current: &CompositionGraph,
    effects: &BTreeMap<String, ObservedEffect>,
    shared: &[SharedDataRelation],
    ledger_state: Option<SemanticDigest>,
) -> Result<CompositionVerificationMeta, CompositionVerificationArtifactError> {
    let revalidation =
        revalidate(previous.unwrap_or(current), current, effects).map_err(|e| graph_error(&e))?;
    let mut meta = CompositionVerificationMeta {
        version: COMPOSITION_VERIFICATION_ARTIFACT_VERSION,
        feature: COMPOSITION_VERIFICATION_ARTIFACT_FEATURE.to_owned(),
        required_features: Vec::new(),
        previous: previous.map(graph_to_wire),
        current: graph_to_wire(current),
        effects: effects_to_wire(effects)?,
        shared_data: shared_to_wire(shared),
        ledger_state: ledger_state.map(|digest| digest.to_hex()),
        revalidation: revalidation.identity().to_hex(),
        claims: claims_to_wire(current, &revalidation, shared)?,
        identity: String::new(),
    };
    meta.identity = compute_identity(&meta)?;
    Ok(meta)
}

// -- public type ----------------------------------------------------------------------------------

/// A validated composition verification: the replayed classification of every
/// claim and the graphs and declarations it was made from.
#[derive(Clone, Debug)]
pub struct CompositionVerificationArtifact {
    previous: Option<CompositionGraph>,
    current: CompositionGraph,
    effects: BTreeMap<String, ObservedEffect>,
    shared: Vec<SharedDataRelation>,
    revalidation: GraphRevalidation,
    identity: String,
    ledger_state: Option<SemanticDigest>,
}

impl CompositionVerificationArtifact {
    /// Classify `current` against `previous` (a first verification when
    /// `None`) and seal the result. The returned artifact passes the checks a
    /// consumer applies.
    ///
    /// `shared` and `ledger_state` declare the evidence ledger's overlap
    /// among the nodes' sources; build them with
    /// [`shared_data_relations`] and `EvidenceLedger::state_digest`.
    ///
    /// # Errors
    /// A graph that does not validate, a relation naming unknown nodes or an
    /// encoding failure.
    pub fn seal(
        previous: Option<&CompositionGraph>,
        current: &CompositionGraph,
        effects: &BTreeMap<String, ObservedEffect>,
        shared: &[SharedDataRelation],
        ledger_state: Option<SemanticDigest>,
    ) -> Result<(Self, CompositionVerificationMeta), CompositionVerificationArtifactError> {
        let meta = build_meta(previous, current, effects, shared, ledger_state)?;
        let artifact = check(&meta, None)?;
        Ok((artifact, meta))
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// Anything [`Self::seal`] refuses, or an oversized payload.
    pub fn to_bytes(
        previous: Option<&CompositionGraph>,
        current: &CompositionGraph,
        effects: &BTreeMap<String, ObservedEffect>,
        shared: &[SharedDataRelation],
        ledger_state: Option<SemanticDigest>,
        artifact_id: &str,
    ) -> Result<Vec<u8>, CompositionVerificationArtifactError> {
        let (_, meta) = Self::seal(previous, current, effects, shared, ledger_state)?;
        encode_meta(&meta, artifact_id)
    }

    /// Serialize this verification again through the checksummed container.
    ///
    /// # Errors
    /// A blank artifact id or an oversized payload.
    pub fn export(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, CompositionVerificationArtifactError> {
        let meta = build_meta(
            self.previous.as_ref(),
            &self.current,
            &self.effects,
            &self.shared,
            self.ledger_state,
        )?;
        encode_meta(&meta, artifact_id)
    }

    /// Consume an artifact by classifying again.
    ///
    /// `expected_identity` is an artifact identity (hex) the consumer
    /// retained independently; when given, a consistently resealed artifact
    /// of another verification is refused.
    ///
    /// # Errors
    /// Corruption, another major version, a bound exceeded, a graph the
    /// library refuses, a stored classification, report identity or relation
    /// that does not replay, or an identity that does not match.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, CompositionVerificationArtifactError> {
        check(&decode_meta(bytes)?, expected_identity)
    }

    /// The replayed classification.
    #[must_use]
    pub const fn revalidation(&self) -> &GraphRevalidation {
        &self.revalidation
    }

    /// The current graph.
    #[must_use]
    pub const fn current(&self) -> &CompositionGraph {
        &self.current
    }

    /// The earlier graph, for a verification made against one.
    #[must_use]
    pub const fn previous(&self) -> Option<&CompositionGraph> {
        self.previous.as_ref()
    }

    /// The observed effects the classification was made under.
    #[must_use]
    pub const fn effects(&self) -> &BTreeMap<String, ObservedEffect> {
        &self.effects
    }

    /// The declared shared-data relations.
    #[must_use]
    pub fn shared_data(&self) -> &[SharedDataRelation] {
        &self.shared
    }

    /// The lineage report of one claim.
    ///
    /// # Errors
    /// A key that is not in the current graph.
    pub fn view(&self, key: &str) -> Result<CompositionView, ViewError> {
        CompositionView::of(&self.current, &self.revalidation, key, &self.shared)
    }

    /// State digest of the evidence ledger the shared-data relations were
    /// declared from, when the producer recorded one.
    #[must_use]
    pub const fn ledger_state(&self) -> Option<SemanticDigest> {
        self.ledger_state
    }

    /// BLAKE3 identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Check the declared relations and the ledger state against a ledger the
    /// consumer holds: the relations are recomputed from its row-level
    /// overlap and must equal the declared ones, and when the artifact
    /// declared a state digest the ledger must be at that state.
    ///
    /// # Errors
    /// A ledger at another state, relations that differ, a source the ledger
    /// does not hold, the work bound or cancellation.
    pub fn verify_ledger(
        &self,
        ledger: &EvidenceLedger,
        cancel: &CancellationToken,
    ) -> Result<(), CompositionVerificationArtifactError> {
        if let Some(state) = self.ledger_state {
            if state != ledger.state_digest() {
                return Err(CompositionVerificationArtifactError::LedgerMismatch("state digest"));
            }
        }
        let recomputed =
            shared_data_relations(&self.current, ledger, cancel).map_err(|e| match e {
                ViewError::LimitExceeded => {
                    CompositionVerificationArtifactError::LimitsExceeded("overlap checks")
                }
                other => malformed(other.to_string()),
            })?;
        if recomputed != self.shared {
            return Err(CompositionVerificationArtifactError::LedgerMismatch(
                "shared-data relations",
            ));
        }
        Ok(())
    }

    /// Validate metadata as a consumer would, for tests that reseal edits.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &CompositionVerificationMeta,
        expected_identity: Option<&str>,
    ) -> Result<Self, CompositionVerificationArtifactError> {
        check(meta, expected_identity)
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut CompositionVerificationMeta) {
        if let Ok(identity) = compute_identity(meta) {
            meta.identity = identity;
        }
    }
}
