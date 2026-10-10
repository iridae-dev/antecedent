//! X8 fixed-population temporal counterfactual artifact (`temporal_counterfactual_v1`).
//!
//! A bounded, checksummed, sectioned container holding one temporal counterfactual answer:
//!
//! * `temporal_counterfactual.meta` (CBOR): version, feature marker, mechanism class, the
//!   point-only claim, the snapshot, the temporal graph, the mechanism fit, the unit and
//!   history ids with their observation times, both named action histories, the
//!   shared-abduction receipt (snapshot, graph, fit, factual and per-world digests, one
//!   exogenous-draw digest per unit), the identity digests and the stored means;
//! * `temporal_counterfactual.values` (little-endian `f64`): for each unit in unit-id order,
//!   the five factual values `[L0, A0, L1, A1, Y]`, then the unit's final outcome under the
//!   plus history and under the minus history.
//!
//! A consumer trusts none of the stored outcomes. It rebuilds the request, abduces each unit's
//! exogenous history again, replays **both** worlds against that same history with the core
//! evaluator, and refuses unless every per-unit outcome, both means, the contrast and the whole
//! receipt are bit-identical. The identity digests are recomputed from the same inputs: a
//! changed action time, unit history, snapshot, mechanism fit or graph is refused even when the
//! artifact was resealed consistently, provided the consumer passes the identity it retained.
//!
//! What this does **not** say: the claim is a point for one fixed population under a
//! linear-Gaussian additive-noise two-slice SCM; no interval is published, a transported
//! counterfactual is not identified, and the Markov and model-adequacy premises are named, not
//! checked. Unknown major versions refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Cursor;

use antecedent_core::ExecutionContext;
use antecedent_counterfactual::temporal_cross_world::{
    FactualUnitHistory, HistoryId, MAX_UNITS, MECHANISM_CLASS, NamedActionHistory, NodeMechanism,
    RefusingWitness, SharedAbductionReceipt, SnapshotId, TemporalCounterfactualError,
    TemporalCounterfactualResult, TemporalCounterfactualSpec, TemporalGraph, TemporalMechanismFit,
    TemporalNode, UnitId, evaluate_temporal_counterfactual,
};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// The artifact major version this reader writes and accepts.
pub const TEMPORAL_COUNTERFACTUAL_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const TEMPORAL_COUNTERFACTUAL_ARTIFACT_FEATURE: &str = "temporal_counterfactual_v1";
/// The only claim of this route.
pub const TEMPORAL_COUNTERFACTUAL_INFERENCE_CLAIM: &str = "point_only";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_TEMPORAL_COUNTERFACTUAL_ARTIFACT_BYTES: usize = 128 * 1024 * 1024;

const ARTIFACT_KIND: &str = "temporal_counterfactual_v1";
const META_SECTION: &str = "temporal_counterfactual_meta";
const VALUES_SECTION: &str = "temporal_counterfactual_values";
/// Reals stored per unit: five factual values and the two counterfactual outcomes.
const REALS_PER_UNIT: usize = 7;

/// The refusal a consumer or producer returns, in the shared structured shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefusalWire {
    /// Registered reason code.
    pub code: String,
    /// Refusing stage.
    pub stage: String,
    /// Namespaced detail, `family.slot`.
    pub detail: String,
    /// Offending unit, node or field, when there is one.
    pub offending: Option<String>,
    /// What the caller can change to proceed, when known.
    pub remedy: Option<String>,
    /// The refuting witness, when the core retained one.
    pub witness: Option<WitnessWire>,
}

/// What a refusal retains about the offending unit or history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessWire {
    /// Offending unit.
    pub unit: Option<String>,
    /// Offending factual history.
    pub history: Option<String>,
    /// Action-history name concerned.
    pub world: Option<String>,
    /// Node whose abduction failed.
    pub node: Option<String>,
    /// Abduced residual.
    pub residual: Option<f64>,
    /// Declared noise half-width it violated.
    pub bound: Option<f64>,
}

impl WitnessWire {
    fn from_witness(witness: &RefusingWitness) -> Self {
        Self {
            unit: witness.unit.as_ref().map(|u| u.as_str().to_owned()),
            history: witness.history.as_ref().map(|h| h.as_str().to_owned()),
            world: witness.world.clone(),
            node: witness.node.map(|n| n.name().to_owned()),
            residual: witness.residual,
            bound: witness.bound,
        }
    }
}

/// Why a temporal counterfactual artifact was refused or could not be produced.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TemporalCounterfactualArtifactError {
    /// The bytes do not decode as this format.
    #[error("temporal counterfactual artifact does not decode: {0}")]
    Decode(String),
    /// Another major version, refused before the payload is interpreted.
    #[error("unsupported temporal counterfactual artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u16,
    },
    /// A feature marker, claim or mechanism class that is not this format's.
    #[error("unsupported temporal counterfactual semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A collection exceeds the format's bound.
    #[error("temporal counterfactual consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored declarations are not a well-formed request.
    #[error("malformed temporal counterfactual artifact: {0}")]
    Malformed(String),
    /// A stored or retained identity field differs from the one the inputs produce.
    #[error("temporal_counterfactual.artifact_changed: {field} changed")]
    IdentityMismatch {
        /// Which identity field differs.
        field: &'static str,
    },
    /// A stored outcome, mean or receipt differs from the recomputed one.
    #[error("stored temporal counterfactual does not replay: {0}")]
    ResultMismatch(&'static str),
    /// The artifact could not be encoded.
    #[error("temporal counterfactual artifact does not encode: {0}")]
    Encode(String),
    /// Cooperative cancellation was observed.
    #[error("cancelled during temporal counterfactual evaluation")]
    Cancelled,
    /// The core evaluator refused.
    #[error("temporal counterfactual refused: {}", .0.detail)]
    Refused(Box<RefusalWire>),
}

impl TemporalCounterfactualArtifactError {
    /// The structured refusal this error carries, for a core refusal and a changed identity.
    #[must_use]
    pub fn refusal(&self) -> Option<RefusalWire> {
        match self {
            Self::Refused(wire) => Some((**wire).clone()),
            Self::IdentityMismatch { field } => Some(RefusalWire {
                code: antecedent_core::reason_code!("route_not_supported").to_owned(),
                stage: "consume".to_owned(),
                detail: "temporal_counterfactual.artifact_changed".to_owned(),
                offending: Some((*field).to_owned()),
                remedy: Some(
                    "consume the artifact the producer exported, or retain the identity of the \
                     inputs you intend"
                        .to_owned(),
                ),
                witness: None,
            }),
            _ => None,
        }
    }
}

impl From<crate::IoError> for TemporalCounterfactualArtifactError {
    fn from(error: crate::IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

fn from_core(error: TemporalCounterfactualError) -> TemporalCounterfactualArtifactError {
    match error {
        TemporalCounterfactualError::Cancelled => TemporalCounterfactualArtifactError::Cancelled,
        TemporalCounterfactualError::Refused(refused) => {
            TemporalCounterfactualArtifactError::Refused(Box::new(RefusalWire {
                code: refused.refusal.code.to_owned(),
                stage: refused.refusal.stage.to_owned(),
                detail: refused.refusal.detail.clone(),
                offending: refused.refusal.offending.clone(),
                remedy: refused.refusal.remedy.map(str::to_owned),
                witness: refused.witness.as_ref().map(WitnessWire::from_witness),
            }))
        }
        other => TemporalCounterfactualArtifactError::Malformed(other.to_string()),
    }
}

fn invalid(detail: &str, offending: &str) -> TemporalCounterfactualArtifactError {
    TemporalCounterfactualArtifactError::Refused(Box::new(RefusalWire {
        code: antecedent_core::reason_code!("invalid_argument").to_owned(),
        stage: "evaluate".to_owned(),
        detail: detail.to_owned(),
        offending: Some(offending.to_owned()),
        remedy: None,
        witness: None,
    }))
}

/// The fixed temporal graph on the wire (node names are `covariate_0`, `action_0`,
/// `covariate_1`, `action_1`, `outcome`).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalGraphWire {
    /// Number of slices.
    pub horizon: u32,
    /// Directed edges `(parent, child)` by node name.
    pub edges: Vec<(String, String)>,
    /// Whether the graph has latent confounding.
    pub latent_confounding: bool,
}

/// One node's linear additive-noise mechanism on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismWire {
    /// The node it generates.
    pub node: String,
    /// Intercept.
    pub intercept: f64,
    /// Parent coefficients by node name.
    pub parent_coefficients: Vec<(String, f64)>,
    /// Optional half-width of the noise support.
    pub noise_halfwidth: Option<f64>,
}

/// The mechanism fit on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitWire {
    /// Stable identity of the fit.
    pub fit_id: String,
    /// The mechanisms of `covariate_0`, `covariate_1` and `outcome`.
    pub mechanisms: Vec<MechanismWire>,
}

/// One unit's factual trajectory on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactualUnitWire {
    /// The unit.
    pub unit: String,
    /// The unit's factual history id.
    pub history: String,
    /// Observation times of the two slices.
    pub times: [u32; 2],
    /// Observed `[L0, A0, L1, A1, Y]`.
    pub values: [f64; 5],
}

/// A named two-slice action history on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionHistoryWire {
    /// Name of the history.
    pub name: String,
    /// Time of each slice's action.
    pub times: [u32; 2],
    /// The actions `[A0, A1]`.
    pub actions: [f64; 2],
    /// The units the world is evaluated for.
    pub units: Vec<String>,
}

/// A complete temporal counterfactual request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalCounterfactualRequestWire {
    /// The fixed temporal graph.
    pub graph: TemporalGraphWire,
    /// The mechanism fit.
    pub fit: FitWire,
    /// Snapshot the factual histories were read from.
    pub snapshot: String,
    /// Factual histories, one per unit.
    pub factual: Vec<FactualUnitWire>,
    /// The added action history.
    pub plus: ActionHistoryWire,
    /// The subtracted action history.
    pub minus: ActionHistoryWire,
}

/// One unit's exogenous-draw digest on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitDrawWire {
    /// The unit.
    pub unit: String,
    /// The unit's factual history.
    pub history: String,
    /// Digest of the unit's abduced exogenous draw.
    pub digest: String,
}

/// The shared-abduction receipt on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptWire {
    /// Snapshot identity.
    pub snapshot: String,
    /// Digest of the temporal graph.
    pub graph_digest: String,
    /// Digest of the mechanism fit.
    pub fit_digest: String,
    /// Digest of the factual histories.
    pub factual_digest: String,
    /// Digest of the plus action history.
    pub plus_digest: String,
    /// Digest of the minus action history.
    pub minus_digest: String,
    /// Per-unit exogenous-draw digests, sorted by unit id.
    pub unit_draws: Vec<UnitDrawWire>,
    /// Digest over all unit draws.
    pub exogenous_digest: String,
    /// Number of units.
    pub n_units: u32,
    /// Number of worlds.
    pub n_worlds: u32,
    /// Horizon.
    pub horizon: u32,
    /// Both worlds read the same draw for every unit.
    pub shared_by_both_worlds: bool,
    /// Overall receipt digest.
    pub digest: String,
}

impl ReceiptWire {
    fn from_receipt(receipt: &SharedAbductionReceipt) -> Self {
        let small = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        Self {
            snapshot: receipt.snapshot.as_str().to_owned(),
            graph_digest: receipt.graph_digest.clone(),
            fit_digest: receipt.fit_digest.clone(),
            factual_digest: receipt.factual_digest.clone(),
            plus_digest: receipt.plus_digest.clone(),
            minus_digest: receipt.minus_digest.clone(),
            unit_draws: receipt
                .unit_draws
                .iter()
                .map(|d| UnitDrawWire {
                    unit: d.unit.as_str().to_owned(),
                    history: d.history.as_str().to_owned(),
                    digest: d.digest.clone(),
                })
                .collect(),
            exogenous_digest: receipt.exogenous_digest.clone(),
            n_units: small(receipt.n_units),
            n_worlds: small(receipt.n_worlds),
            horizon: small(receipt.horizon),
            shared_by_both_worlds: receipt.shared_by_both_worlds,
            digest: receipt.digest.clone(),
        }
    }
}

/// Identity digests of one temporal counterfactual: the receipt's component digests plus a
/// BLAKE3 digest over the canonical request.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalCounterfactualIdentity {
    /// Snapshot id.
    pub snapshot: String,
    /// Temporal graph digest.
    pub graph_digest: String,
    /// Mechanism fit digest.
    pub fit_digest: String,
    /// Unit-history digest (ids, histories, times and observed values).
    pub factual_digest: String,
    /// Plus action history digest (name, times, actions, units).
    pub plus_digest: String,
    /// Minus action history digest.
    pub minus_digest: String,
    /// Digest over every unit's exogenous draw.
    pub exogenous_digest: String,
    /// Overall receipt digest.
    pub receipt_digest: String,
    /// BLAKE3 over the canonical request.
    pub spec_id: String,
}

/// One unit's identifiers and observation times (the values are in the values section).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitMeta {
    /// The unit.
    pub unit: String,
    /// The unit's factual history id.
    pub history: String,
    /// Observation times of the two slices.
    pub times: [u32; 2],
}

/// The CBOR metadata section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalCounterfactualMeta {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Mechanism class of the route.
    pub mechanism_class: String,
    /// Always `point_only`.
    pub inference_claim: String,
    /// The temporal graph.
    pub graph: TemporalGraphWire,
    /// The mechanism fit.
    pub fit: FitWire,
    /// Snapshot id.
    pub snapshot: String,
    /// Units in unit-id order.
    pub units: Vec<UnitMeta>,
    /// The plus action history.
    pub plus: ActionHistoryWire,
    /// The minus action history.
    pub minus: ActionHistoryWire,
    /// The shared-abduction receipt.
    pub receipt: ReceiptWire,
    /// Identity digests.
    pub identity: TemporalCounterfactualIdentity,
    /// Mean final outcome under the plus history.
    pub mean_plus: f64,
    /// Mean final outcome under the minus history.
    pub mean_minus: f64,
    /// Sample mean contrast.
    pub mean_contrast: f64,
}

/// One unit's counterfactual outcomes on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitOutcomeWire {
    /// The unit.
    pub unit: String,
    /// The unit's factual history.
    pub history: String,
    /// The factual final outcome.
    pub factual_outcome: f64,
    /// Final outcome under the plus history.
    pub plus_outcome: f64,
    /// Final outcome under the minus history.
    pub minus_outcome: f64,
}

/// The full, self-describing report handed to host languages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalCounterfactualReportWire {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Mechanism class.
    pub mechanism_class: String,
    /// Always `point_only`.
    pub inference_claim: String,
    /// Snapshot id.
    pub snapshot: String,
    /// Name of the plus action history.
    pub plus_name: String,
    /// Name of the minus action history.
    pub minus_name: String,
    /// Per-unit outcomes, sorted by unit id.
    pub units: Vec<UnitOutcomeWire>,
    /// Mean final outcome under the plus history.
    pub mean_plus: f64,
    /// Mean final outcome under the minus history.
    pub mean_minus: f64,
    /// Sample mean contrast.
    pub mean_contrast: f64,
    /// The shared-abduction receipt.
    pub receipt: ReceiptWire,
    /// Identity digests.
    pub identity: TemporalCounterfactualIdentity,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u16,
}

fn node_of(name: &str) -> Result<TemporalNode, TemporalCounterfactualArtifactError> {
    TemporalNode::ALL
        .iter()
        .copied()
        .find(|node| node.name() == name)
        .ok_or_else(|| invalid("temporal_counterfactual.invalid_graph", name))
}

fn spec_of(
    request: &TemporalCounterfactualRequestWire,
) -> Result<TemporalCounterfactualSpec, TemporalCounterfactualArtifactError> {
    type Checked<T> = Result<T, TemporalCounterfactualArtifactError>;
    let edges = request
        .graph
        .edges
        .iter()
        .map(|(p, c)| -> Checked<(TemporalNode, TemporalNode)> { Ok((node_of(p)?, node_of(c)?)) })
        .collect::<Checked<Vec<_>>>()?;
    let mechanisms = request
        .fit
        .mechanisms
        .iter()
        .map(|m| -> Checked<NodeMechanism> {
            Ok(NodeMechanism {
                node: node_of(&m.node)?,
                intercept: m.intercept,
                parent_coefficients: m
                    .parent_coefficients
                    .iter()
                    .map(|(p, c)| -> Checked<(TemporalNode, f64)> { Ok((node_of(p)?, *c)) })
                    .collect::<Checked<Vec<_>>>()?,
                noise_halfwidth: m.noise_halfwidth,
            })
        })
        .collect::<Checked<Vec<_>>>()?;
    let world = |w: &ActionHistoryWire| NamedActionHistory {
        name: w.name.clone(),
        times: w.times,
        actions: w.actions,
        units: w.units.iter().map(UnitId::new).collect(),
    };
    Ok(TemporalCounterfactualSpec {
        graph: TemporalGraph {
            horizon: usize::try_from(request.graph.horizon).unwrap_or(usize::MAX),
            edges,
            latent_confounding: request.graph.latent_confounding,
        },
        fit: TemporalMechanismFit { fit_id: request.fit.fit_id.clone(), mechanisms },
        snapshot: SnapshotId::new(request.snapshot.as_str()),
        factual: request
            .factual
            .iter()
            .map(|f| FactualUnitHistory {
                unit: UnitId::new(f.unit.as_str()),
                history: HistoryId::new(f.history.as_str()),
                times: f.times,
                values: f.values,
            })
            .collect(),
        plus: world(&request.plus),
        minus: world(&request.minus),
    })
}

/// Length-prefixed canonical byte builder hashed with BLAKE3.
struct Canon(Vec<u8>);

impl Canon {
    fn new(tag: &str) -> Self {
        let mut canon = Self(Vec::new());
        canon.text(tag);
        canon
    }

    fn word(&mut self, word: u64) {
        self.0.extend_from_slice(&word.to_le_bytes());
    }

    fn text(&mut self, text: &str) {
        self.word(text.len() as u64);
        self.0.extend_from_slice(text.as_bytes());
    }

    fn real(&mut self, value: f64) {
        self.word(value.to_bits());
    }

    fn finish(&self) -> String {
        blake3::hash(&self.0).to_hex().to_string()
    }
}

fn spec_id(request: &TemporalCounterfactualRequestWire) -> String {
    let mut c = Canon::new("temporal_counterfactual_v1.spec");
    c.text(&request.snapshot);
    c.word(u64::from(request.graph.horizon));
    c.word(u64::from(request.graph.latent_confounding));
    let mut edges = request.graph.edges.clone();
    edges.sort();
    for (p, ch) in &edges {
        c.text(p);
        c.text(ch);
    }
    c.text(&request.fit.fit_id);
    let mut mechanisms: Vec<&MechanismWire> = request.fit.mechanisms.iter().collect();
    mechanisms.sort_by(|a, b| a.node.cmp(&b.node));
    for m in mechanisms {
        c.text(&m.node);
        c.real(m.intercept);
        let mut coefs = m.parent_coefficients.clone();
        coefs.sort_by(|a, b| a.0.cmp(&b.0));
        for (p, coef) in &coefs {
            c.text(p);
            c.real(*coef);
        }
        match m.noise_halfwidth {
            Some(h) => {
                c.word(1);
                c.real(h);
            }
            None => c.word(0),
        }
    }
    let mut factual: Vec<&FactualUnitWire> = request.factual.iter().collect();
    factual.sort_by(|a, b| a.unit.cmp(&b.unit));
    c.word(factual.len() as u64);
    for f in factual {
        c.text(&f.unit);
        c.text(&f.history);
        c.word(u64::from(f.times[0]));
        c.word(u64::from(f.times[1]));
        for v in f.values {
            c.real(v);
        }
    }
    for w in [&request.plus, &request.minus] {
        c.text(&w.name);
        c.word(u64::from(w.times[0]));
        c.word(u64::from(w.times[1]));
        c.real(w.actions[0]);
        c.real(w.actions[1]);
        let mut units = w.units.clone();
        units.sort();
        for u in &units {
            c.text(u);
        }
    }
    c.finish()
}

fn identity_of(
    request: &TemporalCounterfactualRequestWire,
    receipt: &SharedAbductionReceipt,
) -> TemporalCounterfactualIdentity {
    TemporalCounterfactualIdentity {
        snapshot: receipt.snapshot.as_str().to_owned(),
        graph_digest: receipt.graph_digest.clone(),
        fit_digest: receipt.fit_digest.clone(),
        factual_digest: receipt.factual_digest.clone(),
        plus_digest: receipt.plus_digest.clone(),
        minus_digest: receipt.minus_digest.clone(),
        exogenous_digest: receipt.exogenous_digest.clone(),
        receipt_digest: receipt.digest.clone(),
        spec_id: spec_id(request),
    }
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &TemporalCounterfactualIdentity,
    other: &TemporalCounterfactualIdentity,
) -> Option<&'static str> {
    if stored.snapshot != other.snapshot {
        Some("snapshot")
    } else if stored.graph_digest != other.graph_digest {
        Some("graph")
    } else if stored.fit_digest != other.fit_digest {
        Some("mechanism_fit")
    } else if stored.factual_digest != other.factual_digest {
        Some("unit_history")
    } else if stored.plus_digest != other.plus_digest || stored.minus_digest != other.minus_digest {
        Some("action_history")
    } else if stored.exogenous_digest != other.exogenous_digest {
        Some("shared_abduction")
    } else if stored.receipt_digest != other.receipt_digest {
        Some("receipt")
    } else if stored.spec_id != other.spec_id {
        Some("request")
    } else {
        None
    }
}

fn values_of(
    request: &TemporalCounterfactualRequestWire,
    result: &TemporalCounterfactualResult,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(request.factual.len() * REALS_PER_UNIT * 8);
    for (factual, outcome) in request.factual.iter().zip(&result.units) {
        for v in factual.values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&outcome.plus_outcome.to_le_bytes());
        bytes.extend_from_slice(&outcome.minus_outcome.to_le_bytes());
    }
    bytes
}

fn reals_of(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            let mut word = [0_u8; 8];
            word.copy_from_slice(chunk);
            f64::from_le_bytes(word)
        })
        .collect()
}

/// Encode a metadata section and a values section as a checksummed container.
///
/// Hidden: the producer path is [`TemporalCounterfactualArtifact::to_bytes`]; tests use this
/// to build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &TemporalCounterfactualMeta,
    values: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, TemporalCounterfactualArtifactError> {
    let encode = |e: crate::IoError| TemporalCounterfactualArtifactError::Encode(e.to_string());
    if artifact_id.trim().is_empty() {
        return Err(TemporalCounterfactualArtifactError::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(encode)?;
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: crate::migrate::STABLE_FORMAT,
            minimum_reader_version: crate::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
                .map_err(encode)?,
            artifact_id: artifact_id.into(),
            sections: vec![
                section_descriptor(META_SECTION, "application/cbor", &meta_bytes),
                section_descriptor(VALUES_SECTION, "application/octet-stream", values),
            ],
            provenance: ProvenanceWire {
                note: "temporal_counterfactual_shared_abduction_point_only".into(),
            },
        },
        sections: vec![
            SectionBytes::new(META_SECTION, meta_bytes),
            SectionBytes::new(VALUES_SECTION, values.to_vec()),
        ],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).map_err(encode)?;
    if bytes.len() > MAX_TEMPORAL_COUNTERFACTUAL_ARTIFACT_BYTES {
        return Err(TemporalCounterfactualArtifactError::LimitsExceeded("artifact bytes"));
    }
    Ok(bytes)
}

/// Decode the two sections of a container, refusing another version before the metadata is
/// interpreted.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[doc(hidden)]
pub fn decode_parts(
    bytes: &[u8],
) -> Result<(TemporalCounterfactualMeta, Vec<u8>), TemporalCounterfactualArtifactError> {
    if bytes.len() > MAX_TEMPORAL_COUNTERFACTUAL_ARTIFACT_BYTES {
        return Err(TemporalCounterfactualArtifactError::LimitsExceeded("artifact bytes"));
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
        || manifest.sections.len() != 2
        || manifest.sections[0].id != META_SECTION
        || manifest.sections[1].id != VALUES_SECTION
    {
        return Err(TemporalCounterfactualArtifactError::Malformed(
            "unsupported container layout".into(),
        ));
    }
    let declared = manifest
        .sections
        .iter()
        .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size));
    if declared.is_none_or(|total| total > MAX_TEMPORAL_COUNTERFACTUAL_ARTIFACT_BYTES as u64) {
        return Err(TemporalCounterfactualArtifactError::LimitsExceeded("artifact bytes"));
    }
    let meta_section = reader.load_section(META_SECTION)?;
    let peek: VersionPeek = from_cbor(meta_section.as_bytes())?;
    if peek.version != TEMPORAL_COUNTERFACTUAL_ARTIFACT_VERSION {
        return Err(TemporalCounterfactualArtifactError::UnsupportedVersion {
            version: peek.version,
        });
    }
    let meta: TemporalCounterfactualMeta = from_cbor(meta_section.as_bytes())?;
    let values = reader.load_section(VALUES_SECTION)?;
    Ok((meta, values.as_bytes().to_vec()))
}

fn request_of(
    meta: &TemporalCounterfactualMeta,
    values: &[u8],
) -> Result<TemporalCounterfactualRequestWire, TemporalCounterfactualArtifactError> {
    let n = meta.units.len();
    if n > MAX_UNITS {
        return Err(TemporalCounterfactualArtifactError::LimitsExceeded("units"));
    }
    if values.len() != n * REALS_PER_UNIT * 8 {
        return Err(TemporalCounterfactualArtifactError::Malformed("values section length".into()));
    }
    let reals = reals_of(values);
    let factual = meta
        .units
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let row = &reals[i * REALS_PER_UNIT..i * REALS_PER_UNIT + 5];
            FactualUnitWire {
                unit: u.unit.clone(),
                history: u.history.clone(),
                times: u.times,
                values: [row[0], row[1], row[2], row[3], row[4]],
            }
        })
        .collect();
    Ok(TemporalCounterfactualRequestWire {
        graph: meta.graph.clone(),
        fit: meta.fit.clone(),
        snapshot: meta.snapshot.clone(),
        factual,
        plus: meta.plus.clone(),
        minus: meta.minus.clone(),
    })
}

/// A produced or consumed temporal counterfactual artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalCounterfactualArtifact {
    meta: TemporalCounterfactualMeta,
    request: TemporalCounterfactualRequestWire,
    result: TemporalCounterfactualResult,
}

impl TemporalCounterfactualArtifact {
    /// Evaluate `request` with the core evaluator and seal the answer with its receipt.
    ///
    /// The request is canonicalized to unit-id order.
    ///
    /// # Errors
    /// The core evaluator's refusals (`temporal_counterfactual.unpaired_histories`,
    /// `.time_misaligned`, `.refuting_history`, ...), cancellation and the format's bounds.
    pub fn seal(
        request: &TemporalCounterfactualRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, TemporalCounterfactualArtifactError> {
        if request.factual.len() > MAX_UNITS {
            return Err(TemporalCounterfactualArtifactError::LimitsExceeded("units"));
        }
        let mut request = request.clone();
        request.factual.sort_by(|a, b| a.unit.cmp(&b.unit));
        let spec = spec_of(&request)?;
        let result = evaluate_temporal_counterfactual(&spec, ctx).map_err(from_core)?;
        let meta = TemporalCounterfactualMeta {
            version: TEMPORAL_COUNTERFACTUAL_ARTIFACT_VERSION,
            feature: TEMPORAL_COUNTERFACTUAL_ARTIFACT_FEATURE.to_owned(),
            mechanism_class: MECHANISM_CLASS.to_owned(),
            inference_claim: TEMPORAL_COUNTERFACTUAL_INFERENCE_CLAIM.to_owned(),
            graph: request.graph.clone(),
            fit: request.fit.clone(),
            snapshot: request.snapshot.clone(),
            units: request
                .factual
                .iter()
                .map(|f| UnitMeta {
                    unit: f.unit.clone(),
                    history: f.history.clone(),
                    times: f.times,
                })
                .collect(),
            plus: request.plus.clone(),
            minus: request.minus.clone(),
            receipt: ReceiptWire::from_receipt(&result.receipt),
            identity: identity_of(&request, &result.receipt),
            mean_plus: result.mean_plus,
            mean_minus: result.mean_minus,
            mean_contrast: result.mean_contrast,
        };
        Ok(Self { meta, request, result })
    }

    /// The metadata section.
    #[must_use]
    pub fn meta(&self) -> &TemporalCounterfactualMeta {
        &self.meta
    }

    /// The canonical (unit-id ordered) request.
    #[must_use]
    pub fn request(&self) -> &TemporalCounterfactualRequestWire {
        &self.request
    }

    /// The core result.
    #[must_use]
    pub fn result(&self) -> &TemporalCounterfactualResult {
        &self.result
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &TemporalCounterfactualIdentity {
        &self.meta.identity
    }

    /// The self-describing report for host languages.
    #[must_use]
    pub fn report(&self) -> TemporalCounterfactualReportWire {
        TemporalCounterfactualReportWire {
            version: self.meta.version,
            feature: self.meta.feature.clone(),
            mechanism_class: self.meta.mechanism_class.clone(),
            inference_claim: self.meta.inference_claim.clone(),
            snapshot: self.meta.snapshot.clone(),
            plus_name: self.meta.plus.name.clone(),
            minus_name: self.meta.minus.name.clone(),
            units: self
                .result
                .units
                .iter()
                .map(|u| UnitOutcomeWire {
                    unit: u.unit.as_str().to_owned(),
                    history: u.history.as_str().to_owned(),
                    factual_outcome: u.factual_outcome,
                    plus_outcome: u.plus_outcome,
                    minus_outcome: u.minus_outcome,
                })
                .collect(),
            mean_plus: self.meta.mean_plus,
            mean_minus: self.meta.mean_minus,
            mean_contrast: self.meta.mean_contrast,
            receipt: self.meta.receipt.clone(),
            identity: self.meta.identity.clone(),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, TemporalCounterfactualArtifactError> {
        encode_parts(&self.meta, &values_of(&self.request, &self.result), artifact_id)
    }

    /// Consume an artifact by recomputing both worlds.
    ///
    /// `expected` is an identity the consumer retained independently; when given, every field
    /// must match it, so a resealed change of action time, unit history, snapshot, mechanism fit
    /// or graph is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a changed identity, a stored
    /// outcome or receipt that does not replay, cancellation, or a core refusal.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&TemporalCounterfactualIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, TemporalCounterfactualArtifactError> {
        let (meta, values) = decode_parts(bytes)?;
        if meta.feature != TEMPORAL_COUNTERFACTUAL_ARTIFACT_FEATURE {
            return Err(TemporalCounterfactualArtifactError::UnsupportedSemantics(
                "feature marker",
            ));
        }
        if meta.mechanism_class != MECHANISM_CLASS {
            return Err(TemporalCounterfactualArtifactError::UnsupportedSemantics(
                "mechanism class",
            ));
        }
        if meta.inference_claim != TEMPORAL_COUNTERFACTUAL_INFERENCE_CLAIM {
            return Err(TemporalCounterfactualArtifactError::UnsupportedSemantics(
                "inference claim",
            ));
        }
        let request = request_of(&meta, &values)?;
        let spec = spec_of(&request)?;
        let result = evaluate_temporal_counterfactual(&spec, ctx).map_err(from_core)?;
        let recomputed = identity_of(&request, &result.receipt);
        if let Some(field) = identity_diff(&meta.identity, &recomputed) {
            return Err(TemporalCounterfactualArtifactError::IdentityMismatch { field });
        }
        if meta.receipt != ReceiptWire::from_receipt(&result.receipt) {
            return Err(TemporalCounterfactualArtifactError::ResultMismatch(
                "shared-abduction receipt",
            ));
        }
        let stored = reals_of(&values);
        let outcomes_replay = result.units.iter().enumerate().all(|(i, unit)| {
            let row = &stored[i * REALS_PER_UNIT..(i + 1) * REALS_PER_UNIT];
            row[5].to_bits() == unit.plus_outcome.to_bits()
                && row[6].to_bits() == unit.minus_outcome.to_bits()
        });
        if !outcomes_replay || result.units.len() != meta.units.len() {
            return Err(TemporalCounterfactualArtifactError::ResultMismatch("per-unit outcomes"));
        }
        let means_replay = meta.mean_plus.to_bits() == result.mean_plus.to_bits()
            && meta.mean_minus.to_bits() == result.mean_minus.to_bits()
            && meta.mean_contrast.to_bits() == result.mean_contrast.to_bits();
        if !means_replay {
            return Err(TemporalCounterfactualArtifactError::ResultMismatch("means"));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &recomputed) {
                return Err(TemporalCounterfactualArtifactError::IdentityMismatch { field });
            }
        }
        Ok(Self { meta, request, result })
    }
}
