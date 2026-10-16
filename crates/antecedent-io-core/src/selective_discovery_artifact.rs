//! Portable selective-discovery state (`selective_discovery_v1`).
//!
//! A bounded, checksummed container holding the configuration identities, the statistic
//! receipts and the graph and result identities of one discovery run. It carries no raw
//! data.
//!
//! # What a consumer does
//!
//! A consumer of this crate trusts none of the stored identities. It rebuilds the receipts
//! and the structure, recomputes the receipt identity, the graph identity, the contract
//! identity and the artifact identity, and refuses a stored identity that differs, a newer
//! format version, a feature marker it does not know, any required feature this version does
//! not define, and any inference label other than the one this format states.
//!
//! Replay is separate and needs the raw data: `StaticSelectiveDiscovery::resume` (or the
//! temporal counterpart) in `antecedent-discovery` replays the algorithm over the receipts
//! only and must reach the recorded result identity. A consumer without the data can inspect
//! the artifact and nothing more; absence of data is a refusal to resume, never a waiver.
//!
//! # What it protects against, and what it does not
//!
//! Corruption and edits, sealed or not: an edited receipt that is resealed still has to
//! reproduce the recorded result when replayed, and a consumer that retained the artifact
//! identity independently refuses a consistently resealed artifact of another run. It does
//! not make the discovery true: the label this format carries is `algorithmic_match_only`.
//! No recovery, error-control or interval statement is stored or implied.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CanonicalWriter;
use antecedent_discovery::selective::{
    DiscoveredStructure, MAX_PORTABLE_RECEIPTS, Mark, PortableDiscoveryState,
    SelectiveDiscoveryError, StatisticReceipt, StructureClass, StructureEdge, StructureNode,
    WorkUnits, digest_from_hex, digest_hex,
};
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;

/// The artifact major version this reader writes and accepts.
pub const SELECTIVE_DISCOVERY_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const SELECTIVE_DISCOVERY_ARTIFACT_FEATURE: &str = "selective_discovery_v1";
/// The only inference label this format states.
pub const SELECTIVE_DISCOVERY_INFERENCE: &str = "algorithmic_match_only";
/// Most bytes a selective-discovery artifact may occupy.
pub const MAX_SELECTIVE_DISCOVERY_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
/// Most nodes and most edges a stored structure may declare.
pub const MAX_STRUCTURE_ITEMS: usize = 1 << 16;

const ARTIFACT_KIND: &str = crate::artifact_kinds::SELECTIVE_DISCOVERY;
const META_SECTION: &str = "selective_discovery_meta";
/// Domain tag of the artifact identity stream.
pub const SELECTIVE_DISCOVERY_ARTIFACT_TAG: &str = "antecedent.selective_discovery.artifact";

crate::sectioned_artifact_enum! {
    /// Why a selective-discovery artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    SelectiveDiscoveryArtifactError {
        noun: "selective discovery",
        identity: "selective_discovery.identity_mismatch",
        variants {
            /// The state is not a usable record of one run.
            Refused {
                /// Registered reason code.
                code: &'static str,
                /// Namespaced `selective_discovery.*` detail.
                detail: String,
                /// The declaration or action that would make the state usable.
                message: String,
            },
        }
        display(f) {
            Self::Refused { detail, message, .. } => write!(f, "{detail}: {message}"),
        }
    }
}

impl SelectiveDiscoveryArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::Refused { code, detail, message } => {
                Some((*code, detail.clone(), message.clone()))
            }
            Self::IdentityMismatch { field } => Some((
                antecedent_core::reason_code!("discovery_statistics_invalid"),
                "selective_discovery.identity_mismatch".to_owned(),
                format!("{field} changed"),
            )),
            _ => None,
        }
    }
}

type Failure = SelectiveDiscoveryArtifactError;

fn malformed(text: impl Into<String>) -> Failure {
    Failure::Malformed(text.into())
}

fn refused(error: &SelectiveDiscoveryError) -> Failure {
    match error {
        SelectiveDiscoveryError::Refused { code, detail, remedy } => {
            Failure::Refused { code, detail: (*detail).to_owned(), message: remedy.clone() }
        }
        other => malformed(other.to_string()),
    }
}

// -- wire types ---------------------------------------------------------------------------

/// A structure node on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructureNodeWire {
    /// `0` static, `1` lagged, `2` unfolded, `3` context.
    pub kind: u8,
    /// Variable id.
    pub variable: u32,
    /// Lag magnitude, offset or environment id.
    pub aux: i64,
}

/// A marked edge on the wire; endpoints ascending.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructureEdgeWire {
    /// First endpoint.
    pub a: StructureNodeWire,
    /// Second endpoint.
    pub b: StructureNodeWire,
    /// Mark at `a`: `tail`, `arrow`, `circle` or `conflict`.
    pub at_a: String,
    /// Mark at `b`.
    pub at_b: String,
    /// Search middle-mark tag (`4` = definite).
    pub middle: u8,
}

/// The structure on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructureWire {
    /// `cpdag`, `pag`, `temporal_dag` or `temporal_cpdag`.
    pub class: String,
    /// Nodes ascending.
    pub nodes: Vec<StructureNodeWire>,
    /// Edges ascending.
    pub edges: Vec<StructureEdgeWire>,
}

/// One retained statistic on the wire; floats as bit patterns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptWire {
    /// Coordinate digest (hex).
    pub coordinate: String,
    /// Statistic bits.
    pub statistic_bits: u64,
    /// P-value bits.
    pub p_value_bits: u64,
    /// Degrees-of-freedom bits.
    pub df_bits: u64,
    /// Interval endpoint bits.
    pub interval_bits: Option<(u64, u64)>,
}

/// The producer's work counters on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkWire {
    /// CI tests handed to the test implementation.
    pub ci_tests_attempted: u64,
    /// CI tests that returned a result.
    pub ci_tests_completed: u64,
    /// Requests answered from a retained statistic.
    pub ci_tests_reused: u64,
    /// CI tests the algorithm consumed.
    pub ci_tests_consumed: u64,
    /// Search passes.
    pub search_passes: u64,
    /// Orientation fixed-point drives.
    pub orientation_runs: u64,
    /// Orientation rounds.
    pub orientation_rounds: u64,
    /// Orientations applied.
    pub orientations_applied: u64,
    /// Score evaluations (zero on every delivered route).
    pub score_evaluations: u64,
    /// Algorithm entries (a wrapper count, not a work unit).
    pub algorithm_executions: u64,
}

/// Stored metadata of a selective-discovery artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectiveDiscoveryMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Features a reader must understand. This version defines none.
    pub required_features: Vec<String>,
    /// The inference label; always `algorithmic_match_only`.
    pub inference: String,
    /// Route label.
    pub route: String,
    /// Declared stage inputs as `(name, digest hex)` in canonical name order.
    pub components: Vec<(String, String)>,
    /// Contract identity (hex).
    pub contract_identity: String,
    /// Data identity (hex).
    pub data_identity: String,
    /// Rows or time points of the data.
    pub rows: u64,
    /// Searched variables in search order.
    pub variables: Vec<u32>,
    /// Retained statistics, ascending by coordinate.
    pub receipts: Vec<ReceiptWire>,
    /// Receipt list identity (hex).
    pub receipts_identity: String,
    /// The discovered structure.
    pub structure: StructureWire,
    /// Structure identity (hex).
    pub graph_identity: String,
    /// Result identity (hex).
    pub result_identity: String,
    /// Whether a search bound was exhausted.
    pub search_incomplete: bool,
    /// Whether orientation evidence conflicted.
    pub orientation_conflict: bool,
    /// `(code, message)` of the bounded-search diagnostics.
    pub bounded_search_receipts: Vec<(String, String)>,
    /// The producer's counters.
    pub work: WorkWire,
    /// BLAKE3 identity (hex) of the artifact.
    pub identity: String,
}

// -- state <-> wire -------------------------------------------------------------------------

fn node_to_wire(node: StructureNode) -> StructureNodeWire {
    StructureNodeWire { kind: node.kind, variable: node.variable, aux: node.aux }
}

fn node_from_wire(wire: &StructureNodeWire) -> Result<StructureNode, Failure> {
    if wire.kind > 3 {
        return Err(Failure::UnsupportedSemantics("structure node kind"));
    }
    Ok(StructureNode { kind: wire.kind, variable: wire.variable, aux: wire.aux })
}

/// The wire form of a structure.
#[must_use]
pub fn structure_to_wire(structure: &DiscoveredStructure) -> StructureWire {
    StructureWire {
        class: structure.class().as_str().to_owned(),
        nodes: structure.nodes().iter().copied().map(node_to_wire).collect(),
        edges: structure
            .edges()
            .iter()
            .map(|e| StructureEdgeWire {
                a: node_to_wire(e.a),
                b: node_to_wire(e.b),
                at_a: e.at_a.as_str().to_owned(),
                at_b: e.at_b.as_str().to_owned(),
                middle: e.middle,
            })
            .collect(),
    }
}

fn structure_from_wire(wire: &StructureWire) -> Result<DiscoveredStructure, Failure> {
    if wire.nodes.len() > MAX_STRUCTURE_ITEMS || wire.edges.len() > MAX_STRUCTURE_ITEMS {
        return Err(Failure::LimitsExceeded("structure"));
    }
    let class = StructureClass::from_name(&wire.class)
        .ok_or(Failure::UnsupportedSemantics("structure class"))?;
    let nodes = wire.nodes.iter().map(node_from_wire).collect::<Result<Vec<_>, _>>()?;
    let mut edges = Vec::with_capacity(wire.edges.len());
    for edge in &wire.edges {
        let mark = |name: &str| {
            Mark::from_name(name).ok_or(Failure::UnsupportedSemantics("endpoint mark"))
        };
        edges.push(StructureEdge {
            a: node_from_wire(&edge.a)?,
            b: node_from_wire(&edge.b)?,
            at_a: mark(&edge.at_a)?,
            at_b: mark(&edge.at_b)?,
            middle: edge.middle,
        });
    }
    DiscoveredStructure::from_parts(class, nodes, edges)
        .ok_or_else(|| malformed("structure is not in canonical order or names an unknown node"))
}

fn receipt_to_wire(receipt: &StatisticReceipt) -> ReceiptWire {
    ReceiptWire {
        coordinate: digest_hex(&receipt.coordinate),
        statistic_bits: receipt.statistic.to_bits(),
        p_value_bits: receipt.p_value.to_bits(),
        df_bits: receipt.df.to_bits(),
        interval_bits: receipt.interval.map(|(lo, hi)| (lo.to_bits(), hi.to_bits())),
    }
}

fn receipt_from_wire(wire: &ReceiptWire) -> Result<StatisticReceipt, Failure> {
    let coordinate = digest_from_hex(&wire.coordinate)
        .ok_or_else(|| malformed("a receipt coordinate is not 64 lowercase hex digits"))?;
    Ok(StatisticReceipt {
        coordinate,
        statistic: f64::from_bits(wire.statistic_bits),
        p_value: f64::from_bits(wire.p_value_bits),
        df: f64::from_bits(wire.df_bits),
        interval: wire.interval_bits.map(|(lo, hi)| (f64::from_bits(lo), f64::from_bits(hi))),
    })
}

fn work_to_wire(work: &WorkUnits) -> WorkWire {
    WorkWire {
        ci_tests_attempted: work.ci_tests_attempted,
        ci_tests_completed: work.ci_tests_completed,
        ci_tests_reused: work.ci_tests_reused,
        ci_tests_consumed: work.ci_tests_consumed,
        search_passes: work.search_passes,
        orientation_runs: work.orientation_runs,
        orientation_rounds: work.orientation_rounds,
        orientations_applied: work.orientations_applied,
        score_evaluations: work.score_evaluations,
        algorithm_executions: work.algorithm_executions,
    }
}

fn work_from_wire(wire: &WorkWire) -> WorkUnits {
    WorkUnits {
        ci_tests_attempted: wire.ci_tests_attempted,
        ci_tests_completed: wire.ci_tests_completed,
        ci_tests_reused: wire.ci_tests_reused,
        ci_tests_consumed: wire.ci_tests_consumed,
        search_passes: wire.search_passes,
        orientation_runs: wire.orientation_runs,
        orientation_rounds: wire.orientation_rounds,
        orientations_applied: wire.orientations_applied,
        score_evaluations: wire.score_evaluations,
        algorithm_executions: wire.algorithm_executions,
    }
}

fn hex_digest(text: &str, what: &'static str) -> Result<[u8; 32], Failure> {
    digest_from_hex(text).ok_or_else(|| malformed(format!("{what} is not 64 lowercase hex digits")))
}

fn compute_identity(meta: &SelectiveDiscoveryMeta) -> String {
    let mut writer = CanonicalWriter::new(SELECTIVE_DISCOVERY_ARTIFACT_TAG);
    writer
        .word(u64::from(meta.version))
        .text(&meta.feature)
        .texts(&meta.required_features)
        .text(&meta.inference)
        .text(&meta.route)
        .word(meta.components.len() as u64);
    for (name, digest) in &meta.components {
        writer.text(name).text(digest);
    }
    writer
        .text(&meta.contract_identity)
        .text(&meta.data_identity)
        .word(meta.rows)
        .word(meta.variables.len() as u64);
    for variable in &meta.variables {
        writer.word(u64::from(*variable));
    }
    writer
        .text(&meta.receipts_identity)
        .text(&meta.graph_identity)
        .text(&meta.result_identity)
        .word(u64::from(meta.search_incomplete))
        .word(u64::from(meta.orientation_conflict))
        .word(meta.bounded_search_receipts.len() as u64);
    for (code, message) in &meta.bounded_search_receipts {
        writer.text(code).text(message);
    }
    let w = &meta.work;
    for word in [
        w.ci_tests_attempted,
        w.ci_tests_completed,
        w.ci_tests_reused,
        w.ci_tests_consumed,
        w.search_passes,
        w.orientation_runs,
        w.orientation_rounds,
        w.orientations_applied,
        w.score_evaluations,
        w.algorithm_executions,
    ] {
        writer.word(word);
    }
    writer.finish_hex()
}

// -- consumption --------------------------------------------------------------------------------

fn check(
    meta: &SelectiveDiscoveryMeta,
    expected_identity: Option<&str>,
) -> Result<SelectiveDiscoveryArtifact, Failure> {
    if meta.version != SELECTIVE_DISCOVERY_ARTIFACT_VERSION {
        return Err(Failure::UnsupportedVersion { version: meta.version });
    }
    if meta.feature != SELECTIVE_DISCOVERY_ARTIFACT_FEATURE {
        return Err(Failure::UnsupportedSemantics("feature marker"));
    }
    if !meta.required_features.is_empty() {
        return Err(Failure::UnsupportedSemantics("required feature"));
    }
    if meta.inference != SELECTIVE_DISCOVERY_INFERENCE {
        return Err(Failure::UnsupportedSemantics("inference label"));
    }
    if meta.receipts.len() > MAX_PORTABLE_RECEIPTS {
        return Err(Failure::LimitsExceeded("receipts"));
    }
    let receipts = meta.receipts.iter().map(receipt_from_wire).collect::<Result<Vec<_>, _>>()?;
    let mut components = Vec::with_capacity(meta.components.len());
    for (name, hex) in &meta.components {
        components.push((name.clone(), hex_digest(hex, "a stage input")?));
    }
    let structure = structure_from_wire(&meta.structure)?;
    let graph_identity = hex_digest(&meta.graph_identity, "graph_identity")?;
    if structure.identity() != graph_identity {
        return Err(Failure::IdentityMismatch { field: "graph_identity" });
    }
    let state = PortableDiscoveryState {
        route: meta.route.clone(),
        components,
        contract_identity: hex_digest(&meta.contract_identity, "contract_identity")?,
        data_identity: hex_digest(&meta.data_identity, "data_identity")?,
        rows: meta.rows,
        variables: meta.variables.clone(),
        receipts,
        receipts_identity: hex_digest(&meta.receipts_identity, "receipts_identity")?,
        structure,
        result_identity: hex_digest(&meta.result_identity, "result_identity")?,
        search_incomplete: meta.search_incomplete,
        orientation_conflict: meta.orientation_conflict,
        bounded_search_receipts: meta.bounded_search_receipts.clone(),
        work: work_from_wire(&meta.work),
    };
    state.validate().map_err(|e| refused(&e))?;
    let identity = compute_identity(meta);
    if meta.identity != identity {
        return Err(Failure::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(Failure::IdentityMismatch { field: "retained_identity" });
    }
    Ok(SelectiveDiscoveryArtifact { state, identity })
}

fn meta_of(state: &PortableDiscoveryState) -> SelectiveDiscoveryMeta {
    let mut meta = SelectiveDiscoveryMeta {
        version: SELECTIVE_DISCOVERY_ARTIFACT_VERSION,
        feature: SELECTIVE_DISCOVERY_ARTIFACT_FEATURE.to_owned(),
        required_features: Vec::new(),
        inference: SELECTIVE_DISCOVERY_INFERENCE.to_owned(),
        route: state.route.clone(),
        components: state.components.iter().map(|(n, d)| (n.clone(), digest_hex(d))).collect(),
        contract_identity: digest_hex(&state.contract_identity),
        data_identity: digest_hex(&state.data_identity),
        rows: state.rows,
        variables: state.variables.clone(),
        receipts: state.receipts.iter().map(receipt_to_wire).collect(),
        receipts_identity: digest_hex(&state.receipts_identity),
        structure: structure_to_wire(&state.structure),
        graph_identity: digest_hex(&state.graph_identity()),
        result_identity: digest_hex(&state.result_identity),
        search_incomplete: state.search_incomplete,
        orientation_conflict: state.orientation_conflict,
        bounded_search_receipts: state.bounded_search_receipts.clone(),
        work: work_to_wire(&state.work),
        identity: String::new(),
    };
    meta.identity = compute_identity(&meta);
    meta
}

fn encode_meta(meta: &SelectiveDiscoveryMeta, artifact_id: &str) -> Result<Vec<u8>, Failure> {
    if artifact_id.trim().is_empty() {
        return Err(Failure::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(|error| Failure::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "selective_discovery",
            max_bytes: MAX_SELECTIVE_DISCOVERY_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: META_SECTION,
            content_type: crate::CONTENT_CBOR,
            bytes: &meta_bytes,
        }],
    )
    .map_err(|error| crate::map_artifact_check!(error, SelectiveDiscoveryArtifactError))
}

fn decode_meta(bytes: &[u8]) -> Result<SelectiveDiscoveryMeta, Failure> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_SELECTIVE_DISCOVERY_ARTIFACT_BYTES,
        &[META_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, SelectiveDiscoveryArtifactError))?;
    let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(Failure::Malformed("unsupported container layout".into()));
    };
    crate::decode_with_version(&meta_bytes, |version: u16| {
        if version == SELECTIVE_DISCOVERY_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(Failure::UnsupportedVersion { version })
        }
    })
}

// -- public type --------------------------------------------------------------------------------

/// A validated portable discovery state and the identity that binds it.
#[derive(Clone, Debug)]
pub struct SelectiveDiscoveryArtifact {
    state: PortableDiscoveryState,
    identity: String,
}

impl SelectiveDiscoveryArtifact {
    /// Seal a state. The result passes the checks a consumer applies.
    ///
    /// # Errors
    ///
    /// A state that is not internally consistent.
    pub fn seal(
        state: &PortableDiscoveryState,
    ) -> Result<(Self, SelectiveDiscoveryMeta), SelectiveDiscoveryArtifactError> {
        state.validate().map_err(|e| refused(&e))?;
        let meta = meta_of(state);
        let artifact = check(&meta, None)?;
        Ok((artifact, meta))
    }

    /// Serialize a sealed state through the checksummed container.
    ///
    /// # Errors
    ///
    /// A state that does not seal, an encoding failure or an oversized payload.
    pub fn to_bytes(
        state: &PortableDiscoveryState,
        artifact_id: &str,
    ) -> Result<Vec<u8>, SelectiveDiscoveryArtifactError> {
        let (_, meta) = Self::seal(state)?;
        encode_meta(&meta, artifact_id)
    }

    /// Consume an artifact: rebuild receipts and structure, recompute every identity.
    ///
    /// `expected_identity` is an artifact identity (hex) the consumer retained
    /// independently; when given, a consistently resealed artifact of another run is refused.
    ///
    /// # Errors
    ///
    /// Corruption, another major version, an unknown feature or inference label, a bound
    /// exceeded, a state the library refuses, or a stored identity that does not match.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, SelectiveDiscoveryArtifactError> {
        check(&decode_meta(bytes)?, expected_identity)
    }

    /// Read the stored metadata without validating it: the independent inspection of a
    /// state. Needs no raw data.
    ///
    /// # Errors
    ///
    /// Corruption, another major version or a malformed container.
    pub fn inspect(
        bytes: &[u8],
    ) -> Result<SelectiveDiscoveryMeta, SelectiveDiscoveryArtifactError> {
        decode_meta(bytes)
    }

    /// The validated state.
    #[must_use]
    pub const fn state(&self) -> &PortableDiscoveryState {
        &self.state
    }

    /// The artifact identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Validate metadata as a consumer would, for tests that reseal edits.
    ///
    /// # Errors
    ///
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &SelectiveDiscoveryMeta,
        expected_identity: Option<&str>,
    ) -> Result<Self, SelectiveDiscoveryArtifactError> {
        check(meta, expected_identity)
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut SelectiveDiscoveryMeta) {
        meta.identity = compute_identity(meta);
    }

    /// Serialize metadata as it is, for tests and fixtures that write an edited layout.
    ///
    /// # Errors
    ///
    /// An encoding failure.
    #[doc(hidden)]
    pub fn meta_to_bytes(
        meta: &SelectiveDiscoveryMeta,
        artifact_id: &str,
    ) -> Result<Vec<u8>, SelectiveDiscoveryArtifactError> {
        encode_meta(meta, artifact_id)
    }
}
