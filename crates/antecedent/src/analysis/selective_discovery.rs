//! Selective discovery in the analysis facade: how a discovery run connects to the stage
//! planner, the claim-level invalidation graph, the evidence ledger and the transport
//! refusal surface.
//!
//! The adapters themselves ([`StaticSelectiveDiscovery`], [`TemporalSelectiveDiscovery`])
//! live in `antecedent-discovery`; this module is where their results enter the rest of the
//! library:
//!
//! * [`accepted_graph_stage_identity`] is the only way a discovered structure becomes the
//!   graph input of the stage planner. It requires a resolved structure or an explicitly
//!   declared completion, and refuses a search that exhausted a bound. A representative
//!   member of an equivalence class never becomes a known graph on its own: the completion
//!   is a declaration with its own identity and label, recorded as an assumption.
//! * [`add_discovery_claims`] places the discovery stages in a [`CompositionGraph`] so a
//!   graph, completion or selection change classifies every downstream claim (identification,
//!   prepared fit, transport binding, decision) as still valid, recompute or withdrawn, with
//!   the same states every other upstream change uses.
//! * [`discovery_data_source`] declares the discovery rows in the evidence ledger, so
//!   reuse of the discovery data in a later analysis is an overlap the ledger computes
//!   (shared observations) and a disjoint split is independent.
//! * [`discovered_structure_refusal`] renders the refusal a transport request over a
//!   discovered structure returns.
//!
//! # What is not claimed
//!
//! No node, source or refusal here states that a discovered graph is the true graph, an
//! error rate for it, or an interval for a downstream estimate. A downstream node that rests
//! on a discovered graph carries `PointOnly` calibration; an interval that depends on the
//! discovered graph needs a calibration of the complete discovery-plus-analysis method
//! (including reuse of the discovery rows and any sample split), which is not measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::ops::Range;

use antecedent_contract::ExternalRefusal;
use antecedent_contract::composition_verification::{
    ClaimStanding, CompositionGraph, CompositionNode, DependencyKind, GraphError,
    GraphRevalidation, NodeKind, revalidate,
};
use antecedent_contract::evidence_ledger::{
    Lineage, ObservationId, SourceDeclaration, SourceId, SourceKind,
};
use antecedent_core::recalc::StageIdentity;
use antecedent_core::{CanonicalWriter, SemanticDigest};

pub use antecedent_discovery::selective::{
    CiTestSpec, DataIdentity, DiscoveredStructure, DiscoveredStructureRefusal, DiscoveryOutcome,
    DiscoveryPlan, DiscoveryStage, NativeResult, OrientationWork, PortableDiscoveryState,
    SelectiveDiscoveryError, SelectiveRunReport, StageDecision, StageInputs, StaticAlgorithm,
    StaticDiscoveryContract, StaticSelectiveDiscovery, StatisticReceipt, StatisticStore,
    StructureClass, TemporalAlgorithm, TemporalDiscoveryContract, TemporalSelectiveDiscovery,
    UnresolvedEdge, WorkUnits, request_selection_diagram, static_data_identity,
    temporal_data_identity,
};

const ACCEPTED_GRAPH_LABEL: &str = "discovery_accepted_graph";
const COORDINATE_TAG: &str = "antecedent.selective_discovery.claim_coordinate";
const OBSERVATION_DOMAIN: &str = "selective-discovery-row";

/// A completion of an equivalence class that the caller declares.
///
/// A completion is a choice, made by the caller, of one member of the class (or an explicit
/// graph from background knowledge). It is not something discovery found.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredCompletion {
    /// Human label, recorded as part of the identity.
    pub label: String,
    /// Content identity of the declared graph.
    pub identity: [u8; 32],
}

/// The caller's acceptance of a discovered structure as the causal graph of an analysis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphAcceptance {
    /// Who or what declared the graph (a reviewer, a protocol), recorded in the identity.
    pub declared_by: String,
    /// The completion, required when the structure has unresolved edges.
    pub completion: Option<DeclaredCompletion>,
}

/// The identity to declare for `Stage::Graph` of the stage planner.
///
/// It covers the discovered structure, the declared completion and the declaring party, so a
/// change of any of them reaches identification, prepared fits, transport bindings and
/// decisions through the planner's ordinary dependency rules.
///
/// # Errors
///
/// A search that exhausted a bound, or a structure with unresolved edges and no declared
/// completion. The refusal names every unresolved edge and what would license acceptance.
pub fn accepted_graph_stage_identity(
    outcome: &DiscoveryOutcome,
    acceptance: &GraphAcceptance,
) -> Result<StageIdentity, DiscoveredStructureRefusal> {
    let code = antecedent_core::reason_code!("discovered_structure_unresolved");
    if outcome.search_incomplete {
        return Err(DiscoveredStructureRefusal {
            code,
            detail: "selective_discovery.search_incomplete",
            unresolved_edges: outcome.structure.unresolved_edges(),
            bounded_search_receipts: outcome.bounded_search_receipts.clone(),
            remedy: "the search exhausted a bound, so absent edges are not established; rerun \
                     with larger bounds before accepting a graph"
                .to_owned(),
        });
    }
    let unresolved = outcome.structure.unresolved_edges();
    if !unresolved.is_empty() && acceptance.completion.is_none() {
        return Err(DiscoveredStructureRefusal {
            code,
            detail: "selective_discovery.unresolved_structure",
            remedy: format!(
                "{} edge(s) are not resolved within the equivalence class; declare a \
                 completion (with its own label and identity) or orient them with background \
                 knowledge. A representative member is never taken for the discovered graph",
                unresolved.len()
            ),
            unresolved_edges: unresolved,
            bounded_search_receipts: Vec::new(),
        });
    }
    let (label, identity) =
        acceptance.completion.as_ref().map_or(("", [0u8; 32]), |c| (c.label.as_str(), c.identity));
    Ok(StageIdentity::of(
        ACCEPTED_GRAPH_LABEL,
        &[
            &outcome.structure_identity,
            &identity,
            label.as_bytes(),
            acceptance.declared_by.as_bytes(),
        ],
    ))
}

/// What one discovery run contributes to a claim graph.
#[derive(Clone, Copy, Debug)]
pub struct DiscoverySnapshot<'a> {
    /// The declared stage inputs of the run.
    pub inputs: &'a StageInputs,
    /// The data identity of the run.
    pub data: &'a DataIdentity,
    /// The outcome of the run.
    pub outcome: &'a DiscoveryOutcome,
    /// The declared completion downstream claims rest on, if the caller declared one.
    pub completion: Option<&'a DeclaredCompletion>,
}

/// The node keys [`add_discovery_claims`] created.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryClaimKeys {
    /// The data node.
    pub data: String,
    /// The retained-statistics node.
    pub statistics: String,
    /// The discovered-structure node.
    pub graph: String,
    /// The declared-completion node, when one was declared.
    pub completion: Option<String>,
}

impl DiscoveryClaimKeys {
    /// The node downstream claims depend on: the completion when declared, otherwise the
    /// discovered structure.
    #[must_use]
    pub fn downstream_anchor(&self) -> &str {
        self.completion.as_deref().unwrap_or(&self.graph)
    }
}

fn digest_of(bytes: [u8; 32]) -> SemanticDigest {
    SemanticDigest::from_bytes(bytes)
}

fn graph_assumptions(outcome: &DiscoveryOutcome, route: &str) -> Vec<String> {
    let mut out = vec![
        format!("discovered_structure_class={}", outcome.structure.class().as_str()),
        format!("unresolved_edges={}", outcome.structure.unresolved_edges().len()),
        "faithfulness_assumed".to_owned(),
        "causal_markov_assumed".to_owned(),
        "no_recovery_or_error_control_claim".to_owned(),
    ];
    if !matches!(route, "static_fci") {
        out.push("causal_sufficiency_assumed".to_owned());
    }
    if !outcome.structure.unresolved_edges().is_empty() {
        out.push("equivalence_class_not_a_single_graph".to_owned());
    }
    if outcome.orientation_conflict {
        out.push("orientation_conflict".to_owned());
    }
    for (code, _) in &outcome.bounded_search_receipts {
        out.push(format!("bounded_search:{code}"));
    }
    out
}

/// Add the discovery stages of a run to `graph` under `prefix`.
///
/// Nodes and edges:
///
/// * `{prefix}.data` (data) ← nothing;
/// * `{prefix}.statistics` depends on the data (`data`);
/// * `{prefix}.graph` depends on the statistics (`data`). Its content identity is the result
///   identity and its structural identity the structure identity, so new evidence at an
///   unchanged structure recomputes numbers and keeps the structure, and a changed structure
///   recomputes everything that derived from it. A search that exhausted a bound makes the
///   node unsupported: every claim resting on it is withdrawn, not recomputed;
/// * `{prefix}.completion` (when declared) depends on the graph (`structural_derivation`).
///
/// Downstream claims declare a `structural_derivation` edge to
/// [`DiscoveryClaimKeys::downstream_anchor`].
///
/// # Errors
///
/// A key that already exists in `graph`, or a bound of the graph exceeded.
#[allow(
    clippy::too_many_lines,
    reason = "adds the four related nodes and their edges as one declaration"
)]
pub fn add_discovery_claims(
    graph: &mut CompositionGraph,
    prefix: &str,
    snapshot: &DiscoverySnapshot<'_>,
) -> Result<DiscoveryClaimKeys, GraphError> {
    let keys = DiscoveryClaimKeys {
        data: format!("{prefix}.data"),
        statistics: format!("{prefix}.statistics"),
        graph: format!("{prefix}.graph"),
        completion: snapshot.completion.map(|_| format!("{prefix}.completion")),
    };
    let outcome = snapshot.outcome;
    let route = snapshot.inputs.route();
    let mut coordinate = CanonicalWriter::new(COORDINATE_TAG);
    coordinate.text(route);
    if let Some(population) = snapshot.inputs.component("population") {
        coordinate.bytes(population);
    }
    let coordinate = digest_of(coordinate.finish());

    graph.add_node(
        CompositionNode::new(keys.data.clone(), NodeKind::Data, digest_of(snapshot.data.digest))
            .with_coordinate(coordinate)
            .with_standing(ClaimStanding::native_point()),
    )?;
    let statistics = snapshot.inputs.identity(DiscoveryStage::Statistics);
    graph.add_node(
        CompositionNode::new(
            keys.statistics.clone(),
            NodeKind::Claim,
            digest_of(*statistics.as_bytes()),
        )
        .with_coordinate(coordinate)
        .with_representation("ci_statistics")
        .with_standing(ClaimStanding::native_point()),
    )?;
    let mut node = CompositionNode::new(
        keys.graph.clone(),
        NodeKind::Claim,
        digest_of(outcome.result_identity),
    )
    .with_structure(digest_of(outcome.structure_identity))
    .with_representation(outcome.structure.class().as_str())
    .with_coordinate(coordinate)
    .with_assumptions(graph_assumptions(outcome, route))
    // Discovery is native and reports a structure, not an interval: the standing is
    // stated, not defaulted, so a later calibration claim cannot ride a reused stage.
    .with_standing(ClaimStanding::native_point());
    if outcome.search_incomplete {
        let coordinates: Vec<String> = outcome
            .bounded_search_receipts
            .iter()
            .map(|(code, _)| format!("search_bound_exhausted:{code}"))
            .collect();
        node = node.with_unsupported(coordinates);
    }
    graph.add_node(node)?;
    graph.add_edge(&keys.statistics, &keys.data, DependencyKind::Data)?;
    graph.add_edge(&keys.graph, &keys.statistics, DependencyKind::Data)?;
    if let (Some(key), Some(completion)) = (&keys.completion, snapshot.completion) {
        graph.add_node(
            CompositionNode::new(key.clone(), NodeKind::Claim, digest_of(completion.identity))
                .with_structure(digest_of(completion.identity))
                .with_representation("declared_completion")
                .with_coordinate(coordinate)
                .with_assumptions(vec![
                    "declared_completion_of_equivalence_class".to_owned(),
                    format!("completion_label={}", completion.label),
                ])
                .with_standing(ClaimStanding::native_point()),
        )?;
        graph.add_edge(key, &keys.graph, DependencyKind::StructuralDerivation)?;
    }
    Ok(keys)
}

/// Classify every claim of `current` against `previous` after a discovery change.
///
/// This is [`revalidate`] with no extra observed effects: the discovery nodes carry their
/// own identities, so a new dataset, a new threshold, new background knowledge or a changed
/// completion is visible as an identity or structure change.
///
/// # Errors
///
/// A graph that does not validate.
pub fn revalidate_after_discovery(
    previous: &CompositionGraph,
    current: &CompositionGraph,
) -> Result<GraphRevalidation, GraphError> {
    revalidate(previous, current, &BTreeMap::new())
}

/// Declare the rows a discovery run used as an evidence source.
///
/// Observation identities are keyed by the dataset `snapshot_id` and the row index, never by
/// which columns were searched. Declare the later analysis's rows with the same
/// `snapshot_id`: rows both used overlap (the ledger reports shared observations), and a
/// disjoint row range is independent. Reusing the discovery rows in the analysis is a case
/// of the one overlap model, not a separate one.
///
/// A row range larger than the ledger's per-source bound is declared with undeclared
/// observations, which the ledger treats as unknown overlap, never as independent.
#[must_use]
pub fn discovery_data_source(
    label: &str,
    snapshot_id: &[u8],
    rows: Range<u64>,
) -> SourceDeclaration {
    let declaration = SourceDeclaration::new(label, SourceKind::ObservationSet);
    let count = rows.end.saturating_sub(rows.start);
    if count > antecedent_contract::evidence_ledger::MAX_OBSERVATIONS_PER_SOURCE as u64 {
        return declaration;
    }
    let ids = rows.map(|row| {
        let mut key = Vec::with_capacity(snapshot_id.len() + 8);
        key.extend_from_slice(snapshot_id);
        key.extend_from_slice(&row.to_le_bytes());
        ObservationId::from_key(OBSERVATION_DOMAIN, &key)
    });
    declaration.with_observations(ids)
}

/// Declare an explicit set of rows (not necessarily contiguous) as an evidence source.
///
/// As [`discovery_data_source`], for a row set such as one part of a sample split: the
/// observation identities are keyed by `snapshot_id` and row index, so the discovery part and
/// the analysis part of one split are independent in the ledger, and either part overlaps the
/// unsplit table on exactly the rows it holds.
#[must_use]
pub fn discovery_rows_source(label: &str, snapshot_id: &[u8], rows: &[usize]) -> SourceDeclaration {
    let declaration = SourceDeclaration::new(label, SourceKind::ObservationSet);
    if rows.len() > antecedent_contract::evidence_ledger::MAX_OBSERVATIONS_PER_SOURCE {
        return declaration;
    }
    let ids = rows.iter().map(|&row| {
        let mut key = Vec::with_capacity(snapshot_id.len() + 8);
        key.extend_from_slice(snapshot_id);
        key.extend_from_slice(&(row as u64).to_le_bytes());
        ObservationId::from_key(OBSERVATION_DOMAIN, &key)
    });
    declaration.with_observations(ids)
}

/// Declare a discovered graph as a fitted-model source built from data already used.
///
/// A later analysis that treats the graph as evidence is counted as the discovery rows it
/// came from; the declaration makes that lineage explicit.
#[must_use]
pub fn discovered_graph_source(label: &str, data_sources: Vec<SourceId>) -> SourceDeclaration {
    SourceDeclaration::new(label, SourceKind::FittedModel)
        .with_lineage(Lineage::DerivedFromUsedData { inputs: data_sources })
}

/// Render the refusal of a transport request over a discovered structure.
///
/// The unresolved edges are listed in `offending`; the remedy is fixed text because the
/// remedy field of the refusal surface is static.
#[must_use]
pub fn discovered_structure_refusal(refusal: &DiscoveredStructureRefusal) -> ExternalRefusal {
    let edges: Vec<String> = refusal
        .unresolved_edges
        .iter()
        .map(|u| format!("{}~{}:{}", u.edge.a.label(), u.edge.b.label(), u.reason))
        .collect();
    let mut out = ExternalRefusal::new(refusal.code, "bind", refusal.detail).with_remedy(
        "resolve every unresolved edge or declare the causal graph, then request transport \
             with a declared selection diagram; transport is not computed across an \
             equivalence class",
    );
    if !edges.is_empty() {
        out = out.with_offending(edges.join(";"));
    }
    out
}
