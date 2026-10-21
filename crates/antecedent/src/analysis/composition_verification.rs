//! End-to-end composition verification: classify every downstream claim when
//! an upstream object changes, expose each claim's lineage, and export the
//! result for a fresh-process consumer.
//!
//! The classification is one claim-level invalidation graph
//! ([`CompositionGraph`]) built on the stage planner and the invalidation log
//! the study already keeps; it adds no pipeline stage and no route of its
//! own. Three observations enter it as [`ObservedEffect`]s on the node they
//! concern:
//!
//! * a recalculation plan ([`stage_effects`]): a recomputed stage is a
//!   numerical effect, a refused stage cannot be executed;
//! * the evidence ledger ([`ledger_effects`]): a stale result recomputes, a
//!   withdrawn or replaced result is withdrawn;
//! * the invalidation log ([`invalidation_effects`]) and claim freshness
//!   ([`freshness_effects`]): a query result or model invalidated after a
//!   state version recomputes, one whose premise was removed is withdrawn; a
//!   withdrawn claim is withdrawn, and a claim whose contract identities moved
//!   in any layer recomputes ([`contract_effects`]).
//!
//! # What each class means
//!
//! * `still_valid`: nothing it rests on changed and no license it states was
//!   lost. The earlier numbers and uncertainty stand.
//! * `recompute`: data, evidence, a model or representation version or a
//!   structural derivation it uses changed. The earlier fit and uncertainty
//!   carry no authority; reusing a structural derivation after compatible data
//!   replacement recomputes the numbers and does not retain the old fit.
//! * `withdrawn`: an input it cannot do without was retired, coordinates it
//!   answers are outside the support, a provider's trust no longer holds, a
//!   stage cannot be executed, or the calibration it states does not hold at
//!   its coordinate (a claim that inherited a record from an upstream object
//!   that no longer holds it, or a record established for another target, is
//!   refused). A calibration withdrawal leaves the point value untouched.
//!
//! Stage reuse never proves a claim still licensed: support, provider trust
//! and calibration are rechecked at the current coordinates.
//!
//! # Lifecycle
//!
//! * `refresh` with new outcome-only data **keeps** the structural derivation
//!   (graph, identification, bridge form), **recomputes** every node that
//!   reads the data, and **refuses** to carry a fit or an interval across:
//!   such claims classify as `recompute` with
//!   `structure_reused_fit_not_retained`.
//! * A change of target population, source evidence, selection diagram or
//!   support classifies each transported claim as still valid, recompute or
//!   withdrawn; a calibration record established at one target is never
//!   inherited by a claim about another.
//! * An export carries the graphs, the observed effects, the classification
//!   and the lineage identities; a consumer replays the classification.
//!   Absence of the earlier snapshot makes a first verification: nothing is
//!   reported stale.
//!
//! # What is closed
//!
//! The verification produces no estimate, so it produces no interval,
//! posterior or calibration record: an interval belongs to the route that
//! produced the claim, and a record established for one method at one target
//! never holds for another. A calibration declaration names a record identity
//! and a coordinate that the producer supplies. [`CompositionVerification::verify`]
//! takes the declaration as declared; [`CompositionVerification::verify_against_registry`]
//! resolves every held or inherited record against the shipped coverage
//! records first ([`registry_record_identity`]) and treats one that does not
//! name a currently attesting record as a requirement without a record.
//!
//! # Flagship
//!
//! [`FlagshipWorkflow`] runs one decision on the cluster-randomized trial, the
//! checked temporal period effect, the conjugate cost posterior, and an affine
//! endpoint-to-margin bridge. Each input contributes the variance its route
//! published. The three inputs are estimated apart, so their recorded
//! dependence is independence. The bridge keeps its slope-source covariance.
//! A utility change refits nothing. A prior change refits the cost posterior.
//! A period change refits the temporal mechanisms. Withdrawing the trial
//! withdraws the margin and the decision and leaves the temporal and cost claims.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::ExternalRefusal;
use antecedent_contract::claim::{ClaimEnvelope, ClaimFreshness};
use antecedent_contract::evidence_ledger::{EvidenceLedger, ResultId};
use antecedent_core::recalc::{RecalcPlan, Stage};
use antecedent_core::{
    CancellationToken, CanonicalWriter, ContractIdentities, QueryId, SemanticDigest, StateVersion,
    SupportStatus,
};
use antecedent_io::IoError;
use antecedent_io::composition_verification_artifact::{
    CompositionVerificationArtifact, CompositionVerificationArtifactError,
};
use antecedent_state::{InvalidationLog, InvalidationTarget};

pub use antecedent_contract::composition_verification::{
    BoundaryAssumption, BoundaryTransformation, CalibrationDeclaration, CalibrationLineage,
    CalibrationState, ClaimRevalidation, ClaimStanding, CompositionEdge, CompositionGraph,
    CompositionNode, CompositionView, DependencyKind, GraphError, GraphRevalidation, LineageEdge,
    LineageEntry, NodeKind, NumericalCause, NumericalReason, ObservedEffect, PriorAuthority,
    SharedDataRelation, SupportStanding, TrustStanding, UnsupportedCoordinate, ViewError,
    Withdrawal, WithdrawalKind, shared_data_relations,
};
pub use antecedent_contract::evidence_ledger::Revalidation;

/// A typed, actionable refusal of composition verification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionVerificationRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced detail, `composition_verification.<slot>`.
    pub detail: String,
    /// What was refused.
    pub message: String,
    /// The declaration or evidence that would license the request.
    pub remedy: Option<String>,
}

impl std::fmt::Display for CompositionVerificationRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {}: {}",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.message
        )
    }
}

impl std::error::Error for CompositionVerificationRefusal {}

impl From<CompositionVerificationRefusal> for IoError {
    fn from(refusal: CompositionVerificationRefusal) -> Self {
        Self::Refused {
            code: refusal.code,
            message: format!("{}: {}", refusal.detail, refusal.message),
        }
    }
}

impl CompositionVerificationRefusal {
    fn from_structured(refusal: &ExternalRefusal) -> Self {
        let mut message = format!("refused at stage `{}`", refusal.stage);
        if let Some(offending) = &refusal.offending {
            message.push_str(&format!("; claim `{offending}`"));
        }
        if let Some(expected) = &refusal.expected {
            message.push_str(&format!("; {expected}"));
        }
        if let Some(supplied) = &refusal.supplied {
            message.push_str(&format!("; license lost: {supplied}"));
        }
        Self {
            code: refusal.code,
            detail: refusal.detail.clone(),
            message,
            remedy: refusal.remedy.map(str::to_owned),
        }
    }

    fn graph(error: &GraphError) -> Self {
        Self {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail: "composition_verification.graph_invalid".to_owned(),
            message: error.to_string(),
            remedy: Some(
                "declare nodes with unique keys, dependencies between declared nodes without a \
                 cycle, and a calibration dependency for every inherited calibration"
                    .to_owned(),
            ),
        }
    }

    fn artifact(error: &CompositionVerificationArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, message)) => {
                let remedy = match detail.as_str() {
                    "composition_verification.identity_mismatch" => {
                        "supply the artifact whose recomputed identity equals the retained \
                         identity, or regenerate it from the intended graphs"
                            .to_owned()
                    }
                    "composition_verification.replay_mismatch" => {
                        "regenerate the artifact from the graphs and observed effects; the stored \
                         classification and lineage identities must replay"
                            .to_owned()
                    }
                    _ => message.clone(),
                };
                Self { code, detail, message, remedy: Some(remedy) }
            }
            None => Self {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "composition_verification.artifact_invalid".to_owned(),
                message: error.to_string(),
                remedy: Some("supply an intact composition_verification_v1 artifact".to_owned()),
            },
        }
    }
}

/// The standing of a claim one of the library's own transport routes carried to
/// another population.
///
/// The claim rests on no provider (trust not applicable) and states a point:
/// no interval, posterior or calibration record is claimed, so a moved
/// calibration input cannot be mistaken for a held one. Its support follows the
/// route's label: `extrapolative` is the strongest a carried claim states and
/// reads as supported coordinates (the label itself travels in the claim's
/// identity); a label stronger than that, or one that is not a support label,
/// names the claim's coordinates as unsupported.
#[must_use]
pub fn library_transported_standing(support_label: &str, coordinates: &[String]) -> ClaimStanding {
    let standing = ClaimStanding::native_point();
    match SupportStatus::from_name(support_label) {
        Some(status) if status.severity() <= SupportStatus::Extrapolative.severity() => standing,
        _ => {
            let mut names: Vec<String> = coordinates.to_vec();
            names.sort_unstable();
            names.dedup();
            if names.is_empty() {
                names.push("support_label".to_owned());
            }
            standing.with_unsupported(names)
        }
    }
}

/// The node of a claim a library transport route issued: kind
/// `transported_result`, identity bound to the route and the route's own
/// identity text, coordinate bound to the target population and the route, the
/// route's assumptions, and the standing of
/// [`library_transported_standing`].
///
/// The identity moves exactly when the route's own identity moves, so a changed
/// source model, cohort, declaration or population recomputes the node; the
/// standing is stated, so a calibration record declared on the node later is
/// checked at this coordinate.
#[must_use]
pub fn library_transported_node(
    key: &str,
    route: &str,
    route_identity: &str,
    target_population: &str,
    support_label: &str,
    assumptions: Vec<String>,
) -> CompositionNode {
    let mut identity =
        CanonicalWriter::new("antecedent.composition_verification.library_transport");
    identity.text(route).text(route_identity);
    let mut coordinate =
        CanonicalWriter::new("antecedent.composition_verification.library_transport_coordinate");
    coordinate.text(route).text(target_population);
    CompositionNode::new(
        key,
        NodeKind::TransportedResult,
        SemanticDigest::from_bytes(identity.finish()),
    )
    .with_coordinate(SemanticDigest::from_bytes(coordinate.finish()))
    .with_standing(library_transported_standing(support_label, &[]))
    .with_assumptions(assumptions)
    .with_transformation(route)
}

/// A node whose content identity is the effective identity the plan assigned
/// to `stage`, with the support, trust and calibration standings its producer
/// states. A `Reused` stage retains authority only while those standings are
/// unchanged between two snapshots, so a route builds its nodes here rather
/// than through [`stage_node`] (which states no standing and so reads as a
/// supported point-only claim). `None` when the plan does not contain the
/// stage.
#[must_use]
pub fn stage_node_with_standing(
    key: &str,
    kind: NodeKind,
    plan: &RecalcPlan,
    stage: Stage,
    standing: ClaimStanding,
) -> Option<CompositionNode> {
    stage_node(key, kind, plan, stage).map(|node| node.with_standing(standing))
}

/// A node whose content identity is the effective identity the plan assigned
/// to `stage`: it changes exactly when the stage's own inputs or any stage it
/// depends on change. `None` when the plan does not contain the stage.
///
/// The node states no standing: it is supported, point-only and carries no
/// provider trust. Use [`stage_node_with_standing`] for any claim that states an
/// interval or posterior or rests on a provider.
#[must_use]
pub fn stage_node(
    key: &str,
    kind: NodeKind,
    plan: &RecalcPlan,
    stage: Stage,
) -> Option<CompositionNode> {
    let identity = plan.identity(stage)?;
    Some(CompositionNode::new(key, kind, SemanticDigest::from_bytes(*identity.as_bytes())))
}

/// The observed effects of a recalculation plan: the node bound to a stage
/// takes that stage's status. A reused stage is no effect.
#[must_use]
pub fn stage_effects(
    plan: &RecalcPlan,
    bindings: &[(&str, Stage)],
) -> BTreeMap<String, ObservedEffect> {
    bindings
        .iter()
        .filter_map(|(key, stage)| {
            let status = plan.status(*stage)?;
            ObservedEffect::from_stage_status(status).map(|effect| ((*key).to_owned(), effect))
        })
        .collect()
}

/// The observed effects of ledger standings: a result needing recomputation
/// is a numerical effect, a withdrawn or replaced result a withdrawal.
///
/// # Errors
/// A result id the ledger does not hold.
pub fn ledger_effects(
    ledger: &EvidenceLedger,
    bindings: &[(&str, ResultId)],
) -> Result<BTreeMap<String, ObservedEffect>, CompositionVerificationRefusal> {
    antecedent_contract::composition_verification::ledger_effects(ledger, bindings).map_err(|e| {
        CompositionVerificationRefusal {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail: "composition_verification.unknown_result".to_owned(),
            message: e.to_string(),
            remedy: Some("bind nodes to results registered in the evidence ledger".to_owned()),
        }
    })
}

/// The observed effects of claim freshness: a withdrawn claim is withdrawn,
/// a superseded claim or one whose snapshot no longer matches recomputes.
#[must_use]
pub fn freshness_effects(bindings: &[(&str, &ClaimFreshness)]) -> BTreeMap<String, ObservedEffect> {
    bindings
        .iter()
        .filter_map(|(key, freshness)| {
            ObservedEffect::from_freshness(freshness).map(|effect| ((*key).to_owned(), effect))
        })
        .collect()
}

/// The observed effects of claim contracts against the identities the receiver
/// now holds: every identity layer is compared, so a changed target,
/// identification, program, inference binding or observation recomputes the
/// claim even when the data snapshot is unchanged, and the changed layers are
/// named in the effect's cause. A receiver with no current identities (`None`)
/// falls back to the claim's freshness reading.
#[must_use]
pub fn contract_effects(
    bindings: &[(&str, &ClaimEnvelope, &ClaimFreshness, Option<&ContractIdentities>)],
) -> BTreeMap<String, ObservedEffect> {
    bindings
        .iter()
        .filter_map(|(key, claim, freshness, current)| {
            ObservedEffect::from_claim_contract(claim, freshness, *current)
                .map(|effect| ((*key).to_owned(), effect))
        })
        .collect()
}

/// The observed effects of invalidations recorded after `since`, in the one
/// three-state vocabulary: a node bound to a query result (or a model) the log
/// invalidated recomputes, and one whose premise the log shows removed (a
/// withdrawn constraint) is withdrawn. An invalidation of every result applies
/// to every bound node. The most severe entry per node wins and its cause
/// names the stage input that moved.
#[must_use]
pub fn invalidation_effects(
    log: &InvalidationLog,
    since: StateVersion,
    bindings: &[(&str, InvalidationTarget)],
) -> BTreeMap<String, ObservedEffect> {
    let mut classes: BTreeMap<String, (Revalidation, String)> = BTreeMap::new();
    for entry in log.entries.iter().filter(|e| e.at_version.raw() > since.raw()) {
        let class = entry.revalidation();
        let source = entry.input.map_or_else(
            || "unattributed".to_owned(),
            |input| format!("{}:{}", input.stage, input.identity),
        );
        for (key, target) in bindings {
            let hit =
                matches!(entry.target, InvalidationTarget::AllResults) || entry.target == *target;
            if !hit {
                continue;
            }
            let cause = format!("invalidation.at_version:{}:{source}", entry.at_version.raw());
            match classes.get_mut(*key) {
                Some((held, held_cause)) if class > *held => {
                    *held = class;
                    *held_cause = cause;
                }
                Some(_) => {}
                None => {
                    classes.insert((*key).to_owned(), (class, cause));
                }
            }
        }
    }
    classes
        .into_iter()
        .filter_map(|(key, (class, cause))| {
            ObservedEffect::from_revalidation(class, cause).map(|effect| (key, effect))
        })
        .collect()
}

/// The three-state class of `query` in the invalidation log after `since`:
/// `still_valid` when no entry concerns it, `recompute` for changed data or
/// premises, `withdrawn` when a premise it rested on was removed.
#[must_use]
pub fn query_revalidation(
    log: &InvalidationLog,
    query: QueryId,
    since: StateVersion,
) -> Revalidation {
    log.revalidation_since(query, since)
}

/// The digest under which a shipped coverage record is declared on a node.
#[must_use]
pub fn registry_record_identity(record_id: &str) -> SemanticDigest {
    let mut writer =
        CanonicalWriter::new("antecedent.composition_verification.registry_coverage_record");
    writer.text(record_id);
    SemanticDigest::from_bytes(writer.finish())
}

/// Whether `record` names a shipped coverage record that currently attests the
/// code it measured.
#[must_use]
pub fn registry_record_attests(record: &SemanticDigest) -> bool {
    use antecedent_io::coverage_records_data::{CoverageRecord, RECORDS};
    RECORDS.iter().any(|row| {
        let row: &CoverageRecord = std::borrow::Borrow::borrow(&row);
        registry_record_identity(row.id) == *record
            && antecedent_io::calibration::record_attests_current_code(row)
    })
}

/// A declaration with every record that does not name a currently attesting
/// shipped record replaced by [`CalibrationDeclaration::Required`]: an
/// uncertainty statement is wanted at the node but nothing backs it. A point
/// declaration and a requirement are kept as they are.
#[must_use]
pub fn registry_resolved_calibration(
    declaration: &CalibrationDeclaration,
) -> CalibrationDeclaration {
    match declaration {
        CalibrationDeclaration::Held { record, .. }
        | CalibrationDeclaration::Inherited { record, .. }
            if !registry_record_attests(record) =>
        {
            CalibrationDeclaration::Required
        }
        other => other.clone(),
    }
}

/// The keys of the nodes whose held or inherited calibration record does not
/// name a currently attesting shipped coverage record.
#[must_use]
pub fn unresolved_registry_calibrations(graph: &CompositionGraph) -> Vec<String> {
    graph
        .nodes()
        .filter(|node| registry_resolved_calibration(&node.calibration) != node.calibration)
        .map(|node| node.key.clone())
        .collect()
}

/// A graph whose declared calibrations were resolved against the registry.
fn registry_resolved_graph(
    graph: &CompositionGraph,
) -> Result<CompositionGraph, CompositionVerificationRefusal> {
    let mut resolved = CompositionGraph::new();
    for node in graph.nodes() {
        let mut node = node.clone();
        node.calibration = registry_resolved_calibration(&node.calibration);
        resolved.add_node(node).map_err(|e| CompositionVerificationRefusal::graph(&e))?;
    }
    for edge in graph.edges() {
        resolved
            .add_edge(&edge.downstream, &edge.upstream, edge.kind)
            .map_err(|e| CompositionVerificationRefusal::graph(&e))?;
    }
    Ok(resolved)
}

/// A sealed and replayed composition verification.
#[derive(Clone, Debug)]
pub struct CompositionVerification {
    artifact: CompositionVerificationArtifact,
}

impl CompositionVerification {
    /// Classify `current` against `previous` (a first verification when
    /// `None`) under the observed `effects`.
    ///
    /// With a ledger, the shared-data relations among the nodes' evidence
    /// sources are read from its row-level overlap (an unknown overlap is
    /// reported as an unknown dependence, never as independent) and its state
    /// digest is recorded; without one the lineage reports no relation and
    /// says nothing about independence.
    ///
    /// # Errors
    /// A graph that does not validate, a ledger that does not hold a declared
    /// source, the overlap bound or cancellation.
    pub fn verify(
        previous: Option<&CompositionGraph>,
        current: &CompositionGraph,
        effects: &BTreeMap<String, ObservedEffect>,
        ledger: Option<(&EvidenceLedger, &CancellationToken)>,
    ) -> Result<Self, CompositionVerificationRefusal> {
        current.validate().map_err(|e| CompositionVerificationRefusal::graph(&e))?;
        let (relations, state) = match ledger {
            Some((ledger, cancel)) => {
                let relations =
                    shared_data_relations(current, ledger, cancel).map_err(Self::view_refusal)?;
                (relations, Some(ledger.state_digest()))
            }
            None => (Vec::new(), None),
        };
        let (artifact, _) =
            CompositionVerificationArtifact::seal(previous, current, effects, &relations, state)
                .map_err(|e| CompositionVerificationRefusal::artifact(&e))?;
        Ok(Self { artifact })
    }

    /// As [`Self::verify`], with every held or inherited calibration resolved
    /// against the shipped coverage records first: a record that does not name a
    /// currently attesting registry record ([`registry_record_identity`]) is
    /// treated as a requirement without a record, so the claim reports that a
    /// calibration is required instead of carrying a producer's declaration. The
    /// verification itself measures nothing and produces no interval.
    ///
    /// # Errors
    /// As [`Self::verify`].
    pub fn verify_against_registry(
        previous: Option<&CompositionGraph>,
        current: &CompositionGraph,
        effects: &BTreeMap<String, ObservedEffect>,
        ledger: Option<(&EvidenceLedger, &CancellationToken)>,
    ) -> Result<Self, CompositionVerificationRefusal> {
        let previous = previous.map(registry_resolved_graph).transpose()?;
        let current = registry_resolved_graph(current)?;
        Self::verify(previous.as_ref(), &current, effects, ledger)
    }

    fn view_refusal(error: ViewError) -> CompositionVerificationRefusal {
        match error {
            ViewError::Graph(inner) => CompositionVerificationRefusal::graph(&inner),
            ViewError::Ledger(inner) => {
                let (code, detail, message) = inner.refusal();
                CompositionVerificationRefusal {
                    code,
                    detail: detail.to_owned(),
                    message: message.to_owned(),
                    remedy: Some(
                        "register every source a node names in the evidence ledger".to_owned(),
                    ),
                }
            }
            ViewError::Cancelled => CompositionVerificationRefusal {
                code: antecedent_core::reason_code!("cancelled_no_claim"),
                detail: "composition_verification.cancelled".to_owned(),
                message: "the shared-data query was cancelled; no classification was made"
                    .to_owned(),
                remedy: Some("rerun the verification without cancelling".to_owned()),
            },
            _ => CompositionVerificationRefusal {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "composition_verification.limit_exceeded".to_owned(),
                message: error.to_string(),
                remedy: Some("verify a smaller composition or fewer evidence sources".to_owned()),
            },
        }
    }

    /// The classification of every claim.
    #[must_use]
    pub fn revalidation(&self) -> &GraphRevalidation {
        self.artifact.revalidation()
    }

    /// The classification of one claim.
    #[must_use]
    pub fn classify(&self, key: &str) -> Option<&ClaimRevalidation> {
        self.artifact.revalidation().claim(key)
    }

    /// The lineage report of one claim.
    ///
    /// # Errors
    /// A key that is not in the current graph.
    pub fn view(&self, key: &str) -> Result<CompositionView, CompositionVerificationRefusal> {
        self.artifact.view(key).map_err(Self::view_refusal)
    }

    /// The claim when it is usable as it stands.
    ///
    /// # Errors
    /// The typed refusal of a claim that must be recomputed or whose license
    /// was withdrawn (calibration inherited from an object that no longer
    /// holds it refuses with `cell_not_licensed`).
    pub fn require_valid(
        &self,
        key: &str,
    ) -> Result<&ClaimRevalidation, CompositionVerificationRefusal> {
        self.artifact
            .revalidation()
            .require_valid(key)
            .map_err(|r| CompositionVerificationRefusal::from_structured(&r))
    }

    /// BLAKE3 identity (hex) of the verification.
    #[must_use]
    pub fn identity(&self) -> &str {
        self.artifact.identity()
    }

    /// Write `composition_verification_v1`.
    ///
    /// # Errors
    /// A blank artifact id or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, CompositionVerificationRefusal> {
        self.artifact.export(artifact_id).map_err(|e| CompositionVerificationRefusal::artifact(&e))
    }
}

/// A verification consumed from bytes.
#[derive(Clone, Debug)]
pub struct ConsumedCompositionVerification {
    artifact: CompositionVerificationArtifact,
}

impl ConsumedCompositionVerification {
    /// The replayed classification of every claim.
    #[must_use]
    pub fn revalidation(&self) -> &GraphRevalidation {
        self.artifact.revalidation()
    }

    /// The current graph.
    #[must_use]
    pub const fn current(&self) -> &CompositionGraph {
        self.artifact.current()
    }

    /// The lineage report of one claim.
    ///
    /// # Errors
    /// A key that is not in the current graph.
    pub fn view(&self, key: &str) -> Result<CompositionView, CompositionVerificationRefusal> {
        self.artifact.view(key).map_err(CompositionVerification::view_refusal)
    }

    /// The claim when it is usable as it stands, or its typed refusal.
    ///
    /// # Errors
    /// The refusal of a claim that must be recomputed or was withdrawn.
    pub fn require_valid(
        &self,
        key: &str,
    ) -> Result<&ClaimRevalidation, CompositionVerificationRefusal> {
        self.artifact
            .revalidation()
            .require_valid(key)
            .map_err(|r| CompositionVerificationRefusal::from_structured(&r))
    }

    /// Check the declared shared-data relations and ledger state against a
    /// ledger the consumer holds.
    ///
    /// # Errors
    /// A ledger at another state or whose overlap differs from the declared.
    pub fn verify_ledger(
        &self,
        ledger: &EvidenceLedger,
        cancel: &CancellationToken,
    ) -> Result<(), CompositionVerificationRefusal> {
        self.artifact
            .verify_ledger(ledger, cancel)
            .map_err(|e| CompositionVerificationRefusal::artifact(&e))
    }

    /// BLAKE3 identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        self.artifact.identity()
    }
}

/// Consume `composition_verification_v1`: rebuild both graphs, classify
/// again and require every stored class, reason, withdrawal, authority,
/// calibration state and lineage identity to equal the derivation, against an
/// artifact identity the consumer retained independently of the bytes.
///
/// # Errors
/// Corruption, another major version, a bound exceeded, a graph the library
/// refuses, a stored classification that does not replay, or an identity that
/// does not match.
pub fn consume_composition_verification_artifact(
    bytes: &[u8],
    expected_identity: Option<&str>,
) -> Result<ConsumedCompositionVerification, CompositionVerificationRefusal> {
    CompositionVerificationArtifact::from_bytes(bytes, expected_identity)
        .map(|artifact| ConsumedCompositionVerification { artifact })
        .map_err(|e| CompositionVerificationRefusal::artifact(&e))
}

#[path = "flagship_mixed.rs"]
mod flagship_mixed;

pub use flagship_mixed::{
    DecisionArtifact, DecisionAuthority, DecisionAuthorityBinding, FlagshipDependence,
    FlagshipError, FlagshipWorkflow, LoadedDecision, MixedDecision, PeriodReplay, PriorReplay,
    UtilityReplay, load_decision, replay_decision,
};

#[cfg(test)]
mod flagship_mixed_tests {
    use antecedent_contract::bridge::SourceMappingDependence;
    use antecedent_contract::mixed_module::{CutFeedback, MixedModulePolicy};
    use antecedent_core::ExecutionContext;
    use antecedent_expr::execution_counts::count_static_work;
    use antecedent_prob::fit_counts::count_bayesian_work;
    use antecedent_stats::fit_counts::count_least_squares_solves;

    use super::{FlagshipDependence, FlagshipError, FlagshipWorkflow, Revalidation, load_decision};
    use crate::analysis::LinearDependence;

    /// Replays measure per-thread work counters, so they run on the test's own
    /// thread with a larger stack rather than on the thread `execute` spawns.
    fn on_large_stack(run: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(run)
            .expect("spawn the workflow test")
            .join()
            .expect("the workflow test panicked");
    }

    fn context() -> ExecutionContext {
        ExecutionContext::for_tests(0xF1A6)
    }

    #[test]
    fn decision_keeps_lawful_dependence_and_joint_uncertainty() {
        let workflow = FlagshipWorkflow::execute(&context(), FlagshipDependence::Independent)
            .expect("workflow");
        let decision = workflow.decision().expect("current decision");
        assert_eq!(workflow.cluster_design(), "cluster");
        assert!(workflow.cluster_uncertainty().contains("cluster"));
        assert!(workflow.cluster_variance() > 0.0);
        assert_eq!(workflow.temporal_means(), 2);
        assert!(workflow.cost_draws() > 0);
        assert!(matches!(decision.dependence, LinearDependence::DeclaredIndependent));
        assert!(matches!(
            decision.margin.dependence,
            SourceMappingDependence::SlopeCovariance { covariance, .. }
                if covariance.to_bits() != 0.0f64.to_bits()
        ));
        let mean = decision.joint.value("mean").expect("joint mean");
        let variance = decision.joint.value("variance").expect("joint variance");
        let margin = decision.margin.variance;
        let temporal = decision.temporal_variance;
        let cost = decision.cost_variance;
        let mut expected = 0.0;
        expected += margin;
        expected += temporal;
        expected += cost;
        let mut dropped = 0.0;
        dropped += margin;
        dropped += cost;
        assert!(temporal > 0.0);
        assert_eq!(variance.to_bits(), expected.to_bits());
        assert_ne!(variance.to_bits(), dropped.to_bits());
        assert_ne!(decision.margin.mean.to_bits(), decision.margin.plug_in_mean.to_bits());
        let act = decision.evaluation.actions.iter().find(|action| action.id == "act");
        assert_eq!(act.expect("act").expected_utility.to_bits(), mean.to_bits());
        assert_eq!(decision.joint.route(), "linear_moments");
    }

    #[test]
    fn utility_only_change_performs_zero_causal_refits() {
        on_large_stack(|| {
            let mut workflow =
                FlagshipWorkflow::execute(&context(), FlagshipDependence::Independent)
                    .expect("workflow");
            let effect = workflow.cluster_effect().to_bits();
            let temporal = workflow.temporal_effect().to_bits();
            let temporal_variance =
                workflow.decision().expect("current decision").temporal_variance.to_bits();
            let cost = workflow.cost_mean().to_bits();
            let before = workflow
                .decision()
                .expect("current decision")
                .joint
                .value("mean")
                .expect("mean")
                .to_bits();
            let (((replay, bayesian), temporal_work), solves) = count_least_squares_solves(|| {
                count_static_work(|| {
                    count_bayesian_work(|| workflow.utility_only(3.0, &context()).expect("utility"))
                })
            });
            assert_eq!(solves, 0);
            assert_eq!(bayesian.model_fits, 0);
            assert_eq!(bayesian.posterior_draws, 0);
            assert_eq!(temporal_work.factor_builds, 0);
            assert_eq!(replay.temporal_factor_builds, 0);
            assert_eq!(replay.temporal_identifications, 0);
            assert_eq!(replay.bayesian_model_fits, 0);
            assert_eq!(replay.bayesian_posterior_draws, 0);
            assert_eq!(replay.bayesian_identifications, 0);
            assert_eq!(workflow.cluster_trials(), 1);
            assert_eq!(workflow.cluster_effect().to_bits(), effect);
            assert_eq!(workflow.temporal_effect().to_bits(), temporal);
            assert_eq!(
                workflow.decision().expect("current decision").temporal_variance.to_bits(),
                temporal_variance
            );
            assert_eq!(workflow.cost_mean().to_bits(), cost);
            assert_ne!(
                workflow
                    .decision()
                    .expect("current decision")
                    .joint
                    .value("mean")
                    .expect("mean")
                    .to_bits(),
                before
            );
        });
    }

    #[test]
    fn prior_change_invalidates_only_the_cost_posterior() {
        on_large_stack(|| {
            let mut workflow =
                FlagshipWorkflow::execute(&context(), FlagshipDependence::Independent)
                    .expect("workflow");
            let effect = workflow.cluster_effect().to_bits();
            let temporal = workflow.temporal_effect().to_bits();
            let temporal_variance =
                workflow.decision().expect("current decision").temporal_variance.to_bits();
            let cost = workflow.cost_mean().to_bits();
            let ((replay, temporal_work), bayesian) = count_bayesian_work(|| {
                count_static_work(|| workflow.prior_change(10.0, &context()).expect("prior"))
            });
            assert!(replay.score_recomputed);
            assert!(replay.identification_reused);
            assert!(replay.bayesian_model_fits > 0);
            assert_eq!(replay.bayesian_identifications, 0);
            assert_eq!(bayesian.model_fits, replay.bayesian_model_fits);
            assert_eq!(temporal_work.factor_builds, 0);
            assert_eq!(workflow.cluster_trials(), 1);
            assert_eq!(workflow.cluster_effect().to_bits(), effect);
            assert_eq!(workflow.temporal_effect().to_bits(), temporal);
            assert_eq!(
                workflow.decision().expect("current decision").temporal_variance.to_bits(),
                temporal_variance
            );
            assert_ne!(workflow.cost_mean().to_bits(), cost);
        });
    }

    #[test]
    fn period_change_invalidates_only_the_temporal_products() {
        on_large_stack(|| {
            let mut workflow =
                FlagshipWorkflow::execute(&context(), FlagshipDependence::Independent)
                    .expect("workflow");
            let effect = workflow.cluster_effect().to_bits();
            let cost = workflow.cost_mean().to_bits();
            let (replay, bayesian) = count_bayesian_work(|| {
                workflow.period_change((0, 49), &context()).expect("period")
            });
            assert!(replay.score_recomputed);
            assert!(replay.identification_reused);
            assert!(replay.temporal_factor_builds > 0);
            assert_eq!(replay.temporal_identifications, 0);
            assert_eq!(bayesian.model_fits, 0);
            assert_eq!(bayesian.posterior_draws, 0);
            assert_eq!(workflow.cluster_trials(), 1);
            assert_eq!(workflow.cluster_effect().to_bits(), effect);
            assert_eq!(workflow.cost_mean().to_bits(), cost);
            assert!(workflow.decision().expect("current decision").temporal_variance > 0.0);
        });
    }

    #[test]
    fn withdrawal_removes_the_affected_claim_and_leaves_the_temporal_claim() {
        let mut workflow = FlagshipWorkflow::execute(&context(), FlagshipDependence::Independent)
            .expect("workflow");
        let verification = workflow.withdraw_cluster().expect("withdrawal");
        let class = |key: &str| verification.classify(key).expect(key).class;
        assert_eq!(class("cluster_trial"), Revalidation::Withdrawn);
        assert_eq!(class("margin_bridge"), Revalidation::Withdrawn);
        assert_eq!(class("decision"), Revalidation::Withdrawn);
        assert_eq!(class("temporal_drift"), Revalidation::StillValid);
        assert_eq!(class("bayesian_cost"), Revalidation::StillValid);
    }

    #[test]
    fn loaded_artifact_reproduces_the_decision() {
        let workflow = FlagshipWorkflow::execute(&context(), FlagshipDependence::Independent)
            .expect("workflow");
        let artifact = workflow.export_decision().expect("export");
        let loaded = load_decision(
            &artifact.bytes,
            Some(&artifact.identity),
            workflow.authority(),
            &context(),
        )
        .expect("load");
        let decision = workflow.decision().expect("current decision");
        assert_eq!(loaded.mean.to_bits(), decision.joint.value("mean").expect("mean").to_bits());
        assert_eq!(
            loaded.variance.to_bits(),
            decision.joint.value("variance").expect("variance").to_bits()
        );
        assert_eq!(loaded.evaluation.verdict, decision.evaluation.verdict);
        assert_eq!(loaded.evaluation.contract_identity, decision.evaluation.contract_identity);
    }

    fn independent() -> FlagshipWorkflow {
        FlagshipWorkflow::execute(&context(), FlagshipDependence::Independent).expect("workflow")
    }

    fn declared(
        base: &FlagshipWorkflow,
        margin_temporal: f64,
        margin_cost: f64,
        temporal_cost: f64,
    ) -> FlagshipDependence {
        let decision = base.decision().expect("current decision");
        let (margin, temporal, cost) =
            (decision.margin.variance, decision.temporal_variance, decision.cost_variance);
        FlagshipDependence::Covariance {
            id: "flagship-cross".to_owned(),
            margin_temporal: margin_temporal * (margin * temporal).sqrt(),
            margin_cost: margin_cost * (margin * cost).sqrt(),
            temporal_cost: temporal_cost * (temporal * cost).sqrt(),
        }
    }

    #[test]
    fn declared_cross_covariance_enters_the_decision_variance() {
        let base = independent();
        let dependence = declared(&base, 0.3, -0.2, 0.1);
        let FlagshipDependence::Covariance { margin_temporal, margin_cost, temporal_cost, .. } =
            dependence.clone()
        else {
            panic!("a declared covariance");
        };
        let workflow = FlagshipWorkflow::execute(&context(), dependence).expect("declared");
        let decision = workflow.decision().expect("current decision");
        assert_eq!(
            decision.dependence,
            LinearDependence::Covariance { id: "flagship-cross".to_owned() }
        );
        // Coefficients (1, 1, -1) at benefit 1: Var = sum c_i c_j S_ij
        //   = S_mm + S_tt + S_cc + 2 (S_mt - S_mc - S_tc).
        let expected = decision.margin.variance
            + decision.temporal_variance
            + decision.cost_variance
            + 2.0 * (margin_temporal - margin_cost - temporal_cost);
        let variance = decision.joint.value("variance").expect("joint variance");
        assert!((variance - expected).abs() <= 1.0e-9 * expected.abs(), "{variance} vs {expected}");
        let independent_variance =
            base.decision().expect("current decision").joint.value("variance").expect("variance");
        assert!((variance - independent_variance).abs() > 1.0e-9 * independent_variance);
        assert_eq!(
            decision.joint.value("mean").expect("mean").to_bits(),
            base.decision().expect("current decision").joint.value("mean").expect("mean").to_bits()
        );
    }

    #[test]
    fn a_covariance_that_is_not_positive_semidefinite_is_refused() {
        let base = independent();
        let dependence = declared(&base, 0.9, 0.9, -0.9);
        match FlagshipWorkflow::execute(&context(), dependence) {
            Err(FlagshipError::Dependence(_)) => {}
            Err(other) => panic!("expected a dependence refusal, got {other}"),
            Ok(_) => panic!("a non positive semidefinite covariance was accepted"),
        }
    }

    #[test]
    fn exported_decision_round_trips_the_declared_dependence() {
        let base = independent();
        let base_artifact = base.export_decision().expect("export");
        let workflow = FlagshipWorkflow::execute(&context(), declared(&base, 0.3, -0.2, 0.1))
            .expect("declared");
        let artifact = workflow.export_decision().expect("export");
        assert_ne!(artifact.identity, base_artifact.identity);
        let loaded = load_decision(
            &artifact.bytes,
            Some(&artifact.identity),
            workflow.authority(),
            &context(),
        )
        .expect("load");
        let decision = workflow.decision().expect("current decision");
        assert_eq!(
            loaded.variance.to_bits(),
            decision.joint.value("variance").expect("variance").to_bits()
        );
        assert_eq!(loaded.mean.to_bits(), decision.joint.value("mean").expect("mean").to_bits());
        assert!(
            load_decision(
                &artifact.bytes,
                Some(&base_artifact.identity),
                workflow.authority(),
                &context()
            )
            .is_err()
        );
    }

    fn modular(policy: CutFeedback) -> MixedModulePolicy {
        MixedModulePolicy::declared_modular("shared cost parameter; priors reconciled", policy)
            .expect("a non-blank reconciliation")
    }

    fn reloaded_variance(workflow: &FlagshipWorkflow) -> (f64, String) {
        let artifact = workflow.export_decision().expect("export");
        let loaded = load_decision(
            &artifact.bytes,
            Some(&artifact.identity),
            workflow.authority(),
            &context(),
        )
        .expect("load");
        (loaded.variance, artifact.identity)
    }

    #[test]
    fn cut_and_feedback_give_different_reloaded_variances() {
        let base = independent();
        let dependence = declared(&base, 0.3, -0.2, 0.1);
        let FlagshipDependence::Covariance { margin_temporal, margin_cost, temporal_cost, .. } =
            dependence.clone()
        else {
            panic!("a declared covariance");
        };
        let run = |policy: &MixedModulePolicy| {
            FlagshipWorkflow::execute_with_policy(&context(), dependence.clone(), policy)
                .expect("modular workflow")
        };
        let cut = run(&modular(CutFeedback::Cut));
        let feedback = run(&modular(CutFeedback::Feedback));
        let diagonal = base.decision().expect("current decision").margin.variance
            + base.decision().expect("current decision").temporal_variance
            + base.decision().expect("current decision").cost_variance;
        // Coefficients (1, 1, -1) at benefit 1. Cut zeroes the cost cross-terms:
        //   Var_cut = S_mm + S_tt + S_cc + 2 S_mt.
        // Feedback keeps them: Var_feedback = Var_cut - 2 (S_mc + S_tc).
        let expected_cut = diagonal + 2.0 * margin_temporal;
        let expected_feedback = diagonal + 2.0 * (margin_temporal - margin_cost - temporal_cost);
        let (cut_variance, cut_identity) = reloaded_variance(&cut);
        let (feedback_variance, feedback_identity) = reloaded_variance(&feedback);
        assert!((cut_variance - expected_cut).abs() <= 1.0e-9 * expected_cut.abs());
        assert!((feedback_variance - expected_feedback).abs() <= 1.0e-9 * expected_feedback.abs());
        assert!((cut_variance - feedback_variance).abs() > 1.0e-9 * cut_variance);
        assert_ne!(cut_identity, feedback_identity);
        // Feedback carries the same numbers as the coherent joint, but the policy
        // is part of the sealed claim, so the identities differ.
        let coherent = FlagshipWorkflow::execute(&context(), dependence).expect("coherent");
        let (coherent_variance, coherent_identity) = reloaded_variance(&coherent);
        assert!((coherent_variance - expected_feedback).abs() <= 1.0e-9 * expected_feedback.abs());
        assert_ne!(coherent_identity, feedback_identity);
        assert_ne!(coherent_identity, cut_identity);
    }

    #[test]
    fn feedback_without_a_declared_covariance_is_refused_and_cut_is_independent() {
        match FlagshipWorkflow::execute_with_policy(
            &context(),
            FlagshipDependence::Independent,
            &modular(CutFeedback::Feedback),
        ) {
            Err(FlagshipError::Policy(_)) => {}
            Err(other) => panic!("expected a policy refusal, got {other}"),
            Ok(_) => panic!("feedback over independence was accepted"),
        }
        let base = independent();
        let cut = FlagshipWorkflow::execute_with_policy(
            &context(),
            FlagshipDependence::Independent,
            &modular(CutFeedback::Cut),
        )
        .expect("cut");
        assert_eq!(reloaded_variance(&cut).0.to_bits(), reloaded_variance(&base).0.to_bits());
        assert_eq!(reloaded_variance(&cut).1, reloaded_variance(&base).1);
    }

    #[test]
    fn execute_runs_from_a_default_sized_test_thread() {
        let trials = std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(|| independent().cluster_trials())
            .expect("spawn the workflow test")
            .join()
            .expect("the workflow test panicked");
        assert_eq!(trials, 1);
    }
}
