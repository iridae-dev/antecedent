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
//! The verification states no interval, posterior or calibration record. A
//! calibration declaration names a record identity and a coordinate that the
//! producer supplies; whether the record was measured is the producer's
//! claim and is not checked here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::ExternalRefusal;
use antecedent_contract::claim::{ClaimEnvelope, ClaimFreshness};
use antecedent_contract::evidence_ledger::{EvidenceLedger, ResultId};
use antecedent_core::recalc::{RecalcPlan, Stage};
use antecedent_core::{
    CancellationToken, ContractIdentities, QueryId, SemanticDigest, StateVersion,
};
use antecedent_io::IoError;
use antecedent_io::composition_verification_artifact::{
    CompositionVerificationArtifact, CompositionVerificationArtifactError,
};
use antecedent_state::{InvalidationLog, InvalidationTarget};

pub use antecedent_contract::composition_verification::{
    BoundaryAssumption, BoundaryTransformation, CalibrationDeclaration, CalibrationLineage,
    CalibrationState, ClaimRevalidation, ClaimStanding, CompositionEdge, CompositionGraph, CompositionNode,
    CompositionView, DependencyKind, GraphError, GraphRevalidation, LineageEdge, LineageEntry,
    NodeKind, NumericalCause, NumericalReason, ObservedEffect, PriorAuthority, SharedDataRelation,
    SupportStanding, TrustStanding, UnsupportedCoordinate, ViewError, Withdrawal, WithdrawalKind,
    shared_data_relations,
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
