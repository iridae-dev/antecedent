//! Durable identification-repair artifact (`repair_search_receipt_v1`) and its
//! independent consumer.
//!
//! The artifact stores the repair's premises (the failed contract's family and
//! graph, query and limits, every declared candidate, the objective and the
//! search limits), its data lineage (the base evidence catalog of a transport
//! contract, or the base observed law of a back-door contract) and the report:
//! the unresolved obligations (ids, kinds, source proof step), every candidate
//! subset's classification with its failure reasons, derivation and hypothetical
//! evidence delta, the ranked sufficient subsets and the search receipt
//! (explored, unevaluated, beyond-depth). A premises digest, a data digest and a
//! report digest over both seal it.
//!
//! The consumer first checks the container, the version and the stored limits
//! against its own maxima before any work, then recomputes every digest. It then
//! replays each stored subset's family check from the stored hypothetical delta
//! with the family's own theorem checker (the transport identifier and verifier,
//! or the back-door identifier) and refuses a classification that differs, and
//! finally replays the whole search under the stored limits and accepts only an
//! identical report. A resealed edit of a candidate, an obligation, a
//! classification or the receipt is refused with
//! `repair_artifact.replay_mismatch` naming the section.
//!
//! A search stopped by a budget is retained as such: unevaluated subsets are
//! listed, never classified, and nothing in the artifact states that no repair
//! exists. Replay does not protect against a producer that states a different
//! contract or candidate universe and re-seals honestly (the artifact is then a
//! correct repair report of those stated inputs), and it does not check that the
//! arriving studies will deliver their declared evidence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::assumption::{AssumptionSource, AssumptionStatus};
use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceObligation, EvidenceObligationKind,
    EvidenceObligationSpec, EvidenceRegime, ExecutionContext, IdentityDomain,
    InterventionAssignment, NodeRef, ObligationKind, ObligationProvenance, ObligationRecord,
    ObligationRegime, ObligationScope, SearchLimits, SearchReceipt, SearchStop, VariableId,
    reason_code,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{ClassicalTransportQuery, SidLimits};
use antecedent_io::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::error::IoError;
use antecedent_io::query_wire::ValueWire;
use antecedent_io::reader::ArtifactReader;
use antecedent_io::transport_catalog_wire::{EvidenceCatalogWire, EvidenceRegimeWire};
use antecedent_io::wire::{AdmgWire, ArtifactKind, DagWire, ProvenanceWire, SemanticVersion};
use serde::{Deserialize, Serialize};

use crate::repair::{
    BackdoorRepairFamily, FamilyVerdict, REPAIR_MAX_CANDIDATES, REPAIR_MAX_DEPTH,
    REPAIR_MAX_MEMORY_BYTES, REPAIR_MAX_OPERATIONS, REPAIR_RECEIPT_REGIONS, RepairCancelled,
    RepairClassification, RepairError, RepairFamily, RepairLimits, RepairObjective, RepairOutcome,
    RepairReport, TransportRepairFamily, repair,
};
use crate::{ZTransportFailureSnapshot, ZTransportFailureSnapshotWire, ZTransportRepairFamily};

use crate::study_candidate::{
    DurableStudyCandidate, ExpectedEvidence, StudyCostDeclaration, StudyKind, UnitRules,
};

/// Wire schema version.
pub const REPAIR_ARTIFACT_VERSION: u32 = 1;
/// Artifact kind.
pub const REPAIR_ARTIFACT_KIND: &str = "repair_search_receipt_v1";
/// Largest container, in bytes, a consumer reads.
pub const MAX_REPAIR_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
/// Most outcomes a stored report may carry: every subset of at most
/// [`REPAIR_MAX_DEPTH`] of [`REPAIR_MAX_CANDIDATES`] candidates, plus the
/// refused candidates.
pub const MAX_REPAIR_OUTCOMES: usize = 4096;
/// Most obligations a stored report may carry.
pub const MAX_REPAIR_OBLIGATIONS: usize = 1024;

const BODY_SECTION: &str = "repair_body";

/// A refused artifact: a registered reason code and a stable detail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct RepairArtifactError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `repair_artifact.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl RepairArtifactError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("transport_not_certified"),
            detail: "repair_artifact.invalid_artifact",
            message: message.into(),
        }
    }

    fn unsupported_version(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("transport_not_certified"),
            detail: "repair_artifact.unsupported_version",
            message: message.into(),
        }
    }

    fn bounds(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("route_not_supported"),
            detail: "repair_artifact.bounds_exceeded",
            message: message.into(),
        }
    }

    fn budget(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("transport_budget_cancel"),
            detail: "repair_artifact.budget",
            message: message.into(),
        }
    }

    fn mismatch(section: &str, message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("transport_not_certified"),
            detail: "repair_artifact.replay_mismatch",
            message: format!("{section}: {}", message.into()),
        }
    }

    fn from_io(error: &IoError) -> Self {
        match error {
            IoError::UnsupportedVersion { .. } | IoError::UnsupportedFormat { .. } => {
                Self::unsupported_version(error.to_string())
            }
            IoError::TooLarge => Self::bounds("the repair artifact exceeds the consumer bound"),
            other => Self::invalid(other.to_string()),
        }
    }
}

// ------------------------------------------------------------------- wires

/// One stored evidence obligation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObligationWire {
    /// Declared full scientific coordinates, keyed by structural variable ID.
    /// Legacy structural obligations omit this field rather than inventing units.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quantities: Vec<(u32, antecedent_io::quantity_wire::ScientificQuantityWire)>,
    /// Stable id.
    pub id: String,
    /// Kind name.
    pub kind: String,
    /// Scope name.
    pub scope: String,
    /// Horizon step of a horizon scope.
    pub horizon: Option<u32>,
    /// Variables of the requested law.
    pub variables: Vec<u32>,
    /// Population the evidence is owed in.
    pub population: Option<String>,
    /// Hard-intervention set of the requested regime.
    pub interventions: Vec<u32>,
    /// Conditioning coordinates of the requested regime.
    pub conditioned_on: Vec<u32>,
    /// Whether one joint law is required.
    pub joint: bool,
    /// Why the evidence is owed.
    pub reason: String,
    /// Contract slots the evidence fills.
    pub required_slots: Vec<String>,
    /// Rows to add, for an increase-sample obligation.
    pub min_additional_samples: Option<u64>,
    /// Obligation family or theorem route.
    pub family: String,
    /// Contract or failure the obligation was read from.
    pub source: String,
    /// Source proof step, when known.
    pub proof_step: Option<String>,
}

/// One law a study declares it would deliver.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedEvidenceWire {
    /// Population of the law.
    pub population: String,
    /// Hard-intervention set.
    pub interventions: Vec<u32>,
    /// Concrete intervention levels.
    pub levels: Vec<(u32, ValueWire)>,
    /// Conditioning coordinates.
    pub conditioned_on: Vec<u32>,
    /// Variables of the law.
    pub measured: Vec<u32>,
    /// `None` is a joint law; `Some` lists the separately available marginals.
    pub separate_marginals: Option<Vec<u32>>,
}

/// One declared durable study candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableStudyCandidateWire {
    /// Semantic id (verified on decode).
    pub semantic_id: String,
    /// Human label.
    pub label: String,
    /// Study kind name.
    pub kind: String,
    /// Population.
    pub population: String,
    /// Interventions.
    pub interventions: Vec<u32>,
    /// Measured variables.
    pub measured: Vec<u32>,
    /// Whether the margin is observed jointly.
    pub joint_measurement: bool,
    /// Planned sample size.
    pub sample_size: u64,
    /// Recruitment declaration.
    pub recruitment: String,
    /// Timing declaration.
    pub timing: String,
    /// Unit of analysis.
    pub unit: String,
    /// Cluster identity.
    pub cluster: Option<String>,
    /// Whole-cluster sampling flag.
    pub whole_cluster_sampling: bool,
    /// Cost units.
    pub cost_units: u64,
    /// Cost unit label.
    pub cost_unit_label: String,
    /// Sample budget.
    pub sample_budget: u64,
    /// Declared feasibility.
    pub feasible: bool,
    /// Feasibility notes.
    pub feasibility_notes: Vec<String>,
    /// Expected evidence.
    pub expected_evidence: Vec<ExpectedEvidenceWire>,
    /// External provider.
    pub external_provider: Option<String>,
}

/// One unresolved assumption record of a back-door contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssumptionWire {
    /// Record id.
    pub id: String,
    /// Scope name.
    pub scope: String,
    /// Horizon step of a horizon scope.
    pub horizon: Option<u32>,
    /// Description.
    pub description: String,
    /// Required check.
    pub required_check: Option<String>,
}

/// The transport contract's structure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportPremisesWire {
    /// Causal graph shared by source and target.
    pub graph: AdmgWire,
    /// Selection targets.
    pub selection_targets: Vec<u32>,
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatments.
    pub treatments: Vec<u32>,
    /// Source population.
    pub source: String,
    /// Target population.
    pub target: String,
    /// Identification step limit.
    pub sid_steps: usize,
    /// Identification depth limit.
    pub sid_depth: usize,
}

/// The back-door contract's structure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackdoorPremisesWire {
    /// Causal graph.
    pub graph: DagWire,
    /// Treatment.
    pub treatment: u32,
    /// Outcome.
    pub outcome: u32,
    /// Population of the contract.
    pub population: String,
    /// Unresolved assumption records.
    pub assumptions: Vec<AssumptionWire>,
}

/// Search limits in force.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairLimitsWire {
    /// Operation limit.
    pub operations: usize,
    /// Depth limit (largest subset size).
    pub depth: usize,
    /// Declared memory cap in bytes.
    pub memory_limit_bytes: u64,
}

/// Frozen z-transport proof, evidence state and theorem-verification limits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportPremisesWire {
    /// Independently replayable missing-evidence snapshot.
    pub snapshot: ZTransportFailureSnapshotWire,
    /// Theorem-verification operation bound.
    pub sid_steps: usize,
    /// Theorem-verification depth bound.
    pub sid_depth: usize,
}

/// Everything the report is a function of, except the data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairPremisesWire {
    /// `transport` or `backdoor`.
    pub family: String,
    /// The failed contract's identity.
    pub contract: String,
    /// Objective name.
    pub objective: String,
    /// Transport structure, for the transport family.
    pub transport: Option<TransportPremisesWire>,
    /// Back-door structure, for the back-door family.
    pub backdoor: Option<BackdoorPremisesWire>,
    /// Checked z-transport snapshot and verification limits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_transport: Option<ZTransportPremisesWire>,
    /// Every declared candidate, in semantic-id order.
    pub candidates: Vec<DurableStudyCandidateWire>,
    /// Search limits.
    pub limits: RepairLimitsWire,
}

/// Data lineage: the base evidence the contract failed on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairDataWire {
    /// Base evidence catalog (transport family).
    pub catalog: Option<EvidenceCatalogWire>,
    /// Variables observed jointly with treatment and outcome (back-door family).
    pub base_law: Option<Vec<u32>>,
}

/// The checked derivation behind a sufficient subset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivationWire {
    /// Theorem-specific checker.
    pub checker: String,
    /// Stable derivation facts.
    pub steps: Vec<String>,
    /// Whether independent re-verification passed.
    pub verified: bool,
}

/// One subset's stored outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairOutcomeWire {
    /// Semantic candidate ids, sorted.
    pub candidates: Vec<String>,
    /// `verified_sufficient`, `insufficient`, `not_certified`, `invalid` or
    /// `unevaluated`.
    pub classification: String,
    /// Total cost units.
    pub cost_units: u64,
    /// Shared cost unit label.
    pub cost_unit_label: Option<String>,
    /// Total sample budget.
    pub sample_budget: u64,
    /// Failure or downgrade reasons.
    pub reasons: Vec<String>,
    /// Obligation ids the evidence passes the necessary screen for.
    pub addressed: Vec<String>,
    /// Obligation ids nothing addresses.
    pub unmet: Vec<String>,
    /// Verified derivation.
    pub derivation: Option<DerivationWire>,
    /// The hypothetical evidence delta the family check ran on; `None` for an
    /// invalid candidate or an unevaluated subset.
    pub delta: Option<Vec<EvidenceRegimeWire>>,
}

/// The receipt of a budget stop.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairStopWire {
    /// `search.<stop>`.
    pub stop: String,
    /// Operation limit in force.
    pub operations_limit: usize,
    /// Depth limit in force.
    pub depth_limit: usize,
    /// Memory cap in force.
    pub memory_limit_bytes: Option<u64>,
    /// Operations charged.
    pub operations_consumed: Option<usize>,
    /// Depth reached.
    pub depth_reached: Option<usize>,
    /// Explored subsets.
    pub explored: Vec<String>,
    /// Unevaluated subsets.
    pub unevaluated: Vec<String>,
}

/// What the subset search consumed and left.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairReceiptWire {
    /// Operations charged.
    pub operations_consumed: usize,
    /// Deepest subset size a charge was attempted at.
    pub depth_reached: usize,
    /// Effective memory cap in force.
    pub memory_limit_bytes: u64,
    /// Subsets evaluated, in order.
    pub explored: Vec<String>,
    /// Subsets left unevaluated.
    pub unevaluated: Vec<String>,
    /// Total unevaluated subsets, listed or not.
    pub unevaluated_total: usize,
    /// Supersets of a sufficient subset skipped (ranking only).
    pub dominated_skipped: usize,
    /// Whether subsets larger than the declared depth exist and were not
    /// examined; never a verdict about them.
    pub beyond_declared_depth: bool,
    /// The budget stop, when one ended the search.
    pub stop: Option<RepairStopWire>,
}

/// The stored report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairReportWire {
    /// Unresolved obligations the candidates were screened against.
    pub obligations: Vec<ObligationWire>,
    /// Every outcome, invalid candidates first, then subsets in search order.
    pub outcomes: Vec<RepairOutcomeWire>,
    /// Indices of verified-sufficient outcomes, best first.
    pub ranked_sufficient: Vec<usize>,
    /// The search receipt.
    pub receipt: RepairReceiptWire,
    /// `(code, detail)` when no subset is verified sufficient.
    pub status: Option<(String, String)>,
    /// `repaired`, `none_certified` (nothing certified within the declared
    /// candidates, never an impossibility claim) or `exhausted` (a budget stop
    /// with nothing certified).
    pub outcome: String,
}

/// A durable, independently replayable identification-repair report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairReportArtifact {
    /// Schema version.
    pub version: u32,
    /// `repair_search_receipt_v1`.
    pub kind: String,
    /// Premises.
    pub premises: RepairPremisesWire,
    /// Data lineage.
    pub data: RepairDataWire,
    /// Report.
    pub report: RepairReportWire,
    /// Digest of the premises.
    pub premises_digest: String,
    /// Digest of the data lineage.
    pub data_digest: String,
    /// Digest of both digests and the report.
    pub report_digest: String,
}

/// Maxima a consumer accepts; stored limits above them refuse before any work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepairConsumeLimits {
    /// Operation and depth maxima.
    pub search: SearchLimits,
    /// Memory maximum (further lowered by the context's hard limit).
    pub memory_limit_bytes: u64,
}

impl Default for RepairConsumeLimits {
    fn default() -> Self {
        Self {
            search: SearchLimits { operations: REPAIR_MAX_OPERATIONS, depth: REPAIR_MAX_DEPTH },
            memory_limit_bytes: REPAIR_MAX_MEMORY_BYTES,
        }
    }
}

/// The family a report was produced for, borrowed for export.
#[derive(Clone, Copy, Debug)]
pub enum RepairFamilyRef<'a> {
    /// A catalog-aware classical-transport contract.
    Transport(&'a TransportRepairFamily),
    /// A back-door adjustment contract.
    Backdoor(&'a BackdoorRepairFamily),
    /// A checked single-source z-transport missing-evidence contract.
    ZTransport(&'a ZTransportRepairFamily),
}

impl RepairFamilyRef<'_> {
    fn family(&self) -> &dyn RepairFamily {
        match self {
            Self::Transport(family) => *family,
            Self::Backdoor(family) => *family,
            Self::ZTransport(family) => *family,
        }
    }
}

// ----------------------------------------------------------------- helpers

fn raw(variables: &[VariableId]) -> Vec<u32> {
    variables.iter().map(|v| v.raw()).collect()
}

fn ids(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(VariableId::from_raw).collect()
}

fn strings(values: &[Arc<str>]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

fn arcs(values: &[String]) -> Arc<[Arc<str>]> {
    values.iter().map(|v| Arc::from(v.as_str())).collect()
}

/// Whether `nodes` are exactly the static variables `0..n` in order, the only
/// graph shape the wire (which carries no variable ids) can restore.
fn dense_static(nodes: &[NodeRef]) -> bool {
    nodes.iter().enumerate().all(|(i, node)| {
        u32::try_from(i).is_ok_and(|i| *node == NodeRef::Static(VariableId::from_raw(i)))
    })
}

fn digest<T: Serialize>(domain: IdentityDomain, value: &T) -> Result<String, RepairArtifactError> {
    antecedent_io::identity::digest_wire(domain, value)
        .map(|d| d.to_hex())
        .map_err(|e| RepairArtifactError::invalid(e.to_string()))
}

const fn objective_name(objective: RepairObjective) -> &'static str {
    match objective {
        RepairObjective::MinimizeCost => "minimize_cost",
        RepairObjective::MinimizeSampleBudget => "minimize_sample_budget",
    }
}

fn objective_from_name(name: &str) -> Result<RepairObjective, RepairArtifactError> {
    match name {
        "minimize_cost" => Ok(RepairObjective::MinimizeCost),
        "minimize_sample_budget" => Ok(RepairObjective::MinimizeSampleBudget),
        other => Err(RepairArtifactError::invalid(format!("unknown repair objective {other:?}"))),
    }
}

fn classification_from_name(name: &str) -> Result<RepairClassification, RepairArtifactError> {
    [
        RepairClassification::VerifiedSufficient,
        RepairClassification::Insufficient,
        RepairClassification::NotCertified,
        RepairClassification::Invalid,
        RepairClassification::Unevaluated,
    ]
    .into_iter()
    .find(|c| c.as_str() == name)
    .ok_or_else(|| RepairArtifactError::invalid(format!("unknown classification {name:?}")))
}

fn kind_from_name(name: &str) -> Result<StudyKind, RepairArtifactError> {
    [StudyKind::Experiment, StudyKind::Observation, StudyKind::SampleIncrease]
        .into_iter()
        .find(|k| k.as_str() == name)
        .ok_or_else(|| RepairArtifactError::invalid(format!("unknown study kind {name:?}")))
}

fn scope_parts(scope: ObligationScope) -> (String, Option<u32>) {
    match scope {
        ObligationScope::Horizon { horizon } => ("horizon".to_owned(), Some(horizon)),
        other => (other.as_str().to_owned(), None),
    }
}

fn scope_from_parts(
    name: &str,
    horizon: Option<u32>,
) -> Result<ObligationScope, RepairArtifactError> {
    match (name, horizon) {
        ("program", None) => Ok(ObligationScope::Program),
        ("factor", None) => Ok(ObligationScope::Factor),
        ("atom", None) => Ok(ObligationScope::Atom),
        ("horizon", Some(horizon)) => Ok(ObligationScope::Horizon { horizon }),
        _ => Err(RepairArtifactError::invalid(format!("unknown obligation scope {name:?}"))),
    }
}

/// The classification a family verdict means, given whether an unresolved
/// assumption obligation (never satisfiable by a study) blocks certification.
fn classify_verdict(verdict: &FamilyVerdict, blocked: bool) -> RepairClassification {
    match verdict {
        FamilyVerdict::Sufficient(derivation) if !derivation.verified || blocked => {
            RepairClassification::NotCertified
        }
        FamilyVerdict::Sufficient(_) => RepairClassification::VerifiedSufficient,
        FamilyVerdict::Insufficient { .. } => RepairClassification::Insufficient,
        FamilyVerdict::NotCertified { .. } => RepairClassification::NotCertified,
        FamilyVerdict::Invalid { .. } => RepairClassification::Invalid,
    }
}

// --------------------------------------------------------------- obligations

fn obligation_wire(obligation: &EvidenceObligation) -> ObligationWire {
    let (scope, horizon) = scope_parts(obligation.scope);
    ObligationWire {
        quantities: obligation
            .quantities
            .iter()
            .map(|(variable, quantity)| {
                (
                    variable.raw(),
                    antecedent_io::quantity_wire::ScientificQuantityWire::from(quantity),
                )
            })
            .collect(),
        id: obligation.id.to_string(),
        kind: obligation.kind.as_str().to_owned(),
        scope,
        horizon,
        variables: raw(&obligation.variables),
        population: obligation.population.as_ref().map(ToString::to_string),
        interventions: raw(&obligation.regime.interventions),
        conditioned_on: raw(&obligation.regime.conditioned_on),
        joint: obligation.regime.joint,
        reason: obligation.reason.to_string(),
        required_slots: strings(&obligation.required_slots),
        min_additional_samples: obligation.min_additional_samples,
        family: obligation.provenance.family.to_string(),
        source: obligation.provenance.source.to_string(),
        proof_step: obligation.provenance.proof_step.as_ref().map(ToString::to_string),
    }
}

fn obligation_from_wire(wire: &ObligationWire) -> Result<EvidenceObligation, RepairArtifactError> {
    let kind = EvidenceObligationKind::from_name(&wire.kind).ok_or_else(|| {
        RepairArtifactError::invalid(format!("unknown obligation kind {:?}", wire.kind))
    })?;
    let obligation = EvidenceObligation::try_new(EvidenceObligationSpec {
        quantities: {
            let mut quantities = std::collections::BTreeMap::new();
            for (variable, wire) in &wire.quantities {
                let quantity = antecedent_core::ScientificQuantity::try_from(wire.clone())
                    .map_err(RepairArtifactError::invalid)?;
                if quantities.insert(VariableId::from_raw(*variable), quantity).is_some() {
                    return Err(RepairArtifactError::invalid(
                        "duplicate scientific coordinate variable",
                    ));
                }
            }
            quantities
        },
        kind,
        scope: scope_from_parts(&wire.scope, wire.horizon)?,
        variables: ids(&wire.variables),
        population: wire.population.as_deref().map(Arc::from),
        regime: ObligationRegime {
            interventions: ids(&wire.interventions),
            conditioned_on: ids(&wire.conditioned_on),
            joint: wire.joint,
        },
        reason: Arc::from(wire.reason.as_str()),
        required_slots: arcs(&wire.required_slots),
        min_additional_samples: wire.min_additional_samples,
        provenance: ObligationProvenance {
            family: Arc::from(wire.family.as_str()),
            source: Arc::from(wire.source.as_str()),
            proof_step: wire.proof_step.as_deref().map(Arc::from),
        },
    })
    .map_err(|e| RepairArtifactError::invalid(e.to_string()))?;
    if obligation.id.as_ref() != wire.id {
        return Err(RepairArtifactError::invalid(
            "a stored obligation id does not match its content",
        ));
    }
    Ok(obligation)
}

// --------------------------------------------------------------- candidates

fn evidence_wire(evidence: &ExpectedEvidence) -> ExpectedEvidenceWire {
    ExpectedEvidenceWire {
        population: evidence.population.to_string(),
        interventions: raw(&evidence.interventions),
        levels: evidence
            .intervention_values
            .iter()
            .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
            .collect(),
        conditioned_on: raw(&evidence.conditioned_on),
        measured: raw(&evidence.measured),
        separate_marginals: match &evidence.distribution {
            DistributionAvailability::Joint => None,
            DistributionAvailability::SeparateMarginals { variables } => Some(raw(variables)),
        },
    }
}

fn candidate_wire(candidate: &DurableStudyCandidate) -> DurableStudyCandidateWire {
    DurableStudyCandidateWire {
        semantic_id: candidate.semantic_id().to_string(),
        label: candidate.label.to_string(),
        kind: candidate.kind.as_str().to_owned(),
        population: candidate.population.to_string(),
        interventions: raw(&candidate.interventions),
        measured: raw(&candidate.measured),
        joint_measurement: candidate.joint_measurement,
        sample_size: candidate.sample_size,
        recruitment: candidate.recruitment.to_string(),
        timing: candidate.timing.to_string(),
        unit: candidate.unit_rules.unit.to_string(),
        cluster: candidate.unit_rules.cluster.as_ref().map(ToString::to_string),
        whole_cluster_sampling: candidate.unit_rules.whole_cluster_sampling,
        cost_units: candidate.cost.units,
        cost_unit_label: candidate.cost.unit_label.to_string(),
        sample_budget: candidate.cost.sample_budget,
        feasible: candidate.feasible,
        feasibility_notes: strings(&candidate.feasibility_notes),
        expected_evidence: candidate.expected_evidence.iter().map(evidence_wire).collect(),
        external_provider: candidate.external_provider.as_ref().map(ToString::to_string),
    }
}

fn candidate_from_wire(
    wire: &DurableStudyCandidateWire,
) -> Result<DurableStudyCandidate, RepairArtifactError> {
    let candidate = DurableStudyCandidate {
        label: Arc::from(wire.label.as_str()),
        kind: kind_from_name(&wire.kind)?,
        population: Arc::from(wire.population.as_str()),
        interventions: ids(&wire.interventions),
        measured: ids(&wire.measured),
        joint_measurement: wire.joint_measurement,
        sample_size: wire.sample_size,
        recruitment: Arc::from(wire.recruitment.as_str()),
        timing: Arc::from(wire.timing.as_str()),
        unit_rules: UnitRules {
            unit: Arc::from(wire.unit.as_str()),
            cluster: wire.cluster.as_deref().map(Arc::from),
            whole_cluster_sampling: wire.whole_cluster_sampling,
        },
        cost: StudyCostDeclaration {
            units: wire.cost_units,
            unit_label: Arc::from(wire.cost_unit_label.as_str()),
            sample_budget: wire.sample_budget,
        },
        feasible: wire.feasible,
        feasibility_notes: arcs(&wire.feasibility_notes),
        expected_evidence: wire
            .expected_evidence
            .iter()
            .map(|e| ExpectedEvidence {
                population: Arc::from(e.population.as_str()),
                interventions: ids(&e.interventions),
                intervention_values: e
                    .levels
                    .iter()
                    .map(|(v, x)| InterventionAssignment {
                        variable: VariableId::from_raw(*v),
                        value: x.to_value(),
                    })
                    .collect(),
                conditioned_on: ids(&e.conditioned_on),
                measured: ids(&e.measured),
                distribution: e
                    .separate_marginals
                    .as_ref()
                    .map_or(DistributionAvailability::Joint, |v| {
                        DistributionAvailability::SeparateMarginals { variables: ids(v) }
                    }),
            })
            .collect(),
        external_provider: wire.external_provider.as_deref().map(Arc::from),
    };
    if candidate.semantic_id().as_ref() != wire.semantic_id {
        return Err(RepairArtifactError::invalid(
            "a stored candidate's semantic id does not match its content",
        ));
    }
    Ok(candidate)
}

impl DurableStudyCandidateWire {
    /// Encode a candidate, recording its semantic id.
    #[must_use]
    pub fn from_candidate(candidate: &DurableStudyCandidate) -> Self {
        candidate_wire(candidate)
    }

    /// Decode a candidate, verifying the stored semantic id against its content.
    ///
    /// # Errors
    /// `repair_artifact.invalid_artifact` for an unknown kind or a semantic id
    /// that does not match the content.
    pub fn to_candidate(&self) -> Result<DurableStudyCandidate, RepairArtifactError> {
        candidate_from_wire(self)
    }
}

// ------------------------------------------------------------------ outcomes

fn regimes_wire(
    regimes: &[EvidenceRegime],
) -> Result<Vec<EvidenceRegimeWire>, RepairArtifactError> {
    let catalog = EvidenceCatalog::try_new([], regimes.to_vec(), [], None)
        .map_err(|e| RepairArtifactError::invalid(e.to_string()))?;
    Ok(EvidenceCatalogWire::from_catalog(&catalog).regimes)
}

fn outcome_wire(
    outcome: &RepairOutcome,
    delta: Option<Vec<EvidenceRegimeWire>>,
) -> RepairOutcomeWire {
    RepairOutcomeWire {
        candidates: strings(&outcome.candidates),
        classification: outcome.classification.as_str().to_owned(),
        cost_units: outcome.cost_units,
        cost_unit_label: outcome.cost_unit_label.as_ref().map(ToString::to_string),
        sample_budget: outcome.sample_budget,
        reasons: outcome.reasons.clone(),
        addressed: strings(&outcome.addressed),
        unmet: strings(&outcome.unmet),
        derivation: outcome.derivation.as_ref().map(|d| DerivationWire {
            checker: d.checker.to_owned(),
            steps: d.steps.clone(),
            verified: d.verified,
        }),
        delta,
    }
}

fn stop_wire(receipt: &SearchReceipt) -> RepairStopWire {
    RepairStopWire {
        stop: receipt.stop.code().to_owned(),
        operations_limit: receipt.operations_limit,
        depth_limit: receipt.depth_limit,
        memory_limit_bytes: receipt.memory_limit_bytes,
        operations_consumed: receipt.operations_consumed,
        depth_reached: receipt.depth_reached,
        explored: receipt.explored.clone(),
        unevaluated: receipt.unevaluated.clone(),
    }
}

fn outcome_name(report: &RepairReport) -> &'static str {
    match report.status() {
        None => "repaired",
        Some((_, "identification_repair.budget")) => "exhausted",
        Some(_) => "none_certified",
    }
}

// --------------------------------------------------------------------- build

impl RepairReportArtifact {
    /// Export a finished repair as a sealed `repair_search_receipt_v1` artifact.
    ///
    /// `candidates` are every candidate the repair was declared over (invalid
    /// ones included, so a consumer can replay the refusal), and `family` the
    /// failed contract `report` was produced for.
    ///
    /// # Errors
    /// `repair_artifact.budget` for a report that cancellation stopped (no
    /// consumer could replay it); `repair_artifact.invalid_artifact` for a
    /// report that does not belong to `family` or `candidates`, a graph the wire
    /// cannot restore, or an encoding failure.
    #[allow(clippy::too_many_lines)] // One explicit wire encoding of the report.
    pub fn build(
        family: RepairFamilyRef<'_>,
        candidates: &[DurableStudyCandidate],
        report: &RepairReport,
    ) -> Result<Self, RepairArtifactError> {
        let owner = family.family();
        if owner.family_id() != report.family || owner.contract_id() != report.contract {
            return Err(RepairArtifactError::invalid(
                "the report was not produced for this repair family and contract",
            ));
        }
        if report.receipt.stop.as_ref().is_some_and(|r| r.stop == SearchStop::Cancelled) {
            return Err(RepairArtifactError::budget(
                "a repair stopped by cancellation cannot be replayed; repair again to export it",
            ));
        }
        let mut declared: Vec<&DurableStudyCandidate> = candidates.iter().collect();
        declared.sort_by(|a, b| {
            a.semantic_id().cmp(&b.semantic_id()).then_with(|| a.label.cmp(&b.label))
        });
        let mut by_id: BTreeMap<Arc<str>, &DurableStudyCandidate> = BTreeMap::new();
        for candidate in &declared {
            by_id.entry(candidate.semantic_id()).or_insert(candidate);
        }
        let limits = report.receipt.limits;
        let (transport, backdoor, z_transport, data) = match family {
            RepairFamilyRef::Transport(f) => {
                let diagram = f.diagram();
                if !dense_static(diagram.causal_graph().nodes()) {
                    return Err(RepairArtifactError::invalid(
                        "a transport graph must be the dense static variables 0..n to be stored",
                    ));
                }
                let query = f.query();
                (
                    Some(TransportPremisesWire {
                        graph: antecedent_io::admg_to_wire(diagram.causal_graph())
                            .map_err(|e| RepairArtifactError::invalid(e.to_string()))?,
                        selection_targets: raw(diagram.selection_targets()),
                        outcomes: raw(&query.outcomes),
                        treatments: raw(&query.treatments),
                        source: query.source.to_string(),
                        target: query.target.to_string(),
                        sid_steps: f.sid_limits().steps,
                        sid_depth: f.sid_limits().depth,
                    }),
                    None,
                    None,
                    RepairDataWire {
                        catalog: Some(EvidenceCatalogWire::from_catalog(f.base_catalog())),
                        base_law: None,
                    },
                )
            }
            RepairFamilyRef::Backdoor(f) => {
                if !dense_static(f.dag().nodes()) {
                    return Err(RepairArtifactError::invalid(
                        "a back-door graph must be the dense static variables 0..n to be stored",
                    ));
                }
                let assumptions = f
                    .unresolved_obligations()
                    .iter()
                    .filter(|o| o.provenance.family.as_ref() == "assumption_record")
                    .map(|o| {
                        let (scope, horizon) = scope_parts(o.scope);
                        AssumptionWire {
                            id: o.provenance.source.to_string(),
                            scope,
                            horizon,
                            description: o.reason.to_string(),
                            required_check: o
                                .provenance
                                .proof_step
                                .as_ref()
                                .map(ToString::to_string),
                        }
                    })
                    .collect();
                (
                    None,
                    Some(BackdoorPremisesWire {
                        graph: antecedent_io::dag_to_wire(f.dag())
                            .map_err(|e| RepairArtifactError::invalid(e.to_string()))?,
                        treatment: f.treatment().raw(),
                        outcome: f.outcome().raw(),
                        population: f.population().to_owned(),
                        assumptions,
                    }),
                    None,
                    RepairDataWire { catalog: None, base_law: Some(raw(&f.base_law())) },
                )
            }
            RepairFamilyRef::ZTransport(f) => {
                let snapshot = f
                    .snapshot()
                    .to_wire()
                    .map_err(|e| RepairArtifactError::invalid(e.to_string()))?;
                let data =
                    RepairDataWire { catalog: Some(snapshot.catalog.clone()), base_law: None };
                (
                    None,
                    None,
                    Some(ZTransportPremisesWire {
                        snapshot,
                        sid_steps: f.sid_limits().steps,
                        sid_depth: f.sid_limits().depth,
                    }),
                    data,
                )
            }
        };
        let mut outcomes = Vec::with_capacity(report.outcomes.len());
        for outcome in &report.outcomes {
            let delta = match outcome.classification {
                RepairClassification::Invalid | RepairClassification::Unevaluated => None,
                _ => {
                    let members = outcome
                        .candidates
                        .iter()
                        .map(|id| {
                            by_id.get(id).copied().ok_or_else(|| {
                                RepairArtifactError::invalid(format!(
                                    "an outcome names candidate {id}, which was not declared"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let regimes = match family {
                        RepairFamilyRef::Transport(f) => f.hypothetical_regimes(&members),
                        RepairFamilyRef::Backdoor(f) => f.hypothetical_regimes(&members),
                        RepairFamilyRef::ZTransport(f) => f.hypothetical_regimes(&members),
                    }
                    .map_err(|reasons| RepairArtifactError::invalid(reasons.join("; ")))?;
                    Some(regimes_wire(&regimes)?)
                }
            };
            outcomes.push(outcome_wire(outcome, delta));
        }
        let receipt = &report.receipt;
        RepairReportArtifact {
            version: REPAIR_ARTIFACT_VERSION,
            kind: REPAIR_ARTIFACT_KIND.to_owned(),
            premises: RepairPremisesWire {
                family: owner.family_id().to_owned(),
                contract: owner.contract_id(),
                objective: objective_name(report.objective).to_owned(),
                transport,
                backdoor,
                z_transport,
                candidates: declared.iter().map(|c| candidate_wire(c)).collect(),
                limits: RepairLimitsWire {
                    operations: limits.search.operations,
                    depth: limits.search.depth,
                    memory_limit_bytes: limits.memory_limit_bytes,
                },
            },
            data,
            report: RepairReportWire {
                obligations: report.obligations.iter().map(obligation_wire).collect(),
                outcomes,
                ranked_sufficient: report.ranked_sufficient.clone(),
                receipt: RepairReceiptWire {
                    operations_consumed: receipt.operations_consumed,
                    depth_reached: receipt.depth_reached,
                    memory_limit_bytes: receipt.memory_limit_bytes,
                    explored: receipt.explored.clone(),
                    unevaluated: receipt.unevaluated.clone(),
                    unevaluated_total: receipt.unevaluated_total,
                    dominated_skipped: receipt.dominated_skipped,
                    beyond_declared_depth: receipt.beyond_declared_depth,
                    stop: receipt.stop.as_ref().map(stop_wire),
                },
                status: report.status().map(|(c, d)| (c.to_owned(), d.to_owned())),
                outcome: outcome_name(report).to_owned(),
            },
            premises_digest: String::new(),
            data_digest: String::new(),
            report_digest: String::new(),
        }
        .sealed()
    }

    /// Recompute the premises, data and report digests of this artifact (after an
    /// edit, to re-seal it).
    ///
    /// # Errors
    /// Encoding failure.
    pub fn sealed(mut self) -> Result<Self, RepairArtifactError> {
        self.premises_digest = digest(IdentityDomain::Identification, &self.premises)?;
        self.data_digest = digest(IdentityDomain::Identification, &self.data)?;
        self.report_digest = digest(
            IdentityDomain::Claim,
            &(&self.premises_digest, &self.data_digest, &self.report),
        )?;
        Ok(self)
    }

    /// The stored obligations, each re-validated and its id re-derived.
    ///
    /// # Errors
    /// An obligation whose content does not match its id or violates the
    /// obligation contract.
    pub fn obligations(&self) -> Result<Vec<EvidenceObligation>, RepairArtifactError> {
        self.report.obligations.iter().map(obligation_from_wire).collect()
    }

    /// The stored candidates, each with its semantic id re-derived.
    ///
    /// # Errors
    /// A candidate whose content does not match its stored semantic id.
    pub fn candidates(&self) -> Result<Vec<DurableStudyCandidate>, RepairArtifactError> {
        self.premises.candidates.iter().map(candidate_from_wire).collect()
    }

    // ------------------------------------------------------------- container

    /// Serialize through the checksummed, sectioned container.
    ///
    /// # Errors
    /// A blank artifact id, an oversized payload or an encoding failure.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, RepairArtifactError> {
        if artifact_id.trim().is_empty() {
            return Err(RepairArtifactError::invalid("missing artifact id"));
        }
        let body = to_cbor(self).map_err(|e| RepairArtifactError::from_io(&e))?;
        let encoded = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: antecedent_io::migrate::STABLE_FORMAT,
                minimum_reader_version: antecedent_io::migrate::STABLE_FORMAT,
                artifact_kind: ArtifactKind::Other(REPAIR_ARTIFACT_KIND.into()),
                library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
                    .map_err(|e| RepairArtifactError::from_io(&e))?,
                artifact_id: artifact_id.into(),
                sections: vec![section_descriptor(BODY_SECTION, "application/cbor", &body)],
                provenance: ProvenanceWire { note: "repair_search_receipt".into() },
            },
            sections: vec![SectionBytes::new(BODY_SECTION, body)],
        };
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes).map_err(|e| RepairArtifactError::from_io(&e))?;
        if bytes.len() > MAX_REPAIR_ARTIFACT_BYTES {
            return Err(RepairArtifactError::bounds("the repair artifact is too large"));
        }
        Ok(bytes)
    }

    /// Decode a container under the bounded reader. The decoded artifact is not
    /// trusted until [`Self::consume`] replays it.
    ///
    /// # Errors
    /// `repair_artifact.bounds_exceeded` for an oversized container or section,
    /// `repair_artifact.unsupported_version` for another version or kind, and
    /// `repair_artifact.invalid_artifact` for a corrupt container, another
    /// layout or a report beyond the declared collection bounds.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RepairArtifactError> {
        if bytes.len() > MAX_REPAIR_ARTIFACT_BYTES {
            return Err(RepairArtifactError::bounds("the repair artifact is too large"));
        }
        let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes))
            .map_err(|e| RepairArtifactError::from_io(&e))?;
        let manifest = reader.manifest();
        if manifest.artifact_kind != ArtifactKind::Other(REPAIR_ARTIFACT_KIND.into())
            || manifest.sections.len() != 1
            || manifest.sections[0].id != BODY_SECTION
        {
            return Err(RepairArtifactError::unsupported_version(
                "unsupported repair artifact kind or layout",
            ));
        }
        if manifest.sections[0].uncompressed_size > MAX_REPAIR_ARTIFACT_BYTES as u64 {
            return Err(RepairArtifactError::bounds("the repair body section is too large"));
        }
        let section =
            reader.load_section(BODY_SECTION).map_err(|e| RepairArtifactError::from_io(&e))?;
        let artifact: Self =
            from_cbor(section.as_bytes()).map_err(|e| RepairArtifactError::from_io(&e))?;
        artifact.check_shape()?;
        Ok(artifact)
    }

    fn check_shape(&self) -> Result<(), RepairArtifactError> {
        if self.version != REPAIR_ARTIFACT_VERSION || self.kind != REPAIR_ARTIFACT_KIND {
            return Err(RepairArtifactError::unsupported_version(
                "unsupported repair artifact version",
            ));
        }
        let receipt = &self.report.receipt;
        if self.premises.candidates.len() > REPAIR_MAX_CANDIDATES
            || self.report.outcomes.len() > MAX_REPAIR_OUTCOMES
            || self.report.obligations.len() > MAX_REPAIR_OBLIGATIONS
            || receipt.explored.len() > REPAIR_RECEIPT_REGIONS
            || receipt.unevaluated.len() > REPAIR_RECEIPT_REGIONS
            || receipt.stop.as_ref().is_some_and(|s| {
                s.explored.len() > REPAIR_RECEIPT_REGIONS
                    || s.unevaluated.len() > REPAIR_RECEIPT_REGIONS
            })
        {
            return Err(RepairArtifactError::bounds(
                "a stored collection exceeds the repair artifact's declared bounds",
            ));
        }
        Ok(())
    }
}

// -------------------------------------------------------------------- consume

/// A family rebuilt from stored premises and data.
enum Rebuilt {
    Transport(Box<TransportRepairFamily>),
    Backdoor(Box<BackdoorRepairFamily>),
    ZTransport(Box<ZTransportRepairFamily>),
}

impl Rebuilt {
    fn family(&self) -> &dyn RepairFamily {
        match self {
            Self::Transport(family) => &**family,
            Self::Backdoor(family) => &**family,
            Self::ZTransport(family) => &**family,
        }
    }

    fn reference(&self) -> RepairFamilyRef<'_> {
        match self {
            Self::Transport(family) => RepairFamilyRef::Transport(family),
            Self::Backdoor(family) => RepairFamilyRef::Backdoor(family),
            Self::ZTransport(family) => RepairFamilyRef::ZTransport(family),
        }
    }
}

fn repair_error_to_artifact(error: &RepairError, replay: bool) -> RepairArtifactError {
    if error.code == reason_code!("transport_budget_cancel") {
        RepairArtifactError::budget(format!("the replay stopped: {}", error.message))
    } else if replay {
        RepairArtifactError::invalid(format!("the report does not replay: {error}"))
    } else {
        RepairArtifactError::invalid(error.to_string())
    }
}

fn rebuild_z_transport(
    z: &ZTransportPremisesWire,
    data: &RepairDataWire,
    ctx: &ExecutionContext,
) -> Result<Rebuilt, RepairArtifactError> {
    let limits = SidLimits { steps: z.sid_steps, depth: z.sid_depth };
    let snapshot = ZTransportFailureSnapshot::from_wire(&z.snapshot, limits, ctx)
        .map_err(|e| RepairArtifactError::invalid(e.to_string()))?;
    if data.base_law.is_some() || data.catalog.as_ref() != Some(&z.snapshot.catalog) {
        return Err(RepairArtifactError::invalid(
            "z-transport data must equal the frozen snapshot catalog",
        ));
    }
    Ok(Rebuilt::ZTransport(Box::new(
        ZTransportRepairFamily::try_new(snapshot, limits)
            .map_err(|e| repair_error_to_artifact(&e, true))?,
    )))
}

fn rebuild(
    artifact: &RepairReportArtifact,
    ctx: &ExecutionContext,
) -> Result<Rebuilt, RepairArtifactError> {
    let premises = &artifact.premises;
    let rebuilt = match (
        premises.family.as_str(),
        &premises.transport,
        &premises.backdoor,
        &premises.z_transport,
    ) {
        ("transport", Some(t), None, None) => {
            let catalog = artifact
                .data
                .catalog
                .as_ref()
                .ok_or_else(|| {
                    RepairArtifactError::invalid("a transport artifact stores its base catalog")
                })?
                .to_catalog()
                .map_err(|e| RepairArtifactError::invalid(e.to_string()))?;
            let graph = antecedent_io::admg_from_wire(&t.graph)
                .map_err(|e| RepairArtifactError::invalid(e.to_string()))?;
            let diagram = SelectionDiagram::try_new(graph, ids(&t.selection_targets).to_vec())
                .map_err(|e| RepairArtifactError::invalid(e.to_string()))?;
            let query = ClassicalTransportQuery {
                outcomes: ids(&t.outcomes),
                treatments: ids(&t.treatments),
                source: Arc::from(t.source.as_str()),
                target: Arc::from(t.target.as_str()),
            };
            Rebuilt::Transport(Box::new(
                TransportRepairFamily::try_new(
                    diagram,
                    query,
                    catalog,
                    SidLimits { steps: t.sid_steps, depth: t.sid_depth },
                    ctx,
                )
                .map_err(|e| repair_error_to_artifact(&e, true))?,
            ))
        }
        ("backdoor", None, Some(b), None) => {
            let dag = antecedent_io::dag_from_wire(&b.graph)
                .map_err(|e| RepairArtifactError::invalid(e.to_string()))?;
            let base_law = artifact
                .data
                .base_law
                .as_ref()
                .ok_or_else(|| {
                    RepairArtifactError::invalid("a back-door artifact stores its base law")
                })?
                .iter()
                .copied()
                .map(VariableId::from_raw);
            let mut records = Vec::with_capacity(b.assumptions.len());
            for a in &b.assumptions {
                let mut record = ObligationRecord::new(
                    a.id.as_str(),
                    scope_from_parts(&a.scope, a.horizon)?,
                    AssumptionSource::UserDeclared,
                    ObligationKind::CheckNotRun,
                    AssumptionStatus::Declared,
                    a.description.as_str(),
                );
                if let Some(check) = &a.required_check {
                    record = record.with_required_check(check.as_str());
                }
                records.push(record);
            }
            Rebuilt::Backdoor(Box::new(
                BackdoorRepairFamily::try_new(
                    dag,
                    VariableId::from_raw(b.treatment),
                    VariableId::from_raw(b.outcome),
                    b.population.as_str(),
                    base_law,
                    &records,
                )
                .map_err(|e| repair_error_to_artifact(&e, true))?,
            ))
        }
        ("z_transport", None, None, Some(z)) => rebuild_z_transport(z, &artifact.data, ctx)?,
        _ => {
            return Err(RepairArtifactError::invalid(
                "the artifact names an unknown repair family or an inconsistent premises layout",
            ));
        }
    };
    let family = rebuilt.family();
    if family.family_id() != premises.family || family.contract_id() != premises.contract {
        return Err(RepairArtifactError::mismatch(
            "contract",
            "the stored contract identity differs from the contract its premises describe",
        ));
    }
    Ok(rebuilt)
}

fn first_difference<T: PartialEq>(
    section: &str,
    stored: &T,
    replayed: &T,
) -> Result<(), RepairArtifactError> {
    if stored == replayed {
        Ok(())
    } else {
        Err(RepairArtifactError::mismatch(
            section,
            "the replayed repair differs from the stored report",
        ))
    }
}

impl RepairReportArtifact {
    /// Replay the whole repair under the stored limits and accept only an
    /// identical report.
    ///
    /// Stored limits above `limits` (or a stored memory cap above the smaller
    /// of `limits` and the context's hard limit) refuse before any work. The
    /// stored digests are recomputed, then each stored subset's family check is
    /// replayed from its stored hypothetical delta with the family's own theorem
    /// checker and must reproduce the stored classification, and finally the
    /// whole search is replayed and must reproduce every candidate, obligation,
    /// classification, derivation and receipt. A replay that cancellation stops
    /// is `repair_artifact.budget`: the artifact is neither accepted nor refuted.
    ///
    /// # Errors
    /// `repair_artifact.bounds_exceeded` for unaffordable stored limits;
    /// `repair_artifact.budget` for a replay stopped as above;
    /// `repair_artifact.unsupported_version` for another version or kind;
    /// `repair_artifact.invalid_artifact` for a digest that does not match,
    /// undecodable premises or a replay the repair refuses;
    /// `repair_artifact.replay_mismatch` for any stored section the replay does
    /// not reproduce.
    pub fn consume(
        &self,
        limits: RepairConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<RepairReport, RepairArtifactError> {
        self.check_shape()?;
        let stored = self.premises.limits;
        let memory_cap = ctx
            .memory
            .hard_limit_bytes
            .map_or(limits.memory_limit_bytes, |h| h.min(limits.memory_limit_bytes));
        let effective = self.report.receipt.memory_limit_bytes;
        if stored.operations > limits.search.operations
            || stored.depth > limits.search.depth
            || stored.memory_limit_bytes > memory_cap
            || effective > memory_cap
        {
            return Err(RepairArtifactError::bounds(
                "the artifact's stored repair limits exceed this consumer's maxima",
            ));
        }
        if effective > stored.memory_limit_bytes {
            return Err(RepairArtifactError::invalid(
                "the stored effective memory cap exceeds the declared cap",
            ));
        }
        let resealed = self.clone().sealed()?;
        if resealed.premises_digest != self.premises_digest
            || resealed.data_digest != self.data_digest
            || resealed.report_digest != self.report_digest
        {
            return Err(RepairArtifactError::invalid("a repair artifact digest does not match"));
        }
        let rebuilt = rebuild(self, ctx)?;
        let candidates = self.candidates()?;
        let objective = objective_from_name(&self.premises.objective)?;
        let obligations = self.obligations()?;
        let blocked = obligations.iter().any(|o| !o.satisfiable_by_study());
        self.replay_deltas(&rebuilt, blocked, ctx)?;
        let replayed = repair(
            rebuilt.family(),
            &candidates,
            objective,
            RepairLimits {
                search: SearchLimits { operations: stored.operations, depth: stored.depth },
                memory_limit_bytes: effective,
            },
            ctx,
        )
        .map_err(|e| repair_error_to_artifact(&e, true))?;
        if replayed.receipt.stop.as_ref().is_some_and(|r| r.stop == SearchStop::Cancelled) {
            return Err(RepairArtifactError::budget("the replay was cancelled before it finished"));
        }
        let mut again = Self::build(rebuilt.reference(), &candidates, &replayed)?;
        again.premises.limits = stored;
        first_difference("contract", &again.premises.family, &self.premises.family)?;
        first_difference("objective", &again.premises.objective, &self.premises.objective)?;
        first_difference("candidates", &again.premises.candidates, &self.premises.candidates)?;
        first_difference("contract", &again.premises, &self.premises)?;
        first_difference("data", &again.data, &self.data)?;
        first_difference("obligations", &again.report.obligations, &self.report.obligations)?;
        first_difference("classification", &again.report.outcomes, &self.report.outcomes)?;
        first_difference(
            "classification",
            &again.report.ranked_sufficient,
            &self.report.ranked_sufficient,
        )?;
        first_difference("classification", &again.report.status, &self.report.status)?;
        first_difference("classification", &again.report.outcome, &self.report.outcome)?;
        first_difference("receipt", &again.report.receipt, &self.report.receipt)?;
        Ok(replayed)
    }

    /// Replay every stored subset's family check from its stored hypothetical
    /// delta and require the stored classification.
    fn replay_deltas(
        &self,
        rebuilt: &Rebuilt,
        blocked: bool,
        ctx: &ExecutionContext,
    ) -> Result<(), RepairArtifactError> {
        for outcome in &self.report.outcomes {
            let stored = classification_from_name(&outcome.classification)?;
            let Some(delta) = &outcome.delta else {
                if !matches!(
                    stored,
                    RepairClassification::Invalid | RepairClassification::Unevaluated
                ) {
                    return Err(RepairArtifactError::mismatch(
                        "classification",
                        "a decided subset stores no hypothetical delta",
                    ));
                }
                continue;
            };
            if ctx.cancellation.is_cancelled() {
                return Err(RepairArtifactError::budget("the replay was cancelled"));
            }
            let catalog = EvidenceCatalogWire {
                environments: Vec::new(),
                regimes: delta.clone(),
                bindings: Vec::new(),
                target_sampling: None,
            }
            .to_catalog()
            .map_err(|e| RepairArtifactError::invalid(e.to_string()))?;
            let regimes: Vec<EvidenceRegime> = catalog.regimes.to_vec();
            let verdict = match rebuilt {
                Rebuilt::Transport(family) => family.check_regimes(regimes, ctx),
                Rebuilt::Backdoor(family) => family.check_regimes(&regimes, ctx),
                Rebuilt::ZTransport(family) => family.check_regimes(regimes, ctx),
            }
            .map_err(|RepairCancelled| RepairArtifactError::budget("the replay was cancelled"))?;
            if classify_verdict(&verdict, blocked) != stored {
                return Err(RepairArtifactError::mismatch(
                    "classification",
                    format!(
                        "subset [{}] is stored as {} but its stored hypothetical delta replays \
                         differently",
                        outcome.candidates.join(","),
                        outcome.classification
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod scientific_obligation_tests {
    use super::*;

    #[test]
    fn declared_scientific_coordinates_round_trip_and_tampering_is_refused() {
        let variable = VariableId::from_raw(0);
        let obligation = EvidenceObligation::try_new(EvidenceObligationSpec {
            quantities: std::collections::BTreeMap::from([(
                variable,
                antecedent_core::ScientificQuantity {
                    variable_id: "schema:y".into(),
                    variable_name: "Outcome".into(),
                    role: antecedent_core::QuantityRole::Outcome,
                    units: "kg".into(),
                    population_id: "target".into(),
                    regime_id: "observational".into(),
                    horizon: 2,
                    functional_id: "mean".into(),
                    conditioning: vec![],
                    transform_id: "identity".into(),
                },
            )]),
            kind: EvidenceObligationKind::Measure,
            scope: ObligationScope::Horizon { horizon: 2 },
            variables: Arc::from([variable]),
            population: Some(Arc::from("target")),
            regime: ObligationRegime::observational(false),
            reason: Arc::from("measurement owed"),
            required_slots: Arc::from([Arc::from("coordinate:y")]),
            min_additional_samples: None,
            provenance: ObligationProvenance {
                family: Arc::from("support"),
                source: Arc::from("response"),
                proof_step: None,
            },
        })
        .unwrap();
        let bytes = serde_json::to_vec(&obligation_wire(&obligation)).unwrap();
        let mut stored: ObligationWire = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(obligation_from_wire(&stored).unwrap(), obligation);
        stored.quantities[0].1.units = "lb".into();
        assert!(obligation_from_wire(&stored).is_err());
    }
}
