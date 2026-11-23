//! Wire declarations for durable repair search receipts.

use antecedent_io_core::query_wire::ValueWire;
use antecedent_io_core::transport_catalog_wire::{EvidenceCatalogWire, EvidenceRegimeWire};
use antecedent_io_core::wire::{AdmgWire, DagWire};
use serde::{Deserialize, Serialize};

use crate::ZTransportFailureSnapshotWire;
use crate::identification_repair::SensitivityMethod;

// ------------------------------------------------------------------- wires

/// One stored evidence obligation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObligationWire {
    /// Declared full scientific coordinates, keyed by structural variable ID.
    /// Legacy structural obligations omit this field rather than inventing units.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quantities: Vec<(u32, antecedent_io_core::quantity_wire::ScientificQuantityWire)>,
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

/// A sensitivity finding of a sensitivity contract.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensitivityFindingWire {
    /// `e_value`, `omitted_variable_bias` or `tipping`.
    pub method: SensitivityMethodWire,
    /// Confounder strength at which the conclusion is overturned.
    pub strength: f64,
    /// Whether the conclusion survives the plausible confounding.
    pub robust: bool,
}

/// Name of the analysis behind a sensitivity finding, spelled on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensitivityMethodWire {
    /// E-value.
    EValue,
    /// Omitted-variable bias.
    OmittedVariableBias,
    /// Tipping search.
    Tipping,
}

impl From<SensitivityMethod> for SensitivityMethodWire {
    fn from(method: SensitivityMethod) -> Self {
        match method {
            SensitivityMethod::EValue => Self::EValue,
            SensitivityMethod::OmittedVariableBias => Self::OmittedVariableBias,
            SensitivityMethod::Tipping => Self::Tipping,
        }
    }
}

impl From<SensitivityMethodWire> for SensitivityMethod {
    fn from(method: SensitivityMethodWire) -> Self {
        match method {
            SensitivityMethodWire::EValue => Self::EValue,
            SensitivityMethodWire::OmittedVariableBias => Self::OmittedVariableBias,
            SensitivityMethodWire::Tipping => Self::Tipping,
        }
    }
}

/// The structure of a front-door, instrument, hedge or sensitivity contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdentificationPremisesWire {
    /// Directed graph (front-door, instrument and sensitivity contracts).
    pub graph: Option<DagWire>,
    /// Graph with bidirected edges (hedge contracts).
    pub admg: Option<AdmgWire>,
    /// Treatment.
    pub treatment: u32,
    /// Outcome.
    pub outcome: u32,
    /// Population of the contract.
    pub population: String,
    /// The finding of a sensitivity contract.
    pub finding: Option<SensitivityFindingWire>,
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
    /// `transport`, `backdoor`, `z_transport`, `frontdoor`, `instrument`, `hedge` or
    /// `sensitivity`.
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
    /// Front-door, instrument, hedge or sensitivity structure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identification: Option<IdentificationPremisesWire>,
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
    /// Variables observed jointly with treatment and outcome (back-door, front-door,
    /// instrument, hedge and sensitivity families).
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
