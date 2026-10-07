//! F7 `InverseQueryArtifact` (`inverse_functional_query_v1`).
//!
//! A durable record of one generalized finite inverse decision query and its answer:
//! the declared query (action grid and order, grid scope, constraints, selection
//! rule, tolerance, evaluation budget), the decision contract that declares the
//! actions, the forward evidence behind each feasibility field (finite laws and
//! means embedded compactly), the per-action status table with the *distinct*
//! point, interval-region, identified-set, all-scenario and posterior-probability
//! fields, the selected action, whether the selection is certified and whether
//! the grid was exhaustively decided.
//!
//! Layout of the bounded sectioned container:
//!
//! * `inverse_query.meta` (CBOR): version, feature, claim, contract identity, query,
//!   claim declarations (law metadata, mean coordinates), the evidence structure that
//!   points at those claims, the identity digests and the stored result table;
//! * `inverse_query.contract` (nested `decision_contract_v1` container);
//! * `inverse_query.data` (little-endian `f64`): the draws of every embedded law and
//!   the values of every embedded mean claim, in claim order.
//!
//! Identity keeps the *premises* (contract identity, query, claim declarations and
//! evidence structure) apart from the *data* (the numbers), and binds both in one
//! digest. A consumer trusts none of the stored results: it rebuilds the contract,
//! query and evidence, re-evaluates them through [`evaluate_inverse_query`], and
//! refuses unless the recomputed identity and the recomputed result table equal the
//! stored ones and, when the caller retained an identity of its own, that one too. A
//! resealed change of a target, a tolerance, a draw, a status or the selection is
//! therefore refused.
//!
//! A [`GridScope::ContinuousSample`] artifact never claims global feasibility: a
//! stored `exhaustive_over_declared_set` or `no_feasible_action_in_declared_set` on a
//! continuous sample is refused even before replay.
//!
//! The artifact makes no coverage or calibration claim (`point_only`). The mean of a
//! posterior is never turned into a probability or quantile here: the shared engine
//! refuses that when the query is evaluated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Cursor;

use antecedent_core::ScientificQuantity;
use antecedent_io::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::distribution_artifact::{DistributionArtifact, DistributionMetadata};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::reader::ArtifactReader;
use antecedent_io::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};
use serde::{Deserialize, Serialize};

use crate::decision_artifact::DecisionContractArtifact;
use crate::decision_contract::{DecisionContract, Tail};
use crate::decision_eval::{MeanSource, SourceReceipt};
use crate::decision_structural::{AtomEvidence, StructuralAtom};
use crate::inverse_query::{
    ActionReport, Comparison, ConstraintValue, ExistenceClaim, FeasibilityStatus, ForwardClaim,
    ForwardEvidence, GridScope, IdentifiedSet, IntervalRegion, InverseConstraint, InverseQuery,
    InverseQueryError, InverseResult, MemberStatus, PosteriorFeasibility, SelectionOutcome,
    SelectionRule, evaluate_inverse_query,
};

/// The artifact major version this reader writes and accepts.
pub const INVERSE_QUERY_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const INVERSE_QUERY_ARTIFACT_FEATURE: &str = "inverse_functional_query_v1";
/// The only claim an inverse-query artifact carries.
pub const INVERSE_QUERY_INFERENCE_CLAIM: &str = "point_only";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_INVERSE_QUERY_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Most embedded forward claims (laws and mean grids) of one artifact.
pub const MAX_INVERSE_QUERY_CLAIMS: usize = 256;

const ARTIFACT_KIND: &str = "inverse_functional_query_v1";
const META_SECTION: &str = "inverse_query.meta";
const CONTRACT_SECTION: &str = "inverse_query.contract";
const DATA_SECTION: &str = "inverse_query.data";
const CONTRACT_ARTIFACT_ID: &str = "inverse-query-contract";
const WRONG_CONTRACT: &str = "functional_inverse_query.wrong_contract";
const GLOBAL_CLAIM: &str = "functional_inverse_query.global_feasibility_claim";
const BOUNDS: &str = "functional_inverse_query.bounds_exceeded";

fn refused(detail: &str, text: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("inverse_functional_unsupported"),
        message: format!("{detail}: {text}"),
    }
}

fn wrong_contract(text: &str) -> IoError {
    refused(WRONG_CONTRACT, text)
}

fn wide(count: usize) -> u64 {
    u64::try_from(count).unwrap_or(u64::MAX)
}

// ---------------------------------------------------------------------------
// Wire enums
// ---------------------------------------------------------------------------

/// Side of a target that is feasible.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonWire {
    /// At least the target.
    AtLeast,
    /// At most the target.
    AtMost,
}

impl From<Comparison> for ComparisonWire {
    fn from(value: Comparison) -> Self {
        match value {
            Comparison::AtLeast => Self::AtLeast,
            Comparison::AtMost => Self::AtMost,
        }
    }
}

impl From<ComparisonWire> for Comparison {
    fn from(value: ComparisonWire) -> Self {
        match value {
            ComparisonWire::AtLeast => Self::AtLeast,
            ComparisonWire::AtMost => Self::AtMost,
        }
    }
}

/// Tail of a probability threshold.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TailWire {
    /// `P(U <= t)`.
    Lower,
    /// `P(U >= t)`.
    Upper,
}

impl From<Tail> for TailWire {
    fn from(value: Tail) -> Self {
        match value {
            Tail::Lower => Self::Lower,
            Tail::Upper => Self::Upper,
        }
    }
}

impl From<TailWire> for Tail {
    fn from(value: TailWire) -> Self {
        match value {
            TailWire::Lower => Self::Lower,
            TailWire::Upper => Self::Upper,
        }
    }
}

/// Multiple-action rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionWire {
    /// Earliest feasible action in grid order.
    FirstInGridOrder,
    /// Latest feasible action in grid order.
    LastInGridOrder,
    /// Select only a unique feasible action.
    RequireUnique,
}

impl From<SelectionRule> for SelectionWire {
    fn from(value: SelectionRule) -> Self {
        match value {
            SelectionRule::FirstInGridOrder => Self::FirstInGridOrder,
            SelectionRule::LastInGridOrder => Self::LastInGridOrder,
            SelectionRule::RequireUnique => Self::RequireUnique,
        }
    }
}

impl From<SelectionWire> for SelectionRule {
    fn from(value: SelectionWire) -> Self {
        match value {
            SelectionWire::FirstInGridOrder => Self::FirstInGridOrder,
            SelectionWire::LastInGridOrder => Self::LastInGridOrder,
            SelectionWire::RequireUnique => Self::RequireUnique,
        }
    }
}

/// Grid scope.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GridScopeWire {
    /// The declared finite set is the whole question.
    FiniteEnumeration,
    /// Points sampled from a continuous domain; never a global claim.
    ContinuousSample,
}

impl From<GridScope> for GridScopeWire {
    fn from(value: GridScope) -> Self {
        match value {
            GridScope::FiniteEnumeration => Self::FiniteEnumeration,
            GridScope::ContinuousSample => Self::ContinuousSample,
        }
    }
}

impl From<GridScopeWire> for GridScope {
    fn from(value: GridScopeWire) -> Self {
        match value {
            GridScopeWire::FiniteEnumeration => Self::FiniteEnumeration,
            GridScopeWire::ContinuousSample => Self::ContinuousSample,
        }
    }
}

/// Feasibility classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeasibilityWire {
    /// Every constraint definitely holds.
    Feasible,
    /// A constraint definitely fails.
    Infeasible,
    /// A needed coordinate is masked unsupported.
    Unsupported,
    /// Not evaluated.
    Unevaluated,
    /// Members give conflicting answers.
    StructurallyAmbiguous,
    /// A scenario does not identify the quantity.
    Unidentified,
}

impl From<FeasibilityStatus> for FeasibilityWire {
    fn from(value: FeasibilityStatus) -> Self {
        match value {
            FeasibilityStatus::Feasible => Self::Feasible,
            FeasibilityStatus::Infeasible => Self::Infeasible,
            FeasibilityStatus::Unsupported => Self::Unsupported,
            FeasibilityStatus::Unevaluated => Self::Unevaluated,
            FeasibilityStatus::StructurallyAmbiguous => Self::StructurallyAmbiguous,
            FeasibilityStatus::Unidentified => Self::Unidentified,
        }
    }
}

/// How selection ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionOutcomeWire {
    /// One action was selected.
    Selected,
    /// No action is point-feasible.
    NoFeasibleAction,
    /// More than one action is feasible under `require_unique`.
    MultipleFeasible,
    /// No point claim was supplied.
    NoPointClaim,
}

impl From<SelectionOutcome> for SelectionOutcomeWire {
    fn from(value: SelectionOutcome) -> Self {
        match value {
            SelectionOutcome::Selected => Self::Selected,
            SelectionOutcome::NoFeasibleAction => Self::NoFeasibleAction,
            SelectionOutcome::MultipleFeasible => Self::MultipleFeasible,
            SelectionOutcome::NoPointClaim => Self::NoPointClaim,
        }
    }
}

/// What the point field says about the existence of a feasible action.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExistenceWire {
    /// A feasible witness was found; not a global claim.
    FoundFeasibleAction,
    /// Every action of a finite enumeration is decided and none is feasible.
    NoFeasibleActionInDeclaredSet,
    /// Nothing can be said.
    Undetermined,
}

impl From<ExistenceClaim> for ExistenceWire {
    fn from(value: ExistenceClaim) -> Self {
        match value {
            ExistenceClaim::FoundFeasibleAction => Self::FoundFeasibleAction,
            ExistenceClaim::NoFeasibleActionInDeclaredSet => Self::NoFeasibleActionInDeclaredSet,
            ExistenceClaim::Undetermined => Self::Undetermined,
        }
    }
}

// ---------------------------------------------------------------------------
// Query wire
// ---------------------------------------------------------------------------

/// One typed constraint on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ConstraintWire {
    /// `E[U_a]` against `target`.
    TargetMean {
        /// Target for the mean.
        target: f64,
        /// Feasible side.
        comparison: ComparisonWire,
    },
    /// Left-inverse `p`-quantile of `U_a` against `target`.
    TargetQuantile {
        /// Level in `(0, 1)`.
        p: f64,
        /// Target for the quantile.
        target: f64,
        /// Feasible side.
        comparison: ComparisonWire,
    },
    /// `P(U_a tail outcome_threshold)` against `probability`.
    ProbabilityThreshold {
        /// Outcome threshold.
        outcome_threshold: f64,
        /// Lower or upper tail.
        tail: TailWire,
        /// Probability target in `[0, 1]`.
        probability: f64,
        /// Feasible side of the probability target.
        comparison: ComparisonWire,
    },
}

impl From<&InverseConstraint> for ConstraintWire {
    fn from(value: &InverseConstraint) -> Self {
        match *value {
            InverseConstraint::TargetMean { target, comparison } => {
                Self::TargetMean { target, comparison: comparison.into() }
            }
            InverseConstraint::TargetQuantile { p, target, comparison } => {
                Self::TargetQuantile { p, target, comparison: comparison.into() }
            }
            InverseConstraint::ProbabilityThreshold {
                outcome_threshold,
                tail,
                probability,
                comparison,
            } => Self::ProbabilityThreshold {
                outcome_threshold,
                tail: tail.into(),
                probability,
                comparison: comparison.into(),
            },
        }
    }
}

impl From<ConstraintWire> for InverseConstraint {
    fn from(value: ConstraintWire) -> Self {
        match value {
            ConstraintWire::TargetMean { target, comparison } => {
                Self::TargetMean { target, comparison: comparison.into() }
            }
            ConstraintWire::TargetQuantile { p, target, comparison } => {
                Self::TargetQuantile { p, target, comparison: comparison.into() }
            }
            ConstraintWire::ProbabilityThreshold {
                outcome_threshold,
                tail,
                probability,
                comparison,
            } => Self::ProbabilityThreshold {
                outcome_threshold,
                tail: tail.into(),
                probability,
                comparison: comparison.into(),
            },
        }
    }
}

/// The declared query, without the contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InverseQueryWire {
    /// Ordered action ids: the grid and the tie-rule order.
    pub grid_order: Vec<String>,
    /// Finite enumeration or a continuous sample.
    pub grid_scope: GridScopeWire,
    /// Constraints, all of which must hold.
    pub constraints: Vec<ConstraintWire>,
    /// Multiple-action rule.
    pub selection: SelectionWire,
    /// Non-negative slack on each comparison.
    pub tolerance: f64,
    /// Most functional evaluations; `None` is unbounded.
    pub max_evaluations: Option<u64>,
}

impl InverseQueryWire {
    /// The wire form of a query (the contract is carried separately).
    #[must_use]
    pub fn from_query(query: &InverseQuery) -> Self {
        Self {
            grid_order: query.grid_order.clone(),
            grid_scope: query.grid_scope.into(),
            constraints: query.constraints.iter().map(ConstraintWire::from).collect(),
            selection: query.selection.into(),
            tolerance: query.tolerance,
            max_evaluations: query.max_evaluations.map(wide),
        }
    }

    /// The query this wire declares over `contract`.
    #[must_use]
    pub fn into_query(self, contract: DecisionContract) -> InverseQuery {
        InverseQuery {
            contract,
            grid_order: self.grid_order,
            grid_scope: self.grid_scope.into(),
            constraints: self.constraints.into_iter().map(InverseConstraint::from).collect(),
            selection: self.selection.into(),
            tolerance: self.tolerance,
            max_evaluations: self.max_evaluations.map(|n| usize::try_from(n).unwrap_or(usize::MAX)),
        }
    }
}

// ---------------------------------------------------------------------------
// Result wire
// ---------------------------------------------------------------------------

/// One constraint's value for one action and claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintValueWire {
    /// Index into the query's constraints.
    pub constraint: u64,
    /// The engine's functional value, when answered.
    pub value: Option<f64>,
    /// Standard error; `Some(0.0)` for an exact finite law.
    pub standard_error: Option<f64>,
    /// This constraint's own status.
    pub status: FeasibilityWire,
    /// Why it was not answered, when it was not.
    pub reason: Option<String>,
}

impl From<&ConstraintValue> for ConstraintValueWire {
    fn from(value: &ConstraintValue) -> Self {
        Self {
            constraint: wide(value.constraint),
            value: value.value,
            standard_error: value.standard_error,
            status: value.status.into(),
            reason: value.reason.clone(),
        }
    }
}

/// One scenario or set member's status for one action.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemberStatusWire {
    /// Member identity.
    pub id: String,
    /// Its conjunction status.
    pub status: FeasibilityWire,
    /// Why it was not answered, when it was not.
    pub reason: Option<String>,
}

impl From<&MemberStatus> for MemberStatusWire {
    fn from(value: &MemberStatus) -> Self {
        Self { id: value.id.clone(), status: value.status.into(), reason: value.reason.clone() }
    }
}

/// Probability mass over scenarios for one action; never renormalized.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PosteriorFeasibilityWire {
    /// Mass of scenarios in which the action is feasible.
    pub feasible_mass: f64,
    /// Mass of scenarios in which it is infeasible.
    pub infeasible_mass: f64,
    /// Everything else.
    pub unresolved_mass: f64,
}

impl From<&PosteriorFeasibility> for PosteriorFeasibilityWire {
    fn from(value: &PosteriorFeasibility) -> Self {
        Self {
            feasible_mass: value.feasible_mass,
            infeasible_mass: value.infeasible_mass,
            unresolved_mass: value.unresolved_mass,
        }
    }
}

/// One action across every feasibility field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionReportWire {
    /// Semantic action identity.
    pub id: String,
    /// Position in the declared grid order.
    pub position: u64,
    /// Feasibility on the single forward claim.
    pub point: Option<FeasibilityWire>,
    /// Per-constraint values behind `point`.
    pub point_values: Vec<ConstraintValueWire>,
    /// Feasibility over a published interval region.
    pub interval_region: Option<FeasibilityWire>,
    /// Feasibility over an identified set.
    pub identified_set: Option<FeasibilityWire>,
    /// Per-member statuses behind `identified_set`.
    pub identified_set_members: Vec<MemberStatusWire>,
    /// Feasibility across every declared scenario.
    pub all_scenario: Option<FeasibilityWire>,
    /// Per-scenario statuses behind `all_scenario`.
    pub scenario_members: Vec<MemberStatusWire>,
    /// Posterior mass of feasibility, only with genuine scenario probabilities.
    pub posterior_probability: Option<PosteriorFeasibilityWire>,
}

impl From<&ActionReport> for ActionReportWire {
    fn from(value: &ActionReport) -> Self {
        Self {
            id: value.id.clone(),
            position: wide(value.position),
            point: value.point.map(Into::into),
            point_values: value.point_values.iter().map(Into::into).collect(),
            interval_region: value.interval_region.map(Into::into),
            identified_set: value.identified_set.map(Into::into),
            identified_set_members: value.identified_set_members.iter().map(Into::into).collect(),
            all_scenario: value.all_scenario.map(Into::into),
            scenario_members: value.scenario_members.iter().map(Into::into).collect(),
            posterior_probability: value.posterior_probability.as_ref().map(Into::into),
        }
    }
}

/// Lineage of the point claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceReceiptWire {
    /// Provider object.
    pub provider_id: String,
    /// Data snapshot or exact-law identity.
    pub snapshot_id: String,
    /// RNG identity.
    pub rng_id: String,
    /// Causal-contract identity.
    pub causal_contract_id: String,
}

impl From<&SourceReceipt> for SourceReceiptWire {
    fn from(value: &SourceReceipt) -> Self {
        Self {
            provider_id: value.provider_id.clone(),
            snapshot_id: value.snapshot_id.clone(),
            rng_id: value.rng_id.clone(),
            causal_contract_id: value.causal_contract_id.clone(),
        }
    }
}

/// The stored result table of an inverse query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
// Wire struct mirrors the stored result table, which carries four independent flags.
#[allow(clippy::struct_excessive_bools)]
pub struct InverseResultWire {
    /// Identity of the contract that declared the actions.
    pub contract_identity: String,
    /// The declared grid order.
    pub grid_order: Vec<String>,
    /// The declared grid scope.
    pub grid_scope: GridScopeWire,
    /// The constraints evaluated.
    pub constraints: Vec<ConstraintWire>,
    /// Per-action reports in grid order.
    pub actions: Vec<ActionReportWire>,
    /// Point-feasible action ids in grid order.
    pub feasible_actions: Vec<String>,
    /// The selected action.
    pub selected: Option<String>,
    /// How selection ended.
    pub selection: SelectionOutcomeWire,
    /// Whether every action the rule passed over is definitely infeasible.
    pub selection_certified: bool,
    /// Whether every grid action has a decided point status.
    pub grid_fully_decided: bool,
    /// Existence claim from the point field.
    pub existence: ExistenceWire,
    /// `true` only for a finite enumeration whose every action is decided.
    pub exhaustive_over_declared_set: bool,
    /// Functional evaluations charged.
    pub evaluations_used: u64,
    /// Whether the evaluation limit stopped the search.
    pub budget_exhausted: bool,
    /// Lineage of the point claim.
    pub point_source: Option<SourceReceiptWire>,
    /// What this result is and is not.
    pub scope_note: String,
}

impl From<&InverseResult> for InverseResultWire {
    fn from(value: &InverseResult) -> Self {
        Self {
            contract_identity: value.contract_identity.clone(),
            grid_order: value.grid_order.clone(),
            grid_scope: value.grid_scope.into(),
            constraints: value.constraints.iter().map(ConstraintWire::from).collect(),
            actions: value.actions.iter().map(Into::into).collect(),
            feasible_actions: value.feasible_actions.clone(),
            selected: value.selected.clone(),
            selection: value.selection.into(),
            selection_certified: value.selection_certified,
            grid_fully_decided: value.grid_fully_decided,
            existence: value.existence.into(),
            exhaustive_over_declared_set: value.exhaustive_over_declared_set,
            evaluations_used: wide(value.evaluations_used),
            budget_exhausted: value.budget_exhausted,
            point_source: value.point_source.as_ref().map(Into::into),
            scope_note: value.scope_note.to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// Evidence wire
// ---------------------------------------------------------------------------

/// Declaration of an embedded finite law; its draws are in the data section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LawWire {
    /// Full distribution metadata (identity, shape, weights, support mask).
    pub metadata: DistributionMetadata,
}

/// Declaration of an embedded mean grid; its means are in the data section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeansWire {
    /// One coordinate per mean, in order.
    pub coordinates: Vec<ScientificQuantityWire>,
    /// Provider object that supplied the means.
    pub provider_id: String,
    /// Data snapshot or exact-request identity.
    pub snapshot_id: String,
    /// Causal-contract identity.
    pub causal_contract_id: String,
    /// RNG identity.
    pub rng_id: String,
}

/// One embedded forward claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimWire {
    /// A finite aligned or marginal law.
    Law(Box<LawWire>),
    /// A mean grid, which answers an affine target mean and nothing else.
    Means(MeansWire),
}

/// What is known about one scenario or set member.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtomStateWire {
    /// A law, by index into the claims.
    Evaluated {
        /// Index of the embedded law.
        claim: u64,
    },
    /// The quantity is not identified under this member.
    Unidentified,
    /// The member was not evaluated.
    Unevaluated {
        /// The supplied reason.
        reason: String,
    },
}

/// A scenario or set member.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtomWire {
    /// Stable identity.
    pub id: String,
    /// Genuine probability, when supplied.
    pub probability: Option<f64>,
    /// What is known about it.
    pub state: AtomStateWire,
}

/// A published interval region.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntervalWire {
    /// Index of the lower-endpoint claim.
    pub lower: u64,
    /// Index of the upper-endpoint claim.
    pub upper: u64,
    /// Declared: the endpoint laws bound the functional over the region.
    pub endpoints_bound_functional: bool,
}

/// An identified set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetWire {
    /// Member laws.
    pub members: Vec<AtomWire>,
    /// Declared: the members enumerate the whole identified set.
    pub exhaustive: bool,
}

/// The evidence structure; every number lives behind a claim index.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceWire {
    /// Index of the single point claim.
    pub point: Option<u64>,
    /// Published interval region.
    pub interval_region: Option<IntervalWire>,
    /// Identified set.
    pub identified_set: Option<SetWire>,
    /// Declared structural scenarios.
    pub scenarios: Option<Vec<AtomWire>>,
}

#[derive(Default)]
struct Encoded {
    claims: Vec<ClaimWire>,
    data: Vec<u8>,
    evidence: EvidenceWire,
}

impl Encoded {
    fn push_f64s(&mut self, values: &[f64]) {
        for value in values {
            self.data.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn law(&mut self, artifact: &DistributionArtifact) -> u64 {
        self.push_f64s(artifact.draws());
        self.claims
            .push(ClaimWire::Law(Box::new(LawWire { metadata: artifact.metadata().clone() })));
        wide(self.claims.len() - 1)
    }

    fn claim(&mut self, claim: &ForwardClaim) -> u64 {
        match claim {
            ForwardClaim::Law(artifact) => self.law(artifact),
            ForwardClaim::Means(means) => {
                self.push_f64s(&means.means);
                self.claims.push(ClaimWire::Means(MeansWire {
                    coordinates: means
                        .coordinates
                        .iter()
                        .map(ScientificQuantityWire::from)
                        .collect(),
                    provider_id: means.provider_id.clone(),
                    snapshot_id: means.snapshot_id.clone(),
                    causal_contract_id: means.causal_contract_id.clone(),
                    rng_id: means.rng_id.clone(),
                }));
                wide(self.claims.len() - 1)
            }
        }
    }

    fn atom(&mut self, atom: &StructuralAtom) -> AtomWire {
        let state = match &atom.evidence {
            AtomEvidence::Evaluated(artifact) => {
                AtomStateWire::Evaluated { claim: self.law(artifact) }
            }
            AtomEvidence::Unidentified => AtomStateWire::Unidentified,
            AtomEvidence::Unevaluated(reason) => {
                AtomStateWire::Unevaluated { reason: reason.clone() }
            }
        };
        AtomWire { id: atom.id.clone(), probability: atom.probability, state }
    }

    fn atoms(&mut self, atoms: &[StructuralAtom]) -> Vec<AtomWire> {
        atoms.iter().map(|atom| self.atom(atom)).collect()
    }
}

fn encode_evidence(evidence: &ForwardEvidence) -> Encoded {
    let mut encoded = Encoded::default();
    let point = evidence.point.as_ref().map(|claim| encoded.claim(claim));
    let interval_region = evidence.interval_region.as_ref().map(|region| {
        let lower = encoded.claim(&region.lower);
        let upper = encoded.claim(&region.upper);
        IntervalWire { lower, upper, endpoints_bound_functional: region.endpoints_bound_functional }
    });
    let identified_set = evidence
        .identified_set
        .as_ref()
        .map(|set| SetWire { members: encoded.atoms(&set.members), exhaustive: set.exhaustive });
    let scenarios = evidence.scenarios.as_ref().map(|atoms| encoded.atoms(atoms));
    encoded.evidence = EvidenceWire { point, interval_region, identified_set, scenarios };
    encoded
}

struct DataReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl DataReader<'_> {
    fn take(&mut self, count: usize) -> Result<Vec<f64>, IoError> {
        let end = count
            .checked_mul(8)
            .and_then(|size| self.at.checked_add(size))
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| wrong_contract("data section is shorter than the declared claims"))?;
        let values = self.bytes[self.at..end]
            .chunks_exact(8)
            .map(|chunk| {
                let mut word = [0_u8; 8];
                word.copy_from_slice(chunk);
                f64::from_le_bytes(word)
            })
            .collect();
        self.at = end;
        Ok(values)
    }

    fn exhausted(&self) -> bool {
        self.at == self.bytes.len()
    }
}

fn decode_claims(claims: &[ClaimWire], data: &[u8]) -> Result<Vec<ForwardClaim>, IoError> {
    let mut reader = DataReader { bytes: data, at: 0 };
    let mut decoded = Vec::with_capacity(claims.len());
    for claim in claims {
        decoded.push(match claim {
            ClaimWire::Law(law) => {
                let [rows, columns] = law.metadata.shape;
                let count = rows.checked_mul(columns).ok_or(IoError::TooLarge)?;
                let draws = reader.take(count)?;
                ForwardClaim::Law(Box::new(DistributionArtifact::new(law.metadata.clone(), draws)?))
            }
            ClaimWire::Means(means) => {
                let values = reader.take(means.coordinates.len())?;
                let coordinates = means
                    .coordinates
                    .iter()
                    .map(|wire| {
                        ScientificQuantity::try_from(wire.clone())
                            .map_err(|error| IoError::Convert(error.into()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                ForwardClaim::Means(MeanSource {
                    coordinates,
                    means: values,
                    provider_id: means.provider_id.clone(),
                    snapshot_id: means.snapshot_id.clone(),
                    causal_contract_id: means.causal_contract_id.clone(),
                    rng_id: means.rng_id.clone(),
                })
            }
        });
    }
    if !reader.exhausted() {
        return Err(wrong_contract("data section carries numbers no claim declares"));
    }
    Ok(decoded)
}

fn claim_at(claims: &[ForwardClaim], index: u64) -> Result<ForwardClaim, IoError> {
    usize::try_from(index)
        .ok()
        .and_then(|k| claims.get(k))
        .cloned()
        .ok_or_else(|| wrong_contract("evidence points at a claim that is not embedded"))
}

fn atom_from(claims: &[ForwardClaim], wire: &AtomWire) -> Result<StructuralAtom, IoError> {
    let evidence = match &wire.state {
        AtomStateWire::Evaluated { claim } => match claim_at(claims, *claim)? {
            ForwardClaim::Law(artifact) => AtomEvidence::Evaluated(artifact),
            ForwardClaim::Means(_) => {
                return Err(wrong_contract("a scenario or set member needs a law, not means"));
            }
        },
        AtomStateWire::Unidentified => AtomEvidence::Unidentified,
        AtomStateWire::Unevaluated { reason } => AtomEvidence::Unevaluated(reason.clone()),
    };
    Ok(StructuralAtom { id: wire.id.clone(), probability: wire.probability, evidence })
}

fn atoms_from(claims: &[ForwardClaim], wires: &[AtomWire]) -> Result<Vec<StructuralAtom>, IoError> {
    wires.iter().map(|wire| atom_from(claims, wire)).collect()
}

fn evidence_from(claims: &[ForwardClaim], wire: &EvidenceWire) -> Result<ForwardEvidence, IoError> {
    Ok(ForwardEvidence {
        point: wire.point.map(|index| claim_at(claims, index)).transpose()?,
        interval_region: wire
            .interval_region
            .as_ref()
            .map(|region| {
                Ok::<_, IoError>(IntervalRegion {
                    lower: claim_at(claims, region.lower)?,
                    upper: claim_at(claims, region.upper)?,
                    endpoints_bound_functional: region.endpoints_bound_functional,
                })
            })
            .transpose()?,
        identified_set: wire
            .identified_set
            .as_ref()
            .map(|set| {
                Ok::<_, IoError>(IdentifiedSet {
                    members: atoms_from(claims, &set.members)?,
                    exhaustive: set.exhaustive,
                })
            })
            .transpose()?,
        scenarios: wire.scenarios.as_ref().map(|atoms| atoms_from(claims, atoms)).transpose()?,
    })
}

// ---------------------------------------------------------------------------
// Identity and container
// ---------------------------------------------------------------------------

/// Identity digests; premises and numbers are kept apart.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InverseQueryIdentity {
    /// BLAKE3 of the canonical declarations (contract identity, query, claim
    /// declarations, evidence structure).
    pub premises_digest: String,
    /// BLAKE3 of the numerical section.
    pub data_digest: String,
    /// BLAKE3 of both digests.
    pub digest: String,
}

#[derive(Serialize)]
struct Premises<'a> {
    version: u16,
    feature: &'a str,
    inference_claim: &'a str,
    contract_identity: &'a str,
    query: &'a InverseQueryWire,
    claims: &'a [ClaimWire],
    evidence: &'a EvidenceWire,
}

fn tagged_digest(tag: &str, bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(tag.as_bytes());
    hasher.update(&wide(bytes.len()).to_le_bytes());
    hasher.update(bytes);
    hasher.finalize().to_hex().to_string()
}

fn identity_of(
    contract_identity: &str,
    query: &InverseQueryWire,
    encoded: &Encoded,
) -> Result<InverseQueryIdentity, IoError> {
    let premises = to_cbor(&Premises {
        version: INVERSE_QUERY_ARTIFACT_VERSION,
        feature: INVERSE_QUERY_ARTIFACT_FEATURE,
        inference_claim: INVERSE_QUERY_INFERENCE_CLAIM,
        contract_identity,
        query,
        claims: &encoded.claims,
        evidence: &encoded.evidence,
    })?;
    let premises_digest = tagged_digest("antecedent.inverse_query.premises.v1", &premises);
    let data_digest = tagged_digest("antecedent.inverse_query.data.v1", &encoded.data);
    let mut whole = blake3::Hasher::new();
    whole.update(b"antecedent.inverse_query.identity.v1");
    whole.update(premises_digest.as_bytes());
    whole.update(data_digest.as_bytes());
    Ok(InverseQueryIdentity {
        premises_digest,
        data_digest,
        digest: whole.finalize().to_hex().to_string(),
    })
}

fn identity_diff(
    left: &InverseQueryIdentity,
    right: &InverseQueryIdentity,
) -> Option<&'static str> {
    if left.premises_digest != right.premises_digest {
        Some("premises digest")
    } else if left.data_digest != right.data_digest {
        Some("data digest")
    } else if left.digest != right.digest {
        Some("identity digest")
    } else {
        None
    }
}

/// Stored metadata; the numbers are a separate `f64` section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InverseQueryMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Always [`INVERSE_QUERY_INFERENCE_CLAIM`].
    pub inference_claim: String,
    /// Identity of the decision contract (the contract itself is its own section).
    pub contract_identity: String,
    /// The declared query.
    pub query: InverseQueryWire,
    /// Embedded claim declarations in data order.
    pub claims: Vec<ClaimWire>,
    /// Evidence structure over those claims.
    pub evidence: EvidenceWire,
    /// Identity digests.
    pub identity: InverseQueryIdentity,
    /// The stored per-action status table and selection.
    pub result: InverseResultWire,
}

/// Encode the metadata, the nested contract artifact and the numbers as a
/// checksummed container.
///
/// Hidden: the producer path is [`InverseQueryArtifact::to_bytes`]; tests use this to
/// build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &InverseQueryMeta,
    contract: &[u8],
    data: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    if artifact_id.trim().is_empty() {
        return Err(IoError::Convert("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta)?;
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: antecedent_io::migrate::STABLE_FORMAT,
            minimum_reader_version: antecedent_io::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))?,
            artifact_id: artifact_id.into(),
            sections: vec![
                section_descriptor(META_SECTION, "application/cbor", &meta_bytes),
                section_descriptor(CONTRACT_SECTION, "application/octet-stream", contract),
                section_descriptor(DATA_SECTION, "application/octet-stream", data),
            ],
            provenance: ProvenanceWire { note: "inverse_functional_query_point_only".into() },
        },
        sections: vec![
            SectionBytes::new(META_SECTION, meta_bytes),
            SectionBytes::new(CONTRACT_SECTION, contract.to_vec()),
            SectionBytes::new(DATA_SECTION, data.to_vec()),
        ],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes)?;
    if bytes.len() > MAX_INVERSE_QUERY_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    Ok(bytes)
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u16,
}

/// Decode the three sections of a container, refusing another version before the
/// metadata is interpreted. No semantic validation happens here.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[doc(hidden)]
pub fn decode_parts(bytes: &[u8]) -> Result<(InverseQueryMeta, Vec<u8>, Vec<u8>), IoError> {
    if bytes.len() > MAX_INVERSE_QUERY_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
        || manifest.sections.len() != 3
        || manifest.sections[0].id != META_SECTION
        || manifest.sections[1].id != CONTRACT_SECTION
        || manifest.sections[2].id != DATA_SECTION
    {
        return Err(IoError::Convert("unsupported inverse query artifact layout".into()));
    }
    let declared = manifest
        .sections
        .iter()
        .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size));
    if declared.is_none_or(|total| total > wide(MAX_INVERSE_QUERY_ARTIFACT_BYTES)) {
        return Err(IoError::TooLarge);
    }
    let meta_section = reader.load_section(META_SECTION)?;
    let peek: VersionPeek = from_cbor(meta_section.as_bytes())?;
    if peek.version != INVERSE_QUERY_ARTIFACT_VERSION {
        return Err(IoError::UnsupportedVersion { version: u32::from(peek.version) });
    }
    let meta: InverseQueryMeta = from_cbor(meta_section.as_bytes())?;
    let contract = reader.load_section(CONTRACT_SECTION)?.as_bytes().to_vec();
    let data = reader.load_section(DATA_SECTION)?.as_bytes().to_vec();
    Ok((meta, contract, data))
}

fn query_refusal(error: &InverseQueryError) -> IoError {
    let refusal = error.to_refusal();
    IoError::Refused {
        code: refusal.code,
        message: format!("{}: {}", refusal.detail, refusal.offending.unwrap_or_default()),
    }
}

/// A validated inverse query, its forward evidence, its result table and identity.
#[derive(Clone, Debug)]
pub struct InverseQueryArtifact {
    query: InverseQuery,
    evidence: ForwardEvidence,
    result: InverseResult,
    identity: InverseQueryIdentity,
}

impl InverseQueryArtifact {
    /// Evaluate `query` on `evidence` through [`evaluate_inverse_query`] and seal the
    /// query, the evidence, the result and their identity.
    ///
    /// # Errors
    /// Every refusal of [`evaluate_inverse_query`] (a joint-law requirement the claim
    /// cannot meet, a mean-only claim asked for a quantile or probability, an invalid
    /// grid or constraint), and more embedded claims than
    /// [`MAX_INVERSE_QUERY_CLAIMS`].
    pub fn new(query: InverseQuery, evidence: ForwardEvidence) -> Result<Self, InverseQueryError> {
        let result = evaluate_inverse_query(&query, &evidence)?;
        let encoded = encode_evidence(&evidence);
        if encoded.claims.len() > MAX_INVERSE_QUERY_CLAIMS {
            return Err(InverseQueryError::InvalidParameter("claims"));
        }
        let identity =
            identity_of(&result.contract_identity, &InverseQueryWire::from_query(&query), &encoded)
                .map_err(|_| InverseQueryError::InvalidParameter("artifact_encoding"))?;
        Ok(Self { query, evidence, result, identity })
    }

    /// The query.
    #[must_use]
    pub fn query(&self) -> &InverseQuery {
        &self.query
    }

    /// The forward evidence.
    #[must_use]
    pub fn evidence(&self) -> &ForwardEvidence {
        &self.evidence
    }

    /// The per-action status table, selection and flags.
    #[must_use]
    pub fn result(&self) -> &InverseResult {
        &self.result
    }

    /// The stored form of [`Self::result`].
    #[must_use]
    pub fn result_wire(&self) -> InverseResultWire {
        InverseResultWire::from(&self.result)
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &InverseQueryIdentity {
        &self.identity
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        let encoded = encode_evidence(&self.evidence);
        let contract = DecisionContractArtifact::new(self.query.contract.clone())?
            .to_bytes(CONTRACT_ARTIFACT_ID)?;
        let meta = InverseQueryMeta {
            version: INVERSE_QUERY_ARTIFACT_VERSION,
            feature: INVERSE_QUERY_ARTIFACT_FEATURE.into(),
            inference_claim: INVERSE_QUERY_INFERENCE_CLAIM.into(),
            contract_identity: self.result.contract_identity.clone(),
            query: InverseQueryWire::from_query(&self.query),
            claims: encoded.claims,
            evidence: encoded.evidence,
            identity: self.identity.clone(),
            result: InverseResultWire::from(&self.result),
        };
        encode_parts(&meta, &contract, &encoded.data, artifact_id)
    }

    /// Consume an artifact by re-evaluation.
    ///
    /// The contract, query and evidence are rebuilt, evaluated again through
    /// [`evaluate_inverse_query`], and the recomputed identity and result table must
    /// equal the stored ones. When the caller retained an `expected` identity of its
    /// own, that identity must match as well, so a resealed and self-consistent
    /// semantic change is refused. A continuous sample that stores a global
    /// feasibility claim is refused before replay.
    ///
    /// # Errors
    /// Corruption, another major version (`UnsupportedVersion`), a changed identity or
    /// result, a global claim on a continuous sample
    /// (`functional_inverse_query.global_feasibility_claim`) and any evaluation
    /// refusal, all with code `inverse_functional_unsupported` unless the engine's own
    /// refusal code applies.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&InverseQueryIdentity>,
    ) -> Result<Self, IoError> {
        let (meta, contract_bytes, data) = decode_parts(bytes)?;
        if meta.feature != INVERSE_QUERY_ARTIFACT_FEATURE
            || meta.inference_claim != INVERSE_QUERY_INFERENCE_CLAIM
        {
            return Err(wrong_contract("feature marker or inference claim is not this format's"));
        }
        if meta.claims.len() > MAX_INVERSE_QUERY_CLAIMS {
            return Err(refused(BOUNDS, "too many embedded claims"));
        }
        let contract =
            DecisionContractArtifact::from_bytes(&contract_bytes, &meta.contract_identity)?
                .contract()
                .clone();
        let claims = decode_claims(&meta.claims, &data)?;
        let evidence = evidence_from(&claims, &meta.evidence)?;
        let query = meta.query.clone().into_query(contract);
        if query.grid_scope == GridScope::ContinuousSample
            && (meta.result.exhaustive_over_declared_set
                || meta.result.existence == ExistenceWire::NoFeasibleActionInDeclaredSet)
        {
            return Err(refused(
                GLOBAL_CLAIM,
                "a sample of a continuous domain cannot claim global feasibility or infeasibility",
            ));
        }
        let artifact = Self::new(query, evidence).map_err(|error| query_refusal(&error))?;
        if let Some(field) = identity_diff(&meta.identity, &artifact.identity) {
            return Err(wrong_contract(&format!("stored {field} differs from the recomputed one")));
        }
        if meta.result != artifact.result_wire() {
            return Err(wrong_contract("stored result table does not replay from the evidence"));
        }
        if let Some(retained) = expected {
            if let Some(field) = identity_diff(retained, &artifact.identity) {
                return Err(wrong_contract(&format!(
                    "{field} differs from the consumer's retained identity"
                )));
            }
        }
        Ok(artifact)
    }
}

/// The JSON form of a result table, for bridges that hand it to another language.
///
/// # Errors
/// A serialization failure.
pub fn result_to_json(result: &InverseResult) -> Result<String, IoError> {
    serde_json::to_string(&InverseResultWire::from(result))
        .map_err(|error| IoError::Convert(error.to_string()))
}
