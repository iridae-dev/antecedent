//! Versioned inverse-query, result and evidence wire forms.

use super::{
    ActionReport, Comparison, ConstraintValue, DecisionContract, DistributionMetadata,
    ExistenceClaim, FeasibilityStatus, GridScope, InverseConstraint, InverseQuery, InverseResult,
    MemberStatus, PosteriorFeasibility, ScientificQuantityWire, SelectionOutcome, SelectionRule,
    SourceReceipt, Tail, wide,
};
use serde::{Deserialize, Serialize};

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
#[allow(
    clippy::struct_excessive_bools,
    reason = "each flag is an independent switch of the record; they do not form a state machine"
)]
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
    /// Posterior mean or frequentist point estimate. Absent on an older grid,
    /// which did not name one and is read as a frequentist point estimate.
    #[serde(default = "frequentist_estimate")]
    pub estimate: crate::decision_eval::MeanEstimate,
}

fn frequentist_estimate() -> crate::decision_eval::MeanEstimate {
    crate::decision_eval::MeanEstimate::Frequentist
}

/// One embedded forward claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimWire {
    /// A finite aligned or marginal law.
    Law(Box<LawWire>),
    /// Analytic conditional Gaussian outcome law.
    #[serde(rename = "analytic_gaussian_v1")]
    Gaussian(Box<crate::gaussian_forward_law::GaussianForwardLaw>),
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
