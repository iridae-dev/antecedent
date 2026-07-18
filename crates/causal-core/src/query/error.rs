//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ids::VariableId;

#[derive(Clone, Debug, Eq, PartialEq)]
/// Errors from query construction or validation.
pub enum QueryError {
    /// Treatment and outcome are the same variable.
    TreatmentEqualsOutcome {
        /// Shared id.
        id: VariableId,
    },
    /// Intervention does not target the declared treatment.
    InterventionVariableMismatch {
        /// Expected treatment id.
        expected: VariableId,
        /// Actual intervention target.
        got: VariableId,
    },
    /// Intervention sequence has no unique target variable.
    AmbiguousInterventionTarget,
    /// Effect modifier overlaps treatment or outcome.
    ModifierOverlapsTreatmentOrOutcome,
    /// Sustained window has `until < from`.
    InvalidTemporalWindow {
        /// Window start.
        from: i32,
        /// Window end.
        until: i32,
    },
    /// Horizon must be at least one time step.
    NonPositiveHorizon,
    /// Nested intervention failed validation.
    InvalidIntervention(String),
    /// Counterfactual query has no outcomes.
    EmptyCounterfactualOutcomes,
    /// Anomaly query has no targets.
    EmptyAnomalyTargets,
    /// Anomaly `max_units` must be ≥ 1.
    NonPositiveAnomalyLimit,
    /// Mediation query has no mediators.
    EmptyMediators,
    /// Mediator overlaps treatment or outcome.
    MediatorOverlapsTreatmentOrOutcome,
    /// Conditional effect requires non-empty modifiers.
    EmptyEffectModifiers,
    /// Population selector has no rows.
    EmptyPopulationRows,
    /// Named [`crate::query::PredicateExpr`] has an empty registry key.
    EmptyPredicateName,
    /// [`crate::intervention::TemporalPolicy::Dynamic`] has no single treatment origin.
    DynamicPolicyHasNoTreatmentOffset,
    /// Time-range population has `end <= start`.
    InvalidPopulationTimeRange {
        /// Start.
        start: usize,
        /// End.
        end: usize,
    },
    /// Sequential allocation order is empty.
    EmptyAllocationOrder,
    /// Shapley exact component limit must be ≥ 1.
    NonPositiveShapleyLimit,
    /// Approximate Shapley sample / permutation count must be ≥ 1.
    NonPositiveShapleySamples,
    /// Change attribution `max_components` must be ≥ 1.
    NonPositiveComponentLimit,
    /// Mechanism-change query has no targets.
    EmptyMechanismChangeTargets,
    /// Significance level must be in (0, 1).
    InvalidSignificanceLevel,
    /// Interventional distribution query has no outcomes.
    EmptyDistributionOutcomes,
    /// Path enumeration `max_paths` / `max_len` must be ≥ 1.
    NonPositivePathLimit,
    /// Path node overlaps treatment or outcome.
    PathNodeOverlapsTreatmentOrOutcome,
}

impl core::fmt::Display for QueryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TreatmentEqualsOutcome { id } => {
                write!(f, "treatment and outcome are the same variable {id}")
            }
            Self::InterventionVariableMismatch { expected, got } => {
                write!(f, "intervention targets {got}, expected treatment {expected}")
            }
            Self::AmbiguousInterventionTarget => {
                write!(f, "intervention does not have a unique target variable")
            }
            Self::ModifierOverlapsTreatmentOrOutcome => {
                write!(f, "effect modifier overlaps treatment or outcome")
            }
            Self::InvalidTemporalWindow { from, until } => {
                write!(f, "invalid temporal window [{from}, {until}]")
            }
            Self::NonPositiveHorizon => write!(f, "horizon_steps must be >= 1"),
            Self::InvalidIntervention(msg) => write!(f, "invalid intervention: {msg}"),
            Self::EmptyCounterfactualOutcomes => {
                write!(f, "counterfactual query requires at least one outcome")
            }
            Self::EmptyAnomalyTargets => write!(f, "anomaly attribution requires targets"),
            Self::NonPositiveAnomalyLimit => write!(f, "anomaly max_units must be >= 1"),
            Self::EmptyMediators => write!(f, "mediation query requires mediators"),
            Self::MediatorOverlapsTreatmentOrOutcome => {
                write!(f, "mediator overlaps treatment or outcome")
            }
            Self::EmptyEffectModifiers => {
                write!(f, "conditional effect requires non-empty effect modifiers")
            }
            Self::EmptyPopulationRows => write!(f, "population selector has no rows"),
            Self::EmptyPredicateName => write!(f, "predicate name must be non-empty"),
            Self::DynamicPolicyHasNoTreatmentOffset => {
                write!(f, "TemporalPolicy::Dynamic has no single treatment offset")
            }
            Self::InvalidPopulationTimeRange { start, end } => {
                write!(f, "invalid population time range [{start}, {end})")
            }
            Self::EmptyAllocationOrder => write!(f, "sequential allocation order is empty"),
            Self::NonPositiveShapleyLimit => {
                write!(f, "Shapley max_exact_components must be >= 1")
            }
            Self::NonPositiveShapleySamples => {
                write!(f, "Shapley sample / permutation count must be >= 1")
            }
            Self::NonPositiveComponentLimit => {
                write!(f, "max_components / max_targets must be >= 1")
            }
            Self::EmptyMechanismChangeTargets => {
                write!(f, "mechanism-change detection requires targets")
            }
            Self::InvalidSignificanceLevel => {
                write!(f, "significance level must be in (0, 1)")
            }
            Self::EmptyDistributionOutcomes => {
                write!(f, "interventional distribution requires at least one outcome")
            }
            Self::NonPositivePathLimit => {
                write!(f, "path max_paths / max_len must be >= 1")
            }
            Self::PathNodeOverlapsTreatmentOrOutcome => {
                write!(f, "path node overlaps treatment or outcome")
            }
        }
    }
}

impl std::error::Error for QueryError {}

