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
    /// Distribution conditioning overlaps an outcome or intervention target.
    ConditioningOverlapsOutcomeOrIntervention,
    /// Named / custom population requires a [`super::PopulationRegistry`].
    PopulationRegistryRequired,
    /// Named predicate is not bound in the registry.
    UnknownPredicateName {
        /// Predicate key.
        name: std::sync::Arc<str>,
    },
    /// Custom distribution handle is not bound in the registry.
    UnknownDistributionRef {
        /// Raw distribution id.
        id: u32,
    },
    /// Treated / untreated population requires a treatment column.
    PopulationNeedsTreatment,
    /// Treatment column is not binary 0/1.
    PopulationNonBinaryTreatment,
    /// Keep-mask / weight length mismatch.
    PopulationLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// Predicate row index ≥ `n`.
    PopulationRowOutOfRange {
        /// Offending row.
        row: usize,
        /// Population size.
        n: usize,
    },
    /// Environment-restricted populations need multi-env data (not resolved here).
    PopulationEnvironmentUnsupported,
    /// Distribution weights contain negatives or non-finite values.
    InvalidPopulationWeights,
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
            Self::ConditioningOverlapsOutcomeOrIntervention => {
                write!(f, "distribution conditioning overlaps outcome or intervention")
            }
            Self::PopulationRegistryRequired => {
                write!(f, "named predicate / custom distribution requires a PopulationRegistry")
            }
            Self::UnknownPredicateName { name } => {
                write!(f, "unknown predicate name `{name}`")
            }
            Self::UnknownDistributionRef { id } => {
                write!(f, "unknown DistributionRef({id})")
            }
            Self::PopulationNeedsTreatment => {
                write!(f, "Treated/Untreated population requires a treatment column")
            }
            Self::PopulationNonBinaryTreatment => {
                write!(f, "Treated/Untreated population requires binary 0/1 treatment")
            }
            Self::PopulationLengthMismatch { expected, actual } => {
                write!(f, "population length mismatch: expected {expected}, got {actual}")
            }
            Self::PopulationRowOutOfRange { row, n } => {
                write!(f, "population row {row} out of range for n={n}")
            }
            Self::PopulationEnvironmentUnsupported => {
                write!(f, "Environment target population is not resolved by PopulationRegistry")
            }
            Self::InvalidPopulationWeights => {
                write!(f, "custom distribution weights must be finite and non-negative")
            }
        }
    }
}

impl std::error::Error for QueryError {}

