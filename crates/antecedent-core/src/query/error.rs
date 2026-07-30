//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ids::VariableId;

/// Errors from query construction or validation.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum QueryError {
    /// Treatment and outcome are the same variable.
    #[error("treatment and outcome are the same variable {id}")]
    TreatmentEqualsOutcome {
        /// Shared id.
        id: VariableId,
    },
    /// Intervention does not target the declared treatment.
    #[error("intervention targets {got}, expected treatment {expected}")]
    InterventionVariableMismatch {
        /// Expected treatment id.
        expected: VariableId,
        /// Actual intervention target.
        got: VariableId,
    },
    /// Intervention sequence has no unique target variable.
    #[error("intervention does not have a unique target variable")]
    AmbiguousInterventionTarget,
    /// Effect modifier overlaps treatment or outcome.
    #[error("effect modifier overlaps treatment or outcome")]
    ModifierOverlapsTreatmentOrOutcome,
    /// Sustained window has `until < from`.
    #[error("invalid temporal window [{from}, {until}]")]
    InvalidTemporalWindow {
        /// Window start.
        from: i32,
        /// Window end.
        until: i32,
    },
    /// Horizon must be at least one time step.
    #[error("horizon_steps must be >= 1")]
    NonPositiveHorizon,
    /// Nested intervention failed validation.
    #[error("invalid intervention: {0}")]
    InvalidIntervention(String),
    /// Counterfactual query has no outcomes.
    #[error("counterfactual query requires at least one outcome")]
    EmptyCounterfactualOutcomes,
    /// Anomaly query has no targets.
    #[error("anomaly attribution requires targets")]
    EmptyAnomalyTargets,
    /// Anomaly `max_units` must be ≥ 1.
    #[error("anomaly max_units must be >= 1")]
    NonPositiveAnomalyLimit,
    /// Mediation query has no mediators.
    #[error("mediation query requires mediators")]
    EmptyMediators,
    /// Mediator overlaps treatment or outcome.
    #[error("mediator overlaps treatment or outcome")]
    MediatorOverlapsTreatmentOrOutcome,
    /// Conditional effect requires non-empty modifiers.
    #[error("conditional effect requires non-empty effect modifiers")]
    EmptyEffectModifiers,
    /// Population selector has no rows.
    #[error("population selector has no rows")]
    EmptyPopulationRows,
    /// Named [`crate::query::PredicateExpr`] has an empty registry key.
    #[error("predicate name must be non-empty")]
    EmptyPredicateName,
    /// [`crate::intervention::TemporalPolicy::Dynamic`] has no single treatment origin.
    #[error("TemporalPolicy::Dynamic has no single treatment offset")]
    DynamicPolicyHasNoTreatmentOffset,
    /// Time-range population has `end <= start`.
    #[error("invalid population time range [{start}, {end})")]
    InvalidPopulationTimeRange {
        /// Start.
        start: usize,
        /// End.
        end: usize,
    },
    /// Sequential allocation order is empty.
    #[error("sequential allocation order is empty")]
    EmptyAllocationOrder,
    /// Sequential allocation order contains the same component more than once.
    #[error("sequential allocation order contains duplicate components")]
    DuplicateAllocationComponent,
    /// Shapley exact component limit must be ≥ 1.
    #[error("Shapley max_exact_components must be >= 1")]
    NonPositiveShapleyLimit,
    /// Approximate Shapley sample / permutation count must be ≥ 1.
    #[error("Shapley sample / permutation count must be >= 1")]
    NonPositiveShapleySamples,
    /// Change attribution `max_components` must be ≥ 1.
    #[error("max_components / max_targets must be >= 1")]
    NonPositiveComponentLimit,
    /// Mechanism-change query has no targets.
    #[error("mechanism-change detection requires targets")]
    EmptyMechanismChangeTargets,
    /// Significance level must be in (0, 1).
    #[error("significance level must be in (0, 1)")]
    InvalidSignificanceLevel,
    /// Interventional distribution query has no outcomes.
    #[error("interventional distribution requires at least one outcome")]
    EmptyDistributionOutcomes,
    /// Path enumeration `max_paths` / `max_len` must be ≥ 1.
    #[error("path max_paths / max_len must be >= 1")]
    NonPositivePathLimit,
    /// Path node overlaps treatment or outcome.
    #[error("path node overlaps treatment or outcome")]
    PathNodeOverlapsTreatmentOrOutcome,
    /// Distribution conditioning overlaps an outcome or intervention target.
    #[error("distribution conditioning overlaps outcome or intervention")]
    ConditioningOverlapsOutcomeOrIntervention,
    /// Named / custom population requires a [`super::PopulationRegistry`].
    #[error("named predicate / custom distribution requires a PopulationRegistry")]
    PopulationRegistryRequired,
    /// Named predicate is not bound in the registry.
    #[error("unknown predicate name `{name}`")]
    UnknownPredicateName {
        /// Predicate key.
        name: std::sync::Arc<str>,
    },
    /// Custom distribution handle is not bound in the registry.
    #[error("unknown DistributionRef({id})")]
    UnknownDistributionRef {
        /// Raw distribution id.
        id: u32,
    },
    /// Treated / untreated population requires a treatment column.
    #[error("Treated/Untreated population requires a treatment column")]
    PopulationNeedsTreatment,
    /// Treatment column is not binary 0/1.
    #[error("Treated/Untreated population requires binary 0/1 treatment")]
    PopulationNonBinaryTreatment,
    /// Keep-mask / weight length mismatch.
    #[error("population length mismatch: expected {expected}, got {actual}")]
    PopulationLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// Predicate row index ≥ `n`.
    #[error("population row {row} out of range for n={n}")]
    PopulationRowOutOfRange {
        /// Offending row.
        row: usize,
        /// Population size.
        n: usize,
    },
    /// Environment-restricted populations need multi-env data (not resolved here).
    #[error("Environment target population is not resolved by PopulationRegistry")]
    PopulationEnvironmentUnsupported,
    /// Distribution weights contain negatives or non-finite values.
    #[error("custom distribution weights must be finite and non-negative")]
    InvalidPopulationWeights,
}
