//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ids::VariableId;

use super::fault::{
    AssociationFault, AttributeFault, DiscoverFault, HistoryMatchFault, InterferenceFault,
    InterventionFault, IvBoundsFault, LocalPolynomialRatioFault, LongitudinalRegimeFault,
    PanelDesignFault, PolicyValueFault, ResponseFault, SequentialContrastFault, SimulateFault,
    SurvivalFault, TemporalDesignFault, TransportFault,
};

/// Why a randomized-effect declaration is refused. Each variant is one violated
/// contract; the display text is the stable human-readable statement of it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RandomizedEffectFault {
    /// Refused: multi-arm contrasts do not combine with receipt adjustment, CUPED, ANCOVA, or exact Fisher inference.
    #[error(
        "multi-arm contrasts do not combine with receipt adjustment, CUPED, ANCOVA, or exact Fisher inference"
    )]
    MultiArmCombination,
    /// Refused: multi-arm labels, assignment, probabilities, and distinct unit identities must align.
    #[error("multi-arm labels, assignment, probabilities, and distinct unit identities must align")]
    MultiArmAlignment,
    /// Refused: multi-arm observed actions and known probability rows must agree.
    #[error("multi-arm observed actions and known probability rows must agree")]
    MultiArmProbabilityRows,
    /// Refused: multi-arm positivity failure: every declared arm needs observed support.
    #[error("multi-arm positivity failure: every declared arm needs observed support")]
    MultiArmPositivity,
    /// Refused: randomization inference requires an unadjusted complete two-arm design.
    #[error("randomization inference requires an unadjusted complete two-arm design")]
    ExactTestDesign,
    /// Refused: CACE/LATE requires Bernoulli assignment, row-aligned treatment receipt, and no CUPED adjustment.
    #[error(
        "CACE/LATE requires Bernoulli assignment, row-aligned treatment receipt, and no CUPED adjustment"
    )]
    ReceiptDesign,
    /// Refused: treatment-on-treated requires observed one-sided noncompliance: no control-assigned unit received treatment.
    #[error(
        "treatment-on-treated requires observed one-sided noncompliance: no control-assigned unit received treatment"
    )]
    TreatmentOnTreatedNoncompliance,
    /// Refused: treatment receipt is only valid for a CACE/LATE or treatment-on-treated query.
    #[error("treatment receipt is only valid for a CACE/LATE or treatment-on-treated query")]
    ReceiptWithoutComplier,
    /// Refused: fixed CUPED requires Bernoulli assignment, a distinct pre-assignment covariate, and a finite externally fixed coefficient.
    #[error(
        "fixed CUPED requires Bernoulli assignment, a distinct pre-assignment covariate, and a finite externally fixed coefficient"
    )]
    FixedCupedDesign,
    /// Refused: ANCOVA requires distinct pre-assignment covariates, independent unit-level Bernoulli assignment with a common probability, and residual degrees of freedom.
    #[error(
        "ANCOVA requires distinct pre-assignment covariates, independent unit-level Bernoulli assignment with a common probability, and residual degrees of freedom"
    )]
    AncovaDesign,
    /// Refused: factorial contrasts do not combine with receipt adjustment or CUPED on this route.
    #[error("factorial contrasts do not combine with receipt adjustment or CUPED on this route")]
    FactorialCombination,
    /// Refused: assignment, probabilities, and unit identities must have at least two aligned rows.
    #[error("assignment, probabilities, and unit identities must have at least two aligned rows")]
    RowAlignment,
    /// Refused: assignment inclusion probabilities must be finite and strictly between zero and one.
    #[error("assignment inclusion probabilities must be finite and strictly between zero and one")]
    AssignmentProbabilityRange,
    /// Refused: assignment and outcome unit labels must be non-empty.
    #[error("assignment and outcome unit labels must be non-empty")]
    EmptyUnitLabel,
    /// Refused: retained randomized ITT requires one outcome unit per row.
    #[error("retained randomized ITT requires one outcome unit per row")]
    DuplicateOutcomeUnit,
    /// Refused: individual randomization requires one assignment unit per row.
    #[error("individual randomization requires one assignment unit per row")]
    DuplicateAssignmentUnit,
    /// Refused: treatment arm labels must be distinct and non-empty.
    #[error("treatment arm labels must be distinct and non-empty")]
    ArmLabels,
    /// Refused: both assigned arms must be observed.
    #[error("both assigned arms must be observed")]
    UnobservedArm,
    /// Refused: complete-randomization variance requires at least two units in each arm.
    #[error("complete-randomization variance requires at least two units in each arm")]
    CompleteArmSize,
    /// Refused: complete assignment must match the declared treated count.
    #[error("complete assignment must match the declared treated count")]
    CompleteTreatedCount,
    /// Refused: complete assignment probabilities must equal treated_units / n.
    #[error("complete assignment probabilities must equal treated_units / n")]
    CompleteProbability,
    /// Refused: assignment and probability must be constant within each randomized cluster.
    #[error("assignment and probability must be constant within each randomized cluster")]
    ClusterConstancy,
    /// Refused: cluster-randomization variance requires at least two clusters in each arm.
    #[error("cluster-randomization variance requires at least two clusters in each arm")]
    ClusterArmSize,
    /// Refused: cluster assignment must match declared treated count and inclusion probability.
    #[error("cluster assignment must match declared treated count and inclusion probability")]
    ClusterAllocation,
    /// Refused: stratified block labels and allocations must align with rows.
    #[error("stratified block labels and allocations must align with rows")]
    StratifiedAlignment,
    /// Refused: each block must have a consistent allocation and inclusion probability.
    #[error("each block must have a consistent allocation and inclusion probability")]
    StratifiedConsistency,
    /// Refused: stratified variance requires at least two units per arm in every block.
    #[error("stratified variance requires at least two units per arm in every block")]
    StratifiedArmSize,
    /// Refused: stratified assignment must match declared block counts and inclusion probabilities.
    #[error("stratified assignment must match declared block counts and inclusion probabilities")]
    StratifiedAllocation,
    /// Refused: factorial design requires four cells with at least two units each, aligned second-factor assignment, and distinct labels.
    #[error(
        "factorial design requires four cells with at least two units each, aligned second-factor assignment, and distinct labels"
    )]
    FactorialCells,
    /// Refused: factorial observed cell counts and primary-factor inclusion probabilities must match the fixed four-cell allocation.
    #[error(
        "factorial observed cell counts and primary-factor inclusion probabilities must match the fixed four-cell allocation"
    )]
    FactorialAllocation,
    /// Refused: switchback period labels must be non-empty and aligned with rows.
    #[error("switchback period labels must be non-empty and aligned with rows")]
    SwitchbackLabels,
    /// Refused: switchback periods must be unique within each sequence.
    #[error("switchback periods must be unique within each sequence")]
    SwitchbackDuplicatePeriod,
    /// Refused: switchback variance requires at least two independent sequences.
    #[error("switchback variance requires at least two independent sequences")]
    SwitchbackSequences,
}

/// Errors from query construction or validation.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum QueryError {
    /// Invalid continuous-response configuration.
    #[error("invalid response query: {0}")]
    InvalidResponse(ResponseFault),
    /// Invalid transportability configuration.
    #[error("invalid transport query: {0}")]
    InvalidTransport(TransportFault),
    /// Invalid randomization/interference configuration.
    #[error("invalid interference query: {0}")]
    InvalidInterference(InterferenceFault),
    /// Invalid randomized experiment contract.
    #[error("invalid randomized-effect query: {0}")]
    InvalidRandomizedEffect(PanelDesignFault),
    /// A randomized-effect declaration violates one named contract.
    #[error("invalid randomized-effect query: {0}")]
    RandomizedEffectRefused(RandomizedEffectFault),
    /// Invalid held-out policy value contract.
    #[error("invalid policy-value query: {0}")]
    InvalidPolicyValue(PolicyValueFault),
    /// Invalid survival or competing-risk observation contract.
    #[error("invalid survival query: {0}")]
    InvalidSurvival(SurvivalFault),
    /// Invalid longitudinal regime value or subject-level fold contract.
    #[error("invalid longitudinal-regime query: {0}")]
    InvalidLongitudinalRegime(LongitudinalRegimeFault),
    /// Invalid within-period history match.
    #[error("invalid history-match query: {0}")]
    InvalidHistoryMatch(HistoryMatchFault),
    /// Invalid instrumental-variable bounds declaration.
    #[error("invalid iv-bounds query: {0}")]
    InvalidIvBounds(IvBoundsFault),
    /// Invalid noncausal association declaration.
    #[error("invalid association query: {0}")]
    InvalidAssociation(AssociationFault),
    /// Invalid observational sequential regime contrast.
    #[error("invalid sequential-contrast query: {0}")]
    InvalidSequentialContrast(SequentialContrastFault),
    /// Invalid fixed-cutoff local-polynomial ratio design.
    #[error("invalid local-polynomial ratio query: {0}")]
    InvalidLocalPolynomialRatio(LocalPolynomialRatioFault),
    /// Invalid structure-discovery query.
    #[error("invalid discover query: {0}")]
    InvalidDiscover(DiscoverFault),
    /// Invalid attribution query on a fitted structural model.
    #[error("invalid attribute query: {0}")]
    InvalidAttribute(AttributeFault),
    /// Invalid simulation query on a fitted structural model.
    #[error("invalid simulate query: {0}")]
    InvalidSimulate(SimulateFault),
    /// Invalid sequentially randomized unit-period design.
    #[error("invalid temporal-design query: {0}")]
    InvalidTemporalDesign(TemporalDesignFault),
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
    InvalidIntervention(InterventionFault),
    /// Counterfactual query has no outcomes.
    #[error("counterfactual query requires at least one outcome")]
    EmptyCounterfactualOutcomes,
    /// Anomaly query has no targets.
    #[error("anomaly attribution requires targets")]
    EmptyAnomalyTargets,
    /// Anomaly `max_units` must be ≥ 1.
    #[error("anomaly max_units must be >= 1")]
    NonPositiveAnomalyLimit,
    /// Anomaly fixed reference scale must be finite and strictly positive.
    #[error("anomaly fixed reference requires a finite center and a finite positive scale")]
    InvalidAnomalyReference,
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
    /// A distribution outcome is also an intervention target: `P(A | do(A))`
    /// is a point mass fixed by the query, not an identification problem.
    #[error("distribution outcome {id} is also an intervention target")]
    OutcomeIsInterventionTarget {
        /// Shared id.
        id: VariableId,
    },
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
    /// A cutoff population is a limit, not a set of rows.
    #[error(
        "LocalAtCutoff is the limit population at a running-variable cutoff; it has no row \
         subset and is estimated only by a regression-discontinuity design"
    )]
    PopulationLocalAtCutoffNotRows,
    /// Cutoff of a [`super::TargetPopulation::LocalAtCutoff`] is NaN or infinite.
    #[error("LocalAtCutoff cutoff must be finite")]
    NonFiniteCutoff,
    /// Distribution weights contain negatives or non-finite values.
    #[error("custom distribution weights must be finite and non-negative")]
    InvalidPopulationWeights,
    /// Exceedance threshold is non-finite or the grid is empty / not strictly increasing.
    #[error("outcome functional thresholds must be finite and strictly increasing")]
    InvalidOutcomeFunctional,
}
