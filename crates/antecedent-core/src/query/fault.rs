//! Typed refusals for query declarations.
//!
//! Each query family reports a violated contract as one variant of its fault enum, carried by
//! the matching [`QueryError`](super::QueryError) variant. The display text of a variant is the
//! stable statement of the contract, so callers and tests match on the kind, not on text.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ids::VariableId;
use crate::intervention::InterventionError;

use super::response::{TEMPORAL_OBSERVATION_UNLICENSED, TemporalResponseSpec};

/// Why an attribution declaration on a fitted structural model is refused,
/// carried by [`QueryError::InvalidAttribute`](super::QueryError::InvalidAttribute).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum AttributeFault {
    /// Refused: intrinsic influence needs at least two outer draws, inner draws and replicate batches
    #[error(
        "intrinsic influence needs at least two outer draws, inner draws and replicate batches"
    )]
    IntrinsicDraws,
    /// Refused: feature relevance needs a finite shift, at least one sample and a positive feature limit
    #[error(
        "feature relevance needs a finite shift, at least one sample and a positive feature limit"
    )]
    FeatureRelevanceSampling,
    /// Refused: an anomaly attribution declares no target
    #[error("an anomaly attribution declares no target")]
    AnomalyWithoutTarget,
    /// Refused: a mechanism-change detection declares no target
    #[error("a mechanism-change detection declares no target")]
    MechanismChangeWithoutTarget,
    /// Refused: the interval level must be in (0, 1)
    #[error("the interval level must be in (0, 1)")]
    IntervalLevelRange,
    /// Refused: the source limit must be at least 1
    #[error("the source limit must be at least 1")]
    SourceLimit,
    /// Refused: permutation sampling needs at least two permutations
    #[error("permutation sampling needs at least two permutations")]
    PermutationCount,
    /// Refused: a change attribution needs at least one sample
    #[error("a change attribution needs at least one sample")]
    ChangeSampleCount,
    /// Refused: the robust row cap must be at least 1
    #[error("the robust row cap must be at least 1")]
    RobustRowCap,
    /// Refused: a path decomposition needs at least one source
    #[error("a path decomposition needs at least one source")]
    PathSourcesEmpty,
    /// Refused: feature relevance needs at least one feature, and the outcome is not one
    #[error("feature relevance needs at least one feature, and the outcome is not one")]
    FeatureRelevanceFeatures,
    /// Refused: a comparison structure has no self-loop
    #[error("a comparison structure has no self-loop")]
    ComparisonSelfLoop,
    /// Refused: a structure-change attribution needs at least one sample
    #[error("a structure-change attribution needs at least one sample")]
    StructureChangeSampleCount,
}

/// Why a held-out policy-value or fixed-dose declaration is refused,
/// carried by [`QueryError::InvalidPolicyValue`](super::QueryError::InvalidPolicyValue).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PolicyValueFault {
    /// Refused: fixed dose rules require observed baseline groups
    #[error("fixed dose rules require observed baseline groups")]
    FixedDoseNoGroups,
    /// Refused: fixed dose rules must name every observed baseline group exactly once with finite targets
    #[error(
        "fixed dose rules must name every observed baseline group exactly once with finite targets"
    )]
    FixedDoseGroupCoverage,
    /// Refused: continuous-dose response requires distinct columns, aligned non-empty baseline groups, finite targets, positive bandwidth, local support of at least two, and known or `externally_estimated` density provenance
    #[error(
        "continuous-dose response requires distinct columns, aligned non-empty baseline groups, finite targets, positive bandwidth, local support of at least two, and known or externally_estimated density provenance"
    )]
    ContinuousDoseDeclaration,
    /// Refused: multi-action policy rows, labels, probabilities, and constraints must align
    #[error("multi-action policy rows, labels, probabilities, and constraints must align")]
    MultiActionAlignment,
    /// Refused: multi-action probabilities must be positive and sum to one; costs and budgets must be finite and non-negative
    #[error(
        "multi-action probabilities must be positive and sum to one; costs and budgets must be finite and non-negative"
    )]
    MultiActionProbabilities,
    /// Refused: policy selects an unavailable action
    #[error("policy selects an unavailable action")]
    UnavailableAction,
    /// Refused: policy exceeds its action capacity or budget
    #[error("policy exceeds its action capacity or budget")]
    ActionCapacityExceeded,
    /// Refused: multi-action policy must carry only multi-action inputs and unique evaluation subject IDs
    #[error(
        "multi-action policy must carry only multi-action inputs and unique evaluation subject IDs"
    )]
    MultiActionInputs,
    /// Refused: policy, outcome predictions, ownership IDs, and assignment must align on at least two rows
    #[error(
        "policy, outcome predictions, ownership IDs, and assignment must align on at least two rows"
    )]
    BinaryAlignment,
    /// Refused: randomized propensities must be finite and strictly between zero and one
    #[error("randomized propensities must be finite and strictly between zero and one")]
    PropensityRange,
    /// Refused: predictions, costs, and unique non-empty evaluation subject IDs must be valid
    #[error("predictions, costs, and unique non-empty evaluation subject IDs must be valid")]
    PredictionInputs,
    /// Refused: IPW has no nuisance prediction ownership metadata
    #[error("IPW has no nuisance prediction ownership metadata")]
    IpwOwnershipMetadata,
    /// Refused: uplift bins and training subjects require a positive bin count
    #[error("uplift bins and training subjects require a positive bin count")]
    UpliftBinCount,
    /// Refused: uplift bins must cover evaluation rows and each bin; declared training subjects must be unique and disjoint
    #[error(
        "uplift bins must cover evaluation rows and each bin; declared training subjects must be unique and disjoint"
    )]
    UpliftBinCoverage,
    /// Refused: prediction ownership requires disjoint training IDs or matching excluded-fold metadata
    #[error(
        "prediction ownership requires disjoint training IDs or matching excluded-fold metadata"
    )]
    PredictionOwnership,
    /// Refused: finite-class regret requires 2–16 fixed IPW candidates, selected-policy equality, and disjoint construction subjects
    #[error(
        "finite-class regret requires 2–16 fixed IPW candidates, selected-policy equality, and disjoint construction subjects"
    )]
    FiniteClassRegret,
}

/// Why a nested or cross-world intervention declaration is refused,
/// carried by [`QueryError::InvalidIntervention`](super::QueryError::InvalidIntervention).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum InterventionFault {
    /// Refused: a counterfactual event level must be finite
    #[error("a counterfactual event level must be finite")]
    EventLevelNotFinite,
    /// Refused: a counterfactual event query names at least one world, each addressable by a world index
    #[error(
        "a counterfactual event query names at least one world, each addressable by a world index"
    )]
    EventWorlds,
    /// Refused: a counterfactual event query needs at least one event
    #[error("a counterfactual event query needs at least one event")]
    EventsEmpty,
    /// Refused: a counterfactual event names an unknown world
    #[error("a counterfactual event names an unknown world")]
    EventUnknownWorld,
    /// Refused: treatment and outcome must be distinct
    #[error("treatment and outcome must be distinct")]
    TreatmentOutcomeNotDistinct,
    /// Refused: the counterfactual and the observed treatment levels must be finite and distinct
    #[error("the counterfactual and the observed treatment levels must be finite and distinct")]
    EffectOnTreatedLevels,
    /// Refused: cross-world intervention values must be finite
    #[error("cross-world intervention values must be finite")]
    CrossWorldValueNotFinite,
    /// Refused: a variable is intervened twice in one world
    #[error("a variable is intervened twice in one world")]
    VariableIntervenedTwice,
    /// Refused: an edge route is a self-edge
    #[error("an edge route is a self-edge")]
    SelfEdgeRoute,
    /// Refused: an edge is routed twice
    #[error("an edge is routed twice")]
    EdgeRoutedTwice,
    /// Refused: a cross-world query names at least two worlds, each addressable by a world index
    #[error("a cross-world query names at least two worlds, each addressable by a world index")]
    CrossWorldWorlds,
    /// Refused: an edge route reads an unknown world or its own world
    #[error("an edge route reads an unknown world or its own world")]
    RouteUnknownWorld,
    /// Refused: an observation names an unknown world
    #[error("an observation names an unknown world")]
    ObservationUnknownWorld,
    /// Refused: a cross-world contrast needs two distinct observations
    #[error("a cross-world contrast needs two distinct observations")]
    ContrastObservationsNotDistinct,
    /// Refused: treatment levels must be finite and distinct
    #[error("treatment levels must be finite and distinct")]
    PathSpecificLevels,
    /// Refused: nested treatment levels must be finite and distinct
    #[error("nested treatment levels must be finite and distinct")]
    NestedLevels,
    /// Refused: nested treatment, mediator, and outcome must be distinct
    #[error("nested treatment, mediator, and outcome must be distinct")]
    NestedRolesNotDistinct,
    /// Refused: a temporal conditional modifier cannot be the treatment or the outcome
    #[error("a temporal conditional modifier cannot be the treatment or the outcome")]
    ModifierIsTreatmentOrOutcome,
    /// Refused: an intervened edge is not an edge of the graph.
    #[error("intervened edge {parent} -> {child} is not a graph edge")]
    IntervenedEdgeNotInGraph {
        /// parent.
        parent: VariableId,
        /// child.
        child: VariableId,
    },
    /// Refused: a nested intervention failed its own validation.
    #[error("{0}")]
    Rejected(InterventionError),
}

/// Why a difference-in-differences or synthetic-control panel declaration is refused,
/// carried by [`QueryError::InvalidRandomizedEffect`](super::QueryError::InvalidRandomizedEffect).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PanelDesignFault {
    /// Refused: `DiD` requires at least four aligned rows and metadata vectors
    #[error("DiD requires at least four aligned rows and metadata vectors")]
    DidRows,
    /// Refused: subject and cluster labels must be non-empty
    #[error("subject and cluster labels must be non-empty")]
    BlankLabels,
    /// Refused: augmented panel `DiD` requires nuisance columns
    #[error("augmented panel DiD requires nuisance columns")]
    AugmentedNeedsNuisance,
    /// Refused: augmented panel `DiD` requires distinct columns and one unique subject per row
    #[error("augmented panel DiD requires distinct columns and one unique subject per row")]
    AugmentedColumns,
    /// Refused: nuisance columns require augmented panel `DiD`
    #[error("nuisance columns require augmented panel DiD")]
    NuisanceNeedsAugmented,
    /// Refused: staggered `DiD` requires a selected cohort and period
    #[error("staggered DiD requires a selected cohort and period")]
    StaggeredTarget,
    /// Refused: staggered `DiD` requires aligned positive periods, nonnegative cohorts, and a post-adoption target with an observed pre-period
    #[error(
        "staggered DiD requires aligned positive periods, nonnegative cohorts, and a post-adoption target with an observed pre-period"
    )]
    StaggeredPeriods,
    /// Refused: a heterogeneity-robust specification requires the staggered heterogeneity-robust or covariate-adjusted design
    #[error(
        "a heterogeneity-robust specification requires the staggered heterogeneity-robust or covariate-adjusted design"
    )]
    RobustSpecificationDesign,
    /// Refused: the heterogeneity-robust designs require their specification
    #[error("the heterogeneity-robust designs require their specification")]
    RobustDesignSpecification,
    /// Refused: a rollout requires aligned positive periods, non-negative cohorts with at least one treated cohort, and no selected target or supplied nuisance
    #[error(
        "a rollout requires aligned positive periods, non-negative cohorts with at least one treated cohort, and no selected target or supplied nuisance"
    )]
    RolloutPeriods,
    /// Refused: rollout covariates must be distinct baseline columns other than the outcome
    #[error("rollout covariates must be distinct baseline columns other than the outcome")]
    RolloutCovariates,
    /// Refused: a covariate-adjusted two-period `DiD` needs at least one covariate, the 2x2 estimator and no anticipation
    #[error(
        "a covariate-adjusted two-period DiD needs at least one covariate, the 2x2 estimator and no anticipation"
    )]
    CovariateAdjustedTwoPeriod,
    /// Refused: a whole-procedure resample needs at least two replicates and a rollout or augmented panel design
    #[error(
        "a whole-procedure resample needs at least two replicates and a rollout or augmented panel design"
    )]
    WholeProcedureResample,
    /// Refused: synthetic control requires aligned nonempty units, positive periods, and a treated unit with a positive intervention period
    #[error(
        "synthetic control requires aligned nonempty units, positive periods, and a treated unit with a positive intervention period"
    )]
    SyntheticControlPanel,
    /// Refused: augmented synthetic control requires a finite positive ridge penalty
    #[error("augmented synthetic control requires a finite positive ridge penalty")]
    AugmentedRidge,
    /// Refused: augmented synthetic control cannot be combined with synthetic `DiD`
    #[error("augmented synthetic control cannot be combined with synthetic DiD")]
    AugmentedWithDid,
    /// Refused: a finite sharp-null effect requires declared uniform single-unit randomization
    #[error("a finite sharp-null effect requires declared uniform single-unit randomization")]
    SharpNullRandomization,
    /// Refused: synthetic adoptions must be distinct named units with positive periods and list the treated unit at its intervention period
    #[error(
        "synthetic adoptions must be distinct named units with positive periods and list the treated unit at its intervention period"
    )]
    SyntheticAdoptions,
    /// Refused: synthetic intervals cannot be combined with a unit-randomization test or a ridge augmentation
    #[error(
        "synthetic intervals cannot be combined with a unit-randomization test or a ridge augmentation"
    )]
    IntervalWithRandomizationTest,
    /// Refused: conformal synthetic intervals take one treated unit and the simplex synthetic control; use placebo variance for several units or synthetic `DiD`
    #[error(
        "conformal synthetic intervals take one treated unit and the simplex synthetic control; use placebo variance for several units or synthetic DiD"
    )]
    ConformalIntervalDesign,
    /// Refused: placebo-variance synthetic intervals need at least two replicates
    #[error("placebo-variance synthetic intervals need at least two replicates")]
    PlaceboReplicates,
}

/// Why a within-period history-match declaration is refused,
/// carried by [`QueryError::InvalidHistoryMatch`](super::QueryError::InvalidHistoryMatch).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum HistoryMatchFault {
    /// Refused: declare at least one history column and distinct outcome, treatment, history, period and weight columns
    #[error(
        "declare at least one history column and distinct outcome, treatment, history, period and weight columns"
    )]
    HistoryColumns,
    /// Refused: the caliper must be finite and non-negative
    #[error("the caliper must be finite and non-negative")]
    Caliper,
}

/// Why an interference-query assignment or exposure declaration is refused,
/// carried by [`QueryError::InvalidInterference`](super::QueryError::InvalidInterference).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum InterferenceFault {
    /// Refused: Bernoulli probabilities must be finite and strictly between zero and one
    #[error("Bernoulli probabilities must be finite and strictly between zero and one")]
    BernoulliProbabilities,
    /// Refused: complete randomization requires at least one treated unit
    #[error("complete randomization requires at least one treated unit")]
    CompleteTreatedUnits,
    /// Refused: cluster randomization requires clusters and at least one treated cluster
    #[error("cluster randomization requires clusters and at least one treated cluster")]
    ClusterTreatedClusters,
    /// Refused: two-stage saturation requires aligned clusters, 0 < low < high < 1, and `high_clusters` > 0
    #[error(
        "two-stage saturation requires aligned clusters, 0 < low < high < 1, and high_clusters > 0"
    )]
    TwoStageSaturation,
    /// Refused: observational exposure requires aligned positive propensities and an explicit network exchangeability assumption
    #[error(
        "observational exposure requires aligned positive propensities and an explicit network exchangeability assumption"
    )]
    ObservedExposure,
    /// Refused: exposure levels must be finite/distinct and `probability_draws` must be positive
    #[error("exposure levels must be finite/distinct and probability_draws must be positive")]
    ExposureLevels,
}

/// Why an instrumental-variable bounds declaration is refused,
/// carried by [`QueryError::InvalidIvBounds`](super::QueryError::InvalidIvBounds).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum IvBoundsFault {
    /// Refused: declare distinct outcome, treatment, instrument and weight columns
    #[error("declare distinct outcome, treatment, instrument and weight columns")]
    Columns,
}

/// Why a fixed-cutoff local-polynomial ratio declaration is refused,
/// carried by [`QueryError::InvalidLocalPolynomialRatio`](super::QueryError::InvalidLocalPolynomialRatio).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum LocalPolynomialRatioFault {
    /// Refused: outcome, treatment, and running variable must be distinct
    #[error("outcome, treatment, and running variable must be distinct")]
    VariablesNotDistinct,
    /// Refused: cutoff must be finite and bandwidth positive
    #[error("cutoff must be finite and bandwidth positive")]
    CutoffOrBandwidth,
    /// Refused: a calendar-time block bootstrap needs at least two replicates
    #[error("a calendar-time block bootstrap needs at least two replicates")]
    BootstrapReplicates,
    /// Refused: balance covariates must be distinct columns other than the outcome, treatment and running variable
    #[error(
        "balance covariates must be distinct columns other than the outcome, treatment and running variable"
    )]
    BalanceCovariates,
}

/// Why a longitudinal-regime value declaration is refused,
/// carried by [`QueryError::InvalidLongitudinalRegime`](super::QueryError::InvalidLongitudinalRegime).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum LongitudinalRegimeFault {
    /// Refused: dynamic rule requires a nonempty ID, version, and provenance and a regime-value method
    #[error(
        "dynamic rule requires a nonempty ID, version, and provenance and a regime-value method"
    )]
    DynamicRuleIdentity,
    /// Refused: dimensions overflow
    #[error("dimensions overflow")]
    DimensionsOverflow,
    /// Refused: histories, probabilities, outcomes, subjects, and folds must align
    #[error("histories, probabilities, outcomes, subjects, and folds must align")]
    Alignment,
    /// Refused: method requires aligned finite predictions, observation history, and subject-owned folds
    #[error(
        "method requires aligned finite predictions, observation history, and subject-owned folds"
    )]
    MethodPredictions,
    /// Refused: known fixed outcome predictions require `g_formula`
    #[error("known fixed outcome predictions require g_formula")]
    FixedPredictionsNeedGFormula,
    /// Refused: sequential DR fields require the sequential doubly robust method
    #[error("sequential DR fields require the sequential doubly robust method")]
    SequentialDrFields,
    /// Refused: stabilizing numerator probabilities require the marginal structural model method
    #[error("stabilizing numerator probabilities require the marginal structural model method")]
    StabilizingNumerator,
    /// Refused: observation history must be monotone and agree with endpoint observation
    #[error("observation history must be monotone and agree with endpoint observation")]
    ObservationHistory,
    /// Refused: subject IDs must be unique and nonempty; one outcome row owns the whole history
    #[error("subject IDs must be unique and nonempty; one outcome row owns the whole history")]
    SubjectIds,
    /// Refused: probabilities require a known sequential design or subject-level excluded-fold ownership
    #[error(
        "probabilities require a known sequential design or subject-level excluded-fold ownership"
    )]
    ProbabilityOwnership,
    /// Refused: excluded-fold prediction ownership requires at least two subject folds
    #[error("excluded-fold prediction ownership requires at least two subject folds")]
    ExcludedFoldCount,
    /// Refused: sequential treatment or censoring positivity fails at the declared floor
    #[error("sequential treatment or censoring positivity fails at the declared floor")]
    PositivityFloor,
}

/// Why a continuous-response, temporal-response, derivative and smoothed-dose declaration is refused,
/// carried by [`QueryError::InvalidResponse`](super::QueryError::InvalidResponse).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ResponseFault {
    /// Refused: temporal mediation requires at least one horizon
    #[error("temporal mediation requires at least one horizon")]
    MediationHorizonsEmpty,
    /// Refused: temporal mediation horizons must be strictly increasing
    #[error("temporal mediation horizons must be strictly increasing")]
    MediationHorizonsOrder,
    /// Refused: feedback delay must be at least one tick; zero is contemporaneous transmission
    #[error("feedback delay must be at least one tick; zero is contemporaneous transmission")]
    FeedbackDelay,
    /// Refused: feedback level, coefficients, variance, and initial state must be finite
    #[error("feedback level, coefficients, variance, and initial state must be finite")]
    FeedbackNotFinite,
    /// Refused: feedback innovation variance must not be negative
    #[error("feedback innovation variance must not be negative")]
    FeedbackNegativeVariance,
    /// Refused: temporal response requires at least one horizon
    #[error("temporal response requires at least one horizon")]
    HorizonsEmpty,
    /// Refused: temporal response horizons must be strictly increasing
    #[error("temporal response horizons must be strictly increasing")]
    HorizonsOrder,
    /// Refused: linspace point count exceeds available memory
    #[error("linspace point count exceeds available memory")]
    LinspaceExceedsMemory,
    /// Refused: a response grid requires at least two points
    #[error("a response grid requires at least two points")]
    GridTooShort,
    /// Refused: response-grid values must be finite and strictly increasing
    #[error("response-grid values must be finite and strictly increasing")]
    GridValuesOrder,
    /// Refused: linspace requires finite start < end and at least two points
    #[error("linspace requires finite start < end and at least two points")]
    LinspaceBounds,
    /// Refused: uniform derivative weighting requires finite lower < upper
    #[error("uniform derivative weighting requires finite lower < upper")]
    UniformWeightBounds,
    /// Refused: custom derivative weights must be finite, non-negative, and non-zero
    #[error("custom derivative weights must be finite, non-negative, and non-zero")]
    CustomWeights,
    /// Refused: point derivative requires a finite point and order one or two
    #[error("point derivative requires a finite point and order one or two")]
    PointDerivative,
    /// Refused: log-treatment derivative scales require a positive treatment point
    #[error("log-treatment derivative scales require a positive treatment point")]
    LogScalePoint,
    /// Refused: directional derivative dimensions/values are inconsistent
    #[error("directional derivative dimensions/values are inconsistent")]
    DirectionalDimensions,
    /// Refused: Jacobian dimensions/values are inconsistent
    #[error("Jacobian dimensions/values are inconsistent")]
    JacobianDimensions,
    /// Refused: log-treatment Jacobians require positive treatment coordinates
    #[error("log-treatment Jacobians require positive treatment coordinates")]
    LogScaleJacobian,
    /// Refused: intervention response requires at least one intervention
    #[error("intervention response requires at least one intervention")]
    NoIntervention,
    /// Refused: temporal response dose-by-horizon cell count overflow
    #[error("temporal response dose-by-horizon cell count overflow")]
    DoseHorizonCellOverflow,
    /// Refused: temporal response policy must be pulse or sustained; Dynamic is a `TemporalEffect` spelling, not a `ResponseCurve` cell
    #[error(
        "temporal response policy must be pulse or sustained; Dynamic is a TemporalEffect spelling, not a ResponseCurve cell"
    )]
    DynamicPolicyNotACell,
    /// Refused: a smoothed dose-response transport query needs distinct dose and outcome coordinates, a non-empty grid of distinct finite doses, a positive finite bandwidth, a finite dose support lo < hi, distinct population keys and a density provenance
    #[error(
        "a smoothed dose-response transport query needs distinct dose and outcome coordinates, a non-empty grid of distinct finite doses, a positive finite bandwidth, a finite dose support lo < hi, distinct population keys and a density provenance"
    )]
    SmoothedDoseDeclaration,
    /// Refused: a policy tag other than the licensed pulse and sustained spellings.
    #[error("temporal response policy must be {} or {}; got {}", TemporalResponseSpec::POLICY_PULSE, TemporalResponseSpec::POLICY_SUSTAINED, .got)]
    UnsupportedPolicy {
        /// Name of the refused tag.
        got: String,
    },
    /// Refused: the observation mechanism and assumption pair is not licensed on a temporal curve.
    #[error("{}", TEMPORAL_OBSERVATION_UNLICENSED)]
    TemporalObservationUnlicensed,
}

/// Why an observational sequential regime-contrast declaration is refused,
/// carried by [`QueryError::InvalidSequentialContrast`](super::QueryError::InvalidSequentialContrast).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SequentialContrastFault {
    /// Refused: outcome, both actions and every covariate must be distinct columns
    #[error("outcome, both actions and every covariate must be distinct columns")]
    ColumnsNotDistinct,
    /// Refused: a regime action must be 0 or 1
    #[error("a regime action must be 0 or 1")]
    RegimeAction,
    /// Refused: cross-fitting needs at least two folds and a fold in range for every row
    #[error("cross-fitting needs at least two folds and a fold in range for every row")]
    CrossFitting,
    /// Refused: `min_probability` must be in (0, 0.5]
    #[error("min_probability must be in (0, 0.5]")]
    MinProbability,
}

/// Why a survival or competing-risk observation declaration is refused,
/// carried by [`QueryError::InvalidSurvival`](super::QueryError::InvalidSurvival).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SurvivalFault {
    /// Refused: duration, event, treatment, and entry must name distinct columns
    #[error("duration, event, treatment, and entry must name distinct columns")]
    ColumnsNotDistinct,
    /// Refused: tau must be finite and positive
    #[error("tau must be finite and positive")]
    Horizon,
    /// Refused: target cause must be a positive event code
    #[error("target cause must be a positive event code")]
    TargetCause,
    /// Refused: combined delayed entry and known censoring requires marginal `IndependentGiven`(())
    #[error("combined delayed entry and known censoring requires marginal IndependentGiven(())")]
    EntryWithKnownCensoring,
    /// Refused: known censoring requires aligned columns, increasing times from zero through tau, and a positive probability floor
    #[error(
        "known censoring requires aligned columns, increasing times from zero through tau, and a positive probability floor"
    )]
    KnownCensoring,
    /// Refused: randomized survival requires `IndependentGiven`; conditional censoring also requires known censoring survival
    #[error(
        "randomized survival requires IndependentGiven; conditional censoring also requires known censoring survival"
    )]
    RandomizedCensoring,
}

/// Why a transportability query or evidence-catalog declaration is refused,
/// carried by [`QueryError::InvalidTransport`](super::QueryError::InvalidTransport).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TransportFault {
    /// Refused: `source_experiments` disagree with the evidence catalog
    #[error("source_experiments disagree with the evidence catalog")]
    SourceExperimentsDisagree,
    /// Refused: source and target population keys must be non-empty and distinct
    #[error("source and target population keys must be non-empty and distinct")]
    PopulationKeys,
    /// Refused: source experiment variables must be unique
    #[error("source experiment variables must be unique")]
    SourceExperimentsNotUnique,
    /// Refused: environment identity must be non-empty
    #[error("environment identity must be non-empty")]
    EnvironmentIdentityEmpty,
    /// Refused: categorical cardinality must be positive
    #[error("categorical cardinality must be positive")]
    CategoricalCardinality,
    /// Refused: environment variable coordinates must be unique
    #[error("environment variable coordinates must be unique")]
    CoordinatesNotUnique,
    /// Refused: regime population must be non-empty
    #[error("regime population must be non-empty")]
    RegimePopulationEmpty,
    /// Refused: observational regimes cannot carry hard interventions
    #[error("observational regimes cannot carry hard interventions")]
    ObservationalWithInterventions,
    /// Refused: experimental regimes require a non-empty intervention set
    #[error("experimental regimes require a non-empty intervention set")]
    ExperimentalWithoutInterventions,
    /// Refused: marginals require measured variables
    #[error("marginals require measured variables")]
    MarginalsNeedMeasured,
    /// Refused: invalid or duplicate intervention assignment
    #[error("invalid or duplicate intervention assignment")]
    InterventionAssignment,
    /// Refused: regime study identity must be non-empty
    #[error("regime study identity must be non-empty")]
    StudyIdentityEmpty,
    /// Refused: model artifact identity must be non-empty
    #[error("model artifact identity must be non-empty")]
    ModelArtifactIdentityEmpty,
    /// Refused: recovered law derivation identity must be non-empty
    #[error("recovered law derivation identity must be non-empty")]
    DerivationIdentityEmpty,
    /// Refused: a selected-sample law must name its selection variables
    #[error("a selected-sample law must name its selection variables")]
    SelectedSampleVariables,
    /// Refused: removing a hard intervention and treating its law as observational is not licensed
    #[error("removing a hard intervention and treating its law as observational is not licensed")]
    DeinterventionNotLicensed,
    /// Refused: cannot marginalize an unmeasured variable
    #[error("cannot marginalize an unmeasured variable")]
    MarginalizeUnmeasured,
    /// Refused: conditioning requires a joint law
    #[error("conditioning requires a joint law")]
    ConditioningNeedsJointLaw,
    /// Refused: conditioning projection requires measured variables
    #[error("conditioning projection requires measured variables")]
    ConditioningNeedsMeasured,
    /// Refused: regime population is not a declared environment
    #[error("regime population is not a declared environment")]
    PopulationNotDeclared,
    /// Refused: regime variable is not declared by any environment
    #[error("regime variable is not declared by any environment")]
    VariableNotDeclared,
    /// Refused: duplicate or empty regime label
    #[error("duplicate or empty regime label")]
    RegimeLabel,
    /// Refused: conditioning requires measured coordinates
    #[error("conditioning requires measured coordinates")]
    ConditioningNeedsCoordinates,
    /// Refused: intervention value violates declared domain
    #[error("intervention value violates declared domain")]
    InterventionValueDomain,
    /// Refused: catalog regime ids must be unique
    #[error("catalog regime ids must be unique")]
    RegimeIdsNotUnique,
    /// Refused: catalog environments must have distinct identities
    #[error("catalog environments must have distinct identities")]
    EnvironmentIdentitiesNotDistinct,
    /// Refused: catalog environments declare incompatible domains for the same variable
    #[error("catalog environments declare incompatible domains for the same variable")]
    IncompatibleDomains,
    /// Refused: catalog environments declare incompatible units for the same variable
    #[error("catalog environments declare incompatible units for the same variable")]
    IncompatibleUnits,
    /// Refused: invalid, duplicate, or mismatched regime binding
    #[error("invalid, duplicate, or mismatched regime binding")]
    BindingInvalid,
    /// Refused: regime binding names an unknown regime
    #[error("regime binding names an unknown regime")]
    BindingUnknownRegime,
    /// Refused: regime binding names a regime whose results do not exist
    #[error("regime binding names a regime whose results do not exist")]
    BindingMissingResults,
    /// Refused: catalog delta requires distinct proposed regimes
    #[error("catalog delta requires distinct proposed regimes")]
    DeltaRegimesNotDistinct,
    /// Refused: a list of variables names one twice.
    #[error("{label} must be unique")]
    NotUnique {
        /// Which list.
        label: &'static str,
    },
}

/// Why a noncausal association declaration is refused,
/// carried by [`QueryError::InvalidAssociation`](super::QueryError::InvalidAssociation).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum AssociationFault {
    /// Refused: declare distinct outcome, predictor, conditioning and weight columns; a column cannot take two roles
    #[error(
        "declare distinct outcome, predictor, conditioning and weight columns; a column cannot take two roles"
    )]
    ColumnRoles,
    /// Refused: an association measure relates the outcome to exactly one predictor and holds nothing fixed; use coefficients to adjust for conditioning columns
    #[error(
        "an association measure relates the outcome to exactly one predictor and holds nothing fixed; use coefficients to adjust for conditioning columns"
    )]
    MeasureShape,
    /// Refused: regression coefficients need at least one predictor
    #[error("regression coefficients need at least one predictor")]
    CoefficientsPredictor,
    /// Refused: a predictive model needs at least one predictor
    #[error("a predictive model needs at least one predictor")]
    PredictiveModelPredictor,
    /// Refused: a predictive model is scored over at least 2 folds
    #[error("a predictive model is scored over at least 2 folds")]
    PredictiveModelFolds,
    /// Refused: the recorded refusal reason must be 1 to 256 bytes
    #[error("the recorded refusal reason must be 1 to 256 bytes")]
    RefusalReasonLength,
}

/// Why a structure-discovery declaration is refused,
/// carried by [`QueryError::InvalidDiscover`](super::QueryError::InvalidDiscover).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DiscoverFault {
    /// Refused: the `LiNGAM` prune threshold must be finite and non-negative
    #[error("the LiNGAM prune threshold must be finite and non-negative")]
    LingamPruneThreshold,
    /// Refused: the NOTEARS penalty and threshold must be finite and non-negative
    #[error("the NOTEARS penalty and threshold must be finite and non-negative")]
    NotearsPenalty,
    /// Refused: the GES subset cap must be at least 1
    #[error("the GES subset cap must be at least 1")]
    GesSubsetCap,
    /// Refused: a class posterior collapses the graphs of an exact, order-MCMC, structure-MCMC, CI-screened or bounded-lag template posterior; the exact edge marginals list no graphs
    #[error(
        "a class posterior collapses the graphs of an exact, order-MCMC, structure-MCMC, CI-screened or bounded-lag template posterior; the exact edge marginals list no graphs"
    )]
    ClassPosteriorBase,
    /// Refused: a stability search draws at least 2 resamples
    #[error("a stability search draws at least 2 resamples")]
    StabilityResamples,
    /// Refused: a stability search re-runs PC, FCI, RFCI, GES, `LiNGAM`, NOTEARS, PCMCI, PCMCI+, LPCMCI, J-PCMCI+ or regime PCMCI; a posterior has its own sampling error
    #[error(
        "a stability search re-runs PC, FCI, RFCI, GES, LiNGAM, NOTEARS, PCMCI, PCMCI+, LPCMCI, J-PCMCI+ or regime PCMCI; a posterior has its own sampling error"
    )]
    StabilityBase,
    /// Refused: a structure search needs at least one variable
    #[error("a structure search needs at least one variable")]
    NoVariables,
    /// Refused: a searched variable is listed twice
    #[error("a searched variable is listed twice")]
    RepeatedVariable,
    /// Refused: the significance level must be in (0, 1)
    #[error("the significance level must be in (0, 1)")]
    SignificanceLevel,
    /// Refused: a static search takes no lag; set `max_lag` to 0
    #[error("a static search takes no lag; set max_lag to 0")]
    StaticSearchLag,
    /// Refused: the doubled temporal CI lag window is not representable.
    #[error("the doubled discovery max_lag exceeds u32")]
    DoubledLagOverflow,
    /// Refused: an MCMC schedule needs at least one chain, draw and thin
    #[error("an MCMC schedule needs at least one chain, draw and thin")]
    McmcSchedule,
    /// Refused: a sampled structure posterior needs an MCMC schedule
    #[error("a sampled structure posterior needs an MCMC schedule")]
    PosteriorNeedsSchedule,
}

/// Why a simulation declaration on a fitted structural model is refused,
/// carried by [`QueryError::InvalidSimulate`](super::QueryError::InvalidSimulate).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SimulateFault {
    /// Refused: an interventional sample needs at least one draw
    #[error("an interventional sample needs at least one draw")]
    SampleDraws,
    /// Refused: a policy value needs at least two draws to report its simulation error
    #[error("a policy value needs at least two draws to report its simulation error")]
    PolicyValueDraws,
    /// Refused: unit effects need finite treatment levels
    #[error("unit effects need finite treatment levels")]
    UnitEffectLevels,
    /// Refused: a noise sample needs at least one draw
    #[error("a noise sample needs at least one draw")]
    NoiseSampleDraws,
    /// Refused: replayed noise needs `n_rows` > 0 values for each variable
    #[error("replayed noise needs n_rows > 0 values for each variable")]
    ReplayedNoiseShape,
    /// Refused: replayed noise must be finite
    #[error("replayed noise must be finite")]
    ReplayedNoiseNotFinite,
    /// Refused: a rollout needs at least one row and one step
    #[error("a rollout needs at least one row and one step")]
    RolloutShape,
    /// Refused: rollout history needs `n_rows` values per variable and step
    #[error("rollout history needs n_rows values per variable and step")]
    RolloutHistoryShape,
    /// Refused: a panel replay needs at least one unit and one step
    #[error("a panel replay needs at least one unit and one step")]
    PanelReplayShape,
    /// Refused: the panel needs `n_rows` values per variable at every step
    #[error("the panel needs n_rows values per variable at every step")]
    PanelValuesShape,
    /// Refused: panel history needs `n_rows` values per variable and step
    #[error("panel history needs n_rows values per variable and step")]
    PanelHistoryShape,
    /// Refused: the panel and its history must be finite
    #[error("the panel and its history must be finite")]
    PanelNotFinite,
}

/// Why a sequentially randomized unit-period design declaration is refused,
/// carried by [`QueryError::InvalidTemporalDesign`](super::QueryError::InvalidTemporalDesign).
/// Each variant is one violated contract; the display text is its stable statement.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TemporalDesignFault {
    /// Refused: a whole-regime contrast reads the mean of a continuous outcome without a misclassification correction
    #[error(
        "a whole-regime contrast reads the mean of a continuous outcome without a misclassification correction"
    )]
    WholeRegimeMisclassification,
    /// Refused: the active and reference regimes name one action per period, the same number of periods each, and differ in some period
    #[error(
        "the active and reference regimes name one action per period, the same number of periods each, and differ in some period"
    )]
    RegimeActions,
    /// Refused: unit and period labels must align with at least two rows
    #[error("unit and period labels must align with at least two rows")]
    Alignment,
    /// Refused: unit labels must be non-empty
    #[error("unit labels must be non-empty")]
    BlankUnit,
    /// Refused: each (unit, period) must appear on exactly one row
    #[error("each (unit, period) must appear on exactly one row")]
    DuplicateUnitPeriod,
    /// Refused: `min_probability` must be in (0, 0.5]
    #[error("min_probability must be in (0, 0.5]")]
    MinProbability,
    /// Refused: an outcome misclassification correction applies to the mean of an arm design
    #[error("an outcome misclassification correction applies to the mean of an arm design")]
    MisclassificationDesign,
    /// Refused: the validation counts need true positives and true negatives
    #[error("the validation counts need true positives and true negatives")]
    ValidationCounts,
    /// Refused: the quantile level must be in (0, 1)
    #[error("the quantile level must be in (0, 1)")]
    QuantileLevel,
    /// Refused: the quantile functional is defined for a two-arm design
    #[error("the quantile functional is defined for a two-arm design")]
    QuantileDesign,
    /// Refused: two-arm assignment and probabilities must align with the rows and lie in (0, 1)
    #[error("two-arm assignment and probabilities must align with the rows and lie in (0, 1)")]
    TwoArmAssignment,
    /// Refused: declare at least three distinct non-empty arm labels
    #[error("declare at least three distinct non-empty arm labels")]
    ArmLabels,
    /// Refused: ordered levels are a categorical design
    #[error("ordered levels are a categorical design")]
    OrderedNeedsCategorical,
    /// Refused: every row needs a realized arm in range and a probability row over every arm summing to one
    #[error(
        "every row needs a realized arm in range and a probability row over every arm summing to one"
    )]
    MultiArmRows,
    /// Refused: both factors and their probabilities must align with the rows and lie in (0, 1)
    #[error("both factors and their probabilities must align with the rows and lie in (0, 1)")]
    FactorialAssignment,
    /// Refused: assignment, probabilities and non-empty cluster labels must align with the rows
    #[error("assignment, probabilities and non-empty cluster labels must align with the rows")]
    InterferenceAssignment,
    /// Refused: doses and positive densities must align with the rows, and the bandwidth must be positive
    #[error(
        "doses and positive densities must align with the rows, and the bandwidth must be positive"
    )]
    DoseDensities,
    /// Refused: a dose target needs a finite grid, two distinct doses, or a non-negative curvature bound
    #[error(
        "a dose target needs a finite grid, two distinct doses, or a non-negative curvature bound"
    )]
    DoseTarget,
    /// Refused: regime survival reads each period's event (lag 0) up to a positive horizon
    #[error("regime survival reads each period's event (lag 0) up to a positive horizon")]
    RegimeSurvivalHorizon,
    /// Refused: event, treatment, censoring and confounder columns must be distinct
    #[error("event, treatment, censoring and confounder columns must be distinct")]
    RegimeSurvivalColumns,
    /// Refused: an outcome misclassification correction applies to the recorded-law arm designs
    #[error("an outcome misclassification correction applies to the recorded-law arm designs")]
    MisclassificationRecordedLaw,
    /// Refused: outcome, treatment and history columns of an observational excursion must be distinct
    #[error(
        "outcome, treatment and history columns of an observational excursion must be distinct"
    )]
    ExcursionColumns,
    /// Refused: a whole-regime contrast reads the outcome at the end of follow-up (lag 0)
    #[error("a whole-regime contrast reads the outcome at the end of follow-up (lag 0)")]
    WholeRegimeLag,
    /// Refused: outcome, treatment and confounder columns of a whole-regime contrast must be distinct
    #[error(
        "outcome, treatment and confounder columns of a whole-regime contrast must be distinct"
    )]
    WholeRegimeColumns,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::QueryError;

    #[test]
    fn a_fault_keeps_its_display_text_inside_the_query_error() {
        let error = QueryError::InvalidTransport(TransportFault::SourceExperimentsDisagree);
        assert_eq!(
            error.to_string(),
            "invalid transport query: source_experiments disagree with the evidence catalog"
        );
        let error =
            QueryError::InvalidResponse(ResponseFault::UnsupportedPolicy { got: "dynamic".into() });
        assert_eq!(
            error.to_string(),
            "invalid response query: temporal response policy must be pulse or sustained; got dynamic"
        );
        let error = QueryError::InvalidTransport(TransportFault::NotUnique {
            label: "regime interventions",
        });
        assert_eq!(
            error.to_string(),
            "invalid transport query: regime interventions must be unique"
        );
    }

    #[test]
    fn callers_match_on_the_kind() {
        let error = QueryError::InvalidDiscover(DiscoverFault::StaticSearchLag);
        assert!(matches!(error, QueryError::InvalidDiscover(DiscoverFault::StaticSearchLag)));
        let rejected = QueryError::InvalidIntervention(InterventionFault::Rejected(
            InterventionError::NonFiniteLevel,
        ));
        assert_eq!(rejected.to_string(), "invalid intervention: intervention level must be finite");
        assert_eq!(
            QueryError::InvalidResponse(ResponseFault::TemporalObservationUnlicensed).to_string(),
            format!("invalid response query: {TEMPORAL_OBSERVATION_UNLICENSED}")
        );
    }
}
