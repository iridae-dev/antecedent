//! Estimators, estimation results and strategy identifiers, for configuring or reading the
//! estimation stage of a study.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_data::{NetworkData, NetworkEdge};

pub use antecedent_estimate::{
    CausalPosterior, ConditionalLinearAdjustment, ContinuousResponseOptions, EffectEstimate,
    EstimationError, InterferenceEstimate, OverlapPolicy, ResponseGridInputs,
    TemporalEffectSurface, TemporalLinearPredictor, TemporalMediationEstimator,
    TemporalMediationGrid, TemporalMediationIdentifiedSet, TemporalMediationSlice,
    TemporalMediationUncertainty, TransportEffectEstimate, TransportOverlapDiagnostic,
    TransportOverlapReport, TransportResponseGridEstimate, own_treatment_level,
};

pub use crate::strategy_table::{
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER_ID,
    DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_PATH_ESTIMATOR_ID,
    DEFAULT_PATH_IDENTIFIER_ID, EstimatorId, IdentifierId, StaticEstimateWorkspaces,
    estimand_compatible_with_estimator, identification_status_acceptable, require_identified,
    select_estimand, validate_distribution_pair, validate_path_specific_pair, validate_static_pair,
};

pub use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits,
    ExactSupportRecord, ExactTransportData, LawTolerance,
};

pub use antecedent_estimate::white_innovation_posterior::{
    InnovationWhitenessGate, InnovationWhitenessReport,
};

/// `transport_augmented_response_grid` under the caller's memory, work and cancellation policy.
///
/// The supplied-equation evaluation with no study around it: the entry for a caller that supplies
/// every nuisance itself and owns the execution policy the numerical kernels obey. The result
/// states only the grid means, source effective sample sizes and selection overlap it computed.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::transport_augmented_response_grid_with_context`].
///
/// # Errors
///
/// The refusals of `transport_augmented_response_grid`.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled TransportResponseGrid route, which calls this kernel function with the same inputs"
)]
pub fn transport_augmented_response_grid_with_context(
    identification: &antecedent_identify::transport::TransportIdentification,
    outcome: &[f64],
    trial: &[bool],
    selection_probability: &[f64],
    inputs: &antecedent_estimate::ResponseGridInputs<'_>,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_estimate::TransportResponseGridEstimate, antecedent_estimate::EstimationError>
{
    antecedent_estimate::transport_augmented_response_grid_with_context(
        identification,
        outcome,
        trial,
        selection_probability,
        inputs,
        ctx,
    )
}

/// Evaluate a caller-specified augmented trial-to-target response equation over a grid.
///
/// This function composes the trial-generalization augmented formula of Dahabreh et al. (2019,
/// Web Appendix 2) with a grid-local source correction. `target_regression[g * n + i]` is the
/// source outcome regression at intervention `grid[g]`, evaluated for target row `i`.
/// `observed_regression[i]` is the same regression at the observed source treatment.
/// `response_weight[g * n + i]` is a caller-constructed treatment-density/local
/// smoothing equivalent weight for source row `i`. Signed weights are accepted, as required by
/// local-linear smoothers near boundaries. This primitive evaluates the supplied equation; it
/// does not claim that the composition itself inherits either component method's
/// double-robustness or inference theorem.
///
/// The implementation intentionally does not estimate either nuisance or choose a bandwidth:
/// doing so here would conflate structural transport, treatment support, and nuisance learning.
/// Target-row outcomes and response weights are ignored, but must still be finite so malformed
/// rectangular inputs cannot pass silently.
///
/// `identification` must be the `TransportIdentification` produced for this query by
/// `antecedent_identify::TransportIdentifier::identify`. A transported estimate is only
/// meaningful once a sound formula has been certified for the query being evaluated here;
/// evaluating this grid against an uncertified query would silently manufacture a number for
/// a quantity that was never shown to be identified. When `identification` is
/// `TransportIdentification::NotCertified`, or when it certifies a
/// `TransportFormula::RecursiveFactorization` that this Dahabreh-style grid does not
/// evaluate, this function returns an error rather than an estimate.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::transport_augmented_response_grid`].
///
/// # Errors
///
/// Returns `EstimationError` for empty/mismatched inputs, non-finite values, invalid selection
/// probabilities, non-finite response weights, samples without both source and target rows, an
/// `identification` that is `TransportIdentification::NotCertified`, or a recursive-
/// factorization certificate.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled TransportResponseGrid route, which calls this kernel function with the same inputs"
)]
pub fn transport_augmented_response_grid(
    identification: &antecedent_identify::transport::TransportIdentification,
    outcome: &[f64],
    trial: &[bool],
    selection_probability: &[f64],
    inputs: &antecedent_estimate::ResponseGridInputs<'_>,
) -> Result<antecedent_estimate::TransportResponseGridEstimate, antecedent_estimate::EstimationError>
{
    antecedent_estimate::transport_augmented_response_grid(
        identification,
        outcome,
        trial,
        selection_probability,
        inputs,
    )
}

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Augmented inverse-probability weighting (AIPW / doubly robust) ATE estimator.
pub use antecedent_estimate::AipwAte;
/// Shared analytic SE policy for ATE estimators.
pub use antecedent_estimate::AnalyticSeKind;
/// Honest causal forest CATE (`CausalForest`).
pub use antecedent_estimate::CausalForest;
/// Cell-saturated AIPW for discrete joint `Set` interventions.
pub use antecedent_estimate::CellSaturatedAipw;
/// The one checked nuisance-reuse store.
pub use antecedent_estimate::CrossfitNuisanceCache;
/// Covariate-distance matching estimator.
pub use antecedent_estimate::DistanceMatching;
/// Doubly robust CATE learner (`DrLearner`).
pub use antecedent_estimate::DrLearner;
/// Linear front-door two-stage (product-of-coefficients) regression estimator.
pub use antecedent_estimate::FrontDoorTwoStage;
/// Generalized linear model (logistic) adjustment ATE estimator for binary outcomes.
pub use antecedent_estimate::GlmAdjustmentAte;
/// Learned continuous-outcome trial-to-target transport.
pub use antecedent_estimate::LearnedContinuousOptions;
/// Supplied-propensity observational exposure contrasts on a fixed network.
pub use antecedent_estimate::ObservationalExposureSpec;
/// Declared treatment levels of a propensity problem.
pub use antecedent_estimate::OtherLevels;
/// Propensity stratification estimator.
pub use antecedent_estimate::PropensityStratification;
/// Inverse-probability weighting estimator.
pub use antecedent_estimate::PropensityWeighting;
/// Exact evaluation of a checked observation-recovery formula and of the downstream effect.
pub use antecedent_estimate::RecoveredLaw;
/// Estimation errors.
pub use antecedent_estimate::RefusalFields;
/// Descriptive association of an outcome with predictors.
pub use antecedent_estimate::SPREAD_LEVEL;
/// Declared sampling weights.
pub use antecedent_estimate::SamplingWeights;
/// Cross-fitted AIPW score tables for retargeting and joint inference.
pub use antecedent_estimate::ScoreTable;
/// Classification a fit applied to a numeric treatment: discrete levels or a dose.
pub use antecedent_estimate::TreatmentClassification;
/// Caller declaration that a numeric treatment is levels or a dose.
pub use antecedent_estimate::TreatmentDeclaration;
/// Joint outer bootstrap for catalog-bound empirical transport tables.
pub use antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED;
/// Propensity problem preparation and shared workspace.
pub use antecedent_estimate::default_propensity_overlap;
/// The matched case-control sampling declaration of the conditional odds ratio.
pub use antecedent_estimate::parse_matched_sampling;
/// Supported temporal intervention plans.
pub use antecedent_estimate::plan_from_response_query;
/// Descriptive raw-versus-adjusted comparison and declared reporting-scale transforms.
pub use antecedent_estimate::{
    AdjustedEstimate, ArmSummary, Availability, MeanPair, RawContrast, ReportingScale,
    ReportingTransform, Unavailable,
};
/// Partially linear DML and cross-fitted AIPW via `antecedent-learn`.
pub use antecedent_estimate::{AlphaSelectionRequest, DmlAte, DmlScore, OutcomePriorSource};
/// Finite categorical frequency plug-in for transport joints.
pub use antecedent_estimate::{
    BayesianTransportLawProvider, EmpiricalTableEstimator, EmpiricalTableOptions, RegimeSample,
    StatisticalTransportInput,
};
/// Propensity-score nearest-neighbor matching.
pub use antecedent_estimate::{CaliperScale, PropensityMatching};
/// Factorized-propensity AIPW for bounded discrete joint cells.
pub use antecedent_estimate::{
    CellStatus, FactorizedJointConfig, FactorizedJointFit, JointCellReport, JointContrast,
    declared_joint_nuisance, orderings_for,
};
/// Cluster-aware cross-fitted AIPW: the independence unit is a declared cluster.
pub use antecedent_estimate::{
    ClusterDml, DEFAULT_MIN_CLUSTERS, DEFAULT_MIN_COMPONENTS_PER_FOLD, DeclaredUnits,
};
/// Explicit penalized binary-propensity configuration for cross-fitted AIPW.
pub use antecedent_estimate::{
    DEFAULT_RIDGE_GRID, NuisanceFallback, PropensityNuisance, RidgeTuning,
};
/// `antecedent-estimate` items at the crate root.
pub use antecedent_estimate::{FittedEffect, IdentifiedSetIntervalMethod, LearnerSpec};
/// Bootstrap orchestration: standard errors.
pub use antecedent_estimate::{IntervalRequest, ResamplingReport};
/// Nonparametric estimation of identified interventional distributions via discrete empirical.
pub use antecedent_estimate::{InterventionalDistributionEstimate, ProbabilityInterval};
/// Linear adjustment ATE estimator.
pub use antecedent_estimate::{LinearAdjustmentAte, LinearFitKind};
/// Estimator-side handling of explicit outcome-observation mechanisms.
pub use antecedent_estimate::{
    ObservationEstimatorOptions, ObservationMechanismEstimator, SelectedOutcomeCorrection,
};
/// Sampled observation recovery.
pub use antecedent_estimate::{
    ObservationPattern, ObservationRow, SampledObservationInput, SampledRecoveryConfig,
};
/// Smoothed dose-response transport grid.
pub use antecedent_estimate::{SmoothedDoseInput, SmoothedDoseOptions};
/// Cross-fitted trial-to-nonparticipant transport with explicit sampling design.
pub use antecedent_estimate::{TrialAipwInput, TrialAipwOptions};
/// Instrumental-variable estimators: Wald ratio and two-stage least squares.
pub use antecedent_estimate::{TwoStageLeastSquares, WaldIv};
/// Validated finite-discrete laws.
pub use antecedent_expr::ExactLawError;
/// `antecedent-expr` items at the crate root.
pub use antecedent_expr::{CausalExprArena, ExprId, ExprNode, InterventionAssignment};
/// Distribution providers for compiled expression evaluation.
pub use antecedent_expr::{DistributionProvider, EvalContext, EvalError, FactorSpec};
/// Immutable, structurally checked expression programs.
pub use antecedent_expr::{
    FunctionalProgram, ProgramError, ProgramLimits, ProgramSchema, ProgramVariable,
};

/// `trial_to_target_effect` under the caller's memory, work and cancellation policy.
///
/// The trial-to-target contrast with no study around it: the entry for a caller that supplies
/// the nuisance probabilities itself and owns the execution policy the numerical kernels
/// obey.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::trial_to_target_effect_with_context`].
///
/// # Errors
///
/// The refusals of `trial_to_target_effect`.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled TransportMixed route, which calls this kernel function with the same inputs"
)]
#[allow(clippy::too_many_arguments, reason = "the signature is the kernel function's, unchanged")]
pub fn trial_to_target_effect_with_context(
    identification: &antecedent_identify::transport::TransportIdentification,
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
    outcome_regressions: Option<(&[f64], &[f64])>,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_estimate::TransportEffectEstimate, antecedent_estimate::EstimationError> {
    antecedent_estimate::trial_to_target_effect_with_context(
        identification,
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
        outcome_regressions,
        ctx,
    )
}

/// Put a mean-outcome pair on each declared reporting scale, with the delta-method
/// covariance of the family when the pair carries its joint covariance.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::transform_mean_pair`].
///
/// # Errors
/// `invalid_argument`: no scale requested, a non-finite mean, a mean outside a scale's
/// domain (`descriptive_comparison.transform_domain`), or a covariance that is not finite,
/// has a negative variance or is not positive semidefinite
/// (`descriptive_comparison.invalid_covariance`).
#[deprecated(
    since = "3.1.0",
    note = "compile the computation with antecedent::analysis::descriptive_programs::compile_descriptive(DescriptiveRequest::TransformMeanPair { .. }, ctx)?.evaluate(ctx), which calls this kernel with the same inputs and returns its result unchanged"
)]
pub fn transform_mean_pair(
    pair: &antecedent_estimate::MeanPair,
    scales: &[antecedent_estimate::ReportingScale],
) -> Result<antecedent_estimate::ReportingTransform, antecedent_estimate::EstimationError> {
    antecedent_estimate::transform_mean_pair(pair, scales)
}

/// The unadjusted contrast `mean(Y | T = 1) - mean(Y | T = 0)` of complete rows.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::raw_contrast`].
///
/// # Errors
/// `invalid_argument` (`descriptive_comparison.invalid_data`) for empty, unequal-length,
/// non-finite or non-0/1 treatment input; `arm_not_populated`
/// (`descriptive_comparison.arm_not_populated`) when an arm has no rows;
/// `cancelled_no_claim` (`descriptive_comparison.cancelled`) when `ctx` is cancelled (observed
/// at the first row, every 4096 rows and once after the pass), with no contrast reported.
#[deprecated(
    since = "3.1.0",
    note = "compile the computation with antecedent::analysis::descriptive_programs::compile_descriptive(DescriptiveRequest::RawContrast { .. }, ctx)?.evaluate(ctx), which calls this kernel with the same inputs and returns its result unchanged"
)]
pub fn raw_contrast(
    outcome: &[f64],
    treatment: &[f64],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_estimate::RawContrast, antecedent_estimate::EstimationError> {
    antecedent_estimate::raw_contrast(outcome, treatment, ctx)
}

/// Fit factorized joint-cell scores with caller-supplied nuisance factories.
///
/// Factories receive only selected rows and must declare the required regression or
/// binary-probability task. Their provenance is marked unverified; fit failures stop
/// the operation instead of publishing a partial result. The existing per-cell support
/// checks and point-only inference policy remain in force.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::fit_factorized_joint_cells_with_factories`].
///
/// # Errors
/// Invalid declarations, incompatible factories, invalid predictions, fit failure,
/// cancellation, or a support or resource refusal.
#[deprecated(
    since = "3.1.0",
    note = "declare the learners with StudyBuilder::learner_factories and use Study::compile_joint_cells_learners(..)?.execute(ctx) (route JointCellsLearnerFit), which calls this kernel function with the same inputs"
)]
#[allow(clippy::too_many_arguments, reason = "the signature is the kernel function's, unchanged")]
#[doc(hidden)]
pub fn fit_factorized_joint_cells_with_factories(
    data: &antecedent_data::dataset::TabularData,
    treatments: &[antecedent_core::VariableId],
    outcome: antecedent_core::VariableId,
    adjustment: &[antecedent_core::VariableId],
    orderings: &[Vec<usize>],
    config: &antecedent_estimate::FactorizedJointConfig,
    ctx: &antecedent_core::ExecutionContext,
    outcome_factory: &dyn antecedent_learn::LearnerFactory,
    propensity_factory: &dyn antecedent_learn::LearnerFactory,
) -> Result<antecedent_estimate::FactorizedJointFit, antecedent_estimate::EstimationError> {
    antecedent_estimate::fit_factorized_joint_cells_with_factories(
        data,
        treatments,
        outcome,
        adjustment,
        orderings,
        config,
        ctx,
        outcome_factory,
        propensity_factory,
    )
}

/// Fit a certified trial-to-target AIPW point with caller-supplied nuisance factories.
///
/// Each factory receives only its selected training and prediction rows. Provider
/// validity and prior training overlap are not verified; the result records that
/// boundary. External bootstrap refits are not implemented and are explicitly refused.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::estimate_trial_aipw_with_factories`].
///
/// # Errors
/// Invalid inputs, unsupported inference, incompatible factories, failed fit, or cancellation.
#[deprecated(
    since = "3.1.0",
    note = "declare the learners with StudyBuilder::learner_factories and use Study::compile_trial_aipw_learners(..)?.execute(ctx) (route TrialAipwLearnerTransport), which calls this kernel function with the same inputs"
)]
pub fn estimate_trial_aipw_with_factories(
    id: &antecedent_identify::transport::TransportIdentification,
    input: &antecedent_estimate::TrialAipwInput,
    options: &antecedent_estimate::TrialAipwOptions,
    ctx: &antecedent_core::ExecutionContext,
    outcome: &dyn antecedent_learn::LearnerFactory,
    membership: &dyn antecedent_learn::LearnerFactory,
) -> Result<antecedent_estimate::TrialAipwEstimate, antecedent_estimate::EstimationError> {
    antecedent_estimate::estimate_trial_aipw_with_factories(
        id, input, options, ctx, outcome, membership,
    )
}

/// Execute a smoothed-dose point with caller factories that export portable predictors.
///
/// Each fit is row-isolated and replay uses the portable predictors. Caller factories
/// remain unverified; time-ordered data is refused because this route's folds train on
/// the complement of each held-out fold. Bootstrap refits are not provided here.
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::estimate_smoothed_dose_with_factories`].
///
/// # Errors
/// A validation, capability, portability, time-order, budget, fit or overlap refusal.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled SmoothedDoseLearnerTransport route, which calls this kernel function with the same inputs"
)]
#[doc(hidden)]
pub fn estimate_smoothed_dose_with_factories(
    id: &antecedent_identify::transport::TransportIdentification,
    query: &antecedent_core::SmoothedDoseTransportQuery,
    input: &antecedent_estimate::SmoothedDoseInput,
    options: &antecedent_estimate::SmoothedDoseOptions,
    ctx: &antecedent_core::ExecutionContext,
    outcome: &dyn antecedent_learn::learner::LearnerFactory,
    membership: &dyn antecedent_learn::learner::LearnerFactory,
) -> Result<antecedent_estimate::SmoothedDoseEstimate, antecedent_estimate::EstimationError> {
    antecedent_estimate::estimate_smoothed_dose_with_factories(
        id, query, input, options, ctx, outcome, membership,
    )
}

/// Estimate one contrast under complete high/low cluster allocation followed by
/// independent Bernoulli assignment within each cluster. Exposure probabilities
/// marginalize the high/low allocation of each unit's own cluster and enumerate
/// its incoming-neighbor assignments. Other cluster allocations cancel because
/// partial interference forbids cross-cluster edges.
/// The legacy covariance-free variance remains descriptive. An independent-cluster
/// pointwise interval is supplied separately when both saturation arms and both
/// exposure levels pass their support gates.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::estimate_saturation_interference`].
///
/// # Errors
///
/// Refuses invalid randomization, cross-cluster edges, absent realized exposures,
/// nonpositive exposure probabilities, or neighborhoods above enumeration caps.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled InterferenceDag route, which calls this kernel function with the same inputs"
)]
pub fn estimate_saturation_interference(
    query: &antecedent_core::InterferenceQuery,
    data: &antecedent_data::network::NetworkData,
    assignment: &[bool],
) -> Result<antecedent_estimate::SaturationInterferenceEstimate, antecedent_estimate::EstimationError>
{
    antecedent_estimate::estimate_saturation_interference(query, data, assignment)
}

/// Estimate the whole recovered law and effect from observation-pattern rows and
/// form the requested whole-method interval.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::estimate_sampled_recovery`].
///
/// # Errors
///
/// `sampled_recovery.unrecoverable_pattern` for a zero complete-case cell (the
/// offending patterns are retained) or a refused exact recovery,
/// `sampled_recovery.too_many_failed_replicates`,
/// `sampled_recovery.bounds_exceeded`, `sampled_recovery.invalid_input` and
/// `sampled_recovery.cancelled`.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled SampledObservationRecovery program family, which runs the same kernel call after its unit's support gate"
)]
pub fn estimate_sampled_recovery(
    derivation: &antecedent_identify::recovery::RecoveryDerivation,
    input: &antecedent_estimate::SampledObservationInput,
    config: &antecedent_estimate::SampledRecoveryConfig,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_estimate::SampledRecoveryResult, antecedent_estimate::SampledRecoveryError> {
    antecedent_estimate::estimate_sampled_recovery(derivation, input, config, ctx)
}

/// Estimate an observational exposure contrast with supplied marginal probabilities.
/// Identification requires a caller-declared network exchangeability assumption;
/// this kernel checks topology and positivity but cannot verify exchangeability.
/// A pointwise cluster interval is returned only for known probabilities with
/// sufficient independent and exposed clusters. Externally fitted probabilities
/// retain a descriptive variance because their fitting uncertainty is unknown.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::estimate_observational_exposure`].
///
/// # Errors
///
/// Refuses invalid dimensions, exposure levels, propensities, topology, empirical
/// support, or fewer than two partial-interference clusters.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled InterferenceDag route, which calls this kernel function with the same inputs"
)]
pub fn estimate_observational_exposure(
    data: &antecedent_data::network::NetworkData,
    outcome: antecedent_core::ids::VariableId,
    spec: &antecedent_estimate::ObservationalExposureSpec<'_>,
) -> Result<antecedent_estimate::ObservationalExposureEstimate, antecedent_estimate::EstimationError>
{
    antecedent_estimate::estimate_observational_exposure(data, outcome, spec)
}

/// Run the direct estimator with the caller's memory, work and cancellation policy.
/// The full requested Monte Carlo draw count is used or the operation is refused.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::estimate_interference_with_context`].
///
/// # Errors
///
/// The error of [`antecedent_estimate::estimate_interference_with_context`], unchanged.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled InterferenceDag route, which calls this kernel function with the same inputs"
)]
pub fn estimate_interference_with_context(
    query: &antecedent_core::InterferenceQuery,
    data: &antecedent_data::network::NetworkData,
    assignment: &[bool],
    seed: u64,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_estimate::InterferenceEstimate, antecedent_estimate::EstimationError> {
    antecedent_estimate::estimate_interference_with_context(query, data, assignment, seed, ctx)
}

/// The conditional maximum likelihood odds ratio of matched case-control sets.
///
/// `stratum`, `case` (1 = case, 0 = control) and `exposed` (1 = exposed) are parallel
/// rows. The score is monotone, so the root is bracketed and found by a safeguarded
/// Newton iteration (the information supplies the step, bisection the guarantee);
/// `ctx` is observed on entry and once per iteration.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::conditional_odds_ratio`].
///
/// # Errors
/// `invalid_argument` (`matched_case_control.invalid_data`) for empty or unequal
/// inputs or a value other than exactly 0 or 1; `effect_not_identified`
/// (`matched_case_control.no_informative_sets`) when every set is concordant or
/// outcome-degenerate; `route_not_supported`
/// (`matched_case_control.estimate_not_finite`) when the conditional likelihood has no
/// finite maximizer; `transport_budget_cancel` (`matched_case_control.budget`) when
/// cancelled.
#[deprecated(
    since = "3.1.0",
    note = "compile the computation with antecedent::analysis::descriptive_programs::compile_descriptive(DescriptiveRequest::MatchedOddsRatio { .. }, ctx)?.evaluate(ctx), which calls this kernel with the same inputs and returns its result unchanged"
)]
pub fn conditional_odds_ratio(
    stratum: &[String],
    case: &[f64],
    exposed: &[f64],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_estimate::ConditionalOddsRatio, antecedent_estimate::EstimationError> {
    antecedent_estimate::conditional_odds_ratio(stratum, case, exposed, ctx)
}

/// Set the unadjusted contrast beside an adjusted estimate of the same estimand coding.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_estimate::compare_raw_adjusted`].
///
/// # Errors
/// `route_not_supported` (`descriptive_comparison.estimand_coding_mismatch`) when the
/// adjusted estimate is not a 1-versus-0 difference of means; `population_not_estimable`
/// (`descriptive_comparison.population_not_all_observed`) when it targets another
/// population; `invalid_argument` for a non-finite estimate or standard error; and the
/// `raw_contrast` errors (cancellation included).
#[deprecated(
    since = "3.1.0",
    note = "compile the computation with antecedent::analysis::descriptive_programs::compile_descriptive(DescriptiveRequest::CompareRawAdjusted { .. }, ctx)?.evaluate(ctx), which calls this kernel with the same inputs and returns its result unchanged"
)]
pub fn compare_raw_adjusted(
    outcome: &[f64],
    treatment: &[f64],
    adjusted: antecedent_estimate::AdjustedEstimate,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_estimate::DescriptiveComparison, antecedent_estimate::EstimationError> {
    antecedent_estimate::compare_raw_adjusted(outcome, treatment, adjusted, ctx)
}

/// Deterministic adaptive resampling in waves, with cancellation and failure accounting.
pub mod adaptive_resample_waves {
    pub use antecedent_estimate::adaptive_resample_waves::StopReason;
}
/// Retained adjusted regression.
pub mod adjustment_resume {
    pub use antecedent_estimate::adjustment_resume::count_adjusted_model_fits;
}
/// Independent-unit randomized ANCOVA point and HC0 sandwich variance.
pub mod ancova {
    /// Fit OLS with intercept, assignment, and pre-assignment covariates.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::ancova::fit_ancova`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::ancova::fit_ancova`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled RandomizedEffect route, which calls this kernel function with the same inputs"
    )]
    pub fn fit_ancova(
        outcome: &[f64],
        assignment: &[bool],
        covariates: &[&[f64]],
    ) -> Result<antecedent_estimate::ancova::AncovaFit, &'static str> {
        antecedent_estimate::ancova::fit_ancova(outcome, assignment, covariates)
    }
}
/// Categorical treatments: ordered and unordered regimes.
pub mod categorical_treatment {
    pub use antecedent_estimate::categorical_treatment::CategoricalTreatmentSpec;
}
/// Sampled inference for general counterfactual transport.
pub mod counterfactual_transport_sampled {
    pub use antecedent_estimate::counterfactual_transport_sampled::{
        DatasetOverlap, OverlapDeclaration, SampledAxis, SampledCounterfactualTransportEstimate,
        SampledIntervalOutcome, SampledRegimeCounts, SampledRegimeCountsParts, SampledTableKey,
        SampledTransportLimits, SampledTransportRequest, SamplingDesign, UnseenCellRule,
    };
}
/// Preparation and evaluation of the DAG completions of a supplied CPDAG.
pub mod cpdag_scenarios {
    pub use antecedent_estimate::cpdag_scenarios::CpdagScenarioReport;
}
/// Items of `antecedent-estimate::cyclic_likelihood`.
pub mod cyclic_likelihood {
    /// Sample a zero-mean independent normal parameter prior, restricted to the
    /// template's certified-bijection likelihood domain. `prior_sd` is fixed before fitting.
    /// Structural selection and covariance uncertainty are not included.
    ///
    /// Each chain starts at the template's values plus an independent Gaussian offset of the
    /// declared proposal scale, so the between-chain diagnostics compare distinct starts. The
    /// declared scale is the starting step: warmup tunes the step and the proposal covariance
    /// (`antecedent_prob::metropolis::sample_metropolis_tuned`) and freezes them before the first
    /// retained draw. A fixed isotropic step tuned for one posterior width mixes slowly once more
    /// rows narrow the posterior or its coordinates correlate, and the R-hat gate then refuses.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::cyclic_likelihood::sample_cyclic_posterior`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::cyclic_likelihood::sample_cyclic_posterior`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled CyclicFamily route, which calls this kernel function with the same inputs"
    )]
    pub fn sample_cyclic_posterior(
        template: &antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        parameters: &[antecedent_estimate::cyclic_likelihood::CyclicParameter],
        rows: &[antecedent_model::nonlinear_equilibrium::RegimeRow],
        prior_sd: f64,
        options: antecedent_prob::metropolis::MetropolisOptions,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_estimate::cyclic_likelihood::CyclicPosterior,
        antecedent_model::nonlinear_equilibrium::EquilibriumError,
    > {
        antecedent_estimate::cyclic_likelihood::sample_cyclic_posterior(
            template, parameters, rows, prior_sd, options, ctx,
        )
    }
}
/// Conjugate Gaussian regression of an outcome on a fixed dose basis.
pub mod dose_basis_regression {
    pub use antecedent_estimate::dose_basis_regression::ResidualPrior;
}
/// Items of `antecedent-estimate::experimental_fit`.
pub mod experimental_fit {
    pub use antecedent_estimate::experimental_fit::ExperimentalFitError;
}
/// Items of `antecedent-estimate::external_cyclic_fit`.
pub mod external_cyclic_fit {
    pub use antecedent_estimate::external_cyclic_fit::ExternalFitPolicy;

    /// Fit the declared scalars of an external-capable feedback system from known-regime rows.
    /// Fixed covariance, graph and the declared mechanism family are conditional assumptions; every
    /// proposed system must carry a declared contraction for the likelihood to be a density.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::external_cyclic_fit::fit_external_feedback`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::external_cyclic_fit::fit_external_feedback`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled ExternalFeedbackFit route, which calls this kernel function with the same inputs"
    )]
    pub fn fit_external_feedback(
        template: &antecedent_model::external_mechanism::ExternalFeedbackSystem,
        resolver: Option<&dyn antecedent_model::external_mechanism::TrustedMechanismResolver>,
        parameters: &[antecedent_model::external_mechanism::ExternalFitParameter],
        rows: &[antecedent_model::nonlinear_equilibrium::RegimeRow],
        policy: antecedent_estimate::external_cyclic_fit::ExternalFitPolicy,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_estimate::external_cyclic_fit::ExternalFeedbackFit,
        antecedent_model::nonlinear_equilibrium::EquilibriumError,
    > {
        antecedent_estimate::external_cyclic_fit::fit_external_feedback(
            template, resolver, parameters, rows, policy, ctx,
        )
    }
}
/// Independent-Bernoulli 2×2 factorial Horvitz–Thompson contrasts.
pub mod factorial_bernoulli {
    /// Finite-row cell means and factorial contrasts for two independent Bernoulli factors.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::factorial_bernoulli::estimate_factorial_2x2`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::factorial_bernoulli::estimate_factorial_2x2`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled RandomizedEffect route, which calls this kernel function with the same inputs"
    )]
    pub fn estimate_factorial_2x2(
        outcome: &[f64],
        factor_a: &[bool],
        factor_b: &[bool],
        probability_a: &[f64],
        probability_b: &[f64],
    ) -> Result<antecedent_estimate::factorial_bernoulli::BernoulliFactorial2x2, &'static str> {
        antecedent_estimate::factorial_bernoulli::estimate_factorial_2x2(
            outcome,
            factor_a,
            factor_b,
            probability_a,
            probability_b,
        )
    }
}
/// Scoped instrumentation of factories resolved through the public learner specification.
pub mod fit_counts {
    pub use antecedent_learn::fit_counts::count_resolved_fits;
}
/// Joint Bayesian source-target transport.
pub mod joint_bayesian_transport {
    pub use antecedent_estimate::joint_bayesian_transport::{
        DataIdentity, GaussianPrior, JointPriors, JointTransportModel, JointTransportOptions,
        PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData,
        TransportGraphClass, VaryingBlock,
    };
}
/// Design-native joint response functionals.
pub mod joint_design {
    /// Finite-history adaptive regimes evaluated against treatment-affected state transitions.
    pub mod adaptive_regime {
        pub use antecedent_estimate::joint_design::adaptive_regime::JointAdaptiveRegimeRecipe;
    }
    /// Compiler-owned joint cluster randomization with declared preassignment adjustment.
    pub mod cluster_decision {
        pub use antecedent_estimate::joint_design::cluster_decision::JointClusterRecipe;
    }
    /// Versioned executable design proofs retaining identifying functionals and source data.
    pub mod compiled {
        pub use antecedent_estimate::joint_design::compiled::{
            JointDesignInference, JointDesignRecipe,
        };
    }
    /// Whole-program randomization over independently assigned units or clusters.
    pub mod randomized {
        pub use antecedent_estimate::joint_design::randomized::{
            JointRandomizedDesign, JointRandomizedUnit,
        };
    }
}
/// Finite-lag multichannel nonlinear Gaussian panel mechanisms: the items of
/// `antecedent_estimate::joint_panel`, whose run entries are deprecated wrappers of the kernel
/// functions in favour of the compiled joint-panel routes.
pub mod joint_panel {
    pub use antecedent_estimate::joint_panel::{
        PanelAdstock, PanelAlignedResponses, PanelChannel, PanelChannelHistory,
        PanelChannelParameters, PanelChannelProgram, PanelData, PanelDraw, PanelError, PanelFamily,
        PanelFamilyParameters, PanelFamilyPrior, PanelFeatureInputs, PanelFit, PanelFitDiagnostics,
        PanelFitOptions, PanelGroup, PanelInference, PanelInteractionParameter,
        PanelInteractionSpec, PanelLawResponses, PanelOptimizerOptions, PanelParameters,
        PanelPrior, PanelProgram, PanelProjectedLaw, PanelResponse, PanelSamplingOptions,
        PanelSaturation, PanelSaturationScope, PanelScenario, PanelSpatial, PanelSpatialKernel,
        PanelStreamReceipt, expression, family, validate_fit_options, validate_panel_programs,
    };

    /// Aligned four-component reference-world decompositions of panel responses.
    pub mod decomposition {
        pub use antecedent_estimate::joint_panel::decomposition::{
            ChangeComponent, DecompositionEndpoint, DecompositionJointUncertainty,
            MechanismPeriodMap, PairedPanelPath, PanelDecomposition, PanelDecompositionRequest,
            ReferenceDecomposition, ReferenceDecompositionCovariance, StreamedDecomposition,
        };

        /// Decompose a compiled joint response law whose sixteen coordinates have already
        /// passed the owning scientific compiler's component and world checks.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::decomposition::decompose_hybrid_law`].
        ///
        /// # Errors
        /// Invalid responses/order, resources or cancellation.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointDesign route: JointDesignResult::decompose_reference_components (bootstrap design law), or JointPanelResult::decompose_compiled_components"
        )]
        pub fn decompose_hybrid_law(
            point: [f64; 16],
            draws: &[Option<[f64; 16]>],
            order: [antecedent_estimate::joint_panel::decomposition::ChangeComponent; 4],
            ctx: &antecedent_core::ExecutionContext,
        ) -> Result<
            antecedent_estimate::joint_panel::decomposition::PanelDecomposition,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::decomposition::decompose_hybrid_law(
                point, draws, order, ctx,
            )
        }

        /// Decompose one retained stationary law without inventing a paired independent fit.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::decomposition::decompose_invariant_law`].
        ///
        /// # Errors
        /// Invalid period maps, inputs, numerical failure, resources or cancellation.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelResponse route: JointPanelResult::decompose_reference"
        )]
        pub fn decompose_invariant_law(
            data: &antecedent_estimate::joint_panel::PanelData,
            request: &antecedent_estimate::joint_panel::decomposition::PanelDecompositionRequest,
            periods: &antecedent_estimate::joint_panel::decomposition::MechanismPeriodMap,
            point: &antecedent_estimate::joint_panel::PanelDraw,
            draws: &[Option<antecedent_estimate::joint_panel::PanelDraw>],
            ctx: &antecedent_core::ExecutionContext,
        ) -> Result<
            antecedent_estimate::joint_panel::decomposition::PanelDecomposition,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::decomposition::decompose_invariant_law(
                data, request, periods, point, draws, ctx,
            )
        }

        /// Streamed `decompose_invariant_law`: each draw's paired path is formed, evaluated, folded
        /// into the joint moments and dropped, so neither the paired paths nor the per-draw
        /// decompositions of the whole law are ever held. Moments equal the retained route's.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::decomposition::decompose_invariant_law_streamed`].
        ///
        /// # Errors
        /// Invalid period maps, inputs, numerical failure, resources or cancellation.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelResponse route: JointPanelResult::decompose_reference_streamed"
        )]
        pub fn decompose_invariant_law_streamed(
            data: &antecedent_estimate::joint_panel::PanelData,
            request: &antecedent_estimate::joint_panel::decomposition::PanelDecompositionRequest,
            periods: &antecedent_estimate::joint_panel::decomposition::MechanismPeriodMap,
            point: &antecedent_estimate::joint_panel::PanelDraw,
            draws: &[Option<antecedent_estimate::joint_panel::PanelDraw>],
            ctx: &antecedent_core::ExecutionContext,
        ) -> Result<
            antecedent_estimate::joint_panel::decomposition::StreamedDecomposition,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::decomposition::decompose_invariant_law_streamed(
                data, request, periods, point, draws, ctx,
            )
        }

        /// Evaluate all reference worlds using one compiled interaction plan, preserving
        /// paired parameter/path uncertainty across worlds and components.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::decomposition::decompose_panel_law`].
        ///
        /// # Errors
        /// Refuses mismatched horizons/groups, invalid paths, weights or orders, numerical
        /// overflow, cancellation and exhausted resources. Scientific identification must
        /// be checked by the compiler before these numbers are consumed causally.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelResponse route: JointPanelResult::decompose_reference (the route pairs the fitted law through MechanismPeriodMap; a caller-built parameter path that no fit produced has no compiled form)"
        )]
        pub fn decompose_panel_law(
            data: &antecedent_estimate::joint_panel::PanelData,
            request: &antecedent_estimate::joint_panel::decomposition::PanelDecompositionRequest,
            point: &antecedent_estimate::joint_panel::decomposition::PairedPanelPath,
            draws: &[Option<antecedent_estimate::joint_panel::decomposition::PairedPanelPath>],
            ctx: &antecedent_core::ExecutionContext,
        ) -> Result<
            antecedent_estimate::joint_panel::decomposition::PanelDecomposition,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::decomposition::decompose_panel_law(
                data, request, point, draws, ctx,
            )
        }

        /// Decompose one retained temporal law, preserving model/path/selection slots.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::decomposition::decompose_temporal_law`].
        ///
        /// # Errors
        /// Invalid period maps, inputs, numerical failure, resources or cancellation.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelTimeVarying route: JointPanelResult::decompose_reference on the time-varying route"
        )]
        pub fn decompose_temporal_law(
            data: &antecedent_estimate::joint_panel::PanelData,
            request: &antecedent_estimate::joint_panel::decomposition::PanelDecompositionRequest,
            periods: &antecedent_estimate::joint_panel::decomposition::MechanismPeriodMap,
            point: &antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw,
            draws: &[Option<antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw>],
            ctx: &antecedent_core::ExecutionContext,
        ) -> Result<
            antecedent_estimate::joint_panel::decomposition::PanelDecomposition,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::decomposition::decompose_temporal_law(
                data, request, periods, point, draws, ctx,
            )
        }

        /// Streamed `decompose_temporal_law`, keeping selection and path coupling per draw.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::decomposition::decompose_temporal_law_streamed`].
        ///
        /// # Errors
        /// Invalid period maps, inputs, numerical failure, resources or cancellation.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelTimeVarying route: JointPanelResult::decompose_reference_streamed on the time-varying route"
        )]
        pub fn decompose_temporal_law_streamed(
            data: &antecedent_estimate::joint_panel::PanelData,
            request: &antecedent_estimate::joint_panel::decomposition::PanelDecompositionRequest,
            periods: &antecedent_estimate::joint_panel::decomposition::MechanismPeriodMap,
            point: &antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw,
            draws: &[Option<antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw>],
            ctx: &antecedent_core::ExecutionContext,
        ) -> Result<
            antecedent_estimate::joint_panel::decomposition::StreamedDecomposition,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::decomposition::decompose_temporal_law_streamed(
                data, request, periods, point, draws, ctx,
            )
        }

        /// Exact four-factor Shapley/sequential decomposition of a complete hybrid table.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::decomposition::decompose_values`].
        ///
        /// # Errors
        /// Nonfinite hybrid values, invalid order or arithmetic overflow.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelResponse route: JointPanelResult::decompose_reference or decompose_compiled_components (each ReferenceDecomposition is decompose_values of its hybrids)"
        )]
        pub fn decompose_values(
            hybrids: [f64; 16],
            order: [antecedent_estimate::joint_panel::decomposition::ChangeComponent; 4],
        ) -> Result<
            antecedent_estimate::joint_panel::decomposition::ReferenceDecomposition,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::decomposition::decompose_values(hybrids, order)
        }

        /// Preserve explicit weights from a compiled finite joint law while decomposing it.
        /// Zero-weight failed slots may be retained; positive-weight failures refuse.
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::decomposition::decompose_weighted_hybrid_law`].
        ///
        /// # Errors
        /// Invalid weights, responses/order, resources or cancellation.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelResponse route: JointPanelResult::decompose_compiled_components"
        )]
        pub fn decompose_weighted_hybrid_law(
            point: [f64; 16],
            draws: &[Option<[f64; 16]>],
            weights: Option<&[f64]>,
            order: [antecedent_estimate::joint_panel::decomposition::ChangeComponent; 4],
            ctx: &antecedent_core::ExecutionContext,
        ) -> Result<
            antecedent_estimate::joint_panel::decomposition::PanelDecomposition,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::decomposition::decompose_weighted_hybrid_law(
                point, draws, weights, order, ctx,
            )
        }

        /// Propagate a supplied *joint* hybrid covariance, using exact linear decomposition maps.
        /// The caller must retain the originating method's inferential assumptions.
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::decomposition::propagate_hybrid_covariance`].
        ///
        /// # Errors
        /// Nonfinite/asymmetric covariance, invalid ordering or arithmetic overflow.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointDesign route: JointDesignResult::decompose_reference_components (posterior design law: ScientificReferenceDecomposition::covariance)"
        )]
        pub fn propagate_hybrid_covariance(
            covariance: &[[f64; 16]; 16],
            order: [antecedent_estimate::joint_panel::decomposition::ChangeComponent; 4],
        ) -> Result<
            antecedent_estimate::joint_panel::decomposition::ReferenceDecompositionCovariance,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::decomposition::propagate_hybrid_covariance(
                covariance, order,
            )
        }
    }

    /// Checks of the declared Gaussian restriction on a fitted panel mechanism.
    pub mod restriction {
        pub use antecedent_estimate::joint_panel::restriction::{
            LINEARITY_BINS, MIN_DIAGNOSTIC_INNOVATIONS, PanelLinearityCheck, PanelNormalityCheck,
            PanelRestrictionDiagnostics,
        };

        /// Check the declared Gaussian restriction on a fitted point mechanism.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::restriction::panel_restriction_diagnostics`].
        ///
        /// # Errors
        /// A non-positive or non-finite level, a mechanism that does not match the panel, or a fitted
        /// parameter outside its domain.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelResponse route: PreparedStudy::<JointPanelState>::estimate with declare_joint_panel_gaussian_restriction on the target; the checks are reported on the gaussian-restriction design attempt"
        )]
        pub fn panel_restriction_diagnostics(
            data: &antecedent_estimate::joint_panel::PanelData,
            point: &antecedent_estimate::joint_panel::PanelDraw,
            level: f64,
        ) -> Result<
            antecedent_estimate::joint_panel::restriction::PanelRestrictionDiagnostics,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::restriction::panel_restriction_diagnostics(
                data, point, level,
            )
        }
    }

    /// Change-point, smoothing and selection uncertainty carried into a reference decomposition.
    pub mod selection_uncertainty {
        pub use antecedent_estimate::joint_panel::selection_uncertainty::{
            SelectionDecomposition, SpecificationShare,
        };

        /// Split a temporal decomposition by the specification each aligned draw selected.
        ///
        /// `draws` must be the exact aligned law the decomposition was computed from.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::selection_uncertainty::decomposition_selection`].
        ///
        /// # Errors
        /// Misaligned axes, a decomposed slot whose law slot failed, or no complete slot.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelTimeVarying route: JointPanelResult::decomposition_selection on the time-varying route"
        )]
        pub fn decomposition_selection(
            decomposition: &antecedent_estimate::joint_panel::decomposition::PanelDecomposition,
            point: &antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw,
            draws: &[Option<antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw>],
        ) -> Result<
            antecedent_estimate::joint_panel::selection_uncertainty::SelectionDecomposition,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::selection_uncertainty::decomposition_selection(
                decomposition,
                point,
                draws,
            )
        }
    }

    /// Explicit model-conditional piecewise and smooth attempts on the same panel likelihood.
    pub mod time_varying {
        pub use antecedent_estimate::joint_panel::time_varying::{
            CoarserPanelContrast, PanelExplanation, PanelMeanParameter, TemporalModelChoice,
            TemporalPanelAttempt, TemporalPanelDraw, TemporalPanelFit, TemporalPanelSpec,
            require_coarser_contrast, validate_temporal_spec,
        };

        /// Evaluate every temporal mechanism draw using one immutable compiled expression plan.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::time_varying::evaluate_temporal_law`].
        ///
        /// # Errors
        /// Unsupported extrapolation, invalid axes, point numerical failure, resources or cancellation.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelTimeVarying route: PreparedStudy::<JointPanelState>::estimate with JointPanelRequest::temporal declared"
        )]
        pub fn evaluate_temporal_law(
            data: &antecedent_estimate::joint_panel::PanelData,
            point: &antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw,
            draws: &[Option<antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw>],
            programs: &[antecedent_estimate::joint_panel::PanelProgram],
            ctx: &antecedent_core::ExecutionContext,
        ) -> Result<
            antecedent_estimate::joint_panel::PanelLawResponses,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::time_varying::evaluate_temporal_law(
                data, point, draws, programs, ctx,
            )
        }

        /// Evaluate the same path/random-effect realization in every declared reference world.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::time_varying::evaluate_temporal_programs`].
        ///
        /// # Errors
        /// Refuses implicit extrapolation outside the retained path, scenario mismatch, cancellation and resource failure.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelTimeVarying route: JointPanelResult::point of the time-varying route"
        )]
        pub fn evaluate_temporal_programs(
            data: &antecedent_estimate::joint_panel::PanelData,
            draw: &antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw,
            programs: &[antecedent_estimate::joint_panel::PanelProgram],
            ctx: &antecedent_core::ExecutionContext,
        ) -> Result<
            Vec<antecedent_estimate::joint_panel::PanelResponse>,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::time_varying::evaluate_temporal_programs(
                data, draw, programs, ctx,
            )
        }

        /// Fit both named specification attempts and retain whole-procedure uncertainty.
        ///
        /// Bayesian priors cover the prespecified union of no-change, piecewise and smooth models
        /// before observing data. The numerical point's weak-change rule does not truncate that prior.
        /// Frequentist bootstrap replicates repeat the point's complete selection/fallback procedure.
        ///
        /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
        /// names. This forwards to [`antecedent_estimate::joint_panel::time_varying::fit_temporal_panel`].
        ///
        /// # Errors
        /// Refuses malformed specifications, numerical failure of every required attempt, missing
        /// resource admission or cancellation. No causal-change identification is inferred here.
        #[deprecated(
            since = "3.1.0",
            note = "run the compiled JointPanelTimeVarying route: PreparedStudy::<JointPanelState>::estimate with JointPanelRequest::temporal declared"
        )]
        pub fn fit_temporal_panel(
            data: &antecedent_estimate::joint_panel::PanelData,
            spec: &antecedent_estimate::joint_panel::time_varying::TemporalPanelSpec,
            options: &antecedent_estimate::joint_panel::PanelFitOptions,
            ctx: &antecedent_core::ExecutionContext,
        ) -> Result<
            antecedent_estimate::joint_panel::time_varying::TemporalPanelFit,
            antecedent_estimate::joint_panel::PanelError,
        > {
            antecedent_estimate::joint_panel::time_varying::fit_temporal_panel(
                data, spec, options, ctx,
            )
        }
    }

    /// Evaluate a complete fitted law with one shared compiled expression plan.
    ///
    /// This numerical component preserves the caller's world bindings and failed slots;
    /// scientific authorization remains with the checked compiler that supplied the law.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::joint_panel::evaluate_panel_law`].
    ///
    /// # Errors
    /// Input mismatch, point numerical failure, shared resource refusal or cancellation.
    #[deprecated(
        since = "3.1.0",
        note = "run the compiled JointPanelResponse route: PreparedStudy::<JointPanelState>::estimate, or JointPanelResult::retarget_frozen for new worlds on the same fit"
    )]
    pub fn evaluate_panel_law(
        data: &antecedent_estimate::joint_panel::PanelData,
        point: &antecedent_estimate::joint_panel::PanelDraw,
        draws: &[Option<antecedent_estimate::joint_panel::PanelDraw>],
        programs: &[antecedent_estimate::joint_panel::PanelProgram],
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_estimate::joint_panel::PanelLawResponses,
        antecedent_estimate::joint_panel::PanelError,
    > {
        antecedent_estimate::joint_panel::evaluate_panel_law(data, point, draws, programs, ctx)
    }

    /// Evaluate all declared programs from one complete mechanism/random-effect draw.
    ///
    /// No fitting, identification or implicit future-covariate/initial-law selection occurs.
    /// This returns model-conditional mean response, never predictive innovations or support.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::joint_panel::evaluate_panel_programs`].
    ///
    /// # Errors
    /// Refuses unknown/missing group scenarios, changed parameter dimensions, malformed future
    /// covariates/history, cancellation and failed shared admission.
    #[deprecated(
        since = "3.1.0",
        note = "run the compiled JointPanelResponse route: JointPanelResult::point of PreparedStudy::<JointPanelState>::estimate"
    )]
    pub fn evaluate_panel_programs(
        data: &antecedent_estimate::joint_panel::PanelData,
        draw: &antecedent_estimate::joint_panel::PanelDraw,
        programs: &[antecedent_estimate::joint_panel::PanelProgram],
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        Vec<antecedent_estimate::joint_panel::PanelResponse>,
        antecedent_estimate::joint_panel::PanelError,
    > {
        antecedent_estimate::joint_panel::evaluate_panel_programs(data, draw, programs, ctx)
    }

    /// Fit this numerical mechanism under caller-owned checked identification/target policy.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::joint_panel::fit_panel`].
    ///
    /// # Errors
    /// Explicit input, numerical, shared resource or cancellation failures. Failed optimizer/
    /// convergence diagnostics remain in the fit; they never become a supported uncertainty claim.
    #[deprecated(
        since = "3.1.0",
        note = "run the compiled JointPanelResponse route: PreparedStudy::<JointPanelState>::estimate (Study::prepare_joint_panel without a temporal declaration)"
    )]
    pub fn fit_panel(
        data: &antecedent_estimate::joint_panel::PanelData,
        options: &antecedent_estimate::joint_panel::PanelFitOptions,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_estimate::joint_panel::PanelFit,
        antecedent_estimate::joint_panel::PanelError,
    > {
        antecedent_estimate::joint_panel::fit_panel(data, options, ctx)
    }

    /// Reduce a stationary fitted law to one linear functional per world without ever holding
    /// more than one tile of full response rows. Values equal projecting the full-output rows.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::joint_panel::project_panel_law`].
    ///
    /// # Errors
    /// The refusals of `stream_panel_law`, plus weights that do not cover each horizon.
    #[deprecated(
        since = "3.1.0",
        note = "run the compiled JointPanelResponse route: JointPanelResult::project_streamed, or PreparedStudy::<JointPanelState>::estimate_projected"
    )]
    pub fn project_panel_law(
        data: &antecedent_estimate::joint_panel::PanelData,
        point: &antecedent_estimate::joint_panel::PanelDraw,
        draws: &[Option<antecedent_estimate::joint_panel::PanelDraw>],
        programs: &[antecedent_estimate::joint_panel::PanelProgram],
        weights: &[Vec<f64>],
        tile_draws: usize,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_estimate::joint_panel::PanelProjectedLaw,
        antecedent_estimate::joint_panel::PanelError,
    > {
        antecedent_estimate::joint_panel::project_panel_law(
            data, point, draws, programs, weights, tile_draws, ctx,
        )
    }

    /// Apply already compiled linear response reductions in the retained world order.
    ///
    /// This numerical reduction carries no support, identification or uncertainty support.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::joint_panel::project_panel_responses`].
    ///
    /// # Errors
    /// Refuses missing world weights, mismatched horizons, nonfinite inputs or arithmetic overflow.
    #[deprecated(
        since = "3.1.0",
        note = "run the compiled JointPanelResponse route: JointPanelResult::project_streamed (point and draws), or JointPanelResult::numerical_project_mean_source"
    )]
    pub fn project_panel_responses(
        rows: &[antecedent_estimate::joint_panel::PanelResponse],
        weights: &[Vec<f64>],
    ) -> Result<Vec<f64>, antecedent_estimate::joint_panel::PanelError> {
        antecedent_estimate::joint_panel::project_panel_responses(rows, weights)
    }

    /// `project_panel_law` for a temporal mechanism path.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::joint_panel::project_temporal_law`].
    ///
    /// # Errors
    /// The refusals of `stream_temporal_law`, plus weights that do not cover each horizon.
    #[deprecated(
        since = "3.1.0",
        note = "run the compiled JointPanelTimeVarying route: JointPanelResult::project_streamed on the time-varying route"
    )]
    pub fn project_temporal_law(
        data: &antecedent_estimate::joint_panel::PanelData,
        point: &antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw,
        draws: &[Option<antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw>],
        programs: &[antecedent_estimate::joint_panel::PanelProgram],
        weights: &[Vec<f64>],
        tile_draws: usize,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_estimate::joint_panel::PanelProjectedLaw,
        antecedent_estimate::joint_panel::PanelError,
    > {
        antecedent_estimate::joint_panel::project_temporal_law(
            data, point, draws, programs, weights, tile_draws, ctx,
        )
    }

    /// Stream a stationary fitted law through `sink` in bounded tiles of `tile_draws` draws.
    ///
    /// `sink` receives the index of the tile's first draw and that tile's aligned slots; a
    /// failed slot is `None` at its original position. The tile buffer is reused, so a sink that
    /// keeps a row must clone it.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::joint_panel::stream_panel_law`].
    ///
    /// # Errors
    /// An empty tile, invalid programs or draws, resource refusal, cancellation, or any
    /// error the sink returns.
    #[deprecated(
        since = "3.1.0",
        note = "run the compiled JointPanelResponse route: JointPanelResult::point and draws of PreparedStudy::<JointPanelState>::estimate, or JointPanelResult::project_streamed for a bounded-memory reduction"
    )]
    pub fn stream_panel_law<S>(
        data: &antecedent_estimate::joint_panel::PanelData,
        point: &antecedent_estimate::joint_panel::PanelDraw,
        draws: &[Option<antecedent_estimate::joint_panel::PanelDraw>],
        programs: &[antecedent_estimate::joint_panel::PanelProgram],
        tile_draws: usize,
        ctx: &antecedent_core::ExecutionContext,
        sink: S,
    ) -> Result<
        antecedent_estimate::joint_panel::PanelStreamReceipt,
        antecedent_estimate::joint_panel::PanelError,
    >
    where
        S: FnMut(
                usize,
                &[Option<Vec<antecedent_estimate::joint_panel::PanelResponse>>],
            ) -> Result<(), antecedent_estimate::joint_panel::PanelError>
            + Send,
    {
        antecedent_estimate::joint_panel::stream_panel_law(
            data, point, draws, programs, tile_draws, ctx, sink,
        )
    }

    /// `stream_panel_law` for a temporal mechanism path, keeping selection and path coupling.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::joint_panel::stream_temporal_law`].
    ///
    /// # Errors
    /// The refusals of `stream_panel_law` plus unrequested extrapolation beyond the path.
    #[deprecated(
        since = "3.1.0",
        note = "run the compiled JointPanelTimeVarying route: JointPanelResult::point and draws of the time-varying route, or JointPanelResult::project_streamed"
    )]
    pub fn stream_temporal_law<S>(
        data: &antecedent_estimate::joint_panel::PanelData,
        point: &antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw,
        draws: &[Option<antecedent_estimate::joint_panel::time_varying::TemporalPanelDraw>],
        programs: &[antecedent_estimate::joint_panel::PanelProgram],
        tile_draws: usize,
        ctx: &antecedent_core::ExecutionContext,
        sink: S,
    ) -> Result<
        antecedent_estimate::joint_panel::PanelStreamReceipt,
        antecedent_estimate::joint_panel::PanelError,
    >
    where
        S: FnMut(
                usize,
                &[Option<Vec<antecedent_estimate::joint_panel::PanelResponse>>],
            ) -> Result<(), antecedent_estimate::joint_panel::PanelError>
            + Send,
    {
        antecedent_estimate::joint_panel::stream_temporal_law(
            data, point, draws, programs, tile_draws, ctx, sink,
        )
    }
}
/// Learned joint source-target transport (model-provider row `learned_joint_transport`).
pub mod learned_joint_transport {
    pub use antecedent_estimate::learned_joint_transport::LearnedJointModel;
}
/// Items of `antecedent-estimate::longitudinal_regime`.
pub mod longitudinal_regime {
    use antecedent_estimate::longitudinal_regime::{RegimeKernelError, RegimeValueSummary};
    pub use antecedent_estimate::longitudinal_regime::{SequentialDrInputs, SubjectHistories};

    /// Evaluate a caller-prescribed regime by inverse probability weighting
    /// (`antecedent_estimate::longitudinal_regime::evaluate_regime_value`).
    ///
    /// # Errors
    ///
    /// As the kernel function.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled LongitudinalRegime route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_regime_value(
        outcomes: &[f64],
        histories: &SubjectHistories<'_>,
        actions: &[bool],
    ) -> Result<RegimeValueSummary, RegimeKernelError> {
        antecedent_estimate::longitudinal_regime::evaluate_regime_value(
            outcomes, histories, actions,
        )
    }

    /// Evaluate supplied conditional period rewards under a prescribed regime
    /// (`antecedent_estimate::longitudinal_regime::evaluate_g_formula_value`).
    ///
    /// # Errors
    ///
    /// As the kernel function.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled LongitudinalRegime route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_g_formula_value(
        predictions: &[f64],
        actions: &[bool],
        treatment_probability: &[f64],
        censoring_probability: &[f64],
        subjects: usize,
        periods: usize,
        minimum_probability: f64,
    ) -> Result<RegimeValueSummary, RegimeKernelError> {
        antecedent_estimate::longitudinal_regime::evaluate_g_formula_value(
            predictions,
            actions,
            treatment_probability,
            censoring_probability,
            subjects,
            periods,
            minimum_probability,
        )
    }

    /// The backward-recursive sequential doubly robust regime value
    /// (`antecedent_estimate::longitudinal_regime::evaluate_sequential_dr_value`).
    ///
    /// # Errors
    ///
    /// As the kernel function.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled LongitudinalRegime route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_sequential_dr_value(
        outcomes: &[f64],
        histories: &SubjectHistories<'_>,
        sequential: &SequentialDrInputs<'_>,
    ) -> Result<RegimeValueSummary, RegimeKernelError> {
        antecedent_estimate::longitudinal_regime::evaluate_sequential_dr_value(
            outcomes, histories, sequential,
        )
    }
}
/// Corrected functionals under a declared measurement model.
pub mod measurement {
    pub use antecedent_estimate::measurement::{
        CorrectedExposure, DeclaredEstimate, DetectionDependentPrevalence,
        DetectionFunctionDeclaration, ExposureDraws, ExposureErrorDeclaration, ExposureMeasurement,
        ImputationData, MeasuredEffect, MeasurementError, PosteriorCombination, RatePair,
        SoftExposureCombination, SoftExposureEffect, soft_exposure_reliability_report,
    };

    /// Pool per-draw estimates and their variances with Rubin's rules.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::measurement::pool_measured_effect`].
    ///
    /// # Errors
    /// Fewer than two draws, lengths that differ or non-finite values (`invalid_argument`).
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled MeasuredExposurePooling program family, which runs the same kernel call after its unit's support gate"
    )]
    pub fn pool_measured_effect(
        points: &[f64],
        variances: &[f64],
        naive: Option<f64>,
    ) -> Result<
        antecedent_estimate::measurement::MeasuredEffect,
        antecedent_estimate::measurement::MeasurementError,
    > {
        antecedent_estimate::measurement::pool_measured_effect(points, variances, naive)
    }

    /// Impute the latent exposure `draw_count` times.
    ///
    /// The posterior of every unit is formed from models fitted on a row bootstrap and from
    /// redrawn measurement parameters, so the draws carry the uncertainty of both; the point
    /// posterior (`probability`) uses the full data and the point parameters.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::measurement::impute_exposure`].
    ///
    /// # Errors
    /// An unidentified declaration (`measurement_not_identified`), an outcome that cannot carry an
    /// outcome-specific error (`route_not_supported`), malformed inputs (`invalid_argument`), a
    /// latent-class model that cannot be refitted, or cancellation.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled MeasuredExposureImputation program family, which runs the same kernel call after its unit's support gate"
    )]
    pub fn impute_exposure(
        declaration: &antecedent_estimate::measurement::ExposureMeasurement,
        data: &antecedent_estimate::measurement::ImputationData<'_>,
        draw_count: usize,
        seed: u64,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_estimate::measurement::ExposureDraws,
        antecedent_estimate::measurement::MeasurementError,
    > {
        antecedent_estimate::measurement::impute_exposure(declaration, data, draw_count, seed, ctx)
    }

    /// Fit repeat codings of a binary or ordinal facet as replicates of one latent level.
    ///
    /// Conditional independence of the codings given the latent level is an assumption of the
    /// model. Three or more codings identify it (more for more levels); two need exchangeable
    /// codings and a declared law of the lowest level; fewer refuse.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::measurement::fit_repeat_codings`].
    ///
    /// # Errors
    /// A shape violation, too few codings for the declared levels and constraints, codings that
    /// do not separate the levels, or cancellation refuse.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled RepeatCodingsLatentClass program family, which runs the same kernel call after its unit's support gate"
    )]
    pub fn fit_repeat_codings(
        data: &antecedent_stats::measurement_error::LatentClassData<'_>,
        constraints: &antecedent_stats::measurement_error::LatentClassConstraints,
        settings: &antecedent_stats::measurement_error::LatentClassSettings,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_stats::measurement_error::LatentClassFit,
        antecedent_estimate::measurement::MeasurementError,
    > {
        antecedent_estimate::measurement::fit_repeat_codings(data, constraints, settings, ctx)
    }

    /// Fit a linear outcome model on a per-unit probability of exposure.
    ///
    /// Posterior-mean substitution reports a variance term for the exposure uncertainty; multiple
    /// imputation reports within- and between-imputation variance. Substitution needs the stated
    /// probabilities to be calibrated given the covariates; imputation additionally conditions
    /// each draw on the outcome (see `soft_exposure_outcome_conditioned_draws`).
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::measurement::correct_soft_exposure_effect`].
    ///
    /// # Errors
    /// A shape or range violation, a rank-deficient design, fewer than two imputations or
    /// cancellation refuse.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled SoftExposureCorrection program family, which runs the same kernel call after its unit's support gate"
    )]
    pub fn correct_soft_exposure_effect(
        data: &antecedent_stats::measurement_error::SoftExposureData<'_>,
        combination: &antecedent_estimate::measurement::SoftExposureCombination<'_>,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_estimate::measurement::SoftExposureEffect,
        antecedent_estimate::measurement::MeasurementError,
    > {
        antecedent_estimate::measurement::correct_soft_exposure_effect(data, combination, ctx)
    }

    /// Correct a binary exposure's misclassification, including error that differs by outcome.
    ///
    /// `structure` is `ExposureErrorStructure::Shared` for error that does not depend on the
    /// outcome and `ExposureErrorStructure::ByOutcome` for outcome-specific rates, each declared
    /// or estimated from validation samples (their binomial variances are then propagated).
    /// Rates whose Youden index in either outcome level is at or below `min_youden` are not
    /// identified and refuse.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::measurement::correct_exposure_misclassification`].
    ///
    /// # Errors
    /// Invalid counts, rates that are not identified, a corrected exposure share outside `(0, 1)`
    /// or cancellation refuse.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled ExposureMisclassificationCorrection program family, which runs the same kernel call after its unit's support gate"
    )]
    pub fn correct_exposure_misclassification(
        counts: &antecedent_stats::measurement_error::ExposureOutcomeCounts,
        structure: &antecedent_stats::measurement_error::ExposureErrorStructure,
        min_youden: f64,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_estimate::measurement::CorrectedExposure,
        antecedent_estimate::measurement::MeasurementError,
    > {
        antecedent_estimate::measurement::correct_exposure_misclassification(
            counts, structure, min_youden, ctx,
        )
    }

    /// Prevalence of a condition detected with a probability that depends on a covariate.
    ///
    /// A fully declared function point-identifies the prevalence; a partially declared (monotone,
    /// anchored) one gives sharp bounds. Mean detection net of false positives at or below
    /// `min_net_detection` is not identified and refuses.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::measurement::correct_detection_dependent_prevalence`].
    ///
    /// # Errors
    /// A shape or range violation, a non-identifying detection, an inconsistent envelope or
    /// cancellation refuse.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled DetectionDependentPrevalenceCorrection program family, which runs the same kernel call after its unit's support gate"
    )]
    pub fn correct_detection_dependent_prevalence(
        data: &antecedent_stats::measurement_error::DetectionData<'_>,
        declaration: &antecedent_stats::measurement_error::DetectionDeclaration,
        false_positive: &antecedent_stats::measurement_error::FalsePositiveRate,
        min_net_detection: f64,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_estimate::measurement::DetectionDependentPrevalence,
        antecedent_estimate::measurement::MeasurementError,
    > {
        antecedent_estimate::measurement::correct_detection_dependent_prevalence(
            data,
            declaration,
            false_positive,
            min_net_detection,
            ctx,
        )
    }
}
/// Multiple imputation: impute, fit, pool as one uncertainty path.
pub mod multiple_imputation {
    pub use antecedent_estimate::multiple_imputation::PooledVectorEstimate;
}
/// Continuous eleven-dimensional Bayesian Verma nested-Markov candidate.
pub mod nested_markov_bayesian {
    pub use antecedent_learn::nested_markov_bayesian::{Options, Prior};
}
/// Bounded binary nested-Markov likelihood pilot (record `identify.binary_nested_markov`).
pub mod nested_markov_binary {
    pub use antecedent_estimate::nested_markov_binary::{
        AdmgDeclaration, FitOptions, NestedMarkovInput, Regime, RegimeCounts,
    };
}
/// Items of `antecedent-estimate::ou_mechanisms`.
pub mod ou_mechanisms {
    pub use antecedent_estimate::ou_mechanisms::OuMechanismOutcome;
}
/// Frozen-score doubly robust policy value utility.
pub mod policy_value {
    pub use antecedent_estimate::policy_value::{
        BinaryIpwDesign, BinaryPolicyDesign, BinaryPolicyLimits, MultiActionIpwDesign,
        MultiActionLimits,
    };

    /// Horvitz–Thompson uplift within caller-supplied score bins.
    ///
    /// Distinct from `evaluate_uplift_bins`: a bin may contain one row, and this
    /// function does not form an interval.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::policy_value::uplift_by_score_bin`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::policy_value::uplift_by_score_bin`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled PolicyValue route, which calls this kernel function with the same inputs"
    )]
    pub fn uplift_by_score_bin(
        outcome: &[f64],
        assignment: &[bool],
        score_bin: &[usize],
        propensity: &[f64],
        bin_count: usize,
    ) -> Result<Vec<antecedent_estimate::policy_value::ScoreBinUplift>, &'static str> {
        antecedent_estimate::policy_value::uplift_by_score_bin(
            outcome, assignment, score_bin, propensity, bin_count,
        )
    }

    /// Evaluate fixed held-out binary policy scores using randomized AIPW.
    ///
    /// Propensities and nuisance predictions are caller-supplied design inputs.
    /// Standard errors assume independent evaluation subjects. `mu0` and `mu1` are the
    /// frozen control and treated conditional outcome predictions.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::policy_value::evaluate_policy_value_scores`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::policy_value::evaluate_policy_value_scores`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled PolicyValue route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_policy_value_scores(
        outcome: &[f64],
        design: &antecedent_estimate::policy_value::BinaryPolicyDesign<'_>,
        mu0: &[f64],
        mu1: &[f64],
    ) -> Result<antecedent_estimate::policy_value::PolicyValueScores, &'static str> {
        antecedent_estimate::policy_value::evaluate_policy_value_scores(outcome, design, mu0, mu1)
    }

    /// Known-randomization multi-action policy value with the historical constraint checks.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::policy_value::evaluate_multi_action_ipw_policy`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::policy_value::evaluate_multi_action_ipw_policy`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled PolicyValue route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_multi_action_ipw_policy(
        outcome: &[f64],
        design: &antecedent_estimate::policy_value::MultiActionIpwDesign<'_>,
        limits: &antecedent_estimate::policy_value::MultiActionLimits<'_>,
    ) -> Result<antecedent_estimate::policy_value::MultiActionIpwPolicy, &'static str> {
        antecedent_estimate::policy_value::evaluate_multi_action_ipw_policy(outcome, design, limits)
    }

    /// Known-propensity binary policy value with the historical constraint checks.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::policy_value::evaluate_binary_ipw_policy`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::policy_value::evaluate_binary_ipw_policy`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled PolicyValue route, which calls this kernel function with the same inputs"
    )]
    pub fn evaluate_binary_ipw_policy(
        outcome: &[f64],
        design: &antecedent_estimate::policy_value::BinaryIpwDesign<'_>,
        limits: &antecedent_estimate::policy_value::BinaryPolicyLimits,
    ) -> Result<antecedent_estimate::policy_value::BinaryIpwPolicy, &'static str> {
        antecedent_estimate::policy_value::evaluate_binary_ipw_policy(outcome, design, limits)
    }
}
/// Design-based inference for complete randomization of independent clusters.
pub mod randomized_neyman {
    /// Stratified difference in means under complete randomization within blocks.
    ///
    /// Each block needs two treated and two control units. Block labels are
    /// aggregated in lexicographic order.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::randomized_neyman::stratified_neyman_effect`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::randomized_neyman::stratified_neyman_effect`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled RandomizedEffect route, which calls this kernel function with the same inputs"
    )]
    pub fn stratified_neyman_effect(
        outcome: &[f64],
        assignment: &[bool],
        blocks: &[impl AsRef<str>],
    ) -> Result<antecedent_estimate::randomized_neyman::StratifiedNeymanEffect, &'static str> {
        antecedent_estimate::randomized_neyman::stratified_neyman_effect(
            outcome, assignment, blocks,
        )
    }
}
/// Bayesian counterpart of the randomized-trial arm contrasts.
pub mod randomized_posterior {
    pub use antecedent_estimate::randomized_posterior::{
        ArmBetaPrior, ArmNormalPrior, ArmOutcomeModel,
    };
}
/// Conservative score inference for independently randomized action contrasts.
pub mod randomized_scores {
    /// CUPED residualization with a known-propensity Horvitz–Thompson contrast.
    ///
    /// The slope is the pooled outcome–covariate covariance divided by the
    /// covariate's second moment. Refusal strings match the historical Python entry.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::randomized_scores::cuped_ht_effect`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::randomized_scores::cuped_ht_effect`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled RandomizedEffect route, which calls this kernel function with the same inputs"
    )]
    pub fn cuped_ht_effect(
        outcome: &[f64],
        covariate: &[f64],
        assignment: &[bool],
        propensity: &[f64],
    ) -> Result<antecedent_estimate::randomized_scores::CupedHtEffect, &'static str> {
        antecedent_estimate::randomized_scores::cuped_ht_effect(
            outcome, covariate, assignment, propensity,
        )
    }

    /// Exact Bernoulli randomization test for one to twenty units.
    ///
    /// Every allocation is weighted by its known assignment probability. This is
    /// not the equal-probability complete-randomization test.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::randomized_scores::bernoulli_exact_randomization_test`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::randomized_scores::bernoulli_exact_randomization_test`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled RandomizedEffect route, which calls this kernel function with the same inputs"
    )]
    pub fn bernoulli_exact_randomization_test(
        outcome: &[f64],
        assignment: &[bool],
        propensity: &[f64],
    ) -> Result<antecedent_estimate::randomized_scores::BernoulliExactTest, &'static str> {
        antecedent_estimate::randomized_scores::bernoulli_exact_randomization_test(
            outcome, assignment, propensity,
        )
    }
}
/// Whole-procedure resampling shared by the rollout designs.
pub mod rollout_resampling {
    pub use antecedent_estimate::rollout_resampling::DesignEffect;
}
/// Shared-data covariance of scenario-specific estimates over one row sample.
pub mod scenario_covariance {
    pub use antecedent_estimate::scenario_covariance::ScenarioCovariance;
}
/// Selection differences and invariances behind each scenario answer.
pub mod scenario_invariance_report {
    pub use antecedent_estimate::scenario_invariance_report::{
        EnvelopeSide, ScenarioSetInvarianceReport,
    };
}
/// Bayesian-bootstrap posterior of the mean of per-subject scores.
pub mod score_posterior {
    pub use antecedent_estimate::score_posterior::ScorePosteriorOptions;
}
/// Cross-fitted AIPW score tables for retargeting and joint inference.
pub mod scores {
    pub use antecedent_estimate::scores::ScoreInference;
}
/// Smoothed dose-response transport grid.
pub mod smoothed_dose {
    pub use antecedent_estimate::smoothed_dose::parse_smoothing_kernel;
}
/// Cohort-specific event-time contrasts against never-treated subjects.
pub mod staggered_event_study {
    /// Balanced-panel group-time ATT for post-adoption periods only.
    ///
    /// `estimate` also emits pre-adoption leads and uses different refusal text.
    /// This function keeps the historical group-time refusals.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::staggered_event_study::group_time_att`].
    ///
    /// # Errors
    ///
    /// The error of [`antecedent_estimate::staggered_event_study::group_time_att`], unchanged.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled PanelDid route, which calls this kernel function with the same inputs"
    )]
    pub fn group_time_att(
        outcomes: &[f64],
        subjects: &[impl AsRef<str>],
        periods: &[i64],
        cohorts: &[i64],
        clusters: &[impl AsRef<str>],
    ) -> Result<Vec<antecedent_estimate::staggered_event_study::GroupTimeAtt>, &'static str> {
        antecedent_estimate::staggered_event_study::group_time_att(
            outcomes, subjects, periods, cohorts, clusters,
        )
    }
}
/// Native risk-set summaries for randomized survival outcomes.
pub mod survival {
    pub use antecedent_estimate::survival::{KnownCensoring, SurvivalEndpoint, SurvivalRows};

    /// Evaluate arm curves with a caller supplied, subject-specific censoring survival grid.
    ///
    /// `KnownCensoring::survival` is row-major with one value per subject and time;
    /// each value is `P(C >= t)` so subjects censored at `t` remain in the event risk set.
    /// The probabilities must be known independently of the outcomes used here; this
    /// point kernel does not fit or verify a censoring model. No interval is
    /// constructed by this function. Optional marginally independent left entry
    /// (`SurvivalRows::delayed_entry`) and censoring probabilities remain attached
    /// to their subject in a subject-resampling interval; conditional or informative
    /// entry is not adjusted.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::survival::randomized_survival_ipcw_summary`].
    ///
    /// # Errors
    ///
    /// Returns an error for misaligned rows or times, invalid event codes, absent
    /// arm support through `tau`, or censoring probabilities that violate the
    /// declared positivity and monotonicity contract.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled Survival route, which calls this kernel function with the same inputs"
    )]
    pub fn randomized_survival_ipcw_summary(
        rows: &antecedent_estimate::survival::SurvivalRows<'_>,
        censoring: &antecedent_estimate::survival::KnownCensoring<'_>,
        tau: f64,
        endpoint: antecedent_estimate::survival::SurvivalEndpoint,
    ) -> Result<(antecedent_estimate::survival::RandomizedSurvivalSummary, f64), &'static str> {
        antecedent_estimate::survival::randomized_survival_ipcw_summary(
            rows, censoring, tau, endpoint,
        )
    }
}
/// Sequence-level inference for randomized switchback experiments.
pub mod switchback {
    /// Estimate a switchback ITT from known marginal randomization probabilities.
    ///
    /// Returns `None` on invalid rows, fewer than two independent sequences, or
    /// an unobserved treatment arm across the entire schedule. Both arms need not
    /// appear within each sequence.
    /// Within-sequence outcomes may depend arbitrarily on one another. Identification
    /// additionally requires no treatment carryover and no between-sequence
    /// interference; neither can be established from these observations. Equal
    /// sequence lengths, at least 30 sequences, and 30 observed periods per arm
    /// are required for an interval so that no sequence dominates the
    /// period-weighted mean and both arm contributions are supported.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_estimate::switchback::switchback_itt`].
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled RandomizedEffect route, which calls this kernel function with the same inputs"
    )]
    #[must_use]
    pub fn switchback_itt(
        outcomes: &[f64],
        assignment: &[bool],
        probabilities: &[f64],
        sequence_ids: &[&str],
    ) -> Option<antecedent_estimate::switchback::SwitchbackItt> {
        antecedent_estimate::switchback::switchback_itt(
            outcomes,
            assignment,
            probabilities,
            sequence_ids,
        )
    }
}
/// Bayesian counterpart of the switchback treatment effect.
pub mod switchback_posterior {
    pub use antecedent_estimate::switchback_posterior::{
        CarryoverDeclaration, PeriodAdjustment, SequenceDependence, SwitchbackModel,
        SwitchbackPrior,
    };
}
/// Dependence-preserving sampling interval for the finite action sequence.
pub mod temporal_dependent_interval {
    pub use antecedent_estimate::temporal_dependent_interval::{
        DependentIntervalConfig, IntervalMethod, SequenceHistory, TemporalUnitPanel, UnitHistories,
    };
}
/// Uncertain pre-action state of the finite action sequence.
pub mod temporal_initial_state {
    pub use antecedent_estimate::temporal_initial_state::{
        InitialStateLaw, InitialStatePopulation, InitialStateSpec,
    };
}
/// Semantic refresh of a prepared horizon-two temporal result onto a new observation period.
pub mod temporal_refresh {
    pub use antecedent_estimate::temporal_refresh::{
        MechanismStanding, ObservationCadence, ObservationPeriod, TemporalWindowIdentity,
    };
}
/// Exact point evaluation of one finite temporal transport sequence.
pub mod temporal_transport {
    pub use antecedent_estimate::temporal_transport::TemporalSequenceReport;
}
/// Reports over the identified members of a finite transport scenario set.
pub mod transport_scenarios {
    pub use antecedent_estimate::transport_scenarios::{PreparedScenarioSet, ScenarioSetReport};
}

/// Caller-supplied nuisance learner factories and cross-fitted AIPW predictions.
pub mod external_learner {
    pub use antecedent_learn::aipw_crossfit::AipwCrossFitError;
    pub use antecedent_learn::{
        DesignView, FittedPredictor, IsolatedFactory, LearnError, LearnerCapabilities,
        LearnerFactory, LearnerProvenance, PredictionTask, TargetView,
    };

    /// Cross-fit AIPW nuisances using caller-supplied learner factories.
    ///
    /// The designs must have the same unselected physical rows as `y`, `t`, and the fold
    /// assignment. Factories see row selections into those designs; adapters to external code
    /// must copy only the selected rows before crossing the provider boundary.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_learn::aipw_crossfit::cross_fit_aipw_with_factories`].
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible tasks or capabilities, misaligned inputs, invalid
    /// predictions, failed fits, cancellation, or resource admission failures.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled StaticAteDag route, which calls this kernel function with the same inputs"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the signature is the kernel function's, unchanged"
    )]
    pub fn cross_fit_aipw_with_factories(
        outcome: &dyn antecedent_learn::LearnerFactory,
        treatment: &dyn antecedent_learn::LearnerFactory,
        x_outcome: antecedent_learn::DesignView<'_>,
        x_treat: antecedent_learn::DesignView<'_>,
        y: &[f64],
        t: &[f64],
        (folds, fold_ids): (usize, Vec<u16>),
        ctx: &antecedent_core::ExecutionContext,
        weights: Option<&[f64]>,
    ) -> Result<
        antecedent_learn::aipw_crossfit::AipwPredictions,
        antecedent_learn::aipw_crossfit::AipwCrossFitError,
    > {
        antecedent_learn::aipw_crossfit::cross_fit_aipw_with_factories(
            outcome,
            treatment,
            x_outcome,
            x_treat,
            y,
            t,
            (folds, fold_ids),
            ctx,
            weights,
        )
    }
}
