//! Analysis result artifact.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod bindings;
mod missing_data;
mod structural;
mod uncertainty;
pub use missing_data::{
    ImputationExtra, ImputationReport, ImputationVectorReport, ImputationWald, MissingDataReport,
};
pub use structural::{
    StructuralAggregationPolicy, StructuralResponseAtom, StructuralResponseMixture,
    StructuralWeightBasis,
};
pub(crate) use uncertainty::RESPONSE_BOOTSTRAP_SE;
pub use uncertainty::{
    IntervalBinding, PERCENTILE_95_MIN_REPLICATES, PublishedScalarUncertainty,
    REPORTED_SE_INTERVAL_LEVEL, circular_block_dependence, reported_se_interval_z,
};

use antecedent_attribution::{
    AnomalyScores, ChangeAttributionResult, MechanismChangeDetection, UnitChangeResult,
};
use antecedent_contract::{
    ExecutionPerformanceRecord, LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord,
    ProvenanceGraph,
};
use antecedent_core::{
    CausalResponse, Diagnostic, DiagnosticKind, IdentificationStatus, ResponseEnvelope,
    ResponseValue, VariableId,
};
use antecedent_estimate::{
    CausalPosterior, EffectEstimate, InterferenceEstimate, InterventionalDistributionEstimate,
    TemporalMediationEstimate, TemporalMediationGrid, TransportEffectEstimate,
};
use antecedent_identify::{IdentificationResult, IdentifiedEstimand};
use antecedent_io::{AnalysisTraceWire, DerivationStepWire, assumptions_to_wire};
use antecedent_validate::{PredictiveCheckReport, RefutationReport};
use std::sync::Arc;

use crate::gcm::IteResult;

/// Typed primary result of a study.
#[allow(
    clippy::large_enum_variant,
    reason = "variants are constructed rarely and held singly; boxing would change the public result API"
)]
#[derive(Clone, Debug)]
pub enum PrimaryEstimate {
    /// Conventional scalar effect estimate.
    Effect(EffectEstimate),
    /// This query has a different typed answer (for example policy value).
    NotAnEffect,
}

impl PrimaryEstimate {
    /// Effect payload, absent for query families with a distinct answer shape.
    #[must_use]
    pub fn as_effect(&self) -> Option<&EffectEstimate> {
        match self {
            Self::Effect(effect) => Some(effect),
            Self::NotAnEffect => None,
        }
    }

    /// Mutable effect payload, absent for query families with a distinct answer shape.
    pub fn as_effect_mut(&mut self) -> Option<&mut EffectEstimate> {
        match self {
            Self::Effect(effect) => Some(effect),
            Self::NotAnEffect => None,
        }
    }
}

/// Design metadata retained with a randomized ITT estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomizedEffectEstimate {
    /// Assignment ITT or Wald CACE/LATE contrast under the declared mechanism.
    pub effect: f64,
    /// Query target label.
    pub estimand: Arc<str>,
    /// Outcome assignment ITT when estimating CACE/LATE.
    pub intention_to_treat_effect: Option<f64>,
    /// Treatment-receipt first stage when estimating CACE/LATE.
    pub first_stage_effect: Option<f64>,
    /// Row-aligned treatment receipt when estimating CACE/LATE.
    pub received_treatment: Option<Arc<[bool]>>,
    /// Two-sided Fisher sharp-null p-value for a complete design: exact when every fixed-count
    /// assignment was enumerated, a Monte Carlo permutation p-value otherwise (see
    /// [`Self::randomization_allocations`]).
    pub randomization_p_value: Option<f64>,
    /// Number of fixed-count assignments the p-value averages over: all `C(n, treated)` of them
    /// when they were exhaustively enumerated, otherwise the sampled assignments plus the
    /// observed one.
    pub randomization_allocations: Option<u64>,
    /// Second-factor marginal assignment effect in a fixed 2×2 factorial design.
    pub second_factor_effect: Option<f64>,
    /// Difference of primary-factor effects between second-factor levels.
    pub factorial_interaction: Option<f64>,
    /// Conservative variance for the second-factor marginal effect.
    pub second_factor_variance: Option<f64>,
    /// Conservative variance for the interaction contrast.
    pub factorial_interaction_variance: Option<f64>,
    /// Ordered arm means, variance contributions, and observed counts for multi-arm trials.
    #[allow(
        clippy::type_complexity,
        reason = "labeled arm tuple is part of this public result field's type"
    )]
    pub multi_arm_values: Arc<[(Arc<str>, f64, f64, usize)]>,
    /// Design variance or conservative bound as labeled by `uncertainty`;
    /// for Bernoulli HT this observed-arm estimator is conservative in randomization
    /// expectation and need not exceed the true variance for every assignment.
    /// Switchback uses an independent-sequence sandwich estimate.
    pub variance_upper_bound: f64,
    /// Primary contrast SE when a calibrated pointwise interval is reported.
    pub standard_error: Option<f64>,
    /// Pointwise 95% primary contrast interval, if supported.
    pub interval_95: Option<[f64; 2]>,
    /// Pointwise 95% secondary factorial main-effect interval, if supported.
    pub second_factor_interval_95: Option<[f64; 2]>,
    /// Pointwise 95% factorial interaction interval, if supported.
    pub factorial_interaction_interval_95: Option<[f64; 2]>,
    /// Pointwise 95% action-versus-reference intervals, aligned with arm labels;
    /// the reference arm has no interval.
    pub multi_arm_intervals_95: Arc<[Option<[f64; 2]>]>,
    /// Smallest declared unit-level inclusion probability.
    pub minimum_assignment_probability: f64,
    /// Assignment design label (`bernoulli`, `complete`, or `stratified`).
    pub assignment_design: Arc<str>,
    /// Row-aligned block labels for stratified assignment, empty otherwise.
    pub blocks: Arc<[Arc<str>]>,
    /// Row-aligned period labels for switchback assignment, empty otherwise.
    pub periods: Arc<[Arc<str>]>,
    /// Number of analyzed control assignment units.
    pub control_units: usize,
    /// Number of analyzed treatment assignment units.
    pub treatment_units: usize,
    /// Explicit uncertainty contract for the returned variance.
    pub uncertainty: Arc<str>,
    /// Assignment unit labels in the analyzed row order.
    pub assignment_units: Arc<[Arc<str>]>,
    /// Outcome unit labels in the analyzed row order.
    pub outcome_units: Arc<[Arc<str>]>,
    /// Control and treatment labels.
    pub treatment_arms: (Arc<str>, Arc<str>),
}

/// Two-period panel `DiD` point estimate with cluster-aware uncertainty.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelDidEstimate {
    /// Difference in average subject-level outcome changes.
    pub effect: f64,
    /// Cluster score-sandwich standard error.
    pub standard_error: f64,
    /// Pointwise 95% interval for sufficiently supported scalar panel or repeated-cross-section `DiD`.
    pub interval_95: Option<[f64; 2]>,
    /// Number of treated subjects.
    pub treated_subjects: usize,
    /// Number of comparison subjects.
    pub comparison_subjects: usize,
    /// Total distinct inference clusters.
    pub clusters: usize,
    /// Explicit uncertainty semantics.
    pub uncertainty: Arc<str>,
    /// Cohort-specific contrasts for a retained event study; empty for scalar `DiD`.
    pub event_time_effects: Arc<[antecedent_estimate::staggered_event_study::EventTimeEffect]>,
    /// Pointwise post-adoption intervals aligned with event-time effects;
    /// pre-adoption descriptive contrasts always have no interval.
    pub event_time_intervals_95: Arc<[Option<[f64; 2]>]>,
    /// Propensity range, effective control count, and caller cross-fit declaration.
    pub augmented: Option<(f64, f64, f64, bool)>,
}

/// Balanced-panel synthetic-control point result and donor-support diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticControlEstimate {
    /// Average treated-minus-synthetic post-intervention outcome.
    pub effect: f64,
    /// Pre-intervention root mean squared fit gap.
    pub pre_treatment_rmse: f64,
    /// Donor weights in stable unit-name order.
    pub donor_weights: Arc<[(Arc<str>, f64)]>,
    /// Leave-one-donor-out placebo effects, corresponding to donor order.
    pub placebo_effects: Arc<[f64]>,
    /// Descriptive placebo rank without a calibrated p-value claim.
    pub placebo_rank: f64,
    /// Effective donor count, the reciprocal of summed squared weights.
    pub effective_donors: f64,
    /// Observed pre-intervention periods.
    pub n_pre_periods: usize,
    /// Observed post-intervention periods.
    pub n_post_periods: usize,
    /// Explicit uncertainty semantics.
    pub uncertainty: Arc<str>,
    /// Exact Fisher p-value when uniform one-unit assignment was declared.
    pub randomization_p_value: Option<f64>,
    /// Prespecified constant additive effect tested by the Fisher p-value.
    pub randomization_null_effect: Option<f64>,
    /// Absolute gap for every possible treated unit under the sharp null.
    pub randomization_statistics: Arc<[(Arc<str>, f64)]>,
    /// Unadjusted simplex gap when a donor outcome model corrects the point estimate.
    pub unadjusted_effect: Option<f64>,
    /// Donor outcome-model prediction difference subtracted from the simplex gap.
    pub outcome_model_correction: Option<f64>,
    /// Positive ridge penalty used for the donor outcome model.
    pub augmentation_ridge: Option<f64>,
    /// What an unbalanced panel supplied, when it was read over the treated unit's observed
    /// window.
    pub observed_window: Option<ObservedWindowReport>,
}

/// The support an observed-window synthetic control read of an unbalanced panel.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservedWindowReport {
    /// Pre-intervention periods the treated unit is observed in; the weights are fitted here.
    pub pre_periods: Arc<[i64]>,
    /// Post-intervention periods the treated unit is observed in; the effect averages these.
    pub post_periods: Arc<[i64]>,
    /// Donors observed in every window period.
    pub donors_used: usize,
    /// Units excluded for missing part of the window, with the number of window periods missed.
    pub donors_dropped: Arc<[(Arc<str>, usize)]>,
    /// Observed unit-periods no estimate read.
    pub unread_rows: usize,
}

/// Conditional response grid and optional fixed kernel-smoothed dose-policy value.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousDoseResponseEstimate {
    /// One response and support record for each baseline-group target cell.
    pub points: Arc<[antecedent_estimate::continuous_dose::DoseResponsePoint]>,
    /// Prespecified smoothing window.
    pub bandwidth: f64,
    /// Caller-declared source of dose densities.
    pub density_provenance: Arc<str>,
    /// Fixed group-to-dose policy value, when requested.
    pub fixed_policy: Option<antecedent_estimate::continuous_dose::DosePolicyValue>,
    /// Point-only response or the dose-policy interval method.
    pub uncertainty: Arc<str>,
}

/// Point-only synthetic difference-in-differences result with fitted simplex weights.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticDidEstimate {
    /// Post-treatment difference-in-differences contrast.
    pub effect: f64,
    /// Root mean squared treated-versus-synthetic pre-period gap.
    pub pre_treatment_rmse: f64,
    /// Unit weights over donor units.
    pub donor_weights: Arc<[(Arc<str>, f64)]>,
    /// Time weights over pre-intervention periods.
    pub time_weights: Arc<[(i64, f64)]>,
    /// Number of donor units.
    pub n_donors: usize,
    /// Number of pre-intervention periods.
    pub n_pre_periods: usize,
    /// Number of post-intervention periods.
    pub n_post_periods: usize,
    /// Explicit point-only uncertainty statement.
    pub uncertainty: Arc<str>,
    /// Exact sharp-null p-value under declared uniform single-unit assignment.
    pub randomization_p_value: Option<f64>,
    /// Prespecified constant additive effect tested by the Fisher p-value.
    pub randomization_null_effect: Option<f64>,
    /// Absolute synthetic-DiD contrast for every candidate treated unit.
    pub randomization_statistics: Arc<[(Arc<str>, f64)]>,
    /// Rounding tolerance within which a candidate ties the observed statistic; absent means
    /// strict ranking.
    pub randomization_ranking_tolerance: Option<f64>,
}

/// Local fuzzy-discontinuity or regression-kink ratio with fixed-bandwidth inference.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalPolynomialRatioEstimate {
    /// Ratio of the outcome contrast to the treatment contrast.
    pub effect: f64,
    /// Local outcome jump or slope change.
    pub reduced_form: f64,
    /// Local treatment jump or slope change.
    pub first_stage: f64,
    /// Prespecified running-variable cutoff retained with the result.
    pub cutoff: f64,
    /// Prespecified triangular-kernel bandwidth retained with the result.
    pub bandwidth: f64,
    /// True for a slope-kink contrast; false for a level discontinuity.
    pub kink: bool,
    /// Observations inside the window strictly below the cutoff.
    pub n_left: usize,
    /// Observations inside the window on or above the cutoff.
    pub n_right: usize,
    /// HC0 delta-method standard error for the bias-corrected ratio.
    pub standard_error: f64,
    /// Pointwise normal lower endpoint when the first stage supports a finite ratio interval.
    pub ci_lower: Option<f64>,
    /// Pointwise normal upper endpoint when the first stage supports a finite ratio interval.
    pub ci_upper: Option<f64>,
    /// Descriptive HC0 standard error for the bias-corrected outcome contrast.
    pub reduced_form_standard_error: f64,
    /// Descriptive HC0 standard error for the bias-corrected first stage.
    pub first_stage_standard_error: f64,
    /// Explicit interval construction and fixed-bandwidth dependence.
    pub uncertainty: Arc<str>,
}

/// Excursion contrasts of a sequentially randomized unit-period design, or static-regime
/// survival on subject trajectories (unsupported, uncalibrated).
///
/// One entry per named coordinate. Frequentist results carry standard errors and pointwise
/// intervals (and, for several contrasts or a dose grid, a simultaneous band); Bayesian results
/// carry equal-tailed credible intervals of the retained draws, which ride on the result's
/// posterior. No coverage of any interval has been measured, and the route is not supported.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalDesignEstimate {
    /// Design: `two_arm`, `multi_arm`, `categorical`, `factorial`, `interference`, `dose`,
    /// `regime_survival` or `observational_excursion`.
    pub design: Arc<str>,
    /// Outcome functional: `mean`, `quantile:<level>`, the dose target, or `survival`.
    pub functional: Arc<str>,
    /// Coordinate names.
    pub coordinates: Arc<[Arc<str>]>,
    /// Point estimate (frequentist) or posterior mean (Bayesian) per coordinate.
    pub estimate: Arc<[f64]>,
    /// Standard error (frequentist) or posterior standard deviation (Bayesian).
    pub standard_error: Arc<[f64]>,
    /// Lower endpoint of each pointwise 95% interval.
    pub lower: Arc<[f64]>,
    /// Upper endpoint of each pointwise 95% interval.
    pub upper: Arc<[f64]>,
    /// Row-major covariance across coordinates, when the procedure produced one.
    pub covariance: Option<Arc<[f64]>>,
    /// Coordinates a simultaneous band covers (empty when there is none).
    pub band_coordinates: Arc<[usize]>,
    /// Sup-t critical value of the band.
    pub band_critical_value: Option<f64>,
    /// Lower band envelope, one per band coordinate.
    pub band_lower: Option<Arc<[f64]>>,
    /// Upper band envelope, one per band coordinate.
    pub band_upper: Option<Arc<[f64]>>,
    /// Holm-adjusted two-sided p-values of the contrast family, when the design has one.
    pub holm_adjusted_p: Option<Arc<[f64]>>,
    /// Worst-case smoothing bias per coordinate under the declared curvature bound.
    pub smoothing_bias_bound: Option<Arc<[f64]>>,
    /// Decision-outcome pairs (person-periods for regime survival).
    pub pairs: u64,
    /// Independent units or clusters.
    pub clusters: u64,
    /// Decisions whose outcome period was not observed.
    pub unpaired: u64,
    /// Smallest realized assignment (or regime-consistent) probability.
    pub minimum_probability: f64,
    /// Successful and failed whole-procedure bootstrap replicates, when the interval is a
    /// bootstrap interval.
    pub replicates: Option<(u64, u64)>,
    /// Posterior draws, for a Bayesian result.
    pub draws: Option<u64>,
    /// How the uncertainty was computed.
    pub uncertainty: Arc<str>,
    /// Calibration status of every interval: always `not_measured`.
    pub calibration: Arc<str>,
    /// Always `false`: the route is not supported.
    pub supported: bool,
}

/// Effects of a rollout design with the uncertainty of its whole-procedure path
/// (unsupported; calibration `not_measured`).
///
/// Carries a heterogeneity-robust or covariate-adjusted difference in differences (group-time,
/// aggregated and pre-trend effects), a conformal or multi-unit synthetic control (per-period
/// or per-unit effects), or a local ratio with a selected bandwidth or a calendar-time running
/// variable. The first effect is the scalar summary. Pointwise intervals and simultaneous bands
/// come from the design's own procedure (whole-unit resample, conformal test inversion,
/// placebo variance or block bootstrap), each refitting the whole estimator; none has a
/// coverage record and the route is not supported.
#[derive(Clone, Debug, PartialEq)]
pub struct QuasiRolloutEstimate {
    /// Design: `staggered_did`, `covariate_adjusted_did`, `conformal_synthetic_control`,
    /// `multi_unit_synthetic_control`, `multi_unit_synthetic_did`, `fuzzy_rd`,
    /// `regression_kink`, `temporal_fuzzy_rd` or `temporal_regression_kink`.
    pub design: Arc<str>,
    /// Estimator, comparison, adjustment and aggregation, as `snake_case` words.
    pub method: Arc<str>,
    /// Reported effects; the first is the scalar summary.
    pub effects: Arc<[antecedent_estimate::rollout_resampling::DesignEffect]>,
    /// How the uncertainty was computed.
    pub uncertainty: Arc<str>,
    /// Calibration status of every interval: always `not_measured`.
    pub calibration: Arc<str>,
    /// Requested replicates (zero for a conformal inversion).
    pub replicates_requested: u32,
    /// Replicates whose refit produced every summarized quantity.
    pub replicates_ok: u32,
    /// Replicates whose refit did not.
    pub replicates_failed: u32,
    /// Sup-t critical value of the simultaneous band, when one was formed.
    pub critical_value: Option<f64>,
    /// Named numeric diagnostics (pre-trend statistic, two-way fixed-effects comparison,
    /// density test, covariate balance, selected bandwidth).
    pub diagnostics: Arc<[(Arc<str>, f64)]>,
    /// Always `false`: the route is not supported.
    pub supported: bool,
}

impl QuasiRolloutEstimate {
    /// The scalar summary.
    #[must_use]
    pub fn overall(&self) -> Option<&antecedent_estimate::rollout_resampling::DesignEffect> {
        self.effects.first()
    }

    /// The named numeric diagnostic, when present.
    #[must_use]
    pub fn diagnostic(&self, name: &str) -> Option<f64> {
        self.diagnostics.iter().find(|(key, _)| key.as_ref() == name).map(|(_, value)| *value)
    }
}

/// Within-period history-matched contrast (unsupported, uncalibrated).
///
/// The matched pairs are retained so a loader recomputes every summary.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryMatchEstimate {
    /// Sampling-weighted mean of treated-minus-matched-control outcomes.
    pub contrast: f64,
    /// Abadie-Imbens standard error; absent when a matched unit or donor has no same-arm
    /// neighbour in its period.
    pub standard_error: Option<f64>,
    /// Sum of sampling weights on the matched treated units.
    pub weight_total: f64,
    /// Row indices of matched treated units.
    pub matched_queries: Arc<[usize]>,
    /// Row index of the control each matched query used.
    pub matched_donors: Arc<[usize]>,
    /// Outcome difference of each matched pair.
    pub differences: Arc<[f64]>,
    /// Sampling weight of each matched treated unit.
    pub pair_weights: Arc<[f64]>,
    /// Within-arm outcome variance estimate of each matched treated unit, in pair order; empty
    /// exactly when `standard_error` is absent.
    pub query_variances: Arc<[f64]>,
    /// Within-arm outcome variance estimate of the control each matched query used, in pair
    /// order; empty exactly when `standard_error` is absent.
    pub donor_variances: Arc<[f64]>,
    /// Treated rows with no control inside their period and the caliper.
    pub unmatched_treated: Arc<[usize]>,
    /// Periods in which at least one treated unit matched.
    pub periods_matched: usize,
    /// Calendar period of each matched treated unit, in pair order.
    pub pair_periods: Arc<[i64]>,
    /// Newey-West standard error of the contrast over the matched periods, which allows serial
    /// dependence between periods; absent when fewer than two periods hold a match or a matched
    /// weight is not positive.
    pub serial_standard_error: Option<f64>,
    /// Lower endpoint of the 95% normal interval around the contrast with `serial_standard_error`.
    pub interval_lower: Option<f64>,
    /// Upper endpoint of that interval.
    pub interval_upper: Option<f64>,
    /// Bartlett truncation lag of the Newey-West estimate; zero when it is absent.
    pub serial_bandwidth: usize,
    /// The caliper, when one was declared.
    pub caliper: Option<f64>,
    /// Uncertainty semantics.
    pub uncertainty: Arc<str>,
}

/// Observational two-decision contrast of two static regimes (unsupported, uncalibrated).
///
/// The cross-fit scores are retained so a loader recomputes every summary.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialContrastEstimate {
    /// Active regime actions.
    pub active_regime: [u8; 2],
    /// Control regime actions.
    pub control_regime: [u8; 2],
    /// Cross-fit active minus control.
    pub contrast: f64,
    /// Cross-fit active-regime mean.
    pub active_mean: f64,
    /// Cross-fit control-regime mean.
    pub control_mean: f64,
    /// Subjects scored.
    pub evaluated_rows: u64,
    /// Smallest regime-action probability used in a score.
    pub minimum_action_probability: f64,
    /// Declared positivity floor.
    pub min_probability: f64,
    /// Standard error of the mean contrast score (normal-interval report).
    pub standard_error: Option<f64>,
    /// Normal interval (normal-interval report).
    pub interval: Option<[f64; 2]>,
    /// Nominal central coverage of the interval.
    pub coverage: f64,
    /// Cross-fit active score of each subject.
    pub active_scores: Arc<[f64]>,
    /// Cross-fit control score of each subject.
    pub control_scores: Arc<[f64]>,
    /// Active minus control score of each subject.
    pub contrast_scores: Arc<[f64]>,
    /// Uncertainty semantics.
    pub uncertainty: Arc<str>,
    /// Nuisance fits computed.
    pub fits_computed: u64,
    /// Nuisance fits reused.
    pub fits_reused: u64,
}

/// Longitudinal regime value or additive MSM summary over subject histories.
#[derive(Clone, Debug, PartialEq)]
pub struct LongitudinalRegimeEstimate {
    /// Named method: IPW, g-formula, sequential DR, or additive MSM.
    pub method: Arc<str>,
    /// Exact graphless longitudinal support after observed support and interval checks.
    pub graphless_support_status: Option<crate::support::CellStatus>,
    /// Identity of the caller rule whose actions were frozen in the query.
    pub rule_id: Option<Arc<str>>,
    /// Stable caller-declared rule version.
    pub rule_version: Option<Arc<str>>,
    /// Caller-declared source or provenance of the rule implementation.
    pub rule_provenance: Option<Arc<str>>,
    /// Regime mean, or the additive MSM intercept for the MSM method.
    pub value: f64,
    /// Independent-subject score SE for IPW or subject-clustered intercept SE for MSM.
    pub value_standard_error: Option<f64>,
    /// Pointwise 95% interval for the IPW regime value or MSM intercept when
    /// independent-subject support passes the corresponding calibrated floor.
    pub value_interval_95: Option<[f64; 2]>,
    /// Pointwise 95% intervals for additive MSM period coefficients.
    pub period_intervals_95: Option<Arc<[[f64; 2]]>>,
    /// Specific reason the regime value has no interval, when applicable.
    pub interval_reason: Option<Arc<str>>,
    /// Effective sample size: matching weighted histories for IPW, subject count for g-formula.
    pub effective_sample_size: f64,
    /// Fraction of enrolled subjects with observed matching histories for IPW; one for g-formula predictions.
    pub matched_observed_fraction: f64,
    /// Largest cumulative inverse-probability weight for IPW; one for g-formula.
    pub maximum_weight: f64,
    /// Minimum prescribed action probability across decisions.
    pub minimum_action_probability: f64,
    /// Minimum uncensored probability across decisions.
    pub minimum_censoring_probability: f64,
    /// Explicit uncertainty semantics.
    pub uncertainty: Arc<str>,
    /// Provenance of supplied sequential probabilities.
    pub probability_ownership: Arc<str>,
    /// Additive MSM period coefficients, absent for regime-value methods.
    pub period_effects: Option<Arc<[f64]>>,
    /// Pointwise subject-clustered CR1 standard errors for MSM coefficients.
    pub standard_errors: Option<Arc<[f64]>>,
    /// Stabilizing numerator probabilities used by the MSM.
    pub stabilizing_numerator_probabilities: Option<Arc<[f64]>>,
    /// Number of observed terminal outcomes entering the MSM.
    pub observed_subjects: Option<usize>,
}

/// Doubly robust held-out policy value and paired row-score uncertainty.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyValueEstimate {
    /// Net policy value per evaluation subject.
    pub policy_value: f64,
    /// Net reference value per evaluation subject.
    pub reference_value: f64,
    /// Paired incremental net value.
    pub incremental_value: f64,
    /// Relative value gap, reference minus policy.
    pub relative_value_gap: f64,
    /// Fraction of evaluation rows assigned treatment by the policy.
    pub treatment_rate: f64,
    /// Sum of policy costs over evaluation rows.
    pub total_cost: f64,
    /// Row-score standard errors; assume independent subjects.
    pub policy_standard_error: f64,
    /// Reference-value row-score standard error.
    pub reference_standard_error: f64,
    /// Paired incremental-value row-score standard error.
    pub incremental_standard_error: f64,
    /// Calibrated pointwise 95% intervals when independent held-out evaluation
    /// and empirical assignment support requirements are met.
    pub policy_interval_95: Option<[f64; 2]>,
    /// Paired policy-minus-reference pointwise 95% interval when supported.
    pub incremental_interval_95: Option<[f64; 2]>,
    /// Prediction ownership declaration retained with the result.
    pub prediction_ownership: Arc<str>,
    /// Minimum known randomized propensity.
    pub propensity_min: f64,
    /// Maximum known randomized propensity.
    pub propensity_max: f64,
    /// Method-specific row-score uncertainty semantics.
    pub uncertainty: Arc<str>,
    /// Held-out uplift across descending frozen score bins, when requested.
    pub uplift_bins: Vec<antecedent_estimate::policy_value::UpliftBinScore>,
    /// Point-only action effects versus control within fixed baseline strata.
    pub multi_action_cate: Vec<antecedent_estimate::policy_value::MultiActionCatePoint>,
    /// Simultaneous regret bound relative to the prespecified candidate class.
    pub regret: Option<antecedent_estimate::policy_value::FixedCandidateRegret>,
}

/// Full-grid 95% difference band from randomized subject resampling.
#[derive(Clone, Debug, PartialEq)]
pub struct SurvivalDifferenceBand {
    /// Observed event-time grid, including zero and the horizon.
    pub times: Arc<[f64]>,
    /// Treatment-minus-control curve on the shared grid.
    pub difference: Arc<[f64]>,
    /// Simultaneous lower endpoints.
    pub lower: Arc<[f64]>,
    /// Simultaneous upper endpoints.
    pub upper: Arc<[f64]>,
    /// Bootstrap replicates satisfying the original estimator support contract.
    pub replicates_ok: u32,
    /// Exact graphless simultaneous-band support, when the published band and
    /// arm support match a supported band row. Absent for off-axis bands.
    pub support_status: Option<crate::support::CellStatus>,
}

/// Randomized survival or competing-risk result on a shared time grid.
#[derive(Clone, Debug, PartialEq)]
pub struct SurvivalEstimate {
    /// Event-time grid including zero and the restriction horizon.
    pub times: Arc<[f64]>,
    /// Control-arm survival or target-cause cumulative incidence.
    pub control: Arc<[f64]>,
    /// Treated-arm survival or target-cause cumulative incidence.
    pub treated: Arc<[f64]>,
    /// Control restricted mean survival time for a survival query.
    pub rmst_control: Option<f64>,
    /// Treated restricted mean survival time for a survival query.
    pub rmst_treated: Option<f64>,
    /// Target cause for cumulative incidence; absent for survival/RMST.
    pub target_cause: Option<i64>,
    /// Restriction horizon.
    pub tau: f64,
    /// Smallest event risk set seen in either arm.
    pub minimum_event_risk_set: Option<usize>,
    /// Explicit uncertainty statement for the scalar contrasts.
    pub uncertainty: Arc<str>,
    /// Pointwise 95% subject-bootstrap RMST difference interval, when requested.
    pub rmst_difference_interval: Option<[f64; 2]>,
    /// Pointwise 95% subject-bootstrap difference at the restriction horizon.
    pub difference_at_tau_interval: Option<[f64; 2]>,
    /// Requested arm-stratified subject-bootstrap draws.
    pub bootstrap_replicates_requested: Option<u32>,
    /// Draws satisfying the estimator's original support contract.
    pub bootstrap_replicates_ok: Option<u32>,
    /// Realized control and treated subject counts for an exact graphless support.
    pub assignment_counts: [usize; 2],
    /// Caller-supplied fixed censoring function, when the IPCW route was used.
    pub censoring_survival_provenance: Option<Arc<str>>,
    /// Simultaneous full-grid difference band; separate from scalar intervals.
    pub difference_band: Option<SurvivalDifferenceBand>,
    /// Why an explicit bootstrap request did not produce a full-grid band.
    pub band_unavailable_reason: Option<Arc<str>>,
}

/// Identification certificate retained from the actual execution, including class atoms.
#[derive(Clone, Debug)]
pub struct AnalysisIdentification {
    /// Full point or completion-envelope artifact in its original coordinates.
    pub identification: crate::Identification,
    /// Query whose functional was estimated.
    pub query: antecedent_core::CausalQuery,
    /// Supplied graph class, preserved even when fully oriented.
    pub graph_class: crate::GraphClass,
}

/// Row-weight target population of a retarget, bound to the data snapshot and
/// score table it was computed on.
///
/// The `RowWeights` population references [`Self::target_weights`], so the
/// target identity and every claim over the result name exactly this
/// weighting. The weights travel in the exported contract so a consumer can
/// re-derive the identity.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RowWeightsBinding {
    /// Weights in score-table row order.
    pub weights: std::sync::Arc<[f64]>,
    /// Declared parents of the weights.
    pub depends_on: std::sync::Arc<[VariableId]>,
    /// Rehashable target-weights identity payload.
    pub identity: antecedent_io::TargetWeightsIdentityWire,
    /// Digest of [`Self::identity`].
    pub target_weights: antecedent_core::SemanticDigest,
}

impl StudyResult {
    pub(crate) fn invalidate_authority(&mut self) {
        if matches!(&self.authority, RuntimeRefutationAuthority::Sealed { .. }) {
            self.authority = RuntimeRefutationAuthority::Invalidated;
        }
    }

    /// Record bytes borrowed from host-owned input buffers during ingestion.
    ///
    /// This is host performance accounting; it does not alter the executed target,
    /// scientific result, or checked refutation authority.
    pub fn set_borrowed_input_bytes(&mut self, bytes: Option<u64>) {
        self.view.performance.bytes_borrowed = bytes;
    }

    /// Whether the response's retained checks describe individual completions
    /// rather than establishing a verdict for the whole response.
    #[must_use]
    pub fn response_checks_are_scoped(&self) -> bool {
        use RuntimeRefutationAuthority::Sealed;
        use RuntimeRefutationOrigin::{Route, ScopedCompletion};
        if self.refutations.is_empty() || self.response().is_none() {
            return false;
        }
        matches!(&self.authority, Sealed { origin: ScopedCompletion, .. })
            || (matches!(&self.authority, Sealed { origin: Route, .. })
                && self.structural_response.is_some())
    }

    /// Reservation held with this result's retained output, when accounted by its engine.
    /// This is admitted capacity, not a measurement of process resident memory.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> Option<u64> {
        if self.retained_resources.is_empty() {
            return None;
        }
        self.retained_resources
            .iter()
            .try_fold(0_u64, |bytes, lease| bytes.checked_add(lease.remainder().bytes()))
    }

    /// The posterior predictive replicates the study retained at fit time, when it declared
    /// bounds for them with [`crate::StudyBuilder::bayesian_predictive_replicates`].
    #[must_use]
    pub fn predictive_replicates(&self) -> Option<&crate::analysis::RetainedPredictiveReplicates> {
        self.predictive_replicates.as_deref()
    }

    /// Target population a row-weight retarget recorded, when it differs from
    /// the prepared query.
    #[must_use]
    pub fn retarget_population(&self) -> Option<antecedent_core::TargetPopulation> {
        self.extras.row_weights.as_ref().map(RowWeightsBinding::population)
    }
}

impl RowWeightsBinding {
    /// The population this binding defines.
    #[must_use]
    pub fn population(&self) -> antecedent_core::TargetPopulation {
        antecedent_core::TargetPopulation::RowWeights {
            weights: *self.target_weights.as_bytes(),
            depends_on: std::sync::Arc::clone(&self.depends_on),
        }
    }
}

/// One 95% pointwise interval for a declared interference exposure contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct InterferencePointwiseInterval {
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Independent-cluster Neyman standard error.
    pub standard_error: f64,
    /// Welch--Satterthwaite degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Control/treated clusters or low/high saturation clusters.
    pub first_stage_arm_clusters: [usize; 2],
}

/// Observed support and uncertainty status for a retained interference query.
#[derive(Clone, Debug, PartialEq)]
pub struct InterferenceInference {
    /// Construction identifier.
    pub method: &'static str,
    /// Exact graphless interference support after observed cluster support and
    /// a published pointwise interval pass; `None` when the interval is withheld
    /// or the design/exposure support does not match a supported row.
    pub graphless_support_status: Option<crate::support::CellStatus>,
    /// Pointwise interval, when its support conditions pass.
    pub interval: Option<InterferencePointwiseInterval>,
    /// Reason a point estimate has no interval.
    pub interval_unavailable_reason: Option<&'static str>,
    /// Units observed at the baseline exposure.
    pub from_exposed_units: usize,
    /// Units observed at the active exposure.
    pub to_exposed_units: usize,
    /// Clusters observed at the baseline exposure.
    pub from_exposed_clusters: usize,
    /// Clusters observed at the active exposure.
    pub to_exposed_clusters: usize,
}

/// The retained posterior of an unsupported Bayesian randomized-design row: coordinates, means,
/// equal-tailed 95% credible intervals, the covariance of the joint draws and named summaries.
/// Its uncertainty is `posterior` and its calibration `not_measured`; it is never supported.
pub type PosteriorRowEstimate = antecedent_io::PosteriorRowWire;

/// The typed answer of one query family, published beside a result's scalar
/// estimate.
///
/// A query produces one family of answer, so the families are variants rather
/// than a row of optional fields of which at most one is set. The Bayesian
/// posterior and the atom-level structural response accompany any family and
/// live on [`StudyResult`] itself.
#[allow(
    clippy::large_enum_variant,
    reason = "a result holds exactly one payload; boxing would change the public result API"
)]
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum ResultPayload {
    /// Only the scalar estimate: an average effect, a conditional effect or a
    /// path-specific effect.
    #[default]
    Scalar,
    /// Function-valued causal response for
    /// [`CausalQuery::Response`](antecedent_core::CausalQuery::Response).
    Response(CausalResponse),
    /// Full interventional distribution for
    /// [`CausalQuery::Distribution`](antecedent_core::CausalQuery::Distribution).
    Distribution(InterventionalDistributionEstimate),
    /// Mediation decomposition and its horizon-indexed grid. The grid is
    /// present for temporal mediation, including single-horizon results; it
    /// does not imply a joint posterior.
    Mediation {
        /// Temporal / static mediation decomposition.
        decomposition: Option<TemporalMediationEstimate>,
        /// Horizon-indexed decomposition.
        grid: Option<TemporalMediationGrid>,
    },
    /// Unit-level ITE for a counterfactual query.
    Counterfactual(IteResult),
    /// Unit-level contrasts of a counterfactual query with several outcomes, a joint baseline
    /// or nested interventions.
    CounterfactualContrasts(Box<crate::gcm::CounterfactualContrasts>),
    /// Anomaly scores for an anomaly-attribution query.
    Anomaly(Vec<AnomalyScores>),
    /// Change-attribution result.
    ChangeAttribution(ChangeAttributionResult),
    /// Mechanism-change detections.
    MechanismChange(Vec<MechanismChangeDetection>),
    /// Unit-change attribution.
    UnitChange(UnitChangeResult),
    /// Trial-to-target transport estimate.
    Transport(TransportEffectEstimate),
    /// Design-based interference estimate and its pointwise inference.
    Interference {
        /// Exposure-contrast estimate.
        estimate: Option<InterferenceEstimate>,
        /// Independent-cluster pointwise inference and observed exposure support.
        inference: Option<InterferenceInference>,
    },
    /// Design-based intention-to-treat result for a randomized experiment query.
    RandomizedEffect(RandomizedEffectEstimate),
    /// Balanced two-period panel difference-in-differences result.
    PanelDid(PanelDidEstimate),
    /// Synthetic-control point result with donor and placebo diagnostics.
    SyntheticControl(SyntheticControlEstimate),
    /// Synthetic `DiD` point result and both fitted weight vectors.
    SyntheticDid(SyntheticDidEstimate),
    /// Fixed-bandwidth fuzzy RD or regression-kink point result.
    LocalPolynomialRatio(LocalPolynomialRatioEstimate),
    /// Held-out doubly robust policy evaluation; never an ATE.
    PolicyValue(PolicyValueEstimate),
    /// Conditional continuous-dose response grid; not a policy value.
    ContinuousDoseResponse(ContinuousDoseResponseEstimate),
    /// Randomized right-censored survival or competing-risk curve.
    Survival(SurvivalEstimate),
    /// Unsupported Bayesian posterior of a randomized-design row (survival contrast, fixed
    /// group-dose policy value): means, credible intervals and the joint covariance of the
    /// retained draws, with calibration `not_measured`.
    PosteriorRow(PosteriorRowEstimate),
    /// Prespecified longitudinal regime value.
    LongitudinalRegime(LongitudinalRegimeEstimate),
    /// Observational two-decision regime contrast.
    SequentialContrast(SequentialContrastEstimate),
    /// Within-period history-matched contrast.
    HistoryMatch(HistoryMatchEstimate),
    /// Excursion contrasts of a sequentially randomized design, or regime survival.
    TemporalDesign(Box<TemporalDesignEstimate>),
    /// Sharp bounds on the average effect under a binary instrument: an identified set, never a
    /// point.
    IvBounds(Box<antecedent_estimate::IvBoundsEstimate>),
    /// Noncausal association of the observed data: never an intervention effect, never a
    /// ranking of actions.
    Association(Box<crate::analysis::association::AssociationPayload>),
    /// Rollout design effects with whole-procedure uncertainty.
    QuasiRollout(Box<QuasiRolloutEstimate>),
    /// Structure, structure class or structure posterior a compiled search returned.
    Discovery(Box<crate::analysis::structure_route::DiscoveryPayload>),
    /// Attribution on fitted mechanisms, with their selection record.
    Attribution(Box<crate::analysis::structure_route::AttributionPayload>),
    /// Simulation of fitted mechanisms, with their selection record.
    Simulation(Box<crate::analysis::structure_route::SimulationPayload>),
}

impl ResultPayload {
    /// A mediation payload, or [`Self::Scalar`] when neither part is present.
    #[must_use]
    pub fn from_mediation(
        decomposition: Option<TemporalMediationEstimate>,
        grid: Option<TemporalMediationGrid>,
    ) -> Self {
        if decomposition.is_none() && grid.is_none() {
            Self::Scalar
        } else {
            Self::Mediation { decomposition, grid }
        }
    }

    /// An interference payload, or [`Self::Scalar`] when neither part is present.
    #[must_use]
    pub fn from_interference(
        estimate: Option<InterferenceEstimate>,
        inference: Option<InterferenceInference>,
    ) -> Self {
        if estimate.is_none() && inference.is_none() {
            Self::Scalar
        } else {
            Self::Interference { estimate, inference }
        }
    }

    /// The one payload among `candidates` that is present, or [`Self::Scalar`].
    ///
    /// A query produces one family of answer; more than one present candidate
    /// is a producer bug, refused in debug builds.
    #[must_use]
    pub(crate) fn one_of<const N: usize>(candidates: [Option<Self>; N]) -> Self {
        let mut present = candidates.into_iter().flatten();
        let payload = present.next().unwrap_or(Self::Scalar);
        debug_assert!(present.next().is_none(), "a result carries one payload family");
        payload
    }

    /// The mediation decomposition by value.
    #[must_use]
    pub fn into_mediation(self) -> Option<TemporalMediationEstimate> {
        match self {
            Self::Mediation { decomposition, .. } => decomposition,
            _ => None,
        }
    }

    /// The mediation grid by value.
    #[must_use]
    pub fn into_mediation_grid(self) -> Option<TemporalMediationGrid> {
        match self {
            Self::Mediation { grid, .. } => grid,
            _ => None,
        }
    }

    /// Function-valued response, mutably.
    pub const fn response_mut(&mut self) -> Option<&mut CausalResponse> {
        match self {
            Self::Response(value) => Some(value),
            _ => None,
        }
    }

    /// Mediation decomposition, mutably.
    pub const fn mediation_mut(&mut self) -> Option<&mut TemporalMediationEstimate> {
        match self {
            Self::Mediation { decomposition, .. } => decomposition.as_mut(),
            _ => None,
        }
    }

    /// Mediation decomposition.
    #[must_use]
    pub const fn mediation(&self) -> Option<&TemporalMediationEstimate> {
        match self {
            Self::Mediation { decomposition, .. } => decomposition.as_ref(),
            _ => None,
        }
    }

    /// Horizon-indexed mediation grid.
    #[must_use]
    pub const fn mediation_grid(&self) -> Option<&TemporalMediationGrid> {
        match self {
            Self::Mediation { grid, .. } => grid.as_ref(),
            _ => None,
        }
    }

    /// Anomaly scores.
    #[must_use]
    pub fn anomaly(&self) -> Option<&[AnomalyScores]> {
        match self {
            Self::Anomaly(value) => Some(value),
            _ => None,
        }
    }

    /// Mechanism-change detections.
    #[must_use]
    pub fn mechanism_change(&self) -> Option<&[MechanismChangeDetection]> {
        match self {
            Self::MechanismChange(value) => Some(value),
            _ => None,
        }
    }

    /// Interference estimate.
    #[must_use]
    pub const fn interference_estimate(&self) -> Option<&InterferenceEstimate> {
        match self {
            Self::Interference { estimate, .. } => estimate.as_ref(),
            _ => None,
        }
    }

    /// Interference pointwise inference and exposure support.
    #[must_use]
    pub const fn interference_inference(&self) -> Option<&InterferenceInference> {
        match self {
            Self::Interference { inference, .. } => inference.as_ref(),
            _ => None,
        }
    }
}

// A point-time completion witness. `atom_index` addresses the actual structural
// response produced by this fit; its full digest is independent of enumeration order.
#[derive(Clone, Debug)]
pub(crate) struct TemporalResponseWitness {
    pub(crate) horizon_index: usize,
    pub(crate) horizon: u32,
    pub(crate) atom_index: usize,
    pub(crate) weight: f64,
    pub(crate) completion_digest: [u8; 32],
}

#[derive(Debug)]
pub(crate) struct TemporalResponseWitnessSet {
    pub(crate) entries: Vec<TemporalResponseWitness>,
    pub(crate) _leases: Vec<antecedent_core::execution::AdmissionLease>,
}

#[derive(Debug)]
pub(crate) struct TemporalPosteriorWitnessMember {
    pub(crate) outer_atom: u64,
    pub(crate) outer_weight: f64,
    pub(crate) witnesses: Arc<TemporalResponseWitnessSet>,
    pub(crate) structural: StructuralResponseMixture,
    pub(crate) _resources: Vec<Arc<antecedent_core::execution::AdmissionLease>>,
}

#[derive(Debug)]
pub(crate) struct TemporalPosteriorWitnessSet {
    pub(crate) members: Vec<TemporalPosteriorWitnessMember>,
    pub(crate) _lease: antecedent_core::execution::AdmissionLease,
}

/// The immutable graph source used by an executed posterior response.
#[derive(Debug)]
pub(crate) struct StaticPosteriorSourceWitness {
    pub(crate) source: std::sync::Weak<[antecedent_discovery::AdjacencyMask]>,
    pub(crate) _lease: antecedent_core::execution::AdmissionLease,
}

/// Actual fitted completions retained for same-prepared class refits.
#[derive(Debug)]
pub(crate) struct StaticFittedCompletions {
    pub(crate) entries: Vec<([u8; 32], u64, antecedent_core::IdentificationStatus)>,
    pub(crate) _lease: antecedent_core::execution::AdmissionLease,
}

#[derive(Debug)]
pub(crate) struct StaticClassAtomWitness {
    pub(crate) graph_key: u64,
    pub(crate) outer_weight: u64,
    pub(crate) completions: StaticFittedCompletions,
}

#[derive(Debug)]
pub(crate) struct StaticClassWitnessSet {
    pub(crate) atom_kind: antecedent_discovery::GraphPosteriorAtomKind,
    pub(crate) n_vars: usize,
    pub(crate) identification_cache:
        std::sync::Weak<dyn std::fmt::Debug + Send + Sync + std::panic::RefUnwindSafe>,
    pub(crate) source_atoms: std::sync::Weak<[antecedent_discovery::AdjacencyMask]>,
    pub(crate) source_marks: Option<std::sync::Weak<[antecedent_discovery::AdjacencyMask]>>,
    pub(crate) atoms: Vec<StaticClassAtomWitness>,
    pub(crate) samples: Vec<(u64, u64)>,
    pub(crate) _lease: antecedent_core::execution::AdmissionLease,
}

// Private runtime evidence, installed only by the scoped target operation after it
// has completed its own refits. The selector and original result binding are part
// of the digest, so editing a public report invalidates the marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeRefutationOrigin {
    Route,
    WholeTarget,
    ScopedCompletion,
}

#[derive(Clone, Debug)]
pub(crate) enum RuntimeRefutationAuthority {
    Fresh,
    Sealed { origin: RuntimeRefutationOrigin, owner: Arc<()> },
    Invalidated,
}

/// Caller-readable result fields. Mutable access invalidates checked authority.
#[derive(Clone, Debug)]
pub struct StudyResultView {
    /// Logical program and data classification.
    pub logical_plan: LogicalAnalysisPlanRecord,
    /// Physical plan record.
    pub physical_plan: PhysicalExecutionPlanRecord,
    /// Execution performance and backend facts.
    pub performance: ExecutionPerformanceRecord,
    /// Full identification result.
    pub identification: IdentificationResult,
    /// Complete identification certificate, when supplied.
    pub certificate: Option<AnalysisIdentification>,
    /// The estimand used for the fit.
    pub estimand: IdentifiedEstimand,
    /// Structural atom and mass result.
    pub structural_response: Option<StructuralResponseMixture>,
    /// Bayesian posterior, when present.
    pub posterior: Option<CausalPosterior>,
    /// Refutation reports.
    pub refutations: Vec<RefutationReport>,
    /// Predictive check reports.
    pub predictive_checks: Vec<PredictiveCheckReport>,
    /// Diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Scientific provenance.
    pub provenance: ProvenanceGraph,
    /// Declared structure source.
    pub structure_source: crate::support::StructureSource,
    /// Treatment variable, if applicable.
    pub treatment: Option<VariableId>,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Population bindings.
    pub population_registry: Option<antecedent_core::PopulationRegistry>,
}

impl std::ops::Deref for StudyResult {
    type Target = StudyResultView;
    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

impl std::ops::DerefMut for StudyResult {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.invalidate_authority();
        &mut self.view
    }
}

/// End-to-end analysis result.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct StudyResult {
    pub(crate) retained_resources: Vec<Arc<antecedent_core::execution::AdmissionLease>>,
    // The actual immutable input passed to the executed operation, not a caller's later declaration.
    pub(crate) executed_input: Option<crate::analysis::target_validation::ExecutedInputIdentity>,
    pub(crate) temporal_response_witnesses: Option<Arc<TemporalResponseWitnessSet>>,
    pub(crate) temporal_posterior_witnesses: Option<Arc<TemporalPosteriorWitnessSet>>,
    pub(crate) static_class_witnesses: Option<Arc<StaticClassWitnessSet>>,
    pub(crate) static_posterior_source: Option<Arc<StaticPosteriorSourceWitness>>,
    /// Complete target-native reports retained as descriptive diagnostics only.
    pub(crate) full_target_diagnostics:
        Option<Arc<antecedent_validate::target_native::RetainedTargetReports>>,
    // An operation-issued digest of the checked scope, target and report body. This
    // private marker must never be reconstructed from portable descriptors.
    pub(crate) authority: RuntimeRefutationAuthority,
    pub(crate) view: StudyResultView,
    /// Point estimate + uncertainty (frequentist, or Bayesian posterior mean summary).
    ///
    /// For [`CausalQuery::Distribution`](antecedent_core::CausalQuery::Distribution) this holds the
    /// interventional mean of the first numeric outcome when defined (`ate` field), else NaN.
    /// Its `se_bootstrap` is the mean's bootstrap SE; do not form `ate ± z·se` for a
    /// probability — the bounded per-atom intervals are
    /// [`InterventionalDistributionEstimate::atom_uncertainty`], and a binary outcome's
    /// mean interval is [`InterventionalDistributionEstimate::mean_interval`].
    pub(crate) estimate: PrimaryEstimate,
    /// The typed answer the query family published beside the scalar estimate.
    ///
    /// Read it through [`Self::response`], [`Self::distribution`] and the other
    /// accessors, or match on it directly. The Bayesian [`Self::posterior`] and
    /// the atom [`Self::structural_response`] accompany any family and stay
    /// outside it.
    pub(crate) payload: ResultPayload,
    /// Posterior predictive replicates retained at fit time when the study declared bounds
    /// for them; read through [`Self::predictive_replicates`].
    pub(crate) predictive_replicates: Option<Arc<crate::analysis::RetainedPredictiveReplicates>>,
    /// Support-matrix evidence contract that produced this result.
    ///
    /// A `not_supported` cell refuses the build in a release build; a development build
    /// runs it, carries `not_supported` and lists it in [`Self::would_refuse`]. Downstream consumers must not treat a number as supported unless this is
    /// [`crate::support::CellStatus::Supported`]. `None` when the query is not on
    /// the public axis.
    pub(crate) support_status: Option<crate::support::CellStatus>,
    /// Prepared contract this result was executed under, stamped by the
    /// [`crate::PreparedStudy`] entry point that produced it. `None` for
    /// results no prepared handle executed; those cannot be exported as claims.
    pub(crate) executed_contract: Option<ExecutedContract>,
    /// Identity of the lowered program whose dispatch produced this result.
    pub(crate) lowered_identity: Option<antecedent_core::SemanticDigest>,
    /// The producing program's scientific data and uncertainty classification.
    /// Dedicated response posteriors need not occupy the scalar posterior slot.
    pub(crate) lowered_quadrant: Option<antecedent_validate::target_native::TargetQuadrant>,
    /// The optional report layers of this result.
    pub(crate) extras: ResultExtras,
}

/// The optional report layers of a [`StudyResult`], held together.
#[derive(Clone, Debug)]
pub(crate) struct ResultExtras {
    /// Candidate-selection screen recorded for a prepared batch family.
    pub(crate) candidate_selection: Option<crate::analysis::CandidateSelection>,
    /// How this execution formed its reported interval.
    pub(crate) interval: Option<IntervalBinding>,
    /// Row-weight population recorded by a nonconstant retarget.
    pub(crate) row_weights: Option<RowWeightsBinding>,
    /// Missing-data policy, assumption and row accounting of this execution.
    pub(crate) missing_data: Option<MissingDataReport>,
    /// Names of caller-supplied custom validators that ran on this execution.
    pub(crate) custom_validator_names: Vec<std::sync::Arc<str>>,
    /// What a development build ran although a release build would have refused it. Always
    /// empty in a release build.
    pub(crate) would_refuse: Vec<antecedent_core::WouldRefuse>,
}

impl StudyResult {
    /// What a development build ran although a release build would have refused it, as
    /// `{code, detail, missing}` entries; empty when the route is supported and always empty in
    /// a release build.
    #[must_use]
    pub fn would_refuse(&self) -> &[antecedent_core::WouldRefuse] {
        &self.extras.would_refuse
    }

    /// Record the entries the build and this execution collected (development builds only).
    pub(crate) fn stamp_would_refuse(
        &mut self,
        build: &[antecedent_core::WouldRefuse],
        ctx: &antecedent_core::ExecutionContext,
    ) {
        for entry in build.iter().cloned().chain(ctx.would_refuse()) {
            if !self.extras.would_refuse.contains(&entry) {
                self.extras.would_refuse.push(entry);
            }
        }
    }

    /// The execution contract attached by the engine, when present.
    ///
    /// The stamp is read-only: callers cannot replace an execution's validation suite.
    ///
    /// ```compile_fail
    /// fn replace_stamp(result: &mut antecedent::StudyResult) {
    ///     result.executed_contract = None;
    /// }
    /// ```
    #[must_use]
    pub const fn executed_contract(&self) -> Option<&ExecutedContract> {
        self.executed_contract.as_ref()
    }

    /// The primary estimate; it always agrees with [`Self::payload`].
    #[must_use]
    pub const fn estimate(&self) -> &PrimaryEstimate {
        &self.estimate
    }

    /// Mutable estimate, for tests that deliberately construct a tampered or edge-case result.
    #[doc(hidden)]
    pub fn estimate_mut(&mut self) -> &mut PrimaryEstimate {
        self.invalidate_authority();
        &mut self.estimate
    }

    /// Candidate-selection record, when the study selected among candidates.
    #[must_use]
    pub const fn candidate_selection(&self) -> &Option<crate::analysis::CandidateSelection> {
        &self.extras.candidate_selection
    }

    /// How the reported interval was formed, when the execution reported one.
    #[must_use]
    pub const fn interval(&self) -> &Option<IntervalBinding> {
        &self.extras.interval
    }

    /// Row-weight binding, when the analysis was weighted.
    #[must_use]
    pub const fn row_weights(&self) -> &Option<RowWeightsBinding> {
        &self.extras.row_weights
    }

    /// Missing-data report, when the analysis handled missing values.
    #[must_use]
    pub const fn missing_data(&self) -> &Option<MissingDataReport> {
        &self.extras.missing_data
    }

    /// Names of the custom validators that ran.
    #[must_use]
    pub fn custom_validator_names(&self) -> &[std::sync::Arc<str>] {
        &self.extras.custom_validator_names
    }

    /// The route-specific result payload the estimate was drawn from.
    #[must_use]
    pub const fn payload(&self) -> &ResultPayload {
        &self.payload
    }

    /// Support standing of the estimand's cell, when the route reports one.
    #[must_use]
    pub const fn support_status(&self) -> Option<crate::support::CellStatus> {
        self.support_status
    }

    /// The provenance records behind the route this result executed: the identifier and
    /// estimator of [`StudyResultView::logical_plan`], each with its published sources, the
    /// deviations its record documents, what its bounded review leaves out, and its id and
    /// digest. Ids that name no record are listed as unlinked.
    ///
    /// Descriptive only, like the calibration slot: it changes no number or support status.
    /// [`StudyResultView::provenance`] is the execution's provenance graph, a different thing.
    #[must_use = "the provenance is computed on each call"]
    pub fn method_provenance(&self) -> antecedent_io::method_provenance::MethodProvenanceWire {
        antecedent_io::method_provenance::method_provenance(
            self.logical_plan.identifier.as_deref(),
            self.logical_plan.estimator.as_deref(),
        )
    }

    /// The sources to cite for this result, de-duplicated, each with the caveats of the
    /// records citing it (see [`Self::method_provenance`]).
    #[must_use = "the citations are computed on each call"]
    pub fn citations(&self) -> Vec<antecedent_io::method_provenance::CitationWire> {
        self.method_provenance().citations()
    }

    pub(crate) fn stamp_lowered_program(
        &mut self,
        program: &crate::planner::lowered::LoweredProgram,
    ) {
        self.lowered_identity = program.semantic_identity();
        self.lowered_quadrant = Some(program.refutation_quadrant());
    }

    /// Route-native diagnostic records the result carries, in diagnostic order.
    ///
    /// Passed and failed checks are both included, each with its measured or unreported
    /// unevaluable mass, so the records match what a refusal of the same route reports.
    #[must_use = "dropped records are not on the result"]
    pub fn route_diagnostics(&self) -> Vec<antecedent_core::diagnostic::RouteNativeDiagnostic> {
        self.diagnostics
            .iter()
            .filter_map(antecedent_core::diagnostic::RouteNativeDiagnostic::from_diagnostic)
            .collect()
    }

    /// Identity of the lowered program whose dispatch produced this result.
    ///
    /// `None` for a result that no lowered dispatch produced or whose runtime
    /// callback declaration cannot identify its live implementation.
    #[must_use]
    pub const fn lowered_program_identity(&self) -> Option<antecedent_core::SemanticDigest> {
        self.lowered_identity
    }
}

/// The bibliography of a study: the sources of every result, de-duplicated across results.
///
/// A source cited by several results appears once, with every record that cites it and every
/// caveat those records document. See [`StudyResult::method_provenance`].
#[must_use = "the bibliography is computed on each call"]
pub fn bibliography<'a, I>(results: I) -> Vec<antecedent_io::method_provenance::CitationWire>
where
    I: IntoIterator<Item = &'a StudyResult>,
{
    let provenances: Vec<_> = results.into_iter().map(StudyResult::method_provenance).collect();
    antecedent_io::method_provenance::bibliography(&provenances)
}

/// Reference and by-value accessors of every single-variant payload on [`ResultPayload`], and
/// the matching delegates on [`StudyResult`], from one list.
///
/// `plain` variants hold the value directly; `boxed` variants hold it behind a `Box`.
macro_rules! payload_accessors {
    (
        plain: [$($pvariant:ident => $pname:ident, $pinto:ident: $pty:ty;)*]
        boxed: [$($bvariant:ident => $bname:ident, $binto:ident: $bty:ty;)*]
    ) => {
        impl ResultPayload {
            $(
                #[doc = concat!("The `", stringify!($pname), "` payload, when the answer is one.")]
                #[must_use]
                pub const fn $pname(&self) -> Option<&$pty> {
                    match self {
                        Self::$pvariant(value) => Some(value),
                        _ => None,
                    }
                }

                #[doc = concat!("The `", stringify!($pname), "` payload by value.")]
                #[must_use]
                pub fn $pinto(self) -> Option<$pty> {
                    match self {
                        Self::$pvariant(value) => Some(value),
                        _ => None,
                    }
                }
            )*
            $(
                #[doc = concat!("The `", stringify!($bname), "` payload, when the answer is one.")]
                #[must_use]
                pub fn $bname(&self) -> Option<&$bty> {
                    match self {
                        Self::$bvariant(value) => Some(value),
                        _ => None,
                    }
                }

                #[doc = concat!("The `", stringify!($bname), "` payload by value.")]
                #[must_use]
                pub fn $binto(self) -> Option<$bty> {
                    match self {
                        Self::$bvariant(value) => Some(*value),
                        _ => None,
                    }
                }
            )*
        }

        impl StudyResult {
            $(
                #[doc = concat!("The `", stringify!($pname), "` payload, when the query produced one.")]
                #[must_use]
                pub fn $pname(&self) -> Option<&$pty> {
                    self.payload.$pname()
                }

                #[doc = concat!("The `", stringify!($pname), "` payload by value, consuming the result.")]
                #[must_use]
                pub fn $pinto(self) -> Option<$pty> {
                    self.payload.$pinto()
                }
            )*
            $(
                #[doc = concat!("The `", stringify!($bname), "` payload, when the query produced one.")]
                #[must_use]
                pub fn $bname(&self) -> Option<&$bty> {
                    self.payload.$bname()
                }

                #[doc = concat!("The `", stringify!($bname), "` payload by value, consuming the result.")]
                #[must_use]
                pub fn $binto(self) -> Option<$bty> {
                    self.payload.$binto()
                }
            )*
        }
    };
}

payload_accessors! {
    plain: [
    Response => response, into_response: CausalResponse;
    Distribution => distribution, into_distribution: InterventionalDistributionEstimate;
    Counterfactual => counterfactual, into_counterfactual: IteResult;
    ChangeAttribution => change_attribution, into_change_attribution: ChangeAttributionResult;
    UnitChange => unit_change, into_unit_change: UnitChangeResult;
    Transport => transport, into_transport: TransportEffectEstimate;
    RandomizedEffect => randomized_effect, into_randomized_effect: RandomizedEffectEstimate;
    PanelDid => panel_did, into_panel_did: PanelDidEstimate;
    SyntheticControl => synthetic_control, into_synthetic_control: SyntheticControlEstimate;
    SyntheticDid => synthetic_did, into_synthetic_did: SyntheticDidEstimate;
    LocalPolynomialRatio => local_polynomial_ratio, into_local_polynomial_ratio:
        LocalPolynomialRatioEstimate;
    PolicyValue => policy_value, into_policy_value: PolicyValueEstimate;
    ContinuousDoseResponse => continuous_dose_response, into_continuous_dose_response:
        ContinuousDoseResponseEstimate;
    Survival => survival, into_survival: SurvivalEstimate;
    PosteriorRow => posterior_row, into_posterior_row: PosteriorRowEstimate;
    LongitudinalRegime => longitudinal_regime, into_longitudinal_regime:
        LongitudinalRegimeEstimate;
    SequentialContrast => sequential_contrast, into_sequential_contrast:
        SequentialContrastEstimate;
    HistoryMatch => history_match, into_history_match: HistoryMatchEstimate;
    ]
    boxed: [
    CounterfactualContrasts => counterfactual_contrasts, into_counterfactual_contrasts:
        crate::gcm::CounterfactualContrasts;
    TemporalDesign => temporal_design, into_temporal_design: TemporalDesignEstimate;
    IvBounds => iv_bounds, into_iv_bounds: antecedent_estimate::IvBoundsEstimate;
    Association => association, into_association:
        crate::analysis::association::AssociationPayload;
    QuasiRollout => quasi_rollout, into_quasi_rollout: QuasiRolloutEstimate;
    Discovery => discovery, into_discovery: crate::analysis::structure_route::DiscoveryPayload;
    Attribution => attribution, into_attribution:
        crate::analysis::structure_route::AttributionPayload;
    Simulation => simulation, into_simulation: crate::analysis::structure_route::SimulationPayload;
    ]
}

impl StudyResult {
    /// The mediation decomposition, when the query produced one.
    #[must_use]
    pub fn mediation(&self) -> Option<&TemporalMediationEstimate> {
        self.payload.mediation()
    }

    /// The mediation decomposition by value, consuming the result.
    #[must_use]
    pub fn into_mediation(self) -> Option<TemporalMediationEstimate> {
        self.payload.into_mediation()
    }

    /// The horizon-indexed mediation grid, when the query produced one.
    #[must_use]
    pub fn mediation_grid(&self) -> Option<&TemporalMediationGrid> {
        self.payload.mediation_grid()
    }

    /// The mediation grid by value, consuming the result.
    #[must_use]
    pub fn into_mediation_grid(self) -> Option<TemporalMediationGrid> {
        self.payload.into_mediation_grid()
    }

    /// Anomaly scores when the query was anomaly attribution.
    #[must_use]
    pub fn anomaly(&self) -> Option<&[AnomalyScores]> {
        self.payload.anomaly()
    }

    /// Mechanism-change detections.
    #[must_use]
    pub fn mechanism_change(&self) -> Option<&[MechanismChangeDetection]> {
        self.payload.mechanism_change()
    }

    /// Design-based interference estimate when the query was interference.
    #[must_use]
    pub fn interference(&self) -> Option<&InterferenceEstimate> {
        self.payload.interference_estimate()
    }

    /// Independent-cluster pointwise inference and observed exposure support.
    #[must_use]
    pub fn interference_inference(&self) -> Option<&InterferenceInference> {
        self.payload.interference_inference()
    }
}

/// Prepared contract a result was executed under.
///
/// Export recompiles the contract on the handle and refuses unless every
/// identity layer — including the data snapshot the result was computed on —
/// matches this stamp. The engine records the identity scope privately; callers
/// inspect it with [`Self::identity_scope`] rather than constructing a stamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutedContract {
    /// Every contract identity layer, including the executed data snapshot.
    pub identities: antecedent_core::ContractIdentities,
    /// Snapshot identity before execution-only replay evidence is attached.
    /// This is computed from the same input at execution and permits a cheap
    /// inspection to be compared without replacing the full snapshot identity.
    pub inspection_data_snapshot: antecedent_core::SemanticDigest,
    /// Scope stamped by the prepared execution, independent of mutable result fields.
    pub(crate) identity_scope: crate::planner::lowered::ProgramIdentityScope,
    /// Validation suite whose refutations the result carries.
    pub refute: crate::RefuteSuite,
}

impl ExecutedContract {
    /// Scope recorded by the engine at execution, independent of mutable result fields.
    #[must_use]
    pub const fn identity_scope(&self) -> crate::planner::lowered::ProgramIdentityScope {
        self.identity_scope
    }
}

impl StudyResult {
    /// Primary scalar effect for display and tests.
    ///
    /// Prefer this over reading [`EffectEstimate::ate`] directly when the query may be a
    /// distribution or counterfactual: returns the interventional mean or mean ITE when
    /// present, otherwise the estimate's `ate` field. For a mediation query this is the
    /// queried contrast (total, direct, or mediated), the same scalar the contract and an
    /// exported artifact carry; the other contrasts are on [`Self::mediation`].
    #[must_use]
    pub fn effect(&self) -> f64 {
        if let Some(dist) = self.distribution() {
            return dist.mean;
        }
        if let Some(cf) = self.counterfactual() {
            return cf.mean_ite;
        }
        if let Some(set) = self.counterfactual_contrasts() {
            return set.primary().mean_ite;
        }
        self.estimate.as_effect().map_or(f64::NAN, |estimate| estimate.ate)
    }

    /// Borrow the logical plan record (semantics).
    #[must_use]
    pub fn logical_plan(&self) -> &LogicalAnalysisPlanRecord {
        &self.logical_plan
    }

    /// Borrow the physical plan record (layouts / kernels / batching).
    #[must_use]
    pub fn physical_plan(&self) -> &PhysicalExecutionPlanRecord {
        &self.physical_plan
    }

    /// Build a durable analysis-trace wire payload (assumptions + derivation).
    #[must_use]
    pub fn analysis_trace_wire(&self) -> AnalysisTraceWire {
        let assumptions = self
            .estimate
            .as_effect()
            .map_or(&self.identification.required_assumptions, |estimate| {
                &estimate.provenance.assumptions
            });
        AnalysisTraceWire {
            assumptions: assumptions_to_wire(assumptions),
            derivation: self
                .identification
                .derivation
                .steps
                .iter()
                .map(|s| DerivationStepWire {
                    rule: s.rule.to_string(),
                    detail: s.detail.to_string(),
                })
                .collect(),
            method: self.estimand.method.to_string(),
            adjustment_set: self.estimand.adjustment_set.iter().map(|id| id.raw()).collect(),
            support_status: self
                .support_status
                .map(crate::support::CellStatus::as_str)
                .map(str::to_string),
        }
    }
}

#[cfg(test)]
mod weight_basis_tests {
    use super::StructuralWeightBasis;

    #[test]
    fn as_str_matches_the_wire_spelling() {
        for basis in [
            StructuralWeightBasis::PosteriorProbability,
            StructuralWeightBasis::CompletionEnumeration,
            StructuralWeightBasis::CallerSuppliedClassPrior,
        ] {
            let wire = antecedent_io::StructuralWeightBasisWire::from(basis);
            assert_eq!(serde_json::to_value(wire).unwrap(), basis.as_str());
        }
    }
}

#[cfg(test)]
mod published_scalar_uncertainty_tests {
    use antecedent_core::{AssumptionSet, IntervalMethod};
    use antecedent_estimate::{EffectEstimate, FirstStageDiagnostics, OverlapPolicy};

    use super::PublishedScalarUncertainty;

    fn estimate(ate: f64, se_analytic: f64, se_bootstrap: Option<f64>) -> EffectEstimate {
        let mut estimate = EffectEstimate::new(
            ate,
            se_analytic,
            AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        );
        estimate.uncertainty.bootstrap.se = se_bootstrap;
        estimate
    }

    fn weak_first_stage() -> FirstStageDiagnostics {
        FirstStageDiagnostics {
            f_statistic: 5.0,
            df1: 1,
            df2: 98,
            partial_r2: 0.05,
            anderson_rubin: None,
            uncertainty_withheld: None,
        }
    }

    #[test]
    fn anderson_rubin_set_is_the_published_interval() {
        let mut diagnostics = weak_first_stage();
        diagnostics.anderson_rubin = Some((0.2, 1.8, 0.95));
        let estimate =
            estimate(1.0, f64::NAN, Some(0.2)).with_first_stage_diagnostics(Some(diagnostics));
        let published = PublishedScalarUncertainty::select(&estimate);
        assert!(published.standard_error.is_none());
        assert_eq!(published.method, IntervalMethod::AndersonRubin);
        assert_eq!(published.lower, Some(0.2));
        assert_eq!(published.upper, Some(1.8));
        assert!((published.level - 0.95).abs() < 1e-12);
        assert!(published.withheld_reason.is_none());
    }

    #[test]
    fn weak_iv_non_finite_analytic_withholds_bootstrap() {
        let estimate = estimate(1.0, f64::NAN, Some(0.2))
            .with_first_stage_diagnostics(Some(weak_first_stage()));
        let published = PublishedScalarUncertainty::select(&estimate);
        assert!(published.standard_error.is_none());
        assert_eq!(published.method, IntervalMethod::None);
        assert!(published.withheld_reason.is_some());
    }

    #[test]
    fn finite_analytic_without_bootstrap_publishes_analytic() {
        let estimate = estimate(1.0, 0.15, None);
        let published = PublishedScalarUncertainty::select(&estimate);
        assert_eq!(published.standard_error, Some(0.15));
        assert_eq!(published.method, IntervalMethod::AnalyticSe);
        assert!(published.withheld_reason.is_none());
    }

    #[test]
    fn both_finite_without_first_stage_prefers_bootstrap() {
        let estimate = estimate(1.0, 0.15, Some(0.2));
        let published = PublishedScalarUncertainty::select(&estimate);
        assert_eq!(published.standard_error, Some(0.2));
        assert_eq!(published.method, IntervalMethod::BootstrapSe);
        assert!(published.withheld_reason.is_none());
    }

    #[test]
    fn not_an_effect_exposes_no_effect_payload() {
        let mut estimate = super::PrimaryEstimate::NotAnEffect;
        assert!(estimate.as_effect().is_none());
        assert!(estimate.as_effect_mut().is_none());
    }
}
