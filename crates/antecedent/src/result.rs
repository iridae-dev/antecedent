//! Analysis result artifact.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
    /// switchback uses an independent-sequence sandwich estimate.
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
/// survival on subject trajectories (unlicensed, uncalibrated).
///
/// One entry per named coordinate. Frequentist results carry standard errors and pointwise
/// intervals (and, for several contrasts or a dose grid, a simultaneous band); Bayesian results
/// carry equal-tailed credible intervals of the retained draws, which ride on the result's
/// posterior. No coverage of any interval has been measured, and the route is not licensed.
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
    /// Calibration status of every interval: always `unmeasured`.
    pub calibration: Arc<str>,
    /// Always `false`: the route is not licensed.
    pub licensed: bool,
}

/// Effects of a rollout design with the uncertainty of its whole-procedure path
/// (unlicensed; calibration unmeasured).
///
/// Carries a heterogeneity-robust or covariate-adjusted difference in differences (group-time,
/// aggregated and pre-trend effects), a conformal or multi-unit synthetic control (per-period
/// or per-unit effects), or a local ratio with a selected bandwidth or a calendar-time running
/// variable. The first effect is the scalar summary. Pointwise intervals and simultaneous bands
/// come from the design's own procedure (whole-unit resample, conformal test inversion,
/// placebo variance or block bootstrap), each refitting the whole estimator; none has a
/// coverage record and the route is not licensed.
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
    /// Calibration status of every interval: always `unmeasured`.
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
    /// Always `false`: the route is not licensed.
    pub licensed: bool,
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

/// Within-period history-matched contrast (unlicensed, uncalibrated).
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

/// Observational two-decision contrast of two static regimes (unlicensed, uncalibrated).
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
    /// Exact graphless longitudinal license after observed support and interval checks.
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
    /// Exact graphless simultaneous-band license, when the published band and
    /// arm support match a licensed band row. Absent for off-axis bands.
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
    /// Realized control and treated subject counts for an exact graphless license.
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

/// How an execution treated missing values: the policy, the assumption that policy needs,
/// and the row accounting.
///
/// Present on every result whose route reads rows of a table, including results that dropped
/// nothing, so a reader never has to infer that no row was removed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct MissingDataReport {
    /// Rows total, kept and dropped, per variable and per reason, with the dropped-row digest.
    pub accounting: antecedent_data::RowAccounting,
    /// Whether the caller declared the policy; `false` when the route's own default applied.
    pub declared: bool,
    /// Names of the covariates of the observation model, for responder weighting.
    pub covariates: std::sync::Arc<[std::sync::Arc<str>]>,
    /// The missingness assumption the policy needs, as published in the missing-data guide.
    pub assumption: std::sync::Arc<str>,
    /// Whether the policy differs from the route's default. Only then does a claim over the
    /// result carry the policy, its assumption and the dropped-row digest, so default claims
    /// keep their identity.
    pub non_default: bool,
    /// The impute-fit-pool record, for multiple imputation.
    pub imputation: Option<ImputationReport>,
}

/// How a multiple-imputation result was formed: the declared specification, what was
/// imputed, every completed table's estimate and the pooled variance components.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImputationReport {
    /// The declared specification (model, imputations, sweeps, seed, assumption).
    pub spec: antecedent_data::ImputationSpec,
    /// Unit the parameter draws resampled (`row` or `group`).
    pub unit: std::sync::Arc<str>,
    /// Units resampled.
    pub units: usize,
    /// Cells imputed per variable of the imputation model.
    pub imputed_cells: std::sync::Arc<[(std::sync::Arc<str>, usize)]>,
    /// Estimator fitted on every completed table.
    pub estimator: std::sync::Arc<str>,
    /// `rubin` (frequentist), `stacked_posterior` (Bayesian) or `stacked_graph_posterior` (Bayesian
    /// mixture over declared graphs, whose weights stay inside every table's draws).
    pub pooling: std::sync::Arc<str>,
    /// Each completed table's point estimate, in imputation order.
    pub points: std::sync::Arc<[f64]>,
    /// Within-imputation variance.
    pub within: f64,
    /// Between-imputation variance.
    pub between: f64,
    /// Total variance (Rubin's `T`, or the stacked posterior variance).
    pub total: f64,
    /// Degrees of freedom of the Student-t reference (Barnard–Rubin when the estimator has a
    /// finite complete-data degrees of freedom); `None` for a normal reference or a posterior.
    pub df: Option<f64>,
    /// Complete-data degrees of freedom used by the Barnard–Rubin adjustment.
    pub df_complete: Option<f64>,
    /// Fraction of missing information, for Rubin pooling.
    pub fraction_missing_information: Option<f64>,
    /// Relative increase in variance due to nonresponse, for Rubin pooling.
    pub relative_increase: Option<f64>,
    /// Digest (hex) of the model and design assumptions the pooled estimate rests on: the
    /// imputation model, tables, sweeps, seed, missingness assumption, resampling unit, declared
    /// sampling design and every variable's kind and imputed-cell count. Two reports with
    /// different digests are different analyses.
    pub assumptions_digest: std::sync::Arc<str>,
    /// Canonical declared model-column domains.
    pub imputation_domains:
        std::sync::Arc<[(std::sync::Arc<str>, antecedent_data::ImputationDomain)]>,
    /// Binary model variables and their single observed class under empirical-bootstrap support.
    pub binary_boundary_variables: std::sync::Arc<[(std::sync::Arc<str>, u8)]>,
    /// Neighbouring time steps on each side the imputation model conditioned on, for a series;
    /// zero for exchangeable rows.
    pub lags: usize,
    /// The joint pooling of the answer's coordinates, for a route whose answer has several
    /// (the cells of a conditional effect, a response curve, the levels of a distribution, a
    /// mediation decomposition, the arms of a randomized design, the horizons of a series).
    pub vector: Option<ImputationVectorReport>,
}

/// The Wald test of a pooled answer's coordinates against a stated null (Li, Raghunathan and
/// Rubin's `D1`), referred to an `F` law.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImputationWald {
    /// The hypothesised value of every coordinate.
    pub null: std::sync::Arc<[f64]>,
    /// The pooled statistic.
    pub statistic: f64,
    /// Numerator degrees of freedom, the number of coordinates.
    pub df_numerator: f64,
    /// Denominator degrees of freedom; infinite is the chi-square reference.
    pub df_denominator: f64,
}

/// A quantity of the answer that has no published within-imputation variance, pooled by its
/// mean over the completed tables with the between-imputation variance beside it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImputationExtra {
    /// What the quantity is.
    pub label: std::sync::Arc<str>,
    /// Its value on each completed table, in imputation order.
    pub points: std::sync::Arc<[f64]>,
    /// The mean over the completed tables.
    pub point: f64,
    /// Between-imputation variance of the value: a lower bound on its pooled variance.
    pub between: f64,
}

/// The joint pooling of an answer with several coordinates.
///
/// Matrices are row-major `p x p` over the coordinates in `labels` order. Under Rubin pooling
/// `total = within + (1 + 1/m) between`; under a stacked posterior `total` is the covariance of
/// the equal-weight mixture and `within` carries each coordinate's mean posterior variance on
/// its diagonal.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImputationVectorReport {
    /// Name of each coordinate.
    pub labels: std::sync::Arc<[std::sync::Arc<str>]>,
    /// How the within-imputation covariance was formed: `joint_covariance` (a retained joint
    /// influence covariance), `marginal_variances` (each coordinate's published variance on the
    /// diagonal, no cross-covariance), or `posterior_draws` (a stacked posterior).
    pub covariance_basis: std::sync::Arc<str>,
    /// Every completed table's coordinates, in imputation order.
    pub points: std::sync::Arc<[std::sync::Arc<[f64]>]>,
    /// Pooled coordinates.
    pub point: std::sync::Arc<[f64]>,
    /// Within-imputation covariance.
    pub within: std::sync::Arc<[f64]>,
    /// Between-imputation covariance.
    pub between: std::sync::Arc<[f64]>,
    /// Total covariance.
    pub total: std::sync::Arc<[f64]>,
    /// Degrees of freedom of each coordinate's Student-t reference (Rubin pooling); `None` for a
    /// normal reference or a posterior.
    pub coordinate_df: std::sync::Arc<[Option<f64>]>,
    /// Fraction of missing information of each coordinate (Rubin pooling).
    pub coordinate_fraction_missing_information: std::sync::Arc<[Option<f64>]>,
    /// Average relative increase in variance due to nonresponse (Rubin pooling).
    pub average_relative_increase: Option<f64>,
    /// The joint Wald test that every coordinate is zero, when the within-imputation covariance
    /// is invertible (Rubin pooling).
    pub wald: Option<ImputationWald>,
    /// Quantities of the answer pooled by their mean alone.
    pub extras: std::sync::Arc<[ImputationExtra]>,
}

/// Bitwise equality of two number slices, without building either side's bit vector.
fn same_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Bitwise equality of two optional numbers.
fn same_option_bits(left: Option<f64>, right: Option<f64>) -> bool {
    left.map(f64::to_bits) == right.map(f64::to_bits)
}

/// Bitwise equality of two slices of optional numbers.
fn same_option_slice_bits(left: &[Option<f64>], right: &[Option<f64>]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| same_option_bits(*a, *b))
}

// The `PartialEq` impls below destructure both sides without `..`, so a field added to a
// report is a compile error here until the comparison names it.

impl PartialEq for ImputationWald {
    fn eq(&self, other: &Self) -> bool {
        let Self { null, statistic, df_numerator, df_denominator } = self;
        let Self {
            null: other_null,
            statistic: other_statistic,
            df_numerator: other_df_numerator,
            df_denominator: other_df_denominator,
        } = other;
        same_bits(null, other_null)
            && same_bits(
                &[*statistic, *df_numerator, *df_denominator],
                &[*other_statistic, *other_df_numerator, *other_df_denominator],
            )
    }
}

impl Eq for ImputationWald {}

impl PartialEq for ImputationExtra {
    fn eq(&self, other: &Self) -> bool {
        let Self { label, points, point, between } = self;
        let Self {
            label: other_label,
            points: other_points,
            point: other_point,
            between: other_between,
        } = other;
        label == other_label
            && same_bits(points, other_points)
            && same_bits(&[*point, *between], &[*other_point, *other_between])
    }
}

impl Eq for ImputationExtra {}

impl PartialEq for ImputationVectorReport {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            labels,
            covariance_basis,
            points,
            point,
            within,
            between,
            total,
            coordinate_df,
            coordinate_fraction_missing_information,
            average_relative_increase,
            wald,
            extras,
        } = self;
        let Self {
            labels: other_labels,
            covariance_basis: other_covariance_basis,
            points: other_points,
            point: other_point,
            within: other_within,
            between: other_between,
            total: other_total,
            coordinate_df: other_coordinate_df,
            coordinate_fraction_missing_information: other_fraction,
            average_relative_increase: other_average_relative_increase,
            wald: other_wald,
            extras: other_extras,
        } = other;
        labels == other_labels
            && covariance_basis == other_covariance_basis
            && points.len() == other_points.len()
            && points.iter().zip(other_points.iter()).all(|(a, b)| same_bits(a, b))
            && same_bits(point, other_point)
            && same_bits(within, other_within)
            && same_bits(between, other_between)
            && same_bits(total, other_total)
            && same_option_slice_bits(coordinate_df, other_coordinate_df)
            && same_option_slice_bits(coordinate_fraction_missing_information, other_fraction)
            && same_option_bits(*average_relative_increase, *other_average_relative_increase)
            && wald == other_wald
            && extras == other_extras
    }
}

impl Eq for ImputationVectorReport {}

/// Bitwise equality of the numbers, so the report is `Eq` like the accounting it travels
/// with: two reports are equal when they would serialize identically.
impl PartialEq for ImputationReport {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            spec,
            unit,
            units,
            imputed_cells,
            estimator,
            pooling,
            points,
            within,
            between,
            total,
            df,
            df_complete,
            fraction_missing_information,
            relative_increase,
            assumptions_digest,
            imputation_domains,
            binary_boundary_variables,
            lags,
            vector,
        } = self;
        let Self {
            spec: other_spec,
            unit: other_unit,
            units: other_units,
            imputed_cells: other_imputed_cells,
            estimator: other_estimator,
            pooling: other_pooling,
            points: other_points,
            within: other_within,
            between: other_between,
            total: other_total,
            df: other_df,
            df_complete: other_df_complete,
            fraction_missing_information: other_fraction,
            relative_increase: other_relative_increase,
            assumptions_digest: other_assumptions_digest,
            imputation_domains: other_imputation_domains,
            binary_boundary_variables: other_binary_boundary_variables,
            lags: other_lags,
            vector: other_vector,
        } = other;
        spec == other_spec
            && unit == other_unit
            && units == other_units
            && imputed_cells == other_imputed_cells
            && estimator == other_estimator
            && pooling == other_pooling
            && same_bits(points, other_points)
            && same_bits(
                &[*within, *between, *total],
                &[*other_within, *other_between, *other_total],
            )
            && same_option_bits(*df, *other_df)
            && same_option_bits(*df_complete, *other_df_complete)
            && same_option_bits(*fraction_missing_information, *other_fraction)
            && same_option_bits(*relative_increase, *other_relative_increase)
            && assumptions_digest == other_assumptions_digest
            && imputation_domains == other_imputation_domains
            && binary_boundary_variables == other_binary_boundary_variables
            && lags == other_lags
            && vector == other_vector
    }
}

impl Eq for ImputationReport {}

impl MissingDataReport {
    /// The policy the accounting was produced under.
    #[must_use]
    pub fn policy(&self) -> antecedent_data::MissingPolicy {
        self.accounting.policy
    }
}

/// Meaning of structural atom weights retained on a response result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StructuralWeightBasis {
    /// Probabilities supplied by a graph posterior.
    PosteriorProbability,
    /// Enumeration weights over CPDAG/PAG completions; not posterior probabilities.
    CompletionEnumeration,
    /// Caller-declared mass over class members. Not a graph posterior and not
    /// completion enumeration.
    CallerSuppliedClassPrior,
}

impl StructuralWeightBasis {
    /// Snake-case name used by the bindings and the result artifact wire format.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PosteriorProbability => "posterior_probability",
            Self::CompletionEnumeration => "completion_enumeration",
            Self::CallerSuppliedClassPrior => "caller_supplied_class_prior",
        }
    }
}

impl From<StructuralWeightBasis> for antecedent_io::StructuralWeightBasisWire {
    fn from(basis: StructuralWeightBasis) -> Self {
        match basis {
            StructuralWeightBasis::PosteriorProbability => Self::PosteriorProbability,
            StructuralWeightBasis::CompletionEnumeration => Self::CompletionEnumeration,
            StructuralWeightBasis::CallerSuppliedClassPrior => Self::CallerSuppliedClassPrior,
        }
    }
}

/// How identified structural atoms may be combined.
///
/// Mass-weighting is licensed only when every contributing atom names the same
/// estimand. Completion enumeration is a [`StructuralWeightBasis`], not a
/// reason to treat class members as posterior probability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StructuralAggregationPolicy {
    /// Identified atoms share one estimand identity; a probability-weighted
    /// mean of their values is the same functional.
    SameEstimandWeightedMean,
    /// Identified atoms share one partially identified estimand; publish the
    /// identified set, not a fake scalar mixture.
    IdentifiedSetEnvelope,
    /// Identified atoms name different estimands. No scalar mixture.
    GraphDependentAtoms,
}

impl StructuralAggregationPolicy {
    /// Snake-case name used in diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SameEstimandWeightedMean => "same_estimand_weighted_mean",
            Self::IdentifiedSetEnvelope => "identified_set_envelope",
            Self::GraphDependentAtoms => "graph_dependent_atoms",
        }
    }
}

/// One structural atom in a graph-dependent response.
#[derive(Clone, Debug)]
pub struct StructuralResponseAtom {
    /// Stable key within this result.
    pub graph_key: u64,
    /// Raw atom weight in the declared basis.
    pub weight: f64,
    /// Structural identification/evaluation status.
    pub status: IdentificationStatus,
    /// Numerical response when identified and evaluable.
    pub value: Option<ResponseValue>,
    /// Completion-conditional posterior, without assigning structural probabilities.
    pub posterior: Option<CausalPosterior>,
    /// Full function-valued atom, including its conditional sampling uncertainty.
    pub response: Option<CausalResponse>,
}

/// Structural uncertainty retained separately from sampling uncertainty.
#[derive(Clone, Debug)]
pub struct StructuralResponseMixture {
    /// Interpretation of [`StructuralResponseAtom::weight`].
    pub weight_basis: StructuralWeightBasis,
    /// Every examined structural atom, including nonidentified/unevaluable atoms.
    pub atoms: Vec<StructuralResponseAtom>,
    /// Fraction of total weight with an evaluable identified response.
    pub identified_mass: f64,
    /// Fraction proved or conservatively treated as structurally unidentified.
    pub unidentified_mass: f64,
    /// Fraction identified in theory but not evaluable by the selected estimator.
    pub unevaluable_mass: f64,
    /// Fraction on identified atoms that were never evaluated because the
    /// Interactive latency tier's graph budget left them out of the stratified
    /// subsample. Neither unidentified nor a failed estimate; the four masses
    /// sum to one. Zero outside the Interactive tier.
    pub subsampled_out_mass: f64,
    /// Pointwise range over identified atom point responses.
    pub identified_set: Option<ResponseEnvelope>,
    /// Interval for a scalar [`Self::identified_set`] that adds sampling
    /// uncertainty to the point bounds. Published on class-aware
    /// temporal Pulse / Sustained effects. Frequentist: an Imbens–Manski interval
    /// with per-completion endpoints from the shared circular-block replicates,
    /// covering the true effect with asymptotic probability at least the stated
    /// level whenever it is one retained identified completion's effect.
    /// Bayesian: product-posterior envelope quantiles from independently seeded
    /// per-completion posterior draws (every retained completion's posterior puts
    /// at most `1 − Φ(c)` of its mass outside each endpoint). When the completion
    /// enumeration was capped the interval is still published, flagged
    /// `truncated`, with a warning diagnostic.
    pub identified_set_interval: Option<antecedent_estimate::IdentifiedSetInterval>,
    /// Probability-weighted summary when a scalar `SameEstimandWeightedMean` mix is
    /// published (`E[τ | identified]`). Retained unidentified mass does not
    /// suppress this field; consumers should read `identification.status` and
    /// `unidentified_mass` alongside it.
    pub conditional_on_identified: Option<ResponseValue>,
    /// Whether reported mass covers the full class rather than a capped subset.
    pub full_mass_scope: bool,
    /// Atoms whose identification search was capped before a determination.
    pub truncated_atoms: usize,
    /// How the identified atoms were combined, for routes that aggregate across
    /// graphs. `None` for routes whose atoms are not graph alternatives to be
    /// combined under a policy.
    pub aggregation_policy: Option<StructuralAggregationPolicy>,
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
    /// Exact graphless interference license after observed cluster support and
    /// a published pointwise interval pass; `None` when the interval is withheld
    /// or the design/exposure support does not match a licensed row.
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

/// The retained posterior of an unlicensed Bayesian randomized-design row: coordinates, means,
/// equal-tailed 95% credible intervals, the covariance of the joint draws and named summaries.
/// Its uncertainty is `posterior` and its calibration `unmeasured`; it is never licensed.
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
    /// Unlicensed Bayesian posterior of a randomized-design row (survival contrast, fixed
    /// group-dose policy value): means, credible intervals and the joint covariance of the
    /// retained draws, with calibration `unmeasured`.
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

    /// Whether the family reports no calibrated scalar effect interval: its
    /// uncertainty lives in a dedicated result section.
    #[must_use]
    pub const fn has_dedicated_uncertainty_section(&self) -> bool {
        matches!(
            self,
            Self::RandomizedEffect(_)
                | Self::SyntheticControl(_)
                | Self::SyntheticDid(_)
                | Self::LocalPolynomialRatio(_)
                | Self::PolicyValue(_)
                | Self::ContinuousDoseResponse(_)
                | Self::Survival(_)
                | Self::PosteriorRow(_)
                | Self::LongitudinalRegime(_)
                | Self::SequentialContrast(_)
                | Self::HistoryMatch(_)
                | Self::TemporalDesign(_)
                | Self::IvBounds(_)
                | Self::Association(_)
                | Self::QuasiRollout(_)
        )
    }
}

/// End-to-end analysis result.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct StudyResult {
    pub(crate) retained_resources: Vec<Arc<antecedent_core::execution::AdmissionLease>>,
    /// Logical plan record.
    pub logical_plan: LogicalAnalysisPlanRecord,
    /// Physical plan record.
    pub physical_plan: PhysicalExecutionPlanRecord,
    /// Full identification artifact.
    pub identification: IdentificationResult,
    /// Complete identification certificate, when supplied by the execution path.
    pub certificate: Option<AnalysisIdentification>,
    /// Primary estimand used for estimation.
    pub estimand: IdentifiedEstimand,
    /// Point estimate + uncertainty (frequentist, or Bayesian posterior mean summary).
    ///
    /// For [`CausalQuery::Distribution`](antecedent_core::CausalQuery::Distribution) this holds the
    /// interventional mean of the first numeric outcome when defined (`ate` field), else NaN.
    /// Its `se_bootstrap` is the mean's bootstrap SE; do not form `ate ± z·se` for a
    /// probability — the bounded per-atom intervals are
    /// [`InterventionalDistributionEstimate::atom_uncertainty`], and a binary outcome's
    /// mean interval is [`InterventionalDistributionEstimate::mean_interval`].
    pub(crate) estimate: PrimaryEstimate,
    /// Structural atom/mass result for class-aware or graph-posterior responses.
    pub structural_response: Option<StructuralResponseMixture>,
    /// The typed answer the query family published beside the scalar estimate.
    ///
    /// Read it through [`Self::response`], [`Self::distribution`] and the other
    /// accessors, or match on it directly. The Bayesian [`Self::posterior`] and
    /// the atom [`Self::structural_response`] accompany any family and stay
    /// outside it.
    pub(crate) payload: ResultPayload,
    /// Bayesian posterior when `InferenceMode::Bayesian` was used.
    pub posterior: Option<CausalPosterior>,
    /// Refutation reports (may be empty).
    pub refutations: Vec<RefutationReport>,
    /// Prior/posterior predictive check reports (Bayesian path; may be empty).
    pub predictive_checks: Vec<PredictiveCheckReport>,
    /// Posterior predictive replicates retained at fit time when the study declared bounds
    /// for them; read through [`Self::predictive_replicates`].
    pub(crate) predictive_replicates: Option<Arc<crate::analysis::RetainedPredictiveReplicates>>,
    /// Diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Provenance.
    pub provenance: ProvenanceGraph,
    /// Support-matrix evidence contract that produced this result.
    ///
    /// `licensed` and `allowed_unlicensed` both yield a successful study.
    /// Downstream consumers must not treat a number as licensed unless this is
    /// [`crate::support::CellStatus::Licensed`]. `None` when the query is not on
    /// the public axis.
    pub(crate) support_status: Option<crate::support::CellStatus>,
    /// How the caller supplied structure. Graph-posterior mixtures are never a
    /// single adjustment set, even when every identified atom happens to agree.
    pub structure_source: crate::support::StructureSource,
    /// Performance record.
    pub performance: ExecutionPerformanceRecord,
    /// Treatment variable.
    pub treatment: Option<VariableId>,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Prepared contract this result was executed under, stamped by the
    /// [`crate::PreparedStudy`] entry point that produced it. `None` for
    /// results no prepared handle executed; those cannot be exported as claims.
    pub(crate) executed_contract: Option<ExecutedContract>,
    /// Population bindings in force, when the query names a predicate or a
    /// custom target distribution. Encoding the query needs them.
    pub population_registry: Option<antecedent_core::PopulationRegistry>,
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
}

impl StudyResult {
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

/// Nominal level of the interval a standard-error result reports.
///
/// An [`EffectEstimate`] carries a standard error, not endpoints; every
/// rendering of it (Python views, HTML, the analysis-result artifact's
/// `standard_error`) forms the two-sided `estimate ± 1.96·SE` interval (a Student-t
/// reference where the estimate declares `se_reference_df`, see
/// [`PublishedScalarUncertainty::interval`]).
/// Posterior summaries report the equal-tailed `q025` / `q975` interval at the
/// same level.
pub const REPORTED_SE_INTERVAL_LEVEL: f64 = 0.95;

/// Support diagnostic marking a function-valued response whose interval is a
/// normal interval from the plug-in bootstrap SE (the general-ID
/// `functional.effect` route resamples the whole front-door plug-in each
/// replicate). Its values are `[requested replicates, successful replicates]`.
/// A response carrying it reports [`IntervalMethod::BootstrapSe`] rather than the
/// analytic-SE default; every other response band stays analytic.
pub(crate) const RESPONSE_BOOTSTRAP_SE: &str = "response.bootstrap_se";

/// Fewest successful replicates that earn a nominal 0.95 percentile band or a
/// normal interval from a bootstrap SE.
///
/// Under the (B+1) rule each tail holds `(B+1)·α/2` order statistics. At
/// `α = 0.05` that count exceeds one only for `B ≥ 40` (`B = 39` is exactly
/// one). Statistical transport and the tuple/block bootstrap floor both read
/// this constant.
pub const PERCENTILE_95_MIN_REPLICATES: u32 = 40;

/// Two-sided normal critical value of [`REPORTED_SE_INTERVAL_LEVEL`].
///
/// Every path that forms `estimate ± z·SE` at the published level reads this
/// one quantile rather than writing `normal_ppf(0.975)` again:
/// `0.5 + 0.95 / 2.0` is exactly `0.975` in binary64, so the value is
/// bit-identical to the literal it replaces.
#[must_use]
pub fn reported_se_interval_z() -> f64 {
    antecedent_stats::normal_ppf(0.5 + REPORTED_SE_INTERVAL_LEVEL / 2.0)
}

/// How one reported interval was formed: the construction the calibration
/// match key describes.
///
/// Derived from the result's own content by
/// [`StudyResult::primary_interval_binding`] and
/// [`StudyResult::identified_set_interval_binding`], so it always describes
/// what the execution reported, never what was requested.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct IntervalBinding {
    /// Interval method.
    pub method: antecedent_core::IntervalMethod,
    /// Analytic SE kind recorded by the estimator, when the method is analytic.
    pub se_kind: Option<antecedent_estimate::AnalyticSeKind>,
    /// Nominal coverage level of the reported interval.
    pub level: f64,
    /// Dependence rule of the construction: `iid`, `panel_cluster`, or
    /// `circular_block:<family>` ([`antecedent_estimate::CircularBlockFamily::tag`]).
    pub dependence: &'static str,
    /// Resampling replicates that succeeded (bootstrap, circular block, or
    /// simultaneous-band multipliers), when the interval is resampling-based.
    pub replicates_ok: Option<u32>,
    /// Resampling replicates that failed, when reported.
    pub replicates_failed: Option<u32>,
    /// Posterior draws behind a posterior interval.
    pub posterior_draws: Option<u32>,
}

impl IntervalBinding {
    const fn new(
        method: antecedent_core::IntervalMethod,
        level: f64,
        dependence: &'static str,
    ) -> Self {
        Self {
            method,
            se_kind: None,
            level,
            dependence,
            replicates_ok: None,
            replicates_failed: None,
            posterior_draws: None,
        }
    }

    /// Binding of an execution that reported no interval.
    #[must_use]
    pub const fn none(dependence: &'static str) -> Self {
        Self::new(antecedent_core::IntervalMethod::None, f64::NAN, dependence)
    }
}

/// Dependence label of a circular-block family.
#[must_use]
pub const fn circular_block_dependence(
    family: antecedent_estimate::CircularBlockFamily,
) -> &'static str {
    match family {
        antecedent_estimate::CircularBlockFamily::SingleWindow => "circular_block:single_window",
        antecedent_estimate::CircularBlockFamily::Mediation => "circular_block:mediation",
        antecedent_estimate::CircularBlockFamily::Sequential => "circular_block:sequential",
        antecedent_estimate::CircularBlockFamily::Mixture => "circular_block:mixture",
        antecedent_estimate::CircularBlockFamily::RowBlock => "circular_block:row_block",
    }
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Published scalar standard error and the label of its normal interval.
///
/// One owner of the facade's SE / `± z·SE` rule. Wire `standard_error`,
/// [`StudyResult::primary_interval_binding`], the Uncertainty slot, and the
/// Python copies all read this; they do not prefer `se_bootstrap` on their own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedScalarUncertainty {
    /// The one scalar SE the facade publishes, when licensed and positive-finite.
    pub standard_error: Option<f64>,
    /// Interval method for what was published: a normal interval from
    /// [`Self::standard_error`] ([`antecedent_core::IntervalMethod::AnalyticSe`] or
    /// [`antecedent_core::IntervalMethod::BootstrapSe`]), an Anderson–Rubin set
    /// ([`antecedent_core::IntervalMethod::AndersonRubin`], endpoints in [`Self::lower`] /
    /// [`Self::upper`], no standard error), or [`antecedent_core::IntervalMethod::None`].
    /// Circular-block labeling is applied by
    /// [`StudyResult::primary_interval_binding`] when a block family is recorded.
    pub method: antecedent_core::IntervalMethod,
    /// Nominal level of the published interval, or `NaN` when nothing was published.
    pub level: f64,
    /// Why a scalar SE was withheld, when the estimate is an IV result that
    /// cannot publish a Wald or bootstrap SE and has no Anderson–Rubin set.
    pub withheld_reason: Option<&'static str>,
    /// Lower endpoint of a published Anderson–Rubin set. May be infinite.
    pub lower: Option<f64>,
    /// Upper endpoint of a published Anderson–Rubin set. May be infinite.
    pub upper: Option<f64>,
}

impl PublishedScalarUncertainty {
    /// Select the one published scalar SE for an [`EffectEstimate`].
    ///
    /// Rules:
    /// 1. Only a positive-finite SE is published (`Some` alone is not enough).
    /// 2. When first-stage diagnostics carry an Anderson–Rubin set, publish
    ///    those endpoints and no standard error. Do not fall through to
    ///    `se_bootstrap` or a Wald SE.
    /// 3. When first-stage diagnostics are present and no Anderson–Rubin set
    ///    was formed, do not fall through to `se_bootstrap`. A positive-finite
    ///    analytic SE is still published; otherwise withhold, keeping the
    ///    estimator's reason when it recorded one.
    /// 4. Otherwise prefer a positive-finite bootstrap SE over a positive-finite
    ///    analytic SE (AIPW and peers keep today's preference). IV and NN
    ///    matching do not emit a competing finite pair.
    #[must_use]
    pub fn select(estimate: &EffectEstimate) -> Self {
        use antecedent_core::IntervalMethod as M;

        if let Some(diagnostics) = &estimate.diagnostics.first_stage {
            if let Some((lower, upper, level)) = diagnostics.anderson_rubin {
                if !lower.is_nan()
                    && !upper.is_nan()
                    && level.is_finite()
                    && (0.0..1.0).contains(&level)
                {
                    return Self {
                        standard_error: None,
                        method: M::AndersonRubin,
                        level,
                        withheld_reason: None,
                        lower: Some(lower),
                        upper: Some(upper),
                    };
                }
            }
            if positive_finite(estimate.se_analytic()) {
                return Self {
                    standard_error: Some(estimate.se_analytic()),
                    method: M::AnalyticSe,
                    level: REPORTED_SE_INTERVAL_LEVEL,
                    withheld_reason: None,
                    lower: None,
                    upper: None,
                };
            }
            return Self::withhold_iv(diagnostics.uncertainty_withheld);
        }

        if estimate.se_bootstrap().is_some_and(positive_finite) {
            return Self {
                standard_error: estimate.se_bootstrap(),
                method: M::BootstrapSe,
                level: REPORTED_SE_INTERVAL_LEVEL,
                withheld_reason: None,
                lower: None,
                upper: None,
            };
        }
        if positive_finite(estimate.se_analytic()) {
            return Self {
                standard_error: Some(estimate.se_analytic()),
                method: M::AnalyticSe,
                level: REPORTED_SE_INTERVAL_LEVEL,
                withheld_reason: None,
                lower: None,
                upper: None,
            };
        }
        Self {
            standard_error: None,
            method: M::None,
            level: f64::NAN,
            withheld_reason: None,
            lower: None,
            upper: None,
        }
    }

    /// Two-sided critical value of the published interval for `estimate`: Student-t with
    /// `se_reference_df` degrees of freedom for an analytic SE that declares one (the
    /// cluster-DML few-cluster reference), the normal quantile otherwise.
    #[must_use]
    pub fn critical_value(&self, estimate: &EffectEstimate) -> f64 {
        match (self.method, estimate.uncertainty.analytic.reference_df) {
            (antecedent_core::IntervalMethod::AnalyticSe, Some(df)) if df > 0.0 => {
                antecedent_stats::student_t_ppf(0.5 + self.level / 2.0, df)
            }
            _ => reported_se_interval_z(),
        }
    }

    /// The published `estimate ± critical_value · SE` interval, when a scalar SE was
    /// published.
    #[must_use]
    pub fn interval(&self, estimate: &EffectEstimate) -> Option<(f64, f64)> {
        let se = self.standard_error?;
        let half = self.critical_value(estimate) * se;
        Some((estimate.ate - half, estimate.ate + half))
    }

    fn withhold_iv(reason: Option<&'static str>) -> Self {
        Self {
            standard_error: None,
            method: antecedent_core::IntervalMethod::None,
            level: f64::NAN,
            withheld_reason: Some(reason.unwrap_or(
                "Wald/bootstrap SE not licensed (weak instrument or IV uncertainty withheld)",
            )),
            lower: None,
            upper: None,
        }
    }
}

fn draws_u32(n: usize) -> Option<u32> {
    u32::try_from(n).ok()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the guard admits only finite values in [0, u32::MAX], so the cast neither wraps nor loses sign (fractions truncate by design)"
)]
fn count_u32(value: f64) -> Option<u32> {
    (value.is_finite() && value >= 0.0 && value <= f64::from(u32::MAX)).then_some(value as u32)
}

fn support_values<'a>(response: &'a CausalResponse, id: &str) -> Option<&'a [f64]> {
    response
        .support
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.id.as_ref() == id)
        .map(|diagnostic| diagnostic.values.as_ref())
}

/// A Frequentist temporal response band built from circular-block replicates
/// of the lag-aligned surface (support diagnostic `response.temporal.block_length`).
fn temporal_response_band(response: &CausalResponse, level: f64) -> Option<IntervalBinding> {
    support_values(response, antecedent_estimate::TEMPORAL_RESPONSE_BLOCK_LENGTH)?;
    let mut binding = IntervalBinding::new(
        antecedent_core::IntervalMethod::CircularBlockSe,
        level,
        "circular_block:response",
    );
    if let Some(counts) = support_values(response, "response.observation_block_bootstrap") {
        binding.replicates_ok = counts.get(1).and_then(|n| count_u32(*n));
    }
    Some(binding)
}

impl StudyResult {
    /// The diagnostics that describe whether the data support the estimand (overlap, positivity,
    /// coordinates outside the observed support): exactly those emitted as
    /// [`DiagnosticKind::Support`], whatever their code. The one owner of the Support slot's
    /// diagnostic selection.
    pub fn support_diagnostics(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.iter().filter(|d| d.kind == DiagnosticKind::Support)
    }

    /// Dependence rule the execution's data imposes before any circular-block
    /// family: panel executions cluster by unit, every other modality is `iid`
    /// unless a circular-block construction names its family.
    fn base_dependence(&self) -> &'static str {
        if self.logical_plan.data_classification == antecedent_contract::DataClassification::Panel {
            "panel_cluster"
        } else {
            "iid"
        }
    }

    /// Dependence of a frequentist standard-error interval that names no circular-block
    /// family.
    ///
    /// On time-ordered rows (a series or an event log) an SE that models no serial
    /// correlation is not the `iid` construction of exchangeable rows: its variance ignores
    /// the dependence the data have, and a calibration measured for `iid` rows says nothing
    /// about it. It is labelled `serial_unmodelled`; a Newey-West kernel SE models the serial
    /// correlation and is labelled `serial_hac`. Every other modality keeps
    /// [`Self::base_dependence`]. The posterior bindings keep the base label too: a temporal
    /// posterior's serial-dependence handling (long-run tempering) is part of its construction
    /// and named by its posterior key, not by this dimension.
    fn se_dependence(&self, se_kind: Option<antecedent_estimate::AnalyticSeKind>) -> &'static str {
        use antecedent_contract::DataClassification as C;
        match self.logical_plan.data_classification {
            C::Temporal | C::Event => match se_kind {
                Some(antecedent_estimate::AnalyticSeKind::NeweyWest { .. }) => "serial_hac",
                _ => "serial_unmodelled",
            },
            _ => self.base_dependence(),
        }
    }

    fn posterior_draw_count(&self) -> Option<u32> {
        self.posterior.as_ref().and_then(|posterior| draws_u32(posterior.draws.n_draws))
    }

    /// A frequentist response band formed as a normal interval from the plug-in
    /// bootstrap SE (support diagnostic [`RESPONSE_BOOTSTRAP_SE`]): the general-ID
    /// `functional.effect` route has no exposed influence function, so its
    /// interval is the 199-replicate front-door resample, not an analytic SE.
    fn bootstrap_response_band(
        &self,
        response: &CausalResponse,
        level: f64,
    ) -> Option<IntervalBinding> {
        let values = support_values(response, RESPONSE_BOOTSTRAP_SE)?;
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::BootstrapSe,
            level,
            self.se_dependence(None),
        );
        binding.replicates_ok = values.get(1).and_then(|n| count_u32(*n));
        Some(binding)
    }

    /// Construction of the primary interval this execution reported.
    ///
    /// `bayesian` is the program's inference family: a Bayesian response's
    /// band is a posterior band even when the draws live in a referenced
    /// artifact rather than on the result.
    ///
    /// One owner of the rule, read by the claim's calibration match key:
    ///
    /// 1. A function-valued response reports its `ResponseUncertainty`
    ///    (`None` → [`IntervalMethod::None`](antecedent_core::IntervalMethod::None);
    ///    simultaneous band; identified-envelope band; posterior; a temporal
    ///    circular-block band; otherwise the influence-function band).
    /// 2. An interventional distribution with a binary outcome reports its
    ///    bounded mean interval.
    /// 3. A posterior reports its `q025` / `q975` interval.
    /// 4. A standard error reports `estimate ± 1.96·SE` from
    ///    [`PublishedScalarUncertainty::select`]: the published bootstrap SE
    ///    when that is the selected SE (circular-block when a family is
    ///    recorded), else the published analytic SE with the estimator's
    ///    recorded kind. A bootstrap interval here is a normal interval from
    ///    the bootstrap SE, not a percentile interval (replicates are not
    ///    retained). A requested bootstrap that reported no positive-finite SE
    ///    is not a bootstrap interval; an IV result that withheld Wald/bootstrap
    ///    uncertainty is [`IntervalMethod::None`](antecedent_core::IntervalMethod::None).
    /// 5. When no finite scalar interval exists, a withheld unknown-orientation
    ///    effect reports its scenario max-t band as a simultaneous interval.
    /// 6. Otherwise a reported identified-set interval is the primary interval,
    ///    or nothing was reported.
    #[allow(
        clippy::too_many_lines,
        reason = "the primary-interval choice is one ordered precedence list (cluster and panel intervals, response bands, a withheld unknown-orientation band, the identified set); splitting it would hide the ranking"
    )]
    #[must_use]
    pub fn primary_interval_binding(&self, bayesian: bool) -> IntervalBinding {
        use antecedent_core::{IntervalMethod as M, ResponseUncertainty as U};
        let panel_interval = self.panel_did().is_some_and(|did| did.interval_95.is_some());
        let randomized_cluster_interval = self.randomized_effect().is_some_and(|fit| {
            fit.interval_95.is_some() && fit.assignment_design.as_ref() == "cluster"
        });
        let base = if panel_interval || randomized_cluster_interval {
            "cluster"
        } else if self.panel_did().is_some() {
            "iid"
        } else {
            self.base_dependence()
        };
        if let Some(response) = self.response() {
            // A Bayesian response publishes credible bands; the draws behind a
            // band are in the referenced posterior artifact, not on the result.
            let posterior = bayesian || self.posterior.is_some();
            let mut binding = match &response.uncertainty {
                U::None => return IntervalBinding::none(base),
                U::Posterior { .. } => {
                    IntervalBinding::new(M::PosteriorQuantile, REPORTED_SE_INTERVAL_LEVEL, base)
                }
                U::Scalar { level, .. } | U::PointwiseBand { level, .. } if posterior => {
                    IntervalBinding::new(M::PosteriorQuantile, *level, base)
                }
                U::Scalar { level, .. } | U::PointwiseBand { level, .. } => self
                    .bootstrap_response_band(response, *level)
                    .or_else(|| temporal_response_band(response, *level))
                    .unwrap_or_else(|| {
                        IntervalBinding::new(M::AnalyticSe, *level, self.se_dependence(None))
                    }),
                U::SimultaneousBand { level, replicates, .. } => {
                    let mut binding = IntervalBinding::new(M::SimultaneousBand, *level, base);
                    binding.replicates_ok = Some(*replicates);
                    binding
                }
                U::IdentifiedEnvelopeBand { level, .. } => {
                    IntervalBinding::new(M::IdentifiedSet, *level, base)
                }
            };
            if binding.method == M::PosteriorQuantile {
                binding.posterior_draws = self.posterior_draw_count();
            }
            return binding;
        }
        if let Some(distribution) = self.distribution() {
            if self.posterior.is_none() {
                match distribution.mean_interval {
                    Some(antecedent_estimate::ProbabilityInterval::Bounded { level, .. }) => {
                        let mut binding = IntervalBinding::new(M::BootstrapSe, level, base);
                        binding.replicates_ok = distribution.bootstrap_replicates_ok;
                        binding.replicates_failed = distribution.bootstrap_replicates_failed;
                        return binding;
                    }
                    Some(antecedent_estimate::ProbabilityInterval::Unavailable(_)) => {
                        return IntervalBinding::none(base);
                    }
                    None => {}
                }
            }
        }
        if self.posterior.is_some() {
            let mut binding =
                IntervalBinding::new(M::PosteriorQuantile, REPORTED_SE_INTERVAL_LEVEL, base);
            binding.posterior_draws = self.posterior_draw_count();
            return binding;
        }
        let Some(estimate) = self.estimate.as_effect() else {
            return IntervalBinding::none(base);
        };
        if estimate.ate.is_finite() {
            let published = PublishedScalarUncertainty::select(estimate);
            match published.method {
                M::BootstrapSe => {
                    let (method, dependence) = match estimate.uncertainty.block.family {
                        Some(family) => (M::CircularBlockSe, circular_block_dependence(family)),
                        None => (M::BootstrapSe, self.se_dependence(None)),
                    };
                    let mut binding = IntervalBinding::new(method, published.level, dependence);
                    binding.replicates_ok = estimate.uncertainty.bootstrap.replicates_ok;
                    binding.replicates_failed = estimate.uncertainty.bootstrap.replicates_failed;
                    return binding;
                }
                M::AnalyticSe => {
                    let mut binding = IntervalBinding::new(
                        M::AnalyticSe,
                        published.level,
                        if panel_interval || randomized_cluster_interval {
                            "cluster"
                        } else {
                            self.se_dependence(estimate.uncertainty.analytic.kind)
                        },
                    );
                    binding.se_kind = estimate.uncertainty.analytic.kind;
                    return binding;
                }
                M::AndersonRubin => {
                    return IntervalBinding::new(M::AndersonRubin, published.level, base);
                }
                M::None if published.withheld_reason.is_some() => {
                    return IntervalBinding::none(base);
                }
                _ => {}
            }
        }
        let Some(effect_estimate) = self.estimate.as_effect() else {
            return IntervalBinding::none(base);
        };
        if !effect_estimate.ate.is_finite() {
            if let Some(bands) = effect_estimate.scenarios.intervals.as_deref() {
                if !bands.is_empty() {
                    return IntervalBinding::new(
                        M::SimultaneousBand,
                        REPORTED_SE_INTERVAL_LEVEL,
                        base,
                    );
                }
            }
        }
        self.identified_set_interval_binding().unwrap_or_else(|| IntervalBinding::none(base))
    }

    /// Construction of a reported identified-set interval
    /// (`structural_response.identified_set_interval`), when one was reported.
    ///
    /// A class-aware scalar can report both a frozen-weight point with its
    /// shared-block SE and an Imbens–Manski interval for the identified set;
    /// each states its own calibration.
    #[must_use]
    pub fn identified_set_interval_binding(&self) -> Option<IntervalBinding> {
        use antecedent_estimate::IdentifiedSetIntervalMethod as Method;
        let structural = self.structural_response.as_ref()?;
        let interval = structural.identified_set_interval.as_ref()?;
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::IdentifiedSet,
            interval.level,
            self.base_dependence(),
        );
        match interval.method {
            Method::ImbensManskiSharedBlock => {
                binding.dependence =
                    circular_block_dependence(antecedent_estimate::CircularBlockFamily::Mixture);
                binding.replicates_ok = self
                    .estimate
                    .as_effect()
                    .and_then(|effect| effect.uncertainty.bootstrap.replicates_ok);
                binding.replicates_failed = self
                    .estimate
                    .as_effect()
                    .and_then(|effect| effect.uncertainty.bootstrap.replicates_failed);
            }
            Method::ProductPosteriorEnvelopeQuantile => {
                binding.posterior_draws = structural
                    .atoms
                    .iter()
                    .filter_map(|atom| atom.posterior.as_ref())
                    .filter_map(|posterior| draws_u32(posterior.draws.n_draws))
                    .min()
                    .or_else(|| self.posterior_draw_count());
            }
            _ => {}
        }
        Some(binding)
    }

    /// Construction of a counterfactual's published per-unit intervals
    /// (`counterfactual.unit_effect_intervals`), when the execution formed them.
    #[must_use]
    pub fn unit_effect_interval_binding(&self) -> Option<IntervalBinding> {
        let intervals = match self.counterfactual() {
            Some(single) => single.unit_effect_intervals.as_ref()?,
            None => self.counterfactual_contrasts()?.primary().unit_effect_intervals.as_ref()?,
        };
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::UnitPosteriorQuantile,
            intervals.level,
            self.base_dependence(),
        );
        binding.posterior_draws = self.posterior_draw_count();
        Some(binding)
    }

    /// Construction of a temporal response's published simultaneous band
    /// (support diagnostic `response.simultaneous_band.critical`), when one
    /// accompanies a pointwise band.
    #[must_use]
    pub fn simultaneous_band_binding(&self, bayesian: bool) -> Option<IntervalBinding> {
        let response = self.response()?;
        if matches!(
            response.uncertainty,
            antecedent_core::ResponseUncertainty::SimultaneousBand { .. }
        ) {
            return None;
        }
        let critical = support_values(response, antecedent_estimate::SIMULTANEOUS_BAND_CRITICAL)?;
        let level = *critical.first()?;
        let mut binding = IntervalBinding::new(
            antecedent_core::IntervalMethod::SimultaneousBand,
            level,
            self.primary_interval_binding(bayesian).dependence,
        );
        binding.replicates_ok = critical.get(2).and_then(|n| count_u32(*n));
        if bayesian || self.posterior.is_some() {
            binding.posterior_draws = binding.replicates_ok.take();
        }
        Some(binding)
    }

    /// Every interval this execution reported: the primary interval first,
    /// then an identified-set interval and a simultaneous band when they are
    /// reported beside it. Each states its own calibration on the claim.
    #[must_use]
    pub fn reported_interval_bindings(&self, bayesian: bool) -> Vec<IntervalBinding> {
        let primary = self.primary_interval_binding(bayesian);
        let mut out = vec![primary];
        for extra in [
            self.identified_set_interval_binding(),
            self.simultaneous_band_binding(bayesian),
            self.unit_effect_interval_binding(),
        ]
        .into_iter()
        .flatten()
        {
            if out.iter().all(|seen| seen.method != extra.method) {
                out.push(extra);
            }
        }
        out
    }

    /// Re-derive [`Self::interval`] from the reported content.
    pub(crate) fn rebind_interval(&mut self, bayesian: bool) {
        self.extras.interval = Some(self.primary_interval_binding(bayesian));
    }

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
            allowlist_reason: self
                .support_status
                .and_then(crate::support::CellStatus::allowlist_reason)
                .map(str::to_string),
            allowlist_parent: self
                .support_status
                .and_then(crate::support::CellStatus::allowlist_parent)
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
