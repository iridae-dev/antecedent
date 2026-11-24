//! Estimate of preference parameters from recorded choices read as noisy.
//!
//! The choice model is named in every result ([`ASSUMPTION_LOGIT_CHOICE`]): the probability that a
//! decision-maker chose option `i` of a record is proportional to `exp(lambda * V_i(theta))`, where
//! `V` is the declared value of the option under its recorded expectations and `lambda > 0` is
//! choice consistency (larger means choices follow the value more closely). With
//! `V_i = F_i + sum_k theta_k T_ik` (from [`choice_features`]) this is a conditional logit in
//! `beta = (lambda, lambda * theta)` on the features `(F_i, T_i1, ..., T_iK)`, and
//! `theta_k = beta_k / beta_0`.
//!
//! Only available options enter the fit: an option listed as
//! [`OptionBelief::Unavailable`](crate::choice_record::OptionBelief::Unavailable) existed but could
//! not be chosen, so it teaches nothing about preferences and is not a row of its record's choice
//! set. A record with fewer than two available options is left out of the fit (a choice set needs
//! two alternatives): it counts toward the learnability of no parameter and is not counted in
//! [`ChoiceModelEstimate::n_records`].
//!
//! Stated comparisons ([`BeliefSource::Stated`](crate::choice_record::BeliefSource::Stated):
//! what someone said they would choose, not what they chose) and recorded choices need not share
//! a choice-consistency scale, so one fit never mixes them: a fit whose records hold both is
//! refused ([`ChoiceModelError::StatedAndRecordedMixed`]). Fit them separately, or use the
//! compatible-set method, which does not depend on a noise scale. A fit on stated comparisons
//! alone stands on their named assumption as well as on the logit choice model.
//!
//! What the estimate is: the trade-offs the recorded decision-makers acted on, under the stated
//! choice model and the recorded expectations of each option. What it is not: it is not what the
//! trade-offs should be; the interval's calibration status is resolved from the shipped coverage
//! table under [`calibration_basis`] ([`DistributionCalibration::Measured`] only when a current,
//! attesting record measured this construction at this size, level and covariance, otherwise
//! [`DistributionCalibration::NotMeasured`]), never supplied by a caller; and the measure over
//! parameter values is the normal approximation to the sampling distribution of the estimate,
//! not a posterior.
//!
//! Design. Column 0 is the fixed part `F` and carries the unit of account: unless it differs
//! between the options of at least one record, the scale of `theta` is not determined and the
//! estimate is refused. One further column follows per learnable parameter, in model order. A
//! column *differs* within a record when the spread of its values over the record's options
//! exceeds [`DIFFERENCE_RELATIVE_TOLERANCE`] times the largest absolute value of that column over
//! all options in the fit. A parameter fixed by declaration (`lower == upper`) has been folded
//! into `F` by [`choice_features`]: it is fixed, neither learnable nor listed as not learnable. A
//! free parameter whose term differs in no record is not learnable: it is left out of the fit,
//! listed in [`ChoiceModelEstimate::not_learnable`] and marked unresolved in the measure.
//!
//! Covariance. The inverse observed information when the fit holds one decision-maker. With
//! several, records of one decision-maker are not independent, so the covariance is clustered by
//! decider: the sandwich with per-decider score sums, multiplied by `C / (C - 1)` for `C`
//! deciders (the finite-cluster correction recommended for maximum-likelihood estimators by
//! Cameron and Miller, 2015, *A Practitioner's Guide to Cluster-Robust Inference*). The `C`
//! cluster score sums add up to the score, which is zero at the estimate, so the sandwich has at
//! most `C - 1` degrees of freedom; it is used only when `C - 1` is at least
//! [`CLUSTER_DEGREES_PER_PARAMETER`] times the number of coefficients `q`, that is
//! `C >= 2 q + 1`. Fewer deciders are refused ([`ChoiceModelError::TooFewDecidersToCluster`]);
//! the model covariance is never substituted, since it would treat dependent records as
//! independent. The result says which covariance it uses. The standard error of `theta_k` is the
//! delta-method value `sqrt(V_kk - 2 theta_k V_k0 + theta_k^2 V_00) / beta_0`, and the interval is
//! `theta_hat +- z * se` with `z` the two-sided normal quantile at the level.
//!
//! Measure. Draws `beta = beta_hat + L z`, with `L` the Cholesky factor of the chosen covariance
//! and `z` standard normal from the seeded generator, give
//! `theta_k = beta_k / beta_0`. A draw is kept when `beta_0 > 0` and its point lies in the
//! declared set: each learnable coordinate must lie in its declared range, every relation none of
//! whose terms touches an unresolved parameter must hold (both within [`SET_TOLERANCE`]), and the
//! point must pass the check a measure point with these unresolved parameters passes when the
//! measure is read by an evaluation: some value of the unresolved parameters in their declared
//! ranges completes it to a point of the ranges and linear relations (a small linear program over
//! the slice through the draw). A kept draw is therefore never refused by the evaluation that
//! reads the measure, and every dropped draw counts in the share drawn outside the declared set; a
//! draw whose slice program refuses is dropped. Fixed parameters sit at their fixed value;
//! unresolved parameters sit at the midpoint of their declared range, a placeholder the measure
//! says nothing about.
//!
//! Shape parameters. When the value is not affine in some free parameters (a risk attitude inside
//! an exponential, a breakpoint inside a kink), `V = F(theta_S) + sum_k theta_k T_k(theta_S)` and
//! the model is fitted by [`fit_nonlinear_conditional_logit`]: the profile over the shape
//! parameters, each point an ordinary conditional logit, searched on a Sobol grid over the
//! declared box, refined, and polished by Fisher scoring in the direct parameters
//! `(lambda, theta_L, theta_S)`; the information uses first derivatives of the value only
//! (forward mode through
//! [`PreferenceExpr::evaluate_with_gradient`](crate::preference_model::PreferenceExpr::evaluate_with_gradient)),
//! and the cluster sandwich uses per-decider score sums as above. A shape parameter the
//! within-record differences of the value provably never depend on (every option of every record
//! of the fit has the same rows of the attributes that move with it) is not learnable: it is
//! listed in [`ChoiceModelEstimate::not_learnable`] and left out of the fit, as a linear
//! parameter whose term never differs is. A shape parameter whose profile varies by less than
//! `chi2_1(level) / 2` over its declared range is refused
//! ([`ChoiceModelError::ShapeNotIdentified`]): the linear estimates would rest on an arbitrary
//! shape value. So is an estimate on a declared bound ([`ChoiceModelError::ShapeEstimateAtBound`],
//! the normal approximation does not hold there; the profile likelihood-ratio interval truncated
//! at the bound is reported) and a profile with two maxima separated by a deeper dip
//! ([`ChoiceModelError::ProfileMultimodal`]). Linear parameters then carry Wald intervals from the
//! covariance of the direct parameters, shape parameters profile likelihood-ratio intervals
//! ([`ParameterIntervalMethod`]), and the measure draws the direct parameters from the normal
//! approximation, kept inside the declared set. The calibration key is
//! [`shape_calibration_basis`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt;

use antecedent_contract::ExternalRefusal;
use antecedent_core::{CalibrationBasis, CanonicalWriter, CausalRng, ExecutionContext};
use antecedent_io_core::calibration::{CalibrationBasisWire, calibration_slot};
use antecedent_io_core::contract_section::CalibrationStatus;
use antecedent_io_core::distribution_artifact::DistributionCalibration;
use antecedent_kernels::{cholesky_spd_into, gauss_jordan_solve, norm_inv, standard_normal};
use antecedent_stats::linear_program::LpOptions;
use antecedent_stats::{
    ChoiceUtility, ConditionalLogitDesign, ConditionalLogitError, ConditionalLogitOptions,
    NonlinearChoiceError, NonlinearConditionalLogitFit, NonlinearConditionalLogitOptions,
    SOBOL_DIMENSIONS, ShapeSearchOptions, UtilityNotEvaluable, fit_conditional_logit,
    fit_conditional_logit_with_context, fit_nonlinear_conditional_logit,
};

use crate::choice_record::{
    ChoiceRecordError, ChoiceRecordSet, ChoiceSetFeatures, Pooling, ShapeData, choice_features,
    shape_data,
};
use crate::preference_decision::{MeasureOrigin, PreferenceMeasure, measure_point_accepted};
use crate::preference_model::{PreferenceModelError, RelationSense, ValueModel};
use crate::preference_refusal::{COMPATIBLE_SET, ESTIMATE, refusal};
use crate::signal::at_stage;

/// Identity of the choice-noise assumption every estimate stands on: the probability of choosing
/// an option is proportional to `exp(lambda * V)` in its declared value `V`, with `lambda > 0`.
pub const ASSUMPTION_LOGIT_CHOICE: &str = "assumption.choice_probability_logit_in_declared_value";

/// A design column differs within a record when the spread of its values over the record's
/// options exceeds this fraction of the column's largest absolute value over the fit.
pub const DIFFERENCE_RELATIVE_TOLERANCE: f64 = 1e-9;

/// Absolute tolerance on declared ranges and relations when a measure draw is checked against
/// the declared set.
pub const SET_TOLERANCE: f64 = 1e-9;

/// Query axis of every coverage record of the choice-model parameter intervals.
pub const CALIBRATION_QUERY: &str = "PreferenceTradeOffFromChoices";
/// Estimator of the parameter intervals: the conditional logit ratio `beta_k / beta_0`.
pub const CALIBRATION_ESTIMATOR: &str = "preference.conditional_logit.parameter_ratio";
/// Interval method of the parameter intervals: `theta_hat +- z se` with the delta-method
/// standard error.
pub const CALIBRATION_INTERVAL_METHOD: &str = "wald_delta_method";

/// The calibration key of the parameter intervals of one estimate: the covariance kind, the
/// number of learnable parameters, the records in the fit and the nominal level. The key
/// carries no authority: an interval reads as measured only when the key resolves to a current,
/// attesting coverage record in the shipped table.
#[must_use]
pub fn calibration_basis(
    kind: CovarianceKind,
    learnable: usize,
    records: u64,
    level: f64,
) -> CalibrationBasis {
    let a = |text: &str| std::sync::Arc::<str>::from(text);
    let (se_kind, dependence) = match kind {
        CovarianceKind::Model => ("observed_information", "independent_records"),
        CovarianceKind::ClusteredByDecider => {
            ("clustered_by_decider", "records_clustered_by_decider")
        }
    };
    CalibrationBasis::new(
        [
            a(CALIBRATION_QUERY),
            a("NotApplicable"),
            a("fixed"),
            a("recorded_choices"),
            a("Frequentist"),
            a(CALIBRATION_ESTIMATOR),
            a(CALIBRATION_INTERVAL_METHOD),
            a(se_kind),
            a(dependence),
            a(""),
            std::sync::Arc::from(format!("trade_off_ratio.learnable_{learnable}")),
        ],
        level,
        a("point"),
        records,
        None,
        None,
        0.0,
    )
    .with_validation("not_applicable")
}

/// Estimator of the intervals of an estimate with shape parameters: the direct parameters of the
/// conditional logit profiled over the shape parameters.
pub const CALIBRATION_SHAPE_ESTIMATOR: &str = "preference.nonlinear_conditional_logit.profile";
/// Interval method of an estimate with shape parameters: Wald intervals for the linear
/// parameters, profile likelihood-ratio intervals for the shape parameters.
pub const CALIBRATION_SHAPE_INTERVAL_METHOD: &str = "wald_and_profile_likelihood_ratio";

/// The calibration key of the parameter intervals of an estimate with shape parameters: the
/// covariance kind, the numbers of learnable linear and fitted shape parameters, the records in
/// the fit and the nominal level. As [`calibration_basis`], the key carries no authority.
#[must_use]
pub fn shape_calibration_basis(
    kind: CovarianceKind,
    learnable: usize,
    shape: usize,
    records: u64,
    level: f64,
) -> CalibrationBasis {
    let a = |text: &str| std::sync::Arc::<str>::from(text);
    let (se_kind, dependence) = match kind {
        CovarianceKind::Model => ("expected_information", "independent_records"),
        CovarianceKind::ClusteredByDecider => {
            ("clustered_by_decider", "records_clustered_by_decider")
        }
    };
    CalibrationBasis::new(
        [
            a(CALIBRATION_QUERY),
            a("NotApplicable"),
            a("fixed"),
            a("recorded_choices"),
            a("Frequentist"),
            a(CALIBRATION_SHAPE_ESTIMATOR),
            a(CALIBRATION_SHAPE_INTERVAL_METHOD),
            a(se_kind),
            a(dependence),
            a(""),
            std::sync::Arc::from(format!("shape_trade_off.learnable_{learnable}.shape_{shape}")),
        ],
        level,
        a("point"),
        records,
        None,
        None,
        0.0,
    )
    .with_validation("not_applicable")
}

/// The calibration status of the intervals under `basis`, resolved from the shipped coverage
/// table: measured only when the slot is calibrated, not measured otherwise (no record, a stale
/// record, or an execution outside what the record measured).
fn interval_calibration(basis: &CalibrationBasis) -> DistributionCalibration {
    match calibration_slot(&CalibrationBasisWire::from(basis)).status {
        CalibrationStatus::Calibrated => DistributionCalibration::Measured,
        _ => DistributionCalibration::NotMeasured,
    }
}

/// Degrees of freedom of the clustered covariance required per estimated coefficient: with `C`
/// deciders and `q` coefficients the clustered covariance is used only when
/// `C - 1 >= CLUSTER_DEGREES_PER_PARAMETER * q`.
pub const CLUSTER_DEGREES_PER_PARAMETER: usize = 2;

/// Smallest number of deciders whose clustered covariance is used for `q` coefficients.
const fn clusters_required(q: usize) -> usize {
    CLUSTER_DEGREES_PER_PARAMETER * q + 1
}

/// An earlier column takes part in a dependency when its share of the dependent column, after
/// least-squares projection, exceeds this fraction of the dependent column's norm.
const INVOLVED_RELATIVE: f64 = 1e-6;

/// Controls for [`estimate_choice_model`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChoiceModelOptions {
    /// Two-sided level of each parameter interval; strictly between 0 and 1.
    pub level: f64,
    /// Number of draws from the normal approximation for the measure; at least one.
    pub n_draws: usize,
    /// Seed of the draw generator.
    pub seed: u64,
    /// Controls of the conditional logit fit.
    pub fit: ConditionalLogitOptions,
    /// Controls of the search over shape parameters; read only when the value has shape
    /// parameters.
    pub shape_search: ShapeSearchOptions,
}

impl Default for ChoiceModelOptions {
    fn default() -> Self {
        Self {
            level: 0.95,
            n_draws: 2000,
            seed: 0,
            fit: ConditionalLogitOptions::default(),
            shape_search: ShapeSearchOptions::default(),
        }
    }
}

/// How a parameter interval was formed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterIntervalMethod {
    /// `estimate +- z * standard_error` from the chosen covariance (with the delta method for the
    /// ratio `beta_k / beta_0` when the value has no fitted shape parameter).
    Wald,
    /// `{v : 2 (l_max - l_p(v)) <= chi2_1(level)}` from the profile log-likelihood of a shape
    /// parameter.
    ProfileLikelihood,
}

/// Which covariance of the coefficients an estimate uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CovarianceKind {
    /// Inverse observed information (one decision-maker in the fit).
    Model,
    /// Sandwich with per-decider score sums times `C / (C - 1)` for `C` decision-makers; used
    /// only when `C >= 2 q + 1` for `q` coefficients.
    ClusteredByDecider,
}

impl CovarianceKind {
    const fn tag(self) -> u64 {
        match self {
            Self::Model => 1,
            Self::ClusteredByDecider => 2,
        }
    }
}

/// Estimate of one learnable parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterEstimate {
    /// Identity of the parameter.
    pub parameter_id: String,
    /// Point estimate `beta_k / beta_0`.
    pub estimate: f64,
    /// Delta-method standard error from the chosen covariance.
    pub standard_error: f64,
    /// Two-sided interval at the estimate's level, formed as `interval_method` states; its
    /// coverage is not measured.
    pub interval: (f64, f64),
    /// Declared `(lower, upper)` range.
    pub declared: (f64, f64),
    /// Whether the point estimate lies in the declared range.
    pub inside_declared: bool,
    /// How `interval` was formed.
    pub interval_method: ParameterIntervalMethod,
}

/// Estimate of the trade-offs one group of recorded decision-makers acted on, under the logit
/// choice model ([`ASSUMPTION_LOGIT_CHOICE`]) and the recorded expectations. It is not a statement
/// of what the trade-offs should be.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceModelEstimate {
    /// Canonical digest of the model identity, records digest, decider (or the pooled marker),
    /// level, draw count, seed, fit controls and covariance kind.
    pub identity: String,
    /// Identity of the value model.
    pub model_identity: String,
    /// Digest of the whole record set.
    pub records_digest: String,
    /// The decision-maker of a per-decider fit; `None` for the pooled fit.
    pub decider: Option<String>,
    /// Choice consistency `lambda = beta_0`.
    pub consistency: f64,
    /// Standard error of `consistency` from the chosen covariance.
    pub consistency_standard_error: f64,
    /// Learnable parameters, in model order.
    pub parameters: Vec<ParameterEstimate>,
    /// Identities of free parameters whose term differs in no record, in model order. Parameters
    /// fixed by declaration are not listed.
    pub not_learnable: Vec<String>,
    /// Log-likelihood at the estimate.
    pub log_likelihood: f64,
    /// Log-likelihood with every coefficient zero (each option equally likely).
    pub null_log_likelihood: f64,
    /// Records used in the fit: those with at least two available options. A record with a single
    /// available option is left out and not counted.
    pub n_records: usize,
    /// Distinct decision-makers in the fit.
    pub n_deciders: usize,
    /// Which covariance the standard errors, intervals and measure use.
    pub covariance_kind: CovarianceKind,
    /// The chosen covariance of `beta = (lambda, lambda * theta_learnable...)`, row-major
    /// `(1 + parameters.len())` square.
    pub coefficient_covariance: Vec<f64>,
    /// Level of the parameter intervals.
    pub level: f64,
    /// Calibration status of the intervals, resolved from the shipped coverage table under
    /// [`calibration_basis`]: [`DistributionCalibration::Measured`] when a current, attesting
    /// record measured this construction at this size and level, otherwise
    /// [`DistributionCalibration::NotMeasured`].
    pub interval_calibration: DistributionCalibration,
    /// Draws from the normal approximation to the sampling distribution of the estimate, mapped
    /// to parameter values and kept inside the declared set; not a posterior. Equal weights;
    /// `unresolved` lists the not-learnable parameter indices, whose coordinates (the declared
    /// midpoint) are placeholders.
    pub measure: PreferenceMeasure,
    /// Share of draws dropped because `beta_0 <= 0` or the point lies outside the declared set,
    /// including draws that no value of the unresolved parameters completes to a point of the
    /// set.
    pub share_outside_declared: f64,
    /// Sorted, distinct: parameter assumptions, the record set's assumptions and
    /// [`ASSUMPTION_LOGIT_CHOICE`].
    pub assumptions: Vec<String>,
    /// Identities of the fitted shape parameters, in model order. Empty when the value has no
    /// fitted shape parameter: `coefficient_covariance` is then the covariance of
    /// `beta = (lambda, lambda * theta)`. Otherwise it is the covariance of the direct parameters
    /// `(lambda, theta_learnable..., theta_shape...)`: the learnable linear parameters and then
    /// these shape parameters, each in model order.
    pub shape_parameters: Vec<String>,
}

/// The outcome of one per-decider fit.
#[derive(Clone, Debug, PartialEq)]
pub enum EstimateOutcome {
    /// The estimate.
    Estimated(Box<ChoiceModelEstimate>),
    /// Why this decision-maker's choices give no estimate.
    Refused(ChoiceModelError),
}

/// Estimates for a record set.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceModelResult {
    /// The estimate over every record, present exactly for [`Pooling::Pooled`].
    pub pooled: Option<Box<ChoiceModelEstimate>>,
    /// One outcome per decision-maker, sorted by decider, for [`Pooling::PerDecider`]; empty for
    /// pooled records.
    pub per_decider: Vec<(String, EstimateOutcome)>,
}

/// Why an estimate is refused. `decider` is `None` for the pooled fit.
#[derive(Clone, Debug, PartialEq)]
pub enum ChoiceModelError {
    /// The interval level is not strictly between 0 and 1.
    LevelInvalid {
        /// The level given.
        level: f64,
    },
    /// No measure draws were requested.
    DrawsMissing,
    /// The fit's gradient tolerance or separation bound is not finite and positive.
    FitOptionsInvalid,
    /// The value model is invalid.
    Model(PreferenceModelError),
    /// The records are invalid or do not fit the model.
    Records(ChoiceRecordError),
    /// The records of the fit hold both stated comparisons and recorded choices, which need not
    /// share a choice-consistency scale. Fit them separately, or use the compatible-set method,
    /// which does not depend on a noise scale.
    StatedAndRecordedMixed {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// Number of stated comparisons in the fit.
        stated: usize,
        /// Number of recorded choices in the fit.
        recorded: usize,
    },
    /// The fixed part never differs between the options of a record, so the scale of the
    /// parameters is not determined.
    NoUnitOfAccount {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
    },
    /// Neither the fixed part nor any parameter term differs within any record.
    NothingIdentified {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
    },
    /// The within-record differences of the design are linearly dependent.
    RankDeficient {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// Whether the unit of account (the fixed part) takes part in the dependency.
        unit_of_account: bool,
        /// Identities of the learnable parameters taking part, in model order.
        parameters: Vec<String>,
    },
    /// The choices are perfectly consistent with a value in the family, so the likelihood has
    /// no finite maximum.
    Separated {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
    },
    /// The estimated consistency is not positive: choices run against the unit of account.
    ConsistencyNotPositive {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// The estimated consistency.
        consistency: f64,
    },
    /// The fit did not converge.
    NotConverged {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// Accepted Newton steps.
        iterations: usize,
        /// Largest absolute gradient entry at the last iterate.
        gradient_norm: f64,
    },
    /// The conditional logit fit refused for another reason (such as numerically singular
    /// information).
    Fit {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// The fit's refusal.
        error: ConditionalLogitError,
    },
    /// The records come from several decision-makers, so the covariance must be clustered by
    /// decider, but there are too few of them for the clustered covariance of this many
    /// coefficients: its degrees of freedom (one fewer than the deciders) must be at least
    /// [`CLUSTER_DEGREES_PER_PARAMETER`] per coefficient.
    TooFewDecidersToCluster {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// Distinct decision-makers in the fit.
        deciders: usize,
        /// Coefficients estimated.
        coefficients: usize,
        /// Decision-makers required.
        required: usize,
    },
    /// The chosen covariance has no Cholesky factor.
    CovarianceNotPositiveDefinite {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// Which covariance was chosen.
        kind: CovarianceKind,
    },
    /// Every measure draw had `beta_0 <= 0` or fell outside the declared set.
    NoDrawInsideSet {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// Number of draws made.
        drawn: usize,
    },
    /// The execution context was cancelled between Newton steps, fits or measure draws.
    Cancelled,
    /// The choices are perfectly consistent with a value in the family at a value of the shape
    /// parameters, so the likelihood has no finite maximum.
    SeparatedAtShape {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// The shape parameters and their values where separation was found.
        shape: Vec<(String, f64)>,
    },
    /// The profile log-likelihood of a shape parameter varies by less than `chi2_1(level) / 2`
    /// over its declared range.
    ShapeNotIdentified {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// The shape parameter.
        parameter: String,
        /// Largest minus smallest profile log-likelihood over the range.
        profile_range: f64,
    },
    /// The estimate of a shape parameter lies on a declared bound.
    ShapeEstimateAtBound {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// The shape parameter.
        parameter: String,
        /// The estimate.
        estimate: f64,
        /// The profile likelihood-ratio interval, truncated at the bound.
        interval: (f64, f64),
    },
    /// The profile of a shape parameter has two maxima close to the overall maximum separated by
    /// a deeper dip.
    ProfileMultimodal {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// The shape parameter.
        parameter: String,
        /// The two maximising values.
        modes: (f64, f64),
    },
    /// More shape parameters to fit than the search covers.
    ShapeSearchBudget {
        /// The decision-maker, `None` for the pooled fit.
        decider: Option<String>,
        /// Shape parameters to fit.
        dimensions: usize,
        /// The search limit.
        limit: usize,
    },
}

impl ChoiceModelError {
    /// Stable detail identity of the refusal; a record refusal keeps its own `choice_record.`
    /// detail, the detail the returned refusal carries.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::LevelInvalid { .. } => reason_details::preference_choice_model::LEVEL_INVALID,
            Self::DrawsMissing => reason_details::preference_choice_model::DRAWS_MISSING,
            Self::FitOptionsInvalid => reason_details::preference_choice_model::FIT_OPTIONS_INVALID,
            Self::Model(_) => reason_details::preference_choice_model::MODEL_INVALID,
            Self::Records(inner) => inner.detail(),
            Self::StatedAndRecordedMixed { .. } => {
                reason_details::preference_choice_model::STATED_AND_RECORDED_MIXED
            }
            Self::NoUnitOfAccount { .. } => {
                reason_details::preference_choice_model::UNIT_OF_ACCOUNT_MISSING
            }
            Self::NothingIdentified { .. } => {
                reason_details::preference_choice_model::NOTHING_IDENTIFIED
            }
            Self::RankDeficient { .. } => reason_details::preference_choice_model::RANK_DEFICIENT,
            Self::Separated { .. } | Self::SeparatedAtShape { .. } => {
                reason_details::preference_choice_model::SEPARATED
            }
            Self::ConsistencyNotPositive { .. } => {
                reason_details::preference_choice_model::CONSISTENCY_NOT_POSITIVE
            }
            Self::NotConverged { .. } => reason_details::preference_choice_model::NOT_CONVERGED,
            Self::Fit { .. } => reason_details::preference_choice_model::FIT_REFUSED,
            Self::TooFewDecidersToCluster { .. } => {
                reason_details::preference_choice_model::TOO_FEW_DECIDERS_TO_CLUSTER
            }
            Self::CovarianceNotPositiveDefinite { .. } => {
                reason_details::preference_choice_model::COVARIANCE_NOT_POSITIVE_DEFINITE
            }
            Self::NoDrawInsideSet { .. } => {
                reason_details::preference_choice_model::NO_DRAW_INSIDE_SET
            }
            Self::Cancelled => reason_details::preference_choice_model::CANCELLED,
            Self::ShapeNotIdentified { .. } => {
                reason_details::preference_choice_model::SHAPE_NOT_IDENTIFIED
            }
            Self::ShapeEstimateAtBound { .. } => {
                reason_details::preference_choice_model::SHAPE_ESTIMATE_AT_BOUND
            }
            Self::ProfileMultimodal { .. } => {
                reason_details::preference_choice_model::PROFILE_MULTIMODAL
            }
            Self::ShapeSearchBudget { .. } => {
                reason_details::preference_choice_model::SHAPE_SEARCH_BUDGET
            }
        }
    }
}

fn who(decider: Option<&str>) -> String {
    decider.map_or_else(|| "pooled records".to_owned(), |d| format!("decider `{d}`"))
}

impl fmt::Display for ChoiceModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LevelInvalid { level } => {
                write!(f, "interval level {level} is not strictly between 0 and 1")
            }
            Self::DrawsMissing => write!(f, "at least one measure draw is needed"),
            Self::FitOptionsInvalid => {
                write!(f, "fit gradient tolerance and separation bound must be finite and positive")
            }
            Self::Model(error) => write!(f, "invalid value model ({})", error.detail()),
            Self::Records(error) => write!(f, "choice records refused: {error}"),
            Self::StatedAndRecordedMixed { decider, stated, recorded } => write!(
                f,
                "{}: {stated} stated comparisons and {recorded} recorded choices need not share a noise scale; fit them separately or use the compatible-set method",
                who(decider.as_deref())
            ),
            Self::NoUnitOfAccount { decider } => write!(
                f,
                "{}: the fixed part of the value never differs between options, so the scale of the parameters is not determined",
                who(decider.as_deref())
            ),
            Self::NothingIdentified { decider } => write!(
                f,
                "{}: neither the fixed part nor any parameter term differs between options of any record",
                who(decider.as_deref())
            ),
            Self::RankDeficient { decider, unit_of_account, parameters } => write!(
                f,
                "{}: the choice design is rank-deficient; dependent columns: unit of account {}, parameters {parameters:?}",
                who(decider.as_deref()),
                if *unit_of_account { "involved" } else { "not involved" }
            ),
            Self::Separated { decider } => write!(
                f,
                "{}: choices are perfectly consistent with a value in the family; use the compatible-set method",
                who(decider.as_deref())
            ),
            Self::ConsistencyNotPositive { decider, consistency } => write!(
                f,
                "{}: estimated choice consistency {consistency} is not positive; choices run against the unit of account",
                who(decider.as_deref())
            ),
            Self::NotConverged { decider, iterations, gradient_norm } => write!(
                f,
                "{}: fit did not converge after {iterations} iterations (gradient norm {gradient_norm})",
                who(decider.as_deref())
            ),
            Self::Fit { decider, error } => {
                write!(f, "{}: conditional logit fit refused: {error}", who(decider.as_deref()))
            }
            Self::TooFewDecidersToCluster { decider, deciders, coefficients, required } => write!(
                f,
                "{}: {deciders} decision-makers are too few to cluster the covariance of {coefficients} coefficients; at least {required} are needed",
                who(decider.as_deref())
            ),
            Self::CovarianceNotPositiveDefinite { decider, kind } => write!(
                f,
                "{}: the {kind:?} covariance is not positive definite",
                who(decider.as_deref())
            ),
            Self::NoDrawInsideSet { decider, drawn } => write!(
                f,
                "{}: none of {drawn} draws has positive consistency inside the declared set",
                who(decider.as_deref())
            ),
            Self::Cancelled => write!(f, "estimating the choice model was cancelled"),
            Self::SeparatedAtShape { decider, shape } => write!(
                f,
                "{}: choices are perfectly consistent with a value in the family at shape values {shape:?}; use the compatible-set method",
                who(decider.as_deref())
            ),
            Self::ShapeNotIdentified { decider, parameter, profile_range } => write!(
                f,
                "{}: the profile log-likelihood of shape parameter `{parameter}` varies by only {profile_range} over its declared range",
                who(decider.as_deref())
            ),
            Self::ShapeEstimateAtBound { decider, parameter, estimate, interval } => write!(
                f,
                "{}: shape parameter `{parameter}` is estimated on a declared bound ({estimate}); profile interval {interval:?}",
                who(decider.as_deref())
            ),
            Self::ProfileMultimodal { decider, parameter, modes } => write!(
                f,
                "{}: the profile of shape parameter `{parameter}` has two maxima at {modes:?}",
                who(decider.as_deref())
            ),
            Self::ShapeSearchBudget { decider, dimensions, limit } => write!(
                f,
                "{}: {dimensions} shape parameters exceed the search limit of {limit}",
                who(decider.as_deref())
            ),
        }
    }
}

impl std::error::Error for ChoiceModelError {}

/// Estimate the preference parameters and choice consistency from recorded choices under the
/// logit choice model.
///
/// [`Pooling::Pooled`] fits every record together and reports it in
/// [`ChoiceModelResult::pooled`]; a refusal of that fit is this function's error.
/// [`Pooling::PerDecider`] fits each decision-maker's records separately, each with its own
/// [`EstimateOutcome`], and reports nothing pooled.
///
/// Records with fewer than two available options are left out of every fit. A fit whose records
/// hold both stated comparisons and recorded choices is refused with
/// [`ChoiceModelError::StatedAndRecordedMixed`].
///
/// # Errors
/// - [`ChoiceModelError::LevelInvalid`], [`ChoiceModelError::DrawsMissing`],
///   [`ChoiceModelError::FitOptionsInvalid`] for invalid options;
/// - [`ChoiceModelError::Model`] for an invalid model, [`ChoiceModelError::Records`] for invalid
///   records or records that do not fit the model ([`choice_features`]);
/// - for pooled records, any refusal of the pooled fit.
pub fn estimate_choice_model(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    options: ChoiceModelOptions,
) -> Result<ChoiceModelResult, ChoiceModelError> {
    estimate_all(model, records, options, None)
}

/// [`estimate_choice_model`] under an execution context: the context's cancellation token is
/// read before every fit, every Newton step and every 1024 measure draws. A cancelled run
/// returns [`ChoiceModelError::Cancelled`], never a per-decider refusal.
///
/// # Errors
/// As [`estimate_choice_model`]; [`ChoiceModelError::Cancelled`] once the token is set.
pub fn estimate_choice_model_with_context(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    options: ChoiceModelOptions,
    ctx: &ExecutionContext,
) -> Result<ChoiceModelResult, ChoiceModelError> {
    estimate_all(model, records, options, Some(ctx))
}

pub(crate) fn estimate_all(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    options: ChoiceModelOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<ChoiceModelResult, ChoiceModelError> {
    if !(options.level.is_finite() && options.level > 0.0 && options.level < 1.0) {
        return Err(ChoiceModelError::LevelInvalid { level: options.level });
    }
    if options.n_draws == 0 {
        return Err(ChoiceModelError::DrawsMissing);
    }
    let fit = options.fit;
    if !(fit.gradient_tolerance.is_finite()
        && fit.gradient_tolerance > 0.0
        && fit.separation_bound.is_finite()
        && fit.separation_bound > 0.0)
    {
        return Err(ChoiceModelError::FitOptionsInvalid);
    }
    let search = options.shape_search;
    if search.grid_points == 0
        || search.profile_points < 3
        || search.max_refinements == 0
        || search.max_dimensions > SOBOL_DIMENSIONS
    {
        return Err(ChoiceModelError::FitOptionsInvalid);
    }
    let model_identity = model.identity().map_err(ChoiceModelError::Model)?;
    let records_digest = records.digest().map_err(ChoiceModelError::Records)?;
    if model.shape_split().is_some_and(|split| !split.shape.is_empty()) {
        return estimate_with_shape(model, records, options, ctx, model_identity, records_digest);
    }
    let features = choice_features(model, records).map_err(ChoiceModelError::Records)?;
    let mut assumptions: BTreeSet<String> =
        model.preferences.parameters.iter().map(|p| p.assumption_id.clone()).collect();
    assumptions.extend(records.assumptions());
    assumptions.insert(ASSUMPTION_LOGIT_CHOICE.to_owned());
    let context = Context {
        model,
        options,
        model_identity,
        records_digest,
        n_parameters: features.n_parameters,
        assumptions: assumptions.into_iter().collect(),
        stated: records.stated_records().into_iter().collect(),
        ctx,
    };
    match &records.pooling {
        Pooling::Pooled { .. } => {
            let sets: Vec<&ChoiceSetFeatures> =
                features.sets.iter().filter(|s| informative(s)).collect();
            let estimate = context.estimate(&sets, None)?;
            Ok(ChoiceModelResult { pooled: Some(Box::new(estimate)), per_decider: Vec::new() })
        }
        Pooling::PerDecider => {
            let mut per_decider = Vec::new();
            for decider in records.deciders() {
                let sets: Vec<&ChoiceSetFeatures> = features
                    .sets
                    .iter()
                    .filter(|s| s.decider == decider && informative(s))
                    .collect();
                let outcome = match context.estimate(&sets, Some(decider.as_str())) {
                    Ok(estimate) => EstimateOutcome::Estimated(Box::new(estimate)),
                    Err(ChoiceModelError::Cancelled) => return Err(ChoiceModelError::Cancelled),
                    Err(error) => EstimateOutcome::Refused(error),
                };
                per_decider.push((decider, outcome));
            }
            Ok(ChoiceModelResult { pooled: None, per_decider })
        }
    }
}

/// Whether a record's choice set can enter the fit: at least two available options.
fn informative(set: &ChoiceSetFeatures) -> bool {
    set.option_ids.len() >= 2
}

/// [`estimate_all`] for a value with shape parameters.
fn estimate_with_shape(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    options: ChoiceModelOptions,
    ctx: Option<&ExecutionContext>,
    model_identity: String,
    records_digest: String,
) -> Result<ChoiceModelResult, ChoiceModelError> {
    let data = shape_data(model, records).map_err(ChoiceModelError::Records)?;
    let mut assumptions: BTreeSet<String> =
        model.preferences.parameters.iter().map(|p| p.assumption_id.clone()).collect();
    assumptions.extend(records.assumptions());
    assumptions.insert(ASSUMPTION_LOGIT_CHOICE.to_owned());
    let context = Context {
        model,
        options,
        model_identity,
        records_digest,
        n_parameters: data.n_parameters,
        assumptions: assumptions.into_iter().collect(),
        stated: records.stated_records().into_iter().collect(),
        ctx,
    };
    let informative: Vec<usize> =
        (0..data.records.len()).filter(|&m| data.records[m].options.len() >= 2).collect();
    match &records.pooling {
        Pooling::Pooled { .. } => {
            let estimate = context.estimate_shape(&data, &informative, None)?;
            Ok(ChoiceModelResult { pooled: Some(Box::new(estimate)), per_decider: Vec::new() })
        }
        Pooling::PerDecider => {
            let mut per_decider = Vec::new();
            for decider in records.deciders() {
                let members: Vec<usize> = informative
                    .iter()
                    .copied()
                    .filter(|&m| data.records[m].decider == decider)
                    .collect();
                let outcome = match context.estimate_shape(&data, &members, Some(decider.as_str()))
                {
                    Ok(estimate) => EstimateOutcome::Estimated(Box::new(estimate)),
                    Err(ChoiceModelError::Cancelled) => return Err(ChoiceModelError::Cancelled),
                    Err(error) => EstimateOutcome::Refused(error),
                };
                per_decider.push((decider, outcome));
            }
            Ok(ChoiceModelResult { pooled: None, per_decider })
        }
    }
}

/// The option utilities of the records of one fit under a value with shape parameters.
struct ShapeUtility<'d> {
    data: &'d ShapeData,
    members: &'d [usize],
    /// Learnable linear parameters, in model order.
    learnable: Vec<usize>,
    /// Fitted shape parameters, in model order.
    fitted: Vec<usize>,
    /// Every parameter's value where not fitted: fixed values, and declared midpoints.
    base: Vec<f64>,
    set_starts: Vec<usize>,
    chosen: Vec<usize>,
    clusters: Option<Vec<u32>>,
    /// The record refusal behind the last evaluation that failed.
    failure: RefCell<Option<ChoiceRecordError>>,
}

impl ShapeUtility<'_> {
    fn point(&self, shape: &[f64]) -> Vec<f64> {
        let mut point = self.base.clone();
        for (j, value) in self.fitted.iter().zip(shape) {
            point[*j] = *value;
        }
        point
    }

    fn parts(&self) -> Vec<&crate::preference_model::PreferenceExpr> {
        let form = &self.data.split.form;
        std::iter::once(&form.fixed).chain(self.learnable.iter().map(|j| &form.terms[*j])).collect()
    }

    fn fail(&self, error: ChoiceRecordError) -> UtilityNotEvaluable {
        *self.failure.borrow_mut() = Some(error);
        UtilityNotEvaluable
    }
}

impl ChoiceUtility for ShapeUtility<'_> {
    fn n_linear(&self) -> usize {
        self.learnable.len()
    }

    fn n_shape(&self) -> usize {
        self.fitted.len()
    }

    fn set_starts(&self) -> &[usize] {
        &self.set_starts
    }

    fn chosen(&self) -> &[usize] {
        &self.chosen
    }

    fn clusters(&self) -> Option<&[u32]> {
        self.clusters.as_deref()
    }

    fn features(&self, shape: &[f64]) -> Result<Vec<f64>, UtilityNotEvaluable> {
        let point = self.point(shape);
        let parts = self.parts();
        let mut out = Vec::new();
        for &m in self.members {
            for o in 0..self.data.records[m].options.len() {
                for part in &parts {
                    out.push(self.data.expectation(m, o, part, &point).map_err(|e| self.fail(e))?);
                }
            }
        }
        Ok(out)
    }

    fn shape_jacobian(&self, shape: &[f64]) -> Result<Vec<f64>, UtilityNotEvaluable> {
        let point = self.point(shape);
        let parts = self.parts();
        let mut out = Vec::new();
        for &m in self.members {
            for o in 0..self.data.records[m].options.len() {
                for part in &parts {
                    let (_, gradient) = self
                        .data
                        .expectation_with_gradient(m, o, part, &point)
                        .map_err(|e| self.fail(e))?;
                    out.extend(self.fitted.iter().map(|j| gradient[*j]));
                }
            }
        }
        Ok(out)
    }
}

/// What every fit of one call shares.
struct Context<'a> {
    model: &'a ValueModel,
    options: ChoiceModelOptions,
    model_identity: String,
    records_digest: String,
    n_parameters: usize,
    assumptions: Vec<String>,
    /// Identities of the stated comparisons among the records.
    stated: BTreeSet<String>,
    /// The caller's execution context, when the run is cancellable.
    ctx: Option<&'a ExecutionContext>,
}

/// The conditional logit design of one fit.
struct Design {
    p: usize,
    features: Vec<f64>,
    set_starts: Vec<usize>,
    chosen: Vec<usize>,
    clusters: Option<Vec<u32>>,
    n_deciders: usize,
}

/// What each design column is in one fit.
struct ColumnRoles {
    /// Whether the fixed part differs within some record.
    unit: bool,
    /// Free parameters whose term differs within some record, in model order.
    learnable: Vec<usize>,
    /// Free parameters whose term differs within no record, in model order.
    not_learnable: Vec<usize>,
}

impl Context<'_> {
    fn estimate(
        &self,
        sets: &[&ChoiceSetFeatures],
        decider: Option<&str>,
    ) -> Result<ChoiceModelEstimate, ChoiceModelError> {
        self.observe()?;
        let owned = || decider.map(str::to_owned);
        let stated = sets.iter().filter(|set| self.stated.contains(&set.record_id)).count();
        if stated > 0 && stated < sets.len() {
            return Err(ChoiceModelError::StatedAndRecordedMixed {
                decider: owned(),
                stated,
                recorded: sets.len() - stated,
            });
        }
        let parameters = &self.model.preferences.parameters;
        let columns = self.classify(sets);
        if !columns.unit {
            return Err(if columns.learnable.is_empty() {
                ChoiceModelError::NothingIdentified { decider: owned() }
            } else {
                ChoiceModelError::NoUnitOfAccount { decider: owned() }
            });
        }
        let (learnable, not_learnable) = (columns.learnable, columns.not_learnable);
        let design = build_design(sets, &learnable, self.n_parameters);
        let p = design.p;
        let logit = ConditionalLogitDesign {
            features: &design.features,
            n_features: p,
            set_starts: &design.set_starts,
            chosen: &design.chosen,
            clusters: design.clusters.as_deref(),
        };
        let fit = match self.ctx {
            Some(ctx) => fit_conditional_logit_with_context(&logit, self.options.fit, ctx),
            None => fit_conditional_logit(&logit, self.options.fit),
        }
        .map_err(|error| self.fit_refusal(error, &design, &learnable, decider))?;
        let beta = fit.coefficients;
        let consistency = beta.first().copied().unwrap_or(f64::NAN);
        if consistency.is_nan() || consistency <= 0.0 {
            return Err(ChoiceModelError::ConsistencyNotPositive { decider: owned(), consistency });
        }
        let (kind, covariance) = chosen_covariance(
            design.n_deciders,
            p,
            fit.cluster_covariance,
            fit.covariance,
            decider,
        )?;
        let mut chol = vec![0.0; p * p];
        if cholesky_spd_into(&covariance, p, &mut chol).is_err() {
            return Err(ChoiceModelError::CovarianceNotPositiveDefinite { decider: owned(), kind });
        }
        let estimates = self.parameter_estimates(&beta, &covariance, &learnable);
        let identity = self.identity(decider, kind);
        let draws = self.draw_measure(&beta, &chol, &learnable, &not_learnable)?;
        if draws.points.is_empty() {
            return Err(ChoiceModelError::NoDrawInsideSet {
                decider: owned(),
                drawn: self.options.n_draws,
            });
        }
        Ok(ChoiceModelEstimate {
            identity: identity.clone(),
            model_identity: self.model_identity.clone(),
            records_digest: self.records_digest.clone(),
            decider: owned(),
            consistency,
            consistency_standard_error: covariance[0].max(0.0).sqrt(),
            parameters: estimates,
            not_learnable: not_learnable
                .iter()
                .filter_map(|j| parameters.get(*j).map(|q| q.id.clone()))
                .collect(),
            log_likelihood: fit.log_likelihood,
            null_log_likelihood: fit.null_log_likelihood,
            n_records: sets.len(),
            n_deciders: design.n_deciders,
            covariance_kind: kind,
            coefficient_covariance: covariance,
            level: self.options.level,
            interval_calibration: interval_calibration(&calibration_basis(
                kind,
                learnable.len(),
                u64::try_from(sets.len()).unwrap_or(u64::MAX),
                self.options.level,
            )),
            measure: PreferenceMeasure {
                points: draws.points,
                weights: None,
                labels: None,
                origin: MeasureOrigin::ChoiceEstimate { estimate_identity: identity },
                unresolved: not_learnable,
            },
            share_outside_declared: count_as_f64(draws.dropped)
                / count_as_f64(self.options.n_draws),
            assumptions: self.assumptions.clone(),
            shape_parameters: Vec::new(),
        })
    }

    /// Whether the fixed part differs, and which free parameters are learnable. Parameters fixed
    /// by declaration are in neither list.
    fn classify(&self, sets: &[&ChoiceSetFeatures]) -> ColumnRoles {
        let k_all = self.n_parameters;
        let mut roles = ColumnRoles {
            unit: column_differs(sets, |set, i| set.fixed.get(i).copied()),
            learnable: Vec::new(),
            not_learnable: Vec::new(),
        };
        for (j, parameter) in self.model.preferences.parameters.iter().enumerate() {
            if parameter.lower.total_cmp(&parameter.upper).is_eq() {
                continue;
            }
            if column_differs(sets, |set, i| set.terms.get(i * k_all + j).copied()) {
                roles.learnable.push(j);
            } else {
                roles.not_learnable.push(j);
            }
        }
        roles
    }

    fn fit_refusal(
        &self,
        error: ConditionalLogitError,
        design: &Design,
        learnable: &[usize],
        decider: Option<&str>,
    ) -> ChoiceModelError {
        let decider = decider.map(str::to_owned);
        match error {
            ConditionalLogitError::RankDeficient { columns } => {
                let involved = involved_columns(design, &columns);
                ChoiceModelError::RankDeficient {
                    decider,
                    unit_of_account: involved.contains(&0),
                    parameters: involved
                        .iter()
                        .filter_map(|c| c.checked_sub(1))
                        .filter_map(|c| learnable.get(c))
                        .filter_map(|j| self.model.preferences.parameters.get(*j))
                        .map(|q| q.id.clone())
                        .collect(),
                }
            }
            ConditionalLogitError::Separated { .. } => ChoiceModelError::Separated { decider },
            ConditionalLogitError::NotConverged { iterations, gradient_norm } => {
                ChoiceModelError::NotConverged { decider, iterations, gradient_norm }
            }
            ConditionalLogitError::Cancelled => ChoiceModelError::Cancelled,
            error => ChoiceModelError::Fit { decider, error },
        }
    }

    /// `theta_k = beta_k / beta_0` with its delta-method standard error and normal interval.
    fn parameter_estimates(
        &self,
        beta: &[f64],
        covariance: &[f64],
        learnable: &[usize],
    ) -> Vec<ParameterEstimate> {
        let z = norm_inv(0.5 + 0.5 * self.options.level);
        learnable
            .iter()
            .enumerate()
            .filter_map(|(c, j)| {
                let declared = self.model.preferences.parameters.get(*j)?;
                let estimate = beta[c + 1] / beta[0];
                let standard_error = ratio_variance(covariance, beta, c + 1).max(0.0).sqrt();
                Some(ParameterEstimate {
                    parameter_id: declared.id.clone(),
                    estimate,
                    standard_error,
                    interval: (estimate - z * standard_error, estimate + z * standard_error),
                    declared: (declared.lower, declared.upper),
                    inside_declared: estimate >= declared.lower && estimate <= declared.upper,
                    interval_method: ParameterIntervalMethod::Wald,
                })
            })
            .collect()
    }

    fn identity(&self, decider: Option<&str>, kind: CovarianceKind) -> String {
        let options = &self.options;
        let mut out = CanonicalWriter::new("antecedent.choice_model_estimate.v1");
        out.text(&self.model_identity).text(&self.records_digest);
        match decider {
            Some(decider) => out.word(1).text(decider),
            None => out.word(0),
        };
        out.real(options.level)
            .word(options.n_draws as u64)
            .word(options.seed)
            .word(options.fit.max_iterations as u64)
            .real(options.fit.gradient_tolerance)
            .real(options.fit.separation_bound)
            .word(kind.tag());
        out.finish_hex()
    }

    /// `Err(Cancelled)` once the caller's token is set.
    fn observe(&self) -> Result<(), ChoiceModelError> {
        if self.ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
            return Err(ChoiceModelError::Cancelled);
        }
        Ok(())
    }

    /// Draws of `beta` from `N(beta_hat, L L')` mapped to parameter points; see the module
    /// documentation for which draws are kept. The caller's token is read every 1024 draws.
    fn draw_measure(
        &self,
        beta: &[f64],
        chol: &[f64],
        learnable: &[usize],
        unresolved: &[usize],
    ) -> Result<MeasureDraws, ChoiceModelError> {
        let set = &self.model.preferences;
        let p = beta.len();
        let mut base = set.midpoint();
        for (slot, parameter) in base.iter_mut().zip(&set.parameters) {
            if parameter.lower.total_cmp(&parameter.upper).is_eq() {
                *slot = parameter.lower;
            }
        }
        let checked_relations: Vec<usize> = set
            .relations
            .iter()
            .enumerate()
            .filter(|(_, relation)| relation.terms.iter().all(|(j, _)| !unresolved.contains(j)))
            .map(|(index, _)| index)
            .collect();
        let mut rng = CausalRng::from_seed(self.options.seed);
        let mut normal = vec![0.0; p];
        let mut draw = vec![0.0; p];
        let mut points = Vec::new();
        let mut dropped = 0;
        for index in 0..self.options.n_draws {
            if index % DRAWS_PER_CHECK == 0 {
                self.observe()?;
            }
            for value in &mut normal {
                *value = standard_normal(&mut rng);
            }
            for (i, slot) in draw.iter_mut().enumerate() {
                *slot = beta[i] + (0..=i).map(|j| chol[i * p + j] * normal[j]).sum::<f64>();
            }
            let lambda = draw[0];
            if lambda.is_nan() || lambda <= 0.0 {
                dropped += 1;
                continue;
            }
            let mut point = base.clone();
            for (c, j) in learnable.iter().enumerate() {
                point[*j] = draw[c + 1] / lambda;
            }
            let in_ranges = learnable.iter().all(|j| {
                let declared = &set.parameters[*j];
                let value = point[*j];
                value.is_finite()
                    && value >= declared.lower - SET_TOLERANCE
                    && value <= declared.upper + SET_TOLERANCE
            });
            let in_relations = checked_relations.iter().all(|index| {
                let relation = &set.relations[*index];
                let lhs: f64 = relation
                    .terms
                    .iter()
                    .map(|(j, coefficient)| {
                        coefficient * point.get(*j).copied().unwrap_or(f64::NAN)
                    })
                    .sum();
                match relation.sense {
                    RelationSense::AtLeast => lhs >= relation.bound - SET_TOLERANCE,
                    RelationSense::AtMost => lhs <= relation.bound + SET_TOLERANCE,
                    RelationSense::Equal => (lhs - relation.bound).abs() <= SET_TOLERANCE,
                }
            });
            if in_ranges && in_relations && self.accepted(&point, unresolved) {
                points.push(point);
            } else {
                dropped += 1;
            }
        }
        Ok(MeasureDraws { points, dropped })
    }

    /// Whether an evaluation reading the measure accepts `point` with `unresolved` left open:
    /// the same check, so the two agree by construction. A refusal of the slice's linear
    /// program drops the point. The default linear-program options are the ones this check can
    /// use: the measure is estimated before, and independently of, any evaluation that reads it,
    /// and the estimate carries no evaluation options. An evaluation under other options repeats
    /// the check with its own and refuses a point it does not accept
    /// (`measure_point_outside_set`), so a difference in options is never absorbed silently.
    fn accepted(&self, point: &[f64], unresolved: &[usize]) -> bool {
        measure_point_accepted(&self.model.preferences, point, unresolved, LpOptions::default())
            .unwrap_or(false)
    }
}

/// One parameter's `(index, estimate, standard error, interval, interval method)`.
type EstimateEntry = (usize, f64, f64, (f64, f64), ParameterIntervalMethod);

impl Context<'_> {
    /// The estimate of one fit under a value with shape parameters.
    #[allow(
        clippy::too_many_lines,
        reason = "the checks and outputs of the fit, in the documented order"
    )]
    fn estimate_shape(
        &self,
        data: &ShapeData,
        members: &[usize],
        decider: Option<&str>,
    ) -> Result<ChoiceModelEstimate, ChoiceModelError> {
        self.observe()?;
        let owned = || decider.map(str::to_owned);
        let stated =
            members.iter().filter(|&&m| self.stated.contains(&data.records[m].record_id)).count();
        if stated > 0 && stated < members.len() {
            return Err(ChoiceModelError::StatedAndRecordedMixed {
                decider: owned(),
                stated,
                recorded: members.len() - stated,
            });
        }
        let set = &self.model.preferences;
        let split = &data.split;
        let fitted: Vec<usize> = split
            .shape
            .iter()
            .copied()
            .filter(|&j| data.varies(&data.attributes_moving_with(j), members))
            .collect();
        let learnable: Vec<usize> = split
            .linear
            .iter()
            .copied()
            .filter(|&j| data.varies(&data.term_attributes(j), members))
            .collect();
        let mut base = set.midpoint();
        for (slot, value) in base.iter_mut().zip(&split.fixed) {
            if let Some(value) = value {
                *slot = *value;
            }
        }
        if fitted.is_empty() {
            // No shape parameter moves the within-record differences: the features at any shape
            // value give the same likelihood, and the ordinary fit applies.
            let features = data.features_at(&base).map_err(ChoiceModelError::Records)?;
            let sets: Vec<&ChoiceSetFeatures> =
                members.iter().map(|&m| &features.sets[m]).collect();
            return self.estimate(&sets, decider);
        }
        if !data.varies(&data.fixed_attributes(), members) {
            return Err(ChoiceModelError::NoUnitOfAccount { decider: owned() });
        }
        let unresolved: Vec<usize> = (0..set.dimension())
            .filter(|j| split.fixed[*j].is_none() && !fitted.contains(j) && !learnable.contains(j))
            .collect();
        let deciders: Vec<&str> = members
            .iter()
            .map(|&m| data.records[m].decider.as_str())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut set_starts = vec![0];
        let mut chosen = Vec::with_capacity(members.len());
        for &m in members {
            let last = set_starts.last().copied().unwrap_or(0);
            set_starts.push(last + data.records[m].options.len());
            chosen.push(data.records[m].chosen);
        }
        let clusters = (deciders.len() >= 2).then(|| {
            members
                .iter()
                .map(|&m| {
                    let position =
                        deciders.binary_search(&data.records[m].decider.as_str()).unwrap_or(0);
                    u32::try_from(position).unwrap_or(u32::MAX)
                })
                .collect()
        });
        let utility = ShapeUtility {
            data,
            members,
            learnable: learnable.clone(),
            fitted: fitted.clone(),
            base: base.clone(),
            set_starts,
            chosen,
            clusters,
            failure: RefCell::new(None),
        };
        let bounds: Vec<(f64, f64)> =
            fitted.iter().map(|j| (set.parameters[*j].lower, set.parameters[*j].upper)).collect();
        let options = NonlinearConditionalLogitOptions {
            fit: self.options.fit,
            search: self.options.shape_search,
            level: self.options.level,
        };
        let fit = fit_nonlinear_conditional_logit(&utility, &bounds, options, self.ctx)
            .map_err(|error| self.shape_refusal(error, &utility, decider))?;
        let at_estimate =
            data.features_at(&utility.point(&fit.shape)).map_err(ChoiceModelError::Records)?;
        let sets: Vec<&ChoiceSetFeatures> = members.iter().map(|&m| &at_estimate.sets[m]).collect();
        if !column_differs(&sets, |set, i| set.fixed.get(i).copied()) {
            return Err(ChoiceModelError::NoUnitOfAccount { decider: owned() });
        }
        let n_deciders = deciders.len();
        let q = 1 + learnable.len() + fitted.len();
        let (kind, covariance) = chosen_covariance(
            n_deciders,
            q,
            fit.cluster_covariance.clone(),
            fit.covariance.clone(),
            decider,
        )?;
        let mut chol = vec![0.0; q * q];
        if cholesky_spd_into(&covariance, q, &mut chol).is_err() {
            return Err(ChoiceModelError::CovarianceNotPositiveDefinite { decider: owned(), kind });
        }
        let estimates = self.shape_estimates(&fit, &covariance, &learnable, &fitted);
        let identity = self.shape_identity(decider, kind);
        let draws = self.draw_direct(&fit, &chol, &base, &learnable, &fitted, &unresolved)?;
        if draws.points.is_empty() {
            return Err(ChoiceModelError::NoDrawInsideSet {
                decider: owned(),
                drawn: self.options.n_draws,
            });
        }
        let id = |j: &usize| set.parameters.get(*j).map(|p| p.id.clone());
        Ok(ChoiceModelEstimate {
            identity: identity.clone(),
            model_identity: self.model_identity.clone(),
            records_digest: self.records_digest.clone(),
            decider: owned(),
            consistency: fit.consistency,
            consistency_standard_error: covariance[0].max(0.0).sqrt(),
            parameters: estimates,
            not_learnable: unresolved.iter().filter_map(id).collect(),
            log_likelihood: fit.log_likelihood,
            null_log_likelihood: fit.null_log_likelihood,
            n_records: members.len(),
            n_deciders,
            covariance_kind: kind,
            coefficient_covariance: covariance,
            level: self.options.level,
            interval_calibration: interval_calibration(&shape_calibration_basis(
                kind,
                learnable.len(),
                fitted.len(),
                u64::try_from(members.len()).unwrap_or(u64::MAX),
                self.options.level,
            )),
            measure: PreferenceMeasure {
                points: draws.points,
                weights: None,
                labels: None,
                origin: MeasureOrigin::ChoiceEstimate { estimate_identity: identity },
                unresolved,
            },
            share_outside_declared: count_as_f64(draws.dropped)
                / count_as_f64(self.options.n_draws),
            assumptions: self.assumptions.clone(),
            shape_parameters: fitted.iter().filter_map(id).collect(),
        })
    }

    /// The refusal of a fit with shape parameters, naming parameters by id.
    fn shape_refusal(
        &self,
        error: NonlinearChoiceError,
        utility: &ShapeUtility<'_>,
        decider: Option<&str>,
    ) -> ChoiceModelError {
        use NonlinearChoiceError as N;
        let decider = decider.map(str::to_owned);
        let parameters = &self.model.preferences.parameters;
        let id = |j: usize| parameters.get(j).map(|p| p.id.clone()).unwrap_or_default();
        let fitted = |m: usize| utility.fitted.get(m).map_or_else(String::new, |j| id(*j));
        match error {
            N::InvalidOptions | N::BoundsInvalid { .. } => ChoiceModelError::FitOptionsInvalid,
            N::ShapeSearchBudget { dimensions, limit } => {
                ChoiceModelError::ShapeSearchBudget { decider, dimensions, limit }
            }
            N::Design(error) => ChoiceModelError::Fit { decider, error },
            N::UtilityNotEvaluable { .. } => match utility.failure.borrow_mut().take() {
                Some(record) => ChoiceModelError::Records(record),
                None => ChoiceModelError::Fit {
                    decider,
                    error: ConditionalLogitError::NonFiniteFeature { row: 0, column: 0 },
                },
            },
            N::Separated { shape } => ChoiceModelError::SeparatedAtShape {
                decider,
                shape: utility.fitted.iter().zip(shape).map(|(j, value)| (id(*j), value)).collect(),
            },
            N::NotConverged { iterations, gradient_norm } => {
                ChoiceModelError::NotConverged { decider, iterations, gradient_norm }
            }
            N::ConsistencyNotPositive { consistency } => {
                ChoiceModelError::ConsistencyNotPositive { decider, consistency }
            }
            N::ShapeNotIdentified { dimension, profile_range } => {
                ChoiceModelError::ShapeNotIdentified {
                    decider,
                    parameter: fitted(dimension),
                    profile_range,
                }
            }
            N::ProfileMultimodal { dimension, modes } => {
                ChoiceModelError::ProfileMultimodal { decider, parameter: fitted(dimension), modes }
            }
            N::ShapeEstimateAtBound { dimension, estimate, interval } => {
                ChoiceModelError::ShapeEstimateAtBound {
                    decider,
                    parameter: fitted(dimension),
                    estimate,
                    interval,
                }
            }
            N::RankDeficient { columns } => {
                let n_linear = utility.learnable.len();
                ChoiceModelError::RankDeficient {
                    decider,
                    unit_of_account: columns.contains(&0),
                    parameters: columns
                        .iter()
                        .filter_map(|c| c.checked_sub(1))
                        .filter_map(|c| {
                            if c < n_linear {
                                utility.learnable.get(c).copied()
                            } else {
                                utility.fitted.get(c - n_linear).copied()
                            }
                        })
                        .map(id)
                        .collect(),
                }
            }
            N::Cancelled => ChoiceModelError::Cancelled,
        }
    }

    /// Estimates of the learnable linear parameters (Wald) and the fitted shape parameters
    /// (profile likelihood ratio), in model order.
    fn shape_estimates(
        &self,
        fit: &NonlinearConditionalLogitFit,
        covariance: &[f64],
        learnable: &[usize],
        fitted: &[usize],
    ) -> Vec<ParameterEstimate> {
        let z = norm_inv(0.5 + 0.5 * self.options.level);
        let q = 1 + learnable.len() + fitted.len();
        let standard_error = |index: usize| covariance[index * q + index].max(0.0).sqrt();
        let mut entries: Vec<EstimateEntry> = Vec::new();
        for (c, j) in learnable.iter().enumerate() {
            let estimate = fit.linear[c];
            let se = standard_error(1 + c);
            entries.push((
                *j,
                estimate,
                se,
                (estimate - z * se, estimate + z * se),
                ParameterIntervalMethod::Wald,
            ));
        }
        for (m, j) in fitted.iter().enumerate() {
            let estimate = fit.shape[m];
            let interval = fit.profiles.get(m).map_or((estimate, estimate), |p| p.interval);
            entries.push((
                *j,
                estimate,
                standard_error(1 + learnable.len() + m),
                interval,
                ParameterIntervalMethod::ProfileLikelihood,
            ));
        }
        entries.sort_by_key(|entry| entry.0);
        entries
            .into_iter()
            .filter_map(|(j, estimate, standard_error, interval, interval_method)| {
                let declared = self.model.preferences.parameters.get(j)?;
                Some(ParameterEstimate {
                    parameter_id: declared.id.clone(),
                    estimate,
                    standard_error,
                    interval,
                    declared: (declared.lower, declared.upper),
                    inside_declared: estimate >= declared.lower && estimate <= declared.upper,
                    interval_method,
                })
            })
            .collect()
    }

    /// The fields of [`Self::identity`] followed by the shape search controls, under the shape
    /// fit's own domain tag, so a shape fit and a linear fit never share an identity.
    fn shape_identity(&self, decider: Option<&str>, kind: CovarianceKind) -> String {
        let options = &self.options;
        let search = options.shape_search;
        let mut out = CanonicalWriter::new("antecedent.choice_model_shape_estimate.v1");
        out.text(&self.model_identity).text(&self.records_digest);
        match decider {
            Some(decider) => out.word(1).text(decider),
            None => out.word(0),
        };
        out.real(options.level)
            .word(options.n_draws as u64)
            .word(options.seed)
            .word(options.fit.max_iterations as u64)
            .real(options.fit.gradient_tolerance)
            .real(options.fit.separation_bound)
            .word(kind.tag())
            .word(search.grid_points as u64)
            .word(search.max_dimensions as u64)
            .word(search.profile_points as u64)
            .word(search.max_refinements as u64);
        out.finish_hex()
    }

    /// Draws of the direct parameters `(lambda, theta_learnable, theta_shape)` from the normal
    /// approximation mapped to parameter points and kept inside the declared set, as
    /// [`Self::draw_measure`] keeps them; value relations are checked when no parameter is
    /// unresolved.
    fn draw_direct(
        &self,
        fit: &NonlinearConditionalLogitFit,
        chol: &[f64],
        base: &[f64],
        learnable: &[usize],
        fitted: &[usize],
        unresolved: &[usize],
    ) -> Result<MeasureDraws, ChoiceModelError> {
        let set = &self.model.preferences;
        let mut centre = vec![fit.consistency];
        centre.extend(fit.linear.iter().copied());
        centre.extend(fit.shape.iter().copied());
        let q = centre.len();
        let resolved: Vec<usize> = learnable.iter().chain(fitted).copied().collect();
        let checked_relations: Vec<usize> = set
            .relations
            .iter()
            .enumerate()
            .filter(|(_, relation)| relation.terms.iter().all(|(j, _)| !unresolved.contains(j)))
            .map(|(index, _)| index)
            .collect();
        let mut rng = CausalRng::from_seed(self.options.seed);
        let mut normal = vec![0.0; q];
        let mut draw = vec![0.0; q];
        let mut points = Vec::new();
        let mut dropped = 0;
        for index in 0..self.options.n_draws {
            if index % DRAWS_PER_CHECK == 0 {
                self.observe()?;
            }
            for value in &mut normal {
                *value = standard_normal(&mut rng);
            }
            for (i, slot) in draw.iter_mut().enumerate() {
                *slot = centre[i] + (0..=i).map(|j| chol[i * q + j] * normal[j]).sum::<f64>();
            }
            let lambda = draw[0];
            if lambda.is_nan() || lambda <= 0.0 {
                dropped += 1;
                continue;
            }
            let mut point = base.to_vec();
            for (c, j) in resolved.iter().enumerate() {
                point[*j] = draw[c + 1];
            }
            let in_ranges = resolved.iter().all(|j| {
                let declared = &set.parameters[*j];
                let value = point[*j];
                value.is_finite()
                    && value >= declared.lower - SET_TOLERANCE
                    && value <= declared.upper + SET_TOLERANCE
            });
            let in_relations = checked_relations.iter().all(|index| {
                let relation = &set.relations[*index];
                let lhs: f64 = relation
                    .terms
                    .iter()
                    .map(|(j, coefficient)| {
                        coefficient * point.get(*j).copied().unwrap_or(f64::NAN)
                    })
                    .sum();
                match relation.sense {
                    RelationSense::AtLeast => lhs >= relation.bound - SET_TOLERANCE,
                    RelationSense::AtMost => lhs <= relation.bound + SET_TOLERANCE,
                    RelationSense::Equal => (lhs - relation.bound).abs() <= SET_TOLERANCE,
                }
            });
            let in_values = !unresolved.is_empty()
                || set.value_relations.iter().all(|relation| {
                    self.model
                        .value_relation_gap(relation, &point)
                        .is_ok_and(|gap| gap >= -relation.tolerance - SET_TOLERANCE)
                });
            if in_ranges && in_relations && in_values && self.accepted(&point, unresolved) {
                points.push(point);
            } else {
                dropped += 1;
            }
        }
        Ok(MeasureDraws { points, dropped })
    }
}

/// The covariance a fit with `deciders` decision-makers and `q` coefficients uses: the model
/// covariance for one decision-maker; the clustered covariance for several, when there are at
/// least [`clusters_required`] of them; otherwise a refusal. The fits are given clusters exactly
/// when there are at least two decision-makers, so `clustered` is present then.
fn chosen_covariance(
    deciders: usize,
    q: usize,
    clustered: Option<Vec<f64>>,
    model: Vec<f64>,
    decider: Option<&str>,
) -> Result<(CovarianceKind, Vec<f64>), ChoiceModelError> {
    if deciders < 2 {
        return Ok((CovarianceKind::Model, model));
    }
    let required = clusters_required(q);
    match clustered {
        Some(clustered) if deciders >= required => {
            Ok((CovarianceKind::ClusteredByDecider, clustered))
        }
        _ => Err(ChoiceModelError::TooFewDecidersToCluster {
            decider: decider.map(str::to_owned),
            deciders,
            coefficients: q,
            required,
        }),
    }
}

/// Measure draws made between two reads of the cancellation token.
const DRAWS_PER_CHECK: usize = 1024;

struct MeasureDraws {
    points: Vec<Vec<f64>>,
    dropped: usize,
}

#[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^53")]
const fn count_as_f64(n: usize) -> f64 {
    n as f64
}

/// Whether a design column differs between the options of at least one record, relative to the
/// column's largest absolute value over every option in the fit. `value(set, i)` reads option
/// `i` of a set; a missing value counts as zero.
fn column_differs(
    sets: &[&ChoiceSetFeatures],
    value: impl Fn(&ChoiceSetFeatures, usize) -> Option<f64>,
) -> bool {
    let read = |set: &ChoiceSetFeatures, i: usize| value(set, i).unwrap_or(0.0);
    let scale = sets
        .iter()
        .flat_map(|set| (0..set.option_ids.len()).map(|i| read(set, i).abs()))
        .fold(0.0_f64, f64::max);
    if !(scale.is_finite() && scale > 0.0) {
        return false;
    }
    sets.iter().any(|set| {
        let (low, high) = (0..set.option_ids.len())
            .map(|i| read(set, i))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), v| (low.min(v), high.max(v)));
        high - low > DIFFERENCE_RELATIVE_TOLERANCE * scale
    })
}

/// Column 0 is the fixed part; column `c >= 1` is the term of `learnable[c - 1]`. Clusters are the
/// positions of the deciders in sorted order, given only when there are at least two.
fn build_design(sets: &[&ChoiceSetFeatures], learnable: &[usize], k_all: usize) -> Design {
    let p = 1 + learnable.len();
    let mut features = Vec::new();
    let mut set_starts = vec![0];
    let mut chosen = Vec::with_capacity(sets.len());
    for set in sets {
        let n_options = set.option_ids.len();
        for i in 0..n_options {
            features.push(set.fixed.get(i).copied().unwrap_or(0.0));
            for j in learnable {
                features.push(set.terms.get(i * k_all + j).copied().unwrap_or(0.0));
            }
        }
        set_starts.push(set_starts.last().copied().unwrap_or(0) + n_options);
        chosen.push(set.chosen);
    }
    let deciders: Vec<&str> =
        sets.iter().map(|set| set.decider.as_str()).collect::<BTreeSet<_>>().into_iter().collect();
    let clusters = (deciders.len() >= 2).then(|| {
        sets.iter()
            .map(|set| {
                let position = deciders.binary_search(&set.decider.as_str()).unwrap_or(0);
                // More than `u32::MAX` decision-makers cannot be held in memory as records.
                u32::try_from(position).unwrap_or(u32::MAX)
            })
            .collect()
    });
    Design { p, features, set_starts, chosen, clusters, n_deciders: deciders.len() }
}

/// Columns taking part in the dependencies the fit reported: each dependent column, and every
/// earlier independent column with a non-negligible coefficient when the dependent column's
/// within-record differences are projected on theirs by least squares.
fn involved_columns(design: &Design, dependent: &[usize]) -> BTreeSet<usize> {
    let p = design.p;
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); p];
    for (s, &c) in design.chosen.iter().enumerate() {
        let (start, end) = (design.set_starts[s], design.set_starts[s + 1]);
        let chosen_row = start + c;
        for r in start..end {
            if r == chosen_row {
                continue;
            }
            for (k, column) in columns.iter_mut().enumerate() {
                column.push(design.features[r * p + k] - design.features[chosen_row * p + k]);
            }
        }
    }
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut involved = BTreeSet::new();
    for &k in dependent {
        involved.insert(k);
        let Some(target) = columns.get(k) else { continue };
        let norm = dot(target, target).sqrt();
        if norm.is_nan() || norm <= 0.0 {
            continue;
        }
        let earlier: Vec<usize> = (0..k).filter(|e| !dependent.contains(e)).collect();
        let n = earlier.len();
        if n == 0 {
            continue;
        }
        let mut gram = vec![0.0; n * n];
        let mut rhs = vec![0.0; n];
        for (a, &ea) in earlier.iter().enumerate() {
            rhs[a] = dot(&columns[ea], target);
            for (b, &eb) in earlier.iter().enumerate() {
                gram[a * n + b] = dot(&columns[ea], &columns[eb]);
            }
        }
        if !gauss_jordan_solve(&mut gram, &mut rhs, n, 1) {
            involved.extend(earlier);
            continue;
        }
        for (a, &e) in earlier.iter().enumerate() {
            let share = rhs[a].abs() * dot(&columns[e], &columns[e]).sqrt();
            if share > INVOLVED_RELATIVE * norm {
                involved.insert(e);
            }
        }
    }
    involved
}

/// Delta-method variance of `beta[k] / beta[0]` from the row-major covariance of `beta`:
/// `(V_kk - 2 theta V_k0 + theta^2 V_00) / beta_0^2`.
fn ratio_variance(covariance: &[f64], beta: &[f64], k: usize) -> f64 {
    let p = beta.len();
    let theta = beta[k] / beta[0];
    (covariance[k * p + k] - 2.0 * theta * covariance[k * p] + theta * theta * covariance[0])
        / (beta[0] * beta[0])
}

impl ChoiceModelError {
    /// Structured refusal for a failure to estimate preference parameters from recorded
    /// choices.
    ///
    /// A value-model or choice-record refusal keeps its own refusal; a conditional-logit
    /// refusal is reported under `preference_choice_model.fit_refused` with the fit's detail
    /// and message as the supplied text.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one arm per estimation failure")]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), ESTIMATE, self.detail())
                .remedy(remedy)
        };
        let design = |decider: Option<String>, remedy| {
            refusal(antecedent_core::reason_code!("design_rank_deficient"), ESTIMATE, self.detail())
                .offending_opt(decider)
                .remedy(remedy)
        };
        let fit = |decider: Option<String>, remedy| {
            refusal(antecedent_core::reason_code!("glm_fit_refused"), ESTIMATE, self.detail())
                .offending_opt(decider)
                .remedy(remedy)
        };
        match self {
            Self::LevelInvalid { level } => invalid("supply a level strictly between 0 and 1")
                .expected("(0, 1)")
                .supplied(level.to_string())
                .build(),
            Self::DrawsMissing => invalid("request at least one measure draw").build(),
            Self::FitOptionsInvalid => {
                invalid("supply a finite, positive gradient tolerance and separation bound").build()
            }
            Self::Model(error) => at_stage(error.to_refusal(), ESTIMATE),
            Self::Records(error) => at_stage(error.to_refusal(), ESTIMATE),
            Self::StatedAndRecordedMixed { decider, stated, recorded } => refusal(
                antecedent_core::reason_code!("route_not_supported"),
                ESTIMATE,
                self.detail(),
            )
            .offending_opt(decider.clone())
            .expected("stated comparisons only, or recorded choices only")
            .supplied(format!("{stated} stated, {recorded} recorded"))
            .remedy(
                "fit stated comparisons and recorded choices separately, or use the \
                 compatible-set method, which does not depend on a noise scale",
            )
            .build(),
            Self::NoUnitOfAccount { decider } => design(
                decider.clone(),
                "record choices whose options differ in the fixed part of the value",
            )
            .build(),
            Self::NothingIdentified { decider } => design(
                decider.clone(),
                "record choices whose options differ in the terms of the value",
            )
            .build(),
            Self::RankDeficient { decider, unit_of_account, parameters } => {
                let mut columns = parameters.clone();
                if *unit_of_account {
                    columns.insert(0, "unit_of_account".to_owned());
                }
                design(
                    decider.clone(),
                    "record choices that vary the named parameters apart, or fix one of them",
                )
                .supplied(columns.join(","))
                .build()
            }
            Self::Separated { decider } => fit(decider.clone(), COMPATIBLE_SET).build(),
            Self::ConsistencyNotPositive { decider, consistency } => fit(
                decider.clone(),
                "check the sign of the fixed part of the value, or use the compatible-set method",
            )
            .supplied(consistency.to_string())
            .build(),
            Self::NotConverged { decider, iterations, gradient_norm } => fit(
                decider.clone(),
                "raise the fit's iteration limit, or use the compatible-set method",
            )
            .supplied(format!("{iterations} iterations, gradient norm {gradient_norm}"))
            .build(),
            Self::Fit { decider, error } => fit(decider.clone(), COMPATIBLE_SET)
                .supplied(format!("{}: {error}", error.detail()))
                .build(),
            Self::TooFewDecidersToCluster { decider, deciders, coefficients, required } => refusal(
                antecedent_core::reason_code!("uncertainty_representation_insufficient"),
                ESTIMATE,
                self.detail(),
            )
            .offending_opt(decider.clone())
            .expected(format!(
                "at least {required} decision-makers for {coefficients} coefficients"
            ))
            .supplied(deciders.to_string())
            .remedy(
                "record choices from more decision-makers, fit each decision-maker \
                     separately, or use the compatible-set method",
            )
            .build(),
            Self::CovarianceNotPositiveDefinite { decider, kind } => {
                fit(decider.clone(), "record more choices, or use the compatible-set method")
                    .supplied(format!("{kind:?}"))
                    .build()
            }
            Self::NoDrawInsideSet { decider, drawn } => refusal(
                antecedent_core::reason_code!("uncertainty_representation_insufficient"),
                ESTIMATE,
                self.detail(),
            )
            .offending_opt(decider.clone())
            .supplied(drawn.to_string())
            .remedy("draw more, record more choices, or use the compatible-set method")
            .build(),
            Self::Cancelled => refusal(
                antecedent_core::reason_code!("cancelled_no_claim"),
                ESTIMATE,
                self.detail(),
            )
            .build(),
            Self::SeparatedAtShape { decider, shape } => fit(decider.clone(), COMPATIBLE_SET)
                .supplied(
                    shape
                        .iter()
                        .map(|(id, value)| format!("{id}={value}"))
                        .collect::<Vec<_>>()
                        .join(","),
                )
                .build(),
            Self::ShapeNotIdentified { decider, parameter, profile_range } => design(
                decider.clone(),
                "fix the shape parameter to a single value, record choices that tell its values \
                 apart, or use the compatible-set method",
            )
            .supplied(format!("{parameter}: profile range {profile_range}"))
            .build(),
            Self::ShapeEstimateAtBound { decider, parameter, estimate, interval } => refusal(
                antecedent_core::reason_code!("uncertainty_representation_insufficient"),
                ESTIMATE,
                self.detail(),
            )
            .offending_opt(decider.clone())
            .supplied(format!(
                "{parameter} = {estimate}; profile interval [{}, {}]",
                interval.0, interval.1
            ))
            .remedy(
                "widen the declared range if the bound is not a limit of the preference, or use \
                 the compatible-set method",
            )
            .build(),
            Self::ProfileMultimodal { decider, parameter, modes } => fit(
                decider.clone(),
                "narrow the declared range of the shape parameter to one mode, or use the \
                 compatible-set method",
            )
            .supplied(format!("{parameter}: maxima at {} and {}", modes.0, modes.1))
            .build(),
            Self::ShapeSearchBudget { decider, dimensions, limit } => {
                invalid("fix some shape parameters to single values, or raise the search limit")
                    .offending_opt(decider.clone())
                    .expected(format!("at most {limit}"))
                    .supplied(dimensions.to_string())
                    .build()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_variance_matches_a_finite_difference_gradient() {
        let beta = [1.7, -0.6, 2.3];
        let covariance = [0.30, 0.05, -0.02, 0.05, 0.20, 0.04, -0.02, 0.04, 0.50];
        let h = 1e-6;
        for k in 1..3 {
            let f = |b: &[f64]| b[k] / b[0];
            let gradient: Vec<f64> = (0..3)
                .map(|i| {
                    let mut up = beta;
                    let mut down = beta;
                    up[i] += h;
                    down[i] -= h;
                    (f(&up) - f(&down)) / (2.0 * h)
                })
                .collect();
            let quadratic: f64 = (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .map(|(i, j)| gradient[i] * covariance[i * 3 + j] * gradient[j])
                .sum();
            let analytic = ratio_variance(&covariance, &beta, k);
            assert!((quadratic - analytic).abs() <= 1e-8 * analytic.abs().max(1.0));
        }
    }

    #[test]
    fn dependency_names_every_column_it_involves() {
        // Column 2 equals twice column 1; column 0 is independent of both.
        let design = Design {
            p: 3,
            features: vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, -1.0, -2.0],
            set_starts: vec![0, 2, 4],
            chosen: vec![0, 0],
            clusters: None,
            n_deciders: 1,
        };
        let involved = involved_columns(&design, &[2]);
        assert_eq!(involved.into_iter().collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn a_column_differs_only_beyond_the_relative_tolerance() {
        let set = |fixed: Vec<f64>| ChoiceSetFeatures {
            record_id: "r".to_owned(),
            decider: "d".to_owned(),
            option_ids: vec!["a".to_owned(), "b".to_owned()],
            chosen: 0,
            fixed,
            terms: Vec::new(),
            unavailable: Vec::new(),
        };
        let flat = set(vec![1.0e6, 1.0e6 + 1e-6]);
        let varied = set(vec![1.0, 2.0]);
        assert!(!column_differs(&[&flat], |s, i| s.fixed.get(i).copied()));
        assert!(column_differs(&[&flat, &varied], |s, i| s.fixed.get(i).copied()));
    }
}
