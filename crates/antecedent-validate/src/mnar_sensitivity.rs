//! Missing-not-at-random (MNAR) sensitivity of a two-arm mean contrast.
//!
//! The estimand is the average treatment effect on the full assigned population,
//! `mu_1 - mu_0`, where `mu_a` is the mean outcome of arm `a` over every assigned unit,
//! observed or not. The missing-at-random (MAR) estimate takes the missing outcomes of an
//! arm to follow the observed outcomes of that arm. This module relaxes that single
//! premise by an explicit, one-parameter departure and re-runs the estimand across a
//! declared bounded grid of the parameter. Nothing about the departure can be tested from
//! the data: it is recorded as an assumption with status `untestable`.
//!
//! # Departure models
//!
//! * **Delta-adjustment pattern mixture.** Pattern-mixture factorization
//!   `f(y) = P(R = 1) f(y | R = 1) + P(R = 0) f(y | R = 0)` with the missing-pattern law
//!   the observed-pattern law shifted by `l_a * delta`: `E[Y | R = 0, arm a] =
//!   E[Y | R = 1, arm a] + l_a delta`. Then
//!   `mu_a(delta) = ybar_a + l_a delta f_a`, with `f_a` the missing fraction of the arm.
//!   `delta = 0` is MAR and the effect is linear in `delta` with slope
//!   `l_1 f_1 - l_0 f_0`, so the shift of the effect by `delta` is exactly `delta` times the
//!   missing fraction (times the loading).
//! * **Bounded selection model, exponential tilting.** The odds of observing an outcome
//!   follow `logit P(R = 1 | y, arm a) = alpha_a - t_a y`, which tilts the missing-pattern
//!   law of the arm away from the observed-pattern law:
//!   `f(y | R = 0, a) is proportional to f(y | R = 1, a) exp(t_a y)`, `t_a = l_a theta`.
//!   `theta = 0` is MAR. On the empirical observed outcomes the missing-pattern mean is
//!   `g_a(theta) = sum_i w_i y_i / sum_i w_i`, `w_i = exp(t_a y_i)` (evaluated stabilized
//!   by the arm's extreme outcome), so the tilt is bounded by the observed outcome range:
//!   for every finite `theta` the departure stays inside it. `mu_a = pi_a ybar_a +
//!   (1 - pi_a) g_a(theta)` with `pi_a` the observed fraction of the arm.
//!
//! # Sampling uncertainty
//!
//! At each grid point the estimate is a smooth functional of the observed outcomes and
//! the observed fraction. The delta-method (influence function) variance conditional on
//! the parameter is
//! `(ybar - g)^2 pi (1 - pi) / n + Var_obs(pi y_i + (1 - pi) psi_i) / n_obs`, with `psi_i`
//! the influence of the missing-pattern mean (`y_i - ybar` for the shift,
//! `w_i (y_i - g) / mean(w)` for the tilt); the arms are independent. It is a sampling
//! interval conditional on the parameter value and is reported per grid point, never merged
//! with the assumption range. The assumption range is the envelope of the point effects over
//! the declared parameter range and is never a confidence interval.
//!
//! # Conclusions and tipping points
//!
//! A conclusion is the sign of the effect, whether the conditional sampling interval
//! excludes the null, or whether a declared decision (act when the effect exceeds a
//! threshold) acts. Its tipping point is the smallest departure (searched separately on
//! each side of MAR) at which the conclusion differs from its MAR value: the declared
//! grid locates the first changed grid point and bisection on the conclusion refines the
//! parameter inside the cell, reported with its certified bracket.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

use antecedent_core::{reason_code, AssumptionStatus};
use antecedent_stats::special::normal_ppf;

use crate::mnar_adjusted::{self, AdjData};

/// Most rows of one analysis.
pub const MNAR_MAX_ROWS: usize = 1 << 26;
/// Most grid points.
pub const MNAR_MAX_GRID_POINTS: usize = 4096;
/// Most covariate columns of an adjusted estimator.
pub const MNAR_MAX_COVARIATES: usize = 64;
/// Fewest bootstrap replicates of an adjusted estimator's sampling interval.
pub const MNAR_MIN_BOOTSTRAP: u32 = 20;
/// Most bootstrap replicates of an adjusted estimator's sampling interval.
pub const MNAR_MAX_BOOTSTRAP: u32 = 5000;
/// Sub-steps inside each grid cell scanned for a conclusion change.
const MNAR_CELL_SUBSTEPS: usize = 8;
/// Most declared decisions.
pub const MNAR_MAX_DECISIONS: usize = 64;
/// Largest admitted magnitude of the shift parameter (`delta`), in outcome units.
pub const MNAR_MAX_DELTA: f64 = 1.0e12;
/// Largest admitted magnitude of the tilt parameter (`theta`), per outcome unit.
pub const MNAR_MAX_TILT: f64 = 1.0e3;
/// Smallest admitted relative bracket tolerance.
pub const MNAR_MIN_TOLERANCE: f64 = 1e-14;
/// Largest admitted relative bracket tolerance.
pub const MNAR_MAX_TOLERANCE: f64 = 1e-3;
/// Most `rows x evaluations` of one analysis: the compute budget.
pub const MNAR_MAX_WORK: u64 = 1 << 34;
/// Safety cap on bisection iterations.
const MNAR_MAX_BISECTIONS: usize = 200;
/// Family identity.
pub const MNAR_FAMILY: &str = "missing_outcome_departure_from_mar";
/// Target statement.
pub const MNAR_TARGET: &str =
    "average_treatment_effect_difference_in_arm_means_over_all_assigned_units";
/// Interpretation of the reported range.
pub const MNAR_INTERPRETATION: &str = "assumption range over the declared departure from missing at random; not a confidence interval and not a sampling interval";
/// The only inference claim the range carries.
pub const MNAR_ASSUMPTION_RANGE_CLAIM: &str = "assumption_range";
/// Empirical status of the departure: the data cannot establish it.
pub const MNAR_EMPIRICAL_STATUS: &str = "not_empirically_established";
/// What the per-point sampling interval is conditional on.
pub const MNAR_SAMPLING_BASIS: &str = "sampling interval conditional on the sensitivity parameter (delta method for the difference in means, seeded bootstrap that repeats the imputation fit for adjusted estimators); kept distinct from the assumption range and never combined with it";
/// Stable id of the untestable departure assumption.
pub const MNAR_DEPARTURE_ASSUMPTION_ID: &str = "outcome_missingness_departure_from_mar";
/// Status of the departure assumption.
pub const MNAR_DEPARTURE_ASSUMPTION_STATUS: AssumptionStatus = AssumptionStatus::Untestable;
/// Stable id of the treatment-ignorability assumption the analysis holds fixed.
pub const MNAR_ASSIGNMENT_ASSUMPTION_ID: &str = "treatment_independent_of_potential_outcomes";
/// Method recorded in every result.
pub const MNAR_METHOD: &str = "re-run of the declared estimator (difference in means in closed form; regression adjustment, inverse-propensity weighting or augmented weighting on a table completed by the departure) over a declared parameter grid (pattern-mixture shift; exponential-tilting selection model), with sampling intervals per grid point and tipping points found by sub-cell scanning and bisection on the continuous effect";

/// The departure model of the outcome-missingness mechanism.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MnarModel {
    /// Pattern mixture: missing outcomes are the observed outcomes shifted by `delta`.
    DeltaPatternMixture,
    /// Selection model: the missing-pattern law is the observed law tilted by `exp(theta y)`.
    ExponentialTiltingSelection,
}

impl MnarModel {
    /// Both models, in documentation order.
    pub const ALL: [Self; 2] = [Self::DeltaPatternMixture, Self::ExponentialTiltingSelection];

    /// Stable wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DeltaPatternMixture => "delta_pattern_mixture",
            Self::ExponentialTiltingSelection => "exponential_tilting_selection",
        }
    }

    /// The model named `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|model| model.name() == name)
    }

    /// Largest admitted parameter magnitude.
    #[must_use]
    pub const fn max_magnitude(self) -> f64 {
        match self {
            Self::DeltaPatternMixture => MNAR_MAX_DELTA,
            Self::ExponentialTiltingSelection => MNAR_MAX_TILT,
        }
    }
}

/// The estimators the departure analysis re-runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MnarEstimator {
    /// Difference of the arms' full-population mean outcomes.
    DifferenceInMeans,
    /// Arm-specific outcome regressions on the covariates, averaged over every unit.
    RegressionAdjusted,
    /// Hajek inverse-propensity-weighted contrast with a logistic propensity model.
    InverseProbabilityWeighting,
    /// Augmented inverse-propensity-weighted contrast.
    AugmentedInverseProbabilityWeighting,
}

impl MnarEstimator {
    /// Every supported estimator.
    pub const ALL: [Self; 4] = [
        Self::DifferenceInMeans,
        Self::RegressionAdjusted,
        Self::InverseProbabilityWeighting,
        Self::AugmentedInverseProbabilityWeighting,
    ];

    /// Whether the estimator reads covariates.
    #[must_use]
    pub const fn uses_covariates(self) -> bool {
        !matches!(self, Self::DifferenceInMeans)
    }

    /// Stable wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DifferenceInMeans => "difference_in_means",
            Self::RegressionAdjusted => "regression_adjusted",
            Self::InverseProbabilityWeighting => "ipw",
            Self::AugmentedInverseProbabilityWeighting => "aipw",
        }
    }

    /// The estimator named `name`, or the typed refusal for an unsupported one.
    ///
    /// # Errors
    ///
    /// `mnar_sensitivity.unsupported_estimator`.
    pub fn from_name(name: &str) -> Result<Self, MnarSensitivityError> {
        Self::ALL.into_iter().find(|estimator| estimator.name() == name).ok_or_else(|| {
            refuse(
                reason_code!("route_not_supported"),
                "mnar_sensitivity.unsupported_estimator",
                format!(
                    "the supported estimators are {:?}; `{name}` has no completed-table form to re-run across the grid (it needs a model of the outcome law the departure cannot shift)",
                    Self::ALL.map(Self::name)
                ),
            )
        })
    }
}

/// How the parameter loads onto each arm: the arm's departure is `loading * parameter`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmLoading {
    /// Loading of the treated arm in `[-1, 1]`.
    pub treated: f64,
    /// Loading of the control arm in `[-1, 1]`.
    pub control: f64,
}

impl ArmLoading {
    /// Both arms depart by the parameter.
    #[must_use]
    pub const fn common() -> Self {
        Self { treated: 1.0, control: 1.0 }
    }

    /// Only the treated arm departs.
    #[must_use]
    pub const fn treated_only() -> Self {
        Self { treated: 1.0, control: 0.0 }
    }

    /// Only the control arm departs.
    #[must_use]
    pub const fn control_only() -> Self {
        Self { treated: 0.0, control: 1.0 }
    }

    /// The arms depart in opposite directions.
    #[must_use]
    pub const fn opposite() -> Self {
        Self { treated: 1.0, control: -1.0 }
    }
}

/// The declared outcome-missingness mechanism.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MnarMechanism {
    /// The departure model.
    pub model: MnarModel,
    /// The per-arm loading of the parameter.
    pub loading: ArmLoading,
}

/// A declared decision: act (prefer the treatment) when the effect exceeds a threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct MnarDecision {
    /// Decision name.
    pub name: String,
    /// The effect above which the decision acts.
    pub threshold: f64,
}

/// Analysis settings.
#[derive(Clone, Debug, PartialEq)]
pub struct MnarSensitivitySpec {
    /// The declared mechanism; `None` is refused.
    pub mechanism: Option<MnarMechanism>,
    /// The estimator re-run at every grid point.
    pub estimator: MnarEstimator,
    /// Lower end of the parameter range (finite; at most zero).
    pub lower: f64,
    /// Upper end of the parameter range (finite; at least zero).
    pub upper: f64,
    /// Equally spaced grid points from `lower` to `upper` (2 to [`MNAR_MAX_GRID_POINTS`]).
    pub grid_points: usize,
    /// Level of the per-point sampling interval, in `(0, 1)`.
    pub level: f64,
    /// Declared decisions.
    pub decisions: Vec<MnarDecision>,
    /// Bracket tolerance of the tipping search, relative to the range width.
    pub tolerance: f64,
    /// A requested composition of the sampling interval with the range; always refused.
    pub sampling_composition: Option<String>,
    /// Bootstrap replicates of an adjusted estimator's sampling interval.
    pub bootstrap_replicates: u32,
    /// Seed of the bootstrap resampling stream.
    pub seed: u64,
}

impl MnarSensitivitySpec {
    /// A spec over `[lower, upper]` with a 21-point grid, level 0.95, no decisions.
    #[must_use]
    pub fn new(mechanism: MnarMechanism, lower: f64, upper: f64) -> Self {
        Self {
            mechanism: Some(mechanism),
            estimator: MnarEstimator::DifferenceInMeans,
            lower,
            upper,
            grid_points: 21,
            level: 0.95,
            decisions: Vec::new(),
            tolerance: 1e-12,
            sampling_composition: None,
            bootstrap_replicates: 200,
            seed: 0,
        }
    }
}

/// Typed refusal of the MNAR sensitivity contract.
#[derive(Clone, Debug, PartialEq, Error)]
#[error("{detail}: {message}")]
pub struct MnarSensitivityError {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced `mnar_sensitivity.*` detail.
    pub detail: &'static str,
    /// Human-readable context.
    pub message: String,
}

impl MnarSensitivityError {
    /// Registered reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }
}

fn refuse(
    code: &'static str,
    detail: &'static str,
    message: impl Into<String>,
) -> MnarSensitivityError {
    MnarSensitivityError { code, detail, message: message.into() }
}

/// The data: a binary treatment and an outcome with missing entries.
#[derive(Clone, Copy, Debug)]
pub struct MnarData<'a> {
    /// Treatment per unit, `0.0` or `1.0`, never missing.
    pub treatment: &'a [f64],
    /// Outcome per unit; `None` is a missing outcome.
    pub outcome: &'a [Option<f64>],
    /// Covariate columns (finite, never missing) read by the adjusted estimators; empty for
    /// the difference in means.
    pub covariates: &'a [&'a [f64]],
}

impl<'a> MnarData<'a> {
    /// Data without covariates.
    #[must_use]
    pub const fn new(treatment: &'a [f64], outcome: &'a [Option<f64>]) -> Self {
        Self { treatment, outcome, covariates: &[] }
    }

    /// The same data with covariate columns.
    #[must_use]
    pub const fn with_covariates(mut self, covariates: &'a [&'a [f64]]) -> Self {
        self.covariates = covariates;
        self
    }
}

/// The sampling interval of the effect at one grid point, conditional on the parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MnarSamplingInterval {
    /// Delta-method standard error of the effect at this parameter.
    pub standard_error: f64,
    /// Lower end at [`MnarSensitivitySpec::level`].
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
}

/// The effect at one value of the sensitivity parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct MnarPoint {
    /// The sensitivity parameter.
    pub parameter: f64,
    /// Treated-arm full-population mean.
    pub treated_mean: f64,
    /// Control-arm full-population mean.
    pub control_mean: f64,
    /// The effect, `treated_mean - control_mean`.
    pub effect: f64,
    /// Sampling interval conditional on the parameter.
    pub sampling: MnarSamplingInterval,
    /// Whether each declared decision acts, in declaration order.
    pub acts: Vec<bool>,
}

/// Which conclusion a tipping point concerns.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MnarConclusionKind {
    /// The sign of the effect.
    Sign,
    /// Whether the conditional sampling interval excludes the null.
    ExcludesNull,
    /// A declared decision.
    Decision,
}

impl MnarConclusionKind {
    /// Stable wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sign => "sign",
            Self::ExcludesNull => "excludes_null",
            Self::Decision => "decision",
        }
    }
}

/// The first departure on one side of MAR at which a conclusion changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MnarTippingSide {
    /// The smallest evaluated parameter at which the conclusion differs from MAR's.
    pub parameter: f64,
    /// The largest evaluated parameter before it at which the conclusion is still MAR's
    /// (zero when the first grid point already differs); with `parameter` it brackets the
    /// tipping point.
    pub bracket_lower: f64,
    /// Bisection iterations spent.
    pub iterations: usize,
    /// The conclusion label past the tipping point.
    pub changes_to: i8,
}

/// The tipping points of one conclusion.
#[derive(Clone, Debug, PartialEq)]
pub struct MnarTipping {
    /// The kind of conclusion.
    pub kind: MnarConclusionKind,
    /// The conclusion's name (`sign`, `excludes_null` or the decision's name).
    pub label: String,
    /// The threshold: zero for sign and null, the decision's threshold for a decision.
    pub threshold: f64,
    /// The conclusion label at MAR: sign in `{-1, 0, 1}`; excludes-null `{-1, 0, 1}` for
    /// "interval entirely below, includes, entirely above" the null; decision `{0, 1}`.
    pub at_mar: i8,
    /// First change on the negative side of MAR (`None` when the range does not reach one).
    pub negative: Option<MnarTippingSide>,
    /// First change on the positive side of MAR.
    pub positive: Option<MnarTippingSide>,
}

/// The envelope of the point effects over the declared range: an assumption range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MnarAssumptionRange {
    /// Parameter range the envelope covers.
    pub parameter_lower: f64,
    /// Parameter range the envelope covers.
    pub parameter_upper: f64,
    /// Smallest point effect over the grid.
    pub effect_lower: f64,
    /// Largest point effect over the grid.
    pub effect_upper: f64,
    /// Always `false`: the range is not a confidence interval.
    pub is_confidence_interval: bool,
    /// Always `false`: the departure is not empirically established.
    pub empirically_established: bool,
}

/// An assumption the result rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MnarAssumptionRecord {
    /// Stable id.
    pub id: &'static str,
    /// What is assumed.
    pub statement: &'static str,
    /// Whether the data can test it.
    pub status: AssumptionStatus,
}

/// A declared decision's behavior across the range.
#[derive(Clone, Debug, PartialEq)]
pub struct MnarDecisionReport {
    /// Decision name.
    pub name: String,
    /// Threshold.
    pub threshold: f64,
    /// Whether the decision acts under MAR.
    pub acts_at_mar: bool,
    /// Whether the action differs at any evaluated parameter.
    pub changes_within_range: bool,
    /// Its tipping points.
    pub tipping: MnarTipping,
}

/// Auditable MNAR sensitivity result.
#[derive(Clone, Debug, PartialEq)]
pub struct MnarSensitivityResult {
    /// [`MNAR_FAMILY`].
    pub family: &'static str,
    /// [`MNAR_TARGET`].
    pub target: &'static str,
    /// The departure model.
    pub model: MnarModel,
    /// The per-arm loading.
    pub loading: ArmLoading,
    /// The estimator.
    pub estimator: MnarEstimator,
    /// The MAR estimate (parameter zero).
    pub mar: MnarPoint,
    /// The estimate from the observed outcomes alone (complete cases of the outcome).
    pub observed_case_effect: f64,
    /// Rows analysed.
    pub rows: usize,
    /// Treated rows.
    pub treated_rows: usize,
    /// Treated rows with a missing outcome.
    pub treated_missing: usize,
    /// Control rows with a missing outcome.
    pub control_missing: usize,
    /// The effect as a function of the parameter, ascending in the parameter.
    pub grid: Vec<MnarPoint>,
    /// The assumption range over the declared parameter range.
    pub assumption_range: MnarAssumptionRange,
    /// Tipping points of the sign, of excluding the null, and of every declared decision.
    pub tipping: Vec<MnarTipping>,
    /// Decision changes for the declared decisions.
    pub decisions: Vec<MnarDecisionReport>,
    /// Level of the per-point sampling intervals.
    pub level: f64,
    /// Bracket tolerance of the tipping search.
    pub tolerance: f64,
    /// The assumptions the result rests on.
    pub assumptions: Vec<MnarAssumptionRecord>,
    /// [`MNAR_METHOD`].
    pub method: &'static str,
    /// [`MNAR_INTERPRETATION`].
    pub interpretation: &'static str,
    /// Always [`MNAR_ASSUMPTION_RANGE_CLAIM`].
    pub inference_claim: &'static str,
    /// Always [`MNAR_EMPIRICAL_STATUS`].
    pub empirical_status: &'static str,
    /// [`MNAR_SAMPLING_BASIS`].
    pub sampling_basis: &'static str,
}

/// One arm's observed outcomes and missing count.
struct Arm {
    observed: Vec<f64>,
    missing: usize,
    mean: f64,
    low: f64,
    high: f64,
}

impl Arm {
    fn size(&self) -> usize {
        self.observed.len() + self.missing
    }
}

/// Mean and sampling variance of an arm's full-population mean at departure `param`.
fn arm_mean(arm: &Arm, model: MnarModel, param: f64) -> (f64, f64) {
    let n_obs = arm.observed.len() as f64;
    let n = arm.size() as f64;
    let pi = n_obs / n;
    if arm.missing == 0 {
        let variance = sample_variance(arm.observed.iter().copied()) / n_obs;
        return (arm.mean, variance);
    }
    let (g, influence): (f64, Vec<f64>) = match model {
        MnarModel::DeltaPatternMixture => {
            (arm.mean + param, arm.observed.iter().map(|y| y - arm.mean).collect())
        }
        MnarModel::ExponentialTiltingSelection => {
            let anchor = if param >= 0.0 { arm.high } else { arm.low };
            let weights: Vec<f64> =
                arm.observed.iter().map(|y| (param * (y - anchor)).exp()).collect();
            let total: f64 = weights.iter().sum();
            let g = weights.iter().zip(&arm.observed).map(|(w, y)| w * y).sum::<f64>() / total;
            let mean_weight = total / n_obs;
            let influence =
                weights.iter().zip(&arm.observed).map(|(w, y)| w * (y - g) / mean_weight).collect();
            (g, influence)
        }
    };
    let mean = pi * arm.mean + (1.0 - pi) * g;
    let combined = arm.observed.iter().zip(&influence).map(|(y, psi)| pi * y + (1.0 - pi) * psi);
    let variance = (arm.mean - g).powi(2) * pi * (1.0 - pi) / n + sample_variance(combined) / n_obs;
    (mean, variance)
}

fn sample_variance(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = values.clone().count() as f64;
    let mean = values.clone().sum::<f64>() / n;
    values.map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

struct Prepared {
    treated: Arm,
    control: Arm,
    rows: usize,
    adjusted: Option<AdjData>,
}

fn make_arm(observed: Vec<f64>, missing: usize, label: &str) -> Result<Arm, MnarSensitivityError> {
    if observed.len() < 2 {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.insufficient_observed",
            format!(
                "the {label} arm has {} observed outcome(s); at least 2 are needed to report a sampling interval",
                observed.len()
            ),
        ));
    }
    let mean = observed.iter().sum::<f64>() / observed.len() as f64;
    let low = observed.iter().copied().fold(f64::INFINITY, f64::min);
    let high = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(Arm { observed, missing, mean, low, high })
}

#[allow(clippy::float_cmp, reason = "the treatment is coded exactly 0 or 1")]
fn prepare(
    data: &MnarData<'_>,
    estimator: MnarEstimator,
) -> Result<Prepared, MnarSensitivityError> {
    let rows = data.treatment.len();
    if rows != data.outcome.len() {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.invalid_data",
            format!("{rows} treatments but {} outcomes", data.outcome.len()),
        ));
    }
    if rows == 0 || rows > MNAR_MAX_ROWS {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "mnar_sensitivity.bounds_exceeded",
            format!("{rows} rows; 1 to {MNAR_MAX_ROWS} are admitted"),
        ));
    }
    let (mut treated, mut control) = (Vec::new(), Vec::new());
    let (mut treated_missing, mut control_missing) = (0usize, 0usize);
    for (&t, &y) in data.treatment.iter().zip(data.outcome) {
        let (observed, missing) = if t == 1.0 {
            (&mut treated, &mut treated_missing)
        } else if t == 0.0 {
            (&mut control, &mut control_missing)
        } else {
            return Err(refuse(
                reason_code!("invalid_argument"),
                "mnar_sensitivity.invalid_data",
                format!("treatment {t} is not 0 or 1 (a missing treatment is not admitted)"),
            ));
        };
        match y {
            Some(v) if v.is_finite() => observed.push(v),
            Some(v) => {
                return Err(refuse(
                    reason_code!("invalid_argument"),
                    "mnar_sensitivity.invalid_data",
                    format!("outcome {v} is not finite; a missing outcome is `None`"),
                ));
            }
            None => *missing += 1,
        }
    }
    let covariates = data.covariates;
    if !estimator.uses_covariates() && !covariates.is_empty() {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.covariates_not_used",
            "the difference in means reads no covariates; choose an adjusted estimator or drop them",
        ));
    }
    if estimator.uses_covariates() && covariates.is_empty() {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.covariates_required",
            format!("`{}` adjusts for covariates and none were declared", estimator.name()),
        ));
    }
    if covariates.len() > MNAR_MAX_COVARIATES {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "mnar_sensitivity.bounds_exceeded",
            format!("{} covariates; at most {MNAR_MAX_COVARIATES} are admitted", covariates.len()),
        ));
    }
    if covariates.iter().any(|c| c.len() != rows || c.iter().any(|v| !v.is_finite())) {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.invalid_data",
            "every covariate needs one finite value per row",
        ));
    }
    let treated = make_arm(treated, treated_missing, "treated")?;
    let control = make_arm(control, control_missing, "control")?;
    let width = covariates.len() + 1;
    if estimator.uses_covariates()
        && (treated.observed.len() <= width || control.observed.len() <= width)
    {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.insufficient_observed",
            format!(
                "each arm needs more than {width} observed outcomes to fit the outcome regression"
            ),
        ));
    }
    let adjusted = estimator
        .uses_covariates()
        .then(|| mnar_adjusted::adjusted_data(data.treatment, data.outcome, covariates));
    Ok(Prepared { treated, control, rows, adjusted })
}

#[allow(
    clippy::float_cmp,
    reason = "a loading of exactly zero is the declared absence of a departure"
)]
fn check_spec(spec: &MnarSensitivitySpec) -> Result<MnarMechanism, MnarSensitivityError> {
    if let Some(method) = &spec.sampling_composition {
        return Err(refuse(
            reason_code!("cell_not_licensed"),
            "mnar_sensitivity.composition_not_licensed",
            format!(
                "composing a sampling interval with the assumption range by `{method}` is not licensed"
            ),
        ));
    }
    let Some(mechanism) = spec.mechanism else {
        return Err(refuse(
            reason_code!("required_option_missing"),
            "mnar_sensitivity.mechanism_not_declared",
            "the outcome-missingness mechanism is not declared: name the departure model and the arm loading, because the departure from missing at random cannot be inferred from the data",
        ));
    };
    let loading = mechanism.loading;
    let loading_ok = |l: f64| l.is_finite() && (-1.0..=1.0).contains(&l);
    if !loading_ok(loading.treated) || !loading_ok(loading.control) {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.invalid_loading",
            "arm loadings must be finite and in [-1, 1]",
        ));
    }
    if loading.treated == 0.0 && loading.control == 0.0 {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.invalid_loading",
            "both arm loadings are zero, so the parameter changes nothing",
        ));
    }
    let cap = mechanism.model.max_magnitude();
    if !spec.lower.is_finite()
        || !spec.upper.is_finite()
        || spec.lower.abs() > cap
        || spec.upper.abs() > cap
    {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.unbounded_range",
            format!(
                "the parameter range [{}, {}] must be finite with magnitude at most {cap}: a sensitivity analysis needs a declared bounded range",
                spec.lower, spec.upper
            ),
        ));
    }
    if spec.lower > 0.0 || spec.upper < 0.0 || spec.lower >= spec.upper {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.range_excludes_mar",
            format!(
                "the range [{}, {}] must contain zero (missing at random) and have positive width",
                spec.lower, spec.upper
            ),
        ));
    }
    if !(2..=MNAR_MAX_GRID_POINTS).contains(&spec.grid_points) {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "mnar_sensitivity.bounds_exceeded",
            format!("{} grid points; 2 to {MNAR_MAX_GRID_POINTS} are admitted", spec.grid_points),
        ));
    }
    if !(spec.level > 0.0 && spec.level < 1.0) {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.invalid_level",
            format!("sampling interval level {} must be inside (0, 1)", spec.level),
        ));
    }
    if spec.decisions.len() > MNAR_MAX_DECISIONS {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "mnar_sensitivity.bounds_exceeded",
            format!(
                "{} decisions; at most {MNAR_MAX_DECISIONS} are admitted",
                spec.decisions.len()
            ),
        ));
    }
    if spec.decisions.iter().any(|d| !d.threshold.is_finite() || d.name.is_empty()) {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.invalid_decision",
            "a decision needs a name and a finite threshold",
        ));
    }
    if !spec.tolerance.is_finite()
        || !(MNAR_MIN_TOLERANCE..=MNAR_MAX_TOLERANCE).contains(&spec.tolerance)
    {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.invalid_tolerance",
            format!("tolerance {} is outside [1e-14, 1e-3]", spec.tolerance),
        ));
    }
    if spec.estimator.uses_covariates()
        && !(MNAR_MIN_BOOTSTRAP..=MNAR_MAX_BOOTSTRAP).contains(&spec.bootstrap_replicates)
    {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.invalid_bootstrap",
            format!(
                "{} bootstrap replicates; {MNAR_MIN_BOOTSTRAP} to {MNAR_MAX_BOOTSTRAP} are admitted",
                spec.bootstrap_replicates
            ),
        ));
    }
    Ok(mechanism)
}

/// Evaluation context of one analysis.
struct Context<'a> {
    prepared: &'a Prepared,
    mechanism: MnarMechanism,
    spec: &'a MnarSensitivitySpec,
    z: f64,
}

impl Context<'_> {
    /// The point at `parameter`; the sampling interval is computed only when `with_se` (a
    /// bootstrap is costly) and is NaN otherwise.
    fn try_point(&self, parameter: f64, with_se: bool) -> Result<MnarPoint, MnarSensitivityError> {
        let loading = self.mechanism.loading;
        let model = self.mechanism.model;
        let (treated_mean, control_mean, standard_error) = if let Some(data) =
            self.prepared.adjusted.as_ref()
        {
            let estimator = self.spec.estimator;
            let singular = || {
                refuse(
            reason_code!("invalid_argument"),
            "mnar_sensitivity.design_singular",
                    format!(
                        "`{}` could not be fitted at parameter {parameter}: a singular covariate design, a propensity outside (0, 1) or a propensity model that did not converge",
                        estimator.name()
                    ),
                )
            };
            let propensities = if matches!(
                estimator,
                MnarEstimator::InverseProbabilityWeighting
                    | MnarEstimator::AugmentedInverseProbabilityWeighting
            ) {
                Some(mnar_adjusted::propensity(data).ok_or_else(singular)?)
            } else {
                None
            };
            let (mu1, mu0) = mnar_adjusted::arm_means(
                data,
                propensities.as_deref(),
                model,
                loading,
                parameter,
                estimator,
            )
            .ok_or_else(singular)?;
            let se = if with_se {
                mnar_adjusted::bootstrap_se(
                    data,
                    model,
                    loading,
                    parameter,
                    estimator,
                    self.spec.bootstrap_replicates,
                    self.spec.seed,
                )
                .ok_or_else(|| {
                    refuse(
                        reason_code!("invalid_argument"),
                        "mnar_sensitivity.design_singular",
                        format!(
                            "the bootstrap of `{}` could not be fitted at parameter {parameter}",
                            estimator.name()
                        ),
                    )
                })?
            } else {
                f64::NAN
            };
            (mu1, mu0, se)
        } else {
            let (treated_mean, treated_var) =
                arm_mean(&self.prepared.treated, model, loading.treated * parameter);
            let (control_mean, control_var) =
                arm_mean(&self.prepared.control, model, loading.control * parameter);
            (treated_mean, control_mean, (treated_var + control_var).sqrt())
        };
        let effect = treated_mean - control_mean;
        Ok(MnarPoint {
            parameter,
            treated_mean,
            control_mean,
            effect,
            sampling: MnarSamplingInterval {
                standard_error,
                lower: effect - self.z * standard_error,
                upper: effect + self.z * standard_error,
            },
            acts: self.spec.decisions.iter().map(|d| effect > d.threshold).collect(),
        })
    }

    /// The point at `parameter`, or an all-NaN point (no conclusion changes toward it) when
    /// the estimator cannot be fitted there.
    fn point(&self, parameter: f64, with_se: bool) -> MnarPoint {
        self.try_point(parameter, with_se).unwrap_or_else(|_| MnarPoint {
            parameter,
            treated_mean: f64::NAN,
            control_mean: f64::NAN,
            effect: f64::NAN,
            sampling: MnarSamplingInterval {
                standard_error: f64::NAN,
                lower: f64::NAN,
                upper: f64::NAN,
            },
            acts: vec![false; self.spec.decisions.len()],
        })
    }

    /// Whether the sampling interval is cheap enough to evaluate inside every sub-step.
    fn cheap_interval(&self) -> bool {
        self.prepared.adjusted.is_none()
    }
}

fn sign_label(point: &MnarPoint) -> i8 {
    if point.effect > 0.0 {
        1
    } else if point.effect < 0.0 {
        -1
    } else {
        0
    }
}

fn null_label(point: &MnarPoint) -> i8 {
    if point.sampling.lower > 0.0 {
        1
    } else if point.sampling.upper < 0.0 {
        -1
    } else {
        0
    }
}

fn parameter_grid(spec: &MnarSensitivitySpec) -> Vec<f64> {
    let last = spec.grid_points - 1;
    (0..spec.grid_points)
        .map(|j| match j {
            0 => spec.lower,
            _ if j == last => spec.upper,
            _ => spec.lower + (spec.upper - spec.lower) * j as f64 / last as f64,
        })
        .collect()
}

/// First change of `label` from its value at MAR while walking `cells` outward from MAR.
///
/// Each grid cell is scanned at `substeps` interior points, so a change that reverses inside
/// one grid cell is still seen. The first sub-step at which the conclusion differs brackets
/// the tipping point, which is then refined by bisection on the continuous effect function
/// (the sampling interval, for the excludes-null conclusion) to `tolerance`.
fn first_change(
    context: &Context<'_>,
    label: &impl Fn(&MnarPoint) -> i8,
    needs_se: bool,
    at_mar: i8,
    cells: &[f64],
    substeps: usize,
    tolerance: f64,
) -> Option<MnarTippingSide> {
    let mut previous = 0.0_f64;
    for &next in cells {
        for step in 1..=substeps {
            let probe = if step == substeps {
                next
            } else {
                previous + (next - previous) * step as f64 / substeps as f64
            };
            let mut near = if step == 1 {
                previous
            } else {
                previous + (next - previous) * (step - 1) as f64 / substeps as f64
            };
            if label(&context.point(probe, needs_se)) == at_mar {
                continue;
            }
            let mut far = probe;
            let mut iterations = 0usize;
            while (far - near).abs() > tolerance && iterations < MNAR_MAX_BISECTIONS {
                iterations += 1;
                let middle = near + (far - near) / 2.0;
                if middle == near || middle == far {
                    break;
                }
                if label(&context.point(middle, needs_se)) == at_mar {
                    near = middle;
                } else {
                    far = middle;
                }
            }
            return Some(MnarTippingSide {
                parameter: far,
                bracket_lower: near,
                iterations,
                changes_to: label(&context.point(far, needs_se)),
            });
        }
        previous = next;
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn tipping_of(
    context: &Context<'_>,
    kind: MnarConclusionKind,
    name: &str,
    threshold: f64,
    label: &impl Fn(&MnarPoint) -> i8,
    needs_se: bool,
    grid: &[f64],
    tolerance: f64,
) -> MnarTipping {
    let at_mar = label(&context.point(0.0, needs_se));
    // Sub-cell scanning of an interval conclusion costs a bootstrap per sub-step on the
    // adjusted estimators; there the declared grid is the scan.
    let substeps = if needs_se && !context.cheap_interval() { 1 } else { MNAR_CELL_SUBSTEPS };
    // Each side is walked away from MAR: the grid point nearest zero first.
    let negative_walk: Vec<f64> = grid.iter().rev().copied().filter(|p| *p < 0.0).collect();
    let positive_walk: Vec<f64> = grid.iter().copied().filter(|p| *p > 0.0).collect();
    MnarTipping {
        kind,
        label: name.to_owned(),
        threshold,
        at_mar,
        negative: first_change(
            context,
            label,
            needs_se,
            at_mar,
            &negative_walk,
            substeps,
            tolerance,
        ),
        positive: first_change(
            context,
            label,
            needs_se,
            at_mar,
            &positive_walk,
            substeps,
            tolerance,
        ),
    }
}

/// Re-run the declared estimand across the declared bounded grid of a departure from
/// missing at random, with the tipping points of the treatment conclusions and the changes
/// of the declared decisions.
///
/// # Errors
///
/// A typed [`MnarSensitivityError`]: an undeclared mechanism
/// (`mnar_sensitivity.mechanism_not_declared`), an unsupported estimator
/// (`mnar_sensitivity.unsupported_estimator`, from [`MnarEstimator::from_name`]), an
/// unbounded or MAR-excluding range (`unbounded_range`, `range_excludes_mar`), a
/// requested sampling composition (`composition_not_licensed`), malformed data, loadings,
/// level, decisions or tolerance, or exceeded bounds.
pub fn mnar_sensitivity(
    data: &MnarData<'_>,
    spec: &MnarSensitivitySpec,
) -> Result<MnarSensitivityResult, MnarSensitivityError> {
    let mechanism = check_spec(spec)?;
    let prepared = prepare(data, spec.estimator)?;
    let evaluations = (spec.grid_points as u64)
        + 4 * (MNAR_MAX_BISECTIONS as u64) * (2 + spec.decisions.len() as u64);
    let interval_evaluations = if prepared.adjusted.is_some() {
        u64::from(spec.bootstrap_replicates)
            .saturating_mul(spec.grid_points as u64 + 2 * MNAR_MAX_BISECTIONS as u64)
    } else {
        0
    };
    if (prepared.rows as u64).saturating_mul(evaluations.saturating_add(interval_evaluations))
        > MNAR_MAX_WORK
    {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "mnar_sensitivity.bounds_exceeded",
            format!(
                "{} rows over the grid and tipping searches exceed the work budget",
                prepared.rows
            ),
        ));
    }
    let z = normal_ppf(0.5 + spec.level / 2.0);
    let context = Context { prepared: &prepared, mechanism, spec, z };
    let mar = context.try_point(0.0, true)?;
    let grid_parameters = parameter_grid(spec);
    let grid = grid_parameters
        .iter()
        .map(|&p| context.try_point(p, true))
        .collect::<Result<Vec<MnarPoint>, MnarSensitivityError>>()?;
    let tolerance = spec.tolerance * (spec.upper - spec.lower);
    let mut tipping = vec![
        tipping_of(
            &context,
            MnarConclusionKind::Sign,
            "sign",
            0.0,
            &sign_label,
            false,
            &grid_parameters,
            tolerance,
        ),
        tipping_of(
            &context,
            MnarConclusionKind::ExcludesNull,
            "excludes_null",
            0.0,
            &null_label,
            true,
            &grid_parameters,
            tolerance,
        ),
    ];
    let mut decisions = Vec::with_capacity(spec.decisions.len());
    for (index, decision) in spec.decisions.iter().enumerate() {
        let label = |point: &MnarPoint| i8::from(point.acts[index]);
        let report = tipping_of(
            &context,
            MnarConclusionKind::Decision,
            &decision.name,
            decision.threshold,
            &label,
            false,
            &grid_parameters,
            tolerance,
        );
        decisions.push(MnarDecisionReport {
            name: decision.name.clone(),
            threshold: decision.threshold,
            acts_at_mar: mar.acts[index],
            changes_within_range: grid.iter().any(|point| point.acts[index] != mar.acts[index]),
            tipping: report.clone(),
        });
        tipping.push(report);
    }
    let effects = grid.iter().map(|point| point.effect);
    let assumption_range = MnarAssumptionRange {
        parameter_lower: spec.lower,
        parameter_upper: spec.upper,
        effect_lower: effects.clone().fold(f64::INFINITY, f64::min),
        effect_upper: effects.fold(f64::NEG_INFINITY, f64::max),
        is_confidence_interval: false,
        empirically_established: false,
    };
    let observed_case_effect = prepared.treated.mean - prepared.control.mean;
    Ok(MnarSensitivityResult {
        family: MNAR_FAMILY,
        target: MNAR_TARGET,
        model: mechanism.model,
        loading: mechanism.loading,
        estimator: spec.estimator,
        mar,
        observed_case_effect,
        rows: prepared.rows,
        treated_rows: prepared.treated.size(),
        treated_missing: prepared.treated.missing,
        control_missing: prepared.control.missing,
        grid,
        assumption_range,
        tipping,
        decisions,
        level: spec.level,
        tolerance: spec.tolerance,
        assumptions: assumptions(),
        method: MNAR_METHOD,
        interpretation: MNAR_INTERPRETATION,
        inference_claim: MNAR_ASSUMPTION_RANGE_CLAIM,
        empirical_status: MNAR_EMPIRICAL_STATUS,
        sampling_basis: MNAR_SAMPLING_BASIS,
    })
}

/// The assumptions every result rests on.
#[must_use]
pub fn assumptions() -> Vec<MnarAssumptionRecord> {
    vec![
        MnarAssumptionRecord {
            id: MNAR_DEPARTURE_ASSUMPTION_ID,
            statement: "outcome missingness departs from missing at random by exactly the declared one-parameter model, with the parameter inside the declared range; the data cannot test the departure because the missing outcomes are unobserved",
            status: MNAR_DEPARTURE_ASSUMPTION_STATUS,
        },
        MnarAssumptionRecord {
            id: MNAR_ASSIGNMENT_ASSUMPTION_ID,
            statement: "treatment assignment is independent of the potential outcomes (randomized or ignorable); this analysis varies only the outcome-missingness mechanism and does not relax it",
            status: AssumptionStatus::Declared,
        },
    ]
}
