//! Classical additive error in a continuous column, beside the linear moment
//! correction: regression calibration and SIMEX for linear and logistic
//! outcome models, and the outcome-error case that needs no correction.
//!
//! The error can sit in three places:
//!
//! * **the treatment** (the regressor at coefficient index 1): the reported
//!   functional is the corrected treatment coefficient;
//! * **a covariate**: the error-prone column is still the regressor `W` at
//!   index 1, and the treatment is the error-free covariate at a declared
//!   column, coefficient index `2 + column`; correcting `W` removes the bias
//!   it transmits to the treatment coefficient;
//! * **the outcome**: under an identity link classical outcome error does not
//!   bias a slope and only inflates the residual variance, which the sandwich
//!   variance already includes, so the naive estimate is returned with a
//!   record saying no correction is needed; under a logit link the outcome is
//!   binary and cannot carry additive continuous error, so the request is
//!   refused toward the binary misclassification route.
//!
//! The error variance comes from the model: [`MeasurementParameters::ClassicalError`]
//! declares it (with a standard error), [`MeasurementParameters::Repeated`]
//! estimates it from the analysis sample's own replicates. Any other route, or
//! a repeated route without replicated units, leaves it unidentified and
//! refuses with `measurement_not_identified`.
//!
//! # Uncertainty
//!
//! Linear regression calibration keeps the delta-method variance of
//! [`antecedent_stats::measurement_error::LinearMoments`], including the
//! error-variance estimate's uncertainty. Logistic regression calibration and
//! SIMEX (either link) use a nonparametric bootstrap of the **whole**
//! procedure: rows are resampled with their replicates, the error variance is
//! redrawn from `N(sigma^2, se^2)` truncated at `1e-3 sigma^2` (declared) or
//! re-estimated from the resampled replicates (repeated), and the correction
//! is re-run. Each replicate opens its own keyed random stream from the
//! request's seed, so the result is bitwise reproducible and independent of
//! thread count. The bootstrap needs at least [`MIN_BOOTSTRAP_REPLICATES`]
//! replicates; more than a tenth failing (a reliability at or below the floor
//! or an outcome model that does not fit in a resample) refuses rather than
//! report a variance from the survivors of a selection. The variance is split
//! into the measurement-parameter part, `(d theta / d sigma^2)^2 Var(sigma^2)`
//! with the derivative taken by a central difference under common random
//! numbers, and the remaining sampling part. The standard error is
//! uncalibrated and no interval is produced.
//!
//! References: Cook and Stefanski (1994), doi:10.1080/01621459.1994.10476871;
//! Carroll, Ruppert, Stefanski and Crainiceanu (2006), *Measurement Error in
//! Nonlinear Models*, 2nd ed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::execution::{AdmissionError, AdmissionLease};
use antecedent_core::measurement_model::{
    CorrectedFunctionalRecord, MeasurementModel, MeasurementParameters, MeasurementRoute,
    ParameterUncertainty,
};
use antecedent_core::{
    CausalRng, ChainId, ExecutionContext, FoldId, NextAttempt, ReductionPolicy, ReplicateId,
    RngKey, SemanticOperation,
};
use antecedent_kernels::{standard_normal, unbiased_index};
use antecedent_stats::continuous_error::{
    OutcomeFamily, SimexConfig, regression_calibration, simex,
};
pub use antecedent_stats::continuous_error::{SimexExtrapolant, SimexFit};
use antecedent_stats::measurement_error::{
    ErrorVariance, LinearErrorData, LinearMoments, MeasurementErrorError, VarianceComponents,
    replicate_variance_components,
};

use super::{
    DataBinding, MeasurementError, RecordParts, build_record, check_binding, check_cancelled,
    component, counts_as_reals, digest, require_non_differential, stats_error,
};

/// What to try when the error variance is not identified.
pub const SUPPLY_REPLICATES_OR_VALIDATION: NextAttempt =
    NextAttempt::new("supply_replicates_or_validation", "supply replicates or a validation sample");

/// What to try when a binary outcome is declared to carry continuous error.
pub const USE_BINARY_MISCLASSIFICATION: NextAttempt = NextAttempt::new(
    "binary_misclassification_route",
    "declare the binary outcome's error with the binary misclassification route",
);

/// Fewest bootstrap replicates a bootstrap-based correction accepts.
pub const MIN_BOOTSTRAP_REPLICATES: u32 = 50;
/// Most bootstrap replicates a bootstrap-based correction accepts.
pub const MAX_BOOTSTRAP_REPLICATES: u32 = 100_000;
/// Bootstrap replicates a caller gets when it states none.
pub const DEFAULT_BOOTSTRAP_REPLICATES: u32 = 200;

/// Largest share of bootstrap replicates that may fail before the bootstrap refuses.
const MAX_FAILED_FRACTION: f64 = 0.1;
/// A redrawn error variance is truncated below at this multiple of the estimate.
const ERROR_VARIANCE_DRAW_FLOOR: f64 = 1e-3;
/// Relative step of the error-variance derivative.
const DERIVATIVE_STEP: f64 = 1e-3;
/// Tolerance of an error-prone column against the row means of its replicates.
const ROW_MEAN_TOLERANCE: f64 = 1e-9;
/// Stream tag of this correction's random draws.
const CONTINUOUS_ERROR_OPERATION: SemanticOperation = SemanticOperation::new(0x4D45_5F43_4F4E_5449);
/// Fold of the point estimate's stream (SIMEX remeasurement).
const POINT_FOLD: u64 = 0;
/// Fold of the bootstrap replicates' streams.
const BOOTSTRAP_FOLD: u64 = 1;

/// Which analysed column carries classical additive error.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ErrorLocation {
    /// The treatment, the regressor at coefficient index 1.
    Treatment,
    /// A covariate: the error-prone column is the regressor at index 1 and the
    /// treatment is the error-free covariate `treatment_column`.
    Covariate {
        /// Covariate column of the treatment (coefficient index `2 + column`).
        treatment_column: usize,
    },
    /// The outcome.
    Outcome,
}

impl ErrorLocation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Treatment => "treatment",
            Self::Covariate { .. } => "covariate",
            Self::Outcome => "outcome",
        }
    }
}

/// Link of the outcome model.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OutcomeLink {
    /// Linear outcome model.
    Identity,
    /// Logistic outcome model of a 0/1 outcome.
    Logit,
}

impl OutcomeLink {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Logit => "logit",
        }
    }

    const fn family(self) -> OutcomeFamily {
        match self {
            Self::Identity => OutcomeFamily::Linear,
            Self::Logit => OutcomeFamily::Logistic,
        }
    }

    const fn coefficient_name(self) -> &'static str {
        match self {
            Self::Identity => "linear_coefficient",
            Self::Logit => "logit_coefficient",
        }
    }
}

/// How the error is corrected.
#[derive(Clone, Debug, PartialEq)]
pub enum CorrectionMethod {
    /// Regression calibration: the outcome model on `E[X | W, Z]`.
    RegressionCalibration,
    /// SIMEX (Cook and Stefanski 1994).
    Simex {
        /// Positive, distinct added-error multiples; `lambda = 0` is added.
        lambdas: Vec<f64>,
        /// Remeasurement replicates per `lambda`.
        replicates: u32,
        /// Extrapolant evaluated at `lambda = -1`.
        extrapolant: SimexExtrapolant,
    },
}

impl CorrectionMethod {
    /// SIMEX on the default grid `0.5, 1, 1.5, 2` with a quadratic extrapolant.
    #[must_use]
    pub fn default_simex() -> Self {
        let config = SimexConfig::default();
        Self::Simex {
            lambdas: config.lambdas,
            replicates: config.replicates,
            extrapolant: config.extrapolant,
        }
    }

    fn simex_config(&self) -> Option<SimexConfig> {
        match self {
            Self::RegressionCalibration => None,
            Self::Simex { lambdas, replicates, extrapolant } => Some(SimexConfig {
                lambdas: lambdas.clone(),
                replicates: *replicates,
                extrapolant: *extrapolant,
            }),
        }
    }
}

/// The question asked of the data: where the error is, which outcome model,
/// which correction, and the bootstrap that carries its uncertainty.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousCorrectionSpec {
    /// Column carrying the error.
    pub location: ErrorLocation,
    /// Outcome model link.
    pub link: OutcomeLink,
    /// Correction method.
    pub method: CorrectionMethod,
    /// Bootstrap replicates for logistic calibration and SIMEX; ignored by
    /// linear calibration (delta method) and by outcome error (no correction).
    pub bootstrap_replicates: u32,
    /// Seed of every random stream the correction opens.
    pub seed: u64,
}

/// Columns of one analysis with a continuous error-prone column.
///
/// The design is `(1, regressor, covariates)`. When the error is in the
/// treatment or a covariate, `regressor` is the error-prone `W`; when it is in
/// the outcome, `regressor` is the error-free treatment and `outcome` is the
/// error-prone column.
#[derive(Clone, Copy, Debug)]
pub struct ContinuousErrorData<'a> {
    /// Outcome. May be empty when the error is in the outcome and
    /// `replicate_values` supplies it.
    pub outcome: &'a [f64],
    /// Regressor at coefficient index 1. May be empty when the error is in it
    /// and `replicate_values` supplies it.
    pub regressor: &'a [f64],
    /// Error-free covariates, column-major `n x covariate_count`.
    pub covariates: &'a [f64],
    /// Number of covariate columns.
    pub covariate_count: usize,
    /// Every row's measurements of the error-prone column, back to back. When
    /// given, the error-prone column is their row means (a supplied column
    /// must equal them).
    pub replicate_values: Option<&'a [f64]>,
    /// Measurements averaged into each row of the error-prone column; `None`
    /// means one each. Required with `replicate_values`.
    pub replicate_counts: Option<&'a [u32]>,
}

/// The correction that produced a result.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ContinuousMethodKind {
    /// Regression calibration.
    RegressionCalibration,
    /// SIMEX.
    Simex,
    /// Outcome error under an identity link: the naive estimate is unbiased.
    NoCorrectionNeeded,
}

impl ContinuousMethodKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RegressionCalibration => "regression_calibration",
            Self::Simex => "simex",
            Self::NoCorrectionNeeded => "outcome_error_no_correction",
        }
    }
}

/// What a continuous correction used and how its uncertainty was formed.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousCorrectionSummary {
    /// Correction applied.
    pub method: ContinuousMethodKind,
    /// Outcome model link.
    pub link: OutcomeLink,
    /// Column carrying the error.
    pub location: ErrorLocation,
    /// Error variance of one measurement used; `None` when no correction ran.
    pub error_variance: Option<f64>,
    /// Standard error of that variance (zero when declared exact).
    pub error_variance_standard_error: Option<f64>,
    /// Implied reliability of the error-prone regressor given the covariates.
    pub reliability: Option<f64>,
    /// Bootstrap replicates whose estimate entered the variance; zero for the
    /// delta method and for outcome error.
    pub bootstrap_replicates: u32,
    /// Bootstrap replicates that failed (at most a tenth of those requested).
    pub failed_bootstrap_replicates: u32,
    /// Seed of the random streams; `None` when nothing random ran.
    pub seed: Option<u64>,
}

/// A corrected coefficient under continuous measurement error and its bound
/// record.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedContinuousEffect {
    /// Corrected coefficients on `(1, regressor, covariates)`.
    pub coefficients: Vec<f64>,
    /// Uncorrected coefficients of the same outcome model.
    pub naive_coefficients: Vec<f64>,
    /// Index of the reported coefficient (the treatment's).
    pub coefficient_index: usize,
    /// What the correction used.
    pub summary: ContinuousCorrectionSummary,
    /// The SIMEX path of the point estimate (SIMEX only).
    pub simex: Option<SimexFit>,
    /// Replicate variance components (repeated route only).
    pub replicate_components: Option<VarianceComponents>,
    /// The reported coefficient in each successful bootstrap replicate.
    pub bootstrap_estimates: Vec<f64>,
    /// The reported coefficient, bound to the model identity.
    pub record: CorrectedFunctionalRecord,
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

fn invalid(message: impl Into<String>) -> MeasurementError {
    MeasurementError::new(
        antecedent_core::reason_code!("invalid_argument"),
        "measurement_model.input_invalid",
        message,
    )
}

fn not_identified(message: &str) -> MeasurementError {
    MeasurementError::new(
        antecedent_core::reason_code!("measurement_not_identified"),
        "measurement_model.error_variance_not_identified",
        SUPPLY_REPLICATES_OR_VALIDATION.append_to(message),
    )
}

fn cancelled() -> MeasurementError {
    MeasurementError::new(
        antecedent_core::reason_code!("cancelled_no_claim"),
        "measurement_model.cancelled",
        "cancelled before a corrected functional was formed; this is not a verdict on the data",
    )
}

fn continuous_error(error: &MeasurementErrorError) -> MeasurementError {
    match error {
        MeasurementErrorError::OutcomeModelNotFitted(why) => MeasurementError::new(
            antecedent_core::reason_code!("measurement_not_identified"),
            "measurement_model.outcome_model_not_fitted",
            format!("the corrected outcome model could not be fitted: {why}"),
        ),
        other => stats_error(other),
    }
}

fn admit(ctx: &ExecutionContext, bytes: Option<usize>) -> Result<AdmissionLease, MeasurementError> {
    let bytes = bytes
        .and_then(|b| u64::try_from(b).ok())
        .ok_or_else(|| invalid("the bootstrap buffer size overflows"))?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        AdmissionError::Cancelled => cancelled(),
        other => MeasurementError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "measurement_model.admission_refused",
            format!(
                "the admission pool cannot fund {bytes} bytes of correction and bootstrap buffers: {other}; nothing was allocated"
            ),
        ),
    })
}

// ---------------------------------------------------------------------------
// Columns
// ---------------------------------------------------------------------------

/// Owned, validated columns with the error-prone column resolved.
struct Columns {
    n: usize,
    p: usize,
    outcome: Vec<f64>,
    regressor: Vec<f64>,
    covariates: Vec<f64>,
    counts: Option<Vec<u32>>,
    replicate_values: Option<Vec<f64>>,
    offsets: Vec<usize>,
}

impl Columns {
    fn view(&self) -> LinearErrorData<'_> {
        LinearErrorData {
            y: &self.outcome,
            w: &self.regressor,
            covariates: &self.covariates,
            covariate_count: self.p,
            replicate_counts: self.counts.as_deref(),
        }
    }

    /// The rows `rows` (with repetition), replicates included.
    fn gather(&self, rows: &[usize]) -> Self {
        let n = rows.len();
        let outcome = rows.iter().map(|&r| self.outcome[r]).collect();
        let regressor = rows.iter().map(|&r| self.regressor[r]).collect();
        let mut covariates = Vec::with_capacity(n * self.p);
        for j in 0..self.p {
            let column = &self.covariates[j * self.n..(j + 1) * self.n];
            covariates.extend(rows.iter().map(|&r| column[r]));
        }
        let counts: Option<Vec<u32>> =
            self.counts.as_ref().map(|c| rows.iter().map(|&r| c[r]).collect());
        let mut offsets = Vec::new();
        let replicate_values = match (&self.replicate_values, &self.counts) {
            (Some(values), Some(counts)) => {
                let mut out = Vec::new();
                for &r in rows {
                    offsets.push(out.len());
                    let k = counts[r] as usize;
                    out.extend_from_slice(&values[self.offsets[r]..self.offsets[r] + k]);
                }
                Some(out)
            }
            _ => None,
        };
        Self { n, p: self.p, outcome, regressor, covariates, counts, replicate_values, offsets }
    }
}

fn row_means(values: &[f64], counts: &[u32]) -> Result<(Vec<f64>, Vec<usize>), MeasurementError> {
    let total: usize = counts.iter().map(|&k| k as usize).sum();
    if values.len() != total {
        return Err(invalid("replicate values do not match the replicate counts"));
    }
    if counts.contains(&0) {
        return Err(invalid("every row needs at least one measurement"));
    }
    let mut means = Vec::with_capacity(counts.len());
    let mut offsets = Vec::with_capacity(counts.len());
    let mut at = 0_usize;
    for &k in counts {
        let k = k as usize;
        offsets.push(at);
        means.push(values[at..at + k].iter().sum::<f64>() / k as f64);
        at += k;
    }
    Ok((means, offsets))
}

/// The error-prone column: as supplied, or the row means of its replicates.
fn resolve_error_prone(
    given: &[f64],
    data: &ContinuousErrorData<'_>,
    n: usize,
) -> Result<(Vec<f64>, Vec<usize>), MeasurementError> {
    match (data.replicate_values, data.replicate_counts) {
        (None, _) => {
            if given.len() != n {
                return Err(invalid("the error-prone column differs in length from the others"));
            }
            Ok((given.to_vec(), Vec::new()))
        }
        (Some(_), None) => Err(invalid("replicate values need their per-row replicate counts")),
        (Some(values), Some(counts)) => {
            if counts.len() != n {
                return Err(invalid("replicate counts differ in length from the other columns"));
            }
            let (means, offsets) = row_means(values, counts)?;
            if given.is_empty() {
                return Ok((means, offsets));
            }
            if given.len() != n {
                return Err(invalid("the error-prone column differs in length from the others"));
            }
            let agrees = given
                .iter()
                .zip(&means)
                .all(|(g, m)| (g - m).abs() <= ROW_MEAN_TOLERANCE * (1.0 + m.abs()));
            if !agrees {
                return Err(invalid(
                    "the error-prone column is not the row mean of its replicates; supply the means or leave the column empty",
                ));
            }
            Ok((means, offsets))
        }
    }
}

fn columns(
    data: &ContinuousErrorData<'_>,
    location: ErrorLocation,
) -> Result<Columns, MeasurementError> {
    let in_outcome = location == ErrorLocation::Outcome;
    let n = if in_outcome { data.regressor.len() } else { data.outcome.len() };
    let p = data.covariate_count;
    if p.checked_mul(n) != Some(data.covariates.len()) {
        return Err(invalid("covariates are not n x covariate_count"));
    }
    if let Some(counts) = data.replicate_counts {
        if counts.len() != n {
            return Err(invalid("replicate counts differ in length from the other columns"));
        }
    }
    let (outcome, regressor, offsets) = if in_outcome {
        let (outcome, offsets) = resolve_error_prone(data.outcome, data, n)?;
        (outcome, data.regressor.to_vec(), offsets)
    } else {
        let (regressor, offsets) = resolve_error_prone(data.regressor, data, n)?;
        (data.outcome.to_vec(), regressor, offsets)
    };
    if let ErrorLocation::Covariate { treatment_column } = location {
        if treatment_column >= p {
            return Err(invalid(format!(
                "the treatment column {treatment_column} is not one of the {p} covariates"
            )));
        }
    }
    Ok(Columns {
        n,
        p,
        outcome,
        regressor,
        covariates: data.covariates.to_vec(),
        counts: data.replicate_counts.map(<[u32]>::to_vec),
        replicate_values: data.replicate_values.map(<[f64]>::to_vec),
        offsets,
    })
}

// ---------------------------------------------------------------------------
// Error variance
// ---------------------------------------------------------------------------

enum Source {
    Declared { variance: f64, standard_error: f64 },
    Replicates { error: ErrorVariance, components: VarianceComponents },
}

impl Source {
    fn error(&self) -> Result<ErrorVariance, MeasurementError> {
        match self {
            Self::Declared { variance, standard_error } => {
                ErrorVariance::known(*variance, *standard_error).map_err(|e| stats_error(&e))
            }
            Self::Replicates { error, .. } => Ok(error.clone()),
        }
    }

    fn variance(&self) -> f64 {
        match self {
            Self::Declared { variance, .. } => *variance,
            Self::Replicates { error, .. } => error.variance,
        }
    }

    /// Variance of the error-variance estimate.
    fn estimate_variance(&self) -> f64 {
        match self {
            Self::Declared { standard_error, .. } => standard_error * standard_error,
            Self::Replicates { error, .. } => error.estimate_variance,
        }
    }

    fn uncertainty(&self) -> ParameterUncertainty {
        if self.estimate_variance() > 0.0 {
            ParameterUncertainty::Propagated
        } else {
            ParameterUncertainty::DeclaredExact
        }
    }
}

fn reliability_floor(model: &MeasurementModel) -> Result<f64, MeasurementError> {
    match (&model.route, &model.parameters) {
        (
            MeasurementRoute::ClassicalContinuousError,
            MeasurementParameters::ClassicalError { min_reliability, .. },
        )
        | (
            MeasurementRoute::RepeatedMeasurements,
            MeasurementParameters::Repeated { min_reliability, .. },
        ) => Ok(*min_reliability),
        _ => Err(not_identified(&format!(
            "the model declares route `{}`, which identifies no error variance for a continuous column",
            model.route.as_str()
        ))),
    }
}

fn error_source(model: &MeasurementModel, cols: &Columns) -> Result<Source, MeasurementError> {
    match &model.parameters {
        MeasurementParameters::ClassicalError {
            error_variance,
            error_variance_standard_error,
            ..
        } => Ok(Source::Declared {
            variance: *error_variance,
            standard_error: *error_variance_standard_error,
        }),
        MeasurementParameters::Repeated { min_replicated_units, .. } => {
            let (Some(values), Some(counts)) = (&cols.replicate_values, &cols.counts) else {
                return Err(not_identified(
                    "the repeated-measurement route estimates the error variance from replicates and none were supplied",
                ));
            };
            let replicated = counts.iter().filter(|&&k| k >= 2).count();
            let needed = (*min_replicated_units as usize).max(1);
            if replicated < needed {
                return Err(MeasurementError::new(
                    antecedent_core::reason_code!("measurement_not_identified"),
                    "measurement_model.insufficient_replicates",
                    SUPPLY_REPLICATES_OR_VALIDATION.append_to(&format!(
                        "{replicated} units carry two or more replicates; the model declares at least {needed}"
                    )),
                ));
            }
            let (error, components) = ErrorVariance::from_main_sample_replicates(values, counts)
                .map_err(|e| stats_error(&e))?;
            Ok(Source::Replicates { error, components })
        }
        _ => Err(not_identified("the model's parameters identify no error variance")),
    }
}

// ---------------------------------------------------------------------------
// Corrections
// ---------------------------------------------------------------------------

struct Correction {
    coefficients: Vec<f64>,
    naive: Vec<f64>,
    reliability: f64,
    simex: Option<SimexFit>,
}

fn stream(seed: u64, fold: u64, replicate: u64) -> CausalRng {
    RngKey::new(
        CONTINUOUS_ERROR_OPERATION,
        FoldId::new(fold),
        ChainId::new(0),
        ReplicateId::new(replicate),
        ReductionPolicy::Strict,
    )
    .open(seed)
}

fn run_correction(
    data: &LinearErrorData<'_>,
    family: OutcomeFamily,
    error_variance: f64,
    floor: f64,
    simex_config: Option<&SimexConfig>,
    rng: &mut CausalRng,
) -> Result<Correction, MeasurementErrorError> {
    match simex_config {
        None => regression_calibration(data, family, error_variance, floor).map(|fit| Correction {
            coefficients: fit.coefficients,
            naive: fit.naive_coefficients,
            reliability: fit.reliability,
            simex: None,
        }),
        Some(config) => {
            simex(data, family, error_variance, floor, config, rng).map(|fit| Correction {
                coefficients: fit.coefficients.clone(),
                naive: fit.naive_coefficients.clone(),
                reliability: fit.reliability,
                simex: Some(fit),
            })
        }
    }
}

fn data_identity(cols: &Columns) -> antecedent_core::identity::SemanticDigest {
    let counts = cols.counts.as_deref().map(counts_as_reals).unwrap_or_default();
    digest(&[
        ("outcome", &cols.outcome),
        ("regressor", &cols.regressor),
        ("covariates", &cols.covariates),
        ("replicate_counts", &counts),
        ("replicate_values", cols.replicate_values.as_deref().unwrap_or(&[])),
    ])
}

fn coefficient_index(location: ErrorLocation) -> usize {
    match location {
        ErrorLocation::Treatment | ErrorLocation::Outcome => 1,
        ErrorLocation::Covariate { treatment_column } => 2 + treatment_column,
    }
}

/// Correct one coefficient for classical additive error in a continuous column.
///
/// The checks run in order: cancellation; the model is well formed and bound
/// to this data; the error is declared non-differential; the route identifies
/// an error variance (classical or repeated, else `measurement_not_identified`
/// with the next attempt [`SUPPLY_REPLICATES_OR_VALIDATION`]); the columns
/// agree; a logit outcome model with the error in the outcome is refused
/// toward [`USE_BINARY_MISCLASSIFICATION`]; and the error variance is
/// identified from the parameters or the replicates.
///
/// # Errors
/// Each failed check as a coded [`MeasurementError`]; an implied reliability
/// at or below the model's floor; a rank-deficient design; an outcome model
/// that does not fit; a bootstrap replicate count outside
/// [`MIN_BOOTSTRAP_REPLICATES`]`..=`[`MAX_BOOTSTRAP_REPLICATES`]; more than a
/// tenth of bootstrap replicates failing; an admission pool that cannot fund
/// the buffers; or cancellation.
pub fn correct_continuous_error(
    model: &MeasurementModel,
    binding: &DataBinding,
    data: &ContinuousErrorData<'_>,
    spec: &ContinuousCorrectionSpec,
    ctx: &ExecutionContext,
) -> Result<CorrectedContinuousEffect, MeasurementError> {
    check_cancelled(ctx)?;
    check_binding(model, binding)?;
    require_non_differential(model)?;
    let floor = reliability_floor(model)?;
    let cols = columns(data, spec.location)?;
    match (spec.location, spec.link) {
        (ErrorLocation::Outcome, OutcomeLink::Logit) => Err(MeasurementError::new(
            antecedent_core::reason_code!("route_not_supported"),
            "measurement_model.binary_outcome_continuous_error",
            USE_BINARY_MISCLASSIFICATION.append_to(
                "a logit outcome model has a 0/1 outcome, which cannot carry additive continuous error",
            ),
        )),
        (ErrorLocation::Outcome, OutcomeLink::Identity) => outcome_pass_through(model, &cols, spec),
        _ => {
            let source = error_source(model, &cols)?;
            if let (OutcomeLink::Identity, CorrectionMethod::RegressionCalibration) =
                (spec.link, &spec.method)
            {
                linear_calibration(model, &cols, spec, &source, floor)
            } else {
                bootstrapped(model, &cols, spec, &source, floor, ctx)
            }
        }
    }
}

fn outcome_pass_through(
    model: &MeasurementModel,
    cols: &Columns,
    spec: &ContinuousCorrectionSpec,
) -> Result<CorrectedContinuousEffect, MeasurementError> {
    let exact = ErrorVariance::known(0.0, 0.0).map_err(|e| stats_error(&e))?;
    // The error-free regressor carries no error; the replicate counts belong
    // to the outcome and do not enter the design.
    let view = LinearErrorData { replicate_counts: None, ..cols.view() };
    let fit = LinearMoments::accumulate(&view, &exact, 0.0)
        .and_then(|moments| moments.solve())
        .map_err(|e| continuous_error(&e))?;
    let d = fit.dim;
    let variance = fit.covariance[d + 1];
    let record = build_record(RecordParts {
        model,
        functional_id: "linear_coefficient.outcome_error_no_correction",
        point: fit.coefficients[1],
        naive_point: Some(fit.naive_coefficients[1]),
        variance: Some(variance),
        components: vec![
            component("sampling_including_outcome_error", variance),
            component("measurement_parameter_uncertainty", 0.0),
        ],
        rows: cols.n as u64,
        data_identity: data_identity(cols),
        // The error variance does not enter the slope or its sandwich
        // variance, so there is no parameter uncertainty left out.
        parameter_uncertainty: ParameterUncertainty::Propagated,
    });
    Ok(CorrectedContinuousEffect {
        coefficients: fit.coefficients.clone(),
        naive_coefficients: fit.naive_coefficients.clone(),
        coefficient_index: 1,
        summary: ContinuousCorrectionSummary {
            method: ContinuousMethodKind::NoCorrectionNeeded,
            link: spec.link,
            location: spec.location,
            error_variance: None,
            error_variance_standard_error: None,
            reliability: None,
            bootstrap_replicates: 0,
            failed_bootstrap_replicates: 0,
            seed: None,
        },
        simex: None,
        replicate_components: None,
        bootstrap_estimates: Vec::new(),
        record,
    })
}

fn replicate_components(source: &Source) -> Option<VarianceComponents> {
    match source {
        Source::Replicates { components, .. } => Some(components.clone()),
        Source::Declared { .. } => None,
    }
}

fn linear_calibration(
    model: &MeasurementModel,
    cols: &Columns,
    spec: &ContinuousCorrectionSpec,
    source: &Source,
    floor: f64,
) -> Result<CorrectedContinuousEffect, MeasurementError> {
    let error = source.error()?;
    let fit = LinearMoments::accumulate(&cols.view(), &error, floor)
        .and_then(|moments| moments.solve())
        .map_err(|e| continuous_error(&e))?;
    let index = coefficient_index(spec.location);
    let d = fit.dim;
    let total = fit.covariance[index * d + index];
    let given = fit.covariance_given_error_variance[index * d + index];
    if !(total > 0.0 && given > 0.0) {
        return Err(stats_error(&MeasurementErrorError::NonPositiveVariance(
            "corrected treatment coefficient",
        )));
    }
    let record = build_record(RecordParts {
        model,
        functional_id: "linear_coefficient.regression_calibration",
        point: fit.coefficients[index],
        naive_point: Some(fit.naive_coefficients[index]),
        variance: Some(total),
        components: vec![
            component("sampling_given_measurement_parameters", given),
            component("measurement_parameter_uncertainty", total - given),
        ],
        rows: cols.n as u64,
        data_identity: data_identity(cols),
        parameter_uncertainty: source.uncertainty(),
    });
    Ok(CorrectedContinuousEffect {
        coefficients: fit.coefficients.clone(),
        naive_coefficients: fit.naive_coefficients.clone(),
        coefficient_index: index,
        summary: ContinuousCorrectionSummary {
            method: ContinuousMethodKind::RegressionCalibration,
            link: spec.link,
            location: spec.location,
            error_variance: Some(source.variance()),
            error_variance_standard_error: Some(source.estimate_variance().sqrt()),
            reliability: Some(fit.reliability),
            bootstrap_replicates: 0,
            failed_bootstrap_replicates: 0,
            seed: None,
        },
        simex: None,
        replicate_components: replicate_components(source),
        bootstrap_estimates: Vec::new(),
        record,
    })
}

/// Bytes the bootstrap holds at once: one resampled copy of every column and
/// the replicates, the outcome model's designs and workspace, and the
/// replicate estimates.
fn bootstrap_bytes(cols: &Columns, replicates: u32) -> Option<usize> {
    let n = cols.n;
    let d = cols.p.checked_add(2)?;
    let replicate_len = cols.replicate_values.as_ref().map_or(0, Vec::len);
    let columns = n.checked_mul(cols.p.checked_add(5)?)?;
    let designs = n.checked_mul(d)?.checked_mul(5)?;
    let cells = columns
        .checked_add(designs)?
        .checked_add(replicate_len)?
        .checked_add(replicates as usize)?;
    cells.checked_mul(size_of::<f64>())
}

/// Error variance of one bootstrap replicate.
fn redraw_error_variance(
    source: &Source,
    resampled: &Columns,
    rng: &mut CausalRng,
) -> Result<f64, MeasurementErrorError> {
    match source {
        Source::Declared { variance, standard_error } => {
            if *standard_error > 0.0 {
                let draw = variance + standard_error * standard_normal(rng);
                Ok(draw.max(ERROR_VARIANCE_DRAW_FLOOR * variance).max(0.0))
            } else {
                Ok(*variance)
            }
        }
        Source::Replicates { .. } => {
            let (Some(values), Some(counts)) = (&resampled.replicate_values, &resampled.counts)
            else {
                return Err(MeasurementErrorError::InsufficientReplicates(
                    "a resample lost its replicates",
                ));
            };
            replicate_variance_components(values, counts).map(|(c, _)| c.within_variance)
        }
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one pass: point estimate, bootstrap of the whole procedure, the variance split and the record, sharing the same columns and streams"
)]
fn bootstrapped(
    model: &MeasurementModel,
    cols: &Columns,
    spec: &ContinuousCorrectionSpec,
    source: &Source,
    floor: f64,
    ctx: &ExecutionContext,
) -> Result<CorrectedContinuousEffect, MeasurementError> {
    let requested = spec.bootstrap_replicates;
    if !(MIN_BOOTSTRAP_REPLICATES..=MAX_BOOTSTRAP_REPLICATES).contains(&requested) {
        return Err(invalid(format!(
            "{requested} bootstrap replicates requested; this correction needs between {MIN_BOOTSTRAP_REPLICATES} and {MAX_BOOTSTRAP_REPLICATES}"
        )));
    }
    let simex_config = spec.method.simex_config();
    if let Some(config) = &simex_config {
        config.validate().map_err(|e| stats_error(&e))?;
    }
    let family = spec.link.family();
    let index = coefficient_index(spec.location);
    let sigma2 = source.variance();
    // Held until the result is assembled: every buffer below is funded by it.
    let _lease = admit(ctx, bootstrap_bytes(cols, requested))?;

    let point_at = |error_variance: f64| -> Result<Correction, MeasurementErrorError> {
        // Every evaluation of the point opens the same stream: common random
        // numbers across the error-variance derivative.
        let mut rng = stream(spec.seed, POINT_FOLD, 0);
        run_correction(&cols.view(), family, error_variance, floor, simex_config.as_ref(), &mut rng)
    };
    let point = point_at(sigma2).map_err(|e| continuous_error(&e))?;

    let mut estimates = Vec::with_capacity(requested as usize);
    let mut failed = 0_u32;
    for b in 0..requested {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let mut rng = stream(spec.seed, BOOTSTRAP_FOLD, u64::from(b));
        let rows: Vec<usize> = (0..cols.n).map(|_| unbiased_index(&mut rng, cols.n)).collect();
        let resampled = cols.gather(&rows);
        let outcome = redraw_error_variance(source, &resampled, &mut rng).and_then(|draw| {
            run_correction(&resampled.view(), family, draw, floor, simex_config.as_ref(), &mut rng)
        });
        match outcome {
            Ok(correction) if correction.coefficients[index].is_finite() => {
                estimates.push(correction.coefficients[index]);
            }
            _ => failed += 1,
        }
    }
    let allowed = (f64::from(requested) * MAX_FAILED_FRACTION).floor();
    if f64::from(failed) > allowed || estimates.len() < 2 {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("measurement_not_identified"),
            "measurement_model.bootstrap_unstable",
            format!(
                "{failed} of {requested} bootstrap replicates of the correction failed (a reliability at or below the floor or an outcome model that does not fit in a resample); the procedure is too unstable on these data to report a variance"
            ),
        ));
    }
    let count = estimates.len() as f64;
    let mean = estimates.iter().sum::<f64>() / count;
    let total = estimates.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>() / (count - 1.0);
    if !(total.is_finite() && total > 0.0) {
        return Err(stats_error(&MeasurementErrorError::NonPositiveVariance(
            "bootstrap of the corrected treatment coefficient",
        )));
    }

    let parameter_variance = source.estimate_variance();
    let components = if parameter_variance > 0.0 {
        let step = (DERIVATIVE_STEP * sigma2).max(f64::EPSILON);
        let derivative = if sigma2 > step {
            point_at(sigma2 + step).and_then(|up| {
                point_at(sigma2 - step)
                    .map(|down| (up.coefficients[index] - down.coefficients[index]) / (2.0 * step))
            })
        } else {
            point_at(sigma2 + step)
                .map(|up| (up.coefficients[index] - point.coefficients[index]) / step)
        };
        match derivative {
            Ok(slope) if slope.is_finite() => {
                let part = (slope * slope * parameter_variance).min(total);
                vec![
                    component("sampling_given_measurement_parameters", total - part),
                    component("measurement_parameter_uncertainty", part),
                ]
            }
            _ => vec![component("bootstrap_including_measurement_parameters", total)],
        }
    } else {
        vec![
            component("sampling_given_measurement_parameters", total),
            component("measurement_parameter_uncertainty", 0.0),
        ]
    };
    let method = if simex_config.is_some() {
        ContinuousMethodKind::Simex
    } else {
        ContinuousMethodKind::RegressionCalibration
    };
    let functional_id = format!("{}.{}", spec.link.coefficient_name(), method.as_str());
    let record = build_record(RecordParts {
        model,
        functional_id: &functional_id,
        point: point.coefficients[index],
        naive_point: Some(point.naive[index]),
        variance: Some(total),
        components,
        rows: cols.n as u64,
        data_identity: data_identity(cols),
        parameter_uncertainty: source.uncertainty(),
    });
    Ok(CorrectedContinuousEffect {
        coefficients: point.coefficients,
        naive_coefficients: point.naive,
        coefficient_index: index,
        summary: ContinuousCorrectionSummary {
            method,
            link: spec.link,
            location: spec.location,
            error_variance: Some(sigma2),
            error_variance_standard_error: Some(parameter_variance.sqrt()),
            reliability: Some(point.reliability),
            bootstrap_replicates: u32::try_from(estimates.len()).unwrap_or(u32::MAX),
            failed_bootstrap_replicates: failed,
            seed: Some(spec.seed),
        },
        simex: point.simex,
        replicate_components: replicate_components(source),
        bootstrap_estimates: estimates,
        record,
    })
}
