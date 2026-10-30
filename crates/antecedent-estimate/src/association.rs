//! Descriptive association of an outcome with predictors: conditional laws, regression,
//! association measures and out-of-fold predictive models, with their uncertainty and the
//! declared rule that picks among them.
//!
//! Nothing here is causal. Every quantity describes the observed joint law of the columns, among
//! the units that were observed, with whatever confounding, selection and reverse direction that
//! law carries. The estimator reads the outcome scale and the data modality, applies one
//! declared selection rule ([`select_analysis`]) and reports:
//!
//! * a conditional mean (real or count outcome) or probability (0/1 outcome) in each stratum of
//!   the predictors, with a linearised standard error;
//! * regression coefficients of the predictors: linear (real outcome), logistic (0/1) or Poisson
//!   (count), with a sandwich standard error (heteroskedasticity-robust for rows,
//!   Newey-West for a series);
//! * the Pearson or distance correlation of the outcome with one predictor;
//! * the coefficients of a predictive model together with its out-of-fold predictions, which
//!   the caller scores with the predictive-assessment proper scores.
//!
//! Uncertainty is stated as what it is. Frequentist: the analytic standard error, replaced by the
//! standard deviation of resampling replicates when replicates are requested (a row bootstrap
//! for rows under independent sampling, a circular-block bootstrap for a dependent series), and
//! the percentile spread of those replicates. Bayesian: the closed-form conjugate
//! Normal-Inverse-Gamma posterior for a Gaussian outcome on independent, equally weighted rows,
//! and the Bayesian bootstrap (Dirichlet row weights, or Dirichlet weights on contiguous blocks
//! for a series) for every other analysis. No interval here is calibrated: no coverage record
//! measures any construction, so a spread or posterior interval carries the label
//! `unmeasured`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{
    AssociationFunctional, AssociationMeasure, AssociationOutcome, AssociationQuery,
    DeclaredSamplingDesign, ExecutionContext, StreamDomain, ValueType, VariableId, reason_code,
};
use antecedent_data::{
    CategoryDomain, ColumnView, OwnedColumnarStorage, TableView, fill_circular_block_indexes,
};
use antecedent_kernels::{
    bayesian_bootstrap_weights, sample_std, shuffled_fold_assignment, sigmoid, unbiased_index,
};
use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace, QuantileRule,
    SandwichKind, distance_correlation, equal_tail_interval_sorted, fit_glm_weighted,
    score_coefficient_covariance,
};

use crate::error::EstimationError;
use crate::gaussian_linear_posterior::{GaussianLinearPrior, conjugate_posterior_from_residuals};
use crate::temporal_block::{circular_fixed_b_scale, dependence_block_length};
use crate::util::SamplingDeclaration;

/// Identifier of the declared selection rule, recorded on every result.
pub const SELECTION_RULE: &str = "association_selection_rule";
/// Level of the percentile spread and of the credible interval.
pub const SPREAD_LEVEL: f64 = 0.95;
/// Strata a conditional law always reports; a larger sample admits `rows / 4` (an average of
/// four rows per stratum).
pub const MAX_STRATA: usize = 4096;
/// Distinct values of a numeric column crossed into strata as they are; a column with more is
/// coarsened into this many equal-count bins.
pub const MAX_STRATUM_LEVELS: usize = 64;
/// Fewest resampling replicates or posterior draws a spread or interval is formed from.
pub const MIN_REPLICATES: usize = 20;

/// Stream index of the row-bootstrap resamples.
const BOOTSTRAP_STREAM: u64 = 0xA550_C001;
/// Stream index of the Bayesian-bootstrap weight draws.
const POSTERIOR_STREAM: u64 = 0xA550_C002;
/// Stream index of the held-out fold assignment.
const FOLD_STREAM: u64 = 0xA550_C003;
/// Value tolerance when reading a column as an indicator or a count.
const CODE_TOLERANCE: f64 = 1e-12;
/// Smallest fitted probability or mean a score is evaluated at.
const SCORE_FLOOR: f64 = 1e-12;

/// The measurement scale of the outcome the rule read.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OutcomeKind {
    /// A real-valued outcome.
    Continuous,
    /// A 0/1 indicator.
    Binary,
    /// A non-negative integer count.
    Count,
}

impl OutcomeKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Continuous => "continuous",
            Self::Binary => "binary",
            Self::Count => "count",
        }
    }

    const fn family(self) -> GlmFamily {
        match self {
            Self::Continuous => GlmFamily::GaussianIdentity,
            Self::Binary => GlmFamily::BinomialLogit,
            Self::Count => GlmFamily::PoissonLog,
        }
    }
}

/// Whether the rows are independent units or the consecutive points of one series.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Modality {
    /// Rows of a table, sampled as independent units.
    CrossSection,
    /// Consecutive points of one dependent series.
    Series,
}

impl Modality {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CrossSection => "cross_section",
            Self::Series => "series",
        }
    }
}

/// The analysis the selection rule names.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AssociationAnalysis {
    /// Conditional mean of a real or count outcome in each stratum.
    StratifiedMean,
    /// Conditional probability of a 0/1 outcome in each stratum.
    StratifiedProbability,
    /// Linear regression.
    LinearRegression,
    /// Logistic regression.
    LogisticRegression,
    /// Poisson regression.
    PoissonRegression,
    /// Pearson correlation of the outcome with one predictor.
    PearsonCorrelation,
    /// Distance correlation of the outcome with one predictor.
    DistanceCorrelation,
    /// Linear predictive model scored out of fold.
    PredictiveLinear,
    /// Logistic predictive model scored out of fold.
    PredictiveLogistic,
    /// Poisson predictive model scored out of fold.
    PredictivePoisson,
}

impl AssociationAnalysis {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StratifiedMean => "stratified_mean",
            Self::StratifiedProbability => "stratified_probability",
            Self::LinearRegression => "linear_regression",
            Self::LogisticRegression => "logistic_regression",
            Self::PoissonRegression => "poisson_regression",
            Self::PearsonCorrelation => "pearson_correlation",
            Self::DistanceCorrelation => "distance_correlation",
            Self::PredictiveLinear => "predictive_linear",
            Self::PredictiveLogistic => "predictive_logistic",
            Self::PredictivePoisson => "predictive_poisson",
        }
    }

    /// Whether the analysis fits a generalised linear model.
    #[must_use]
    pub const fn fits_model(self) -> bool {
        matches!(
            self,
            Self::LinearRegression
                | Self::LogisticRegression
                | Self::PoissonRegression
                | Self::PredictiveLinear
                | Self::PredictiveLogistic
                | Self::PredictivePoisson
        )
    }

    /// Whether the analysis reports a conditional law by stratum.
    #[must_use]
    pub const fn is_stratified(self) -> bool {
        matches!(self, Self::StratifiedMean | Self::StratifiedProbability)
    }

    /// Whether the analysis scores a predictive model out of fold.
    #[must_use]
    pub const fn is_predictive(self) -> bool {
        matches!(self, Self::PredictiveLinear | Self::PredictiveLogistic | Self::PredictivePoisson)
    }

    const fn family(self) -> Option<GlmFamily> {
        match self {
            Self::LinearRegression | Self::PredictiveLinear => Some(GlmFamily::GaussianIdentity),
            Self::LogisticRegression | Self::PredictiveLogistic => Some(GlmFamily::BinomialLogit),
            Self::PoissonRegression | Self::PredictivePoisson => Some(GlmFamily::PoissonLog),
            _ => None,
        }
    }
}

/// What the declared selection rule decided, and why.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssociationSelection {
    /// Identifier of the rule ([`SELECTION_RULE`]).
    pub rule: &'static str,
    /// The functional the query requested (`auto` lets the rule pick).
    pub requested: &'static str,
    /// Outcome scale the rule read.
    pub outcome_kind: OutcomeKind,
    /// Where the outcome scale came from.
    pub outcome_source: &'static str,
    /// Data modality.
    pub modality: Modality,
    /// The analysis picked.
    pub analysis: AssociationAnalysis,
    /// Why the rule picked it.
    pub reason: &'static str,
    /// Held-out folds of a predictive analysis.
    pub folds: Option<u32>,
}

/// A reported quantity: a stratum value, a coefficient, a correlation.
#[derive(Clone, Debug, PartialEq)]
pub struct AssociationTerm {
    /// Name of the quantity.
    pub label: String,
    /// Scale of the estimate.
    pub scale: &'static str,
    /// The estimate on the observed rows.
    pub estimate: f64,
    /// Published standard error; `None` when the construction has none.
    pub standard_error: Option<f64>,
    /// Name of the standard-error construction.
    pub standard_error_method: Option<&'static str>,
    /// Percentile spread of the resampling replicates (uncalibrated).
    pub spread: Option<(f64, f64)>,
    /// Posterior summary.
    pub posterior: Option<TermPosterior>,
    /// Rows behind the quantity.
    pub rows: usize,
}

/// Posterior summary of one quantity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TermPosterior {
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub sd: f64,
    /// Lower end of the equal-tailed interval.
    pub lower: f64,
    /// Upper end of the equal-tailed interval.
    pub upper: f64,
}

/// Resampling behind the standard errors and spreads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResamplingRecord {
    /// `bootstrap` or `block_bootstrap`.
    pub method: &'static str,
    /// `row` or `calendar_block`.
    pub unit: &'static str,
    /// Replicates requested.
    pub replicates_requested: u32,
    /// Replicates every quantity of which was finite.
    pub replicates_ok: u32,
    /// Block length of a circular-block resample.
    pub block_length: Option<usize>,
}

/// Posterior behind the summaries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosteriorRecord {
    /// `conjugate_normal_inverse_gamma` or `bayesian_bootstrap`.
    pub method: &'static str,
    /// Draws behind the summaries; zero for a closed-form marginal.
    pub draws: usize,
}

/// One held-out case of a predictive model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldOutCase {
    /// Observed outcome.
    pub outcome: f64,
    /// Sampling weight of the case.
    pub weight: f64,
    /// Predicted mean (real outcome), probability (0/1 outcome) or rate (count).
    pub predicted: f64,
    /// Predictive standard deviation of a Gaussian outcome.
    pub predicted_sd: f64,
    /// The training-sample mean, probability or rate: the baseline forecast.
    pub baseline: f64,
    /// The training-sample standard deviation of a Gaussian outcome.
    pub baseline_sd: f64,
}

/// Out-of-fold predictions of a predictive model.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictionFit {
    /// Held-out folds.
    pub folds: u32,
    /// `random_assignment` or `consecutive_blocks`.
    pub fold_design: &'static str,
    /// The held-out cases, in row order.
    pub cases: Vec<HeldOutCase>,
}

/// A descriptive association and the evidence behind it.
#[derive(Clone, Debug, PartialEq)]
pub struct AssociationEstimate {
    /// What the selection rule decided.
    pub selection: AssociationSelection,
    /// The reported quantities.
    pub terms: Vec<AssociationTerm>,
    /// Complete rows behind the answer.
    pub rows: usize,
    /// Rows left out for a missing cell or weight.
    pub rows_dropped: usize,
    /// Resampling behind the standard errors and spreads.
    pub resampling: Option<ResamplingRecord>,
    /// Posterior behind the summaries.
    pub posterior: Option<PosteriorRecord>,
    /// Out-of-fold predictions.
    pub prediction: Option<PredictionFit>,
    /// The premises the description rests on and the readings it does not support.
    pub assumptions: Vec<&'static str>,
}

fn refuse_invalid(message: impl Into<String>) -> EstimationError {
    EstimationError::refused(reason_code!("invalid_argument"), message)
}

fn refuse_unsupported(message: impl Into<String>) -> EstimationError {
    EstimationError::refused(reason_code!("route_not_supported"), message)
}

/// `value` is exactly 0 or exactly 1.
fn is_indicator(value: f64) -> bool {
    let bits = (value + 0.0).to_bits();
    bits == 0.0_f64.to_bits() || bits == 1.0_f64.to_bits()
}

/// `value` is a non-negative integer within rounding.
fn is_count(value: f64) -> bool {
    value.is_finite() && value >= 0.0 && (value - value.round()).abs() <= CODE_TOLERANCE
}

fn is_discrete_type(value_type: &ValueType) -> bool {
    matches!(value_type, ValueType::Binary | ValueType::Categorical | ValueType::Ordinal)
}

fn schema_error(error: &impl std::fmt::Display) -> EstimationError {
    EstimationError::refused(reason_code!("unknown_variable"), error.to_string())
}

fn row_is_analysed(storage: &OwnedColumnarStorage, row: usize) -> bool {
    storage.analysis_mask().is_none_or(|mask| mask.is_valid(row))
}

/// Read the outcome scale and pick the analysis.
///
/// The rule is declared and data-independent except for one reading of the outcome column.
///
/// 1. *Outcome scale.* A declared scale wins. Otherwise a schema `binary` column is binary;
///    a column whose observed complete values are exactly 0 and 1 (both present) is binary;
///    a `count` or unspecified column of non-negative integers is a count; every other real
///    column is continuous. A categorical, ordinal or vector outcome is refused.
/// 2. *Functional.* An explicit functional is honoured: a conditional law is a stratified mean
///    (real or count outcome) or probability (0/1); coefficients are linear, logistic or Poisson
///    regression by the outcome scale; a measure is the Pearson or distance correlation; a
///    predictive model is the linear, logistic or Poisson model by the outcome scale. `auto`
///    picks the conditional law when there is no predictor or every predictor and conditioning
///    column is discrete in the schema, and the regression of the outcome scale otherwise.
/// 3. *Modality.* The analysis does not change with the modality; the standard error and the
///    resampling do (Newey-West and circular blocks for a series).
///
/// # Errors
///
/// `invalid_argument` for a column the data lacks, an outcome the rule cannot read, or a
/// functional that does not fit the columns (a categorical predictor of a correlation, a
/// numeric column with too many values for strata).
pub fn select_analysis(
    storage: &OwnedColumnarStorage,
    query: &AssociationQuery,
    modality: Modality,
) -> Result<AssociationSelection, EstimationError> {
    query.validate().map_err(|error| refuse_invalid(error.to_string()))?;
    let schema = storage.schema();
    for id in query.columns() {
        schema.get(id).map_err(|error| schema_error(&error))?;
    }
    let (outcome_kind, outcome_source) = read_outcome_scale(storage, query)?;
    let discrete = |ids: &[VariableId]| -> Result<bool, EstimationError> {
        let mut all = true;
        for id in ids {
            all &= is_discrete_type(&schema.get(*id).map_err(|e| schema_error(&e))?.value_type);
        }
        Ok(all)
    };
    let all_discrete = discrete(&query.predictors)? && discrete(&query.conditioning)?;
    let strata = if outcome_kind == OutcomeKind::Binary {
        AssociationAnalysis::StratifiedProbability
    } else {
        AssociationAnalysis::StratifiedMean
    };
    let regression = match outcome_kind {
        OutcomeKind::Continuous => AssociationAnalysis::LinearRegression,
        OutcomeKind::Binary => AssociationAnalysis::LogisticRegression,
        OutcomeKind::Count => AssociationAnalysis::PoissonRegression,
    };
    let predictive = match outcome_kind {
        OutcomeKind::Continuous => AssociationAnalysis::PredictiveLinear,
        OutcomeKind::Binary => AssociationAnalysis::PredictiveLogistic,
        OutcomeKind::Count => AssociationAnalysis::PredictivePoisson,
    };
    let (analysis, reason, folds) = match query.functional {
        AssociationFunctional::Auto if query.predictors.is_empty() => {
            (strata, "no predictor: the marginal law of the outcome", None)
        }
        AssociationFunctional::Auto if all_discrete => (
            strata,
            "every predictor and conditioning column is discrete: the law in each stratum",
            None,
        ),
        AssociationFunctional::Auto => {
            (regression, "a numeric predictor: the regression of the outcome scale", None)
        }
        AssociationFunctional::Conditional => (strata, "a conditional law was requested", None),
        AssociationFunctional::Coefficients => {
            (regression, "regression coefficients were requested", None)
        }
        AssociationFunctional::Measure(AssociationMeasure::Pearson) => {
            (AssociationAnalysis::PearsonCorrelation, "a Pearson correlation was requested", None)
        }
        AssociationFunctional::Measure(AssociationMeasure::Distance) => {
            (AssociationAnalysis::DistanceCorrelation, "a distance correlation was requested", None)
        }
        AssociationFunctional::Predictive { folds } => {
            (predictive, "a predictive model was requested", Some(folds))
        }
    };
    if matches!(
        analysis,
        AssociationAnalysis::PearsonCorrelation | AssociationAnalysis::DistanceCorrelation
    ) {
        let predictor = query.predictors[0];
        let spec = schema.get(predictor).map_err(|e| schema_error(&e))?;
        if matches!(spec.value_type, ValueType::Categorical | ValueType::Ordinal) {
            return Err(refuse_invalid(format!(
                "the predictor `{}` is categorical: a correlation needs a numeric predictor. \
                 Next attempt: ask for the conditional law, or code the predictor numerically",
                spec.name
            )));
        }
    }
    Ok(AssociationSelection {
        rule: SELECTION_RULE,
        requested: query.functional.as_str(),
        outcome_kind,
        outcome_source,
        modality,
        analysis,
        reason,
        folds,
    })
}

fn read_outcome_scale(
    storage: &OwnedColumnarStorage,
    query: &AssociationQuery,
) -> Result<(OutcomeKind, &'static str), EstimationError> {
    match query.outcome_scale {
        AssociationOutcome::Continuous => return Ok((OutcomeKind::Continuous, "declared")),
        AssociationOutcome::Binary => return Ok((OutcomeKind::Binary, "declared")),
        AssociationOutcome::Count => return Ok((OutcomeKind::Count, "declared")),
        AssociationOutcome::Schema => {}
    }
    let spec = storage.schema().get(query.outcome).map_err(|e| schema_error(&e))?;
    match spec.value_type {
        ValueType::Binary => return Ok((OutcomeKind::Binary, "schema_binary")),
        ValueType::Categorical | ValueType::Ordinal | ValueType::Vector { .. } => {
            return Err(refuse_invalid(format!(
                "the outcome `{}` is categorical or vector valued: an association needs a real, \
                 0/1 or count outcome. Next attempt: code one level as a 0/1 indicator column",
                spec.name
            )));
        }
        ValueType::Continuous | ValueType::Count | ValueType::Unspecified => {}
    }
    let values = storage.float64_values(query.outcome).map_err(EstimationError::from)?;
    let observed: Vec<f64> = values
        .iter()
        .enumerate()
        .filter(|(row, value)| value.is_finite() && row_is_analysed(storage, *row))
        .map(|(_, value)| *value)
        .collect();
    let has_zero = observed.iter().any(|v| is_indicator(*v) && v.abs() < 0.5);
    let has_one = observed.iter().any(|v| is_indicator(*v) && v.abs() > 0.5);
    if observed.iter().all(|v| is_indicator(*v)) && has_zero && has_one {
        return Ok((OutcomeKind::Binary, "observed_zero_one"));
    }
    if matches!(spec.value_type, ValueType::Count | ValueType::Unspecified)
        && !observed.is_empty()
        && observed.iter().all(|v| is_count(*v))
    {
        return Ok((OutcomeKind::Count, "observed_count"));
    }
    Ok((OutcomeKind::Continuous, "default_continuous"))
}

/// The values of one predictor or conditioning column over the complete rows.
#[derive(Clone, Debug)]
enum ColumnValues {
    Numeric(Vec<f64>),
    Levels { codes: Vec<u32>, labels: Vec<String>, reference: Option<u32> },
}

#[derive(Clone, Debug)]
struct PreparedColumn {
    name: String,
    values: ColumnValues,
}

/// A model design: column-major matrix, its names and the columns reported.
#[derive(Clone, Debug)]
struct Design {
    matrix: Vec<f64>,
    ncols: usize,
    names: Vec<String>,
    report: Vec<usize>,
}

/// The strata of a conditional law.
#[derive(Clone, Debug)]
struct Strata {
    labels: Vec<String>,
    of_row: Vec<usize>,
    counts: Vec<usize>,
}

/// The complete rows of the columns an association reads, with its design.
#[derive(Clone, Debug)]
pub struct PreparedAssociation {
    selection: AssociationSelection,
    rows: usize,
    rows_dropped: usize,
    outcome_name: String,
    y: Vec<f64>,
    weight: Vec<f64>,
    unit_weights: bool,
    predictors: Vec<PreparedColumn>,
    conditioning: Vec<PreparedColumn>,
    design: Option<Design>,
    strata: Option<Strata>,
}

impl PreparedAssociation {
    /// What the selection rule decided.
    #[must_use]
    pub const fn selection(&self) -> &AssociationSelection {
        &self.selection
    }

    /// Complete rows behind the answer.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Rows left out for a missing cell or weight.
    #[must_use]
    pub const fn rows_dropped(&self) -> usize {
        self.rows_dropped
    }
}

struct RawColumn {
    name: String,
    categorical: bool,
    numeric: Vec<f64>,
    codes: Vec<u32>,
    labels: Vec<String>,
    reference: Option<u32>,
}

fn level_labels(domain: &CategoryDomain) -> Vec<String> {
    domain.levels.iter().map(|level| level.label.to_string()).collect()
}

fn read_raw_column(
    storage: &OwnedColumnarStorage,
    id: VariableId,
) -> Result<RawColumn, EstimationError> {
    let name = storage.schema().get(id).map_err(|e| schema_error(&e))?.name.to_string();
    if let ColumnView::Categorical(column) = storage.column(id).map_err(EstimationError::from)? {
        let rows = column.len();
        let mut codes = Vec::with_capacity(rows);
        for row in 0..rows {
            codes.push(if column.validity.is_valid(row) {
                column.codes[row].raw()
            } else {
                u32::MAX
            });
        }
        return Ok(RawColumn {
            name,
            categorical: true,
            numeric: Vec::new(),
            codes,
            labels: level_labels(&column.domain),
            reference: column.domain.reference.map(antecedent_data::CategoryCode::raw),
        });
    }
    let numeric = storage.float64_values(id).map_err(EstimationError::from)?;
    Ok(RawColumn {
        name,
        categorical: false,
        numeric,
        codes: Vec::new(),
        labels: Vec::new(),
        reference: None,
    })
}

fn raw_is_valid(raw: &RawColumn, row: usize) -> bool {
    if raw.categorical { raw.codes[row] != u32::MAX } else { raw.numeric[row].is_finite() }
}

fn gather_column(raw: &RawColumn, kept: &[usize]) -> PreparedColumn {
    let values = if raw.categorical {
        ColumnValues::Levels {
            codes: kept.iter().map(|&row| raw.codes[row]).collect(),
            labels: raw.labels.clone(),
            reference: raw.reference,
        }
    } else {
        ColumnValues::Numeric(kept.iter().map(|&row| raw.numeric[row]).collect())
    };
    PreparedColumn { name: raw.name.clone(), values }
}

/// Reads a sampling-weight column over every row; `None` rows are invalid.
fn read_weights(
    storage: &OwnedColumnarStorage,
    id: VariableId,
) -> Result<Vec<Option<f64>>, EstimationError> {
    let values = storage.float64_values(id).map_err(EstimationError::from)?;
    let mut weights = Vec::with_capacity(values.len());
    for value in values {
        if value.is_nan() {
            weights.push(None);
        } else if value.is_finite() && value >= 0.0 {
            weights.push(Some(value));
        } else {
            return Err(EstimationError::refused(
                reason_code!("sampling_weights_invalid"),
                "the association weights must be finite and non-negative",
            ));
        }
    }
    Ok(weights)
}

/// Distinct sorted values of a numeric column as strata levels.
fn numeric_levels(_name: &str, values: &[f64]) -> Result<(Vec<u32>, Vec<String>), EstimationError> {
    let mut distinct: Vec<f64> = values.to_vec();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup_by(|a, b| a.to_bits() == b.to_bits());
    if distinct.len() > MAX_STRATUM_LEVELS {
        // Strata of that many levels are mostly empty: coarsen into equal-count bins whose
        // labels name their closed value range.
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let total = sorted.len();
        let edges: Vec<usize> =
            (0..=MAX_STRATUM_LEVELS).map(|bin| (bin * total) / MAX_STRATUM_LEVELS).collect();
        let mut upper: Vec<f64> = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        for bin in 0..MAX_STRATUM_LEVELS {
            let (lo, hi) = (edges[bin], edges[bin + 1]);
            if lo >= hi {
                continue;
            }
            let top = sorted[hi - 1];
            // Equal values never straddle a bin boundary: skip a bin the previous one covers.
            if upper.last().is_some_and(|last| top <= *last) {
                continue;
            }
            labels.push(format!("[{}, {top}]", sorted[lo]));
            upper.push(top);
        }
        let codes = values
            .iter()
            .map(|value| {
                let position = upper.partition_point(|top| top < value);
                u32::try_from(position.min(upper.len() - 1)).unwrap_or(u32::MAX)
            })
            .collect();
        return Ok((codes, labels));
    }
    let codes = values
        .iter()
        .map(|value| {
            let position = distinct.partition_point(|candidate| candidate < value);
            u32::try_from(position).unwrap_or(u32::MAX)
        })
        .collect();
    Ok((codes, distinct.iter().map(|v| format!("{v}")).collect()))
}

fn expand_column(column: &PreparedColumn, rows: usize, into: &mut Vec<(String, Vec<f64>)>) {
    match &column.values {
        ColumnValues::Numeric(values) => into.push((column.name.clone(), values.clone())),
        ColumnValues::Levels { codes, labels, reference } => {
            let mut present: Vec<u32> = codes.clone();
            present.sort_unstable();
            present.dedup();
            // The level the domain declares as its reference is the omitted one when the rows
            // hold it; otherwise the lowest level present is.
            let omitted =
                reference.filter(|r| present.contains(r)).or_else(|| present.first().copied());
            for &level in present.iter().filter(|&&level| Some(level) != omitted) {
                let label =
                    labels.get(level as usize).map_or_else(|| level.to_string(), Clone::clone);
                let indicator =
                    (0..rows).map(|row| if codes[row] == level { 1.0 } else { 0.0 }).collect();
                into.push((format!("{}={label}", column.name), indicator));
            }
        }
    }
}

fn build_design(
    rows: usize,
    predictors: &[PreparedColumn],
    conditioning: &[PreparedColumn],
) -> Design {
    let mut columns: Vec<(String, Vec<f64>)> = vec![("intercept".to_owned(), vec![1.0; rows])];
    for column in predictors {
        expand_column(column, rows, &mut columns);
    }
    let reported = columns.len();
    for column in conditioning {
        expand_column(column, rows, &mut columns);
    }
    let ncols = columns.len();
    let mut matrix = Vec::with_capacity(rows * ncols);
    let mut names = Vec::with_capacity(ncols);
    for (name, values) in columns {
        names.push(name);
        matrix.extend(values);
    }
    Design { matrix, ncols, names, report: (0..reported).collect() }
}

fn build_strata(
    rows: usize,
    predictors: &[PreparedColumn],
    conditioning: &[PreparedColumn],
) -> Result<Strata, EstimationError> {
    let mut per_variable: Vec<(String, Vec<u32>, Vec<String>)> = Vec::new();
    for column in predictors.iter().chain(conditioning) {
        match &column.values {
            ColumnValues::Numeric(values) => {
                let (codes, labels) = numeric_levels(&column.name, values)?;
                per_variable.push((column.name.clone(), codes, labels));
            }
            ColumnValues::Levels { codes, labels, .. } => {
                per_variable.push((column.name.clone(), codes.clone(), labels.clone()));
            }
        }
    }
    let mut index: BTreeMap<Vec<u32>, usize> = BTreeMap::new();
    let mut keys: Vec<Vec<u32>> = Vec::new();
    for row in 0..rows {
        let key: Vec<u32> = per_variable.iter().map(|(_, codes, _)| codes[row]).collect();
        if !index.contains_key(&key) {
            index.insert(key.clone(), 0);
            keys.push(key);
        }
    }
    if keys.len() > MAX_STRATA.max(rows / 4) {
        return Err(refuse_invalid(format!(
            "the predictors cross into {} strata (an average of fewer than four rows each). \
             Next attempt: ask for the regression coefficients, or coarsen the columns",
            keys.len()
        )));
    }
    keys.sort();
    for (position, key) in keys.iter().enumerate() {
        index.insert(key.clone(), position);
    }
    let mut of_row = Vec::with_capacity(rows);
    let mut counts = vec![0_usize; keys.len()];
    for row in 0..rows {
        let key: Vec<u32> = per_variable.iter().map(|(_, codes, _)| codes[row]).collect();
        let position = index[&key];
        of_row.push(position);
        counts[position] += 1;
    }
    let labels = keys
        .iter()
        .map(|key| {
            if key.is_empty() {
                return "all_rows".to_owned();
            }
            per_variable
                .iter()
                .zip(key)
                .map(|((name, _, labels), &code)| {
                    let label =
                        labels.get(code as usize).map_or_else(|| code.to_string(), Clone::clone);
                    format!("{name}={label}")
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    Ok(Strata { labels, of_row, counts })
}

/// Bayesian settings of an association.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BayesianSettings {
    /// Posterior draws of the Bayesian bootstrap.
    pub draws: usize,
    /// Scale of the weakly informative coefficient prior, in standard deviations of the outcome
    /// per standard deviation of the predictor.
    pub prior_scale: f64,
}

/// Descriptive association estimator.
#[derive(Clone, Debug)]
pub struct AssociationEstimator {
    /// Resampling replicates for the frequentist spread (0 = analytic standard errors only).
    pub bootstrap_replicates: u32,
    /// The sampling design the caller declared; a design the data cannot honour is refused.
    pub sampling_design: SamplingDeclaration,
    /// Rows of a table or the points of one series.
    pub modality: Modality,
    /// Bayesian settings; `None` is the frequentist reading.
    pub bayesian: Option<BayesianSettings>,
}

/// A fitted model on the rows of positive weight.
struct WeightedFit {
    coefficients: Vec<f64>,
    kept: Vec<usize>,
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    family: GlmFamily,
}

struct TermSpec {
    label: String,
    scale: &'static str,
    rows: usize,
}

/// Draws of the statistic vector, one entry per replicate, a quantity that did not fit `NaN`.
struct ReplicateDraws {
    draws: Vec<Vec<f64>>,
    block_length: Option<usize>,
}

impl AssociationEstimator {
    /// A frequentist estimator with analytic standard errors and no declared design.
    #[must_use]
    pub const fn new(modality: Modality) -> Self {
        Self {
            bootstrap_replicates: 0,
            sampling_design: SamplingDeclaration::Undeclared,
            modality,
            bayesian: None,
        }
    }

    crate::inference::with_bootstrap_replicates!();

    /// Declare the sampling design of the rows.
    #[must_use]
    pub const fn with_sampling_design(mut self, design: SamplingDeclaration) -> Self {
        self.sampling_design = design;
        self
    }

    /// Report a posterior instead of a standard error.
    #[must_use]
    pub const fn with_bayesian(mut self, settings: BayesianSettings) -> Self {
        self.bayesian = Some(settings);
        self
    }

    /// Refuse a declared design the data cannot honour.
    ///
    /// # Errors
    ///
    /// `route_not_supported` naming the design to declare: independent subjects for a table,
    /// a dependent time series for a series.
    pub fn check_design(&self) -> Result<(), EstimationError> {
        use DeclaredSamplingDesign as Design;
        match (self.modality, self.sampling_design) {
            (_, SamplingDeclaration::Undeclared)
            | (
                Modality::CrossSection,
                SamplingDeclaration::Declared(Design::IndependentSubjectTrajectories),
            )
            | (Modality::Series, SamplingDeclaration::Declared(Design::DependentTimeSeries)) => {
                Ok(())
            }
            (Modality::CrossSection, SamplingDeclaration::Declared(_)) => Err(refuse_unsupported(
                "the association table carries one row per unit and no cluster or series labels, \
                 so a row-level draw is allowed only under the independent-subject design. Next \
                 attempt: declare independent subjects, or pass a series for a dependent design",
            )),
            (Modality::Series, SamplingDeclaration::Declared(_)) => Err(refuse_unsupported(
                "the points of one series are serially dependent, so a draw must resample \
                 contiguous blocks under the dependent-time-series design. Next attempt: declare \
                 a dependent time series, or pass a table of independent units",
            )),
        }
    }

    /// Read the columns of `query` from `storage` and apply the selection rule.
    ///
    /// # Errors
    ///
    /// A column that is missing, an outcome the rule cannot read or that violates its scale (a
    /// 0/1 outcome holding another value, a count that is negative or fractional), too few
    /// complete rows, a design the data cannot honour, or an analysis the weights or the row count
    /// do not allow.
    pub fn prepare(
        &self,
        storage: &OwnedColumnarStorage,
        query: &AssociationQuery,
    ) -> Result<PreparedAssociation, EstimationError> {
        let selection = select_analysis(storage, query, self.modality)?;
        let rows_total = storage.row_count();
        let outcome = storage.float64_values(query.outcome).map_err(EstimationError::from)?;
        let weights = query.weight.map(|id| read_weights(storage, id)).transpose()?;
        let raw_predictors = query
            .predictors
            .iter()
            .map(|id| read_raw_column(storage, *id))
            .collect::<Result<Vec<_>, _>>()?;
        let raw_conditioning = query
            .conditioning
            .iter()
            .map(|id| read_raw_column(storage, *id))
            .collect::<Result<Vec<_>, _>>()?;
        let mut kept = Vec::new();
        for row in 0..rows_total {
            let complete = row_is_analysed(storage, row)
                && outcome[row].is_finite()
                && weights.as_ref().is_none_or(|w| w[row].is_some())
                && raw_predictors.iter().chain(&raw_conditioning).all(|c| raw_is_valid(c, row));
            if complete {
                kept.push(row);
            }
        }
        if kept.is_empty() {
            return Err(refuse_invalid(
                "no complete row holds the outcome, the predictors, the conditioning columns and \
                 the weight",
            ));
        }
        let y: Vec<f64> = kept.iter().map(|&row| outcome[row]).collect();
        check_outcome_values(&y, selection.outcome_kind)?;
        let weight: Vec<f64> = match &weights {
            Some(column) => kept.iter().map(|&row| column[row].unwrap_or(0.0)).collect(),
            None => vec![1.0; kept.len()],
        };
        if weight.iter().all(|w| *w <= 0.0) {
            return Err(EstimationError::refused(
                reason_code!("sampling_weights_invalid"),
                "every complete row has zero weight",
            ));
        }
        let unit_weights = query.weight.is_none();
        let predictors: Vec<PreparedColumn> =
            raw_predictors.iter().map(|raw| gather_column(raw, &kept)).collect();
        let conditioning: Vec<PreparedColumn> =
            raw_conditioning.iter().map(|raw| gather_column(raw, &kept)).collect();
        let rows = kept.len();
        let outcome_name =
            storage.schema().get(query.outcome).map_err(|e| schema_error(&e))?.name.to_string();
        let mut prepared = PreparedAssociation {
            selection,
            rows,
            rows_dropped: rows_total - rows,
            outcome_name,
            y,
            weight,
            unit_weights,
            predictors,
            conditioning,
            design: None,
            strata: None,
        };
        let analysis = prepared.selection.analysis;
        if analysis.is_stratified() {
            prepared.strata =
                Some(build_strata(rows, &prepared.predictors, &prepared.conditioning)?);
        } else if analysis.fits_model() {
            let design = build_design(rows, &prepared.predictors, &prepared.conditioning);
            if rows <= design.ncols {
                return Err(refuse_invalid(format!(
                    "the model has {} coefficients but only {rows} complete rows",
                    design.ncols
                )));
            }
            prepared.design = Some(design);
        } else if rows < 3 {
            return Err(refuse_invalid("a correlation needs at least three complete rows"));
        }
        if analysis == AssociationAnalysis::DistanceCorrelation {
            if query.weight.is_some() {
                return Err(refuse_unsupported(
                    "the distance correlation has no weighted form. Next attempt: omit the weight \
                     column, or ask for the Pearson correlation",
                ));
            }
            if self.bayesian.is_some() {
                return Err(refuse_unsupported(
                    "the Bayesian bootstrap reweights rows and the distance correlation has no \
                     weighted form. Next attempt: use frequentist inference, or ask for the \
                     Pearson correlation",
                ));
            }
        }
        Ok(prepared)
    }
}

fn check_outcome_values(y: &[f64], kind: OutcomeKind) -> Result<(), EstimationError> {
    match kind {
        OutcomeKind::Continuous => Ok(()),
        OutcomeKind::Binary => {
            if y.iter().all(|v| is_indicator(*v)) {
                Ok(())
            } else {
                Err(refuse_invalid(
                    "a binary outcome must be coded exactly 0 and 1. Next attempt: recode the \
                     column, or declare the outcome scale continuous",
                ))
            }
        }
        OutcomeKind::Count => {
            if y.iter().all(|v| is_count(*v)) {
                Ok(())
            } else {
                Err(refuse_invalid(
                    "a count outcome must be a non-negative integer. Next attempt: declare the \
                     outcome scale continuous",
                ))
            }
        }
    }
}

fn weighted_sum(values: &[f64], weights: &[f64]) -> f64 {
    values.iter().zip(weights).map(|(v, w)| v * w).sum()
}

fn weighted_correlation(x: &[f64], y: &[f64], w: &[f64]) -> f64 {
    let total: f64 = w.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return f64::NAN;
    }
    let mx = weighted_sum(x, w) / total;
    let my = weighted_sum(y, w) / total;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for ((xi, yi), wi) in x.iter().zip(y).zip(w) {
        let (dx, dy) = (xi - mx, yi - my);
        sxy += wi * dx * dy;
        sxx += wi * dx * dx;
        syy += wi * dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 { f64::NAN } else { sxy / (sxx * syy).sqrt() }
}

/// Bartlett-window Newey-West variance of the sum of an influence series.
fn newey_west_variance(psi: &[f64], lag: usize) -> f64 {
    let mut total: f64 = psi.iter().map(|v| v * v).sum();
    for l in 1..=lag.min(psi.len().saturating_sub(1)) {
        let weight = 1.0 - l as f64 / (lag as f64 + 1.0);
        let cross: f64 = psi[l..].iter().zip(psi).map(|(a, b)| a * b).sum();
        total += 2.0 * weight * cross;
    }
    total.max(0.0)
}

/// The Newey-West truncation lag `floor(4 (n / 100)^(2/9))`, at least 1 and below `n`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the floor of a small positive rule value, clamped to the row count"
)]
fn newey_west_lag(rows: usize) -> usize {
    let rule = (4.0 * (rows as f64 / 100.0).powf(2.0 / 9.0)).floor().max(1.0) as usize;
    rule.min(rows.saturating_sub(1)).max(1)
}

impl AssociationEstimator {
    fn layout(prepared: &PreparedAssociation) -> Vec<TermSpec> {
        let analysis = prepared.selection.analysis;
        if let Some(strata) = &prepared.strata {
            let scale = if analysis == AssociationAnalysis::StratifiedProbability {
                "probability"
            } else {
                "outcome_mean"
            };
            return strata
                .labels
                .iter()
                .zip(&strata.counts)
                .map(|(label, &rows)| TermSpec { label: label.clone(), scale, rows })
                .collect();
        }
        if let Some(design) = &prepared.design {
            let scale = match analysis.family() {
                Some(GlmFamily::BinomialLogit) => "log_odds",
                Some(GlmFamily::PoissonLog) => "log_rate",
                _ => "outcome_units",
            };
            return design
                .report
                .iter()
                .map(|&c| TermSpec { label: design.names[c].clone(), scale, rows: prepared.rows })
                .collect();
        }
        let predictor = prepared.predictors.first().map_or("predictor", |c| c.name.as_str());
        let (name, scale) = if analysis == AssociationAnalysis::DistanceCorrelation {
            ("distance_correlation", "distance_correlation")
        } else {
            ("pearson_correlation", "correlation")
        };
        vec![TermSpec {
            label: format!("{name}({}, {predictor})", prepared.outcome_name),
            scale,
            rows: prepared.rows,
        }]
    }

    fn numeric_predictor(prepared: &PreparedAssociation) -> Result<&[f64], EstimationError> {
        match prepared.predictors.first().map(|c| &c.values) {
            Some(ColumnValues::Numeric(values)) => Ok(values),
            _ => Err(refuse_invalid("a correlation needs a numeric predictor")),
        }
    }

    /// The reported statistic vector under the row multipliers `mult`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a multiplier is a small non-negative resample count rounded to an integer"
    )]
    fn statistics(
        prepared: &PreparedAssociation,
        mult: &[f64],
    ) -> Result<Vec<f64>, EstimationError> {
        let analysis = prepared.selection.analysis;
        let effective: Vec<f64> = prepared.weight.iter().zip(mult).map(|(w, m)| w * m).collect();
        if let Some(strata) = &prepared.strata {
            let mut numerator = vec![0.0; strata.labels.len()];
            let mut denominator = vec![0.0; strata.labels.len()];
            for (row, &weight) in effective.iter().enumerate() {
                let k = strata.of_row[row];
                numerator[k] += weight * prepared.y[row];
                denominator[k] += weight;
            }
            return Ok(numerator
                .iter()
                .zip(&denominator)
                .map(|(n, d)| if *d > 0.0 { n / d } else { f64::NAN })
                .collect());
        }
        if analysis.fits_model() {
            let fit = Self::fit_with_weights(prepared, &effective)?;
            let design = prepared.design.as_ref().ok_or_else(|| refuse_invalid("no design"))?;
            return Ok(design.report.iter().map(|&c| fit.coefficients[c]).collect());
        }
        let x = Self::numeric_predictor(prepared)?;
        if analysis == AssociationAnalysis::PearsonCorrelation {
            return Ok(vec![weighted_correlation(x, &prepared.y, &effective)]);
        }
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for row in 0..prepared.rows {
            let copies = mult[row].round();
            if copies >= 1.0 {
                for _ in 0..copies as usize {
                    xs.push(x[row]);
                    ys.push(prepared.y[row]);
                }
            }
        }
        // The distance correlation centres its inputs itself and is undefined (`NaN`, refused by
        // the caller) for a series without spread, as the Pearson correlation is.
        Ok(vec![distance_correlation(&xs, &ys)])
    }

    /// Fit the model on the rows of positive effective weight.
    fn fit_with_weights(
        prepared: &PreparedAssociation,
        effective: &[f64],
    ) -> Result<WeightedFit, EstimationError> {
        let design = prepared.design.as_ref().ok_or_else(|| refuse_invalid("no design"))?;
        let family = prepared
            .selection
            .analysis
            .family()
            .ok_or_else(|| refuse_invalid("the analysis fits no model"))?;
        let kept: Vec<usize> = (0..prepared.rows).filter(|&row| effective[row] > 0.0).collect();
        let (n, p) = (kept.len(), design.ncols);
        if n <= p {
            return Err(EstimationError::refused(
                reason_code!("design_rank_deficient"),
                format!("the model has {p} coefficients but {n} rows of positive weight"),
            ));
        }
        let mut x = Vec::with_capacity(n * p);
        for column in 0..p {
            let source = &design.matrix[column * prepared.rows..(column + 1) * prepared.rows];
            x.extend(kept.iter().map(|&row| source[row]));
        }
        let y: Vec<f64> = kept.iter().map(|&row| prepared.y[row]).collect();
        let w: Vec<f64> = kept.iter().map(|&row| effective[row]).collect();
        let mut workspace = LeastSquaresWorkspace::default();
        let fit = fit_glm_weighted(
            family,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: p, y: &y },
            &w,
            &FaerBackend,
            &mut workspace,
            &GlmOptions::default().without_separation_ridge(),
        )
        .map_err(|error| {
            EstimationError::refused(
                reason_code!("design_rank_deficient"),
                format!(
                    "the model could not be fitted: {error}. Next attempt: drop a collinear \
                     predictor or a level with no rows"
                ),
            )
        })?;
        fit.require_ok().map_err(|error| {
            EstimationError::refused(
                reason_code!("design_rank_deficient"),
                format!(
                    "the fit is degenerate: {error}. Next attempt: drop the predictor that \
                     separates the outcome, or ask for the conditional law"
                ),
            )
        })?;
        Ok(WeightedFit { coefficients: fit.coefficients, kept, x, y, w, family })
    }

    /// Sandwich covariance of a fitted model.
    fn sandwich(&self, fit: &WeightedFit, ncols: usize) -> Result<Vec<f64>, EstimationError> {
        let n = fit.kept.len();
        let mut score = vec![0.0; n];
        let mut information = vec![0.0; n];
        for row in 0..n {
            let eta: f64 = (0..ncols).map(|c| fit.x[c * n + row] * fit.coefficients[c]).sum();
            let mean = fit.family.mean_from_eta(eta);
            let (u, w) = match fit.family {
                GlmFamily::BinomialLogit => {
                    let mean = mean.clamp(1e-9, 1.0 - 1e-9);
                    (fit.y[row] - mean, (mean * (1.0 - mean)).max(1e-12))
                }
                GlmFamily::PoissonLog => {
                    let mean = mean.max(1e-12);
                    (fit.y[row] - mean, mean)
                }
                _ => (fit.y[row] - mean, 1.0),
            };
            score[row] = u * fit.w[row];
            information[row] = w * fit.w[row];
        }
        let kind = match self.modality {
            Modality::CrossSection => SandwichKind::Hc1,
            Modality::Series => SandwichKind::NeweyWest { lag: newey_west_lag(n) },
        };
        Ok(score_coefficient_covariance(&fit.x, n, ncols, &score, &information, kind)?)
    }

    /// Analytic standard errors of the reported quantities and the construction's name.
    fn analytic(
        &self,
        prepared: &PreparedAssociation,
    ) -> Result<(Vec<Option<f64>>, Option<&'static str>), EstimationError> {
        let analysis = prepared.selection.analysis;
        if let Some(strata) = &prepared.strata {
            let means = Self::statistics(prepared, &vec![1.0; prepared.rows])?;
            let series = self.modality == Modality::Series;
            let mut errors = Vec::with_capacity(means.len());
            for (k, mean) in means.iter().enumerate() {
                let total: f64 = (0..prepared.rows)
                    .filter(|&r| strata.of_row[r] == k)
                    .map(|r| prepared.weight[r])
                    .sum();
                // Rows of zero weight are not part of the sample the mean describes: the
                // regression routes drop them, so the degrees-of-freedom divisor counts only
                // the rows that carry weight.
                let count = (0..prepared.rows)
                    .filter(|&r| strata.of_row[r] == k && prepared.weight[r] > 0.0)
                    .count();
                if count < 2 || total.is_nan() || total <= 0.0 {
                    errors.push(None);
                    continue;
                }
                let psi: Vec<f64> = (0..prepared.rows)
                    .map(|r| {
                        if strata.of_row[r] == k {
                            prepared.weight[r] * (prepared.y[r] - mean) / total
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let variance = if series {
                    newey_west_variance(&psi, newey_west_lag(prepared.rows))
                } else {
                    let raw: f64 = psi.iter().map(|v| v * v).sum();
                    raw * count as f64 / (count as f64 - 1.0)
                };
                errors.push(Some(variance.max(0.0).sqrt()));
            }
            let method = if series { "newey_west_influence" } else { "linearised_influence" };
            return Ok((errors, Some(method)));
        }
        if analysis.fits_model() {
            let design = prepared.design.as_ref().ok_or_else(|| refuse_invalid("no design"))?;
            let fit = Self::fit_with_weights(prepared, &prepared.weight)?;
            let covariance = self.sandwich(&fit, design.ncols)?;
            let errors = design
                .report
                .iter()
                .map(|&c| Some(covariance[c * design.ncols + c].max(0.0).sqrt()))
                .collect();
            let method = match self.modality {
                Modality::CrossSection => "sandwich_hc1",
                Modality::Series => "sandwich_newey_west",
            };
            return Ok((errors, Some(method)));
        }
        if analysis == AssociationAnalysis::PearsonCorrelation
            && self.modality == Modality::CrossSection
        {
            let r = Self::statistics(prepared, &vec![1.0; prepared.rows])?[0];
            let total: f64 = prepared.weight.iter().sum();
            let squares: f64 = prepared.weight.iter().map(|w| w * w).sum();
            let effective_rows = total * total / squares;
            let se = ((1.0 - r * r) / (effective_rows - 1.0).max(1.0).sqrt()).abs();
            return Ok((vec![Some(se)], Some("normal_theory_delta")));
        }
        Ok((vec![None; Self::layout(prepared).len()], None))
    }
}

fn cancelled() -> EstimationError {
    EstimationError::refused(
        reason_code!("cancelled_no_claim"),
        "the association was cancelled before every replicate ran; no spread is reported and the \
         stop is not a verdict on the data",
    )
}

fn finite_column(draws: &[Vec<f64>], k: usize) -> Vec<f64> {
    draws.iter().map(|draw| draw[k]).filter(|value| value.is_finite()).collect()
}

fn percentile_spread(mut values: Vec<f64>) -> Option<(f64, f64)> {
    if values.len() < MIN_REPLICATES {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(equal_tail_interval_sorted(&values, SPREAD_LEVEL, QuantileRule::Interpolated))
}

impl AssociationEstimator {
    /// Draws of the statistic vector under row or circular-block resamples.
    fn bootstrap_draws(
        &self,
        prepared: &PreparedAssociation,
        ctx: &ExecutionContext,
    ) -> Result<ReplicateDraws, EstimationError> {
        let rows = prepared.rows;
        let replicates = self.bootstrap_replicates as usize;
        let terms = Self::layout(prepared).len();
        let _admitted = crate::util::admit_array::<f64>(ctx, replicates, terms.max(1))?;
        let block_length = (self.modality == Modality::Series).then(|| {
            let mean = prepared.y.iter().sum::<f64>() / rows as f64;
            let centred: Vec<f64> = prepared.y.iter().map(|v| v - mean).collect();
            dependence_block_length(1, rows, &[centred.as_slice()]).clamp(1, rows)
        });
        let mut draws = Vec::with_capacity(replicates);
        let mut index = vec![0_usize; rows];
        for replicate in 0..replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            if let Some(length) = block_length {
                let mut rng = ctx.rng.stream_for(
                    StreamDomain::TemporalBlock,
                    antecedent_core::derive_stream(BOOTSTRAP_STREAM, replicate as u64),
                );
                fill_circular_block_indexes(rows, length, &mut rng, &mut index)
                    .map_err(EstimationError::from)?;
            } else {
                let mut rng = ctx.rng.stream_for(
                    StreamDomain::Resample,
                    antecedent_core::derive_stream(BOOTSTRAP_STREAM, replicate as u64),
                );
                for slot in &mut index {
                    *slot = unbiased_index(&mut rng, rows);
                }
            }
            let mut multiplier = vec![0.0_f64; rows];
            for &source in &index {
                multiplier[source] += 1.0;
            }
            draws.push(
                Self::statistics(prepared, &multiplier).unwrap_or_else(|_| vec![f64::NAN; terms]),
            );
        }
        Ok(ReplicateDraws { draws, block_length })
    }

    /// Draws of the statistic vector under Dirichlet row (or block) weights.
    fn posterior_draws(
        &self,
        prepared: &PreparedAssociation,
        settings: BayesianSettings,
        ctx: &ExecutionContext,
    ) -> Result<ReplicateDraws, EstimationError> {
        if settings.draws < MIN_REPLICATES {
            return Err(EstimationError::refused(
                reason_code!("posterior_draws_below_supported"),
                "the Bayesian bootstrap of an association needs at least 20 draws to form an \
                 interval. Next attempt: raise n_draws",
            ));
        }
        let rows = prepared.rows;
        let terms = Self::layout(prepared).len();
        let _admitted = crate::util::admit_array::<f64>(ctx, settings.draws, terms.max(1))?;
        let block_length = (self.modality == Modality::Series).then(|| {
            let mean = prepared.y.iter().sum::<f64>() / rows as f64;
            let centred: Vec<f64> = prepared.y.iter().map(|v| v - mean).collect();
            dependence_block_length(1, rows, &[centred.as_slice()]).clamp(1, rows)
        });
        let mut draws = Vec::with_capacity(settings.draws);
        for draw in 0..settings.draws {
            if ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            let mut rng = ctx.rng.stream_for(
                StreamDomain::Bayesian,
                antecedent_core::derive_stream(POSTERIOR_STREAM, draw as u64),
            );
            let multiplier = match block_length {
                None => bayesian_bootstrap_weights(&mut rng, rows, false),
                Some(length) => {
                    let blocks = rows.div_ceil(length);
                    let block_weights = bayesian_bootstrap_weights(&mut rng, blocks, false);
                    (0..rows).map(|row| block_weights[row / length]).collect()
                }
            };
            draws.push(
                Self::statistics(prepared, &multiplier).unwrap_or_else(|_| vec![f64::NAN; terms]),
            );
        }
        Ok(ReplicateDraws { draws, block_length })
    }

    /// Whether the closed-form conjugate posterior applies: a Gaussian outcome on independent
    /// rows with no sampling-weight column.
    fn conjugate_applies(prepared: &PreparedAssociation) -> bool {
        bayesian_method(&prepared.selection, !prepared.unit_weights)
            == "conjugate_normal_inverse_gamma"
    }

    /// Closed-form Normal-Inverse-Gamma marginals of the reported quantities.
    fn conjugate(
        prepared: &PreparedAssociation,
        settings: BayesianSettings,
    ) -> Result<Vec<TermPosterior>, EstimationError> {
        let rows = prepared.rows as f64;
        let mean_y = prepared.y.iter().sum::<f64>() / rows;
        let variance_y = prepared.y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / rows;
        if !(variance_y.is_finite() && variance_y > 0.0) {
            return Err(refuse_invalid(
                "the outcome is constant, so its spread has no scale to put a prior on",
            ));
        }
        let scale_squared = settings.prior_scale * settings.prior_scale;
        let summarise = |posterior: &crate::gaussian_linear_posterior::GaussianLinearPosterior,
                         weights: &[f64]|
         -> Result<TermPosterior, EstimationError> {
            let marginal = posterior
                .linear_functional(weights)
                .ok_or_else(|| refuse_invalid("the posterior has no marginal for a quantity"))?;
            let summary = marginal.summary(SPREAD_LEVEL);
            Ok(TermPosterior {
                mean: summary.mean,
                sd: summary.standard_deviation,
                lower: summary.interval[0],
                upper: summary.interval[1],
            })
        };
        if let Some(strata) = &prepared.strata {
            let strata_count = strata.labels.len();
            let (mut count, mut sum) = (vec![0.0; strata_count], vec![0.0; strata_count]);
            let mut members: Vec<Vec<usize>> = vec![Vec::new(); strata_count];
            for row in 0..prepared.rows {
                let k = strata.of_row[row];
                count[k] += 1.0;
                sum[k] += prepared.y[row];
                members[k].push(row);
            }
            let prior = GaussianLinearPrior {
                coefficient_precision: vec![1.0 / scale_squared],
                variance_shape: 1.0,
                variance_scale: variance_y,
            };
            let mut terms = Vec::with_capacity(strata_count);
            for k in 0..strata_count {
                // The scale is formed from the residual about the posterior mean, not from
                // the sum of squares, which cancels for an outcome far from zero.
                let posterior = conjugate_posterior_from_residuals(
                    &[count[k]],
                    &[sum[k]],
                    strata.counts[k],
                    &prior,
                    |mean| members[k].iter().map(|&row| (prepared.y[row] - mean[0]).powi(2)).sum(),
                )?;
                terms.push(summarise(&posterior, &[1.0])?);
            }
            return Ok(terms);
        }
        let design = prepared.design.as_ref().ok_or_else(|| refuse_invalid("no design"))?;
        let (p, n) = (design.ncols, prepared.rows);
        let column = |c: usize| &design.matrix[c * n..(c + 1) * n];
        let mut gram = vec![0.0; p * p];
        let mut xty = vec![0.0; p];
        for a in 0..p {
            xty[a] = weighted_sum(column(a), &prepared.y);
            for b in a..p {
                let value: f64 = column(a).iter().zip(column(b)).map(|(u, v)| u * v).sum();
                gram[a * p + b] = value;
                gram[b * p + a] = value;
            }
        }
        let mut precision = vec![1.0 / scale_squared; p];
        for (a, slot) in precision.iter_mut().enumerate().skip(1) {
            let mean = column(a).iter().sum::<f64>() / rows;
            let variance = column(a).iter().map(|v| (v - mean).powi(2)).sum::<f64>() / rows;
            if !(variance.is_finite() && variance > 0.0) {
                return Err(EstimationError::refused(
                    reason_code!("design_rank_deficient"),
                    format!("the column `{}` is constant", design.names[a]),
                ));
            }
            *slot = variance / scale_squared;
        }
        let prior = GaussianLinearPrior {
            coefficient_precision: precision,
            variance_shape: 1.0,
            variance_scale: variance_y,
        };
        let posterior = conjugate_posterior_from_residuals(&gram, &xty, n, &prior, |mean| {
            (0..n)
                .map(|row| {
                    let fitted: f64 = (0..p).map(|c| column(c)[row] * mean[c]).sum();
                    (prepared.y[row] - fitted).powi(2)
                })
                .sum()
        })?;
        design
            .report
            .iter()
            .map(|&c| {
                let mut weights = vec![0.0; p];
                weights[c] = 1.0;
                summarise(&posterior, &weights)
            })
            .collect()
    }

    /// Out-of-fold predictions of a predictive analysis.
    fn out_of_fold(
        &self,
        prepared: &PreparedAssociation,
        ctx: &ExecutionContext,
    ) -> Result<PredictionFit, EstimationError> {
        let folds = prepared.selection.folds.unwrap_or(5) as usize;
        let rows = prepared.rows;
        if rows < 2 * folds {
            return Err(refuse_invalid(format!(
                "{rows} complete rows cannot fill {folds} held-out folds of two cases. Next \
                 attempt: ask for fewer folds"
            )));
        }
        let (assignment, fold_design) = match self.modality {
            Modality::CrossSection => {
                let mut rng = ctx.rng.stream_for(StreamDomain::Estimate, FOLD_STREAM);
                let assignment = shuffled_fold_assignment(&mut rng, rows, folds, None)
                    .map_err(|error| refuse_invalid(error.to_string()))?;
                (assignment, "random_assignment")
            }
            Modality::Series => {
                ((0..rows).map(|row| row * folds / rows).collect(), "consecutive_blocks")
            }
        };
        let design = prepared.design.as_ref().ok_or_else(|| refuse_invalid("no design"))?;
        let family = prepared.selection.outcome_kind.family();
        let mut cases: Vec<Option<HeldOutCase>> = vec![None; rows];
        for fold in 0..folds {
            if ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            let training: Vec<f64> = (0..rows)
                .map(|row| if assignment[row] == fold { 0.0 } else { prepared.weight[row] })
                .collect();
            let fit = Self::fit_with_weights(prepared, &training)?;
            let total: f64 = fit.w.iter().sum();
            let count = fit.kept.len() as f64;
            let baseline = weighted_sum(&fit.y, &fit.w) / total;
            let residual: f64 = (0..fit.kept.len())
                .map(|i| {
                    let eta: f64 = (0..design.ncols)
                        .map(|c| fit.x[c * fit.kept.len() + i] * fit.coefficients[c])
                        .sum();
                    fit.w[i] * (fit.y[i] - family.mean_from_eta(eta)).powi(2)
                })
                .sum();
            let predictive_sd = (residual / (total * (1.0 - design.ncols as f64 / count)))
                .max(0.0)
                .sqrt()
                .max(SCORE_FLOOR);
            let baseline_sd =
                (fit.y.iter().zip(&fit.w).map(|(y, w)| w * (y - baseline).powi(2)).sum::<f64>()
                    / total)
                    .max(0.0)
                    .sqrt()
                    .max(SCORE_FLOOR);
            for row in (0..rows).filter(|&row| assignment[row] == fold) {
                let eta: f64 = (0..design.ncols)
                    .map(|c| design.matrix[c * rows + row] * fit.coefficients[c])
                    .sum();
                cases[row] = Some(HeldOutCase {
                    outcome: prepared.y[row],
                    weight: prepared.weight[row],
                    predicted: match fit.family {
                        GlmFamily::BinomialLogit => {
                            sigmoid(eta).clamp(SCORE_FLOOR, 1.0 - SCORE_FLOOR)
                        }
                        GlmFamily::PoissonLog => eta.exp().max(SCORE_FLOOR),
                        _ => eta,
                    },
                    predicted_sd: predictive_sd,
                    baseline: match fit.family {
                        GlmFamily::BinomialLogit => baseline.clamp(SCORE_FLOOR, 1.0 - SCORE_FLOOR),
                        GlmFamily::PoissonLog => baseline.max(SCORE_FLOOR),
                        _ => baseline,
                    },
                    baseline_sd,
                });
            }
        }
        Ok(PredictionFit {
            folds: prepared.selection.folds.unwrap_or(5),
            fold_design,
            cases: cases.into_iter().flatten().collect(),
        })
    }

    /// The description of the prepared rows: the point, its uncertainty, and the premises.
    ///
    /// # Errors
    ///
    /// A declared design the data cannot honour, a Bayesian request that also asks for
    /// replicates, a fit that is degenerate (a collinear design, a separated outcome), a
    /// constant column of a correlation, too few draws, or cancellation.
    pub fn estimate(
        &self,
        prepared: &PreparedAssociation,
        ctx: &ExecutionContext,
    ) -> Result<AssociationEstimate, EstimationError> {
        self.check_design()?;
        if self.bayesian.is_some() && self.bootstrap_replicates != 0 {
            return Err(EstimationError::refused(
                reason_code!("option_not_applicable"),
                "the Bayesian association draws its posterior with n_draws and reads no bootstrap \
                 replicates. Next attempt: omit bootstrap_replicates, or use frequentist inference \
                 for a resampling spread",
            ));
        }
        let rows = prepared.rows;
        let point = Self::statistics(prepared, &vec![1.0; rows])?;
        if point.iter().any(|value| !value.is_finite()) {
            return Err(refuse_invalid(
                "a reported quantity is not defined on these rows (a constant column has no \
                 correlation). Next attempt: drop the constant column",
            ));
        }
        let layout = Self::layout(prepared);
        let (analytic, analytic_method) = self.analytic(prepared)?;
        let mut terms: Vec<AssociationTerm> = layout
            .into_iter()
            .enumerate()
            .map(|(k, spec)| AssociationTerm {
                label: spec.label,
                scale: spec.scale,
                estimate: point[k],
                standard_error: analytic[k],
                standard_error_method: analytic[k].and(analytic_method),
                spread: None,
                posterior: None,
                rows: spec.rows,
            })
            .collect();
        let mut resampling = None;
        let mut posterior = None;
        match self.bayesian {
            None if self.bootstrap_replicates > 0 => {
                resampling = Some(self.publish_bootstrap(prepared, &mut terms, ctx)?);
            }
            None => {}
            Some(settings) if Self::conjugate_applies(prepared) => {
                let marginals = Self::conjugate(prepared, settings)?;
                for (term, marginal) in terms.iter_mut().zip(marginals) {
                    term.posterior = Some(marginal);
                }
                posterior =
                    Some(PosteriorRecord { method: "conjugate_normal_inverse_gamma", draws: 0 });
            }
            Some(settings) => {
                let drawn = self.posterior_draws(prepared, settings, ctx)?;
                for (k, term) in terms.iter_mut().enumerate() {
                    let mut values = finite_column(&drawn.draws, k);
                    if values.len() < MIN_REPLICATES {
                        return Err(EstimationError::refused(
                            reason_code!("posterior_draws_below_supported"),
                            format!(
                                "only {} of {} Bayesian-bootstrap draws fit for `{}`; the \
                                 posterior is withheld. Next attempt: drop the predictor that \
                                 separates the outcome, or raise n_draws",
                                values.len(),
                                settings.draws,
                                term.label
                            ),
                        ));
                    }
                    values.sort_by(f64::total_cmp);
                    let (lower, upper) = equal_tail_interval_sorted(
                        &values,
                        SPREAD_LEVEL,
                        QuantileRule::Interpolated,
                    );
                    term.posterior = Some(TermPosterior {
                        mean: values.iter().sum::<f64>() / values.len() as f64,
                        sd: sample_std(&values),
                        lower,
                        upper,
                    });
                }
                posterior =
                    Some(PosteriorRecord { method: "bayesian_bootstrap", draws: settings.draws });
            }
        }
        let prediction = if prepared.selection.analysis.is_predictive() {
            Some(self.out_of_fold(prepared, ctx)?)
        } else {
            None
        };
        Ok(AssociationEstimate {
            selection: prepared.selection.clone(),
            terms,
            rows,
            rows_dropped: prepared.rows_dropped,
            resampling,
            posterior,
            prediction,
            assumptions: descriptive_assumptions(&prepared.selection),
        })
    }

    /// Publish the frequentist bootstrap summaries and their construction record.
    fn publish_bootstrap(
        &self,
        prepared: &PreparedAssociation,
        terms: &mut [AssociationTerm],
        ctx: &ExecutionContext,
    ) -> Result<ResamplingRecord, EstimationError> {
        let rows = prepared.rows;
        let drawn = self.bootstrap_draws(prepared, ctx)?;
        let series = self.modality == Modality::Series;
        let scale = drawn.block_length.map_or(1.0, |length| circular_fixed_b_scale(length, rows));
        let attempted = drawn.draws.len();
        let mut ok = attempted;
        for (k, term) in terms.iter_mut().enumerate() {
            let values = finite_column(&drawn.draws, k);
            ok = ok.min(values.len());
            if values.len() >= 2 && values.len() * 2 >= attempted {
                term.standard_error = Some(sample_std(&values) * scale);
                term.standard_error_method =
                    Some(if series { "bootstrap_circular_block" } else { "bootstrap_row" });
            }
            term.spread = percentile_spread(values);
        }
        Ok(ResamplingRecord {
            method: if series { "block_bootstrap" } else { "bootstrap" },
            unit: if series { "calendar_block" } else { "row" },
            replicates_requested: self.bootstrap_replicates,
            replicates_ok: u32::try_from(ok).unwrap_or(u32::MAX),
            block_length: drawn.block_length,
        })
    }
}

/// The posterior construction a Bayesian reading of `selection` uses.
///
/// The closed-form Normal-Inverse-Gamma marginals apply to a Gaussian outcome on independent
/// rows with no sampling-weight column: conditional means by stratum, linear regression and the
/// linear predictive model. Every other analysis, a weighted study and a dependent series use the
/// Bayesian bootstrap (Dirichlet weights on rows, or on contiguous blocks of a series).
#[must_use]
pub fn bayesian_method(selection: &AssociationSelection, weighted: bool) -> &'static str {
    let gaussian = selection.outcome_kind == OutcomeKind::Continuous
        && matches!(
            selection.analysis,
            AssociationAnalysis::StratifiedMean
                | AssociationAnalysis::LinearRegression
                | AssociationAnalysis::PredictiveLinear
        );
    if gaussian && !weighted && selection.modality == Modality::CrossSection {
        "conjugate_normal_inverse_gamma"
    } else {
        "bayesian_bootstrap"
    }
}

/// The premises a description rests on and the readings it does not support.
#[must_use]
pub fn descriptive_assumptions(selection: &AssociationSelection) -> Vec<&'static str> {
    let mut premises = vec![
        "descriptive_only: the quantities describe the observed joint law of the columns among \
         the units observed; they are not the effect of changing any variable and rank no action",
        "no_confounding_adjustment: conditioning columns are held fixed arithmetically; omitted \
         common causes, selection into the sample and the direction of dependence are not ruled out",
    ];
    premises.push(match selection.modality {
        Modality::CrossSection => {
            "independent_units: the rows are independent draws from the population described, or \
             are weighted to it"
        }
        Modality::Series => {
            "stationary_dependent_series: one series with short-range dependence, resampled in \
             circular blocks; trend and regime change are not modelled"
        }
    });
    premises.push(match selection.analysis {
        AssociationAnalysis::StratifiedMean | AssociationAnalysis::StratifiedProbability => {
            "strata_are_levels_observed: each stratum is described by the rows observed in it; an \
             unobserved stratum has no value"
        }
        AssociationAnalysis::PearsonCorrelation | AssociationAnalysis::DistanceCorrelation => {
            "association_measure: a correlation summarises dependence between two columns and says \
             nothing about its direction"
        }
        AssociationAnalysis::PredictiveLinear
        | AssociationAnalysis::PredictiveLogistic
        | AssociationAnalysis::PredictivePoisson => {
            "held_out_scores: the scores describe the prediction of held-out rows by models \
             fitted on the other rows; they are not evidence of a causal relation"
        }
        AssociationAnalysis::LinearRegression
        | AssociationAnalysis::LogisticRegression
        | AssociationAnalysis::PoissonRegression => {
            "declared_form: the coefficients describe the declared linear, log-odds or log-rate \
             form; under another true form they describe its best approximation in that family"
        }
    });
    premises.push(
        "uncalibrated_uncertainty: no coverage record measures the standard errors, spreads or \
         posterior intervals reported here",
    );
    premises
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "test fixtures compare exact constants and index with small literals"
)]
mod tests {
    use antecedent_data::{TabularData, TimeSeriesData};

    use super::*;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn ctx(seed: u64) -> ExecutionContext {
        ExecutionContext::for_tests(seed)
    }

    /// Deterministic pseudo-noise in `[-0.5, 0.5)`.
    fn noise(i: usize) -> f64 {
        ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5
    }

    fn table(columns: &[(&str, Vec<f64>)]) -> TabularData {
        TabularData::from_f64_columns(columns.iter().map(|(name, values)| (*name, &values[..])))
            .unwrap()
    }

    fn simple_regression_data() -> (TabularData, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..60).map(|i| (i as f64) / 6.0).collect();
        let y: Vec<f64> = x.iter().enumerate().map(|(i, x)| 1.0 + 2.0 * x + noise(i)).collect();
        (table(&[("x", x.clone()), ("y", y.clone())]), x, y)
    }

    #[test]
    fn resampling_posterior_and_fold_streams_are_distinct_and_repeatable() {
        let factory = &ctx(21).rng;
        let sample = |domain, key| factory.stream_for(domain, key).next_u64();
        let fold = sample(StreamDomain::Estimate, FOLD_STREAM);
        let boot =
            |i| sample(StreamDomain::Resample, antecedent_core::derive_stream(BOOTSTRAP_STREAM, i));
        let posterior =
            |i| sample(StreamDomain::Bayesian, antecedent_core::derive_stream(POSTERIOR_STREAM, i));
        assert_ne!(boot(2), fold);
        assert_ne!(boot(3), posterior(0));
        assert_eq!(boot(2), boot(2));
        assert_eq!(posterior(0), posterior(0));
    }

    #[test]
    fn linear_coefficients_and_the_robust_error_match_the_closed_form() {
        let (data, x, y) = simple_regression_data();
        let query = AssociationQuery::new(v(1), [v(0)]);
        let estimator = AssociationEstimator::new(Modality::CrossSection);
        let prepared = estimator.prepare(data.storage(), &query).unwrap();
        assert_eq!(prepared.selection().analysis, AssociationAnalysis::LinearRegression);
        assert_eq!(prepared.selection().outcome_source, "default_continuous");
        let estimate = estimator.estimate(&prepared, &ctx(1)).unwrap();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let slope = sxy / sxx;
        let intercept = my - slope * mx;
        assert_eq!(estimate.terms.len(), 2);
        assert!((estimate.terms[0].estimate - intercept).abs() < 1e-8);
        assert!((estimate.terms[1].estimate - slope).abs() < 1e-8);
        // HC1 variance of the slope of a simple regression.
        let meat: f64 = x
            .iter()
            .zip(&y)
            .map(|(a, b)| {
                let e = b - intercept - slope * a;
                (a - mx).powi(2) * e * e
            })
            .sum();
        let expected = (n / (n - 2.0) * meat).sqrt() / sxx;
        let published = estimate.terms[1].standard_error.unwrap();
        assert!((published - expected).abs() < 1e-8, "{published} vs {expected}");
        assert_eq!(estimate.terms[1].standard_error_method, Some("sandwich_hc1"));
        assert!(estimate.resampling.is_none() && estimate.posterior.is_none());
        assert!(estimate.terms.iter().all(|t| t.spread.is_none()));
        assert!(estimate.assumptions.iter().any(|a| a.starts_with("descriptive_only")));
    }

    #[test]
    fn conditional_means_are_group_means_with_the_sample_standard_error() {
        let g: Vec<f64> = (0..40).map(|i| f64::from(i % 2 == 0)).collect();
        let y: Vec<f64> = g.iter().enumerate().map(|(i, g)| 3.0 * g + noise(i)).collect();
        let data = table(&[("g", g.clone()), ("y", y.clone())]);
        let query =
            AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::Conditional);
        let estimator = AssociationEstimator::new(Modality::CrossSection);
        let prepared = estimator.prepare(data.storage(), &query).unwrap();
        assert_eq!(prepared.selection().analysis, AssociationAnalysis::StratifiedMean);
        let estimate = estimator.estimate(&prepared, &ctx(2)).unwrap();
        assert_eq!(estimate.terms.len(), 2);
        for term in &estimate.terms {
            let level = if term.label == "g=1" { 1.0 } else { 0.0 };
            let rows: Vec<f64> =
                g.iter().zip(&y).filter(|(a, _)| **a == level).map(|(_, b)| *b).collect();
            let n = rows.len() as f64;
            let mean = rows.iter().sum::<f64>() / n;
            let variance = rows.iter().map(|b| (b - mean).powi(2)).sum::<f64>() / (n - 1.0);
            assert!((term.estimate - mean).abs() < 1e-10, "{}", term.label);
            let se = term.standard_error.unwrap();
            assert!((se - (variance / n).sqrt()).abs() < 1e-10, "{se}");
            assert_eq!(term.scale, "outcome_mean");
        }
    }

    #[test]
    fn conditional_mean_single_positive_weight_has_no_analytic_standard_error() {
        let data = table(&[
            ("g", vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
            ("y", vec![1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ]);
        let query =
            AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::Conditional);
        let estimator = AssociationEstimator::new(Modality::CrossSection);
        let mut prepared = estimator.prepare(data.storage(), &query).unwrap();
        prepared.weight = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        prepared.unit_weights = false;
        let (errors, _) = estimator.analytic(&prepared).unwrap();
        assert!(errors[0].is_some_and(|se| se > 0.0));
        assert_eq!(errors[1], None);
    }

    #[test]
    fn a_binary_outcome_gets_conditional_probabilities_and_logistic_log_odds_of_a_two_by_two() {
        // 2x2 table: P(y=1 | x=0) = 20/100, P(y=1 | x=1) = 60/100.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (level, ones) in [(0.0, 20), (1.0, 60)] {
            for i in 0..100 {
                x.push(level);
                y.push(f64::from(i < ones));
            }
        }
        let data = table(&[("x", x), ("y", y)]);
        let estimator = AssociationEstimator::new(Modality::CrossSection);
        let strata =
            AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::Conditional);
        let prepared = estimator.prepare(data.storage(), &strata).unwrap();
        assert_eq!(prepared.selection().outcome_kind, OutcomeKind::Binary);
        assert_eq!(prepared.selection().analysis, AssociationAnalysis::StratifiedProbability);
        let estimate = estimator.estimate(&prepared, &ctx(3)).unwrap();
        assert!((estimate.terms[0].estimate - 0.2).abs() < 1e-12);
        assert!((estimate.terms[1].estimate - 0.6).abs() < 1e-12);
        let se0 = estimate.terms[0].standard_error.unwrap();
        assert!((se0 - (0.2_f64 * 0.8 / 99.0).sqrt()).abs() < 1e-12, "{se0}");
        let logistic = AssociationQuery::new(v(1), [v(0)]);
        let prepared = estimator.prepare(data.storage(), &logistic).unwrap();
        assert_eq!(prepared.selection().analysis, AssociationAnalysis::LogisticRegression);
        let fit = estimator.estimate(&prepared, &ctx(3)).unwrap();
        let log_odds = |p: f64| (p / (1.0 - p)).ln();
        assert!((fit.terms[0].estimate - log_odds(0.2)).abs() < 1e-6);
        assert!((fit.terms[1].estimate - (log_odds(0.6) - log_odds(0.2))).abs() < 1e-6);
        assert_eq!(fit.terms[1].scale, "log_odds");
    }

    #[test]
    fn the_selection_rule_is_deterministic_and_reads_the_outcome_scale() {
        let counts: Vec<f64> = (0..50).map(|i| f64::from((i * 7) % 5)).collect();
        let reals: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.37 + noise(i as usize)).collect();
        let flags: Vec<f64> = (0..50).map(|i| f64::from(i % 3 == 0)).collect();
        let data = table(&[("x", reals.clone()), ("c", counts), ("b", flags), ("r", reals)]);
        let storage = data.storage();
        let pick = |outcome: u32, functional: AssociationFunctional| {
            let query = AssociationQuery::new(v(outcome), [v(0)]).with_functional(functional);
            select_analysis(storage, &query, Modality::CrossSection).unwrap().analysis
        };
        // Observed 0/1 is binary, a non-negative-integer float column typed continuous stays
        // continuous, and the same query twice picks the same analysis.
        assert_eq!(pick(2, AssociationFunctional::Auto), AssociationAnalysis::LogisticRegression);
        assert_eq!(pick(1, AssociationFunctional::Auto), AssociationAnalysis::LinearRegression);
        assert_eq!(pick(3, AssociationFunctional::Auto), AssociationAnalysis::LinearRegression);
        assert_eq!(pick(3, AssociationFunctional::Auto), pick(3, AssociationFunctional::Auto));
        assert_eq!(
            pick(2, AssociationFunctional::Predictive { folds: 5 }),
            AssociationAnalysis::PredictiveLogistic
        );
        assert_eq!(
            pick(3, AssociationFunctional::Measure(AssociationMeasure::Distance)),
            AssociationAnalysis::DistanceCorrelation
        );
        // A declared scale wins over the values.
        let declared =
            AssociationQuery::new(v(1), [v(0)]).with_outcome_scale(AssociationOutcome::Count);
        let selection = select_analysis(storage, &declared, Modality::CrossSection).unwrap();
        assert_eq!(selection.analysis, AssociationAnalysis::PoissonRegression);
        assert_eq!(selection.outcome_source, "declared");
        // No predictor is the marginal law; the selection records its reason.
        let marginal = AssociationQuery::new(v(3), Vec::<VariableId>::new());
        let selection = select_analysis(storage, &marginal, Modality::CrossSection).unwrap();
        assert_eq!(selection.analysis, AssociationAnalysis::StratifiedMean);
        assert!(selection.reason.starts_with("no predictor"));
        assert_eq!(selection.rule, SELECTION_RULE);
    }

    #[test]
    fn a_count_outcome_fits_a_poisson_model() {
        let x: Vec<f64> = (0..80).map(|i| f64::from(i % 4)).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, x)| ((0.3 + 0.4 * x).exp() + noise(i) * 2.0).round().max(0.0))
            .collect();
        let data = table(&[("x", x), ("y", y)]);
        let query =
            AssociationQuery::new(v(1), [v(0)]).with_outcome_scale(AssociationOutcome::Count);
        let estimator = AssociationEstimator::new(Modality::CrossSection);
        let prepared = estimator.prepare(data.storage(), &query).unwrap();
        assert_eq!(prepared.selection().analysis, AssociationAnalysis::PoissonRegression);
        let estimate = estimator.estimate(&prepared, &ctx(4)).unwrap();
        assert_eq!(estimate.terms[1].scale, "log_rate");
        assert!((estimate.terms[1].estimate - 0.4).abs() < 0.3, "{}", estimate.terms[1].estimate);
    }

    #[test]
    fn correlations_match_the_closed_form_and_distance_sees_nonlinear_dependence() {
        let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 10.0 - 3.95).collect();
        let y: Vec<f64> = x.iter().map(|a| a * a).collect();
        let data = table(&[("x", x.clone()), ("y", y.clone())]);
        let estimator = AssociationEstimator::new(Modality::CrossSection);
        let pearson = AssociationQuery::new(v(1), [v(0)])
            .with_functional(AssociationFunctional::Measure(AssociationMeasure::Pearson));
        let prepared = estimator.prepare(data.storage(), &pearson).unwrap();
        let r = estimator.estimate(&prepared, &ctx(5)).unwrap().terms[0].estimate;
        assert!(r.abs() < 1e-9, "a symmetric quadratic has no linear association: {r}");
        let distance = AssociationQuery::new(v(1), [v(0)])
            .with_functional(AssociationFunctional::Measure(AssociationMeasure::Distance));
        let prepared = estimator.prepare(data.storage(), &distance).unwrap();
        let estimate = estimator.estimate(&prepared, &ctx(5)).unwrap();
        assert!(estimate.terms[0].estimate > 0.4, "{}", estimate.terms[0].estimate);
        assert!(estimate.terms[0].standard_error.is_none());
        let weighted = table(&[("x", x), ("y", y), ("w", vec![1.0; 80])]);
        let with_weight = distance.clone().with_weight(v(2));
        let refusal = estimator.prepare(weighted.storage(), &with_weight).unwrap_err();
        assert!(refusal.to_string().contains("Next attempt"), "{refusal}");
    }

    #[test]
    fn the_row_bootstrap_is_reproducible_and_reports_an_uncalibrated_spread() {
        let (data, _, _) = simple_regression_data();
        let query = AssociationQuery::new(v(1), [v(0)]);
        let estimator =
            AssociationEstimator::new(Modality::CrossSection).with_bootstrap_replicates(80);
        let prepared = estimator.prepare(data.storage(), &query).unwrap();
        let first = estimator.estimate(&prepared, &ctx(6)).unwrap();
        let again = estimator.estimate(&prepared, &ctx(6)).unwrap();
        assert_eq!(first, again, "a fixed seed reproduces the replicates");
        let record = first.resampling.unwrap();
        assert_eq!((record.method, record.unit), ("bootstrap", "row"));
        assert!(record.replicates_ok >= 70);
        let slope = &first.terms[1];
        assert_eq!(slope.standard_error_method, Some("bootstrap_row"));
        let (lower, upper) = slope.spread.unwrap();
        assert!(lower < slope.estimate && slope.estimate < upper);
    }

    #[test]
    fn a_series_resamples_circular_blocks_and_refuses_a_design_it_cannot_honour() {
        let n = 120;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = 0.8 * x[t - 1] + noise(t);
            y[t] = 0.5 * x[t] + 0.6 * y[t - 1] + noise(t + 1000);
        }
        let series =
            TimeSeriesData::from_f64_columns([("x", &x[..]), ("y", &y[..])], 1_000_000_000)
                .unwrap();
        let query = AssociationQuery::new(v(1), [v(0)]);
        let estimator = AssociationEstimator::new(Modality::Series)
            .with_bootstrap_replicates(60)
            .with_sampling_design(SamplingDeclaration::Declared(
                DeclaredSamplingDesign::DependentTimeSeries,
            ));
        let prepared = estimator.prepare(series.storage(), &query).unwrap();
        let estimate = estimator.estimate(&prepared, &ctx(7)).unwrap();
        let record = estimate.resampling.unwrap();
        assert_eq!((record.method, record.unit), ("block_bootstrap", "calendar_block"));
        let block = record.block_length.unwrap();
        assert!(block >= 2 && block <= n / 3, "dependent blocks, not rows: {block}");
        assert_eq!(estimate.terms[1].standard_error_method, Some("bootstrap_circular_block"));
        assert!(estimate.assumptions.iter().any(|a| a.starts_with("stationary_dependent_series")));
        // Independent subjects cannot describe one series; a dependent design cannot describe a table.
        let wrong = AssociationEstimator::new(Modality::Series).with_sampling_design(
            SamplingDeclaration::Declared(DeclaredSamplingDesign::IndependentSubjectTrajectories),
        );
        assert!(
            wrong.estimate(&prepared, &ctx(7)).unwrap_err().to_string().contains("Next attempt")
        );
        let table_design = AssociationEstimator::new(Modality::CrossSection).with_sampling_design(
            SamplingDeclaration::Declared(DeclaredSamplingDesign::DependentTimeSeries),
        );
        assert!(table_design.check_design().is_err());
        // The analytic error of a series is the Newey-West sandwich.
        let analytic = AssociationEstimator::new(Modality::Series);
        let plain = analytic.estimate(&prepared, &ctx(7)).unwrap();
        assert_eq!(plain.terms[1].standard_error_method, Some("sandwich_newey_west"));
    }

    #[test]
    fn the_conjugate_posterior_tracks_ordinary_least_squares_and_the_bootstrap_covers_other_outcomes()
     {
        let (data, _, _) = simple_regression_data();
        let query = AssociationQuery::new(v(1), [v(0)]);
        let frequentist = AssociationEstimator::new(Modality::CrossSection);
        let prepared = frequentist.prepare(data.storage(), &query).unwrap();
        let ols = frequentist.estimate(&prepared, &ctx(8)).unwrap();
        let bayesian = AssociationEstimator::new(Modality::CrossSection)
            .with_bayesian(BayesianSettings { draws: 400, prior_scale: 100.0 });
        let posterior = bayesian.estimate(&prepared, &ctx(8)).unwrap();
        assert_eq!(posterior.posterior.unwrap().method, "conjugate_normal_inverse_gamma");
        let slope = posterior.terms[1].posterior.unwrap();
        assert!((slope.mean - ols.terms[1].estimate).abs() < 0.01, "{}", slope.mean);
        assert!(slope.lower < slope.mean && slope.mean < slope.upper && slope.sd > 0.0);
        // A 0/1 outcome has no conjugate form here: the Bayesian bootstrap runs.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (level, ones) in [(0.0, 25), (1.0, 55)] {
            for i in 0..100 {
                x.push(level);
                y.push(f64::from(i < ones));
            }
        }
        let binary = table(&[("x", x), ("y", y)]);
        let logistic = bayesian.prepare(binary.storage(), &query).unwrap();
        let draws = bayesian.estimate(&logistic, &ctx(9)).unwrap();
        assert_eq!(draws.posterior.unwrap().method, "bayesian_bootstrap");
        let slope = draws.terms[1].posterior.unwrap();
        assert!(slope.lower < 1.3 && 1.3 < slope.upper, "{slope:?}");
        // The Bayesian reading takes no replicate count.
        let both = bayesian.clone().with_bootstrap_replicates(10);
        assert!(both.estimate(&prepared, &ctx(8)).is_err());
    }

    #[test]
    fn a_predictive_model_returns_out_of_fold_cases_and_blocks_a_series() {
        let (data, _, _) = simple_regression_data();
        let query = AssociationQuery::new(v(1), [v(0)])
            .with_functional(AssociationFunctional::Predictive { folds: 5 });
        let estimator = AssociationEstimator::new(Modality::CrossSection);
        let prepared = estimator.prepare(data.storage(), &query).unwrap();
        let estimate = estimator.estimate(&prepared, &ctx(10)).unwrap();
        let fit = estimate.prediction.unwrap();
        assert_eq!((fit.folds, fit.fold_design, fit.cases.len()), (5, "random_assignment", 60));
        let model: f64 = fit.cases.iter().map(|c| (c.outcome - c.predicted).powi(2)).sum();
        let baseline: f64 = fit.cases.iter().map(|c| (c.outcome - c.baseline).powi(2)).sum();
        assert!(model < baseline / 10.0, "the slope predicts held-out rows: {model} vs {baseline}");
        let series = TimeSeriesData::from_f64_columns(
            [
                ("x", &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0][..]),
                ("y", &[0.1, 1.2, 1.9, 3.2, 3.9, 5.1, 6.0, 7.2][..]),
            ],
            1_000,
        )
        .unwrap();
        let blocked = AssociationEstimator::new(Modality::Series);
        let query = AssociationQuery::new(v(1), [v(0)])
            .with_functional(AssociationFunctional::Predictive { folds: 2 });
        let prepared = blocked.prepare(series.storage(), &query).unwrap();
        let fit = blocked.estimate(&prepared, &ctx(11)).unwrap().prediction.unwrap();
        assert_eq!(fit.fold_design, "consecutive_blocks");
    }
}
