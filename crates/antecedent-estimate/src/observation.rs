//! Estimator-side handling of explicit outcome-observation mechanisms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalResponse, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    IdentificationStatus, ObservationAssumption, ObservationSpec, ResponseFunctional,
    ResponseQuery, ResponseUncertainty, SupportDiagnostic, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_stats::{
    FaerBackend, GaussianObservation, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace,
    fit_glm, fit_observation_logistic, gaussian_observation_log_likelihood, kaplan_meier_ipcw,
    selected_outcome_pseudo_values,
};

use crate::{ContinuousResponseEstimator, EstimationError};

/// Selected-outcome correction used after fitting the observation model.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SelectedOutcomeCorrection {
    /// Inverse-probability weighting, `R Y / p(X)`.
    Ipw,
    /// Augmented inverse-probability weighting with a Gaussian linear outcome nuisance.
    Aipw,
}

/// Numerical choices for observation-mechanism estimation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservationEstimatorOptions {
    /// Selected-outcome correction.
    pub selected_correction: SelectedOutcomeCorrection,
    /// Floor for fitted observation probabilities.
    pub observation_probability_floor: f64,
    /// Floor for estimated censoring survival.
    pub censoring_survival_floor: f64,
    /// Deterministic row-index folds for augmented-path nuisance cross-fitting.
    ///
    /// Used by [`SelectedOutcomeCorrection::Aipw`] only; see [`ObservationMechanismEstimator`]
    /// for why the inverse-probability path keeps its in-sample fit.
    pub crossfit_folds: usize,
}

impl Default for ObservationEstimatorOptions {
    fn default() -> Self {
        Self {
            selected_correction: SelectedOutcomeCorrection::Aipw,
            observation_probability_floor: 0.01,
            censoring_survival_floor: 0.01,
            crossfit_folds: 5,
        }
    }
}

/// Out-of-fold selected-outcome nuisances, one prediction per source row.
struct CrossFittedSelectedNuisances {
    probabilities: Vec<f64>,
    outcome_predictions: Vec<f64>,
}

/// Pseudo-outcome representation produced by a supported observation mechanism.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationAdjustedOutcome {
    /// One pseudo-value per source row.
    ///
    /// Values already incorporate inverse-probability weighting and must not be weighted a
    /// second time by [`Self::weights`].
    pub values: Vec<f64>,
    /// Observation/IPCW weights exposed for diagnostics only.
    /// Unobserved/censored rows have zero weight.
    pub weights: Vec<f64>,
    /// Stable method identifier.
    pub method: Arc<str>,
}

/// Explicit observation-mechanism estimator.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObservationMechanismEstimator {
    /// Numerical and method options.
    pub options: ObservationEstimatorOptions,
}

impl ObservationMechanismEstimator {
    /// Construct with explicit options.
    #[must_use]
    pub const fn new(options: ObservationEstimatorOptions) -> Self {
        Self { options }
    }

    /// Produce IPW/AIPW or IPCW pseudo-outcomes for a response query.
    ///
    /// Selected outcomes require exactly one explicit
    /// [`ObservationAssumption::OutcomeIndependentGiven`] claim and fit a logistic
    /// observation model on those variables. Right/left censoring supports only an empty
    /// conditional-independence set because this implementation uses an unconditional
    /// Kaplan–Meier censoring distribution. `delayed_entry` is supported for right censoring;
    /// left censoring with delayed entry and interval censoring/truncation are refused here.
    ///
    /// Under [`SelectedOutcomeCorrection::Aipw`] both nuisances are cross-fit over
    /// `crossfit_folds` deterministic row-index folds, so no row's pseudo-value is built
    /// from a model that saw it — the sample-splitting condition the AIPW double-robustness
    /// argument assumes. A fold that cannot support either model is refused, never silently
    /// refit in sample. [`SelectedOutcomeCorrection::Ipw`] deliberately keeps its in-sample
    /// maximum-likelihood propensity, which is the published estimator.
    ///
    /// Cross-fitting removes the in-sample nuisance bias; it does not by itself license an
    /// interval, and this path still publishes no standard error.
    ///
    /// # Errors
    ///
    /// Unsupported mechanism/assumption combinations, malformed columns, positivity failure,
    /// or nuisance-model failure.
    pub fn adjusted_outcome(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        delayed_entry: Option<VariableId>,
    ) -> Result<ObservationAdjustedOutcome, EstimationError> {
        query.validate()?;
        self.validate_options()?;
        match &query.observation {
            ObservationSpec::Selected { observed, indicator, .. } => {
                if delayed_entry.is_some() {
                    return Err(EstimationError::unsupported(
                        "delayed entry applies only to right-censoring IPCW",
                    ));
                }
                self.selected(data, query, *observed, *indicator)
            }
            ObservationSpec::RightCensored { observed, censoring, event, .. } => {
                self.censored(data, query, *observed, *censoring, *event, delayed_entry, false)
            }
            ObservationSpec::LeftCensored { observed, censoring, event, .. } => {
                if delayed_entry.is_some() {
                    return Err(EstimationError::unsupported(
                        "delayed entry is not defined for left-censoring sign reversal",
                    ));
                }
                self.censored(data, query, *observed, *censoring, *event, None, true)
            }
            ObservationSpec::Complete => Err(EstimationError::unsupported(
                "complete outcomes do not require observation-mechanism correction",
            )),
            ObservationSpec::IntervalCensored { .. } | ObservationSpec::Truncated { .. } => {
                Err(EstimationError::unsupported(
                    "interval censoring and truncation require the opt-in Gaussian likelihood",
                ))
            }
        }
    }

    /// Estimate an observation-adjusted scalar mean-response curve.
    ///
    /// This composition first constructs the licensed selected-outcome IPW/AIPW or marginal
    /// censoring IPCW pseudo-outcome, then applies the continuous-treatment response estimator
    /// to that pseudo-outcome. For selected outcomes, the declared
    /// [`ObservationAssumption::OutcomeIndependentGiven`] set must include the treatment and
    /// every causal adjustment variable. This containment makes the observation correction
    /// conditionally valid for the downstream response regression. Marginal Kaplan–Meier IPCW
    /// continues to require the stronger unconditional observation-independence claim.
    ///
    /// The returned point curve does **not** carry the complete-data response bands: those bands
    /// omit uncertainty from the estimated observation mechanism and would overstate precision.
    /// Joint nuisance-and-curve uncertainty is therefore reported as [`ResponseUncertainty::None`]
    /// with an explicit diagnostic warning.
    ///
    /// # Errors
    ///
    /// Non-curve response functionals, unsupported observation mechanisms or assumptions,
    /// selected-outcome conditioning that omits treatment/adjustment variables, and any
    /// observation or response-estimation failure are refused.
    pub fn estimate_mean_curve(
        &self,
        response_estimator: &ContinuousResponseEstimator,
        data: &TabularData,
        query: &ResponseQuery,
        delayed_entry: Option<VariableId>,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
    ) -> Result<CausalResponse, EstimationError> {
        let (outcome, treatment) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => (*outcome, treatment.variable),
            _ => {
                return Err(EstimationError::unsupported(
                    "observation-adjusted response composition currently supports MeanCurve only",
                ));
            }
        };
        if let ObservationSpec::Selected { .. } = query.observation {
            let conditioning = exact_outcome_independence(query)?;
            if !conditioning.contains(&treatment)
                || response_estimator
                    .adjustment_set
                    .iter()
                    .any(|variable| !conditioning.contains(variable))
            {
                return Err(EstimationError::unsupported(
                    "selected-outcome response correction requires OutcomeIndependentGiven to include the treatment and every causal adjustment variable",
                ));
            }
        }
        if response_estimator.options.simultaneous_replicates.is_some() {
            return Err(EstimationError::unsupported(
                "simultaneous bands are unavailable for observation-adjusted response curves",
            ));
        }
        let adjusted = self.adjusted_outcome(data, query, delayed_entry)?;
        let adjusted_data = data
            .with_replaced_float(outcome, Arc::from(adjusted.values))
            .map_err(EstimationError::from)?;
        let mut complete_query = query.clone();
        complete_query.observation = ObservationSpec::Complete;
        complete_query.observation_assumptions = Arc::from([]);
        let mut response = response_estimator.estimate_identified(
            &adjusted_data,
            &complete_query,
            identification_status,
            assumptions,
        )?;
        let (minimum_weight, maximum_weight, effective_sample_size) =
            diagnostic_weight_summary(&adjusted.weights);
        response.uncertainty = ResponseUncertainty::None;
        response.provenance_id = Arc::from("estimate.response.observation_adjusted");
        response.support.diagnostics.push(SupportDiagnostic {
            id: Arc::from("response.observation_adjustment_weights"),
            values: Arc::from([minimum_weight, maximum_weight, effective_sample_size]),
            detail: Arc::from(
                "minimum positive weight, maximum weight, and Kish effective sample size; weights are diagnostic only and were already incorporated into the pseudo-outcome",
            ),
        });
        response.support.warnings.push(Diagnostic::new(
            "response.observation_joint_uncertainty_unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "point estimate includes observation correction; uncertainty is omitted because complete-data curve bands do not account for the estimated observation mechanism",
        ));
        response.support.warnings.push(Diagnostic::new(
            "response.observation_adjustment_method",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            adjusted.method,
        ));
        Ok(response)
    }

    /// Evaluate the opt-in Gaussian likelihood for the query's observation mechanism.
    ///
    /// The query must explicitly declare
    /// `ObservationAssumption::Structural("gaussian_observation_likelihood")`. `means` must
    /// contain one latent-outcome mean per source row. Selected outcomes are not supported by
    /// this likelihood path.
    ///
    /// # Errors
    ///
    /// Missing structural opt-in, unsupported mechanism, malformed data, or invalid likelihood.
    pub fn gaussian_log_likelihood(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        means: &[f64],
        sigma: f64,
    ) -> Result<f64, EstimationError> {
        query.validate()?;
        if means.len() != data.row_count() {
            return Err(EstimationError::unsupported(
                "Gaussian observation means must align with source rows",
            ));
        }
        let opted_in = query.observation_assumptions.iter().any(|assumption| {
            matches!(assumption, ObservationAssumption::Structural(name) if name.as_ref() == "gaussian_observation_likelihood")
        });
        if !opted_in {
            return Err(EstimationError::unsupported(
                "Gaussian censoring/truncation likelihood requires an explicit structural opt-in",
            ));
        }
        let observations = gaussian_observations(data, &query.observation)?;
        Ok(gaussian_observation_log_likelihood(&observations, means, sigma)?)
    }

    fn validate_options(&self) -> Result<(), EstimationError> {
        if !self.options.observation_probability_floor.is_finite()
            || !(0.0..0.5).contains(&self.options.observation_probability_floor)
            || !self.options.censoring_survival_floor.is_finite()
            || !(0.0..1.0).contains(&self.options.censoring_survival_floor)
            || self.options.crossfit_folds < 2
        {
            return Err(EstimationError::unsupported("invalid observation-estimator options"));
        }
        Ok(())
    }

    fn selected(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        observed_id: VariableId,
        indicator_id: VariableId,
    ) -> Result<ObservationAdjustedOutcome, EstimationError> {
        let conditioning = exact_outcome_independence(query)?;
        if conditioning.iter().any(|id| *id == observed_id || *id == indicator_id) {
            return Err(EstimationError::unsupported(
                "observation-model conditions cannot include observed outcome or indicator",
            ));
        }
        let observed = data.float64_values(observed_id)?;
        let indicator = data.float64_values(indicator_id)?;
        let covariates = read_complete_columns(data, conditioning)?;
        if indicator.iter().all(|&r| r == 1.0) {
            if observed.iter().any(|value| !value.is_finite()) {
                return Err(EstimationError::unsupported("selected outcomes must be finite"));
            }
            return Ok(ObservationAdjustedOutcome {
                values: observed,
                weights: vec![1.0; data.row_count()],
                method: Arc::from("observation.selected.complete_collapse.v1"),
            });
        }
        let (probabilities, outcome_predictions) = match self.options.selected_correction {
            // Plain IPW keeps the in-sample maximum-likelihood propensity. That is the
            // published estimator, and estimating the propensity in sample is what makes it
            // efficient — cross-fitting here would deviate from the citation to no end.
            SelectedOutcomeCorrection::Ipw => {
                let fit = fit_observation_logistic(
                    &indicator,
                    &covariates,
                    conditioning.len(),
                    self.options.observation_probability_floor,
                )?;
                (fit.probabilities, None)
            }
            // The augmented path is where sample splitting earns its keep: the AIPW
            // double-robustness and asymptotic-linearity arguments assume the nuisances were
            // not fit on the row they are evaluated at.
            SelectedOutcomeCorrection::Aipw => {
                let nuisances =
                    self.crossfit_selected_nuisances(&observed, &indicator, &covariates)?;
                (nuisances.probabilities, Some(nuisances.outcome_predictions))
            }
        };
        let values = selected_outcome_pseudo_values(
            &observed,
            &indicator,
            &probabilities,
            outcome_predictions.as_deref(),
        )?;
        let weights = indicator
            .iter()
            .zip(&probabilities)
            .map(|(&r, &p)| if r == 1.0 { 1.0 / p } else { 0.0 })
            .collect();
        Ok(ObservationAdjustedOutcome {
            values,
            weights,
            method: Arc::from(match self.options.selected_correction {
                SelectedOutcomeCorrection::Ipw => "observation.selected.logistic_ipw.v1",
                SelectedOutcomeCorrection::Aipw => "observation.selected.crossfit_logistic_aipw.v1",
            }),
        })
    }

    /// Fit both selected-outcome nuisances out of fold.
    ///
    /// Folds are the deterministic row-index folds used elsewhere in this crate. Every row
    /// receives a probability and an outcome prediction from a model fit without it. A fold
    /// whose training rows cannot support either model is refused rather than quietly
    /// falling back to an in-sample fit, which would reintroduce exactly the bias the
    /// splitting removes.
    fn crossfit_selected_nuisances(
        &self,
        observed: &[f64],
        indicator: &[f64],
        covariates: &[f64],
    ) -> Result<CrossFittedSelectedNuisances, EstimationError> {
        let n = indicator.len();
        let folds = self.options.crossfit_folds;
        let ncols = if n == 0 { 0 } else { covariates.len() / n };
        if folds > n {
            return Err(EstimationError::unsupported(
                "cross-fitting folds cannot exceed observed rows",
            ));
        }
        let mut probabilities = vec![f64::NAN; n];
        let mut outcome_predictions = vec![f64::NAN; n];
        for fold in 0..folds {
            let train: Vec<usize> = (0..n).filter(|i| i % folds != fold).collect();
            let valid: Vec<usize> = (0..n).filter(|i| i % folds == fold).collect();
            if valid.is_empty() {
                continue;
            }
            let train_indicator: Vec<f64> = train.iter().map(|&i| indicator[i]).collect();
            let train_covariates = subset_colmajor(covariates, n, ncols, &train);
            let fit = fit_observation_logistic(
                &train_indicator,
                &train_covariates,
                ncols,
                self.options.observation_probability_floor,
            )
            .map_err(|_| {
                EstimationError::unsupported(
                    "a cross-fitting fold cannot support the observation model; reduce crossfit_folds or supply more rows covering both observed and unobserved outcomes",
                )
            })?;
            let train_observed: Vec<f64> = train.iter().map(|&i| observed[i]).collect();
            let coefficients = fit_selected_outcome_regression(
                &train_observed,
                &train_indicator,
                &train_covariates,
                ncols,
            )?;
            for &row in &valid {
                let features = covariate_row(covariates, n, ncols, row);
                probabilities[row] = fit.probability_at(&features)?;
                outcome_predictions[row] = predict_linear(&coefficients, &features);
            }
        }
        if probabilities.iter().chain(&outcome_predictions).any(|value| !value.is_finite()) {
            return Err(EstimationError::unsupported(
                "cross-fitted observation nuisances produced a non-finite prediction",
            ));
        }
        Ok(CrossFittedSelectedNuisances { probabilities, outcome_predictions })
    }

    #[allow(clippy::too_many_arguments)]
    fn censored(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        observed_id: VariableId,
        censoring_id: VariableId,
        event_id: VariableId,
        delayed_entry: Option<VariableId>,
        reverse: bool,
    ) -> Result<ObservationAdjustedOutcome, EstimationError> {
        require_unconditional_censoring_independence(query)?;
        let observed = data.float64_values(observed_id)?;
        let censoring = data.float64_values(censoring_id)?;
        let event = data.float64_values(event_id)?;
        if observed.iter().chain(&censoring).any(|v| !v.is_finite()) {
            return Err(EstimationError::unsupported(
                "censoring times and recorded outcomes must be finite",
            ));
        }
        for i in 0..observed.len() {
            let compatible =
                if reverse { observed[i] >= censoring[i] } else { observed[i] <= censoring[i] };
            if !compatible || (event[i] == 0.0 && observed[i] != censoring[i]) {
                return Err(EstimationError::unsupported(
                    "recorded outcome is incompatible with its censoring value/event",
                ));
            }
        }
        let transformed: Vec<f64> =
            observed.iter().map(|&value| if reverse { -value } else { value }).collect();
        let entry_values = delayed_entry.map(|id| data.float64_values(id)).transpose()?;
        let weights = kaplan_meier_ipcw(
            &transformed,
            &event,
            entry_values.as_deref(),
            self.options.censoring_survival_floor,
        )?;
        let values = observed.iter().zip(&weights).map(|(&y, &w)| y * w).collect();
        Ok(ObservationAdjustedOutcome {
            values,
            weights,
            method: Arc::from(if reverse {
                "observation.left_censored.km_ipcw_sign_reversal.v1"
            } else if delayed_entry.is_some() {
                "observation.right_censored.km_ipcw_delayed_entry.v1"
            } else {
                "observation.right_censored.km_ipcw.v1"
            }),
        })
    }
}

fn diagnostic_weight_summary(weights: &[f64]) -> (f64, f64, f64) {
    let minimum =
        weights.iter().copied().filter(|weight| *weight > 0.0).fold(f64::INFINITY, f64::min);
    let maximum = weights.iter().copied().fold(0.0_f64, f64::max);
    let sum = weights.iter().sum::<f64>();
    let sum_squares = weights.iter().map(|weight| weight * weight).sum::<f64>();
    let effective_sample_size = if sum_squares > 0.0 { sum * sum / sum_squares } else { 0.0 };
    (if minimum.is_finite() { minimum } else { 0.0 }, maximum, effective_sample_size)
}

fn exact_outcome_independence(query: &ResponseQuery) -> Result<&[VariableId], EstimationError> {
    let mut claims =
        query.observation_assumptions.iter().filter_map(|assumption| match assumption {
            ObservationAssumption::OutcomeIndependentGiven(vars) => Some(vars.as_ref()),
            _ => None,
        });
    let Some(first) = claims.next() else {
        return Err(EstimationError::unsupported(
            "selected-outcome correction requires OutcomeIndependentGiven",
        ));
    };
    if claims.next().is_some() || query.observation_assumptions.len() != 1 {
        return Err(EstimationError::unsupported(
            "selected-outcome correction requires exactly one supported observation assumption",
        ));
    }
    Ok(first)
}

fn require_unconditional_censoring_independence(
    query: &ResponseQuery,
) -> Result<(), EstimationError> {
    if query.observation_assumptions.len() != 1 {
        return Err(EstimationError::unsupported(
            "Kaplan-Meier IPCW requires exactly one unconditional independence assumption",
        ));
    }
    match &query.observation_assumptions[0] {
        ObservationAssumption::IndependentGiven(vars)
        | ObservationAssumption::OutcomeIndependentGiven(vars)
            if vars.is_empty() =>
        {
            Ok(())
        }
        _ => Err(EstimationError::unsupported(
            "Kaplan-Meier IPCW cannot adjust conditional censoring; the declared set must be empty",
        )),
    }
}

fn read_complete_columns(
    data: &TabularData,
    variables: &[VariableId],
) -> Result<Vec<f64>, EstimationError> {
    let mut values = Vec::with_capacity(data.row_count() * variables.len());
    for &variable in variables {
        let column = data.float64_values(variable)?;
        if column.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::unsupported(
                "observation-model covariates must be completely observed and finite",
            ));
        }
        values.extend(column);
    }
    Ok(values)
}

/// Column-major covariates for a subset of rows, preserving column order.
fn subset_colmajor(covariates: &[f64], n: usize, ncols: usize, rows: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; rows.len() * ncols];
    for col in 0..ncols {
        for (position, &row) in rows.iter().enumerate() {
            out[col * rows.len() + position] = covariates[col * n + row];
        }
    }
    out
}

/// One row's covariate values, excluding the intercept.
fn covariate_row(covariates: &[f64], n: usize, ncols: usize, row: usize) -> Vec<f64> {
    (0..ncols).map(|col| covariates[col * n + row]).collect()
}

/// Linear prediction from an intercept-leading coefficient vector.
fn predict_linear(coefficients: &[f64], features: &[f64]) -> f64 {
    coefficients[0]
        + coefficients[1..].iter().zip(features).map(|(beta, value)| beta * value).sum::<f64>()
}

/// Selected-outcome regression coefficients, intercept first.
///
/// Returns coefficients rather than fitted values so the caller decides which rows the model
/// is evaluated on; returning in-sample predictions would make out-of-fold use impossible.
fn fit_selected_outcome_regression(
    observed: &[f64],
    indicator: &[f64],
    covariates: &[f64],
    ncols: usize,
) -> Result<Vec<f64>, EstimationError> {
    let n = observed.len();
    let rows: Vec<usize> = (0..n).filter(|&i| indicator[i] == 1.0).collect();
    if rows.len() <= ncols + 1 {
        return Err(EstimationError::unsupported(
            "too few selected rows for augmented outcome regression",
        ));
    }
    let train_n = rows.len();
    let mut train_x = vec![1.0; train_n * (ncols + 1)];
    for col in 0..ncols {
        for (r, &source) in rows.iter().enumerate() {
            train_x[(col + 1) * train_n + r] = covariates[col * n + source];
        }
    }
    let train_y: Vec<f64> = rows.iter().map(|&i| observed[i]).collect();
    if train_y.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::unsupported("selected outcomes must be finite"));
    }
    let mut workspace = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::GaussianIdentity,
        GlmDesignRef { x_colmajor: &train_x, nrows: train_n, ncols: ncols + 1, y: &train_y },
        &FaerBackend,
        &mut workspace,
        &GlmOptions::default(),
    )?;
    fit.require_ok()?;
    Ok(fit.coefficients)
}

fn gaussian_observations(
    data: &TabularData,
    spec: &ObservationSpec,
) -> Result<Vec<GaussianObservation>, EstimationError> {
    Ok(match spec {
        ObservationSpec::Complete => {
            return Err(EstimationError::unsupported(
                "complete Gaussian outcomes use the ordinary complete-data likelihood",
            ));
        }
        ObservationSpec::Selected { .. } => {
            return Err(EstimationError::unsupported(
                "selected outcomes use logistic IPW/AIPW, not the Gaussian observation likelihood",
            ));
        }
        ObservationSpec::RightCensored { observed, event, .. } => {
            let y = data.float64_values(*observed)?;
            let delta = data.float64_values(*event)?;
            binary_events(&delta)?;
            y.into_iter()
                .zip(delta)
                .map(|(value, d)| {
                    if d == 1.0 {
                        GaussianObservation::Exact(value)
                    } else {
                        GaussianObservation::RightCensored(value)
                    }
                })
                .collect()
        }
        ObservationSpec::LeftCensored { observed, event, .. } => {
            let y = data.float64_values(*observed)?;
            let delta = data.float64_values(*event)?;
            binary_events(&delta)?;
            y.into_iter()
                .zip(delta)
                .map(|(value, d)| {
                    if d == 1.0 {
                        GaussianObservation::Exact(value)
                    } else {
                        GaussianObservation::LeftCensored(value)
                    }
                })
                .collect()
        }
        ObservationSpec::IntervalCensored { lower, upper, .. } => {
            let lower = data.float64_values(*lower)?;
            let upper = data.float64_values(*upper)?;
            lower
                .into_iter()
                .zip(upper)
                .map(|(lower, upper)| GaussianObservation::IntervalCensored { lower, upper })
                .collect()
        }
        ObservationSpec::Truncated { observed, lower, upper, .. } => {
            let value = data.float64_values(*observed)?;
            let lower = lower.map(|id| data.float64_values(id)).transpose()?;
            let upper = upper.map(|id| data.float64_values(id)).transpose()?;
            (0..data.row_count())
                .map(|i| GaussianObservation::Truncated {
                    value: value[i],
                    lower: lower.as_ref().map_or(f64::NEG_INFINITY, |v| v[i]),
                    upper: upper.as_ref().map_or(f64::INFINITY, |v| v[i]),
                })
                .collect()
        }
    })
}

fn binary_events(events: &[f64]) -> Result<(), EstimationError> {
    if events.iter().all(|&event| event == 0.0 || event == 1.0) {
        Ok(())
    } else {
        Err(EstimationError::unsupported("censoring event indicators must be binary"))
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        ContinuousDomain, GridSpec, ObservationAssumption, ResponseFunctional, ResponseQuery,
    };

    use super::*;

    fn response_query(outcome: VariableId, treatment: VariableId) -> ResponseQuery {
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome,
            treatment: ContinuousDomain::new(treatment, GridSpec::Values(Arc::from([-0.1, 0.1]))),
        })
    }

    /// Selection depends on `x`, so a complete-case mean is biased and only a correction
    /// that uses the observation model recovers `E[Y] = 2 + 3 E[x]`.
    fn selection_biased_sample(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let r: Vec<f64> =
            x.iter().enumerate().map(|(i, &x)| f64::from(x > 0.25 || i % 3 == 0)).collect();
        let y: Vec<f64> = x
            .iter()
            .zip(&r)
            .map(|(&x, &r)| if r == 1.0 { 2.0 + 3.0 * x } else { f64::NAN })
            .collect();
        (x, r, y)
    }

    fn selected_query() -> ResponseQuery {
        response_query(VariableId::from_raw(1), VariableId::from_raw(0)).with_observation(
            ObservationSpec::Selected {
                latent: VariableId::from_raw(1),
                observed: VariableId::from_raw(1),
                indicator: VariableId::from_raw(2),
            },
            [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(3)]))],
        )
    }

    fn selected_table(x: &[f64], y: &[f64], r: &[f64]) -> TabularData {
        TabularData::from_f64_columns([("a", x), ("y", y), ("r", r), ("x", x)]).unwrap()
    }

    #[test]
    fn selected_aipw_nuisances_are_fit_without_the_row_they_are_applied_to() {
        // The property cross-fitting exists to provide. Row 0 falls in fold 0, so the
        // probability used for it must come from a model fit on the rows outside fold 0 —
        // not from the all-rows fit, which is what this path used to do.
        let n = 100;
        let (x, r, y) = selection_biased_sample(n);
        let data = selected_table(&x, &y, &r);
        let estimator = ObservationMechanismEstimator::default();
        let adjusted = estimator.adjusted_outcome(&data, &selected_query(), None).unwrap();
        assert_eq!(adjusted.method.as_ref(), "observation.selected.crossfit_logistic_aipw.v1");

        let folds = estimator.options.crossfit_folds;
        let train: Vec<usize> = (0..n).filter(|i| i % folds != 0).collect();
        let train_indicator: Vec<f64> = train.iter().map(|&i| r[i]).collect();
        let train_covariates: Vec<f64> = train.iter().map(|&i| x[i]).collect();
        let out_of_fold = fit_observation_logistic(
            &train_indicator,
            &train_covariates,
            1,
            estimator.options.observation_probability_floor,
        )
        .unwrap()
        .probability_at(&[x[0]])
        .unwrap();

        let in_sample =
            fit_observation_logistic(&r, &x, 1, estimator.options.observation_probability_floor)
                .unwrap()
                .probabilities[0];

        assert_eq!(r[0], 1.0, "row 0 must be selected for its weight to be 1/p");
        let used = 1.0 / adjusted.weights[0];
        assert!(
            (used - out_of_fold).abs() < 1e-9,
            "row 0 used p={used}, expected the out-of-fold p={out_of_fold}"
        );
        assert!(
            (used - in_sample).abs() > 1e-12,
            "out-of-fold and in-sample probabilities coincide; the test cannot tell them apart"
        );
    }

    #[test]
    fn crossfit_selected_aipw_recovers_the_latent_mean_under_biased_selection() {
        let n = 150;
        let (x, r, y) = selection_biased_sample(n);
        let data = selected_table(&x, &y, &r);
        let adjusted = ObservationMechanismEstimator::default()
            .adjusted_outcome(&data, &selected_query(), None)
            .unwrap();

        let truth = x.iter().map(|&x| 2.0 + 3.0 * x).sum::<f64>() / n as f64;
        let corrected = adjusted.values.iter().sum::<f64>() / n as f64;
        let complete_case = {
            let selected: Vec<f64> =
                y.iter().zip(&r).filter(|&(_, &r)| r == 1.0).map(|(&y, _)| y).collect();
            selected.iter().sum::<f64>() / selected.len() as f64
        };
        assert!(
            (corrected - truth).abs() < 1e-6,
            "cross-fitted AIPW gave {corrected}, truth {truth}"
        );
        assert!(
            (complete_case - truth).abs() > 0.1,
            "the complete-case mean must be visibly biased or this proves nothing"
        );
    }

    #[test]
    fn a_fold_that_cannot_support_the_observation_model_is_refused() {
        // Every unobserved row sits in one fold, so the other folds train on selected rows
        // only and the logistic model has no contrast. Refusing is the point: silently
        // refitting in sample would return exactly the biased answer cross-fitting removes.
        let n = 60usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let r: Vec<f64> = (0..n).map(|i| f64::from(i % 5 != 0)).collect();
        let y: Vec<f64> = x
            .iter()
            .zip(&r)
            .map(|(&x, &r)| if r == 1.0 { 2.0 + 3.0 * x } else { f64::NAN })
            .collect();
        let data = selected_table(&x, &y, &r);
        let error = ObservationMechanismEstimator::default()
            .adjusted_outcome(&data, &selected_query(), None)
            .unwrap_err();
        assert!(error.to_string().contains("cross-fitting fold"), "got {error}");
    }

    #[test]
    fn selected_aipw_requires_and_uses_explicit_outcome_independence() {
        let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 79.0).collect();
        let r: Vec<f64> = (0..80).map(|i| f64::from(i % 3 != 0)).collect();
        let y: Vec<f64> = x
            .iter()
            .zip(&r)
            .map(|(&x, &r)| if r == 1.0 { 2.0 + 3.0 * x } else { f64::NAN })
            .collect();
        let data = TabularData::from_f64_columns([
            ("a", x.as_slice()),
            ("y", y.as_slice()),
            ("r", r.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let mut query = response_query(VariableId::from_raw(1), VariableId::from_raw(0));
        query = query.with_observation(
            ObservationSpec::Selected {
                latent: VariableId::from_raw(1),
                observed: VariableId::from_raw(1),
                indicator: VariableId::from_raw(2),
            },
            [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(3)]))],
        );
        let adjusted =
            ObservationMechanismEstimator::default().adjusted_outcome(&data, &query, None).unwrap();
        assert!(
            adjusted.values.iter().zip(&x).all(|(&got, &x)| (got - (2.0 + 3.0 * x)).abs() < 1e-8)
        );
    }

    #[test]
    fn selected_aipw_composes_into_point_curve_without_invalid_bands() {
        let a: Vec<f64> = (0..80).map(|i| f64::from(i) / 79.0).collect();
        let r: Vec<f64> = (0..80).map(|i| f64::from(i % 3 != 0)).collect();
        let y: Vec<f64> = a
            .iter()
            .zip(&r)
            .map(|(&a, &r)| if r == 1.0 { 2.0 + 3.0 * a } else { f64::NAN })
            .collect();
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("r", r.as_slice()),
        ])
        .unwrap();
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.2, 0.8])),
            ),
        })
        .with_observation(
            ObservationSpec::Selected {
                latent: VariableId::from_raw(1),
                observed: VariableId::from_raw(1),
                indicator: VariableId::from_raw(2),
            },
            [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(0)]))],
        );
        let response = ObservationMechanismEstimator::default()
            .estimate_mean_curve(
                &ContinuousResponseEstimator::new(Arc::from([])),
                &data,
                &query,
                None,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        assert_eq!(response.uncertainty, ResponseUncertainty::None);
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.observation_adjusted");
        assert!(
            response.support.warnings.iter().any(|warning| warning.code.as_ref()
                == "response.observation_joint_uncertainty_unavailable")
        );
    }

    #[test]
    fn selected_curve_refuses_missing_treatment_in_observation_conditioning() {
        let values: Vec<f64> = (0..40).map(f64::from).collect();
        let selected = vec![1.0; values.len()];
        let data = TabularData::from_f64_columns([
            ("a", values.as_slice()),
            ("y", values.as_slice()),
            ("r", selected.as_slice()),
            ("x", values.as_slice()),
        ])
        .unwrap();
        let query = response_query(VariableId::from_raw(1), VariableId::from_raw(0))
            .with_observation(
                ObservationSpec::Selected {
                    latent: VariableId::from_raw(1),
                    observed: VariableId::from_raw(1),
                    indicator: VariableId::from_raw(2),
                },
                [ObservationAssumption::OutcomeIndependentGiven(Arc::from([
                    VariableId::from_raw(3),
                ]))],
            );
        let error = ObservationMechanismEstimator::default()
            .estimate_mean_curve(
                &ContinuousResponseEstimator::new(Arc::from([])),
                &data,
                &query,
                None,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("include the treatment"));
    }

    #[test]
    fn conditional_km_claim_fails_closed() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let event = [0.0, 1.0, 1.0, 1.0];
        let data = TabularData::from_f64_columns([
            ("a", values.as_slice()),
            ("y", values.as_slice()),
            ("c", values.as_slice()),
            ("d", event.as_slice()),
        ])
        .unwrap();
        let query = response_query(VariableId::from_raw(1), VariableId::from_raw(0))
            .with_observation(
                ObservationSpec::RightCensored {
                    latent: VariableId::from_raw(1),
                    observed: VariableId::from_raw(1),
                    censoring: VariableId::from_raw(2),
                    event: VariableId::from_raw(3),
                },
                [ObservationAssumption::IndependentGiven(Arc::from([VariableId::from_raw(0)]))],
            );
        let error = ObservationMechanismEstimator::default()
            .adjusted_outcome(&data, &query, None)
            .unwrap_err();
        assert!(error.to_string().contains("cannot adjust conditional censoring"));
    }

    #[test]
    fn gaussian_interval_likelihood_is_explicitly_opt_in() {
        let lower = [-1.0, 0.0];
        let upper = [0.0, 1.0];
        let treatment = [0.0, 1.0];
        let data = TabularData::from_f64_columns([
            ("a", treatment.as_slice()),
            ("lo", lower.as_slice()),
            ("hi", upper.as_slice()),
        ])
        .unwrap();
        let base = response_query(VariableId::from_raw(1), VariableId::from_raw(0));
        let spec = ObservationSpec::IntervalCensored {
            latent: VariableId::from_raw(1),
            lower: VariableId::from_raw(1),
            upper: VariableId::from_raw(2),
        };
        let without = base.clone().with_observation(spec.clone(), Arc::from([]));
        assert!(
            ObservationMechanismEstimator::default()
                .gaussian_log_likelihood(&data, &without, &[0.0, 0.0], 1.0)
                .is_err()
        );
        let with = base.with_observation(
            spec,
            [ObservationAssumption::Structural(Arc::from("gaussian_observation_likelihood"))],
        );
        assert!(
            ObservationMechanismEstimator::default()
                .gaussian_log_likelihood(&data, &with, &[0.0, 0.0], 1.0)
                .unwrap()
                .is_finite()
        );
    }

    #[test]
    fn no_selection_and_no_censoring_collapse_to_observed_values() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let selected = [1.0; 4];
        let far_censor = [10.0; 4];
        let data = TabularData::from_f64_columns([
            ("a", values.as_slice()),
            ("y", values.as_slice()),
            ("r", selected.as_slice()),
            ("c", far_censor.as_slice()),
        ])
        .unwrap();
        let selected_query = response_query(VariableId::from_raw(1), VariableId::from_raw(0))
            .with_observation(
                ObservationSpec::Selected {
                    latent: VariableId::from_raw(1),
                    observed: VariableId::from_raw(1),
                    indicator: VariableId::from_raw(2),
                },
                [ObservationAssumption::OutcomeIndependentGiven(Arc::from([]))],
            );
        let right_query = response_query(VariableId::from_raw(1), VariableId::from_raw(0))
            .with_observation(
                ObservationSpec::RightCensored {
                    latent: VariableId::from_raw(1),
                    observed: VariableId::from_raw(1),
                    censoring: VariableId::from_raw(3),
                    event: VariableId::from_raw(2),
                },
                [ObservationAssumption::IndependentGiven(Arc::from([]))],
            );
        let estimator = ObservationMechanismEstimator::default();
        let selected_adjusted = estimator.adjusted_outcome(&data, &selected_query, None).unwrap();
        let right_adjusted = estimator.adjusted_outcome(&data, &right_query, None).unwrap();
        assert_eq!(selected_adjusted.values, values);
        assert_eq!(right_adjusted.values, values);
        assert_eq!(selected_adjusted.weights, vec![1.0; 4]);
        assert_eq!(right_adjusted.weights, vec![1.0; 4]);
    }
}
