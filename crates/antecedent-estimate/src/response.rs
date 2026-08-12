//! Complete-observation estimators for continuous causal responses.
//!
//! The scalar curve path uses deterministic cross-fitting, Gaussian additive
//! nuisance regressions, the Kennedy-style doubly robust pseudo-outcome, and a
//! Gaussian local-polynomial smoother. Average derivatives use the corresponding
//! Gaussian treatment-score Riesz representer. Multivariate derivatives are
//! deliberately low-dimensional, model-dependent plug-in functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalResponse, DerivativeScale, DerivativeWeighting, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, IdentificationStatus, Intervention,
    MAX_NONPARAMETRIC_RESPONSE_DIM, ObservationSpec, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, StochasticPolicy, SupportDiagnostic,
    SupportRegion, SupportReport, SupportStatus, TargetPopulation, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_stats::{
    FaerBackend, GamOptions, GamWorkspace, SmoothSpec, fit_gam, gaussian_density,
    gaussian_local_quadratic, gaussian_local_quadratic_influence, normal_ppf, predict_gam,
    silverman_bandwidth,
};

use crate::EstimationError;

/// Configuration for [`ContinuousResponseEstimator`].
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousResponseOptions {
    /// Deterministic row-index folds used for nuisance cross-fitting.
    pub folds: usize,
    /// Cubic B-spline basis count for each additive nuisance term.
    pub nuisance_basis: usize,
    /// Roughness penalty for nuisance smooths.
    pub nuisance_lambda: f64,
    /// Optional response-kernel bandwidth; normal-reference bandwidth when absent.
    pub bandwidth: Option<f64>,
    /// Local ESS below which support is classified as weak.
    pub minimum_local_ess: f64,
    /// Pointwise confidence level.
    pub confidence_level: f64,
    /// Wild-multiplier replicates for a fixed-grid simultaneous band.
    /// `None` preserves pointwise-band behavior.
    pub simultaneous_replicates: Option<u32>,
    /// Deterministic seed for simultaneous-band multipliers.
    pub multiplier_seed: u64,
}

impl Default for ContinuousResponseOptions {
    fn default() -> Self {
        Self {
            folds: 5,
            nuisance_basis: 6,
            nuisance_lambda: 1.0,
            bandwidth: None,
            minimum_local_ess: 20.0,
            confidence_level: 0.95,
            simultaneous_replicates: None,
            multiplier_seed: 0xA17E_CEDE_0500,
        }
    }
}

/// Continuous-response estimator with a caller-supplied valid adjustment set.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousResponseEstimator {
    /// Adjustment variables licensed by identification.
    pub adjustment_set: Arc<[VariableId]>,
    /// Numerical and diagnostic options.
    pub options: ContinuousResponseOptions,
}

impl ContinuousResponseEstimator {
    /// Construct with default numerical options.
    #[must_use]
    pub fn new(adjustment_set: impl Into<Arc<[VariableId]>>) -> Self {
        Self {
            adjustment_set: adjustment_set.into(),
            options: ContinuousResponseOptions::default(),
        }
    }

    /// Estimate a response already licensed by the identification layer.
    ///
    /// Only complete observations and the all-observed target are supported here.
    /// Missing rows in any used column are dropped jointly. The estimator never
    /// chooses or validates the causal adequacy of `adjustment_set`.
    ///
    /// # Errors
    ///
    /// Invalid/unsupported queries, non-point identification, insufficient complete
    /// observations, invalid options, or nuisance/smoothing failures.
    pub fn estimate_identified(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
    ) -> Result<CausalResponse, EstimationError> {
        self.validate(query, identification_status)?;
        let (value, uncertainty, support, provenance_id) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                let (value, uncertainty, support) =
                    self.mean_curve(data, *outcome, treatment.variable, &treatment.grid.values()?)?;
                let provenance =
                    if matches!(uncertainty, ResponseUncertainty::SimultaneousBand { .. }) {
                        "estimate.response.kennedy_dr_simultaneous"
                    } else {
                        "estimate.response.kennedy_dr"
                    };
                (value, uncertainty, support, provenance)
            }
            ResponseFunctional::PointDerivative { outcome, treatment, at, order, scale } => {
                let (value, uncertainty, support) =
                    self.point_derivative(data, *outcome, *treatment, *at, *order, *scale)?;
                (value, uncertainty, support, "estimate.response.point_derivative")
            }
            ResponseFunctional::AverageDerivative { outcome, treatment, weighting } => {
                let (value, uncertainty, support) =
                    self.average_derivative(data, *outcome, *treatment, weighting)?;
                (value, uncertainty, support, "estimate.response.riesz_ade")
            }
            ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                let (value, uncertainty, support) =
                    self.jacobian(data, outcomes, treatments, at, *scale)?;
                (value, uncertainty, support, "estimate.response.gam_derivative")
            }
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
                let (value, uncertainty, support) =
                    self.directional_derivative(data, outcomes, treatments, at, direction)?;
                (value, uncertainty, support, "estimate.response.gam_derivative")
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                let (value, uncertainty, support) =
                    self.intervention_response(data, *outcome, interventions)?;
                (value, uncertainty, support, "estimate.response.intervention_gcomp")
            }
        };
        Ok(CausalResponse {
            estimand: query.functional.clone(),
            identification_status,
            estimate: ResponseIdentification::PointIdentified(value),
            uncertainty,
            support,
            assumptions,
            provenance_id: Arc::from(provenance_id),
        })
    }

    fn validate(
        &self,
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
    ) -> Result<(), EstimationError> {
        query.validate()?;
        if query.observation != ObservationSpec::Complete {
            return Err(EstimationError::unsupported(
                "continuous-response estimator currently requires complete observations",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::TargetPopulation);
        }
        if !matches!(
            identification_status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "continuous-response estimation requires point identification",
            });
        }
        let o = &self.options;
        if o.folds < 2
            || o.nuisance_basis < 4
            || !o.nuisance_lambda.is_finite()
            || o.nuisance_lambda < 0.0
            || o.bandwidth.is_some_and(|h| !h.is_finite() || h <= 0.0)
            || !o.minimum_local_ess.is_finite()
            || o.minimum_local_ess <= 0.0
            || !o.confidence_level.is_finite()
            || !(0.0..1.0).contains(&o.confidence_level)
            || o.simultaneous_replicates.is_some_and(|replicates| replicates < 100)
            || (o.simultaneous_replicates.is_some() && o.bandwidth.is_none())
        {
            return Err(EstimationError::unsupported("invalid continuous-response options"));
        }
        Ok(())
    }

    fn mean_curve(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatment: VariableId,
        grid: &[f64],
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        let sample = CompleteSample::read(data, outcome, &[treatment], &self.adjustment_set)?;
        let pseudo = self.cross_fitted_pseudo_outcome(&sample)?;
        let bandwidth = self.options.bandwidth.unwrap_or(silverman_bandwidth(&sample.treatments)?);
        let mut mean = Vec::with_capacity(grid.len());
        let mut lower = Vec::with_capacity(grid.len());
        let mut upper = Vec::with_capacity(grid.len());
        let mut ess = Vec::with_capacity(grid.len());
        let mut density = Vec::with_capacity(grid.len());
        let mut influences = Vec::with_capacity(grid.len());
        let mut robust_se = Vec::with_capacity(grid.len());
        let z = normal_ppf(0.5 + self.options.confidence_level / 2.0);
        for &at in grid {
            let fit =
                gaussian_local_quadratic_influence(&sample.treatments, &pseudo, at, bandwidth)?;
            let point = fit.point;
            mean.push(point.value);
            lower.push(point.value - z * point.standard_error);
            upper.push(point.value + z * point.standard_error);
            ess.push(point.local_ess);
            density.push(
                point.weight_sum
                    / (sample.len() as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt()),
            );
            influences.push(fit.influences);
            robust_se.push(fit.robust_standard_error);
        }
        let support =
            support_report(grid, &sample.treatments, &ess, density, self.options.minimum_local_ess);
        let uncertainty = if let Some(replicates) = self.options.simultaneous_replicates {
            simultaneous_multiplier_band(
                &mean,
                &influences,
                &robust_se,
                self.options.confidence_level,
                replicates,
                self.options.multiplier_seed,
            )?
        } else {
            ResponseUncertainty::PointwiseBand {
                level: self.options.confidence_level,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            }
        };
        Ok((
            ResponseValue::Surface {
                grid: Arc::from(grid.to_vec()),
                dimension: 1,
                mean: Arc::from(mean),
            },
            uncertainty,
            support,
        ))
    }

    fn intervention_response(
        &self,
        data: &TabularData,
        outcome: VariableId,
        interventions: &[Intervention],
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        let mut treatments = Vec::with_capacity(interventions.len());
        for intervention in interventions {
            let Some(variable) = intervention.primary_variable() else {
                return Err(EstimationError::unsupported(
                    "intervention-response estimation requires one target per intervention",
                ));
            };
            if treatments.contains(&variable) {
                return Err(EstimationError::unsupported(
                    "intervention-response targets must be unique",
                ));
            }
            if matches!(intervention, Intervention::Soft { .. } | Intervention::Sequence(_)) {
                return Err(EstimationError::unsupported(
                    "soft and sequenced intervention responses require a structural model",
                ));
            }
            treatments.push(variable);
        }
        let sample = CompleteSample::read(data, outcome, &treatments, &self.adjustment_set)?;
        let rows: Vec<usize> = (0..sample.len()).collect();
        let fit = self.fit_outcome(&sample, &rows)?;
        let stochastic = interventions
            .iter()
            .any(|intervention| matches!(intervention, Intervention::Stochastic { .. }));
        let draws = if stochastic { 256 } else { 1 };
        let mut total = 0.0;
        for row_index in 0..sample.len() {
            let factual = sample.raw_row(row_index);
            for draw in 0..draws {
                let mut row = factual.clone();
                for (column, intervention) in interventions.iter().enumerate() {
                    row[column] = intervention_level(intervention, factual[column], draw, column)?;
                }
                total += predict_one(&fit, &row)?;
            }
        }
        let estimate = total / (sample.len() * draws) as f64;
        let minima: Vec<f64> =
            (0..treatments.len()).map(|column| sample.treatment_column_range(column).0).collect();
        let maxima: Vec<f64> =
            (0..treatments.len()).map(|column| sample.treatment_column_range(column).1).collect();
        Ok((
            ResponseValue::Scalar(estimate),
            ResponseUncertainty::None,
            SupportReport {
                status: SupportStatus::Extrapolative,
                query_region: SupportRegion {
                    minima: Arc::from(minima.clone()),
                    maxima: Arc::from(maxima.clone()),
                },
                diagnostics: vec![SupportDiagnostic {
                    id: Arc::from("response.intervention_observed_bounds"),
                    values: Arc::from(minima.into_iter().chain(maxima).collect::<Vec<_>>()),
                    detail: Arc::from(
                        "observed minima followed by maxima; policy support is not certified",
                    ),
                }],
                warnings: vec![Diagnostic::new(
                    "response.intervention_plugin_model_dependent",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "intervention response uses additive-GAM g-computation; joint policy support and statistical uncertainty are not certified",
                )],
            },
        ))
    }

    fn point_derivative(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatment: VariableId,
        at: f64,
        order: u8,
        scale: DerivativeScale,
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        let sample = CompleteSample::read(data, outcome, &[treatment], &self.adjustment_set)?;
        let pseudo = self.cross_fitted_pseudo_outcome(&sample)?;
        let bandwidth = self.options.bandwidth.unwrap_or(silverman_bandwidth(&sample.treatments)?);
        let point = gaussian_local_quadratic(&sample.treatments, &pseudo, at, bandwidth)?;
        let estimate = transform_point_derivative(
            point.value,
            point.first_derivative,
            point.second_derivative,
            at,
            order,
            scale,
        )?;
        let derivative_se = if order == 1 {
            point.first_derivative_standard_error
        } else {
            point.second_derivative_standard_error
        };
        let standard_error = match (order, scale) {
            (1 | 2, DerivativeScale::Identity) => derivative_se,
            (1, DerivativeScale::LogTreatment) => at.abs() * derivative_se,
            // Log-outcome scales and transformed second derivatives also need
            // coefficient covariance; a partial delta interval would overclaim.
            _ => f64::NAN,
        };
        let support = support_report(
            &[at],
            &sample.treatments,
            &[point.local_ess],
            vec![
                point.weight_sum
                    / (sample.len() as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt()),
            ],
            self.options.minimum_local_ess,
        );
        let uncertainty = if standard_error.is_finite() {
            let z = normal_ppf(0.5 + self.options.confidence_level / 2.0);
            ResponseUncertainty::Scalar {
                standard_error,
                level: self.options.confidence_level,
                lower: estimate - z * standard_error,
                upper: estimate + z * standard_error,
            }
        } else {
            ResponseUncertainty::None
        };
        Ok((ResponseValue::Scalar(estimate), uncertainty, support))
    }

    fn average_derivative(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatment: VariableId,
        weighting: &DerivativeWeighting,
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        if !matches!(weighting, DerivativeWeighting::Observed) {
            return Err(EstimationError::unsupported(
                "Gaussian-score Riesz ADE currently supports observed-law weighting only",
            ));
        }
        let sample = CompleteSample::read(data, outcome, &[treatment], &self.adjustment_set)?;
        let scores = self.cross_fitted_ade_scores(&sample)?;
        let effective_n = scores.len() as f64;
        let estimate = scores.iter().sum::<f64>() / effective_n;
        let variance = scores.iter().map(|v| (v - estimate).powi(2)).sum::<f64>()
            / (effective_n - 1.0).max(1.0);
        let se = (variance / effective_n.max(1.0)).sqrt();
        let z = normal_ppf(0.5 + self.options.confidence_level / 2.0);
        let (minimum, maximum) = range(&sample.treatments);
        let support = SupportReport {
            status: if effective_n < self.options.minimum_local_ess {
                SupportStatus::WeakOverlap
            } else {
                SupportStatus::Supported
            },
            query_region: SupportRegion {
                minima: Arc::from([minimum]),
                maxima: Arc::from([maximum]),
            },
            diagnostics: vec![SupportDiagnostic {
                id: Arc::from("response.weighted_effective_sample_size"),
                values: Arc::from([effective_n]),
                detail: Arc::from("Kish effective sample size under derivative target weights"),
            }],
            warnings: Vec::new(),
        };
        Ok((
            ResponseValue::Scalar(estimate),
            ResponseUncertainty::Scalar {
                standard_error: se,
                level: self.options.confidence_level,
                lower: estimate - z * se,
                upper: estimate + z * se,
            },
            support,
        ))
    }

    fn jacobian(
        &self,
        data: &TabularData,
        outcomes: &[VariableId],
        treatments: &[VariableId],
        at: &[f64],
        scale: DerivativeScale,
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        if treatments.len() > MAX_NONPARAMETRIC_RESPONSE_DIM {
            return Err(EstimationError::unsupported(
                "plug-in response Jacobian supports at most two treatments",
            ));
        }
        let mut values = Vec::with_capacity(outcomes.len() * treatments.len());
        let mut all_treatments = Vec::new();
        for &outcome in outcomes {
            let sample = CompleteSample::read(data, outcome, treatments, &self.adjustment_set)?;
            if all_treatments.is_empty() {
                all_treatments.clone_from(&sample.treatment_matrix);
            }
            let (level, gradient) = self.plugin_gradient(&sample, at)?;
            for (j, &raw) in gradient.iter().enumerate() {
                values.push(transform_derivative(raw, at[j], level, scale)?);
            }
        }
        let support = multivariate_support(at, &all_treatments, treatments.len());
        Ok((
            ResponseValue::Jacobian {
                outcomes: outcomes.len(),
                treatments: treatments.len(),
                values: Arc::from(values),
            },
            ResponseUncertainty::None,
            support,
        ))
    }

    fn directional_derivative(
        &self,
        data: &TabularData,
        outcomes: &[VariableId],
        treatments: &[VariableId],
        at: &[f64],
        direction: &[f64],
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        if treatments.len() > MAX_NONPARAMETRIC_RESPONSE_DIM {
            return Err(EstimationError::unsupported(
                "plug-in directional derivative supports at most two treatments",
            ));
        }
        let mut values = Vec::with_capacity(outcomes.len());
        let mut all_treatments = Vec::new();
        for &outcome in outcomes {
            let sample = CompleteSample::read(data, outcome, treatments, &self.adjustment_set)?;
            if all_treatments.is_empty() {
                all_treatments.clone_from(&sample.treatment_matrix);
            }
            let (_, gradient) = self.plugin_gradient(&sample, at)?;
            values.push(gradient.iter().zip(direction).map(|(a, b)| a * b).sum());
        }
        Ok((
            ResponseValue::Vector(Arc::from(values)),
            ResponseUncertainty::None,
            multivariate_support(at, &all_treatments, treatments.len()),
        ))
    }

    fn cross_fitted_pseudo_outcome(
        &self,
        sample: &CompleteSample,
    ) -> Result<Vec<f64>, EstimationError> {
        let n = sample.len();
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        let mut pseudo = vec![0.0; n];
        for fold in 0..self.options.folds {
            let train: Vec<usize> = (0..n).filter(|i| i % self.options.folds != fold).collect();
            let valid: Vec<usize> = (0..n).filter(|i| i % self.options.folds == fold).collect();
            let outcome_fit = self.fit_outcome(sample, &train)?;
            let treatment_fit = self.fit_treatment(sample, &train)?;
            let sigma = treatment_sigma(sample, &train, treatment_fit.as_ref())?;
            for &i in &valid {
                let observed_x = sample.raw_row(i);
                let mu_observed = predict_one(&outcome_fit, &observed_x)?;
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => predict_one(fit, &sample.adjustment_row(i))?,
                    None => sample.train_treatment_mean(&train),
                };
                let conditional_density =
                    gaussian_density(sample.treatment_matrix[i], treatment_mean, sigma).max(1e-8);
                let mut marginal_density = 0.0;
                let mut marginal_mu = 0.0;
                for &j in &train {
                    let treatment_mean_j = match treatment_fit.as_ref() {
                        Some(fit) => predict_one(fit, &sample.adjustment_row(j))?,
                        None => sample.train_treatment_mean(&train),
                    };
                    marginal_density +=
                        gaussian_density(sample.treatment_matrix[i], treatment_mean_j, sigma);
                    let mut row = sample.raw_row(j);
                    row[0] = sample.treatment_matrix[i];
                    marginal_mu += predict_one(&outcome_fit, &row)?;
                }
                marginal_density /= train.len() as f64;
                marginal_mu /= train.len() as f64;
                pseudo[i] = marginal_mu
                    + (sample.outcome[i] - mu_observed) * marginal_density / conditional_density;
            }
        }
        Ok(pseudo)
    }

    fn cross_fitted_ade_scores(
        &self,
        sample: &CompleteSample,
    ) -> Result<Vec<f64>, EstimationError> {
        let n = sample.len();
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        let mut scores = vec![0.0; n];
        let treatment_range = range(&sample.treatments).1 - range(&sample.treatments).0;
        let step = (treatment_range * 1e-4).max(1e-7);
        for fold in 0..self.options.folds {
            let train: Vec<usize> = (0..n).filter(|i| i % self.options.folds != fold).collect();
            let outcome_fit = self.fit_outcome(sample, &train)?;
            let treatment_fit = self.fit_treatment(sample, &train)?;
            let sigma = treatment_sigma(sample, &train, treatment_fit.as_ref())?;
            let treatment_mean_constant = sample.train_treatment_mean(&train);
            for i in (0..n).filter(|i| i % self.options.folds == fold) {
                let row = sample.raw_row(i);
                let mu = predict_one(&outcome_fit, &row)?;
                let mut plus = row.clone();
                let mut minus = row.clone();
                plus[0] += step;
                minus[0] -= step;
                let derivative = (predict_one(&outcome_fit, &plus)?
                    - predict_one(&outcome_fit, &minus)?)
                    / (2.0 * step);
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => predict_one(fit, &sample.adjustment_row(i))?,
                    None => treatment_mean_constant,
                };
                let riesz = (sample.treatments[i] - treatment_mean) / (sigma * sigma);
                scores[i] = derivative + riesz * (sample.outcome[i] - mu);
            }
        }
        Ok(scores)
    }

    fn plugin_gradient(
        &self,
        sample: &CompleteSample,
        at: &[f64],
    ) -> Result<(f64, Vec<f64>), EstimationError> {
        let rows: Vec<usize> = (0..sample.len()).collect();
        let fit = self.fit_outcome(sample, &rows)?;
        let mut base_sum = 0.0;
        let mut gradient = vec![0.0; at.len()];
        for i in 0..sample.len() {
            let mut row = sample.raw_row(i);
            row[..at.len()].copy_from_slice(at);
            base_sum += predict_one(&fit, &row)?;
            for j in 0..at.len() {
                let range_j = sample.treatment_column_range(j);
                let step = ((range_j.1 - range_j.0) * 1e-4).max(1e-7);
                let mut plus = row.clone();
                let mut minus = row.clone();
                plus[j] += step;
                minus[j] -= step;
                gradient[j] +=
                    (predict_one(&fit, &plus)? - predict_one(&fit, &minus)?) / (2.0 * step);
            }
        }
        let n = sample.len() as f64;
        Ok((base_sum / n, gradient.into_iter().map(|v| v / n).collect()))
    }

    fn fit_outcome(
        &self,
        sample: &CompleteSample,
        rows: &[usize],
    ) -> Result<antecedent_stats::GamFit, EstimationError> {
        let x = sample.raw_subset(rows);
        let y: Vec<f64> = rows.iter().map(|&i| sample.outcome[i]).collect();
        fit_additive(
            &x,
            rows.len(),
            sample.raw_cols,
            &y,
            self.options.nuisance_basis,
            self.options.nuisance_lambda,
        )
    }

    fn fit_treatment(
        &self,
        sample: &CompleteSample,
        rows: &[usize],
    ) -> Result<Option<antecedent_stats::GamFit>, EstimationError> {
        if sample.adjustment_cols == 0 {
            return Ok(None);
        }
        let x = sample.adjustment_subset(rows);
        let y: Vec<f64> = rows.iter().map(|&i| sample.treatments[i]).collect();
        fit_additive(
            &x,
            rows.len(),
            sample.adjustment_cols,
            &y,
            self.options.nuisance_basis,
            self.options.nuisance_lambda,
        )
        .map(Some)
    }
}

#[derive(Clone, Debug)]
struct CompleteSample {
    outcome: Vec<f64>,
    treatments: Vec<f64>,
    treatment_matrix: Vec<f64>,
    adjustment: Vec<f64>,
    treatment_cols: usize,
    adjustment_cols: usize,
    raw_cols: usize,
}

impl CompleteSample {
    fn read(
        data: &TabularData,
        outcome: VariableId,
        treatments: &[VariableId],
        adjustment: &[VariableId],
    ) -> Result<Self, EstimationError> {
        let duplicate_treatment =
            treatments.iter().enumerate().any(|(i, value)| treatments[i + 1..].contains(value));
        let duplicate_adjustment =
            adjustment.iter().enumerate().any(|(i, value)| adjustment[i + 1..].contains(value));
        if treatments.is_empty()
            || duplicate_treatment
            || duplicate_adjustment
            || adjustment.iter().any(|v| treatments.contains(v) || *v == outcome)
        {
            return Err(EstimationError::unsupported(
                "treatments and adjustments must be unique and adjustments distinct from outcome",
            ));
        }
        let y = data.float64_values(outcome)?;
        let treatment_values: Vec<Vec<f64>> =
            treatments.iter().map(|&v| data.float64_values(v)).collect::<Result<_, _>>()?;
        let adjustment_values: Vec<Vec<f64>> =
            adjustment.iter().map(|&v| data.float64_values(v)).collect::<Result<_, _>>()?;
        let keep: Vec<usize> = (0..data.row_count())
            .filter(|&i| {
                y[i].is_finite()
                    && treatment_values.iter().all(|c| c[i].is_finite())
                    && adjustment_values.iter().all(|c| c[i].is_finite())
            })
            .collect();
        if keep.len() < 20 {
            return Err(EstimationError::unsupported(
                "continuous-response estimation requires at least 20 complete rows",
            ));
        }
        let outcome = keep.iter().map(|&i| y[i]).collect();
        let mut treatment_matrix = Vec::with_capacity(keep.len() * treatments.len());
        for column in &treatment_values {
            treatment_matrix.extend(keep.iter().map(|&i| column[i]));
        }
        let treatments = treatment_matrix[..keep.len()].to_vec();
        let mut adjustment_matrix = Vec::with_capacity(keep.len() * adjustment.len());
        for column in &adjustment_values {
            adjustment_matrix.extend(keep.iter().map(|&i| column[i]));
        }
        Ok(Self {
            outcome,
            treatments,
            treatment_matrix,
            adjustment: adjustment_matrix,
            treatment_cols: treatment_values.len(),
            adjustment_cols: adjustment_values.len(),
            raw_cols: treatment_values.len() + adjustment_values.len(),
        })
    }

    fn len(&self) -> usize {
        self.outcome.len()
    }

    fn raw_row(&self, row: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.raw_cols);
        for col in 0..self.treatment_cols {
            out.push(self.treatment_matrix[col * self.len() + row]);
        }
        out.extend(self.adjustment_row(row));
        out
    }

    fn adjustment_row(&self, row: usize) -> Vec<f64> {
        (0..self.adjustment_cols).map(|col| self.adjustment[col * self.len() + row]).collect()
    }

    fn raw_subset(&self, rows: &[usize]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows.len() * self.raw_cols);
        for col in 0..self.treatment_cols {
            out.extend(rows.iter().map(|&row| self.treatment_matrix[col * self.len() + row]));
        }
        for col in 0..self.adjustment_cols {
            out.extend(rows.iter().map(|&row| self.adjustment[col * self.len() + row]));
        }
        out
    }

    fn adjustment_subset(&self, rows: &[usize]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows.len() * self.adjustment_cols);
        for col in 0..self.adjustment_cols {
            out.extend(rows.iter().map(|&row| self.adjustment[col * self.len() + row]));
        }
        out
    }

    fn train_treatment_mean(&self, rows: &[usize]) -> f64 {
        rows.iter().map(|&i| self.treatments[i]).sum::<f64>() / rows.len() as f64
    }

    fn treatment_column_range(&self, col: usize) -> (f64, f64) {
        range(&self.treatment_matrix[col * self.len()..(col + 1) * self.len()])
    }
}

fn fit_additive(
    x: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    basis: usize,
    lambda: f64,
) -> Result<antecedent_stats::GamFit, EstimationError> {
    let specs: Vec<SmoothSpec> =
        (0..ncols).map(|col| SmoothSpec::new(col, basis, lambda)).collect();
    let mut workspace = GamWorkspace::default();
    Ok(fit_gam(x, nrows, ncols, y, &specs, &GamOptions::default(), &FaerBackend, &mut workspace)?)
}

fn predict_one(fit: &antecedent_stats::GamFit, raw_row: &[f64]) -> Result<f64, EstimationError> {
    Ok(predict_gam(fit, raw_row, 1, raw_row.len())?[0])
}

fn treatment_sigma(
    sample: &CompleteSample,
    train: &[usize],
    fit: Option<&antecedent_stats::GamFit>,
) -> Result<f64, EstimationError> {
    let mean = sample.train_treatment_mean(train);
    let rss = if let Some(fit) = fit {
        fit.residuals.iter().map(|v| v * v).sum::<f64>()
    } else {
        train.iter().map(|&i| (sample.treatments[i] - mean).powi(2)).sum()
    };
    let sigma = (rss / (train.len().saturating_sub(1).max(1) as f64)).sqrt();
    if !sigma.is_finite() || sigma <= f64::EPSILON {
        return Err(EstimationError::unsupported(
            "Gaussian treatment nuisance has degenerate residual variance",
        ));
    }
    Ok(sigma)
}

fn ensure_crossfit_size(n: usize, folds: usize, basis: usize) -> Result<(), EstimationError> {
    if folds > n {
        return Err(EstimationError::unsupported(
            "cross-fitting folds cannot exceed complete observations",
        ));
    }
    let smallest_train = n - n.div_ceil(folds);
    if smallest_train <= basis + 2 {
        return Err(EstimationError::unsupported(
            "too few complete rows for requested cross-fitting and nuisance basis",
        ));
    }
    Ok(())
}

fn simultaneous_multiplier_band(
    mean: &[f64],
    influences: &[Vec<f64>],
    standard_errors: &[f64],
    level: f64,
    replicates: u32,
    seed: u64,
) -> Result<ResponseUncertainty, EstimationError> {
    let Some(sample_size) = influences.first().map(Vec::len) else {
        return Err(EstimationError::unsupported(
            "simultaneous bands require a non-empty response grid",
        ));
    };
    if influences.iter().any(|row| row.len() != sample_size)
        || standard_errors.iter().any(|se| !se.is_finite() || *se <= f64::EPSILON)
    {
        return Err(EstimationError::unsupported(
            "simultaneous bands require finite non-degenerate influence standard errors",
        ));
    }
    let mut state = seed;
    let mut maxima = Vec::with_capacity(replicates as usize);
    let mut multipliers = vec![0.0; sample_size];
    for _ in 0..replicates {
        for multiplier in &mut multipliers {
            state = splitmix64(state);
            *multiplier = if state & 1 == 0 { -1.0 } else { 1.0 };
        }
        let maximum = influences
            .iter()
            .zip(standard_errors)
            .map(|(coordinate, se)| {
                coordinate
                    .iter()
                    .zip(&multipliers)
                    .map(|(influence, multiplier)| influence * multiplier)
                    .sum::<f64>()
                    .abs()
                    / se
            })
            .fold(0.0_f64, f64::max);
        maxima.push(maximum);
    }
    maxima.sort_by(f64::total_cmp);
    let mut index = 0usize;
    while index + 1 < maxima.len()
        && f64::from(u32::try_from(index + 1).unwrap_or(u32::MAX)) / f64::from(replicates) < level
    {
        index += 1;
    }
    let critical = maxima[index];
    let lower = mean
        .iter()
        .zip(standard_errors)
        .map(|(estimate, se)| estimate - critical * se)
        .collect::<Vec<_>>();
    let upper = mean
        .iter()
        .zip(standard_errors)
        .map(|(estimate, se)| estimate + critical * se)
        .collect::<Vec<_>>();
    Ok(ResponseUncertainty::SimultaneousBand {
        level,
        lower: Arc::from(lower),
        upper: Arc::from(upper),
        replicates,
    })
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn intervention_level(
    intervention: &Intervention,
    factual: f64,
    draw: usize,
    policy_index: usize,
) -> Result<f64, EstimationError> {
    let numeric = |value: &antecedent_core::Value| {
        value.as_f64().filter(|number| number.is_finite()).ok_or_else(|| {
            EstimationError::unsupported("intervention response requires finite numeric values")
        })
    };
    let value = match intervention {
        Intervention::Set { value, .. } => numeric(value)?,
        Intervention::Shift { delta, .. } => factual + numeric(delta)?,
        Intervention::Stochastic { policy, .. } => {
            // A distinct deterministic SplitMix64 stream coordinate for every
            // (Monte Carlo draw, policy) pair avoids silently imposing a shared
            // rank/comonotone coupling across joint stochastic interventions.
            let state = (draw as u64)
                .wrapping_mul(0xD2B7_4407_B1CE_6E93)
                .wrapping_add((policy_index as u64).wrapping_mul(0xCA5A_8263_9512_1157))
                .wrapping_add(0xA17E_CEDE_0500_0001);
            let random = splitmix64(state);
            let quantile = ((random >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0);
            match policy {
                StochasticPolicy::Bernoulli { p } => {
                    if quantile < *p {
                        1.0
                    } else {
                        0.0
                    }
                }
                StochasticPolicy::Gaussian { mean, variance } => {
                    if !mean.is_finite() {
                        return Err(EstimationError::unsupported(
                            "Gaussian intervention mean must be finite",
                        ));
                    }
                    mean + variance.sqrt() * normal_ppf(quantile)
                }
                StochasticPolicy::Categorical { probs } => {
                    let total: f64 = probs.iter().sum();
                    let threshold = quantile * total;
                    let mut cumulative = 0.0;
                    let mut category = probs.len() - 1;
                    for (index, probability) in probs.iter().enumerate() {
                        cumulative += probability;
                        if threshold < cumulative {
                            category = index;
                            break;
                        }
                    }
                    category as f64
                }
                _ => {
                    return Err(EstimationError::unsupported(
                        "unsupported stochastic intervention policy",
                    ));
                }
            }
        }
        Intervention::Soft { .. } | Intervention::Sequence(_) => {
            return Err(EstimationError::unsupported(
                "soft and sequenced intervention responses require a structural model",
            ));
        }
        _ => {
            return Err(EstimationError::unsupported("unsupported intervention-response policy"));
        }
    };
    if !value.is_finite() {
        return Err(EstimationError::unsupported(
            "intervention response produced a non-finite treatment value",
        ));
    }
    Ok(value)
}

fn transform_derivative(
    derivative: f64,
    treatment: f64,
    response: f64,
    scale: DerivativeScale,
) -> Result<f64, EstimationError> {
    let value = match scale {
        DerivativeScale::Identity => derivative,
        DerivativeScale::LogTreatment => treatment * derivative,
        DerivativeScale::LogOutcome => {
            if response <= 0.0 {
                return Err(EstimationError::unsupported(
                    "log-outcome derivative scale requires a positive fitted response",
                ));
            }
            derivative / response
        }
        DerivativeScale::LogLog => {
            if response <= 0.0 {
                return Err(EstimationError::unsupported(
                    "elasticity requires a positive fitted response",
                ));
            }
            treatment * derivative / response
        }
    };
    Ok(value)
}

fn transform_point_derivative(
    response: f64,
    first: f64,
    second: f64,
    treatment: f64,
    order: u8,
    scale: DerivativeScale,
) -> Result<f64, EstimationError> {
    if order == 1 {
        return transform_derivative(first, treatment, response, scale);
    }
    if matches!(scale, DerivativeScale::LogOutcome | DerivativeScale::LogLog) && response <= 0.0 {
        return Err(EstimationError::unsupported(
            "log-outcome derivative scale requires a positive fitted response",
        ));
    }
    Ok(match scale {
        DerivativeScale::Identity => second,
        DerivativeScale::LogTreatment => treatment * first + treatment * treatment * second,
        DerivativeScale::LogOutcome => second / response - (first / response).powi(2),
        DerivativeScale::LogLog => {
            treatment * first / response
                + treatment * treatment * (second / response - (first / response).powi(2))
        }
    })
}

fn support_report(
    points: &[f64],
    observed: &[f64],
    ess: &[f64],
    density: Vec<f64>,
    minimum_ess: f64,
) -> SupportReport {
    let (minimum, maximum) = range(observed);
    let outside = points.iter().any(|v| *v < minimum || *v > maximum);
    let weak = ess.iter().any(|v| *v < minimum_ess);
    let status = if outside {
        SupportStatus::OutsideEmpiricalSupport
    } else if weak {
        SupportStatus::WeakOverlap
    } else {
        SupportStatus::Supported
    };
    let warnings = if outside {
        vec![Diagnostic::new(
            "response.outside_empirical_support",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "at least one requested response coordinate is outside observed treatment support",
        )]
    } else if weak {
        vec![Diagnostic::new(
            "response.weak_local_overlap",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "at least one requested response coordinate has low local effective sample size",
        )]
    } else {
        Vec::new()
    };
    SupportReport {
        status,
        query_region: SupportRegion {
            minima: Arc::from([points.iter().copied().fold(f64::INFINITY, f64::min)]),
            maxima: Arc::from([points.iter().copied().fold(f64::NEG_INFINITY, f64::max)]),
        },
        diagnostics: vec![
            SupportDiagnostic {
                id: Arc::from("response.local_ess"),
                values: Arc::from(ess.to_vec()),
                detail: Arc::from("Kish effective sample size of Gaussian local weights"),
            },
            SupportDiagnostic {
                id: Arc::from("response.local_density"),
                values: Arc::from(density),
                detail: Arc::from("Gaussian-kernel marginal treatment-density estimate"),
            },
        ],
        warnings,
    }
}

fn multivariate_support(at: &[f64], treatment_matrix: &[f64], dimensions: usize) -> SupportReport {
    let n = treatment_matrix.len() / dimensions;
    let mut minima = Vec::with_capacity(dimensions);
    let mut maxima = Vec::with_capacity(dimensions);
    let mut outside = false;
    for (j, &point) in at.iter().enumerate() {
        let (lo, hi) = range(&treatment_matrix[j * n..(j + 1) * n]);
        minima.push(lo);
        maxima.push(hi);
        outside |= point < lo || point > hi;
    }
    SupportReport {
        status: if outside {
            SupportStatus::OutsideEmpiricalSupport
        } else {
            SupportStatus::Extrapolative
        },
        query_region: SupportRegion {
            minima: Arc::from(at.to_vec()),
            maxima: Arc::from(at.to_vec()),
        },
        diagnostics: vec![SupportDiagnostic {
            id: Arc::from("response.marginal_observed_bounds"),
            values: Arc::from(minima.into_iter().chain(maxima).collect::<Vec<_>>()),
            detail: Arc::from(
                "per-treatment minima followed by maxima; joint support is not established",
            ),
        }],
        warnings: vec![Diagnostic::new(
            "response.plugin_jacobian_model_dependent",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "multivariate derivative uses an additive GAM plug-in and marginal support checks",
        )],
    }
}

fn range(values: &[f64]) -> (f64, f64) {
    values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        ContinuousDomain, GridSpec, Intervention, ResponseFunctional, ResponseQuery,
        StochasticPolicy, Value,
    };
    use antecedent_data::TabularData;

    use super::*;

    fn confounded_curve(n: usize) -> (TabularData, VariableId, VariableId, VariableId) {
        let mut a = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            let noise = ((i * 37 % 101) as f64 / 100.0 - 0.5) * 0.3;
            let treatment = 0.7 * z + noise;
            x.push(z);
            a.push(treatment);
            y.push(1.0 + 2.0 * treatment + 0.8 * z + 0.05 * (i as f64).sin());
        }
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        (data, VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2))
    }

    #[test]
    fn dr_curve_calibrates_linear_response_and_reports_support() {
        let (data, a, y, x) = confounded_curve(500);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
        });
        let response = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            response.estimate
        else {
            panic!("expected surface");
        };
        assert!((mean[2] - mean[0] - 1.6).abs() < 0.25, "means={mean:?}");
        assert_ne!(response.support.status, SupportStatus::OutsideEmpiricalSupport);
        assert_eq!(response.support.diagnostics[0].values.len(), 3);
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.kennedy_dr");
    }

    #[test]
    fn simultaneous_band_is_deterministic_and_contains_pointwise_curve() {
        let (data, a, y, x) = confounded_curve(500);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
        });
        let mut estimator = ContinuousResponseEstimator::new([x]);
        estimator.options.bandwidth = Some(0.35);
        estimator.options.simultaneous_replicates = Some(200);
        let first = estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let second = estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        assert_eq!(first, second);
        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            &first.estimate
        else {
            panic!("expected surface");
        };
        let ResponseUncertainty::SimultaneousBand { lower, upper, replicates, .. } =
            &first.uncertainty
        else {
            panic!("expected simultaneous band");
        };
        assert_eq!(*replicates, 200);
        assert!(
            mean.iter().zip(lower.iter()).zip(upper.iter()).all(|((m, lo), hi)| lo < m && m < hi)
        );
        assert_eq!(first.provenance_id.as_ref(), "estimate.response.kennedy_dr_simultaneous");
    }

    #[test]
    fn intervention_response_executes_set_shift_and_stochastic_policies() {
        let (data, a, y, x) = confounded_curve(500);
        let estimate = |intervention| {
            let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: Arc::from([intervention]),
            });
            ContinuousResponseEstimator::new([x])
                .estimate_identified(
                    &data,
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                )
                .unwrap()
        };
        let set = estimate(Intervention::set(a, Value::f64(0.25)));
        let shift = estimate(Intervention::shift(a, Value::f64(0.25)));
        let stochastic =
            estimate(Intervention::stochastic(a, StochasticPolicy::gaussian(0.25, 0.01)));
        let scalar = |response: &CausalResponse| match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
            _ => panic!("expected scalar"),
        };
        assert!((scalar(&set) - 1.5).abs() < 0.2);
        assert!((scalar(&stochastic) - scalar(&set)).abs() < 0.1);
        assert!((scalar(&shift) - 1.5).abs() < 0.2);
        assert_eq!(set.support.status, SupportStatus::Extrapolative);
        assert_eq!(set.provenance_id.as_ref(), "estimate.response.intervention_gcomp");
    }

    #[test]
    fn intervention_response_fails_closed_for_soft_interventions() {
        let (data, a, y, x) = confounded_curve(100);
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([Intervention::soft(
                a,
                antecedent_core::MechanismOverride::named("replacement", Arc::from([])),
            )]),
        });
        let error = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("structural model"));
    }

    #[test]
    fn riesz_ade_calibrates_linear_slope() {
        let (data, a, y, x) = confounded_curve(600);
        let query = ResponseQuery::new(ResponseFunctional::AverageDerivative {
            outcome: y,
            treatment: a,
            weighting: DerivativeWeighting::Observed,
        });
        let response = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
            response.estimate
        else {
            panic!("expected scalar");
        };
        assert!((value - 2.0).abs() < 0.2, "ade={value}");
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.riesz_ade");
    }

    #[test]
    fn point_elasticity_applies_scale_transform() {
        let (data, a, y, x) = confounded_curve(500);
        let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: y,
            treatment: a,
            at: 0.4,
            order: 1,
            scale: DerivativeScale::LogLog,
        });
        let response = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
            response.estimate
        else {
            panic!("expected scalar");
        };
        assert!(value.is_finite() && value > 0.2 && value < 0.7, "elasticity={value}");
    }

    #[test]
    fn second_derivative_uses_chain_rule_on_log_log_scale() {
        // m(a)=a² at a=2: d² log(m)/d(log(a))² = 0.
        let value =
            transform_point_derivative(4.0, 4.0, 2.0, 2.0, 2, DerivativeScale::LogLog).unwrap();
        assert!(value.abs() < 1e-12);
    }

    #[test]
    fn plugin_jacobian_recovers_low_dimensional_slopes() {
        let n = 400;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut y1 = Vec::with_capacity(n);
        let mut y2 = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            let av = z + 0.2 * (i as f64 * 0.7).sin();
            let bv = -0.4 * z + 0.3 * (i as f64 * 1.1).cos();
            a.push(av);
            b.push(bv);
            x.push(z);
            y1.push(1.0 + 2.0 * av - 0.5 * bv + z);
            y2.push(-1.0 + 0.25 * av + 1.5 * bv - 0.7 * z);
        }
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("y1", y1.as_slice()),
            ("y2", y2.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let query = ResponseQuery::new(ResponseFunctional::Jacobian {
            outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            at: Arc::from([0.0, 0.0]),
            scale: DerivativeScale::Identity,
        });
        let response = ContinuousResponseEstimator::new([VariableId::from_raw(4)])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::IdentifiedUnderParametricRestrictions,
                AssumptionSet::new(),
            )
            .unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values, .. }) =
            response.estimate
        else {
            panic!("expected Jacobian");
        };
        for (got, expected) in values.iter().zip([2.0, -0.5, 0.25, 1.5]) {
            assert!((got - expected).abs() < 0.25, "got={got}, expected={expected}");
        }
        assert_eq!(response.support.status, SupportStatus::Extrapolative);
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.gam_derivative");
    }
}
