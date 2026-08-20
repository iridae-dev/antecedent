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
    FaerBackend, GamOptions, GamWorkspace, LocalQuadraticWorkspace, SmoothSpec, StatsError,
    fit_gam, gaussian_density, gaussian_local_quadratic,
    gaussian_local_quadratic_influence_prechecked, normal_ppf, silverman_bandwidth,
};

use crate::EstimationError;
use crate::util::range;

/// Lower clamp on the fitted conditional treatment density in the Kennedy weight.
///
/// A clamped row has an unbounded inverse weight, so every clamp is counted and
/// surfaced as a positivity diagnostic rather than absorbed silently.
const CONDITIONAL_DENSITY_FLOOR: f64 = 1e-8;

/// Largest cartesian-product support the exact discrete intervention mixture will evaluate.
///
/// Every combination costs one GAM prediction per row, so this bounds the work at roughly
/// the same order as the Monte-Carlo path it replaced. Joint interventions on a handful of
/// binary or small-categorical variables stay well inside it.
const MAX_EXACT_MIXTURE_COMBINATIONS: usize = 4096;

/// Cross-fitted Kennedy pseudo-outcome with its positivity accounting.
struct PseudoOutcome {
    values: Vec<f64>,
    density_floor_rows: usize,
}

/// Cross-fitted average-derivative scores with the Riesz representer that built them.
struct AverageDerivativeScores {
    scores: Vec<f64>,
    riesz_weights: Vec<f64>,
}

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
    /// Export per-row pseudo-outcomes and influence values as support diagnostics.
    ///
    /// Off by default; the estimate is identical either way. When set, three
    /// channels are appended to `support.diagnostics`, for `N` retained rows
    /// and `G` grid points:
    ///
    /// - `response.row_index` — 0-based positions of retained rows in the data
    ///   as received (after caller preprocessing, before the all-finite row
    ///   filter); exact integers stored as `f64`.
    /// - `response.row_pseudo_outcome` — cross-fitted Kennedy pseudo-outcomes
    ///   `φ_i`, outcome units, aligned with `row_index`. Fold = retained-row
    ///   position mod `folds`; nuisances for row `i` exclude `i`'s fold.
    /// - `response.row_influence` — `G * N`, grid-major (`value[g*N + i]`):
    ///   the local-WLS influence of row `i` on the fitted level at grid point
    ///   `g`, `ψ = w_i · [(XᵀWX)⁻¹]₀ · x_i · (φ_i − x_iᵀβ̂)`. Sums to zero per
    ///   grid point (WLS normal equation); reported pointwise
    ///   `SE(g) = √(Σ_i ψ²)`, and both bands are `m̂ ± c·SE` with these values.
    ///
    /// These are diagnostics conditional on the estimator's construction: the
    /// pseudo-outcomes are treated as fixed data, so nuisance and bandwidth
    /// uncertainty are NOT inside the influences, and `ψ` is not the Kennedy
    /// estimator's semiparametric efficient influence function. Channel ids,
    /// alignment, and layout are stable; values may change when the internal
    /// construction changes. Full contract:
    /// `docs/causal-responses.md#row-diagnostic-export-contract`.
    pub export_row_diagnostics: bool,
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
            export_row_diagnostics: false,
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
        if matches!(query.functional, ResponseFunctional::PointDerivative { .. })
            && o.bandwidth.is_none()
        {
            // Silverman's rule is a level/KDE rate. Using it for m'/m'' oversmooths the
            // derivative toward zero while still publishing Identity-scale sandwich SEs —
            // the same silent-undersmoothing hole simultaneous bands already refuse.
            return Err(EstimationError::unsupported(
                "point derivatives require an explicit bandwidth; Silverman's rule is not an undersmoothing rule for m' or m''",
            ));
        }
        if o.folds < 2
            || o.nuisance_basis < 4
            || !o.nuisance_lambda.is_finite()
            || o.nuisance_lambda < 0.0
            || o.bandwidth.is_some_and(|h| !h.is_finite() || h <= 0.0)
            || !o.minimum_local_ess.is_finite()
            || o.minimum_local_ess <= 0.0
            || !o.confidence_level.is_finite()
            // A level of exactly 0 yields z = 0 and a zero-width interval that would be
            // reported as a band at level 0 rather than refused.
            || o.confidence_level <= 0.0
            || o.confidence_level >= 1.0
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
        let PseudoOutcome { values: pseudo, density_floor_rows } =
            self.cross_fitted_pseudo_outcome(&sample)?;
        let bandwidth = self.options.bandwidth.unwrap_or(silverman_bandwidth(&sample.treatments)?);
        let mut mean = Vec::with_capacity(grid.len());
        let mut lower = Vec::with_capacity(grid.len());
        let mut upper = Vec::with_capacity(grid.len());
        let mut ess = Vec::with_capacity(grid.len());
        let mut density = Vec::with_capacity(grid.len());
        let mut influences = Vec::with_capacity(grid.len());
        let mut robust_se = Vec::with_capacity(grid.len());
        let z = normal_ppf(0.5 + self.options.confidence_level / 2.0);
        if sample.treatments.iter().chain(&pseudo).any(|v| !v.is_finite()) {
            return Err(
                StatsError::Shape { message: "local quadratic inputs must be finite" }.into()
            );
        }
        let mut local = LocalQuadraticWorkspace::default();
        for &at in grid {
            let fit = gaussian_local_quadratic_influence_prechecked(
                &mut local,
                &sample.treatments,
                &pseudo,
                at,
                bandwidth,
            )?;
            let point = fit.point;
            mean.push(point.value);
            // The pointwise band uses the same influence-based standard error the
            // simultaneous band standardizes by, so the two are nested by
            // construction rather than being two different variance estimates at
            // the same nominal level.
            lower.push(point.value - z * fit.robust_standard_error);
            upper.push(point.value + z * fit.robust_standard_error);
            ess.push(point.local_ess);
            density.push(
                point.weight_sum
                    / (sample.len() as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt()),
            );
            influences.push(fit.influences);
            robust_se.push(fit.robust_standard_error);
        }
        let mut support = support_report(
            grid,
            &sample.treatments,
            &ess,
            density,
            self.options.minimum_local_ess,
            density_floor_rows,
        );
        if self.options.export_row_diagnostics {
            let n = sample.len();
            let flat_influences: Vec<f64> =
                influences.iter().flat_map(|row| row.iter().copied()).collect();
            // Downstream result layers require finite diagnostic values; refuse
            // rather than publish a non-finite influence into the export channel.
            if flat_influences.iter().any(|value| !value.is_finite()) {
                return Err(EstimationError::unsupported(
                    "row-diagnostic export encountered a non-finite influence value",
                ));
            }
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from("response.row_index"),
                values: Arc::from(
                    sample.keep.iter().map(|&index| index as f64).collect::<Vec<_>>(),
                ),
                detail: Arc::from("original dataframe row index of each retained complete row"),
            });
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from("response.row_pseudo_outcome"),
                values: Arc::from(pseudo),
                detail: Arc::from("cross-fitted Kennedy pseudo-outcome per retained row"),
            });
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from("response.row_influence"),
                values: Arc::from(flat_influences),
                detail: Arc::from(format!(
                    "row-major by grid point, grid_len={}, n={n}: value[g*N + i]",
                    grid.len()
                )),
            });
        }
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
        let mut gam_ws = GamWorkspace::default();
        let fit = self.fit_outcome(&sample, &rows, &mut gam_ws)?;
        // Discrete policies (Set/Shift/Bernoulli/Categorical) are integrated exactly as a
        // finite mixture. Monte Carlo through a continuous spline would treat categorical
        // codes as ordered coordinates and approximate a sum that has a closed form.
        let estimate = if interventions.iter().any(intervention_needs_monte_carlo) {
            let draws = 256;
            let mut total = 0.0;
            let mut factual = vec![0.0; sample.raw_cols];
            let mut row = vec![0.0; sample.raw_cols];
            for row_index in 0..sample.len() {
                sample.write_raw_row(row_index, &mut factual);
                for draw in 0..draws {
                    row.copy_from_slice(&factual);
                    for (column, intervention) in interventions.iter().enumerate() {
                        row[column] =
                            intervention_level(intervention, factual[column], draw, column)?;
                    }
                    total += predict_one(&fit, &row)?;
                }
            }
            total / (sample.len() * draws) as f64
        } else {
            exact_discrete_intervention_mean(&fit, &sample, interventions)?
        };
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
        let PseudoOutcome { values: pseudo, density_floor_rows } =
            self.cross_fitted_pseudo_outcome(&sample)?;
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
        let mut support = support_report(
            &[at],
            &sample.treatments,
            &[point.local_ess],
            vec![
                point.weight_sum
                    / (sample.len() as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt()),
            ],
            self.options.minimum_local_ess,
            density_floor_rows,
        );
        if !standard_error.is_finite() {
            // Withholding the interval is deliberate, but a silently absent interval is
            // indistinguishable from one the caller never asked for. Say why.
            support.warnings.push(Diagnostic::new(
                "response.derivative_interval_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "no interval is reported for this derivative order and scale: the delta-method transform needs the full coefficient covariance, and a partial interval would understate uncertainty",
            ));
        }
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
        let AverageDerivativeScores { scores, riesz_weights } =
            self.cross_fitted_ade_scores(&sample)?;
        let rows = scores.len() as f64;
        let estimate = scores.iter().sum::<f64>() / rows;
        let variance = scores.iter().map(|v| (v - estimate).powi(2)).sum::<f64>() / (rows - 1.0);
        let se = (variance / rows).sqrt();
        let z = normal_ppf(0.5 + self.options.confidence_level / 2.0);
        let (minimum, maximum) = range(&sample.treatments);
        // Kish effective sample size of the Riesz representer. The raw row count is
        // not a weighted ESS: it cannot fall when the representer concentrates on a
        // handful of treatment-tail rows, which is exactly the failure this reports.
        let absolute_sum = riesz_weights.iter().map(|weight| weight.abs()).sum::<f64>();
        let square_sum = riesz_weights.iter().map(|weight| weight * weight).sum::<f64>();
        let effective_n =
            if square_sum > 0.0 { absolute_sum * absolute_sum / square_sum } else { 0.0 };
        let weak = effective_n < self.options.minimum_local_ess;
        let support = SupportReport {
            status: if weak { SupportStatus::WeakOverlap } else { SupportStatus::Supported },
            query_region: SupportRegion {
                minima: Arc::from([minimum]),
                maxima: Arc::from([maximum]),
            },
            diagnostics: vec![SupportDiagnostic {
                id: Arc::from("response.weighted_effective_sample_size"),
                values: Arc::from([effective_n, rows]),
                detail: Arc::from(
                    "Kish effective sample size of the Riesz representer, then complete rows",
                ),
            }],
            warnings: if weak {
                vec![Diagnostic::new(
                    "response.weak_riesz_overlap",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "the average-derivative Riesz representer concentrates on few rows; the estimate is driven by the treatment tail",
                )]
            } else {
                Vec::new()
            },
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
    ) -> Result<PseudoOutcome, EstimationError> {
        let n = sample.len();
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        let mut pseudo = vec![0.0; n];
        let mut density_floor_rows = 0usize;
        let mut gam_ws = GamWorkspace::default();
        let mut adj_row = vec![0.0; sample.adjustment_cols];
        let mut raw_row = vec![0.0; sample.raw_cols];
        for fold in 0..self.options.folds {
            let train: Vec<usize> = (0..n).filter(|i| i % self.options.folds != fold).collect();
            let valid: Vec<usize> = (0..n).filter(|i| i % self.options.folds == fold).collect();
            let outcome_fit = self.fit_outcome(sample, &train, &mut gam_ws)?;
            let treatment_fit = self.fit_treatment(sample, &train, &mut gam_ws)?;
            let sigma = treatment_sigma(sample, &train, treatment_fit.as_ref())?;
            // The training-row treatment means do not depend on the validation row;
            // computing them once per fold avoids |valid| x |train| spline expansions.
            let constant_treatment_mean = sample.train_treatment_mean(&train);
            let train_treatment_means: Vec<f64> = match treatment_fit.as_ref() {
                Some(fit) => {
                    let mut means = Vec::with_capacity(train.len());
                    for &j in &train {
                        sample.write_adjustment_row(j, &mut adj_row);
                        means.push(predict_one(fit, &adj_row)?);
                    }
                    means
                }
                None => vec![constant_treatment_mean; train.len()],
            };
            // The outcome nuisance is additive, so a counterfactual prediction
            // decomposes exactly: μ̂(a, X_j) = μ̂(A_j, X_j) − f_T(A_j) + f_T(a),
            // where f_T is the centered treatment smooth. Averaging over the
            // training rows therefore needs one covariate offset per fold plus a
            // single smooth evaluation per validation row — O(n) per fold instead
            // of the |valid|×|train| full-prediction double loop.
            let treat_smooth = outcome_fit.smooth_for_raw_col(0).ok_or_else(|| {
                EstimationError::unsupported("outcome nuisance is missing its treatment smooth")
            })?;
            let mut covariate_offset = 0.0;
            for (position, &j) in train.iter().enumerate() {
                let treat_partial =
                    outcome_fit.smooth_partial(treat_smooth, sample.treatments[j])?;
                covariate_offset += outcome_fit.fitted[position] - treat_partial;
            }
            covariate_offset /= train.len() as f64;
            for &i in &valid {
                sample.write_raw_row(i, &mut raw_row);
                let mu_observed = predict_one(&outcome_fit, &raw_row)?;
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => {
                        sample.write_adjustment_row(i, &mut adj_row);
                        predict_one(fit, &adj_row)?
                    }
                    None => constant_treatment_mean,
                };
                let raw_density =
                    gaussian_density(sample.treatment_matrix[i], treatment_mean, sigma);
                if !raw_density.is_finite() || raw_density <= CONDITIONAL_DENSITY_FLOOR {
                    density_floor_rows += 1;
                }
                let conditional_density = raw_density.max(CONDITIONAL_DENSITY_FLOOR);
                let mut marginal_density = 0.0;
                for &train_mean in &train_treatment_means {
                    marginal_density +=
                        gaussian_density(sample.treatment_matrix[i], train_mean, sigma);
                }
                marginal_density /= train.len() as f64;
                let marginal_mu = covariate_offset
                    + outcome_fit.smooth_partial(treat_smooth, sample.treatment_matrix[i])?;
                pseudo[i] = marginal_mu
                    + (sample.outcome[i] - mu_observed) * marginal_density / conditional_density;
            }
        }
        Ok(PseudoOutcome { values: pseudo, density_floor_rows })
    }

    fn cross_fitted_ade_scores(
        &self,
        sample: &CompleteSample,
    ) -> Result<AverageDerivativeScores, EstimationError> {
        let n = sample.len();
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        let mut scores = vec![0.0; n];
        let mut riesz_weights = vec![0.0; n];
        let mut gam_ws = GamWorkspace::default();
        let mut row = vec![0.0; sample.raw_cols];
        let mut plus = vec![0.0; sample.raw_cols];
        let mut minus = vec![0.0; sample.raw_cols];
        let mut adj_row = vec![0.0; sample.adjustment_cols];
        let treatment_range = range(&sample.treatments).1 - range(&sample.treatments).0;
        for fold in 0..self.options.folds {
            let train: Vec<usize> = (0..n).filter(|i| i % self.options.folds != fold).collect();
            let outcome_fit = self.fit_outcome(sample, &train, &mut gam_ws)?;
            let treatment_fit = self.fit_treatment(sample, &train, &mut gam_ws)?;
            let sigma = treatment_sigma(sample, &train, treatment_fit.as_ref())?;
            let treatment_mean_constant = sample.train_treatment_mean(&train);
            for i in (0..n).filter(|i| i % self.options.folds == fold) {
                sample.write_raw_row(i, &mut row);
                let mu = predict_one(&outcome_fit, &row)?;
                let step = finite_difference_step(sample.treatments[i], treatment_range);
                plus.copy_from_slice(&row);
                minus.copy_from_slice(&row);
                plus[0] += step;
                minus[0] -= step;
                let gap = plus[0] - minus[0];
                if !gap.is_finite() || gap <= 0.0 {
                    return Err(EstimationError::unsupported(
                        "finite-difference step collapsed under floating-point precision",
                    ));
                }
                let derivative =
                    (predict_one(&outcome_fit, &plus)? - predict_one(&outcome_fit, &minus)?) / gap;
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => {
                        sample.write_adjustment_row(i, &mut adj_row);
                        predict_one(fit, &adj_row)?
                    }
                    None => treatment_mean_constant,
                };
                let riesz = (sample.treatments[i] - treatment_mean) / (sigma * sigma);
                riesz_weights[i] = riesz;
                scores[i] = derivative + riesz * (sample.outcome[i] - mu);
            }
        }
        Ok(AverageDerivativeScores { scores, riesz_weights })
    }

    fn plugin_gradient(
        &self,
        sample: &CompleteSample,
        at: &[f64],
    ) -> Result<(f64, Vec<f64>), EstimationError> {
        let rows: Vec<usize> = (0..sample.len()).collect();
        let mut gam_ws = GamWorkspace::default();
        let fit = self.fit_outcome(sample, &rows, &mut gam_ws)?;
        let mut base_sum = 0.0;
        let mut gradient = vec![0.0; at.len()];
        let mut row = vec![0.0; sample.raw_cols];
        let mut plus = vec![0.0; sample.raw_cols];
        let mut minus = vec![0.0; sample.raw_cols];
        for i in 0..sample.len() {
            sample.write_raw_row(i, &mut row);
            row[..at.len()].copy_from_slice(at);
            base_sum += predict_one(&fit, &row)?;
            for j in 0..at.len() {
                let range_j = sample.treatment_column_range(j);
                let step = finite_difference_step(at[j], range_j.1 - range_j.0);
                plus.copy_from_slice(&row);
                minus.copy_from_slice(&row);
                plus[j] += step;
                minus[j] -= step;
                let gap = plus[j] - minus[j];
                if !gap.is_finite() || gap <= 0.0 {
                    return Err(EstimationError::unsupported(
                        "finite-difference step collapsed under floating-point precision",
                    ));
                }
                gradient[j] += (predict_one(&fit, &plus)? - predict_one(&fit, &minus)?) / gap;
            }
        }
        let n = sample.len() as f64;
        Ok((base_sum / n, gradient.into_iter().map(|v| v / n).collect()))
    }

    fn fit_outcome(
        &self,
        sample: &CompleteSample,
        rows: &[usize],
        workspace: &mut GamWorkspace,
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
            workspace,
        )
    }

    fn fit_treatment(
        &self,
        sample: &CompleteSample,
        rows: &[usize],
        workspace: &mut GamWorkspace,
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
            workspace,
        )
        .map(Some)
    }
}

#[derive(Clone, Debug)]
struct CompleteSample {
    /// Original dataframe row index of each retained complete row.
    keep: Vec<usize>,
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
            keep,
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

    #[cfg(test)]
    fn raw_row(&self, row: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.raw_cols];
        self.write_raw_row(row, &mut out);
        out
    }

    fn write_raw_row(&self, row: usize, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.raw_cols);
        // Column-major treatment/adjustment layouts: index by col deliberately.
        #[allow(clippy::needless_range_loop)]
        for col in 0..self.treatment_cols {
            out[col] = self.treatment_matrix[col * self.len() + row];
        }
        #[allow(clippy::needless_range_loop)]
        for col in 0..self.adjustment_cols {
            out[self.treatment_cols + col] = self.adjustment[col * self.len() + row];
        }
    }

    fn write_adjustment_row(&self, row: usize, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.adjustment_cols);
        #[allow(clippy::needless_range_loop)]
        for col in 0..self.adjustment_cols {
            out[col] = self.adjustment[col * self.len() + row];
        }
    }

    #[cfg(test)]
    fn adjustment_row(&self, row: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.adjustment_cols];
        self.write_adjustment_row(row, &mut out);
        out
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
    workspace: &mut GamWorkspace,
) -> Result<antecedent_stats::GamFit, EstimationError> {
    let specs: Vec<SmoothSpec> =
        (0..ncols).map(|col| SmoothSpec::new(col, basis, lambda)).collect();
    // Response nuisances use a longer backfitting budget than the GAM default: the default
    // 100-iteration / 1e-6 tolerance combination routinely returns converged=false on the
    // cross-fitted Kennedy fixtures while the fit itself is already stable enough to use.
    let fit = fit_gam(
        x,
        nrows,
        ncols,
        y,
        &specs,
        &GamOptions { max_iter: 500, tol: 1e-6 },
        &FaerBackend,
        workspace,
    )?;
    // Observation logistics already call `GlmFit::require_ok`. An unfinished backfit after the
    // extended budget is refused rather than published into a Kennedy curve or ADE.
    if !fit.converged {
        return Err(EstimationError::unsupported(
            "additive GAM nuisance did not converge; refuse rather than publish an unfinished fit",
        ));
    }
    Ok(fit)
}

fn predict_one(fit: &antecedent_stats::GamFit, raw_row: &[f64]) -> Result<f64, EstimationError> {
    Ok(fit.predict_row(raw_row)?)
}

fn treatment_sigma(
    sample: &CompleteSample,
    train: &[usize],
    fit: Option<&antecedent_stats::GamFit>,
) -> Result<f64, EstimationError> {
    let mean = sample.train_treatment_mean(train);
    let (rss, denominator) = if let Some(fit) = fit {
        // The residual scale of a penalized GAM uses effective degrees of freedom, not
        // n−1. Dividing by n−1 understates σ whenever edf > 1, which peaks the Kennedy
        // conditional density and inflates the Gaussian-score Riesz representer.
        let df = (train.len() as f64 - fit.edf_approx).max(1.0);
        (fit.residuals.iter().map(|v| v * v).sum::<f64>(), df)
    } else {
        (
            train.iter().map(|&i| (sample.treatments[i] - mean).powi(2)).sum(),
            train.len().saturating_sub(1).max(1) as f64,
        )
    };
    let sigma = (rss / denominator).sqrt();
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

fn intervention_needs_monte_carlo(intervention: &Intervention) -> bool {
    matches!(
        intervention,
        Intervention::Stochastic { policy: StochasticPolicy::Gaussian { .. }, .. }
    )
}

/// One atom of a discrete intervention law: absolute level, or additive shift of the factual.
#[derive(Clone, Copy, Debug)]
enum DiscreteAtom {
    /// Absolute treatment level with mixture weight.
    Level { value: f64, weight: f64 },
    /// Additive shift of the unit's factual treatment (weight is always 1).
    Shift { delta: f64 },
}

/// Exact finite-mixture g-computation for Set/Shift/Bernoulli/Categorical policies.
///
/// Bernoulli and Categorical are summed over their support with the declared probabilities
/// rather than Monte-Carlo sampled through a continuous smoother. That avoids treating
/// unordered category codes as ordered coordinates along a spline.
fn exact_discrete_intervention_mean(
    fit: &antecedent_stats::GamFit,
    sample: &CompleteSample,
    interventions: &[Intervention],
) -> Result<f64, EstimationError> {
    let supports: Vec<Vec<DiscreteAtom>> =
        interventions.iter().map(discrete_intervention_support).collect::<Result<_, _>>()?;
    // The mixture is a cartesian product across interventions, so its cost is exponential in
    // how many discrete policies are joined. The Monte-Carlo path it replaced was bounded at
    // a fixed draw count, so without a budget here a query that used to return in
    // milliseconds can run for hours. Refuse rather than silently reverting to an
    // approximation the caller did not ask for.
    let combinations = supports
        .iter()
        .try_fold(1usize, |product, support| product.checked_mul(support.len()))
        .filter(|product| *product <= MAX_EXACT_MIXTURE_COMBINATIONS);
    if combinations.is_none() {
        return Err(EstimationError::unsupported(
            "joint discrete intervention support exceeds the exact-mixture budget; intervene on fewer variables or coarsen the category supports",
        ));
    }
    let mut total = 0.0;
    let mut row = vec![0.0; sample.raw_cols];
    for row_index in 0..sample.len() {
        sample.write_raw_row(row_index, &mut row);
        total += mixture_expectation(fit, &mut row, &supports, 0, 1.0)?;
    }
    Ok(total / sample.len() as f64)
}

fn discrete_intervention_support(
    intervention: &Intervention,
) -> Result<Vec<DiscreteAtom>, EstimationError> {
    let numeric = |value: &antecedent_core::Value| {
        value.as_f64().filter(|number| number.is_finite()).ok_or_else(|| {
            EstimationError::unsupported("intervention response requires finite numeric values")
        })
    };
    match intervention {
        Intervention::Set { value, .. } => {
            Ok(vec![DiscreteAtom::Level { value: numeric(value)?, weight: 1.0 }])
        }
        Intervention::Shift { delta, .. } => {
            Ok(vec![DiscreteAtom::Shift { delta: numeric(delta)? }])
        }
        Intervention::Stochastic { policy: StochasticPolicy::Bernoulli { p }, .. } => {
            if !p.is_finite() || !(0.0..=1.0).contains(p) {
                return Err(EstimationError::unsupported(
                    "Bernoulli intervention probability must lie in [0, 1]",
                ));
            }
            Ok([(0.0, 1.0 - p), (1.0, *p)]
                .into_iter()
                .filter(|(_, w)| *w > 0.0)
                .map(|(value, weight)| DiscreteAtom::Level { value, weight })
                .collect())
        }
        Intervention::Stochastic { policy: StochasticPolicy::Categorical { probs }, .. } => {
            let total: f64 = probs.iter().sum();
            if !total.is_finite()
                || total <= 0.0
                || probs.iter().any(|p| !p.is_finite() || *p < 0.0)
            {
                return Err(EstimationError::unsupported(
                    "Categorical intervention probabilities must be finite and non-negative",
                ));
            }
            Ok(probs
                .iter()
                .enumerate()
                .filter(|(_, p)| **p > 0.0)
                .map(|(index, p)| DiscreteAtom::Level { value: index as f64, weight: p / total })
                .collect())
        }
        _ => Err(EstimationError::unsupported(
            "exact discrete intervention mixture does not cover this policy",
        )),
    }
}

fn mixture_expectation(
    fit: &antecedent_stats::GamFit,
    row: &mut [f64],
    supports: &[Vec<DiscreteAtom>],
    column: usize,
    weight: f64,
) -> Result<f64, EstimationError> {
    if !(weight.is_finite() && weight >= 0.0) {
        return Err(EstimationError::unsupported(
            "intervention mixture weight must be finite and non-negative",
        ));
    }
    if column == supports.len() {
        return Ok(weight * predict_one(fit, row)?);
    }
    let mut sum = 0.0;
    let factual = row[column];
    for atom in &supports[column] {
        let saved = row[column];
        let branch = match *atom {
            DiscreteAtom::Level { value, weight: atom_weight } => {
                row[column] = value;
                atom_weight
            }
            DiscreteAtom::Shift { delta } => {
                row[column] = factual + delta;
                1.0
            }
        };
        sum += mixture_expectation(fit, row, supports, column + 1, weight * branch)?;
        row[column] = saved;
    }
    Ok(sum)
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
    density_floor_rows: usize,
) -> SupportReport {
    let (minimum, maximum) = range(observed);
    let outside = points.iter().any(|v| *v < minimum || *v > maximum);
    let weak = ess.iter().any(|v| *v < minimum_ess);
    // A clamped conditional density is a positivity failure on the nuisance side:
    // it does not move the requested coordinate outside the observed range, but the
    // curve is then driven by an inverse weight the data never supported.
    let clamped = density_floor_rows > 0;
    let status = if outside {
        SupportStatus::OutsideEmpiricalSupport
    } else if weak || clamped {
        SupportStatus::WeakOverlap
    } else {
        SupportStatus::Supported
    };
    let mut warnings = Vec::new();
    if outside {
        warnings.push(Diagnostic::new(
            "response.outside_empirical_support",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "at least one requested response coordinate is outside observed treatment support",
        ));
    } else if weak {
        warnings.push(Diagnostic::new(
            "response.weak_local_overlap",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "at least one requested response coordinate has low local effective sample size",
        ));
    }
    if clamped {
        warnings.push(Diagnostic::new(
            "response.conditional_density_floored",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "at least one row hit the conditional treatment-density floor; the doubly robust weight for those rows is bounded by the floor, not estimated from data",
        ));
    }
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
            SupportDiagnostic {
                id: Arc::from("response.conditional_density_floor_rows"),
                values: Arc::from([density_floor_rows as f64]),
                detail: Arc::from(
                    "rows whose fitted conditional treatment density hit the positivity floor",
                ),
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
        warnings: {
            let mut warnings = vec![Diagnostic::new(
                "response.plugin_jacobian_model_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "multivariate derivative uses an additive GAM plug-in and marginal support checks",
            )];
            if outside {
                // The cubic B-spline basis is clamped at its boundary knots, so the
                // fitted surface is constant outside the fitted range and its finite
                // difference is identically zero there. That zero is a property of the
                // basis, not evidence of a flat response, and must not be read as one.
                warnings.push(Diagnostic::new(
                    "response.clamped_basis_derivative",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "at least one coordinate is outside the fitted range, where the clamped spline basis makes the plug-in derivative exactly zero by construction",
                ));
            }
            warnings
        },
    }
}


/// Finite-difference step that stays above floating-point ulp at `|at|`.
///
/// A step of only `range · 1e-4` (floored at 1e-7) collapses when `|at|` is huge relative to
/// the observed range: `at ± step` rounds to `at`, the central difference is exactly zero, and
/// a slope of 2 can be reported as 0 with a normal-looking SE.
fn finite_difference_step(at: f64, treatment_range: f64) -> f64 {
    let scale = at.abs().max(treatment_range.abs()).max(1.0);
    (treatment_range.abs() * 1e-4).max(scale * 1e-8).max(1e-7)
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
    fn pseudo_outcome_additive_hoist_matches_brute_force_double_loop() {
        // The O(n) covariate-offset form must agree with the definitional
        // |valid|×|train| double loop (full counterfactual prediction per pair)
        // up to floating-point re-association.
        let (data, a, y, x) = confounded_curve(160);
        let estimator = ContinuousResponseEstimator::new([x]);
        let sample = CompleteSample::read(&data, y, &[a], &estimator.adjustment_set).unwrap();
        let fast = estimator.cross_fitted_pseudo_outcome(&sample).unwrap();

        let n = sample.len();
        let folds = estimator.options.folds;
        let mut brute = vec![0.0; n];
        for fold in 0..folds {
            let train: Vec<usize> = (0..n).filter(|i| i % folds != fold).collect();
            let valid: Vec<usize> = (0..n).filter(|i| i % folds == fold).collect();
            let mut gam_ws = GamWorkspace::default();
            let outcome_fit = estimator.fit_outcome(&sample, &train, &mut gam_ws).unwrap();
            let treatment_fit = estimator.fit_treatment(&sample, &train, &mut gam_ws).unwrap();
            let sigma = treatment_sigma(&sample, &train, treatment_fit.as_ref()).unwrap();
            let constant_mean = sample.train_treatment_mean(&train);
            for &i in &valid {
                let mu_observed = predict_one(&outcome_fit, &sample.raw_row(i)).unwrap();
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => predict_one(fit, &sample.adjustment_row(i)).unwrap(),
                    None => constant_mean,
                };
                let raw_density =
                    gaussian_density(sample.treatment_matrix[i], treatment_mean, sigma);
                let conditional_density = raw_density.max(CONDITIONAL_DENSITY_FLOOR);
                let mut marginal_density = 0.0;
                let mut marginal_mu = 0.0;
                for &j in &train {
                    let mean_j = match treatment_fit.as_ref() {
                        Some(fit) => predict_one(fit, &sample.adjustment_row(j)).unwrap(),
                        None => constant_mean,
                    };
                    marginal_density += gaussian_density(sample.treatment_matrix[i], mean_j, sigma);
                    let mut row = sample.raw_row(j);
                    row[0] = sample.treatment_matrix[i];
                    marginal_mu += predict_one(&outcome_fit, &row).unwrap();
                }
                marginal_density /= train.len() as f64;
                marginal_mu /= train.len() as f64;
                brute[i] = marginal_mu
                    + (sample.outcome[i] - mu_observed) * marginal_density / conditional_density;
            }
        }
        for (i, (&fast_i, &brute_i)) in fast.values.iter().zip(&brute).enumerate() {
            assert!(
                (fast_i - brute_i).abs() <= 1e-9 * brute_i.abs().max(1.0),
                "row {i}: fast={fast_i} brute={brute_i}"
            );
        }
    }

    #[test]
    fn simultaneous_band_is_never_narrower_than_the_pointwise_band() {
        let (data, a, y, x) = confounded_curve(500);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
        });
        let run = |replicates: Option<u32>| {
            let mut estimator = ContinuousResponseEstimator::new([x]);
            estimator.options.bandwidth = Some(0.35);
            estimator.options.simultaneous_replicates = replicates;
            estimator
                .estimate_identified(
                    &data,
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                )
                .unwrap()
        };
        let ResponseUncertainty::PointwiseBand { lower: p_lo, upper: p_hi, .. } =
            run(None).uncertainty
        else {
            panic!("expected pointwise band");
        };
        let ResponseUncertainty::SimultaneousBand { lower: s_lo, upper: s_hi, .. } =
            run(Some(400)).uncertainty
        else {
            panic!("expected simultaneous band");
        };
        // Both bands standardize by the same influence-based standard error, so the
        // simultaneous band can never be the tighter statement at the same level.
        for index in 0..3 {
            assert!(
                s_lo[index] <= p_lo[index] + 1e-12 && s_hi[index] >= p_hi[index] - 1e-12,
                "simultaneous band narrower than pointwise at {index}"
            );
        }
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
    fn joint_discrete_mixture_beyond_the_budget_is_refused() {
        // The mixture is a cartesian product, so cost is exponential in the number of joined
        // discrete policies. Four 14-level categoricals is already ~38k combinations per row
        // — seconds of work — and nothing stops a caller going further. Refuse instead of
        // running for hours or silently falling back to Monte Carlo.
        let (data, a, y, x) = confounded_curve(200);
        let run = |levels: usize| {
            let probs: Arc<[f64]> = Arc::from(vec![1.0 / levels as f64; levels]);
            let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: Arc::from([Intervention::stochastic(
                    a,
                    StochasticPolicy::Categorical { probs },
                )]),
            });
            ContinuousResponseEstimator::new([x]).estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
        };
        assert!(run(MAX_EXACT_MIXTURE_COMBINATIONS).is_ok());
        let error = run(MAX_EXACT_MIXTURE_COMBINATIONS + 1).unwrap_err();
        assert!(error.to_string().contains("exact-mixture budget"), "got {error}");
    }

    #[test]
    fn categorical_intervention_is_an_exact_finite_mixture_not_monte_carlo() {
        // Under an additive model, a Categorical policy on codes {0,1} with weights
        // (0.25, 0.75) must equal 0.25·μ(0) + 0.75·μ(1). Monte Carlo through a continuous
        // spline only approximates that sum and would treat the codes as ordered coordinates.
        let (data, a, y, x) = confounded_curve(500);
        let estimate = |intervention| {
            let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: Arc::from([intervention]),
            });
            let response = ContinuousResponseEstimator::new([x])
                .estimate_identified(
                    &data,
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                )
                .unwrap();
            match response.estimate {
                ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => value,
                _ => panic!("expected scalar"),
            }
        };
        let at0 = estimate(Intervention::set(a, Value::f64(0.0)));
        let at1 = estimate(Intervention::set(a, Value::f64(1.0)));
        let mixture =
            estimate(Intervention::stochastic(a, StochasticPolicy::categorical([0.25, 0.75])));
        let expected = 0.25 * at0 + 0.75 * at1;
        assert!(
            (mixture - expected).abs() < 1e-10,
            "categorical g-comp={mixture}, exact mixture={expected}"
        );
    }

    #[test]
    fn treatment_sigma_uses_gam_edf_not_n_minus_one() {
        // Discriminating residual-scale check: with edf > 1, dividing by n−1 understates σ.
        let n = 80usize;
        let mut treatments = Vec::with_capacity(n);
        let mut adjustment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        for i in 0..n {
            let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            adjustment.push(z);
            treatments.push(0.8 * z + 0.1 * (i as f64 * 0.3).sin());
            outcome.push(1.0 + treatments[i] + 0.5 * z);
        }
        let sample = CompleteSample {
            keep: (0..n).collect(),
            outcome,
            treatments: treatments.clone(),
            treatment_matrix: treatments,
            adjustment,
            treatment_cols: 1,
            adjustment_cols: 1,
            raw_cols: 2,
        };
        let train: Vec<usize> = (0..n).collect();
        let mut gam_ws = GamWorkspace::default();
        let fit = ContinuousResponseEstimator::new([VariableId::from_raw(0)])
            .fit_treatment(&sample, &train, &mut gam_ws)
            .unwrap()
            .expect("adjustment present");
        assert!(fit.edf_approx > 1.0 + 1e-6, "edf={}", fit.edf_approx);
        let got = treatment_sigma(&sample, &train, Some(&fit)).unwrap();
        let rss: f64 = fit.residuals.iter().map(|r| r * r).sum();
        let wrong = (rss / (n - 1) as f64).sqrt();
        let right = (rss / (n as f64 - fit.edf_approx).max(1.0)).sqrt();
        assert!(
            (got - right).abs() < 1e-12,
            "treatment_sigma={got} should use edf denominator ({right})"
        );
        assert!(
            (got - wrong).abs() > 1e-6,
            "edf and n-1 denominators coincide; the test cannot discriminate"
        );
        assert!(got > wrong, "edf-aware σ must exceed the n-1 understatement");
    }

    #[test]
    fn row_diagnostics_export_is_opt_in_and_aligned_to_retained_rows() {
        let (data, a, y, x) = confounded_curve(200);
        // Poison one row so a retained-row index gap is observable in the export.
        let mut columns: Vec<Vec<f64>> =
            (0..3).map(|c| data.float64_values(VariableId::from_raw(c)).unwrap()).collect();
        columns[1][7] = f64::NAN;
        let data = TabularData::from_f64_columns([
            ("a", columns[0].as_slice()),
            ("y", columns[1].as_slice()),
            ("x", columns[2].as_slice()),
        ])
        .unwrap();
        let grid = [-0.4, 0.0, 0.4];
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from(grid))),
        });
        let run = |export: bool| {
            let mut estimator = ContinuousResponseEstimator::new([x]);
            estimator.options.export_row_diagnostics = export;
            estimator
                .estimate_identified(
                    &data,
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                )
                .unwrap()
        };
        let off = run(false);
        assert!(
            off.support.diagnostics.iter().all(|d| !d.id.starts_with("response.row_")),
            "row diagnostics must be absent when the flag is off"
        );
        let on = run(true);
        let diagnostic = |id: &str| {
            on.support
                .diagnostics
                .iter()
                .find(|d| d.id.as_ref() == id)
                .unwrap_or_else(|| panic!("missing diagnostic {id}"))
        };
        let n = 199; // one NaN row dropped
        let row_index = diagnostic("response.row_index");
        assert_eq!(row_index.values.len(), n);
        let expected: Vec<f64> = (0..200).filter(|&i| i != 7).map(f64::from).collect();
        assert_eq!(row_index.values.as_ref(), expected.as_slice());
        let pseudo = diagnostic("response.row_pseudo_outcome");
        assert_eq!(pseudo.values.len(), n);
        assert!(pseudo.values.iter().all(|v| v.is_finite()));
        let influence = diagnostic("response.row_influence");
        assert_eq!(influence.values.len(), grid.len() * n);
        assert!(influence.values.iter().all(|v| v.is_finite()));
        assert!(influence.detail.contains("grid_len=3"));
        // The export must not perturb the estimate itself.
        assert_eq!(off.estimate, on.estimate);
        assert_eq!(off.uncertainty, on.uncertainty);
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
        let mut estimator = ContinuousResponseEstimator::new([x]);
        estimator.options.bandwidth = Some(0.35);
        let response = estimator
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
    fn point_derivative_refuses_default_silverman_bandwidth() {
        let (data, a, y, x) = confounded_curve(200);
        let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: y,
            treatment: a,
            at: 0.0,
            order: 1,
            scale: DerivativeScale::Identity,
        });
        let error = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("explicit bandwidth"), "got {error}");
    }

    #[test]
    fn finite_difference_step_scales_with_absolute_level() {
        // At |A|=1e12 with a tiny observed range, the old range·1e-4 floor of 1e-7 is
        // smaller than a ulp and collapses at±step to at.
        let step = finite_difference_step(1e12, 1.0);
        assert!(step > 1e-7);
        let plus = 1e12 + step;
        let minus = 1e12 - step;
        assert!((plus - minus).abs() > 0.0);
        assert!(plus > minus);
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
