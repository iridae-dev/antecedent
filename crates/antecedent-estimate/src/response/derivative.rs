//! Point, average, Jacobian and directional response derivatives.

use std::sync::Arc;

use antecedent_core::{
    DerivativeScale, DerivativeWeighting, Diagnostic, DiagnosticKind, DiagnosticScope,
    DiagnosticSeverity, ResponseUncertainty, ResponseValue, SupportDiagnostic, SupportRegion,
    SupportReport, SupportStatus, VariableId,
};
use antecedent_data::TabularData;
use antecedent_stats::{GamWorkspace, normal_ppf, silverman_bandwidth};

use crate::EstimationError;
use crate::util::range;

use super::{
    AverageDerivativeScores, ContinuousResponseEstimator, CrossFitFold, PseudoOutcome,
    bias_corrected_interval_note, delta_method_interval_note, delta_method_standard_error,
    ensure_crossfit_size, fieller_elasticity_interval, fieller_interval_note, local_design_refusal,
    multivariate_support, plugin::PluginGradientKind, plugin::plugin_gradient_interval_withheld,
    plugin::predict_one, push_outcome_tail_diagnostic, sample::CompleteSample, support_report,
    transform_derivative, transform_point_derivative, transform_point_derivative_gradient,
    treatment_sigma,
};

impl ContinuousResponseEstimator {
    #[allow(
        clippy::too_many_lines,
        reason = "local fit, bias-corrected coordinate, then the delta-method and Fieller interval choice; the stages share the local influence"
    )]
    pub(super) fn point_derivative(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatment: VariableId,
        at: f64,
        order: u8,
        scale: DerivativeScale,
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        let sample = CompleteSample::read(data, outcome, &[treatment], &self.adjustment_set)?;
        let PseudoOutcome { values: pseudo, density_floor_rows, .. } =
            self.cross_fitted_pseudo_outcome(&sample)?;
        let bandwidth = self.options.bandwidth.unwrap_or(silverman_bandwidth(&sample.treatments)?);
        let local = antecedent_stats::gaussian_local_quadratic_influence(
            &sample.treatments,
            &pseudo,
            at,
            bandwidth,
        )
        .map_err(|err| local_design_refusal(err, &sample.treatments, at, bandwidth))?;
        let point = local.point;
        let estimate = transform_point_derivative(
            point.value,
            point.first_derivative,
            point.second_derivative,
            at,
            order,
            scale,
        )?;
        // The published interval is robust bias-corrected (CCT, pilot bandwidth
        // = h): centered at the higher-order coordinate (local-cubic slope for
        // order 1, local-quartic curvature for order 2) and studentized by its
        // own sandwich. The conventional local-quadratic interval ignores the
        // O(h²·m''') smoothing bias of m̂' (and the O(h²·m'''') bias of m̂'')
        // and under-covers the true derivative at an MSE-sized bandwidth
        // (tests/derivative_calibration.rs).
        let corrected = antecedent_stats::gaussian_local_quadratic_bias_corrected(
            &sample.treatments,
            &pseudo,
            at,
            bandwidth,
            None,
        )
        .map_err(|err| local_design_refusal(err, &sample.treatments, at, bandwidth))?;
        let corrected_estimate = transform_point_derivative(
            corrected.value,
            corrected.first_derivative,
            corrected.second_derivative,
            at,
            order,
            scale,
        )?;
        let derivative_se = if order == 1 {
            corrected.robust_first_derivative_standard_error
        } else {
            corrected.robust_second_derivative_standard_error
        };
        // A directly published coordinate SE (identity level/curvature, or the
        // log-treatment order-1 scale-up `|at|·SE(m')`) versus a nonlinear
        // transform of the coordinates that takes the full delta method.
        let delta_transformed = !matches!(
            (order, scale),
            (1 | 2, DerivativeScale::Identity) | (1, DerivativeScale::LogTreatment)
        );
        let standard_error = match (order, scale) {
            (1 | 2, DerivativeScale::Identity) => derivative_se,
            (1, DerivativeScale::LogTreatment) => at.abs() * derivative_se,
            // Every other order/scale is a nonlinear transform of the local
            // coordinates θ = (m, m', m''); its interval is the full delta method
            // on the joint coordinate covariance Σ_θ, not a partial one. The
            // gradient is evaluated at the bias-corrected coordinates the interval
            // is centred on; Σ_θ uses the same bias-corrected local-cubic
            // slope and local-quartic level/curvature as that center.
            _ => {
                let gradient = transform_point_derivative_gradient(
                    corrected.value,
                    corrected.first_derivative,
                    corrected.second_derivative,
                    at,
                    order,
                    scale,
                );
                delta_method_standard_error(&corrected.coefficient_covariance, &gradient)
            }
        };
        // A ratio of estimated coordinates needs denominator uncertainty in
        // the interval itself. A symmetric delta interval can badly under-cover
        // when the response level is small even though that level is positive.
        // Fieller inverts the joint normal test for a*m'/m. If its confidence
        // set is unbounded, a finite scalar interval cannot represent it.
        let fieller = (order == 1 && scale == DerivativeScale::LogLog).then(|| {
            fieller_elasticity_interval(
                corrected.value,
                corrected.first_derivative,
                at,
                &corrected.coefficient_covariance,
                normal_ppf(0.5 + self.options.confidence_level / 2.0),
            )
        });
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
        push_outcome_tail_diagnostic(&mut support, &sample.outcome);
        if !standard_error.is_finite() {
            // The interval is published for every order and scale now; a
            // non-finite delta-method variance is a degenerate local fit (a
            // rank-deficient covariance or a transform singularity), not a
            // deliberate withholding. Say why the one interval is absent.
            support.warnings.push(Diagnostic::new(
                "response.derivative_interval_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "no interval is reported: the delta-method variance of this transformed derivative is not finite at this coordinate (a degenerate local covariance or a transform singularity)",
            ));
        }
        let uncertainty = if standard_error.is_finite()
            && fieller.as_ref().is_some_and(Option::is_none)
        {
            support.warnings.push(Diagnostic::new(
                "response.derivative_interval_unbounded",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "the Fieller confidence set for elasticity is unbounded because the response level is not separated from zero by its joint-covariance confidence region; no finite scalar interval is reported",
            ));
            ResponseUncertainty::None
        } else if standard_error.is_finite() {
            if fieller.is_some() {
                support.warnings.push(fieller_interval_note(estimate, corrected_estimate));
            } else if delta_transformed {
                support.warnings.push(delta_method_interval_note(estimate, corrected_estimate));
            } else {
                support.warnings.push(bias_corrected_interval_note(
                    false,
                    estimate,
                    corrected_estimate,
                ));
            }
            let z = normal_ppf(0.5 + self.options.confidence_level / 2.0);
            let (lower, upper) = fieller.flatten().unwrap_or((
                corrected_estimate - z * standard_error,
                corrected_estimate + z * standard_error,
            ));
            ResponseUncertainty::Scalar {
                standard_error,
                level: self.options.confidence_level,
                lower,
                upper,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            }
        } else {
            ResponseUncertainty::None
        };
        Ok((ResponseValue::Scalar(estimate), uncertainty, support))
    }

    pub(super) fn average_derivative(
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
        let mut support = SupportReport {
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
                scope: DiagnosticScope::Global,
            }],
            warnings: if weak {
                vec![Diagnostic::new(
                    "response.weak_riesz_overlap",
                    DiagnosticKind::Support,
                    DiagnosticSeverity::Warning,
                    "the average-derivative Riesz representer concentrates on few rows; the estimate is driven by the treatment tail",
                )]
            } else {
                Vec::new()
            },
            point_status: None,
        };
        push_outcome_tail_diagnostic(&mut support, &sample.outcome);
        Ok((
            ResponseValue::Scalar(estimate),
            ResponseUncertainty::Scalar {
                standard_error: se,
                level: self.options.confidence_level,
                lower: estimate - z * se,
                upper: estimate + z * se,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            },
            support,
        ))
    }

    pub(super) fn jacobian(
        &self,
        data: &TabularData,
        outcomes: &[VariableId],
        treatments: &[VariableId],
        at: &[f64],
        scale: DerivativeScale,
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        let run = self.plugin_gradient_run(
            data,
            outcomes,
            treatments,
            at,
            PluginGradientKind::Jacobian,
            false,
        )?;
        let mut values = Vec::with_capacity(outcomes.len() * treatments.len());
        for (level, gradient) in run.levels.iter().zip(&run.gradients) {
            for (j, &raw) in gradient.iter().enumerate() {
                values.push(transform_derivative(raw, at[j], *level, scale)?);
            }
        }
        let mut support =
            multivariate_support(at, &run.all_treatments, treatments.len(), values.len());
        support.warnings.push(plugin_gradient_interval_withheld());
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

    pub(super) fn directional_derivative(
        &self,
        data: &TabularData,
        outcomes: &[VariableId],
        treatments: &[VariableId],
        at: &[f64],
        direction: &[f64],
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        let run = self.plugin_gradient_run(
            data,
            outcomes,
            treatments,
            at,
            PluginGradientKind::Directional,
            false,
        )?;
        let values: Vec<f64> = run
            .gradients
            .iter()
            .map(|gradient| gradient.iter().zip(direction).map(|(a, b)| a * b).sum())
            .collect();
        let mut support =
            multivariate_support(at, &run.all_treatments, treatments.len(), values.len());
        support.warnings.push(plugin_gradient_interval_withheld());
        Ok((ResponseValue::Vector(Arc::from(values)), ResponseUncertainty::None, support))
    }

    pub(super) fn cross_fitted_ade_scores(
        &self,
        sample: &CompleteSample,
    ) -> Result<AverageDerivativeScores, EstimationError> {
        let n = sample.len();
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        let mut scores = vec![0.0; n];
        let mut riesz_weights = vec![0.0; n];
        let mut gam_ws = GamWorkspace::default();
        let mut row = vec![0.0; sample.raw_cols];
        let mut adj_row = vec![0.0; sample.adjustment_cols];
        for fold in 0..self.options.folds {
            let train: Vec<usize> = (0..n).filter(|i| i % self.options.folds != fold).collect();
            let outcome_fit = self.fit_outcome(sample, &train, &mut gam_ws)?;
            let treatment_fit = self.fit_treatment(sample, &train, &mut gam_ws)?;
            let sigma = treatment_sigma(sample, &train, treatment_fit.as_ref())?;
            let treatment_mean_constant = sample.train_treatment_mean(&train);
            let treat_smooth = outcome_fit.smooth_for_raw_col(0).ok_or_else(|| {
                EstimationError::unsupported("outcome nuisance is missing its treatment smooth")
            })?;
            for i in (0..n).filter(|i| i % self.options.folds == fold) {
                sample.write_raw_row(i, &mut row);
                let mu = predict_one(&outcome_fit, &row)?;
                // Additive μ(a, x) = α + f_T(a) + Σ g_k(x_k), so ∂μ/∂a = f_T'(a)
                // and is identical for every covariate row. The clamped evaluation
                // is constant outside the knot interior; the analytic derivative
                // is therefore exactly 0 there (`response.clamped_basis_derivative`).
                let derivative =
                    outcome_fit.smooth_derivative(treat_smooth, sample.treatments[i])?;
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

    /// One Bayesian-bootstrap draw of the cross-fitted Riesz ADE: every fold refits
    /// the outcome and treatment nuisances on its training rows under the draw's row
    /// weights and scores only its held-out rows, then the weighted mean of the
    /// held-out scores `∂_a μ̂ + α̂ (Y − μ̂)` is returned. Held-out scoring keeps the
    /// draws on the same footing as the cross-fitted point estimate, without the
    /// leverage shrinkage of in-sample residuals.
    pub(super) fn weighted_cross_fitted_ade(
        &self,
        sample: &CompleteSample,
        folds: &[CrossFitFold],
        weights: &[f64],
    ) -> Result<f64, EstimationError> {
        let n = sample.len();
        if weights.len() != n {
            return Err(EstimationError::stats_msg("ADE weights length must match complete rows"));
        }
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        let mut gam_ws = GamWorkspace::default();
        let mut row = vec![0.0; sample.raw_cols];
        let mut adj_row = vec![0.0; sample.adjustment_cols];
        let mut train_weights: Vec<f64> = Vec::new();
        let mut num = 0.0;
        let mut den = 0.0;
        for fold in folds {
            let train = fold.train.as_slice();
            train_weights.clear();
            train_weights.extend(train.iter().map(|&i| weights[i]));
            let train_weight_sum: f64 = train_weights.iter().sum();
            if !train_weight_sum.is_finite() || train_weight_sum <= 0.0 {
                return Err(EstimationError::stats_msg("fold training weights are degenerate"));
            }
            let (outcome_fit, treatment_fit, sigma) =
                Self::fit_train_nuisances(sample, fold, Some(&train_weights), &mut gam_ws)?;
            let treatment_mean_constant = train
                .iter()
                .zip(&train_weights)
                .map(|(&i, w)| w * sample.treatments[i])
                .sum::<f64>()
                / train_weight_sum;
            let treat_smooth = outcome_fit.smooth_for_raw_col(0).ok_or_else(|| {
                EstimationError::unsupported("outcome nuisance is missing its treatment smooth")
            })?;
            for &i in &fold.valid {
                sample.write_raw_row(i, &mut row);
                let mu = predict_one(&outcome_fit, &row)?;
                let derivative =
                    outcome_fit.smooth_derivative(treat_smooth, sample.treatments[i])?;
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => {
                        sample.write_adjustment_row(i, &mut adj_row);
                        predict_one(fit, &adj_row)?
                    }
                    None => treatment_mean_constant,
                };
                let alpha = (sample.treatments[i] - treatment_mean) / (sigma * sigma);
                num += weights[i] * (derivative + alpha * (sample.outcome[i] - mu));
                den += weights[i];
            }
        }
        if !den.is_finite() || den <= 0.0 || !num.is_finite() {
            return Err(EstimationError::stats_msg("weighted ADE draw was non-finite"));
        }
        Ok(num / den)
    }
}
