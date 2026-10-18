//! Mean-curve and intervention-response estimation.

use std::sync::Arc;

use antecedent_core::{
    DiagnosticScope, Intervention, ResponseUncertainty, ResponseValue, SupportDiagnostic,
    SupportRegion, SupportReport, SupportStatus, VariableId,
};
use antecedent_data::TabularData;
use antecedent_stats::{
    LocalQuadraticWorkspace, StatsError, gaussian_local_quadratic_adjusted_influence_prechecked,
    normal_ppf, silverman_bandwidth, student_t_ppf,
};

use crate::EstimationError;

use super::{
    ContinuousResponseEstimator, PseudoOutcome, ResponseInfluence, additive_policy_rows,
    exact_discrete_intervention_rows, intervention_needs_quadrature, local_design_refusal,
    plugin::intervention_plugin_influence, plugin::intervention_plugin_warnings,
    push_outcome_tail_diagnostic, push_pseudo_outcome_winsor_shift, sample::CompleteSample,
    sample::row_ids_u32, simultaneous_multiplier_band, support_report,
};

impl ContinuousResponseEstimator {
    #[allow(
        clippy::too_many_lines,
        reason = "pseudo-outcome, local fit and influence per grid point, then the interval; the grid loop writes the shared influence buffer in place"
    )]
    pub(super) fn mean_curve(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatment: VariableId,
        grid: &[f64],
    ) -> Result<
        (ResponseValue, ResponseUncertainty, SupportReport, ResponseInfluence),
        EstimationError,
    > {
        let sample = CompleteSample::read(data, outcome, &[treatment], &self.adjustment_set)?;
        let PseudoOutcome { values: pseudo, density_floor_rows, covariate_centered } =
            self.cross_fitted_pseudo_outcome(&sample)?;
        let bandwidth = self.options.bandwidth.unwrap_or(silverman_bandwidth(&sample.treatments)?);
        let mut mean = Vec::with_capacity(grid.len());
        let mut lower = Vec::with_capacity(grid.len());
        let mut upper = Vec::with_capacity(grid.len());
        let mut ess = Vec::with_capacity(grid.len());
        let mut density = Vec::with_capacity(grid.len());
        let n = sample.len();
        let g = grid.len();
        // One G×n buffer: local-polynomial IFs plus the grid-constant covariate
        // term written in place (ENG-012). The additive-μ claim that makes the
        // extra term the same at every `a` is disclosed below (SUS-002).
        let mut influences = vec![0.0; g * n];
        let mut robust_se = Vec::with_capacity(g);
        let mut variance_df = Vec::with_capacity(g);
        let mut pointwise_critical = 0.0_f64;
        if sample.treatments.iter().chain(&pseudo).any(|v| !v.is_finite()) {
            return Err(
                StatsError::Shape { message: "local quadratic inputs must be finite" }.into()
            );
        }
        // Per-row marginalization contribution `(c_i − c̄) / n` (see the loop).
        // Centering over rows keeps every grid column summing to zero exactly,
        // as the local WLS term does, because the cross-fit folds center `c`
        // on their own training rows.
        let covariate_term: Vec<f64> = {
            let n_rows = n as f64;
            let center = covariate_centered.iter().sum::<f64>() / n_rows;
            covariate_centered.iter().map(|c| (c - center) / n_rows).collect()
        };
        let mut local = LocalQuadraticWorkspace::default();
        for (g_idx, &at) in grid.iter().enumerate() {
            let fit = gaussian_local_quadratic_adjusted_influence_prechecked(
                &mut local,
                &sample.treatments,
                &pseudo,
                at,
                bandwidth,
            )
            .map_err(|err| local_design_refusal(err, &sample.treatments, at, bandwidth))?;
            let point = fit.point;
            mean.push(point.value);
            // Kennedy et al. (2017, Thm. 3): the pseudo-outcome's marginalization
            // `∫μ̂(a, x) dP_n(x)` averages every row's covariates, so each row also
            // moves the fitted level through `e₀ᵀD⁻¹ P_n[g K {μ̂(·, X_i) − m̂(·)}]`.
            // With an additive μ̂ that bracket is the constant `ĥ(X_i) − mean ĥ`
            // and `P_n[g K] = D e₀`, so the term is exactly `covariate_term[i]`,
            // the same at every grid point. Treating the pseudo-outcomes as fixed
            // data drops it; without it, calibration measured 0.80–0.86 pointwise
            // coverage at nominal 0.90 without it.
            let row = &mut influences[g_idx * n..(g_idx + 1) * n];
            for (slot, (local_if, centered)) in
                row.iter_mut().zip(fit.influences.iter().zip(&covariate_term))
            {
                *slot = local_if + centered;
            }
            let standard_error = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            // The pointwise band uses the same influence-based standard error the
            // simultaneous band standardizes by, so the two are nested by
            // construction rather than being two different variance estimates at
            // the same nominal level.
            let df = fit
                .variance_degrees_of_freedom
                .expect("adjusted local fit supplies variance degrees of freedom");
            let critical = student_t_ppf(0.5 + self.options.confidence_level / 2.0, df);
            pointwise_critical = pointwise_critical.max(critical);
            lower.push(point.value - critical * standard_error);
            upper.push(point.value + critical * standard_error);
            variance_df.push(df);
            ess.push(point.local_ess);
            density.push(
                point.weight_sum / (n as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt()),
            );
            robust_se.push(standard_error);
        }
        let mut support = support_report(
            grid,
            &sample.treatments,
            &ess,
            density,
            self.options.minimum_local_ess,
            density_floor_rows,
        );
        push_outcome_tail_diagnostic(&mut support, &sample.outcome);
        push_pseudo_outcome_winsor_shift(
            &mut support,
            &sample.treatments,
            &pseudo,
            grid,
            &mean,
            bandwidth,
            &mut local,
        );
        support.warnings.push(antecedent_core::Diagnostic::new(
            "response.kennedy_dr.additive_covariate_if",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "the grid-constant Kennedy IF covariate term (c_i − c̄)/n assumes the fitted \
             additive outcome restriction; it is not re-derived for treatment–covariate \
             interactions in μ̂",
        ));
        support.diagnostics.push(SupportDiagnostic {
                id: Arc::from("response.local_variance_degrees_of_freedom"),
                values: Arc::from(variance_df),
                detail: Arc::from("pointwise Student intervals with Satterthwaite df=tr(A)^2/tr(A^2) for the leverage-corrected local residual variance quadratic form; Gaussian working approximation conditional on the local design"),
                scope: DiagnosticScope::Global,
            });
        if self.options.export_row_diagnostics {
            let flat_influences = influences.clone();
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
                scope: DiagnosticScope::Global,
            });
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from("response.row_pseudo_outcome"),
                values: Arc::from(pseudo),
                detail: Arc::from("cross-fitted Kennedy pseudo-outcome per retained row"),
                scope: DiagnosticScope::Global,
            });
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from("response.row_influence"),
                values: Arc::from(flat_influences),
                detail: Arc::from(format!(
                    "row-major by grid point, grid_len={}, n={n}: value[g*N + i]",
                    grid.len()
                )),
                scope: DiagnosticScope::Global,
            });
        }
        let uncertainty = if let Some(replicates) = self.options.simultaneous_replicates {
            let mut band = simultaneous_multiplier_band(
                &mean,
                &influences,
                n,
                &robust_se,
                self.options.confidence_level,
                replicates,
                self.options.multiplier_seed,
            )?;
            // Keep the unchanged multiplier construction, with a caller-owned
            // floor covering every marginal Student interval.
            if let ResponseUncertainty::SimultaneousBand { lower, upper, .. } = &mut band {
                *lower = mean
                    .iter()
                    .zip(&robust_se)
                    .zip(lower.iter())
                    .map(|((value, se), bound)| bound.min(value - pointwise_critical * se))
                    .collect();
                *upper = mean
                    .iter()
                    .zip(&robust_se)
                    .zip(upper.iter())
                    .map(|((value, se), bound)| bound.max(value + pointwise_critical * se))
                    .collect();
            }
            band
        } else {
            ResponseUncertainty::PointwiseBand {
                level: self.options.confidence_level,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            }
        };
        let row_index = row_ids_u32(&sample.keep)?;
        // Local-polynomial exports are contributions to the estimate (order 1/n).
        // Shared IF covariance expects unnormalised row scores (order 1).
        let scale = n as f64;
        let columns = (0..g)
            .map(|g_idx| influences[g_idx * n..(g_idx + 1) * n].iter().map(|v| v * scale).collect())
            .collect();
        let scores = ResponseInfluence { columns, row_index };
        Ok((
            ResponseValue::Surface {
                grid: Arc::from(grid.to_vec()),
                dimension: 1,
                mean: Arc::from(mean),
            },
            uncertainty,
            support,
            scores,
        ))
    }

    pub(super) fn intervention_response(
        &self,
        data: &TabularData,
        outcome: VariableId,
        interventions: &[Intervention],
    ) -> Result<
        (ResponseValue, ResponseUncertainty, SupportReport, ResponseInfluence),
        EstimationError,
    > {
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
        let (fit, target_fallback) = self.fit_outcome_target(&sample)?;
        // Discrete policies (Set/Shift/Bernoulli/Categorical) are integrated exactly as a
        // finite mixture. Monte Carlo through a continuous spline would treat categorical
        // codes as ordered coordinates and approximate a sum that has a closed form.
        // Gaussian policies are integrated by Gauss–Hermite quadrature: the outcome
        // model is additive, so E[μ(A_1, …, A_k, X)] depends only on each policy's own
        // marginal law and the same rule serves every row, with no sampling error.
        let row_means = if interventions.iter().any(intervention_needs_quadrature) {
            additive_policy_rows(&fit, &sample, interventions)?
        } else {
            exact_discrete_intervention_rows(&fit, &sample, interventions)?
        };
        let n = row_means.len() as f64;
        let estimate = row_means.iter().sum::<f64>() / n;
        let psi = intervention_plugin_influence(&fit, &sample, interventions, &row_means)?;
        let se = if n > 1.0 {
            (psi.iter().map(|x| x * x).sum::<f64>() / (n * (n - 1.0))).sqrt()
        } else {
            f64::NAN
        };
        let level = self.options.confidence_level;
        let z = normal_ppf(0.5 + level / 2.0);
        let scores =
            ResponseInfluence { columns: vec![psi], row_index: row_ids_u32(&sample.keep)? };
        let minima: Vec<f64> =
            (0..treatments.len()).map(|column| sample.treatment_column_range(column).0).collect();
        let maxima: Vec<f64> =
            (0..treatments.len()).map(|column| sample.treatment_column_range(column).1).collect();
        Ok((
            ResponseValue::Scalar(estimate),
            ResponseUncertainty::Scalar {
                standard_error: se,
                level,
                lower: estimate - z * se,
                upper: estimate + z * se,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            },
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
                    scope: DiagnosticScope::Global,
                }],
                warnings: intervention_plugin_warnings(target_fallback.as_deref()),
                // One scalar answers the whole (possibly joint) regime, so it has exactly
                // one coordinate and one label: the summary itself, never a per-treatment claim.
                point_status: Some(Arc::from([SupportStatus::Extrapolative])),
            },
            scores,
        ))
    }
}
