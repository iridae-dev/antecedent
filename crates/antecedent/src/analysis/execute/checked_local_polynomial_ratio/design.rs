//! The local ratio with its design layer: an MSE-optimal plug-in bandwidth paired with the
//! robust bias-corrected interval, a calendar-time running variable with a within-side block
//! bootstrap, and the density and covariate-balance diagnostics.
//!
//! A result states support only when a `local_polynomial_ratio` support row admits its design,
//! interval procedure, window rows, first-stage strength and replicates; the check is the one
//! the artifact loader repeats on the stored section. Every other result is unsupported.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, effect_quantity, finish_design_posterior_as,
};
use super::super::rollout_finish::{
    RolloutPayloadParts, RolloutReport, finish_rollout, named, rollout_payload,
};
use super::{
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, EstimatorId, ExecutionContext,
    IdentifierId, Strategy, StudyResult, TableView, TabularData,
};
use antecedent_core::LocalBandwidthRule;
use antecedent_estimate::local_ratio_design::{
    LocalRatioColumns, LocalRatioDesign, LocalRatioDesignFit, fit_local_ratio_design,
};
use antecedent_estimate::rollout_resampling::{DesignEffect, DesignEffectKind};

/// Plan id, identifier and estimator a compiled local-ratio query records: the design-layer
/// estimators when options are declared, otherwise the fixed-bandwidth ones.
pub(in crate::analysis::execute) fn local_ratio_query_plan_ids(
    query: &antecedent_core::LocalPolynomialRatioQuery,
) -> (String, String, String) {
    let Some(options) = &query.options else {
        return super::local_polynomial_ratio_plan_ids(query.kink);
    };
    let label = if query.kink { "regression_kink" } else { "fuzzy_rd" };
    let temporal = options.temporal_replicates.is_some();
    let plan = if temporal { format!("quasi.temporal_{label}") } else { format!("quasi.{label}") };
    let estimator = match (options.bandwidth_rule, temporal) {
        (LocalBandwidthRule::Declared, false) => "declared_bandwidth_rbc_with_diagnostics",
        (LocalBandwidthRule::MseOptimal, false) => "mse_optimal_bandwidth_rbc",
        (LocalBandwidthRule::Declared, true) => "declared_bandwidth_rbc_block_bootstrap",
        (LocalBandwidthRule::MseOptimal, true) => "mse_optimal_bandwidth_rbc_block_bootstrap",
    };
    (
        plan,
        IdentifierId::for_local_ratio(query.kink).as_str().to_string(),
        format!("quasi.{label}.{estimator}"),
    )
}

/// State support on a local-design result whose rollout section a support row admits, and
/// replace the route's unsupported note with the scope of that row.
///
/// `selected_bandwidth` is the query's MSE-optimal rule. The section is read in its artifact
/// form, so the loader's recomputation sees exactly what is decided here.
fn local_design_state_support(result: &mut StudyResult, selected_bandwidth: bool) {
    let crate::result::ResultPayload::QuasiRollout(rollout) = &mut result.payload else {
        return;
    };
    let section = antecedent_io::analysis_result_artifact::QuasiRolloutWire::from(&**rollout);
    if !antecedent_io::analysis_result_artifact::rollout_support::local_design_row_admits(
        selected_bandwidth,
        &section,
    ) {
        return;
    }
    rollout.supported = true;
    result.diagnostics.retain(|diagnostic| {
        !matches!(
            diagnostic.code.as_ref(),
            "estimate.quasi.rollout.unsupported" | "estimate.quasi.bayesian.unsupported"
        )
    });
    result.diagnostics.push(Diagnostic::new(
        "estimate.quasi.local_ratio.design_support",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "supported design: a local_polynomial_ratio support row admits this design, interval \
         procedure, window rows on each side, first-stage estimate over its standard error and \
         replicate count; the identifying premises, and for calendar time the dependence of \
         the errors, stay declared",
    ));
}

/// The first-stage estimate and its HC0 standard error at `bandwidth`, with the window rows on
/// each side, as named diagnostics; empty when the fit is refused.
fn local_design_first_stage_facts(
    columns: &LocalRatioColumns<'_>,
    cutoff: f64,
    bandwidth: f64,
    kink: bool,
) -> Vec<(String, f64)> {
    antecedent_estimate::local_polynomial_ratio::fit_local_polynomial_ratio(
        columns.running,
        columns.outcome,
        columns.treatment,
        cutoff,
        bandwidth,
        kink,
    )
    .map_or_else(
        |_| Vec::new(),
        |fit| {
            named(&[
                ("first_stage", fit.first_stage),
                ("first_stage_standard_error", fit.first_stage_standard_error),
                ("rows_left", fit.n_left as f64),
                ("rows_right", fit.n_right as f64),
            ])
        },
    )
}

/// The selected bandwidth of a Bayesian local ratio that declares the MSE-optimal rule, using
/// the frequentist pilot ratio at the declared bandwidth to linearize the outcome.
pub(super) fn selected_bandwidth(
    query: &antecedent_core::LocalPolynomialRatioQuery,
    columns: &LocalRatioColumns<'_>,
    ctx: &ExecutionContext,
) -> Result<f64, CausalError> {
    let fit = fit_local_ratio_design(
        columns,
        &[],
        query.cutoff,
        query.bandwidth,
        query.kink,
        LocalRatioDesign { select_bandwidth: true, temporal_replicates: None, density_test: false },
        ctx,
    )
    .map_err(CausalError::Estimate)?;
    Ok(fit.bandwidth.bandwidth)
}

impl super::CheckedLocalPolynomialRatioOperation {
    /// Execute the local ratio with its declared design layer.
    pub(super) fn execute_design(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let Some(options) = self.query.options.as_ref() else {
            return Err(CausalError::Compile {
                message: "the local-ratio design layer requires its options".into(),
            });
        };
        let column = |id| -> Result<&[f64], CausalError> {
            match data.column(id).map_err(CausalError::from)? {
                antecedent_data::ColumnView::Float64(column) => Ok(column.values.as_slice()),
                _ => Err(CausalError::Unsupported {
                    message: "local ratio requires continuous columns",
                }),
            }
        };
        let columns = LocalRatioColumns {
            running: column(self.query.running)?,
            outcome: column(self.query.outcome)?,
            treatment: column(self.query.treatment)?,
        };
        let covariates = options
            .balance_covariates
            .iter()
            .map(|covariate| column(*covariate))
            .collect::<Result<Vec<_>, _>>()?;
        let design = LocalRatioDesign {
            select_bandwidth: options.bandwidth_rule == LocalBandwidthRule::MseOptimal,
            temporal_replicates: options.temporal_replicates,
            density_test: options.density_test,
        };
        let fit = fit_local_ratio_design(
            &columns,
            &covariates,
            self.query.cutoff,
            self.query.bandwidth,
            self.query.kink,
            design,
            ctx,
        )
        .map_err(CausalError::Estimate)?;
        let mut report = self.design_report(&fit);
        report.diagnostics.push(("rows".to_string(), columns.running.len() as f64));
        let mut result = finish_rollout(
            self,
            data,
            self.query.outcome,
            Strategy::new(IdentifierId::for_local_ratio(self.query.kink), EstimatorId::LocalRatio),
            report,
        );
        local_design_state_support(
            &mut result,
            options.bandwidth_rule == LocalBandwidthRule::MseOptimal,
        );
        Ok(result)
    }

    /// The within-side block Bayesian-bootstrap posterior of a local ratio in calendar time.
    #[allow(
        clippy::too_many_lines,
        reason = "one calendar-time posterior: the block Bayesian bootstrap, its summary and the rollout payload read the same fit"
    )]
    pub(super) fn execute_bayesian_temporal(
        &self,
        data: &TabularData,
        config: &super::BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = std::time::Instant::now();
        let column = |id| -> Result<&[f64], CausalError> {
            match data.column(id).map_err(CausalError::from)? {
                antecedent_data::ColumnView::Float64(column) => Ok(column.values.as_slice()),
                _ => Err(CausalError::Unsupported {
                    message: "local ratio requires continuous columns",
                }),
            }
        };
        let columns = LocalRatioColumns {
            running: column(self.query.running)?,
            outcome: column(self.query.outcome)?,
            treatment: column(self.query.treatment)?,
        };
        let selected = self
            .query
            .options
            .as_ref()
            .is_some_and(|options| options.bandwidth_rule == LocalBandwidthRule::MseOptimal);
        let bandwidth = if selected {
            selected_bandwidth(&self.query, &columns, ctx)?
        } else {
            self.query.bandwidth
        };
        let draws = u32::try_from(config.n_draws).unwrap_or(u32::MAX);
        let posterior = antecedent_estimate::local_ratio_design::posterior_temporal_local_ratio(
            &columns,
            self.query.cutoff,
            bandwidth,
            self.query.kink,
            draws,
            ctx,
        )
        .map_err(CausalError::Estimate)?;
        let label = if self.query.kink { "regression_kink" } else { "fuzzy_rd" };
        let summary = posterior.summary;
        // A ratio's draws have no finite mean when the first stage is uncertain, so the point
        // is the posterior median.
        let mut sorted = posterior.draws.clone();
        sorted.sort_by(f64::total_cmp);
        let median = antecedent_estimate::rollout_resampling::quantile_sorted(&sorted, 0.5);
        let overall = DesignEffect {
            interval: Some(summary.interval),
            ..DesignEffect::point(DesignEffectKind::Overall, median)
        };
        let design = DesignPosterior {
            columns: vec![(effect_quantity("local_ratio"), posterior.draws.clone())],
            point: median,
            point_sd: summary.sd,
            n_obs: columns.running.len() as u64,
            interval_withheld: false,
            backend: "bayesian_bootstrap.within_side_block_dirichlet",
            model: format!(
                "within-side block Bayesian bootstrap of the errors at the fixed time design: \
                 flat-Dirichlet weights on consecutive blocks of {} time-ordered AR(1)-prewhitened \
                 innovations redraw each side's errors, the prewhitening coefficient is drawn \
                 from its large-sample posterior, and the robust bias-corrected local ratio is \
                 refit per draw; serial dependence within a block is kept, across block \
                 boundaries its AR(1) part is",
                posterior.block_length
            ),
            assumptions: vec![(
                "bayesian_block_bootstrap_serial_dependence",
                "the errors about a smooth time path are stationary within each side, their \
                 serial dependence beyond an AR(1) fit is shorter than a block, and blocks of \
                 innovations are exchangeable draws within each side"
                    .to_string(),
            )],
            diagnostics: vec![Diagnostic::new(
                "estimate.quasi.bayesian_local_ratio.temporal_blocks",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "{label} in calendar time at bandwidth {bandwidth}; {} of {} block \
                     Bayesian-bootstrap draws produced a finite ratio",
                    posterior.draws_ok, posterior.requested
                ),
            )],
            notes: vec![format!("block_length={}", posterior.block_length)],
            sampler: None,
        };
        let mut result = finish_design_posterior_as(
            DesignFinish {
                context: &self.result_context,
                data,
                physical: &self.physical,
                identification: &self.identification,
                estimand: &self.estimand,
                outcome: self.query.outcome,
                config,
                started,
            },
            design,
            Strategy::new(
                IdentifierId::for_local_ratio(self.query.kink),
                EstimatorId::BayesianLocalRatio,
            ),
        )?;
        result.payload = rollout_payload(RolloutPayloadParts {
            design: if self.query.kink { "temporal_regression_kink" } else { "temporal_fuzzy_rd" },
            method: format!(
                "local_{}_rbc.within_side_block_bayesian_bootstrap",
                if self.query.kink { "quartic_slope" } else { "quadratic_cubic" }
            ),
            effects: vec![overall],
            uncertainty: "bayesian_within_side_block_bootstrap_credible_interval",
            replicates: [
                posterior.requested,
                posterior.draws_ok,
                posterior.requested.saturating_sub(posterior.draws_ok),
            ],
            critical_value: None,
            diagnostics: {
                let mut diagnostics = named(&[
                    ("bandwidth", bandwidth),
                    ("block_length", posterior.block_length as f64),
                    ("posterior_sd", summary.sd),
                ]);
                diagnostics.extend(local_design_first_stage_facts(
                    &columns,
                    self.query.cutoff,
                    bandwidth,
                    self.query.kink,
                ));
                diagnostics
            },
        });
        local_design_state_support(&mut result, selected);
        Ok(result)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "a linear assembly of the design effect, its diagnostics and its notes from one fit"
    )]
    fn design_report(&self, fit: &LocalRatioDesignFit) -> RolloutReport {
        let temporal = fit.temporal.as_ref();
        let (standard_error, interval) = if let Some(record) = temporal {
            (record.ratio.standard_error, record.ratio.interval)
        } else {
            let available = fit.fit.standard_error.is_finite() && fit.fit.standard_error > 0.0;
            (
                available.then_some(fit.fit.standard_error),
                available.then_some([fit.fit.ci_lower, fit.fit.ci_upper]),
            )
        };
        let overall = DesignEffect {
            standard_error,
            interval,
            treated: fit.fit.n_right,
            comparison: fit.fit.n_left,
            ..DesignEffect::point(DesignEffectKind::Overall, fit.fit.estimate)
        };
        let mut diagnostics = named(&[
            ("bandwidth", fit.bandwidth.bandwidth),
            ("bandwidth_capped_at_declared", if fit.bandwidth.capped { 1.0 } else { 0.0 }),
            ("reduced_form", fit.fit.reduced_form),
            ("first_stage", fit.fit.first_stage),
            ("reduced_form_standard_error", fit.fit.reduced_form_standard_error),
            ("first_stage_standard_error", fit.fit.first_stage_standard_error),
            ("hc0_ratio_standard_error", fit.fit.standard_error),
            ("rows_left", fit.fit.n_left as f64),
            ("rows_right", fit.fit.n_right as f64),
        ]);
        if let Some(optimal) = fit.bandwidth.mse_optimal {
            diagnostics.push(("mse_optimal_bandwidth".to_string(), optimal));
        }
        if let Some(density) = fit.density {
            diagnostics.push(("density_log_difference".to_string(), density.log_difference));
            diagnostics.push(("density_standard_error".to_string(), density.standard_error));
            diagnostics.push(("density_p_value".to_string(), density.p_value));
        }
        for balance in &fit.balance {
            let name = self
                .query
                .options
                .as_ref()
                .and_then(|options| options.balance_covariates.get(balance.index))
                .map_or_else(|| balance.index.to_string(), |id| id.raw().to_string());
            diagnostics.push((format!("balance_jump_variable_{name}"), balance.jump));
            diagnostics.push((format!("balance_z_variable_{name}"), balance.z));
        }
        if let Some(record) = temporal {
            diagnostics.push(("block_length".to_string(), record.block_length as f64));
            // The artifact check recomputes the Student-t interval from these degrees.
            if let Some(degrees) = record.reference_degrees_of_freedom {
                diagnostics.push((
                    antecedent_io::analysis_result_artifact::rollout_support::STUDENT_REFERENCE_DIAGNOSTIC
                        .to_string(),
                    degrees,
                ));
            }
        }
        let label = if self.query.kink { "regression_kink" } else { "fuzzy_rd" };
        let design = match (temporal.is_some(), self.query.kink) {
            (true, true) => "temporal_regression_kink",
            (true, false) => "temporal_fuzzy_rd",
            (false, true) => "regression_kink",
            (false, false) => "fuzzy_rd",
        };
        let mut notes = vec![Diagnostic::new(
            "estimate.quasi.local_ratio.design_layer",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "{label} at bandwidth {}{}; robust bias-corrected local ratio; {}",
                fit.bandwidth.bandwidth,
                fit.bandwidth.mse_optimal.map_or_else(String::new, |optimal| format!(
                    " (plug-in MSE-optimal {optimal}{})",
                    if fit.bandwidth.capped { ", capped at the declared bandwidth" } else { "" }
                )),
                if temporal.is_some() {
                    "calendar-time rows: within-side moving-block bootstrap of the prewhitened \
                     errors refits the whole local ratio and its interquartile standard error \
                     gives the Student-t interval"
                } else {
                    "independent rows: HC0 delta-method interval of the bias-corrected ratio"
                }
            ),
        )];
        if let Some(density) = fit.density {
            notes.push(Diagnostic::new(
                "diagnostic.quasi.local_ratio.density_test",
                DiagnosticKind::Scientific,
                if density.p_value < 0.05 {
                    DiagnosticSeverity::Warning
                } else {
                    DiagnosticSeverity::Info
                },
                format!(
                    "running-variable density log difference at the cutoff {:.6} (se {:.6}, \
                     p {:.4}); a manipulation diagnostic, not a test that establishes continuity \
                     of potential outcomes",
                    density.log_difference, density.standard_error, density.p_value
                ),
            ));
        }
        if fit.balance.iter().any(|balance| balance.z.abs() > antecedent_stats::NORMAL_Q975) {
            notes.push(Diagnostic::new(
                "diagnostic.quasi.local_ratio.covariate_imbalance",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "a declared baseline covariate jumps at the cutoff beyond its pointwise 95% \
                 band; local continuity is doubtful",
            ));
        }
        if let Some(p_value) = temporal.and_then(|record| record.serial_dependence_rejected) {
            notes.push(Diagnostic::new(
                "diagnostic.quasi.local_ratio.serial_dependence_beyond_ar1",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "the innovations of the AR(1)-prewhitened errors are serially dependent at                      lags two to six (Ljung-Box p {p_value:.2e}), so the AR(1) error model of the                      window does not hold and no interval is published; the point estimate is                      unchanged. Next attempt: a design whose errors are AR(1) about the time                      path, or a longer window of the same process"
                ),
            ));
        }
        let (uncertainty, replicates) = match temporal {
            Some(record) => (
                "within_side_moving_block_bootstrap_refit_iqr_se_student_t_interval",
                [record.requested, record.replicates_ok, record.replicates_failed],
            ),
            None if fit.bandwidth.mse_optimal.is_some() || fit.bandwidth.capped => {
                ("rbc_hc0_delta_normal_at_selected_bandwidth", [0, 0, 0])
            }
            None => ("rbc_hc0_delta_normal_fixed_bandwidth", [0, 0, 0]),
        };
        RolloutReport {
            design,
            method: format!(
                "local_{}_rbc.{}",
                if self.query.kink { "quartic_slope" } else { "quadratic_cubic" },
                if fit.bandwidth.mse_optimal.is_some() || fit.bandwidth.capped {
                    "mse_optimal_plugin_bandwidth"
                } else {
                    "declared_bandwidth"
                }
            ),
            effects: vec![overall],
            uncertainty,
            replicates,
            critical_value: None,
            diagnostics,
            notes,
        }
    }
}
