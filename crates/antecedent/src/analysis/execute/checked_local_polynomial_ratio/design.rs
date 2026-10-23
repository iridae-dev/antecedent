//! The local ratio with its design layer: an MSE-optimal plug-in bandwidth paired with the
//! robust bias-corrected interval, a calendar-time running variable with a within-side block
//! bootstrap, and the density and covariate-balance diagnostics.
//!
//! The result is unlicensed and its calibration is unmeasured; the existing fixed-bandwidth
//! licence row covers only the declared-bandwidth independent-row route.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::rollout_finish::{RolloutReport, finish_rollout, named};
use super::{
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext, StudyResult,
    TableView, TabularData,
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
    (plan, format!("quasi.{label}.local_ratio"), format!("quasi.{label}.{estimator}"))
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
        Ok(finish_rollout(self, data, self.query.outcome, self.design_report(&fit)))
    }

    fn design_report(&self, fit: &LocalRatioDesignFit) -> RolloutReport {
        let temporal = fit.temporal.as_ref();
        let (standard_error, interval) = match temporal {
            Some(record) => (record.ratio.standard_error, record.ratio.interval),
            None => {
                let available = fit.fit.standard_error.is_finite() && fit.fit.standard_error > 0.0;
                (
                    available.then_some(fit.fit.standard_error),
                    available.then_some([fit.fit.ci_lower, fit.fit.ci_upper]),
                )
            }
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
                    "calendar-time rows: within-side moving-block bootstrap refits the whole local \
                     ratio and its interquartile standard error gives the normal interval"
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
        let (uncertainty, replicates) = match temporal {
            Some(record) => (
                "within_side_moving_block_bootstrap_refit_iqr_se_normal_interval",
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
