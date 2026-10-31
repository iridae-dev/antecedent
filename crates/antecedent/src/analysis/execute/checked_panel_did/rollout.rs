//! Heterogeneity-robust and covariate-adjusted rollouts, and the whole-cluster resample of the
//! augmented panel.
//!
//! A rollout reads its own panel layout: a declared `available_case` policy keeps every
//! complete unit-period (each 2×2 then uses the units observed at both of its periods), a
//! declared `balanced_panel` policy keeps the units complete in every period through the
//! shared unbalanced-panel preparation, and a structurally unbalanced panel (absent rows) is
//! read as it is by every estimator: the Sun–Abraham saturated regression uses only the
//! observed unit-periods that have a supported comparison and reports what it dropped. The
//! declared observation process is recorded as an untestable assumption, and a process
//! declared informative is refused because no route here corrects for it.
//!
//! Uncertainty is a whole-procedure resample of the inference clusters (whole subjects under a
//! declared independent-subject design; a dependent single series is refused), and every
//! interval is unlicensed with calibration `unmeasured`.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, effect_quantity, finish_design_posterior, scalar_quantity,
};
use super::super::rollout_finish::{
    RolloutReport, finish_rollout, named, rollout_payload, split_report,
};
use super::{
    Arc, CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext,
    StudyResult, TableView, TabularData,
};
use crate::analysis::missing_data::MissingRoute;
use antecedent_core::{
    DeclaredSamplingDesign, DidAggregation, DidResamplingScheme, DidSamplingDesign,
    PanelObservationProcess, RolloutDidEstimator,
};
use antecedent_data::{MissingPolicy, PanelLayout, PreparationMode};
use antecedent_estimate::EstimationError;
use antecedent_estimate::heterogeneity_robust_did::{
    RolloutDidFit, RolloutDidSpec, RolloutPanel, RolloutResampling, estimate_rollout_did,
    posterior_rollout_did,
};
use antecedent_estimate::rollout_resampling::UnitWeighting;

/// The unit weighting of a declared resampling scheme.
pub(super) const fn weighting(scheme: DidResamplingScheme) -> UnitWeighting {
    match scheme {
        DidResamplingScheme::ClusterBootstrap => UnitWeighting::Resample,
        DidResamplingScheme::ClusterMultiplier => UnitWeighting::Multiplier,
    }
}

/// The resampling units of a declared sampling design: the query's clusters, or each subject
/// under independent subject trajectories.
pub(super) fn resampling_units(
    sampling: Option<DeclaredSamplingDesign>,
    subjects: &Arc<[Arc<str>]>,
    clusters: &Arc<[Arc<str>]>,
) -> Result<Arc<[Arc<str>]>, CausalError> {
    match sampling {
        Some(DeclaredSamplingDesign::DependentTimeSeries) => Err(crate::compile_reason!(
            "option_not_applicable",
            "a difference-in-differences resample draws whole units of a panel, and the declared \
             dependent_time_series design is one dependent series. Next attempt: declare \
             clustered_panel or independent_subject_trajectories for a panel of units"
        )),
        Some(DeclaredSamplingDesign::IndependentSubjectTrajectories) => Ok(Arc::clone(subjects)),
        Some(DeclaredSamplingDesign::ClusteredPanel) | None => Ok(Arc::clone(clusters)),
    }
}

fn estimation(error: EstimationError) -> CausalError {
    CausalError::Estimate(error)
}

/// Posterior sample SD; invalid or insufficient draws cannot license uncertainty.
fn rollout_draw_sd(draws: &[f64]) -> f64 {
    if draws.iter().any(|draw| !draw.is_finite()) {
        return f64::NAN;
    }
    let sd = antecedent_stats::sample_std(draws);
    if sd.is_finite() { sd } else { f64::NAN }
}

impl super::CheckedPanelDidOperation {
    /// Execute a heterogeneity-robust or covariate-adjusted rollout.
    pub(super) fn execute_rollout(
        &self,
        data: &TabularData,
        y: &[f64],
        observed: &[bool],
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let Some(robust) = self.query.robust.as_ref() else {
            return Err(CausalError::Compile {
                message: "a rollout requires its heterogeneity-robust specification".into(),
            });
        };
        if robust.observation == PanelObservationProcess::Informative {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "the observation process is declared informative (attrition or late entry depends \
                 on outcomes or untreated trends within cohort) with no bound on how far, and \
                 nothing identifies the effect under unbounded departures from the observed \
                 unit-periods. Next attempt: declare attrition_shift_bound, the largest amount (in \
                 outcome units) by which the treated units a cohort fails to observe may differ in \
                 change from those it observes, for an assumption-dependent range; or declare \
                 observation=ignorable_given_cohort if the chance of being observed is unrelated to \
                 the untreated outcome trend within cohort"
            ));
        }
        let n = y.len();
        let mut complete = observed.to_vec();
        let mut covariates = Vec::with_capacity(robust.covariates.len());
        for variable in robust.covariates.iter() {
            let antecedent_data::ColumnView::Float64(column) =
                data.column(*variable).map_err(CausalError::from)?
            else {
                return Err(CausalError::Unsupported {
                    message: "rollout covariates must be continuous columns",
                });
            };
            for (row, seen) in complete.iter_mut().enumerate() {
                *seen = *seen && column.validity.is_valid(row);
            }
            covariates.push(column.values.as_slice().to_vec());
        }
        let kept = self.rollout_rows(data, &complete, robust.covariates.as_ref())?;
        let outcome: Vec<f64> = y
            .iter()
            .zip(&kept)
            .map(|(value, keep)| if *keep { *value } else { f64::NAN })
            .collect();
        let units =
            resampling_units(self.extra.sampling, &self.query.subjects, &self.query.clusters)?;
        let spec = RolloutDidSpec {
            estimator: robust.estimator,
            comparison: robust.comparison,
            nuisance: robust.nuisance,
            base_period: robust.base_period,
            aggregation: robust.aggregation,
            anticipation: i64::from(robust.anticipation),
        };
        let panel = RolloutPanel {
            outcome: &outcome,
            kept: &kept,
            units: &self.query.subjects,
            clusters: &units,
            periods: &self.query.periods,
            cohorts: &self.query.cohorts,
            covariates: &covariates,
        };
        let resampling = self.query.resampling.map(|resampling| RolloutResampling {
            weighting: weighting(resampling.scheme),
            replicates: resampling.replicates,
        });
        if let Some(config) = self.extra.bayesian.as_ref() {
            let mut result = self.rollout_posterior(data, &panel, &spec, config, n, ctx)?;
            result.extras.missing_data = Some(self.missing_report(data, &complete, &kept)?);
            return Ok(result);
        }
        let fit = estimate_rollout_did(&panel, &spec, resampling, ctx).map_err(estimation)?;
        let mut result = finish_rollout(self, data, self.query.outcome, self.report(&fit, n));
        result.extras.missing_data = Some(self.missing_report(data, &complete, &kept)?);
        Ok(result)
    }

    /// The Bayesian-bootstrap posterior of the rollout, with its credible intervals and bands
    /// on the rollout payload.
    fn rollout_posterior(
        &self,
        data: &TabularData,
        panel: &RolloutPanel<'_>,
        spec: &RolloutDidSpec,
        config: &super::BayesianConfig,
        rows: usize,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = super::Instant::now();
        let draws = u32::try_from(config.n_draws).unwrap_or(u32::MAX);
        let posterior = posterior_rollout_did(panel, spec, draws, ctx).map_err(estimation)?;
        let fit = &posterior.fit;
        let mut report = self.report(fit, rows);
        report.uncertainty = "bayesian_bootstrap_whole_unit_dirichlet_credible_intervals_and_bands";
        let (parts, notes) = split_report(report);
        let overall_draws =
            posterior.columns.first().map(|(_, column)| column.clone()).unwrap_or_default();
        let point = fit.overall.estimate;
        let point_sd = rollout_draw_sd(&overall_draws);
        let columns = posterior
            .columns
            .iter()
            .enumerate()
            .map(|(index, (label, column))| {
                let kind = if index == 0 { effect_quantity("att") } else { scalar_quantity(label) };
                (kind, column.clone())
            })
            .collect();
        let design = DesignPosterior {
            columns,
            point,
            point_sd,
            n_obs: fit.units as u64,
            interval_withheld: !point_sd.is_finite(),
            backend: "bayesian_bootstrap.whole_unit_dirichlet",
            model: "Bayesian bootstrap: flat-Dirichlet weights on whole independent units refit                     every group-time cell, nuisance model and aggregation per draw; the posterior                     is joint across cells and aggregations"
                .to_string(),
            assumptions: vec![(
                "bayesian_bootstrap_unit_distribution",
                "the independent units are exchangeable draws from a population whose distribution                  carries a Dirichlet-process posterior with vanishing concentration; no parametric                  outcome model is assumed"
                    .to_string(),
            )],
            diagnostics: notes,
            notes: vec![format!(
                "draws_requested={} draws_complete={}",
                posterior.requested, posterior.draws_ok
            )],
            sampler: None,
        };
        let mut result = finish_design_posterior(
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
        )?;
        result.payload = rollout_payload(parts);
        result.treatment = None;
        result.support_status = None;
        Ok(result)
    }

    /// The rows a rollout reads under the declared missing-data policy.
    fn rollout_rows(
        &self,
        data: &TabularData,
        complete: &[bool],
        covariates: &[antecedent_core::VariableId],
    ) -> Result<Vec<bool>, CausalError> {
        if complete.iter().all(|seen| *seen) {
            return Ok(complete.to_vec());
        }
        let supported = MissingRoute::Panel.supported();
        let Some(declaration) = self
            .extra
            .declaration
            .as_ref()
            .filter(|declaration| declaration.policy != MissingPolicy::ErrorOnMissing)
        else {
            let mut variables = vec![self.query.outcome];
            variables.extend_from_slice(covariates);
            let cell = antecedent_data::first_missing_cell(data, &variables)?.ok_or(
                CausalError::Unsupported { message: "rollout validity changed during execution" },
            )?;
            return Err(estimation(antecedent_estimate::missing_policy::policy_required(
                &cell,
                MissingRoute::Panel.label(),
                &supported[1..],
            )));
        };
        match declaration.policy {
            MissingPolicy::AvailableCase => Ok(complete.to_vec()),
            MissingPolicy::BalancedPanel => {
                let refused = |error: antecedent_data::PanelPreparationError| {
                    estimation(EstimationError::refused(error.code(), error.message().to_string()))
                };
                let report = PanelLayout::assess(
                    &self.query.subjects[..],
                    &self.query.periods[..],
                    complete,
                )
                .map_err(refused)?;
                let prepared = report.prepare(PreparationMode::BalancedUnits).map_err(refused)?;
                Ok(prepared.keep().to_vec())
            }
            policy => Err(super::panel_policy_refusal(
                policy,
                "a rollout reads complete unit-periods (available_case) or complete units \
                 (balanced_panel); it has no estimator for this policy",
            )),
        }
    }

    /// The payload and notes of a fitted rollout.
    #[allow(
        clippy::too_many_lines,
        reason = "a linear assembly of the rollout effects, diagnostics and notes from one fit"
    )]
    fn report(&self, fit: &RolloutDidFit, rows: usize) -> RolloutReport {
        let robust = self.query.robust.clone().unwrap_or_default();
        let two_period = self.query.design == DidSamplingDesign::CovariateAdjustedTwoPeriod;
        let mut method = format!(
            "{}.{}.{}.{}",
            robust.estimator.as_str(),
            robust.comparison.as_str(),
            if robust.covariates.is_empty() { "unadjusted" } else { robust.nuisance.as_str() },
            robust.aggregation.as_str()
        );
        if robust.estimator != RolloutDidEstimator::Imputation {
            method.push_str(&format!(".base_{}", robust.base_period.as_str()));
        }
        if robust.anticipation > 0 {
            method.push_str(&format!(".anticipation_{}", robust.anticipation));
        }
        let mut effects =
            Vec::with_capacity(1 + fit.aggregated.len() + fit.group_time.len() + fit.leads.len());
        effects.push(fit.overall.clone());
        if robust.aggregation != DidAggregation::Simple {
            effects.extend(fit.aggregated.iter().cloned());
        }
        effects.extend(fit.group_time.iter().cloned());
        effects.extend(fit.leads.iter().cloned());
        let mut diagnostics = named(&[
            ("units", fit.units as f64),
            ("treated_units", fit.treated_units as f64),
            ("never_treated_units", fit.never_treated_units as f64),
            ("clusters", fit.clusters as f64),
            ("rows_read", fit.rows as f64),
            ("rows_supplied", rows as f64),
            ("balanced_panel", if fit.balanced { 1.0 } else { 0.0 }),
        ]);
        if let Some(twfe) = fit.twfe_static {
            diagnostics.push(("twfe_static_coefficient".to_string(), twfe));
        }
        if let PanelObservationProcess::InformativeBounded(bound) = robust.observation {
            // Assumption-dependent range, not a confidence interval: the sampling intervals
            // below are conditional on the shift and do not cover it.
            let (shift, slope) = (bound.get(), fit.attrition_slope);
            let estimate = fit.overall.estimate;
            diagnostics.push(("attrition_shift_bound".to_string(), shift));
            diagnostics.push(("attrition_slope".to_string(), slope));
            diagnostics.push(("assumption_range_lower".to_string(), estimate - shift * slope));
            diagnostics.push(("assumption_range_upper".to_string(), estimate + shift * slope));
            if slope > 0.0 {
                diagnostics.push(("attrition_tipping_shift".to_string(), estimate.abs() / slope));
            }
        }
        if let Some(statistic) = fit.pretrend_max_abs_t {
            diagnostics.push(("pretrend_max_abs_t".to_string(), statistic));
        }
        if let Some(covers) = fit.pretrend_band_covers_zero {
            diagnostics
                .push(("pretrend_band_covers_zero".to_string(), if covers { 1.0 } else { 0.0 }));
        }
        if !fit.balanced {
            diagnostics.push(("dropped_rows".to_string(), fit.dropped_rows as f64));
            diagnostics.push(("dropped_units".to_string(), fit.dropped_units as f64));
        }
        for support in &fit.cohort_support {
            let cohort = support.cohort;
            for (field, value) in [
                ("units", support.units),
                ("rows", support.rows),
                ("units_at_base", support.units_at_base),
                ("post_cells", support.post_cells),
                ("pre_cells", support.pre_cells),
                ("dropped_rows", support.dropped_rows),
            ] {
                diagnostics.push((format!("cohort_{cohort}_{field}"), value as f64));
            }
        }
        if fit.unimputed_treated_observations > 0 {
            diagnostics.push((
                "unimputed_treated_observations".to_string(),
                fit.unimputed_treated_observations as f64,
            ));
        }
        let mut notes = vec![Diagnostic::new(
            "identification.quasi.rollout.assumptions",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            if two_period {
                "parallel untreated trends conditional on the baseline covariates, overlap of the \
                 covariate distributions and no anticipation are declared identifying \
                 assumptions; two periods permit no pre-trend check"
                    .to_string()
            } else {
                format!(
                    "cohort-specific parallel untreated trends against {} units, {} and the \
                     estimator's own model are declared identifying assumptions; the pre-trend \
                     leads are falsification diagnostics, not a test that establishes them",
                    robust.comparison.as_str().replace('_', "-"),
                    if robust.anticipation == 0 {
                        "no anticipation".to_string()
                    } else {
                        format!("anticipation of at most {} periods", robust.anticipation)
                    }
                )
            },
        )];
        if let Some(twfe) = fit.twfe_static {
            notes.push(Diagnostic::new(
                "diagnostic.quasi.rollout.two_way_fixed_effects_comparison",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "the static two-way fixed-effects coefficient is {twfe:.6} against the \
                     heterogeneity-robust summary {:.6}; a gap indicates effects that vary across \
                     cohorts or with exposure, where the two-way coefficient weights cells \
                     improperly",
                    fit.overall.estimate
                ),
            ));
        }
        if !fit.balanced {
            notes.push(Diagnostic::new(
                "diagnostic.quasi.rollout.unbalanced_panel_support",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "the panel is unbalanced and was read as observed ({} unit-periods of {} \
                     units); the untestable assumption that a unit's chance of being observed is \
                     unrelated to its untreated outcome trend within cohort is declared, not \
                     checked. {} unit-periods and {} units carried no supported comparison and \
                     were dropped; per-cohort support is in the cohort_* diagnostics",
                    fit.rows, fit.units, fit.dropped_rows, fit.dropped_units
                ),
            ));
        }
        if let PanelObservationProcess::InformativeBounded(bound) = robust.observation {
            notes.push(Diagnostic::new(
                "sensitivity.quasi.rollout.attrition_assumption_range",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "the observation process is declared informative: the treated units a cohort \
                     does not observe may change by at most {} (outcome units) more or less than \
                     those it observes, and comparison units are taken as missing at random. The \
                     overall effect then lies in [{:.6}, {:.6}], an assumption-dependent range \
                     (slope {:.6} per unit of shift) that is not a confidence interval and is \
                     distinct from the sampling uncertainty; the sign changes at a shift of {}",
                    bound.get(),
                    fit.overall.estimate - bound.get() * fit.attrition_slope,
                    fit.overall.estimate + bound.get() * fit.attrition_slope,
                    fit.attrition_slope,
                    if fit.attrition_slope > 0.0 {
                        format!("{:.6}", fit.overall.estimate.abs() / fit.attrition_slope)
                    } else {
                        "none (no cohort is missing a cell)".to_string()
                    }
                ),
            ));
        }
        if !fit.dropped_cohorts.is_empty() {
            notes.push(Diagnostic::new(
                "diagnostic.quasi.rollout.dropped_cohorts",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "cohorts {:?} have no period before adoption (minus anticipation) in the panel \
                     and contribute no group-time effect",
                    fit.dropped_cohorts
                ),
            ));
        }
        let (uncertainty, replicates, critical_value) = match &fit.resampling {
            Some(record) => (
                match record.weighting {
                    UnitWeighting::Resample => {
                        "whole_cluster_bootstrap_refit_iqr_se_pointwise_and_sup_t_band"
                    }
                    UnitWeighting::Multiplier => {
                        "whole_cluster_multiplier_refit_iqr_se_pointwise_and_sup_t_band"
                    }
                },
                [record.requested, record.replicates_ok, record.replicates_failed],
                record.critical_value,
            ),
            None => ("point_only_no_resample_requested", [0, 0, 0], None),
        };
        RolloutReport {
            design: if two_period { "covariate_adjusted_did" } else { "staggered_did" },
            method,
            effects,
            uncertainty,
            replicates,
            critical_value,
            diagnostics,
            notes,
        }
    }
}

#[cfg(test)]
mod posterior_sd_tests {
    use super::rollout_draw_sd;

    #[test]
    fn posterior_sd_is_centered_on_draws_and_requires_finite_sample() {
        let expected = 2.0_f64.sqrt();
        assert!((rollout_draw_sd(&[1.0, 3.0]) - expected).abs() < 1e-12);
        let point = 100.0_f64;
        let wrong_point_centered = ((1.0 - point).powi(2) + (3.0 - point).powi(2)).sqrt();
        assert!((rollout_draw_sd(&[1.0, 3.0]) - wrong_point_centered).abs() > 100.0);
        assert!((rollout_draw_sd(&[101.0, 103.0]) - expected).abs() < 1e-12);
        assert_eq!(rollout_draw_sd(&[4.0, 4.0]), 0.0);
        for invalid in [&[][..], &[1.0][..], &[1.0, f64::NAN][..], &[1.0, f64::INFINITY][..]] {
            assert!(rollout_draw_sd(invalid).is_nan());
        }
    }
}
