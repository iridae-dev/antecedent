//! Heterogeneity-robust and covariate-adjusted rollouts, and the whole-cluster resample of the
//! augmented panel.
//!
//! A rollout reads its own panel layout: a declared `available_case` policy keeps every
//! complete unit-period (each 2×2 then uses the units observed at both of its periods), a
//! declared `balanced_panel` policy keeps the units complete in every period through the
//! shared unbalanced-panel preparation, and a structurally unbalanced panel (absent rows) is
//! read as it is by the estimators that allow it. The Sun–Abraham estimator refuses an
//! unbalanced panel and names the next attempt.
//!
//! Uncertainty is a whole-procedure resample of the inference clusters (whole subjects under a
//! declared independent-subject design; a dependent single series is refused), and every
//! interval is unlicensed with calibration `unmeasured`.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::rollout_finish::{RolloutReport, finish_rollout, named};
use super::{
    Arc, CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext,
    StudyResult, TableView, TabularData,
};
use crate::analysis::missing_data::MissingRoute;
use antecedent_core::{
    DeclaredSamplingDesign, DidAggregation, DidResamplingScheme, DidSamplingDesign,
    RolloutDidEstimator,
};
use antecedent_data::{MissingPolicy, PanelLayout, PreparationMode};
use antecedent_estimate::EstimationError;
use antecedent_estimate::heterogeneity_robust_did::{
    RolloutDidFit, RolloutDidSpec, RolloutPanel, RolloutResampling, estimate_rollout_did,
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
        let fit = estimate_rollout_did(&panel, &spec, resampling, ctx).map_err(estimation)?;
        let mut result = finish_rollout(self, data, self.query.outcome, self.report(&fit, n));
        result.missing_data = Some(self.missing_report(data, &complete, &kept)?);
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
        if let Some(statistic) = fit.pretrend_max_abs_t {
            diagnostics.push(("pretrend_max_abs_t".to_string(), statistic));
        }
        if let Some(covers) = fit.pretrend_band_covers_zero {
            diagnostics
                .push(("pretrend_band_covers_zero".to_string(), if covers { 1.0 } else { 0.0 }));
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
