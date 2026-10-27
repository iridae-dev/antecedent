// SPDX-License-Identifier: MIT OR Apache-2.0

//! Execution of a pulse, sustained or conditional contrast on a temporal ADMG.
//!
//! The contrast is identified on the unrolled window of the template (the claim the plan
//! retained, or a fresh identification), the series is read into the lag-aligned window table,
//! and the identified functional is evaluated by the discrete plug-in on that table. The windows
//! overlap, so the frequentist bootstrap resamples circular blocks of consecutive windows and the
//! Bayesian bootstrap shares one Dirichlet weight per block; neither is ever drawn row by row.

use super::{
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext,
    FunctionalEffect, GraphRef, IdentifiedExecuteExtras, IdentifiedExecuteFinish, Instant,
    PhysicalExecutionPlan, RefuteSuite, Strategy, StudyResult, TemporalEffectQuery, TimeSeriesData,
    VariableId, elapsed_ns, report_identify_compute, run_functional,
};
use crate::analysis::temporal_admg_contrast::{
    identify_window_contrast, used_variables, window_coordinates, window_table,
};
use crate::planner::RouteDefaults;
use antecedent_data::TableView;
use antecedent_estimate::RowResampling;
use antecedent_identify::temporal_admg::plan_temporal_admg_window;

/// Stable code of the disclosure every window result carries.
const WINDOW_DISCLOSURE: &str = "estimate.temporal_admg.window";

impl super::Study {
    /// How the rows of a window table are resampled and reweighted: circular blocks of
    /// consecutive windows, never shorter than the window, under the declared sampling design.
    ///
    /// # Errors
    ///
    /// A declared independent-subject or clustered-panel design when a draw will run: the table
    /// has no subject or cluster labels, and a row-level draw is refused under both.
    pub(super) fn window_row_resampling(
        &self,
        span: u32,
    ) -> Result<(RowResampling, antecedent_estimate::util::SamplingDeclaration), CausalError> {
        let (resampling, design) = self.functional_row_resampling()?;
        Ok((
            match resampling {
                RowResampling::SeriesBlocks => RowResampling::WindowBlocks { span },
                other => other,
            },
            design,
        ))
    }

    /// Identify and estimate a pulse, sustained or conditional contrast on a `TemporalAdmg`.
    ///
    /// # Errors
    ///
    /// The contrast is not identified on the window (the refusal names the hedge by variable and
    /// period), the series holds a missing value or too few complete windows, a custom validator
    /// was attached (none applies to a window contrast), or the declared sampling design refuses
    /// the draw.
    pub(super) fn execute_temporal_admg_contrast(
        &self,
        series: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if !self.declared.custom_validators.is_empty() {
            return Err(crate::unsupported_reason!(
                "validators_not_applicable",
                "a custom validator refutes one scalar average effect and none applies to a \
                 contrast identified on the unrolled window of a temporal ADMG. Next attempt: \
                 drop the validator, or analyse the contrast on a TemporalDag when the lag \
                 structure has no latent confounding"
            ));
        }
        let started = Instant::now();
        let GraphRef::TemporalAdmg(template) = self.graph.graph_ref() else {
            return Err(CausalError::Compile {
                message: "a temporal ADMG contrast requires a supplied TemporalAdmg".into(),
            });
        };
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::ADMG)?;
        let variable_count = u32::try_from(series.schema().len()).map_err(|_| {
            CausalError::Compile { message: "series schema exceeds the variable id range".into() }
        })?;
        // The product compile retained or the prepared handle froze, else a fresh identification.
        let (identification, estimand, identify_cached) =
            self.identify_admg_contrast(physical, template, query, variable_count, ctx)?;
        let plan = plan_temporal_admg_window(template, query)?;
        let window = template.unroll_window_over(plan.periods, variable_count)?;
        let table =
            window_table(series, &window, &used_variables(template, query), &ctx.kernel_policy)?;
        let coordinates = window_coordinates(&window, &plan, query);
        let (resampling, design) = self.window_row_resampling(plan.periods)?;
        let fitter = FunctionalEffect {
            bootstrap_replicates: self.settings.bootstrap_replicates,
            ..FunctionalEffect::new()
        }
        .with_row_resampling(resampling, design);
        let prepared = fitter
            .prepare(
                &table,
                &estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
                &coordinates,
            )
            .map_err(CausalError::from)?;
        let (estimate, posterior) = run_functional(
            &fitter,
            &prepared,
            &self.settings.inference,
            identification.status,
            ctx,
        )?;
        // The window-native suite refits the same plug-in on stretches of the windows, rotates
        // the treatment as a placebo and (under full validation) compares the two halves.
        let refutations = if self.settings.refute == RefuteSuite::None {
            Vec::new()
        } else {
            let treatment_columns: Vec<VariableId> = plan
                .treated
                .iter()
                .filter_map(|&period| window.coordinate(query.treatment, period))
                .collect();
            antecedent_validate::functional::refute_window_contrast(
                &table,
                &identification,
                &estimand,
                &coordinates,
                &treatment_columns,
                plan.periods,
                estimate.ate,
                self.settings.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?
        };
        let disclosure = window_disclosure(
            &plan,
            table.row_count(),
            fitter.row_resampling.block_length(table.row_count()),
            query.effect_modifiers.len(),
        );
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let cancelled = estimate.uncertainty.bootstrap.cancelled;
        let early_stopped = estimate.uncertainty.bootstrap.early_stopped;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .extra_diagnostics(vec![disclosure])
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(bootstrap_ok)
            .cancelled(cancelled)
            .early_stopped(early_stopped)
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            }),
        ))
    }
    /// Reuse the exact horizon proof before identifying this template's window.
    fn identify_admg_contrast(
        &self,
        physical: &PhysicalExecutionPlan,
        template: &antecedent_graph::TemporalAdmg,
        query: &TemporalEffectQuery,
        variable_count: u32,
        ctx: &ExecutionContext,
    ) -> Result<(super::IdentificationResult, super::IdentifiedEstimand, bool), CausalError> {
        let reused = self.reuse_temporal_identification(physical).and_then(|(cache, prepared)| {
            cache
                .get(query.horizon_steps)
                .map(|entry| (entry.identification.clone(), entry.estimand.clone(), prepared))
        });
        Ok(if let Some(found) = reused {
            found
        } else {
            report_identify_compute(ctx);
            let contrast = identify_window_contrast(template, query, variable_count)?;
            (contrast.identification, contrast.estimand, false)
        })
    }
}

/// The disclosure of how the window was read: the premises a window contrast adds to the ones
/// the identification carries, stated where a reader of the result will find them.
pub(super) fn window_disclosure(
    plan: &antecedent_identify::temporal_admg::TemporalAdmgWindowPlan,
    rows: usize,
    block_length: Option<usize>,
    modifiers: usize,
) -> Diagnostic {
    let block = block_length.map_or_else(|| "none".to_owned(), |length| length.to_string());
    Diagnostic::new(
        WINDOW_DISCLOSURE,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "the contrast was identified on the {}-period unrolling of the temporal ADMG (a \
             cause that precedes the window is a latent common cause of its in-window effects) \
             and evaluated on {rows} overlapping windows of the series by a plug-in of the \
             identified functional. Every window is read as a draw from the same law, so the \
             series is assumed stationary over the template. Windows overlap, so uncertainty \
             resamples circular blocks of {block} consecutive windows (never shorter than the \
             window), the declared dependent-series design; the result is unlicensed and no \
             coverage record names it",
            plan.periods,
        ),
    )
    .with_fields([
        ("periods", plan.periods.to_string()),
        ("windows", rows.to_string()),
        ("block_length", block),
        ("treated_periods", plan.treated.len().to_string()),
        ("outcome_period", plan.outcome_period.to_string()),
        ("modifiers", modifiers.to_string()),
    ])
}
