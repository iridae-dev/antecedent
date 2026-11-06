//! Anomaly and distribution-change uncertainty for checked attribution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AttributionProcedure, CausalError, CausalPosterior, CausalQuery, CheckedAttributionOperation,
    Diagnostic, EffectEstimate, ExecutionContext, OverlapPolicy, TableView, TabularData,
    VariableId, admit_posterior_columns, nan_effect, parametric_scm_identification,
};

impl CheckedAttributionOperation {
    /// Frequentist mechanism-refit bootstrap of the first target's mean anomaly score:
    /// the point is that mean and the interval its bootstrap standard error.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "the bounded operation keeps its execution inputs and resource accounting together"
    )]
    pub(super) fn anomaly_bootstrap(
        &self,
        query: &antecedent_core::AnomalyAttributionQuery,
        data: &TabularData,
        fitted: &crate::gcm::FittedGcm,
        observed_scores: &[antecedent_attribution::AnomalyScores],
        ctx: &ExecutionContext,
    ) -> Result<(EffectEstimate, Option<Diagnostic>), CausalError> {
        let AttributionProcedure::RowBootstrap { replicates } = self.procedure else {
            return Ok((nan_effect(), None));
        };
        let Some(primary) = observed_scores.first() else {
            return Ok((nan_effect(), None));
        };
        let n = data.row_count();
        let n_rows = n.checked_mul(2).ok_or_else(|| CausalError::Resource {
            message: "anomaly bootstrap row count overflow".to_owned(),
        })?;
        let _admitted = super::super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, n_rows)?;
        let unit = vec![1.0_f64; n];
        let point = super::super::attribution_path::dirichlet_weighted_mean(
            &primary.scores,
            &primary.rows,
            &unit,
        );
        let registry = crate::gcm::MechanismRegistry::standard();
        let scorer_query = if self.temporal_input.is_some() {
            query.clone()
        } else {
            antecedent_core::AnomalyAttributionQuery::new(query.targets.clone(), query.max_units)
                .with_reference(query.reference)
        };
        let cancelled = || ctx.cancellation.is_cancelled();
        let mut values = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
        let mut failed = 0_u32;
        for replicate in 0..replicates {
            if cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            let mut rng =
                super::super::gcm_bootstrap::replicate_stream(ctx, 0xA110_B007, replicate);
            let counts = self.draw_row_counts(
                n,
                &mut rng,
                self.full_temporal_runs(),
                u64::from(replicate),
                ctx,
            )?;
            let store = match registry.refit_frequency_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                &counts,
                &cancelled,
            ) {
                Ok(store) => store,
                Err(_) if cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(error @ antecedent_model::ModelError::ResourceLimit { .. })
                    if self.temporal_input.is_some() =>
                {
                    return Err(CausalError::from(error));
                }
                Err(_) => {
                    failed += 1;
                    continue;
                }
            };
            match self.score_anomalies(
                &fitted.model.clone().with_mechanisms(store),
                data,
                &scorer_query,
                ctx,
            ) {
                Ok(draw_scores) => {
                    // The resampled population's mean score: the replicate's row counts
                    // weight the per-row scores, as the Dirichlet weights do on the
                    // Bayesian cell.
                    let value = draw_scores
                        .iter()
                        .find(|score| score.target == primary.target)
                        .map_or(f64::NAN, |score| {
                            super::super::attribution_path::dirichlet_weighted_mean(
                                &score.scores,
                                &score.rows,
                                &counts,
                            )
                        });
                    if self.temporal_input.is_some() && !value.is_finite() {
                        failed += 1;
                    } else {
                        values.push(value);
                    }
                }
                Err(_) if cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. }))
                    if self.temporal_input.is_some() =>
                {
                    return Err(error);
                }
                Err(_) => failed += 1,
            }
        }
        let accounting = super::super::gcm_bootstrap::accounting(&values, failed, replicates);
        let mut estimate = EffectEstimate::new(
            point,
            f64::NAN,
            antecedent_core::AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        );
        estimate.uncertainty.bootstrap = accounting;
        let diagnostic = self.bootstrap_disclosure(
            &format!("mean anomaly score of target v{}", primary.target.raw()),
            n,
            &mut estimate,
        );
        Ok((estimate, Some(diagnostic)))
    }

    /// Frequentist mechanism-refit bootstrap of the total distribution change: each
    /// replicate draws row counts within each population and refits both.
    pub(super) fn change_bootstrap(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        model: &crate::gcm::CompiledCausalModel,
        options: &antecedent_attribution::DistributionChangeOptions,
        ctx: &ExecutionContext,
    ) -> Result<Option<(antecedent_estimate::BootstrapAccounting, Diagnostic)>, CausalError> {
        let AttributionProcedure::RowBootstrap { replicates } = self.procedure else {
            return Ok(None);
        };
        let baseline_rows = antecedent_attribution::resolve_rows(data, &query.baseline)
            .map_err(CausalError::rejected)?;
        let comparison_rows = antecedent_attribution::resolve_rows(data, &query.comparison)
            .map_err(CausalError::rejected)?;
        let baseline_n = baseline_rows.len();
        let comparison_n = comparison_rows.len();
        let baseline_runs = self.selected_temporal_runs(&baseline_rows, ctx)?;
        let comparison_runs = self.selected_temporal_runs(&comparison_rows, ctx)?;
        let rows = baseline_n.checked_add(comparison_n).ok_or_else(|| CausalError::Resource {
            message: "change bootstrap row count overflow".to_owned(),
        })?;
        let _admitted = super::super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, rows)?;
        let mut totals = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
        let mut failed = 0_u32;
        for replicate in 0..replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            let mut rng =
                super::super::gcm_bootstrap::replicate_stream(ctx, 0xC4A6_B007, replicate);
            let baseline_counts = self.draw_row_counts(
                baseline_n,
                &mut rng,
                baseline_runs
                    .as_ref()
                    .map(super::super::temporal_attribution::SelectedRuns::ranges),
                antecedent_core::derive_stream(u64::from(replicate), 0),
                ctx,
            )?;
            let comparison_counts = self.draw_row_counts(
                comparison_n,
                &mut rng,
                comparison_runs
                    .as_ref()
                    .map(super::super::temporal_attribution::SelectedRuns::ranges),
                antecedent_core::derive_stream(u64::from(replicate), 1),
                ctx,
            )?;
            match self.change_attribution(
                model,
                data,
                query,
                options,
                Some(antecedent_attribution::PopulationRefit::Counts {
                    baseline: &baseline_counts,
                    comparison: &comparison_counts,
                }),
                ctx,
            ) {
                Ok(draw) => totals.push(draw.total_change),
                Err(_) if ctx.cancellation.is_cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(
                    error @ (antecedent_attribution::AttributionError::Cancelled
                    | antecedent_attribution::AttributionError::Budget { .. }),
                ) if self.temporal_input.is_some() => return Err(CausalError::from(error)),
                Err(antecedent_attribution::AttributionError::Model(
                    error @ antecedent_model::ModelError::ResourceLimit { .. },
                )) if self.temporal_input.is_some() => return Err(CausalError::from(error)),
                Err(_) => failed += 1,
            }
        }
        let accounting = super::super::gcm_bootstrap::accounting(&totals, failed, replicates);
        let diagnostic = match &self.lag_aligned {
            Some(_) => super::super::gcm_bootstrap::bootstrap_diagnostic_over(
                "total distribution change",
                &format!(
                    "v2 run-aware circular-block bootstrap (nominal block {}; disconnected segments use derived streams and wrap only within their own run)",
                    self.temporal_block_length(baseline_n.min(comparison_n))
                ),
                &accounting,
            ),
            None => super::super::gcm_bootstrap::bootstrap_diagnostic(
                "total distribution change",
                &accounting,
            ),
        };
        Ok(Some((accounting, diagnostic)))
    }

    pub(super) fn anomaly_posterior(
        &self,
        query: &antecedent_core::AnomalyAttributionQuery,
        data: &TabularData,
        fitted: &crate::gcm::FittedGcm,
        observed_scores: &[antecedent_attribution::AnomalyScores],
        outcome: VariableId,
        ctx: &ExecutionContext,
    ) -> Result<Option<CausalPosterior>, CausalError> {
        let AttributionProcedure::SharedDirichletRowWeights { draws } = self.procedure else {
            return Ok(None);
        };
        let _admitted = admit_posterior_columns(ctx, draws, observed_scores.len())?;
        let mut columns = observed_scores
            .iter()
            .map(|score| {
                (format!("mean_anomaly_score[{}]", score.target.raw()), Vec::with_capacity(draws))
            })
            .collect::<Vec<_>>();
        let registry = crate::gcm::MechanismRegistry::standard();
        let scorer_query = if self.temporal_input.is_some() {
            query.clone()
        } else {
            antecedent_core::AnomalyAttributionQuery::new(query.targets.clone(), query.max_units)
                .with_reference(query.reference)
        };
        let mut rng = ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xA110_7A7E);
        for replicate in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let weights = self.draw_row_weights(
                data.row_count(),
                &mut rng,
                self.full_temporal_runs(),
                replicate as u64,
                ctx,
            )?;
            let store = registry
                .refit_weighted(&fitted.model, data, &fitted.assignments, &weights)
                .map_err(|error| {
                    if self.temporal_input.is_some() {
                        CausalError::from(error)
                    } else {
                        CausalError::rejected(error)
                    }
                })?;
            let draw_scores = self.score_anomalies(
                &fitted.model.clone().with_mechanisms(store),
                data,
                &scorer_query,
                ctx,
            )?;
            if draw_scores.len() != columns.len() {
                return Err(CausalError::Compile {
                    message: "Bayesian anomaly draw changed target set".into(),
                });
            }
            for (column, score) in columns.iter_mut().zip(draw_scores) {
                // Bayesian bootstrap of the mean anomaly score under the shared Dirichlet
                // row weights, for both references: the factual scores do not depend on the
                // mechanism refit, so an unweighted mean would be identical every draw (a
                // degenerate posterior). See `execute_anomaly` for the full rationale.
                column.1.push(super::super::attribution_path::dirichlet_weighted_mean(
                    &score.scores,
                    &score.rows,
                    &weights,
                ));
            }
        }
        let identification = if self.temporal_input.is_some() {
            self.cache.identification.clone()
        } else {
            parametric_scm_identification(
                CausalQuery::AnomalyAttribution(query.clone()),
                outcome,
                outcome,
            )
            .0
        };
        Ok(Some(super::super::attribution_path::attribution_posterior(
            columns,
            identification.required_assumptions,
            identification.status,
            super::super::ANOMALY_DIRICHLET_BACKEND,
            false,
        )?))
    }

    pub(super) fn change_posterior(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        model: &crate::gcm::CompiledCausalModel,
        options: &antecedent_attribution::DistributionChangeOptions,
        ctx: &ExecutionContext,
    ) -> Result<Option<CausalPosterior>, CausalError> {
        let AttributionProcedure::SharedDirichletRowWeights { draws } = self.procedure else {
            return Ok(None);
        };
        let posterior_columns =
            self.change_posterior_columns(query, data, model, options, draws, ctx)?;
        let identification = if self.temporal_input.is_some() {
            self.cache.identification.clone()
        } else {
            parametric_scm_identification(
                CausalQuery::ChangeAttribution(query.clone()),
                query.outcome,
                query.outcome,
            )
            .0
        };
        Ok(Some(super::super::attribution_path::attribution_posterior(
            posterior_columns,
            identification.required_assumptions,
            identification.status,
            super::super::CHANGE_DIRICHLET_BACKEND,
            true,
        )?))
    }

    /// The posterior columns of a Dirichlet row-weight bootstrap of the distribution change:
    /// the total first, then one column per component, one weight vector per population per
    /// draw.
    fn change_posterior_columns(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        model: &crate::gcm::CompiledCausalModel,
        options: &antecedent_attribution::DistributionChangeOptions,
        draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, Vec<f64>)>, CausalError> {
        let baseline_rows = antecedent_attribution::resolve_rows(data, &query.baseline)
            .map_err(CausalError::rejected)?;
        let comparison_rows = antecedent_attribution::resolve_rows(data, &query.comparison)
            .map_err(CausalError::rejected)?;
        let baseline_n = baseline_rows.len();
        let comparison_n = comparison_rows.len();
        let baseline_runs = self.selected_temporal_runs(&baseline_rows, ctx)?;
        let comparison_runs = self.selected_temporal_runs(&comparison_rows, ctx)?;
        let mut rng = ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xC4A6_7A7E);
        let _totals_admitted = admit_posterior_columns(ctx, draws, 1)?;
        let mut component_admitted = None;
        let mut components: Option<Vec<(String, Vec<f64>)>> = None;
        let mut totals = Vec::with_capacity(draws);
        for replicate in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let baseline_weights = self.draw_row_weights(
                baseline_n,
                &mut rng,
                baseline_runs
                    .as_ref()
                    .map(super::super::temporal_attribution::SelectedRuns::ranges),
                antecedent_core::derive_stream(replicate as u64, 0),
                ctx,
            )?;
            let comparison_weights = self.draw_row_weights(
                comparison_n,
                &mut rng,
                comparison_runs
                    .as_ref()
                    .map(super::super::temporal_attribution::SelectedRuns::ranges),
                antecedent_core::derive_stream(replicate as u64, 1),
                ctx,
            )?;
            let draw = self
                .change_attribution(
                    model,
                    data,
                    query,
                    options,
                    Some(antecedent_attribution::PopulationRefit::RowWeights {
                        baseline: &baseline_weights,
                        comparison: &comparison_weights,
                    }),
                    ctx,
                )
                .map_err(CausalError::rejected)?;
            totals.push(draw.total_change);
            let columns = components.get_or_insert_with(|| {
                draw.contributions
                    .iter()
                    .map(|c| {
                        (
                            format!("component[{}]", c.component.variable().raw()),
                            Vec::with_capacity(draws),
                        )
                    })
                    .collect()
            });
            if draw.contributions.len() != columns.len() {
                return Err(CausalError::Compile {
                    message: "Bayesian change attribution draw changed component set".into(),
                });
            }
            if component_admitted.is_none() {
                // The component count is known once the first draw returns.
                component_admitted = Some(admit_posterior_columns(ctx, draws, columns.len())?);
            }
            for (column, component) in columns.iter_mut().zip(draw.contributions.iter()) {
                column.1.push(component.contribution);
            }
        }
        let mut posterior_columns = vec![("total_change".to_owned(), totals)];
        posterior_columns.extend(components.unwrap_or_default());
        Ok(posterior_columns)
    }
}
