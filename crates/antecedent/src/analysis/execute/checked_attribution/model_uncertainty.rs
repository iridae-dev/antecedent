//! Fitted-model attribution resampling and uncertainty.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AttributionProcedure, CausalError, CausalPosterior, CausalQuery, CheckedAttributionOperation,
    Diagnostic, EffectEstimate, ExecutionContext, Replicates, SampledWeights, TableView,
    TabularData, admit_posterior_columns, attribute_intrinsic_target,
    attribution_quantities_weighted, change_quantities, difference_measure,
    disclose_model_bootstrap, intrinsic_posterior_columns, population_refit,
};

impl CheckedAttributionOperation {
    /// The posterior over the mechanisms of a fitted-model target: the Dirichlet-weighted
    /// refits of the attribution, or the refit contributions the intrinsic-influence engine
    /// returned.
    pub(super) fn model_posterior(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        answer: &super::super::structural_route::ModelAnswer,
        draws: usize,
        observed: Option<&super::super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<CausalPosterior, CausalError> {
        let (columns, has_primary_effect) = if attribute_intrinsic_target(query) {
            (intrinsic_posterior_columns(&answer.slot)?, false)
        } else {
            let replicates = self.attribute_replicates(query, data, draws, true, observed, ctx)?;
            let primary = replicates.names.first().is_some_and(|name| name == "total_change");
            (replicates.columns(), primary)
        };
        super::super::attribution_path::attribution_posterior(
            columns,
            self.cache.identification.required_assumptions.clone(),
            self.cache.identification.status,
            super::super::ATTRIBUTE_DIRICHLET_BACKEND,
            has_primary_effect,
        )
    }

    /// The mechanism-refit bootstrap of a fitted-model target: the standard error of the
    /// primary quantity on the estimate, and every quantity's standard error in a diagnostic.
    pub(super) fn model_bootstrap(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        replicates: u32,
        answer: &mut super::super::structural_route::ModelAnswer,
        observed: Option<&super::super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let draws = usize::try_from(replicates).unwrap_or(usize::MAX);
        let outcome = self.attribute_replicates(query, data, draws, false, observed, ctx)?;
        let diagnostic = disclose_model_bootstrap(
            &outcome,
            replicates,
            &mut answer.estimate,
            |target, estimate| self.bootstrap_disclosure(target, data.row_count(), estimate),
        );
        answer.diagnostics.push(diagnostic);
        Ok(())
    }

    /// Recompute a fitted-model attribution on `replicates` refits of the mechanisms: each
    /// under fresh Dirichlet (row or block) weights when `bayesian`, else fresh bootstrap
    /// counts. The mechanism families are the ones selected on the original rows, as on the
    /// anomaly and change cells.
    ///
    /// A failed replicate is an error under weights and failed mass under counts; it is never
    /// redrawn.
    pub(super) fn attribute_replicates(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        replicates: usize,
        bayesian: bool,
        observed: Option<&super::super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<Replicates, CausalError> {
        let (attribute, simulate) = match query {
            CausalQuery::Attribute(attribute) => (Some(attribute), None),
            CausalQuery::Simulate(simulate) => (None, Some(simulate)),
            _ => {
                return Err(CausalError::Compile {
                    message: "a mechanism-refit replicate needs an attribution or rollout query"
                        .into(),
                });
            }
        };
        let fitted = self.mechanisms.fit(&self.graph, data)?;
        let mut stream =
            ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xA77B_7A7E);
        let mut out = Replicates { names: Vec::new(), rows: Vec::new(), failed: 0 };
        let mut admitted = None;
        for replicate in 0..replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            let mut counts_stream;
            let rng = if bayesian {
                &mut stream
            } else {
                counts_stream = super::super::gcm_bootstrap::replicate_stream(
                    ctx,
                    0xA77B_B007,
                    u32::try_from(replicate).unwrap_or(u32::MAX),
                );
                &mut counts_stream
            };
            let replicate = match (attribute, simulate) {
                (Some(attribute), _) => self.attribute_replicate(
                    attribute,
                    &fitted,
                    data,
                    bayesian,
                    rng,
                    replicate as u64,
                    ctx,
                ),
                (None, Some(simulate)) => self.simulate_replicate(
                    simulate,
                    &fitted,
                    data,
                    bayesian,
                    rng,
                    replicate as u64,
                    observed,
                    ctx,
                ),
                (None, None) => unreachable!("a replicate query is an attribution or a rollout"),
            };
            match replicate {
                Ok(values) => {
                    if out.rows.is_empty() {
                        out.names = values.iter().map(|(name, _)| name.clone()).collect();
                        admitted = Some(admit_posterior_columns(ctx, replicates, out.names.len())?);
                    } else if values.len() != out.names.len()
                        || values.iter().zip(&out.names).any(|((name, _), first)| name != first)
                    {
                        return Err(CausalError::Compile {
                            message: "a mechanism-refit replicate changed the attribution's \
                                      components"
                                .into(),
                        });
                    }
                    out.rows.push(values.into_iter().map(|(_, value)| value).collect());
                }
                Err(_) if ctx.cancellation.is_cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. }))
                    if self.temporal_input.is_some() =>
                {
                    return Err(error);
                }
                Err(error) if bayesian => return Err(error),
                Err(_) => out.failed += 1,
            }
        }
        drop(admitted);
        if out.rows.len() < 2 && bayesian {
            return Err(CausalError::Compile {
                message: "Bayesian attribution draws have inconsistent shape".into(),
            });
        }
        Ok(out)
    }

    /// One replicate of a temporal rollout: the mechanisms refit under fresh block weights (or
    /// block counts), the dynamics rolled forward from the last rows of the series, and the mean
    /// of every variable at every step read off the rollout.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "the bounded operation keeps its execution inputs and resource accounting together"
    )]
    pub(super) fn simulate_replicate(
        &self,
        simulate: &antecedent_core::SimulateQuery,
        fitted: &crate::gcm::FittedGcm,
        data: &TabularData,
        bayesian: bool,
        rng: &mut antecedent_core::CausalRng,
        replicate: u64,
        observed: Option<&super::super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, f64)>, CausalError> {
        let n = data.row_count();
        let weights = if bayesian {
            self.draw_row_weights(n, rng, self.full_temporal_runs(), replicate, ctx)?
        } else {
            self.draw_row_counts(n, rng, self.full_temporal_runs(), replicate, ctx)?
        };
        let registry = crate::gcm::MechanismRegistry::standard();
        let cancelled = || ctx.cancellation.is_cancelled();
        let store = if bayesian {
            registry.refit_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                &weights,
                &cancelled,
            )
        } else {
            registry.refit_frequency_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                &weights,
                &cancelled,
            )
        }
        .map_err(|error| {
            if self.temporal_input.is_some() {
                CausalError::from(error)
            } else {
                CausalError::rejected(error)
            }
        })?;
        let refit = crate::gcm::FittedGcm {
            model: fitted.model.clone().with_mechanisms(store),
            assignments: fitted.assignments.clone(),
        };
        let outcome = match &self.lag_aligned {
            Some(aligned) => super::super::structural_route::rollout_on_unfolding(
                &refit,
                data,
                aligned,
                observed,
                self.temporal_budget.as_ref(),
                simulate,
                ctx,
            )?,
            None => crate::analysis::structure_route::run_simulate(&refit, data, simulate, ctx)?,
        };
        let nodes = self.lag_aligned.as_ref().map_or(
            refit.model.n_nodes(),
            super::super::temporal_attribution::LagAlignedScm::contemporaneous_count,
        );
        let mut quantities = crate::analysis::structure_route::simulation_quantities(
            &outcome, nodes,
        )
        .ok_or_else(|| CausalError::Compile {
            message: "this simulation has no mechanism-refit replicate".into(),
        })?;
        // The means above reuse the query's seed in every refit (common random numbers keep the
        // refit spread free of Monte Carlo noise). The predictive law of a path needs fresh
        // innovations in every refit, so one path is rolled out under a seed drawn per replicate.
        if let antecedent_core::SimulationTarget::TemporalRollout {
            schedule,
            n_rows,
            n_steps,
            window,
            history,
            ..
        } = &simulate.target
        {
            let single = antecedent_core::SimulateQuery::new(
                antecedent_core::SimulationTarget::TemporalRollout {
                    schedule: schedule.clone(),
                    // A supplied history has one value per row, so it keeps its row count.
                    n_rows: if history.is_empty() { 1 } else { *n_rows },
                    n_steps: *n_steps,
                    window: *window,
                    history: history.clone(),
                    seed: rng.next_u64(),
                },
            );
            let path_outcome = match &self.lag_aligned {
                Some(aligned) => super::super::structural_route::rollout_on_unfolding(
                    &refit,
                    data,
                    aligned,
                    observed,
                    self.temporal_budget.as_ref(),
                    &single,
                    ctx,
                )?,
                None => crate::analysis::structure_route::run_simulate(&refit, data, &single, ctx)?,
            };
            quantities.extend(
                crate::analysis::structure_route::simulated_path_quantities(&path_outcome, nodes)
                    .ok_or_else(|| CausalError::Compile {
                    message: "a rollout replicate has no simulated path".into(),
                })?,
            );
        }
        Ok(quantities)
    }

    /// One replicate of a fitted-model attribution on mechanisms refit under fresh weights.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "the bounded operation keeps its execution inputs and resource accounting together"
    )]
    pub(super) fn attribute_replicate(
        &self,
        attribute: &antecedent_core::AttributeQuery,
        fitted: &crate::gcm::FittedGcm,
        data: &TabularData,
        bayesian: bool,
        rng: &mut antecedent_core::CausalRng,
        replicate: u64,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, f64)>, CausalError> {
        use antecedent_core::{AttributeTarget, ChangeEstimator};
        let draw = |op: &Self,
                    n: usize,
                    positions: Option<&[usize]>,
                    rng: &mut antecedent_core::CausalRng,
                    key: u64| {
            let partition =
                positions.map(|rows| op.selected_temporal_runs(rows, ctx)).transpose()?.flatten();
            let runs = partition
                .as_ref()
                .map(super::super::temporal_attribution::SelectedRuns::ranges)
                .or_else(|| if positions.is_none() { op.full_temporal_runs() } else { None });
            if bayesian {
                op.draw_row_weights(n, rng, runs, key, ctx)
            } else {
                op.draw_row_counts(n, rng, runs, key, ctx)
            }
        };
        let draw_populations = |change: &antecedent_core::ChangeAttributionQuery,
                                rng: &mut antecedent_core::CausalRng|
         -> Result<(SampledWeights, SampledWeights), CausalError> {
            let baseline_rows = antecedent_attribution::resolve_rows(data, &change.baseline)
                .map_err(CausalError::rejected)?;
            let comparison_rows = antecedent_attribution::resolve_rows(data, &change.comparison)
                .map_err(CausalError::rejected)?;
            let baseline = draw(
                self,
                baseline_rows.len(),
                Some(&baseline_rows),
                rng,
                antecedent_core::derive_stream(replicate, 0),
            )?;
            let comparison = draw(
                self,
                comparison_rows.len(),
                Some(&comparison_rows),
                rng,
                antecedent_core::derive_stream(replicate, 1),
            )?;
            Ok((baseline, comparison))
        };
        match &attribute.target {
            AttributeTarget::DistributionChange { change, estimator }
            | AttributeTarget::RootCauses { change, estimator } => {
                let (baseline, comparison) = draw_populations(change, rng)?;
                let refit = population_refit(bayesian, &baseline, &comparison);
                let result = match estimator {
                    ChangeEstimator::Sampled { measure, n_samples, seed } => {
                        let options = antecedent_attribution::DistributionChangeOptions {
                            measure: difference_measure(*measure),
                            n_samples: *n_samples,
                            seed: *seed,
                        };
                        if bayesian {
                            antecedent_attribution::distribution_change_with_row_weights(
                                &fitted.model,
                                data,
                                change,
                                &options,
                                &baseline,
                                &comparison,
                                ctx,
                            )
                        } else {
                            antecedent_attribution::distribution_change_with_frequency_weights(
                                &fitted.model,
                                data,
                                change,
                                &options,
                                &baseline,
                                &comparison,
                                ctx,
                            )
                        }
                    }
                    ChangeEstimator::Robust { max_rows, n_samples, seed } => {
                        antecedent_attribution::distribution_change_robust_with_refit(
                            &fitted.model,
                            data,
                            change,
                            &antecedent_attribution::RobustChangeOptions {
                                max_rows: *max_rows,
                                n_samples: *n_samples,
                                seed: *seed,
                            },
                            &refit,
                            ctx,
                        )
                    }
                }
                .map_err(CausalError::rejected)?;
                Ok(change_quantities(&result))
            }
            AttributeTarget::StructureChange { change, comparison, measure, n_samples, seed } => {
                let (baseline, comparison_weights) = draw_populations(change, rng)?;
                let refit = population_refit(bayesian, &baseline, &comparison_weights);
                let comparison_model =
                    crate::analysis::structure_route::structure_comparison_model(
                        &self.graph,
                        comparison,
                    )?;
                let result = antecedent_attribution::structure_change_with_refit(
                    &fitted.model,
                    &comparison_model,
                    data,
                    change,
                    &antecedent_attribution::StructureChangeOptions {
                        measure: difference_measure(*measure),
                        n_samples: *n_samples,
                        seed: *seed,
                    },
                    &refit,
                    ctx,
                )
                .map_err(CausalError::rejected)?;
                Ok(change_quantities(&result))
            }
            AttributeTarget::MechanismChange { .. } => Err(CausalError::Unsupported {
                message: "mechanism-change detection is a test, not a smooth functional of the \
                          refitted mechanisms",
            }),
            AttributeTarget::IntrinsicInfluence(_) => Err(CausalError::Compile {
                message: "intrinsic influence draws its refits inside the engine".into(),
            }),
            _ => {
                // Paths, feature relevance, anomaly scores and unit change read the fitted
                // mechanisms (and, for the last three, the rows): refit them on the whole
                // table under one weight vector and recompute.
                let weights = draw(self, data.row_count(), None, rng, replicate)?;
                self.whole_table_replicate(attribute, fitted, data, bayesian, &weights, ctx)
            }
        }
    }

    /// The quantities of an attribution recomputed on mechanisms refit on the whole table under
    /// one weight vector.
    pub(super) fn whole_table_replicate(
        &self,
        attribute: &antecedent_core::AttributeQuery,
        fitted: &crate::gcm::FittedGcm,
        data: &TabularData,
        bayesian: bool,
        weights: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, f64)>, CausalError> {
        let registry = crate::gcm::MechanismRegistry::standard();
        let store = if bayesian {
            registry.refit_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                weights,
                &|| ctx.cancellation.is_cancelled(),
            )
        } else {
            registry.refit_frequency_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                weights,
                &|| ctx.cancellation.is_cancelled(),
            )
        }
        .map_err(|error| {
            if self.temporal_input.is_some() {
                CausalError::from(error)
            } else {
                CausalError::rejected(error)
            }
        })?;
        let refit = crate::gcm::FittedGcm {
            model: fitted.model.clone().with_mechanisms(store),
            assignments: fitted.assignments.clone(),
        };
        let outcome = crate::analysis::structure_route::run_attribute_with_fit(
            &refit,
            &self.graph,
            data,
            attribute,
            None,
            crate::analysis::structure_route::ChangeReading::default(),
            None,
            None,
            ctx,
        )?;
        attribution_quantities_weighted(&outcome, weights).ok_or_else(|| CausalError::Compile {
            message: "this attribution has no mechanism-refit replicate".into(),
        })
    }

    /// Draw count published on the result: absent for point procedures.
    pub(super) fn published_draws(&self) -> Option<u32> {
        match self.procedure {
            AttributionProcedure::SharedDirichletRowWeights { draws } => {
                Some(u32::try_from(draws).unwrap_or(u32::MAX))
            }
            AttributionProcedure::Point | AttributionProcedure::RowBootstrap { .. } => None,
        }
    }

    /// One frequentist replicate's row counts over `n` rows: multinomial single rows on a
    /// static table, circular blocks of consecutive times on a lag-aligned unfolding.
    pub(super) fn draw_row_counts(
        &self,
        n: usize,
        rng: &mut antecedent_core::CausalRng,
        runs: Option<&[std::ops::Range<usize>]>,
        replicate: u64,
        ctx: &ExecutionContext,
    ) -> Result<SampledWeights, CausalError> {
        match (&self.lag_aligned, runs, &self.temporal_budget) {
            (Some(_), Some(runs), Some(budget)) => {
                let mut budget = budget.clone();
                antecedent_attribution::temporal_resampling::bootstrap_run_counts(
                    n,
                    self.temporal_block_length(n),
                    runs,
                    replicate,
                    rng,
                    &mut budget,
                    ctx,
                )
                .map(SampledWeights::Run)
                .map_err(CausalError::from)
            }
            (Some(_), None, _) => Err(CausalError::Compile {
                message: "temporal resampling lost its row partition".into(),
            }),
            (None, _, _) => {
                Ok(SampledWeights::Legacy(super::super::gcm_bootstrap::bootstrap_counts(rng, n)))
            }
            _ => Err(CausalError::Compile {
                message: "temporal resampling lost its shared budget".into(),
            }),
        }
    }

    pub(super) fn selected_temporal_runs(
        &self,
        positions: &[usize],
        ctx: &ExecutionContext,
    ) -> Result<Option<super::super::temporal_attribution::SelectedRuns>, CausalError> {
        match (&self.temporal_input, &self.temporal_budget) {
            (Some(input), Some(budget)) => {
                let mut budget = budget.clone();
                Ok(Some(input.selected_runs(positions, &mut budget, ctx)?))
            }
            (None, None) => Ok(None),
            _ => Err(CausalError::Compile {
                message: "temporal resampling lost its row owner".into(),
            }),
        }
    }

    pub(super) fn temporal_block_length(&self, n: usize) -> usize {
        self.temporal_input.as_ref().map_or_else(
            || self.lag_aligned.as_ref().map_or(1, |aligned| aligned.block_length(n)),
            |input| input.block_length(n),
        )
    }

    pub(super) fn full_temporal_runs(&self) -> Option<&[std::ops::Range<usize>]> {
        self.temporal_input
            .as_ref()
            .map(super::super::temporal_attribution::CompleteWindowInput::runs)
    }

    /// The bootstrap disclosure of `target` over `n` rows, recording the row-block
    /// family on `estimate` when the rows are one lag-aligned series.
    pub(super) fn bootstrap_disclosure(
        &self,
        target: &str,
        n: usize,
        estimate: &mut EffectEstimate,
    ) -> Diagnostic {
        let accounting = estimate.uncertainty.bootstrap;
        match &self.lag_aligned {
            Some(_) => {
                estimate.uncertainty.block.family =
                    Some(antecedent_estimate::CircularBlockFamily::RowBlock);
                super::super::gcm_bootstrap::bootstrap_diagnostic_over(
                    target,
                    &format!(
                        "v2 run-aware circular-block bootstrap (nominal block {}; disconnected segments use derived streams and wrap only within their own run)",
                        self.temporal_block_length(n)
                    ),
                    &accounting,
                )
            }
            None => super::super::gcm_bootstrap::bootstrap_diagnostic(target, &accounting),
        }
    }

    /// One draw's row weights over `n` rows: Dirichlet row weights on a static table,
    /// block weights on the consecutive times of a lag-aligned unfolding.
    pub(super) fn draw_row_weights(
        &self,
        n: usize,
        rng: &mut antecedent_core::CausalRng,
        runs: Option<&[std::ops::Range<usize>]>,
        replicate: u64,
        ctx: &ExecutionContext,
    ) -> Result<SampledWeights, CausalError> {
        match (&self.lag_aligned, runs, &self.temporal_budget) {
            (Some(_), Some(runs), Some(budget)) => {
                let mut budget = budget.clone();
                antecedent_attribution::temporal_resampling::bayesian_run_weights(
                    n,
                    self.temporal_block_length(n),
                    runs,
                    replicate,
                    rng,
                    &mut budget,
                    ctx,
                )
                .map(SampledWeights::Run)
                .map_err(CausalError::from)
            }
            (Some(_), None, _) => Err(CausalError::Compile {
                message: "temporal resampling lost its row partition".into(),
            }),
            (None, _, _) => Ok(SampledWeights::Legacy(
                super::super::attribution_path::dirichlet_row_weights(n, rng),
            )),
            _ => Err(CausalError::Compile {
                message: "temporal resampling lost its shared budget".into(),
            }),
        }
    }
}
