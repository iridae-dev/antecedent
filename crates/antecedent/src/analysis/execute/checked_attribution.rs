//! Builder-independent execution for frequentist DAG attribution operations.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalPosterior, CausalQuery, Dag, DataInput, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext, GcmSlot,
    IdentificationResult, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedResultContext, IdentifierId, InferenceMode, Instant, OverlapPolicy,
    PhysicalExecutionPlan, Strategy, Study, StudyResult, TableView, TabularData, TimeSeriesData,
    VariableId, anomaly_attribution_query_with, attribute_distribution_change, bayesian_draw_count,
    elapsed_ns, finish_identified_execute_with_context, fit_gcm, nan_effect,
    parametric_scm_identification,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::proof_supplies_method_and_functional;

#[derive(Clone, Debug)]
enum AttributionTarget {
    Anomaly(antecedent_core::AnomalyAttributionQuery),
    Change(antecedent_core::ChangeAttributionQuery),
    /// A fitted-model route (attribution, simulation, mechanism change or unit
    /// change) that runs on the mechanisms its program fits.
    Model(CausalQuery),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AttributionProcedure {
    Point,
    SharedDirichletRowWeights {
        draws: usize,
    },
    /// Frequentist whole-procedure multinomial row bootstrap (static DAG only).
    RowBootstrap {
        replicates: u32,
    },
}

/// A frozen GCM attribution target and its prepare-time identification contract.
#[derive(Clone)]
pub(crate) struct CheckedAttributionOperation {
    graph: Dag,
    target: AttributionTarget,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    inference: InferenceMode,
    procedure: AttributionProcedure,
    estimator: EstimatorId,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    result_context: IdentifiedResultContext,
    /// Present when the source graph was a `TemporalDag`. The static `graph` is its unfolding.
    lag_aligned: Option<super::temporal_attribution::LagAlignedScm>,
    /// Mechanism fit of a fitted-model route.
    mechanisms: Arc<crate::analysis::structure_route::MechanismProgram>,
}

impl std::fmt::Debug for CheckedAttributionOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedAttributionOperation")
            .field("target", &self.target)
            .field("estimand", &self.cache.estimand)
            .field("identification_status", &self.cache.identification.status)
            .field("identifier", &IdentifierId::GcmParametric)
            .field("inference", &self.inference)
            .field("procedure", &self.procedure)
            .field("estimator", &self.estimator)
            .finish_non_exhaustive()
    }
}

impl CheckedAttributionOperation {
    pub(crate) fn checked(
        study: &Study,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        cache: &Arc<crate::analysis::prepared::CachedStaticIdentification>,
    ) -> Result<Self, CausalError> {
        let (graph, lag_aligned) = attribution_graph(study)?;
        let target = match &study.query {
            CausalQuery::AnomalyAttribution(query) => {
                query.validate()?;
                AttributionTarget::Anomaly(query.clone())
            }
            CausalQuery::ChangeAttribution(query) => {
                query.validate()?;
                AttributionTarget::Change(query.clone())
            }
            query @ (CausalQuery::MechanismChange(_)
            | CausalQuery::UnitChange(_)
            | CausalQuery::Attribute(_)
            | CausalQuery::Simulate(_)) => {
                query.validate()?;
                AttributionTarget::Model(query.clone())
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "checked attribution supports only anomaly, change and fitted-model \
                              targets"
                        .into(),
                });
            }
        };
        let fitted_model = matches!(target, AttributionTarget::Model(_));
        let target_query = target.query();
        if !study.custom_validators.is_empty() {
            return Err(CausalError::Unsupported {
                message: "checked attribution does not support custom validators",
            });
        }
        let (estimator, procedure) = attribution_procedure(&target, study)?;
        let structure_ok = study.structure_source == crate::support::StructureSource::Explicit
            || (fitted_model
                && study.structure_source == crate::support::StructureSource::Accepted);
        if !structure_ok
            || (fitted_model && lag_aligned.is_some())
            || cache.identification.query != target_query
            || physical.logical.query != target_query
            || !physical.binding().names(IdentifierId::GcmParametric, estimator)
        {
            return Err(CausalError::Compile {
                message: "attribution target, source graph, identification, or estimator does not match the checked operation".into(),
            });
        }
        let estimand = &cache.estimand;
        if !proof_supplies_method_and_functional(&cache.identification, estimand) {
            return Err(CausalError::Compile {
                message: "attribution estimand is absent from its identification result".into(),
            });
        }
        Ok(Self {
            graph: graph.clone(),
            target,
            cache: Arc::clone(cache),
            inference: study.inference.clone(),
            procedure,
            estimator,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
            lag_aligned,
            mechanisms: study.mechanisms.clone().unwrap_or_else(|| {
                Arc::new(crate::analysis::structure_route::MechanismProgram::standard())
            }),
        })
    }

    /// Fit the frozen unfolding on a fresh lag-aligned table.
    ///
    /// # Errors
    ///
    /// The operation was sealed on a static DAG, or fitting and scoring fail.
    pub(crate) fn execute_series(
        &self,
        series: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let aligned = self.lag_aligned.as_ref().ok_or_else(|| CausalError::Compile {
            message: "temporal attribution lost its lag-aligned unfolding".into(),
        })?;
        let table = aligned.table(series)?;
        self.execute(&table, ctx)
    }

    pub(crate) fn query(&self) -> CausalQuery {
        self.target.query()
    }

    pub(crate) fn graph(&self) -> &Dag {
        &self.graph
    }

    pub(crate) fn identification(&self) -> &IdentificationResult {
        &self.cache.identification
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.result_context.query != self.query()
            || self.cache.identification.query != self.query()
            || !proof_supplies_method_and_functional(
                &self.cache.identification,
                &self.cache.estimand,
            )
        {
            return Err(CausalError::Compile {
                message: "attribution operation lost its checked target or identification binding"
                    .into(),
            });
        }
        match &self.target {
            AttributionTarget::Anomaly(query) => self.execute_anomaly(query, data, ctx),
            AttributionTarget::Change(query) => self.execute_change(query, data, ctx),
            AttributionTarget::Model(query) => self.execute_model(query, data, ctx),
        }
    }

    /// Fit the program's mechanisms and run the fitted-model target on them.
    fn execute_model(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let answer = super::structural_route::run_model_query(
            &self.mechanisms,
            &self.graph,
            data,
            query,
            ctx,
        )?;
        let (treatment, outcome) = super::gcm_query_vars(query)?;
        Ok(finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.cache.identification.clone(),
                self.cache.estimand.clone(),
                answer.estimate,
                Strategy::new(IdentifierId::GcmParametric, self.estimator),
                treatment,
                outcome,
            )
            .identify_cached(true)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                diagnostics: Some(answer.diagnostics),
                gcm: Some(answer.slot),
                empty_provenance: true,
                ..Default::default()
            }),
        ))
    }

    /// Draw count published on the result: absent for point procedures.
    fn published_draws(&self) -> Option<u32> {
        match self.procedure {
            AttributionProcedure::SharedDirichletRowWeights { draws } => {
                Some(u32::try_from(draws).unwrap_or(u32::MAX))
            }
            AttributionProcedure::Point | AttributionProcedure::RowBootstrap { .. } => None,
        }
    }

    /// One frequentist replicate's row counts over `n` rows: multinomial single rows on a
    /// static table, circular blocks of consecutive times on a lag-aligned unfolding.
    fn draw_row_counts(&self, n: usize, rng: &mut antecedent_core::CausalRng) -> Vec<f64> {
        match &self.lag_aligned {
            Some(aligned) => aligned.block_row_counts(n, rng),
            None => super::gcm_bootstrap::bootstrap_counts(rng, n),
        }
    }

    /// The bootstrap disclosure of `target` over `n` rows, recording the row-block
    /// family on `estimate` when the rows are one lag-aligned series.
    fn bootstrap_disclosure(
        &self,
        target: &str,
        n: usize,
        estimate: &mut EffectEstimate,
    ) -> Diagnostic {
        let accounting = estimate.uncertainty.bootstrap;
        match &self.lag_aligned {
            Some(aligned) => {
                estimate.uncertainty.block.family =
                    Some(antecedent_estimate::CircularBlockFamily::RowBlock);
                super::gcm_bootstrap::block_bootstrap_diagnostic(
                    target,
                    aligned.block_length(n),
                    &accounting,
                )
            }
            None => super::gcm_bootstrap::bootstrap_diagnostic(target, &accounting),
        }
    }

    /// One draw's row weights over `n` rows: Dirichlet row weights on a static table,
    /// block weights on the consecutive times of a lag-aligned unfolding.
    fn draw_row_weights(&self, n: usize, rng: &mut antecedent_core::CausalRng) -> Vec<f64> {
        match &self.lag_aligned {
            Some(aligned) => aligned.block_row_weights(n, rng),
            None => super::attribution_path::dirichlet_row_weights(n, rng),
        }
    }

    /// Fits the GCM, scores every row against the query reference, and publishes the anomaly attribution.
    fn execute_anomaly(
        &self,
        query: &antecedent_core::AnomalyAttributionQuery,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let fitted = fit_gcm(self.graph.clone(), data)?;
        // Score all rows while honoring the query's reference (Empirical / Fixed).
        let scorer_query =
            antecedent_core::AnomalyAttributionQuery::new(query.targets.clone(), query.max_units)
                .with_reference(query.reference);
        let scores = anomaly_attribution_query_with(&fitted.model, data, &scorer_query, ctx)?;
        let outcome = *query.targets.first().ok_or_else(|| CausalError::Compile {
            message: "checked anomaly target has no outcome variable".into(),
        })?;
        let posterior = self.anomaly_posterior(query, data, &fitted, &scores, outcome, ctx)?;
        let (estimate, bootstrap_diagnostic) =
            self.anomaly_bootstrap(query, data, &fitted, &scores, ctx)?;
        let (estimate, posterior, mut extra_diagnostics) =
            publish_lag_aligned(self.lag_aligned.is_some(), estimate, posterior);
        extra_diagnostics.extend(bootstrap_diagnostic);
        if posterior.is_some() {
            let disclosure = match query.reference {
                antecedent_core::AnomalyReference::Empirical => {
                    "Posterior is the Bayesian bootstrap of the mean anomaly score: each draw reweights the per-row scores by one shared Dirichlet(1,…,1) row-weight vector while holding the observed marginal reference (median / 1.4826·MAD) fixed, so the interval is for the mean relative to that realized sample's reference, not a fixed population functional."
                }
                antecedent_core::AnomalyReference::Fixed { .. } => {
                    "Posterior is the Bayesian bootstrap of the population mean anomaly score against the injected fixed reference: each draw reweights the per-row scores by one shared Dirichlet(1,…,1) row-weight vector, so the interval is a credible interval for the fixed functional μ_A = E_Y[−log 2Φ(−|Y − center|/scale)]."
                }
            };
            extra_diagnostics.push(bayesian_attribution_diagnostic(disclosure));
        }
        Ok(finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.cache.identification.clone(),
                self.cache.estimand.clone(),
                estimate,
                Strategy::new(IdentifierId::GcmParametric, self.estimator),
                outcome,
                outcome,
            )
            .identify_cached(true)
            .extra_diagnostics(extra_diagnostics)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                posterior,
                n_draws: self.published_draws(),
                gcm: Some(GcmSlot::Anomaly(scores)),
                empty_provenance: true,
                ..Default::default()
            }),
        ))
    }

    /// Fits the GCM and publishes the distribution-change attribution with its optional posterior.
    fn execute_change(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let fitted = fit_gcm(self.graph.clone(), data)?;
        let change = attribute_distribution_change(
            &fitted.model,
            data,
            query,
            &antecedent_attribution::DistributionChangeOptions::default(),
            ctx,
        )?;
        let mut estimate = EffectEstimate::new(
            change.total_change,
            f64::NAN,
            antecedent_core::AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        );
        let posterior = self.change_posterior(query, data, &fitted.model, ctx)?;
        let bootstrap_diagnostic = self.change_bootstrap(query, data, &fitted.model, ctx)?.map(
            |(accounting, diagnostic)| {
                estimate.uncertainty.bootstrap = accounting;
                if self.lag_aligned.is_some() {
                    estimate.uncertainty.block.family =
                        Some(antecedent_estimate::CircularBlockFamily::RowBlock);
                }
                diagnostic
            },
        );
        if let Some(posterior) = &posterior {
            if let Some(index) = posterior.effect_column() {
                estimate.ate = posterior.summaries.mean[index];
                estimate.uncertainty.analytic.se = posterior.summaries.sd[index];
            }
        }
        let (estimate, posterior, mut extra_diagnostics) =
            publish_lag_aligned(self.lag_aligned.is_some(), estimate, posterior);
        extra_diagnostics.extend(bootstrap_diagnostic);
        if posterior.is_some() {
            extra_diagnostics.push(bayesian_attribution_diagnostic("Each population has one Dirichlet(1,…,1) row-weight vector per draw shared across mechanisms; total and Shapley components use the same weighted attribution."));
        }
        Ok(finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.cache.identification.clone(),
                self.cache.estimand.clone(),
                estimate,
                Strategy::new(IdentifierId::GcmParametric, self.estimator),
                query.outcome,
                query.outcome,
            )
            .identify_cached(true)
            .extra_diagnostics(extra_diagnostics)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                posterior,
                n_draws: self.published_draws(),
                gcm: Some(GcmSlot::Change(change)),
                empty_provenance: true,
                ..Default::default()
            }),
        ))
    }

    /// Frequentist mechanism-refit bootstrap of the first target's mean anomaly score:
    /// the point is that mean and the interval its bootstrap standard error.
    fn anomaly_bootstrap(
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
        let _admitted = super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, n_rows)?;
        let unit = vec![1.0_f64; n];
        let point =
            super::attribution_path::dirichlet_weighted_mean(&primary.scores, &primary.rows, &unit);
        let registry = crate::gcm::MechanismRegistry::standard();
        let scorer_query =
            antecedent_core::AnomalyAttributionQuery::new(query.targets.clone(), query.max_units)
                .with_reference(query.reference);
        let cancelled = || ctx.cancellation.is_cancelled();
        let mut values = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
        let mut failed = 0_u32;
        for replicate in 0..replicates {
            if cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            let mut rng = super::gcm_bootstrap::replicate_stream(ctx, 0xA110_B007, replicate);
            let counts = self.draw_row_counts(n, &mut rng);
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
                        stage: super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(_) => {
                    failed += 1;
                    continue;
                }
            };
            match anomaly_attribution_query_with(
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
                            super::attribution_path::dirichlet_weighted_mean(
                                &score.scores,
                                &score.rows,
                                &counts,
                            )
                        });
                    values.push(value);
                }
                Err(_) if cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(_) => failed += 1,
            }
        }
        let accounting = super::gcm_bootstrap::accounting(&values, failed, replicates);
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
    fn change_bootstrap(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        model: &crate::gcm::CompiledCausalModel,
        ctx: &ExecutionContext,
    ) -> Result<Option<(antecedent_estimate::BootstrapAccounting, Diagnostic)>, CausalError> {
        let AttributionProcedure::RowBootstrap { replicates } = self.procedure else {
            return Ok(None);
        };
        let baseline_n = antecedent_attribution::resolve_rows(data, &query.baseline)
            .map_err(CausalError::rejected)?
            .len();
        let comparison_n = antecedent_attribution::resolve_rows(data, &query.comparison)
            .map_err(CausalError::rejected)?
            .len();
        let rows = baseline_n.checked_add(comparison_n).ok_or_else(|| CausalError::Resource {
            message: "change bootstrap row count overflow".to_owned(),
        })?;
        let _admitted = super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, rows)?;
        let options = antecedent_attribution::DistributionChangeOptions::default();
        let mut totals = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
        let mut failed = 0_u32;
        for replicate in 0..replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            let mut rng = super::gcm_bootstrap::replicate_stream(ctx, 0xC4A6_B007, replicate);
            let baseline_counts = self.draw_row_counts(baseline_n, &mut rng);
            let comparison_counts = self.draw_row_counts(comparison_n, &mut rng);
            match antecedent_attribution::distribution_change_with_frequency_weights(
                model,
                data,
                query,
                &options,
                &baseline_counts,
                &comparison_counts,
                ctx,
            ) {
                Ok(draw) => totals.push(draw.total_change),
                Err(_) if ctx.cancellation.is_cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(_) => failed += 1,
            }
        }
        let accounting = super::gcm_bootstrap::accounting(&totals, failed, replicates);
        let diagnostic = match &self.lag_aligned {
            Some(aligned) => super::gcm_bootstrap::block_bootstrap_diagnostic(
                "total distribution change",
                aligned.block_length(baseline_n.min(comparison_n)),
                &accounting,
            ),
            None => {
                super::gcm_bootstrap::bootstrap_diagnostic("total distribution change", &accounting)
            }
        };
        Ok(Some((accounting, diagnostic)))
    }

    fn anomaly_posterior(
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
        let scorer_query =
            antecedent_core::AnomalyAttributionQuery::new(query.targets.clone(), query.max_units)
                .with_reference(query.reference);
        let mut rng = ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xA110_7A7E);
        for _ in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let weights = self.draw_row_weights(data.row_count(), &mut rng);
            let store = registry
                .refit_weighted(&fitted.model, data, &fitted.assignments, &weights)
                .map_err(CausalError::rejected)?;
            let draw_scores = anomaly_attribution_query_with(
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
                column.1.push(super::attribution_path::dirichlet_weighted_mean(
                    &score.scores,
                    &score.rows,
                    &weights,
                ));
            }
        }
        let (identification, _) = parametric_scm_identification(
            CausalQuery::AnomalyAttribution(query.clone()),
            outcome,
            outcome,
        );
        Ok(Some(super::attribution_path::attribution_posterior(
            columns,
            identification.required_assumptions,
            identification.status,
            super::ANOMALY_DIRICHLET_BACKEND,
            false,
        )?))
    }

    fn change_posterior(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        model: &crate::gcm::CompiledCausalModel,
        ctx: &ExecutionContext,
    ) -> Result<Option<CausalPosterior>, CausalError> {
        let AttributionProcedure::SharedDirichletRowWeights { draws } = self.procedure else {
            return Ok(None);
        };
        let baseline_n = antecedent_attribution::resolve_rows(data, &query.baseline)
            .map_err(CausalError::rejected)?
            .len();
        let comparison_n = antecedent_attribution::resolve_rows(data, &query.comparison)
            .map_err(CausalError::rejected)?
            .len();
        let options = antecedent_attribution::DistributionChangeOptions::default();
        let mut rng = ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xC4A6_7A7E);
        let _totals_admitted = admit_posterior_columns(ctx, draws, 1)?;
        let mut component_admitted = None;
        let mut components: Option<Vec<(String, Vec<f64>)>> = None;
        let mut totals = Vec::with_capacity(draws);
        for _ in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let baseline_weights = self.draw_row_weights(baseline_n, &mut rng);
            let comparison_weights = self.draw_row_weights(comparison_n, &mut rng);
            let draw = antecedent_attribution::distribution_change_with_row_weights(
                model,
                data,
                query,
                &options,
                &baseline_weights,
                &comparison_weights,
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
        let (identification, _) = parametric_scm_identification(
            CausalQuery::ChangeAttribution(query.clone()),
            query.outcome,
            query.outcome,
        );
        Ok(Some(super::attribution_path::attribution_posterior(
            posterior_columns,
            identification.required_assumptions,
            identification.status,
            super::CHANGE_DIRICHLET_BACKEND,
            true,
        )?))
    }
}

/// Lease `draws * columns` posterior values on the pool before the columns are filled.
fn admit_posterior_columns(
    ctx: &ExecutionContext,
    draws: usize,
    columns: usize,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    let bytes = draws
        .checked_mul(columns)
        .and_then(|cells| cells.checked_mul(size_of::<f64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| CausalError::Resource {
            message: "attribution posterior size overflow".to_owned(),
        })?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: super::super::stage::STAGE_ESTIMATE_POINT }
        }
        other => CausalError::Resource { message: other.to_string() },
    })
}

/// A `TemporalDag` publishes the point, and for Bayesian the posterior mean, with no interval.
fn publish_lag_aligned(
    lag_aligned: bool,
    mut estimate: EffectEstimate,
    posterior: Option<CausalPosterior>,
) -> (EffectEstimate, Option<CausalPosterior>, Vec<Diagnostic>) {
    if !lag_aligned {
        return (estimate, posterior, Vec::new());
    }
    let mut diagnostics = vec![Diagnostic::new(
        "gcm.attribution.lag_aligned",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "attribution ran on the lag-aligned static unfolding of the TemporalDag",
    )];
    if let Some(posterior) = &posterior {
        let mean = posterior.summaries.mean.first().copied().unwrap_or(f64::NAN);
        if mean.is_finite() {
            estimate.ate = mean;
            estimate.uncertainty.analytic.se =
                posterior.summaries.sd.first().copied().unwrap_or(f64::NAN);
        }
        diagnostics.push(Diagnostic::new(
            "gcm.attribution.block_bayesian_bootstrap",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the lag-aligned rows are consecutive times of one dependent series, so each draw's \
             Dirichlet weights fall on contiguous blocks of times (one Exp(1) weight per block \
             of at least the lag span plus one rows, shared by every row of the block); every \
             mechanism is refit under those weights; no coverage record measures this interval",
        ));
    }
    (estimate, posterior, diagnostics)
}

fn bayesian_attribution_diagnostic(detail: &str) -> Diagnostic {
    Diagnostic::new(
        "gcm.attribution.bayesian",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "Posterior is conditional on mechanism-family selection from the original unweighted data. {detail} It does not address model misspecification or causal identification."
        ),
    )
}

impl AttributionTarget {
    fn query(&self) -> CausalQuery {
        match self {
            Self::Anomaly(query) => CausalQuery::AnomalyAttribution(query.clone()),
            Self::Change(query) => CausalQuery::ChangeAttribution(query.clone()),
            Self::Model(query) => query.clone(),
        }
    }
}

/// Read-only target and procedure retained by a checked GCM attribution plan.
#[derive(Clone, Debug)]
pub struct CheckedAttributionInfo {
    /// Exact anomaly or distribution-change target.
    pub query: CausalQuery,
    /// Prepare-time identifier and estimator.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Prepare-time estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Canonical supplied DAG edges.
    pub graph_edges: std::sync::Arc<[(u32, u32)]>,
    /// Identification status fixed at preparation.
    pub identification_status: antecedent_identify::IdentificationStatus,
}

impl Inspect for CheckedAttributionOperation {
    type Info = CheckedAttributionInfo;

    fn info(&self) -> CheckedAttributionInfo {
        let edges = self.graph().directed_edge_pairs_sorted();
        CheckedAttributionInfo {
            query: self.query(),
            identifier: IdentifierId::GcmParametric,
            estimator: self.estimator,
            graph_edges: edges.into(),
            identification_status: self.identification().status,
        }
    }
}

/// Resolves the static DAG, unfolding a `TemporalDag` over the supplied series when the source graph is temporal.
fn attribution_graph(
    study: &Study,
) -> Result<(Dag, Option<super::temporal_attribution::LagAlignedScm>), CausalError> {
    Ok(if let Some(dag) = study.graph.as_dag() {
        (dag.clone(), None)
    } else if let Some(temporal) = study.graph.as_temporal_dag() {
        let (DataInput::Temporal(series) | DataInput::Event(series)) = &study.data else {
            return Err(CausalError::Unsupported {
                message: "temporal attribution requires a time series",
            });
        };
        let aligned = super::temporal_attribution::LagAlignedScm::unfold(temporal, series)?;
        let dag = aligned.dag.clone();
        (dag, Some(aligned))
    } else {
        return Err(CausalError::Unsupported {
            message: "checked attribution requires a supplied DAG or TemporalDag",
        });
    })
}

/// Selects the estimator and inference procedure for the target, refusing prior-bearing Bayesian configurations.
fn attribution_procedure(
    target: &AttributionTarget,
    study: &Study,
) -> Result<(EstimatorId, AttributionProcedure), CausalError> {
    Ok(match (target, &study.inference) {
        // A fitted-model target reports the fitted model's answer under either mode.
        (AttributionTarget::Model(_), _) => (EstimatorId::GcmFit, AttributionProcedure::Point),
        (_, InferenceMode::Frequentist) => {
            // A requested replicate count is the mechanism-refit bootstrap: multinomial
            // rows of independent subjects on a static DAG, circular blocks of consecutive
            // times on a lag-aligned TemporalDag unfolding.
            if study.bootstrap_replicates > 0 && study.graph.as_dag().is_some() {
                super::gcm_bootstrap::require_subject_resampling(study.sampling_design)?;
                (
                    EstimatorId::GcmFit,
                    AttributionProcedure::RowBootstrap { replicates: study.bootstrap_replicates },
                )
            } else if study.bootstrap_replicates > 0
                && matches!(study.graph.class(), crate::accepted::GraphClass::TemporalDag)
            {
                super::gcm_bootstrap::require_series_resampling(study.sampling_design)?;
                (
                    EstimatorId::GcmFit,
                    AttributionProcedure::RowBootstrap { replicates: study.bootstrap_replicates },
                )
            } else {
                (EstimatorId::GcmFit, AttributionProcedure::Point)
            }
        }
        (AttributionTarget::Anomaly(_), InferenceMode::Bayesian(config)) => {
            if config.prior.is_some()
                || config.prior_artifact.is_some()
                || config.external_compose.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "Bayesian attribution uses its shared-row Bayesian bootstrap and does not accept coefficient priors or external prior composition",
                });
            }
            (
                EstimatorId::GcmFitBayesian,
                AttributionProcedure::SharedDirichletRowWeights {
                    draws: bayesian_draw_count(&study.inference)?,
                },
            )
        }
        (AttributionTarget::Change(_), InferenceMode::Bayesian(config)) => {
            if config.prior.is_some()
                || config.prior_artifact.is_some()
                || config.external_compose.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "Bayesian attribution uses its shared-row Bayesian bootstrap and does not accept coefficient priors or external prior composition",
                });
            }
            (
                EstimatorId::GcmAttributionBayesian,
                AttributionProcedure::SharedDirichletRowWeights {
                    draws: bayesian_draw_count(&study.inference)?,
                },
            )
        }
    })
}

#[cfg(test)]
mod bayesian_checked_attribution_tests {
    use super::*;
    use crate::analysis::builder::RefuteSuite;
    use antecedent_core::{AnomalyAttributionQuery, CausalQuery, PopulationSelector, VariableId};
    use antecedent_data::TabularData;
    use antecedent_graph::DenseNodeId;

    fn context() -> ExecutionContext {
        ExecutionContext::for_tests(0xBA71_5EED)
    }

    fn anomaly_fixture() -> (TabularData, Dag) {
        let x: Vec<f64> = (0..20).map(f64::from).collect();
        let y: Vec<f64> =
            (0..20).map(|i| if i == 19 { 200.0 } else { 2.0 * f64::from(i) }).collect();
        let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])
            .expect("fixture data");
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (data, graph)
    }

    fn change_fixture() -> (TabularData, Dag) {
        let x: Vec<f64> = (0..80).map(|i| f64::from(i % 40) * 0.1).collect();
        let y: Vec<f64> = (0..80)
            .map(|i| {
                let x = f64::from(i % 40) * 0.1;
                if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x }
            })
            .collect();
        let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])
            .expect("fixture data");
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (data, graph)
    }

    fn checked_operation(
        data: &TabularData,
        graph: Dag,
        query: &CausalQuery,
        estimator: EstimatorId,
    ) -> CheckedAttributionOperation {
        let ctx = context();
        let builder = crate::Study::tabular(data.clone())
            .graph(graph)
            .query(query.clone())
            .inference(InferenceMode::Bayesian(crate::BayesianConfig::conjugate().n_draws(24)))
            .refute(RefuteSuite::None);
        let study = builder.build().expect("study build");
        let physical = study.plan(&ctx).expect("physical plan");
        let outcome = match query {
            CausalQuery::AnomalyAttribution(query) => *query.targets.first().unwrap(),
            CausalQuery::ChangeAttribution(query) => query.outcome,
            _ => unreachable!(),
        };
        let (identification, estimand) =
            parametric_scm_identification(query.clone(), outcome, outcome);
        let cache = Arc::new(crate::analysis::prepared::CachedStaticIdentification {
            identification,
            estimand,
        });
        let operation =
            CheckedAttributionOperation::checked(&study, &Arc::new(physical.clone()), &cache)
                .expect("Bayesian attribution plan");
        let info = operation.info();
        assert_eq!((info.identifier, info.estimator), (IdentifierId::GcmParametric, estimator));
        drop(study);
        operation
    }

    #[test]
    fn anomaly_posterior_executes_from_retained_operation_after_builder_drop() {
        let (data, graph) = anomaly_fixture();
        let query = CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
            [VariableId::from_raw(1)],
            100,
        ));
        let operation = checked_operation(&data, graph, &query, EstimatorId::GcmFitBayesian);
        let result = operation.execute(&data, &context()).expect("retained execution");
        let posterior = result.posterior.as_ref().expect("shared-row posterior");
        assert_eq!(posterior.draws.n_draws, 24);
        assert_eq!(posterior.draws.schema.n_quantities(), 1);
        assert!(posterior.summaries.mean[0].is_finite() && posterior.summaries.mean[0] > 0.0);
        let scores = result.anomaly().expect("anomaly scores");
        let target = scores.iter().find(|scores| scores.target == VariableId::from_raw(1)).unwrap();
        let top_row = target.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(target.rows[top_row], 19, "known outlier remains the highest-scoring row");
        assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.attribution.bayesian"));
    }

    #[test]
    fn change_posterior_keeps_total_and_shapley_components_coupled() {
        let (data, graph) = change_fixture();
        let query = CausalQuery::ChangeAttribution(antecedent_core::ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        ));
        let operation =
            checked_operation(&data, graph, &query, EstimatorId::GcmAttributionBayesian);
        let result = operation.execute(&data, &context()).expect("retained execution");
        let posterior = result.posterior.as_ref().expect("shared-population posterior");
        assert_eq!(posterior.draws.n_draws, 24);
        assert!(posterior.summaries.mean[0].is_finite());
        let truth: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../conformance/estimate/staged_attribution/expected.json"
        ))
        .unwrap();
        let expected = truth["change"]["total_change"].as_f64().unwrap();
        let tolerance = truth["change"]["tolerance"].as_f64().unwrap();
        assert!((posterior.summaries.mean[0] - expected).abs() < tolerance);
        for draw in 0..posterior.draws.n_draws {
            let total = posterior.draws.get(draw, 0).unwrap();
            let sum = (1..posterior.draws.schema.n_quantities())
                .map(|quantity| posterior.draws.get(draw, quantity).unwrap())
                .sum::<f64>();
            assert!((sum - total).abs() < 1e-8, "Shapley components must sum to total per draw");
        }
        assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.attribution.bayesian"));
    }
}
