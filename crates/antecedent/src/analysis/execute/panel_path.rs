// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    pub(super) fn execute_panel(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let id_res = TemporalBackdoorIdentifier::new()
            .identify_temporal(graph, query)
            .map_err(CausalError::from)?;
        let identification = id_res.result;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::TemporalLinearAdjustment)?;

        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let (prep, cluster_ids, panel_times) = estimator
            .prepare_panel(
                panel,
                &estimand,
                query,
                &id_res.indexer,
                self.split.as_ref(),
                &ctx.kernel_policy,
            )
            .map_err(CausalError::from)?;
        let max_lag = query.max_history_lag.unwrap_or(1).max(1) as usize;

        let (estimate, mut posterior, estimate_artifact, estimate_op) = match &self.inference {
            InferenceMode::Bayesian(cfg) => {
                let mut bayes = BayesianTemporalGcomp {
                    inner: BayesianGComputationAte {
                        backend: cfg.backend,
                        likelihood: cfg.likelihood,
                        n_draws: cfg.n_draws,
                        seed: ctx.rng.master_seed(),
                        overlap: OverlapPolicy::ExplicitOverride,
                        prior_scale: cfg.prior_scale,
                        prior: None,
                    },
                };
                let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
                let (resolved_prior, conflict_summary) =
                    resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
                bayes.inner.prior = resolved_prior;
                let mut ws = BayesianGCompWorkspace::default();
                let mut posterior = bayes
                    .fit(&bprep, identification.status, &mut ws, ctx)
                    .map_err(CausalError::from)?;
                if let Some(summary) = conflict_summary {
                    posterior = with_conflict_summary(posterior, summary);
                }
                let estimate = effect_from_posterior(&posterior)?;
                (
                    estimate,
                    Some(posterior),
                    "estimate.bayesian_temporal_gcomp.panel",
                    "estimate.bayesian.temporal.gcomp.panel",
                )
            }
            InferenceMode::Frequentist => {
                estimator.inner.cluster_ids = Some(cluster_ids);
                estimator.inner.panel_times = Some(panel_times);
                estimator.inner.se_kind = AnalyticSeKind::PanelClusterHac { lag: max_lag };
                let mut workspace = EstimationWorkspace::default();
                let estimate = estimator
                    .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
                    .map_err(CausalError::from)?;
                (
                    estimate,
                    None,
                    "estimate.temporal_linear_adjustment.panel",
                    "estimate.temporal.linear.adjustment.panel",
                )
            }
        };

        let provenance = provenance_pair(
            (
                "identify.temporal_backdoor",
                "identify.temporal.backdoor.unfolded",
                &[],
                &identification.required_assumptions,
            ),
            (
                estimate_artifact,
                estimate_op,
                &["identify.temporal_backdoor"],
                &estimate.assumptions,
            ),
        );

        let mut diagnostics = Vec::new();
        let stacked = stack_panel_tabular(panel).map_err(CausalError::from)?;
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut refute_ws = EstimationWorkspace::default();
        let temporal_ctx = TemporalRefitContext {
            indexer: &id_res.indexer,
            temporal_query: query,
            split: self.split.as_ref(),
            kernel_policy: &ctx.kernel_policy,
            time_index: None,
            panel: Some(panel),
        };
        let mut refutations = run_refuters(
            &stacked,
            &estimand,
            &ate_q,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            if posterior.is_some() {
                "bayesian.temporal.gcomp"
            } else {
                "temporal.linear.adjustment"
            },
            &self.custom_validators,
            Some(temporal_ctx),
        )?;

        // Panel Bayesian: α-grid under Full when external compose is present (mirror temporal).
        if matches!(self.refute, RefuteSuite::Full) {
            if let (InferenceMode::Bayesian(cfg), Some(post)) = (&self.inference, &posterior) {
                let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
                let mut est = BayesianTemporalGcomp {
                    inner: BayesianGComputationAte {
                        backend: cfg.backend,
                        likelihood: cfg.likelihood,
                        n_draws: cfg.n_draws,
                        seed: ctx.rng.master_seed(),
                        overlap: OverlapPolicy::ExplicitOverride,
                        prior_scale: cfg.prior_scale,
                        prior: None,
                    },
                };
                let mut ws = BayesianGCompWorkspace::default();
                if let Some(ext) = cfg.external_compose.as_ref() {
                    est.inner.prior = Some(ext.composed.prior.clone());
                } else {
                    est.inner.prior = resolve_bayesian_prior(cfg, &bprep)?;
                }
                let (summary, sens) = evaluate_bayesian_prior_sensitivity(
                    cfg,
                    &est.inner,
                    &bprep,
                    identification.status,
                    post,
                    &mut ws,
                    ctx,
                )?;
                refutations.push(sens.to_report(&summary, estimate.ate));
                posterior = Some(with_prior_sensitivity(post.clone(), summary));
            }
        }

        if let Some(cs) = posterior.as_ref().and_then(|p| p.conflict_summary.as_ref()) {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }
}
