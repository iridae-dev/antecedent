// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::CausalAnalysis {
    pub(super) fn execute_temporal(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
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
        let prep = estimator
            .prepare(
                data,
                &estimand,
                query,
                &id_res.indexer,
                self.split.as_ref(),
                &ctx.kernel_policy,
            )
            .map_err(CausalError::from)?;

        let (estimate, posterior, estimate_artifact, estimate_op) = match &self.inference {
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
                    "estimate.bayesian_temporal_gcomp",
                    "estimate.bayesian.temporal.gcomp",
                )
            }
            InferenceMode::Frequentist => {
                let mut workspace = EstimationWorkspace::default();
                let estimate = estimator
                    .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
                    .map_err(CausalError::from)?;
                (
                    estimate,
                    None,
                    "estimate.temporal_linear_adjustment",
                    "estimate.temporal.linear.adjustment",
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
        if physical
            .logical
            .record
            .discovery_algorithm
            .as_deref()
            .is_some_and(|a| a.contains("pag_completed_to_dag") || a.contains("completed_to_dag"))
        {
            diagnostics.push(Diagnostic::new(
                "temporal.pag.completed_to_dag",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "TemporalPag completed to TemporalDag before temporal.backdoor \
                 (completion path; not class-aware temporal PAG identification)",
            ));
        }
        let tabular = TabularData::new(data.storage().clone());
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut refute_ws = EstimationWorkspace::default();
        let temporal_ctx = TemporalRefitContext {
            indexer: &id_res.indexer,
            temporal_query: query,
            split: self.split.as_ref(),
            kernel_policy: &ctx.kernel_policy,
            time_index: Some(data.time_index()),
            panel: None,
        };
        let mut refutations = run_refuters(
            &tabular,
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

        // Bayesian temporal: prior/posterior PPC + prior sensitivity on Full (mirror static).
        let mut posterior = posterior;
        if matches!(&self.inference, InferenceMode::Bayesian(_))
            && !matches!(self.refute, RefuteSuite::None)
        {
            if let Some(ref post) = posterior {
                const PPC_ALPHA: f64 = 0.05;
                let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
                let prior_rep = PriorPredictiveCheck {
                    n_sims: 200,
                    seed: ctx.rng.master_seed(),
                    ..PriorPredictiveCheck::new()
                }
                .check(&bprep, ctx)
                .map_err(CausalError::from)?;
                refutations.push(prior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));

                let post_rep = PosteriorPredictiveCheck::new()
                    .check(&bprep, post)
                    .map_err(CausalError::from)?;
                refutations.push(post_rep.to_refutation_report(estimate.ate, PPC_ALPHA));

                if matches!(self.refute, RefuteSuite::Full) {
                    let cfg = match &self.inference {
                        InferenceMode::Bayesian(c) => c.clone(),
                        InferenceMode::Frequentist => unreachable!(),
                    };
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
                        est.inner.prior = resolve_bayesian_prior(&cfg, &bprep)?;
                    }
                    let (summary, sens) = evaluate_bayesian_prior_sensitivity(
                        &cfg,
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

    /// Panel temporal effect: identify on the shared graph, estimate on stacked units
    /// with [`AnalyticSeKind::PanelClusterHac`] and per-unit `cluster_ids`.
    ///
    /// Bayesian mode fits [`BayesianTemporalGcomp`] on the stacked lag-aligned design
    /// (no hierarchical unit random effects; cluster-HAC is frequentist-only).
    pub(super) fn execute_temporal_mediation(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let identification = TemporalMediationIdentifier {
            allow_natural_controlled_alias: true,
            ..TemporalMediationIdentifier::new()
        }
        .identify(graph, query)
        .map_err(CausalError::from)?;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::TemporalMediation)?;
        let mut est = TemporalMediationEstimator::new();
        est.allow_natural_controlled_alias = true;
        let mediation = est.estimate(data, &estimand, query, ctx).map_err(CausalError::from)?;
        let estimate = mediation.effect.clone();
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        let provenance = provenance_pair(
            (
                "identify.temporal_mediation",
                "identify.temporal_mediation",
                &[],
                &identification.required_assumptions,
            ),
            (
                "estimate.temporal_mediation",
                "estimate.temporal_mediation",
                &["identify.temporal_mediation"],
                &estimate.assumptions,
            ),
        );
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: Some(mediation),
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
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
