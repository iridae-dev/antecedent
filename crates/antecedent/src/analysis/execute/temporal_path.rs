// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    pub(crate) fn mediation_adjustment(
        &self,
        graph: &TemporalDag,
        query: &antecedent_core::MediationQuery,
        horizon: u32,
    ) -> Result<Arc<[antecedent_data::LaggedColumn]>, CausalError> {
        if let Some(entry) =
            self.temporal_identification_cache.as_deref().and_then(|cache| cache.get(horizon))
        {
            return Ok(lagged_adjustment_from_entry(entry));
        }
        let ider = TemporalMediationIdentifier {
            allow_natural_controlled_alias: true,
            ..TemporalMediationIdentifier::new()
        };
        let (_, temporal) =
            ider.identify_with_horizon(graph, query, horizon).map_err(CausalError::from)?;
        Ok(lagged_adjustment_from_temporal(&temporal))
    }

    pub(super) fn execute_temporal(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identification, estimand, indexer, identify_cached) = if let Some(cache) =
            self.temporal_identification_cache.as_deref()
        {
            let entry = cache.get(query.horizon_steps).ok_or_else(|| CausalError::Compile {
                message: format!(
                    "prepared temporal identification missing horizon {}",
                    query.horizon_steps
                ),
            })?;
            (entry.identification.clone(), entry.estimand.clone(), entry.indexer.clone(), true)
        } else {
            let id_res = TemporalBackdoorIdentifier::new()
                .identify_temporal(graph, query)
                .map_err(CausalError::from)?;
            let estimand = select_estimand(
                &id_res.result,
                if matches!(query.policy, antecedent_core::TemporalPolicy::Sustained { from, until } if from != until)
                {
                    EstimatorId::TemporalSequentialGcomp
                } else {
                    EstimatorId::TemporalLinearAdjustment
                },
            )?;
            (id_res.result, estimand, id_res.indexer, false)
        };
        require_identified(&identification)?;
        if matches!(query.policy, antecedent_core::TemporalPolicy::Sustained { from, until } if from != until)
        {
            if self.refute != RefuteSuite::None || self.split.is_some() {
                return Err(CausalError::Unsupported {
                    message: "multi-step sustained g-computation currently requires validation=none and no discovery-estimation split",
                });
            }
            let bayes = if let InferenceMode::Bayesian(cfg) = &self.inference {
                if cfg.prior.is_some()
                    || cfg.prior_artifact.is_some()
                    || cfg.external_compose.is_some()
                {
                    return Err(CausalError::Unsupported {
                        message: "multi-step sustained inference requires isotropic per-mechanism priors",
                    });
                }
                Some(bayesian_gcomp(cfg, ctx))
            } else {
                None
            };
            let mut assumptions = identification.required_assumptions.clone();
            assumptions.push(antecedent_core::AssumptionRecord {
                assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
                    id: Arc::from("temporal.sequential.linear_sem"), description: Arc::from("linear additive mechanisms on the identified unfolded DAG; every sustained time is intervened on; frequentist intervals use a shared moving-block row bootstrap, Bayesian intervals share each stationary mechanism posterior across time copies, fitting its unique complete observed rows once"),
                }),
                source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("temporal.sequential.gcomp") },
                scope: antecedent_core::AssumptionScope::Estimation, status: antecedent_core::AssumptionStatus::Declared,
            });
            let (estimate, posterior) =
                antecedent_estimate::temporal_sequential::estimate_sustained_window(
                    data,
                    graph,
                    &indexer,
                    &estimand,
                    query,
                    identification.status,
                    assumptions,
                    self.bootstrap_replicates,
                    bayes.as_ref(),
                    ctx,
                )
                .map_err(CausalError::from)?;
            let bootstrap_replicates_ok = estimate.bootstrap_replicates_ok;
            let cancelled = estimate.bootstrap_cancelled;
            return Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
                physical, identification, estimand, estimate,
                identifier_id: IdentifierId::TemporalBackdoorUnfolded, estimator_id: EstimatorId::TemporalSequentialGcomp,
                treatment: query.treatment, outcome: query.outcome, identify_cached,
                extra_diagnostics: vec![Diagnostic::new("estimate.temporal.sustained_window", DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                    "the contrast propagates through all intervened times in topological order; no one-node collapse; no analytic SE is asserted for the frequentist sequential fit")],
                refutations: Vec::new(), distribution: None, mediation: None,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok, cancelled, early_stopped: false,
                extras: IdentifiedExecuteExtras { posterior, ..Default::default() },
            }));
        }

        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let prep = estimator
            .prepare(data, &estimand, query, &indexer, self.split.as_ref(), &ctx.kernel_policy)
            .map_err(CausalError::from)?;

        let (estimate, posterior, estimate_artifact, estimate_op) = match &self.inference {
            InferenceMode::Bayesian(cfg) => {
                let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
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

        let mut diagnostics = Vec::new();
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
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
            indexer: &indexer,
            temporal_query: query,
            split: self.split.as_ref(),
            kernel_policy: &ctx.kernel_policy,
            time_index: Some(data.time_index()),
            panel: None,
        };
        let (mut refutations, na_diagnostics) = run_refuters(
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
        diagnostics.extend(na_diagnostics);

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
                    let InferenceMode::Bayesian(cfg) = &self.inference else { unreachable!() };
                    posterior = Some(apply_temporal_prior_sensitivity(
                        cfg,
                        &bprep,
                        identification.status,
                        post,
                        estimate.ate,
                        ctx,
                        &mut refutations,
                    )?);
                }
            }
        }

        if let Some(cs) = posterior.as_ref().and_then(|p| p.conflict_summary.as_ref()) {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let certificate = crate::Identification::Point {
            result: identification.clone(),
            temporal_indexer: Some(indexer.clone()),
            strategy: IdentifierId::TemporalBackdoorUnfolded,
            structure_version: self.graph.version(),
        };
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TemporalBackdoorUnfolded,
            estimator_id: EstimatorId::TemporalLinearAdjustment,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: Some(certificate),
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(estimate_artifact, estimate_op)),
                posterior,
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
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
    ) -> Result<StudyResult, CausalError> {
        if let InferenceMode::Bayesian(cfg) = &self.inference {
            return self.execute_bayesian_mediation(data, graph, query, physical, cfg, ctx);
        }
        let started = Instant::now();
        let (cache, identify_cached) =
            if let Some(cache) = self.temporal_identification_cache.clone() {
                (cache, true)
            } else {
                report_identify_compute(ctx);
                (
                    Arc::new(crate::analysis::prepared::identify_temporal_mediation_horizons(
                        graph,
                        query,
                        EstimatorId::TemporalMediation,
                    )?),
                    false,
                )
            };
        let clicks = mediation_horizon_clicks(&cache, query, |horizon| {
            self.mediation_adjustment(graph, query, horizon)
        })?;
        let mut extra_diagnostics = Vec::new();
        if mediation_horizon_z_differs(&clicks) {
            extra_diagnostics.push(Diagnostic::new(
                "identify.temporal_mediation.horizon_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "adjustment sets differ across requested horizons; each contrast uses I(h) \
                 identified for that horizon, not a shared max-horizon set",
            ));
        }
        let est = TemporalMediationEstimator::new().with_allow_natural_controlled_alias(true);
        let mut published = None;
        for click in &clicks {
            require_identified(&click.identification)?;
            let mut qh = query.clone();
            qh.horizons = Arc::from([click.horizon]);
            let mediation = est
                .estimate_with_adjustment(data, &click.estimand, &qh, &click.adjustment, &[], ctx)
                .map_err(CausalError::from)?;
            if published.is_none() {
                published = Some((click, qh, mediation));
            }
        }
        let (click, qh, mediation) = published.ok_or_else(|| CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        })?;
        let estimate = mediation.effect.clone();
        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::mediation::refute_temporal_mediation_adjusted(
                data,
                &click.estimand,
                &qh,
                &mediation,
                self.refute == RefuteSuite::Full,
                &click.adjustment,
                ctx,
            )
            .map_err(CausalError::from)?
        };
        let certificate = crate::Identification::Point {
            result: click.identification.clone(),
            temporal_indexer: Some(click.indexer.clone()),
            strategy: IdentifierId::Frontdoor,
            structure_version: self.graph.version(),
        };
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification: click.identification.clone(),
            estimand: click.estimand.clone(),
            estimate,
            identifier_id: IdentifierId::Frontdoor,
            estimator_id: EstimatorId::TemporalMediation,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation: Some(mediation),
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: Some(certificate),
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_mediation",
                    "identify.temporal_mediation",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal_mediation",
                    "estimate.temporal_mediation",
                )),
                ..Default::default()
            },
        }))
    }

    fn execute_bayesian_mediation(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        cfg: &BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use antecedent_estimate::bayesian_mediation::{
            compose_temporal_mediation, prepare_temporal_mediation_adjusted,
            require_gaussian_mediation,
        };
        let started = Instant::now();
        if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
            return Err(CausalError::Unsupported {
                message: "Bayesian mediation currently supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
            });
        }
        let (cache, identify_cached) =
            if let Some(cache) = self.temporal_identification_cache.clone() {
                (cache, true)
            } else {
                report_identify_compute(ctx);
                (
                    Arc::new(crate::analysis::prepared::identify_temporal_mediation_horizons(
                        graph,
                        query,
                        EstimatorId::BayesianTemporalMediation,
                    )?),
                    false,
                )
            };
        let clicks = mediation_horizon_clicks(&cache, query, |horizon| {
            self.mediation_adjustment(graph, query, horizon)
        })?;
        if clicks.is_empty() {
            return Err(CausalError::Compile {
                message: "temporal mediation requires at least one horizon".into(),
            });
        }
        let mut extra_diagnostics = Vec::new();
        if mediation_horizon_z_differs(&clicks) {
            extra_diagnostics.push(Diagnostic::new(
                "identify.temporal_mediation.horizon_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "adjustment sets differ across requested horizons; each contrast uses I(h) \
                 identified for that horizon, not a shared max-horizon set",
            ));
        }
        // Click every I(h) so a multi-horizon Bayesian run does not estimate
        // h=1 under I(2)'s Z. The scalar API publishes the first horizon.
        let mut published = None;
        for click in &clicks {
            require_identified(&click.identification)?;
            let mut qh = query.clone();
            qh.horizons = Arc::from([click.horizon]);
            let preparations = prepare_temporal_mediation_adjusted(
                data,
                &click.estimand,
                &qh,
                &click.adjustment,
                ctx,
            )
            .map_err(CausalError::from)?;
            if published.is_none() {
                published = Some((qh, preparations));
            }
        }
        let click = &clicks[0];
        let identification = click.identification.clone();
        let estimand = click.estimand.clone();
        let adjustment = Arc::clone(&click.adjustment);
        let (qh, preparations) = published.ok_or_else(|| CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        })?;
        let estimator = bayesian_gcomp(cfg, ctx);
        require_gaussian_mediation(&estimator).map_err(CausalError::from)?;
        let fit = |scale: f64| -> Result<Vec<CausalPosterior>, CausalError> {
            preparations
                .iter()
                .enumerate()
                .map(|(i, prep)| {
                    let mut est = estimator.clone();
                    est.prior_scale = scale;
                    est.seed = est.seed.wrapping_add(if i == 0 { 0 } else { 0xBA71_u64 });
                    est.fit(
                        prep,
                        identification.status,
                        &mut BayesianGCompWorkspace::default(),
                        ctx,
                    )
                    .map_err(CausalError::from)
                })
                .collect()
        };
        let mechanisms = fit(cfg.prior_scale)?;
        let mut posterior =
            compose_temporal_mediation(&mechanisms[0], &mechanisms[1], &qh, identification.status)
                .map_err(CausalError::from)?;
        let estimate = effect_from_posterior(&posterior)?;
        let mediation = TemporalMediationEstimate {
            effect: estimate.clone(),
            total: Some(posterior.summaries.mean[1]),
            direct: Some(posterior.summaries.mean[2]),
            mediated: Some(posterior.summaries.mean[3]),
        };
        let mut refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::mediation::refute_temporal_mediation_adjusted(
                data,
                &estimand,
                &qh,
                &mediation,
                self.refute == RefuteSuite::Full,
                &adjustment,
                ctx,
            )
            .map_err(CausalError::from)?
        };
        let mut predictive_checks = Vec::new();
        if self.refute != RefuteSuite::None {
            for (i, (prep, post)) in preparations.iter().zip(&mechanisms).enumerate() {
                let prior = PriorSet::weakly_informative(prep.design.ncols);
                let prior = estimator.prior.clone().unwrap_or_else(|| {
                    let mut prior = prior;
                    prior.specs = vec![antecedent_prob::PriorSpec::GaussianCoefficients(
                        antecedent_prob::GaussianCoefficientPrior::isotropic(
                            prep.design.ncols,
                            cfg.prior_scale,
                        ),
                    )];
                    prior
                });
                let prior_rep = PriorPredictiveCheck::new()
                    .check_with_prior(prep, &prior, ctx)
                    .map_err(CausalError::from)?;
                let post_rep =
                    PosteriorPredictiveCheck::new().check(prep, post).map_err(CausalError::from)?;
                for report in [prior_rep, post_rep] {
                    let mut r = report.to_refutation_report(estimate.ate, 0.05);
                    r.refuter = Arc::from(format!("mediation.mechanism_{i}.{}", r.refuter));
                    refutations.push(r);
                    predictive_checks.push(report);
                }
            }
        }
        if self.refute == RefuteSuite::Full {
            let sensitivity = antecedent_validate::PriorSensitivity::standard_grid();
            let mut means = Vec::new();
            let mut sds = Vec::new();
            for &scale in sensitivity.scales.iter() {
                let posts = fit(scale)?;
                let post =
                    compose_temporal_mediation(&posts[0], &posts[1], &qh, identification.status)
                        .map_err(CausalError::from)?;
                means.push(post.summaries.mean[0]);
                sds.push(post.summaries.sd[0]);
            }
            let summary = antecedent_prob::PriorSensitivitySummary {
                prior_scales: sensitivity.scales.clone(),
                alphas: Arc::from([]),
                effect_means: Arc::from(means),
                effect_sds: Arc::from(sds),
            };
            refutations.push(sensitivity.to_report(&summary, estimate.ate));
            posterior = with_prior_sensitivity(posterior, summary);
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical, identification: identification.clone(), estimand, estimate,
            identifier_id: IdentifierId::Frontdoor, estimator_id: EstimatorId::BayesianTemporalMediation,
            treatment: query.treatment, outcome: query.outcome, identify_cached,
            extra_diagnostics: {
                extra_diagnostics.push(Diagnostic::new("estimate.mediation.bayesian", DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                    "independent Gaussian mediator and outcome mechanisms; total = direct + mediated for every posterior draw; natural effects use the linear no-interaction alias"));
                extra_diagnostics
            },
            refutations, distribution: None, mediation: Some(mediation),
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None, cancelled: false, early_stopped: false,
            extras: IdentifiedExecuteExtras { posterior: Some(posterior), predictive_checks,
                certificate: Some(crate::Identification::Point {
                    result: identification.clone(),
                    temporal_indexer: Some(click.indexer.clone()),
                    strategy: IdentifierId::Frontdoor,
                    structure_version: self.graph.version(),
                }),
                identify_provenance: Some(provenance_ids("identify.temporal_mediation", "identify.temporal_mediation")),
                ..Default::default() },
        }))
    }

    pub(super) fn execute_temporal_response(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let Some(temporal) = query.temporal.as_ref() else {
            return Err(CausalError::Compile {
                message: "temporal response route requires TemporalResponseSpec".into(),
            });
        };
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let schedule = match antecedent_estimate::plan_from_response_query(query) {
            Ok(Some(antecedent_estimate::TemporalInterventionPlan::Sequential { overlays })) => {
                Some(overlays.iter().map(|o| (o.variable, o.offset)).collect::<Vec<_>>())
            }
            Ok(_) => None,
            Err(error) => return Err(CausalError::from(error)),
        };
        let (cache, identify_cached) =
            if let Some(cache) = self.temporal_identification_cache.clone() {
                (cache, true)
            } else {
                (
                    Arc::new(crate::analysis::prepared::identify_temporal_response_horizons(
                        graph,
                        treatment,
                        outcome,
                        temporal,
                        &query.target_population,
                        if matches!(self.inference, InferenceMode::Bayesian(_)) {
                            EstimatorId::TemporalResponseBayesian
                        } else {
                            EstimatorId::TemporalResponseGcomp
                        },
                        schedule.as_deref(),
                    )?),
                    false,
                )
            };
        let mut aligned = Vec::with_capacity(temporal.horizons.len());
        for &horizon in temporal.horizons.iter() {
            let entry = cache.get(horizon).ok_or_else(|| CausalError::Compile {
                message: format!("temporal identification missing horizon {horizon}"),
            })?;
            require_identified(&entry.identification)?;
            aligned.push(entry);
        }
        let first = aligned[0];
        let (aggregate_status, mut aggregate_assumptions) =
            aggregate_temporal_horizon_evidence(aligned.iter().map(|entry| &entry.identification))?;
        let mut identification = first.identification.clone();
        identification.status = aggregate_status;
        identification.required_assumptions = aggregate_assumptions.clone();
        let estimand = first.estimand.clone();
        let identifications: Vec<_> =
            aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();

        query
            .require_licensed_temporal_observation()
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let mut working_query = query.clone();
        let mut observation_adjusted = None;
        let series_owned = if query.observation == ObservationSpec::Complete {
            None
        } else {
            if matches!(self.inference, InferenceMode::Bayesian(_)) {
                return Err(CausalError::Unsupported {
                    message: "Bayesian temporal response requires complete observations",
                });
            }
            if self.observation_delayed_entry.is_some() {
                return Err(CausalError::Unsupported {
                    message: "delayed entry is not licensed for temporal observation",
                });
            }
            let adjustment = contemporaneous_adjustment_variables(&aligned);
            let (series, adjusted) = ObservationMechanismEstimator::new(self.observation_options)
                .adjust_temporal_series(data, query, &adjustment)
                .map_err(CausalError::from)?;
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
            append_temporal_observation_assumptions(query, &mut aggregate_assumptions);
            working_query.observation = ObservationSpec::Complete;
            working_query.observation_assumptions = Arc::from([]);
            observation_adjusted = Some(adjusted);
            Some(series)
        };
        let data = series_owned.as_ref().unwrap_or(data);

        if let Some(antecedent_estimate::TemporalInterventionPlan::Sequential { overlays }) =
            antecedent_estimate::plan_from_response_query(query).map_err(CausalError::from)?
        {
            return self.execute_temporal_sequence_response(
                data,
                graph,
                query,
                temporal,
                &overlays,
                &aligned,
                identification,
                estimand,
                treatment,
                outcome,
                aggregate_status,
                aggregate_assumptions,
                identify_cached,
                physical,
                started,
                ctx,
                observation_adjusted.as_ref(),
            );
        }

        // Surface SEs follow the Study bootstrap / replicate contract so Pulse,
        // single-step Sustained, and the dose × horizon surface report comparable
        // uncertainty. Replicates = 0 keeps the analytic OLS linear-functional SE.
        let mut estimator = TemporalResponseEstimator::new();
        estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
        let mut response = if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
                return Err(CausalError::Unsupported { message: "Bayesian temporal response prior transfer requires a horizon-specific mapping" });
            }
            let mut bayes = bayesian_gcomp(cfg, ctx);
            bayes.prior.clone_from(&cfg.prior);
            estimator.estimate_bayesian(data, &identifications, &working_query, aggregate_status, aggregate_assumptions, &bayes, ctx)
        } else { estimator.estimate(data, &identifications, &working_query, aggregate_status, aggregate_assumptions, ctx) }.map_err(CausalError::from)?;
        if let Some(adjusted) = observation_adjusted.as_ref() {
            apply_temporal_observation_result(&mut response, adjusted);
        }
        // The estimator's public API accepts one aggregate status for callers
        // that have a homogeneous set of horizon witnesses. This execution path
        // has the richer per-horizon records, so retain their actual statuses
        // instead of repeating the first/aggregate label across the surface.
        if let Some(per_horizon) = response.horizon_identification.as_mut() {
            for (record, entry) in Arc::make_mut(per_horizon).iter_mut().zip(&aligned) {
                record.status = entry.identification.status;
            }
        }

        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = Vec::new();
        for entry in &aligned {
            diagnostics.extend(entry.identification.diagnostics.iter().cloned());
        }
        if horizon_adjustment_sets_differ(&aligned) {
            diagnostics.push(Diagnostic::new(
                "identify.temporal_response.horizon_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "adjustment sets differ across requested horizons; each cell uses I(h) \
                 identified for that horizon, not a shared max-horizon set",
            ));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "refute.temporal_response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters are not applicable to a function-valued temporal response",
        ));
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued; the scalar effect summary is not applicable \
                 and the result is carried by the response payload",
            ));
        }
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TemporalBackdoorUnfolded,
            estimator_id: if matches!(self.inference, InferenceMode::Bayesian(_)) {
                EstimatorId::TemporalResponseBayesian
            } else {
                EstimatorId::TemporalResponseGcomp
            },
            treatment,
            outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    fn execute_temporal_sequence_response(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &ResponseQuery,
        temporal: &antecedent_core::TemporalResponseSpec,
        overlays: &[antecedent_estimate::SequentialNodeOverlay],
        aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        treatment: VariableId,
        outcome: VariableId,
        aggregate_status: IdentificationStatus,
        mut assumptions: antecedent_core::AssumptionSet,
        identify_cached: bool,
        physical: &PhysicalExecutionPlan,
        started: Instant,
        ctx: &ExecutionContext,
        observation_adjusted: Option<&antecedent_estimate::ObservationAdjustedOutcome>,
    ) -> Result<StudyResult, CausalError> {
        if self.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "multi-step Sequence overlays require no discovery-estimation split",
            });
        }
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "multi-step Sequence inference requires isotropic per-mechanism priors",
                });
            }
            Some(bayesian_gcomp(cfg, ctx))
        } else {
            None
        };
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::ParametricRestriction(
                antecedent_core::ParametricAssumption {
                    id: Arc::from("temporal.sequential.linear_sem"),
                    description: Arc::from(
                        "linear additive mechanisms on the identified unfolded DAG; each Sequence \
                         step is a Set / Soft constant / Soft shift overlay; no last-step collapse",
                    ),
                },
            ),
            source: antecedent_core::AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("temporal.sequential.gcomp"),
            },
            scope: antecedent_core::AssumptionScope::Estimation,
            status: antecedent_core::AssumptionStatus::Declared,
        });
        let mut mean = Vec::with_capacity(temporal.horizons.len());
        let mut lower = Vec::with_capacity(temporal.horizons.len());
        let mut upper = Vec::with_capacity(temporal.horizons.len());
        let mut horizons = Vec::with_capacity(temporal.horizons.len());
        let mut last_posterior = None;
        let z = 1.959963984540054;
        for (horizon_steps, entry) in temporal.horizons.iter().copied().zip(aligned.iter()) {
            require_identified(&entry.identification)?;
            let outcome_offset = i32::try_from(horizon_steps.saturating_sub(1)).unwrap_or(i32::MAX);
            let (effect, posterior) = antecedent_estimate::estimate_sequence_overlays(
                data,
                graph,
                &entry.indexer,
                &entry.estimand,
                outcome,
                outcome_offset,
                overlays,
                entry.identification.status,
                assumptions.clone(),
                self.bootstrap_replicates,
                bayes.as_ref(),
                ctx,
            )
            .map_err(CausalError::from)?;
            last_posterior = posterior.clone();
            let (point, se) = if let Some(post) = posterior {
                let eq = 0;
                (post.summaries.mean[eq], post.summaries.sd[eq])
            } else {
                (effect.ate, effect.se_bootstrap.unwrap_or(effect.se_analytic))
            };
            mean.push(point);
            if se.is_finite() {
                lower.push(point - z * se);
                upper.push(point + z * se);
            } else {
                lower.push(f64::NAN);
                upper.push(f64::NAN);
            }
            horizons.push(antecedent_core::HorizonIdentification {
                horizon: horizon_steps,
                status: entry.identification.status,
                method: Arc::clone(&entry.estimand.method),
                adjustment: Arc::from(named_adjustment_keys(entry)),
            });
        }
        let eval_level = overlays
            .iter()
            .find(|overlay| overlay.variable == treatment)
            .map(|overlay| overlay.assigned(0.0))
            .or_else(|| overlays.first().map(|overlay| overlay.assigned(0.0)))
            .unwrap_or(0.0);
        let support = antecedent_core::SupportReport {
            status: antecedent_core::SupportStatus::Supported,
            query_region: antecedent_core::SupportRegion {
                minima: Arc::from([
                    eval_level,
                    f64::from(temporal.horizons.first().copied().unwrap_or(1)),
                ]),
                maxima: Arc::from([
                    eval_level,
                    f64::from(temporal.horizons.last().copied().unwrap_or(1)),
                ]),
            },
            diagnostics: vec![antecedent_core::SupportDiagnostic {
                id: Arc::from("response.temporal.sequence_overlay"),
                values: Arc::from(
                    overlays
                        .iter()
                        .flat_map(|overlay| {
                            [
                                f64::from(overlay.variable.raw()),
                                f64::from(overlay.offset),
                                overlay.assigned(0.0),
                            ]
                        })
                        .collect::<Vec<_>>(),
                ),
                detail: Arc::from(
                    "sequential Sequence overlays as [variable, offset, assigned, …]; \
                     not a last-step collapse",
                ),
            }],
            warnings: Vec::new(),
            point_status: Some(Arc::from(vec![
                antecedent_core::SupportStatus::Supported;
                temporal.horizons.len()
            ])),
        };
        let mut response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: aggregate_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(
                    temporal.horizons.iter().map(|&h| f64::from(h)).collect::<Vec<_>>(),
                ),
                dimension: 1,
                mean: Arc::from(mean.clone()),
            }),
            uncertainty: ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.intervention_gcomp"),
            horizon_identification: Some(Arc::from(horizons)),
            interaction_structurally_zero: false,
        };
        if let Some(adjusted) = observation_adjusted {
            apply_temporal_observation_result(&mut response, adjusted);
        }
        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = vec![Diagnostic::new(
            "estimate.temporal.sequence_overlay",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "multi-step / joint Sequence runs as overlays on unfolded sequential g-computation; \
             identifier remains temporal.backdoor.unfolded; no last-step collapse",
        )];
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "refute.temporal_response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters are not applicable to a function-valued temporal response",
        ));
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TemporalBackdoorUnfolded,
            estimator_id: if matches!(self.inference, InferenceMode::Bayesian(_)) {
                EstimatorId::TemporalResponseBayesian
            } else {
                EstimatorId::TemporalResponseGcomp
            },
            treatment,
            outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                posterior: last_posterior,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_temporal_class(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            return Err(CausalError::Unsupported {
                message: "class-aware TemporalCpdag/TemporalPag pulse is Frequentist only",
            });
        }
        if matches!(query.policy, antecedent_core::TemporalPolicy::Sustained { from, until } if from != until)
        {
            return Err(CausalError::Compile {
                message: "class-aware TemporalCpdag/TemporalPag is Pulse and single-step \
                          Sustained only"
                    .into(),
            });
        }
        let started = Instant::now();
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(DEFAULT_PAG_IDENTIFIER_ID.as_str());
        let identifier_id: IdentifierId = identifier.parse()?;
        let (bundle, identify_cached) =
            if let Some(cache) = self.temporal_class_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                report_identify_compute(ctx);
                (self.identify_temporal_class(identifier_id, query)?, false)
            };
        let envelope = &bundle.envelope.envelope;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::Compile {
                message:
                    "temporal class-aware effect not identified (no identified mass in envelope)"
                        .into(),
            });
        }
        let mut diagnostics =
            vec![temporal_class_envelope_diagnostic(envelope, self.graph.class())];
        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut refute_atoms = Vec::new();
        for (i, (case, indexer)) in
            envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            let mut estimator = TemporalLinearAdjustment::new();
            estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
            estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
            let prep = estimator
                .prepare(data, &estimand, query, indexer, self.split.as_ref(), &ctx.kernel_policy)
                .map_err(CausalError::from)?;
            let mut workspace = EstimationWorkspace::default();
            let estimate = estimator
                .fit(&prep, &mut workspace, ctx, case.result.required_assumptions.clone())
                .map_err(CausalError::from)?;
            let w = case.weight.0;
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic));
            total_w += w;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                assumptions = estimate.assumptions.clone();
            }
            refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: w,
                estimand,
                indexer: Some(indexer.clone()),
            });
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "temporal class-aware envelope had no estimable identified cases".into(),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "temporal class-aware envelope missing estimand".into(),
        })?;
        let estimate = EffectEstimate::new(
            weighted_ate / total_w,
            mix_weighted_analytic_se(se_items),
            assumptions,
            OverlapPolicy::ExplicitOverride,
        );
        let identification = envelope_to_identification_result_for(
            envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        let tabular = TabularData::new(data.storage().clone());
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = run_envelope_effect_refuters(
            &tabular,
            &ate_q,
            &estimate,
            &refute_atoms,
            &mut refute_ws,
            ctx,
            self.refute,
            "temporal.linear.adjustment",
            &self.custom_validators,
            Some(query),
            self.split.as_ref(),
            Some(data.time_index()),
        )?;
        diagnostics.extend(na_diagnostics);
        diagnostics.push(envelope_se_omits_between_atom_variance());
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id: EstimatorId::TemporalLinearAdjustment,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: bundle.envelope.clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }

    fn identify_temporal_class(
        &self,
        identifier_id: IdentifierId,
        query: &TemporalEffectQuery,
    ) -> Result<crate::analysis::prepared::CachedTemporalClassIdentification, CausalError> {
        let envelope = match self.graph.class() {
            GraphClass::TemporalCpdag => {
                let cpdag = self.graph.as_temporal_cpdag().ok_or_else(|| CausalError::Compile {
                    message: "TemporalCpdag execute missing supplied graph".into(),
                })?;
                identify_temporal_cpdag(identifier_id, cpdag, query)?
            }
            GraphClass::TemporalPag => {
                let pag = self.graph.as_temporal_pag().ok_or_else(|| CausalError::Compile {
                    message: "TemporalPag execute missing supplied graph".into(),
                })?;
                identify_temporal_pag(identifier_id, pag, query)?
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "class-aware temporal execute requires TemporalCpdag or TemporalPag",
                });
            }
        };
        Ok(crate::analysis::prepared::CachedTemporalClassIdentification { envelope })
    }
}

fn temporal_class_envelope_diagnostic<G>(
    envelope: &IdentificationEnvelope<G>,
    class: GraphClass,
) -> Diagnostic {
    let code = if class == GraphClass::TemporalCpdag {
        "identify.temporal_cpdag.envelope"
    } else {
        "identify.temporal_pag.envelope"
    };
    Diagnostic::new(
        code,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "generalized.adjustment envelope: identified_mass={}, unidentified_mass={}, cases={}, limitations={:?}",
            envelope.identified_weight.0,
            envelope.unidentified_weight.0,
            envelope.cases.len(),
            envelope.critical_graph_features,
        ),
    )
}

fn aggregate_temporal_horizon_evidence<'a>(
    identifications: impl IntoIterator<Item = &'a IdentificationResult>,
) -> Result<(IdentificationStatus, antecedent_core::AssumptionSet), CausalError> {
    let mut aggregate_status = IdentificationStatus::NonparametricallyIdentified;
    let mut assumptions = antecedent_core::AssumptionSet::new();
    let mut saw_any = false;
    for identification in identifications {
        saw_any = true;
        match identification.status {
            IdentificationStatus::NonparametricallyIdentified => {}
            IdentificationStatus::IdentifiedUnderParametricRestrictions => {
                aggregate_status = IdentificationStatus::IdentifiedUnderParametricRestrictions;
            }
            status => {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal response requires point identification at every horizon; got {status:?}"
                    ),
                });
            }
        }
        for record in &identification.required_assumptions.entries {
            if !assumptions.entries.contains(record) {
                assumptions.push(record.clone());
            }
        }
    }
    if !saw_any {
        return Err(CausalError::Compile {
            message: "temporal response requires at least one horizon identification".into(),
        });
    }
    Ok((aggregate_status, assumptions))
}

pub(super) struct MediationHorizonClick {
    pub(super) horizon: u32,
    pub(super) identification: IdentificationResult,
    pub(super) estimand: IdentifiedEstimand,
    pub(super) indexer: TemporalIndexer,
    pub(super) adjustment: Arc<[antecedent_data::LaggedColumn]>,
}

pub(super) fn mediation_horizon_clicks(
    cache: &crate::analysis::prepared::CachedTemporalIdentification,
    query: &antecedent_core::MediationQuery,
    mut adjustment_for: impl FnMut(u32) -> Result<Arc<[antecedent_data::LaggedColumn]>, CausalError>,
) -> Result<Vec<MediationHorizonClick>, CausalError> {
    let mut clicks = Vec::with_capacity(query.horizons.len());
    for &horizon in query.horizons.iter() {
        let entry = cache.get(horizon).ok_or_else(|| CausalError::Compile {
            message: format!(
                "prepared temporal mediation identification missing horizon {horizon}"
            ),
        })?;
        clicks.push(MediationHorizonClick {
            horizon,
            identification: entry.identification.clone(),
            estimand: entry.estimand.clone(),
            indexer: entry.indexer.clone(),
            adjustment: adjustment_for(horizon)?,
        });
    }
    if clicks.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        });
    }
    Ok(clicks)
}

pub(super) fn lagged_adjustment_from_entry(
    entry: &crate::analysis::prepared::CachedTemporalHorizonIdentification,
) -> Arc<[antecedent_data::LaggedColumn]> {
    let outcome_offset = i32::try_from(entry.horizon.saturating_sub(1)).unwrap_or(0);
    entry
        .estimand
        .adjustment_set
        .iter()
        .filter_map(|&dense| {
            let key = entry.indexer.key_of(dense.raw()).ok()?;
            lagged_column_relative_to_outcome(key, outcome_offset)
        })
        .collect::<Vec<_>>()
        .into()
}

fn lagged_column_relative_to_outcome(
    key: antecedent_core::TemporalNodeKey,
    outcome_offset: i32,
) -> Option<antecedent_data::LaggedColumn> {
    let lag = outcome_offset.checked_sub(key.offset)?;
    if lag < 0 {
        return None;
    }
    Some(antecedent_data::LaggedColumn {
        variable: key.variable,
        lag: antecedent_core::Lag::from_raw(u32::try_from(lag).ok()?),
    })
}

pub(super) fn mediation_horizon_z_differs(clicks: &[MediationHorizonClick]) -> bool {
    let Some(first) = clicks.first() else {
        return false;
    };
    clicks.iter().skip(1).any(|click| click.adjustment.as_ref() != first.adjustment.as_ref())
}

fn lagged_adjustment_from_temporal(
    temporal: &antecedent_identify::TemporalIdentificationResult,
) -> Arc<[antecedent_data::LaggedColumn]> {
    let outcome_offset = match &temporal.result.query {
        CausalQuery::TemporalEffect(q) => q.outcome_offset(),
        CausalQuery::Mediation(q) => {
            i32::try_from(q.horizons.first().copied().unwrap_or(1).saturating_sub(1)).unwrap_or(0)
        }
        _ => 0,
    };
    temporal
        .result
        .estimands
        .first()
        .map(|estimand| {
            estimand
                .adjustment_set
                .iter()
                .filter_map(|&dense| {
                    let key = temporal.indexer.key_of(dense.raw()).ok()?;
                    lagged_column_relative_to_outcome(key, outcome_offset)
                })
                .collect::<Vec<_>>()
                .into()
        })
        .unwrap_or_else(|| Arc::from([]))
}

fn horizon_adjustment_sets_differ(
    entries: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
) -> bool {
    let Some(first) = entries.first() else {
        return false;
    };
    let first_z = named_adjustment_keys(first);
    entries.iter().skip(1).any(|entry| named_adjustment_keys(entry) != first_z)
}

fn contemporaneous_adjustment_variables(
    entries: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
) -> Vec<VariableId> {
    let mut adjustment = Vec::new();
    for entry in entries {
        for &dense in entry.estimand.adjustment_set.iter() {
            let Ok(key) = entry.indexer.key_of(dense.raw()) else {
                continue;
            };
            if !adjustment.contains(&key.variable) {
                adjustment.push(key.variable);
            }
        }
    }
    adjustment
}

fn named_adjustment_keys(
    entry: &crate::analysis::prepared::CachedTemporalHorizonIdentification,
) -> Vec<antecedent_core::TemporalNodeKey> {
    let mut keys: Vec<_> = entry
        .estimand
        .adjustment_set
        .iter()
        .filter_map(|&dense| entry.indexer.key_of(dense.raw()).ok())
        .collect();
    keys.sort();
    keys
}

fn append_temporal_observation_assumptions(
    query: &ResponseQuery,
    assumptions: &mut antecedent_core::AssumptionSet,
) {
    for claim in query.observation_assumptions.iter() {
        let (id, description) = match claim {
            ObservationAssumption::IndependentGiven(vars) => (
                "observation.independent_given",
                format!("observation/censoring independent given {vars:?}"),
            ),
            ObservationAssumption::OutcomeIndependentGiven(vars) => (
                "observation.outcome_independent_given",
                format!("observation independent of latent outcome given {vars:?}"),
            ),
            ObservationAssumption::Structural(model) => {
                ("observation.structural", format!("structural observation model {model}"))
            }
        };
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: id.into(),
                description: description.into(),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Untestable,
        });
    }
}

fn apply_temporal_observation_result(
    response: &mut CausalResponse,
    adjusted: &antecedent_estimate::ObservationAdjustedOutcome,
) {
    let (minimum_weight, maximum_weight, effective_sample_size) = {
        let minimum = adjusted
            .weights
            .iter()
            .copied()
            .filter(|weight| *weight > 0.0)
            .fold(f64::INFINITY, f64::min);
        let maximum = adjusted.weights.iter().copied().fold(0.0_f64, f64::max);
        let sum = adjusted.weights.iter().sum::<f64>();
        let sum_squares = adjusted.weights.iter().map(|weight| weight * weight).sum::<f64>();
        let ess = if sum_squares > 0.0 { sum * sum / sum_squares } else { 0.0 };
        (if minimum.is_finite() { minimum } else { 0.0 }, maximum, ess)
    };
    response.uncertainty = ResponseUncertainty::None;
    response.provenance_id = Arc::from("estimate.temporal_response.observation_adjusted");
    response.support.diagnostics.push(antecedent_core::SupportDiagnostic {
        id: Arc::from("response.observation_adjustment_weights"),
        values: Arc::from([minimum_weight, maximum_weight, effective_sample_size]),
        detail: Arc::from(
            "minimum positive weight, maximum weight, and Kish effective sample size; weights are diagnostic only and were already incorporated into the pseudo-outcome",
        ),
    });
    response.support.warnings.push(Diagnostic::new(
        "response.observation_joint_uncertainty_unavailable",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "point estimate includes observation correction; uncertainty is omitted because complete-data curve bands do not account for the estimated observation mechanism",
    ));
    response.support.warnings.push(Diagnostic::new(
        "response.observation_adjustment_method",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        adjusted.method.clone(),
    ));
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
        AssumptionStatus,
    };

    use super::*;

    fn horizon_result(
        status: IdentificationStatus,
        evidence: &'static str,
    ) -> IdentificationResult {
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let mut assumptions = AssumptionSet::new();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from(evidence),
                description: Arc::from("horizon-specific evidence"),
            },
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("test.temporal_horizon"),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        IdentificationResult::from_parts(
            status,
            CausalQuery::AverageEffect(query),
            Vec::new(),
            CausalExprArena::new(),
            DerivationTrace::default(),
            assumptions,
            Vec::new(),
            IdentificationPerformanceRecord::default(),
            None,
        )
    }

    #[test]
    fn temporal_horizon_evidence_is_unioned_and_never_upgraded() {
        let first =
            horizon_result(IdentificationStatus::NonparametricallyIdentified, "horizon.one");
        let second = horizon_result(
            IdentificationStatus::IdentifiedUnderParametricRestrictions,
            "horizon.two",
        );
        let (status, assumptions) = aggregate_temporal_horizon_evidence([&first, &second]).unwrap();
        assert_eq!(status, IdentificationStatus::IdentifiedUnderParametricRestrictions);
        for evidence in ["horizon.one", "horizon.two"] {
            assert!(assumptions.entries.iter().any(|record| {
                matches!(&record.assumption, Assumption::Custom { id, .. } if id.as_ref() == evidence)
            }));
        }

        let partial = horizon_result(IdentificationStatus::PartiallyIdentified, "horizon.partial");
        let error = aggregate_temporal_horizon_evidence([&first, &partial]).unwrap_err();
        assert!(error.to_string().contains("at every horizon"));
    }
}
