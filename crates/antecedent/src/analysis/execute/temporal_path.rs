// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    pub(crate) fn mediation_adjustment(
        &self,
        graph: &TemporalDag,
        query: &antecedent_core::MediationQuery,
    ) -> Result<Arc<[antecedent_data::LaggedColumn]>, CausalError> {
        if let Some(adjustment) = &self.mediation_adjustment_cache {
            return Ok(Arc::clone(adjustment));
        }
        let nodes = TemporalMediationIdentifier::adjustment_nodes(graph, query)
            .map_err(CausalError::from)?;
        Ok(nodes
            .iter()
            .map(|key| antecedent_data::LaggedColumn {
                variable: key.variable,
                lag: antecedent_core::Lag::from_raw(key.offset.unsigned_abs()),
            })
            .collect::<Vec<_>>()
            .into())
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
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = TemporalMediationIdentifier {
                    allow_natural_controlled_alias: true,
                    ..TemporalMediationIdentifier::new()
                }
                .identify(graph, query)
                .map_err(CausalError::from)?;
                let estimand = select_estimand(&identification, EstimatorId::TemporalMediation)?;
                Ok((identification, estimand))
            })?;
        require_identified(&identification)?;
        let adjustment = self.mediation_adjustment(graph, query)?;
        let est = TemporalMediationEstimator::new().with_allow_natural_controlled_alias(true);
        let mediation = est
            .estimate_with_adjustment(data, &estimand, query, &adjustment, &[], ctx)
            .map_err(CausalError::from)?;
        let estimate = mediation.effect.clone();
        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::mediation::refute_temporal_mediation_adjusted(
                data,
                &estimand,
                query,
                &mediation,
                self.refute == RefuteSuite::Full,
                &adjustment,
                ctx,
            )
            .map_err(CausalError::from)?
        };
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::Frontdoor,
            estimator_id: EstimatorId::TemporalMediation,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: Some(mediation),
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
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
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let id = TemporalMediationIdentifier {
                    allow_natural_controlled_alias: true,
                    ..TemporalMediationIdentifier::new()
                }
                .identify(graph, query)
                .map_err(CausalError::from)?;
                let estimand = select_estimand(&id, EstimatorId::BayesianTemporalMediation)?;
                Ok((id, estimand))
            })?;
        let adjustment = self.mediation_adjustment(graph, query)?;
        let preparations =
            prepare_temporal_mediation_adjusted(data, &estimand, query, &adjustment, ctx)
                .map_err(CausalError::from)?;
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
        let mut posterior = compose_temporal_mediation(
            &mechanisms[0],
            &mechanisms[1],
            query,
            identification.status,
        )
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
                query,
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
                    compose_temporal_mediation(&posts[0], &posts[1], query, identification.status)
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
            physical, identification, estimand, estimate,
            identifier_id: IdentifierId::Frontdoor, estimator_id: EstimatorId::BayesianTemporalMediation,
            treatment: query.treatment, outcome: query.outcome, identify_cached,
            extra_diagnostics: vec![Diagnostic::new("estimate.mediation.bayesian", DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                "independent Gaussian mediator and outcome mechanisms; total = direct + mediated for every posterior draw; natural effects use the linear no-interaction alias")],
            refutations, distribution: None, mediation: Some(mediation),
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None, cancelled: false, early_stopped: false,
            extras: IdentifiedExecuteExtras { posterior: Some(posterior), predictive_checks,
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
        let (aggregate_status, aggregate_assumptions) =
            aggregate_temporal_horizon_evidence(aligned.iter().map(|entry| &entry.identification))?;
        let mut identification = first.identification.clone();
        identification.status = aggregate_status;
        identification.required_assumptions = aggregate_assumptions.clone();
        let estimand = first.estimand.clone();
        let identifications: Vec<_> =
            aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();

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
            estimator.estimate_bayesian(data, &identifications, query, aggregate_status, aggregate_assumptions, &bayes, ctx)
        } else { estimator.estimate(data, &identifications, query, aggregate_status, aggregate_assumptions, ctx) }.map_err(CausalError::from)?;
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

fn horizon_adjustment_sets_differ(
    entries: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
) -> bool {
    let Some(first) = entries.first() else {
        return false;
    };
    let first_z = named_adjustment_keys(first);
    entries.iter().skip(1).any(|entry| named_adjustment_keys(entry) != first_z)
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
