// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    pub(super) fn execute_bayesian(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut clock = super::super::stage::StageClock::new();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        let identifier_id: IdentifierId = identifier.parse()?;
        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query) — rd is never consulted on this path —
        // all frozen there, so reuse is exact and observable via the
        // `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(self.identification_cache.as_deref(), || {
                let identification = identify_static(identifier_id, graph, query)?;
                let estimand = select_estimand(&identification, EstimatorId::BayesianGcomp)?;
                Ok((identification, estimand))
            })?;
        clock.finish(super::super::stage::STAGE_IDENTIFY);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Identify {
                identification: identification.clone(),
                estimand: estimand.clone(),
            },
        );

        let full_cols = data.schema().len();
        let (data_est, query_est, estimand_est) = project_for_ate_estimate(data, query, &estimand)?;
        let projected_cols = data_est.schema().len();

        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let mut est = bayesian_gcomp(&cfg, ctx);
        clock.begin(ctx, super::super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        let prep = est.prepare(&data_est, &estimand_est, &query_est).map_err(CausalError::from)?;
        let (resolved_prior, conflict_summary) =
            resolve_bayesian_prior_with_conflict(&cfg, &prep, Some(ctx))?;
        est.prior = resolved_prior;
        let mut ws = BayesianGCompWorkspace::default();
        let mut posterior =
            est.fit(&prep, identification.status, &mut ws, ctx).map_err(CausalError::from)?;
        if let Some(summary) = conflict_summary {
            posterior = with_conflict_summary(posterior, summary);
        }
        let estimate = effect_from_posterior(&posterior)?;
        clock.finish(super::super::stage::STAGE_ESTIMATE_POINT);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Point { estimate: estimate.clone() },
        );
        clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
        clock.finish(super::super::stage::STAGE_UNCERTAINTY);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Uncertainty { estimate: estimate.clone() },
        );

        let mut extra_diagnostics = Vec::new();
        if let Some(d) = projection_diagnostic(full_cols, projected_cols) {
            extra_diagnostics.push(d);
        }
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut extra_diagnostics, cs);
        }

        clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
        let mut refute_ws = EstimationWorkspace::default();
        let (mut refutations, na_diagnostics) = match self.refute {
            RefuteSuite::None => (Vec::new(), Vec::new()),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => run_refuters(
                &data_est,
                &estimand_est,
                &query_est,
                &estimate,
                &mut refute_ws,
                None,
                ctx,
                self.refute,
                "bayesian.gcomp",
                &self.custom_validators,
                None,
            )?,
        };
        extra_diagnostics.extend(na_diagnostics);
        // Prior + posterior PPC whenever refute is enabled (full PredictiveCheckReport retained).
        let mut predictive_checks = Vec::new();
        if !matches!(self.refute, RefuteSuite::None) {
            const PPC_ALPHA: f64 = 0.05;
            let ppc_prior = est
                .prior
                .clone()
                .unwrap_or_else(|| PriorSet::weakly_informative(prep.design.ncols));
            let prior_rep = PriorPredictiveCheck {
                n_sims: 200,
                seed: ctx.rng.master_seed(),
                ..PriorPredictiveCheck::new()
            }
            .check_with_prior(&prep, &ppc_prior, ctx)
            .map_err(CausalError::from)?;
            refutations.push(prior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(prior_rep);

            let post_rep = PosteriorPredictiveCheck::new()
                .check(&prep, &posterior)
                .map_err(CausalError::from)?;
            refutations.push(post_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(post_rep);
        }
        // Prior sensitivity / MCMC stay behind the full suite (Shared Bayesian UX).
        // Mode-select: α-multiplier grid when an external composed prior is present;
        // isotropic scale grid otherwise (avoids clearing banked priors).
        if matches!(self.refute, RefuteSuite::Full) {
            let (summary, sens) = evaluate_bayesian_prior_sensitivity(
                &cfg,
                &est,
                &prep,
                identification.status,
                &posterior,
                &mut ws,
                ctx,
            )?;
            refutations.push(sens.to_report(&summary, estimate.ate));
            posterior = with_prior_sensitivity(posterior, summary);

            let suite = ValidationSuite::new().with(ValidatorId::McmcDiagnostics);
            let mut bayes_ctx = BayesianSuiteContext::new(
                &est,
                &prep,
                &posterior,
                identification.status,
                &mut ws,
                estimate.ate,
            );
            let outcomes = suite.run_bayesian(&mut bayes_ctx, ctx).map_err(CausalError::from)?;
            refutations.extend(ValidationSuite::reports_only(&outcomes));
            extra_diagnostics.extend(validator_not_applicable_diagnostics(&outcomes));
        }
        clock.finish(super::super::stage::STAGE_VALIDATE);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Validate {
                refutations: refutations.clone(),
                predictive_checks: predictive_checks.clone(),
            },
        );

        let n_draws = u32::try_from(posterior.draws.n_draws).ok();
        let early_stopped = posterior.early_stopped;
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id: EstimatorId::BayesianGcomp,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: clock.wall_time_ns(),
            bootstrap_replicates_ok: None,
            cancelled: clock.cancelled(),
            early_stopped,
            extras: IdentifiedExecuteExtras {
                stage_timings_ns: clock.timings(),
                posterior: Some(posterior),
                n_draws,
                predictive_checks,
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_pag_bayesian(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        envelope: &IdentificationEnvelope<Pag>,
        identification: IdentificationResult,
        started: Instant,
    ) -> Result<StudyResult, CausalError> {
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let mut est = bayesian_gcomp(&cfg, ctx);

        let mut weights = Vec::new();
        let mut flags = Vec::new();
        let mut keys = Vec::new();
        let mut fit_atoms = Vec::new();
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut envelope_prior: Option<PriorSet> = None;
        let mut envelope_conflict: Option<antecedent_prob::ConflictSummary> = None;
        for (i, case) in envelope.cases.iter().enumerate() {
            let key = i as u64 + 1;
            keys.push(key);
            weights.push(case.weight.0);
            if identification_status_ok_for_case(case.result.status)
                && !case.result.estimands.is_empty()
            {
                flags.push(GraphIdentFlag::Identified);
                let mut estimand = select_estimand(&case.result, EstimatorId::BayesianGcomp)?;
                if estimand.method.as_ref().starts_with("generalized.adjustment") {
                    estimand.method = Arc::from("backdoor.adjustment");
                }
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand.clone());
                }
                fit_atoms.push((key, estimand, case.result.status));
            } else {
                flags.push(GraphIdentFlag::Unidentified);
            }
        }
        // Prepare once before Interactive subsample (0.6.0 eligibility), stash
        // so kept atoms are not prepared a second time. PAG keys are unique
        // (1..n); still use entry() so the stash is key-safe.
        let mut prepared = std::collections::HashMap::with_capacity(fit_atoms.len());
        for (i, (key, estimand, _)) in fit_atoms.iter().enumerate() {
            let prep = est.prepare(data, estimand, query).map_err(CausalError::from)?;
            if i == 0 {
                let (resolved, conflict) = resolve_envelope_prior_anchor(&cfg, &prep, ctx)?;
                envelope_prior = resolved;
                envelope_conflict = conflict;
            }
            prepared.entry(*key).or_insert(prep);
        }
        let graphs = WeightedGraphSamples::new(weights, flags, keys)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let mut subsample_notes = Vec::new();
        let graphs = maybe_interactive_subsample_graphs(
            self.latency_mode,
            graphs,
            ctx,
            &mut subsample_notes,
        )?;
        let keep = identified_envelope_keys(&graphs);
        let mut ws = BayesianGCompWorkspace::default();
        let mut per_graph = Vec::new();
        for (key, _estimand, status) in fit_atoms {
            if !keep.contains(&key) {
                continue;
            }
            // Graph-posterior keys may collide (shared adjacency masks). Aggregation
            // indexes draws by key, so fit each kept key once; later duplicates skip.
            let Some(prep) = prepared.remove(&key) else {
                continue;
            };
            est.prior.clone_from(&envelope_prior);
            let posterior = est.fit(&prep, status, &mut ws, ctx).map_err(CausalError::from)?;
            per_graph.push(envelope_draws_from_posterior(key, &posterior)?);
        }
        let mut posterior = aggregate_effect_envelope(
            &graphs,
            &per_graph,
            InferenceDiagnostics::analytic("pag_envelope"),
            EnvelopeOptions::default(),
        )
        .map_err(CausalError::from)?;
        if let Some(summary) = envelope_conflict {
            posterior = with_conflict_summary(posterior, summary);
        }
        let estimate = effect_from_posterior(&posterior)?;
        let estimand = primary_estimand.or(envelope.invariant.clone()).ok_or_else(|| {
            CausalError::Compile { message: "PAG Bayesian envelope missing estimand".into() }
        })?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(subsample_notes);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(Diagnostic::new(
            "estimate.pag.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", posterior.unidentified_mass),
        ));
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let mut refute_ws = EstimationWorkspace::default();
        let refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                let (reports, na_diagnostics) = run_refuters(
                    data,
                    &estimand,
                    query,
                    &estimate,
                    &mut refute_ws,
                    None,
                    ctx,
                    self.refute,
                    "bayesian.gcomp",
                    &self.custom_validators,
                    None,
                )?;
                diagnostics.extend(na_diagnostics);
                reports
            }
        };
        if matches!(self.refute, RefuteSuite::Full) {
            // PPC suite needs a single-graph fit context; skip with diagnostic when envelope-only.
            diagnostics.push(Diagnostic::new(
                "refute.bayesian.ppc.skipped",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "Bayesian PPC suite skipped for multi-graph PAG envelope; effect refuters ran on mixture mean",
            ));
        }

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id: EstimatorId::BayesianGcomp,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.bayesian_gcomp",
                    "estimate.aggregate_effect_envelope",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }

    /// Compile path for a supplied graph posterior ([`crate::StudyBuilder::graph_posterior`]).
    ///
    /// `self.graph` is not consulted here beyond its class (it only carries the
    /// placeholder shape `stub_accepted_graph_for` built at `build()` time) — the logical
    /// plan needs a structure argument for row-count / classification bookkeeping only.
    /// Real identification happens per-graph, against the posterior atoms, in
    /// [`Self::execute_graph_posterior_bayesian`] / [`Self::execute_dbn_posterior_bayesian`].
    ///
    /// # Errors
    ///
    /// `InferenceMode::Frequentist` (a graph posterior is a mixture over structures;
    /// only Bayesian inference can combine per-graph effect draws into an envelope), or
    /// an unsupported data/query combination (graph-posterior analysis supports tabular
    /// average-effect or temporal-effect queries only).
    pub(super) fn compile_graph_posterior(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        if matches!(self.inference, InferenceMode::Frequentist) {
            return Err(CausalError::Unsupported {
                message: "graph-posterior discovery requires inference=Bayesian for effect mixture",
            });
        }
        match (&self.data, &self.query) {
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q)) => {
                let n_vars =
                    u32::try_from(data.schema().len()).map_err(|_| CausalError::Compile {
                        message: "too many variables for graph-posterior compile".into(),
                    })?;
                let stub = Dag::with_variables(n_vars);
                let identifier = Arc::from("backdoor.adjustment");
                let estimator = Arc::from("bayesian.gcomp");
                let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph: &stub,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                logical.record.discovery_algorithm = Some(
                    self.graph_posterior
                        .as_ref()
                        .and_then(|gp| gp.algorithm.clone())
                        .unwrap_or_else(|| Arc::from("graph_posterior")),
                );
                logical.compile_physical(ctx)
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
            ) => {
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                let mut logical = compile_logical_temporal_effect_classified(
                    data,
                    &TemporalDag::empty(),
                    q,
                    self.split,
                    false,
                    class,
                )?;
                logical.record.discovery_algorithm = Some(
                    self.graph_posterior
                        .as_ref()
                        .and_then(|gp| gp.algorithm.clone())
                        .unwrap_or_else(|| Arc::from("dbn_posterior")),
                );
                logical.compile_physical(ctx)
            }
            _ => Err(CausalError::Unsupported {
                message: "graph-posterior analysis supports tabular average-effect or \
                          temporal-effect queries only",
            }),
        }
    }

    /// Execute a supplied static graph posterior into a Bayesian effect envelope.
    ///
    /// Identifies and fits each posterior atom independently; atoms that fail to
    /// identify (or fail to build a valid DAG from their adjacency mask) are marked
    /// [`GraphIdentFlag::Unidentified`] and contribute their posterior weight to
    /// [`antecedent_estimate::CausalPosterior::unidentified_mass`] rather than being
    /// dropped or having their mass redistributed onto the identified atoms.
    ///
    /// # Errors
    ///
    /// `InferenceMode::Frequentist`, or any per-graph identification/estimation
    /// infrastructure failure (envelope aggregation, missing effect column, …).
    pub(super) fn execute_graph_posterior_bayesian(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::Unsupported {
                    message: "graph-posterior discovery requires inference=Bayesian for effect mixture",
                });
            }
        };
        let mut est = bayesian_gcomp(&cfg, ctx);

        let mut weights = Vec::with_capacity(gp.n_graphs);
        let mut flags = Vec::with_capacity(gp.n_graphs);
        let mut keys = Vec::with_capacity(gp.n_graphs);
        let mut fit_atoms = Vec::new();
        let mut ident_cache: std::collections::HashMap<
            u64,
            Option<(IdentifiedEstimand, IdentificationStatus, IdentificationResult)>,
        > = std::collections::HashMap::new();
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut primary_identification: Option<IdentificationResult> = None;
        let mut envelope_prior: Option<PriorSet> = None;
        let mut envelope_conflict: Option<antecedent_prob::ConflictSummary> = None;

        for i in 0..gp.n_graphs {
            if ctx.cancellation.is_cancelled() {
                for j in i..gp.n_graphs {
                    keys.push(gp.graph_keys[j]);
                    weights.push(gp.weights[j]);
                    flags.push(GraphIdentFlag::Unidentified);
                }
                break;
            }
            if let Some(p) = &ctx.progress {
                #[allow(clippy::cast_precision_loss)]
                p.report(i as f64 / gp.n_graphs.max(1) as f64, "envelope.identify");
            }
            let mask = gp.adjacency[i];
            let key = gp.graph_keys[i];
            keys.push(key);
            weights.push(gp.weights[i]);
            let cached = if let Some(hit) = ident_cache.get(&mask) {
                hit.clone()
            } else {
                let resolved = (|| -> Result<
                    Option<(IdentifiedEstimand, IdentificationStatus, IdentificationResult)>,
                    CausalError,
                > {
                    let Ok(dag) = dag_from_adjacency_mask(mask, gp.n_vars) else {
                        return Ok(None);
                    };
                    let Ok(identification) = identify_static(DEFAULT_IDENTIFIER_ID, &dag, query)
                    else {
                        return Ok(None);
                    };
                    if !identification_status_ok_for_case(identification.status)
                        || identification.estimands.is_empty()
                    {
                        return Ok(None);
                    }
                    let estimand = select_estimand(&identification, EstimatorId::BayesianGcomp)?;
                    Ok(Some((estimand, identification.status, identification)))
                })()?;
                ident_cache.insert(mask, resolved.clone());
                resolved
            };
            if let Some((estimand, status, identification)) = cached {
                flags.push(GraphIdentFlag::Identified);
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand.clone());
                    primary_identification = Some(identification);
                }
                fit_atoms.push((key, estimand, status));
            } else {
                flags.push(GraphIdentFlag::Unidentified);
            }
        }

        // Prepare once before Interactive subsample (0.6.0 eligibility), stash
        // so kept atoms are not prepared a second time. Keys may collide when
        // several atoms share an adjacency mask — keep the first prep per key.
        let mut prepared = std::collections::HashMap::with_capacity(fit_atoms.len());
        for (i, (key, estimand, _)) in fit_atoms.iter().enumerate() {
            let prep = est.prepare(data, estimand, query).map_err(CausalError::from)?;
            if i == 0 {
                let (resolved, conflict) = resolve_envelope_prior_anchor(&cfg, &prep, ctx)?;
                envelope_prior = resolved;
                envelope_conflict = conflict;
            }
            prepared.entry(*key).or_insert(prep);
        }
        let graphs = WeightedGraphSamples::new(weights, flags, keys)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let mut subsample_notes = Vec::new();
        let graphs = maybe_interactive_subsample_graphs(
            self.latency_mode,
            graphs,
            ctx,
            &mut subsample_notes,
        )?;
        let keep = identified_envelope_keys(&graphs);
        let mut ws = BayesianGCompWorkspace::default();
        let mut per_graph = Vec::new();
        for (key, _estimand, status) in fit_atoms {
            if !keep.contains(&key) {
                continue;
            }
            // Aggregation indexes draws by key; fit each kept key once.
            let Some(prep) = prepared.remove(&key) else {
                continue;
            };
            est.prior.clone_from(&envelope_prior);
            let posterior = est.fit(&prep, status, &mut ws, ctx).map_err(CausalError::from)?;
            per_graph.push(envelope_draws_from_posterior(key, &posterior)?);
        }
        let mut posterior = aggregate_effect_envelope(
            &graphs,
            &per_graph,
            InferenceDiagnostics::analytic("graph_posterior_envelope"),
            EnvelopeOptions::default(),
        )
        .map_err(CausalError::from)?;
        if let Some(summary) = envelope_conflict {
            posterior = with_conflict_summary(posterior, summary);
        }
        let estimate = effect_from_posterior(&posterior)?;
        let identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "graph-posterior envelope: no identified graph atoms".into(),
        })?;
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "graph-posterior envelope: missing estimand".into(),
        })?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(subsample_notes);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(Diagnostic::new(
            "estimate.graph_posterior.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", posterior.unidentified_mass),
        ));
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let mut refute_ws = EstimationWorkspace::default();
        let refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                let (reports, na_diagnostics) = run_refuters(
                    data,
                    &estimand,
                    query,
                    &estimate,
                    &mut refute_ws,
                    None,
                    ctx,
                    self.refute,
                    "bayesian.gcomp",
                    &self.custom_validators,
                    None,
                )?;
                diagnostics.extend(na_diagnostics);
                reports
            }
        };

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::BackdoorAdjustment,
            estimator_id: EstimatorId::BayesianGcomp,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.aggregate_effect_envelope",
                    "estimate.bayesian_gcomp",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }

    /// Execute a supplied DBN (temporal) graph posterior into a Bayesian effect envelope.
    ///
    /// Same envelope discipline as [`Self::execute_graph_posterior_bayesian`], adapted to
    /// per-atom lag masks: each atom's contemporaneous + lagged adjacency is completed into
    /// a [`antecedent_graph::TemporalDag`], identified via [`TemporalBackdoorIdentifier`],
    /// and fit with [`BayesianTemporalGcomp`].
    ///
    /// # Errors
    ///
    /// `InferenceMode::Frequentist`, a posterior missing per-atom lag masks / `max_lag`
    /// (i.e. not actually a DBN posterior), or per-graph estimation infrastructure failure.
    pub(super) fn execute_dbn_posterior_bayesian(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::Unsupported {
                    message: "DBN graph-posterior discovery requires inference=Bayesian for effect mixture",
                });
            }
        };
        let lag_masks = gp.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
            message: "DBN posterior missing per-atom lag masks".into(),
        })?;
        let max_lag = gp.max_lag.ok_or_else(|| CausalError::Compile {
            message: "DBN posterior missing max_lag".into(),
        })?;
        let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();

        let mut bayes = bayesian_temporal_gcomp(&cfg, ctx);

        let mut weights = Vec::with_capacity(gp.n_graphs);
        let mut flags = Vec::with_capacity(gp.n_graphs);
        let mut keys = Vec::with_capacity(gp.n_graphs);
        let mut fit_atoms = Vec::new();
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut primary_identification: Option<IdentificationResult> = None;
        let mut envelope_prior: Option<PriorSet> = None;
        let mut envelope_conflict: Option<antecedent_prob::ConflictSummary> = None;

        for i in 0..gp.n_graphs {
            if ctx.cancellation.is_cancelled() {
                for j in i..gp.n_graphs {
                    keys.push(gp.graph_keys[j]);
                    weights.push(gp.weights[j]);
                    flags.push(GraphIdentFlag::Unidentified);
                }
                break;
            }
            if let Some(p) = &ctx.progress {
                #[allow(clippy::cast_precision_loss)]
                p.report(i as f64 / gp.n_graphs.max(1) as f64, "envelope.identify");
            }
            let cmask = gp.adjacency[i];
            let lmask = lag_masks[i];
            let key = gp.graph_keys[i];
            keys.push(key);
            weights.push(gp.weights[i]);
            let Ok(tdag) = temporal_dag_from_dbn_masks(cmask, lmask, gp.n_vars, max_lag, &vars)
            else {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            };
            let Ok(id_res) = TemporalBackdoorIdentifier::new().identify_temporal(&tdag, query)
            else {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            };
            let identification = id_res.result;
            if !identification_status_ok_for_case(identification.status)
                || identification.estimands.is_empty()
            {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            }
            let Ok(estimand) =
                select_estimand(&identification, EstimatorId::TemporalLinearAdjustment)
            else {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            };
            flags.push(GraphIdentFlag::Identified);
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                primary_identification = Some(identification.clone());
            }
            fit_atoms.push((key, estimand, identification, id_res.indexer));
        }

        // Soft prepare+fit before Interactive subsample (0.6.0): demote failures
        // so stratified selection only chooses among atoms that already produced
        // draws. Prior anchors on the first successful prepare.
        let mut per_graph = Vec::new();
        let mut ws = BayesianGCompWorkspace::default();
        for (key, estimand, identification, indexer) in fit_atoms {
            let mut temporal_est = TemporalLinearAdjustment::new();
            temporal_est.inner.overlap = OverlapPolicy::ExplicitOverride;
            let Ok(prep) = temporal_est.prepare(
                data,
                &estimand,
                query,
                &indexer,
                self.split.as_ref(),
                &ctx.kernel_policy,
            ) else {
                if let Some(idx) = keys.iter().position(|&k| k == key) {
                    flags[idx] = GraphIdentFlag::Unidentified;
                }
                continue;
            };
            let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
            if envelope_prior.is_none() {
                let (resolved, conflict) = resolve_envelope_prior_anchor(&cfg, &bprep, ctx)?;
                envelope_prior = resolved;
                envelope_conflict = conflict;
            }
            bayes.inner.prior.clone_from(&envelope_prior);
            let Ok(posterior) = bayes.fit(&bprep, identification.status, &mut ws, ctx) else {
                if let Some(idx) = keys.iter().position(|&k| k == key) {
                    flags[idx] = GraphIdentFlag::Unidentified;
                }
                continue;
            };
            match envelope_draws_from_posterior(key, &posterior) {
                Ok(draws) => per_graph.push(draws),
                Err(_) => {
                    if let Some(idx) = keys.iter().position(|&k| k == key) {
                        flags[idx] = GraphIdentFlag::Unidentified;
                    }
                }
            }
        }

        let graphs = WeightedGraphSamples::new(weights, flags, keys)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let mut subsample_notes = Vec::new();
        let (graphs, per_graph) = maybe_interactive_envelope_subsample(
            self.latency_mode,
            graphs,
            per_graph,
            ctx,
            &mut subsample_notes,
        )?;
        let mut posterior = aggregate_effect_envelope(
            &graphs,
            &per_graph,
            InferenceDiagnostics::analytic("dbn_posterior_envelope"),
            EnvelopeOptions::default(),
        )
        .map_err(CausalError::from)?;
        if let Some(summary) = envelope_conflict {
            posterior = with_conflict_summary(posterior, summary);
        }
        let estimate = effect_from_posterior(&posterior)?;
        let identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "DBN posterior envelope: no identified graph atoms".into(),
        })?;
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "DBN posterior envelope: missing estimand".into(),
        })?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(subsample_notes);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(Diagnostic::new(
            "estimate.dbn_posterior.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", posterior.unidentified_mass),
        ));
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TemporalBackdoorUnfolded,
            estimator_id: EstimatorId::BayesianGcomp,
            treatment: query.treatment,
            outcome: query.outcome,
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
                    "discover.dbn_posterior",
                    "dbn_posterior",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.aggregate_effect_envelope",
                    "estimate.bayesian.temporal.gcomp",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }
}

fn envelope_draws_from_posterior(
    key: u64,
    posterior: &CausalPosterior,
) -> Result<GraphEffectDraws, CausalError> {
    let col = posterior.effect_column().ok_or_else(|| CausalError::Compile {
        message: "Bayesian posterior missing effect column".into(),
    })?;
    let draws =
        posterior.draws.column(col).map_err(|e| CausalError::Compile { message: e.to_string() })?;
    Ok(GraphEffectDraws { graph_key: key, effect_draws: Arc::from(draws.to_vec()) })
}
