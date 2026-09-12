// SPDX-License-Identifier: MIT OR Apache-2.0

use super::sequential_validation::{SequentialValidationAtom, validate_sequential};
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
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let estimator_id =
            if conditional { EstimatorId::BayesianConditional } else { EstimatorId::BayesianGcomp };
        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query) — rd is never consulted on this path —
        // all frozen there, so reuse is exact and observable via the
        // `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static(identifier_id, graph, query)?;
                let estimand = select_estimand(&identification, estimator_id)?;
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
        let prep = if conditional {
            let q = antecedent_core::ConditionalEffectQuery::try_new(query_est.clone())
                .map_err(|e| CausalError::Compile { message: e.to_string() })?;
            est.prepare_conditional(&data_est, &estimand_est, &q)
        } else {
            est.prepare(&data_est, &estimand_est, &query_est)
        }
        .map_err(CausalError::from)?;
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
                estimator_id.as_str(),
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
            estimator_id,
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

    pub(super) fn execute_pag_bayesian<G>(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        envelope: &IdentificationEnvelope<G>,
        identification: IdentificationResult,
        identify_cached: bool,
        started: Instant,
        envelope_diagnostic: Diagnostic,
        class_tag: &str,
    ) -> Result<StudyResult, CausalError> {
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let mut est = bayesian_gcomp(&cfg, ctx);
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let estimator_id =
            if conditional { EstimatorId::BayesianConditional } else { EstimatorId::BayesianGcomp };

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
                let estimand = select_estimand(&case.result, estimator_id)?;
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
            let prep = if conditional {
                let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
                est.prepare_conditional(data, estimand, &q)
            } else {
                est.prepare(data, estimand, query)
            }
            .map_err(CausalError::from)?;
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
        let mut atoms = Vec::new();
        for (key, estimand, status) in fit_atoms {
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
            let weight = identified_weight_for_key(&graphs, key);
            atoms.push(EnvelopeAtomFit {
                key,
                prep,
                posterior,
                status,
                weight,
                estimand,
                indexer: None,
            });
        }
        let mut posterior = aggregate_effect_envelope(
            &graphs,
            &per_graph,
            InferenceDiagnostics::analytic(format!("{class_tag}_envelope")),
            EnvelopeOptions::default(),
        )
        .map_err(CausalError::from)?;
        if let Some(summary) = envelope_conflict {
            posterior = with_conflict_summary(posterior, summary);
        }
        let estimate = effect_from_posterior(&posterior)?;
        let estimand = primary_estimand.or(envelope.invariant.clone()).ok_or_else(|| {
            CausalError::Compile {
                message: format!("{class_tag} Bayesian envelope missing estimand"),
            }
        })?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(envelope_diagnostic);
        diagnostics.extend(subsample_notes);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(Diagnostic::new(
            format!("estimate.{class_tag}.envelope"),
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", posterior.unidentified_mass),
        ));
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let mut refute_ws = EstimationWorkspace::default();
        let mut refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                let (reports, mix_diagnostics) = run_envelope_effect_refuters(
                    data,
                    query,
                    &estimate,
                    &envelope_refute_atoms(&atoms),
                    &mut refute_ws,
                    ctx,
                    self.refute,
                    estimator_id.as_str(),
                    &self.custom_validators,
                    None,
                    self.split.as_ref(),
                    None,
                )?;
                diagnostics.extend(mix_diagnostics);
                reports
            }
        };
        let predictive_checks = run_envelope_bayesian_full_validation(
            self.refute,
            &cfg,
            &est,
            &atoms,
            &mut posterior,
            estimate.ate,
            ctx,
            &mut refutations,
            &mut diagnostics,
        )?;

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id,
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
                estimate_provenance: Some(provenance_ids(
                    "estimate.bayesian_gcomp",
                    "estimate.aggregate_effect_envelope",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                predictive_checks,
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
    /// [`Self::execute_graph_posterior_bayesian`],
    /// [`Self::execute_graph_posterior_frequentist`], or
    /// [`Self::execute_dbn_posterior_bayesian`].
    ///
    /// # Errors
    ///
    /// `InferenceMode::Frequentist` on a temporal (DBN) posterior — that combiner
    /// is 1.7 — or an unsupported data/query combination (graph-posterior analysis
    /// supports tabular average-effect, temporal-effect, or temporal-mediation
    /// queries only). Response mixtures remain unscheduled post-1.x.
    pub(super) fn compile_graph_posterior(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PhysicalExecutionPlan, CausalError> {
        match (&self.data, &self.query) {
            (DataInput::Tabular(data), CausalQuery::Response(q)) => {
                let n_vars =
                    u32::try_from(data.schema().len()).map_err(|_| CausalError::Compile {
                        message: "too many variables for graph-posterior compile".into(),
                    })?;
                let stub = Dag::with_variables(n_vars);
                let (identifier, estimator) = self.resolve_response_pair(q);
                let mut logical = compile_logical_static_response(StaticResponseCompileInput {
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
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q)) => {
                let n_vars =
                    u32::try_from(data.schema().len()).map_err(|_| CausalError::Compile {
                        message: "too many variables for graph-posterior compile".into(),
                    })?;
                let stub = Dag::with_variables(n_vars);
                let identifier = Arc::from("backdoor.adjustment");
                let estimator = match &self.inference {
                    InferenceMode::Frequentist => Arc::from("linear.adjustment.ate"),
                    InferenceMode::Bayesian(_) => Arc::from("bayesian.gcomp"),
                };
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
                if is_multi_step_sustained(q) {
                    logical.record.estimator = Some(Arc::from("temporal.sequential.gcomp"));
                }
                logical.record.validation_suite = self.validation_suite_id();
                logical.record.discovery_algorithm = Some(
                    self.graph_posterior
                        .as_ref()
                        .and_then(|gp| gp.algorithm.clone())
                        .unwrap_or_else(|| Arc::from("dbn_posterior")),
                );
                logical.compile_physical(ctx)
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Mediation(q)) => {
                q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                let mut logical = compile_logical_temporal_effect_classified(
                    data,
                    &TemporalDag::empty(),
                    &TemporalEffectQuery::pulse(q.treatment, q.outcome, 1.0),
                    self.split,
                    false,
                    class,
                )?;
                logical.record.plan_id = Arc::from("temporal_mediation");
                logical.record.identifier = Some(Arc::from("temporal.mediation"));
                logical.record.estimator = Some(Arc::from("temporal.mediation.bayesian"));
                logical.record.validation_suite = self.validation_suite_id();
                logical.record.query_variables = Arc::from([q.treatment, q.outcome]);
                logical.query = CausalQuery::Mediation(q.clone());
                logical.record.discovery_algorithm = Some(
                    self.graph_posterior
                        .as_ref()
                        .and_then(|gp| gp.algorithm.clone())
                        .unwrap_or_else(|| Arc::from("dbn_posterior")),
                );
                logical.compile_physical(ctx)
            }
            _ => Err(CausalError::Unsupported {
                message: "graph-posterior analysis supports tabular average-effect, \
                          temporal-effect, or temporal-mediation queries only",
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

        let (identified, identify_cached) =
            if let Some(cache) = self.graph_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_graph_posterior_identification_cache(
                        gp, query, ctx,
                    )?,
                    false,
                )
            };
        let graphs = identified.graphs.clone();
        let fit_atoms: Vec<_> = identified
            .atoms
            .iter()
            .map(|atom| (atom.key, atom.estimand.clone(), atom.identification.clone()))
            .collect();
        // Interactive subsampling can demote the first structurally identified
        // atom before estimation. Keep the shared prior anchored to that
        // original first atom below, but anchor the public estimand and
        // identification to the first atom that actually contributes draws.
        let mut primary_estimand = None;
        let mut primary_identification = None;
        let mut envelope_prior: Option<PriorSet> = None;
        let mut envelope_conflict: Option<antecedent_prob::ConflictSummary> = None;

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
        let mut atoms = Vec::new();
        for (key, estimand, identification) in fit_atoms {
            if !keep.contains(&key) {
                continue;
            }
            // Aggregation indexes draws by key; fit each kept key once.
            let Some(prep) = prepared.remove(&key) else {
                continue;
            };
            est.prior.clone_from(&envelope_prior);
            let posterior =
                est.fit(&prep, identification.status, &mut ws, ctx).map_err(CausalError::from)?;
            per_graph.push(envelope_draws_from_posterior(key, &posterior)?);
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                primary_identification = Some(identification.clone());
            }
            let weight = identified_weight_for_key(&graphs, key);
            atoms.push(EnvelopeAtomFit {
                key,
                prep,
                posterior,
                status: identification.status,
                weight,
                estimand,
                indexer: None,
            });
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
        let mut refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                let (reports, mix_diagnostics) = run_envelope_effect_refuters(
                    data,
                    query,
                    &estimate,
                    &envelope_refute_atoms(&atoms),
                    &mut refute_ws,
                    ctx,
                    self.refute,
                    "bayesian.gcomp",
                    &self.custom_validators,
                    None,
                    self.split.as_ref(),
                    None,
                )?;
                diagnostics.extend(mix_diagnostics);
                reports
            }
        };
        let predictive_checks = run_envelope_bayesian_full_validation(
            self.refute,
            &cfg,
            &est,
            &atoms,
            &mut posterior,
            estimate.ate,
            ctx,
            &mut refutations,
            &mut diagnostics,
        )?;

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
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.aggregate_effect_envelope",
                    "estimate.bayesian_gcomp",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                predictive_checks,
                ..Default::default()
            },
        }))
    }

    /// Execute a supplied static graph posterior into a Frequentist effect mixture.
    ///
    /// Same atoms and unidentified-mass rule as
    /// [`Self::execute_graph_posterior_bayesian`]: identified atoms contribute
    /// `linear.adjustment.ate` point estimates, mixed as
    /// `E[τ | identified] = Σ w_i τ_i / identified_mass`. Unidentified mass is
    /// retained in the envelope diagnostic and is not redistributed.
    ///
    /// # Errors
    ///
    /// No identified atoms, or any per-graph identification/estimation failure.
    pub(super) fn execute_graph_posterior_frequentist(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if !matches!(self.inference, InferenceMode::Frequentist) {
            return Err(CausalError::Unsupported {
                message: "execute_graph_posterior_frequentist requires inference=Frequentist",
            });
        }
        let estimator_id = EstimatorId::LinearAdjustmentAte;
        let estimator = estimator_id.as_str();
        let (identified, identify_cached) =
            if let Some(cache) = self.graph_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_graph_posterior_identification_cache(
                        gp, query, ctx,
                    )?,
                    false,
                )
            };
        let mut subsample_notes = Vec::new();
        let graphs = maybe_interactive_subsample_graphs(
            self.latency_mode,
            identified.graphs.clone(),
            ctx,
            &mut subsample_notes,
        )?;
        let keep = identified_envelope_keys(&graphs);

        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand = None;
        let mut primary_identification = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut refute_atoms = Vec::new();
        for atom in identified.atoms.iter() {
            if !keep.contains(&atom.key) {
                continue;
            }
            let mut case_ws = StaticEstimateWorkspaces::default();
            let case_spec = self
                .estimator_spec
                .clone()
                .unwrap_or(crate::estimator_spec::EstimatorSpec::Default(estimator_id));
            let estimate = estimate_static_effect(
                &case_spec,
                data,
                &atom.estimand,
                query,
                atom.identification.required_assumptions.clone(),
                self.bootstrap_replicates,
                self.overlap_policy,
                self.population_registry.as_ref(),
                ctx,
                &mut case_ws,
            )?;
            let w = identified_weight_for_key(&graphs, atom.key);
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic));
            total_w += w;
            if primary_estimand.is_none() {
                primary_estimand = Some(atom.estimand.clone());
                primary_identification = Some(atom.identification.clone());
                assumptions = estimate.assumptions.clone();
            }
            refute_atoms.push(EnvelopeRefuteAtom {
                key: atom.key,
                weight: w,
                estimand: atom.estimand.clone(),
                indexer: None,
            });
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "graph-posterior envelope: no identified graph atoms".into(),
            });
        }
        let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "graph-posterior envelope: no identified graph atoms".into(),
        })?;
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "graph-posterior envelope: missing estimand".into(),
        })?;
        let estimate = EffectEstimate::new(
            weighted_ate / total_w,
            mix_weighted_analytic_se(se_items),
            assumptions,
            OverlapPolicy::ExplicitOverride,
        );
        let unidentified_mass: f64 = graphs
            .weights
            .iter()
            .zip(graphs.identified.iter())
            .filter(|(_, flag)| **flag != GraphIdentFlag::Identified)
            .map(|(weight, _)| *weight)
            .sum();
        let contributing: Vec<&IdentifiedEstimand> =
            refute_atoms.iter().map(|atom| &atom.estimand).collect();
        identification.status =
            graph_posterior_mixture_status(unidentified_mass, &contributing, identification.status);

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(subsample_notes);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(envelope_se_omits_between_atom_variance());
        diagnostics.push(Diagnostic::new(
            "estimate.graph_posterior.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "identified_mass={total_w}, unidentified_mass={unidentified_mass}, atoms={}",
                refute_atoms.len()
            ),
        ));

        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = run_envelope_effect_refuters(
            data,
            query,
            &estimate,
            &refute_atoms,
            &mut refute_ws,
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
            None,
            self.split.as_ref(),
            None,
        )?;
        diagnostics.extend(na_diagnostics);

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::BackdoorAdjustment,
            estimator_id,
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
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.linear_adjustment",
                    "estimate.linear_adjustment_ate",
                )),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }

    /// Execute a supplied DBN posterior with fixed graph weights and a shared
    /// outer circular-block bootstrap across every contributing atom.
    pub(super) fn execute_dbn_posterior_frequentist(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if is_multi_step_sustained(query) && self.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "multi-step sustained g-computation currently requires no discovery-estimation split",
            });
        }
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let (identified, identify_cached) =
            if let Some(cache) = self.dbn_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_dbn_posterior_identification_cache(
                        gp, &vars, query, ctx,
                    )?,
                    false,
                )
            };
        let mut contexts = Vec::new();
        let mut point_sum = 0.0;
        let mut point_mass = 0.0;
        let mut primary = None;
        for atom in identified.atoms.iter() {
            let weight = identified_weight_for_key(&identified.graphs, atom.key);
            if weight <= 0.0 {
                continue;
            }
            let estimate = fit_frequentist_dbn_atom(data, gp, &vars, atom, query, ctx)?;
            point_sum += weight * estimate.ate;
            point_mass += weight;
            if primary.is_none() {
                primary = Some((
                    atom.estimand.clone(),
                    atom.identification.clone(),
                    estimate.assumptions.clone(),
                ));
            }
            contexts.push((atom, weight));
        }
        let (estimand, mut identification, assumptions) =
            primary.ok_or_else(|| CausalError::Compile {
                message: "Frequentist DBN posterior has no estimable identified atom".into(),
            })?;
        let point = point_sum / point_mass;
        let mut bootstrap = Vec::new();
        let mut attempted = 0u32;
        let n = data.row_count();
        let block_length =
            (gp.max_lag.unwrap_or(1) as usize + 1).max(integer_cube_root_ceil(n)).min(n);
        let plan = antecedent_data::ResamplingPlan::CircularBlock { length: block_length };
        let mut index_scratch = Vec::with_capacity(n);
        for replicate in 0..self.bootstrap_replicates {
            if ctx.cancellation.is_cancelled() {
                break;
            }
            attempted += 1;
            let mut rng = ctx.rng.stream(0xDBF0_0000 + u64::from(replicate));
            let sampled =
                antecedent_data::resample_timeseries(data, plan, &mut rng, &mut index_scratch)
                    .map_err(CausalError::from)?;
            let mut sum = 0.0;
            let mut failed = false;
            for (atom, weight) in &contexts {
                if let Ok(estimate) =
                    fit_frequentist_dbn_atom(&sampled, gp, &vars, atom, query, ctx)
                {
                    sum += *weight * estimate.ate;
                } else {
                    failed = true;
                    break;
                }
            }
            if !failed && sum.is_finite() {
                bootstrap.push(sum / point_mass);
            }
        }
        let se = if bootstrap_has_enough_successes(bootstrap.len(), attempted as usize) {
            let mean = bootstrap.iter().sum::<f64>() / bootstrap.len() as f64;
            (bootstrap.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
                / (bootstrap.len() - 1) as f64)
                .sqrt()
        } else {
            f64::NAN
        };
        if identified.graphs.unidentified_mass() > 0.0 {
            identification.status = IdentificationStatus::GraphDependent;
        }
        let estimate = EffectEstimate::from_parts(
            point,
            f64::NAN,
            se.is_finite().then_some(se),
            (self.bootstrap_replicates > 0)
                .then_some(u32::try_from(bootstrap.len()).unwrap_or(u32::MAX)),
            (self.bootstrap_replicates > 0).then_some(
                attempted.saturating_sub(u32::try_from(bootstrap.len()).unwrap_or(u32::MAX)),
            ),
            ctx.cancellation.is_cancelled(),
            false,
            assumptions,
            OverlapPolicy::ExplicitOverride,
            None,
            None,
        );
        let refute_atoms = contexts
            .iter()
            .map(|(atom, weight)| EnvelopeRefuteAtom {
                key: atom.key,
                weight: *weight,
                estimand: atom.estimand.clone(),
                indexer: Some(atom.indexer.clone()),
            })
            .collect::<Vec<_>>();
        let tabular = TabularData::new(data.storage().clone());
        let ate_query = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let (refutations, mut diagnostics) =
            if self.refute == RefuteSuite::None && self.custom_validators.is_empty() {
                (Vec::new(), Vec::new())
            } else if is_multi_step_sustained(query) {
                let validation_atoms = contexts
                    .iter()
                    .map(|(atom, weight)| {
                        Ok(SequentialValidationAtom {
                            weight: *weight,
                            graph: crate::analysis::prepared::temporal_dag_from_dbn_atom(
                                gp, atom.key, &vars,
                            )?,
                            indexer: atom.indexer.clone(),
                            estimand: atom.estimand.clone(),
                            status: atom.identification.status,
                            estimate: fit_frequentist_dbn_atom(data, gp, &vars, atom, query, ctx)?,
                            mechanisms: Vec::new(),
                        })
                    })
                    .collect::<Result<Vec<_>, CausalError>>()?;
                let (reports, notes, _) = validate_sequential(
                    data,
                    query,
                    &validation_atoms,
                    self.refute,
                    &self.custom_validators,
                    None,
                    None,
                    estimate.ate,
                    ctx,
                )?;
                (reports, notes)
            } else {
                run_envelope_effect_refuters(
                    &tabular,
                    &ate_query,
                    &estimate,
                    &refute_atoms,
                    &mut EstimationWorkspace::default(),
                    ctx,
                    self.refute,
                    "temporal.linear.adjustment",
                    &self.custom_validators,
                    Some(query),
                    self.split.as_ref(),
                    Some(data.time_index()),
                )?
            };
        diagnostics.push(Diagnostic::new(
            "estimate.dbn_posterior.frequentist",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "fixed graph weights; identified_mass={}; unidentified_mass={}; shared \
                 circular-block replicates={}; attempted={attempted}; bands require two successes and at most half failed attempts",
                point_mass,
                identified.graphs.unidentified_mass(),
                bootstrap.len()
            ),
        ));
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TemporalBackdoorUnfolded,
            estimator_id: if is_multi_step_sustained(query) {
                EstimatorId::TemporalSequentialGcomp
            } else {
                EstimatorId::TemporalLinearAdjustment
            },
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: (self.bootstrap_replicates > 0)
                .then_some(u32::try_from(bootstrap.len()).unwrap_or(u32::MAX)),
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "discover.dbn_posterior",
                    "dbn_posterior",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.dbn_posterior.frequentist",
                    "estimate.temporal.linear.adjustment",
                )),
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
    /// and fit with [`BayesianTemporalGcomp`] (Pulse / single-step Sustained) or
    /// sequential g-computation (multi-step Sustained; no last-step collapse).
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
        let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
        let multi_step = is_multi_step_sustained(query);
        if multi_step {
            if self.split.is_some() {
                return Err(CausalError::Unsupported {
                    message: "multi-step sustained g-computation currently requires no discovery-estimation split",
                });
            }
            if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "multi-step sustained inference requires isotropic per-mechanism priors",
                });
            }
        }

        let mut bayes = bayesian_temporal_gcomp(&cfg, ctx);
        let sequential_bayes = if multi_step { Some(bayesian_gcomp(&cfg, ctx)) } else { None };

        let (identified, identify_cached) =
            if let Some(cache) = self.dbn_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_dbn_posterior_identification_cache(
                        gp, &vars, query, ctx,
                    )?,
                    false,
                )
            };
        let keys = identified.graphs.graph_keys.to_vec();
        let mut flags = identified.graphs.identified.to_vec();
        let fit_atoms: Vec<_> = identified
            .atoms
            .iter()
            .map(|atom| {
                (atom.key, atom.estimand.clone(), atom.identification.clone(), atom.indexer.clone())
            })
            .collect();
        // The first structurally identified atom can still fail soft estimation
        // and be demoted below. Anchor the public result on the first atom that
        // actually contributes draws, not on a discarded fit.
        let mut atom_contexts = Vec::new();
        let mut atoms = Vec::new();
        let mut envelope_prior: Option<PriorSet> = None;
        let mut envelope_conflict: Option<antecedent_prob::ConflictSummary> = None;

        // Soft prepare+fit before Interactive subsample (0.6.0): demote failures
        // so stratified selection only chooses among atoms that already produced
        // draws. Prior anchors on the first successful prepare.
        let mut per_graph = Vec::new();
        let mut sequential_atoms = Vec::new();
        let mut ws = BayesianGCompWorkspace::default();
        let mut prepare_demoted = 0usize;
        let mut fit_demoted = 0usize;
        let mut draws_demoted = 0usize;
        for (key, estimand, identification, indexer) in fit_atoms {
            if multi_step {
                let Ok(graph) =
                    crate::analysis::prepared::temporal_dag_from_dbn_atom(gp, key, &vars)
                else {
                    if let Some(idx) = keys.iter().position(|&k| k == key) {
                        flags[idx] = GraphIdentFlag::Unidentified;
                    }
                    prepare_demoted += 1;
                    continue;
                };
                let mut assumptions = identification.required_assumptions.clone();
                assumptions.push(antecedent_core::AssumptionRecord {
                    assumption: antecedent_core::Assumption::ParametricRestriction(
                        antecedent_core::ParametricAssumption {
                            id: Arc::from("temporal.sequential.linear_sem"),
                            description: Arc::from(
                                "linear additive mechanisms on the identified unfolded DAG; \
                                 every sustained time is intervened on; Bayesian intervals share \
                                 each stationary mechanism posterior across time copies",
                            ),
                        },
                    ),
                    source: antecedent_core::AssumptionSource::AlgorithmDefault {
                        algorithm: Arc::from("temporal.sequential.gcomp"),
                    },
                    scope: antecedent_core::AssumptionScope::Estimation,
                    status: antecedent_core::AssumptionStatus::Declared,
                });
                let mut mechanisms = Vec::new();
                let Ok((atom_estimate, Some(posterior))) =
                    antecedent_estimate::temporal_sequential::estimate_sustained_window_with_validation(
                        data,
                        &graph,
                        &indexer,
                        &estimand,
                        query,
                        identification.status,
                        assumptions,
                        self.bootstrap_replicates,
                        sequential_bayes.as_ref(),
                        ctx,
                        Some(&mut mechanisms),
                    )
                else {
                    if let Some(idx) = keys.iter().position(|&k| k == key) {
                        flags[idx] = GraphIdentFlag::Unidentified;
                    }
                    fit_demoted += 1;
                    continue;
                };
                if let Ok(draws) = envelope_draws_from_posterior(key, &posterior) {
                    sequential_atoms.push(SequentialValidationAtom {
                        weight: identified_weight_for_key(&identified.graphs, key),
                        graph,
                        indexer: indexer.clone(),
                        estimand: estimand.clone(),
                        status: identification.status,
                        estimate: atom_estimate,
                        mechanisms,
                    });
                    atom_contexts.push((
                        key,
                        estimand.clone(),
                        identification.clone(),
                        indexer.clone(),
                    ));
                    per_graph.push(draws);
                } else {
                    if let Some(idx) = keys.iter().position(|&k| k == key) {
                        flags[idx] = GraphIdentFlag::Unidentified;
                    }
                    draws_demoted += 1;
                }
                continue;
            }
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
                prepare_demoted += 1;
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
                fit_demoted += 1;
                continue;
            };
            if let Ok(draws) = envelope_draws_from_posterior(key, &posterior) {
                atom_contexts.push((
                    key,
                    estimand.clone(),
                    identification.clone(),
                    indexer.clone(),
                ));
                per_graph.push(draws);
                atoms.push(EnvelopeAtomFit {
                    key,
                    prep: bprep,
                    posterior,
                    status: identification.status,
                    weight: identified_weight_for_key(&identified.graphs, key),
                    estimand,
                    indexer: Some(indexer),
                });
            } else {
                if let Some(idx) = keys.iter().position(|&k| k == key) {
                    flags[idx] = GraphIdentFlag::Unidentified;
                }
                draws_demoted += 1;
            }
        }

        let graphs = WeightedGraphSamples::new(
            Arc::clone(&identified.graphs.weights),
            flags,
            Arc::clone(&identified.graphs.graph_keys),
        )
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let mut subsample_notes = Vec::new();
        let (graphs, per_graph) = maybe_interactive_envelope_subsample(
            self.latency_mode,
            graphs,
            per_graph,
            ctx,
            &mut subsample_notes,
        )?;
        let keep = identified_envelope_keys(&graphs);
        atoms.retain(|atom| keep.contains(&atom.key));
        let (_, estimand, identification, _) = atom_contexts
            .into_iter()
            .find(|(key, _, _, _)| keep.contains(key))
            .ok_or_else(|| CausalError::Compile {
                message: "DBN posterior envelope has no contributing context".into(),
            })?;
        for atom in &mut atoms {
            atom.weight = identified_weight_for_key(&graphs, atom.key);
        }
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
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(subsample_notes);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(Diagnostic::new(
            "estimate.dbn_posterior.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", posterior.unidentified_mass),
        ));
        diagnostics.push(Diagnostic::new(
            "estimate.dbn_posterior.atom_demotion",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            identified.identify_demotion.summary(prepare_demoted, fit_demoted, draws_demoted),
        ));
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }
        if multi_step {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal.sustained_window",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "the contrast propagates through all intervened times in topological order; \
                 no one-node collapse; no analytic SE is asserted for the sequential fit",
            ));
        }

        let tabular = TabularData::new(data.storage().clone());
        let ate_query = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let (mut refutations, notes, sequential_predictive) = if multi_step {
            validate_sequential(
                data,
                query,
                &sequential_atoms,
                self.refute,
                &self.custom_validators,
                sequential_bayes.as_ref(),
                Some(&mut posterior),
                estimate.ate,
                ctx,
            )?
        } else {
            let (reports, notes) = run_envelope_effect_refuters(
                &tabular,
                &ate_query,
                &estimate,
                &envelope_refute_atoms(&atoms),
                &mut EstimationWorkspace::default(),
                ctx,
                self.refute,
                "bayesian.temporal.gcomp",
                &self.custom_validators,
                Some(query),
                self.split.as_ref(),
                Some(data.time_index()),
            )?;
            (reports, notes, Vec::new())
        };
        diagnostics.extend(notes);
        let predictive_checks = if multi_step {
            sequential_predictive
        } else {
            run_envelope_bayesian_full_validation(
                self.refute,
                &cfg,
                &bayes.inner,
                &atoms,
                &mut posterior,
                estimate.ate,
                ctx,
                &mut refutations,
                &mut diagnostics,
            )?
        };

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TemporalBackdoorUnfolded,
            estimator_id: if multi_step {
                EstimatorId::TemporalSequentialGcomp
            } else {
                EstimatorId::BayesianGcomp
            },
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
                identify_provenance: Some(provenance_ids(
                    "discover.dbn_posterior",
                    "dbn_posterior",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.aggregate_effect_envelope",
                    if multi_step {
                        "estimate.temporal.sequential.gcomp"
                    } else {
                        "estimate.bayesian.temporal.gcomp"
                    },
                )),
                posterior: Some(posterior),
                predictive_checks,
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }

    /// Mix a DBN posterior into a Bayesian temporal-mediation envelope.
    ///
    /// Each atom uses that atom's per-horizon `I(h)` cache. Unidentified atoms
    /// keep their mass. Priors do not upgrade identification.
    pub(super) fn execute_dbn_posterior_mediation(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use antecedent_estimate::bayesian_mediation::{
            compose_temporal_mediation, prepare_temporal_mediation_adjusted,
            require_gaussian_mediation,
        };

        if query.horizons.len() > 1 {
            let started = Instant::now();
            // Identify once for the complete request, then project the same
            // atom cache independently for each horizon (fresh and prepared).
            let cache = if let Some(cache) = &self.dbn_posterior_identification_cache {
                Arc::clone(cache)
            } else {
                let variables = data.schema().variables().iter().map(|v| v.id).collect::<Vec<_>>();
                Arc::new(
                    crate::analysis::prepared::build_dbn_posterior_mediation_identification_cache(
                        gp, &variables, query, ctx,
                    )?,
                )
            };
            let mut horizon_study = self.clone();
            horizon_study.dbn_posterior_identification_cache = Some(Arc::clone(&cache));

            let mut slices = Vec::with_capacity(query.horizons.len());
            let mut first = None;
            let mut diagnostics = vec![Diagnostic::new(
                "estimate.dbn_posterior.mediation_multi_horizon",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "each horizon is independently identified and graph-mixed; summaries are \
                 pointwise and do not define a joint horizon posterior",
            )];
            let mut refutations = Vec::new();
            let mut predictive_checks = Vec::new();
            for &horizon in query.horizons.iter() {
                let mut horizon_query = query.clone();
                horizon_query.horizons = Arc::from([horizon]);
                let horizon_cache = cache.mediation_horizon(horizon)?;
                if horizon_cache.atoms.is_empty() {
                    let notes = vec![
                        Diagnostic::new(
                            "estimate.dbn_posterior.envelope",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Warning,
                            "unidentified_mass=1",
                        ),
                        Diagnostic::new(
                            "estimate.dbn_posterior.atom_demotion",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Info,
                            horizon_cache.identify_demotion.summary(0, 0, 0),
                        ),
                    ];
                    slices.push(antecedent_estimate::TemporalMediationSlice {
                        horizon,
                        identification_status: IdentificationStatus::NotIdentified,
                        method: Arc::from("temporal_mediation.unidentified"),
                        adjustment: Arc::from([]),
                        estimate: TemporalMediationEstimate {
                            effect: nan_effect(),
                            total: None,
                            direct: None,
                            mediated: None,
                        },
                        uncertainty: antecedent_estimate::TemporalMediationUncertainty::Unavailable,
                        identified_set: None,
                        diagnostics: notes.clone(),
                    });
                    diagnostics.extend(notes);
                    continue;
                }
                let result = horizon_study.execute_dbn_posterior_mediation(
                    data,
                    gp,
                    &horizon_query,
                    physical,
                    ctx,
                )?;
                let slice = result
                    .mediation_grid
                    .as_ref()
                    .and_then(|grid| grid.slices.first())
                    .ok_or_else(|| CausalError::Compile {
                        message: "DBN mediation horizon omitted its grid slice".into(),
                    })?
                    .clone();
                slices.push(slice);
                for mut report in result.refutations {
                    report.refuter = Arc::from(format!("horizon.{horizon}.{}", report.refuter));
                    refutations.push(report);
                }
                predictive_checks.extend(result.predictive_checks);
                diagnostics.extend(result.diagnostics);
                if first.is_none() {
                    first = Some((result.identification, result.estimand));
                }
            }
            let (mut identification, estimand) = first.ok_or_else(|| CausalError::Compile {
                message: "DBN mediation requires at least one horizon".into(),
            })?;
            identification.status = most_conservative_identification_status(
                slices.iter().map(|slice| slice.identification_status),
            );
            diagnostics.push(Diagnostic::new(
                "identify.temporal_mediation.multi_horizon_status",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "parent identification.status is the most conservative requested-horizon \
                 status; mediation_grid is authoritative per horizon",
            ));
            return Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
                physical,
                identification,
                estimand,
                estimate: nan_effect(),
                identifier_id: IdentifierId::Frontdoor,
                estimator_id: EstimatorId::BayesianTemporalMediation,
                treatment: query.treatment,
                outcome: query.outcome,
                identify_cached: self.dbn_posterior_identification_cache.is_some(),
                extra_diagnostics: Vec::new(),
                refutations,
                distribution: None,
                mediation: None,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras {
                    mediation_grid: Some(antecedent_estimate::TemporalMediationGrid {
                        slices: Arc::from(slices),
                        joint_posterior: false,
                    }),
                    predictive_checks,
                    diagnostics: Some(diagnostics),
                    ..Default::default()
                },
            }));
        }
        let started = Instant::now();
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::Unsupported {
                    message: "DBN graph-posterior discovery requires inference=Bayesian for effect mixture",
                });
            }
        };
        if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
            return Err(CausalError::Unsupported {
                message: "Bayesian mediation currently supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
            });
        }
        let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
        let estimator = bayesian_gcomp(&cfg, ctx);
        require_gaussian_mediation(&estimator).map_err(CausalError::from)?;

        let (identified, identify_cached) =
            if let Some(cache) = self.dbn_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_dbn_posterior_mediation_identification_cache(
                        gp, &vars, query, ctx,
                    )?,
                    false,
                )
            };
        let horizon = query.horizons.first().copied().ok_or_else(|| CausalError::Compile {
            message: "DBN mediation requires at least one horizon".into(),
        })?;
        let identified = identified.mediation_horizon(horizon)?;
        let keys = identified.graphs.graph_keys.to_vec();
        let mut flags = identified.graphs.identified.to_vec();

        let mut atom_contexts = Vec::new();
        let mut per_graph = Vec::new();
        let mut refute_atoms = Vec::new();
        let mut prepare_demoted = 0usize;
        let mut fit_demoted = 0usize;
        let mut draws_demoted = 0usize;
        let mut distinct_z = false;
        let mut first_z: Option<Arc<[antecedent_data::LaggedColumn]>> = None;
        let mut horizon_dependent = false;

        for atom in identified.atoms.iter() {
            let Some(horizons) = atom.horizons.as_ref() else {
                if let Some(idx) = keys.iter().position(|&k| k == atom.key) {
                    flags[idx] = GraphIdentFlag::Unidentified;
                }
                prepare_demoted += 1;
                continue;
            };
            let Ok(clicks) =
                super::temporal_path::mediation_horizon_clicks(horizons, query, |horizon| {
                    horizons
                        .get(horizon)
                        .map(super::temporal_path::lagged_adjustment_from_entry)
                        .ok_or_else(|| CausalError::Compile {
                            message: format!(
                                "DBN mediation atom missing I({horizon}) for key {}",
                                atom.key
                            ),
                        })
                })
            else {
                if let Some(idx) = keys.iter().position(|&k| k == atom.key) {
                    flags[idx] = GraphIdentFlag::Unidentified;
                }
                prepare_demoted += 1;
                continue;
            };
            if clicks.is_empty() {
                if let Some(idx) = keys.iter().position(|&k| k == atom.key) {
                    flags[idx] = GraphIdentFlag::Unidentified;
                }
                prepare_demoted += 1;
                continue;
            }
            horizon_dependent |= super::temporal_path::mediation_horizon_z_differs(&clicks);
            let mut published = None;
            let mut click_ok = true;
            for click in &clicks {
                if require_identified(&click.identification).is_err() {
                    click_ok = false;
                    break;
                }
                let mut qh = query.clone();
                qh.horizons = Arc::from([click.horizon]);
                let Ok(preparations) = prepare_temporal_mediation_adjusted(
                    data,
                    &click.estimand,
                    &qh,
                    &click.adjustment,
                    ctx,
                ) else {
                    click_ok = false;
                    break;
                };
                if published.is_none() {
                    published = Some((click, qh, preparations));
                }
            }
            if !click_ok {
                if let Some(idx) = keys.iter().position(|&k| k == atom.key) {
                    flags[idx] = GraphIdentFlag::Unidentified;
                }
                prepare_demoted += 1;
                continue;
            }
            let (click, qh, preparations) = published.expect("non-empty clicks");
            match first_z.as_ref() {
                Some(z) if z.as_ref() != click.adjustment.as_ref() => distinct_z = true,
                None => first_z = Some(Arc::clone(&click.adjustment)),
                _ => {}
            }
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
                            click.identification.status,
                            &mut BayesianGCompWorkspace::default(),
                            ctx,
                        )
                        .map_err(CausalError::from)
                    })
                    .collect()
            };
            let Ok(mechanisms) = fit(cfg.prior_scale) else {
                if let Some(idx) = keys.iter().position(|&k| k == atom.key) {
                    flags[idx] = GraphIdentFlag::Unidentified;
                }
                fit_demoted += 1;
                continue;
            };
            let Ok(composed) = compose_temporal_mediation(
                &mechanisms[0],
                &mechanisms[1],
                &qh,
                click.identification.status,
            ) else {
                if let Some(idx) = keys.iter().position(|&k| k == atom.key) {
                    flags[idx] = GraphIdentFlag::Unidentified;
                }
                fit_demoted += 1;
                continue;
            };
            if let Ok(draws) = envelope_draws_from_posterior(atom.key, &composed) {
                atom_contexts.push((
                    atom.key,
                    click.estimand.clone(),
                    click.identification.clone(),
                    click.indexer.clone(),
                ));
                per_graph.push(draws);
                let weight = identified_weight_for_key(&identified.graphs, atom.key);
                refute_atoms.push(DbnMediationAtom {
                    key: atom.key,
                    weight,
                    estimand: click.estimand.clone(),
                    adjustment: Arc::clone(&click.adjustment),
                    query: qh,
                    preparations,
                    mechanisms,
                    composed,
                });
            } else {
                if let Some(idx) = keys.iter().position(|&k| k == atom.key) {
                    flags[idx] = GraphIdentFlag::Unidentified;
                }
                draws_demoted += 1;
            }
        }

        let graphs = WeightedGraphSamples::new(
            Arc::clone(&identified.graphs.weights),
            flags,
            Arc::clone(&identified.graphs.graph_keys),
        )
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let mut subsample_notes = Vec::new();
        let (graphs, per_graph) = maybe_interactive_envelope_subsample(
            self.latency_mode,
            graphs,
            per_graph,
            ctx,
            &mut subsample_notes,
        )?;
        let keep = identified_envelope_keys(&graphs);
        refute_atoms.retain(|atom| keep.contains(&atom.key));
        let (_, estimand, mut identification, indexer) = atom_contexts
            .into_iter()
            .find(|(key, _, _, _)| keep.contains(key))
            .ok_or_else(|| CausalError::Compile {
                message: "DBN posterior envelope has no contributing context".into(),
            })?;
        for atom in &mut refute_atoms {
            atom.weight = identified_weight_for_key(&graphs, atom.key);
        }
        let mut posterior = aggregate_effect_envelope(
            &graphs,
            &per_graph,
            InferenceDiagnostics::analytic("dbn_posterior_envelope"),
            EnvelopeOptions::default(),
        )
        .map_err(CausalError::from)?;
        // The envelope must preserve structural restrictions and disclose this
        // horizon's unidentified graph mass, independently of other horizons.
        if posterior.unidentified_mass > 0.0 {
            identification.status = IdentificationStatus::GraphDependent;
        }
        posterior.identification = identification.status;
        let estimate = effect_from_posterior(&posterior)?;
        let mut mediation = mix_dbn_mediation_estimate(&estimate, &refute_atoms);
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(subsample_notes);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(Diagnostic::new(
            "estimate.dbn_posterior.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", posterior.unidentified_mass),
        ));
        diagnostics.push(Diagnostic::new(
            "estimate.dbn_posterior.atom_demotion",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            identified.identify_demotion.summary(prepare_demoted, fit_demoted, draws_demoted),
        ));
        diagnostics.push(Diagnostic::new(
            "identify.dbn_posterior.per_atom_horizon",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "each graph atom uses that atom's I(h); adjustment sets are not unioned across atoms",
        ));
        if horizon_dependent {
            diagnostics.push(Diagnostic::new(
                "identify.temporal_mediation.horizon_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "adjustment sets differ across requested horizons; each contrast uses I(h) \
                 identified for that horizon, not a shared max-horizon set",
            ));
        }
        if distinct_z {
            diagnostics.push(Diagnostic::new(
                "identify.dbn_posterior.atom_horizon_sets_differ",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "contributing DBN atoms have distinct I(h) adjustment sets at the published horizon",
            ));
        }
        diagnostics.push(Diagnostic::new(
            "estimate.mediation.bayesian",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "independent Gaussian mediator and outcome mechanisms; total = direct + mediated for every posterior draw; natural effects use the linear no-interaction alias",
        ));

        let mut refutations = Vec::new();
        if self.refute != RefuteSuite::None {
            let (reports, notes) = mix_dbn_mediation_refuters(
                data,
                &mediation,
                &refute_atoms,
                self.refute == RefuteSuite::Full,
                ctx,
            )?;
            refutations.extend(reports);
            diagnostics.extend(notes);
        }
        let predictive_checks = run_dbn_mediation_bayesian_validation(
            self.refute,
            &cfg,
            &estimator,
            &refute_atoms,
            &mut posterior,
            estimate.ate,
            ctx,
            &mut refutations,
            &mut diagnostics,
        )?;
        let requested_summary = antecedent_estimate::MediationPosteriorSummary {
            mean: posterior.summaries.mean[0],
            standard_deviation: posterior.summaries.sd[0],
            q025: posterior.summaries.q025[0],
            q975: posterior.summaries.q975[0],
        };
        let total_summary = dbn_mediation_component_summary(&graphs, &refute_atoms, 1)?;
        let direct_summary = dbn_mediation_component_summary(&graphs, &refute_atoms, 2)?;
        let mediated_summary = dbn_mediation_component_summary(&graphs, &refute_atoms, 3)?;
        mediation.total = Some(total_summary.mean);
        mediation.direct = Some(direct_summary.mean);
        mediation.mediated = Some(mediated_summary.mean);
        let mut adjustment = estimand
            .adjustment_set
            .iter()
            .filter_map(|id| indexer.key_of(id.raw()).ok())
            .collect::<Vec<_>>();
        adjustment.sort();
        let mediation_grid = antecedent_estimate::TemporalMediationGrid {
            slices: Arc::from([antecedent_estimate::TemporalMediationSlice {
                horizon: query.horizons[0],
                identification_status: identification.status,
                method: Arc::clone(&estimand.method),
                adjustment: Arc::from(adjustment),
                estimate: mediation.clone(),
                uncertainty: antecedent_estimate::TemporalMediationUncertainty::BayesianPointwise {
                    requested: requested_summary,
                    total: total_summary,
                    direct: direct_summary,
                    mediated: mediated_summary,
                    n_draws: posterior.draws.n_draws,
                    backend: Arc::clone(&posterior.diagnostics.backend_id),
                },
                identified_set: None,
                diagnostics: diagnostics.clone(),
            }]),
            joint_posterior: false,
        };
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification: identification.clone(),
            estimand,
            estimate,
            identifier_id: IdentifierId::Frontdoor,
            estimator_id: EstimatorId::BayesianTemporalMediation,
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
                    "discover.dbn_posterior",
                    "dbn_posterior",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.aggregate_effect_envelope",
                    "estimate.bayesian_temporal_mediation",
                )),
                posterior: Some(posterior),
                mediation_grid: Some(mediation_grid),
                predictive_checks,
                diagnostics: Some(diagnostics),
                certificate: Some(crate::Identification::Point {
                    result: identification,
                    temporal_indexer: Some(indexer),
                    strategy: IdentifierId::Frontdoor,
                    structure_version: self.graph.version(),
                }),
                ..Default::default()
            },
        }))
    }
}

fn fit_frequentist_dbn_atom(
    data: &TimeSeriesData,
    graph_posterior: &GraphPosterior,
    variables: &[VariableId],
    atom: &crate::analysis::prepared::CachedDbnPosteriorAtomIdentification,
    query: &TemporalEffectQuery,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
    if is_multi_step_sustained(query) {
        let graph = crate::analysis::prepared::temporal_dag_from_dbn_atom(
            graph_posterior,
            atom.key,
            variables,
        )?;
        let mut assumptions = atom.identification.required_assumptions.clone();
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::ParametricRestriction(
                antecedent_core::ParametricAssumption {
                    id: Arc::from("temporal.sequential.linear_sem"),
                    description: Arc::from(
                        "linear additive mechanisms on each identified DBN atom; fixed posterior \
                         graph weights; uncertainty uses shared outer circular-block replicates",
                    ),
                },
            ),
            source: antecedent_core::AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("temporal.sequential.gcomp"),
            },
            scope: antecedent_core::AssumptionScope::Estimation,
            status: antecedent_core::AssumptionStatus::Declared,
        });
        return antecedent_estimate::temporal_sequential::estimate_sustained_window(
            data,
            &graph,
            &atom.indexer,
            &atom.estimand,
            query,
            atom.identification.status,
            assumptions,
            0,
            None,
            ctx,
        )
        .map(|(estimate, _)| estimate)
        .map_err(CausalError::from);
    }
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.bootstrap_replicates = 0;
    estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
    let prepared = estimator
        .prepare(data, &atom.estimand, query, &atom.indexer, None, &ctx.kernel_policy)
        .map_err(CausalError::from)?;
    estimator
        .fit(
            &prepared,
            &mut EstimationWorkspace::default(),
            ctx,
            atom.identification.required_assumptions.clone(),
        )
        .map_err(CausalError::from)
}

struct DbnMediationAtom {
    key: u64,
    weight: f64,
    estimand: IdentifiedEstimand,
    adjustment: Arc<[antecedent_data::LaggedColumn]>,
    query: antecedent_core::MediationQuery,
    preparations: [PreparedBayesianProblem; 2],
    mechanisms: Vec<CausalPosterior>,
    composed: CausalPosterior,
}

fn dbn_mediation_component_summary(
    graphs: &WeightedGraphSamples,
    atoms: &[DbnMediationAtom],
    quantity: usize,
) -> Result<antecedent_estimate::MediationPosteriorSummary, CausalError> {
    let per_graph = atoms
        .iter()
        .map(|atom| {
            let draws = atom
                .composed
                .draws
                .column(quantity)
                .map_err(|error| CausalError::Compile { message: error.to_string() })?;
            Ok(GraphEffectDraws { graph_key: atom.key, effect_draws: Arc::from(draws.to_vec()) })
        })
        .collect::<Result<Vec<_>, CausalError>>()?;
    let posterior = aggregate_effect_envelope(
        graphs,
        &per_graph,
        InferenceDiagnostics::analytic("dbn_mediation_component"),
        EnvelopeOptions::default(),
    )
    .map_err(CausalError::from)?;
    Ok(antecedent_estimate::MediationPosteriorSummary {
        mean: posterior.summaries.mean[0],
        standard_deviation: posterior.summaries.sd[0],
        q025: posterior.summaries.q025[0],
        q975: posterior.summaries.q975[0],
    })
}

fn mix_dbn_mediation_estimate(
    estimate: &EffectEstimate,
    atoms: &[DbnMediationAtom],
) -> TemporalMediationEstimate {
    let mut total = 0.0;
    let mut direct = 0.0;
    let mut mediated = 0.0;
    let mut w = 0.0;
    for atom in atoms {
        if atom.weight <= 0.0 || atom.composed.summaries.mean.len() < 4 {
            continue;
        }
        w += atom.weight;
        total += atom.weight * atom.composed.summaries.mean[1];
        direct += atom.weight * atom.composed.summaries.mean[2];
        mediated += atom.weight * atom.composed.summaries.mean[3];
    }
    TemporalMediationEstimate {
        effect: estimate.clone(),
        total: (w > 0.0).then_some(total / w),
        direct: (w > 0.0).then_some(direct / w),
        mediated: (w > 0.0).then_some(mediated / w),
    }
}

fn mix_dbn_mediation_refuters(
    data: &TimeSeriesData,
    mediation: &TemporalMediationEstimate,
    atoms: &[DbnMediationAtom],
    full: bool,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let mut order = Vec::new();
    let mut by_refuter: std::collections::HashMap<
        Arc<str>,
        Vec<(f64, antecedent_validate::RefutationReport)>,
    > = std::collections::HashMap::new();
    for atom in atoms {
        if atom.weight <= 0.0 {
            continue;
        }
        let reports = antecedent_validate::mediation::refute_temporal_mediation_adjusted(
            data,
            &atom.estimand,
            &atom.query,
            mediation,
            full,
            &atom.adjustment,
            ctx,
        )
        .map_err(CausalError::from)?;
        for report in reports {
            let bucket = by_refuter.entry(Arc::clone(&report.refuter)).or_insert_with(|| {
                order.push(Arc::clone(&report.refuter));
                Vec::new()
            });
            bucket.push((atom.weight, report));
        }
    }
    let mut mixed = Vec::with_capacity(order.len());
    for id in order {
        let Some(items) = by_refuter.get(&id) else {
            continue;
        };
        let borrowed: Vec<(f64, &antecedent_validate::RefutationReport)> =
            items.iter().map(|(w, r)| (*w, r)).collect();
        if let Some(report) = antecedent_validate::RefutationReport::mixture_weighted(&borrowed) {
            mixed.push(report);
        }
    }
    let atom_keys: String =
        atoms.iter().map(|atom| format!("{:x}", atom.key)).collect::<Vec<_>>().join(",");
    Ok((
        mixed,
        vec![Diagnostic::new(
            "refute.envelope.effect_mixture",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "mediation refuters evaluated each contributing graph atom [{atom_keys}] against \
                 the mixture effect using that atom's I(h); reports mix by envelope mass"
            ),
        )],
    ))
}

fn run_dbn_mediation_bayesian_validation(
    refute: RefuteSuite,
    cfg: &crate::inference::BayesianConfig,
    estimator: &BayesianGComputationAte,
    atoms: &[DbnMediationAtom],
    mixture_posterior: &mut CausalPosterior,
    estimate_ate: f64,
    ctx: &ExecutionContext,
    refutations: &mut Vec<antecedent_validate::RefutationReport>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<Vec<PredictiveCheckReport>, CausalError> {
    const PPC_ALPHA: f64 = 0.05;
    if matches!(refute, RefuteSuite::None) || atoms.is_empty() {
        return Ok(Vec::new());
    }
    let mut predictive_checks = Vec::new();
    let mut prior_items = Vec::new();
    let mut post_items = Vec::new();
    for atom in atoms {
        for (i, (prep, post)) in atom.preparations.iter().zip(&atom.mechanisms).enumerate() {
            let prior = estimator.prior.clone().unwrap_or_else(|| {
                let mut prior = PriorSet::weakly_informative(prep.design.ncols);
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
            prior_items.push((atom.weight, i, prior_rep));
            post_items.push((atom.weight, i, post_rep));
        }
    }
    for (label, items) in [("prior", &prior_items), ("posterior", &post_items)] {
        let _ = label;
        let borrowed: Vec<(f64, &PredictiveCheckReport)> =
            items.iter().map(|(w, _, r)| (*w, r)).collect();
        if let Some(mixed) = PredictiveCheckReport::mixture_weighted(&borrowed) {
            refutations.push(mixed.to_refutation_report(estimate_ate, PPC_ALPHA));
            predictive_checks.push(mixed);
        }
    }
    if matches!(refute, RefuteSuite::Full) {
        let sensitivity = antecedent_validate::PriorSensitivity::standard_grid();
        let mut owned = Vec::new();
        for atom in atoms {
            let mut means = Vec::new();
            let mut sds = Vec::new();
            for &scale in sensitivity.scales.iter() {
                let posts: Result<Vec<_>, _> = atom
                    .preparations
                    .iter()
                    .enumerate()
                    .map(|(i, prep)| {
                        let mut est = estimator.clone();
                        est.prior_scale = scale;
                        est.seed = est.seed.wrapping_add(if i == 0 { 0 } else { 0xBA71_u64 });
                        est.fit(
                            prep,
                            atom.composed.identification,
                            &mut BayesianGCompWorkspace::default(),
                            ctx,
                        )
                        .map_err(CausalError::from)
                    })
                    .collect();
                let posts = posts?;
                let post = compose_temporal_mediation_for_atom(atom, &posts)?;
                means.push(post.summaries.mean[0]);
                sds.push(post.summaries.sd[0]);
            }
            owned.push((
                atom.weight,
                antecedent_prob::PriorSensitivitySummary {
                    prior_scales: sensitivity.scales.clone(),
                    alphas: Arc::from([]),
                    effect_means: Arc::from(means),
                    effect_sds: Arc::from(sds),
                },
            ));
        }
        let items: Vec<_> = owned.iter().map(|(w, s)| (*w, s)).collect();
        if let Some(mixed) = mix_prior_sensitivity_summaries(&items) {
            refutations.push(sensitivity.to_report(&mixed, estimate_ate));
            *mixture_posterior = with_prior_sensitivity(mixture_posterior.clone(), mixed);
        }
    }
    let atom_keys: String =
        atoms.iter().map(|atom| format!("{:x}", atom.key)).collect::<Vec<_>>().join(",");
    diagnostics.push(Diagnostic::new(
        "refute.bayesian.ppc.envelope",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "mechanism PPC evaluated per identified mediation atom [{atom_keys}]; reports mix \
             by graph posterior mass"
        ),
    ));
    Ok(predictive_checks)
}

fn compose_temporal_mediation_for_atom(
    atom: &DbnMediationAtom,
    posts: &[CausalPosterior],
) -> Result<CausalPosterior, CausalError> {
    use antecedent_estimate::bayesian_mediation::compose_temporal_mediation;
    compose_temporal_mediation(&posts[0], &posts[1], &atom.query, atom.composed.identification)
        .map_err(CausalError::from)
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
