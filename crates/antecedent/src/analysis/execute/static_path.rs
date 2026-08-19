// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::estimator_spec::EstimatorSpec;

impl super::Study {
    pub(super) fn execute_static(
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
        let estimator = physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;

        // rd.sharp has no graph-based identification step; dispatch to its
        // own path before touching `graph`.
        if matches!(estimator_id, EstimatorId::RdSharp) {
            return self.execute_rd(data, query, physical, ctx);
        }
        if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            return self.execute_bayesian(data, graph, query, physical, ctx);
        }

        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query, rd), all frozen there, so reuse is
        // exact and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(self.identification_cache.as_deref(), || {
                let rd =
                    self.rd.map(|c| SharpRdConfig::new(c.running_variable, c.cutoff, c.bandwidth));
                let identification = identify_static_query_with_rd(
                    identifier_id,
                    graph,
                    &CausalQuery::AverageEffect(query.clone()),
                    rd,
                )?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;
        let assumptions = identification.required_assumptions.clone();
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

        // Point estimate first (no bootstrap); uncertainty stage fills SE separately.
        clock.begin(ctx, super::super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let mut estimate_ws = StaticEstimateWorkspaces::default();
        // A caller-configured estimator wins; otherwise select by id and let the
        // study fill bootstrap/overlap defaults. The builder refuses the ambiguous
        // case (both set) at `build()` time, so there is nothing to reconcile here.
        let estimator_spec =
            self.estimator_spec.clone().unwrap_or(EstimatorSpec::Default(estimator_id));
        let point = estimate_static_effect(
            &estimator_spec,
            &data_est,
            &estimand_est,
            &query_est,
            assumptions.clone(),
            0, // point stage: no bootstrap
            self.overlap_policy,
            self.population_registry.as_ref(),
            ctx,
            &mut estimate_ws,
        )?;
        clock.finish(super::super::stage::STAGE_ESTIMATE_POINT);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Point { estimate: point.clone() },
        );

        // Uncertainty: bootstrap fills (real work when replicates > 0).
        let estimate = if self.bootstrap_replicates == 0 {
            if ctx.cancellation.is_cancelled() {
                clock.mark_cancelled();
                point
            } else {
                clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
                clock.finish(super::super::stage::STAGE_UNCERTAINTY);
                super::super::stage::emit_stage(
                    self.stage_sink.as_ref(),
                    &super::super::stage::StageEvent::Uncertainty { estimate: point.clone() },
                );
                point
            }
        } else if matches!(estimator_id, EstimatorId::LinearAdjustmentAte) {
            // Reuse warmed OLS workspace: re-prepare + attach bootstrap without refitting point.
            let cancelled_before = ctx.cancellation.is_cancelled();
            if cancelled_before {
                clock.mark_cancelled();
                if let Some(p) = &ctx.progress {
                    p.report(0.55, super::super::stage::STAGE_UNCERTAINTY);
                }
                point
            } else {
                clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
                // Reuse the caller's configured estimator when there is one, so the
                // warm-workspace bootstrap path cannot silently diverge from the
                // point-estimate path above.
                let est = if let EstimatorSpec::LinearAdjustmentAte(cfg) = &estimator_spec {
                    (**cfg).clone()
                } else {
                    let mut est = LinearAdjustmentAte::new();
                    est.bootstrap_replicates = self.bootstrap_replicates;
                    est.overlap = OverlapPolicy::ExplicitOverride;
                    est
                };
                let prep =
                    est.prepare(&data_est, &estimand_est, &query_est).map_err(CausalError::from)?;
                let filled = est
                    .attach_bootstrap(&prep, &mut estimate_ws.linear, ctx, point)
                    .map_err(CausalError::from)?;
                let cancelled = filled.bootstrap_cancelled || ctx.cancellation.is_cancelled();
                if cancelled {
                    clock.mark_cancelled();
                } else {
                    clock.finish(super::super::stage::STAGE_UNCERTAINTY);
                }
                super::super::stage::emit_stage(
                    self.stage_sink.as_ref(),
                    &super::super::stage::StageEvent::Uncertainty { estimate: filled.clone() },
                );
                filled
            }
        } else {
            // Non-linear static estimators: re-run with bootstrap for uncertainty fills.
            let cancelled_before = ctx.cancellation.is_cancelled();
            if cancelled_before {
                clock.mark_cancelled();
                if let Some(p) = &ctx.progress {
                    p.report(0.55, super::super::stage::STAGE_UNCERTAINTY);
                }
                point
            } else {
                clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
                let filled = estimate_static_effect(
                    &estimator_spec,
                    &data_est,
                    &estimand_est,
                    &query_est,
                    assumptions,
                    self.bootstrap_replicates,
                    self.overlap_policy,
                    self.population_registry.as_ref(),
                    ctx,
                    &mut estimate_ws,
                )?;
                let cancelled = filled.bootstrap_cancelled || ctx.cancellation.is_cancelled();
                if cancelled {
                    clock.mark_cancelled();
                } else {
                    clock.finish(super::super::stage::STAGE_UNCERTAINTY);
                }
                super::super::stage::emit_stage(
                    self.stage_sink.as_ref(),
                    &super::super::stage::StageEvent::Uncertainty { estimate: filled.clone() },
                );
                filled
            }
        };

        let cancelled = estimate.bootstrap_cancelled || clock.cancelled();

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        if identify_cached {
            diagnostics.push(Diagnostic::new(
                "exec.identify.cached",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Info,
                "identification reused from the prepare-time cache".to_string(),
            ));
        }
        if let Some(d) = projection_diagnostic(full_cols, projected_cols) {
            diagnostics.push(d);
        }

        let refutations = if cancelled {
            Vec::new()
        } else {
            clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
            let prop_scratch = match estimator_id {
                EstimatorId::Aipw => &mut estimate_ws.aipw.propensity,
                _ => &mut estimate_ws.propensity.propensity,
            };
            let reports = run_refuters(
                &data_est,
                &estimand_est,
                &query_est,
                &estimate,
                &mut estimate_ws.linear,
                Some(prop_scratch),
                ctx,
                self.refute,
                estimator,
                &self.custom_validators,
                None,
            )?;
            clock.finish(super::super::stage::STAGE_VALIDATE);
            super::super::stage::emit_stage(
                self.stage_sink.as_ref(),
                &super::super::stage::StageEvent::Validate {
                    refutations: reports.clone(),
                    predictive_checks: Vec::new(),
                },
            );
            reports
        };

        let (id_artifact, id_op) = identify_provenance_step(identifier_id);
        let (est_artifact, est_op) = estimate_provenance_step(estimator_id);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let early_stopped = estimate.bootstrap_early_stopped;
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
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
            wall_time_ns: clock.wall_time_ns(),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: clock.timings(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: bootstrap_ok,
            n_draws: None,
            cancelled: clock.cancelled(),
            early_stopped,
        }))
    }

    /// Identify + plug-in estimate for an interventional distribution.
    pub(super) fn execute_distribution(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::InterventionalDistributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(DEFAULT_DISTRIBUTION_IDENTIFIER);
        let estimator =
            physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_DISTRIBUTION_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if !matches!(estimator_id, EstimatorId::FunctionalDistribution) {
            return Err(CausalError::Compile {
                message: format!(
                    "Distribution execute requires estimator functional.distribution; got {estimator}"
                ),
            });
        }

        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query), all frozen there, so reuse is exact
        // and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(self.identification_cache.as_deref(), || {
                let cq = CausalQuery::Distribution(query.clone());
                let identification = identify_static_query(identifier_id, graph, &cq)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;

        let est = FunctionalDistribution {
            bootstrap_replicates: self.bootstrap_replicates,
            ..FunctionalDistribution::new()
        };
        let prepared = est
            .prepare(
                data,
                query,
                &estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
            )
            .map_err(CausalError::from)?;
        let mut ws = FunctionalDistributionWorkspace::default();
        let dist = est.estimate(&prepared, &[], &mut ws, ctx).map_err(CausalError::from)?;

        let estimate = EffectEstimate::from_parts(
            dist.mean,
            dist.se_analytic,
            dist.se_bootstrap,
            dist.bootstrap_replicates_ok,
            dist.bootstrap_replicates_failed,
            dist.bootstrap_cancelled,
            dist.bootstrap_early_stopped,
            dist.assumptions.clone(),
            dist.overlap,
            None,
            dist.retained_memory_bytes,
        );

        let treatment =
            query.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
                || CausalError::Compile {
                    message: "distribution query missing intervention target".into(),
                },
            )?;
        let outcome = *query.outcomes.first().ok_or_else(|| CausalError::Compile {
            message: "distribution query missing outcome".into(),
        })?;
        let early_stopped = estimate.bootstrap_early_stopped;
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let cancelled = estimate.bootstrap_cancelled;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        if identify_cached {
            diagnostics.push(Diagnostic::new(
                "exec.identify.cached",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Info,
                "identification reused from the prepare-time cache".to_string(),
            ));
        }

        let mut refute_ws = EstimationWorkspace::default();
        let ate_q = AverageEffectQuery::binary_ate(treatment, outcome);
        let refutations = if estimate.ate.is_finite() {
            run_refuters(
                data,
                &estimand,
                &ate_q,
                &estimate,
                &mut refute_ws,
                None,
                ctx,
                self.refute,
                estimator,
                &self.custom_validators,
                None,
            )?
        } else {
            diagnostics.push(Diagnostic::new(
                "refute.distribution.skipped",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "effect refuters skipped: interventional mean is not a finite scalar",
            ));
            Vec::new()
        };

        let (id_artifact, id_op) = identify_provenance_step(identifier_id);
        let (est_artifact, est_op) = estimate_provenance_step(estimator_id);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: Some(dist),
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment,
            outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: bootstrap_ok,
            n_draws: None,
            cancelled,
            early_stopped,
        }))
    }

    /// Identify + plug-in estimate for a path-specific natural effect.
    pub(super) fn execute_path_specific(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::PathSpecificEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_PATH_IDENTIFIER);
        let estimator =
            physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_PATH_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "PathSpecific execute requires estimator functional.effect; got {estimator}"
                ),
            });
        }

        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query), all frozen there, so reuse is exact
        // and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(self.identification_cache.as_deref(), || {
                let cq = CausalQuery::PathSpecific(query.clone());
                let identification = identify_static_query(identifier_id, graph, &cq)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;

        let mut extra = vec![query.treatment, query.outcome];
        extra.extend(query.path_nodes.iter().copied());
        let est = FunctionalEffect {
            bootstrap_replicates: self.bootstrap_replicates,
            ..FunctionalEffect::new()
        };
        let prepared = est
            .prepare(
                data,
                &estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
                &extra,
            )
            .map_err(CausalError::from)?;
        let mut ws = FunctionalDistributionWorkspace::default();
        let estimate = est.estimate(&prepared, &mut ws, ctx).map_err(CausalError::from)?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        if identify_cached {
            diagnostics.push(Diagnostic::new(
                "exec.identify.cached",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Info,
                "identification reused from the prepare-time cache".to_string(),
            ));
        }

        let mut refute_ws = EstimationWorkspace::default();
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let refutations = run_refuters(
            data,
            &estimand,
            &ate_q,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
            None,
        )?;

        let (id_artifact, id_op) = identify_provenance_step(identifier_id);
        let (est_artifact, est_op) = estimate_provenance_step(estimator_id);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
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

    /// Bayesian g-computation execute path.
    pub(super) fn execute_rd(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let rd = self.rd.ok_or_else(|| CausalError::Compile {
            message: "estimator \"rd.sharp\" requires builder.rd_config(running_variable, cutoff, bandwidth)".into(),
        })?;
        let identification = SharpRdIdentifier::new(SharpRdConfig::new(
            rd.running_variable,
            rd.cutoff,
            rd.bandwidth,
        ))
        .identify(CausalQuery::AverageEffect(query.clone()))
        .map_err(CausalError::from)?;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::RdSharp)?;

        let mut est =
            SharpRegressionDiscontinuity::new(rd.running_variable, rd.cutoff, rd.bandwidth);
        est.bootstrap_replicates = self.bootstrap_replicates;
        let prep = est.prepare(data, &estimand, query).map_err(CausalError::from)?;
        let mut ws = RdWorkspace::default();
        let estimate = est
            .fit(&prep, &mut ws, ctx, identification.required_assumptions.clone())
            .map_err(CausalError::from)?;

        let mut diagnostics = vec![overlap_diagnostic(estimate.overlap)];
        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            "rd.sharp",
            &self.custom_validators,
            None,
        )?;

        let provenance = provenance_pair(
            ("identify.rd_design", "identify.rd_sharp", &[], &identification.required_assumptions),
            ("estimate.rd", "estimate.rd_sharp", &["identify.rd_design"], &estimate.assumptions),
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

    pub(super) fn execute_conditional(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier, _) = self.resolve_conditional_pair();
        let identifier_id: IdentifierId = identifier.parse()?;
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query.inner), all frozen there, so reuse is
        // exact and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(self.identification_cache.as_deref(), || {
                let identification = identify_static(identifier_id, graph, &query.inner)?;
                let estimand =
                    select_estimand(&identification, EstimatorId::ConditionalLinearAdjustment)?;
                Ok((identification, estimand))
            })?;
        let est = ConditionalLinearAdjustment::new();
        let estimate = est.estimate(data, &estimand, query, ctx).map_err(CausalError::from)?;
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        if identify_cached {
            diagnostics.push(Diagnostic::new(
                "exec.identify.cached",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Info,
                "identification reused from the prepare-time cache".to_string(),
            ));
        }
        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            data,
            &estimand,
            &query.inner,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            "conditional.linear.adjustment",
            &self.custom_validators,
            None,
        )?;
        let (id_artifact, id_op) = identify_provenance_step(identifier_id);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (
                "estimate.conditional_linear",
                "estimate.conditional_linear_adjustment",
                &[id_artifact],
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
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.inner.treatment,
            outcome: query.inner.outcome,
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

    pub(super) fn execute_static_mediation_total(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if !matches!(query.contrast, MediationContrast::Total) {
            return Err(CausalError::Unsupported {
                message: "static Mediation supports only MediationContrast::Total via front-door",
            });
        }
        let ate = AverageEffectQuery::new(
            query.treatment,
            query.outcome,
            Arc::from([]),
            query.control.clone(),
            query.active.clone(),
            query.target_population.clone(),
        );
        let identification = identify_static(IdentifierId::Frontdoor, graph, &ate)?;
        let estimand = select_estimand(&identification, EstimatorId::FrontDoorTwoStage)?;
        let mut estimate_ws = StaticEstimateWorkspaces::default();
        let estimate = estimate_static_effect(
            &EstimatorSpec::Default(EstimatorId::FrontDoorTwoStage),
            data,
            &estimand,
            &ate,
            identification.required_assumptions.clone(),
            self.bootstrap_replicates,
            self.overlap_policy,
            self.population_registry.as_ref(),
            ctx,
            &mut estimate_ws,
        )?;
        let mediation = TemporalMediationEstimate {
            effect: estimate.clone(),
            total: Some(estimate.ate),
            direct: None,
            mediated: None,
        };
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        let refutations = run_refuters(
            data,
            &estimand,
            &ate,
            &estimate,
            &mut estimate_ws.linear,
            None,
            ctx,
            self.refute,
            "frontdoor.two_stage",
            &self.custom_validators,
            None,
        )?;
        let provenance = provenance_pair(
            ("identify.frontdoor", "identify.frontdoor", &[], &identification.required_assumptions),
            (
                "estimate.frontdoor",
                "estimate.frontdoor_two_stage",
                &["identify.frontdoor"],
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
