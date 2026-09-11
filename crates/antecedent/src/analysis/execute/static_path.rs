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
        if let Some(background) = self.tiered.clone() {
            return self.execute_tiered_average(
                data,
                query,
                physical,
                ctx,
                &background,
                identifier_id,
                estimator_id,
            );
        }

        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query, rd), all frozen there, so reuse is
        // exact and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
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
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            &data_est,
            query_est.outcome,
            &query_est.outcome_functional,
        )?;
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

        let (refutations, na_diagnostics) = if cancelled {
            (Vec::new(), Vec::new())
        } else {
            clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
            let prop_scratch = match estimator_id {
                EstimatorId::Aipw => &mut estimate_ws.aipw.propensity,
                _ => &mut estimate_ws.propensity.propensity,
            };
            let (reports, na_diagnostics) = run_refuters(
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
            (reports, na_diagnostics)
        };

        let mut extra_diagnostics =
            if let Some(d) = projection_diagnostic(full_cols, projected_cols) {
                vec![d]
            } else {
                Vec::new()
            };
        extra_diagnostics.extend(na_diagnostics);
        let estimate = super::super::helpers::attach_average_functional_grid(
            estimate,
            data,
            query,
            &estimand,
            &mut extra_diagnostics,
            estimator_id,
            self,
        )?;
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let early_stopped = estimate.bootstrap_early_stopped;
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
            bootstrap_replicates_ok: bootstrap_ok,
            cancelled: clock.cancelled(),
            early_stopped,
            extras: IdentifiedExecuteExtras {
                stage_timings_ns: clock.timings(),
                ..Default::default()
            },
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
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
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
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let cancelled = estimate.bootstrap_cancelled;
        let early_stopped = estimate.bootstrap_early_stopped;

        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_distribution(
                data,
                query,
                &identification,
                &estimand,
                &dist,
                self.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?
        };

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: Some(dist),
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: bootstrap_ok,
            cancelled,
            early_stopped,
            extras: IdentifiedExecuteExtras::default(),
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
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
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

        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_path(
                data,
                query,
                &identification,
                &estimand,
                estimate.ate,
                self.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?
        };

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
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
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
        // Sharp RD is the documented identify-per-click exception: prepare()
        // stores no cache for it, so every click computes identification.
        report_identify_compute(ctx);
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

        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, extra_diagnostics) = run_refuters(
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

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::RdSharp,
            estimator_id: EstimatorId::RdSharp,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached: false,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
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
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            return self.execute_bayesian(data, graph, &query.inner, physical, ctx);
        }
        let started = Instant::now();
        let (identifier, _) = self.resolve_conditional_pair();
        let identifier_id: IdentifierId = identifier.parse()?;
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query.inner), all frozen there, so reuse is
        // exact and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static(identifier_id, graph, &query.inner)?;
                let estimand =
                    select_estimand(&identification, EstimatorId::ConditionalLinearAdjustment)?;
                Ok((identification, estimand))
            })?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.inner.outcome,
            &query.inner.outcome_functional,
        )?;
        let est = ConditionalLinearAdjustment::new();
        let mut mean_query = query.clone();
        mean_query.inner.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
        let estimate =
            est.estimate(&data_est, &estimand, &mean_query, ctx).map_err(CausalError::from)?;
        let estimate = super::super::helpers::attach_conditional_functional_grid(
            estimate, data, query, &estimand, ctx,
        )?;
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, mut extra_diagnostics) = run_refuters(
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
        if let Some(diagnostic) = super::super::helpers::conditional_quantile_grid_diagnostic(
            data,
            query,
            estimand.adjustment_set.iter().copied(),
        )? {
            extra_diagnostics.push(diagnostic);
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id: EstimatorId::ConditionalLinearAdjustment,
            treatment: query.inner.treatment,
            outcome: query.inner.outcome,
            identify_cached,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        }))
    }

    /// Cpdag/Pag ConditionalEffect via the same generalized-adjustment envelope as ATE.
    pub(super) fn execute_class_conditional(
        &self,
        data: &TabularData,
        query: &antecedent_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier, estimator) = self.resolve_class_conditional_pair();
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        match self.graph.class() {
            GraphClass::Pag => {
                let pag = physical.static_pag().ok_or_else(|| CausalError::Compile {
                    message: "PAG ConditionalEffect execute missing resolved static PAG".into(),
                })?;
                let (envelope, identification, identify_cached) =
                    if let Some(cache) = self.pag_identification_cache.as_deref() {
                        (cache.envelope.clone(), cache.identification.clone(), true)
                    } else {
                        report_identify_compute(ctx);
                        let envelope = identify_pag(identifier_id, pag, &query.inner)?;
                        let identification = envelope_to_identification_result_for(
                            &envelope,
                            CausalQuery::ConditionalEffect(query.clone()),
                        );
                        (envelope, identification, false)
                    };
                self.finish_class_conditional(
                    data,
                    query,
                    physical,
                    ctx,
                    &envelope,
                    identification,
                    identify_cached,
                    super::pag_path::pag_envelope_diagnostic(&envelope),
                    identifier_id,
                    estimator_id,
                    started,
                    "pag",
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::Envelope {
                            envelope,
                            strategy: identifier_id,
                            structure_version: self.graph.version(),
                        },
                    )
                })
            }
            GraphClass::Cpdag => {
                let cpdag = self.graph.as_cpdag().ok_or_else(|| CausalError::Compile {
                    message: "CPDAG ConditionalEffect execute missing supplied graph".into(),
                })?;
                let (envelope, identification, identify_cached) =
                    if let Some(cache) = self.cpdag_identification_cache.as_deref() {
                        (cache.envelope.clone(), cache.identification.clone(), true)
                    } else {
                        report_identify_compute(ctx);
                        let envelope = identify_cpdag(identifier_id, cpdag, &query.inner)?;
                        let identification = envelope_to_identification_result_for(
                            &envelope,
                            CausalQuery::ConditionalEffect(query.clone()),
                        );
                        (envelope, identification, false)
                    };
                self.finish_class_conditional(
                    data,
                    query,
                    physical,
                    ctx,
                    &envelope,
                    identification,
                    identify_cached,
                    super::pag_path::cpdag_envelope_diagnostic(&envelope),
                    identifier_id,
                    estimator_id,
                    started,
                    "cpdag",
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::CpdagEnvelope {
                            envelope,
                            strategy: identifier_id,
                            structure_version: self.graph.version(),
                        },
                    )
                })
            }
            _ => Err(CausalError::Unsupported {
                message: "class-aware ConditionalEffect execute requires a Cpdag or Pag",
            }),
        }
    }

    fn finish_class_conditional<G>(
        &self,
        data: &TabularData,
        query: &antecedent_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        envelope: &IdentificationEnvelope<G>,
        identification: IdentificationResult,
        identify_cached: bool,
        envelope_diag: Diagnostic,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
        started: Instant,
        class_tag: &str,
    ) -> Result<StudyResult, CausalError> {
        if query.inner.outcome_functional.quantile_level().is_some()
            && envelope.unidentified_weight.0 > 1e-12
        {
            return Err(CausalError::Unsupported {
                message: "conditional quantile mixture requires zero unidentified completion mass",
            });
        }
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            if matches!(self.inference, InferenceMode::Bayesian(_))
                || matches!(estimator_id, EstimatorId::BayesianConditional)
            {
                return self.execute_pag_nonidentified_prior(
                    &query.inner,
                    physical,
                    ctx,
                    envelope,
                    identification,
                    identify_cached,
                    started,
                    envelope_diag,
                    class_tag,
                );
            }
            return Err(CausalError::Compile {
                message: format!(
                    "{class_tag} ConditionalEffect not identified (no identified mass in envelope)"
                ),
            });
        }
        if matches!(self.inference, InferenceMode::Bayesian(_))
            || matches!(estimator_id, EstimatorId::BayesianConditional)
        {
            return self.execute_pag_bayesian(
                data,
                &query.inner,
                physical,
                ctx,
                envelope,
                identification,
                identify_cached,
                started,
                envelope_diag,
                class_tag,
            );
        }

        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.inner.outcome,
            &query.inner.outcome_functional,
        )?;
        let mut diagnostics = vec![envelope_diag];
        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut atom_ifs = Vec::new();
        let mut atom_weights = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut refute_atoms = Vec::new();
        let mut grid_atoms = Vec::new();
        let est = ConditionalLinearAdjustment::new();
        for (i, case) in envelope.cases.iter().enumerate() {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let mut estimand = select_estimand(&case.result, estimator_id)?;
            if estimand.method.as_ref().starts_with("generalized.adjustment") {
                estimand.method = Arc::from("backdoor.adjustment");
            }
            let mut mean_query = query.clone();
            mean_query.inner.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
            let estimate =
                est.estimate(&data_est, &estimand, &mean_query, ctx).map_err(CausalError::from)?;
            let w = case.weight.0;
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic));
            if let Some(inf) = estimate
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, &query.inner, &estimand, inf))
            {
                atom_ifs.push(inf);
                atom_weights.push(w);
            }
            total_w += w;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                assumptions = estimate.assumptions.clone();
            }
            grid_atoms.push((w, estimand.clone()));
            refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: w,
                estimand,
                indexer: None,
            });
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: format!("{class_tag} ConditionalEffect envelope had no estimable cases"),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: format!("{class_tag} ConditionalEffect envelope missing estimand"),
        })?;
        let n_contributing = se_items.len();
        let se = if atom_ifs.len() == n_contributing {
            mix_static_envelope_se(&atom_ifs, &atom_weights)
        } else {
            mix_weighted_analytic_se(se_items)
        };
        let mut estimate = EffectEstimate::new(
            weighted_ate / total_w,
            se,
            assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        if atom_ifs.len() == n_contributing {
            if let Some(inf) = mixed_static_influence(&atom_ifs, &atom_weights) {
                estimate.influence = Some(inf);
            }
        }
        if query.inner.outcome_functional.thresholds().is_some()
            || query.inner.outcome_functional.quantile_level().is_some()
        {
            estimate = super::attach_class_conditional_functional_grid(
                estimate,
                data,
                query,
                &grid_atoms,
                ctx,
            )?;
        }
        if let Some(diagnostic) = super::super::helpers::conditional_quantile_grid_diagnostic(
            data,
            query,
            grid_atoms.iter().flat_map(|(_, e)| e.adjustment_set.iter().copied()),
        )? {
            diagnostics.push(diagnostic);
        }
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = run_envelope_effect_refuters(
            data,
            &query.inner,
            &estimate,
            &refute_atoms,
            &mut refute_ws,
            ctx,
            self.refute,
            estimator_id.as_str(),
            &self.custom_validators,
            None,
            self.split.as_ref(),
            None,
        )?;
        diagnostics.extend(na_diagnostics);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        if !estimate.se_analytic.is_finite() {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.inner.treatment,
            outcome: query.inner.outcome,
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
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
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
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static_query(
                    IdentifierId::PathSpecificNatural,
                    graph,
                    &CausalQuery::Mediation(query.clone()),
                )?;
                let estimand =
                    select_estimand(&identification, EstimatorId::StaticMediationLinear)?;
                Ok((identification, estimand))
            })?;
        let mediation = antecedent_estimate::estimate_static_mediation(
            data,
            graph,
            query,
            identification.required_assumptions.clone(),
            self.bootstrap_replicates,
            &[],
            ctx,
        )?;
        let estimate = mediation.effect.clone();
        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::mediation::refute_static_mediation(
                data,
                graph,
                query,
                &mediation,
                self.refute == RefuteSuite::Full,
                ctx,
            )?
        };
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::PathSpecificNatural,
            estimator_id: EstimatorId::StaticMediationLinear,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: Some(mediation),
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: (self.bootstrap_replicates > 1)
                .then_some(self.bootstrap_replicates),
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_tiered_average(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        background: &antecedent_graph::TieredBackground,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identification, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = antecedent_identify::identify_tiered(background, query)?;
                let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                    CausalError::Compile {
                        message: "tiered identification returned no estimand".into(),
                    }
                })?;
                Ok((identification, estimand))
            })
            .map(|(id, _, cached)| (id, cached))?;
        if background.within_tier == antecedent_graph::WithinTier::Unknown {
            return self.finish_tiered_unknown(
                data,
                query,
                physical,
                ctx,
                identification,
                identifier_id,
                estimator_id,
                identify_cached,
                started,
            );
        }
        let mut estimand = identification.estimands[0].clone();
        if estimand.method.as_ref().starts_with("generalized.adjustment")
            || estimand.method.as_ref().starts_with("tiered.")
        {
            estimand.method = Arc::from("backdoor.adjustment");
        }
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.outcome,
            &query.outcome_functional,
        )?;
        let mut ws = StaticEstimateWorkspaces::default();
        let spec = self.estimator_spec.clone().unwrap_or(EstimatorSpec::Default(estimator_id));
        let estimate = estimate_static_effect(
            &spec,
            &data_est,
            &estimand,
            query,
            identification.required_assumptions.clone(),
            self.bootstrap_replicates,
            self.overlap_policy,
            self.population_registry.as_ref(),
            ctx,
            &mut ws,
        )?;
        let mut extra_diagnostics = Vec::new();
        let estimate = super::super::helpers::attach_average_functional_grid(
            estimate,
            data,
            query,
            &estimand,
            &mut extra_diagnostics,
            estimator_id,
            self,
        )?;
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            estimator_id.as_str(),
            &self.custom_validators,
            None,
        )?;
        extra_diagnostics.extend(na_diagnostics);
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
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_tiered_unknown(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        identification: IdentificationResult,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
        identify_cached: bool,
        started: Instant,
    ) -> Result<StudyResult, CausalError> {
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.outcome,
            &query.outcome_functional,
        )?;
        let spec = self.estimator_spec.clone().unwrap_or(EstimatorSpec::Default(estimator_id));
        let mut atom_ifs = Vec::new();
        let mut atom_ws = Vec::new();
        let mut scenarios = Vec::new();
        let mut primary = None;
        let mut assumptions = identification.required_assumptions.clone();
        for estimand in &identification.estimands {
            let mut tagged = estimand.clone();
            tagged.method = Arc::from("backdoor.adjustment");
            let mut ws = StaticEstimateWorkspaces::default();
            let estimate = estimate_static_effect(
                &spec,
                &data_est,
                &tagged,
                query,
                identification.required_assumptions.clone(),
                self.bootstrap_replicates,
                self.overlap_policy,
                self.population_registry.as_ref(),
                ctx,
                &mut ws,
            )?;
            scenarios.push((
                scenarios.len() as u64,
                antecedent_core::ResponseValue::Scalar(estimate.ate),
            ));
            if let Some(inf) = estimate
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, query, &tagged, inf))
            {
                atom_ifs.push(inf);
                atom_ws.push(0.5);
            }
            if primary.is_none() {
                primary = Some(tagged);
                assumptions = estimate.assumptions;
            }
        }
        let refs: Vec<&[f64]> = atom_ifs.iter().map(Vec::as_slice).collect();
        let mut extra_diagnostics = Vec::new();
        let covariance = if refs.len() == scenarios.len() && refs.len() >= 2 {
            if let Ok(cov) = antecedent_estimate::joint_influence_covariance(&refs, None) {
                Some(cov)
            } else {
                extra_diagnostics.push(Diagnostic::new(
                        "tiered.unknown.joint_if.unavailable",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Warning,
                        "tiered Unknown joint IF covariance could not be formed; scenario intervals are omitted",
                    ));
                None
            }
        } else {
            if !refs.is_empty() && refs.len() != scenarios.len() {
                extra_diagnostics.push(Diagnostic::new(
                    "tiered.unknown.joint_if.unavailable",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "tiered Unknown joint IF refused: every canonical scenario must supply an aligned influence function",
                ));
            }
            None
        };
        let estimate =
            EffectEstimate::new(f64::NAN, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride);
        let estimand = primary.ok_or_else(|| CausalError::Compile {
            message: "tiered Unknown envelope had no estimand".into(),
        })?;
        let mut estimate = estimate.with_joint_covariance(covariance.clone());
        let values: Vec<f64> = scenarios
            .iter()
            .filter_map(|(_, v)| {
                if let antecedent_core::ResponseValue::Scalar(v) = v { Some(*v) } else { None }
            })
            .collect();
        if let Some(cov) = &covariance {
            let c = antecedent_estimate::max_t_critical(cov, 0.95, 4096, 15)?;
            estimate.scenario_intervals = Some(
                values
                    .iter()
                    .enumerate()
                    .map(|(j, v)| (v - c * cov.se(j), v + c * cov.se(j)))
                    .collect::<Vec<_>>()
                    .into(),
            );
        }
        estimate.scenario_effects = Some(values.into());
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
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        }))
    }
}
