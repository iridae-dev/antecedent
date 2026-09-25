//! Sealed Bayesian DAG conditional-effect g-computation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Bayesian CATE target and its checked DAG proof, prior, and validation plan.
#[derive(Clone)]
pub(crate) struct CheckedBayesianConditionalOperation {
    query: antecedent_core::ConditionalEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    inference: InferenceMode,
    refute: RefuteSuite,
    graph: Dag,
    context: IdentifiedResultContext,
    physical: PhysicalExecutionPlan,
    latency: Option<LatencyMode>,
    validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    stage_sink: Option<Arc<dyn super::super::stage::StageResultSink>>,
}

impl std::fmt::Debug for CheckedBayesianConditionalOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedBayesianConditionalOperation")
            .field("query", &self.query)
            .field("identification", &self.identification)
            .field("estimand", &self.estimand)
            .field("refute", &self.refute)
            .field("graph", &self.graph)
            .finish_non_exhaustive()
    }
}

impl CheckedBayesianConditionalOperation {
    pub(crate) fn set_stage_sink(
        &mut self,
        sink: Option<Arc<dyn super::super::stage::StageResultSink>>,
    ) {
        self.stage_sink = sink;
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn checked(
        graph: &Dag,
        query: antecedent_core::ConditionalEffectQuery,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        inference: InferenceMode,
        refute: RefuteSuite,
        context: IdentifiedResultContext,
        physical: PhysicalExecutionPlan,
        latency: Option<LatencyMode>,
        validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
        stage_sink: Option<Arc<dyn super::super::stage::StageResultSink>>,
    ) -> Result<Self, CausalError> {
        let InferenceMode::Bayesian(_) = inference else {
            return Err(CausalError::Compile {
                message: "checked Bayesian CATE requires its prepared Bayesian configuration"
                    .into(),
            });
        };
        let CausalQuery::ConditionalEffect(target) = &context.query else {
            return Err(CausalError::Compile {
                message: "Bayesian conditional result context has a different query family".into(),
            });
        };
        if target != &query
            || physical.logical.query != context.query
            || context.graph_class != GraphClass::Dag
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::BayesianConditional.as_str())
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::BackdoorAdjustment.as_str())
            || !matches!(
                context.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            || !matches!(refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
        {
            return Err(CausalError::Compile {
                message:
                    "Bayesian CATE target, graph source, or procedure changed after identification"
                        .into(),
            });
        }
        if query.inner.target_population != antecedent_core::TargetPopulation::AllObserved
            || !matches!(query.inner.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
        {
            return Err(CausalError::Unsupported {
                message: "checked Bayesian DAG conditional effects currently require a mean outcome over AllObserved",
            });
        }
        if identification.average_effect() != Some(&query.inner)
            || !identification.estimands.iter().any(|candidate| {
                candidate.functional == estimand.functional
                    && candidate.method == estimand.method
                    && candidate.adjustment_set == estimand.adjustment_set
            })
        {
            return Err(CausalError::Compile {
                message: "Bayesian CATE estimand is not supplied by its retained proof".into(),
            });
        }
        let (checked_identification, checked_estimand) = select_claim(
            identify_static(IdentifierId::BackdoorAdjustment, graph, &query.inner)?,
            EstimatorId::BayesianConditional,
        )?;
        if checked_identification.status != identification.status
            || checked_identification.query != identification.query
            || checked_estimand.functional != estimand.functional
            || checked_estimand.method != estimand.method
            || checked_estimand.adjustment_set != estimand.adjustment_set
        {
            return Err(CausalError::Compile {
                message: "Bayesian CATE proof is not justified by its retained graph".into(),
            });
        }
        Ok(Self {
            query,
            identification,
            estimand,
            inference,
            refute,
            graph: graph.clone(),
            context,
            physical,
            latency,
            validators,
            stage_sink,
        })
    }

    pub(crate) fn query(&self) -> &antecedent_core::ConditionalEffectQuery {
        &self.query
    }

    pub(crate) fn identification(&self) -> &IdentificationResult {
        &self.identification
    }

    pub(crate) fn estimand(&self) -> &IdentifiedEstimand {
        &self.estimand
    }

    pub(crate) fn inference(&self) -> &InferenceMode {
        &self.inference
    }

    pub(crate) fn validation(&self) -> RefuteSuite {
        self.refute
    }

    pub(crate) fn modifier_roles(&self) -> &[VariableId] {
        &self.query.inner.effect_modifiers
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut clock = super::super::stage::StageClock::new();
        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        let identification = self.identification.clone();
        let estimand = self.estimand.clone();
        clock.finish(super::super::stage::STAGE_IDENTIFY);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Identify {
                identification: identification.clone(),
                estimand: estimand.clone(),
            },
        );
        let InferenceMode::Bayesian(config) = &self.inference else {
            unreachable!("checked constructor requires Bayesian inference")
        };
        let mut fitter = bayesian_gcomp(config, ctx);
        let prepared =
            fitter.prepare_conditional(data, &estimand, &self.query).map_err(CausalError::from)?;
        let (prior, conflict) = resolve_bayesian_prior_with_conflict(config, &prepared, Some(ctx))?;
        fitter.prior = prior;
        let mut workspace = BayesianGCompWorkspace::default();
        clock.begin(ctx, super::super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        let mut posterior = fitter
            .fit(&prepared, identification.status, &mut workspace, ctx)
            .map_err(CausalError::from)?;
        if let Some(summary) = conflict {
            posterior = with_conflict_summary(posterior, summary);
        }
        let likelihood =
            if fitter.backend == antecedent_estimate::BayesianBackendKind::ConjugateGaussian {
                antecedent_prob::BayesLikelihood::GaussianIdentity
            } else {
                fitter.likelihood
            };
        posterior
            .assumptions
            .push(super::bayesian_path::gcomp_outcome_model_assumption(likelihood, true));
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
        let mut refutations = Vec::new();
        let mut diagnostics = Vec::new();
        let mut predictive_checks = Vec::new();
        clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
        let mut refute_workspace = EstimationWorkspace::default();
        if self.refute != RefuteSuite::None {
            let mean_query = {
                let mut query = self.query.inner.clone();
                query.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
                query
            };
            let (reports, not_applicable) = run_refuters(
                data,
                &estimand,
                &mean_query,
                &estimate,
                &mut refute_workspace,
                None,
                ctx,
                self.refute,
                EstimatorId::BayesianConditional.as_str(),
                &self.validators,
                None,
            )?;
            refutations = reports;
            diagnostics.extend(not_applicable);
            const PPC_ALPHA: f64 = 0.05;
            let sims = super::super::latency::predictive_check_sims(self.latency);
            let prior = fitter.prior_in_force(prepared.design.ncols);
            let prior_rep = PriorPredictiveCheck::for_estimator(&fitter, ctx)
                .with_n_sims(sims)
                .check_with_prior(&prepared, &prior, ctx)
                .map_err(CausalError::from)?;
            refutations.push(prior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(prior_rep);
            let posterior_rep = PosteriorPredictiveCheck::for_estimator(&fitter, ctx)
                .with_n_sims(sims)
                .check(&prepared, &posterior)
                .map_err(CausalError::from)?;
            refutations.push(posterior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(posterior_rep);
        }
        if self.refute == RefuteSuite::Full {
            let (summary, sensitivity) = evaluate_bayesian_prior_sensitivity(
                config,
                &fitter,
                &prepared,
                identification.status,
                &posterior,
                &mut workspace,
                ctx,
            )?;
            refutations.push(sensitivity.to_report(&summary, estimate.ate));
            posterior = with_prior_sensitivity(posterior, summary);
            let suite = ValidationSuite::new().with(ValidatorId::McmcDiagnostics);
            let mut bayes_context = BayesianSuiteContext::new(
                &fitter,
                &prepared,
                &posterior,
                identification.status,
                &mut workspace,
                estimate.ate,
            );
            let outcomes =
                suite.run_bayesian(&mut bayes_context, ctx).map_err(CausalError::from)?;
            refutations.extend(ValidationSuite::reports_only(&outcomes));
            diagnostics.extend(validator_not_applicable_diagnostics(&outcomes));
        }
        clock.finish(super::super::stage::STAGE_VALIDATE);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Validate {
                refutations: refutations.clone(),
                predictive_checks: predictive_checks.clone(),
            },
        );
        let draws = u32::try_from(posterior.draws.n_draws).ok();
        Ok(finish_identified_execute_with_context(
            &self.context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification,
                estimand,
                estimate,
                identifier_id: IdentifierId::BackdoorAdjustment,
                estimator_id: EstimatorId::BayesianConditional,
                treatment: self.query.inner.treatment,
                outcome: self.query.inner.outcome,
                identify_cached: true,
                extra_diagnostics: diagnostics,
                refutations,
                distribution: None,
                mediation: None,
                wall_time_ns: clock.wall_time_ns(),
                bootstrap_replicates_ok: None,
                cancelled: ctx.cancellation.is_cancelled(),
                early_stopped: posterior.early_stopped,
                extras: IdentifiedExecuteExtras {
                    stage_timings_ns: clock.timings(),
                    posterior: Some(posterior),
                    n_draws: draws,
                    predictive_checks,
                    bootstrap_replicates_requested: Some(None),
                    ..Default::default()
                },
            },
        ))
    }
}
