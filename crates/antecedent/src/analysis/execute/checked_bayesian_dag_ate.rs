//! Sealed Bayesian DAG average-effect g-computation execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// The causal proof, prior, procedure, validation, and result contract selected
/// during preparation. Estimation uses this value and current rows only; it
/// never consults the `Study` that produced it.
#[derive(Clone)]
pub(crate) struct CheckedBayesianDagAteExecution {
    graph: Dag,
    operation: super::super::prepared::CheckedBayesianGcompOperation,
    context: IdentifiedResultContext,
    physical: PhysicalExecutionPlan,
    config: BayesianConfig,
    validation: RefuteSuite,
    latency: Option<LatencyMode>,
    validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    stage_sink: Option<Arc<dyn super::super::stage::StageResultSink>>,
}

impl std::fmt::Debug for CheckedBayesianDagAteExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedBayesianDagAteExecution")
            .field("graph", &self.graph)
            .field("query", &self.operation.query())
            .field("identification", &self.operation.identification())
            .field("estimand", &self.operation.estimand())
            .field("procedure", &EstimatorId::BayesianGcomp)
            .field("validation", &self.validation)
            .finish_non_exhaustive()
    }
}

impl CheckedBayesianDagAteExecution {
    pub(crate) fn checked(
        graph: &Dag,
        operation: super::super::prepared::CheckedBayesianGcompOperation,
        context: IdentifiedResultContext,
        physical: PhysicalExecutionPlan,
        validation: RefuteSuite,
        latency: Option<LatencyMode>,
        validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
        stage_sink: Option<Arc<dyn super::super::stage::StageResultSink>>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::AverageEffect(target) = &context.query else {
            return Err(CausalError::Compile {
                message: "Bayesian DAG effect result context has a different query family".into(),
            });
        };
        if target != operation.query()
            || physical.logical.query != context.query
            || context.graph_class != GraphClass::Dag
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::BayesianGcomp.as_str())
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::BackdoorAdjustment.as_str())
        {
            return Err(CausalError::Compile {
                message:
                    "Bayesian DAG effect target, graph, or procedure changed after identification"
                        .into(),
            });
        }
        let InferenceMode::Bayesian(config) = operation.inference() else {
            return Err(CausalError::Compile {
                message: "Bayesian DAG effect requires its prepared Bayesian configuration".into(),
            });
        };
        if operation.identification().average_effect() != Some(target)
            || !operation.identification().estimands.iter().any(|candidate| {
                candidate.functional == operation.estimand().functional
                    && candidate.method == operation.estimand().method
                    && candidate.adjustment_set == operation.estimand().adjustment_set
            })
        {
            return Err(CausalError::Compile {
                message: "Bayesian DAG effect estimand is not supplied by its retained proof"
                    .into(),
            });
        }
        if !matches!(
            operation.identification().status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(CausalError::Unsupported {
                message: "Bayesian DAG effect requires a point-identified target",
            });
        }
        if !matches!(target.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            || target.target_population != antecedent_core::TargetPopulation::AllObserved
        {
            return Err(CausalError::Unsupported {
                message: "checked Bayesian DAG g-computation currently licenses only a mean ATE over AllObserved",
            });
        }
        let checked_identification =
            identify_static(IdentifierId::BackdoorAdjustment, graph, target)?;
        let (checked_identification, checked_estimand) =
            select_claim(checked_identification, EstimatorId::BayesianGcomp)?;
        if checked_identification.status != operation.identification().status
            || checked_identification.query != operation.identification().query
            || checked_estimand.functional != operation.estimand().functional
            || checked_estimand.method != operation.estimand().method
            || checked_estimand.adjustment_set != operation.estimand().adjustment_set
        {
            return Err(CausalError::Compile {
                message: "Bayesian DAG effect proof is not justified by its retained graph".into(),
            });
        }
        let config = config.clone();
        Ok(Self {
            graph: graph.clone(),
            operation,
            context,
            physical,
            config,
            validation,
            latency,
            validators,
            stage_sink,
        })
    }

    pub(crate) fn operation(&self) -> &super::super::prepared::CheckedBayesianGcompOperation {
        &self.operation
    }

    pub(crate) fn validation(&self) -> RefuteSuite {
        self.validation
    }

    pub(crate) fn custom_validator_names(&self) -> Vec<Arc<str>> {
        self.validators.iter().map(|validator| Arc::from(validator.name())).collect()
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut clock = super::super::stage::StageClock::new();
        let query = self.operation.query();
        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        let identification = self.operation.identification().clone();
        let estimand = self.operation.estimand().clone();
        clock.finish(super::super::stage::STAGE_IDENTIFY);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Identify {
                identification: identification.clone(),
                estimand: estimand.clone(),
            },
        );
        let full_cols = data.schema().len();
        let (projected, query_est, estimand_est) =
            project_for_ate_estimate(data, query, &estimand)?;
        let projected_cols = projected.schema().len();

        clock.begin(ctx, super::super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        let mut fitter = bayesian_gcomp(&self.config, ctx);
        let prepared =
            fitter.prepare(&projected, &estimand_est, &query_est).map_err(CausalError::from)?;
        let (prior, conflict) =
            resolve_bayesian_prior_with_conflict(&self.config, &prepared, Some(ctx))?;
        fitter.prior = prior;
        let mut workspace = BayesianGCompWorkspace::default();
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
            .push(super::bayesian_path::gcomp_outcome_model_assumption(likelihood, false));
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
        let mut diagnostics = Vec::new();
        if let Some(projection) = projection_diagnostic(full_cols, projected_cols) {
            diagnostics.push(projection);
        }
        if let Some(conflict) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, conflict);
        }

        clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
        let mut workspace_refute = EstimationWorkspace::default();
        let (mut refutations, not_applicable) = match self.validation {
            RefuteSuite::None => (Vec::new(), Vec::new()),
            suite => run_refuters(
                &projected,
                &estimand_est,
                &query_est,
                &estimate,
                &mut workspace_refute,
                None,
                ctx,
                suite,
                EstimatorId::BayesianGcomp.as_str(),
                &self.validators,
                None,
            )?,
        };
        diagnostics.extend(not_applicable);

        let mut predictive_checks = Vec::new();
        if self.validation != RefuteSuite::None {
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
        if self.validation == RefuteSuite::Full {
            let (summary, sensitivity) = evaluate_bayesian_prior_sensitivity(
                &self.config,
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
        let early_stopped = posterior.early_stopped;
        let args = IdentifiedExecuteFinish {
            physical: &self.physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::BackdoorAdjustment,
            estimator_id: EstimatorId::BayesianGcomp,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached: true,
            extra_diagnostics: diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: clock.wall_time_ns(),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped,
            extras: IdentifiedExecuteExtras {
                stage_timings_ns: clock.timings(),
                posterior: Some(posterior),
                n_draws: draws,
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        };
        Ok(finish_identified_execute_with_context(&self.context, Some(data), args))
    }
}
