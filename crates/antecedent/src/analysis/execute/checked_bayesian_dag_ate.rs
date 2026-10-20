//! Sealed Bayesian DAG average-effect g-computation execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianConfig, BayesianGCompWorkspace, BayesianSuiteContext, CausalError,
    CausalPosterior, CausalQuery, Dag, Diagnostic, EstimationWorkspace, EstimatorId,
    ExecutionContext, GraphClass, IdentificationStatus, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifiedResultContext, IdentifierId, InferenceMode, LatencyMode,
    PhysicalExecutionPlan, PosteriorPredictiveCheck, PredictiveCheckReport, PriorPredictiveCheck,
    RefutationRun, RefuteSuite, Strategy, StudyResult, TableView, TabularData, ValidationSuite,
    ValidatorId, bayesian_gcomp, effect_from_posterior, evaluate_bayesian_prior_sensitivity,
    finish_identified_execute_with_context, identify_static, project_for_ate_estimate,
    projection_diagnostic, push_conflict_diagnostics, resolve_bayesian_prior_with_conflict,
    select_claim, validator_not_applicable_diagnostics, with_conflict_summary,
    with_prior_sensitivity,
};
use crate::analysis::proof_support::{proof_supplies, verify_proof_against_graph};

/// Fitted point stage state the validate stage reuses.
struct CheckedBayesianDagAtePoint {
    fitter: antecedent_estimate::BayesianGComputationAte,
    prepared: antecedent_estimate::PreparedBayesianProblem,
    workspace: BayesianGCompWorkspace,
    posterior: CausalPosterior,
    estimate: antecedent_estimate::EffectEstimate,
}

/// Borrowed inputs of the validate stage.
#[derive(Clone, Copy)]
struct CheckedBayesianDagAteValidation<'a> {
    data: &'a TabularData,
    estimand: &'a antecedent_identify::IdentifiedEstimand,
    query: &'a antecedent_core::AverageEffectQuery,
    estimate: &'a antecedent_estimate::EffectEstimate,
    fitter: &'a antecedent_estimate::BayesianGComputationAte,
    prepared: &'a antecedent_estimate::PreparedBayesianProblem,
    status: IdentificationStatus,
}

/// The causal proof, prior, procedure, validation, and result contract selected
/// during preparation. Estimation uses this value and current rows only; it
/// never consults the `Study` that produced it.
#[derive(Clone)]
pub(crate) struct CheckedBayesianDagAteExecution {
    graph: Dag,
    operation: super::super::prepared::CheckedBayesianGcompOperation,
    context: IdentifiedResultContext,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    config: BayesianConfig,
    validation: RefuteSuite,
    latency: Option<LatencyMode>,
    validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    stage_sink: Option<Arc<dyn super::super::stage::StageResultSink>>,
    draw_controls: Option<crate::PosteriorDrawControls>,
}

/// The frozen premises of [`CheckedBayesianDagAteExecution::checked`], carried together.
pub(crate) struct CheckedBayesianDagAteExecutionPremises<'a> {
    pub(crate) graph: &'a Dag,
    pub(crate) operation: super::super::prepared::CheckedBayesianGcompOperation,
    pub(crate) context: IdentifiedResultContext,
    pub(crate) physical: std::sync::Arc<PhysicalExecutionPlan>,
    pub(crate) validation: RefuteSuite,
    pub(crate) latency: Option<LatencyMode>,
    pub(crate) validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    pub(crate) stage_sink: Option<Arc<dyn super::super::stage::StageResultSink>>,
    pub(crate) draw_controls: Option<crate::PosteriorDrawControls>,
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
    pub(crate) fn with_inference_config(
        &self,
        config: BayesianConfig,
    ) -> Result<Self, CausalError> {
        if config.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
            || config.likelihood != antecedent_prob::BayesLikelihood::GaussianIdentity
            || config.n_draws < 2
            || !config.prior_scale.is_finite()
            || config.prior_scale <= 0.0
        {
            return Err(CausalError::Unsupported {
                message: "retained Bayesian recalculation requires conjugate Gaussian inference",
            });
        }
        let mut next = self.clone();
        next.operation.inference = InferenceMode::Bayesian(config.clone());
        next.context.inference = InferenceMode::Bayesian(config.clone());
        next.config = config;
        Ok(next)
    }

    pub(crate) fn set_stage_sink(
        &mut self,
        sink: Option<Arc<dyn super::super::stage::StageResultSink>>,
    ) {
        self.stage_sink = sink;
    }

    /// Replace the caller validators run by this sealed operation. Names are
    /// checked by the prepared handle before the swap.
    pub(crate) fn set_custom_validators(
        &mut self,
        validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    ) {
        self.validators = validators;
    }

    pub(crate) fn checked(
        premises: CheckedBayesianDagAteExecutionPremises<'_>,
    ) -> Result<Self, CausalError> {
        let CheckedBayesianDagAteExecutionPremises {
            graph,
            operation,
            context,
            physical,
            validation,
            latency,
            validators,
            stage_sink,
            draw_controls,
        } = premises;
        let CausalQuery::AverageEffect(target) = &context.query else {
            return Err(CausalError::Compile {
                message: "Bayesian DAG effect result context has a different query family".into(),
            });
        };
        if target != operation.query()
            || physical.logical.query != context.query
            || context.graph_class != GraphClass::Dag
            || !physical
                .binding()
                .names(IdentifierId::BackdoorAdjustment, EstimatorId::BayesianGcomp)
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
            || !proof_supplies(operation.identification(), operation.estimand())
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
        verify_proof_against_graph(
            &checked_identification,
            &checked_estimand,
            operation.identification(),
            operation.estimand(),
            "Bayesian DAG effect proof is not justified by its retained graph",
        )?;
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
            draw_controls,
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

    /// Point and uncertainty stages: prepare the design, resolve the prior, fit, and record
    /// the outcome-model assumption, with the stage clock and events around them.
    fn estimate_point(
        &self,
        projected: &TabularData,
        estimand_est: &antecedent_identify::IdentifiedEstimand,
        query_est: &antecedent_core::AverageEffectQuery,
        status: IdentificationStatus,
        clock: &mut super::super::stage::StageClock,
        ctx: &ExecutionContext,
    ) -> Result<CheckedBayesianDagAtePoint, CausalError> {
        clock.begin(ctx, super::super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        let mut fitter = bayesian_gcomp(&self.config, ctx);
        let prepared =
            fitter.prepare(projected, estimand_est, query_est).map_err(CausalError::from)?;
        let (prior, conflict) =
            resolve_bayesian_prior_with_conflict(&self.config, &prepared, Some(ctx))?;
        fitter.prior = prior;
        let mut workspace = BayesianGCompWorkspace::default();
        let controlled = self.draw_controls.is_some();
        let mut posterior =
            fitter.fit(&prepared, status, &mut workspace, ctx).map_err(|error| {
                super::bayesian_path::chain_refusal_under_draw_controls(
                    controlled,
                    &self.config,
                    error,
                )
            })?;
        if controlled {
            super::bayesian_path::enforce_draw_controls(&mut posterior)?;
        }
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
        Ok(CheckedBayesianDagAtePoint { fitter, prepared, workspace, posterior, estimate })
    }

    /// Validate stage: refutation suite, prior / posterior predictive checks, and under the
    /// full suite prior sensitivity plus MCMC diagnostics.
    fn validate(
        &self,
        input: &CheckedBayesianDagAteValidation<'_>,
        mut posterior: CausalPosterior,
        workspace: &mut BayesianGCompWorkspace,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<
        (CausalPosterior, Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>),
        CausalError,
    > {
        let CheckedBayesianDagAteValidation {
            data,
            estimand,
            query,
            estimate,
            fitter,
            prepared,
            status,
        } = *input;
        let mut workspace_refute = EstimationWorkspace::default();
        let (mut refutations, not_applicable) = match self.validation {
            RefuteSuite::None => (Vec::new(), Vec::new()),
            suite => RefutationRun {
                data,
                estimand,
                query,
                estimate,
                suite,
                estimator: EstimatorId::BayesianGcomp.as_str(),
                custom: &self.validators,
                temporal: None,
            }
            .reports(&mut workspace_refute, None, ctx)?,
        };
        diagnostics.extend(not_applicable);

        let mut predictive_checks = Vec::new();
        if self.validation != RefuteSuite::None {
            const PPC_ALPHA: f64 = 0.05;
            let sims = super::super::latency::predictive_check_sims(self.latency);
            let prior = fitter.prior_in_force(prepared.design.ncols);
            let prior_rep = PriorPredictiveCheck::for_estimator(fitter, ctx)
                .with_n_sims(sims)
                .check_with_prior(prepared, &prior, ctx)
                .map_err(CausalError::from)?;
            refutations.push(prior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(prior_rep);
            let posterior_rep = PosteriorPredictiveCheck::for_estimator(fitter, ctx)
                .with_n_sims(sims)
                .check(prepared, &posterior)
                .map_err(CausalError::from)?;
            refutations.push(posterior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(posterior_rep);
        }
        if self.validation == RefuteSuite::Full {
            let (summary, sensitivity) = evaluate_bayesian_prior_sensitivity(
                &self.config,
                fitter,
                prepared,
                status,
                &posterior,
                workspace,
                ctx,
            )?;
            refutations.push(sensitivity.to_report(&summary, estimate.ate));
            posterior = with_prior_sensitivity(posterior, summary);
            let suite = ValidationSuite::new().with(ValidatorId::McmcDiagnostics);
            let mut bayes_context = BayesianSuiteContext::new(
                fitter,
                prepared,
                &posterior,
                status,
                workspace,
                estimate.ate,
            );
            let outcomes =
                suite.run_bayesian(&mut bayes_context, ctx).map_err(CausalError::from)?;
            refutations.extend(ValidationSuite::reports_only(&outcomes));
            diagnostics.extend(validator_not_applicable_diagnostics(&outcomes));
        }
        Ok((posterior, refutations, predictive_checks))
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

        let CheckedBayesianDagAtePoint { fitter, prepared, mut workspace, posterior, estimate } =
            self.estimate_point(
                &projected,
                &estimand_est,
                &query_est,
                identification.status,
                &mut clock,
                ctx,
            )?;
        let mut diagnostics = Vec::new();
        if let Some(projection) = projection_diagnostic(full_cols, projected_cols) {
            diagnostics.push(projection);
        }
        if let Some(conflict) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, conflict);
        }
        super::bayesian_path::push_passed_route_diagnostics(&mut diagnostics, &posterior);

        clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
        let (posterior, refutations, predictive_checks) = self.validate(
            &CheckedBayesianDagAteValidation {
                data: &projected,
                estimand: &estimand_est,
                query: &query_est,
                estimate: &estimate,
                fitter: &fitter,
                prepared: &prepared,
                status: identification.status,
            },
            posterior,
            &mut workspace,
            &mut diagnostics,
            ctx,
        )?;
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
        let args = IdentifiedExecuteFinish::new(
            &self.physical,
            identification,
            estimand,
            estimate,
            Strategy::new(IdentifierId::BackdoorAdjustment, EstimatorId::BayesianGcomp),
            query.treatment,
            query.outcome,
        )
        .identify_cached(true)
        .extra_diagnostics(diagnostics)
        .refutations(refutations)
        .wall_time_ns(clock.wall_time_ns())
        .cancelled(ctx.cancellation.is_cancelled())
        .early_stopped(early_stopped)
        .extras(IdentifiedExecuteExtras {
            stage_timings_ns: clock.timings(),
            posterior: Some(posterior),
            n_draws: draws,
            predictive_checks,
            bootstrap_replicates_requested: Some(None),
            ..Default::default()
        });
        Ok(finish_identified_execute_with_context(&self.context, Some(data), args))
    }
}
