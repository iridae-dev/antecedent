//! Sealed Bayesian DAG conditional-effect g-computation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianGCompWorkspace, BayesianSuiteContext, CausalError, CausalPosterior, CausalQuery,
    Dag, Diagnostic, EstimationWorkspace, EstimatorId, ExecutionContext, GraphClass,
    IdentificationResult, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedResultContext, IdentifierId, InferenceMode, LatencyMode, PhysicalExecutionPlan,
    PosteriorPredictiveCheck, PredictiveCheckReport, PriorPredictiveCheck, RefutationRun,
    RefuteSuite, Strategy, StudyResult, TableView, TabularData, ValidationSuite, ValidatorId,
    VariableId, bayesian_gcomp, effect_from_posterior, evaluate_bayesian_prior_sensitivity,
    finish_identified_execute_with_context, identify_static, project_for_ate_estimate,
    projection_diagnostic, resolve_bayesian_prior_with_conflict, select_claim,
    validator_not_applicable_diagnostics, with_conflict_summary, with_prior_sensitivity,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::{
    RetainedProof, proof_supplies, verify_proof_against_graph, verify_retained_static,
};

const PPC_ALPHA: f64 = 0.05;

/// Fitted point stage state the validate stage reuses.
struct CheckedBayesianConditionalPoint {
    fitter: antecedent_estimate::BayesianGComputationAte,
    prepared: antecedent_estimate::PreparedBayesianProblem,
    workspace: BayesianGCompWorkspace,
    posterior: CausalPosterior,
    estimate: antecedent_estimate::EffectEstimate,
}

/// Borrowed inputs of the validate stage.
#[derive(Clone, Copy)]
struct CheckedBayesianConditionalValidation<'a> {
    data: &'a TabularData,
    estimand: &'a IdentifiedEstimand,
    query: &'a antecedent_core::AverageEffectQuery,
    estimate: &'a antecedent_estimate::EffectEstimate,
    config: &'a crate::inference::BayesianConfig,
    fitter: &'a antecedent_estimate::BayesianGComputationAte,
    prepared: &'a antecedent_estimate::PreparedBayesianProblem,
    status: antecedent_identify::IdentificationStatus,
}

/// Bayesian CATE target and its checked DAG proof, prior, and validation plan.
#[derive(Clone)]
pub(crate) struct CheckedBayesianConditionalOperation {
    query: antecedent_core::ConditionalEffectQuery,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    inference: InferenceMode,
    refute: RefuteSuite,
    graph: Dag,
    context: IdentifiedResultContext,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    latency: Option<LatencyMode>,
    validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    stage_sink: Option<Arc<dyn super::super::stage::StageResultSink>>,
}

/// The frozen premises of [`CheckedBayesianConditionalOperation::checked`], carried together.
pub(crate) struct CheckedBayesianConditionalOperationPremises<'a> {
    pub(crate) graph: &'a Dag,
    pub(crate) query: antecedent_core::ConditionalEffectQuery,
    pub(crate) cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    pub(crate) inference: InferenceMode,
    pub(crate) refute: RefuteSuite,
    pub(crate) context: IdentifiedResultContext,
    pub(crate) physical: std::sync::Arc<PhysicalExecutionPlan>,
    pub(crate) latency: Option<LatencyMode>,
    pub(crate) validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    pub(crate) stage_sink: Option<Arc<dyn super::super::stage::StageResultSink>>,
}

impl std::fmt::Debug for CheckedBayesianConditionalOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedBayesianConditionalOperation")
            .field("query", &self.query)
            .field("identification", &self.cache.identification)
            .field("estimand", &self.cache.estimand)
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

    /// Replace the caller validators run by this sealed operation. Names are
    /// checked by the prepared handle before the swap.
    pub(crate) fn set_custom_validators(
        &mut self,
        validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    ) {
        self.validators = validators;
    }

    pub(crate) fn checked(
        premises: CheckedBayesianConditionalOperationPremises<'_>,
    ) -> Result<Self, CausalError> {
        let CheckedBayesianConditionalOperationPremises {
            graph,
            query,
            cache,
            inference,
            refute,
            context,
            physical,
            latency,
            validators,
            stage_sink,
        } = premises;
        let (identification, estimand) = (&cache.identification, &cache.estimand);
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
            || !physical
                .binding()
                .names(IdentifierId::BackdoorAdjustment, EstimatorId::BayesianConditional)
            || !matches!(
                context.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            || !refute.is_point()
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
            || !proof_supplies(identification, estimand)
        {
            return Err(CausalError::Compile {
                message: "Bayesian CATE estimand is not supplied by its retained proof".into(),
            });
        }
        const PROOF_REFUSED: &str = "Bayesian CATE proof is not justified by its retained graph";
        if verify_retained_static(&physical, identification, estimand, PROOF_REFUSED)?
            == RetainedProof::NotRetained
        {
            let (checked_identification, checked_estimand) = select_claim(
                identify_static(IdentifierId::BackdoorAdjustment, graph, &query.inner)?,
                EstimatorId::BayesianConditional,
            )?;
            verify_proof_against_graph(
                &checked_identification,
                &checked_estimand,
                identification,
                estimand,
                PROOF_REFUSED,
            )?;
        }
        Ok(Self {
            query,
            cache,
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
        &self.cache.identification
    }

    pub(crate) fn estimand(&self) -> &IdentifiedEstimand {
        &self.cache.estimand
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

    /// Point and uncertainty stages: prepare the conditional design, resolve the prior, fit,
    /// and record the outcome-model assumption, with the stage clock and events around them.
    fn estimate_point(
        &self,
        projected: &TabularData,
        estimand_est: &IdentifiedEstimand,
        query_est: &antecedent_core::AverageEffectQuery,
        status: antecedent_identify::IdentificationStatus,
        clock: &mut super::super::stage::StageClock,
        ctx: &ExecutionContext,
    ) -> Result<CheckedBayesianConditionalPoint, CausalError> {
        let InferenceMode::Bayesian(config) = &self.inference else {
            return Err(CausalError::Compile {
                message: "checked Bayesian CATE lost its Bayesian inference configuration".into(),
            });
        };
        let conditional_est = antecedent_core::ConditionalEffectQuery::try_new(query_est.clone())?;
        let mut fitter = bayesian_gcomp(config, ctx);
        let prepared = fitter
            .prepare_conditional(projected, estimand_est, &conditional_est)
            .map_err(CausalError::from)?;
        let (prior, conflict) = resolve_bayesian_prior_with_conflict(config, &prepared, Some(ctx))?;
        fitter.prior = prior;
        let mut workspace = BayesianGCompWorkspace::default();
        clock.begin(ctx, super::super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        let mut posterior =
            fitter.fit(&prepared, status, &mut workspace, ctx).map_err(CausalError::from)?;
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
        Ok(CheckedBayesianConditionalPoint { fitter, prepared, workspace, posterior, estimate })
    }

    /// Validate stage: refutation suite and prior / posterior predictive checks when the
    /// suite is on, and under the full suite prior sensitivity plus MCMC diagnostics.
    fn validate(
        &self,
        input: &CheckedBayesianConditionalValidation<'_>,
        mut posterior: CausalPosterior,
        workspace: &mut BayesianGCompWorkspace,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<
        (CausalPosterior, Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>),
        CausalError,
    > {
        let CheckedBayesianConditionalValidation {
            data,
            estimand,
            query,
            estimate,
            config,
            fitter,
            prepared,
            status,
        } = *input;
        let mut refutations = Vec::new();
        let mut predictive_checks = Vec::new();
        let mut refute_workspace = EstimationWorkspace::default();
        if self.refute != RefuteSuite::None {
            let mean_query = {
                let mut query = query.clone();
                query.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
                query
            };
            let (reports, not_applicable) = RefutationRun {
                data,
                estimand,
                query: &mean_query,
                estimate,
                suite: self.refute,
                estimator: EstimatorId::BayesianConditional.as_str(),
                custom: &self.validators,
                temporal: None,
            }
            .reports(&mut refute_workspace, None, ctx)?;
            refutations = reports;
            diagnostics.extend(not_applicable);
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
        if self.refute == RefuteSuite::Full {
            let (summary, sensitivity) = evaluate_bayesian_prior_sensitivity(
                config, fitter, prepared, status, &posterior, workspace, ctx,
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
        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        let identification = self.cache.identification.clone();
        let estimand = self.cache.estimand.clone();
        clock.finish(super::super::stage::STAGE_IDENTIFY);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Identify {
                identification: identification.clone(),
                estimand: estimand.clone(),
            },
        );
        let InferenceMode::Bayesian(config) = &self.inference else {
            return Err(CausalError::Compile {
                message: "checked Bayesian CATE lost its Bayesian inference configuration".into(),
            });
        };
        // Fit on the identified columns only, as the one-shot route does;
        // the retained estimand keeps the caller's ids for the result.
        let full_cols = data.schema().len();
        let (projected, query_est, estimand_est) =
            project_for_ate_estimate(data, &self.query.inner, &estimand)?;
        let projected_cols = projected.schema().len();
        let CheckedBayesianConditionalPoint {
            fitter,
            prepared,
            mut workspace,
            posterior,
            estimate,
        } = self.estimate_point(
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
        clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
        let (posterior, refutations, predictive_checks) = self.validate(
            &CheckedBayesianConditionalValidation {
                data: &projected,
                estimand: &estimand_est,
                query: &query_est,
                estimate: &estimate,
                config,
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
        Ok(finish_identified_execute_with_context(
            &self.context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::BackdoorAdjustment, EstimatorId::BayesianConditional),
                self.query.inner.treatment,
                self.query.inner.outcome,
            )
            .identify_cached(true)
            .extra_diagnostics(diagnostics)
            .refutations(refutations)
            .wall_time_ns(clock.wall_time_ns())
            .cancelled(ctx.cancellation.is_cancelled())
            .early_stopped(posterior.early_stopped)
            .extras(IdentifiedExecuteExtras {
                stage_timings_ns: clock.timings(),
                posterior: Some(posterior),
                n_draws: draws,
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }
}

/// Inspection receipt for a prepared Bayesian DAG conditional-effect route.
#[derive(Clone, Debug)]
pub struct CheckedBayesianConditionalInfo {
    query: antecedent_core::ConditionalEffectQuery,
    inference: crate::inference::InferenceMode,
    validation: crate::analysis::builder::RefuteSuite,
    modifier_roles: std::sync::Arc<[antecedent_core::VariableId]>,
    identification_status: antecedent_identify::IdentificationStatus,
    estimand: antecedent_expr::IdentifiedEstimand,
}

impl CheckedBayesianConditionalInfo {
    /// Conditional causal target retained by preparation.
    #[must_use]
    pub fn query(&self) -> &antecedent_core::ConditionalEffectQuery {
        &self.query
    }
    /// Bayesian prior and inference settings retained by preparation.
    #[must_use]
    pub fn inference(&self) -> &crate::inference::InferenceMode {
        &self.inference
    }
    /// Validation suite retained by preparation.
    #[must_use]
    pub fn validation(&self) -> crate::analysis::builder::RefuteSuite {
        self.validation
    }
    /// Semantic modifier variables retained by preparation.
    #[must_use]
    pub fn modifier_roles(&self) -> &[antecedent_core::VariableId] {
        &self.modifier_roles
    }
    /// Identification status retained with the target.
    #[must_use]
    pub fn identification_status(&self) -> antecedent_identify::IdentificationStatus {
        self.identification_status
    }
    /// Identified target functional and adjustment roles.
    #[must_use]
    pub fn estimand(&self) -> &antecedent_expr::IdentifiedEstimand {
        &self.estimand
    }
}

impl Inspect for CheckedBayesianConditionalOperation {
    type Info = CheckedBayesianConditionalInfo;

    fn info(&self) -> CheckedBayesianConditionalInfo {
        CheckedBayesianConditionalInfo {
            query: self.query().clone(),
            inference: self.inference().clone(),
            validation: self.validation(),
            modifier_roles: Arc::from(self.modifier_roles()),
            identification_status: self.identification().status,
            estimand: self.estimand().clone(),
        }
    }
}
