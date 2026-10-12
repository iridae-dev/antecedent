//! Sealed Bayesian quadratic-basis ATE execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::bayesian_basis_path::{
    BasisColumns, BasisEstimatorSettings, BasisRows, basis_posterior, basis_prediction_diagnostic,
};
use super::{
    Arc, AverageEffectQuery, BayesianConfig, CausalError, CausalQuery, Dag, EstimatorId,
    ExecutionContext, GraphClass, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedResultContext, IdentifierId, InferenceMode, Instant, PhysicalExecutionPlan, Strategy,
    StudyResult, TableView, TabularData, bayesian_draw_count, elapsed_ns,
    finish_identified_execute_with_context, identify_static_query, project_for_ate_estimate,
    select_claim,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::{proof_supplies, verify_proof_against_graph};
use antecedent_core::{TargetPopulation, VariableId};
use antecedent_learn::{BasisTargetPopulation, BayesianBasisGComputation, BayesianBasisSpec};

/// The checked graph proof, target, prior and physical plan for the licensed basis ATE.
#[derive(Clone)]
pub(crate) struct CheckedBayesianBasisAteExecution {
    graph: Dag,
    query: AverageEffectQuery,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    config: BayesianConfig,
    context: IdentifiedResultContext,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    schema: Arc<[(VariableId, Arc<str>)]>,
}

impl std::fmt::Debug for CheckedBayesianBasisAteExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedBayesianBasisAteExecution")
            .field("query", &self.query)
            .field("identification", &self.cache.identification)
            .field("estimand", &self.cache.estimand)
            .field("estimator", &EstimatorId::BayesianBasisGcomp)
            .field("graph", &self.graph)
            .finish_non_exhaustive()
    }
}

impl CheckedBayesianBasisAteExecution {
    pub(crate) fn with_inference_config(
        &self,
        config: BayesianConfig,
    ) -> Result<Self, CausalError> {
        if config.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
            || config.likelihood != antecedent_prob::BayesLikelihood::GaussianIdentity
            || config.n_draws < 2
            || !config.prior_scale.is_finite()
            || config.prior_scale <= 0.0
            || config.prior.is_some()
            || config.prior_artifact.is_some()
            || config.external_compose.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp requires its declared conjugate Gaussian shrinkage prior",
            });
        }
        let mut next = self.clone();
        next.context.inference = InferenceMode::Bayesian(config.clone());
        next.config = config;
        Ok(next)
    }

    pub(crate) fn checked(
        data: &TabularData,
        graph: &Dag,
        query: AverageEffectQuery,
        cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
        inference: InferenceMode,
        context: IdentifiedResultContext,
        physical: std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let (identification, estimand) = (&cache.identification, &cache.estimand);
        let CausalQuery::AverageEffect(target) = &context.query else {
            return Err(CausalError::Compile {
                message: "Bayesian basis result context has a different query family".into(),
            });
        };
        if target != &query
            || physical.logical.query != context.query
            || context.graph_class != GraphClass::Dag
            || !matches!(context.structure_source, crate::support::StructureSource::Explicit)
            || !physical
                .binding()
                .names(IdentifierId::BackdoorAdjustment, EstimatorId::BayesianBasisGcomp)
        {
            return Err(CausalError::Compile {
                message: "Bayesian basis target, graph, or procedure changed after identification"
                    .into(),
            });
        }
        if query.target_population != TargetPopulation::AllObserved
            || !matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            || !matches!(query.control, antecedent_core::Intervention::Set { variable, ref value }
                if variable == query.treatment && value.as_f64() == Some(0.0))
            || !matches!(query.active, antecedent_core::Intervention::Set { variable, ref value }
                if variable == query.treatment && value.as_f64() == Some(1.0))
        {
            return Err(CausalError::Unsupported {
                message: "checked Bayesian basis ATE requires binary mean ATE over AllObserved",
            });
        }
        let InferenceMode::Bayesian(config) = inference else {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp requires inference=Bayesian",
            });
        };
        if config.likelihood != antecedent_prob::BayesLikelihood::GaussianIdentity
            || config.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
            || config.prior.is_some()
            || config.prior_artifact.is_some()
            || config.external_compose.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp requires its declared conjugate Gaussian shrinkage prior",
            });
        }
        if identification.average_effect() != Some(&query)
            || !proof_supplies(identification, estimand)
        {
            return Err(CausalError::Compile {
                message: "Bayesian basis estimand is not supplied by its retained proof".into(),
            });
        }
        let (checked_identification, checked_estimand) = select_claim(
            identify_static_query(
                IdentifierId::BackdoorAdjustment,
                graph,
                &CausalQuery::AverageEffect(query.clone()),
            )?,
            EstimatorId::BayesianBasisGcomp,
        )?;
        verify_proof_against_graph(
            &checked_identification,
            &checked_estimand,
            identification,
            estimand,
            "Bayesian basis proof is not justified by its retained graph",
        )?;
        Ok(Self {
            graph: graph.clone(),
            query,
            cache,
            config,
            context,
            physical,
            schema: Arc::from(
                data.schema()
                    .variables()
                    .iter()
                    .map(|variable| (variable.id, Arc::clone(&variable.name)))
                    .collect::<Vec<_>>(),
            ),
        })
    }

    pub(crate) fn inference(&self) -> InferenceMode {
        InferenceMode::Bayesian(self.config.clone())
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let current_schema: Vec<_> = data
            .schema()
            .variables()
            .iter()
            .map(|variable| (variable.id, Arc::clone(&variable.name)))
            .collect();
        if current_schema.as_slice() != self.schema.as_ref() {
            return Err(CausalError::Compile {
                message: "Bayesian basis ATE refresh changed semantic variable IDs or column names"
                    .into(),
            });
        }
        let started = Instant::now();
        let query = &self.query;
        let identification = self.cache.identification.clone();
        let estimand = self.cache.estimand.clone();
        let (data_est, query_est, estimand_est) = project_for_ate_estimate(data, query, &estimand)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            &data_est,
            query_est.outcome,
            &query_est.outcome_functional,
        )?;
        let rows = BasisRows::complete(
            &data_est,
            BasisColumns {
                treatment: query_est.treatment,
                outcome: query_est.outcome,
                adjustment_set: &estimand_est.adjustment_set,
            },
            "bayesian.basis.gcomp row index exceeds row id capacity",
        )?;
        let model = BayesianBasisGComputation;
        let fit = model.fit(
            &rows.as_basis_data(),
            BasisTargetPopulation::AllObserved,
            BayesianBasisSpec {
                prior_sd: self.config.prior_scale,
                n_draws: bayesian_draw_count(&InferenceMode::Bayesian(self.config.clone()))?,
                seed: ctx.rng.master_seed(),
            },
            ctx,
        )?;

        let n_draws = fit.ate_draws.len();
        let (posterior, estimate) = basis_posterior(
            &fit,
            &rows,
            &identification,
            BasisEstimatorSettings {
                prior_scale: self.config.prior_scale,
                treatment_contrast: 1.0,
                estimator_id: model.estimator_id(),
            },
            "Gaussian identity-link outcome regression with quadratic covariate terms and treatment-by-covariate interactions.",
        )?;
        let diagnostics = vec![basis_prediction_diagnostic(&fit)];
        let args = IdentifiedExecuteFinish::new(
            &self.physical,
            identification,
            estimand_est,
            estimate,
            Strategy::new(IdentifierId::BackdoorAdjustment, EstimatorId::BayesianBasisGcomp),
            query.treatment,
            query.outcome,
        )
        .identify_cached(true)
        .extra_diagnostics(diagnostics)
        .wall_time_ns(elapsed_ns(started))
        .cancelled(ctx.cancellation.is_cancelled())
        .extras(IdentifiedExecuteExtras {
            posterior: Some(posterior),
            n_draws: Some(u32::try_from(n_draws).unwrap_or(u32::MAX)),
            bootstrap_replicates_requested: Some(None),
            estimate_provenance: Some((
                Arc::from("estimate.bayesian_basis_gcomp"),
                Arc::from(model.estimator_id()),
            )),
            ..Default::default()
        });
        Ok(finish_identified_execute_with_context(&self.context, Some(data), args))
    }
}

/// Inspection receipt for the checked Bayesian quadratic-basis ATE route.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedBayesianBasisAteInfo {
    /// Target query retained by preparation.
    pub query: antecedent_core::AverageEffectQuery,
    /// Back-door adjustment variables used by the basis model.
    pub adjustment_set: std::sync::Arc<[antecedent_core::VariableId]>,
    /// Number of posterior draws used by execution.
    pub posterior_draws: usize,
    /// Isotropic coefficient-prior scale retained by preparation.
    pub prior_scale: f64,
}

impl Inspect for CheckedBayesianBasisAteExecution {
    type Info = CheckedBayesianBasisAteInfo;

    fn info(&self) -> CheckedBayesianBasisAteInfo {
        CheckedBayesianBasisAteInfo {
            query: self.query.clone(),
            adjustment_set: Arc::clone(&self.cache.estimand.adjustment_set),
            posterior_draws: self.config.n_draws,
            prior_scale: self.config.prior_scale,
        }
    }
}
