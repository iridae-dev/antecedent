//! Sealed Bayesian quadratic-basis conditional-effect execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::bayesian_basis_path::{
    BasisColumns, BasisEstimatorSettings, BasisRows, basis_posterior,
};
use super::{
    Arc, BayesianConfig, CausalError, CausalQuery, Dag, EstimatorId, ExecutionContext, GraphClass,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedResultContext, IdentifierId,
    InferenceMode, Instant, PhysicalExecutionPlan, Strategy, StudyResult, TableView, TabularData,
    VariableId, bayesian_draw_count, elapsed_ns, finish_identified_execute_with_context,
    identify_static, select_claim,
};
use crate::analysis::proof_support::verify_proof_against_graph;
use antecedent_core::ConditionalEffectQuery;
use antecedent_learn::{BasisTargetPopulation, BayesianBasisGComputation, BayesianBasisSpec};

/// Checked CATE plan for the basis estimator. This initial contract conditions
/// on every adjustment variable, matching the row-wise predictions produced by
/// the fitted basis model.
#[derive(Clone)]
pub(crate) struct CheckedBayesianBasisCateExecution {
    graph: Dag,
    query: ConditionalEffectQuery,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    config: BayesianConfig,
    context: IdentifiedResultContext,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    schema: Arc<[(VariableId, Arc<str>)]>,
}

impl std::fmt::Debug for CheckedBayesianBasisCateExecution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedBayesianBasisCateExecution")
            .field("query", &self.query)
            .field("identification", &self.cache.identification)
            .field("estimand", &self.cache.estimand)
            .field("estimator", &EstimatorId::BayesianBasisGcomp)
            .field("graph", &self.graph)
            .finish_non_exhaustive()
    }
}

impl CheckedBayesianBasisCateExecution {
    pub(crate) fn checked(
        data: &TabularData,
        graph: &Dag,
        query: ConditionalEffectQuery,
        cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
        inference: InferenceMode,
        context: IdentifiedResultContext,
        physical: std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let (identification, estimand) = (&cache.identification, &cache.estimand);
        let CausalQuery::ConditionalEffect(target) = &context.query else {
            return Err(CausalError::Compile {
                message: "Bayesian basis CATE context has a different query family".into(),
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
                message:
                    "Bayesian basis CATE target, graph, or procedure changed after identification"
                        .into(),
            });
        }
        let inner = &query.inner;
        if inner.target_population != antecedent_core::TargetPopulation::AllObserved
            || !matches!(inner.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            || !matches!(inner.control, antecedent_core::Intervention::Set { variable, ref value } if variable == inner.treatment && value.as_f64() == Some(0.0))
            || !matches!(inner.active, antecedent_core::Intervention::Set { variable, ref value } if variable == inner.treatment && value.as_f64() == Some(1.0))
        {
            return Err(CausalError::Unsupported {
                message: "checked Bayesian basis CATE requires binary mean effects over AllObserved",
            });
        }
        let InferenceMode::Bayesian(config) = inference else {
            return Err(CausalError::unsupported_inference(
                "bayesian.basis.gcomp requires inference=Bayesian",
            ));
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
        let mut modifiers = inner.effect_modifiers.to_vec();
        let mut adjustment = estimand.adjustment_set.to_vec();
        modifiers.sort_unstable();
        adjustment.sort_unstable();
        if modifiers.is_empty() || modifiers != adjustment {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp CATE requires effect modifiers to equal the complete adjustment set",
            });
        }
        let (checked_identification, checked_estimand) = select_claim(
            identify_static(IdentifierId::BackdoorAdjustment, graph, inner)?,
            EstimatorId::BayesianBasisGcomp,
        )?;
        verify_proof_against_graph(
            &checked_identification,
            &checked_estimand,
            identification,
            estimand,
            "Bayesian basis CATE proof is not justified by its retained graph",
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
                    .map(|v| (v.id, Arc::clone(&v.name)))
                    .collect::<Vec<_>>(),
            ),
        })
    }

    pub(crate) fn query(&self) -> &ConditionalEffectQuery {
        &self.query
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let schema: Vec<_> =
            data.schema().variables().iter().map(|v| (v.id, Arc::clone(&v.name))).collect();
        if schema.as_slice() != self.schema.as_ref() {
            return Err(CausalError::Compile {
                message:
                    "Bayesian basis CATE refresh changed semantic variable IDs or column names"
                        .into(),
            });
        }
        let inner = &self.query.inner;
        let rows = BasisRows::complete(
            data,
            BasisColumns {
                treatment: inner.treatment,
                outcome: inner.outcome,
                adjustment_set: &self.cache.estimand.adjustment_set,
            },
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
            &self.cache.identification,
            BasisEstimatorSettings {
                prior_scale: self.config.prior_scale,
                treatment_contrast: 1.0,
                estimator_id: model.estimator_id(),
            },
            "Gaussian identity-link outcome regression with quadratic covariate terms and treatment-by-covariate interactions.",
        )?;
        let args = IdentifiedExecuteFinish::new(
            &self.physical,
            self.cache.identification.clone(),
            self.cache.estimand.clone(),
            estimate,
            Strategy::new(IdentifierId::BackdoorAdjustment, EstimatorId::BayesianBasisGcomp),
            inner.treatment,
            inner.outcome,
        )
        .identify_cached(true)
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
