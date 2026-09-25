//! Sealed Bayesian quadratic-basis conditional-effect execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    ConditionalEffectQuery, ParametricAssumption, PriorAssumption,
};
use antecedent_learn::{BasisTargetPopulation, BayesianBasisGComputation, BayesianBasisSpec};

/// Checked CATE plan for the basis estimator. This initial contract conditions
/// on every adjustment variable, matching the row-wise predictions produced by
/// the fitted basis model.
#[derive(Clone)]
pub(crate) struct CheckedBayesianBasisCateExecution {
    graph: Dag,
    query: ConditionalEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    config: BayesianConfig,
    context: IdentifiedResultContext,
    physical: PhysicalExecutionPlan,
    schema: Arc<[(VariableId, Arc<str>)]>,
}

impl std::fmt::Debug for CheckedBayesianBasisCateExecution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedBayesianBasisCateExecution")
            .field("query", &self.query)
            .field("identification", &self.identification)
            .field("estimand", &self.estimand)
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
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        inference: InferenceMode,
        context: IdentifiedResultContext,
        physical: PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::ConditionalEffect(target) = &context.query else {
            return Err(CausalError::Compile {
                message: "Bayesian basis CATE context has a different query family".into(),
            });
        };
        if target != &query
            || physical.logical.query != context.query
            || context.graph_class != GraphClass::Dag
            || !matches!(context.structure_source, crate::support::StructureSource::Explicit)
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::BayesianBasisGcomp.as_str())
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::BackdoorAdjustment.as_str())
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
        if checked_identification.status != identification.status
            || checked_identification.query != identification.query
            || checked_estimand.functional != estimand.functional
            || checked_estimand.method != estimand.method
            || checked_estimand.adjustment_set != estimand.adjustment_set
        {
            return Err(CausalError::Compile {
                message: "Bayesian basis CATE proof is not justified by its retained graph".into(),
            });
        }
        Ok(Self {
            graph: graph.clone(),
            query,
            identification,
            estimand,
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
        let treatment = data.float64_values(inner.treatment)?;
        let outcome = data.float64_values(inner.outcome)?;
        let covariates = self
            .estimand
            .adjustment_set
            .iter()
            .copied()
            .map(|v| data.float64_values(v))
            .collect::<Result<Vec<_>, _>>()?;
        let rows: Vec<_> = (0..data.row_count())
            .filter(|r| {
                treatment[*r].is_finite()
                    && outcome[*r].is_finite()
                    && covariates.iter().all(|x| x[*r].is_finite())
            })
            .collect();
        if rows.len() < 4 {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp requires at least four complete observed rows",
            });
        }
        let t: Vec<_> = rows.iter().map(|r| treatment[*r]).collect();
        let y: Vec<_> = rows.iter().map(|r| outcome[*r]).collect();
        let x: Vec<_> = covariates.iter().flat_map(|col| rows.iter().map(|r| col[*r])).collect();
        let row_ids: Vec<u32> = rows
            .iter()
            .map(|r| {
                u32::try_from(*r).map_err(|_| CausalError::Compile {
                    message: "basis CATE row ID exceeds capacity".into(),
                })
            })
            .collect::<Result<_, _>>()?;
        let model = BayesianBasisGComputation;
        let fit = model
            .fit(
                &t,
                &y,
                &x,
                covariates.len(),
                Some(&row_ids),
                None,
                BasisTargetPopulation::AllObserved,
                BayesianBasisSpec {
                    prior_sd: self.config.prior_scale,
                    n_draws: bayesian_draw_count(&InferenceMode::Bayesian(self.config.clone()))?,
                    seed: ctx.rng.master_seed(),
                },
                ctx,
            )
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let n_draws = fit.ate_draws.len();
        let mut quantities = Vec::with_capacity(rows.len() + 1);
        quantities.push(antecedent_prob::PosteriorQuantityKind::Effect { name: Arc::from("ate") });
        quantities.extend(row_ids.iter().map(|row| {
            antecedent_prob::PosteriorQuantityKind::Effect {
                name: Arc::from(format!("cate.row.{row}")),
            }
        }));
        let mut columns = Vec::with_capacity((rows.len() + 1) * n_draws);
        columns.extend_from_slice(&fit.ate_draws);
        for row in 0..rows.len() {
            for draw in 0..n_draws {
                columns.push(fit.predictions.values[draw * 3 * rows.len() + 2 * rows.len() + row]);
            }
        }
        let draws = antecedent_prob::PosteriorDraws::from_column_major(
            antecedent_prob::PosteriorSchema { quantities: Arc::from(quantities) },
            n_draws,
            Arc::<[f64]>::from(columns),
        )
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let summaries = draws.summarize();
        let mut assumptions = self.identification.required_assumptions.clone();
        assumptions.push(AssumptionRecord { assumption: Assumption::ParametricRestriction(ParametricAssumption { id: Arc::from("bayesian.basis_gcomp.quadratic_outcome"), description: Arc::from("Gaussian identity-link outcome regression with quadratic covariate terms and treatment-by-covariate interactions.") }), source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(model.estimator_id()) }, scope: AssumptionScope::Estimation, status: AssumptionStatus::Declared });
        assumptions.push(AssumptionRecord { assumption: Assumption::PriorRestriction(PriorAssumption { id: Arc::from(format!("gaussian_coefficients_isotropic_sd_{}", self.config.prior_scale)), description: Arc::from("Every basis coefficient, including the intercept, has a zero-centered Gaussian prior scaled by residual standard deviation.") }), source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(model.estimator_id()) }, scope: AssumptionScope::Estimation, status: AssumptionStatus::Declared });
        let posterior = CausalPosterior {
            draws,
            summaries: summaries.clone(),
            identification: self.identification.status,
            prior_sensitivity: None,
            conflict_summary: None,
            diagnostics: antecedent_prob::InferenceDiagnostics::analytic("conjugate_gaussian"),
            assumptions: assumptions.clone(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            unevaluable_mass: 0.0,
            early_stopped: false,
            treatment_contrast: Some(1.0),
        };
        let mut estimate = EffectEstimate::new(
            summaries.mean[0],
            summaries.sd[0],
            assumptions,
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(rows.len() as u64);
        estimate.cate =
            Some(Arc::from((0..rows.len()).map(|r| summaries.mean[r + 1]).collect::<Vec<_>>()));
        let args = IdentifiedExecuteFinish {
            physical: &self.physical,
            identification: self.identification.clone(),
            estimand: self.estimand.clone(),
            estimate,
            identifier_id: IdentifierId::BackdoorAdjustment,
            estimator_id: EstimatorId::BayesianBasisGcomp,
            treatment: inner.treatment,
            outcome: inner.outcome,
            identify_cached: true,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                posterior: Some(posterior),
                n_draws: Some(u32::try_from(n_draws).unwrap_or(u32::MAX)),
                bootstrap_replicates_requested: Some(None),
                estimate_provenance: Some((
                    Arc::from("estimate.bayesian_basis_gcomp"),
                    Arc::from(model.estimator_id()),
                )),
                ..Default::default()
            },
        };
        Ok(finish_identified_execute_with_context(&self.context, Some(data), args))
    }
}
