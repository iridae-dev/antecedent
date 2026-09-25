//! Sealed Bayesian quadratic-basis ATE execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    ParametricAssumption, PriorAssumption, TargetPopulation, VariableId,
};
use antecedent_learn::{BasisTargetPopulation, BayesianBasisGComputation, BayesianBasisSpec};

/// The checked graph proof, target, prior and physical plan for the licensed basis ATE.
#[derive(Clone)]
pub(crate) struct CheckedBayesianBasisAteExecution {
    graph: Dag,
    query: AverageEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    config: BayesianConfig,
    context: IdentifiedResultContext,
    physical: PhysicalExecutionPlan,
    schema: Arc<[(VariableId, Arc<str>)]>,
}

impl std::fmt::Debug for CheckedBayesianBasisAteExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedBayesianBasisAteExecution")
            .field("query", &self.query)
            .field("identification", &self.identification)
            .field("estimand", &self.estimand)
            .field("estimator", &EstimatorId::BayesianBasisGcomp)
            .field("graph", &self.graph)
            .finish_non_exhaustive()
    }
}

impl CheckedBayesianBasisAteExecution {
    pub(crate) fn checked(
        data: &TabularData,
        graph: &Dag,
        query: AverageEffectQuery,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        inference: InferenceMode,
        context: IdentifiedResultContext,
        physical: PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::AverageEffect(target) = &context.query else {
            return Err(CausalError::Compile {
                message: "Bayesian basis result context has a different query family".into(),
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
            || !identification.estimands.iter().any(|candidate| {
                candidate.functional == estimand.functional
                    && candidate.method == estimand.method
                    && candidate.adjustment_set == estimand.adjustment_set
            })
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
        if checked_identification.status != identification.status
            || checked_identification.query != identification.query
            || checked_estimand.functional != estimand.functional
            || checked_estimand.method != estimand.method
            || checked_estimand.adjustment_set != estimand.adjustment_set
        {
            return Err(CausalError::Compile {
                message: "Bayesian basis proof is not justified by its retained graph".into(),
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
                    .map(|variable| (variable.id, Arc::clone(&variable.name)))
                    .collect::<Vec<_>>(),
            ),
        })
    }

    pub(crate) fn info(&self) -> super::super::prepared::CheckedBayesianBasisAteInfo {
        super::super::prepared::CheckedBayesianBasisAteInfo {
            query: self.query.clone(),
            adjustment_set: Arc::clone(&self.estimand.adjustment_set),
            posterior_draws: self.config.n_draws,
            prior_scale: self.config.prior_scale,
        }
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
        let identification = self.identification.clone();
        let estimand = self.estimand.clone();
        let (data_est, query_est, estimand_est) = project_for_ate_estimate(data, query, &estimand)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            &data_est,
            query_est.outcome,
            &query_est.outcome_functional,
        )?;
        let treatment = data_est.float64_values(query_est.treatment)?;
        let outcome = data_est.float64_values(query_est.outcome)?;
        let covariates = estimand_est
            .adjustment_set
            .iter()
            .copied()
            .map(|variable| data_est.float64_values(variable))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = (0..data_est.row_count())
            .filter(|row| {
                treatment[*row].is_finite()
                    && outcome[*row].is_finite()
                    && covariates.iter().all(|column| column[*row].is_finite())
            })
            .collect::<Vec<_>>();
        if rows.len() < 4 {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp requires at least four complete observed rows",
            });
        }
        let treatment = rows.iter().map(|row| treatment[*row]).collect::<Vec<_>>();
        let outcome = rows.iter().map(|row| outcome[*row]).collect::<Vec<_>>();
        let covariates_colmajor = covariates
            .iter()
            .flat_map(|column| rows.iter().map(|row| column[*row]))
            .collect::<Vec<_>>();
        let row_ids = rows
            .iter()
            .map(|row| {
                u32::try_from(*row).map_err(|_| CausalError::Compile {
                    message: "bayesian.basis.gcomp row index exceeds row id capacity".into(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let model = BayesianBasisGComputation;
        let fit = model
            .fit(
                &treatment,
                &outcome,
                &covariates_colmajor,
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
            .map_err(|error| CausalError::Compile { message: error.to_string().into() })?;

        let n_draws = fit.ate_draws.len();
        let mut quantities = Vec::with_capacity(rows.len() + 1);
        quantities.push(antecedent_prob::PosteriorQuantityKind::Effect { name: Arc::from("ate") });
        quantities.extend((0..rows.len()).map(|row| {
            antecedent_prob::PosteriorQuantityKind::Effect {
                name: Arc::from(format!("cate.row.{}", row_ids[row])),
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
        .map_err(|error| CausalError::Compile { message: error.to_string().into() })?;
        let summaries = draws.summarize();
        let (control, active) = (0.0, 1.0);
        let mut assumptions = identification.required_assumptions.clone();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(ParametricAssumption {
                id: Arc::from("bayesian.basis_gcomp.quadratic_outcome"),
                description: Arc::from("Gaussian identity-link outcome regression with quadratic covariate terms and treatment-by-covariate interactions."),
            }),
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(model.estimator_id()) },
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
        assumptions.push(AssumptionRecord {
            assumption: Assumption::PriorRestriction(PriorAssumption {
                id: Arc::from(format!("gaussian_coefficients_isotropic_sd_{}", self.config.prior_scale)),
                description: Arc::from("Every basis coefficient, including the intercept, has a zero-centered Gaussian prior scaled by residual standard deviation."),
            }),
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(model.estimator_id()) },
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
        let posterior = CausalPosterior {
            draws,
            summaries: summaries.clone(),
            identification: identification.status,
            prior_sensitivity: None,
            conflict_summary: None,
            diagnostics: antecedent_prob::InferenceDiagnostics::analytic("conjugate_gaussian"),
            assumptions: assumptions.clone(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            unevaluable_mass: 0.0,
            early_stopped: false,
            treatment_contrast: Some(active - control),
        };
        let mut estimate = EffectEstimate::new(
            summaries.mean[0],
            summaries.sd[0],
            assumptions,
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(rows.len() as u64);
        estimate.cate =
            Some(Arc::from((0..rows.len()).map(|row| summaries.mean[row + 1]).collect::<Vec<_>>()));
        let diagnostics = vec![Diagnostic::new(
            "estimate.bayesian_basis_gcomp.prediction_diagnostics",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "draws={n_draws}; rows={}; targets={}; min={}; max={}; model={}; prior={}",
                fit.predictions.diagnostics.row_count,
                fit.predictions.diagnostics.target_count,
                fit.predictions.diagnostics.minimum,
                fit.predictions.diagnostics.maximum,
                fit.predictions.provenance.model_id,
                fit.predictions.provenance.prior_id,
            ),
        )];
        let args = IdentifiedExecuteFinish {
            physical: &self.physical,
            identification,
            estimand: estimand_est,
            estimate,
            identifier_id: IdentifierId::BackdoorAdjustment,
            estimator_id: EstimatorId::BayesianBasisGcomp,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached: true,
            extra_diagnostics: diagnostics,
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
