//! Sealed Bayesian-bootstrap orthogonal ATE execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_estimate::bayesian_robust_ate::{
    BayesianRobustAteInput, BayesianRobustAteOptions, estimate_bayesian_robust_ate,
    plan_bayesian_robust_ate_folds,
};

/// Prepared Bayesian robust ATE, including its fixed row and fold identities.
#[derive(Clone)]
pub(crate) struct CheckedBayesianRobustAteExecution {
    graph: Dag,
    query: AverageEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    context: IdentifiedResultContext,
    physical: PhysicalExecutionPlan,
    schema: Arc<[(VariableId, Arc<str>)]>,
    row_ids: Arc<[u64]>,
    fold_ids: Arc<[u16]>,
    options: BayesianRobustAteOptions,
}

impl std::fmt::Debug for CheckedBayesianRobustAteExecution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedBayesianRobustAteExecution")
            .field("query", &self.query)
            .field("identification", &self.identification)
            .field("estimand", &self.estimand)
            .field("row_count", &self.row_ids.len())
            .field("fold_count", &self.options.folds)
            .field("graph", &self.graph)
            .finish_non_exhaustive()
    }
}

impl CheckedBayesianRobustAteExecution {
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
                message: "Bayesian robust ATE context has a different query family".into(),
            });
        };
        if target != &query
            || physical.logical.query != context.query
            || context.graph_class != GraphClass::Dag
            || !matches!(context.structure_source, crate::support::StructureSource::Explicit)
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::BayesianRobustAte.as_str())
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::BackdoorAdjustment.as_str())
        {
            return Err(CausalError::Compile {
                message:
                    "Bayesian robust ATE target, graph, or procedure changed after identification"
                        .into(),
            });
        }
        if query.target_population != antecedent_core::TargetPopulation::AllObserved
            || query.outcome_functional != antecedent_core::OutcomeFunctional::Mean
            || !query.effect_modifiers.is_empty()
            || query.control
                != antecedent_core::Intervention::set(
                    query.treatment,
                    antecedent_core::Value::f64(0.0),
                )
            || query.active
                != antecedent_core::Intervention::set(
                    query.treatment,
                    antecedent_core::Value::f64(1.0),
                )
        {
            return Err(CausalError::Unsupported {
                message: "bayesian.robust_ate supports only a binary mean ATE for AllObserved",
            });
        }
        let InferenceMode::Bayesian(config) = inference else {
            return Err(CausalError::Unsupported {
                message: "bayesian.robust_ate requires Bayesian inference",
            });
        };
        if config.prior.is_some()
            || config.prior_artifact.is_some()
            || config.external_compose.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "bayesian.robust_ate uses a Bayesian-bootstrap row law and does not accept coefficient or transferred priors",
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
                message: "Bayesian robust ATE estimand is not supplied by its retained proof"
                    .into(),
            });
        }
        let (checked_identification, checked_estimand) = select_claim(
            identify_static_query_with_rd(
                IdentifierId::BackdoorAdjustment,
                graph,
                &CausalQuery::AverageEffect(query.clone()),
                None,
            )?,
            EstimatorId::BayesianRobustAte,
        )?;
        if checked_identification.status != identification.status
            || checked_identification.query != identification.query
            || checked_estimand.functional != estimand.functional
            || checked_estimand.method != estimand.method
            || checked_estimand.adjustment_set != estimand.adjustment_set
        {
            return Err(CausalError::Compile {
                message: "Bayesian robust ATE proof is not justified by its retained graph".into(),
            });
        }
        let options = BayesianRobustAteOptions { draws: config.n_draws, ..Default::default() };
        let input = make_input(data, &query, &estimand)?;
        let fold_ids = plan_bayesian_robust_ate_folds(&input, options.folds)
            .map_err(|error| CausalError::Compile { message: error.to_string().into() })?;
        Ok(Self {
            graph: graph.clone(),
            query,
            identification,
            estimand,
            context,
            physical,
            schema: Arc::from(
                data.schema()
                    .variables()
                    .iter()
                    .map(|v| (v.id, Arc::clone(&v.name)))
                    .collect::<Vec<_>>(),
            ),
            row_ids: Arc::from(input.row_ids),
            fold_ids: Arc::from(fold_ids),
            options,
        })
    }

    pub(crate) fn query(&self) -> &AverageEffectQuery {
        &self.query
    }
    pub(crate) fn posterior_draws(&self) -> usize {
        self.options.draws
    }
    pub(crate) fn adjustment_set(&self) -> &[VariableId] {
        &self.estimand.adjustment_set
    }
    pub(crate) fn row_ids(&self) -> &[u64] {
        &self.row_ids
    }
    pub(crate) fn fold_ids(&self) -> &[u16] {
        &self.fold_ids
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let current_schema: Vec<_> =
            data.schema().variables().iter().map(|v| (v.id, Arc::clone(&v.name))).collect();
        if current_schema.as_slice() != self.schema.as_ref() {
            return Err(CausalError::Compile {
                message:
                    "Bayesian robust ATE refresh changed semantic variable IDs or column names"
                        .into(),
            });
        }
        let started = Instant::now();
        let input = make_input(data, &self.query, &self.estimand)?;
        if input.row_ids.as_slice() != self.row_ids.as_ref() {
            return Err(CausalError::Compile {
                message: "Bayesian robust ATE refresh changed its prepared row identity set".into(),
            });
        }
        let rebound_folds = plan_bayesian_robust_ate_folds(&input, self.options.folds)
            .map_err(|error| CausalError::Compile { message: error.to_string().into() })?;
        if rebound_folds.as_slice() != self.fold_ids.as_ref() {
            return Err(CausalError::Compile {
                message: "Bayesian robust ATE refresh changed its fixed fold assignment".into(),
            });
        }
        let kernel =
            estimate_bayesian_robust_ate(&input, self.options, ctx).map_err(CausalError::from)?;
        if kernel.fold_ids.as_slice() != self.fold_ids.as_ref() {
            return Err(CausalError::Compile {
                message: "Bayesian robust ATE kernel changed the retained fold assignment".into(),
            });
        }
        let mut assumptions = self.identification.required_assumptions.clone();
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
                id: Arc::from("bayesian.robust_ate.modular_bootstrap_pushforward"),
                description: Arc::from(format!("Posterior draws are modular Bayesian-bootstrap pushforwards: each Exp(1) row-weight draw refits cross-fitted nuisances and reweights the orthogonal AIPW score. {}. Fixed {} folds and clipping; no repeated-sampling calibration is implied.", kernel.model_identity, input_and_folds_note(&kernel.fold_ids))),
            }),
            source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("bayesian.robust_ate") },
            scope: antecedent_core::AssumptionScope::Estimation,
            status: antecedent_core::AssumptionStatus::Declared,
        });
        let draws = antecedent_prob::PosteriorDraws::from_column_major(
            antecedent_prob::PosteriorSchema {
                quantities: Arc::from([antecedent_prob::PosteriorQuantityKind::Effect {
                    name: Arc::from("ate"),
                }]),
            },
            kernel.draws.len(),
            Arc::<[f64]>::from(kernel.draws.clone()),
        )
        .map_err(|error| CausalError::Compile { message: error.to_string().into() })?;
        let posterior = CausalPosterior {
            summaries: draws.summarize(),
            draws,
            identification: self.identification.status,
            prior_sensitivity: None,
            conflict_summary: None,
            diagnostics: InferenceDiagnostics::analytic(
                "bayesian.robust_ate.modular_bootstrap_pushforward",
            ),
            assumptions: assumptions.clone(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            unevaluable_mass: 0.0,
            early_stopped: false,
            treatment_contrast: None,
        };
        let estimate = effect_from_posterior(&posterior)?;
        let diagnostic = Diagnostic::new(
            "estimate.bayesian.robust_ate.modular_bootstrap_pushforward",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "{}; interval is a modular bootstrap pushforward conditional on fixed row IDs, folds, linear nuisance bases and penalties",
                kernel.interval_interpretation
            ),
        );
        let args = IdentifiedExecuteFinish {
            physical: &self.physical,
            identification: self.identification.clone(),
            estimand: self.estimand.clone(),
            estimate,
            identifier_id: IdentifierId::BackdoorAdjustment,
            estimator_id: EstimatorId::BayesianRobustAte,
            treatment: self.query.treatment,
            outcome: self.query.outcome,
            identify_cached: true,
            extra_diagnostics: vec![diagnostic],
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                posterior: Some(posterior),
                n_draws: Some(u32::try_from(kernel.draws.len()).unwrap_or(u32::MAX)),
                estimate_provenance: Some((
                    Arc::from("estimate.bayesian_robust_ate"),
                    Arc::from("estimate.bayesian_robust_ate"),
                )),
                ..Default::default()
            },
        };
        Ok(finish_identified_execute_with_context(&self.context, Some(data), args))
    }
}

fn make_input(
    data: &TabularData,
    query: &AverageEffectQuery,
    estimand: &IdentifiedEstimand,
) -> Result<BayesianRobustAteInput, CausalError> {
    let (data, query, estimand) = project_for_ate_estimate(data, query, estimand)?;
    let raw_treatment = data.float64_values(query.treatment)?;
    if raw_treatment.iter().any(|value| *value != 0.0 && *value != 1.0) {
        return Err(CausalError::Unsupported {
            message: "bayesian.robust_ate requires treatment values coded exactly 0/1",
        });
    }
    let treatment = raw_treatment.iter().map(|value| *value == 1.0).collect();
    let outcome = data.float64_values(query.outcome)?;
    let covariates = estimand
        .adjustment_set
        .iter()
        .copied()
        .map(|variable| data.float64_values(variable))
        .collect::<Result<Vec<_>, _>>()?;
    let n = data.row_count();
    Ok(BayesianRobustAteInput { row_ids: (0..n as u64).collect(), treatment, outcome, covariates })
}

fn input_and_folds_note(folds: &[u16]) -> String {
    let count = folds.iter().copied().max().map_or(0, |max| usize::from(max) + 1);
    format!(
        "linear weighted ridge outcome and logistic weighted ridge propensity nuisances, {count}"
    )
}
