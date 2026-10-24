//! Sealed Bayesian-bootstrap orthogonal ATE execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AverageEffectQuery, CausalError, CausalPosterior, CausalQuery, Dag, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EstimatorId, ExecutionContext, GraphClass,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedResultContext,
    IdentifierId, InferenceDiagnostics, InferenceMode, Instant, PhysicalExecutionPlan, Strategy,
    StudyResult, TableView, TabularData, VariableId, effect_from_posterior, elapsed_ns,
    finish_identified_execute_with_context, identify_static_query_with_rd,
    project_for_ate_estimate, select_claim,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::{
    RetainedProof, proof_supplies, verify_proof_against_graph, verify_retained_static,
};
use antecedent_estimate::bayesian_robust_ate::{
    BayesianRobustAteInput, BayesianRobustAteOptions, estimate_bayesian_robust_ate,
    plan_bayesian_robust_ate_folds,
};

/// Prepared Bayesian robust ATE, including its fixed row and fold identities.
#[derive(Clone)]
pub(crate) struct CheckedBayesianRobustAteExecution {
    graph: Dag,
    query: AverageEffectQuery,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    context: IdentifiedResultContext,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    schema: Arc<[(VariableId, Arc<str>)]>,
    row_ids: Arc<[u64]>,
    fold_ids: Arc<[u16]>,
    options: BayesianRobustAteOptions,
}

impl std::fmt::Debug for CheckedBayesianRobustAteExecution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedBayesianRobustAteExecution")
            .field("query", &self.query)
            .field("identification", &self.cache.identification)
            .field("estimand", &self.cache.estimand)
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
        cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
        inference: InferenceMode,
        context: IdentifiedResultContext,
        physical: std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let (identification, estimand) = (&cache.identification, &cache.estimand);
        let CausalQuery::AverageEffect(target) = &context.query else {
            return Err(CausalError::Compile {
                message: "Bayesian robust ATE context has a different query family".into(),
            });
        };
        if target != &query
            || physical.logical.query != context.query
            || context.graph_class != GraphClass::Dag
            || !matches!(context.structure_source, crate::support::StructureSource::Explicit)
            || !physical
                .binding()
                .names(IdentifierId::BackdoorAdjustment, EstimatorId::BayesianRobustAte)
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
            return Err(CausalError::unsupported_inference(
                "bayesian.robust_ate requires Bayesian inference",
            ));
        };
        if config.prior.is_some()
            || config.prior_artifact.is_some()
            || config.external_compose.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "bayesian.robust_ate uses a Bayesian-bootstrap row law and does not accept coefficient or transferred priors",
            });
        }
        verify_robust_ate_proof(graph, &query, identification, estimand, &physical)?;
        let options = BayesianRobustAteOptions { draws: config.n_draws, ..Default::default() };
        let input = make_input(data, &query, estimand)?;
        let fold_ids =
            plan_bayesian_robust_ate_folds(&input, options.folds).map_err(CausalError::rejected)?;
        Ok(Self {
            graph: graph.clone(),
            query,
            cache,
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
        &self.cache.estimand.adjustment_set
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
        let input = make_input(data, &self.query, &self.cache.estimand)?;
        if input.row_ids.as_slice() != self.row_ids.as_ref() {
            return Err(CausalError::Compile {
                message: "Bayesian robust ATE refresh changed its prepared row identity set".into(),
            });
        }
        let rebound_folds = plan_bayesian_robust_ate_folds(&input, self.options.folds)
            .map_err(CausalError::rejected)?;
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
        let mut assumptions = self.cache.identification.required_assumptions.clone();
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
        )?;
        let posterior = CausalPosterior {
            summaries: draws.summarize(),
            draws,
            identification: self.cache.identification.status,
            prior_sensitivity: None,
            conflict_summary: None,
            diagnostics: InferenceDiagnostics::analytic(super::ROBUST_ATE_BOOTSTRAP_BACKEND),
            assumptions: assumptions.clone(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            unevaluable_mass: 0.0,
            early_stopped: false,
            treatment_contrast: None,
            preparation: None,
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
        let args = IdentifiedExecuteFinish::new(
            &self.physical,
            self.cache.identification.clone(),
            self.cache.estimand.clone(),
            estimate,
            Strategy::new(IdentifierId::BackdoorAdjustment, EstimatorId::BayesianRobustAte),
            self.query.treatment,
            self.query.outcome,
        )
        .identify_cached(true)
        .extra_diagnostics(vec![diagnostic])
        .wall_time_ns(elapsed_ns(started))
        .cancelled(ctx.cancellation.is_cancelled())
        .extras(IdentifiedExecuteExtras {
            posterior: Some(posterior),
            n_draws: Some(u32::try_from(kernel.draws.len()).unwrap_or(u32::MAX)),
            estimate_provenance: Some((
                Arc::from("estimate.bayesian_robust_ate"),
                Arc::from("estimate.bayesian_robust_ate"),
            )),
            ..Default::default()
        });
        Ok(finish_identified_execute_with_context(&self.context, Some(data), args))
    }
}

/// The retained identification supplies the query's estimand, and the back-door proof
/// re-derives the same claim from the retained graph.
fn verify_robust_ate_proof(
    graph: &Dag,
    query: &AverageEffectQuery,
    identification: &antecedent_identify::IdentificationResult,
    estimand: &IdentifiedEstimand,
    physical: &PhysicalExecutionPlan,
) -> Result<(), CausalError> {
    if identification.average_effect() != Some(query) || !proof_supplies(identification, estimand) {
        return Err(CausalError::Compile {
            message: "Bayesian robust ATE estimand is not supplied by its retained proof".into(),
        });
    }
    const PROOF_REFUSED: &str = "Bayesian robust ATE proof is not justified by its retained graph";
    if verify_retained_static(physical, identification, estimand, PROOF_REFUSED)?
        == RetainedProof::Sealed
    {
        return Ok(());
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
    verify_proof_against_graph(
        &checked_identification,
        &checked_estimand,
        identification,
        estimand,
        PROOF_REFUSED,
    )
}

#[expect(clippy::float_cmp, reason = "the Bernoulli treatment contract requires exact 0/1 coding")]
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

/// Target and fixed row/fold bindings for the checked robust Bayesian ATE.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedBayesianRobustAteInfo {
    /// Mean binary ATE retained by preparation.
    pub query: antecedent_core::AverageEffectQuery,
    /// Number of bootstrap pushforward draws requested.
    pub posterior_draws: usize,
    /// Back-door adjustment variables bound to the nuisance design.
    pub adjustment_set: std::sync::Arc<[antecedent_core::VariableId]>,
    /// Stable row identities bound to the prepared fold plan.
    pub row_ids: std::sync::Arc<[u64]>,
    /// Treatment-stratified fold identity in row order.
    pub fold_ids: std::sync::Arc<[u16]>,
}

impl Inspect for CheckedBayesianRobustAteExecution {
    type Info = CheckedBayesianRobustAteInfo;

    fn info(&self) -> CheckedBayesianRobustAteInfo {
        CheckedBayesianRobustAteInfo {
            query: self.query().clone(),
            posterior_draws: self.posterior_draws(),
            adjustment_set: Arc::from(self.adjustment_set()),
            row_ids: Arc::from(self.row_ids()),
            fold_ids: Arc::from(self.fold_ids()),
        }
    }
}
