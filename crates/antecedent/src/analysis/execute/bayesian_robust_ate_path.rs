// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AverageEffectQuery, CausalError, CausalPosterior, CausalQuery, Dag, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EstimatorId, ExecutionContext, IdentifiedExecuteFinish,
    InferenceDiagnostics, InferenceMode, Instant, PhysicalExecutionPlan, Strategy, StudyResult,
    TableView, TabularData, effect_from_posterior, elapsed_ns, identification_from_cache_or,
    identify_static_query_with_rd, project_for_ate_estimate, select_claim,
};
use crate::planner::RouteDefaults;
use antecedent_estimate::bayesian_robust_ate::{
    BayesianRobustAteInput, BayesianRobustAteOptions, estimate_bayesian_robust_ate,
};

impl super::Study {
    pub(super) fn execute_bayesian_robust_ate(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        Self::ensure_bayesian_robust_ate_admitted(&self.inference, query)?;
        let identifier = physical.binding().identifier_or(RouteDefaults::STATIC_ATE)?;
        let estimator = EstimatorId::BayesianRobustAte;
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identified = identify_static_query_with_rd(
                    identifier,
                    graph,
                    &CausalQuery::AverageEffect(query.clone()),
                    None,
                )?;
                select_claim(identified, estimator)
            })?;
        let (data_est, query_est, estimand_est) = project_for_ate_estimate(data, query, &estimand)?;
        let input = robust_ate_input(&data_est, &query_est)?;
        let InferenceMode::Bayesian(cfg) = &self.inference else {
            return Err(CausalError::Compile {
                message: "bayesian.robust_ate requires Bayesian inference".into(),
            });
        };
        if cfg.has_transfer() {
            return Err(CausalError::Unsupported {
                message: "bayesian.robust_ate uses a Bayesian-bootstrap row law and does not accept coefficient or transferred priors",
            });
        }
        let kernel = estimate_bayesian_robust_ate(
            &input,
            BayesianRobustAteOptions { draws: cfg.n_draws, ..BayesianRobustAteOptions::default() },
            ctx,
        )
        .map_err(CausalError::from)?;
        let posterior = robust_ate_posterior(&kernel, &identification)?;
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
        let result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand_est,
                estimate,
                Strategy::new(identifier, estimator),
                query_est.treatment,
                query_est.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(vec![diagnostic])
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled()),
        );
        let mut result = result;
        result.posterior = Some(posterior);
        Ok(result)
    }

    /// Admission of the robust ATE route: Bayesian inference and a binary all-observed mean
    /// ATE query with no effect modifiers.
    fn ensure_bayesian_robust_ate_admitted(
        inference: &InferenceMode,
        query: &AverageEffectQuery,
    ) -> Result<(), CausalError> {
        if !inference.is_bayesian() {
            return Err(CausalError::Unsupported {
                message: "bayesian.robust_ate requires Bayesian inference",
            });
        }
        if query.target_population != antecedent_core::TargetPopulation::AllObserved
            || query.outcome_functional != antecedent_core::OutcomeFunctional::Mean
            || query.effect_modifiers.len() > 0
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
        Ok(())
    }
}

/// Row-aligned robust-ATE input: the 0/1 treatment, the outcome, and every other column
/// as a baseline covariate.
#[expect(clippy::float_cmp, reason = "the Bernoulli treatment contract requires exact 0/1 coding")]
fn robust_ate_input(
    data_est: &TabularData,
    query_est: &AverageEffectQuery,
) -> Result<BayesianRobustAteInput, CausalError> {
    let n = data_est.row_count();
    let t_raw = data_est.float64_values(query_est.treatment).map_err(CausalError::rejected)?;
    let treatment = t_raw.iter().map(|t| *t == 1.0).collect::<Vec<_>>();
    if t_raw.iter().any(|t| *t != 0.0 && *t != 1.0) {
        return Err(CausalError::Unsupported {
            message: "bayesian.robust_ate requires treatment values coded exactly 0/1",
        });
    }
    let outcome = data_est.float64_values(query_est.outcome).map_err(CausalError::rejected)?;
    let covariate_ids = (0..data_est.schema().len())
        .map(|i| {
            u32::try_from(i).map(antecedent_core::VariableId::from_raw).map_err(|_| {
                CausalError::Compile {
                    message: "Bayesian robust ATE schema exceeds variable id capacity".into(),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let covariates = covariate_ids
        .into_iter()
        .filter(|id| *id != query_est.treatment && *id != query_est.outcome)
        .map(|id| data_est.float64_values(id).map_err(CausalError::rejected))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(BayesianRobustAteInput { row_ids: (0..n as u64).collect(), treatment, outcome, covariates })
}

/// Posterior of the modular Bayesian-bootstrap pushforward, with its declared assumptions.
fn robust_ate_posterior(
    kernel: &antecedent_estimate::bayesian_robust_ate::BayesianRobustAteResult,
    identification: &super::IdentificationResult,
) -> Result<CausalPosterior, CausalError> {
    let mut assumptions = identification.required_assumptions.clone();
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
            id: Arc::from("bayesian.robust_ate.modular_bootstrap_pushforward"),
            description: Arc::from(format!("Posterior draws are modular Bayesian-bootstrap pushforwards: each Exp(1) row-weight draw refits cross-fitted nuisances and reweights the orthogonal AIPW score. {}. Fixed {} folds and clipping; no repeated-sampling calibration is implied.", kernel.model_identity, input_and_folds_note(&kernel.fold_ids))),
        }),
        source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("bayesian.robust_ate") },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let schema = antecedent_prob::PosteriorSchema {
        quantities: Arc::from([antecedent_prob::PosteriorQuantityKind::Effect {
            name: Arc::from("ate"),
        }]),
    };
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        kernel.draws.len(),
        Arc::<[f64]>::from(kernel.draws.clone()),
    )?;
    let posterior = CausalPosterior {
        summaries: draws.summarize(),
        draws,
        identification: identification.status,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: InferenceDiagnostics::analytic(super::ROBUST_ATE_BOOTSTRAP_BACKEND),
        assumptions,
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: None,
    };
    Ok(posterior)
}

fn input_and_folds_note(folds: &[u16]) -> String {
    let count = folds.iter().copied().max().map_or(0, |m| usize::from(m) + 1);
    format!(
        "linear weighted ridge outcome and logistic weighted ridge propensity nuisances, {count}"
    )
}
