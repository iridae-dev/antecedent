// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
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
        if !matches!(self.inference, InferenceMode::Bayesian(_)) {
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
        let identifier: IdentifierId =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER).parse()?;
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
        let n = data_est.row_count();
        let t_raw = data_est
            .float64_values(query_est.treatment)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let treatment = t_raw.iter().map(|t| *t == 1.0).collect::<Vec<_>>();
        if t_raw.iter().any(|t| *t != 0.0 && *t != 1.0) {
            return Err(CausalError::Unsupported {
                message: "bayesian.robust_ate requires treatment values coded exactly 0/1",
            });
        }
        let outcome = data_est
            .float64_values(query_est.outcome)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let covariates = (0..data_est.schema().len())
            .map(|i| antecedent_core::VariableId::from_raw(i as u32))
            .filter(|id| *id != query_est.treatment && *id != query_est.outcome)
            .map(|id| {
                data_est
                    .float64_values(id)
                    .map_err(|e| CausalError::Compile { message: e.to_string() })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let cfg = match &self.inference {
            InferenceMode::Bayesian(cfg) => cfg,
            _ => unreachable!(),
        };
        if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
            return Err(CausalError::Unsupported {
                message: "bayesian.robust_ate uses a Bayesian-bootstrap row law and does not accept coefficient or transferred priors",
            });
        }
        let input = BayesianRobustAteInput {
            row_ids: (0..n as u64).collect(),
            treatment,
            outcome,
            covariates,
        };
        let kernel = estimate_bayesian_robust_ate(
            &input,
            BayesianRobustAteOptions { draws: cfg.n_draws, ..BayesianRobustAteOptions::default() },
            ctx,
        )
        .map_err(CausalError::from)?;
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
        )
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let posterior = CausalPosterior {
            summaries: draws.summarize(),
            draws,
            identification: identification.status,
            prior_sensitivity: None,
            conflict_summary: None,
            diagnostics: InferenceDiagnostics::analytic(
                "bayesian.robust_ate.modular_bootstrap_pushforward",
            ),
            assumptions,
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
        let result = self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand: estimand_est,
            estimate,
            identifier_id: identifier,
            estimator_id: estimator,
            treatment: query_est.treatment,
            outcome: query_est.outcome,
            identify_cached,
            extra_diagnostics: vec![diagnostic],
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        });
        let mut result = result;
        result.posterior = Some(posterior);
        Ok(result)
    }
}

fn input_and_folds_note(folds: &[u16]) -> String {
    let count = folds.iter().copied().max().map_or(0, |m| usize::from(m) + 1);
    format!(
        "linear weighted ridge outcome and logistic weighted ridge propensity nuisances, {count}"
    )
}
