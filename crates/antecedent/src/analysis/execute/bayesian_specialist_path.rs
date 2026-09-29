//! Staged execution for model-scoped Bayesian IV and sharp-RD estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_estimate::bayesian_iv::fit_bayesian_iv_joint_fixed_loading;
use antecedent_estimate::bayesian_rd::fit_bayesian_sharp_rd;
use antecedent_prob::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};

/// Model-scope checks shared by the ordinary and checked Bayesian IV routes.
pub(super) fn check_bayesian_iv_scope(
    estimand: &IdentifiedEstimand,
    config: &BayesianConfig,
) -> Result<(), CausalError> {
    if estimand.instruments.len() != 1 || !estimand.adjustment_set.is_empty() {
        return Err(CausalError::Compile{message:"iv.bayesian_joint_linear currently supports exactly one instrument and no exogenous adjustment covariates".into()});
    }
    if config.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
        || config.prior.is_some()
        || config.prior_artifact.is_some()
        || config.external_compose.is_some()
    {
        return Err(CausalError::Compile{message:"iv.bayesian_joint_linear requires its built-in conjugate model; transferred and custom priors are not supported".into()});
    }
    Ok(())
}

/// Model-scope checks shared by the ordinary and checked Bayesian sharp-RD routes.
pub(super) fn check_bayesian_rd_scope(
    query: &AverageEffectQuery,
    config: &BayesianConfig,
) -> Result<(), CausalError> {
    if !query.effect_modifiers.is_empty() || !query.outcome_functional.is_mean() {
        return Err(CausalError::Compile {
            message: "rd.bayesian_local_linear supports only an unmodified mean outcome".into(),
        });
    }
    if config.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
        || config.prior.is_some()
        || config.prior_artifact.is_some()
        || config.external_compose.is_some()
    {
        return Err(CausalError::Compile{message:"rd.bayesian_local_linear requires its built-in conjugate model; transferred and custom priors are not supported".into()});
    }
    Ok(())
}

/// Identification products and model configuration a Bayesian specialist
/// execution reads; the checked route supplies these from its retained plan.
#[derive(Clone, Copy)]
pub(super) struct BayesianSpecialistInputs<'a> {
    pub(super) identification: &'a IdentificationResult,
    pub(super) estimand: &'a IdentifiedEstimand,
    pub(super) config: &'a BayesianConfig,
    pub(super) identify_cached: bool,
}

impl Study {
    pub(super) fn execute_bayesian_iv(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let identification = identify_static_query(
            IdentifierId::Iv,
            graph,
            &CausalQuery::AverageEffect(query.clone()),
        )?;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::BayesianIvJointLinear)?;
        let InferenceMode::Bayesian(config) = &self.inference else {
            return Err(CausalError::Compile {
                message: "iv.bayesian_joint_linear requires Bayesian inference".into(),
            });
        };
        self.execute_bayesian_iv_identified(
            data,
            query,
            physical,
            BayesianSpecialistInputs {
                identification: &identification,
                estimand: &estimand,
                config,
                identify_cached: false,
            },
            ctx,
        )
    }

    /// Joint linear Gaussian IV posterior from already-identified products.
    pub(super) fn execute_bayesian_iv_identified(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        inputs: BayesianSpecialistInputs<'_>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let BayesianSpecialistInputs { identification, estimand, config, identify_cached } = inputs;
        let estimator_id = EstimatorId::BayesianIvJointLinear;
        check_bayesian_iv_scope(estimand, config)?;
        let frequentist = antecedent_estimate::TwoStageLeastSquares::new();
        let prep = frequentist.prepare(data, estimand, query).map_err(CausalError::from)?;
        let n = prep.nrows;
        let z: Vec<f64> = (0..n).map(|r| prep.instruments_matrix[n + r]).collect();
        let result = fit_bayesian_iv_joint_fixed_loading(
            &z,
            &prep.treatment,
            &prep.outcome,
            config.prior_scale,
            1.0,
            1.0,
            config.n_draws,
            ctx.rng.master_seed(),
            10.0,
        )
        .map_err(CausalError::from)?;
        let effect_draws: Vec<f64> =
            result.effect_draws.iter().map(|v| v * prep.treatment_delta).collect();
        let mut posterior = effect_posterior(
            &effect_draws,
            identification.status,
            identification.required_assumptions.clone(),
            Some(prep.treatment_delta),
            "bayesian_iv.joint_fixed_loading",
        );
        posterior.diagnostics.notes.push(Arc::from(format!("interval=equal_tailed_95 first_stage_f={:.4}; joint Gaussian structural posterior with fixed unit disturbance loading, declared stage variances=1.0, and prior_scale={}",result.first_stage_f,config.prior_scale)));
        let effect = posterior.summaries.mean[0];
        let sd = posterior.summaries.sd[0];
        let estimate = EffectEstimate::new(
            effect,
            sd,
            identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(n as u64);
        let diagnostics = vec![Diagnostic::new(
            "bayesian_iv.model_scope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "joint linear Gaussian structural model; fixed unit disturbance loading and declared stage variances of 1.0; one instrument; no adjustment covariates",
        )];
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification: identification.clone(),
            estimand: estimand.clone(),
            estimate,
            identifier_id: IdentifierId::Iv,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: diagnostics,
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                posterior: Some(posterior),
                n_draws: Some(u32::try_from(config.n_draws).unwrap_or(u32::MAX)),
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_bayesian_rd(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let rd=self.rd.ok_or_else(||CausalError::Compile{message:"rd.bayesian_local_linear requires builder.rd_config(running_variable, cutoff, bandwidth)".into()})?;
        let identification = SharpRdIdentifier::new(SharpRdConfig::new(
            rd.running_variable,
            rd.cutoff,
            rd.bandwidth,
        ))
        .identify_on(graph, CausalQuery::AverageEffect(query.clone()))
        .map_err(CausalError::from)?;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::BayesianRdLocalLinear)?;
        let InferenceMode::Bayesian(config) = &self.inference else {
            return Err(CausalError::Compile {
                message: "rd.bayesian_local_linear requires Bayesian inference".into(),
            });
        };
        self.execute_bayesian_rd_identified(
            data,
            query,
            physical,
            rd,
            BayesianSpecialistInputs {
                identification: &identification,
                estimand: &estimand,
                config,
                identify_cached: false,
            },
            ctx,
        )
    }

    /// Fixed-bandwidth local-linear sharp-RD posterior from already-identified products.
    pub(super) fn execute_bayesian_rd_identified(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        rd: crate::RdConfig,
        inputs: BayesianSpecialistInputs<'_>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let BayesianSpecialistInputs { identification, estimand, config, identify_cached } = inputs;
        let estimator_id = EstimatorId::BayesianRdLocalLinear;
        check_bayesian_rd_scope(query, config)?;
        let ids = [query.treatment, query.outcome, rd.running_variable];
        let mask = data.complete_case_mask(&ids).map_err(CausalError::from)?;
        let t = data.float64_masked(query.treatment, &mask).map_err(CausalError::from)?;
        let y = data.float64_masked(query.outcome, &mask).map_err(CausalError::from)?;
        let r = data.float64_masked(rd.running_variable, &mask).map_err(CausalError::from)?;
        let result = fit_bayesian_sharp_rd(
            &r,
            &t,
            &y,
            rd.cutoff,
            rd.bandwidth,
            config.prior_scale,
            config.n_draws,
            ctx.rng.master_seed(),
        )
        .map_err(CausalError::from)?;
        let mut posterior = effect_posterior(
            &result.jump_draws,
            identification.status,
            identification.required_assumptions.clone(),
            Some(1.0),
            "bayesian_rd.fixed_bandwidth_local_linear",
        );
        posterior.diagnostics.notes.push(Arc::from(format!(
            "interval=equal_tailed_95 fixed_bandwidth={} prior_sd={}",
            rd.bandwidth, config.prior_scale
        )));
        let estimate = EffectEstimate::new(
            posterior.summaries.mean[0],
            posterior.summaries.sd[0],
            identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(result.n_window as u64);
        let diagnostics = vec![Diagnostic::new(
            "bayesian_rd.model_scope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "sharp-assignment local-linear Gaussian model; cutoff={}; fixed bandwidth={}; plug-in outcome variance",
                rd.cutoff, rd.bandwidth
            ),
        )];
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification: identification.clone(),
            estimand: estimand.clone(),
            estimate,
            identifier_id: IdentifierId::RdSharp,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: diagnostics,
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                posterior: Some(posterior),
                n_draws: Some(u32::try_from(config.n_draws).unwrap_or(u32::MAX)),
                ..Default::default()
            },
        }))
    }
}

fn effect_posterior(
    draws: &[f64],
    identification: antecedent_core::IdentificationStatus,
    assumptions: antecedent_core::AssumptionSet,
    contrast: Option<f64>,
    backend: &str,
) -> CausalPosterior {
    let schema = PosteriorSchema {
        quantities: Arc::from([PosteriorQuantityKind::Effect { name: Arc::from("ate") }]),
    };
    let samples = PosteriorDraws::from_column_major(schema, draws.len(), Arc::<[f64]>::from(draws))
        .expect("posterior draw shape matches one effect quantity");
    let summaries = samples.summarize();
    CausalPosterior {
        draws: samples,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: InferenceDiagnostics::analytic(backend),
        assumptions,
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: contrast,
    }
}
