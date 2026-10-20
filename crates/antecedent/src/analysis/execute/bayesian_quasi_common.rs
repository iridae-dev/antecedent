//! Shared assembly of the Bayesian design-based routes: panel difference-in-differences,
//! synthetic control and the fuzzy-jump / kink local ratio.
//!
//! These routes are explicitly unlicensed and uncalibrated. A posterior here is the declared
//! outcome model's belief about the same estimand the frequentist row estimates. It never
//! supplies or tests an identifying premise: the design's premises stay declared in the
//! result's assumptions, and no calibration or coverage evidence covers the credible interval.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianConfig, CausalError, CausalPosterior, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, IdentificationResult, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedResultContext, IdentifierId,
    InferenceDiagnostics, Instant, OverlapPolicy, PhysicalExecutionPlan, Strategy, StudyResult,
    TabularData, VariableId, elapsed_ns, finish_identified_execute_with_context,
};
use antecedent_prob::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};

/// Sampler record of a Metropolis-within-Gibbs fit, kept on the posterior's diagnostics.
pub(super) struct SamplerRecord {
    pub(super) chains: u32,
    pub(super) max_split_rhat: f64,
    pub(super) min_bulk_ess: f64,
    pub(super) acceptance_rate: f64,
}

/// What one Bayesian design fit reports before it is assembled into a result.
pub(super) struct DesignPosterior {
    /// Posterior columns; the first `Effect` column is the primary quantity.
    pub(super) columns: Vec<(PosteriorQuantityKind, Vec<f64>)>,
    /// Reported scalar.
    pub(super) point: f64,
    /// Posterior standard deviation of the scalar, `NaN` when none is reported.
    pub(super) point_sd: f64,
    /// Rows or subjects behind the posterior.
    pub(super) n_obs: u64,
    /// The interval is withheld: the posterior carries draws but no credible interval is published.
    pub(super) interval_withheld: bool,
    /// Backend identifier of the posterior.
    pub(super) backend: &'static str,
    /// Model scope sentence.
    pub(super) model: String,
    /// Model-level assumptions: identifier and description, recorded as declared.
    pub(super) assumptions: Vec<(&'static str, String)>,
    /// Route-specific diagnostics.
    pub(super) diagnostics: Vec<Diagnostic>,
    /// Notes kept on the posterior's inference diagnostics.
    pub(super) notes: Vec<String>,
    /// Sampler summary of a simulated posterior.
    pub(super) sampler: Option<SamplerRecord>,
}

/// The retained facts a Bayesian design fit is finished with.
#[derive(Clone, Copy)]
pub(super) struct DesignFinish<'a> {
    pub(super) context: &'a IdentifiedResultContext,
    pub(super) data: &'a TabularData,
    pub(super) physical: &'a PhysicalExecutionPlan,
    pub(super) identification: &'a IdentificationResult,
    pub(super) estimand: &'a IdentifiedEstimand,
    pub(super) outcome: VariableId,
    pub(super) config: &'a BayesianConfig,
    pub(super) started: Instant,
}

/// Refuse a configuration these routes' built-in Gaussian models cannot honor.
pub(super) fn require_built_in_model(
    config: &BayesianConfig,
    route: &str,
) -> Result<(), CausalError> {
    if config.backend == antecedent_estimate::BayesianBackendKind::ConjugateGaussian
        && !config.has_transfer()
    {
        return Ok(());
    }
    Err(crate::compile_reason!(
        "option_not_applicable",
        "{route} fits its own built-in Gaussian model, so a backend choice, a transferred prior \
         or a custom prior set does not apply. Next attempt: Bayesian inference with the \
         conjugate backend, prior_scale and n_draws only, and no prior artifact"
    ))
}

/// Typed refusal of a design option the Bayesian counterpart does not read, naming the
/// next attempt.
pub(super) fn unread_option(route: &str, option: &str, next: &str) -> CausalError {
    crate::compile_reason!(
        "route_not_supported",
        "{route} does not read {option}. Next attempt: {next}"
    )
}

/// Assemble a posterior from equal-length named columns.
pub(super) fn assemble_posterior(
    columns: &[(PosteriorQuantityKind, Vec<f64>)],
    identification: &IdentificationResult,
    assumptions: antecedent_core::AssumptionSet,
    diagnostics: InferenceDiagnostics,
) -> Result<CausalPosterior, CausalError> {
    let n_draws = columns.first().map_or(0, |(_, draws)| draws.len());
    if n_draws < 2 || columns.iter().any(|(_, draws)| draws.len() != n_draws) {
        return Err(CausalError::Compile {
            message: "a Bayesian design route produced posterior columns of unequal length".into(),
        });
    }
    let mut values = Vec::with_capacity(n_draws * columns.len());
    for (_, draws) in columns {
        values.extend_from_slice(draws);
    }
    let schema = PosteriorSchema {
        quantities: columns.iter().map(|(kind, _)| kind.clone()).collect::<Vec<_>>().into(),
    };
    let samples = PosteriorDraws::from_column_major(schema, n_draws, Arc::<[f64]>::from(values))
        .map_err(CausalError::from)?;
    let summaries = samples.summarize();
    Ok(CausalPosterior {
        draws: samples,
        summaries,
        identification: identification.status,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics,
        assumptions,
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: None,
        preparation: None,
    })
}

/// Declared model assumption of an estimation stage; the design's identifying premises are
/// carried unchanged beside it.
pub(super) fn model_assumption(
    backend: &str,
    id: &str,
    description: &str,
) -> antecedent_core::AssumptionRecord {
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::Custom {
            id: Arc::from(id),
            description: Arc::from(description),
        },
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from(backend),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// Finish a Bayesian design fit into the study's result: an unlicensed posterior whose
/// declared identifying premises stay declared.
pub(super) fn finish_design_posterior(
    args: DesignFinish<'_>,
    fit: DesignPosterior,
) -> Result<StudyResult, CausalError> {
    finish_design_posterior_as(
        args,
        fit,
        Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedHt),
    )
}

/// [`finish_design_posterior`] recording the identifier and estimator the route actually ran.
pub(super) fn finish_design_posterior_as(
    args: DesignFinish<'_>,
    fit: DesignPosterior,
    strategy: Strategy,
) -> Result<StudyResult, CausalError> {
    let DesignFinish {
        context,
        data,
        physical,
        identification,
        estimand,
        outcome,
        config,
        started,
    } = args;
    let mut posterior_assumptions = identification.required_assumptions.clone();
    for (id, description) in &fit.assumptions {
        posterior_assumptions.push(model_assumption(fit.backend, id, description));
    }
    let mut inference = InferenceDiagnostics::analytic(fit.backend);
    inference.notes = fit.notes.iter().map(|note| Arc::<str>::from(note.as_str())).collect();
    if let Some(sampler) = &fit.sampler {
        inference.n_chains = Some(sampler.chains);
        inference.rhat_max = Some(sampler.max_split_rhat);
        inference.ess_bulk_min = Some(sampler.min_bulk_ess);
        inference.mean_accept_prob = Some(sampler.acceptance_rate);
    }
    let posterior =
        assemble_posterior(&fit.columns, identification, posterior_assumptions, inference)?;
    let n_draws = posterior.draws.n_draws;
    let estimate = EffectEstimate::new(
        fit.point,
        fit.point_sd,
        identification.required_assumptions.clone(),
        OverlapPolicy::ExplicitOverride,
    )
    .with_n_obs(fit.n_obs);
    let mut diagnostics = vec![
        Diagnostic::new(
            "estimate.quasi.bayesian.unlicensed",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "unlicensed and uncalibrated Bayesian route: no coverage or calibration evidence \
             covers this posterior, and the prior supplies no identifying premise",
        ),
        Diagnostic::new(
            "estimate.quasi.bayesian.model_scope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("{}; prior_scale={}; n_draws={n_draws}", fit.model, config.prior_scale),
        ),
    ];
    diagnostics.extend(fit.diagnostics);
    let finish = IdentifiedExecuteFinish::new(
        physical,
        identification.clone(),
        estimand.clone(),
        estimate,
        strategy,
        outcome,
        outcome,
    )
    .extra_diagnostics(diagnostics)
    .wall_time_ns(elapsed_ns(started))
    .extras(IdentifiedExecuteExtras {
        posterior: Some(posterior),
        n_draws: Some(u32::try_from(n_draws).unwrap_or(u32::MAX)),
        ..Default::default()
    });
    let mut result = finish_identified_execute_with_context(context, Some(data), finish);
    result.treatment = None;
    result.support_status = None;
    if fit.interval_withheld {
        result.interval = None;
    }
    Ok(result)
}

/// A named effect quantity.
pub(super) fn effect_quantity(name: impl AsRef<str>) -> PosteriorQuantityKind {
    PosteriorQuantityKind::Effect { name: Arc::from(name.as_ref()) }
}

/// A named descriptive scalar quantity.
pub(super) fn scalar_quantity(name: impl AsRef<str>) -> PosteriorQuantityKind {
    PosteriorQuantityKind::Scalar { name: Arc::from(name.as_ref()) }
}
