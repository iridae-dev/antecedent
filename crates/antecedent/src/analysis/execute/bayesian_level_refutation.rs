// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bayesian refuter state of a scalar intervention level on a static DAG.
//!
//! The Frequentist intervention level runs the plugin-level suite: overlap under
//! `cheap`, overlap plus sampling stability of the level under `full`. A Bayesian
//! level runs that same plugin-level suite against its posterior-mean level and adds
//! the checks that belong to a posterior:
//!
//! * `cheap` and `full`: the prior predictive check and the posterior predictive
//!   check of the outcome regression the level is computed from (the same linear
//!   response design the posterior was fitted on);
//! * `full` only: level-native prior sensitivity (the whole response is refitted
//!   at each point of the prior grid and the posterior-mean **level** is read back,
//!   not the dose coefficient) and the MCMC diagnostics of the primary fit.
//!
//! The state executes and is unlicensed: no evidence row or coverage record names
//! it, the support matrix reports the cell `refused`, and a refuter verdict here is
//! a check, not a calibration claim.

use super::response_path::response_prior_targets;
use super::{
    Arc, AverageEffectQuery, BayesianConfig, BayesianGCompWorkspace, BayesianGComputationAte,
    BayesianSuiteContext, CausalError, ContinuousResponseEstimator, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimationWorkspace, EstimatorId, ExecutionContext,
    IdentificationStatus, IdentifiedEstimand, LatencyMode, OverlapPolicy, PosteriorPredictiveCheck,
    PredictiveCheckReport, PriorPredictiveCheck, RefutationRun, RefuteSuite, ResponseQuery,
    TabularData, ValidationSuite, ValidatorId, VariableId, bayesian_gcomp,
    validator_not_applicable_diagnostics,
};

/// Two-sided level of the predictive-check pass/fail rule (as the Bayesian ATE route).
const PPC_ALPHA: f64 = 0.05;

/// Borrowed inputs of one Bayesian intervention-level refutation.
pub(super) struct BayesianLevelRefutationInput<'a> {
    /// Rows the response was estimated on.
    pub(super) data: &'a TabularData,
    /// Retained intervention-response query.
    pub(super) query: &'a ResponseQuery,
    /// Identified target whose adjustment set the response used.
    pub(super) estimand: &'a IdentifiedEstimand,
    /// Response fitter fixed to that adjustment set and the study's response options.
    pub(super) fitter: &'a ContinuousResponseEstimator,
    /// Bayesian estimator with the prior that was in force for the published level.
    pub(super) estimator: &'a BayesianGComputationAte,
    /// Identification status the response was estimated under.
    pub(super) status: IdentificationStatus,
    /// Assumptions carried from identification.
    pub(super) assumptions: &'a antecedent_core::AssumptionSet,
    /// Published posterior-mean level.
    pub(super) level: f64,
    /// Posterior standard deviation of the level.
    pub(super) level_sd: f64,
    /// Primary intervention coordinate.
    pub(super) treatment: VariableId,
    /// Outcome.
    pub(super) outcome: VariableId,
    /// Requested suite.
    pub(super) suite: RefuteSuite,
    /// Latency tier, which sets the predictive-check simulation budget.
    pub(super) latency: Option<LatencyMode>,
    /// Run the plugin-level suite here. `false` when the caller already ran it on
    /// the published level (a graph-posterior mixture runs it on the pooled level).
    pub(super) plugin_level: bool,
    /// Scalar (value, posterior SD) a refitted response is read back as: the level for
    /// an intervention response, the first component for a derivative.
    pub(super) summarize: fn(&antecedent_core::CausalResponse) -> (f64, f64),
    /// What the state refutes, for its disclosure (`intervention-level`, `derivative`).
    pub(super) target_label: &'static str,
}

/// Reports, predictive checks and notes of one Bayesian intervention-level refutation.
#[derive(Default)]
pub(super) struct BayesianLevelRefutation {
    pub(super) refutations: Vec<antecedent_validate::RefutationReport>,
    pub(super) predictive_checks: Vec<PredictiveCheckReport>,
    pub(super) diagnostics: Vec<Diagnostic>,
}

/// The Bayesian estimator a static response fits with, prior transfer resolved the
/// same way the response route resolves it.
///
/// # Errors
///
/// A transferred prior without a response-specific mapping, or prior resolution.
pub(super) fn bayesian_response_estimator_in_force(
    fitter: &ContinuousResponseEstimator,
    data: &TabularData,
    query: &ResponseQuery,
    config: &BayesianConfig,
    ctx: &ExecutionContext,
) -> Result<BayesianGComputationAte, CausalError> {
    if config.has_transferred_source() && config.prior_mapping.is_none() && config.prior.is_none() {
        return Err(CausalError::Unsupported {
            message: "Bayesian response prior transfer requires a response-specific \
                      mapping (dose/intervention coordinate to outcome-regression \
                      coefficients); incompatible catalogs must not fall back to \
                      isotropic priors",
        });
    }
    let mut bayes = bayesian_gcomp(config, ctx);
    if config.has_transferred_source() {
        let (outcome, treatments) = response_prior_targets(&query.functional)?;
        let prep = fitter
            .prepare_linear_response_problem(data, outcome, &treatments)
            .map_err(CausalError::from)?;
        let (resolved, _) =
            crate::inference::resolve_bayesian_prior_with_conflict(config, &prep, Some(ctx))?;
        bayes.prior = resolved;
    } else {
        bayes.prior.clone_from(&config.prior);
    }
    Ok(bayes)
}

/// Run the Bayesian refuter state of a scalar intervention level.
///
/// `RefuteSuite::None` returns nothing. `Cheap` runs the cheap state and `Full` the
/// full state. Callers pass `PlaceboAndRcc` only where the Frequentist level runs it;
/// it is treated as the cheap state (the plugin-level suite maps it to overlap only).
///
/// # Errors
///
/// Cancellation, a design the outcome regression cannot be prepared on, or a
/// refit failure on the primary fit or the prior-sensitivity grid.
#[allow(
    clippy::too_many_lines,
    reason = "one Bayesian refuter state: the primary fit, the predictive checks and the prior-sensitivity grid read the same prepared design"
)]
pub(super) fn refute_bayesian_intervention_level(
    input: &BayesianLevelRefutationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<BayesianLevelRefutation, CausalError> {
    let mut out = BayesianLevelRefutation::default();
    if matches!(input.suite, RefuteSuite::None) {
        return Ok(out);
    }
    if !input.level.is_finite() {
        out.diagnostics.push(Diagnostic::new(
            "refute.response.bayesian.no_scalar_level",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the Bayesian refuter state reads a scalar intervention level; this response has none",
        ));
        return Ok(out);
    }
    check_cancelled(ctx)?;
    if input.plugin_level {
        // Plugin-level suite against the posterior-mean level: overlap is a property of
        // the data and the adjustment set; sampling stability refits the level.
        let ate_query = AverageEffectQuery::binary_ate(input.treatment, input.outcome);
        let plugin_estimate = EffectEstimate::new(
            input.level,
            input.level_sd,
            input.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut refute_ws = EstimationWorkspace::default();
        let (reports, notes) = RefutationRun {
            data: input.data,
            estimand: input.estimand,
            query: &ate_query,
            estimate: &plugin_estimate,
            suite: input.suite,
            estimator: EstimatorId::ResponseBayesian.as_str(),
            custom: &[],
            temporal: None,
        }
        .plugin_level(&mut refute_ws, ctx)?;
        out.refutations.extend(reports);
        out.diagnostics.extend(notes);
    }

    // Predictive checks of the outcome regression the level is computed from.
    check_cancelled(ctx)?;
    let (outcome, treatments) = response_prior_targets(&input.query.functional)?;
    let prep = input
        .fitter
        .prepare_linear_response_problem(input.data, outcome, &treatments)
        .map_err(CausalError::from)?;
    let mut bayes_ws = BayesianGCompWorkspace::default();
    let posterior =
        input.estimator.fit(&prep, input.status, &mut bayes_ws, ctx).map_err(CausalError::from)?;
    let sims = super::super::latency::predictive_check_sims(input.latency);
    let prior_in_force = input.estimator.prior_in_force(prep.design.ncols);
    let prior_rep = PriorPredictiveCheck::for_estimator(input.estimator, ctx)
        .with_n_sims(sims)
        .check_with_prior(&prep, &prior_in_force, ctx)
        .map_err(CausalError::from)?;
    out.refutations.push(prior_rep.to_refutation_report(input.level, PPC_ALPHA));
    out.predictive_checks.push(prior_rep);
    let post_rep = PosteriorPredictiveCheck::for_estimator(input.estimator, ctx)
        .with_n_sims(sims)
        .check(&prep, &posterior)
        .map_err(CausalError::from)?;
    out.refutations.push(post_rep.to_refutation_report(input.level, PPC_ALPHA));
    out.predictive_checks.push(post_rep);

    if matches!(input.suite, RefuteSuite::Full) {
        let (sensitivity, summary) = level_prior_sensitivity(input, ctx)?;
        out.refutations.push(sensitivity.to_report(&summary, input.level));

        check_cancelled(ctx)?;
        let suite = ValidationSuite::new().with(ValidatorId::McmcDiagnostics);
        let mut bayes_ctx = BayesianSuiteContext::new(
            input.estimator,
            &prep,
            &posterior,
            input.status,
            &mut bayes_ws,
            input.level,
        );
        let outcomes = suite.run_bayesian(&mut bayes_ctx, ctx).map_err(CausalError::from)?;
        out.refutations.extend(ValidationSuite::reports_only(&outcomes));
        out.diagnostics.extend(validator_not_applicable_diagnostics(&outcomes));
    }
    let full = matches!(input.suite, RefuteSuite::Full);
    out.diagnostics.push(Diagnostic::new(
        "refute.response.bayesian_state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "Bayesian {} refuter state ({}): {}prior and posterior predictive checks of the \
             outcome regression{}. It executes unlicensed: no coverage or evidence record \
             names it, and the verdicts are checks, not a calibration claim.",
            input.target_label,
            if full { "full" } else { "cheap" },
            match (input.plugin_level, full) {
                (true, true) =>
                    "plugin-level overlap and sampling stability of the \
                                 posterior-mean level, ",
                (true, false) => "plugin-level overlap on the posterior-mean level, ",
                (false, _) => "",
            },
            if full {
                ", prior sensitivity refitting the whole response at each grid point and \
                 reading back the target itself, and MCMC diagnostics"
            } else {
                ""
            },
        ),
    ));
    Ok(out)
}

/// Refit the whole response at each prior-grid point and read back the posterior-mean
/// level. The grid perturbs the prior in force: variance multipliers around a resolved
/// prior, the isotropic scale grid when none was supplied.
fn level_prior_sensitivity(
    input: &BayesianLevelRefutationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<
    (antecedent_validate::PriorSensitivity, antecedent_prob::PriorSensitivitySummary),
    CausalError,
> {
    let grid = antecedent_validate::PriorSensitivity::for_estimator(input.estimator);
    let resolved = !grid.variance_multipliers.is_empty();
    let points: Vec<f64> =
        if resolved { grid.variance_multipliers.to_vec() } else { grid.scales.to_vec() };
    let mut means = Vec::with_capacity(points.len());
    let mut sds = Vec::with_capacity(points.len());
    for point in points {
        check_cancelled(ctx)?;
        let estimator = if resolved {
            let prior = input.estimator.prior.as_ref().ok_or_else(|| CausalError::Compile {
                message: "level prior sensitivity lost the prior in force".into(),
            })?;
            BayesianGComputationAte {
                n_draws: input.estimator.n_draws,
                prior: Some(scale_coefficient_variances(prior, point)?),
                ..input.estimator.clone()
            }
        } else {
            BayesianGComputationAte {
                prior_scale: point,
                n_draws: input.estimator.n_draws,
                prior: None,
                ..input.estimator.clone()
            }
        };
        let response = input
            .fitter
            .estimate_bayesian(
                input.data,
                input.query,
                input.status,
                input.assumptions.clone(),
                &estimator,
                ctx,
            )
            .map_err(CausalError::from)?;
        let (mean, sd) = (input.summarize)(&response);
        means.push(mean);
        sds.push(sd);
    }
    let summary = antecedent_prob::PriorSensitivitySummary {
        family: if resolved {
            antecedent_prob::PriorSensitivityFamily::ResolvedPriorVariance
        } else {
            antecedent_prob::PriorSensitivityFamily::IsotropicScale
        },
        prior_scales: if resolved { Arc::from([]) } else { Arc::clone(&grid.scales) },
        alphas: Arc::from([]),
        variance_multipliers: if resolved {
            Arc::clone(&grid.variance_multipliers)
        } else {
            Arc::from([])
        },
        effect_means: Arc::from(means),
        effect_sds: Arc::from(sds),
    };
    Ok((grid, summary))
}

/// Copy of `prior` with every Gaussian coefficient variance multiplied by `multiplier`;
/// coefficient means, residual-variance and restriction specs are kept.
fn scale_coefficient_variances(
    prior: &antecedent_prob::PriorSet,
    multiplier: f64,
) -> Result<antecedent_prob::PriorSet, CausalError> {
    if !multiplier.is_finite() || multiplier <= 0.0 {
        return Err(CausalError::Compile {
            message: "prior sensitivity variance multiplier must be finite and positive".into(),
        });
    }
    let mut scaled = prior.clone();
    let mut found = false;
    for spec in &mut scaled.specs {
        if let antecedent_prob::PriorSpec::GaussianCoefficients(coefficients) = spec {
            coefficients.variance =
                coefficients.variance.iter().map(|variance| variance * multiplier).collect();
            found = true;
        }
    }
    if !found {
        return Err(CausalError::Unsupported {
            message: "the prior in force has no Gaussian coefficient block for level prior \
                      sensitivity to perturb",
        });
    }
    Ok(scaled)
}

fn check_cancelled(ctx: &ExecutionContext) -> Result<(), CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
    }
    Ok(())
}
