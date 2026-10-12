//! The one fit-and-assemble implementation of a `TemporalDag` mediation.
//!
//! The sealed operation ([`super::CheckedTemporalMediationOperation`]) and the
//! one-shot route both resolve their per-horizon proofs, gate their own
//! licence, and call [`fit_temporal_mediation`]. The linear mechanism fits, the
//! block bootstrap, the Gaussian posterior composition, the refuters and the
//! result envelope live here only, so a route cannot drift from the other in
//! seeds, diagnostics or provenance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::temporal_path::{
    MediationHorizonClick, mediation_click_adjustment_keys, mediation_horizon_z_differs,
    temporal_dependence_se_diagnostics,
};
use super::{
    Arc, BayesianConfig, BayesianGCompWorkspace, BayesianGComputationAte, CausalError,
    CausalPosterior, Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId,
    ExecutionContext, IdentificationResult, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifierId, MediationRoute, PhysicalExecutionPlan,
    PosteriorPredictiveCheck, PredictiveCheckReport, PriorPredictiveCheck, RefuteSuite,
    TemporalIndexer, TemporalMediationEstimate, TemporalMediationEstimator, TimeSeriesData,
    bayesian_gcomp, effect_from_posterior, mediation_block_stream, mediation_mechanism_seed,
    most_conservative_identification_status, nan_effect, provenance_ids, require_identified,
    with_prior_sensitivity,
};

/// The inference a mediation fit runs under.
pub(super) enum MediationInference<'a> {
    Frequentist { bootstrap_replicates: u32 },
    Bayesian(&'a BayesianConfig),
}

impl MediationInference<'_> {
    pub(super) fn estimator_id(&self) -> EstimatorId {
        match self {
            Self::Frequentist { .. } => EstimatorId::TemporalMediation,
            Self::Bayesian(_) => EstimatorId::BayesianTemporalMediation,
        }
    }
}

/// Refuse a coefficient prior the two isotropic mechanism fits cannot take.
pub(super) fn require_isotropic_mediation_prior(
    config: &BayesianConfig,
) -> Result<(), CausalError> {
    if config.prior.is_some()
        || config.prior_artifact.is_some()
        || config.external_compose.is_some()
    {
        return Err(CausalError::Unsupported {
            message: "Bayesian temporal mediation supports isotropic mechanism priors; shared or transferred coefficient priors are not licensed",
        });
    }
    Ok(())
}

/// Everything one mediation fit reads, resolved by the caller.
pub(super) struct TemporalMediationInputs<'a> {
    pub data: &'a TimeSeriesData,
    pub query: &'a antecedent_core::MediationQuery,
    pub horizons: &'a [MediationHorizonClick],
    pub inference: MediationInference<'a>,
    pub validation: RefuteSuite,
}

/// An assembled mediation fit, before the result envelope.
pub(super) struct TemporalMediationFit {
    estimator_id: EstimatorId,
    estimate: EffectEstimate,
    mediation: Option<TemporalMediationEstimate>,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    grid: antecedent_estimate::TemporalMediationGrid,
    refutations: Vec<antecedent_validate::RefutationReport>,
    predictive_checks: Vec<PredictiveCheckReport>,
    diagnostics: Vec<Diagnostic>,
    posterior: Option<CausalPosterior>,
    n_draws: Option<u32>,
    bootstrap_replicates_ok: Option<u32>,
    cancelled: bool,
    single: bool,
}

impl TemporalMediationFit {
    pub(super) fn finish_args<'a>(
        self,
        physical: &'a PhysicalExecutionPlan,
        query: &antecedent_core::MediationQuery,
        identify_cached: bool,
        structure_version: u32,
        wall_time_ns: u64,
    ) -> IdentifiedExecuteFinish<'a> {
        let certificate = self.single.then(|| crate::Identification::Point {
            result: self.identification.clone(),
            temporal_indexer: Some(self.indexer.clone()),
            strategy: IdentifierId::Frontdoor,
            structure_version,
        });
        IdentifiedExecuteFinish::new(
            physical,
            self.identification,
            self.estimand,
            self.estimate,
            IdentifierId::Frontdoor,
            self.estimator_id,
            query.treatment,
            query.outcome,
        )
        .identify_cached(identify_cached)
        .extra_diagnostics(self.diagnostics)
        .refutations(self.refutations)
        .mediation(self.mediation)
        .wall_time_ns(wall_time_ns)
        .bootstrap_replicates_ok(self.bootstrap_replicates_ok)
        .cancelled(self.cancelled)
        .extras(IdentifiedExecuteExtras {
            certificate,
            mediation_grid: Some(self.grid),
            identify_provenance: Some(provenance_ids(
                "identify.temporal_mediation",
                "identify.temporal_mediation",
            )),
            posterior: self.posterior,
            predictive_checks: self.predictive_checks,
            n_draws: self.n_draws,
            ..Default::default()
        })
    }
}

/// One horizon's fit, before the grid is assembled.
struct HorizonFit {
    mediation: TemporalMediationEstimate,
    uncertainty: antecedent_estimate::TemporalMediationUncertainty,
    posterior: Option<CausalPosterior>,
    refutations: Vec<antecedent_validate::RefutationReport>,
    predictive_checks: Vec<PredictiveCheckReport>,
    diagnostics: Vec<Diagnostic>,
    replicates_ok: Option<u32>,
    cancelled: bool,
}

/// Fit and assemble the mediation of every requested horizon.
pub(super) fn fit_temporal_mediation(
    inputs: &TemporalMediationInputs<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalMediationFit, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_ESTIMATE_POINT });
    }
    let TemporalMediationInputs { query, horizons, .. } = *inputs;
    let first = horizons.first().ok_or_else(|| CausalError::Compile {
        message: "temporal mediation requires at least one horizon".into(),
    })?;
    let single = horizons.len() == 1;
    let bayesian = match &inputs.inference {
        MediationInference::Bayesian(config) => {
            require_isotropic_mediation_prior(config)?;
            Some(*config)
        }
        MediationInference::Frequentist { .. } => None,
    };
    let estimator = bayesian.map(|config| bayesian_gcomp(config, ctx));
    if let Some(estimator) = estimator.as_ref() {
        antecedent_estimate::bayesian_mediation::require_gaussian_mediation(estimator)
            .map_err(CausalError::from)?;
    }
    let mut diagnostics = Vec::new();
    if mediation_horizon_z_differs(horizons) {
        diagnostics.push(Diagnostic::new(
            "identify.temporal_mediation.horizon_dependent",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "adjustment sets differ across requested horizons; each contrast uses S(h) = I(h) \
             plus mediator-outcome confounders identified for that horizon, not a shared \
             max-horizon set",
        ));
    }
    if bayesian.is_some() {
        diagnostics.push(Diagnostic::new(
            "estimate.mediation.bayesian",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "independent Gaussian mediator and outcome mechanisms; total = direct + mediated for every posterior draw; NaturalDirect and NaturalIndirect use the licensed additive linear no-treatment-mediator-interaction alias",
        ));
    }

    let mut slices = Vec::with_capacity(horizons.len());
    let mut refutations = Vec::new();
    let mut predictive_checks = Vec::new();
    let mut first_posterior = None;
    let mut replicates_ok: Option<u32> = None;
    let mut cancelled = false;
    for click in horizons {
        require_identified(&click.identification)?;
        let mut horizon_query = query.clone();
        horizon_query.horizons = Arc::from([click.horizon]);
        let fit = match (&inputs.inference, estimator.as_ref()) {
            (MediationInference::Frequentist { bootstrap_replicates }, _) => {
                fit_frequentist_horizon(
                    inputs,
                    click,
                    &horizon_query,
                    *bootstrap_replicates,
                    single,
                    ctx,
                )?
            }
            (MediationInference::Bayesian(config), Some(estimator)) => {
                fit_bayesian_horizon(inputs, click, &horizon_query, config, estimator, ctx)?
            }
            (MediationInference::Bayesian(_), None) => unreachable!("estimator built above"),
        };
        let mut reports = fit.refutations;
        if !single {
            for report in &mut reports {
                report.refuter = Arc::from(format!("horizon.{}.{}", click.horizon, report.refuter));
            }
        }
        refutations.extend(reports);
        predictive_checks.extend(fit.predictive_checks);
        diagnostics.extend(fit.diagnostics);
        replicates_ok = match (replicates_ok, fit.replicates_ok) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        cancelled |= fit.cancelled;
        slices.push(antecedent_estimate::TemporalMediationSlice {
            horizon: click.horizon,
            identification_status: click.identification.status,
            method: Arc::clone(&click.estimand.method),
            adjustment: mediation_click_adjustment_keys(click),
            estimate: fit.mediation,
            uncertainty: fit.uncertainty,
            identified_set: None,
            diagnostics: click.identification.diagnostics.clone(),
        });
        if first_posterior.is_none() {
            first_posterior = fit.posterior;
        }
    }

    let head = slices.first().ok_or_else(|| CausalError::Compile {
        message: "temporal mediation requires at least one horizon".into(),
    })?;
    let estimate = if single { head.estimate.effect.clone() } else { nan_effect() };
    let mediation = single.then(|| head.estimate.clone());
    let mut identification = first.identification.clone();
    if !single {
        identification.status = most_conservative_identification_status(
            slices.iter().map(|slice| slice.identification_status),
        );
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_mediation.multi_horizon",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "all requested horizons are retained in mediation_grid, each with its own pointwise \
             uncertainty and no joint cross-horizon uncertainty; the scalar estimate and scalar \
             mediation fields are intentionally absent because no horizon is authoritative",
        ));
        diagnostics.push(Diagnostic::new(
            "identify.temporal_mediation.multi_horizon_status",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "parent identification.status is the most conservative requested-horizon \
             status; mediation_grid is authoritative per horizon",
        ));
    }
    let grid = antecedent_estimate::TemporalMediationGrid {
        slices: Arc::from(slices),
        joint_posterior: false,
    };
    let n_draws = match (&inputs.inference, single) {
        (MediationInference::Bayesian(config), true) => u32::try_from(config.n_draws).ok(),
        _ => None,
    };
    let bootstrap_replicates_ok = match &inputs.inference {
        MediationInference::Frequentist { bootstrap_replicates } if *bootstrap_replicates > 0 => {
            replicates_ok
        }
        _ => None,
    };
    Ok(TemporalMediationFit {
        estimator_id: inputs.inference.estimator_id(),
        estimate,
        mediation,
        identification,
        estimand: first.estimand.clone(),
        indexer: first.indexer.clone(),
        grid,
        refutations,
        predictive_checks,
        diagnostics,
        posterior: if single { first_posterior } else { None },
        n_draws,
        bootstrap_replicates_ok,
        cancelled,
        single,
    })
}

fn fit_frequentist_horizon(
    inputs: &TemporalMediationInputs<'_>,
    click: &MediationHorizonClick,
    horizon_query: &antecedent_core::MediationQuery,
    bootstrap_replicates: u32,
    single: bool,
    ctx: &ExecutionContext,
) -> Result<HorizonFit, CausalError> {
    let estimator = TemporalMediationEstimator::new().with_allow_natural_controlled_alias(true);
    // The iid analytic SEs stay NaN for every contrast; one shared circular-block
    // bootstrap over lag-aligned rows supplies the Total/Direct/Mediated SEs.
    let (mut mediation, block) = estimator
        .estimate_with_block_bootstrap(
            inputs.data,
            &click.estimand,
            horizon_query,
            &click.adjustment,
            antecedent_estimate::BlockBootstrap {
                replicates: bootstrap_replicates,
                stream_base: mediation_block_stream(MediationRoute::Temporal, click.horizon),
            },
            ctx,
        )
        .map_err(CausalError::from)?;
    let cancelled = mediation.effect.uncertainty.bootstrap.cancelled;
    let mut diagnostics = Vec::new();
    if block.replicates_attempted > 0 {
        mediation.effect = mediation
            .effect
            .clone()
            .with_block_family(antecedent_estimate::CircularBlockFamily::Mediation);
        let horizon_label =
            if single { String::new() } else { format!("horizon {}: ", click.horizon) };
        diagnostics.extend(temporal_dependence_se_diagnostics(
            antecedent_estimate::CircularBlockFamily::Mediation,
            block.block_length,
            block.rows,
            block.kernel_bias,
            block.effective_rows,
            true,
            &format!(
                "{horizon_label}iid analytic SEs are NaN for Total, Direct and Mediated; one \
                 shared circular-block replicate of consecutive lag-aligned rows refits all \
                 three mechanism regressions, {}/{} replicates",
                block.replicates_ok, block.replicates_attempted,
            ),
        ));
    }
    let refutations = if inputs.validation == RefuteSuite::None {
        Vec::new()
    } else {
        antecedent_validate::mediation::refute_temporal_mediation_adjusted(
            inputs.data,
            &click.estimand,
            horizon_query,
            &mediation,
            inputs.validation == RefuteSuite::Full,
            &click.adjustment,
            ctx,
        )
        .map_err(CausalError::from)?
    };
    let replicates_ok = Some(block.replicates_ok);
    let uncertainty = if block.replicates_attempted > 0 {
        antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
            requested: mediation.effect.se_bootstrap(),
            block,
        }
    } else {
        antecedent_estimate::TemporalMediationUncertainty::FrequentistPointwise {
            standard_error: mediation
                .effect
                .se_analytic()
                .is_finite()
                .then_some(mediation.effect.se_analytic()),
        }
    };
    Ok(HorizonFit {
        mediation,
        uncertainty,
        posterior: None,
        refutations,
        predictive_checks: Vec::new(),
        diagnostics,
        replicates_ok,
        cancelled,
    })
}

fn fit_bayesian_horizon(
    inputs: &TemporalMediationInputs<'_>,
    click: &MediationHorizonClick,
    horizon_query: &antecedent_core::MediationQuery,
    config: &BayesianConfig,
    estimator: &BayesianGComputationAte,
    ctx: &ExecutionContext,
) -> Result<HorizonFit, CausalError> {
    use antecedent_estimate::bayesian_mediation::{
        compose_temporal_mediation, prepare_temporal_mediation_adjusted,
    };
    let status = click.identification.status;
    let natural_alias = matches!(
        inputs.query.contrast,
        antecedent_core::MediationContrast::NaturalDirect
            | antecedent_core::MediationContrast::NaturalIndirect
    );
    let preparations = prepare_temporal_mediation_adjusted(
        inputs.data,
        &click.estimand,
        horizon_query,
        &click.adjustment,
        ctx,
    )
    .map_err(CausalError::from)?;
    // Each mechanism fits from its own `(master seed, horizon, mechanism)` stream
    // on a clone, so no fit depends on another's seed or on the order of fits.
    let fit = |scale: f64| -> Result<Vec<CausalPosterior>, CausalError> {
        preparations
            .iter()
            .enumerate()
            .map(|(mechanism, prep)| {
                let mut fitted = estimator.clone();
                fitted.prior_scale = scale;
                fitted.seed =
                    mediation_mechanism_seed(ctx.rng.master_seed(), click.horizon, mechanism);
                fitted
                    .fit(prep, status, &mut BayesianGCompWorkspace::default(), ctx)
                    .map_err(CausalError::from)
            })
            .collect()
    };
    let mechanisms = fit(config.prior_scale)?;
    let mut posterior =
        compose_temporal_mediation(&mechanisms[0], &mechanisms[1], horizon_query, status)
            .map_err(CausalError::from)?;
    if natural_alias {
        posterior.assumptions.push(antecedent_estimate::linear_no_interaction_restriction());
    }
    let mut effect = effect_from_posterior(&posterior)?;
    let point_estimate = TemporalMediationEstimate {
        effect: effect.clone(),
        total: Some(posterior.summaries.mean[1]),
        direct: Some(posterior.summaries.mean[2]),
        mediated: Some(posterior.summaries.mean[3]),
    };
    let mut refutations = Vec::new();
    let mut predictive_checks = Vec::new();
    if inputs.validation != RefuteSuite::None {
        refutations.extend(
            antecedent_validate::mediation::refute_temporal_mediation_adjusted(
                inputs.data,
                &click.estimand,
                horizon_query,
                &point_estimate,
                inputs.validation == RefuteSuite::Full,
                &click.adjustment,
                ctx,
            )
            .map_err(CausalError::from)?,
        );
        for (mechanism, (prep, post)) in preparations.iter().zip(&mechanisms).enumerate() {
            // Mediation refuses prior transfer, so the isotropic mechanism prior is
            // the prior in force.
            let prior = estimator.prior_in_force(prep.design.ncols);
            let prior_check = PriorPredictiveCheck::for_estimator(estimator, ctx)
                .check_with_prior(prep, &prior, ctx)
                .map_err(CausalError::from)?;
            let posterior_check = PosteriorPredictiveCheck::for_estimator(estimator, ctx)
                .check(prep, post)
                .map_err(CausalError::from)?;
            for report in [prior_check, posterior_check] {
                let mut refutation = report.to_refutation_report(effect.ate, 0.05);
                refutation.refuter =
                    Arc::from(format!("mediation.mechanism_{mechanism}.{}", refutation.refuter));
                refutations.push(refutation);
                predictive_checks.push(report);
            }
        }
    }
    if inputs.validation == RefuteSuite::Full {
        let sensitivity = antecedent_validate::PriorSensitivity::standard_grid();
        let mut means = Vec::new();
        let mut sds = Vec::new();
        for &scale in sensitivity.scales.iter() {
            let scaled = fit(scale)?;
            let scaled_posterior =
                compose_temporal_mediation(&scaled[0], &scaled[1], horizon_query, status)
                    .map_err(CausalError::from)?;
            means.push(scaled_posterior.summaries.mean[0]);
            sds.push(scaled_posterior.summaries.sd[0]);
        }
        let summary = antecedent_prob::PriorSensitivitySummary {
            family: antecedent_prob::PriorSensitivityFamily::IsotropicScale,
            prior_scales: sensitivity.scales.clone(),
            alphas: Arc::from([]),
            variance_multipliers: Arc::from([]),
            effect_means: Arc::from(means),
            effect_sds: Arc::from(sds),
        };
        refutations.push(sensitivity.to_report(&summary, effect.ate));
        posterior = with_prior_sensitivity(posterior, summary);
        effect = effect_from_posterior(&posterior)?;
    }
    let summary = |column: usize| antecedent_estimate::MediationPosteriorSummary {
        mean: posterior.summaries.mean[column],
        standard_deviation: posterior.summaries.sd[column],
        q025: posterior.summaries.q025[column],
        q975: posterior.summaries.q975[column],
    };
    let uncertainty = antecedent_estimate::TemporalMediationUncertainty::BayesianPointwise {
        requested: summary(0),
        total: summary(1),
        direct: summary(2),
        mediated: summary(3),
        n_draws: posterior.draws.n_draws,
        backend: Arc::clone(&posterior.diagnostics.backend_id),
    };
    Ok(HorizonFit {
        mediation: TemporalMediationEstimate {
            effect,
            total: Some(posterior.summaries.mean[1]),
            direct: Some(posterior.summaries.mean[2]),
            mediated: Some(posterior.summaries.mean[3]),
        },
        uncertainty,
        posterior: Some(posterior),
        refutations,
        predictive_checks,
        diagnostics: Vec::new(),
        replicates_ok: None,
        cancelled: false,
    })
}
