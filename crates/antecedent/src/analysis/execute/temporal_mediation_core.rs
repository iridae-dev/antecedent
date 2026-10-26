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
    PosteriorPredictiveCheck, PredictiveCheckReport, PriorPredictiveCheck, RefuteSuite, Strategy,
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
    if config.has_transfer() {
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
    /// Declared handling of missing time points, when the caller declared one.
    pub missing: Option<&'a antecedent_data::MissingDeclaration>,
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
    /// Windows dropped for a missing time point, when the fit accounts for them.
    pub(super) missing_data: Option<crate::result::MissingDataReport>,
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
            Strategy::new(IdentifierId::Frontdoor, self.estimator_id),
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
#[allow(
    clippy::too_many_lines,
    reason = "every horizon shares the gap policy, the design and the replicate loop of one mediation fit"
)]
pub(super) fn fit_temporal_mediation(
    inputs: &TemporalMediationInputs<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalMediationFit, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_ESTIMATE_POINT });
    }
    let TemporalMediationInputs { horizons, .. } = *inputs;
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
    let mut diagnostics = mediation_preamble_diagnostics(horizons, bayesian.is_some());
    let gap_variables = mediation_variables(inputs);
    let first_gap = antecedent_data::first_missing_cell(inputs.data, &gap_variables)?;
    let gaps = first_gap.is_some();
    if let Some(cell) = &first_gap {
        admit_mediation_gaps(inputs, cell)?;
    }

    let mut slices = Vec::with_capacity(horizons.len());
    let mut refutations = Vec::new();
    let mut predictive_checks = Vec::new();
    let mut first_posterior = None;
    let mut replicates_ok: Option<u32> = None;
    let mut cancelled = false;
    for click in horizons {
        require_identified(&click.identification)?;
        let fit = fit_mediation_horizon(inputs, estimator.as_ref(), click, single, gaps, ctx)?;
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
        diagnostics.extend(multi_horizon_mediation_diagnostics());
    }
    let grid = antecedent_estimate::TemporalMediationGrid {
        slices: Arc::from(slices),
        joint_posterior: false,
    };
    let n_draws = match (&inputs.inference, single) {
        (MediationInference::Bayesian(config), true) => u32::try_from(config.n_draws).ok(),
        _ => None,
    };
    let bootstrap_requested = matches!(
        inputs.inference,
        MediationInference::Frequentist { bootstrap_replicates } if bootstrap_replicates > 0
    );
    let bootstrap_replicates_ok = bootstrap_requested.then_some(replicates_ok).flatten();
    let missing_data = if gaps {
        // The first horizon's windows; later horizons read other lags of the same gaps.
        let selection = TemporalMediationEstimator::new()
            .with_allow_natural_controlled_alias(true)
            .window_selection(
                inputs.data,
                &first.estimand,
                &{
                    let mut query = inputs.query.clone();
                    query.horizons = Arc::from([first.horizon]);
                    query
                },
                &first.adjustment,
                ctx,
            )
            .map_err(CausalError::from)?;
        let dropped = selection.dropped_times().into_iter().map(|time| time as u64).collect();
        let accounting = antecedent_data::RowAccounting::from_dropped(
            antecedent_data::MissingPolicy::CompleteCase,
            selection.complete.len(),
            dropped,
            missing_series_points(inputs.data, &gap_variables)?,
        );
        Some(crate::analysis::missing_data::report(
            crate::analysis::missing_data::MissingRoute::Temporal,
            true,
            Arc::from([]),
            accounting,
        ))
    } else {
        None
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
        missing_data,
    })
}

/// Series variables a mediation fit reads: treatment, outcome, mediator and every horizon's
/// adjustment columns.
fn mediation_variables(inputs: &TemporalMediationInputs<'_>) -> Vec<antecedent_core::VariableId> {
    let mut variables = vec![inputs.query.treatment, inputs.query.outcome];
    for click in inputs.horizons {
        variables.extend(click.estimand.mediators.iter().copied());
        variables.extend(click.adjustment.iter().map(|column| column.variable));
    }
    let mut seen = Vec::new();
    variables.retain(|variable| {
        let new = !seen.contains(variable);
        seen.push(*variable);
        new
    });
    variables
}

/// Admit a series with gaps only under a declared complete-case policy without refuters (the
/// mediation refuters refit through contiguous windows). The Bayesian mechanisms are tempered
/// by the segment-wise long-run variance of the intact windows.
fn admit_mediation_gaps(
    inputs: &TemporalMediationInputs<'_>,
    cell: &antecedent_data::MissingCell,
) -> Result<(), CausalError> {
    use crate::analysis::missing_data::MissingRoute;
    use antecedent_data::MissingPolicy;
    let route = MissingRoute::Temporal;
    let supported = route.supported();
    let Some(declaration) =
        inputs.missing.filter(|declaration| declaration.policy == MissingPolicy::CompleteCase)
    else {
        if inputs.missing.is_none() {
            return Err(CausalError::Data(antecedent_data::DataError::IncompleteSeries {
                id: Some(cell.id),
                message: "missing values in series; lagged temporal analysis requires complete series (declare missing=\"complete_case\" to drop the lag windows that read a gap)",
            }));
        }
        return Err(CausalError::Estimate(antecedent_estimate::missing_policy::policy_required(
            cell,
            route.label(),
            &supported[1..],
        )));
    };
    let refuse = |reason: &str| {
        CausalError::Estimate(antecedent_estimate::missing_policy::policy_not_applicable(
            declaration.policy,
            route.label(),
            reason,
            &supported[..1],
        ))
    };
    if inputs.validation != RefuteSuite::None
        && matches!(inputs.inference, MediationInference::Bayesian(_))
    {
        return Err(refuse(
            "the mediation predictive checks read consecutive rows; request no refuters on a Bayesian fit of a series with gaps",
        ));
    }
    Ok(())
}

/// Missing time points per variable.
fn missing_series_points(
    data: &TimeSeriesData,
    variables: &[antecedent_core::VariableId],
) -> Result<Vec<(Arc<str>, usize)>, CausalError> {
    use antecedent_data::TableView;
    variables
        .iter()
        .map(|&id| {
            let validity = data.column(id)?.validity();
            let name = Arc::clone(&data.schema().get(id)?.name);
            Ok((name, (0..data.row_count()).filter(|&row| !validity.is_valid(row)).count()))
        })
        .collect()
}

/// The horizon-dependent adjustment note and the Bayesian mechanism note that
/// open a mediation fit's diagnostics.
fn mediation_preamble_diagnostics(
    horizons: &[MediationHorizonClick],
    bayesian: bool,
) -> Vec<Diagnostic> {
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
    if bayesian {
        diagnostics.push(Diagnostic::new(
            "estimate.mediation.bayesian",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "independent Gaussian mediator and outcome mechanisms; total = direct + mediated for every posterior draw; NaturalDirect and NaturalIndirect use the licensed additive linear no-treatment-mediator-interaction alias",
        ));
    }
    diagnostics
}

/// The notes a multi-horizon mediation carries: no scalar estimate, and the
/// parent status is the most conservative horizon's.
fn multi_horizon_mediation_diagnostics() -> [Diagnostic; 2] {
    [
        Diagnostic::new(
            "estimate.temporal_mediation.multi_horizon",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "all requested horizons are retained in mediation_grid, each with its own pointwise \
             uncertainty and no joint cross-horizon uncertainty; the scalar estimate and scalar \
             mediation fields are intentionally absent because no horizon is authoritative",
        ),
        Diagnostic::new(
            "identify.temporal_mediation.multi_horizon_status",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "parent identification.status is the most conservative requested-horizon \
             status; mediation_grid is authoritative per horizon",
        ),
    ]
}

/// One horizon's fit under the study's inference mode.
fn fit_mediation_horizon(
    inputs: &TemporalMediationInputs<'_>,
    estimator: Option<&BayesianGComputationAte>,
    click: &MediationHorizonClick,
    single: bool,
    gaps: bool,
    ctx: &ExecutionContext,
) -> Result<HorizonFit, CausalError> {
    let mut horizon_query = inputs.query.clone();
    horizon_query.horizons = Arc::from([click.horizon]);
    match (&inputs.inference, estimator) {
        (MediationInference::Frequentist { bootstrap_replicates }, _) => fit_frequentist_horizon(
            inputs,
            click,
            &horizon_query,
            *bootstrap_replicates,
            single,
            gaps,
            ctx,
        ),
        (MediationInference::Bayesian(config), Some(estimator)) => {
            fit_bayesian_horizon(inputs, click, &horizon_query, config, estimator, gaps, ctx)
        }
        (MediationInference::Bayesian(_), None) => Err(CausalError::Compile {
            message: "Bayesian temporal mediation horizon has no estimator".into(),
        }),
    }
}

fn fit_frequentist_horizon(
    inputs: &TemporalMediationInputs<'_>,
    click: &MediationHorizonClick,
    horizon_query: &antecedent_core::MediationQuery,
    bootstrap_replicates: u32,
    single: bool,
    gaps: bool,
    ctx: &ExecutionContext,
) -> Result<HorizonFit, CausalError> {
    let estimator = TemporalMediationEstimator::new()
        .with_allow_natural_controlled_alias(true)
        .with_allow_gaps(gaps);
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
        antecedent_validate::mediation::refute_temporal_mediation_with_gaps(
            inputs.data,
            &click.estimand,
            horizon_query,
            &mediation,
            inputs.validation == RefuteSuite::Full,
            &click.adjustment,
            gaps,
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

/// Each mechanism fits from its own `(master seed, horizon, mechanism)` stream
/// on a clone, so no fit depends on another's seed or on the order of fits.
fn fit_mediation_mechanisms(
    estimator: &BayesianGComputationAte,
    preparations: &[antecedent_estimate::PreparedBayesianProblem],
    (horizon, status): (u32, antecedent_core::IdentificationStatus),
    prior_scale: f64,
    ctx: &ExecutionContext,
) -> Result<Vec<CausalPosterior>, CausalError> {
    let fitted = preparations
        .iter()
        .enumerate()
        .map(|(mechanism, prep)| {
            let mut fitted = estimator.clone();
            fitted.prior_scale = prior_scale;
            fitted.seed = mediation_mechanism_seed(ctx.rng.master_seed(), horizon, mechanism);
            fitted
                .fit(prep, status, &mut BayesianGCompWorkspace::default(), ctx)
                .map_err(CausalError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The mediator and outcome share the series: their draws are one joint law.
    Ok(antecedent_estimate::bayesian_mediation::couple_mediation_mechanisms(
        preparations,
        fitted,
        horizon as usize,
        mediation_mechanism_seed(ctx.rng.master_seed(), horizon, preparations.len()),
        ctx,
    ))
}

fn fit_bayesian_horizon(
    inputs: &TemporalMediationInputs<'_>,
    click: &MediationHorizonClick,
    horizon_query: &antecedent_core::MediationQuery,
    config: &BayesianConfig,
    estimator: &BayesianGComputationAte,
    gaps: bool,
    ctx: &ExecutionContext,
) -> Result<HorizonFit, CausalError> {
    use antecedent_estimate::bayesian_mediation::{
        compose_temporal_mediation_mechanisms, prepare_temporal_mediation_with_gaps,
    };
    let status = click.identification.status;
    let natural_alias = matches!(
        inputs.query.contrast,
        antecedent_core::MediationContrast::NaturalDirect
            | antecedent_core::MediationContrast::NaturalIndirect
    );
    let preparations = prepare_temporal_mediation_with_gaps(
        inputs.data,
        &click.estimand,
        horizon_query,
        &click.adjustment,
        ctx,
        gaps,
    )
    .map_err(CausalError::from)?;
    let fit = |scale: f64| {
        fit_mediation_mechanisms(estimator, &preparations, (click.horizon, status), scale, ctx)
    };
    let mechanisms = fit(config.prior_scale)?;
    let mut posterior = compose_temporal_mediation_mechanisms(&mechanisms, horizon_query, status)
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
            antecedent_validate::mediation::refute_temporal_mediation_with_gaps(
                inputs.data,
                &click.estimand,
                horizon_query,
                &point_estimate,
                inputs.validation == RefuteSuite::Full,
                &click.adjustment,
                gaps,
                ctx,
            )
            .map_err(CausalError::from)?,
        );
        predictive_checks = mechanism_predictive_checks(
            estimator,
            (&preparations, &mechanisms),
            effect.ate,
            &mut refutations,
            ctx,
        )?;
    }
    if inputs.validation == RefuteSuite::Full {
        let (report, summary) =
            mediation_prior_sensitivity(&fit, horizon_query, status, effect.ate)?;
        refutations.push(report);
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

/// Refits the mechanisms over the standard prior-scale grid and summarizes the
/// effect's sensitivity to the isotropic prior scale.
fn mediation_prior_sensitivity(
    fit: &impl Fn(f64) -> Result<Vec<CausalPosterior>, CausalError>,
    horizon_query: &antecedent_core::MediationQuery,
    status: antecedent_core::IdentificationStatus,
    effect_ate: f64,
) -> Result<
    (antecedent_validate::RefutationReport, antecedent_prob::PriorSensitivitySummary),
    CausalError,
> {
    use antecedent_estimate::bayesian_mediation::compose_temporal_mediation_mechanisms;
    let sensitivity = antecedent_validate::PriorSensitivity::standard_grid();
    let mut means = Vec::new();
    let mut sds = Vec::new();
    for &scale in sensitivity.scales.iter() {
        let scaled = fit(scale)?;
        let scaled_posterior =
            compose_temporal_mediation_mechanisms(&scaled, horizon_query, status)
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
    Ok((sensitivity.to_report(&summary, effect_ate), summary))
}

/// Prior and posterior predictive checks of each mechanism fit; each check's
/// refutation report is appended, labelled by mechanism.
fn mechanism_predictive_checks(
    estimator: &BayesianGComputationAte,
    (preparations, mechanisms): (
        &[antecedent_estimate::PreparedBayesianProblem],
        &[CausalPosterior],
    ),
    effect_ate: f64,
    refutations: &mut Vec<antecedent_validate::RefutationReport>,
    ctx: &ExecutionContext,
) -> Result<Vec<PredictiveCheckReport>, CausalError> {
    let mut predictive_checks = Vec::new();
    for (mechanism, (prep, post)) in preparations.iter().zip(mechanisms).enumerate() {
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
            let mut refutation = report.to_refutation_report(effect_ate, 0.05);
            refutation.refuter =
                Arc::from(format!("mediation.mechanism_{mechanism}.{}", refutation.refuter));
            refutations.push(refutation);
            predictive_checks.push(report);
        }
    }
    Ok(predictive_checks)
}
